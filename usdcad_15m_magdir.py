"""USDCAD 15-MIN — E2 DIRECTION-CONDITIONED-ON-MAGNITUDE lever (RUN the sign-invariance test).

SCOPE: USDCAD · 15m · NY session. Fork of usdjpy_15m_magdir.py (PAIR=USDCAD). Train the base NY GBM
direction model AND a separate MAGNITUDE model predicting P(|ret_15m| >= train-Q75); on held-out NY moved
bars, bucket by the magnitude model's PREDICTED-magnitude quartile (Q1 low ... Q4 high) and report per
quartile: direction moved-AUC + selective win-rate.

WHY THIS MIGHT NOT BE NULL FOR USDCAD (the petrocurrency angle): the sign-invariance theorem
(arXiv:2512.15720) says magnitude/vol gates SIZE not SIGN -> expected NULL (flat dir-AUC across mag
quartiles). BUT USDCAD's big moves are disproportionately OIL-DRIVEN (WTI shocks -> sharp, directional CAD
moves), so the HIGH-magnitude bucket could plausibly carry CLEANER sign than the low-mag grind. This is the
one lever where the petrocurrency mechanism gives a genuine reason to expect a non-flat profile. RUN it; let
the number decide.

EVAL: deriv-faithful fixed-15m sign (ties LOSE), NY decision rows, per-year moved-AUC + cov3% side_eval,
nonoverlap gap=900. Pre-registered falsifier in result JSON BEFORE held-out.
Usage: ~/binary-algo-venv/bin/python usdcad_15m_magdir.py [stride=6] [leaves=127]
"""
import os, sys, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
import sessions
from usdcad_15m_base import build, side_eval, boot, mk_lgb, BE, SPL, FEATS

PAIR = "USDCAD"; HOR = 15; STEP = 60; GAP = HOR * STEP
FEAT = H.FEAT_DIR
KEY = "USDCAD.15m.ny"
RESULT = "usdcad_15m_magdir_result.json"
MAG_Q = 0.75

def _argint(i, default):
    return int(sys.argv[i]) if len(sys.argv) > i and str(sys.argv[i]).isdigit() else default

TR_STRIDE = _argint(1, 6)
NUM_LEAVES = _argint(2, 127)

def build_mag(years, stride=1):
    """MIRROR of usdcad_15m_base.build(), additionally returning |fwd_ret| (absret) aligned to (X,y,moved,ts)."""
    Xs = []; ys = []; mv = []; tss = []; ars = []
    for y in years:
        p = f"{FEAT}/{PAIR}_{y}.parquet"
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p, columns=FEATS + ["close"])
        d = d[~d.index.duplicated(keep="last")]
        c = d["close"].values.astype(float)
        ts = d.index.values.astype("datetime64[s]").astype("int64"); n = len(d)
        contig = np.zeros(n, bool); contig[:n - HOR] = (ts[HOR:] - ts[:-HOR]) == GAP
        fr = np.full(n, np.nan); fr[:n - HOR] = c[HOR:] / c[:-HOR] - 1.0
        X = d[FEATS].astype("float32")
        keepf = X.isna().mean(axis=1).values < 0.5
        valid = contig & np.isfinite(fr) & keepf
        moved = valid & (fr != 0.0)
        idx = np.where(valid)[0]
        if stride > 1:
            idx = idx[::stride]
        Xs.append(X.iloc[idx]); ys.append((fr[idx] > 0).astype(int))
        mv.append(moved[idx]); tss.append(ts[idx]); ars.append(np.abs(fr[idx]))
    X = pd.concat(Xs)
    return X, np.concatenate(ys), np.concatenate(mv), np.concatenate(tss), np.concatenate(ars)

def ny_mask(ts):
    return sessions.session_mask(ts, "ny")

def main():
    t0 = time.time()
    res = {
        "key": KEY,
        "lever": "E2 direction-conditioned-on-magnitude (base NY dir GBM x magnitude-quartile bucketing); petrocurrency angle",
        "settlement": "deriv-faithful sign(close[t+15]-close[t]); ties LOSE; BE=0.541; gap=900s nonoverlap",
        "splits": SPL, "tr_stride": TR_STRIDE, "num_leaves": NUM_LEAVES, "mag_target_quantile": MAG_Q,
        "falsifier": {
            "registered_utc": "pre-OOS",
            "KILL_if": ("in fewer than 2 of {test24,test25,oos} does Q4(HIGH-mag) NY dir moved-AUC exceed BOTH "
                        "(a) Q1(LOW-mag) by >0.010 AND (b) ~0.532 (USDCAD NY incumbent AUC) — AND no year's Q4 "
                        "cov3% win-rate CI95-lower clears 0.541 in >=2 years"),
            "expected": ("NULL per sign-invariance (mag gates SIZE not SIGN) UNLESS the petrocurrency/oil angle holds "
                         "(oil-driven big moves carry cleaner sign) — then Q4 lifts. RUN it; let the number decide."),
            "incumbent": {"ny_signal_auc": 0.532, "ny_winrate": "0.58-0.60", "BE": BE},
        },
        "years": {},
    }
    json.dump(res, open(RESULT, "w"), indent=2)

    Xtr, ytr, mtr, tstr, atr = build_mag(SPL["train"], TR_STRIDE)
    Xva, yva, mva, tsv, ava = build_mag(SPL["val"], 1)
    ny_tr = ny_mask(tstr); ny_va = ny_mask(tsv)
    magthr = float(np.quantile(atr[mtr & ny_tr], MAG_Q))
    ymag_tr = (atr >= magthr).astype(int); ymag_va = (ava >= magthr).astype(int)
    dir_tr = mtr & ny_tr; dir_va = mva & ny_va
    mag_tr = ny_tr; mag_va = ny_va
    print(f"[magdir15m USDCAD] stride={TR_STRIDE} dir_train(NY moved)={int(dir_tr.sum()):,} "
          f"mag_train(NY)={int(mag_tr.sum()):,} val(NY moved)={int(dir_va.sum()):,} magQ75={magthr:.2e} build={time.time()-t0:.0f}s", flush=True)

    Ld = mk_lgb(num_leaves=NUM_LEAVES)
    Ld.fit(Xtr[dir_tr], ytr[dir_tr], eval_set=[(Xva[dir_va], yva[dir_va])], eval_metric="auc",
           callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    Lm = mk_lgb(num_leaves=NUM_LEAVES)
    Lm.fit(Xtr[mag_tr], ymag_tr[mag_tr], eval_set=[(Xva[mag_va], ymag_va[mag_va])], eval_metric="auc",
           callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    pva_d = Ld.predict_proba(Xva)[:, 1]; pva_m = Lm.predict_proba(Xva)[:, 1]
    val_dir_auc = float(roc_auc_score(yva[dir_va], pva_d[dir_va]))
    val_mag_auc = float(roc_auc_score(ymag_va[mag_va], pva_m[mag_va]))
    print(f"[magdir15m USDCAD] VAL NY dir-AUC={val_dir_auc:.4f}  VAL NY magAUC={val_mag_auc:.4f} {time.time()-t0:.0f}s", flush=True)
    res["val_or_signal_auc"] = val_dir_auc; res["val_mag_auc"] = val_mag_auc
    qedges = list(np.quantile(pva_m[dir_va], [0.25, 0.50, 0.75]))
    res["mag_quartile_edges_val"] = [float(q) for q in qedges]
    del Xtr, Xva, ytr, yva, mtr, mva, atr, ava, ymag_tr, ymag_va

    bucket_names = ["Q1_low", "Q2", "Q3", "Q4_high"]
    for w in ("test24", "test25", "oos"):
        Xw, yw, mw, tsw, aw = build_mag(SPL[w], 1)
        pdir = Ld.predict_proba(Xw)[:, 1]; pmag = Lm.predict_proba(Xw)[:, 1]
        ny = ny_mask(tsw); nymoved = ny & mw
        ymag_w = (aw >= magthr).astype(int)
        mag_auc = float(roc_auc_score(ymag_w[ny], pmag[ny])) if ny.sum() > 50 else float("nan")
        auc_all = float(roc_auc_score(yw[nymoved], pdir[nymoved])) if nymoved.sum() > 50 else float("nan")
        up_rate = float(yw[nymoved].mean()) if nymoved.sum() > 0 else float("nan")
        conf_ny = np.abs(pdir[ny] - 0.5); thr3 = float(np.quantile(conf_ny, 0.97)) if ny.sum() > 0 else 1.0
        gate3 = side_eval(pdir[ny], yw[ny], mw[ny], tsw[ny], thr3)
        g3 = gate3["COMBINED"] if gate3 else {"n": 0, "wr": float("nan"), "ci": [float("nan")] * 2}
        edges = [-np.inf] + list(qedges) + [np.inf]; quart = {}
        for bi, bname in enumerate(bucket_names):
            lo, hi = edges[bi], edges[bi + 1]
            bmask = nymoved & (pmag > lo) & (pmag <= hi)
            if bmask.sum() > 50:
                bauc = float(roc_auc_score(yw[bmask], pdir[bmask]))
                bm_ny = ny & (pmag > lo) & (pmag <= hi)
                conf_b = np.abs(pdir[bm_ny] - 0.5); cthr = float(np.quantile(conf_b, 0.70)) if bm_ny.sum() > 30 else 1.0
                bg = side_eval(pdir[bm_ny], yw[bm_ny], mw[bm_ny], tsw[bm_ny], cthr)
                bgc = bg["COMBINED"] if bg else {"n": 0, "wr": float("nan"), "ci": [float("nan")] * 2}
            else:
                bauc = float("nan"); bgc = {"n": 0, "wr": float("nan"), "ci": [float("nan")] * 2}
            quart[bname] = {"n_moved": int(bmask.sum()), "dir_moved_auc": bauc, "sel_wr": bgc["wr"], "sel_n": bgc["n"], "sel_ci": bgc["ci"]}
        res["years"][w] = {"auc": auc_all, "moved_up_rate": up_rate, "tripwire_ok": bool(0.47 <= up_rate <= 0.53),
                           "mag_auc": mag_auc, "cov3_wr": g3["wr"], "cov3_n": g3["n"], "cov3_ci": g3["ci"], "quartiles": quart}
        print(f"=== {w} === NY dir-AUC={auc_all:.4f} up={up_rate:.4f} magAUC={mag_auc:.4f} | cov3 wr={g3['wr']:.4f} n{g3['n']}", flush=True)
        for bname in bucket_names:
            q = quart[bname]; print(f"        {bname:8} dir-AUC={q['dir_moved_auc']:.4f} (n {q['n_moved']}) | sel wr={q['sel_wr']:.4f} n{q['sel_n']} CI[{q['sel_ci'][0]:.3f},{q['sel_ci'][1]:.3f}]", flush=True)
        del Xw, yw, mw, tsw, aw, pdir, pmag

    AUC_BASE = 0.532; AUC_MARGIN = 0.010
    q4_beats = []; q4_wr_clears = []
    for w in ("test24", "test25", "oos"):
        q = res["years"][w]["quartiles"]
        a1 = q["Q1_low"]["dir_moved_auc"]; a4 = q["Q4_high"]["dir_moved_auc"]
        if np.isfinite(a1) and np.isfinite(a4) and (a4 > a1 + AUC_MARGIN) and (a4 > AUC_BASE): q4_beats.append(w)
        ci_lo = q["Q4_high"]["sel_ci"][0]
        if np.isfinite(ci_lo) and ci_lo >= BE: q4_wr_clears.append(w)
    auc_beats = [w for w in ("test24", "test25", "oos") if np.isfinite(res["years"][w]["auc"]) and res["years"][w]["auc"] > AUC_BASE]
    killed = not (len(q4_beats) >= 2 or len(q4_wr_clears) >= 2)
    res["verdict"] = {"KILLED": bool(killed), "beats_base_auc": bool(len(auc_beats) >= 2),
                      "q4_beats_q1_and_base_years": q4_beats, "q4_cov_wr_clears_BE_years": q4_wr_clears,
                      "full_ny_auc_beats_base_years": auc_beats,
                      "note": "E2: petrocurrency angle = does the HIGH-mag (oil-driven) bucket carry cleaner sign? SURVIVES if Q4 dir-AUC beats Q1+.010 & >.532 in >=2yr OR Q4 cov-WR CI-lo clears BE in >=2yr."}
    json.dump(res, open(RESULT, "w"), indent=2)
    print(f"\n[magdir15m USDCAD] VERDICT: {'KILLED' if killed else 'SURVIVED'} (q4_beats={q4_beats}; q4_wr_clears={q4_wr_clears}; auc_beats={auc_beats}) -> {RESULT} total={time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
