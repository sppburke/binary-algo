"""USDJPY 15-MIN — E2 DIRECTION-CONDITIONED-ON-MAGNITUDE lever (RUN the sign-invariance test).

SCOPE: USDJPY · 15m · NY session (America/New_York 08-17, DST-correct via sessions.session_mask).
LEVER (method E2): train the base NY GBM direction model (mk_lgb on NY moved bars), AND a separate
MAGNITUDE model that predicts P(|ret_15m| >= train-Q75). On held-out NY moved bars, bucket bars by the
magnitude model's PREDICTED-magnitude quartile (Q1 low ... Q4 high) and report, per quartile:
  - DIRECTION moved-AUC  (base dir model's signal vs the deriv-faithful up/down label)
  - selective win-rate    (side_eval at the in-bucket confidence gate)

THE TEST (sign-invariance, theorem arXiv:2512.15720): magnitude/state/volatility gates are EXPECTED to
gate move SIZE, not SIGN. So the EXPECTED result is NULL: direction AUC FLAT across magnitude quartiles
(high-mag bars NOT more sign-predictable). We RUN it faithfully and let the number decide.

EVAL DISCIPLINE (non-negotiable, inherited from usdjpy_15m_base):
  * FEATURES stay causal/continuous; only the DECISION rows are restricted to the NY session.
  * The EVAL label is ALWAYS the deriv-faithful fixed-15m sign = sign(close[t+15]-close[t]) with ties
    (move==0) LOSING. This is exactly what build() returns via (y, moved).
  * Per held-out year (test24/test25/oos): moved-AUC of the dir signal on NY moved bars, AND a selective
    win-rate at cov3% via side_eval (nonoverlap gap=900 built in). Sanity: NY moved up-rate ~0.47-0.53.
  * Pre-register the FALSIFIER in the result JSON BEFORE reading held-out; then write the verdict.

INCUMBENT to beat: certified NY own-pair GBM signal AUC ~.539 / win-rate ~.58-.60. A lever SURVIVES only
if held-out moved-AUC exceeds ~.539 OR its NY cov3% win-rate CI-lower clears 0.541 in >=2 years (here,
specifically the HIGH-magnitude-quartile direction must do so — a flat profile is the null KILL).

MEMORY/COMPUTE: load the 239 base feats via the base contiguity logic at stride>=6 for train; held-out
years loaded full (stride 1) but one year at a time and freed. Peak RAM < ~3GB, runtime < ~10min.

Usage: ~/binary-algo-venv/bin/python usdjpy_15m_magdir.py [stride=6] [leaves=127]
"""
import os, sys, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
import sessions
from usdjpy_15m_base import build, side_eval, boot, mk_lgb, BE, SPL, FEATS

PAIR = "USDJPY"; HOR = 15; STEP = 60; GAP = HOR * STEP   # 15m -> 900s horizon, gap 900s
FEAT = H.FEAT_DIR
KEY = "USDJPY.15m.ny"
RESULT = "usdjpy_15m_magdir_result.json"
MAG_Q = 0.75                                            # magnitude target: |ret_15m| >= train-Q75 (== m10_magdir)


def _argint(i, default):
    return int(sys.argv[i]) if len(sys.argv) > i and str(sys.argv[i]).isdigit() else default


TR_STRIDE = _argint(1, 6)
NUM_LEAVES = _argint(2, 127)


def build_mag(years, stride=1):
    """MIRROR of usdjpy_15m_base.build() row-for-row, additionally returning |fwd_ret| (absret) so the
    magnitude label/AUC is perfectly aligned with (X, y, moved, ts). Same contiguity / NaN / stride logic
    as build(), so X here is identical to build()'s X — we re-derive absret on the SAME rows. Features are
    causal (<=t); the magnitude label |ret_15m| is forward and used ONLY as a learning/eval target, never
    fed back into X."""
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

    # ---- pre-register FALSIFIER BEFORE any held-out read ----
    res = {
        "key": KEY,
        "lever": "E2 direction-conditioned-on-magnitude (base NY dir GBM x magnitude-quartile bucketing)",
        "model": ("base direction = mk_lgb on NY MOVED bars (239 base feats, 15m own-clock sign label); "
                  "magnitude = mk_lgb predicting P(|ret_15m| >= train-Q75); held-out NY moved bars bucketed "
                  "by PREDICTED-magnitude quartile, direction moved-AUC + cov3% side_eval reported per quartile"),
        "settlement": "deriv-faithful sign(close[t+15]-close[t]); ties (move==0) LOSE; BE=0.541; gap=900s nonoverlap",
        "splits": SPL, "tr_stride": TR_STRIDE, "num_leaves": NUM_LEAVES, "mag_target_quantile": MAG_Q,
        "falsifier": {
            "registered_utc": "pre-OOS",
            "KILL_if": ("direction moved-AUC is FLAT across predicted-magnitude quartiles "
                        "(HIGH-mag Q4 NOT more sign-predictable than LOW-mag Q1) — concretely: in fewer than 2 of "
                        "{test24,test25,oos} does the Q4(HIGH-mag) NY direction moved-AUC exceed BOTH (a) the Q1(LOW-mag) "
                        "AUC by >0.010 AND (b) ~0.539 (incumbent NY signal AUC) — AND no year's Q4 cov3% win-rate "
                        "CI95-lower clears 0.541 in >=2 years"),
            "expected": ("NULL per sign-invariance theorem arXiv:2512.15720 (magnitude gates SIZE not SIGN); "
                         "flat direction-AUC across mag quartiles => KILL. RUN it; let the number decide."),
            "incumbent": {"ny_signal_auc": 0.539, "ny_winrate": "0.58-0.60", "BE": BE},
        },
        "years": {},
    }
    json.dump(res, open(RESULT, "w"), indent=2)

    # ---- TRAIN: direction (NY moved bars) + magnitude (|ret|>=train-Q75), features causal ----
    Xtr, ytr, mtr, tstr, atr = build_mag(SPL["train"], TR_STRIDE)
    Xva, yva, mva, tsv, ava = build_mag(SPL["val"], 1)
    ny_tr = ny_mask(tstr); ny_va = ny_mask(tsv)

    # magnitude threshold from TRAIN (NY moved bars define the magnitude regime we'll bucket within)
    magthr = float(np.quantile(atr[mtr & ny_tr], MAG_Q))
    ymag_tr = (atr >= magthr).astype(int)
    ymag_va = (ava >= magthr).astype(int)

    dir_tr = mtr & ny_tr          # direction: train/early-stop on NY MOVED bars (ties excluded)
    dir_va = mva & ny_va
    mag_tr = ny_tr                # magnitude: all NY valid bars (ties carry size info too)
    mag_va = ny_va
    print(f"[magdir15m] stride={TR_STRIDE} leaves={NUM_LEAVES} dir_train(NY moved)={int(dir_tr.sum()):,} "
          f"mag_train(NY)={int(mag_tr.sum()):,} val(NY moved)={int(dir_va.sum()):,} magQ75={magthr:.2e} "
          f"build={time.time()-t0:.0f}s", flush=True)

    Ld = mk_lgb(num_leaves=NUM_LEAVES)
    Ld.fit(Xtr[dir_tr], ytr[dir_tr], eval_set=[(Xva[dir_va], yva[dir_va])], eval_metric="auc",
           callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    Lm = mk_lgb(num_leaves=NUM_LEAVES)
    Lm.fit(Xtr[mag_tr], ymag_tr[mag_tr], eval_set=[(Xva[mag_va], ymag_va[mag_va])], eval_metric="auc",
           callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])

    pva_d = Ld.predict_proba(Xva)[:, 1]
    pva_m = Lm.predict_proba(Xva)[:, 1]
    val_dir_auc = float(roc_auc_score(yva[dir_va], pva_d[dir_va]))
    val_mag_auc = float(roc_auc_score(ymag_va[mag_va], pva_m[mag_va]))
    print(f"[magdir15m] VAL NY dir moved-AUC={val_dir_auc:.4f}  VAL NY magAUC={val_mag_auc:.4f}  "
          f"dir_best={Ld.best_iteration_} mag_best={Lm.best_iteration_} {time.time()-t0:.0f}s", flush=True)
    res["val_or_signal_auc"] = val_dir_auc
    res["val_mag_auc"] = val_mag_auc

    # freeze the predicted-magnitude quartile edges on VAL NY moved bars (so held-out buckets are causal)
    qedges = list(np.quantile(pva_m[dir_va], [0.25, 0.50, 0.75]))
    res["mag_quartile_edges_val"] = [float(q) for q in qedges]
    print(f"[magdir15m] FROZEN predicted-mag quartile edges (VAL NY moved): {[round(q,4) for q in qedges]}", flush=True)

    # free train arrays before held-out loop
    del Xtr, Xva, ytr, yva, mtr, mva, atr, ava, ymag_tr, ymag_va

    # ---- held-out per year: NY moved bars, bucket by predicted-magnitude quartile ----
    bucket_names = ["Q1_low", "Q2", "Q3", "Q4_high"]
    for w in ("test24", "test25", "oos"):
        Xw, yw, mw, tsw, aw = build_mag(SPL[w], 1)
        pdir = Ld.predict_proba(Xw)[:, 1]
        pmag = Lm.predict_proba(Xw)[:, 1]
        ny = ny_mask(tsw)
        nymoved = ny & mw

        ymag_w = (aw >= magthr).astype(int)
        mag_auc = float(roc_auc_score(ymag_w[ny], pmag[ny])) if ny.sum() > 50 else float("nan")
        # full-NY direction moved-AUC (the lever's signal as a DIRECTION predictor)
        auc_all = float(roc_auc_score(yw[nymoved], pdir[nymoved])) if nymoved.sum() > 50 else float("nan")
        up_rate = float(yw[nymoved].mean()) if nymoved.sum() > 0 else float("nan")
        # cov3% selective win-rate on ALL NY bars (decision rows = NY; ties charged as losses inside side_eval)
        conf_ny = np.abs(pdir[ny] - 0.5)
        thr3 = float(np.quantile(conf_ny, 0.97)) if ny.sum() > 0 else 1.0
        gate3 = side_eval(pdir[ny], yw[ny], mw[ny], tsw[ny], thr3)
        g3 = gate3["COMBINED"] if gate3 else {"n": 0, "wr": float("nan"), "ci": [float("nan")] * 2}

        # per predicted-magnitude quartile: direction moved-AUC + in-bucket cov-WR
        edges = [-np.inf] + list(qedges) + [np.inf]
        quart = {}
        for bi, bname in enumerate(bucket_names):
            lo, hi = edges[bi], edges[bi + 1]
            bmask = nymoved & (pmag > lo) & (pmag <= hi)
            if bmask.sum() > 50:
                bauc = float(roc_auc_score(yw[bmask], pdir[bmask]))
                # selective win-rate within this magnitude bucket: top-conf, nonoverlap, ties lose
                bm_ny = ny & (pmag > lo) & (pmag <= hi)
                conf_b = np.abs(pdir[bm_ny] - 0.5)
                cthr = float(np.quantile(conf_b, 0.70)) if bm_ny.sum() > 30 else 1.0
                bg = side_eval(pdir[bm_ny], yw[bm_ny], mw[bm_ny], tsw[bm_ny], cthr)
                bgc = bg["COMBINED"] if bg else {"n": 0, "wr": float("nan"), "ci": [float("nan")] * 2}
            else:
                bauc = float("nan"); bgc = {"n": 0, "wr": float("nan"), "ci": [float("nan")] * 2}
            quart[bname] = {"n_moved": int(bmask.sum()), "dir_moved_auc": bauc,
                            "sel_wr": bgc["wr"], "sel_n": bgc["n"], "sel_ci": bgc["ci"]}

        res["years"][w] = {
            "auc": auc_all,                          # full-NY direction moved-AUC (the lever signal)
            "moved_up_rate": up_rate,
            "tripwire_ok": bool(0.47 <= up_rate <= 0.53),
            "mag_auc": mag_auc,
            "cov3_wr": g3["wr"], "cov3_n": g3["n"], "cov3_ci": g3["ci"],
            "quartiles": quart,
        }
        print(f"=== {w} === NY-moved dir-AUC={auc_all:.4f} up-rate={up_rate:.4f} magAUC={mag_auc:.4f} "
              f"| cov3 COMB n{g3['n']} wr={g3['wr']:.4f} CI[{g3['ci'][0]:.3f},{g3['ci'][1]:.3f}]", flush=True)
        for bname in bucket_names:
            q = quart[bname]
            print(f"        {bname:8} dir-AUC={q['dir_moved_auc']:.4f} (n_moved {q['n_moved']}) | "
                  f"sel wr={q['sel_wr']:.4f} n{q['sel_n']} "
                  f"CI[{q['sel_ci'][0]:.3f},{q['sel_ci'][1]:.3f}]", flush=True)
        del Xw, yw, mw, tsw, aw, pdir, pmag

    # ---- apply FALSIFIER (sign-invariance test) ----
    AUC_BASE = 0.539; AUC_MARGIN = 0.010
    q4_beats = []          # years where Q4(high) AUC beats Q1(low)+margin AND >0.539
    q4_wr_clears = []      # years where Q4 cov-WR CI-lower clears BE
    for w in ("test24", "test25", "oos"):
        q = res["years"][w]["quartiles"]
        a1 = q["Q1_low"]["dir_moved_auc"]; a4 = q["Q4_high"]["dir_moved_auc"]
        if np.isfinite(a1) and np.isfinite(a4) and (a4 > a1 + AUC_MARGIN) and (a4 > AUC_BASE):
            q4_beats.append(w)
        ci_lo = q["Q4_high"]["sel_ci"][0]
        if np.isfinite(ci_lo) and ci_lo >= BE:
            q4_wr_clears.append(w)

    # also track the simpler "lever signal beats base AUC" (full-NY direction moved-AUC > .539) per year
    auc_beats = [w for w in ("test24", "test25", "oos")
                 if np.isfinite(res["years"][w]["auc"]) and res["years"][w]["auc"] > AUC_BASE]

    high_mag_more_predictable = len(q4_beats) >= 2
    wr_survives = len(q4_wr_clears) >= 2
    killed = not (high_mag_more_predictable or wr_survives)

    res["verdict"] = {
        "KILLED": bool(killed),
        "beats_base_auc": bool(len(auc_beats) >= 2),
        "q4_beats_q1_and_base_years": q4_beats,
        "q4_cov_wr_clears_BE_years": q4_wr_clears,
        "full_ny_auc_beats_base_years": auc_beats,
        "note": ("E2 sign-invariance test: HIGH-magnitude-quartile bars are %s more sign-predictable than "
                 "LOW-magnitude bars. %s. Expected NULL per theorem arXiv:2512.15720 (magnitude gates SIZE not "
                 "SIGN). SURVIVES only if Q4 dir-AUC beats Q1+0.010 AND >0.539 in >=2 yrs, OR Q4 cov-WR CI-lo "
                 "clears 0.541 in >=2 yrs.") % (
                    "" if (high_mag_more_predictable or wr_survives) else "NOT",
                    "FLAT direction-AUC across magnitude quartiles (null confirmed)" if killed
                    else "magnitude quartile DOES lift direction predictability"),
    }
    json.dump(res, open(RESULT, "w"), indent=2)
    print(f"\n[magdir15m] VERDICT: {'KILLED' if killed else 'SURVIVED'} "
          f"(q4_beats={q4_beats}; q4_wr_clears={q4_wr_clears}; auc_beats={auc_beats}) "
          f"-> {RESULT}  total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
