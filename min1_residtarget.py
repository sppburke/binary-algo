"""EXPERIMENT #3 - RESIDUALIZED TARGET (highest remaining direction prior).

Never tried: the USD-basket residual was only ever a FEATURE; the LABEL stayed raw EURUSD sign.
Here the LABEL becomes label_resid = sign(eu_ret_H - beta_t*basket_ret_H), where:
  - eu_ret_H     = forward H-bar EURUSD log return (the thing we settle on, in raw terms)
  - basket_ret_H = forward H-bar USD-weakness basket return (EURUSD-equivalent direction)
  - beta_t       = CAUSAL rolling OLS slope of realized eu_ret_H on basket_ret_H using only
                   (eu,basket) forward-return pairs that FULLY RESOLVED strictly before t
                   (window ends at bar t-H so the most recent pair's outcome was observed by t).

We train the cross-pair LGBM (m5_xpair build_xp + augment) to predict label_resid, MX_HOR=15 first
(most tradeable), then MX_HOR=1. We evaluate, per-year 2024/2025/2026, deriv-faithful (non-overlap,
ties lose, mid-to-mid, CI95, NY gate, moved-bars-only):
  (a) residual-sign accuracy of the residual model;
  (b) AGREEMENT GATE: the UN-residualized raw EURUSD-sign accuracy on the subset where the residual
      model's directional vote AND the raw-direction model AGREE (a two-edge gate).

PRE-REGISTERED FALSIFIER (kill):
  residual-sign OOS < 0.55  OR  agreement-subset 2025 acc does not beat raw xpair 0.534 with CI clearing it.

Memory-safe: subsample train <=120k, one parquet at a time (build_xp does this), del big frames, small models.
A 15m CPCV runs concurrently; keep n_jobs modest.
"""
import sys, os, json, time, gc
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

import m5_xpair as MX
from m5_xpair import (FEAT, PAIRS, USD_BASE, NONEU, LB, SPL,
                      eu_equiv_sign, nonoverlap_chrono, boot, augment, xp_cols, feat_cols)

OUT = "/home/sean/git/binary-algo/min1_residtarget_result.json"
RAW_XPAIR_BASELINE = 0.534      # the binding 2025 raw-direction number to beat
BREAKEVEN = 0.541               # deriv mid-to-mid ties-lose breakeven
TRAIN_CAP = 120_000             # subsample cap per memory-safety
BETA_WIN = 500                  # rolling-OLS window (in resolved forward-return pairs)
BETA_MINP = 100                 # min pairs before a beta is trusted


# ------------------------------------------------------------------ causal beta + residual label
def _causal_beta(eu_fwd, bask_fwd, secs, H, win=BETA_WIN, minp=BETA_MINP):
    """Rolling OLS slope beta_t of eu_fwd on bask_fwd, using only pairs RESOLVED strictly before t.

    A forward pair anchored at bar j (eu_fwd[j], bask_fwd[j]) settles at bar j+H, i.e. at time secs[j]+H*60.
    For bar t we may only use pairs whose settlement time <= secs[t]. With contiguous minute bars the most
    recent admissible anchor is t-H. We therefore compute beta over anchors [t-H-win, t-H).
    Implemented with running sums shifted by H so there is ZERO look-ahead.
    """
    n = len(eu_fwd)
    x = bask_fwd.astype(np.float64); y = eu_fwd.astype(np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    xz = np.where(ok, x, 0.0); yz = np.where(ok, y, 0.0)
    o = ok.astype(np.float64)
    # prefix sums over ANCHORS (index = anchor bar). cumulative through anchor i inclusive.
    cN  = np.cumsum(o)
    cX  = np.cumsum(xz)
    cY  = np.cumsum(yz)
    cXX = np.cumsum(xz * xz)
    cXY = np.cumsum(xz * yz)
    def windowed(cum):
        # sum over anchors (hi-win, hi]  -> cum[hi] - cum[hi-win]; cum is 0-based inclusive prefix
        out = np.full(n, np.nan)
        return cum, out
    beta = np.full(n, np.nan)
    # admissible anchor upper bound for bar t is a_hi = t-H-1 (inclusive). window lower a_lo = a_hi-win+1.
    for t in range(n):
        a_hi = t - H - 1               # last anchor whose outcome resolved by secs[t]
        if a_hi < 0:
            continue
        a_lo = a_hi - win + 1
        # inclusive prefix at index i = cX[i]; sum over [a_lo, a_hi] = cX[a_hi] - (cX[a_lo-1] if a_lo>0 else 0)
        def seg(c):
            hi = c[a_hi]
            lo = c[a_lo - 1] if a_lo > 0 else 0.0
            return hi - lo
        N = seg(cN)
        if N < minp:
            continue
        Sx = seg(cX); Sy = seg(cY); Sxx = seg(cXX); Sxy = seg(cXY)
        denom = N * Sxx - Sx * Sx
        if denom <= 1e-18:
            continue
        beta[t] = (N * Sxy - Sx * Sy) / denom
    return beta


def build_resid(years, H, stride=1):
    """Like m5_xpair.build_xp but ALSO compute forward basket return, causal beta, and the residual label.

    Returns the same XP frame with extra cols:
       _fwd      raw forward H-bar EURUSD log return (already in build_xp)
       _fwdbask  forward H-bar USD-weakness basket return (EURUSD-equiv)
       _beta     causal rolling OLS slope (eu_fwd ~ bask_fwd)
       _yresid   residual-sign label = (eu_fwd - beta*bask_fwd) > 0      [the NEW label]
       _y        raw EURUSD-sign label (unchanged, for the agreement gate)
    Rows kept: build_xp's moved-bars-only AND finite beta AND non-zero residual (ties lose).
    """
    parts = []
    for y in years:
        # build ONE year at a time (memory safety): reuse m5_xpair.build_xp for a single year
        F = MX.build_xp([y], stride=1)          # stride applied after we attach labels
        if F is None or len(F) == 0:
            continue
        secs = F["_ts"].values.astype("int64")
        eu_fwd = F["_fwd"].values.astype(np.float64)   # forward H-bar EURUSD return (moved-only already)
        # forward basket return aligned to the SAME anchors. build_xp dropped invalid rows, so recompute
        # the forward basket directly from the per-pair closes for THIS year, then align by timestamp.
        fwdbask = _forward_basket(y, H, secs)
        beta = _causal_beta(eu_fwd, fwdbask, secs, H)
        resid = eu_fwd - beta * fwdbask
        F = F.assign(_fwdbask=fwdbask, _beta=beta, _resid=resid,
                     _yresid=(resid > 0).astype(float))
        keep = np.isfinite(beta) & np.isfinite(fwdbask) & (resid != 0.0)
        F = F.loc[keep]
        if stride > 1:
            F = F.iloc[::stride]
        parts.append(F)
        del F; gc.collect()
    if not parts:
        return None
    return pd.concat(parts)


def _forward_basket(year, H, want_secs):
    """Forward H-bar USD-weakness basket return for `year`, returned aligned to `want_secs` (the XP anchors).

    Basket = mean over NONEU pairs of eu_equiv_sign(p)*(lr[t+H]-lr[t]), contiguity-checked (exactly H*60s).
    Built from the same close parquets m5_xpair uses, intersection-joined on timestamp (NOT ffilled)."""
    cl = {}
    for p in PAIRS:
        d = pd.read_parquet(f"{FEAT}/{p}_{year}.parquet", columns=["close"])
        d = d[~d.index.duplicated(keep="last")]
        cl[p] = d["close"]
    df = pd.DataFrame(cl).dropna()                 # intersection of the 7 pairs (matches build_xp)
    idx = df.index
    secs = idx.values.astype("datetime64[s]").astype("int64")
    n = len(df)
    lr = {p: np.log(df[p].values) for p in PAIRS}
    contig = np.zeros(n, dtype=bool)
    if n > H:
        contig[:n - H] = (secs[H:] - secs[:-H]) == H * 60
    bask = np.full(n, np.nan)
    cols = []
    for p in NONEU:
        fr = np.full(n, np.nan)
        if n > H:
            fr[:n - H] = lr[p][H:] - lr[p][:-H]
        cols.append(eu_equiv_sign(p) * fr)
    bm = np.nanmean(np.vstack(cols), axis=0)
    bask = np.where(contig, bm, np.nan)
    s = pd.Series(bask, index=secs)
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(want_secs).values


# ------------------------------------------------------------------ evaluation helpers
def _sel_acc(pred_up, y, ts, gate, thr, conf):
    """non-overlap selective accuracy with NY gate + confidence thr. y is the relevant 0/1 label."""
    m = gate & (conf >= thr)
    sel = nonoverlap_chrono(ts, m, gap=H_GAP)
    if len(sel) < 20:
        return len(sel), float("nan"), (float("nan"), float("nan")), np.array([])
    corr = (pred_up[sel].astype(int) == y[sel].astype(int)).astype(float)
    lo, hi = boot(corr)
    return len(sel), float(corr.mean()), (lo, hi), corr


def run(H, mode="xp", stride_train=8):
    global H_GAP
    H_GAP = H * 60
    os.environ["MX_HOR"] = str(H)
    # reload m5_xpair module-level HOR/GAP_S so build_xp uses this horizon
    MX.HOR = H; MX.GAP_S = H * 60
    import importlib
    # build_xp reads MX.HOR via the module global captured at def-time? It reads HOR at call time (global). ok.
    t0 = time.time()
    print(f"\n######## H={H} mode={mode} ########", flush=True)

    TR = build_resid(SPL["train"], H, stride=stride_train)
    if TR is None:
        return {"error": "no train"}
    if len(TR) > TRAIN_CAP:
        TR = TR.sample(n=TRAIN_CAP, random_state=7).sort_index()
    VA = build_resid(SPL["val"], H, stride=2)
    # CRITICAL: strip ALL label/target-derived underscore cols so they NEVER enter the feature set.
    # m5_xpair.xp_cols only drops {_y,_ts,_fwd,hour}; build_resid added _fwdbask,_beta,_resid,_yresid
    # which are deterministic functions of the FORWARD return -> pure leakage if used as features.
    xpc = [c for c in xp_cols(TR) if not c.startswith("_")]
    TR = augment(TR, SPL["train"], mode); VA = augment(VA, SPL["val"], mode)
    cols = feat_cols(mode, TR, xpc)
    print(f"[H{H}] train={len(TR):,} val={len(VA):,} feats={len(cols)} "
          f"resid-uprate(train)={TR['_yresid'].mean():.4f} raw-uprate(train)={TR['_y'].mean():.4f} "
          f"build={time.time()-t0:.0f}s", flush=True)

    # ----- model A: residual-sign model (the NEW label) -----
    yA = TR["_yresid"].astype(int).values
    XtrA = TR[cols].astype("float32"); XvaA = VA[cols].astype("float32")
    yvaA = VA["_yresid"].astype(int).values
    LA = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=63,
                            min_child_samples=400, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                            reg_lambda=20, n_estimators=2500, n_jobs=10, verbosity=-1)
    LA.fit(XtrA, yA, eval_set=[(XvaA, yvaA)], eval_metric="auc",
           callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)])
    pvaA = LA.predict_proba(XvaA)[:, 1]
    aucvA = roc_auc_score(yvaA, pvaA)

    # ----- model B: raw EURUSD-sign model (for the agreement gate) -----
    yB = TR["_y"].astype(int).values
    yvaB = VA["_y"].astype(int).values
    LB_ = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=63,
                             min_child_samples=400, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                             reg_lambda=20, n_estimators=2500, n_jobs=10, verbosity=-1)
    LB_.fit(XtrA, yB, eval_set=[(XvaA, yvaB)], eval_metric="auc",
            callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)])
    pvaB = LB_.predict_proba(XvaA)[:, 1]
    aucvB = roc_auc_score(yvaB, pvaB)
    print(f"[H{H}] VAL AUC resid={aucvA:.4f} raw={aucvB:.4f} best_iter A={LA.best_iteration_} B={LB_.best_iteration_}", flush=True)

    del TR, XtrA; gc.collect()

    # ----- freeze operating point on VAL by WORST-VAL-HALF (per discipline; never VAL-acc-max) -----
    tsv = VA["_ts"].values.astype("int64"); gny = VA["sess_ny"].values > 0.5
    confA = np.abs(pvaA - 0.5)
    # agreement: both models point the SAME way; gate confidence = min(|pA-.5|,|pB-.5|)
    dirA = (pvaA > 0.5); dirB = (pvaB > 0.5)
    agree_v = (dirA == dirB)
    confAGR = np.minimum(np.abs(pvaA - 0.5), np.abs(pvaB - 0.5))

    def worst_half_thr(conf, ylabel, pred_up, target_cov_list=(0.10, 0.05, 0.03, 0.02), extra_mask=None):
        """Choose thr maximizing the WORSE of two VAL time-halves' selective acc (constrained n)."""
        mid = np.median(tsv)
        h1 = tsv < mid; h2 = ~h1
        best = None
        base = gny.copy()
        if extra_mask is not None:
            base = base & extra_mask
        for cov in target_cov_list:
            pool = conf[base]
            if pool.size < 500:
                continue
            thr = float(np.quantile(pool, 1 - cov))
            accs = []
            for hmask in (h1, h2):
                m = base & hmask & (conf >= thr)
                sel = nonoverlap_chrono(tsv, m, gap=H_GAP)
                if len(sel) < 60:
                    accs = None; break
                accs.append((pred_up[sel].astype(int) == ylabel[sel].astype(int)).mean())
            if accs is None:
                continue
            worst = min(accs)
            if best is None or worst > best[0]:
                # full-VAL acc at this thr for reporting
                m = base & (conf >= thr); sel = nonoverlap_chrono(tsv, m, gap=H_GAP)
                fullacc = (pred_up[sel].astype(int) == ylabel[sel].astype(int)).mean() if len(sel) else float("nan")
                best = (worst, cov, thr, len(sel), fullacc)
        return best

    frA = worst_half_thr(confA, yvaA, dirA)
    frAGR = worst_half_thr(confAGR, yvaB, dirB, extra_mask=agree_v)   # gate B's RAW sign on agreement subset
    print(f"[H{H}] FROZEN resid: {frA}   agree-gate(rawsign): {frAGR}", flush=True)

    # ----- per-year held-out evaluation -----
    res = {"H": H, "mode": mode, "val_auc_resid": float(aucvA), "val_auc_raw": float(aucvB),
           "train_resid_uprate": float(yA.mean()), "raw_xpair_baseline": RAW_XPAIR_BASELINE,
           "breakeven": BREAKEVEN, "frozen_resid": None, "frozen_agree": None, "years": {}}
    if frA is not None:
        res["frozen_resid"] = {"worsthalf": frA[0], "cov": frA[1], "thr": frA[2], "n": frA[3], "val_fullacc": frA[4]}
    if frAGR is not None:
        res["frozen_agree"] = {"worsthalf": frAGR[0], "cov": frAGR[1], "thr": frAGR[2], "n": frAGR[3], "val_fullacc": frAGR[4]}

    thrA = frA[2] if frA is not None else 0.0
    thrAGR = frAGR[2] if frAGR is not None else 0.0

    YEARMAP = {"2024": ["2024"], "2025": ["2025"], "2026": ["2026"]}
    for yr, yrs in YEARMAP.items():
        D = build_resid(yrs, H, stride=1)
        if D is None or len(D) == 0:
            res["years"][yr] = {"error": "empty"}; continue
        XD = D[cols].astype("float32")
        prA = LA.predict_proba(XD)[:, 1]
        prB = LB_.predict_proba(XD)[:, 1]
        yResid = D["_yresid"].astype(int).values
        yRaw = D["_y"].astype(int).values
        ts = D["_ts"].values.astype("int64"); g = D["sess_ny"].values > 0.5
        # moved-bars sanity: raw EURUSD up-rate on this window (should be ~0.49, no ffill mirage)
        uprate = float(yRaw.mean())

        confA_ = np.abs(prA - 0.5)
        # (a) residual-sign accuracy at frozen resid thr
        nA, accA, ciA, _ = _sel_acc((prA > 0.5), yResid, ts, g, thrA, confA_)
        # full-coverage residual AUC + raw AUC
        aucA = roc_auc_score(yResid, prA) if len(np.unique(yResid)) > 1 else float("nan")
        aucB = roc_auc_score(yRaw, prB) if len(np.unique(yRaw)) > 1 else float("nan")

        # (b) AGREEMENT GATE: raw EURUSD-sign acc where dirA==dirB, gate-conf=min(|pA-.5|,|pB-.5|)
        dA = (prA > 0.5); dB = (prB > 0.5)
        agree = (dA == dB)
        confAGR_ = np.minimum(np.abs(prA - 0.5), np.abs(prB - 0.5))
        nAGR, accAGR, ciAGR, _ = _sel_acc(dB.astype(int), yRaw, ts, g & agree, thrAGR, confAGR_)
        # also a NO-conf-threshold agreement view (all agreeing NY moved bars, non-overlap)
        nAGR0, accAGR0, ciAGR0, _ = _sel_acc(dB.astype(int), yRaw, ts, g & agree, 0.0, confAGR_)

        res["years"][yr] = {
            "n_rows": int(len(D)), "raw_uprate_moved": uprate,
            "auc_resid": float(aucA), "auc_raw": float(aucB),
            "resid_sign": {"n": nA, "acc": accA, "ci95": [ciA[0], ciA[1]]},
            "agree_gate_rawsign": {"n": nAGR, "acc": accAGR, "ci95": [ciAGR[0], ciAGR[1]]},
            "agree_gate_noconf": {"n": nAGR0, "acc": accAGR0, "ci95": [ciAGR0[0], ciAGR0[1]]},
        }
        print(f"=== {yr} === uprate={uprate:.4f} AUCresid={aucA:.4f} AUCraw={aucB:.4f} | "
              f"resid_sign n{nA} acc={accA:.3f} CI[{ciA[0]:.3f},{ciA[1]:.3f}] | "
              f"agree_rawsign n{nAGR} acc={accAGR:.3f} CI[{ciAGR[0]:.3f},{ciAGR[1]:.3f}] | "
              f"agree_noconf n{nAGR0} acc={accAGR0:.3f} CI[{ciAGR0[0]:.3f},{ciAGR0[1]:.3f}]", flush=True)
        del D, XD; gc.collect()

    # ----- falsifier verdict -----
    r25 = res["years"].get("2025", {})
    r26 = res["years"].get("2026", {})
    resid_oos = r26.get("resid_sign", {}).get("acc", float("nan"))   # 2026 = OOS
    agr25 = r25.get("agree_gate_rawsign", {})
    agr25_acc = agr25.get("acc", float("nan"))
    agr25_lo = agr25.get("ci95", [float("nan"), float("nan")])[0]
    kill_resid = (not np.isfinite(resid_oos)) or (resid_oos < 0.55)
    kill_agree = (not np.isfinite(agr25_acc)) or (agr25_acc <= RAW_XPAIR_BASELINE) or \
                 (not np.isfinite(agr25_lo)) or (agr25_lo <= RAW_XPAIR_BASELINE)
    killed = kill_resid or kill_agree
    res["falsifier"] = {
        "rule": "kill if resid-sign OOS(2026)<0.55 OR agree-subset 2025 acc<=0.534 or CI lo<=0.534",
        "resid_oos_2026": resid_oos, "kill_resid_branch": bool(kill_resid),
        "agree_2025_acc": agr25_acc, "agree_2025_ci_lo": agr25_lo, "kill_agree_branch": bool(kill_agree),
        "KILLED": bool(killed),
    }
    print(f"[H{H}] FALSIFIER: resid_oos={resid_oos:.4f} (kill={kill_resid}) | "
          f"agree25={agr25_acc:.4f} ci_lo={agr25_lo:.4f} vs {RAW_XPAIR_BASELINE} (kill={kill_agree}) | "
          f"KILLED={killed}", flush=True)
    del LA, LB_, VA; gc.collect()
    return res


def main():
    results = {"experiment": "min1_residtarget", "design": "label=sign(eu_fwd - causal_beta*basket_fwd)",
               "beta_win": BETA_WIN, "beta_minp": BETA_MINP, "train_cap": TRAIN_CAP,
               "discipline": "nonoverlap, ties-lose, NY gate, moved-only, worst-VAL-half select, CI95",
               "runs": {}}
    # MX_HOR=15 FIRST (most tradeable), then MX_HOR=1
    for H, stride in ((15, 6), (1, 30)):
        try:
            results["runs"][f"H{H}"] = run(H, mode="xp", stride_train=stride)
        except Exception as e:
            import traceback; traceback.print_exc()
            results["runs"][f"H{H}"] = {"error": str(e)}
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"[saved] {OUT}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
