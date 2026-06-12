"""IMPROVE / COMBINE @15m: cross-pair USD-residual signal (the 5m edge SOURCE, m5xp) retargeted to MX_HOR=15,
blended with the certified frozen base book. Genuinely distinct from D7 (exp_15m_v3 was exog-FEATURES added to
the base; this is the cross-pair-NATIVE primary). REASONED mechanism: base book = EURUSD's own MTF features;
cross-pair = USD-basket residual + order-flow → potentially DECORRELATED errors → a blend could lift the floor.
GOAL discipline: MEASURE the prediction correlation FIRST; only blend if decorrelated.

Trains the cross-pair primary in-memory (NO save — must not clobber the frozen 5m m5xp_* artifacts). Aligns
book and cross-pair predictions by timestamp, applies the BOOK's comp(bb_width)xNY gate, side-splits
book-alone / xpair-alone / 0.5-blend at fixed cov{0.05,0.10}. Compares binding-2025 sides to the certified
floor (UP .5475 / DOWN .5486). Breakeven 0.541.

Usage: MX_HOR=15 python m15_xpair_blend.py   (the script sets MX_HOR=15 itself)
"""
import os
os.environ["MX_HOR"] = "15"   # MUST precede the m5_xpair import (HOR is read at import)
import json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import m5_xpair as MX
import m15_production as M15

BE = 0.541
COVS = (0.05, 0.10)
YEARS = (("test24", "2024"), ("test25", "2025"), ("oos", "2026"))


def train_xpair(stride=4, sub=150_000):
    TR = MX.build_xp(MX.SPL["train"], stride); VA = MX.build_xp(MX.SPL["val"])
    xpc = MX.xp_cols(TR)
    TR = MX.augment(TR, MX.SPL["train"], "xpof"); VA = MX.augment(VA, MX.SPL["val"], "xpof")
    cols = MX.feat_cols("xpof", TR, xpc)
    if len(TR) > sub:
        TR = TR.iloc[np.linspace(0, len(TR) - 1, sub).astype(int)]
    ytr = TR["_y"].astype(int).values
    P = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=127, min_child_samples=400,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=20, n_estimators=3000, n_jobs=20, verbosity=-1)
    yva = VA["_y"].astype(int).values
    P.fit(TR[cols].astype("float32"), ytr, eval_set=[(VA[cols].astype("float32"), yva)], eval_metric="auc",
          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    print(f"[xpair15] primary VAL AUC={roc_auc_score(yva, P.predict_proba(VA[cols].astype('float32'))[:,1]):.4f}", flush=True)
    return P, cols


def book_pred_year(p, L, G, C, year):
    """frozen m15 book predictions on `year`, returned aligned by ts with gate fields + label."""
    D = M15.load([year])
    pr = M15._dirproba(p, L, G, C, D)
    ts = D.index.values.astype("datetime64[s]").astype("int64")
    out = pd.DataFrame({"ts": ts, "book_pr": pr, "y": D["_y"].astype(int).values,
                        "bbw": D["15m_bb_width"].values.astype(float), "ny": D["sess_ny"].values.astype(float) > 0.5})
    del D
    return out


def side_split(p, y, ts, gate, cthr):
    m = gate & (np.abs(p - 0.5) >= cthr)
    sel = MX.nonoverlap_chrono(ts, m)
    pred = (p[sel] > 0.5).astype(int); yy = y[sel]
    out = {}
    for side, nm in ((1, "UP"), (0, "DOWN")):
        ss = pred == side
        if ss.sum() < 5:
            out[nm] = {"n": int(ss.sum()), "acc": None}; continue
        corr = (pred[ss] == yy[ss]).astype(float); lo, hi = MX.boot(corr)
        out[nm] = {"n": int(ss.sum()), "acc": round(float(corr.mean()), 4), "ci_lo": round(lo, 4),
                   "up_rate": round(float(yy[ss].mean()), 4)}
    return out


def main():
    t0 = time.time()
    P, cols = train_xpair()
    print(f"[xpair15] trained {time.time()-t0:.0f}s", flush=True)
    bp = M15._load()  # p,L,G,C
    bthr = bp[0]["bb_width_thr"]

    # VAL: align to pick conf thresholds on the BLEND/each signal
    VAxp = MX.augment(MX.build_xp(MX.SPL["val"]), MX.SPL["val"], "xpof")
    xpv = P.predict_proba(VAxp[cols].astype("float32"))[:, 1]; tsv_xp = VAxp["_ts"].values.astype("int64")
    bv = book_pred_year(bp[0], bp[1], bp[2], bp[3], "2022")
    bv2 = book_pred_year(bp[0], bp[1], bp[2], bp[3], "2023")
    bv = pd.concat([bv, bv2], ignore_index=True)
    # align VAL book<->xpair by ts
    xpmap_v = dict(zip(tsv_xp, xpv))
    bv["xp_pr"] = bv["ts"].map(xpmap_v)
    bv = bv.dropna(subset=["xp_pr"])
    gate_v = bv["bbw"].values <= bthr; gate_v &= bv["ny"].values
    corr_all = float(np.corrcoef(bv["book_pr"].values, bv["xp_pr"].values)[0, 1])
    corr_gate = float(np.corrcoef(bv["book_pr"].values[gate_v], bv["xp_pr"].values[gate_v])[0, 1])
    print(f"[xpair15] CORR(book_pr, xpair_pr) all={corr_all:.3f} in-gate={corr_gate:.3f} (decorrelated if <~0.5)", flush=True)

    signals = {"book": lambda b, x: b, "xpair": lambda b, x: x, "blend": lambda b, x: 0.5 * b + 0.5 * x}
    # conf thresholds per signal at fixed cov, chosen on VAL in-gate
    thrs = {}
    for sn, f in signals.items():
        s = f(bv["book_pr"].values, bv["xp_pr"].values); conf = np.abs(s - 0.5)
        thrs[sn] = {cov: float(np.quantile(conf[gate_v], 1 - cov)) for cov in COVS}

    res = {"MX_HOR": 15, "certified_floor": {"UP": 0.5475, "DOWN": 0.5486}, "breakeven": BE,
           "corr_book_xpair": {"all": corr_all, "in_gate": corr_gate},
           "xpair_val_auc_note": "see log", "signals": {sn: {} for sn in signals}}
    # test years
    yr_book = {}
    yr_xp = {}
    for w, yr in YEARS:
        b = book_pred_year(bp[0], bp[1], bp[2], bp[3], yr)
        Dxp = MX.augment(MX.build_xp([yr]), [yr], "xpof")
        xp = P.predict_proba(Dxp[cols].astype("float32"))[:, 1]; tsx = Dxp["_ts"].values.astype("int64")
        b["xp_pr"] = b["ts"].map(dict(zip(tsx, xp)))
        b = b.dropna(subset=["xp_pr"])
        yr_book[yr] = b
        del Dxp

    print("\n=== forward side-split @cov0.10 (binding 2025) — book vs xpair vs blend ===", flush=True)
    for sn, f in signals.items():
        for cov in COVS:
            per = {}
            for w, yr in YEARS:
                b = yr_book[yr]
                s = f(b["book_pr"].values, b["xp_pr"].values)
                gate = (b["bbw"].values <= bthr) & b["ny"].values
                per[yr] = side_split(s, b["y"].values, b["ts"].values, gate, thrs[sn][cov])
            res["signals"][sn][f"cov{cov}"] = per
        r = res["signals"][sn]["cov0.1"]
        def c(yr, sd):
            v = r[yr][sd]; return f"{v['acc']:.3f}(n{v['n']})" if v.get("acc") else "NA"
        print(f"  {sn:6} cov0.10: 2024 U{c('2024','UP')} D{c('2024','DOWN')} | "
              f"2025 U{c('2025','UP')} D{c('2025','DOWN')} | 2026 U{c('2026','UP')} D{c('2026','DOWN')}", flush=True)
    json.dump(res, open("m15_xpair_blend_result.json", "w"), indent=1)
    print(f"\n[xpair15] floors UP .5475 DOWN .5486 | frozen-fwd UP .725/.607/.674 DOWN .653/.559/.650", flush=True)
    print(f"[xpair15] -> m15_xpair_blend_result.json ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
