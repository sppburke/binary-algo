"""SIDE-SPLIT of the FROZEN 5m books -> the measured (EURUSD, 5m, UP) and (5m, DOWN) keys.
Sweep ledger EURUSD_5m rows 0a (m5xp) and 0b (m5stack). Inference-only (no training): replicate each book's
backtest gating, take the independent (nonoverlap_chrono 300s) trades, split by PREDICTED direction, score per
year (2024/2025/2026 = the test24/test25/oos windows) moved-bars-only with bootstrap CI95. Deriv-faithful:
the book's own settlement (wall-clock-contiguous 300s close-to-close, label _y=(fwd>0), tie bars dropped at
build). Mirrors min1_updown (60s) / min2_updown (120s). Breakeven 0.541.

Usage: python m5_updown.py [m5xp|m5stack|both]
"""
import sys, json, numpy as np
import m5_xpair as MX

YEARS = (("test24", "2024"), ("test25", "2025"), ("oos", "2026"))


def side_eval(per_year_trades):
    """per_year_trades: {label: (pred, y, fwd)} -> nested per-year per-side stats (moved-bars-only)."""
    out = {}
    for label, (pred, y, fwd) in per_year_trades.items():
        for side, name in ((1, "UP"), (0, "DOWN"), (None, "COMBINED")):
            sel = (pred == side) if side is not None else np.ones(len(pred), bool)
            moved = np.abs(fwd[sel]) > 0  # _y already moved-only, so this is all-true; kept for discipline parity
            if moved.sum() < 5:
                out[f"{label}_{name}"] = {"n": int(sel.sum()), "n_moved": int(moved.sum()), "acc": None,
                                          "ci": [None, None], "out_up_rate": None}
                continue
            corr = (pred[sel][moved] == y[sel][moved]).astype(float)
            lo, hi = MX.boot(corr)
            out[f"{label}_{name}"] = {"n": int(sel.sum()), "n_moved": int(moved.sum()),
                                      "acc": float(corr.mean()), "ci": [lo, hi],
                                      "out_up_rate": float(y[sel][moved].mean())}
    return out


def split_m5xp():
    import m5_xpair_production as XP
    p, P, M = XP._load()
    cols = p["primary_feats"]; mcols = p["meta_feats"]; THR = p["meta_thr"]
    per_year = {}
    for w, label in YEARS:
        D = MX.build_xp(XP.SPL[w]); D = MX.augment(D, XP.SPL[w], XP.MODE)
        pr = P.predict(D[cols].astype("float32")); y = D["_y"].astype(int).values
        fwd = D["_fwd"].values.astype(float)
        sm = M.predict(XP._Xmeta(D, pr, mcols)); ts = D["_ts"].values.astype("int64")
        ny = D["sess_ny"].values > 0.5
        m = ny & (sm >= THR); sel = MX.nonoverlap_chrono(ts, m)
        per_year[label] = ((pr[sel] > 0.5).astype(int), y[sel], fwd[sel])
        del D
    return per_year, {"book": "EURUSD.m5xp.v1", "gate": "sess_ny & meta>=%.4f" % THR}


def split_m5stack():
    """Side-split the cross-horizon stack: bet = dir15 (15m ensemble sign on the 5-min outcome) when
    sess_ny & stack_meta>=thr. Replicates m5_stack2 inference at the frozen q0.98 threshold.
    fwd is all-moved by construction (build_xp drops fwd==0), so a ones placeholder is faithful for the
    moved-mask; out_up_rate is computed from y (the realized 5-min sign), not fwd."""
    import lightgbm as lgb, json as _j
    MODELS = "/home/sean/git/binary-algo/models"
    sp = _j.load(open(f"{MODELS}/m5stack_EURUSD_strategy.json"))
    THR = sp["meta_thr"]
    M = lgb.Booster(model_file=f"{MODELS}/m5stack_EURUSD_meta_lgb.txt")
    import m5_stack2 as S2
    _p15, L15, G15, C15 = S2.load15(); s5, P5 = S2.load5()
    per_year = {}
    for w, label in YEARS:
        W = S2.features(w, L15, G15, C15, s5, P5)
        ts = W["ts"]; ny = W["ny"]; y = W["y"]; dir15 = W["dir15"]; Xm = W["Xm"]
        sm = M.predict(Xm)
        m = ny & (sm >= THR); sel = MX.nonoverlap_chrono(ts, m, 300)
        per_year[label] = (dir15[sel].astype(int), y[sel].astype(int), np.ones(len(sel)))
    return per_year, {"book": "EURUSD.m5stack.v1", "gate": "sess_ny & stack_meta>=%.4f" % THR}


def run(which):
    splitter = {"m5xp": split_m5xp, "m5stack": split_m5stack}[which]
    per_year, meta = splitter()
    allpred = np.concatenate([per_year[l][0] for _, l in YEARS])
    allfwd = np.concatenate([per_year[l][2] for _, l in YEARS])
    ally = np.concatenate([per_year[l][1] for _, l in YEARS])
    upr = float(ally[np.abs(allfwd) > 0].mean())
    print(f"[{which}] indep trades={len(allpred)} moved up-rate(all)={upr:.4f} pred-up share={allpred.mean():.3f}", flush=True)
    res = side_eval(per_year)
    for k, v in res.items():
        print(f"  {k:16} n={v['n']:>4} moved={v['n_moved']:>4} acc={v['acc']} CI={v['ci']} out_up_rate={v['out_up_rate']}", flush=True)
    return {"book": meta["book"], "horizon_s": 300, "breakeven": 0.541, "gate": meta["gate"],
            "settlement": "book-native: wall-clock-contiguous 300s close-to-close, _y=(fwd>0), ties dropped at build; nonoverlap_chrono 300s; moved-only; per-year CI95",
            "moved_up_rate_all": upr, "pred_up_share": float(allpred.mean()), "per_year_side": res}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    targets = ["m5xp", "m5stack"] if which == "both" else [which]
    import os
    out = {}
    if os.path.exists("m5_updown_result.json"):
        try:
            out = json.load(open("m5_updown_result.json"))
        except Exception:
            out = {}
    for t in targets:
        out[t] = run(t)
    json.dump(out, open("m5_updown_result.json", "w"), indent=1)
    print("[m5_updown] -> m5_updown_result.json", flush=True)


if __name__ == "__main__":
    main()
