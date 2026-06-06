"""BAR-PATTERN CNN — step 4: does the bar-image CNN ADD to what we currently have?

The only 60s structure is the compression-release dip-buy REVERSION regime (the incumbent UP filter, OOS .613).
Question: gated to THAT regime, does the bar-image CNN's directional call beat (a) breakeven 0.541, (b) the 0.65
target, and (c) the pure-reversion baseline (the incumbent mechanism, CNN ignored)? Reuses the trained CNN
predictions (barcnn_pred_<variant>.npz); computes the 1-min-bar analog of the min1 regime, ties-strict, CPCV.

Policies on pooled held-out test+oos, within compression regime (bbw30<=train-q67 & rel>=train-q70):
  P1 = CNN direction, reversion-consistent gate (sign(p-0.5)==-sign(ret5)), selective by CNN confidence
  P2 = PURE REVERSION baseline (bet against ret5; CNN ignored) — the number P1 must beat to be a bar-pattern win
Run: ~/binary-algo-venv/bin/python barcnn_regime.py [variant=ohlc]
"""
import sys, json, numpy as np, pandas as pd
from barcnn_cpcv import cpcv_side, nonoverlap_chrono
ROOT = "/media/sean/CORSAIR/binary-algo"; OUT = f"{ROOT}/ohlc_cache"
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "ohlc"
BREAKEVEN, TARGET = 0.541, 0.65


def regime_arrays(sp):
    b = pd.read_parquet(f"{OUT}/EURUSD_1m_{sp}.parquet")
    c = b["close"]; r1 = c.pct_change()
    bbw30 = (r1.rolling(30).std() * np.sqrt(30)).values
    rel = ((r1.rolling(5).std() * np.sqrt(5)) / (pd.Series(bbw30, index=c.index) + 1e-12)).values
    ret5 = c.pct_change(5).values
    return b["t"].values.astype("int64"), bbw30, rel, ret5


def main():
    # train regime thresholds (valid moved bars)
    bt = pd.read_parquet(f"{OUT}/EURUSD_1m_train.parquet")
    ct = bt["close"]; r1 = ct.pct_change()
    bbw30_tr = (r1.rolling(30).std() * np.sqrt(30)).values
    rel_tr = ((r1.rolling(5).std() * np.sqrt(5)) / (pd.Series(bbw30_tr, index=ct.index) + 1e-12)).values
    vt = bt["valid"].values & (bt["mag"].values > 0)
    qb = float(np.nanpercentile(bbw30_tr[vt], 67))
    rq = float(np.nanpercentile(rel_tr[vt & (bbw30_tr <= qb)], 70))
    rqt = float(np.nanpercentile(rel_tr[vt & (bbw30_tr <= qb) & (rel_tr >= rq)], 80))
    print(f"[regime] train qb(bbw30,67%)={qb:.3e} rq(rel,70%)={rq:.3f} rqt(rel,80%)={rqt:.3f}", flush=True)

    # map regime onto the CNN predictions by ts
    d = np.load(f"{ROOT}/barcnn_pred_{VARIANT}.npz")
    reg = {}
    for sp in ("test", "oos"):
        t, bbw, rel, ret5 = regime_arrays(sp)
        reg[sp] = dict(zip(t.tolist(), zip(bbw.tolist(), rel.tolist(), ret5.tolist())))
    ts = np.concatenate([d["test_ts"], d["oos_ts"]]).astype("int64")
    y = np.concatenate([d["test_y"], d["oos_y"]]).astype("int64")
    p = np.concatenate([d["test_p"], d["oos_p"]]).astype("float32")
    src = ["test"] * len(d["test_ts"]) + ["oos"] * len(d["oos_ts"])
    bbw = np.array([reg[s].get(int(t), (np.nan,)*3)[0] for s, t in zip(src, ts)])
    rel = np.array([reg[s].get(int(t), (np.nan,)*3)[1] for s, t in zip(src, ts)])
    ret5 = np.array([reg[s].get(int(t), (np.nan,)*3)[2] for s, t in zip(src, ts)])
    o = np.argsort(ts); ts, y, p, bbw, rel, ret5 = ts[o], y[o], p[o], bbw[o], rel[o], ret5[o]

    comp = np.isfinite(bbw) & np.isfinite(rel) & np.isfinite(ret5) & (bbw <= qb) & (rel >= rqt)
    yr = pd.to_datetime(ts, unit="s", utc=True).year.values
    out = {"variant": VARIANT, "breakeven": BREAKEVEN, "target": TARGET, "regime": "compression(bbw30<=q67 & rel>=q80)",
           "n_in_regime": int(comp.sum()), "P1_cnn_reversion_gated": {}, "P2_pure_reversion_baseline": {}}

    # P2 — pure reversion (bet against ret5), no CNN; selective by |ret5| within regime
    pred2 = (ret5 < 0).astype(int)              # bet UP after a down 5-bar move
    conf2 = np.abs(ret5)
    # P1 — CNN direction, reversion-consistent; selective by CNN confidence within regime
    pred1 = (p > 0.5).astype(int); conf1 = np.abs(p - 0.5)
    revcons = np.sign(p - 0.5) == -np.sign(ret5)

    for cov in (0.10, 0.05, 0.02):
        thr2 = np.quantile(conf2[comp], 1 - cov)
        m2 = comp & (conf2 >= thr2); sel2 = nonoverlap_chrono(ts, m2)
        thr1 = np.quantile(conf1[comp & revcons], 1 - cov) if (comp & revcons).sum() > 50 else 1.0
        m1 = comp & revcons & (conf1 >= thr1); sel1 = nonoverlap_chrono(ts, m1)
        def yr_acc(sel, pred):
            r = {}
            for Y in (2024, 2025, 2026):
                s = sel[yr[sel] == Y]
                if len(s) < 20: r[str(Y)] = {"n": int(len(s)), "acc": None}; continue
                corr = (pred[s] == y[s]).astype(float)
                r[str(Y)] = {"n": int(len(s)), "acc": round(float(corr.mean()), 4)}
            return r
        # CPCV on each
        w1 = (pred1[sel1] == y[sel1]).astype(float); w2 = (pred2[sel2] == y[sel2]).astype(float)
        out["P1_cnn_reversion_gated"][f"cov{cov}"] = {"n": int(len(sel1)), "per_year": yr_acc(sel1, pred1),
            "cpcv": cpcv_side(ts[sel1], w1) if len(sel1) >= 60 else None}
        out["P2_pure_reversion_baseline"][f"cov{cov}"] = {"n": int(len(sel2)), "per_year": yr_acc(sel2, pred2),
            "cpcv": cpcv_side(ts[sel2], w2) if len(sel2) >= 60 else None}

    json.dump(out, open(f"{ROOT}/barcnn_regime_{VARIANT}_result.json", "w"), indent=1)
    print(f"\n=== REGIME-GATED [{VARIANT}] (in-regime n={int(comp.sum())}) ===", flush=True)
    for pol in ("P2_pure_reversion_baseline", "P1_cnn_reversion_gated"):
        print(f"-- {pol} --", flush=True)
        for cv, v in out[pol].items():
            py = {k: w["acc"] for k, w in v["per_year"].items()}
            p10 = v["cpcv"]["path_p10"] if v["cpcv"] else None
            clr = v["cpcv"]["frac_paths_clear_0.541"] if v["cpcv"] else None
            print(f"   {cv:8s} n={v['n']:>5} per-year={py} CPCV_p10={p10} frac_clear={clr}", flush=True)
    print(f"-> barcnn_regime_{VARIANT}_result.json", flush=True)


if __name__ == "__main__":
    main()
