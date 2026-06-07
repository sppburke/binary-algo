"""LEVER 2 RESOLUTION — is the decode_s1 embedding's small magnitude lift just TIME-OF-DAY?

Lever 2 (kronos_embed.py) found the 512-d decode_s1 embedding adds a small REAL magnitude lift over the rv baseline
[-pe,rv30,rv120] (+.005-.008 AUC, every CI95 excl 0). The embedding carries an additive TIME embedding (kronos.py:298),
so the hypothesis is the lift = intraday vol SEASONALITY the embedding re-derives — which the cheap §7 deseasonalized-RV
would capture without a Kronos dependency.

DECISIVE TEST (reuses kronos_embed's npz; no GPU, no re-extraction). 4-way paired CPCV ablation per horizon:
  base          = [-pe, rv30, rv120]
  base+tod      = base + time-of-day [hour, minute_of_day, dow, sin(t), cos(t)]   (UTC; captures London/NY/Asia vol clock)
  base+emb      = base + 512-d decode_s1 embedding                                 (the Lever 2 feature)
  base+tod+emb  = base + tod + emb
KEY metric: dAUC(base+tod+emb − base+tod) — does the embedding add ANYTHING beyond the clock?
  If ~0 (CI incl 0 or < +0.002): Lever 2's lift WAS time-of-day -> drop Kronos, use deseasonalized RV.
  If still clearly +: the embedding carries non-clock magnitude info worth the Kronos forward pass.
Also report dAUC(base+tod − base) (does clock help at all?) and the original dAUC(base+emb − base).

Run: ~/binary-algo-venv/bin/python kronos_embed_tod.py [src_tag=embed_main]
"""
import sys, os, json, math, itertools, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = "/media/sean/CORSAIR/binary-algo"
SRC = sys.argv[1] if len(sys.argv) > 1 else "embed_main"
NPZ = f"{ROOT}/kronos_embed_{SRC}.npz"; RES = f"{ROOT}/kronos_embed_tod_{SRC}_result.json"
RNG = np.random.default_rng(7); T0 = time.time()
def hb(m): print(f"HB[{time.time()-T0:.0f}s] {m}", flush=True)


def cpcv_groups(n, ng=8):
    e = np.linspace(0, n, ng + 1).astype(int); g = np.zeros(n, np.int8)
    for k in range(ng): g[e[k]:e[k + 1]] = k
    return g

def cpcv_paths(g, ts, horizon_min, ng=8, k=2):
    hs = horizon_min * 60; alli = np.arange(len(g))
    for combo in itertools.combinations(range(ng), k):
        tm = np.isin(g, combo); te = alli[tm]; tr = alli[~tm]
        keep = np.ones(len(tr), bool); tt = ts[tr]
        for grp in combo:
            gi = alli[g == grp]; lo, hi = ts[gi[0]], ts[gi[-1]]
            keep &= ~((tt >= lo - hs) & (tt <= hi + hs))
        yield combo, tr[keep], te

def mk_lgb(n=600):
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.03, num_leaves=255,
        min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10,
        n_estimators=n, n_jobs=20, verbosity=-1)

def boot_ci(x, nb=5000, seed=11):
    x = np.asarray(x, float); rng = np.random.default_rng(seed); n = len(x)
    a = np.array([x[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)

def tod_features(ts):
    dt = pd.to_datetime(ts, unit="s", utc=True)
    mod = (dt.hour * 60 + dt.minute).values.astype(np.float32)
    return np.column_stack([dt.hour.values.astype(np.float32), mod, dt.dayofweek.values.astype(np.float32),
                            np.sin(2 * np.pi * mod / 1440.0), np.cos(2 * np.pi * mod / 1440.0)]).astype(np.float32)


def main():
    if not os.path.exists(NPZ): hb(f"missing {NPZ}"); return
    d = np.load(NPZ); f = d["filled"]; ts = d["pick_t"][f]
    base3 = d["base3"][f]; emb = d["emb"][f]; horizons = list(d["horizons"])
    o = np.argsort(ts, kind="stable"); ts, base3, emb = ts[o], base3[o], emb[o]
    tod = tod_features(ts); n = len(ts)
    hb(f"{n} windows; emb_d={emb.shape[1]}; tod_d={tod.shape[1]}")
    from sklearn.metrics import roc_auc_score
    g = cpcv_groups(n, 8)
    Xbase = base3; Xtod = np.column_stack([base3, tod]); Xemb = np.column_stack([base3, emb])
    Xte = np.column_stack([base3, tod, emb])
    out = {"src": SRC, "n_windows": int(n), "note": "4-way: does decode_s1 embedding add beyond time-of-day?",
           "by_horizon": {}}
    for hk, H in enumerate(horizons):
        aret = d[f"aret_{H}"][f][o]; ok = np.isfinite(aret) & np.isfinite(emb).all(1)
        au_b, au_t, au_e, au_te = [], [], [], []
        for combo, tri, tei in cpcv_paths(g, ts, H):
            tr = tri[ok[tri]]; te = tei[ok[tei]]
            if len(tr) > 100_000: tr = np.sort(RNG.choice(tr, 100_000, replace=False))
            if len(tr) < 300 or len(te) < 150: continue
            thr = np.nanquantile(aret[tr], 0.75); ytr = (aret[tr] >= thr).astype(int); yte = (aret[te] >= thr).astype(int)
            if ytr.mean() in (0.0, 1.0) or yte.mean() in (0.0, 1.0): continue
            for X, acc in ((Xbase, au_b), (Xtod, au_t), (Xemb, au_e), (Xte, au_te)):
                m = mk_lgb(); m.fit(X[tr], ytr); acc.append(roc_auc_score(yte, m.predict_proba(X[te])[:, 1]))
        if not au_b: out["by_horizon"][str(H)] = {"note": "no valid paths"}; continue
        b, t, e, te2 = map(np.array, (au_b, au_t, au_e, au_te))
        d_tod = t - b; d_emb = e - b; d_emb_beyond_tod = te2 - t
        ci_beyond = boot_ci(d_emb_beyond_tod)
        verdict = ("EMB ADDS BEYOND CLOCK" if (d_emb_beyond_tod.mean() > 0.002 and ci_beyond[0] > 0)
                   else "LIFT IS TIME-OF-DAY (drop Kronos)")
        out["by_horizon"][str(H)] = dict(
            n_paths=len(b),
            auc_base=round(float(b.mean()), 4), auc_base_tod=round(float(t.mean()), 4),
            auc_base_emb=round(float(e.mean()), 4), auc_base_tod_emb=round(float(te2.mean()), 4),
            dAUC_tod_over_base=round(float(d_tod.mean()), 4), dAUC_tod_ci=list(boot_ci(d_tod)),
            dAUC_emb_over_base=round(float(d_emb.mean()), 4), dAUC_emb_ci=list(boot_ci(d_emb)),
            dAUC_emb_beyond_tod=round(float(d_emb_beyond_tod.mean()), 4), dAUC_emb_beyond_tod_ci=list(ci_beyond),
            VERDICT=verdict)
        hb(f"H={H}m base {b.mean():.4f} | +tod {t.mean():.4f}(d{d_tod.mean():+.4f}) | +emb {e.mean():.4f}(d{d_emb.mean():+.4f}) "
           f"| +tod+emb {te2.mean():.4f} | EMB BEYOND CLOCK d{d_emb_beyond_tod.mean():+.4f} CI{ci_beyond} -> {verdict}")
    json.dump(out, open(RES, "w"), indent=1); hb(f"DONE -> {RES}")


if __name__ == "__main__":
    main()
