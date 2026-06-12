"""Q1 (last on-disk row): magnitude x direction gate @15m. Does conditioning the certified direction trades on
predicted move-SIZE lift the per-side hit-rate? Sign-invariance theorem (arXiv:2512.15720) + the 5m magspot null
+ the 15m coverage curve (edge tracks CONFIDENCE not magnitude) all predict NULL — but the backlog queued it to
CONFIRM at 15m (move composition differs). FAST-KILL sized to the ~.10 prior.

Trains a 15m magnitude model (target |ret_15m| >= train-median |ret|; leakage-safe feats<=t), buckets the frozen
cross-pair book's gated direction trades into LOW/HIGH predicted-magnitude, and reports per-side hit-rate per
bucket per year. Falsifier: KILL unless some magnitude bucket lifts a side's BINDING-year hit-rate CI95-lo above
the un-bucketed cross-pair incumbent by >1 SE, up-rate in [.47,.53]. Breakeven 0.541.

Usage: python m15_magdir.py
"""
import os
os.environ["MX_HOR"] = "15"
import json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
import m5_xpair as MX

HOR = 15
base = list(H.feature_cols("EURUSD"))
SUB = 150_000
BE = 0.541
MODELS = "/home/sean/git/binary-algo/models"


def load_mag(years, stride=1):
    parts = []
    for y in years:
        p = f"{H.FEAT_DIR}/EURUSD_{y}.parquet"
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p, columns=base + H.META_COLS); df = df[~df.index.duplicated(keep="last")]
        c = df["close"].values; n = len(c)
        secs = df.index.values.astype("datetime64[s]").astype("int64")
        contig = np.zeros(n, bool)
        if n > HOR:
            contig[:n - HOR] = (secs[HOR:] - secs[:-HOR]) == HOR * 60
        fwd = np.full(n, np.nan); fwd[:n - HOR] = c[HOR:]; ret = fwd / c - 1.0
        valid = contig & np.isfinite(ret) & (ret != 0)
        d = df.loc[valid, base].copy(); d["_absr"] = np.abs(ret[valid]); d["_ts"] = secs[valid]
        parts.append(d.iloc[::stride] if stride > 1 else d)
    return pd.concat(parts)


def main():
    t0 = time.time()
    TR = load_mag([str(y) for y in range(2012, 2022)], stride=3)
    med = float(np.median(TR["_absr"].values))
    if len(TR) > SUB:
        TR = TR.iloc[np.linspace(0, len(TR) - 1, SUB).astype(int)]
    ymag = (TR["_absr"].values >= med).astype(int)
    Mg = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.03, num_leaves=127, min_child_samples=400,
                            subsample=0.8, colsample_bytree=0.5, reg_lambda=20, n_estimators=600, n_jobs=20, verbosity=-1)
    Mg.fit(TR[base].astype("float32").values, ymag)
    print(f"[magdir] magnitude model trained (median|ret|={med:.2e}) {time.time()-t0:.0f}s", flush=True)

    # frozen cross-pair direction book
    strat = json.load(open(f"{MODELS}/m15xp_EURUSD_strategy.json")); cols = strat["primary_feats"]
    B = lgb.Booster(model_file=f"{MODELS}/m15xp_EURUSD_primary_lgb.txt")
    bthr = strat["bb_width_thr"]; cthr = strat["conf_thr"]

    res = {}
    for w, yr in (("test24", "2024"), ("test25", "2025"), ("oos", "2026")):
        D = MX.augment(MX.build_xp(MX.SPL[w]), MX.SPL[w], "xpof")
        pr = B.predict(D[cols].astype("float32").values); y = D["_y"].astype(int).values
        pm = Mg.predict_proba(D[base].astype("float32").values)[:, 1]   # predicted P(large move)
        ts = D["_ts"].values.astype("int64"); bbw = D["15m_bb_width"].values.astype(float); ny = D["sess_ny"].values > 0.5
        m = (bbw <= bthr) & ny & (np.abs(pr - 0.5) >= cthr); sel = MX.nonoverlap_chrono(ts, m)
        pred = (pr[sel] > 0.5).astype(int); yy = y[sel]; pmag = pm[sel]
        magmed = np.median(pmag)
        for bucket, bmask in (("LOWmag", pmag < magmed), ("HIGHmag", pmag >= magmed)):
            for side, nm in ((1, "UP"), (0, "DOWN")):
                ss = bmask & (pred == side)
                if ss.sum() < 5:
                    res[f"{yr}_{bucket}_{nm}"] = {"n": int(ss.sum()), "acc": None}; continue
                corr = (pred[ss] == yy[ss]).astype(float); lo, hi = MX.boot(corr)
                res[f"{yr}_{bucket}_{nm}"] = {"n": int(ss.sum()), "acc": round(float(corr.mean()), 4), "ci_lo": round(lo, 4)}
        del D
    print("\n=== per-year per-bucket per-side hit-rate (cross-pair trades; LOW vs HIGH predicted magnitude) ===", flush=True)
    for yr in ("2024", "2025", "2026"):
        for nm in ("UP", "DOWN"):
            lo = res.get(f"{yr}_LOWmag_{nm}", {}); hi = res.get(f"{yr}_HIGHmag_{nm}", {})
            print(f"  {yr} {nm:5} LOWmag acc={lo.get('acc')} (n{lo.get('n')}) | HIGHmag acc={hi.get('acc')} (n{hi.get('n')})", flush=True)
    out = {"exp": "Q1 magnitude x direction @15m", "median_abs_ret": med, "breakeven": BE,
           "incumbent_xpair_refit_p10": {"UP": 0.5673, "DOWN": 0.5742},
           "falsifier": "KILL unless a magnitude bucket lifts a side's binding-year hit-rate CI-lo > un-bucketed incumbent +1SE",
           "per_year_bucket_side": res,
           "verdict": "see analysis — expected NULL (sign-invariance); magnitude gates SIZE not SIGN"}
    json.dump(out, open("m15_magdir_result.json", "w"), indent=1)
    print(f"[magdir] -> m15_magdir_result.json ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
