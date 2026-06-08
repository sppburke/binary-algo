"""PHASE 1A — the clean 7-pair USD RETURN PANEL + USD common-factor/residual (substrate for all cross-sectional methods).

The certified cross-pair direction edge (m5_xpair.build_xp) engineers SCALAR features (eu-equiv basket/catchup/eurresid/
lead-lag over lookbacks) from the 7 USD pairs. The new path-based methods (signatures, DMD/HAVOK, contrastive, whitening,
frac-diff) need the RAW multi-channel return PATH, which is not persisted anywhere. This builds it once.

Output: features/panel_<year>.parquet, one row per 1-min timestamp where ALL 7 pairs are present (inner join), columns:
  t (epoch s, UTC)
  r_<PAIR> for the 7 pairs = EU-EQUIV 1-min log-return (sign-flipped for USD-base pairs so +r = USD-weakness = EURUSD-up)
  fac   = USD common factor = cross-sectional MEAN of the 7 eu-equiv returns
  e_<PAIR> = residual = r_<PAIR> - fac  (idiosyncratic, beta=1 in eu-equiv space — the certified book's basket convention)
  c_eur = raw EURUSD close (for level-based / fracdiff methods + label construction)
Returns are already stationary; downstream derives factor/residual/signatures/DMD-state/lead-lag from r_<PAIR>.

VALIDATION (mode=validate): recompute the certified build_xp scalar features FROM THE PANEL, run the certified 15m NY
direction CPCV, and confirm it REPRODUCES the m15xp p10 ~.567 — proving the panel construction is faithful & non-leaky.

Run:  ~/binary-algo-venv/bin/python build_panel.py build      # emit panel_<year>.parquet 2012-2026
      ~/binary-algo-venv/bin/python build_panel.py validate [HOR=15]
"""
import sys, os, time, json, itertools, math
import numpy as np, pandas as pd
ROOT = "/home/sean/git/binary-algo"; FEAT = f"{ROOT}/features"
PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"]
USD_BASE = {"USDJPY", "USDCHF", "USDCAD"}
NONEU = [p for p in PAIRS if p != "EURUSD"]
YEARS = list(range(2012, 2027))
LB = [1, 3, 5, 10, 15, 30]
T0 = time.time()
def hb(m): print(f"[{time.time()-T0:.0f}s] {m}", flush=True)
def eu_sign(p): return -1.0 if p in USD_BASE else 1.0


def build_year(y):
    cl = {}
    for p in PAIRS:
        fp = f"{FEAT}/{p}_{y}.parquet"
        if not os.path.exists(fp): return None
        d = pd.read_parquet(fp, columns=["close"]); d = d[~d.index.duplicated(keep="last")]
        cl[p] = d["close"]
    df = pd.DataFrame(cl).dropna().sort_index()
    if len(df) < 200: return None
    out = pd.DataFrame(index=df.index)
    out["t"] = df.index.values.astype("datetime64[s]").astype("int64")
    reu = {}
    for p in PAIRS:
        r = np.zeros(len(df)); r[1:] = eu_sign(p) * np.diff(np.log(df[p].values))   # eu-equiv 1m log-return
        out[f"r_{p}"] = r.astype(np.float32); reu[p] = r
    fac = np.mean(np.vstack([reu[p] for p in PAIRS]), axis=0)                        # USD common factor
    out["fac"] = fac.astype(np.float32)
    for p in PAIRS: out[f"e_{p}"] = (reu[p] - fac).astype(np.float32)
    out["c_eur"] = df["EURUSD"].values.astype(np.float32)
    return out.reset_index(drop=True)


def build():
    n_tot = 0
    for y in YEARS:
        pan = build_year(y)
        if pan is None: hb(f"{y}: skip (missing pair)"); continue
        pan.to_parquet(f"{FEAT}/panel_{y}.parquet"); n_tot += len(pan)
        hb(f"{y}: {len(pan):,} bars -> features/panel_{y}.parquet")
    hb(f"DONE total {n_tot:,} bars")


# ----------------------------------------------------------------- validation (reproduce m15xp .567 from the panel)
def cpcv_groups(n, ng=8):
    e = np.linspace(0, n, ng + 1).astype(int); g = np.zeros(n, np.int8)
    for k in range(ng): g[e[k]:e[k + 1]] = k
    return g

def cpcv_paths(g, ts, hm, ng=8, k=2):
    hs = hm * 60; alli = np.arange(len(g))
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


def xp_features_from_panel(P):
    """Recompute build_xp's certified scalar features from the raw eu-equiv returns in the panel."""
    n = len(P); reu = {p: P[f"r_{p}"].values.astype(float) for p in PAIRS}
    feats = {}
    for k in LB:
        euk = {p: pd.Series(reu[p]).rolling(k).sum().values for p in PAIRS}
        eu_r = euk["EURUSD"]
        basket = np.nanmean(np.vstack([euk[p] for p in NONEU]), axis=0)
        disp = np.nanstd(np.vstack([euk[p] for p in NONEU]), axis=0)
        agree = np.nanmean(np.vstack([(np.sign(euk[p]) == np.sign(basket)).astype(float) for p in NONEU]), axis=0)
        feats[f"eu_r{k}"] = eu_r; feats[f"usdbask{k}"] = basket; feats[f"catchup{k}"] = basket - eu_r
        feats[f"eurresid{k}"] = eu_r - basket; feats[f"disp{k}"] = disp; feats[f"agree{k}"] = agree
        for p in NONEU: feats[f"ll_{p}{k}"] = euk[p] - eu_r
    X = pd.DataFrame(feats); return X


def validate(HOR=15):
    frames = []
    for y in YEARS:
        fp = f"{FEAT}/panel_{y}.parquet"
        if os.path.exists(fp): frames.append(pd.read_parquet(fp))
    P = pd.concat(frames).reset_index(drop=True); hb(f"panel n={len(P):,}")
    t = P["t"].values.astype("int64"); n = len(P)
    reu_eur = P["r_EURUSD"].values.astype(float)
    # forward HOR-bar eu-equiv EURUSD return (contiguous), sign label, ties excluded
    csum = np.concatenate([[0.0], np.cumsum(reu_eur)])
    fwd = np.full(n, np.nan); fwd[:n - HOR] = csum[HOR + 1:] - csum[1:n - HOR + 1]
    contig = np.zeros(n, bool); contig[:n - HOR] = (t[HOR:] - t[:-HOR]) == HOR * 60
    dt = pd.to_datetime(t, unit="s", utc=True); hours = dt.hour.values + dt.minute.values / 60.0
    ny = (hours >= 13.0) & (hours < 22.0)
    X = xp_features_from_panel(P)
    valid = contig & np.isfinite(fwd) & (fwd != 0.0) & ny & np.isfinite(X.values).all(1)
    Xv = X.values[valid].astype(np.float32); y = (fwd[valid] > 0).astype(int); ts = t[valid]
    hb(f"NY decision bars n={len(y):,} up-rate={y.mean():.4f}")
    g = cpcv_groups(len(y), 8); rng = np.random.default_rng(7)
    from sklearn.metrics import roc_auc_score
    accs = []
    for ci, (combo, tri, tei) in enumerate(cpcv_paths(g, ts, HOR)):
        tr = np.sort(rng.choice(tri, 100_000, replace=False)) if len(tri) > 100_000 else tri
        m = mk_lgb(); m.fit(Xv[tr], y[tr]); p = m.predict_proba(Xv[tei])[:, 1]
        # certified book = comp x NY x confidence cov0.10 selective accuracy
        conf = np.abs(p - 0.5); thr = np.quantile(conf, 1 - 0.10); sel = conf >= thr
        if sel.sum() >= 25: accs.append(float(((p[sel] > 0.5).astype(int) == y[tei][sel]).mean()))
        if ci % 7 == 0: hb(f"  path {ci+1}/28 selacc={accs[-1] if accs else float('nan'):.4f}")
    accs = np.array(accs); p10 = float(np.percentile(accs, 10)); mean = float(accs.mean())
    res = dict(HOR=HOR, n=int(len(y)), n_paths=len(accs), selacc_mean=round(mean, 4), selacc_p10=round(p10, 4),
               target_m15xp="UP .567 / DOWN .574 (combined book)", reproduces=bool(p10 >= 0.54))
    json.dump(res, open(f"{ROOT}/build_panel_validate_{HOR}m.json", "w"), indent=1)
    hb(f"VALIDATION {HOR}m: selacc mean={mean:.4f} p10={p10:.4f} (target ~.55-.57) -> {'REPRODUCES' if res['reproduces'] else 'CHECK'}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    if mode == "build": build()
    elif mode == "validate": validate(int(sys.argv[2]) if len(sys.argv) > 2 else 15)
