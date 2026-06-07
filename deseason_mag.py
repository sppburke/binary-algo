"""DESEASONALIZED-RV upgrade to the CERTIFIED magnitude model (cpcv_certify.py § magnitude).

The time-of-day decomposition (kronos_embed_tod.py) proved raw clock features add a real CPCV-robust magnitude lift
(+0.0048 at 10m, +0.0087 at 30m) and at 30m BEAT the Kronos embedding — for FREE. This builds the principled version:
DESEASONALIZED realized variance. rv30/rv120 mix the strong FX intraday vol seasonality (London/NY overlap spikes,
Asia lull) into the level; we estimate a CAUSAL seasonal profile and feed (a) the expected vol at this minute-of-day
and (b) the deseasonalized ratio = rv/seasonal (ABNORMAL vol), so the model predicts the abnormal component.

Faithful to the CERTIFICATION: identical to cpcv_certify.load_magnitude (perm-entropy d4 / rv30 / rv120, target
|ret_H|>=train-Q75, contiguity, full pooled 2012-2026 1-min bars), identical CPCV (8 groups, k=2 -> 28 purged paths,
purge+embargo=1 horizon), identical LGBM (mk_lgb 600), identical deflation. CPU-only -> runs on the FULL dataset (no
subsample of decision bars; only TRAIN rows are subsampled to 100k per fit, exactly like cpcv_certify).

LEAKAGE GUARD (the trap): the seasonal profile is estimated PER CPCV PATH on its TRAIN block ONLY (bincount of rv by
minute-of-day), then applied to train+test rows by each row's own minute-of-day. No test row ever enters the profile.
All arms share the SAME per-path train subsample so the comparison is PAIRED.

Arms (features added ON TOP of base = [-pe, rv30, rv120]):
  base       : (the certified model — sanity anchor, expect path-mean AUC ~0.74)
  +tod       : [hour, minute_of_day, dow, sin, cos]            (raw clock — the validated lift)
  +seasonal  : [seas_rv30, seas_rv120]                          (expected vol at this minute, causal)
  +deseason  : [rv30/seas_rv30, rv120/seas_rv120]              (ABNORMAL vol ratio)
  +seas_des  : seasonal + deseason
  +all       : tod + seasonal + deseason

Run: ~/binary-algo-venv/bin/python deseason_mag.py [HOR=30]   ->  deseason_mag_<HOR>m_result.json
"""
import os, sys, json, time, math, itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H

FEAT = H.FEAT_DIR; PAIR = "EURUSD"
YEARS = list(range(2012, 2027))
HOR = int(sys.argv[1]) if len(sys.argv) > 1 else 30
RNG = np.random.default_rng(7); SUBSAMPLE = 100_000
N_GROUPS = 8; K_TEST = 2; N_TRIALS = 70; CORR_VAL_OOS = -0.54; MAG_BAR = 0.55
RES = f"/media/sean/CORSAIR/binary-algo/deseason_mag_{HOR}m_result.json"
T0 = time.time()
def hb(m): print(f"HB[{time.time()-T0:.0f}s] {m}", flush=True)


def _contig_fwd(idx_secs, c, hor):
    n = len(c); contig = np.zeros(n, bool)
    if n > hor: contig[:n-hor] = (idx_secs[hor:] - idx_secs[:-hor]) == hor*60
    fwd = np.full(n, np.nan); fwd[:n-hor] = c[hor:]; ret = fwd / c - 1.0
    return ret, contig


def perm_entropy(r, d=4, tau=1, W=120):
    N = len(r); Lg = (d-1)*tau
    if N <= Lg+1: return np.full(N, np.nan)
    idx = np.arange(N-Lg)[:, None] + np.arange(0, d*tau, tau)[None, :]
    order = np.argsort(r[idx], axis=1, kind="stable"); code = (order*(d**np.arange(d))).sum(1).astype(np.int32)
    M = len(code); nb = d**d
    oh = np.zeros((M, nb), dtype=np.float32); oh[np.arange(M), code] = 1.0
    cs = np.cumsum(oh, axis=0); cnt = cs.copy(); cnt[W:] = cs[W:]-cs[:-W]
    pp = cnt/np.maximum(cnt.sum(1, keepdims=True), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(pp > 0, pp*np.log(pp), 0.0), axis=1)/math.log(math.factorial(d))
    out = np.full(N, np.nan); out[Lg:Lg+M] = ent; out[:Lg+W] = np.nan; return out


def load(hor):
    """Identical to cpcv_certify.load_magnitude + raw rv30/rv120 (un-negated) and clock fields, for deseasonalization."""
    pes, r30s, r120s, arets, tss = [], [], [], [], []
    for y in YEARS:
        p = f"{FEAT}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        df = pd.read_parquet(p, columns=["close"]); df = df[~df.index.duplicated(keep="last")].sort_index()
        c = df["close"].values.astype(float); secs = df.index.values.astype("datetime64[s]").astype("int64")
        ret, contig = _contig_fwd(secs, c, hor)
        r = np.zeros(len(c)); r[1:] = np.diff(np.log(c))
        pe = perm_entropy(r, 4, 1, 120); rs = pd.Series(r)
        rv30 = rs.rolling(30).std().values; rv120 = rs.rolling(120).std().values
        valid = contig & np.isfinite(ret) & np.isfinite(pe) & np.isfinite(rv30) & np.isfinite(rv120)
        pes.append(pe[valid].astype(np.float32)); r30s.append(rv30[valid].astype(np.float32))
        r120s.append(rv120[valid].astype(np.float32)); arets.append(np.abs(ret[valid]).astype(np.float32))
        tss.append(secs[valid]); del df
    pe = np.concatenate(pes); rv30 = np.concatenate(r30s); rv120 = np.concatenate(r120s)
    aret = np.concatenate(arets); ts = np.concatenate(tss)
    o = np.argsort(ts, kind="stable")
    pe, rv30, rv120, aret, ts = pe[o], rv30[o], rv120[o], aret[o], ts[o]
    dt = pd.to_datetime(ts, unit="s", utc=True)
    mod = (dt.hour*60 + dt.minute).values.astype(np.int32)        # minute-of-day 0..1439
    tod = np.column_stack([dt.hour.values, mod, dt.dayofweek.values,
                           np.sin(2*np.pi*mod/1440.0), np.cos(2*np.pi*mod/1440.0)]).astype(np.float32)
    return pe, rv30, rv120, aret, ts, mod, tod


def cpcv_groups(n, ng):
    e = np.linspace(0, n, ng+1).astype(int); g = np.zeros(n, np.int8)
    for k in range(ng): g[e[k]:e[k+1]] = k
    return g

def cpcv_paths(g, ts, horizon_min, ng=N_GROUPS, k=K_TEST):
    hs = horizon_min*60; emb = horizon_min*60; alli = np.arange(len(g))
    for combo in itertools.combinations(range(ng), k):
        tm = np.isin(g, combo); te = alli[tm]; tr = alli[~tm]
        keep = np.ones(len(tr), bool); tt = ts[tr]
        for grp in combo:
            gi = alli[g == grp]; lo, hi = ts[gi[0]], ts[gi[-1]]
            keep &= ~((tt >= lo - hs) & (tt <= hi + emb))
        yield combo, tr[keep], te

def mk_lgb(n=600):
    return lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.03, num_leaves=255,
        min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10,
        n_estimators=n, n_jobs=20, verbosity=-1)

def deflated(dist, bar):
    a = np.asarray(dist, float); a = a[np.isfinite(a)]
    mu = float(a.mean()); sd = float(a.std(ddof=1)); p10 = float(np.percentile(a, 10))
    emax = sd*math.sqrt(2*math.log(max(N_TRIALS, 2))); de = mu - abs(CORR_VAL_OOS)*emax
    return dict(path_mean=round(mu, 4), path_p10=round(p10, 4), path_min=round(float(a.min()), 4),
                deflated_expectation=round(de, 4), p10_clears_bar=bool(p10 > bar), deflated_clears_bar=bool(de > bar))

def boot_ci(x, nb=5000, seed=11):
    x = np.asarray(x, float); rng = np.random.default_rng(seed); n = len(x)
    a = np.array([x[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)

def seasonal_profile(mod_tr, val_tr):
    """Causal: mean(val) per minute-of-day over TRAIN only; empty bins -> global train mean."""
    s = np.bincount(mod_tr, weights=val_tr, minlength=1440)
    c = np.bincount(mod_tr, minlength=1440)
    gm = val_tr.mean()
    prof = np.where(c > 0, s/np.maximum(c, 1), gm)
    return prof.astype(np.float32)


ARMS = ["base", "+tod", "+seasonal", "+deseason", "+seas_des", "+all"]

def build_arm(arm, base, tod, seas30, seas120, des30, des120):
    if arm == "base": return base
    if arm == "+tod": return np.column_stack([base, tod])
    if arm == "+seasonal": return np.column_stack([base, seas30, seas120])
    if arm == "+deseason": return np.column_stack([base, des30, des120])
    if arm == "+seas_des": return np.column_stack([base, seas30, seas120, des30, des120])
    if arm == "+all": return np.column_stack([base, tod, seas30, seas120, des30, des120])


def main():
    hb(f"loading magnitude frame hor={HOR}m ...")
    pe, rv30, rv120, aret, ts, mod, tod = load(HOR)
    n = len(aret); hb(f"pooled n={n:,}")
    base_full = np.column_stack([-pe, rv30, rv120]).astype(np.float32)
    g = cpcv_groups(n, N_GROUPS)
    aucs = {a: [] for a in ARMS}; lifts = {a: [] for a in ARMS}
    for ci, (combo, tri, tei) in enumerate(cpcv_paths(g, ts, HOR)):
        # seasonal profile from FULL train (causal), then subsample train rows for fitting (shared across arms)
        prof30 = seasonal_profile(mod[tri], rv30[tri]); prof120 = seasonal_profile(mod[tri], rv120[tri])
        seas30 = prof30[mod]; seas120 = prof120[mod]
        des30 = rv30/(seas30+1e-12); des120 = rv120/(seas120+1e-12)
        tr = np.sort(RNG.choice(tri, SUBSAMPLE, replace=False)) if len(tri) > SUBSAMPLE else tri
        thr = np.nanquantile(aret[tr], 0.75)
        ytr = (aret[tr] >= thr).astype(int); yte = (aret[tei] >= thr).astype(int)
        if ytr.mean() in (0.0, 1.0) or yte.mean() in (0.0, 1.0): continue
        for arm in ARMS:
            Xtr = build_arm(arm, base_full[tr], tod[tr], seas30[tr], seas120[tr], des30[tr], des120[tr])
            Xte = build_arm(arm, base_full[tei], tod[tei], seas30[tei], seas120[tei], des30[tei], des120[tei])
            m = mk_lgb(); m.fit(Xtr, ytr); pp = m.predict_proba(Xte)[:, 1]
            aucs[arm].append(roc_auc_score(yte, pp))
            q = np.quantile(pp, [0.9, 0.1]); hi = pp >= q[0]; loq = pp <= q[1]
            lifts[arm].append(float(aret[tei][hi].mean()/max(aret[tei][loq].mean(), 1e-12)))
            del m, pp
        if ci % 7 == 0: hb(f"  path {ci+1}/28 base AUC={aucs['base'][-1]:.4f} +deseason={aucs['+deseason'][-1]:.4f} +all={aucs['+all'][-1]:.4f}")
    base = np.array(aucs["base"])
    out = {"horizon_min": HOR, "n_windows": int(n), "n_paths": len(base), "mag_bar": MAG_BAR,
           "scope": "FULL 2012-2026 pooled (faithful to cpcv_certify); causal per-path seasonal profile (train-only)",
           "falsifier": "arm beats base iff paired mean dAUC>0 with CI95 excl 0; PROMOTE best robust arm",
           "by_arm": {}}
    for arm in ARMS:
        au = np.array(aucs[arm]); dd = au - base; ci = boot_ci(dd) if arm != "base" else [0.0, 0.0]
        out["by_arm"][arm] = dict(auc_mean=round(float(au.mean()), 4), auc_p10=round(float(np.percentile(au, 10)), 4),
            auc_min=round(float(au.min()), 4), lift_mean=round(float(np.mean(lifts[arm])), 3),
            dAUC_vs_base_mean=round(float(dd.mean()), 4), dAUC_vs_base_ci95=list(ci),
            beats_base=bool(arm != "base" and dd.mean() > 0 and ci[0] > 0), deflated=deflated(au, MAG_BAR))
        v = out["by_arm"][arm]
        hb(f"{arm:11s} AUC {v['auc_mean']:.4f}(p10 {v['auc_p10']:.4f}) lift {v['lift_mean']:.2f}x  dAUC {v['dAUC_vs_base_mean']:+.4f} CI{v['dAUC_vs_base_ci95']}  beats_base={v['beats_base']}")
    best = max([a for a in ARMS if a != "base"], key=lambda a: out["by_arm"][a]["dAUC_vs_base_mean"])
    out["best_arm"] = best
    json.dump(out, open(RES, "w"), indent=1); hb(f"DONE best={best} -> {RES}")


if __name__ == "__main__":
    main()
