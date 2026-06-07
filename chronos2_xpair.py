"""LEVER 3 — Chronos-2 GROUP-ATTENTION on the 7-pair USD panel → cross-sectional DIRECTION (+ magnitude).

The certified ≥5m DIRECTION edge is CROSS-SECTIONAL (7-USD-pair common factor / lead-lag), and every SINGLE-PAIR model
reads ~.50 on direction (Kronos zero-shot/FT/dispersion/embed, GBM, ARF, RFF, GRU, bar-CNN). Chronos-2
(amazon/chronos-2, 119.5M, group-attention) is the only mainstream TSFM whose core mechanism MATCHES this: fed a
multivariate panel `(batch, n_variates, history)` with n_variates>1, "information is shared among the variates"
(cross-attention) — so EURUSD's representation/forecast is conditioned on the other 6 USD pairs.

We feed the L-bar close panel of [EURUSD,GBPUSD,AUDUSD,NZDUSD,USDJPY,USDCHF,USDCAD] (EURUSD=variate 0) and test the
EURUSD forward DIRECTION three ways, all deriv-faithful (ties LOSE, breakeven .541, CPCV, per-year CI, per-session):
  (A) FORECAST sign  — median forecast EURUSD close(+H) vs close(now); does cross-conditioned forecast carry sign?
  (B) EMBED -> GBM    — Chronos-2 encoder embedding (EURUSD variate + cross-sectional pool, last patch, 768-d each)
                        as features for a LGBM direction classifier (the analog to the certified cross-pair GBM book).
  (C) per-SESSION     — both broken down NY/LDN/Asia (the edge is NY-concentrated if it's the same lever).
Plus MAGNITUDE: forecast quantile spread (q90-q10) as a forward-vol feature, paired-ablated vs the rv baseline.

One predict_quantiles(pred_len=30) + one embed per window cover all horizons {1,5,10,15,30}. ~26 win/s on the 8GB GPU.

PRE-REGISTERED FALSIFIER:
  DIRECTION (A/B) at horizon H: SURVIVES iff CPCV path-p10 accuracy ≥ 0.541 (deriv breakeven) on ≥1 of {pooled, NY},
    OR AUC p10 > 0.52 with per-year CI95-lo>0.5. Else KILLED — Chronos-2 group attention does NOT recover the
    cross-sectional sign at the model level (the edge needs the engineered USD-residual/lead-lag features, not a
    generic TSFM). Realistic bar: matches/﻿adds to the certified book, not necessarily beats it.
  MAGNITUDE: paired mean ΔAUC(base+spread − base) > +0.005 with CI95 excl 0.

Run (extraction GPU+fast, ablation CPU; resume from cache npz):
  ~/binary-algo-venv/bin/python chronos2_xpair.py [N_PER_YR=3000] [L=512] [tag]
Artifacts: chronos2_xpair_<tag>.npz, chronos2_xpair_<tag>_result.json.
"""
import sys, os, json, time, math, itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ROOT = "/media/sean/CORSAIR/binary-algo"; FEAT = f"{ROOT}/features"
PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"]   # EURUSD = variate 0
N_PER_YR = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
L        = int(sys.argv[2]) if len(sys.argv) > 2 else 512
TAG      = sys.argv[3] if len(sys.argv) > 3 else f"c2_N{N_PER_YR}_L{L}"
MODEL    = "amazon/chronos-2"
YEARS    = list(range(2012, 2027))
HORIZONS = [1, 5, 10, 15, 30]; H_MAX = max(HORIZONS)
STEP = 60; BATCH = 32; CKPT_BATCHES = 20; PRED_LEN = H_MAX
QL = [0.1, 0.25, 0.5, 0.75, 0.9]                                   # median idx 2, q10 idx 0, q90 idx 4
NPZ = f"{ROOT}/chronos2_xpair_{TAG}.npz"; RES = f"{ROOT}/chronos2_xpair_{TAG}_result.json"
BREAKEVEN = 0.541; DIR_AUC_BAR = 0.52; MAG_BAR = 0.55
N_TRIALS = 70; CORR_VAL_OOS = -0.54
RNG = np.random.default_rng(7)
T0 = time.time()
def hb(m): print(f"HB[{time.time()-T0:.0f}s] {m}", flush=True)


def perm_entropy(r, d=4, tau=1, W=120):
    N = len(r); Llag = (d - 1) * tau
    if N <= Llag + 1: return np.full(N, np.nan)
    idx = np.arange(N - Llag)[:, None] + np.arange(0, d * tau, tau)[None, :]
    order = np.argsort(r[idx], axis=1, kind="stable"); code = (order * (d ** np.arange(d))).sum(1).astype(np.int32)
    M = len(code); nb = d ** d
    oh = np.zeros((M, nb), dtype=np.float32); oh[np.arange(M), code] = 1.0
    cs = np.cumsum(oh, axis=0); cnt = cs.copy(); cnt[W:] = cs[W:] - cs[:-W]
    pp = cnt / np.maximum(cnt.sum(1, keepdims=True), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(pp > 0, pp * np.log(pp), 0.0), axis=1) / math.log(math.factorial(d))
    out = np.full(N, np.nan); out[Llag:Llag + M] = ent; out[:Llag + W] = np.nan; return out


def session_mask_local(ts, name):
    from sessions import session_mask
    return session_mask(ts, name)


def load_panel():
    """Aligned 1-min close panel for the 7 pairs over 2012-2026 (inner join on common timestamps)."""
    series = {}
    for p in PAIRS:
        cols = []
        for y in YEARS:
            fp = f"{FEAT}/{p}_{y}.parquet"
            if os.path.exists(fp):
                df = pd.read_parquet(fp, columns=["close"])
                df = df[~df.index.duplicated(keep="last")]
                cols.append(df["close"])
        s = pd.concat(cols).sort_index(); s = s[~s.index.duplicated(keep="last")]
        series[p] = s
    panel = pd.concat([series[p].rename(p) for p in PAIRS], axis=1, join="inner").dropna().sort_index()
    return panel


def select_decision_bars(t):
    n = len(t); ar = np.arange(n); loi = ar - L + 1; hii = ar + H_MAX
    base = (loi >= 0) & (hii < n)
    ctx_ok = np.zeros(n, bool); ctx_ok[base] = (t[ar[base]] - t[loi[base]]) == (L - 1) * STEP
    fwd_ok = np.zeros(n, bool); fwd_ok[base] = (t[hii[base]] - t[ar[base]]) == H_MAX * STEP
    elig = np.where(base & ctx_ok & fwd_ok)[0]
    yr = pd.to_datetime(t, unit="s", utc=True).year.values
    pick = []
    for Y in sorted(set(yr[elig].tolist())):
        ey = elig[yr[elig] == Y]
        if len(ey) > N_PER_YR: ey = ey[np.linspace(0, len(ey) - 1, N_PER_YR).astype(int)]
        pick.append(ey)
    pick = np.sort(np.concatenate(pick))
    take = []; bu = -1; gap = H_MAX * STEP
    for i in pick:
        if t[i] < bu: continue
        take.append(i); bu = int(t[i]) + gap
    return np.array(take, dtype=int)


# ============================================================= extraction (Chronos-2 predict + embed)
def extract():
    panel = load_panel(); t = panel.index.values.astype("datetime64[s]").astype("int64")
    P = panel.values.astype(np.float32)                            # (n, 7)
    n = len(panel); hb(f"panel n={n:,} pairs={PAIRS} span {panel.index[0]}..{panel.index[-1]}")
    pick = select_decision_bars(t); Np = len(pick); hb(f"decision bars (nonoverlap gap={H_MAX*STEP}s): {Np}")
    yr = pd.to_datetime(t[pick], unit="s", utc=True).year.values

    ceur = P[:, 0].astype(float)                                   # EURUSD close
    r = np.zeros(n); r[1:] = np.diff(np.log(ceur)); rs = pd.Series(r)
    pe = perm_entropy(r, 4, 1, 120); rv30 = rs.rolling(30).std().values; rv120 = rs.rolling(120).std().values
    base3 = np.column_stack([-pe[pick], rv30[pick], rv120[pick]]).astype(np.float32)
    retH = {H: np.log(ceur[pick + H] / ceur[pick]).astype(np.float32) for H in HORIZONS}   # forward log-ret per H

    EMB = 768
    emb_eur = np.full((Np, EMB), np.nan, np.float32)               # EURUSD variate, last patch
    emb_pool = np.full((Np, EMB), np.nan, np.float32)              # cross-sectional mean over 7 variates, last patch
    fc_ret = np.full((Np, len(HORIZONS)), np.nan, np.float32)      # median forecast cum log-ret to horizon H
    fc_spread = np.full((Np, len(HORIZONS)), np.nan, np.float32)   # forecast q90-q10 spread at H (rel)
    filled = np.zeros(Np, bool)
    if os.path.exists(NPZ):
        d = np.load(NPZ)
        if len(d["pick_t"]) == Np and np.array_equal(d["pick_t"], t[pick]):
            emb_eur = d["emb_eur"].copy(); emb_pool = d["emb_pool"].copy(); fc_ret = d["fc_ret"].copy()
            fc_spread = d["fc_spread"].copy(); filled = d["filled"].copy()
            hb(f"RESUME: {int(filled.sum())}/{Np} done")

    import torch; torch.set_num_threads(8)
    from chronos import Chronos2Pipeline
    DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = Chronos2Pipeline.from_pretrained(MODEL, device_map=DEV)
    hb(f"loaded chronos-2 119.5M on {DEV}; L={L} pred_len={PRED_LEN}")
    hidx = [H - 1 for H in HORIZONS]

    def save():
        tmp = NPZ.replace(".npz", ".tmp.npz")
        np.savez(tmp, pick_t=t[pick], year=yr, base3=base3, emb_eur=emb_eur, emb_pool=emb_pool,
                 fc_ret=fc_ret, fc_spread=fc_spread, filled=filled,
                 **{f"ret_{H}": retH[H] for H in HORIZONS}, horizons=np.array(HORIZONS))
        os.replace(tmp, NPZ)

    nb = 0
    for s in range(0, Np, BATCH):
        chunk = pick[s:s + BATCH]
        if filled[s:s + len(chunk)].all(): continue
        x = np.stack([P[i - L + 1:i + 1].T for i in chunk]).astype(np.float32)   # (B, 7, L)  EURUSD = row 0
        q, _ = pipe.predict_quantiles(x, prediction_length=PRED_LEN, quantile_levels=QL)   # list of (7,30,5)
        emb, _ = pipe.embed(x)                                                   # list of (7,34,768)
        for j, i in enumerate(chunk):
            qj = q[j].cpu().numpy() if hasattr(q[j], "cpu") else np.asarray(q[j])  # (7,30,5)
            c0 = float(P[i, 0])
            med = qj[0, :, 2]; q10 = qj[0, :, 0]; q90 = qj[0, :, 4]               # EURUSD variate
            for hk, hh in enumerate(hidx):
                fc_ret[s + j, hk] = math.log(max(med[hh], 1e-9) / max(c0, 1e-9))
                fc_spread[s + j, hk] = (q90[hh] - q10[hh]) / max(c0, 1e-9)
            ej = emb[j].float().cpu().numpy()                                     # (7,34,768)
            emb_eur[s + j] = ej[0, -1, :]
            emb_pool[s + j] = ej[:, -1, :].mean(axis=0)
            filled[s + j] = True
        nb += 1
        if nb % CKPT_BATCHES == 0: save(); hb(f"  extracted {int(filled.sum())}/{Np} (ckpt)")
    save(); hb(f"extraction DONE {int(filled.sum())}/{Np} -> {NPZ}")


# ============================================================= CPCV + helpers
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

def mk_lgb(n_estimators=600):
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.03, num_leaves=255,
        min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10,
        n_estimators=n_estimators, n_jobs=20, verbosity=-1)

def boot_ci(x, nb=4000, seed=11):
    x = np.asarray(x, float); rng = np.random.default_rng(seed); n = len(x)
    if n < 5: return (float("nan"), float("nan"))
    a = np.array([x[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)

def pct10(x): return round(float(np.percentile(x, 10)), 4)


def ablate():
    if not os.path.exists(NPZ): hb("no cache — extract first"); return
    d = np.load(NPZ); f = d["filled"]; ts = d["pick_t"][f]
    base3 = d["base3"][f]; emb_eur = d["emb_eur"][f]; emb_pool = d["emb_pool"][f]
    fc_ret = d["fc_ret"][f]; fc_spread = d["fc_spread"][f]; horizons = list(d["horizons"])
    o = np.argsort(ts, kind="stable")
    ts, base3, emb_eur, emb_pool, fc_ret, fc_spread = ts[o], base3[o], emb_eur[o], emb_pool[o], fc_ret[o], fc_spread[o]
    n = len(ts); hb(f"ablation on {n} windows")
    from sklearn.metrics import roc_auc_score
    g = cpcv_groups(n, 8)
    sess = {snm: session_mask_local(ts, snm) for snm in ("ny", "ldn", "asia")}
    yrv = pd.to_datetime(ts, unit="s", utc=True).year.values
    out = {"tag": TAG, "model": MODEL, "L": L, "N_PER_YR": N_PER_YR, "n_windows": int(n), "pairs": PAIRS,
           "breakeven": BREAKEVEN, "scope": "7-pair group-attention; deriv-faithful direction + magnitude ablation",
           "falsifier": "DIR survives iff CPCV acc p10>=0.541 on pooled or NY, OR AUC p10>0.52 w/ per-yr CI-lo>0.5; MAG iff paired dAUC>+.005 CI excl 0",
           "by_horizon": {}}
    Xemb = np.column_stack([emb_eur, emb_pool])                    # 1536-d cross-pair representation
    for hk, H in enumerate(horizons):
        ret = d[f"ret_{H}"][f][o]
        nz = ret != 0.0                                            # ties dropped
        yb = (ret > 0).astype(int)
        # ---------- (A) forecast-sign direction
        fcr = fc_ret[:, hk]; fc_call = (fcr > 0).astype(int)
        accA, aucA = [], []; accA_ny = []
        # ---------- (B) embed->GBM direction
        accB, aucB = [], []; accB_ny = []
        # ---------- (MAG) forecast spread paired ablation
        base_au, sp_au, dlift = [], [], []
        for combo, tri, tei in cpcv_paths(g, ts, H):
            tem = tei[nz[tei]]
            # (A) forecast sign — no fit, just score on test
            if len(tem) >= 100:
                ya = yb[tem]
                if 0 < ya.mean() < 1:
                    accA.append(float((fc_call[tem] == ya).mean())); aucA.append(roc_auc_score(ya, fcr[tem]))
                    nyt = tem[sess["ny"][tem]]
                    if len(nyt) >= 50 and 0 < yb[nyt].mean() < 1: accA_ny.append(float((fc_call[nyt] == yb[nyt]).mean()))
            # (B) embed -> GBM
            trm = tri[nz[tri]]
            if len(trm) > 100_000: trm = np.sort(RNG.choice(trm, 100_000, replace=False))
            if len(trm) >= 300 and len(tem) >= 150 and 0 < yb[trm].mean() < 1 and 0 < yb[tem].mean() < 1:
                mb = mk_lgb(); mb.fit(Xemb[trm], yb[trm]); pb = mb.predict_proba(Xemb[tem])[:, 1]
                accB.append(float(((pb > 0.5).astype(int) == yb[tem]).mean())); aucB.append(roc_auc_score(yb[tem], pb))
                nyt = tem[sess["ny"][tem]]
                if len(nyt) >= 50 and 0 < yb[nyt].mean() < 1:
                    accB_ny.append(float(((pb[sess["ny"][tem]] > 0.5).astype(int) == yb[nyt]).mean()))
            # (MAG) base vs base+spread
            am = np.isfinite(fc_spread[:, hk]); trm2 = tri[am[tri]]; tem2 = tei[am[tei]]
            aret = np.abs(ret)
            if len(trm2) > 100_000: trm2 = np.sort(RNG.choice(trm2, 100_000, replace=False))
            if len(trm2) >= 300 and len(tem2) >= 150:
                thr = np.nanquantile(aret[trm2], 0.75); ytr = (aret[trm2] >= thr).astype(int); yte = (aret[tem2] >= thr).astype(int)
                if 0 < ytr.mean() < 1 and 0 < yte.mean() < 1:
                    Xb_tr, Xb_te = base3[trm2], base3[tem2]
                    Xs_tr = np.column_stack([base3[trm2], fc_spread[trm2, hk]]); Xs_te = np.column_stack([base3[tem2], fc_spread[tem2, hk]])
                    mbb = mk_lgb(); mbb.fit(Xb_tr, ytr); pbb = mbb.predict_proba(Xb_te)[:, 1]
                    mss = mk_lgb(); mss.fit(Xs_tr, ytr); pss = mss.predict_proba(Xs_te)[:, 1]
                    base_au.append(roc_auc_score(yte, pbb)); sp_au.append(roc_auc_score(yte, pss))
        def summ(acc, auc, acc_ny):
            if not acc: return None
            acc = np.array(acc); auc = np.array(auc)
            d_ = dict(acc_mean=round(float(acc.mean()), 4), acc_p10=pct10(acc), auc_mean=round(float(auc.mean()), 4),
                      auc_p10=pct10(auc), n_paths=len(acc))
            if acc_ny: d_["acc_ny_mean"] = round(float(np.mean(acc_ny)), 4); d_["acc_ny_p10"] = pct10(np.array(acc_ny))
            surv = (d_["acc_p10"] >= BREAKEVEN) or (d_.get("acc_ny_p10", 0) >= BREAKEVEN) or (d_["auc_p10"] > DIR_AUC_BAR)
            d_["VERDICT"] = "SURVIVES" if surv else "KILLED"
            return d_
        mag = None
        if base_au:
            ba = np.array(base_au); sa = np.array(sp_au); dd = sa - ba; ci = boot_ci(dd)
            mag = dict(base_auc_mean=round(float(ba.mean()), 4), base_plus_spread_auc_mean=round(float(sa.mean()), 4),
                       dAUC_mean=round(float(dd.mean()), 4), dAUC_ci95=list(ci),
                       VERDICT=("SURVIVES" if (dd.mean() > 0.005 and ci[0] > 0) else "KILLED"))
        out["by_horizon"][str(H)] = dict(
            dir_forecast=summ(accA, aucA, accA_ny), dir_embed_gbm=summ(accB, aucB, accB_ny), magnitude_spread=mag)
        A = out["by_horizon"][str(H)]["dir_forecast"]; Bd = out["by_horizon"][str(H)]["dir_embed_gbm"]
        hb(f"H={H}m DIR-fc acc {A['acc_mean'] if A else '-'}(NYp10 {A.get('acc_ny_p10','-') if A else '-'}) AUC {A['auc_mean'] if A else '-'}->{A['VERDICT'] if A else '-'} "
           f"| DIR-emb acc {Bd['acc_mean'] if Bd else '-'}(NYp10 {Bd.get('acc_ny_p10','-') if Bd else '-'}) AUC {Bd['auc_mean'] if Bd else '-'}->{Bd['VERDICT'] if Bd else '-'} "
           f"| MAG {mag['VERDICT'] if mag else '-'} dAUC {mag['dAUC_mean'] if mag else '-'}")
    json.dump(out, open(RES, "w"), indent=1); hb(f"ablation DONE -> {RES}")
    return out


if __name__ == "__main__":
    extract()
    ablate()
