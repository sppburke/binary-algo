"""LEVER 2 — Kronos decode_s1 HIDDEN STATE as a frozen feature for the GBM.

Kronos' transformer produces a 512-d post-norm context representation per bar (kronos.py:305-308, decode_s1 returns
(s1_logits, x) with x = [B, L, d_model=512]). We take x[:, -1, :] — the learned representation AT the decision bar
(it already carries time-of-day via the additive time embedding, kronos.py:298) — and test it as a frozen feature:

  Does Kronos' learned 512-d representation carry MAGNITUDE info beyond the certified backward-rv baseline
  [-pe, rv30, rv120], or DIRECTION (sign) info beyond chance, from EURUSD's OWN OHLCV?

Honest prior (after Lever 1 killed + every single-pair direction null): single-pair representation is expected to
share Kronos' blind spots — the magnitude edge is already in cheap rv, the direction edge is cross-sectional (7-pair).
But the embedding is a DIFFERENT object than per-path dispersion and is cheap to test: ONE forward pass per window
(tokenize -> decode_s1), NO autoregression, NO K-sampling -> ~10-50x faster than Lever 1.

Faithful by construction — reuses cpcv_certify's target/CPCV/LGBM/deflation, same nonoverlap subsample design as
kronos_disp.py. Tests BOTH targets per horizon H in HORIZONS:
  MAGNITUDE: paired CPCV ablation, baseline {-pe,rv30,rv120} vs baseline+512emb, plus 512emb-ALONE magAUC.
  DIRECTION: 512emb-ALONE -> P(ret_H>0) CPCV AUC (ties dropped); single-pair, expected ~.50.

PRE-REGISTERED FALSIFIER:
  MAGNITUDE H: KILL unless base path-mean AUC>=0.60 AND paired mean dAUC(base+emb - base)>+0.005 with CI95 excl 0.
  DIRECTION H: KILL unless emb-alone CPCV AUC p10 > 0.52 (a real ranking edge over coin-flip).

Run (extraction GPU+fast, ablation CPU; both resume from cache npz):
  ~/binary-algo-venv/bin/python kronos_embed.py [N_PER_YR=4000] [L=256] [model] [tag]
Artifacts: kronos_embed_<tag>.npz (512-d emb + baseline + labels, checkpointed) ; kronos_embed_<tag>_result.json.
"""
import sys, os, json, time, math, itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/sean/git/Kronos")

ROOT = "/media/sean/CORSAIR/binary-algo"; OUT = f"{ROOT}/ohlc_cache"
N_PER_YR = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
L        = int(sys.argv[2]) if len(sys.argv) > 2 else 256
MODEL    = sys.argv[3] if len(sys.argv) > 3 else "NeoQuasar/Kronos-small"
TAG      = sys.argv[4] if len(sys.argv) > 4 else f"embed_N{N_PER_YR}_L{L}"
TOK_ID   = "NeoQuasar/Kronos-Tokenizer-base"
HORIZONS = [1, 5, 10, 15, 30]; H_MAX = max(HORIZONS)
STEP = 60; BATCH = 32; CKPT_BATCHES = 30; EMB_D = 512
NPZ = f"{ROOT}/kronos_embed_{TAG}.npz"; RES = f"{ROOT}/kronos_embed_{TAG}_result.json"
MAG_BAR = 0.55; DIR_AUC_BAR = 0.52
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


def load_bars():
    frames = [pd.read_parquet(f"{OUT}/EURUSD_1m_{sp}.parquet") for sp in ("train", "val", "test", "oos")]
    B = pd.concat(frames).reset_index(drop=True)
    return B.drop_duplicates(subset="t").sort_values("t").reset_index(drop=True)


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


# ============================================================= extraction (decode_s1 -> 512-d, checkpointed)
def extract():
    B = load_bars(); t = B["t"].values.astype("int64")
    O = B["open"].values.astype(float); Hg = B["high"].values.astype(float)
    Lw = B["low"].values.astype(float); C = B["close"].values.astype(float); V = B["vol"].values.astype(float)
    n = len(B); hb(f"bars n={n:,}")
    pick = select_decision_bars(t); Np = len(pick); hb(f"decision bars (nonoverlap gap={H_MAX*STEP}s): {Np}")
    yr = pd.to_datetime(t[pick], unit="s", utc=True).year.values

    r = np.zeros(n); r[1:] = np.diff(np.log(C)); rs = pd.Series(r)
    pe = perm_entropy(r, 4, 1, 120); rv30 = rs.rolling(30).std().values; rv120 = rs.rolling(120).std().values
    base3 = np.column_stack([-pe[pick], rv30[pick], rv120[pick]]).astype(np.float32)
    retH = {H: (C[pick + H] / C[pick] - 1.0).astype(np.float32) for H in HORIZONS}
    aretH = {H: np.abs(retH[H]) for H in HORIZONS}

    emb = np.full((Np, EMB_D), np.nan, np.float32); filled = np.zeros(Np, bool); start = 0
    if os.path.exists(NPZ):
        d = np.load(NPZ)
        if len(d["pick_t"]) == Np and np.array_equal(d["pick_t"], t[pick]):
            emb = d["emb"].copy(); filled = d["filled"].copy(); start = int(filled.sum())
            hb(f"RESUME: {start}/{Np} already extracted")

    import torch; torch.set_num_threads(8)
    from model import Kronos, KronosTokenizer, KronosPredictor
    from model.kronos import calc_time_stamps
    DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
    tok = KronosTokenizer.from_pretrained(TOK_ID); mdl = Kronos.from_pretrained(MODEL).to(DEV).eval()
    pred = KronosPredictor(mdl, tok, device=DEV, max_context=max(L, 256))
    clip = pred.clip; pcols = pred.price_cols + [pred.vol_col, pred.amt_vol]
    hb(f"loaded {sum(p.numel() for p in mdl.parameters())/1e6:.1f}M params on {DEV}; emb_d={EMB_D}")

    def save():
        tmp = NPZ.replace(".npz", ".tmp.npz")
        np.savez(tmp, pick_t=t[pick], year=yr, base3=base3, emb=emb, filled=filled,
                 **{f"ret_{H}": retH[H] for H in HORIZONS}, **{f"aret_{H}": aretH[H] for H in HORIZONS},
                 horizons=np.array(HORIZONS))
        os.replace(tmp, NPZ)

    nb = 0
    for s in range(0, Np, BATCH):
        chunk = pick[s:s + BATCH]
        if filled[s:s + len(chunk)].all(): continue
        xb, sb = [], []
        for i in chunk:
            sl = slice(i - L + 1, i + 1)                                   # context INCLUDES decision bar i
            x = np.column_stack([O[sl], Hg[sl], Lw[sl], C[sl], V[sl], np.zeros(L)]).astype(np.float32)
            mu = x.mean(0); sd = x.std(0); xn = np.clip((x - mu) / (sd + 1e-5), -clip, clip)   # predict_batch norm
            xb.append(xn)
            sb.append(calc_time_stamps(pd.Series(pd.to_datetime(t[sl], unit="s", utc=True))).values.astype(np.float32))
        import torch
        with torch.no_grad():
            xt = torch.from_numpy(np.stack(xb)).to(DEV)                    # (B, L, 6)
            st = torch.from_numpy(np.stack(sb)).to(DEV)                    # (B, L, 5)
            s1, s2 = tok.encode(xt, half=True)                            # (B, L) each
            _, ctx = mdl.decode_s1(s1, s2, st)                            # ctx (B, L, 512)
            e = ctx[:, -1, :].float().cpu().numpy()                       # (B, 512) decision-bar representation
        emb[s:s + len(chunk)] = e; filled[s:s + len(chunk)] = True
        nb += 1
        if nb % CKPT_BATCHES == 0: save(); hb(f"  extracted {int(filled.sum())}/{Np} (ckpt)")
    save(); hb(f"extraction DONE {int(filled.sum())}/{Np} -> {NPZ}")


# ============================================================= CPCV + deflation (same as cpcv_certify / kronos_disp)
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

def deflated(headline, dist, bar):
    a = np.asarray(dist, float); a = a[np.isfinite(a)]
    mu = float(a.mean()); sd = float(a.std(ddof=1)) if len(a) > 1 else float("nan")
    p10 = float(np.percentile(a, 10)); emax = sd * math.sqrt(2 * math.log(max(N_TRIALS, 2))) if np.isfinite(sd) else float("nan")
    de = mu - abs(CORR_VAL_OOS) * emax
    return dict(path_mean=round(mu, 4), path_p10=round(p10, 4), path_min=round(float(a.min()), 4),
                deflated_expectation=round(de, 4), p10_clears_bar=bool(p10 > bar), deflated_clears_bar=bool(de > bar))

def boot_ci(x, nb=5000, seed=11):
    x = np.asarray(x, float); rng = np.random.default_rng(seed); n = len(x)
    a = np.array([x[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)


def ablate():
    if not os.path.exists(NPZ): hb("no cache — extract first"); return
    d = np.load(NPZ); filled = d["filled"]; ts = d["pick_t"][filled]
    base3 = d["base3"][filled]; emb = d["emb"][filled]; horizons = list(d["horizons"])
    order = np.argsort(ts, kind="stable"); ts = ts[order]; base3 = base3[order]; emb = emb[order]
    n = len(ts); hb(f"ablation on {n} windows; emb_d={emb.shape[1]}")
    from sklearn.metrics import roc_auc_score
    g = cpcv_groups(n, 8)
    out = {"tag": TAG, "model": MODEL, "L": L, "N_PER_YR": N_PER_YR, "n_windows": int(n), "emb_d": int(emb.shape[1]),
           "scope": "SUBSAMPLE ablation (nonoverlap, 2021-2026 1m cache) — single-pair decode_s1 feature test",
           "falsifier_mag": "KILL unless base mean AUC>=0.60 AND paired mean dAUC>+0.005 CI95 excl 0",
           "falsifier_dir": "KILL unless emb-alone CPCV AUC p10>0.52", "mag_bar": MAG_BAR, "dir_auc_bar": DIR_AUC_BAR,
           "by_horizon": {}}
    for hi, H in enumerate(horizons):
        aret = d[f"aret_{H}"][filled][order]; ret = d[f"ret_{H}"][filled][order]
        okm = np.isfinite(aret) & np.isfinite(emb).all(1)
        ba, ea, eo, dlift = [], [], [], []                                # mag: base, base+emb, emb-only
        da_dir = []                                                       # dir: emb-only AUC
        for combo, tri, tei in cpcv_paths(g, ts, H):
            trm = tri[okm[tri]]; tem = tei[okm[tei]]
            if len(trm) > 100_000: trm = np.sort(RNG.choice(trm, 100_000, replace=False))
            if len(trm) < 300 or len(tem) < 150: continue
            # ---- magnitude
            thr = np.nanquantile(aret[trm], 0.75); ytr = (aret[trm] >= thr).astype(int); yte = (aret[tem] >= thr).astype(int)
            if ytr.mean() in (0.0, 1.0) or yte.mean() in (0.0, 1.0): continue
            Xb_tr, Xb_te = base3[trm], base3[tem]
            Xe_tr, Xe_te = np.column_stack([base3[trm], emb[trm]]), np.column_stack([base3[tem], emb[tem]])
            mb = mk_lgb(); mb.fit(Xb_tr, ytr); pb = mb.predict_proba(Xb_te)[:, 1]
            me = mk_lgb(); me.fit(Xe_tr, ytr); pe2 = me.predict_proba(Xe_te)[:, 1]
            mo = mk_lgb(); mo.fit(emb[trm], ytr); po = mo.predict_proba(emb[tem])[:, 1]
            ba.append(roc_auc_score(yte, pb)); ea.append(roc_auc_score(yte, pe2)); eo.append(roc_auc_score(yte, po))
            q = np.quantile(pe2, [0.9, 0.1]); dlift.append(float(aret[tem][pe2 >= q[0]].mean() / max(aret[tem][pe2 <= q[1]].mean(), 1e-12)))
            # ---- direction (emb-only; ties dropped)
            dmtr = ret[trm] != 0.0; dmte = ret[tem] != 0.0
            if dmtr.sum() > 300 and dmte.sum() > 150:
                yd = (ret[trm][dmtr] > 0).astype(int); yvd = (ret[tem][dmte] > 0).astype(int)
                if yd.mean() not in (0.0, 1.0) and yvd.mean() not in (0.0, 1.0):
                    mdd = mk_lgb(); mdd.fit(emb[trm][dmtr], yd); pdd = mdd.predict_proba(emb[tem][dmte])[:, 1]
                    da_dir.append(roc_auc_score(yvd, pdd))
        if not ba: out["by_horizon"][str(H)] = {"note": "no valid paths"}; continue
        ba, ea, eo = np.array(ba), np.array(ea), np.array(eo); dd = ea - ba; dci = boot_ci(dd)
        base_p10 = float(np.percentile(ba, 10)); emb_p10 = float(np.percentile(ea, 10))
        cond1 = float(ba.mean()) >= 0.60; cond2 = (float(dd.mean()) > 0.005) and (dci[0] > 0.0); cond3 = emb_p10 >= base_p10
        mag_verdict = "SURVIVES" if (cond1 and cond2 and cond3) else "KILLED"
        dir_block = None
        if da_dir:
            da = np.array(da_dir); dir_p10 = float(np.percentile(da, 10))
            dir_block = dict(emb_only_auc_mean=round(float(da.mean()), 4), emb_only_auc_p10=round(dir_p10, 4),
                             n_paths=len(da), VERDICT=("SURVIVES" if dir_p10 > DIR_AUC_BAR else "KILLED"))
        out["by_horizon"][str(H)] = dict(
            n_paths=len(ba),
            magnitude=dict(base_auc_mean=round(float(ba.mean()), 4), base_auc_p10=round(base_p10, 4),
                           base_plus_emb_auc_mean=round(float(ea.mean()), 4), base_plus_emb_auc_p10=round(emb_p10, 4),
                           emb_only_auc_mean=round(float(eo.mean()), 4),
                           dAUC_mean=round(float(dd.mean()), 4), dAUC_ci95=list(dci),
                           disp_lift_mean=round(float(np.mean(dlift)), 3), VERDICT=mag_verdict),
            direction=dir_block)
        dl = f"dir emb-only AUC {dir_block['emb_only_auc_mean']}(p10 {dir_block['emb_only_auc_p10']})->{dir_block['VERDICT']}" if dir_block else "dir n/a"
        hb(f"H={H}m MAG base {ba.mean():.4f}->+emb {ea.mean():.4f} dAUC {dd.mean():+.4f} CI{dci} embonly {eo.mean():.4f} ->{mag_verdict} | {dl}")
    json.dump(out, open(RES, "w"), indent=1); hb(f"ablation DONE -> {RES}")
    return out


if __name__ == "__main__":
    extract()
    ablate()
