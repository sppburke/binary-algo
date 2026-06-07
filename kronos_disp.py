"""LEVER 1 — Kronos per-path DISPERSION as a forward-volatility / MAGNITUDE feature.

Kronos already samples K OHLCV paths per window and DISCARDS the dispersion (kronos.py:467 mean-collapses the
sample_count paths before they ever reach us; kronos_mtf.py reads only the mean close). This harness taps the
per-path tensor BEFORE the collapse and turns its dispersion into FORWARD-looking volatility features, then asks the
pre-registered question:

  Does Kronos forward per-path dispersion ADD magnitude-predictive power BEYOND the backward-looking realized-vol
  features (rv30, rv120) that already carry the certified 30m magnitude edge (cpcv_certify.py, |ret30|>=train-Q75,
  CPCV mean AUC 0.744 deflated 0.712)?

Faithful by construction — it reuses cpcv_certify's EXACT target (|ret_H| vs train-Q75), baseline predictors
([-pe, rv30, rv120]), CPCV design (8 groups, k=2 -> C(8,2)=28 purged paths, purge+embargo=1 label-horizon), LGBM
(mk_lgb 600), and deflation. The ONLY scope reduction vs the full certification: decision bars are a NONOVERLAPPING
SUBSAMPLE (N_PER_YR/yr, gap=H_max) over the available 1-min OHLCV cache (2021-2026) so K-sample x 30-step Kronos
generation is tractable on the 8GB GPU. This is the LEVER test, not a re-certification; if it WINS we scale up.

From ONE pred_len=H_MAX generation we derive dispersion features for EVERY sub-horizon H in HORIZONS (cumulative to
step H) and run the ablation per H against that H's own |ret_H| target. Per-path denormalization is done in PRICE
space per sample (kronos.py collapses in NORMALIZED space, which is wrong for dispersion).

PRE-REGISTERED FALSIFIER (per horizon H): KILL the dispersion lever unless
  (1) baseline {-pe,rv30,rv120} arm reproduces a real edge (path-mean AUC >= 0.60) — sanity that rv carries signal;
  (2) paired mean dAUC(+disp - base) over the 28 CPCV paths > +0.005 with bootstrap CI95 EXCLUDING 0;
  (3) +disp path-p10 >= baseline path-p10 (no worst-path regression).
If dAUC CI includes 0 (or <0.005), the dispersion just DUPLICATES backward rv -> KILLED.

Run (extraction is GPU + long; ablation is CPU + fast; both resume from the cache npz):
  ~/binary-algo-venv/bin/python kronos_disp.py [N_PER_YR=2500] [K=24] [L=256] [H_MAX=30] [model] [tag]
Artifacts: kronos_disp_<tag>.npz (per-window dispersion + baseline + labels, checkpointed) ;
           kronos_disp_<tag>_result.json (per-horizon ablation + deflation + verdict).
"""
import sys, os, json, time, math, itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/sean/git/Kronos")

ROOT = "/media/sean/CORSAIR/binary-algo"; OUT = f"{ROOT}/ohlc_cache"
N_PER_YR = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
K        = int(sys.argv[2]) if len(sys.argv) > 2 else 24
L        = int(sys.argv[3]) if len(sys.argv) > 3 else 256
H_MAX    = int(sys.argv[4]) if len(sys.argv) > 4 else 30
MODEL    = sys.argv[5] if len(sys.argv) > 5 else "NeoQuasar/Kronos-small"
TAG      = sys.argv[6] if len(sys.argv) > 6 else f"disp_K{K}_N{N_PER_YR}_L{L}"
TOK_ID   = "NeoQuasar/Kronos-Tokenizer-base"
HORIZONS = [h for h in (1, 5, 10, 15, 30) if h <= H_MAX]
STEP = 60; TOPP = 0.9; BATCH = 16; CKPT_BATCHES = 50
NPZ = f"{ROOT}/kronos_disp_{TAG}.npz"; RES = f"{ROOT}/kronos_disp_{TAG}_result.json"
DISP_NAMES = ["term_std", "term_absmean", "term_q9010", "term_iqr", "range_mean", "wpv_mean"]
ND = len(DISP_NAMES)
BREAKEVEN_BAR = 0.55          # magnitude "0.5 + meaningful edge" bar (matches cpcv_certify mag bar)
N_TRIALS = 70; CORR_VAL_OOS = -0.54
RNG = np.random.default_rng(7)
T0 = time.time()
def hb(m): print(f"HB[{time.time()-T0:.0f}s] {m}", flush=True)


# ============================================================= baseline features (IDENTICAL to cpcv_certify.load_magnitude)
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


# ============================================================= per-path Kronos generation (un-collapsed; price space)
def _ar_samples(tokenizer, model, x, x_stamp, y_stamp, max_context, pred_len, clip, T, top_k, top_p, sample_count):
    """COPY of kronos.auto_regressive_inference, returning the FULL per-path tensor (B, sample_count, total_seq, 6)
    in NORMALIZED space — i.e. WITHOUT the np.mean(axis=1) collapse at kronos.py:467. Everything else is identical."""
    import torch
    from model.kronos import sample_from_logits
    with torch.no_grad():
        x = torch.clip(x, -clip, clip)
        device = x.device
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2)).to(device)
        x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2)).to(device)
        y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2)).to(device)
        x_token = tokenizer.encode(x, half=True)
        initial_seq_len = x.size(1); batch_size = x_token[0].size(0); total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)
        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)
        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]
        for i in range(pred_len):
            current_seq_len = initial_seq_len + i; window_len = min(current_seq_len, max_context)
            if current_seq_len <= max_context:
                input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
            else:
                input_tokens = [pre_buffer, post_buffer]
            context_end = current_seq_len; context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()
            s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
            s1_logits = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)
            s2_logits = model.decode_s2(context, sample_pre); s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)
            generated_pre[:, i] = sample_pre.squeeze(-1); generated_post[:, i] = sample_post.squeeze(-1)
            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1); post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1)); post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1); post_buffer[:, -1] = sample_post.squeeze(-1)
        full_pre = torch.cat([x_token[0], generated_pre], dim=1); full_post = torch.cat([x_token[1], generated_post], dim=1)
        context_start = max(0, total_seq_len - max_context)
        input_tokens = [full_pre[:, context_start:total_seq_len].contiguous(), full_post[:, context_start:total_seq_len].contiguous()]
        z = tokenizer.decode(input_tokens, half=True)
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))          # (B, sample_count, total_seq, 6)  — NO np.mean
        return z.cpu().numpy()


def kronos_path_samples(pred, dfl, xtl, ytl, pred_len, sample_count, top_p):
    """Replicate predict_batch's per-series normalization, generate UN-collapsed samples, denormalize PER-SAMPLE in
    price space. Returns (B, sample_count, pred_len, 6) numpy in PRICE space. Mirrors kronos.py:597-657."""
    import torch
    from model.kronos import calc_time_stamps
    pcols = pred.price_cols + [pred.vol_col, pred.amt_vol]
    xs, xss, yss, means, stds = [], [], [], [], []
    for df, xt, yt in zip(dfl, xtl, ytl):
        df = df.copy()
        if pred.vol_col not in df.columns: df[pred.vol_col] = 0.0; df[pred.amt_vol] = 0.0
        if pred.amt_vol not in df.columns: df[pred.amt_vol] = df[pred.vol_col] * df[pred.price_cols].mean(axis=1)
        x = df[pcols].values.astype(np.float32)
        xst = calc_time_stamps(xt).values.astype(np.float32); yst = calc_time_stamps(yt).values.astype(np.float32)
        mu = np.mean(x, axis=0); sd = np.std(x, axis=0)
        xn = np.clip((x - mu) / (sd + 1e-5), -pred.clip, pred.clip)
        xs.append(xn); xss.append(xst); yss.append(yst); means.append(mu); stds.append(sd)
    xb = torch.from_numpy(np.stack(xs).astype(np.float32)).to(pred.device)
    xsb = torch.from_numpy(np.stack(xss).astype(np.float32)).to(pred.device)
    ysb = torch.from_numpy(np.stack(yss).astype(np.float32)).to(pred.device)
    z = _ar_samples(pred.tokenizer, pred.model, xb, xsb, ysb, pred.max_context, pred_len,
                    pred.clip, 1.0, 0, top_p, sample_count)            # (B, K, total_seq, 6) normalized
    z = z[:, :, -pred_len:, :]                                          # keep only the forward bars
    out = np.empty_like(z)
    for i in range(len(dfl)):                                           # denormalize PER-SAMPLE (price space)
        out[i] = z[i] * (stds[i] + 1e-5) + means[i]
    return out


def disp_features(samp, c0):
    """samp: (K, pred_len, 6) PRICE space for one window. c0 = entry close. Return (n_horizons, ND) cumulative
    forward dispersion features per horizon in HORIZONS. Cols: open0 high1 low2 close3 vol4 amount5."""
    K_, P_, _ = samp.shape
    close = samp[:, :, 3]; high = samp[:, :, 1]; low = samp[:, :, 2]
    feats = np.full((len(HORIZONS), ND), np.nan, np.float32)
    for hi, H in enumerate(HORIZONS):
        s = slice(0, H)
        rT = np.log(np.maximum(close[:, H - 1], 1e-12) / max(c0, 1e-12))      # terminal log-ret per path
        term_std = float(np.std(rT)); term_absmean = float(np.mean(np.abs(rT)))
        q10, q25, q75, q90 = np.percentile(rT, [10, 25, 75, 90])
        term_q9010 = float(q90 - q10); term_iqr = float(q75 - q25)
        rng = (np.max(high[:, s], axis=1) - np.min(low[:, s], axis=1)) / max(c0, 1e-12)   # per-path path range frac
        range_mean = float(np.mean(rng))
        lc = np.log(np.maximum(np.concatenate([np.full((K_, 1), c0), close[:, s]], axis=1), 1e-12))
        step = np.diff(lc, axis=1)                                            # per-step log-rets along each path
        wpv = np.std(step, axis=1) if H > 1 else np.abs(step[:, 0])
        wpv_mean = float(np.mean(wpv))
        feats[hi] = [term_std, term_absmean, term_q9010, term_iqr, range_mean, wpv_mean]
    return feats


# ============================================================= decision-bar selection
def load_bars():
    frames = [pd.read_parquet(f"{OUT}/EURUSD_1m_{sp}.parquet") for sp in ("train", "val", "test", "oos")]
    B = pd.concat(frames).reset_index(drop=True)
    B = B.drop_duplicates(subset="t").sort_values("t").reset_index(drop=True)
    return B


def select_decision_bars(t):
    n = len(t); ar = np.arange(n)
    loi = ar - L + 1; hii = ar + H_MAX
    base = (loi >= 0) & (hii < n)
    ctx_ok = np.zeros(n, bool); ctx_ok[base] = (t[ar[base]] - t[loi[base]]) == (L - 1) * STEP    # gap-free context (covers rv120/pe)
    fwd_ok = np.zeros(n, bool); fwd_ok[base] = (t[hii[base]] - t[ar[base]]) == H_MAX * STEP        # gap-free forward (covers all H labels)
    elig = np.where(base & ctx_ok & fwd_ok)[0]
    yr = pd.to_datetime(t, unit="s", utc=True).year.values
    pick = []
    for Y in sorted(set(yr[elig].tolist())):
        ey = elig[yr[elig] == Y]
        if len(ey) > N_PER_YR: ey = ey[np.linspace(0, len(ey) - 1, N_PER_YR).astype(int)]
        pick.append(ey)
    pick = np.sort(np.concatenate(pick))
    take = []; bu = -1; gap = H_MAX * STEP                                                        # nonoverlap by H_max
    for i in pick:
        if t[i] < bu: continue
        take.append(i); bu = int(t[i]) + gap
    return np.array(take, dtype=int)


# ============================================================= extraction (GPU, checkpointed)
def extract():
    B = load_bars(); t = B["t"].values.astype("int64")
    O = B["open"].values.astype(float); Hg = B["high"].values.astype(float)
    Lw = B["low"].values.astype(float); C = B["close"].values.astype(float); V = B["vol"].values.astype(float)
    n = len(B); hb(f"bars n={n:,} span {pd.to_datetime(t[0],unit='s')}..{pd.to_datetime(t[-1],unit='s')}")
    pick = select_decision_bars(t); Np = len(pick); hb(f"decision bars (nonoverlap gap={H_MAX*STEP}s): {Np}")
    yr = pd.to_datetime(t[pick], unit="s", utc=True).year.values

    # baseline features computed on full series (decision bars have gap-free trailing window -> values valid there)
    r = np.zeros(n); r[1:] = np.diff(np.log(C)); rs = pd.Series(r)
    pe = perm_entropy(r, 4, 1, 120); rv30 = rs.rolling(30).std().values; rv120 = rs.rolling(120).std().values
    base3 = np.column_stack([-pe[pick], rv30[pick], rv120[pick]]).astype(np.float32)             # [-pe, rv30, rv120]
    aretH = {H: np.abs(C[pick + H] / C[pick] - 1.0).astype(np.float32) for H in HORIZONS}        # |ret_H| label per horizon

    disp = np.full((Np, len(HORIZONS), ND), np.nan, np.float32); filled = np.zeros(Np, bool)
    start = 0
    if os.path.exists(NPZ):
        d = np.load(NPZ)
        if len(d["pick_t"]) == Np and np.array_equal(d["pick_t"], t[pick]):
            disp = d["disp"].copy(); filled = d["filled"].copy(); start = int(filled.sum())
            hb(f"RESUME: {start}/{Np} windows already extracted")
        else:
            hb("existing npz mismatches current pick params -> fresh extraction")

    import torch; torch.set_num_threads(8)
    from model import Kronos, KronosTokenizer, KronosPredictor
    DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
    tok = KronosTokenizer.from_pretrained(TOK_ID); mdl = Kronos.from_pretrained(MODEL)
    pred = KronosPredictor(mdl, tok, device=DEV, max_context=max(L, 256))
    hb(f"loaded {sum(p.numel() for p in mdl.parameters())/1e6:.1f}M params on {DEV}; K={K} pred_len={H_MAX}")

    cols = ["open", "high", "low", "close", "volume", "amount"]
    def save():
        tmp = NPZ.replace(".npz", ".tmp.npz")           # np.savez appends .npz; keep tmp ending in .npz so os.replace finds it
        np.savez(tmp, pick_t=t[pick], year=yr, base3=base3, disp=disp, filled=filled,
                 **{f"aret_{H}": aretH[H] for H in HORIZONS},
                 horizons=np.array(HORIZONS), disp_names=np.array(DISP_NAMES))
        os.replace(tmp, NPZ)

    nb = 0
    for s in range(0, Np, BATCH):
        chunk = pick[s:s + BATCH]
        if filled[s:s + len(chunk)].all():                              # already done (resume)
            continue
        dfl, xtl, ytl = [], [], []
        for i in chunk:
            sl = slice(i - L + 1, i + 1)                                # context INCLUDES decision bar i; close C[i] = entry ref
            dfl.append(pd.DataFrame({"open": O[sl], "high": Hg[sl], "low": Lw[sl], "close": C[sl],
                                     "volume": V[sl], "amount": 0.0})[cols])
            xtl.append(pd.Series(pd.to_datetime(t[sl], unit="s", utc=True)))
            ytl.append(pd.Series(pd.to_datetime(t[i + 1:i + 1 + H_MAX], unit="s", utc=True)))   # forward H_MAX bars
        samp = kronos_path_samples(pred, dfl, xtl, ytl, H_MAX, K, TOPP)  # (B,K,H_MAX,6) price space
        for j, i in enumerate(chunk):
            disp[s + j] = disp_features(samp[j], float(C[i])); filled[s + j] = True
        nb += 1
        if nb % CKPT_BATCHES == 0:
            save(); hb(f"  extracted {int(filled.sum())}/{Np}  (ckpt saved)")
    save(); hb(f"extraction DONE {int(filled.sum())}/{Np} -> {NPZ}")


# ============================================================= CPCV + deflation (re-implemented from cpcv_certify)
def cpcv_groups(n, ng=8):
    e = np.linspace(0, n, ng + 1).astype(int); g = np.zeros(n, np.int8)
    for k in range(ng): g[e[k]:e[k + 1]] = k
    return g

def cpcv_paths(g, ts, horizon_min, ng=8, k=2):
    hs = horizon_min * 60; emb = horizon_min * 60; alli = np.arange(len(g))
    for combo in itertools.combinations(range(ng), k):
        tm = np.isin(g, combo); te = alli[tm]; tr = alli[~tm]
        keep = np.ones(len(tr), bool); tt = ts[tr]
        for grp in combo:
            gi = alli[g == grp]; lo, hi = ts[gi[0]], ts[gi[-1]]
            keep &= ~((tt >= lo - hs) & (tt <= hi + emb))
        yield combo, tr[keep], te

def mk_lgb(n_estimators=600):
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.03, num_leaves=255,
        min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10,
        n_estimators=n_estimators, n_jobs=20, verbosity=-1)

def deflated(headline, dist, bar, n_trials=N_TRIALS, corr=CORR_VAL_OOS):
    from scipy.stats import norm
    a = np.asarray(dist, float); a = a[np.isfinite(a)]
    mu = float(a.mean()); sd = float(a.std(ddof=1)) if len(a) > 1 else float("nan")
    p10 = float(np.percentile(a, 10)); emax = sd * math.sqrt(2 * math.log(max(n_trials, 2))) if np.isfinite(sd) else float("nan")
    anti = abs(corr) * emax if np.isfinite(emax) else float("nan"); de = mu - anti
    return dict(headline=headline, bar=bar, path_mean=round(mu, 4), path_p10=round(p10, 4),
                path_min=round(float(a.min()), 4), path_max=round(float(a.max()), 4),
                emax_inflation=round(emax, 4), neg_corr_penalty=round(anti, 4),
                deflated_expectation=round(de, 4), p10_clears_bar=bool(p10 > bar),
                deflated_exp_clears_bar=bool(de > bar))

def boot_ci(x, nb=5000, seed=11):
    x = np.asarray(x, float); rng = np.random.default_rng(seed); n = len(x)
    a = np.array([x[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)


def ablate():
    if not os.path.exists(NPZ): hb(f"no {NPZ} — run extraction first"); return
    d = np.load(NPZ); filled = d["filled"]; ts = d["pick_t"][filled]
    base3 = d["base3"][filled]; disp = d["disp"][filled]; horizons = list(d["horizons"])
    order = np.argsort(ts, kind="stable"); ts = ts[order]; base3 = base3[order]; disp = disp[order]
    n = len(ts); hb(f"ablation on {n} extracted windows; horizons={horizons}")
    out = {"tag": TAG, "model": MODEL, "K": K, "L": L, "N_PER_YR": N_PER_YR, "H_MAX": H_MAX,
           "n_windows": int(n), "disp_names": DISP_NAMES, "bar": BREAKEVEN_BAR,
           "scope": "SUBSAMPLE ablation (nonoverlap, 2021-2026 1m cache) — lever test, NOT a full re-certification",
           "falsifier": "KILL horizon H unless base path-mean AUC>=0.60 AND paired mean dAUC>+0.005 with CI95 excl 0 AND +disp p10>=base p10",
           "by_horizon": {}}
    from sklearn.metrics import roc_auc_score
    g = cpcv_groups(n, 8)
    for hi, H in enumerate(horizons):
        aret = d[f"aret_{H}"][filled][order]
        Xb = base3; Xd = np.column_stack([base3, disp[:, hi, :]])
        ok = np.isfinite(Xd).all(1) & np.isfinite(aret)
        base_aucs, disp_aucs, dlift = [], [], []
        for combo, tri, tei in cpcv_paths(g, ts, H):
            tri = tri[ok[tri]]; tei = tei[ok[tei]]
            if len(tri) > 100_000: tri = np.sort(RNG.choice(tri, 100_000, replace=False))
            if len(tri) < 200 or len(tei) < 100: continue
            thr = np.nanquantile(aret[tri], 0.75)
            ytr = (aret[tri] >= thr).astype(int); yte = (aret[tei] >= thr).astype(int)
            if ytr.mean() in (0.0, 1.0) or yte.mean() in (0.0, 1.0): continue
            mb = mk_lgb(); mb.fit(Xb[tri], ytr); pb = mb.predict_proba(Xb[tei])[:, 1]
            md = mk_lgb(); md.fit(Xd[tri], ytr); pdd = md.predict_proba(Xd[tei])[:, 1]
            ab = roc_auc_score(yte, pb); ad = roc_auc_score(yte, pdd)
            base_aucs.append(ab); disp_aucs.append(ad)
            q = np.quantile(pdd, [0.9, 0.1]); hiq = pdd >= q[0]; loq = pdd <= q[1]
            dlift.append(float(aret[tei][hiq].mean() / max(aret[tei][loq].mean(), 1e-12)))
        if not base_aucs:
            out["by_horizon"][str(H)] = {"note": "no valid paths"}; continue
        ba = np.array(base_aucs); da = np.array(disp_aucs); dd = da - ba
        dci = boot_ci(dd); base_p10 = float(np.percentile(ba, 10)); disp_p10 = float(np.percentile(da, 10))
        # full-data dispersion importance + correlation to rv30 (orthogonality check)
        thr_all = np.nanquantile(aret[ok], 0.75); yall = (aret[ok] >= thr_all).astype(int)
        mfull = mk_lgb(); mfull.fit(Xd[ok], yall); gains = mfull.booster_.feature_importance("gain")
        names = ["neg_pe", "rv30", "rv120"] + DISP_NAMES
        imp = {nm: round(float(gi), 1) for nm, gi in zip(names, gains)}
        rv30col = base3[ok, 1]
        corr_rv30 = {DISP_NAMES[c]: round(float(np.corrcoef(disp[ok, hi, c], rv30col)[0, 1]), 3) for c in range(ND)}
        cond1 = float(ba.mean()) >= 0.60; cond2 = (float(dd.mean()) > 0.005) and (dci[0] > 0.0); cond3 = disp_p10 >= base_p10
        verdict = "SURVIVES" if (cond1 and cond2 and cond3) else "KILLED"
        out["by_horizon"][str(H)] = dict(
            n_paths=len(ba), base_auc_mean=round(float(ba.mean()), 4), base_auc_p10=round(base_p10, 4),
            disp_auc_mean=round(float(da.mean()), 4), disp_auc_p10=round(disp_p10, 4),
            dAUC_mean=round(float(dd.mean()), 4), dAUC_ci95=list(dci), dAUC_min=round(float(dd.min()), 4),
            dAUC_max=round(float(dd.max()), 4), disp_lift_mean=round(float(np.mean(dlift)), 3),
            base_deflated=deflated(float(ba.mean()), ba, BREAKEVEN_BAR),
            disp_deflated=deflated(float(da.mean()), da, BREAKEVEN_BAR),
            feature_gain=imp, disp_corr_rv30=corr_rv30,
            cond_base_real=cond1, cond_dAUC_positive=cond2, cond_no_p10_regress=cond3, VERDICT=verdict)
        hb(f"H={H}m: base {ba.mean():.4f}(p10 {base_p10:.4f}) -> +disp {da.mean():.4f}(p10 {disp_p10:.4f}) "
           f"dAUC={dd.mean():+.4f} CI{dci} lift={np.mean(dlift):.2f}x -> {verdict}")
    json.dump(out, open(RES, "w"), indent=1)
    hb(f"ablation DONE -> {RES}")
    return out


if __name__ == "__main__":
    extract()
    ablate()
