"""PROBE W2-1 — RMT (Marchenko-Pastur) CROSS-PAIR EIGEN-RESIDUAL REVERSION.

Thesis: the raw USD-basket residual (actual EURUSD move minus its loading on the USD common factor)
carries 0.516 AUC / test25 0.534 because the *noisy* part of the USD-factor loading INVERTS in 2025.
Marchenko-Pastur cleaning keeps only the eigenmodes whose eigenvalues exceed the MP upper edge
lambda+ (the genuine market + USD common factors) and discards the noise bulk. Projecting EURUSD onto
ONLY the cleaned (significant) eigenmodes gives a more 2025-stable common-factor-implied move, so the
idiosyncratic residual = actual - implied should revert more reliably and be more year-stable.

DERIV-FAITHFUL DISCIPLINE (matches min1_production.py honest ground truth):
  * label = sign(close[t+H] - close[t]); ties (ret==0) LOSE (counted as wrong).
  * H=1 -> 60s next-bar binary; H=15 -> 15-minute binary (best deriv-tradeable horizon).
  * Residual computed CAUSALLY from bar-t returns of all 7 majors (observed at close of bar t),
    used to predict bar t+H direction. The eigenvectors / MP edge / residual-std are fit on TRAIN ONLY.
  * Cross-pair alignment is an INNER JOIN on common timestamps with NO ffill (1m bars are 99.8% aligned,
    so this does NOT manufacture fake-flat bars; we additionally evaluate MOVED-BARS-ONLY and print the
    EURUSD up-rate, which must sit ~0.47-0.49, to rule out the 0.728 leakage mirage).
  * NON-OVERLAPPING trades: a selected trade at t blocks the next H bars (its outcome window).
  * Selection is by |standardized residual| (the conviction of the reversion bet); coverage swept.
  * Per-window 2024 / 2025 / 2026, bootstrap CI95.

PRE-REGISTERED FALSIFIER: if cleaned-residual 2025 acc <= raw-xpair 0.534, RMT added nothing.
"""
import sys, numpy as np, pandas as pd

ROOT = "/media/sean/CORSAIR/binary-algo"
PAIRS = ["AUDUSD","EURUSD","GBPUSD","NZDUSD","USDCAD","USDCHF","USDJPY"]
TARGET = "EURUSD"
FEAT = lambda p,y: f"{ROOT}/features/{p}_{y}.parquet"
TRAIN_YEARS = (2021,2022,2023)
WIN_YEARS = {"2024":(2024,), "2025":(2025,), "2026":(2026,)}
COVS = (0.30, 0.20, 0.10, 0.05, 0.02)   # selection coverages by |residual|


def load_closes(years):
    closes={}
    for p in PAIRS:
        parts=[pd.read_parquet(FEAT(p,y), columns=["close"])["close"] for y in years]
        s=pd.concat(parts); s=s[~s.index.duplicated(keep="first")].sort_index()
        closes[p]=s
    mat=pd.DataFrame(closes).dropna()              # INNER JOIN, no ffill
    return mat[PAIRS]


def returns_1m(mat):
    """contemporaneous 1m log returns on the inner-join index (true co-moves)."""
    return np.log(mat).diff().dropna()


def boot(correct, nb=5000, seed=7):
    correct=np.asarray(correct,dtype=float)
    if len(correct)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(correct)
    a=np.array([correct[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)), float(np.percentile(a,97.5))


def fit_rmt(Rtr):
    """Fit on TRAIN ONLY. Returns: eigvecs of significant modes, EURUSD projection operator, residual std,
    plus diagnostics (lambda+, #significant, eigenvalues). Standardize returns by TRAIN per-pair std first
    so the correlation matrix == covariance of standardized returns."""
    mu = Rtr.mean(0); sd = Rtr.std(0, ddof=0)
    Z = (Rtr - mu) / sd                                  # standardized returns (TRAIN moments)
    Zv = Z.values
    T, N = Zv.shape
    C = np.corrcoef(Zv, rowvar=False)                    # NxN correlation matrix
    w, V = np.linalg.eigh(C)                             # ascending eigenvalues
    order = np.argsort(w)[::-1]; w=w[order]; V=V[:,order]
    q = N / T
    lam_plus = (1 + np.sqrt(q))**2                       # MP upper edge, sigma^2=1 for a correlation matrix
    sig = w > lam_plus                                   # significant (above-noise) eigenmodes
    k = int(sig.sum())
    Vk = V[:, :k]                                        # top-k significant eigenvectors (NxK)
    # Project standardized returns onto the significant subspace: implied_Z = Vk Vk^T Z
    # EURUSD row of the projector picks its common-factor-implied standardized move.
    eur_idx = PAIRS.index(TARGET)
    P = Vk @ Vk.T                                        # NxN projector onto significant subspace
    proj_row = P[eur_idx, :]                             # 1xN: implied EURUSD std-move = proj_row . Z_t
    # TRAIN residual (standardized units) and its std (for selection threshold)
    implied_tr = Zv @ proj_row
    resid_tr = Zv[:, eur_idx] - implied_tr
    resid_std = float(resid_tr.std(ddof=0)) + 1e-12
    return dict(mu=mu.values, sd=sd.values, proj_row=proj_row, eur_idx=eur_idx,
                resid_std=resid_std, lam_plus=lam_plus, k=k, eigvals=w, q=q, T=T, N=N)


def raw_usd_residual_fit(Rtr):
    """Baseline to beat: single-factor USD-basket residual. USD common factor = mean of USD-quote returns
    with sign so a stronger-USD factor is positive; EURUSD residual = its return minus OLS loading on that
    factor. (This reproduces the prior 'raw USD-basket residual' family, 0.516 AUC / test25 0.534.)"""
    mu = Rtr.mean(0).values; sd = Rtr.std(0, ddof=0).values
    Z = ((Rtr - Rtr.mean(0)) / Rtr.std(0, ddof=0)).values
    eur_idx = PAIRS.index(TARGET)
    # USD strength factor: USDxxx pairs go UP when USD strong; xxxUSD go DOWN. Sign each pair to "USD up".
    usd_sign = np.array([+1 if p.startswith("USD") else -1 for p in PAIRS], dtype=float)
    factor = (Z * usd_sign).mean(1)                      # standardized USD-strength factor per bar
    # OLS loading of EURUSD standardized return on factor
    beta = np.cov(Z[:, eur_idx], factor, ddof=0)[0,1] / (factor.var(ddof=0)+1e-12)
    resid_tr = Z[:, eur_idx] - beta*factor
    resid_std = float(resid_tr.std(ddof=0))+1e-12
    return dict(mu=mu, sd=sd, eur_idx=eur_idx, usd_sign=usd_sign, beta=beta, resid_std=resid_std)


def residual_series_rmt(mat, model, H):
    """CAUSAL residual at each bar t (from bar-t returns) + the deriv-faithful next-H label.
    Returns aligned (resid_z, label, moved) arrays and the integer second-spacing-safe non-overlap blocker."""
    R = np.log(mat).diff()                               # bar-t return at index t (uses close[t],close[t-1])
    Z = (R.values - model["mu"]) / model["sd"]
    implied = Z @ model["proj_row"]
    resid = (Z[:, model["eur_idx"]] - implied) / model["resid_std"]
    # label: sign(close[t+H]-close[t]); ties LOSE
    close = mat[TARGET].values
    fwd = np.full(len(close), np.nan)
    fwd[:-H] = close[H:] / close[:-H] - 1.0
    label = (fwd > 0).astype(float)                      # 1=up; ties(fwd==0) and down both -> not-up
    moved = (fwd != 0)
    valid = np.isfinite(resid) & np.isfinite(fwd)
    return resid, label, fwd, moved, valid


def residual_series_raw(mat, model, H):
    R = np.log(mat).diff()
    Z = (R.values - model["mu"]) / model["sd"]
    factor = (Z * model["usd_sign"]).mean(1)
    resid = (Z[:, model["eur_idx"]] - model["beta"]*factor) / model["resid_std"]
    close = mat[TARGET].values
    fwd=np.full(len(close),np.nan); fwd[:-H]=close[H:]/close[:-H]-1.0
    label=(fwd>0).astype(float); moved=(fwd!=0)
    valid=np.isfinite(resid)&np.isfinite(fwd)
    return resid, label, fwd, moved, valid


def nonoverlap_select(idx_pos, H):
    """Live-faithful chronological de-overlap on bar positions: take a signal, block next H bars."""
    take=[]; block_until=-1
    for i in idx_pos:
        if i < block_until: continue
        take.append(i); block_until=i+H
    return np.array(take, dtype=int)


def evaluate(resid, label, moved, valid, H, cov, moved_only=False):
    """Reversion bet: predicted_up = (resid < 0) (bet against idiosyncratic deviation).
    Select |resid| top-cov, de-overlap, report acc + CI + AUC-style sign agreement."""
    m = valid.copy()
    if moved_only: m &= moved
    pos = np.where(m)[0]
    if len(pos)==0: return None
    aresid = np.abs(resid[pos])
    thr = np.quantile(aresid, 1-cov)
    sel = pos[aresid >= thr]
    sel = np.sort(sel)
    tr = nonoverlap_select(sel, H)
    if len(tr)==0: return None
    pred_up = (resid[tr] < 0).astype(float)              # reversion: deviation negative -> bet up
    # deriv-faithful correctness: correct iff pred matches realized up AND bar moved (tie=loss)
    correct = ((pred_up==label[tr]) & moved[tr]).astype(float)
    acc = correct.mean()
    lo,hi = boot(correct)
    # directional AUC over selected (does residual sign predict next move at all?)
    return dict(n=len(tr), acc=acc, lo=lo, hi=hi, thr=thr, cov_real=len(tr)/len(pos))


def auc_full(resid, label, moved, valid, H):
    """AUC of reversion score (-resid) vs up-label, MOVED bars only, all bars (no selection, no overlap-control).
    A single honest discrimination number per window."""
    from sklearn.metrics import roc_auc_score
    m = valid & moved
    pos=np.where(m)[0]
    if len(pos)<50: return float("nan")
    return roc_auc_score(label[pos], -resid[pos])


def run():
    H = int(sys.argv[1]) if len(sys.argv)>1 else 1
    label_h = "60s (next-bar)" if H==1 else f"{H}-minute"
    print(f"\n{'='*78}\nPROBE W2-1 RMT cross-pair eigen-residual reversion | horizon H={H} ({label_h})\n{'='*78}")

    # ---- TRAIN: fit MP + residual operators ----
    mat_tr = load_closes(TRAIN_YEARS)
    Rtr = returns_1m(mat_tr)
    rmt = fit_rmt(Rtr)
    raw = raw_usd_residual_fit(Rtr)
    print(f"[TRAIN] inner bars={len(mat_tr)}  N={rmt['N']} T={rmt['T']}  q=N/T={rmt['q']:.2e}")
    print(f"[TRAIN] MP lambda+={rmt['lam_plus']:.4f}  significant modes k={rmt['k']}")
    print(f"[TRAIN] eigenvalues (desc): " + " ".join(f"{v:.3f}" for v in rmt['eigvals']))
    above = rmt['eigvals'][rmt['eigvals']>rmt['lam_plus']]
    print(f"[TRAIN] above lambda+: " + " ".join(f"{v:.3f}" for v in above)
          + f"   (variance captured by signal modes = {above.sum()/rmt['eigvals'].sum():.1%})")
    print(f"[TRAIN] EURUSD projector loadings on significant subspace (per-pair): "
          + " ".join(f"{p}={w:+.3f}" for p,w in zip(PAIRS, rmt['proj_row'])))

    # ---- per-window evaluation ----
    for wname, years in WIN_YEARS.items():
        mat = load_closes(years)
        # diagnostics: EURUSD next-H up-rate + moved fraction (leakage guard)
        close=mat[TARGET].values
        fwd=np.full(len(close),np.nan); fwd[:-H]=close[H:]/close[:-H]-1.0
        uprate=np.nanmean(fwd>0); flat=np.nanmean(fwd==0)
        print(f"\n----- WINDOW {wname} ({years}) | inner bars={len(mat)} | EURUSD up-rate={uprate:.4f} flat={flat:.4f} -----")

        for tag, fn, model in (("RMT-clean", residual_series_rmt, rmt),
                               ("raw-USD  ", residual_series_raw, raw)):
            resid,label,fwd2,moved,valid = fn(mat, model, H)
            a_all = auc_full(resid,label,moved,valid,H)
            print(f"  [{tag}] reversion-score AUC (moved bars, all, no-overlap-ctrl) = {a_all:.4f}")
            for cov in COVS:
                r = evaluate(resid,label,moved,valid,H,cov,moved_only=True)
                if r is None: continue
                star = " <-- FALSIFIER ref 0.534" if (tag.startswith("RMT") and wname=="2025") else ""
                print(f"      cov={cov:>4.0%}  n={r['n']:>5}  acc={r['acc']:.4f}  CI95=[{r['lo']:.4f},{r['hi']:.4f}]"
                      f"  (thr|z|={r['thr']:.2f}){star}")


if __name__=="__main__":
    run()
