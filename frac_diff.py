"""PHASE 3 / T2 — FRACTIONAL DIFFERENTIATION (FFD, López de Prado ch.5). GENUINELY UNTRIED (grep fracdiff = 0 hits).

The certified cross-pair edge is built on INTEGER-differenced (1-bar) log-returns, which are stationary but DESTROY the
long-memory level co-movement where a slow USD trend lives. Fractional differentiation with the SMALLEST d* that passes
ADF keeps maximal memory while being stationary — "stationarity WITH memory." The hypothesis: an FFD USD factor/residual
exposes a slow cross-pair trend the 1-bar return panel cannot see.

Fixed-Width Window FD (FFD): weights w_0=1, w_k = -w_{k-1}·(d-k+1)/k, truncated where |w_k| < thresh (so the window is
FIXED width -> every output uses the same number of past points -> no expanding-window leakage, fully causal). d* chosen
as the smallest d on a grid whose ADF p-value < 0.05 on the TRAIN block ONLY (then frozen — no per-fold refit).

This module is the pure transform + d*-selection. Integration (build FFD USD factor/residual panel, feed run_direction
at 15m/30m NY, gate by forward holdout) lives in the Phase-4 direction driver once the panel is validation-confirmed.

  from frac_diff import ffd_weights, ffd, min_ffd_d
  d_star, info = min_ffd_d(logclose_train)          # smallest stationary d on train
  x = ffd(logclose_full, d_star)                    # causal FFD of the full series (NaN warmup = window-1)
"""
import numpy as np


def ffd_weights(d, thresh=1e-5, max_k=10000):
    """Fixed-width FFD weights for order d, truncated where |w_k| < thresh. Returns w[0..K] (w[0]=1, most-recent last)."""
    w = [1.0]; k = 1
    while k < max_k:
        wk = -w[-1] * (d - k + 1) / k
        if abs(wk) < thresh: break
        w.append(wk); k += 1
    return np.array(w[::-1])                                # reverse: oldest..newest so np.dot with a trailing window works


def ffd(series, d, thresh=1e-5):
    """Causal fractional differentiation of a 1-D series at order d. Output[i] = Σ_k w_k · x[i-K+1+k]; NaN for i<K-1."""
    x = np.asarray(series, float); w = ffd_weights(d, thresh); K = len(w); n = len(x)
    out = np.full(n, np.nan)
    if n >= K:
        # sliding dot product via stride trick
        from numpy.lib.stride_tricks import sliding_window_view
        win = sliding_window_view(x, K)                    # shape (n-K+1, K), each row = [x[i-K+1..i]]
        out[K-1:] = win @ w
    return out


def _adf_p(x):
    """ADF p-value (statsmodels). Lower = more stationary; < 0.05 rejects unit root."""
    from statsmodels.tsa.stattools import adfuller
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 50: return 1.0
    try: return float(adfuller(x, maxlag=1, regression="c", autolag=None)[1])
    except Exception: return 1.0


def min_ffd_d(series_train, grid=None, thresh=1e-5, adf_target=0.05, corr_floor=0.0):
    """Smallest d on `grid` whose FFD(train) ADF p-value < adf_target. Reports the memory retained (corr of FFD vs level).
    Pass series_train = TRAIN-only log-close; freeze the returned d* for all downstream (no per-fold refit -> no leakage)."""
    if grid is None: grid = np.round(np.arange(0.0, 1.01, 0.05), 2)
    x = np.asarray(series_train, float); rows = []
    chosen = None
    for d in grid:
        f = ffd(x, d, thresh); fv = f[np.isfinite(f)]
        if len(fv) < 100: rows.append((float(d), 1.0, 0.0)); continue
        p = _adf_p(fv)
        lev = x[np.isfinite(f)]
        corr = float(np.corrcoef(fv, lev)[0, 1]) if fv.std() > 0 else 0.0   # memory retained vs the raw level
        rows.append((float(d), round(p, 5), round(corr, 4)))
        if chosen is None and p < adf_target and abs(corr) >= corr_floor:
            chosen = float(d)
    info = {"grid": [{"d": r[0], "adf_p": r[1], "corr_vs_level": r[2]} for r in rows],
            "d_star": chosen, "window_len": int(len(ffd_weights(chosen if chosen is not None else 1.0, thresh)))}
    return chosen, info


def _selfcheck():
    """A random WALK (unit root, non-stationary) should need d>0 to pass ADF; FFD at d* keeps strong level-correlation
    (memory) where full d=1 differencing (returns) has ~0 correlation with the level."""
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.standard_normal(20000)) + 100.0           # I(1) log-price proxy
    d_star, info = min_ffd_d(walk)
    # contrast memory: corr(level) at d* vs at d=1 (plain differencing)
    f1 = ffd(walk, 1.0); c1 = float(np.corrcoef(f1[np.isfinite(f1)], walk[np.isfinite(f1)])[0, 1])
    at = next(g for g in info["grid"] if g["d"] == d_star)
    print(f"random walk: d*={d_star} (window={info['window_len']})  ADF_p@d*={at['adf_p']}  corr_vs_level@d*={at['corr_vs_level']}")
    print(f"contrast: plain differencing d=1 -> corr_vs_level={c1:+.4f} (memory destroyed)")
    print("ADF p across grid:", [(g["d"], g["adf_p"]) for g in info["grid"][:8]], "...")
    ok = d_star is not None and 0.0 < d_star <= 1.0 and at["adf_p"] < 0.05 and abs(at["corr_vs_level"]) > abs(c1)
    print("SELFCHECK", "PASS" if ok else "FAIL", "(d* stationary AND retains more memory than plain returns)")


if __name__ == "__main__":
    _selfcheck()
