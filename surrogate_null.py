"""PHASE 1B — phase-randomized / IAAFT SURROGATE-NULL gate (reusable validation for nonlinear/TDA/signature features).

A feature can score well purely because it re-encodes linear autocorrelation/spectrum, not genuine nonlinear structure.
Surrogate data testing (Theiler 1992): generate surrogates that PRESERVE the power spectrum (hence autocorrelation) but
DESTROY nonlinear/phase structure, recompute the feature's OOS metric on each, and require the REAL metric to exceed the
surrogate null (e.g. 95th percentile). Two surrogate types:
  - phase_randomize: FFT, randomize phases, iFFT — preserves spectrum, destroys nonlinearity (fastest).
  - iaaft: iterative amplitude-adjusted FT — also preserves the amplitude DISTRIBUTION (stricter, for heavy-tailed returns).
For a CROSS-PAIR cloud, surrogate each pair INDEPENDENTLY to also kill genuine cross-pair coupling (tests whether a
cross-sectional feature uses real coupling vs each series' own spectrum).

Usage:
  from surrogate_null import phase_randomize, iaaft, surrogate_gate
  res = surrogate_gate(feature_metric_fn, series, n=300, kind="iaaft", seed=7)
  # feature_metric_fn(x)->scalar OOS metric (e.g. magAUC of the feature built from series x). res.passes if real>null p95.
"""
import numpy as np


def phase_randomize(x, rng):
    """Phase-randomized surrogate: same power spectrum, randomized Fourier phases. Preserves autocorrelation."""
    x = np.asarray(x, float); n = len(x); X = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2 * np.pi, len(X)); ph[0] = 0.0
    if n % 2 == 0: ph[-1] = 0.0                                   # Nyquist real
    Xs = np.abs(X) * np.exp(1j * ph)
    return np.fft.irfft(Xs, n=n) + x.mean()


def iaaft(x, rng, n_iter=100):
    """Iterative Amplitude-Adjusted FT surrogate: preserves BOTH the amplitude distribution AND the power spectrum."""
    x = np.asarray(x, float); n = len(x)
    amp = np.abs(np.fft.rfft(x - x.mean())); sorted_x = np.sort(x)
    s = rng.permutation(x)                                        # random start with same values
    for _ in range(n_iter):
        S = np.fft.rfft(s - s.mean()); ph = np.angle(S)
        s = np.fft.irfft(amp * np.exp(1j * ph), n=n) + x.mean()   # impose spectrum
        ranks = np.argsort(np.argsort(s)); s = sorted_x[ranks]    # impose amplitude distribution
    return s


def surrogate_gate(metric_fn, series, n=300, kind="iaaft", seed=7, n_iter=100):
    """Return real metric vs surrogate-null distribution. `series` may be a 1-D array OR a list of arrays (cross-pair:
    each surrogated independently). metric_fn takes the same shape and returns a scalar OOS metric."""
    rng = np.random.default_rng(seed)
    real = float(metric_fn(series))
    multi = isinstance(series, (list, tuple))
    surr_fn = iaaft if kind == "iaaft" else phase_randomize
    null = []
    for _ in range(n):
        if multi:
            s = [surr_fn(x, rng, n_iter) if kind == "iaaft" else surr_fn(x, rng) for x in series]
        else:
            s = surr_fn(series, rng, n_iter) if kind == "iaaft" else surr_fn(series, rng)
        null.append(float(metric_fn(s)))
    null = np.array(null)
    p95 = float(np.percentile(null, 95)); pct = float((null < real).mean())
    return dict(real=round(real, 5), null_mean=round(float(null.mean()), 5), null_p95=round(p95, 5),
                null_std=round(float(null.std()), 5), real_percentile=round(pct, 4),
                passes=bool(real > p95), kind=kind, n=n)


def _selfcheck():
    """Sanity: a LINEAR-autocorrelation metric should NOT pass (surrogates preserve it); a NONLINEAR metric SHOULD."""
    rng = np.random.default_rng(1)
    # AR(1): linear. lag-1 autocorr is preserved by surrogates -> should NOT pass.
    n = 4000; x = np.zeros(n)
    for i in range(1, n): x[i] = 0.7 * x[i - 1] + rng.standard_normal()
    lin = lambda z: float(np.corrcoef(np.asarray(z)[:-1], np.asarray(z)[1:])[0, 1])
    # nonlinear: |x| autocorrelation (volatility clustering) destroyed by phase-randomization -> SHOULD pass on a GARCH-like
    e = rng.standard_normal(n); s = np.ones(n)
    for i in range(1, n): s[i] = np.sqrt(0.1 + 0.85 * s[i - 1] ** 2 * 1 + 0.1 * e[i - 1] ** 2)
    g = e * s
    nl = lambda z: float(np.corrcoef(np.abs(np.asarray(z))[:-1], np.abs(np.asarray(z))[1:])[0, 1])
    print("linear lag-1 autocorr (should NOT pass):", surrogate_gate(lin, x, n=100, kind="phase"))
    print("nonlinear |.| autocorr (SHOULD pass):  ", surrogate_gate(nl, g, n=100, kind="phase"))


if __name__ == "__main__":
    _selfcheck()
