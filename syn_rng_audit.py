#!/usr/bin/env python3
"""Randomness / structure audit of a Deriv synthetic-index tick series.

Goal is NOT to "pass/fail" a generic RNG battery (a CSPRNG will pass) but to answer two
concrete questions for THIS research program:
  (1) Is there any exploitable serial structure -- in DIRECTION (sign/return autocorr) or,
      more importantly, in MAGNITUDE (|r| / r^2 autocorr = volatility clustering)?
  (2) Does the repo's magnitude edge (rv-window predicts future |move|) transfer to this
      synthetic? On FX, corr(rv_t, future|r|) ~= +0.4. On a constant-vol IID generator it
      should be ~0. This single number decides whether the "synthetic magnitude" fork lives.

At n~1.5M, p-values are uninformative (everything is "significant"), so we report EFFECT
SIZES and compare to an IID-Gaussian control of the same length (the statistical noise floor).

Pure numpy/scipy (no statsmodels). Usage:
    python3 syn_rng_audit.py syn_data/R_100_ticks.npz
"""
import json, sys, os
import numpy as np
from scipy import stats
from scipy.special import erfc

SEC_PER_YEAR = 365.25 * 24 * 3600


def fft_acf(x, nlags):
    """Autocorrelation rho[0..nlags] via FFT (fast for large n)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    m = 1 << (int(np.ceil(np.log2(2 * n))))
    f = np.fft.rfft(x, n=m)
    acov = np.fft.irfft(f * np.conj(f), n=m)[: nlags + 1].real
    return acov / acov[0]


def ljung_box(r, lags_report):
    nlags = max(lags_report)
    n = len(r)
    rho = fft_acf(r, nlags)
    k = np.arange(1, nlags + 1)
    Q_cum = n * (n + 2) * np.cumsum((rho[1:] ** 2) / (n - k))
    out = {}
    for L in lags_report:
        out[L] = {"Q": float(Q_cum[L - 1]), "p": float(stats.chi2.sf(Q_cum[L - 1], L))}
    return rho, out


def runs_test_ww(bits):
    """Wald-Wolfowitz runs test on a 0/1 array."""
    bits = bits.astype(np.int8)
    n = len(bits); n1 = int(bits.sum()); n0 = n - n1
    if n1 == 0 or n0 == 0:
        return {"runs": None, "z": None, "p": None}
    runs = 1 + int(np.sum(bits[1:] != bits[:-1]))
    mu = 2.0 * n1 * n0 / n + 1.0
    var = 2.0 * n1 * n0 * (2.0 * n1 * n0 - n) / (n ** 2 * (n - 1))
    z = (runs - mu) / np.sqrt(var)
    return {"runs": runs, "z": float(z), "p": float(2 * stats.norm.sf(abs(z))),
            "up_fraction": n1 / n}


# ---------- NIST-style bit tests (subset), alpha = 0.01 ----------
def nist_monobit(b):
    s = int((2 * b.astype(np.int64) - 1).sum())
    p = float(stats.norm.sf(abs(s) / np.sqrt(len(b))) * 2)
    return {"test": "monobit", "p": p, "pass": p >= 0.01}


def nist_block_frequency(b, M=20000):
    n = len(b); N = n // M
    if N < 1:
        return {"test": "block_frequency", "p": None, "pass": None}
    blocks = b[: N * M].reshape(N, M).mean(axis=1)
    chisq = 4.0 * M * np.sum((blocks - 0.5) ** 2)
    p = float(stats.gamma.sf(chisq / 2.0, N / 2.0))  # = igamc(N/2, chisq/2)
    return {"test": "block_frequency", "p": p, "pass": p >= 0.01, "N_blocks": int(N), "M": M}


def nist_runs(b):
    n = len(b); pi = b.mean()
    if abs(pi - 0.5) >= 2 / np.sqrt(n):
        return {"test": "runs", "p": 0.0, "pass": False, "note": "failed monobit prereq"}
    V = 1 + int(np.sum(b[1:] != b[:-1]))
    num = abs(V - 2 * n * pi * (1 - pi))
    den = 2 * np.sqrt(2 * n) * pi * (1 - pi)
    p = float(erfc(num / den))
    return {"test": "runs", "p": p, "pass": p >= 0.01}


def nist_dft_spectral(b):
    n = len(b)
    x = 2 * b.astype(np.float64) - 1
    M = np.abs(np.fft.rfft(x))[: n // 2]
    T = np.sqrt(np.log(1 / 0.05) * n)
    N0 = 0.95 * n / 2.0
    N1 = float(np.sum(M < T))
    d = (N1 - N0) / np.sqrt(n * 0.95 * 0.05 / 4.0)
    p = float(erfc(abs(d) / np.sqrt(2)))
    return {"test": "dft_spectral", "p": p, "pass": p >= 0.01}


def nist_approx_entropy(b, m=2):
    n = len(b)
    def phi(mm):
        if mm == 0:
            return 0.0
        # pattern counts over circular sequence
        seq = np.concatenate([b, b[: mm - 1]])
        # encode m-bit windows as integers
        idx = np.zeros(n, dtype=np.int64)
        for j in range(mm):
            idx = (idx << 1) | seq[j : j + n]
        counts = np.bincount(idx, minlength=1 << mm).astype(np.float64)
        c = counts / n
        c = c[c > 0]
        return float(np.sum(c * np.log(c)))
    apen = phi(m) - phi(m + 1)
    chisq = 2.0 * n * (np.log(2) - apen)
    p = float(stats.gamma.sf(chisq / 2.0, (1 << (m - 1)) / 1.0))
    return {"test": "approx_entropy", "p": p, "pass": p >= 0.01, "apen": apen}


def nist_cusum(b):
    x = 2 * b.astype(np.float64) - 1
    s = np.cumsum(x)
    z = np.max(np.abs(s))
    n = len(b)
    # forward cusum p-value (NIST formula, truncated series)
    k = np.arange(int((-n / z + 1) / 4), int((n / z - 1) / 4) + 1)
    t1 = np.sum(stats.norm.cdf((4 * k + 1) * z / np.sqrt(n)) - stats.norm.cdf((4 * k - 1) * z / np.sqrt(n)))
    k2 = np.arange(int((-n / z - 3) / 4), int((n / z - 1) / 4) + 1)
    t2 = np.sum(stats.norm.cdf((4 * k2 + 3) * z / np.sqrt(n)) - stats.norm.cdf((4 * k2 + 1) * z / np.sqrt(n)))
    p = float(1 - t1 + t2)
    return {"test": "cusum", "p": max(0.0, min(1.0, p)), "pass": (max(0.0, min(1.0, p)) >= 0.01)}


def bit_battery(b, label):
    b = b.astype(np.int8)
    res = [nist_monobit(b), nist_block_frequency(b), nist_runs(b),
           nist_dft_spectral(b), nist_approx_entropy(b, 2), nist_cusum(b)]
    n_pass = sum(1 for r in res if r.get("pass") is True)
    n_tot = sum(1 for r in res if r.get("pass") is not None)
    return {"stream": label, "n_bits": int(len(b)), "passed": n_pass, "of": n_tot, "tests": res}


def rolling_std(r, W):
    """Rolling std over window W (population), aligned so rv[t] uses r[t-W+1..t]."""
    c = np.cumsum(np.concatenate([[0.0], r]))
    c2 = np.cumsum(np.concatenate([[0.0], r * r]))
    s = c[W:] - c[:-W]
    s2 = c2[W:] - c2[:-W]
    var = s2 / W - (s / W) ** 2
    return np.sqrt(np.maximum(var, 0.0))  # length n-W+1


def magnitude_transfer_probe(r, W=30, H=30):
    """Does the repo's magnitude signal transfer? corr(rv over past W, mean|r| over next H).
    On FX this is ~ +0.4. On constant-vol IID it should be ~0."""
    L = len(r)
    if L <= W + H + 1000:
        return {"corr_rv_future_absr": None}
    rv = rolling_std(r, W)                    # length L-W+1; rv[i] summarizes r[i .. i+W-1]
    absr = np.abs(r)
    cab = np.cumsum(np.concatenate([[0.0], absr]))   # length L+1; cab[j]=sum(absr[:j])
    nvalid = L - W - H + 1                     # future window r[i+W .. i+W+H-1] must exist
    fut = (cab[W + H: W + H + nvalid] - cab[W: W + nvalid]) / H
    rv_al = rv[:nvalid]
    cc = float(np.corrcoef(rv_al, fut)[0, 1])
    # also: autocorr of rv itself (vol-of-vol persistence) and of r^2
    rho_rv = fft_acf(rv, 50)
    return {"W": W, "H": H, "corr_rv_future_absr": cc,
            "rv_acf_lag1": float(rho_rv[1]), "rv_acf_max_1to50": float(np.max(np.abs(rho_rv[1:])))}


def control_noise_floor(n, seed_shift):
    """Max|ACF| over lags 1..50 for an IID-Gaussian series of length n (the statistical floor).
    seed via deterministic Generator (no Math.random dependency)."""
    g = np.random.default_rng(12345 + seed_shift)
    z = g.standard_normal(n)
    rho_r = fft_acf(z, 50)
    rho_r2 = fft_acf(z * z, 50)
    return {"max_abs_acf_r": float(np.max(np.abs(rho_r[1:]))),
            "max_abs_acf_r2": float(np.max(np.abs(rho_r2[1:])))}


def audit(path):
    d = np.load(path)
    P, T = d["prices"].astype(np.float64), d["times"].astype(np.int64)
    sym = os.path.basename(path).replace("_ticks.npz", "")
    n = len(P)
    dt = np.diff(T)
    dt_med = float(np.median(dt))
    r = np.diff(np.log(P))                 # log returns
    n_zero = int(np.sum(r == 0))
    rep = {"symbol": sym, "n_ticks": int(n), "span_days": float((T[-1] - T[0]) / 86400),
           "dt_median_s": dt_med, "dt_min_s": int(dt.min()), "dt_max_s": int(dt.max()),
           "dt_regular_frac": float(np.mean(dt == dt_med))}

    # --- quantization ---
    pip = 0.01
    steps = np.round(np.diff(P) / pip).astype(np.int64)
    rep["quantization"] = {"pip": pip,
                           "frac_price_on_grid": float(np.mean(np.abs(np.round(P / pip) - P / pip) < 1e-6)),
                           "abs_step_pips_p50": int(np.percentile(np.abs(steps), 50)),
                           "abs_step_pips_p99": int(np.percentile(np.abs(steps), 99)),
                           "frac_zero_step": float(np.mean(steps == 0))}

    # --- return distribution ---
    ticks_per_year = SEC_PER_YEAR / dt_med
    ann_vol = float(np.std(r) * np.sqrt(ticks_per_year))
    samp = r if n <= 400000 else r[:: max(1, n // 400000)]  # KS/AD on a subsample (n cap)
    ks = stats.kstest((samp - samp.mean()) / samp.std(), "norm")
    rep["returns"] = {
        "mean": float(np.mean(r)), "std": float(np.std(r)),
        "skew": float(stats.skew(r)), "excess_kurtosis": float(stats.kurtosis(r)),
        "annualized_vol_implied": ann_vol,  # ~1.0 would confirm "100% annualized"
        "ticks_per_year": float(ticks_per_year),
        "ks_normal_stat": float(ks.statistic), "ks_normal_p": float(ks.pvalue),
        "jarque_bera_p": float(stats.jarque_bera(samp).pvalue), "n_zero_returns": n_zero}

    # --- IID / autocorr (effect sizes) ---
    lags = [1, 2, 5, 10, 20, 50]
    rho_r, lb_r = ljung_box(r, lags)
    rho_abs, lb_abs = ljung_box(np.abs(r), lags)
    rho_r2, lb_r2 = ljung_box(r * r, lags)
    floor = control_noise_floor(n - 1, 0)
    rep["autocorr"] = {
        "noise_floor_iid": floor,  # max|ACF| expected from pure IID at this n
        "returns":       {"acf_lag1": float(rho_r[1]), "acf_lag2": float(rho_r[2]),
                          "max_abs_acf_1to50": float(np.max(np.abs(rho_r[1:51]))),
                          "ljung_box": lb_r},
        "abs_returns":   {"acf_lag1": float(rho_abs[1]),
                          "max_abs_acf_1to50": float(np.max(np.abs(rho_abs[1:51]))),
                          "ljung_box": lb_abs},
        "sq_returns":    {"acf_lag1": float(rho_r2[1]),
                          "max_abs_acf_1to50": float(np.max(np.abs(rho_r2[1:51]))),
                          "ljung_box": lb_r2}}

    # --- magnitude-edge transfer probe (the decisive economic test) ---
    rep["magnitude_transfer"] = {
        "W30_H30": magnitude_transfer_probe(r, 30, 30),
        "W120_H30": magnitude_transfer_probe(r, 120, 30)}

    # --- sign / runs + NIST bit batteries ---
    nz = r != 0
    sign_bits = (r[nz] > 0).astype(np.int8)
    rep["sign"] = runs_test_ww(sign_bits)
    price_int = np.round(P / pip).astype(np.int64)
    lsb_bits = (price_int & 1).astype(np.int8)
    rep["nist"] = [bit_battery(sign_bits, "return_sign_bits"),
                   bit_battery(lsb_bits, "price_lsb_bits")]

    # --- spectral periodicity of returns (top peaks) ---
    x = r - r.mean()
    Pxx = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x))
    k = np.argsort(Pxx[1:])[::-1][:5] + 1
    med = np.median(Pxx[1:])
    rep["spectral_top_peaks"] = [{"period_ticks": float(1 / freqs[i]) if freqs[i] > 0 else None,
                                  "power_over_median": float(Pxx[i] / med)} for i in k]
    return rep


def verdict(rep):
    a = rep["autocorr"]; floor = a["noise_floor_iid"]
    mt = rep["magnitude_transfer"]["W30_H30"]["corr_rv_future_absr"]
    flags = []
    # direction structure?
    if a["returns"]["max_abs_acf_1to50"] > 5 * floor["max_abs_acf_r"]:
        flags.append(f"RETURN autocorr {a['returns']['max_abs_acf_1to50']:.4f} >> IID floor {floor['max_abs_acf_r']:.4f}: DIRECTION structure")
    # magnitude / vol clustering?
    if a["sq_returns"]["max_abs_acf_1to50"] > 5 * floor["max_abs_acf_r2"]:
        flags.append(f"SQUARED-return autocorr {a['sq_returns']['max_abs_acf_1to50']:.4f} >> IID floor {floor['max_abs_acf_r2']:.4f}: VOL CLUSTERING")
    if mt is not None and abs(mt) > 0.05:
        flags.append(f"MAGNITUDE EDGE TRANSFERS: corr(rv,future|r|)={mt:.3f} (FX is ~+0.4)")
    return flags if flags else ["NO exploitable serial structure above the IID noise floor (consistent with CSPRNG constant-vol)"]


if __name__ == "__main__":
    rep = audit(sys.argv[1])
    out = sys.argv[1].replace("_ticks.npz", "_rng_audit.json")
    rep["verdict"] = verdict(rep)
    json.dump(rep, open(out, "w"), indent=2)
    # printed summary
    r = rep
    print(f"\n===== {r['symbol']}  n={r['n_ticks']:,}  span={r['span_days']:.1f}d  dt={r['dt_median_s']:.0f}s (regular {r['dt_regular_frac']*100:.2f}%) =====")
    rt = r["returns"]
    print(f"returns: std={rt['std']:.2e}  skew={rt['skew']:+.3f}  exkurt={rt['excess_kurtosis']:+.3f}  "
          f"ANN_VOL_implied={rt['annualized_vol_implied']:.3f} (1.0=100%)  KS_p={rt['ks_normal_p']:.2e}  zeros={rt['n_zero_returns']}")
    a = r["autocorr"]; f = a["noise_floor_iid"]
    print(f"IID floor (max|ACF| over 50 lags): r={f['max_abs_acf_r']:.4f}  r^2={f['max_abs_acf_r2']:.4f}")
    print(f"  returns  : acf1={a['returns']['acf_lag1']:+.4f}  max|acf|={a['returns']['max_abs_acf_1to50']:.4f}")
    print(f"  |returns|: acf1={a['abs_returns']['acf_lag1']:+.4f}  max|acf|={a['abs_returns']['max_abs_acf_1to50']:.4f}")
    print(f"  returns^2: acf1={a['sq_returns']['acf_lag1']:+.4f}  max|acf|={a['sq_returns']['max_abs_acf_1to50']:.4f}")
    mt = r["magnitude_transfer"]
    print(f"MAGNITUDE TRANSFER corr(rv,future|r|): W30={mt['W30_H30']['corr_rv_future_absr']}  W120={mt['W120_H30']['corr_rv_future_absr']}  (FX~+0.4)")
    print(f"sign: up_frac={r['sign']['up_fraction']:.5f}  runs_z={r['sign']['z']:+.2f}  runs_p={r['sign']['p']:.3f}")
    for bb in r["nist"]:
        print(f"NIST [{bb['stream']}]: passed {bb['passed']}/{bb['of']}  " +
              " ".join(f"{t['test']}={'P' if t.get('pass') else 'F'}" for t in bb["tests"]))
    print("VERDICT:")
    for v in r["verdict"]:
        print("  -", v)
    print(f"saved -> {out}")
