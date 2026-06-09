#!/usr/bin/env python3
"""Point-process / hazard analysis of Deriv engineered indices (Crash, Boom, Jump).

The gross stats already confirm the documented structure (uni-directional drift + rare
opposite spikes for Crash/Boom; symmetric periodic jumps for Jump). The ONLY economically
interesting question is whether the spike TIMING is predictable:

  * Memoryless (geometric/Bernoulli per tick): CV of inter-spike gaps ~= 1, FLAT hazard,
    gaps un-autocorrelated, fits geometric. => timing UNPREDICTABLE, no timing edge.
  * Structured: refractory period (min gap >> 1, CV < 1), RISING hazard with age
    (a spike becomes "due"), or autocorrelated gaps. => timing PARTIALLY PREDICTABLE.

Also reports the documented-frequency check (mean gap vs the index's nominal N ticks).
Pure numpy/scipy. Usage:
    python3 syn_spike_audit.py syn_data/CRASH1000_ticks.npz [nominal_N]
"""
import json, sys, os
import numpy as np
from scipy import stats


def detect_spikes(r, k_thresh=10.0, merge=2):
    """Detect spike events. direction inferred from skew: Crash => big negative r,
    Boom => big positive r, Jump (near-symmetric) => either sign. Returns (idx, direction)."""
    absr = np.abs(r)
    med = np.median(absr[absr > 0])
    thresh = k_thresh * med
    sk = stats.skew(r)
    if sk < -2:        # Crash: rare large NEGATIVE
        cand = np.where(r < -thresh)[0]; direction = "down(crash)"
    elif sk > 2:       # Boom: rare large POSITIVE
        cand = np.where(r > thresh)[0]; direction = "up(boom)"
    else:              # Jump: symmetric, either sign
        cand = np.where(absr > thresh)[0]; direction = "either(jump)"
    if len(cand) == 0:
        return cand, direction, thresh, med
    # merge spikes that span consecutive ticks into one event
    groups = np.split(cand, np.where(np.diff(cand) > merge)[0] + 1)
    idx = np.array([g[0] for g in groups])
    return idx, direction, thresh, med


def hazard_analysis(gaps):
    """Empirical hazard h(k) = P(end at age k | reached age k). Flat => memoryless."""
    maxg = int(gaps.max())
    counts = np.bincount(gaps, minlength=maxg + 1).astype(np.float64)  # counts[k]=#gaps==k
    surv = np.cumsum(counts[::-1])[::-1]                                # surv[k]=#gaps>=k
    hazard = np.divide(counts, surv, out=np.zeros_like(counts), where=surv > 0)
    # regress hazard on age k over the support where survivors are reasonably many
    kmax = int(np.percentile(gaps, 95))
    ks = np.arange(1, max(2, kmax) + 1)
    w = surv[ks]                                  # weight by # at risk
    h = hazard[ks]
    valid = w >= 20
    slope = pval = None
    if valid.sum() >= 5:
        # weighted least squares of h ~ a + b*k
        kk, hh, ww = ks[valid].astype(float), h[valid], w[valid]
        W = ww / ww.sum()
        kbar = np.sum(W * kk); hbar = np.sum(W * hh)
        cov = np.sum(W * (kk - kbar) * (hh - hbar))
        var = np.sum(W * (kk - kbar) ** 2)
        slope = float(cov / var) if var > 0 else None
        # significance via unweighted OLS t-stat on the valid points (conservative)
        if slope is not None and valid.sum() >= 8:
            res = stats.linregress(kk, hh)
            pval = float(res.pvalue)
    return {"hazard_slope_per_tick": slope, "hazard_slope_pval": pval,
            "mean_hazard": float(np.mean(h[valid])) if valid.any() else None,
            "kmax_tested": int(kmax)}


def audit_spikes(path, nominal_N=None):
    d = np.load(path)
    P = d["prices"].astype(np.float64); T = d["times"].astype(np.int64)
    sym = os.path.basename(path).replace("_ticks.npz", "")
    r = np.diff(np.log(P))
    idx, direction, thresh, med = detect_spikes(r)
    rep = {"symbol": sym, "n_ticks": len(P), "direction": direction,
           "spike_thresh_logret": float(thresh), "median_abs_ret": float(med),
           "n_spikes": int(len(idx))}
    if len(idx) < 10:
        rep["note"] = "too few spikes detected to characterize timing"
        return rep
    gaps = np.diff(idx)
    rep["mean_gap_ticks"] = float(gaps.mean())
    rep["nominal_N"] = nominal_N
    if nominal_N:
        rep["mean_gap_over_nominal"] = float(gaps.mean() / nominal_N)
    rep["cv_gaps"] = float(gaps.std() / gaps.mean())     # ~1 => memoryless
    rep["min_gap"] = int(gaps.min())
    rep["p05_gap"] = float(np.percentile(gaps, 5))
    rep["median_gap"] = float(np.median(gaps))
    rep["max_gap"] = int(gaps.max())
    # gap autocorrelation (does a short gap predict the next gap?)
    g = gaps - gaps.mean()
    rep["gap_acf_lag1"] = float(np.sum(g[1:] * g[:-1]) / np.sum(g * g)) if len(g) > 2 else None
    rep["gap_acf_noise_floor"] = float(1.96 / np.sqrt(len(gaps)))  # |acf| below this = noise
    # KS vs geometric(p=1/mean_gap)  (memoryless null)
    p_geom = 1.0 / gaps.mean()
    rep["ks_vs_geometric_p"] = float(stats.kstest(gaps, "geom", args=(p_geom,)).pvalue)
    # mean spike magnitude vs drift (documented ~30x for jump)
    spike_ret = np.abs(r[idx])
    rep["spike_mag_over_median"] = float(np.median(spike_ret) / med)
    rep.update({"hazard": hazard_analysis(gaps)})
    # spike-direction "due" predictability summary
    cv, hz = rep["cv_gaps"], rep["hazard"]["hazard_slope_per_tick"]
    flags = []
    if cv < 0.7:
        flags.append(f"CV(gaps)={cv:.2f} < 0.7: gaps MORE REGULAR than memoryless (refractory/periodic) -> timing partly predictable")
    if rep["min_gap"] > 0.3 * rep["mean_gap_ticks"]:
        flags.append(f"min_gap={rep['min_gap']} is a large fraction of mean {rep['mean_gap_ticks']:.0f}: REFRACTORY floor")
    if hz is not None and rep["hazard"]["hazard_slope_pval"] is not None and rep["hazard"]["hazard_slope_pval"] < 0.01 and hz > 0:
        flags.append(f"RISING hazard (slope={hz:.2e}/tick, p={rep['hazard']['hazard_slope_pval']:.1e}): spike becomes DUE with age -> exploitable")
    if rep["gap_acf_lag1"] is not None and abs(rep["gap_acf_lag1"]) > max(0.05, rep["gap_acf_noise_floor"]):
        flags.append(f"gap autocorr lag1={rep['gap_acf_lag1']:.3f} > noise floor {rep['gap_acf_noise_floor']:.3f}: gaps serially dependent")
    rep["timing_verdict"] = flags if flags else ["MEMORYLESS timing (CV~1, flat hazard, geometric, no gap autocorr) -> spike timing UNPREDICTABLE"]
    return rep


if __name__ == "__main__":
    path = sys.argv[1]
    nominal = int(sys.argv[2]) if len(sys.argv) > 2 else None
    rep = audit_spikes(path, nominal)
    out = path.replace("_ticks.npz", "_spike_audit.json")
    json.dump(rep, open(out, "w"), indent=2)
    print(f"\n===== {rep['symbol']}  spikes={rep.get('n_spikes')}  dir={rep['direction']} =====")
    if rep.get("n_spikes", 0) >= 10:
        print(f"mean gap = {rep['mean_gap_ticks']:.1f} ticks" +
              (f"  (nominal {nominal}, ratio {rep['mean_gap_over_nominal']:.2f})" if nominal else "") +
              f" | spike mag = {rep['spike_mag_over_median']:.1f}x median tick")
        print(f"CV(gaps)={rep['cv_gaps']:.3f} (1.0=memoryless)  min_gap={rep['min_gap']}  median={rep['median_gap']:.0f}  max={rep['max_gap']}")
        print(f"gap_acf_lag1={rep['gap_acf_lag1']:+.4f}  KS_vs_geometric_p={rep['ks_vs_geometric_p']:.3f}")
        hz = rep["hazard"]
        print(f"hazard slope={hz['hazard_slope_per_tick']}  p={hz['hazard_slope_pval']}  mean_hazard={hz['mean_hazard']}")
        print("TIMING VERDICT:")
        for v in rep["timing_verdict"]:
            print("  -", v)
    else:
        print(rep.get("note"))
    print(f"saved -> {out}")
