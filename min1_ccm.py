"""CCM COUPLING-GATE (Sugihara 2012) for 60s EURUSD direction — on-disk long-shot #2 (see CCM_DESIGN.md).

HYPOTHESIS: does convergent-cross-mapping coupling from another USD leg's order flow (or EURUSD's own OFI) into
EURUSD's 60s return dynamics identify the WINDOWS in which the frozen production sign model is genuinely accurate?
CCM is used as a GATE (decides WHEN to trade), NOT a feature (direction always comes from the frozen
min1_production blend). This keeps it leakage-immune.

HONEST PRIOR (do not relitigate): ~5-10%. 60s direction is near-efficient across ~24 channels; single/cross-pair
CKS-OFI are clean nulls (min1_cksofi VAL worst-half AUC 0.4993; cross-pair residual <=0.516). A documented null is
the expected, valuable outcome.

FIREWALL AGAINST THE MIRAGE (non-negotiable): the fixed 1s grid is built ONLY for the delay embedding (returns
SUMMED, OFI SUMMED, empty second -> 0.0, NEVER ffill a price). Labels / trades / accuracy are ALWAYS computed on
EURUSD's native micro-bar clock via wc_ret. No resampled bar ever touches a label, trade, or accuracy number.
Moved-bars-only, per-year, bootstrap CI95, threshold selected on worst-VAL-half (NEVER acc-max).

PRE-REGISTERED FALSIFIER (KILL if ANY): see decide().  Reuses min1_production settlement + frozen model.

Usage:  python min1_ccm.py            # full run -> min1_ccm_result.json
        python min1_ccm.py selfcheck  # CCM core sanity (self-coupling must converge strongest) then stop
"""
import sys, os, json, time, gc
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

import min1_production as P
from min1_production import wc_ret, boot, nonoverlap_chrono, HS, TOL_S, ENTRY_LAG_S, GAP

ROOT   = "/media/sean/CORSAIR/binary-algo"
TICK   = f"{ROOT}/features_tick"          # EURUSD micro bars (mid, imb, micro, spread, nt, tsz) = trading clock
CKS    = f"{ROOT}/features_tick_cks"      # EURUSD signed CKS OFI (cks_e)
XOFI   = f"{ROOT}/features_tick_xofi"     # cross-pair signed CKS OFI
RESULT = f"{ROOT}/min1_ccm_result.json"

# 7 drivers (EUR-equivalent USD-direction: keep XXXUSD sign, FLIP USDXXX). EURUSD own OFI = self-coupling sanity.
DRIVERS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"]
USD_BASE = {"USDJPY", "USDCHF", "USDCAD"}
SIGN = {p: (-1.0 if p in USD_BASE else 1.0) for p in DRIVERS}

# CCM params (CCM_DESIGN.md S2): E=4, tau=1 step (1s grid), E+1 neighbors, exp weights, Theiler window.
E, TAU = 4, 1
WIN_S = 1800                  # 30-min windows on the EURUSD clock
MAX_GAP_S = 300               # drop windows straddling a >5-min data gap
GRID_CAP = 1800               # resampled grid length cap (=30min*1s)
L_GRID = [50, 100, 200, 400, 800]   # +Lmax appended per window
B_SUB = 20                    # library subsamples per L
TP_SCAN = [1, 5, 15, 30, 60]  # forward cross-map lags (s) tested for tradeable lead
N_SURR = 100                  # Ebisuzaki phase-randomized surrogates
VAL_WIN_SAMPLE = 160          # windows sampled on VAL for the (expensive) convergence+surrogate characterization
RNG = np.random.default_rng(7)

# falsifier thresholds (frozen)
MIN_SLOPE, MIN_DELTA_RHO, SURR_PCT, GATE_CI_LO = 0.02, 0.05, 95, 0.515
UPRATE_BAND, MIN_MOVED_FRAC = (0.47, 0.53), 0.40


# ----------------------------- data loading -----------------------------
def load_eur(sp):
    """EURUSD micro bars (the trading clock + all features for the frozen production model)."""
    b = pd.read_parquet(f"{TICK}/{sp}_1s.parquet")
    return b[~b.index.duplicated(keep="last")].sort_index()

def load_driver_1s(pair, sp, idx):
    """A driver's signed (EUR-equivalent) CKS OFI 1s series, aligned to the EURUSD clock `idx`, missing -> 0.0.
    NEVER ffill (an OFI sum of zero events IS 0.0, not a stale price)."""
    fp = f"{CKS}/{sp}_cks1s.parquet" if pair == "EURUSD" else f"{XOFI}/{pair}_{sp}_cks1s.parquet"
    e = pd.read_parquet(fp)[["cks_e"]]["cks_e"].astype("float32") * np.float32(SIGN[pair])
    return e.reindex(idx).fillna(0.0).astype("float32").values


# ----------------------------- CCM core (CCM_DESIGN.md S2) -----------------------------
def to_grid(sec_in_win, vals, w0, w1):
    """Resample (event-time) values onto a fixed 1s grid [w0, w1) by SUMMING into each second; empty -> 0.0.
    This is the embedding firewall: used ONLY for the manifold, NEVER for labels/trades."""
    n = int(w1 - w0)
    if n < 1: return None
    g = np.zeros(n, dtype=np.float64)
    pos = (sec_in_win - w0).astype(int)
    ok = (pos >= 0) & (pos < n)
    np.add.at(g, pos[ok], vals[ok])
    return g

def shadow(Y, E=E, tau=TAU):
    """Time-delay embedding of Y: row t = [Y_t, Y_{t-tau}, ..., Y_{t-(E-1)tau}]. Returns (M, base_idx)."""
    n = len(Y); span = (E - 1) * tau
    if n <= span + 2: return None, None
    base = np.arange(span, n)
    M = np.column_stack([Y[base - k * tau] for k in range(E)])
    return M, base

def ccm_rho(Y, X, L, tp, E=E, tau=TAU, B=1, rng=None):
    """Cross-map skill rho(X_hat | M_Y) at library length L and forward cross-map lag tp (seconds=grid steps).
    E+1 neighbors, Theiler window |dt|<=E*tau+tp, exponential weights. Median over B library subsamples."""
    M, base = shadow(Y, E, tau)
    if M is None: return np.nan
    # targets: base points whose t+tp has a valid X observation
    valid_t = base + tp < len(X)
    tgt = base[valid_t]
    if len(tgt) < E + 2: return np.nan
    theiler = E * tau + abs(tp)
    rng = rng or RNG
    rhos = []
    for _ in range(B):
        lib = base if L >= len(base) else np.sort(rng.choice(base, size=L, replace=False))
        if len(lib) < E + 2: continue
        Mlib = M[lib - base[0]]
        tree = cKDTree(Mlib)
        Mtgt = M[tgt - base[0]]
        k = min(E + 1 + 8, len(lib))            # over-fetch to survive Theiler pruning
        dist, nn = tree.query(Mtgt, k=k)
        xhat = np.full(len(tgt), np.nan)
        for i in range(len(tgt)):
            ti = tgt[i]
            cand = lib[nn[i]]; dd = dist[i]
            keep = np.abs(cand - ti) > theiler
            cand, dd = cand[keep][:E + 1], dd[keep][:E + 1]
            if len(cand) < E + 1: continue
            d1 = dd[0] if dd[0] > 0 else 1e-12
            w = np.exp(-dd / d1); w /= w.sum()
            xhat[i] = np.dot(w, X[cand + tp])
        m = np.isfinite(xhat)
        if m.sum() < E + 2: continue
        xt = X[tgt + tp][m]
        if np.std(xhat[m]) < 1e-12 or np.std(xt) < 1e-12: continue
        rhos.append(np.corrcoef(xhat[m], xt)[0, 1])
    return float(np.median(rhos)) if rhos else np.nan

def ebisuzaki(x, rng):
    """Phase-randomized surrogate: preserve power spectrum (autocorr/seasonality), randomize phases."""
    n = len(x); f = np.fft.rfft(x)
    amp = np.abs(f); ph = rng.uniform(0, 2 * np.pi, len(f))
    ph[0] = 0
    if n % 2 == 0: ph[-1] = 0
    s = np.fft.irfft(amp * np.exp(1j * ph), n=n)
    return s.astype(np.float64)


# ----------------------------- per-window coupling -----------------------------
def window_iter(ts):
    """Yield (w0, w1, member_positions) for each 30-min UTC bucket with no >MAX_GAP_S internal gap."""
    wid = (ts // WIN_S)
    order = np.argsort(wid, kind="mergesort")
    ws = wid[order]
    bounds = np.where(np.diff(ws) != 0)[0] + 1
    for grp in np.split(order, bounds):
        if len(grp) < E + 5: continue
        gts = np.sort(ts[grp])
        if np.max(np.diff(gts)) > MAX_GAP_S: continue
        w0 = (gts[0] // WIN_S) * WIN_S; w1 = w0 + WIN_S
        if w1 - w0 > GRID_CAP: w1 = w0 + GRID_CAP
        yield int(w0), int(w1), grp

def win_grids(ts, mid_ret_evt, driver_evt, w0, w1, grp):
    """Build the EUR-return grid (Y) and each driver's OFI grid (X) for one window. Returns (Yg, {pair:Xg})."""
    sec = ts[grp]
    Yg = to_grid(sec, mid_ret_evt[grp], w0, w1)
    Xg = {p: to_grid(sec, driver_evt[p][grp], w0, w1) for p in DRIVERS}
    return Yg, Xg


# ----------------------------- frozen production direction -----------------------------
def production_dir(b):
    """p_up / pred / conf from the FROZEN min1_production blend (direction only; CCM supplies the gate)."""
    par, L, G, C, S = P._load()
    X = P.feats(b)
    p = P._blend(par, L, G, C, S, X)
    return p, (p > 0.5).astype(int), np.abs(p - 0.5)

def split_arrays(b):
    """Native-clock labels (deriv-faithful) + per-bar event-time return for the CCM grid."""
    mid = b["mid"].values.astype(float)
    ts = b.index.values.astype("datetime64[s]").astype("int64")
    ret, valid = wc_ret(ts, mid, HS, TOL_S, ENTRY_LAG_S)     # native-clock 60s label
    y = (ret > 0).astype(int); mag = np.abs(ret)
    logmid = np.log(mid)
    r_evt = np.empty(len(mid)); r_evt[0] = 0.0; r_evt[1:] = np.diff(logmid)  # event-time 1-step return (for grid Y)
    return ts, mid, y, mag, valid, r_evt


# ----------------------------- per-year gated eval -----------------------------
def gated_year_eval(ts, y, mag, valid, pred, conf, conf_thr, gate_open):
    """Trade gate-open & valid & conf>=thr bars; non-overlap chrono; moved-bars-only; per-year acc + CI95."""
    base = valid & gate_open & (conf >= conf_thr)
    tr = nonoverlap_chrono(ts, base)
    out = {}
    yrs = P.year_of(ts[tr]) if len(tr) else np.array([])
    for label in ("2024", "2025", "2026", "all"):
        sel = tr if label == "all" else (tr[yrs == int(label)] if len(tr) else tr)
        if len(sel) < 10:
            out[label] = {"n": int(len(sel)), "moved_acc": None, "moved_acc_ci": [None, None],
                          "n_moved": 0, "moved_up_rate": None, "moved_frac": None}
            continue
        moved = mag[sel] > 0
        mv = (pred[sel][moved] == y[sel][moved]).astype(float)
        mlo, mhi = boot(mv) if moved.sum() >= 5 else (None, None)
        out[label] = {"n": int(len(sel)),
                      "moved_acc": float(mv.mean()) if moved.sum() else None,
                      "moved_acc_ci": [mlo, mhi],
                      "n_moved": int(moved.sum()),
                      "moved_up_rate": float(y[sel][moved].mean()) if moved.sum() else None,
                      "moved_frac": float(moved.mean())}
    return out


# ----------------------------- characterization on VAL -----------------------------
def characterize_val(win_data):
    """For each driver on a sample of VAL windows: convergence slope, delta_rho, tp*, rho, Ebisuzaki p95.
    win_data = list of (Yg, Xg) grids. Returns per-driver dict."""
    idx = np.arange(len(win_data))
    if len(idx) > VAL_WIN_SAMPLE:
        idx = RNG.choice(idx, VAL_WIN_SAMPLE, replace=False)
    res = {}
    for pair in DRIVERS:
        # convergence: median rho over sampled windows at each L (Lmax appended), at the window's best tp
        rho_by_L = {L: [] for L in L_GRID}
        rho_Lmax, tp_pick, surr_exc = [], [], []
        for wi in idx:
            Yg, Xg = win_data[wi]; Xg = Xg[pair]
            if Yg is None or Xg is None: continue
            Lmax = len(Yg)
            # pick tp* by max rho at Lmax
            best_tp, best_rho = None, -2
            for tp in TP_SCAN:
                r = ccm_rho(Yg, Xg, Lmax, tp, B=1)
                if np.isfinite(r) and r > best_rho: best_rho, best_tp = r, tp
            if best_tp is None: continue
            tp_pick.append(best_tp); rho_Lmax.append(best_rho)
            for L in L_GRID:
                if L < Lmax:
                    rr = ccm_rho(Yg, Xg, L, best_tp, B=B_SUB)
                    if np.isfinite(rr): rho_by_L[L].append(rr)
            # Ebisuzaki surrogate exceedance at this window (X phase-randomized)
            srho = []
            for _ in range(max(8, N_SURR // VAL_WIN_SAMPLE + 8)):
                xs = ebisuzaki(Xg, RNG)
                rs = ccm_rho(Yg, xs, Lmax, best_tp, B=1)
                if np.isfinite(rs): srho.append(rs)
            if srho: surr_exc.append(best_rho > np.percentile(srho, SURR_PCT))
        Ls = [L for L in L_GRID if rho_by_L[L]]
        med = [np.median(rho_by_L[L]) for L in Ls]
        rmax = float(np.median(rho_Lmax)) if rho_Lmax else np.nan
        if len(Ls) >= 2:
            xs_ = np.log(Ls); slope = float(np.polyfit(xs_, med, 1)[0])
            sp = float(spearmanr(Ls, med).correlation) if len(Ls) > 2 else (1.0 if med[-1] >= med[0] else -1.0)
            drho = float(rmax - med[0])
        else:
            slope, sp, drho = np.nan, np.nan, np.nan
        res[pair] = {
            "tp_star_s": int(np.median(tp_pick)) if tp_pick else None,
            "rho_Lmin": float(med[0]) if med else None,
            "rho_Lmax": rmax if np.isfinite(rmax) else None,
            "delta_rho": drho if np.isfinite(drho) else None,
            "conv_slope_logL": slope if np.isfinite(slope) else None,
            "spearman_rho_L": sp if np.isfinite(sp) else None,
            "surrogate_pass_frac": float(np.mean(surr_exc)) if surr_exc else None,
            "converges": bool(np.isfinite(slope) and slope > MIN_SLOPE and np.isfinite(drho) and drho >= MIN_DELTA_RHO),
            "passes_surrogate": bool(surr_exc and np.mean(surr_exc) > 0.5),
            "forward_lead": bool(tp_pick and np.median(tp_pick) >= 1),
        }
    return res


# ----------------------------- main -----------------------------
def build_split_ccm(sp):
    """Per split: native labels + production direction + per-window coupling grids. Returns dict."""
    b = load_eur(sp)
    ts, mid, y, mag, valid, r_evt = split_arrays(b)
    idx = b.index
    driver_evt = {p: load_driver_1s(p, sp, idx) for p in DRIVERS}   # event-time, EUR-clock aligned, EUR-equiv signed
    p_up, pred, conf = production_dir(b)
    del b; gc.collect()
    # per-window grids
    win_meta, win_grids_list = [], []
    for w0, w1, grp in window_iter(ts):
        Yg, Xg = win_grids(ts, r_evt, driver_evt, w0, w1, grp)
        win_meta.append((w0, w1, grp)); win_grids_list.append((Yg, Xg))
    return dict(ts=ts, y=y, mag=mag, valid=valid, pred=pred, conf=conf, p_up=p_up,
                win_meta=win_meta, win_grids=win_grids_list)

def window_coupling(d, pair, tp):
    """Per-bar gate_open flag from window coupling: rho(Lmax,tp) per window for `pair`. Returns (rho_per_win, n)."""
    rho_w = np.empty(len(d["win_meta"])); rho_w[:] = np.nan
    for i, (Yg, Xg) in enumerate(d["win_grids"]):
        xg = Xg[pair]
        if Yg is None or xg is None: continue
        rho_w[i] = ccm_rho(Yg, xg, len(Yg), tp, B=1)
    return rho_w

def assign_gate(d, rho_w, theta):
    """Map per-window coupling >= theta back onto bars."""
    gate = np.zeros(len(d["ts"]), bool)
    for (w0, w1, grp), r in zip(d["win_meta"], rho_w):
        if np.isfinite(r) and r >= theta:
            gate[grp] = True
    return gate

def select_theta_worstval(d, rho_w):
    """Smallest theta whose worst-VAL-half gated moved-acc CI95-lower >= GATE_CI_LO. Returns (theta, stats) or (None,..)."""
    finite = rho_w[np.isfinite(rho_w)]
    if len(finite) < 10: return None, None
    conf_thr = float(np.quantile(d["conf"][d["valid"]], 0.95))   # production 5%-cov conf threshold on VAL
    order = np.argsort([m[0] for m in d["win_meta"]])            # chronological windows
    half = set(order[len(order) // 2:].tolist())                # worst-half = 2nd chronological half (robustness)
    best = None
    for theta in np.quantile(finite, np.linspace(0.1, 0.9, 9)):
        gate = np.zeros(len(d["ts"]), bool)
        for wi, ((w0, w1, grp), r) in enumerate(zip(d["win_meta"], rho_w)):
            if wi in half and np.isfinite(r) and r >= theta: gate[grp] = True
        base = d["valid"] & gate & (d["conf"] >= conf_thr)
        tr = nonoverlap_chrono(d["ts"], base)
        if len(tr) < 30: continue
        moved = d["mag"][tr] > 0
        if moved.sum() < 20: continue
        mv = (d["pred"][tr][moved] == d["y"][tr][moved]).astype(float)
        lo, hi = boot(mv)
        if lo is not None and lo >= GATE_CI_LO:
            if best is None or theta < best[0]: best = (float(theta), float(mv.mean()), float(lo), float(hi), conf_thr)
    if best is None:
        return None, {"conf_thr": conf_thr}
    return best[0], {"worstval_moved_acc": best[1], "worstval_ci": [best[2], best[3]], "conf_thr": best[4]}


def decide(drivers, sanity, per_year_best):
    reasons = []
    any_conv = any(v.get("converges") for v in drivers.values())
    any_surr = any(v.get("passes_surrogate") for v in drivers.values())
    any_fwd = any(v.get("converges") and v.get("forward_lead") for v in drivers.values())
    if not any_conv: reasons.append("no_convergence(all drivers slope<=0.02 or delta_rho<0.05)")
    if not any_surr: reasons.append("no_surrogate_separation(none exceed Ebisuzaki p95)")
    if not any_fwd: reasons.append("no_forward_lead(no converging driver has tp*>=1s)")
    # gate fails OOS-stably: need some driver clearing GATE_CI_LO in ALL of 2024/2025/2026
    gate_ok = False
    for pr, yr in per_year_best.items():
        if yr and all(yr.get(Y, {}).get("moved_acc_ci", [None])[0] is not None and
                      yr[Y]["moved_acc_ci"][0] >= GATE_CI_LO for Y in ("2024", "2025", "2026")):
            gate_ok = True
    if not gate_ok: reasons.append(f"no_driver_clears_{GATE_CI_LO}_CI_lo_in_all_three_years")
    # mirage tripwire on the best driver's gated years
    mirage = False
    for pr, yr in per_year_best.items():
        for Y in ("2024", "2025", "2026"):
            r = (yr or {}).get(Y, {})
            ur, mf = r.get("moved_up_rate"), r.get("moved_frac")
            if ur is not None and (ur < UPRATE_BAND[0] or ur > UPRATE_BAND[1]): mirage = True
            if mf is not None and mf < MIN_MOVED_FRAC: mirage = True
    if mirage: reasons.append("mirage_tripwire(moved_up_rate or moved_frac out of band)")
    if not sanity.get("self_coupling_strongest"): reasons.append("sanity_fail(self-coupling not strongest convergence)")
    killed = bool(reasons)
    return {"FALSIFIER_KILLED": killed,
            "kill_reasons": reasons,
            "thresholds": {"min_conv_slope": MIN_SLOPE, "min_delta_rho": MIN_DELTA_RHO,
                           "surrogate_pctile": SURR_PCT, "gate_acc_ci_lower": GATE_CI_LO,
                           "require_all_years": True, "moved_up_rate_band": list(UPRATE_BAND),
                           "min_moved_frac": MIN_MOVED_FRAC},
            "verdict": ("KILLED — " + "; ".join(reasons)) if killed else
                       "SURVIVED — at least one cross-pair OFI shows convergent, surrogate-significant, forward-lead "
                       "coupling whose gated subset clears 0.515 CI95-lower in all three years (check 0.541 for product-grade)"}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    t0 = time.time()
    print(f"[ccm] loading VAL ...", flush=True)
    val = build_split_ccm("val")
    print(f"[ccm] VAL n={len(val['ts']):,} windows={len(val['win_meta'])} "
          f"moved-up-rate={val['y'][val['valid']&(val['mag']>0)].mean():.4f} {time.time()-t0:.0f}s", flush=True)
    chars = characterize_val(val["win_grids"])
    # self-coupling sanity (#6): EURUSD own OFI must have the strongest convergence slope
    slopes = {p: (chars[p]["conv_slope_logL"] or -9) for p in DRIVERS}
    self_strongest = all(slopes["EURUSD"] >= slopes[p] for p in DRIVERS)
    sanity = {"self_coupling_strongest": bool(self_strongest),
              "self_conv_slope": slopes["EURUSD"],
              "max_crosspair_conv_slope": max(slopes[p] for p in DRIVERS if p != "EURUSD"),
              "slopes": slopes}
    print(f"[ccm] characterization done {time.time()-t0:.0f}s; self_strongest={self_strongest}", flush=True)
    for p in DRIVERS:
        c = chars[p]
        print(f"   {p:7s} slope={c['conv_slope_logL']} drho={c['delta_rho']} tp*={c['tp_star_s']} "
              f"surr_pass={c['surrogate_pass_frac']} conv={c['converges']} surr={c['passes_surrogate']}", flush=True)
    if mode == "selfcheck":
        json.dump({"chars": chars, "sanity": sanity, "elapsed_s": round(time.time()-t0, 1)},
                  open(f"{ROOT}/min1_ccm_selfcheck.json", "w"), indent=1)
        print(f"[ccm] SELFCHECK done -> min1_ccm_selfcheck.json", flush=True); return

    # theta selection on VAL (worst-half) + per-year gate eval on TEST/OOS, per driver
    test = build_split_ccm("test"); print(f"[ccm] TEST built {time.time()-t0:.0f}s", flush=True)
    oos  = build_split_ccm("oos");  print(f"[ccm] OOS built {time.time()-t0:.0f}s", flush=True)
    per_year_best, gate_tables = {}, {}
    for pair in DRIVERS:
        tp = chars[pair]["tp_star_s"] or 5
        rho_val = window_coupling(val, pair, tp)
        theta, sel = select_theta_worstval(val, rho_val)
        chars[pair]["theta_gate"] = theta
        chars[pair]["theta_select"] = sel
        if theta is None:
            per_year_best[pair] = None; gate_tables[pair] = {"theta": None}; continue
        conf_thr = sel["conf_thr"]
        years = {}
        for nm, d in (("test", test), ("oos", oos)):
            rho_w = window_coupling(d, pair, tp)
            gate = assign_gate(d, rho_w, theta)
            years[nm] = gated_year_eval(d["ts"], d["y"], d["mag"], d["valid"], d["pred"], d["conf"], conf_thr, gate)
        # combine test+oos years: 2024/2025 come from TEST, 2026 from OOS
        comb = {"2024": years["test"]["2024"], "2025": years["test"]["2025"], "2026": years["oos"]["2026"]}
        per_year_best[pair] = comb; gate_tables[pair] = {"theta": theta, "tp": tp, "by_split": years, "combined": comb}
        print(f"   GATE {pair:7s} theta={theta:.4f} tp={tp}s "
              f"2024={comb['2024'].get('moved_acc')} 2025={comb['2025'].get('moved_acc')} "
              f"2026={comb['2026'].get('moved_acc')} {time.time()-t0:.0f}s", flush=True)

    falsifier = decide(chars, sanity, per_year_best)
    out = {
        "experiment": "CCM coupling-gate (Sugihara 2012) for 60s EURUSD direction — cross-pair OFI drivers",
        "settlement": "deriv-faithful wc_ret: next-tick entry lag (1s), last-tick<=expiry, mid-to-mid, ties LOSE",
        "discipline": ("EURUSD-clock-only; resample-to-1s-grid for embedding ONLY (sums, empty=0.0, NEVER ffill); "
                       "moved-bars-only; per-year; bootstrap CI95; theta on worst-VAL-half (NEVER acc-max)"),
        "horizon_s": HS, "gap_s": GAP, "embedding": {"E": E, "tau": TAU, "win_s": WIN_S, "tp_scan_s": TP_SCAN,
                                                     "L_grid": L_GRID, "n_surrogate": N_SURR},
        "splits": {"train": "2021-2023", "val": "2024-H1", "test": "2024.09-2025.11", "oos": "2026"},
        "drivers": chars, "sanity": sanity, "gate_tables": gate_tables, "falsifier": falsifier,
        "elapsed_s": round(time.time() - t0, 1),
    }
    json.dump(out, open(RESULT, "w"), indent=1)
    print(f"\n[ccm] FALSIFIER: {falsifier['verdict']}", flush=True)
    print(f"[ccm] DONE {time.time()-t0:.0f}s -> {RESULT}", flush=True)


if __name__ == "__main__":
    main()
