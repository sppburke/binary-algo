"""PHASE 4 — unified CROSS-SECTIONAL DIRECTION harness on the faithfulness-certified 7-pair panel.

Tests mechanism-matched direction feature FAMILIES against the certified cross-pair book, each gated by the frozen-past
forward holdout (NOVEL_METHODS_RESEARCH §3): an arm must BEAT the base xp book selacc forward (no decay) AND clear
breakeven .541 standalone. Generic TSFMs read ~.50 here; these are matched to the engineered lead-lag edge.

Families (CLI arg `which`):
  D1 sig   — DEPTH-2 LEAD-LAG SIGNATURE cross-terms (hand-rolled numpy, no iisignature). Over a trailing W-bar window of
             the 7 eu-equiv cumulative-return paths, the level-2 iterated integral S^{ij}=∫∫_{s<t}dX^i dX^j and its
             antisymmetric LÉVY AREA A^{ij}=½(S^{ij}−S^{ji}) between EURUSD and each peer directly encode signed lead-lag /
             quadratic covariation. Falsifier: beats book p10 CI95 AND a lead/lag-shuffle MUST degrade it (else not lead-lag).
  D6 havok — FROZEN-BASIS HAVOK (Hankel Koopman): delay-embed the USD common-factor trend, SVD on TRAIN -> freeze spatial
             modes U, causally project to coords v_1..v_{r-1} + intermittent FORCING v_r (signed precursor) + phase.
             Falsifier: signed feature beats .541, book-gating lifts p10, up-rate∈[.47,.53], surrogate-null.

Each family runs forward_holdout (direction, cov0.10 NY selacc) on arms {base, base+fam, famonly} + a SHUFFLE control
(D1: lead/lag time-shuffle; D6: phase-randomized factor) to prove the signal is mechanism-specific, not spectral.

Run: ~/binary-algo-venv/bin/python xsec_direction.py <sig|havok> [HOR=15]   -> xsec_direction_<which>_<HOR>m_result.json
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from numpy.lib.stride_tricks import sliding_window_view
from fwd_holdout import forward_holdout
import m5_xpair as MX

FEAT = MX.FEAT; PAIRS = MX.PAIRS; NONEU = MX.NONEU
WHICH = sys.argv[1] if len(sys.argv) > 1 else "sig"
HOR = int(sys.argv[2]) if len(sys.argv) > 2 else 15
YEARS = list(range(2012, 2027)); TRAIN_MAX = 2023
SIG_W = 30                                                          # signature window (bars)
HV_Q = 60; HV_R = 8; HV_TREND = 30                                 # HAVOK: delays, modes, factor-trend window
T0 = time.time()
def hb(m): print(f"[{time.time()-T0:.0f}s] {m}", flush=True)


def load_panel():
    frames = [pd.read_parquet(f"{FEAT}/panel_{y}.parquet") for y in YEARS if os.path.exists(f"{FEAT}/panel_{y}.parquet")]
    P = pd.concat(frames).reset_index(drop=True)
    P = P.sort_values("t").drop_duplicates("t", keep="last").reset_index(drop=True)
    return P


def make_labels(P, hor):
    """Forward hor-bar eu-equiv EURUSD signed return, wall-clock contiguous, NY session. Returns fwd, ny, valid_base."""
    t = P["t"].values.astype("int64"); n = len(P); reu = P["r_EURUSD"].values.astype(float)
    csum = np.concatenate([[0.0], np.cumsum(reu)])
    fwd = np.full(n, np.nan); fwd[:n - hor] = csum[hor + 1:] - csum[1:n - hor + 1]
    contig = np.zeros(n, bool); contig[:n - hor] = (t[hor:] - t[:-hor]) == hor * 60
    dt = pd.to_datetime(t, unit="s", utc=True); h = dt.hour.values + dt.minute.values / 60.0
    ny = (h >= 13.0) & (h < 22.0)
    return fwd, contig, ny, t


# ---------------------------------------------------------------- D1: depth-2 lead-lag signature (pure numpy)
def sig_features(P, W=SIG_W, shuffle=False, seed=0):
    """Level-2 signature cross-terms between EURUSD and each peer over a trailing W-bar window of eu-equiv return paths.
    X^i_k = cumulative eu-equiv return; increments dX = per-bar return. S^{ij}_t = Σ (X^i_k − X^i_{t−W}) dX^j_k over the
    window; Lévy area A^{ij}=½(S^{ij}−S^{ji}). shuffle = circular-shift each peer by a random offset (ROTATION surrogate):
    destroys cross-pair LEAD-LAG alignment while preserving each series' own autocorrelation — the mechanism-specificity
    null. If the signal is genuine lead-lag, the shuffled arm must collapse toward noise (vectorized view, no per-row copy)."""
    n = len(P); rng = np.random.default_rng(seed)
    r = {p: P[f"r_{p}"].values.astype(np.float64) for p in PAIRS}
    feats = {}
    eur = r["EURUSD"]
    dEUR = sliding_window_view(eur, W)                              # (n-W+1, W) per-bar EURUSD returns in window (view)
    XEUR = np.cumsum(dEUR, axis=1) - dEUR                           # X^EUR_k − X^EUR_0 = accumulated BEFORE step k (left point)
    lvl1_eur = dEUR.sum(1)                                          # total EURUSD move over window (= level-1)
    for p in NONEU:
        rp = r[p]
        if shuffle:
            off = int(rng.integers(W + 1, n - W - 1)); rp = np.roll(rp, off)        # rotation surrogate: kill lead-lag
        dP = sliding_window_view(rp, W)                            # view (no copy)
        XP = np.cumsum(dP, axis=1) - dP
        S_eur_p = (XEUR * dP).sum(1)                                # ∫ dEUR dP  (EURUSD leads P)
        S_p_eur = (XP * dEUR).sum(1)                                # ∫ dP dEUR  (P leads EURUSD)
        area = 0.5 * (S_eur_p - S_p_eur)                            # Lévy area: signed lead-lag EURUSD vs P
        for nm, v in [(f"sig_S_eur_{p}", S_eur_p), (f"sig_S_{p}_eur", S_p_eur), (f"sig_area_{p}", area)]:
            col = np.full(n, np.nan); col[W - 1:] = v; feats[nm] = col.astype(np.float32)
        del XP, S_eur_p, S_p_eur, area
    pad = np.full(n, np.nan); pad[W - 1:] = lvl1_eur; feats["sig_lvl1_eur"] = pad.astype(np.float32)
    return pd.DataFrame(feats)


# ---------------------------------------------------------------- D6: frozen-basis HAVOK
def havok_features(P, q=HV_Q, r=HV_R, trend=HV_TREND, train_mask=None, phase_rand=False, seed=0):
    """Delay-embed the USD common-factor trend; SVD on TRAIN Hankel -> freeze U; causal project to v_1..v_{r-1}+forcing v_r."""
    n = len(P); fac = P["fac"].values.astype(np.float64)
    sig = pd.Series(fac).rolling(trend).sum().values               # USD common-factor trend (smooth signal for Koopman)
    sig = np.nan_to_num(sig, nan=0.0)
    if phase_rand:                                                 # phase-randomize the signal (kills nonlinear dynamics)
        from surrogate_null import phase_randomize
        sig = phase_randomize(sig, np.random.default_rng(seed))
    H = sliding_window_view(sig, q)                                # (n-q+1, q) delay vectors (rows end at index k+q-1)
    if train_mask is None: train_mask = np.ones(len(H), bool)
    else: train_mask = train_mask[q - 1:]
    Htr = H[train_mask]
    Htr = Htr - Htr.mean(0, keepdims=True)
    # SVD of train Hankel (rows=samples) -> right singular vectors are the q-dim spatial modes
    U, S, Vt = np.linalg.svd(Htr, full_matrices=False)             # Vt: (q, q); take first r modes
    modes = Vt[:r]                                                  # (r, q) frozen spatial basis
    coords = (H - Htr.mean(0, keepdims=True)) @ modes.T            # (n-q+1, r) causal coords for ALL rows
    feats = {}
    pad = lambda v: np.concatenate([np.full(q - 1, np.nan), v]).astype(np.float32)
    for i in range(r - 1): feats[f"hv_v{i+1}"] = pad(coords[:, i])
    feats["hv_force"] = pad(coords[:, r - 1])                       # v_r = intermittent forcing (signed precursor)
    feats["hv_force_abs"] = pad(np.abs(coords[:, r - 1]))
    feats["hv_phase"] = pad(np.arctan2(coords[:, 1], coords[:, 0]))  # leading-mode oscillation phase
    return pd.DataFrame(feats)


def base_xp_features():
    """Certified base xp features + _ts + sess_ny + _fwd, concatenated across years, deduped — aligned to panel order."""
    MX.HOR = HOR; MX.GAP_S = HOR * 60
    parts = [MX.build_xp([str(y)]) for y in YEARS if os.path.exists(f"{FEAT}/EURUSD_{y}.parquet")]
    B = pd.concat(parts); B = B[~B.index.duplicated(keep="last")].sort_index()
    return B


def main():
    hb(f"which={WHICH} HOR={HOR}m : loading panel ...")
    P = load_panel(); hb(f"panel n={len(P):,}")
    fwd, contig, ny, t = make_labels(P, HOR)

    hb("building base xp features ...")
    B = base_xp_features()
    Bidx_ts = B["_ts"].values.astype("int64")
    # map base rows to panel rows by timestamp
    pos = pd.Series(np.arange(len(P)), index=P["t"].values)
    keep_b = np.isin(Bidx_ts, P["t"].values)
    B = B.iloc[keep_b]; brows = pos.loc[B["_ts"].values.astype("int64")].values
    xpc = [c for c in MX.xp_cols(B) if c in B.columns]
    Xb_full = np.full((len(P), len(xpc)), np.nan, np.float32); Xb_full[brows] = B[xpc].astype(np.float32).values
    hb(f"base mapped rows={len(brows):,} feats={len(xpc)}")

    hb(f"building family={WHICH} features ...")
    if WHICH == "sig":
        F = sig_features(P, SIG_W); Fsh = sig_features(P, SIG_W, shuffle=True, seed=1)
    elif WHICH == "havok":
        trmask = pd.to_datetime(t, unit="s", utc=True).year.values <= TRAIN_MAX
        F = havok_features(P, train_mask=trmask)
        Fsh = havok_features(P, train_mask=trmask, phase_rand=True, seed=1)
    else:
        raise SystemExit("which must be sig|havok")
    Xf = F.values.astype(np.float32); Xfsh = Fsh.values.astype(np.float32)
    hb(f"family feats={Xf.shape[1]} (+shuffle control)")

    fin = (np.isfinite(Xb_full).all(1) & np.isfinite(Xf).all(1) & np.isfinite(Xfsh).all(1)
           & np.isfinite(fwd) & (fwd != 0) & contig & ny)
    hb(f"NY decision bars n={fin.sum():,} up-rate={(fwd[fin] > 0).mean():.4f}")
    Xb, Xf, Xfsh, tgt, tsm = Xb_full[fin], Xf[fin], Xfsh[fin], fwd[fin], t[fin]

    arms = {"base": Xb, f"+{WHICH}": np.column_stack([Xb, Xf]).astype(np.float32),
            f"{WHICH}only": Xf, f"{WHICH}_shuf": Xfsh}
    res = forward_holdout(arms, target=tgt, ts=tsm, mode="direction", train_max=TRAIN_MAX,
                          test_years=(2024, 2025, 2026), cov=0.10, verbose=True)
    out = {"which": WHICH, "horizon_min": HOR, "params": dict(SIG_W=SIG_W, HV_Q=HV_Q, HV_R=HV_R, HV_TREND=HV_TREND),
           "design": "frozen-past forward holdout, direction cov0.10 NY selacc; base=certified xp book + shuffle control",
           "falsifier": f"+{WHICH} beats base selacc >=2 fwd years (no decay) AND {WHICH}only clears .541 AND real >> shuffle",
           "by_arm": res["by_arm"], "deltas": res["deltas"], "deployable": res["deployable"]}
    json.dump(out, open(f"/media/sean/CORSAIR/binary-algo/xsec_direction_{WHICH}_{HOR}m_result.json", "w"), indent=1)
    hb(f"DONE -> xsec_direction_{WHICH}_{HOR}m_result.json")


if __name__ == "__main__":
    main()
