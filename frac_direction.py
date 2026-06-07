"""PHASE 3 / T2 applied to DIRECTION — does a FRACTIONALLY-DIFFERENCED USD factor/residual add to the certified
cross-pair direction book? (NOVEL_METHODS_RESEARCH T2: "build USD factor/residual on FFD series + lead-lag 1-3, feed
run_direction at 15m/30m NY, gate per-year forward holdout.")

Hypothesis: the certified edge uses 1-bar (integer-differenced) returns, which destroy the long-memory LEVEL co-movement
where a slow USD trend lives. FFD at the smallest ADF-stationary d* keeps that memory. If a slow cross-pair trend has
directional content, an FFD factor/residual should add to the base cross-pair book on a frozen-past forward holdout.

LEAKAGE-SAFE: d*_p chosen on TRAIN log-close ONLY (2012-2023), per pair, FROZEN (no per-fold refit). FFD is a causal
fixed-width-window convolution applied PER YEAR (warmup NaN at each year start). Label = forward HOR-bar EURUSD sign,
wall-clock contiguous, ties dropped, NY session. Arms gated by fwd_holdout (direction mode, cov0.10 selective accuracy).

Arms:  base   = certified xp features (build_xp)            <- the anchor (must match the book)
       +ffd   = base + FFD factor/residual/lead-lag         <- does FFD ADD?
       ffdonly= FFD features alone                          <- does FFD carry standalone signal?

Run: ~/binary-algo-venv/bin/python frac_direction.py [HOR=15]   ->  frac_direction_<HOR>m_result.json
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from frac_diff import ffd, min_ffd_d, ffd_weights
from fwd_holdout import forward_holdout
import m5_xpair as MX

FEAT = MX.FEAT; PAIRS = MX.PAIRS; USD_BASE = MX.USD_BASE; NONEU = MX.NONEU
def eu_sign(p): return -1.0 if p in USD_BASE else 1.0
HOR = int(sys.argv[1]) if len(sys.argv) > 1 else 15
YEARS = list(range(2012, 2027)); TRAIN_MAX = 2023
LAGS = [1, 2, 3]; THRESH = 1e-4                                  # thresh caps FFD window length (tractable conv)
T0 = time.time()
def hb(m): print(f"[{time.time()-T0:.0f}s] {m}", flush=True)


def read_closes(y):
    cl = {}
    for p in PAIRS:
        fp = f"{FEAT}/{p}_{y}.parquet"
        if not os.path.exists(fp): return None
        d = pd.read_parquet(fp, columns=["close"]); d = d[~d.index.duplicated(keep="last")].sort_index()
        cl[p] = d["close"]
    df = pd.DataFrame(cl).dropna()
    return df if len(df) > 5000 else None


def select_dstar():
    """Per-pair smallest ADF-stationary d* on the TRAIN log-close (concatenated 2012-2023)."""
    dstars = {}
    for p in PAIRS:
        segs = []
        for y in range(2012, TRAIN_MAX + 1):
            fp = f"{FEAT}/{p}_{y}.parquet"
            if not os.path.exists(fp): continue
            d = pd.read_parquet(fp, columns=["close"]); d = d[~d.index.duplicated(keep="last")].sort_index()
            segs.append(np.log(d["close"].values)[::5])           # decimate 5x for a fast ADF d*-scan
        lc = np.concatenate(segs)
        grid = np.round(np.arange(0.1, 0.91, 0.1), 2)
        dstar, info = min_ffd_d(lc, grid=grid, thresh=THRESH)
        dstars[p] = dstar if dstar is not None else 0.4
        hb(f"  d*[{p}]={dstars[p]} (win={len(ffd_weights(dstars[p], THRESH))})")
    return dstars


def build_ffd_year(df, dstars):
    """FFD features for one year. Returns DataFrame indexed by timestamp."""
    idx = df.index; n = len(df)
    ffd_eu = {}                                                    # eu-equiv FFD level per pair
    for p in PAIRS:
        lc = np.log(df[p].values)
        ffd_eu[p] = eu_sign(p) * ffd(lc, dstars[p], THRESH)
    F = np.vstack([ffd_eu[p] for p in PAIRS])
    factor = np.nanmean(F, axis=0)                                 # USD common FFD factor (eu-equiv)
    resid = {p: ffd_eu[p] - factor for p in PAIRS}
    feats = {"ffd_eur": ffd_eu["EURUSD"], "ffd_factor": factor, "ffd_resid_eur": resid["EURUSD"]}
    for p in NONEU:
        feats[f"ffd_resid_{p}"] = resid[p]                         # peer idiosyncratic FFD level
        for L in LAGS:
            lead = np.full(n, np.nan); lead[L:] = (resid[p] - resid["EURUSD"])[:-L]   # peer-vs-EUR FFD gap, lagged (lead-lag)
            feats[f"ffd_ll_{p}_{L}"] = lead
    return pd.DataFrame(feats, index=idx)


def main():
    MX.HOR = HOR; MX.GAP_S = HOR * 60                              # build_xp binds HOR at import — override the module attr
    hb(f"HOR={HOR}m (MX.HOR={MX.HOR}) : selecting per-pair d* on train log-close ...")
    dstars = select_dstar()
    hb("building base xp + ffd features per year ...")
    base_parts, ffd_parts = [], []
    for y in YEARS:
        df = read_closes(y)
        if df is None: hb(f"  {y}: skip"); continue
        bx = MX.build_xp([str(y)])                                 # certified base xp features (+ _y,_ts,_fwd, gate cols)
        fx = build_ffd_year(df, dstars)
        base_parts.append(bx); ffd_parts.append(fx)
        hb(f"  {y}: base {len(bx):,} ffd {len(fx):,}")
    B = pd.concat(base_parts); Ff = pd.concat(ffd_parts)
    B = B[~B.index.duplicated(keep="last")].sort_index()          # dedupe so .loc aligns 1:1 (no row expansion)
    Ff = Ff[~Ff.index.duplicated(keep="last")].sort_index()
    common = B.index.intersection(Ff.index)
    B = B.loc[common]; Ff = Ff.loc[common]
    assert len(B) == len(Ff), f"align mismatch {len(B)} vs {len(Ff)}"
    hb(f"aligned n={len(common):,}")

    xpc = [c for c in MX.xp_cols(B) if c in B.columns]             # certified model feature names
    ffc = list(Ff.columns)
    ts = B["_ts"].values.astype("int64"); fwd = B["_fwd"].values   # signed forward HOR return (NaN/0 already dropped in build_xp)
    ny = B["sess_ny"].values > 0.5
    Xb = B[xpc].astype(np.float32).values
    Xf = Ff.astype(np.float32).values
    fin = np.isfinite(Xb).all(1) & np.isfinite(Xf).all(1) & np.isfinite(fwd) & (fwd != 0) & ny
    hb(f"NY decision bars (finite base+ffd) n={fin.sum():,} up-rate={(fwd[fin] > 0).mean():.4f}")
    Xb, Xf, tgt, tsm = Xb[fin], Xf[fin], fwd[fin], ts[fin]

    arms = {"base": Xb, "+ffd": np.column_stack([Xb, Xf]).astype(np.float32), "ffdonly": Xf}
    res = forward_holdout(arms, target=tgt, ts=tsm, mode="direction", train_max=TRAIN_MAX,
                          test_years=(2024, 2025, 2026), cov=0.10, verbose=True)
    out = {"horizon_min": HOR, "design": "frozen-past forward holdout, direction cov0.10 selacc, NY; base=certified xp book",
           "d_star": dstars, "thresh": THRESH,
           "falsifier": "+ffd beats base selacc in >=2 forward years AND mean dSelacc>0 (no decay); ffdonly must clear .541",
           "by_arm": res["by_arm"], "deltas": res["deltas"], "deployable": res["deployable"]}
    json.dump(out, open(f"/media/sean/CORSAIR/binary-algo/frac_direction_{HOR}m_result.json", "w"), indent=1)
    hb(f"DONE -> frac_direction_{HOR}m_result.json")


if __name__ == "__main__":
    main()
