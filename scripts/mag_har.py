"""PHASE 2 — HAR / realized-measure MAGNITUDE workhorses (the econometric-vol canon the research critic flagged missing).

Tests whether jump-decomposed HAR (bipower), realized SEMIVARIANCE (RS+/RS-, Patton-Sheppard good/bad vol), realized
QUARTICITY (HARQ attenuation), and the signed-jump add FORWARD-ROBUST magnitude signal ON TOP of the certified base model
[-perm_entropy, rv30, rv120]. All are within-window return-moment decompositions -> forward-robust BY CONSTRUCTION (no
calendar/seasonal memorization), the opposite of the trap-#9 time-of-day lever.

GATE (NOVEL_METHODS_RESEARCH §0/§2): the DECISIVE test is the frozen-past FORWARD HOLDOUT (fwd_holdout.py), not pooled CPCV.
Falsifier per family: beat base-rv forward by +0.005 AUC in >=2 of {2024,2025,2026} AND >=2 of horizons {10,15,30}; HARQ
additionally must not be collinear with rv (|corr|<0.9). Winners (if any) escalate to a pooled-CPCV + deflation confirm.

Realized measures over rolling window W on 1-min log-returns r (HAR scales W in {30,120,480} = ~0.5h/2h/8h):
  RV_W  = Σ r²                          (realized variance, sum-of-squares; base uses rolling STD instead)
  BV_W  = (π/2) Σ |r_i||r_{i-1}|         (bipower variation — jump-robust continuous variation)
  RQ_W  = (W/3) Σ r⁴                    (realized quarticity — RV's measurement-error scale, HARQ)
  RS±_W = Σ r²·1(r≷0)                   (realized semivariance, upside/downside)
  Jump_W= max(RV_W−BV_W, 0);  SJ_W = RS⁺_W−RS⁻_W   (jump size;  signed jump = directional vol asymmetry)

Run: ~/binary-algo-venv/bin/python mag_har.py            # horizons 10,15,30 ; -> mag_har_result.json
"""
import os, sys, json, time, math, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from fwd_holdout import forward_holdout, mk_lgb
import harness as H

FEAT = H.FEAT_DIR; PAIR = "EURUSD"; YEARS = list(range(2012, 2027))
HORIZONS = [10, 15, 30]; WINS = [30, 120, 480]
T0 = time.time()
def hb(m): print(f"[{time.time()-T0:.0f}s] {m}", flush=True)


def perm_entropy(r, d=4, tau=1, W=120):
    N = len(r); Lg = (d-1)*tau
    if N <= Lg+1: return np.full(N, np.nan)
    idx = np.arange(N-Lg)[:, None] + np.arange(0, d*tau, tau)[None, :]
    order = np.argsort(r[idx], axis=1, kind="stable"); code = (order*(d**np.arange(d))).sum(1).astype(np.int32)
    M = len(code); nb = d**d
    oh = np.zeros((M, nb), dtype=np.float32); oh[np.arange(M), code] = 1.0
    cs = np.cumsum(oh, axis=0); cnt = cs.copy(); cnt[W:] = cs[W:]-cs[:-W]
    pp = cnt/np.maximum(cnt.sum(1, keepdims=True), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(pp > 0, pp*np.log(pp), 0.0), axis=1)/math.log(math.factorial(d))
    out = np.full(N, np.nan); out[Lg:Lg+M] = ent; out[:Lg+W] = np.nan; return out


def realized_measures(r):
    """All rolling realized measures from 1-min log-returns r (np array). Returns dict of float32 columns (NaN warmup)."""
    s = pd.Series(r); r2 = s*s; absr = s.abs()
    bp = absr * absr.shift(1)                                      # |r_i||r_{i-1}| for bipower
    pos2 = r2.where(s > 0, 0.0); neg2 = r2.where(s < 0, 0.0)
    out = {}
    for W in WINS:
        RV = r2.rolling(W).sum()
        BV = (math.pi/2.0) * bp.rolling(W).sum()
        RQ = (W/3.0) * (r2*r2).rolling(W).sum()
        RSp = pos2.rolling(W).sum(); RSm = neg2.rolling(W).sum()
        jump = np.maximum(RV - BV, 0.0)
        out[f"lRV{W}"]   = np.log(RV + 1e-18)                      # log realized variance (proper sum-of-squares form)
        out[f"lBV{W}"]   = np.log(BV + 1e-18)                      # log continuous (jump-robust) variation
        out[f"jump{W}"]  = jump                                    # jump variation size
        out[f"jfrac{W}"] = jump / (RV + 1e-18)                     # jump fraction of total variance
        out[f"RSp{W}"]   = RSp; out[f"RSm{W}"] = RSm               # up / down semivariance
        out[f"SJ{W}"]    = RSp - RSm                               # signed jump (good−bad vol)
        out[f"RSasym{W}"]= (RSp - RSm) / (RV + 1e-18)             # semivariance asymmetry, scale-free
        out[f"sqRQ{W}"]  = np.sqrt(RQ)                             # HARQ attenuation scale
        out[f"rvQ{W}"]   = np.log(RV + 1e-18) * np.sqrt(RQ)        # HARQ interaction lRV·√RQ
    return {k: v.values.astype(np.float32) if hasattr(v, "values") else np.asarray(v, np.float32) for k, v in out.items()}


def _contig_fwd(idx_secs, c, hor):
    n = len(c); contig = np.zeros(n, bool)
    if n > hor: contig[:n-hor] = (idx_secs[hor:] - idx_secs[:-hor]) == hor*60
    fwd = np.full(n, np.nan); fwd[:n-hor] = c[hor:]; ret = fwd / c - 1.0
    return ret, contig


def load_all():
    """Per-year: base [-pe,rv30,rv120], all realized measures, ts, close, and forward |ret| for each horizon.
    Validity = finite(base) & finite(all-measures); per-horizon target validity adds contiguity & finite ret."""
    cols = {}; bases = []; tss = []; arets = {h: [] for h in HORIZONS}; valids_extra = []
    meas_keys = None
    for y in YEARS:
        p = f"{FEAT}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        df = pd.read_parquet(p, columns=["close"]); df = df[~df.index.duplicated(keep="last")].sort_index()
        c = df["close"].values.astype(float); secs = df.index.values.astype("datetime64[s]").astype("int64")
        r = np.zeros(len(c)); r[1:] = np.diff(np.log(c))
        pe = perm_entropy(r, 4, 1, 120); rs = pd.Series(r)
        rv30 = rs.rolling(30).std().values; rv120 = rs.rolling(120).std().values
        meas = realized_measures(r)
        if meas_keys is None: meas_keys = list(meas.keys())
        base = np.column_stack([-pe, rv30, rv120]).astype(np.float32)
        M = np.column_stack([meas[k] for k in meas_keys]).astype(np.float32)
        finite_base = np.isfinite(base).all(1) & np.isfinite(M).all(1)
        bases.append(base)
        tss.append(secs); valids_extra.append(finite_base)
        for h in HORIZONS:
            ret, contig = _contig_fwd(secs, c, h)
            ar = np.full(len(c), np.nan); ok = contig & np.isfinite(ret)
            ar[ok] = np.abs(ret[ok]); arets[h].append(ar.astype(np.float32))
        cols.setdefault("M", []).append(M)
        del df
    base = np.concatenate(bases); ts = np.concatenate(tss); M = np.concatenate(cols["M"])
    fin = np.concatenate(valids_extra)
    aret = {h: np.concatenate(arets[h]) for h in HORIZONS}
    o = np.argsort(ts, kind="stable")
    base, ts, M, fin = base[o], ts[o], M[o], fin[o]
    aret = {h: aret[h][o] for h in HORIZONS}
    return base, M, dict(zip(meas_keys, M.T)), meas_keys, aret, ts, fin


def main():
    hb("loading magnitude substrate + realized measures ...")
    base, M, mdict, mkeys, aret, ts, fin = load_all()
    hb(f"pooled n={len(ts):,} (finite base+measures: {fin.sum():,}) ; measures={len(mkeys)}")

    def cols(*keys): return np.column_stack([mdict[k] for k in keys]).astype(np.float32)
    # arm feature blocks added ON TOP of base
    har   = cols(*[f"lRV{W}" for W in WINS])                                   # multi-scale realized variance (log)
    jumpf = cols("lBV120", "jump120", "jfrac120", "jump480", "jfrac480")       # continuous/jump decomposition
    semiv = cols("RSp120", "RSm120", "SJ120", "RSasym120", "SJ480", "RSasym480")
    harq  = cols("sqRQ120", "rvQ120", "sqRQ480", "rvQ480")                     # realized-quarticity attenuation

    arms_extra = {"+har": har, "+jump": jumpf, "+semivar": semiv, "+harq": harq,
                  "+all": np.column_stack([har, jumpf, semiv, harq]).astype(np.float32)}

    # HARQ collinearity guard (must be < 0.9 vs rv120 = base col 2)
    rv120 = base[:, 2]
    collin = {k: round(float(np.corrcoef(np.nan_to_num(mdict[k][fin]), rv120[fin])[0, 1]), 3)
              for k in ["sqRQ120", "rvQ120", "lRV120", "RSp120", "RSm120", "SJ120"]}
    hb(f"collinearity vs rv120: {collin}")

    result = {"design": "frozen-past forward holdout (train<=2023 -> 2024/2025/2026), base=[-pe,rv30,rv120]",
              "falsifier": "+0.005 AUC vs base in >=2 forward years AND >=2 horizons; HARQ |corr rv120|<0.9",
              "collinearity_vs_rv120": collin, "by_horizon": {}}
    for h in HORIZONS:
        m = fin & np.isfinite(aret[h])
        b = base[m]; tgt = aret[h][m]; tsm = ts[m]
        arms = {"base": b}
        for nm, blk in arms_extra.items():
            arms[nm] = np.column_stack([b, blk[m]]).astype(np.float32)
        hb(f"=== horizon {h}m : n={m.sum():,} ===")
        res = forward_holdout(arms, target=tgt, ts=tsm, mode="magnitude", train_max=2023,
                              test_years=(2024, 2025, 2026), verbose=True)
        result["by_horizon"][str(h)] = {"by_arm": res["by_arm"], "deltas": res["deltas"], "deployable": res["deployable"]}

    # cross-horizon verdict: an arm "passes" if deployable (no decay) in >=2 horizons AND mean forward dAUC>=+0.005
    verdict = {}
    for nm in arms_extra:
        depl = sum(int(result["by_horizon"][str(h)]["deployable"].get(nm, False)) for h in HORIZONS)
        ds = [d for h in HORIZONS for d in result["by_horizon"][str(h)]["deltas"].get(nm, {}).values()]
        mean_d = round(float(np.mean(ds)), 4) if ds else 0.0
        passes = bool(depl >= 2 and mean_d >= 0.005)
        if nm == "+harq": passes = passes and abs(collin["rvQ120"]) < 0.9 and abs(collin["sqRQ120"]) < 0.9
        verdict[nm] = dict(deployable_horizons=depl, mean_fwd_dAUC=mean_d, PASSES=passes)
    result["verdict"] = verdict
    json.dump(result, open(f"/home/sean/git/binary-algo/mag_har_result.json", "w"), indent=1)
    hb("VERDICT: " + " | ".join(f"{k} {v['mean_fwd_dAUC']:+.4f} depl{v['deployable_horizons']}/3 {'PASS' if v['PASSES'] else 'kill'}" for k, v in verdict.items()))
    hb("DONE -> mag_har_result.json")


if __name__ == "__main__":
    main()
