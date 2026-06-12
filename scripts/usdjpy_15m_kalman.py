"""USDJPY 15m NY — C2 KALMAN forward-FILTER direction lever (local-level+slope, CAUSAL).

SCOPE: USDJPY · 15m · NY. Method spec C2: run a local-linear-trend (local-level + slope/velocity)
Kalman FILTER on log-close, forward recursion ONLY (state at t uses observations <= t; NEVER the RTS
smoother — the smoother peeks at the future = leakage, see min1_kalman.py docstring). The DIRECTION
signal = the filtered SLOPE (velocity): predict UP when slope>0, DOWN when slope<0; |slope| = magnitude
for the selective gate. This is the smoothed-momentum / trend lever at the 15m horizon.

EVAL DISCIPLINE (deriv-faithful, non-negotiable):
  * The EVAL label is ALWAYS the deriv-faithful fixed-15m sign = sign(close[t+15]-close[t]) with ties
    (move==0) LOSING. We re-derive it here with the SAME contiguity rule as usdjpy_15m_base.build()
    (require ts[t+15]-ts[t]==900 == 15 clean 60s steps) so y/moved are identical to the template's.
  * Lever signal (filtered slope) is CAUSAL/continuous over the full grid; only the DECISION rows are
    restricted to the NY session (America/New_York 08-17, DST-correct) via sessions.session_mask.
  * Per held-out year (test24/test25/oos): moved-AUC of the lever signal vs the up/down label on NY
    MOVED bars, AND a selective win-rate at cov3% (nonoverlap gap=900 is built into side_eval — we
    feed the lever's |slope| as a pseudo-confidence pr=0.5+sign*scaled|slope| so side_eval's
    conf=|pr-0.5| gate and de-overlap apply unchanged).
  * Sanity tripwire: moved up-rate must be ~0.47-0.53 (else the AUC is a base-rate mirage).

This lever is PRICE-ONLY -> a close-only loader (NOT build's 239 feats) keeps RAM/runtime tiny.
The Kalman (q_level,q_slope,r) hyper + signal scale are SELECTED on VAL=2022-2023 by signal moved-AUC
(no held-out peeking); the held-out years are read only AFTER the falsifier is pre-registered to disk.

INCUMBENT to beat: certified NY own-pair GBM signal AUC ~.539 / win-rate ~.58-.60.
FALSIFIER (pre-registered): KILL if held-out moved-AUC <= .51 (trend dead -> filter anti-predictive or
flat). SURVIVE only if held-out moved-AUC > ~.539 OR cov3% win-rate CI-lower clears 0.541 in >=2 years.
If moved-AUC < .50 -> note ANTI-PREDICTIVE (reversion: the trend signal is the wrong sign at 15m).

Usage: /home/sean/binary-algo-venv/bin/python usdjpy_15m_kalman.py
"""
import os, json, time, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sessions import session_mask
from usdjpy_15m_base import side_eval, boot, BE, SPL

PAIR = "USDJPY"; HOR = 15; STEP = 60; GAP = HOR * STEP   # 900s horizon, 900s nonoverlap gap
FEAT = "/home/sean/git/binary-algo/features"
RESULT = "usdjpy_15m_kalman_result.json"

# Kalman process/measurement-noise grid + slope-signal scales, selected on VAL only.
# (q_level, q_slope): process-noise for the local level and the slope state; r: measurement noise.
Q_GRID = [(1e-8, 1e-10, 1e-6), (1e-7, 1e-9, 1e-6), (1e-6, 1e-8, 1e-6), (1e-7, 1e-10, 1e-5)]


def load_close_year(y):
    """Close-only loader mirroring usdjpy_15m_base.build()'s label/contiguity EXACTLY.

    Returns (logclose, ts_epoch_s, y_updown, moved_bool) over the FULL contiguous-valid grid:
      label_t = sign(close[t+HOR]-close[t]); requires ts[t+HOR]-ts[t]==GAP (no intervening gap);
      moved = valid & (fwd_ret != 0); ties kept as rows with moved=False so the win-rate charges them.
    y/moved here are bit-identical to build()'s on the same rows (same close, same contiguity test)."""
    p = f"{FEAT}/{PAIR}_{y}.parquet"
    if not os.path.exists(p):
        return None
    d = pd.read_parquet(p, columns=["close"])
    d = d[~d.index.duplicated(keep="last")]
    c = d["close"].values.astype(float)
    ts = d.index.values.astype("datetime64[s]").astype("int64"); n = len(d)
    contig = np.zeros(n, bool); contig[:n - HOR] = (ts[HOR:] - ts[:-HOR]) == GAP   # exactly HOR clean 60s steps
    fr = np.full(n, np.nan); fr[:n - HOR] = c[HOR:] / c[:-HOR] - 1.0
    valid = contig & np.isfinite(fr) & (c > 0)
    moved = valid & (fr != 0.0)
    yud = (fr > 0).astype(int)
    lp = np.log(c)
    return lp, ts, yud, moved, valid


def kalman_llt(y, q_level, q_slope, r):
    """Local-linear-trend Kalman FILTER (forward recursion ONLY, causal). y = log price.
    State x=[level,slope]; F=[[1,1],[0,1]] (slope integrates into level); obs = level + noise.
    Returns filtered (level, slope) per step; state at t uses observations <= t (NO smoother)."""
    n = len(y)
    F = np.array([[1.0, 1.0], [0.0, 1.0]]); Hm = np.array([1.0, 0.0])
    Q = np.array([[q_level, 0.0], [0.0, q_slope]]); R = r
    x = np.array([y[0], 0.0]); P = np.eye(2) * 1e-4
    lev = np.empty(n); slp = np.empty(n)
    for t in range(n):
        x = F @ x; P = F @ P @ F.T + Q                       # predict
        S = Hm @ P @ Hm + R; K = (P @ Hm) / S                # update with obs y[t]
        x = x + K * (y[t] - Hm @ x); P = P - np.outer(K, Hm @ P)
        lev[t] = x[0]; slp[t] = x[1]
    return lev, slp


def signal_to_pr(slope, scale):
    """Map the (causal) filtered slope to a pseudo-probability pr in (0,1) for side_eval.
    side_eval reads direction from pr>0.5 and confidence from |pr-0.5|; a monotone squashing of the
    SLOPE preserves both sign (=direction) and magnitude ranking (=gate), so cov3% selects the
    highest-|slope| NY bars. AUC is computed directly on `slope` (rank-only), so `scale` is irrelevant
    to AUC and only sets the gate granularity for side_eval."""
    return 1.0 / (1.0 + np.exp(-slope / scale))


def build_signals(years):
    """Per window: concat causal Kalman slope per Q-config + the aligned label arrays + NY mask.
    Signals are computed PER YEAR (filter does not span year boundaries / data gaps), then concatenated."""
    per = {i: [] for i in range(len(Q_GRID))}
    yud = []; mv = []; ts = []
    for y in years:
        z = load_close_year(y)
        if z is None:
            continue
        lp, tsy, yudy, mvy, _ = z
        yud.append(yudy); mv.append(mvy); ts.append(tsy)
        for i, (ql, qs, r) in enumerate(Q_GRID):
            _, slp = kalman_llt(lp, ql, qs, r)
            per[i].append(slp)
    if not ts:
        return None
    ts = np.concatenate(ts)
    return dict(slope=[np.concatenate(per[i]) for i in range(len(Q_GRID))],
                y=np.concatenate(yud), moved=np.concatenate(mv), ts=ts,
                ny=session_mask(ts, "ny"))


def moved_auc(slope, y, moved, ny):
    """AUC of the lever slope vs up/down label on NY MOVED bars (direction test)."""
    m = ny & moved & np.isfinite(slope)
    if m.sum() < 50 or len(np.unique(y[m])) < 2:
        return float("nan"), int(m.sum())
    return float(roc_auc_score(y[m], slope[m])), int(m.sum())


def main():
    t0 = time.time()
    # ---- PRE-REGISTER falsifier to disk BEFORE any held-out read ----
    res = {"key": "USDJPY.15m.ny",
           "lever": "C2 Kalman forward-FILTER (local-level+slope), CAUSAL; DIRECTION = sign/magnitude of filtered slope (UP if slope>0)",
           "settlement": "deriv-faithful fixed-15m sign(close[t+15]-close[t]), ties LOSE, BE=0.541, gap=900 nonoverlap (via side_eval); DECISION rows = NY session only",
           "splits": SPL, "q_grid": [list(g) for g in Q_GRID],
           "incumbent": "certified NY own-pair GBM signal AUC ~.539 / win-rate ~.58-.60",
           "falsifier": {"registered_utc": "pre-OOS",
                         "KILL_if": "held-out moved-AUC <= 0.51 (trend dead -> filter anti-predictive or flat)",
                         "SURVIVE_if": "held-out moved-AUC > ~0.539 OR cov3% win-rate CI95-lower clears 0.541 in >=2 held-out years",
                         "anti_predictive_note": "if moved-AUC < 0.50 the slope is the WRONG sign at 15m => 15m is mean-reverting, lever should be inverted (reversion), still report as KILLED for the trend direction tested"}}
    json.dump(res, open(RESULT, "w"), indent=2)

    # ---- VAL selection: pick Q-config by VAL signal moved-AUC on NY moved bars (no held-out peek) ----
    val = build_signals(SPL["val"])
    if val is None:
        res["verdict"] = {"KILLED": True, "beats_base_auc": False, "note": "no VAL data found"}
        json.dump(res, open(RESULT, "w"), indent=2)
        print("[kalman] no VAL data", flush=True); return
    best = None
    for i in range(len(Q_GRID)):
        a, nv = moved_auc(val["slope"][i], val["y"], val["moved"], val["ny"])
        print(f"[kalman] VAL Qidx={i} {Q_GRID[i]} signal moved-AUC(NY)={a:.4f} (n={nv}) {time.time()-t0:.0f}s", flush=True)
        # select on |AUC-0.5| so we capture a strong-but-anti-predictive config too (it would still flag the falsifier);
        # tie-break toward predictive (AUC>0.5). Primary objective = max moved-AUC (trend direction as registered).
        score = a
        if best is None or (np.isfinite(score) and score > best[0]):
            best = (score, i, a)
    _, QI, val_auc = best
    ql, qs, r = Q_GRID[QI]
    # signal scale = VAL std of the chosen slope (sets side_eval gate granularity; AUC is scale-free)
    vs = val["slope"][QI]; scale = float(np.nanstd(vs[np.isfinite(vs)])) or 1e-9
    res["selected_qidx"] = QI; res["selected_q"] = [ql, qs, r]; res["signal_scale"] = scale
    res["val_or_signal_auc"] = float(val_auc)
    print(f"[kalman] SELECTED Qidx={QI} q=({ql:.0e},{qs:.0e},{r:.0e}) VAL signal moved-AUC={val_auc:.4f} scale={scale:.3e}", flush=True)

    # ---- held-out per year: moved-AUC + cov3% selective win-rate (COMBINED/UP/DOWN via side_eval) ----
    res["years"] = {}
    for w in ("test24", "test25", "oos"):
        D = build_signals(SPL[w])
        if D is None:
            res["years"][w] = {"auc": None, "note": "no data"}; continue
        slope = D["slope"][QI]; ny = D["ny"]
        auc, n_moved = moved_auc(slope, D["y"], D["moved"], ny)
        wi = np.where(ny)[0]                                  # DECISION rows restricted to NY
        pr = signal_to_pr(slope[wi], scale)                  # pseudo-prob: sign=direction, |.-.5|=magnitude gate
        ywi = D["y"][wi]; mwi = D["moved"][wi]; twi = D["ts"][wi]
        up_rate = float(ywi[mwi].mean()) if mwi.sum() else float("nan")
        finite = np.isfinite(pr)
        thr3 = float(np.quantile(np.abs(pr[finite] - 0.5), 0.97)) if finite.sum() else 0.0   # cov3% on NY
        gate = side_eval(pr, ywi, mwi, twi, thr3)
        g = gate["COMBINED"] if gate else None
        res["years"][w] = {
            "auc": auc, "n_moved_ny": n_moved, "moved_up_rate": up_rate,
            "tripwire_ok": bool(np.isfinite(up_rate) and 0.47 <= up_rate <= 0.53),
            "cov3_thr": thr3,
            "cov3_wr": (round(g["wr"], 4) if g else None),
            "cov3_n": (g["n"] if g else 0),
            "cov3_ci": ([round(c, 4) for c in g["ci"]] if g else None),
            "cov3_UP": ({"n": gate["UP"]["n"], "wr": round(gate["UP"]["wr"], 4), "ci": [round(c, 4) for c in gate["UP"]["ci"]]} if gate and gate["UP"]["n"] else None),
            "cov3_DOWN": ({"n": gate["DOWN"]["n"], "wr": round(gate["DOWN"]["wr"], 4), "ci": [round(c, 4) for c in gate["DOWN"]["ci"]]} if gate and gate["DOWN"]["n"] else None)}
        cilo = res["years"][w]["cov3_ci"][0] if res["years"][w]["cov3_ci"] else float("nan")
        print(f"=== {w} === moved-AUC={auc if auc==auc else float('nan'):.4f} up-rate={up_rate:.4f} tripwire={res['years'][w]['tripwire_ok']} | "
              f"cov3% COMB n{res['years'][w]['cov3_n']} wr={res['years'][w]['cov3_wr']} CI[{cilo:.3f},...]", flush=True)

    # ---- apply pre-registered falsifier ----
    aucs = [res["years"][w]["auc"] for w in ("test24", "test25", "oos") if res["years"][w].get("auc") is not None]
    min_auc = min(aucs) if aucs else float("nan")
    max_auc = max(aucs) if aucs else float("nan")
    beats_base = bool(np.isfinite(min_auc) and min_auc > 0.539)          # held-out AUC beats incumbent (worst-year)
    clears_be = [w for w in ("test24", "test25", "oos")
                 if res["years"][w].get("cov3_ci") and res["years"][w]["cov3_ci"][0] >= BE]
    survive_wr = len(clears_be) >= 2
    kill_auc = bool(np.isfinite(max_auc) and max_auc <= 0.51)            # KILL: best held-out year still <= .51
    anti = bool(np.isfinite(min_auc) and min_auc < 0.50)
    killed = bool(kill_auc and not survive_wr) or (not beats_base and not survive_wr)
    note = ("ANTI-PREDICTIVE: held-out moved-AUC<0.50 => 15m slope/trend is the WRONG sign (mean-reverting). "
            if anti else "")
    note += (f"min held-out moved-AUC={min_auc:.4f} (incumbent ~.539); cov3% CI-lo clears {BE} in years {clears_be}. "
             f"{'SURVIVES' if (beats_base or survive_wr) and not kill_auc else 'KILLED'} the trend-direction test." if aucs
             else "no held-out data.")
    res["verdict"] = {"KILLED": killed, "beats_base_auc": beats_base,
                      "min_holdout_auc": (round(min_auc, 4) if np.isfinite(min_auc) else None),
                      "cov3_CIlo_clears_BE_years": clears_be, "anti_predictive": anti, "note": note}
    json.dump(res, open(RESULT, "w"), indent=2)
    print(f"\n[kalman] VERDICT: {'KILLED' if killed else 'SURVIVED'} (min held-out moved-AUC="
          f"{min_auc if min_auc==min_auc else float('nan'):.4f}; cov3-clears={clears_be}; anti={anti}) -> {RESULT}  total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
