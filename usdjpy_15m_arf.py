"""USDJPY 15-MIN — C5 ONLINE ADAPTIVE FOREST direction lever (the efficiency keystone / ceiling confirmator).

SCOPE: USDJPY · 15m · NY session. Sibling of min1_online.py (the 1m online KEYSTONE @ AUC .50) retargeted
to USDJPY 15m. METHOD (C5): an ADAPTIVE, continuously-retraining model — river ARFClassifier + ADWIN drift
detection — trained PREQUENTIALLY (predict-then-learn) in strict chronological order on a COMPACT CAUSAL
feature subset. Warm-up learn-only on 2012-21, then predict-then-learn through 2022..2026 bars. If river is
unavailable we fall back to a PERIODIC-REFIT LGBM proxy (refit every ~1 month of bars on a trailing window).

WHY THIS LEVER, AND THE INTERPRETATION: an online learner with an explicit drift detector answers "is the
held-out wall stale-model drift, or genuine 15m efficiency?". The 1m sibling landed at AUC ~.50 (efficiency).
BUT USDJPY 15m is already CERTIFIED (own-pair NY GBM ~.539 AUC extractable, win-rate ~.58-.60) — so here the
online forest is a CONFIRMATOR OF THE CEILING, not of total efficiency: if it merely matches ~.50 that is the
*online learner's* residual, not proof the horizon is dead. It SURVIVES only if it beats the frozen incumbent.

EVAL DISCIPLINE (non-negotiable, from usdjpy_15m_base + the campaign rule):
  - FEATURES stay causal/continuous; only DECISION rows are restricted to the NY session (sessions.session_mask,
    DST-correct America/New_York 08-17).
  - The EVAL LABEL is ALWAYS the deriv-faithful fixed-15m sign = sign(close[t+15]-close[t]) with ties LOSING.
    This is exactly what base build() returns via (y on moved, moved bool). We do NOT re-derive it.
  - Per held-out year (test24/test25/oos): moved-AUC of the lever signal vs the up/down label on NY moved bars,
    AND a selective win-rate at cov3% via base side_eval (nonoverlap gap=900s built in). Sanity tripwire:
    moved up-rate must be ~0.47-0.53 (else flag the mirage).

INCUMBENT: certified NY own-pair GBM signal AUC ~.539 / win-rate ~.58-.60. SURVIVES only if held-out moved-AUC
exceeds ~.539 in a held-out year, OR NY cov3% win-rate CI-lower clears 0.541 in >=2 years.

Pre-registered falsifier is written to the result JSON BEFORE the held-out read.

Usage: ~/binary-algo-venv/bin/python usdjpy_15m_arf.py [stride]
  (stride>=6 passed to base build(); ARF additionally sub-streams via ON_STRIDE for throughput.)
"""
import os, sys, json, time, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import harness as H
from sessions import session_mask
from usdjpy_15m_base import build, side_eval, covcurve, boot, mk_lgb, BE, SPL, FEATS

PAIR = "USDJPY"; HOR = 15; GAP = 900
KEY = "arf"
# base build() stride (>=6 per memory budget); ARF sub-streams every ON_STRIDE-th NY-eval bar for throughput
def _argint(i, d): return int(sys.argv[i]) if len(sys.argv) > i and str(sys.argv[i]).isdigit() else d
TR_STRIDE = _argint(1, 6)
ON_STRIDE = int(os.environ.get("ON_STRIDE", "1"))   # extra stride on the prequential decision stream
RESULT = f"usdjpy_15m_{KEY}_result.json"

# Compact, CAUSAL feature subset (same family min1_online.py used; all 33 verified present in USDJPY base feats).
# Multi-TF momentum / vol / range / trend / session — kept small for online throughput.
COMPACT = ["1m_ret_1", "1m_ret_3", "1m_ret_6", "5m_ret_1", "5m_ret_3", "15m_ret_1",
           "1m_rsi", "5m_rsi", "15m_rsi", "1m_bb_width", "5m_bb_width", "15m_bb_width", "1m_atr_pct", "5m_atr_pct",
           "1m_dist_ema20", "5m_dist_ema20", "1m_rangepos_24", "5m_rangepos_24", "1m_autocorr_10", "5m_autocorr_10",
           "1m_macd_hist", "5m_macd_hist", "1m_bb_pctb", "5m_bb_pctb", "mtf_trend_align", "mtf_rsi_mean", "vol_z",
           "sess_ny", "sess_london", "sess_overlap", "hour_sin", "hour_cos", "gap_prev"]
COLS = [c for c in COMPACT if c in FEATS]            # defensive intersect with the 239 base names

# ---------------------------------------------------------------------------
# river availability probe (decides ARF vs periodic-refit LGBM proxy)
try:
    from river import forest, drift
    HAVE_RIVER = True
except Exception as _e:                              # pragma: no cover
    HAVE_RIVER = False
    _RIVER_ERR = repr(_e)


def _clean(df):
    """river/proxy ingestion: replace inf, fill NaN with 0.0 (min1_online.py rule), float32."""
    return df[COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")


def _stack(years, stride):
    """Pull base build() (deriv-faithful y/moved/ts on 239 feats), slice the COMPACT subset, sort chronologically.
    Returns Xc(float32 ndarray of COLS), y, moved, ts — all aligned, time-ordered."""
    X, y, mv, ts = build(years, stride)
    Xc = _clean(X).values
    order = np.argsort(ts, kind="stable")            # strict chronological order for prequential streaming
    return Xc[order], y[order], mv[order], ts[order]


def _arf_stream(Xw, Xe, ye, tse, t0, horizon_s=900):
    """Warm-up learn-only on Xw, then prequential predict-then-learn over Xe with a LEAKAGE-CORRECT
    DEFERRED-LEARN queue: a bar's 15m label (= sign(close[t+15]-close[t])) is only learned once its
    horizon has ELAPSED (realize_ts = ts+horizon_s <= current decision ts). This prevents the prior leak
    where learning bar i's future-dependent label before predicting overlapping bars i+1..i+14 leaked the
    next 15 minutes of price. Return p1 cache for Xe."""
    from collections import deque
    model = forest.ARFClassifier(n_models=5, max_depth=12, grace_period=300,
                                 drift_detector=drift.ADWIN(), warning_detector=drift.ADWIN(), seed=0)
    for i in range(len(Xw)):
        model.learn_one({c: float(v) for c, v in zip(COLS, Xw[i])}, int(_WARM_Y[i]))
        if i and i % 50000 == 0: print(f"[arf] warm {i}/{len(Xw)} {time.time()-t0:.0f}s", flush=True)
    print(f"[arf] warm-up done on {len(Xw)} bars {time.time()-t0:.0f}s", flush=True)
    pc = np.full(len(Xe), np.nan)
    q = deque()   # (realize_ts, xdict, label) — learn only after the horizon elapses
    for i in range(len(Xe)):
        now = int(tse[i])
        while q and q[0][0] <= now:                       # flush all labels realized by `now`
            _, xq, yq = q.popleft(); model.learn_one(xq, yq)
        xd = {c: float(v) for c, v in zip(COLS, Xe[i])}
        pp = model.predict_proba_one(xd); pc[i] = pp.get(1, 0.5) if pp else 0.5
        q.append((now + horizon_s, xd, int(ye[i])))       # DEFER learning until realized (no look-ahead)
        if i and i % 100000 == 0: print(f"[arf] prequential {i}/{len(Xe)} {time.time()-t0:.0f}s", flush=True)
    print(f"[arf] prequential done (deferred-learn, horizon {horizon_s}s) {time.time()-t0:.0f}s", flush=True)
    return pc


def _proxy_stream(Xw, yw, Xe, ye, tse, t0, refit_every_days=30, trail_days=540):
    """Periodic-refit LGBM proxy (river fallback): refit every ~refit_every_days of bars on a trailing window
    of the seen-so-far stream, then predict-then-extend. Adaptive in the same spirit as ARF but batch-refit."""
    refit_s = refit_every_days * 86400; trail_s = trail_days * 86400
    # seen pool starts with the warm-up; we extend it as we walk the eval stream (predict-then-learn order preserved)
    pool_X = [Xw]; pool_y = [yw]; pool_t = [np.full(len(Xw), tse[0] - 1, dtype=np.int64)] if len(Xw) else []
    pc = np.full(len(Xe), np.nan)
    clf = None; next_refit = tse[0] if len(tse) else 0
    sx = np.vstack(pool_X) if pool_X else np.empty((0, len(COLS)), np.float32)
    sy = np.concatenate(pool_y) if pool_y else np.empty(0, int)
    st = np.concatenate(pool_t) if pool_t else np.empty(0, np.int64)
    pend_X = []; pend_y = []; pend_t = []
    for i in range(len(Xe)):
        now = int(tse[i])
        if clf is None or now >= next_refit:
            if pend_X:                                # fold only pending rows whose label is REALIZED by `now` (leakage-correct)
                pt = np.array(pend_t, np.int64); rdy = pt <= now
                if rdy.any():
                    sx = np.vstack([sx, np.vstack([pend_X[k] for k in np.where(rdy)[0]])])
                    sy = np.concatenate([sy, np.array([pend_y[k] for k in np.where(rdy)[0]])])
                    st = np.concatenate([st, pt[rdy]])
                pend_X = [pend_X[k] for k in np.where(~rdy)[0]]; pend_y = [pend_y[k] for k in np.where(~rdy)[0]]; pend_t = [pend_t[k] for k in np.where(~rdy)[0]]
            keep = (st >= (now - trail_s)) & (st <= now)  # trailing window, realized labels only
            tx, ty = sx[keep], sy[keep]
            if len(np.unique(ty)) == 2 and len(ty) >= 2000:
                clf = mk_lgb(n=400, num_leaves=63); clf.fit(tx, ty)
            next_refit = now + refit_s
            print(f"[proxy] refit @{pd.to_datetime(now, unit='s')} ntrain={len(ty)} {time.time()-t0:.0f}s", flush=True)
        pc[i] = clf.predict_proba(Xe[i:i+1])[:, 1][0] if clf is not None else 0.5
        pend_X.append(Xe[i:i+1]); pend_y.append(int(ye[i])); pend_t.append(now + 900)   # realize_ts = ts+horizon (leakage-correct: only usable once label realized)
    return pc


def main():
    t0 = time.time()
    river_used = HAVE_RIVER
    lever = ("river ARFClassifier(n_models=5,max_depth=12,grace=300)+ADWIN, prequential predict-then-learn, "
             "compact causal 33-feat subset, NY decision rows") if river_used else \
            ("periodic-refit LGBM proxy (river unavailable): refit ~monthly on a trailing 540d window, "
             "predict-then-learn, compact 33-feat subset, NY decision rows")

    # ---- pre-register falsifier BEFORE any held-out read ----
    res = {"key": "USDJPY.15m.ny", "lever": lever, "engine": "river-arf" if river_used else "lgbm-proxy",
           "method": "C5 online adaptive forest (efficiency keystone / ceiling confirmator)",
           "settlement": "deriv-faithful 15m sign label, ties LOSE, BE 0.541, gap=900s nonoverlap_chrono; NY decision rows",
           "splits": SPL, "tr_stride": TR_STRIDE, "on_stride": ON_STRIDE, "n_compact_feats": len(COLS),
           "compact_feats": COLS,
           "falsifier": {"registered_utc": "pre-OOS",
               "KILL_if": ("held-out moved-AUC <= 0.51 in EVERY year (the efficiency/null reading) "
                           "OR it fails to beat the certified incumbent: no held-out year moved-AUC > 0.539 "
                           "AND NY cov3% win-rate CI95-lower clears 0.541 in < 2 years"),
               "SURVIVE_if": "held-out moved-AUC > ~0.539 in a year, OR NY cov3% WR CI-lo >= 0.541 in >=2 years",
               "rationale": ("1m online sibling -> AUC ~.50 (efficiency). USDJPY 15m IS certified (own-pair NY GBM "
                             "~.539 AUC, WR ~.58-.60), so an online forest here CONFIRMS THE CEILING, not total "
                             "efficiency: matching ~.50 just bounds the online learner's residual. Sign-invariance "
                             "note (arXiv:2512.15720): state/vol gates move SIZE not SIGN — but we RUN the direction "
                             "test faithfully and let AUC<=.51 falsify if it is only a magnitude signal.")},
           "incumbent": {"certified_ny_gbm_auc": 0.539, "certified_ny_gbm_wr": "~.58-.60"}}
    json.dump(res, open(RESULT, "w"), indent=2)

    if not river_used:
        res["river_error"] = _RIVER_ERR
        print(f"[arf] river UNAVAILABLE ({_RIVER_ERR}) -> periodic-refit LGBM proxy", flush=True)

    # ---- build streams: warm-up (train 2012-21) learn-only; eval = val+held-out, predict-then-learn ----
    global _WARM_Y
    Xw, _WARM_Y, _mw, _tw = _stack(SPL["train"], TR_STRIDE)
    EVAL_YEARS = SPL["val"] + SPL["test24"] + SPL["test25"] + SPL["oos"]
    Xe, ye, me, tse = _stack(EVAL_YEARS, 1)
    if ON_STRIDE > 1:                                  # extra throughput stride on the prequential stream
        Xe, ye, me, tse = Xe[::ON_STRIDE], ye[::ON_STRIDE], me[::ON_STRIDE], tse[::ON_STRIDE]
    yr = pd.to_datetime(tse, unit="s").year.values
    print(f"[arf] engine={'river-arf' if river_used else 'lgbm-proxy'} stride={TR_STRIDE} on_stride={ON_STRIDE} "
          f"feats={len(COLS)} warm={len(Xw):,} eval={len(Xe):,} build={time.time()-t0:.0f}s", flush=True)

    # ---- adaptive stream: prequential predict-then-learn over the WHOLE eval span ----
    if river_used:
        pc = _arf_stream(Xw, Xe, ye, tse, t0)
    else:
        pc = _proxy_stream(Xw, _WARM_Y, Xe, ye, tse, t0)

    # ---- VAL signal AUC (NY moved bars) — diagnostic, NOT a gate selector (online learner is gate-free) ----
    ny = session_mask(tse, "ny")
    vmask = np.isin(yr, [int(y) for y in SPL["val"]]) & ny & me & np.isfinite(pc)
    val_auc = float(roc_auc_score(ye[vmask], pc[vmask])) if vmask.sum() > 50 else float("nan")
    res["val_or_signal_auc"] = val_auc
    print(f"[arf] VAL(NY,moved) signal-AUC={val_auc:.4f} n={int(vmask.sum())} {time.time()-t0:.0f}s", flush=True)

    # ---- fixed cov3% gate (online learner has no fit-time threshold; use the cov3 quantile of NY conf) ----
    COV = 0.03

    # ---- held-out per year: moved-AUC + cov3% selective WR (NY decision rows), via base side_eval/covcurve ----
    res["years"] = {}
    for w, yy in (("test24", 2024), ("test25", 2025), ("oos", 2026)):
        m = np.isin(yr, [yy]) & ny & np.isfinite(pc)   # NY decision rows for this held-out year
        idx = np.where(m)[0]
        pr = pc[idx]; ywi = ye[idx]; mwi = me[idx]; twi = tse[idx]
        auc = float(roc_auc_score(ywi[mwi], pr[mwi])) if mwi.sum() > 20 else float("nan")
        up_rate = float(ywi[mwi].mean()) if mwi.sum() > 0 else float("nan")
        conf = np.abs(pr - 0.5)
        thr = float(np.quantile(conf, 1 - COV)) if len(conf) else 0.0
        gate = side_eval(pr, ywi, mwi, twi, thr)        # cov3% selective win-rate, nonoverlap gap=900 built in
        cc = covcurve(pr, ywi, mwi, twi)
        g = gate["COMBINED"] if gate else {"n": 0, "wr": float("nan"), "ci": [float("nan")] * 2}
        u = gate["UP"] if gate else {"n": 0, "wr": float("nan"), "ci": [float("nan")] * 2}
        d = gate["DOWN"] if gate else {"n": 0, "wr": float("nan"), "ci": [float("nan")] * 2}
        res["years"][w] = {"auc": auc, "moved_up_rate": up_rate, "n_moved": int(mwi.sum()),
                           "cov3_thr": thr, "cov3_wr": g["wr"], "cov3_n": g["n"], "cov3_ci": g["ci"],
                           "gate": gate, "covcurve": cc,
                           "tripwire_ok": bool(0.47 <= up_rate <= 0.53)}
        print(f"=== {w} === moved-AUC={auc:.4f} up-rate={up_rate:.4f} (tw_ok={0.47<=up_rate<=0.53}) | "
              f"cov3%: COMB n{g['n']} wr={g['wr']:.4f} CI[{g['ci'][0]:.3f},{g['ci'][1]:.3f}] | "
              f"UP n{u['n']} {u['wr']:.4f} CI[{u['ci'][0]:.3f},{u['ci'][1]:.3f}] | "
              f"DOWN n{d['n']} {d['wr']:.4f} CI[{d['ci'][0]:.3f},{d['ci'][1]:.3f}]", flush=True)

    # ---- apply pre-registered falsifier ----
    aucs = {w: res["years"][w]["auc"] for w in ("test24", "test25", "oos")}
    beats_auc = [w for w, a in aucs.items() if np.isfinite(a) and a > 0.539]
    cov3_clears = [w for w in ("test24", "test25", "oos")
                   if res["years"][w]["gate"] and res["years"][w]["gate"]["COMBINED"]["ci"][0] >= BE]
    all_null = all((not np.isfinite(a)) or a <= 0.51 for a in aucs.values())
    beats_base = (len(beats_auc) >= 1) or (len(cov3_clears) >= 2)
    killed = (not beats_base) or all_null
    res["verdict"] = {"KILLED": bool(killed), "beats_base_auc": bool(len(beats_auc) >= 1),
                      "years_auc_gt_0539": beats_auc, "cov3_COMBINED_CIlo_clears_BE": cov3_clears,
                      "all_years_auc_le_0.51": bool(all_null),
                      "note": ("Online adaptive forest as a CEILING CONFIRMATOR for the already-certified USDJPY "
                               "15m NY direction edge. SURVIVES only if held-out moved-AUC beats the certified "
                               "~.539 incumbent in >=1 year, OR cov3% WR CI-lo clears 0.541 in >=2 years. "
                               "AUC~.50 every year => the online learner bounds its own residual at the ceiling "
                               "(NOT proof of total efficiency, since the frozen GBM is certified).")}
    json.dump(res, open(RESULT, "w"), indent=2, default=float)
    print(f"\n[arf] VERDICT: {'KILLED' if killed else 'SURVIVED'} "
          f"(val_auc={val_auc:.4f}; auc>.539 yrs={beats_auc}; cov3-clears={cov3_clears}; all-null={all_null}) "
          f"-> {RESULT}  total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
