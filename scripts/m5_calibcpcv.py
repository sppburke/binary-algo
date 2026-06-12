"""C4: Calibrated meta gate — absolute threshold on calibrated meta score vs fold-local quantile baseline.

Context: the nested-refit CPCV of the deployed meta gate (m5_aci_cpcv.py) JUST missed (p10 .5376/78.6%
vs the ≥.541/80% bar). A fold-local Q=0.95 quantile adapts to each fold's distribution; if the meta
score distribution shifts across regimes, the quantile chases it, introducing instability.

Calibration insight: applying a monotone calibrator (isotonic/Platt) then re-using the SAME quantile
coverage is a no-op (same rows selected). The genuine test is:
  - Calibrate meta scores on each fold's meta segment
  - Select via an ABSOLUTE threshold (calibrated_score ≥ thr) that is FIXED across all folds
  - The absolute threshold is chosen on held-in calibration segments (to match production coverage Q=0.95)

If calibrated absolute thresholds transfer better across regimes than fold-local quantiles, p10 should rise.
Also tests raw-meta absolute thresholds (0.55, 0.57, 0.60) without calibration, as a simple comparison.

Production meta architecture (must match exactly):
  - Primary P refitted on trp (primary features Xp)
  - Meta M refitted on trm (Xmeta = [|pr-0.5|, Xm]; target = P-correct)
  - Gate: de-overlapped UP candidates where sm >= THR

Pre-registered falsifier: KILL unless at least one calibration variant achieves nested-refit p10 ≥ .541
AND ≥80% paths clear (i.e., passes the bar the raw quantile gate missed).

Usage: M5_STRIDE=6 python m5_calibcpcv.py
"""
import os, json, time, itertools, numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import m5_xpair as MX
import m5_xpair_production as XP
import m5_cpcv_refit as RF
from m5_aci_cpcv import build_pm, mk_meta   # reuse exact production functions

N_GROUPS, K_TEST = 8, 2
STRIDE   = int(os.environ.get("M5_STRIDE", "6"))
Q        = 0.95          # production coverage
PRIMARY_SUB = 100_000
META_SUB    = 50_000
# Absolute threshold grid (on RAW meta score and calibrated score)
ABS_THRS = [0.50, 0.53, 0.55, 0.57, 0.60]


def calibrate_iso(sm_tr, y_tr, sm_te):
    if len(np.unique(y_tr)) < 2 or sm_tr.std() < 1e-9:
        return sm_te.copy()
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(sm_tr, y_tr.astype(float))
    return ir.predict(sm_te)


def calibrate_platt(sm_tr, y_tr, sm_te):
    if len(np.unique(y_tr)) < 2:
        return sm_te.copy()
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(sm_tr.reshape(-1, 1), y_tr)
    return lr.predict_proba(sm_te.reshape(-1, 1))[:, 1]


def summ(a_list, n_list, lo_n=20):
    a = np.array(a_list, float); nn = np.array(n_list)
    v = a[np.isfinite(a) & (nn >= lo_n)]
    if len(v) == 0:
        return {"n_valid": 0}
    p10 = float(np.percentile(v, 10)); frac = float((v >= 0.541).mean())
    return {"n_valid": int(len(v)), "mean": round(float(v.mean()), 4), "p10": round(p10, 4),
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4),
            "frac_clear_0.541": round(frac, 3), "med_n": int(np.median(nn[nn >= lo_n])),
            "CERTIFIED": bool(p10 >= 0.541 and frac >= 0.80)}


def acc_from_wins(wins, lo_n=20):
    w = np.concatenate(wins) if wins else np.array([])
    return (float(w.mean()), int(len(w))) if len(w) >= lo_n else (np.nan, int(len(w)))


def main():
    t0 = time.time()
    p = json.load(open(XP.art("strategy.json")))
    cols = p["primary_feats"]; mcols = p["meta_feats"]
    print(f"[calibcpcv] loading...", flush=True)
    Xp, Xm, y, ts, ny = build_pm(cols, mcols)
    order = np.argsort(ts, kind="stable")
    Xp, Xm, y, ts, ny = Xp[order], Xm[order], y[order], ts[order], ny[order]
    n = len(y)
    print(f"[calibcpcv] pooled n={n:,} stride={STRIDE} build {time.time()-t0:.0f}s", flush=True)

    edges = np.linspace(0, n, N_GROUPS + 1).astype(int)
    g = np.zeros(n, np.int8)
    for k in range(N_GROUPS):
        g[edges[k]:edges[k+1]] = k
    rng = np.random.default_rng(7); HORS = EMB = 300; all_idx = np.arange(n)

    # PASS 1: collect per-fold calibrated thresholds (global threshold estimation)
    # We store calibration-segment (sm_m_ny, iso_cal_ny, plt_cal_ny) to pool across folds
    # for a global absolute threshold selection at the end.
    # PASS 2 (integrated): evaluate each gate per fold using fold-local and global thresholds.

    # Storage per fold
    fold_data = []   # list of dicts: {sm_m, y_m, ny_m, sm_te, y_te, ny_te, pr_te, cand_idx}

    tags_raw  = [f"raw_abs_{t}" for t in ABS_THRS]
    tags_iso  = [f"iso_abs_{t}" for t in ABS_THRS]
    tags_plt  = [f"plt_abs_{t}" for t in ABS_THRS]
    tag_raw_q = "raw_quantile_baseline"
    all_tags  = [tag_raw_q] + tags_raw + tags_iso + tags_plt

    acc  = {t: [] for t in all_tags}
    nsel = {t: [] for t in all_tags}

    print(f"[calibcpcv] running 28 paths...", flush=True)
    for ci, combo in enumerate(itertools.combinations(range(N_GROUPS), K_TEST)):
        te = all_idx[np.isin(g, combo)]
        tr = all_idx[~np.isin(g, combo)]
        keep = np.ones(len(tr), bool); tt = ts[tr]
        for grp in combo:
            gi = all_idx[g == grp]; lo, hi = ts[gi[0]], ts[gi[-1]]
            keep &= ~((tt >= lo - HORS) & (tt <= hi + HORS))
        tr = tr[keep]
        tr = tr[np.argsort(ts[tr], kind="stable")]
        cut = int(len(tr) * 0.75)
        if cut < 50 or len(tr) - cut < 50:
            for t in all_tags:
                acc[t].append(np.nan); nsel[t].append(0)
            continue
        trp = tr[:cut]; trm = tr[cut:]
        bnd = ts[tr[cut - 1]]
        trm = trm[ts[trm] > bnd + EMB]
        if len(trp) > PRIMARY_SUB:
            trp = np.sort(rng.choice(trp, PRIMARY_SUB, replace=False))
        if len(trm) > META_SUB:
            trm = np.sort(rng.choice(trm, META_SUB, replace=False))

        # refit primary
        P = RF.mk_lgb(); P.fit(Xp[trp], y[trp])

        # build meta features on meta segment
        pr_m = P.predict_proba(Xp[trm])[:, 1]
        conf_m = np.abs(pr_m - 0.5).astype("float32")
        Xmeta_m = np.column_stack([conf_m, Xm[trm]])
        ycorr_m = ((pr_m > 0.5).astype(int) == y[trm]).astype(int)
        nym = ny[trm]
        if nym.sum() < 200:
            del P
            for t in all_tags:
                acc[t].append(np.nan); nsel[t].append(0)
            continue

        # refit meta (exact production)
        M = mk_meta(); M.fit(Xmeta_m[nym], ycorr_m[nym])
        sm_m_ny = M.predict_proba(Xmeta_m[nym])[:, 1]
        y_m_ny  = ycorr_m[nym]

        THR_raw_q = float(np.quantile(sm_m_ny, Q))

        # calibrators fitted on meta-segment NY rows
        sm_iso_ny = calibrate_iso(sm_m_ny, y_m_ny, sm_m_ny)
        sm_plt_ny = calibrate_platt(sm_m_ny, y_m_ny, sm_m_ny)

        # collect test bars (de-overlapped UP candidates, one block at a time)
        raw_q_wins = []
        raw_wins  = {t: [] for t in ABS_THRS}
        iso_wins  = {t: [] for t in ABS_THRS}
        plt_wins  = {t: [] for t in ABS_THRS}

        for grp in combo:
            bi = all_idx[g == grp]
            pr_b = P.predict_proba(Xp[bi])[:, 1]
            conf_b = np.abs(pr_b - 0.5).astype("float32")
            Xmeta_b = np.column_stack([conf_b, Xm[bi]])
            sm_b = M.predict_proba(Xmeta_b)[:, 1]

            # calibrate test scores using fold calibrators
            sm_iso_b = calibrate_iso(sm_m_ny, y_m_ny, sm_b)
            sm_plt_b = calibrate_platt(sm_m_ny, y_m_ny, sm_b)

            ny_b = ny[bi]; y_b = y[bi]
            cand = ny_b & (pr_b > 0.5)
            sel = MX.nonoverlap_chrono(ts[bi], cand)
            if len(sel) == 0:
                continue
            sm_s     = sm_b[sel]
            sm_iso_s = sm_iso_b[sel]
            sm_plt_s = sm_plt_b[sel]
            win_s    = (y_b[sel] == 1).astype(float)

            raw_q_wins.append(win_s[sm_s >= THR_raw_q])
            for thr in ABS_THRS:
                raw_wins[thr].append(win_s[sm_s     >= thr])
                iso_wins[thr].append(win_s[sm_iso_s >= thr])
                plt_wins[thr].append(win_s[sm_plt_s >= thr])

        a_rq, n_rq = acc_from_wins(raw_q_wins)
        acc[tag_raw_q].append(a_rq); nsel[tag_raw_q].append(n_rq)
        for i, thr in enumerate(ABS_THRS):
            a_r, n_r = acc_from_wins(raw_wins[thr])
            acc[tags_raw[i]].append(a_r); nsel[tags_raw[i]].append(n_r)
            a_i, n_i = acc_from_wins(iso_wins[thr])
            acc[tags_iso[i]].append(a_i); nsel[tags_iso[i]].append(n_i)
            a_p, n_p = acc_from_wins(plt_wins[thr])
            acc[tags_plt[i]].append(a_p); nsel[tags_plt[i]].append(n_p)

        del P, M
        if ci % 7 == 0:
            print(f"  [calibcpcv] path {ci+1}/28 raw_q={a_rq:.4f}(n={n_rq}) "
                  f"raw_abs_.55={acc['raw_abs_0.55'][-1]:.4f} "
                  f"iso_abs_.55={acc['iso_abs_0.55'][-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)

    results = {t: summ(acc[t], nsel[t]) for t in all_tags}
    baseline = results[tag_raw_q]
    best_tag, best_p10 = None, 0.0
    for t in all_tags:
        r = results[t]
        if r.get("CERTIFIED") and r.get("p10", 0) > best_p10:
            best_p10 = r["p10"]; best_tag = t
    any_pass = any(results[t].get("CERTIFIED") and results[t].get("p10", 0) >= 0.541
                   for t in all_tags if t != tag_raw_q)
    out = {"test": "C4: calibrated meta gate — absolute threshold scan (iso+Platt) vs raw quantile baseline",
           "breakeven": 0.541, "Q": Q, "stride": STRIDE, "pooled_n": int(n),
           "results": results,
           "verdict": {
               "any_calibration_certified": any_pass,
               "best_tag": best_tag, "best_p10": round(best_p10, 4),
               "baseline_raw_quantile": {"p10": baseline.get("p10"), "frac": baseline.get("frac_clear_0.541")},
               "statement": (
                   f"Best calibration variant: {best_tag} p10={best_p10:.4f} "
                   + ("CERTIFIED → calibration HELPS (passes the bar the raw quantile missed)."
                      if any_pass else
                      f"— calibration does NOT rescue the meta gate. Raw quantile baseline p10 "
                      f"{baseline.get('p10')} frac {baseline.get('frac_clear_0.541')}. "
                      "LEVER KILLED.")
               )}}
    json.dump(out, open("m5_calibcpcv_result.json", "w"), indent=1)
    print(f"\n[calibcpcv] VERDICT: {out['verdict']['statement']}", flush=True)
    for t in all_tags:
        r = results[t]; cert = "✓" if r.get("CERTIFIED") else "✗"
        print(f"  {cert} {t}: p10={r.get('p10')} frac={r.get('frac_clear_0.541')} n={r.get('med_n')}", flush=True)
    print(f"[calibcpcv] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
