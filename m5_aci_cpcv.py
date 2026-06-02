"""NESTED-REFIT CPCV of the DEPLOYED m5xp two-stage gate + the ACI policy on top (the open follow-up to the
primary-only refit cert). The existing m5_cpcv_refit / m5_refit_tightcov refit ONLY the primary GBM and gated on
*primary-confidence cover* — they never tested the book's ACTUAL deployed gate (NY & meta>=THR) nor the ACI
adaptive gate. This refits BOTH stages on each purged-combinatorial train fold, exactly mirroring production
(primary on the inner-75% by time, orthogonal meta-labeler on the purged inner-25%, THR set by the production
val-quantile on that meta segment), then evaluates THREE gates on the held-out test fold:
  (1) FIXED   : sm >= THR_fold            (the deployed book gate)
  (2) ACI w*  : online adaptive threshold targeting selective win-rate w*  (Gibbs-Candes 2021; m5_conformal.aci_gate)
Selective win-rate is measured on de-overlapped (nonoverlap_chrono) UP candidates exactly as production/m5_conformal.

CERTIFY a gate iff selective path_p10 >= 0.541 AND >=80% of valid paths clear 0.541.  ACI IMPROVES iff it is
certified AND (p10 >= fixed p10) AND (coverage med_n >= fixed) — i.e. ACI is at least as robust at >= coverage.
Settlement: book-native contiguous-300s, moved-only (_y), ties LOSE. Purge+embargo = 1 horizon (300s).

Usage: M5_STRIDE=6 python m5_aci_cpcv.py     (M5_STRIDE=40 for a fast smoke test)
"""
import os, json, time, itertools, numpy as np
import lightgbm as lgb
import m5_xpair as MX
import m5_xpair_production as XP
import m5_cpcv_refit as RF
import m5_conformal as CF

N_GROUPS, K_TEST = 8, 2
STRIDE = int(os.environ.get("M5_STRIDE", "6"))
WSTARS = (0.55, 0.56, 0.57)
PRIMARY_SUB = 100_000
META_SUB = 120_000
ALL_YEARS = [str(y) for y in range(2012, 2027)]


def build_pm(cols, mcols):
    """Pooled cross-pair matrix, ONE YEAR AT A TIME (no cross-year dep -> equals all-years build at low peak RAM).
    Returns primary X, meta-feature X (raw agree/disp/comp60/OF_ cols; conf is added per-fold from refit pr), y, ts, ny."""
    import gc
    Xp, Xm, ys, tss, nys = [], [], [], [], []
    for yr in ALL_YEARS:
        D = MX.build_xp([yr], stride=STRIDE)
        if len(D) == 0:
            continue
        D = MX.augment(D, [yr], XP.MODE)
        miss = [c for c in mcols if c not in D.columns]
        if miss:
            raise SystemExit(f"meta cols missing from build for {yr}: {miss[:6]}")
        Xp.append(D[cols].astype("float32").to_numpy())
        Xm.append(D[mcols].astype("float32").to_numpy())
        ys.append(D["_y"].astype(np.int8).values)
        tss.append(D["_ts"].values.astype("int64"))
        nys.append((D["sess_ny"].values > 0.5))
        del D; gc.collect()
    return (np.concatenate(Xp), np.concatenate(Xm), np.concatenate(ys),
            np.concatenate(tss), np.concatenate(nys))


def mk_meta():
    """Orthogonal meta-labeler, matching production params (num_leaves=15 small, heavily regularized)."""
    return lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=15,
        min_child_samples=500, subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=20,
        n_estimators=300, n_jobs=20, verbosity=-1)


def summ(accs, ns, lo_n=20):
    a = np.array(accs, float); nn = np.array(ns); v = a[np.isfinite(a) & (nn >= lo_n)]
    if len(v) == 0:
        return {"n_valid": 0}
    p10 = float(np.percentile(v, 10)); frac = float((v >= 0.541).mean())
    return {"n_valid": int(len(v)), "mean": round(float(v.mean()), 4), "p10": round(p10, 4),
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4),
            "frac_clear_0.541": round(frac, 3), "med_n": int(np.median(nn[nn >= lo_n])),
            "CERTIFIED": bool(p10 >= 0.541 and frac >= 0.80)}


def main():
    t0 = time.time()
    p = json.load(open(XP.art("strategy.json")))
    cols = p["primary_feats"]; mcols = p["meta_feats"]; Q = float(p.get("val_q", 0.95))
    Xp, Xm, y, ts, ny = build_pm(cols, mcols)
    order = np.argsort(ts, kind="stable")
    Xp, Xm, y, ts, ny = Xp[order], Xm[order], y[order], ts[order], ny[order]
    n = len(y)
    print(f"[aci-cpcv] pooled n={n:,} stride={STRIDE} q={Q} feats p={len(cols)} m={len(mcols)} build {time.time()-t0:.0f}s", flush=True)
    edges = np.linspace(0, n, N_GROUPS + 1).astype(int); g = np.zeros(n, np.int8)
    for k in range(N_GROUPS):
        g[edges[k]:edges[k+1]] = k
    rng = np.random.default_rng(7); HORS = EMB = 300; all_idx = np.arange(n)
    fixed_acc, fixed_n = [], []
    aci_acc = {w: [] for w in WSTARS}; aci_n = {w: [] for w in WSTARS}
    for ci, combo in enumerate(itertools.combinations(range(N_GROUPS), K_TEST)):
        te = all_idx[np.isin(g, combo)]; tr = all_idx[~np.isin(g, combo)]
        keep = np.ones(len(tr), bool); tt = ts[tr]
        for grp in combo:
            gi = all_idx[g == grp]; lo, hi = ts[gi[0]], ts[gi[-1]]
            keep &= ~((tt >= lo - HORS) & (tt <= hi + EMB))
        tr = tr[keep]
        # inner chronological split: primary on first 75% by time, meta on purged last 25% (mirrors prod train/VAL)
        tr = tr[np.argsort(ts[tr], kind="stable")]
        cut = int(len(tr) * 0.75)
        if cut < 50 or len(tr) - cut < 50:
            fixed_acc.append(np.nan); fixed_n.append(0)
            for w in WSTARS: aci_acc[w].append(np.nan); aci_n[w].append(0)
            continue
        trp = tr[:cut]; trm = tr[cut:]
        bnd = ts[tr[cut-1]]
        trm = trm[ts[trm] > bnd + EMB]                       # purge meta segment off primary segment
        if len(trp) > PRIMARY_SUB: trp = np.sort(rng.choice(trp, PRIMARY_SUB, replace=False))
        if len(trm) > META_SUB: trm = np.sort(rng.choice(trm, META_SUB, replace=False))
        # ---- refit primary on trp ----
        P = RF.mk_lgb(); P.fit(Xp[trp], y[trp])
        # ---- refit meta on trm (orthogonal axes; target = primary directional call correct), NY rows ----
        pr_m = P.predict_proba(Xp[trm])[:, 1]; conf_m = np.abs(pr_m - 0.5).astype("float32")
        Xmeta_m = np.column_stack([conf_m, Xm[trm]]); ycorr = ((pr_m > 0.5).astype(int) == y[trm]).astype(int)
        nym = ny[trm]
        if nym.sum() < 200:
            del P; fixed_acc.append(np.nan); fixed_n.append(0)
            for w in WSTARS: aci_acc[w].append(np.nan); aci_n[w].append(0)
            continue
        M = mk_meta(); M.fit(Xmeta_m[nym], ycorr[nym])
        sm_m = M.predict_proba(Xmeta_m)[:, 1]
        THR = float(np.quantile(sm_m[nym], Q))               # production: meta thr = VAL-segment quantile
        # ---- evaluate on the test fold, block by block (contiguous in time), de-overlapped UP candidates ----
        f_wins = []; a_wins = {w: [] for w in WSTARS}
        for grp in combo:
            bi = all_idx[g == grp]
            pr_b = P.predict_proba(Xp[bi])[:, 1]; conf_b = np.abs(pr_b - 0.5).astype("float32")
            sm_b = M.predict_proba(np.column_stack([conf_b, Xm[bi]]))[:, 1]
            ts_b = ts[bi]; y_b = y[bi]; ny_b = ny[bi]
            cand = ny_b & (pr_b > 0.5)                        # UP candidates
            sel = MX.nonoverlap_chrono(ts_b, cand)            # independent (de-overlapped) UP candidates
            if len(sel) == 0:
                continue
            sm_s = sm_b[sel]; win_s = (y_b[sel] == 1).astype(float)   # chronological stream
            f_wins.append(win_s[sm_s >= THR])                # FIXED gate
            for w in WSTARS:
                ta = CF.aci_gate(sm_s, win_s, w)             # ACI online gate
                a_wins[w].append(win_s[ta])
        fw = np.concatenate(f_wins) if f_wins else np.array([])
        fixed_acc.append(float(fw.mean()) if len(fw) >= 20 else np.nan); fixed_n.append(int(len(fw)))
        for w in WSTARS:
            aw = np.concatenate(a_wins[w]) if a_wins[w] else np.array([])
            aci_acc[w].append(float(aw.mean()) if len(aw) >= 20 else np.nan); aci_n[w].append(int(len(aw)))
        del P, M, pr_m, sm_m
        if ci % 7 == 0:
            print(f"  [aci-cpcv] path {ci+1}/28 combo{combo} THR={THR:.3f} fixed={fixed_acc[-1]} n={fixed_n[-1]} "
                  f"aci.57={aci_acc[0.57][-1]} n={aci_n[0.57][-1]} ({time.time()-t0:.0f}s)", flush=True)
    out = {"test": "(5m,UP) NESTED-REFIT CPCV: deployed meta gate + ACI policy (both stages refit per path)",
           "breakeven": 0.541, "stride": STRIDE, "q": Q, "pooled_n": int(n),
           "FIXED": summ(fixed_acc, fixed_n), "ACI": {f"wstar_{w}": summ(aci_acc[w], aci_n[w]) for w in WSTARS}}
    f = out["FIXED"]; best = None
    for w in WSTARS:
        a = out["ACI"][f"wstar_{w}"]
        if a.get("n_valid", 0) and a.get("CERTIFIED") and f.get("CERTIFIED") \
                and a["p10"] >= f["p10"] and a["med_n"] >= f["med_n"]:
            if best is None or a["med_n"] > best[1]:
                best = (w, a["med_n"], a)
    out["verdict"] = {
        "fixed_meta_gate_certified": bool(f.get("CERTIFIED")),
        "aci_certified_any": bool(any(out["ACI"][f"wstar_{w}"].get("CERTIFIED") for w in WSTARS)),
        "aci_improves_on_fixed": bool(best),
        "statement": (
            (f"DEPLOYED meta gate SURVIVES nested refit (p10 {f['p10']}, {f['frac_clear_0.541']:.0%} paths). "
             if f.get("CERTIFIED") else
             f"DEPLOYED meta gate does NOT survive nested refit (p10 {f.get('p10')}, {f.get('frac_clear_0.541')} paths). ")
            + (f"ACI w*={best[0]} ALSO certified and >= fixed at >= coverage (p10 {best[2]['p10']}, med_n {best[2]['med_n']} vs fixed {f.get('med_n')}) -> ACI improvement is CPCV-robust."
               if best else
               ("ACI is certified but does NOT strictly improve on the fixed gate under refit (no w* beats fixed p10 at >= coverage)."
                if any(out["ACI"][f"wstar_{w}"].get("CERTIFIED") for w in WSTARS) else
                "ACI is NOT certified under nested refit.")))}
    json.dump(out, open("m5_aci_cpcv_result.json", "w"), indent=1)
    print(f"[aci-cpcv] FIXED: {f}", flush=True)
    for w in WSTARS:
        print(f"[aci-cpcv] ACI w*={w}: {out['ACI'][f'wstar_{w}']}", flush=True)
    print(f"[aci-cpcv] VERDICT: {out['verdict']['statement']}", flush=True)


if __name__ == "__main__":
    main()
