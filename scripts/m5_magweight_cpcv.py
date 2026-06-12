"""B1: NESTED-REFIT CPCV of the POW=0.5 |return|-weighted (magweight) primary — UP AND DOWN — at the certified
PRIMARY-CONFIDENCE cover (per the A1/m5_aci_cpcv finding: the meta gate underperforms confidence selection, so the
robust operating point is the tight top-cover by |pr-0.5|). Tests whether the forward POW=0.5 result (UP 2025 .578,
DOWN-rescue 2024/25 .605/.559 with NO 2026 collapse — m5_magweight_result.json) SURVIVES per-fold refit or is a
single-split artifact like the ACI 2025 .584 was. Refits the magweighted primary (sample_weight=(|fwd|/med)^POW,
clipped) on each of 28 purged-combinatorial folds; evaluates UP-sel and DOWN-sel at covs {0.05,0.10,0.15}.

CERTIFY a (side,cov) iff selective path_p10 >= 0.541 AND >=80% of valid paths clear 0.541. Compare to incumbent
m5xp (m5_refit_tightcov_result.json): UP cov0.05 p10 .553/96% (CERT) ; DOWN cov0.05 p10 .542/93% (marginal).
PRE-REGISTERED FALSIFIER: KILL the UP claim unless POW=0.5 UP cov0.05 p10 >= .553 & >=80% (i.e. at least matches
the unweighted incumbent). KILL the DOWN-RESCUE unless POW=0.5 DOWN cov0.05 p10 >= .541 & >=80% AND p10 > incumbent
DOWN .542 (a genuine improvement, not the same marginal). Settlement: book-native, moved-only (_y), ties LOSE.

Usage: M5_STRIDE=6 M5_POW=0.5 python m5_magweight_cpcv.py
"""
import os, json, time, itertools, numpy as np
from sklearn.metrics import roc_auc_score
import m5_xpair as MX
import m5_xpair_production as XP
import m5_cpcv_refit as RF

N_GROUPS, K_TEST = 8, 2
STRIDE = int(os.environ.get("M5_STRIDE", "6"))
POW = float(os.environ.get("M5_POW", "0.5"))
COVS = [0.05, 0.10, 0.15]
SUBSAMPLE = 100_000
ALL_YEARS = [str(y) for y in range(2012, 2027)]


def magweight(fwd, pow_=POW):
    a = np.abs(fwd).astype(float); med = np.median(a[a > 0]) or 1e-9
    return np.clip((a / med) ** pow_, 0.1, 10.0)


def build_fwd(cols):
    """Pooled cross-pair matrix one year at a time; also returns _fwd (signed fwd return) for the magnitude weights."""
    import gc
    Xs, ys, tss, nys, fws = [], [], [], [], []
    for yr in ALL_YEARS:
        D = MX.build_xp([yr], stride=STRIDE)
        if len(D) == 0:
            continue
        D = MX.augment(D, [yr], XP.MODE)
        Xs.append(D[cols].astype("float32").to_numpy())
        ys.append(D["_y"].astype(np.int8).values)
        tss.append(D["_ts"].values.astype("int64"))
        nys.append((D["sess_ny"].values > 0.5))
        fws.append(D["_fwd"].astype("float32").values)
        del D; gc.collect()
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(tss),
            np.concatenate(nys), np.concatenate(fws))


def summ(a_list, n_list):
    a = np.array(a_list, float); nn = np.array(n_list); v = a[np.isfinite(a) & (nn >= 20)]
    if len(v) == 0:
        return {"n_valid": 0}
    p10 = float(np.percentile(v, 10)); frac = float((v >= 0.541).mean())
    return {"n_valid": int(len(v)), "mean": round(float(v.mean()), 4), "p10": round(p10, 4),
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4),
            "frac_clear_0.541": round(frac, 3), "med_n": int(np.median(nn[nn >= 20])),
            "CERTIFIED": bool(p10 >= 0.541 and frac >= 0.80)}


def main():
    t0 = time.time()
    p = json.load(open(XP.art("strategy.json"))); cols = p["primary_feats"]
    X, y, ts, ny, fwd = build_fwd(cols)
    order = np.argsort(ts, kind="stable"); X, y, ts, ny, fwd = X[order], y[order], ts[order], ny[order], fwd[order]
    n = len(y)
    print(f"[magw-cpcv] pooled n={n:,} stride={STRIDE} POW={POW} feats={len(cols)} build {time.time()-t0:.0f}s", flush=True)
    edges = np.linspace(0, n, N_GROUPS + 1).astype(int); g = np.zeros(n, np.int8)
    for k in range(N_GROUPS):
        g[edges[k]:edges[k+1]] = k
    rng = np.random.default_rng(7); HORS = EMB = 300; all_idx = np.arange(n)
    SIDES = [("UP", 1), ("DOWN", 0)]
    acc = {(s, c): [] for s, _ in SIDES for c in COVS}
    nsel = {(s, c): [] for s, _ in SIDES for c in COVS}
    aucs = []
    for ci, combo in enumerate(itertools.combinations(range(N_GROUPS), K_TEST)):
        te = all_idx[np.isin(g, combo)]; tr = all_idx[~np.isin(g, combo)]
        keep = np.ones(len(tr), bool); tt = ts[tr]
        for grp in combo:
            gi = all_idx[g == grp]; lo, hi = ts[gi[0]], ts[gi[-1]]
            keep &= ~((tt >= lo - HORS) & (tt <= hi + EMB))
        tr = tr[keep]
        if len(tr) > SUBSAMPLE:
            tr = np.sort(rng.choice(tr, SUBSAMPLE, replace=False))
        m = RF.mk_lgb(); m.fit(X[tr], y[tr], sample_weight=magweight(fwd[tr]))
        pr = m.predict_proba(X[te])[:, 1]; yte = y[te]; aucs.append(roc_auc_score(yte, pr))
        conf = np.abs(pr - 0.5)
        for sname, sv in SIDES:
            gate = ny[te] & ((pr > 0.5) if sv == 1 else (pr < 0.5))
            for c in COVS:
                if gate.sum() >= 40:
                    cthr = np.quantile(conf[gate], 1 - c); sel = gate & (conf >= cthr)
                    if sel.sum() >= 20:
                        acc[(sname, c)].append(float((yte[sel] == sv).mean()))
                        nsel[(sname, c)].append(int(sel.sum())); continue
                acc[(sname, c)].append(np.nan); nsel[(sname, c)].append(0)
        del m, pr
        if ci % 7 == 0:
            print(f"  [magw-cpcv] path {ci+1}/28 AUC={aucs[-1]:.4f} UP@.05={acc[('UP',0.05)][-1]} "
                  f"DOWN@.05={acc[('DOWN',0.05)][-1]} ({time.time()-t0:.0f}s)", flush=True)
    out = {"test": f"(5m) POW={POW} magweight NESTED-REFIT CPCV — UP+DOWN at primary-confidence cover",
           "breakeven": 0.541, "stride": STRIDE, "POW": POW, "pooled_n": int(n),
           "auc_mean": round(float(np.mean(aucs)), 4), "per_side_cov": {}}
    for sname, _ in SIDES:
        for c in COVS:
            out["per_side_cov"][f"{sname}_cov{c}"] = summ(acc[(sname, c)], nsel[(sname, c)])
    up_cert = any(out["per_side_cov"][f"UP_cov{c}"].get("CERTIFIED") for c in COVS)
    down_cert = any(out["per_side_cov"][f"DOWN_cov{c}"].get("CERTIFIED") for c in COVS)
    d05 = out["per_side_cov"]["DOWN_cov0.05"]
    down_beats_incumbent = bool(d05.get("CERTIFIED") and d05.get("p10", 0) > 0.542)
    out["verdict"] = {
        "UP_certified": up_cert, "DOWN_certified": down_cert, "DOWN_beats_incumbent_marginal": down_beats_incumbent,
        "statement": (f"POW={POW} magweight under nested refit @ confidence cover: "
                      f"UP {'CERTIFIED' if up_cert else 'NOT certified'} (vs incumbent UP cov0.05 p10 .553/96%); "
                      f"DOWN {'CERTIFIED' if down_cert else 'NOT certified'}"
                      f"{' and BEATS the marginal incumbent DOWN .542 -> genuine DOWN rescue' if down_beats_incumbent else ' (incumbent DOWN cov0.05 p10 .542 marginal)'}.")}
    json.dump(out, open("m5_magweight_cpcv_result.json", "w"), indent=1)
    for k, v in out["per_side_cov"].items():
        print(f"  {k}: {v}", flush=True)
    print(f"[magw-cpcv] VERDICT: {out['verdict']['statement']}", flush=True)


if __name__ == "__main__":
    main()
