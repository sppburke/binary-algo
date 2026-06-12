"""Purpose-built UP and DOWN SPECIALISTS @MX_HOR=2 (goal step (c)) — closes the pipeline gap with a RUN, not an
argument. Specialist form = orthogonal META-LABELER on the cross-pair primary (the program's Tier-1 finding is that
separately-TRAINED side models are WORSE — subset-training kills ranking — so the meta-labeler gate is the right
specialist). Trains, per side, M_side = P(primary correct | primary predicts side) from agreement/dispersion/comp/
OF + confidence (NOT the 331 raw), freezes the meta threshold by worst-VAL-half, evaluates held-out side-split.

Pre-registered falsifier: KILL a side if its meta-gated held-out acc does not clear 0.545 in >=2 of {2024,25,26}
years (n>=25). Context: the cross-pair refit-CPCV already showed UP p10 .5096 / DOWN .5124 (per-fold gate-tuned,
flat) — a meta-gate is a richer gate, so a frozen win here would be an overfit-gate upper bound, not a refit floor.
Usage: python min2_spec.py
"""
import os
os.environ["MX_HOR"] = "2"
import json, time, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import m5_xpair as MX

HOR = 2; GAP_S = HOR * 60; BE = 0.541
GATE_FEAT = "1m_bb_width"


def yr(ts):
    return (np.asarray(ts, dtype="datetime64[s]").astype("datetime64[Y]").astype(int) + 1970)


def boot(c, nb=4000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def meta_feats(F):
    return [c for c in F.columns if c.startswith("agree") or c.startswith("disp") or c == "comp60" or c.startswith("OF_")]


def _Xmeta(F, pr, mcols):
    conf = np.abs(pr - 0.5).astype("float32")
    return np.column_stack([conf] + [F[c].values.astype("float32") for c in mcols])


def main():
    t0 = time.time()
    TR = MX.build_xp(MX.SPL["train"], stride=4); VA = MX.build_xp(MX.SPL["val"])
    xpc = MX.xp_cols(TR)
    TR = MX.augment(TR, MX.SPL["train"], "xpof"); VA = MX.augment(VA, MX.SPL["val"], "xpof")
    cols = MX.feat_cols("xpof", TR, xpc)
    print(f"[spec] train={len(TR):,} val={len(VA):,} feats={len(cols)} build {time.time()-t0:.0f}s", flush=True)
    ytr = TR["_y"].astype(int).values; yva = VA["_y"].astype(int).values
    P = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=127, min_child_samples=400,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=20, n_estimators=3000,
                           n_jobs=20, verbosity=-1)
    P.fit(TR[cols].astype("float32"), ytr, eval_set=[(VA[cols].astype("float32"), yva)], eval_metric="auc",
          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    pva = P.predict_proba(VA[cols].astype("float32"))[:, 1]
    mcols = meta_feats(VA)
    Xm = _Xmeta(VA, pva, mcols); ycorr = ((pva > 0.5).astype(int) == yva).astype(int)
    tsv = VA["_ts"].values.astype("int64"); vyr = yr(tsv); bbv = VA[GATE_FEAT].values.astype("float32")
    nyv = VA["sess_ny"].values > 0.5; predv = (pva > 0.5).astype(int)
    res = {"test": "purpose-built UP/DOWN meta-labeler specialists @MX_HOR=2 (cross-pair primary)", "breakeven": BE,
           "falsifier": {"KILL_side_if": "meta-gated held-out acc < 0.545 in >=2 of 3 years (n>=25)"}, "sides": {}}
    for side, sval in (("UP", 1), ("DOWN", 0)):
        sidev = predv == sval
        if sidev.sum() < 500:
            res["sides"][side] = {"verdict": "n/a (too few VAL side rows)"}; continue
        M = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=15,
                               min_child_samples=1000, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                               reg_lambda=20, n_estimators=400, n_jobs=20, verbosity=-1)
        M.fit(Xm[sidev], ycorr[sidev])
        sm = M.predict_proba(Xm)[:, 1]
        # freeze meta thr by worst-VAL-half (min over VAL years), gate = side & NY & comp & meta>=thr
        vyears = sorted(set(vyr.tolist())); bthr = np.nanpercentile(bbv[np.isfinite(bbv)], 33)
        best = None
        for q in (0.80, 0.90, 0.95):
            thr = float(np.quantile(sm[sidev], q)); accs = []; ntot = 0
            for Y in vyears:
                m = sidev & nyv & (bbv <= bthr) & (sm >= thr) & (vyr == Y); sel = MX.nonoverlap_chrono(tsv, m, GAP_S)
                if len(sel) < 25:
                    accs = None; break
                accs.append(((pva[sel] > 0.5).astype(int) == yva[sel]).mean()); ntot += len(sel)
            if accs is None or ntot < 80:
                continue
            wh = min(accs)
            if best is None or wh > best[0]:
                best = (wh, thr, q)
        if best is None:
            res["sides"][side] = {"verdict": "KILL: no stable VAL meta gate (thin)"}; continue
        wh, THR, Q = best
        per_year = {}
        for w, yl in (("test24", 2024), ("test25", 2025), ("oos", 2026)):
            D = MX.build_xp(MX.SPL[w]); D = MX.augment(D, MX.SPL[w], "xpof")
            for c in cols:
                if c not in D.columns:
                    D[c] = np.nan
            pr = P.predict_proba(D[cols].astype("float32"))[:, 1]; y = D["_y"].astype(int).values
            ts = D["_ts"].values.astype("int64"); bb = D[GATE_FEAT].values.astype("float32")
            ny = D["sess_ny"].values > 0.5; pred = (pr > 0.5).astype(int)
            smh = M.predict_proba(_Xmeta(D, pr, mcols))[:, 1]
            m = (pred == sval) & ny & (bb <= bthr) & (smh >= THR); sel = MX.nonoverlap_chrono(ts, m, GAP_S)
            if len(sel) >= 25:
                corr = ((pr[sel] > 0.5).astype(int) == y[sel]).astype(float); lo, hi = boot(corr)
                per_year[str(yl)] = {"n": int(len(sel)), "acc": round(float(corr.mean()), 4), "ci": [round(lo, 4), round(hi, 4)]}
            else:
                per_year[str(yl)] = {"n": int(len(sel)), "acc": None}
        clears = sum(1 for Y in ("2024", "2025", "2026") if (per_year[Y].get("acc") or 0) >= 0.545)
        verdict = "SURVIVES->refit-CPCV" if clears >= 2 else "KILL"
        res["sides"][side] = {"val_worst_half": round(wh, 4), "meta_q": Q, "per_year": per_year,
                              "years_ge_0.545": clears, "verdict": verdict}
        print(f"[spec] {side}: VAL worst-half={wh:.4f} q{Q} | "
              + " ".join(f"{Y}:{per_year[Y]['acc']}(n{per_year[Y]['n']})" for Y in ("2024", "2025", "2026"))
              + f" | {verdict}", flush=True)
    json.dump(res, open("min2_spec_result.json", "w"), indent=1)
    print(f"[spec] DONE {time.time()-t0:.0f}s -> min2_spec_result.json", flush=True)


if __name__ == "__main__":
    main()
