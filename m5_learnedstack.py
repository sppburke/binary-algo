"""C5: LEARNED STACK WEIGHT (GBM ⊕ MLP, logistic stacker trained on VAL predictions).

EXP-2 used a fixed 50/50 blend which was null (+.001 on UP 2025). The MLP is genuinely decorrelated
(corr=0.694) but WEAKER than GBM (MLPens UP2025 .5743 < GBM .5765). A logistic stacker learns the
optimal weight α directly from VAL-set predictions, instead of fixing α=0.5. Since MLP < GBM, the
learned α should approach 1 (GBM-alone); the stacker CANNOT extract more than the convex hull unless
the MLP has complementary information in a nonlinear sense.

Protocol (forward-split, FAST — full CPCV would be 28×500s=~4hrs):
  - GBM: frozen production primary (Xp features)
  - MLP: M=5 seed-ensemble trained on train split (same as EXP-2)
  - Stacker: logistic regression on VAL-set [gbm_val_pr, mlp_val_pr] → y_val
  - Apply stacker to test24/test25/oos at the meta gate
  - Also test: logistic stacker on [gbm_pr, mlp_pr, gbm_pr*mlp_pr] (interaction term)

Pre-registered falsifier: KILL unless stacked UP 2025 > 0.5776 (EXP-2 BLEND incumbent) OR at the
primary-confidence cov0.05 cover the stacker p10 > 0.553 on a quick 3-fold check.
(Since EXP-2 50/50 blend was null, expect stacker to be null or marginally better; confirm, then KILL.)

Usage: python m5_learnedstack.py
"""
import json, time, numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import m5_xpair as MX
import m5_xpair_production as XP
from m5_deep import MLP, load
from m5_deep_ens import train_one, predict

torch.manual_seed(0); np.random.seed(0)
M = 5
SPL = XP.SPL


def boot(c, nb=3000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    return tuple(float(x) for x in np.percentile(
        [c[rng.integers(0, n, n)].mean() for _ in range(nb)], [2.5, 97.5]))


def main():
    t0 = time.time()
    p = json.load(open(XP.art("strategy.json"))); cols = p["primary_feats"]; d = len(cols)

    # Load training data
    Xtr, ytr, _, _ = load(SPL["train"], cols, 6)
    Xva, yva, _, _ = load(SPL["val"], cols)
    if len(Xtr) > 150_000:
        idx = np.random.choice(len(Xtr), 150_000, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    print(f"[stack] train={len(Xtr):,} val={len(Xva):,} M={M} {time.time()-t0:.0f}s", flush=True)

    # Train M=5 MLPs (same as EXP-2)
    models = []
    for s in range(M):
        b = train_one(Xtr, ytr, Xva, yva, d, 100 + s)
        models.append(b[2:])
        print(f"[stack] MLP seed {s} VAL AUC={b[0]:.4f} {time.time()-t0:.0f}s", flush=True)

    # Load frozen GBM + meta gate
    pp, P_gbm, Mm = XP._load()
    gcols = pp["primary_feats"]; mcols = pp["meta_feats"]; THR = pp["meta_thr"]

    # Get VAL predictions (stacker training set)
    D_va = MX.build_xp(SPL["val"]); D_va = MX.augment(D_va, SPL["val"], XP.MODE)
    gbm_va  = P_gbm.predict(D_va[gcols].astype("float32"))
    mlp_va  = np.mean([predict(st, mu, sd, D_va[cols].astype("float32").to_numpy(), d)
                       for (st, mu, sd) in models], axis=0)
    y_va    = D_va["_y"].astype(int).values
    sm_va   = Mm.predict(XP._Xmeta(D_va, gbm_va, mcols))
    ny_va   = D_va["sess_ny"].values > 0.5
    gate_va = ny_va & (sm_va >= THR)    # meta gate on VAL = stacker training set

    # Fit stackers on gated VAL bars only (UP bars only for the UP-prediction task)
    Xstk_va = np.column_stack([gbm_va[gate_va], mlp_va[gate_va]])
    Xstk_va_int = np.column_stack([Xstk_va, (Xstk_va[:, 0] * Xstk_va[:, 1]).reshape(-1, 1)])
    y_stk_va = y_va[gate_va]

    stk_lin = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
    stk_lin.fit(Xstk_va, y_stk_va)
    stk_int = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
    stk_int.fit(Xstk_va_int, y_stk_va)
    print(f"[stack] stacker coefs (lin): {stk_lin.coef_} intercept {stk_lin.intercept_}", flush=True)
    del D_va

    # Evaluate on test splits at the meta gate
    res = {}
    for w, label in (("test24", "2024"), ("test25", "2025"), ("oos", "2026")):
        D = MX.build_xp(SPL[w]); D = MX.augment(D, SPL[w], XP.MODE)
        Xw  = D[cols].astype("float32").to_numpy()
        gbm = P_gbm.predict(D[gcols].astype("float32"))
        mlp = np.mean([predict(st, mu, sd, Xw, d) for (st, mu, sd) in models], axis=0)
        sm  = Mm.predict(XP._Xmeta(D, gbm, mcols))
        ts  = D["_ts"].values.astype("int64")
        ny  = D["sess_ny"].values > 0.5
        y   = D["_y"].astype(int).values
        sel = MX.nonoverlap_chrono(ts, ny & (sm >= THR))

        gbm_s = gbm[sel]; mlp_s = mlp[sel]; y_s = y[sel]
        blend_50 = 0.5 * gbm_s + 0.5 * mlp_s

        Xs_lin = np.column_stack([gbm_s, mlp_s])
        Xs_int = np.column_stack([gbm_s, mlp_s, gbm_s * mlp_s])
        stk_lin_pr = stk_lin.predict_proba(Xs_lin)[:, 1]
        stk_int_pr = stk_int.predict_proba(Xs_int)[:, 1]

        for name, pr in (("GBM", gbm_s), ("MLPens", mlp_s), ("BLEND50", blend_50),
                         ("STK_lin", stk_lin_pr), ("STK_int", stk_int_pr)):
            up = pr > 0.5
            if up.sum() < 5:
                res[f"{label}_{name}_UP"] = {"n": int(up.sum()), "acc": None}
                continue
            corr = (y_s[up] == 1).astype(float)
            lo, hi = boot(corr)
            res[f"{label}_{name}_UP"] = {"n": int(up.sum()), "acc": round(float(corr.mean()), 4),
                                         "ci": [round(lo, 4), round(hi, 4)]}
        del D

    def up(label, src):
        return res.get(f"{label}_{src}_UP", {}).get("acc")

    stk_lin_25 = up("2025", "STK_lin"); stk_int_25 = up("2025", "STK_int")
    blend_25   = up("2025", "BLEND50")
    gbm_25     = up("2025", "GBM")
    improves   = (stk_lin_25 or 0) > (blend_25 or 0) and (stk_lin_25 or 0) > (gbm_25 or 0)

    out = {"exp": "C5 learned stack weight (GBM ⊕ MLP, logistic stacker on VAL)", "M_seeds": M,
           "stacker_lin_coefs": stk_lin.coef_.tolist(), "stacker_lin_intercept": stk_lin.intercept_.tolist(),
           "per_year": res,
           "verdict": {
               "stk_lin_improves_over_blend50_2025": bool(improves),
               "statement": (
                   f"STK_lin 2025={stk_lin_25} STK_int 2025={stk_int_25} vs BLEND50 {blend_25} GBM {gbm_25} — "
                   + ("IMPROVES: learned weight beats 50/50 and GBM-alone on binding year → escalate to CPCV."
                      if improves else
                      "NULL: logistic stacker does not beat 50/50 blend on 2025 → KILLED (mechanically expected: MLP < GBM so α→1=GBM-alone).")
               )}}
    json.dump(out, open("m5_learnedstack_result.json", "w"), indent=1)
    for k, v in res.items():
        if v.get("acc") is not None:
            print(f"  {k:22} n={v['n']} acc={v['acc']} CI={v['ci']}", flush=True)
    print(f"[stack] VERDICT: {out['verdict']['statement']}", flush=True)
    print(f"[stack] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
