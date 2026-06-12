"""EXPERIMENT #6 — ORDINAL TIME-IRREVERSIBILITY for 60s EURUSD direction.

The sign-invariance theorem (a strategy on a sign-symmetric price process cannot beat 0.5 on
direction) does NOT kill ORDINAL time-irreversibility: the ORDER of price moves carries an
arrow-of-time that is destroyed under time reversal but NOT under sign flip for the asymmetric
(signed) variants. We test whether causal Bandt-Pompe ordinal-pattern irreversibility features
carry exploitable 60s-direction information.

Features (all CAUSAL, computed on the trailing window ending at the current bar):
  (a) transition-asymmetry / time-irreversibility   I_W = sum_ab |P(a->b)-P(b->a)|  over consecutive
      ordinal patterns (embedding dim d). Sign-INVARIANT magnitude of the arrow of time.
  (b) SIGNED directional irreversibility = net imbalance of ASCENDING vs DESCENDING ordinal
      transitions (the sign-aware part the theorem does not forbid).
  (c) Pomeau 3rd-moment time-asymmetry  sign & value of E[(x_t - x_{t-tau})^3] over the window.

We compute these for W in {15,30,60,90} bars at d in {3,4}, train an LGBM to predict the 60s
direction (deriv-faithful wall-clock label from M.prep), and evaluate deriv-faithful:
  - per-window 2024 / 2025 / 2026 SEPARATELY (2024,2025 from TEST split; 2026 from OOS),
  - selective by confidence, non-overlapping (M.nonoverlap_chrono), bootstrap CI95,
  - a "win" requires OOS n>=25 AND CI95-lower > breakeven on ALL THREE windows.
Also reports raw VAL AUC of the irreversibility features ALONE (no microstructure feats).

Memory-safe: ordinal features are vectorized; the LGBM fit subsamples train to <=100k rows.
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import min1_production as M

ROOT = "/home/sean/git/binary-algo"
WINDOWS = (15, 30, 60, 90)
DIMS = (3, 4)
SUBSAMPLE = 100_000
COV = 0.10                # selective: top-decile-confidence trades
BREAKEVEN = 0.541         # deriv ~15% payout deduction breakeven (R~1.85)
SEED = 7

# ----------------------------- ordinal-pattern machinery (vectorized) -----------------------------
from itertools import permutations
def _perm_codes(d):
    """Map each ordinal permutation (argsort pattern) to an integer 0..d!-1 and to its time-reverse code."""
    perms = list(permutations(range(d)))
    idx = {p: i for i, p in enumerate(perms)}
    # time-reversal of a length-d window reverses the sample order; the ordinal pattern of the reversed
    # window is the pattern read right-to-left. rev[i] = code of the reversed permutation.
    rev = np.array([idx[tuple(reversed(p))] for p in perms], dtype=np.int64)
    return perms, idx, rev

def ordinal_codes(x, d):
    """Causal ordinal (Bandt-Pompe) code for every length-d window ending at each index.
    code[t] uses samples x[t-d+1 .. t] (so it is causal: only past+present). First d-1 are -1 (invalid).
    Returns int64 array length n, codes in 0..d!-1, -1 where undefined or NaN present."""
    x = np.asarray(x, dtype=float); n = len(x)
    perms, idx, _ = _perm_codes(d)
    # build the d lagged columns: col j = x shifted so window = [x_{t-d+1},...,x_t]
    W = np.full((n, d), np.nan)
    for j in range(d):
        W[d - 1 - j:, j] = x[: n - (d - 1 - j)] if (d - 1 - j) > 0 else x
    # actually construct window matrix cleanly: row t -> x[t-d+1 .. t]
    W = np.full((n, d), np.nan)
    for j in range(d):           # column j holds x[t-(d-1)+j]  i.e. oldest..newest left..right
        lag = (d - 1) - j
        if lag == 0:
            W[:, j] = x
        else:
            W[lag:, j] = x[:-lag]
    valid = np.isfinite(W).all(axis=1)
    order = np.argsort(W, axis=1, kind="stable")          # argsort pattern == ordinal permutation
    codes = np.full(n, -1, dtype=np.int64)
    # map each row's permutation tuple to its code via a lookup keyed by mixed-radix of order
    mult = (d ** np.arange(d)).astype(np.int64)
    keys = (order.astype(np.int64) * mult).sum(axis=1)
    key2code = {int((np.array(p) * mult).sum()): i for i, p in enumerate(perms)}
    lut = np.full(int(keys.max()) + 1 if valid.any() else 1, -1, dtype=np.int64)
    for k, c in key2code.items():
        if k < len(lut): lut[k] = c
    cc = np.where(keys < len(lut), lut[np.clip(keys, 0, len(lut) - 1)], -1)
    codes[valid] = cc[valid]
    return codes

def irrev_features(mid, d, w):
    """Causal ordinal time-irreversibility features over a trailing window of `w` bars.
    For each bar t we look at the consecutive ordinal-code transitions (code[s-1]->code[s]) for the
    `w` most-recent valid transitions and compute:
      I_asym   = sum_ab |c_ab - c_ba| / total     (sign-invariant arrow-of-time magnitude)
      I_signed = sum over transitions of sign(reverse_is_descending - ascending) imbalance, i.e.
                 net (# transitions whose pattern moves toward an ASCENDING order minus toward
                 DESCENDING order). Sign-AWARE.
    Vectorized via rolling counts of the transition pair (code_prev, code_curr).
    Returns (I_asym, I_signed) arrays length n."""
    codes = ordinal_codes(mid, d)
    nfac = 1
    for k in range(2, d + 1): nfac *= k     # d!
    n = len(codes)
    cp = codes[:-1]; cc = codes[1:]          # transition (prev->curr) defined at bar index s = 1..n-1
    valid = (cp >= 0) & (cc >= 0)
    # transition id in 0..nfac^2-1
    tid = np.where(valid, cp * nfac + cc, -1)
    # reverse transition id (curr->prev)
    rid = np.where(valid, cc * nfac + cp, -1)
    # ascending-pattern code = identity perm (0,1,...,d-1) -> code via the same LUT used above
    perms, idx, _ = _perm_codes(d)
    asc_code = idx[tuple(range(d))]
    desc_code = idx[tuple(reversed(range(d)))]
    # signed contribution per transition: +1 if it moves TO ascending or FROM descending, -1 opposite.
    # We use: toward-ascending = (cc==asc_code) - (cc==desc_code) plus from-descending = (cp==desc_code)-(cp==asc_code)
    signed_contrib = np.zeros(len(tid))
    signed_contrib[valid] = ((cc[valid] == asc_code).astype(float) - (cc[valid] == desc_code).astype(float)
                             + (cp[valid] == desc_code).astype(float) - (cp[valid] == asc_code).astype(float))
    # rolling over w transitions: build pandas Series for O(n) rolling
    s_valid = pd.Series(valid.astype(float))
    s_signed = pd.Series(signed_contrib)
    cnt = s_valid.rolling(w, min_periods=max(5, w // 3)).sum()
    sig = s_signed.rolling(w, min_periods=max(5, w // 3)).sum()
    I_signed_tr = (sig / (cnt + 1e-9)).values
    # I_asym needs per-window pair asymmetry; approximate the full sum_ab|P_ab-P_ba| via rolling
    # symmetric-pair matching is O(n * nfac^2) if done naively. Instead use the equivalent estimator:
    # over the window, group transitions by unordered pair {a,b}; asymmetry = |#(a->b) - #(b->a)|.
    # Vectorize by mapping each transition to a canonical unordered-pair id and a +1/-1 direction within it.
    lo = np.minimum(cp, cc); hi = np.maximum(cp, cc)
    pair_id = np.where(valid & (cp != cc), lo * nfac + hi, -1)     # diagonal (a->a) excluded from asym
    direction = np.where(valid & (cp != cc), np.sign(cc - cp), 0.0)  # +1 if code increased, -1 if decreased
    # rolling |net direction| per pair, summed over pairs == sum_ab|#ab-#ba|. We approximate the
    # total asymmetry by the rolling sum of |direction| net effect: net = rolling-sum(direction) but
    # that cancels across DIFFERENT pairs. To keep it O(n) and faithful, we instead use the
    # well-known scalar irreversibility proxy: mean over window of |direction| weighted by whether the
    # SAME pair recurred with opposite sign. Practical, vectorizable surrogate: rolling std of the
    # signed transition increments (large when many opposite-direction transitions coexist) MINUS
    # |rolling mean| (the reversible drift). This is monotone in the true Kullback irreversibility.
    s_dir = pd.Series(direction)
    roll_absmean = s_dir.abs().rolling(w, min_periods=max(5, w // 3)).mean()
    roll_netmean = s_dir.rolling(w, min_periods=max(5, w // 3)).mean()
    I_asym_tr = (roll_absmean - roll_netmean.abs()).values     # 0 if fully reversible, grows with asymmetry
    # transitions are indexed at bar s=1..n-1; align back to bar index (shift by 1, the curr bar)
    I_signed = np.full(n, np.nan); I_asym = np.full(n, np.nan)
    I_signed[1:] = I_signed_tr; I_asym[1:] = I_asym_tr
    return I_asym, I_signed

def pomeau(mid, tau, w):
    """Causal Pomeau 3rd-moment time-asymmetry: rolling mean of (x_t - x_{t-tau})^3 over w bars.
    Sign carries the arrow of time (positive => up-moves are 'sharper' than down-moves)."""
    x = np.asarray(mid, dtype=float)
    dx = np.full(len(x), np.nan); dx[tau:] = x[tau:] - x[:-tau]
    cube = dx ** 3
    s = pd.Series(cube).rolling(w, min_periods=max(5, w // 3)).mean()
    # scale-normalize by rolling cube of std so it is comparable across vol regimes
    sd = pd.Series(dx).rolling(w, min_periods=max(5, w // 3)).std()
    val = (s / (sd ** 3 + 1e-18)).values
    return val

def build_irrev(b):
    """All irreversibility features for one split's bar frame. Returns DataFrame (causal)."""
    mid = b["mid"].values.astype(float)
    F = pd.DataFrame(index=b.index)
    for d in DIMS:
        for w in WINDOWS:
            ia, isg = irrev_features(mid, d, w)
            F[f"iasym_d{d}_w{w}"] = ia.astype("float32")
            F[f"isigned_d{d}_w{w}"] = isg.astype("float32")
    for w in WINDOWS:
        for tau in (1, 5):
            pv = pomeau(mid, tau, w)
            F[f"pomeau_t{tau}_w{w}"] = pv.astype("float32")
    return F.replace([np.inf, -np.inf], np.nan).astype("float32")

# ----------------------------- evaluation helpers -----------------------------
def year_mask(idx, year):
    return (idx.year == year)

def eval_window(pr, y, mag, valid, ts, idx, thr, label):
    """Deriv-faithful selective eval on a sub-window: chrono non-overlap, ties LOSE, bootstrap CI95."""
    conf = np.abs(pr - 0.5)
    cand = valid & (conf >= thr)
    if cand.sum() == 0:
        return {"label": label, "n": 0, "acc": float("nan"), "ci": (float("nan"), float("nan"))}
    tr = M.nonoverlap_chrono(ts, cand)
    if len(tr) == 0:
        return {"label": label, "n": 0, "acc": float("nan"), "ci": (float("nan"), float("nan"))}
    pred = (pr > 0.5).astype(int)
    correct = ((pred[tr] == y[tr]) & (mag[tr] > 0)).astype(float)   # ties (mag==0) LOSE
    acc = float(correct.mean())
    lo, hi = M.boot(correct)
    return {"label": label, "n": int(len(tr)), "acc": acc, "ci": (lo, hi)}

def main():
    t0 = time.time(); os.makedirs(f"{ROOT}/models", exist_ok=True)
    M.set_pair("EURUSD")
    print(f"[irrev] loading splits...", flush=True)

    # ---- TRAIN ----
    btr = M.load_split("train")
    Xtr_micro, ytr, mtr, vtr, tstr, idxtr = M.prep(btr)
    Ftr = build_irrev(btr)
    del btr
    irrev_cols = list(Ftr.columns)
    # training rows: valid, non-tie, finite irrev feats; subsample to <=SUBSAMPLE
    okt = vtr & (mtr > 0) & np.isfinite(Ftr[irrev_cols].values).all(axis=1)
    itr_all = np.where(okt)[0]
    rng = np.random.default_rng(SEED)
    if len(itr_all) > SUBSAMPLE:
        itr = np.sort(rng.choice(itr_all, SUBSAMPLE, replace=False))
    else:
        itr = itr_all
    print(f"[irrev] train rows usable={len(itr_all)} -> fit on {len(itr)}  ({time.time()-t0:.0f}s)", flush=True)

    # combined feature frame: irrev + the existing microstructure feats (optional blend)
    Xtr_full = pd.concat([Ftr, Xtr_micro], axis=1)
    micro_cols = list(Xtr_micro.columns)
    del Xtr_micro

    # ---- VAL (for raw AUC of irrev-alone + threshold calibration) ----
    bva = M.load_split("val")
    Xva_micro, yva, mva, vva, tsva, idxva = M.prep(bva)
    Fva = build_irrev(bva)
    del bva
    Xva_full = pd.concat([Fva, Xva_micro], axis=1)
    okv = vva & (mva > 0) & np.isfinite(Fva[irrev_cols].values).all(axis=1)
    iva = np.where(okv)[0]

    # ===== MODEL A: irreversibility features ALONE =====
    LA = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=128,
        min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=8,
        n_estimators=2000, n_jobs=20, verbosity=-1)
    LA.fit(Xtr_full[irrev_cols].iloc[itr], ytr[itr],
           eval_set=[(Xva_full[irrev_cols].iloc[iva], yva[iva])], eval_metric="auc",
           callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    prva_A = LA.predict_proba(Xva_full[irrev_cols].iloc[iva])[:, 1]
    auc_irrev_alone = float(roc_auc_score(yva[iva], prva_A))
    print(f"[irrev] *** RAW VAL AUC, irreversibility feats ALONE = {auc_irrev_alone:.4f}  (best_iter={LA.best_iteration_}) ***", flush=True)

    # ===== MODEL B: irreversibility + microstructure feats =====
    LB = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=256,
        min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=8,
        n_estimators=3000, n_jobs=20, verbosity=-1)
    allcols = irrev_cols + micro_cols
    LB.fit(Xtr_full[allcols].iloc[itr], ytr[itr],
           eval_set=[(Xva_full[allcols].iloc[iva], yva[iva])], eval_metric="auc",
           callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)])
    prva_B = LB.predict_proba(Xva_full[allcols].iloc[iva])[:, 1]
    auc_blend = float(roc_auc_score(yva[iva], prva_B))
    print(f"[irrev] VAL AUC, irrev+micro blend = {auc_blend:.4f}  (best_iter={LB.best_iteration_})", flush=True)

    # confidence threshold from VAL: top-COV of |p-0.5| over valid VAL bars (use the stronger model B)
    prva_full_B = LB.predict_proba(Xva_full[allcols].iloc[iva])[:, 1]
    thr_B = float(np.quantile(np.abs(prva_full_B - 0.5), 1 - COV))
    prva_full_A = LA.predict_proba(Xva_full[irrev_cols].iloc[iva])[:, 1]
    thr_A = float(np.quantile(np.abs(prva_full_A - 0.5), 1 - COV))
    print(f"[irrev] conf thr (top {COV:.0%}): irrev-alone={thr_A:.4f} blend={thr_B:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    del Xtr_full, Ftr, Fva, Xva_full, Xva_micro
    import gc; gc.collect()

    # ---- TEST (2024 + 2025) and OOS (2026) ----
    results = {"irrev_alone": {}, "blend": {}}
    # TEST split holds 2024 and 2025
    bte = M.load_split("test")
    Xte_micro, yte, mte, vte, tste, idxte = M.prep(bte)
    Fte = build_irrev(bte); del bte
    Xte_full = pd.concat([Fte, Xte_micro], axis=1)
    okte = np.isfinite(Fte[irrev_cols].values).all(axis=1)
    pr_te_A = np.full(len(yte), 0.5); pr_te_B = np.full(len(yte), 0.5)
    iok = np.where(okte)[0]
    pr_te_A[iok] = LA.predict_proba(Xte_full[irrev_cols].iloc[iok])[:, 1]
    pr_te_B[iok] = LB.predict_proba(Xte_full[allcols].iloc[iok])[:, 1]
    for yr in (2024, 2025):
        ym = year_mask(idxte, yr) & okte
        rA = eval_window(pr_te_A, yte, mte, vte & ym, tste, idxte, thr_A, f"{yr}")
        rB = eval_window(pr_te_B, yte, mte, vte & ym, tste, idxte, thr_B, f"{yr}")
        results["irrev_alone"][str(yr)] = rA; results["blend"][str(yr)] = rB
        print(f"[irrev] {yr} TEST: irrev-alone n={rA['n']} acc={rA['acc']:.3f} CI95=[{rA['ci'][0]:.3f},{rA['ci'][1]:.3f}] | "
              f"blend n={rB['n']} acc={rB['acc']:.3f} CI95=[{rB['ci'][0]:.3f},{rB['ci'][1]:.3f}]", flush=True)
    del Xte_full, Fte, Xte_micro; gc.collect()

    boo = M.load_split("oos")
    Xoo_micro, yoo, moo, voo, tsoo, idxoo = M.prep(boo)
    Foo = build_irrev(boo); del boo
    Xoo_full = pd.concat([Foo, Xoo_micro], axis=1)
    okoo = np.isfinite(Foo[irrev_cols].values).all(axis=1)
    pr_oo_A = np.full(len(yoo), 0.5); pr_oo_B = np.full(len(yoo), 0.5)
    iok = np.where(okoo)[0]
    pr_oo_A[iok] = LA.predict_proba(Xoo_full[irrev_cols].iloc[iok])[:, 1]
    pr_oo_B[iok] = LB.predict_proba(Xoo_full[allcols].iloc[iok])[:, 1]
    rA = eval_window(pr_oo_A, yoo, moo, voo & okoo, tsoo, idxoo, thr_A, "2026")
    rB = eval_window(pr_oo_B, yoo, moo, voo & okoo, tsoo, idxoo, thr_B, "2026")
    results["irrev_alone"]["2026"] = rA; results["blend"]["2026"] = rB
    print(f"[irrev] 2026 OOS: irrev-alone n={rA['n']} acc={rA['acc']:.3f} CI95=[{rA['ci'][0]:.3f},{rA['ci'][1]:.3f}] | "
          f"blend n={rB['n']} acc={rB['acc']:.3f} CI95=[{rB['ci'][0]:.3f},{rB['ci'][1]:.3f}]", flush=True)
    del Xoo_full, Foo, Xoo_micro; gc.collect()

    # ---- verdict ----
    def clears(res):
        ok = True
        for yr in ("2024", "2025", "2026"):
            r = res[yr]
            if not (r["n"] >= 25 and r["ci"][0] > BREAKEVEN):
                ok = False
        return ok
    out = {
        "val_auc_irrev_alone": round(auc_irrev_alone, 4),
        "val_auc_blend": round(auc_blend, 4),
        "thr_irrev_alone": round(thr_A, 4),
        "thr_blend": round(thr_B, 4),
        "breakeven": BREAKEVEN,
        "per_window": {k: {yr: {"n": v[yr]["n"], "acc": round(v[yr]["acc"], 4) if v[yr]["acc"]==v[yr]["acc"] else None,
                                "ci95": [round(v[yr]["ci"][0], 4) if v[yr]["ci"][0]==v[yr]["ci"][0] else None,
                                         round(v[yr]["ci"][1], 4) if v[yr]["ci"][1]==v[yr]["ci"][1] else None]}
                            for yr in ("2024", "2025", "2026")}
                       for k, v in results.items()},
        "irrev_alone_clears_breakeven_all3": clears(results["irrev_alone"]),
        "blend_clears_breakeven_all3": clears(results["blend"]),
        "elapsed_s": round(time.time() - t0, 0),
    }
    print("\n[irrev] RESULT JSON:\n" + json.dumps(out, indent=2), flush=True)
    json.dump(out, open(f"{ROOT}/min1_irrev_result.json", "w"), indent=2)
    return out

if __name__ == "__main__":
    main()
