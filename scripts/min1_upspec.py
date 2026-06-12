"""1-MIN DEDICATED SIDE-SPECIALISTS (the full version of the up-vs-down idea).

The asymmetry diagnostic (min1_updown.py) showed the symmetric book's edge lives entirely on the UP (dip-buy) side
(2025 .584 / 2026 .613) while the DOWN side is dead (~0.52). Here we train a model SPECIFICALLY for each side, instead of
filtering the symmetric model's predictions:
  - UP-specialist: trained ONLY on dip-buy setups (compression regime AND recent down-move, ret300<0) to predict whether the
    next-60s move is UP (the bounce). Trade UP on its confident calls.
  - DOWN-specialist: trained ONLY on rally-sell setups (compression AND ret300>0) to predict DOWN. (Control — expected dead.)
Selection: freeze each side's confidence threshold on VAL (2024-H1) by worst-VAL-half stability; judge on EACH of 2024/2025/2026
separately, deriv-faithful (non-overlap 60s, ties LOSE), bootstrap CI95. No leakage: specialists trained 2021-2023, frozen, judged
held-out. A side clears the goal only if n>=25 AND CI95 lower>0.65 on ALL THREE windows. Reuses the frozen compression gate params.
"""
import sys, json, time, numpy as np, pandas as pd
import lightgbm as lgb
import min1_production as M

def spec_model():
    return lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.03, num_leaves=128, min_child_samples=200,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=10, n_estimators=600,
                              n_jobs=20, verbosity=-1)

def boot(c, nb=5000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def main():
    t0 = time.time()
    p, *_ = M._load()
    qb = p["bbw1800_q67"]; rqp = p["rel_tighten"]; FN = p["feature_names"]; STR = 5
    print(f"[upspec] gate bbw<={qb:.2e} rel>={rqp:.3f}; training side-specialists on dip/rally setups {time.time()-t0:.0f}s", flush=True)

    def setups(sp):
        b = M.load_split(sp); X, y, mag, valid, ts, idx = M.prep(b)
        r300 = X["ret300"].values; bbw = X["bbw1800"].values; rel = X["rel_ratio"].values
        comp = valid & (bbw <= qb) & (rel >= rqp)
        up_s = comp & (r300 < 0); dn_s = comp & (r300 > 0)
        return X[FN], y, mag, ts, idx.year.values, up_s, dn_s

    # TRAIN specialists on 2021-2023 setup bars
    Xtr, ytr, mtr, _, _, up_tr, dn_tr = setups("train")
    iu = np.where(up_tr & (mtr > 0))[0][::STR]; idn = np.where(dn_tr & (mtr > 0))[0][::STR]
    print(f"[upspec] train up-setups={len(iu)} (bounce rate {ytr[iu].mean():.3f}) dn-setups={len(idn)} (up rate {ytr[idn].mean():.3f}) {time.time()-t0:.0f}s", flush=True)
    Mu = spec_model().fit(Xtr.iloc[iu], ytr[iu])           # P(up) on dip setups -> bet UP when high
    Md = spec_model().fit(Xtr.iloc[idn], ytr[idn])         # P(up) on rally setups -> bet DOWN when low
    del Xtr
    print(f"[upspec] specialists trained {time.time()-t0:.0f}s", flush=True)

    # cache per-split slim predictions
    def predict(sp):
        X, y, mag, ts, yr, up_s, dn_s = setups(sp)
        pu = Mu.predict_proba(X)[:, 1]; pd_ = Md.predict_proba(X)[:, 1]
        return dict(y=y, mag=mag, ts=ts, yr=yr, up_s=up_s, dn_s=dn_s, pu=pu.astype("float32"), pd=pd_.astype("float32"))
    VA = predict("val"); TE = predict("test"); OO = predict("oos")
    print(f"[upspec] predictions ready {time.time()-t0:.0f}s", flush=True)
    vord = np.argsort(VA["ts"]); h = len(vord) // 2
    vh1 = np.zeros(len(VA["ts"]), bool); vh1[vord[:h]] = True; vh2 = ~vh1
    WIN = [("2024", TE, 2024), ("2025", TE, 2025), ("2026", OO, 2026)]

    def up_acc(D, thr, ymask=None):
        cand = D["up_s"] & (D["pu"] >= thr)
        if ymask is not None: cand = cand & ymask
        sel = M.nonoverlap_chrono(D["ts"], cand)
        if len(sel) == 0: return np.array([])
        return ((D["y"][sel] == 1) & (D["mag"][sel] > 0)).astype(float)   # bet UP
    def dn_acc(D, thr, ymask=None):
        cand = D["dn_s"] & (D["pd"] <= 1.0 - thr)
        if ymask is not None: cand = cand & ymask
        sel = M.nonoverlap_chrono(D["ts"], cand)
        if len(sel) == 0: return np.array([])
        return ((D["y"][sel] == 0) & (D["mag"][sel] > 0)).astype(float)   # bet DOWN

    def select_thr(accfn):
        """worst-VAL-half-stable threshold over a confidence grid (require both halves n>=20)."""
        pool = VA["pu"] if accfn is up_acc else VA["pd"]
        grid = [float(np.quantile(pool, q)) for q in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)]
        best = None
        for thr in grid:
            c1 = accfn(VA, thr, vh1); c2 = accfn(VA, thr, vh2)
            if len(c1) < 20 or len(c2) < 20: continue
            hm = min(c1.mean(), c2.mean())
            if best is None or hm > best[0]: best = (hm, thr)
        return best

    for sidename, accfn in (("UP-specialist (dip-buy)", up_acc), ("DOWN-specialist (rally-sell, control)", dn_acc)):
        b = select_thr(accfn)
        if b is None: print(f"\n[{sidename}] no eligible VAL threshold"); continue
        hm, thr = b
        print(f"\n[{sidename}] frozen thr={thr:.3f} (VAL worst-half={hm:.3f}):", flush=True)
        accs = []
        for w, D, yy in WIN:
            c = accfn(D, thr, D["yr"] == yy)
            if len(c) < 5: print(f"    {w}: n{len(c)} (thin)"); accs.append(np.nan); continue
            lo, hi = boot(c); accs.append(c.mean())
            flag = "  <-- clears 0.65!" if (len(c) >= 25 and lo > 0.65) else ""
            print(f"    {w}: {c.mean():.3f} (n{len(c)}, CI[{lo:.3f},{hi:.3f}]){flag}", flush=True)
        fl = np.nanmin(accs)
        print(f"    FLOOR across windows = {fl:.3f}", flush=True)
    print(f"\n[upspec] DONE {time.time()-t0:.0f}s  (vs symmetric-book up-side 2024 .520 / 2025 .584 / 2026 .613)", flush=True)

if __name__ == "__main__":
    main()
