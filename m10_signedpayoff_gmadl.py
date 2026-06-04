"""N18 (bounded) — SIGN-COUPLED GMADL loss head @ MX_HOR=10, the one distinct loss family untested at 10m.
Compact retarget of m5_signedpayoff_torch.py HEAD A (GMADL lgb fobj only; skips the slow torch RRL head — that is
escalated only if this shows life). GMADL couples the model's SIGNED score z to the SIGNED forward return R:
    L_i = -sigma(a*R*z)*|R|^b      (a sign-WRONG large move penalized MORE than a sign-right one)
This is NOT magweight (symmetric |ret|^POW reweight of BCE — KILLED here) nor a relabel (N19 — KILLED here): the
R*z coupling is explicit. grad/hess wrt raw margin z (per m5_signedpayoff docstring), hess floored |.|+eps.

Eval: NY gate + conf-cover COV=0.05, nonoverlap_chrono(600), per-year side-split 2024/25/26, ties LOSE (raw _y),
selection on WORST-VAL-HALF. Sweep a in {50,100}, b in {1,2}.
INCUMBENT (certified EURUSD.m10xp.v1): refit p10 UP .5863 / DOWN .5683; forward cov10 2025 UP .605 / DOWN .574.
PRE-REGISTERED FALSIFIER (written BEFORE OOS): KILL the sign-coupled-loss family at 10m unless some (a,b) beats the
incumbent on the BINDING 2025 year on >=1 side (UP-2025 >= .605 OR DOWN-2025 >= .574) at healthy n with up-rate in
[.47,.53]. Else: loss-reopt family exhausted at 10m (consistent with magweight + N19 KILLs on the 2025 wall).
    python m10_signedpayoff_gmadl.py
"""
import os
os.environ["MX_HOR"] = "10"
import json, time, numpy as np
import lightgbm as lgb
import m5_xpair as MX

MODE = "xpof"; SPL = MX.SPL; GAP = 600; COV = 0.05; BE = 0.541
OUT = "/media/sean/CORSAIR/binary-algo/m10_signedpayoff_gmadl_result.json"
INC = {"UP_2025": 0.605, "DOWN_2025": 0.574}


def gmadl_fobj_builder(R, a, b):
    absRb = np.abs(R) ** b
    aR = a * R
    def fobj(preds, dtrain):
        z = preds
        u = aR * z
        s = 1.0 / (1.0 + np.exp(-np.clip(u, -30, 30)))
        sp = s * (1.0 - s)
        grad = -absRb * sp * aR
        hess = -absRb * (aR ** 2) * sp * (1.0 - 2.0 * s)
        hess = np.abs(hess) + 1e-6
        return grad, hess
    return fobj


def boot(c, nb=3000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    return tuple(float(x) for x in np.percentile([c[rng.integers(0, n, n)].mean() for _ in range(nb)], [2.5, 97.5]))


def side_year(raw, y, ts, ny, cover_thr, side):
    """side=1 UP (z>0), 0 DOWN (z<0). conf=|raw|. cover_thr frozen on worst-VAL-half (or per-window if None)."""
    conf = np.abs(raw)
    g = ny & ((raw > 0) == (side == 1))
    if g.sum() < 20: return {"n": 0, "win": None, "ci": [None, None]}
    cthr = cover_thr if cover_thr is not None else float(np.quantile(conf[g], 1 - COV))
    m = g & (conf >= cthr); sel = MX.nonoverlap_chrono(ts, m, GAP)
    if len(sel) < 10: return {"n": len(sel), "win": None, "ci": [None, None]}
    pred = side  # all selected are this side
    cc = ((1 if side == 1 else 0) == y[sel]).astype(float)
    lo, hi = boot(cc)
    return {"n": len(sel), "win": round(float(cc.mean()), 4), "ci": [round(lo, 4), round(hi, 4)],
            "out_up_rate": round(float(y[sel].mean()), 4)}


def worst_val_half_cover(raw, y, ts, ny, side):
    conf = np.abs(raw); g = ny & ((raw > 0) == (side == 1)); mid = np.median(ts); h1 = ts < mid
    best = None
    for cov in (0.10, 0.05, 0.03):
        pool = conf[g]
        if pool.size < 400: continue
        thr = float(np.quantile(pool, 1 - cov)); accs = []; ok = True
        for hm in (h1, ~h1):
            m = g & hm & (conf >= thr); sel = MX.nonoverlap_chrono(ts, m, GAP)
            if len(sel) < 40: ok = False; break
            accs.append(((1 if side == 1 else 0) == y[sel]).mean())
        if not ok: continue
        w = min(accs)
        if best is None or w > best[1]: best = (thr, w, cov)
    return best


def main():
    t0 = time.time()
    stub = {"experiment": "m10_signedpayoff_gmadl", "incumbent": INC, "breakeven": BE,
            "PRE_REGISTERED_FALSIFIER": "KILL sign-coupled-loss family @10m unless some (a,b) beats incumbent on "
            "binding 2025 (UP>=.605 OR DOWN>=.574) at healthy n, up-rate in [.47,.53].", "status": "PRE-REGISTERED",
            "grid": [], "VERDICT": None}
    json.dump(stub, open(OUT, "w"), indent=1)
    TR = MX.augment(MX.build_xp(SPL["train"], 4), SPL["train"], MODE)
    VA = MX.augment(MX.build_xp(SPL["val"]), SPL["val"], MODE)
    xpc = MX.xp_cols(TR); cols = MX.feat_cols(MODE, TR, xpc)
    Rtr = TR["_fwd"].values.astype(float); ytr = TR["_y"].astype(int).values
    Xtr = TR[cols].astype("float32").values; Xva = VA[cols].astype("float32").values
    yva = VA["_y"].astype(int).values; tsv = VA["_ts"].values.astype("int64"); nyv = VA["sess_ny"].values > 0.5
    print(f"[gmadl10] train={len(TR):,} val={len(VA):,} feats={len(cols)} {time.time()-t0:.0f}s", flush=True)
    dtr = lgb.Dataset(Xtr, label=ytr, free_raw_data=False)
    years = {}
    for w in ("test24", "test25", "oos"):
        D = MX.augment(MX.build_xp(SPL[w]), SPL[w], MODE)
        years[w] = (D[cols].astype("float32").values, D["_y"].astype(int).values,
                    D["_ts"].values.astype("int64"), D["sess_ny"].values > 0.5)
        del D
    grid = []
    for a in (50.0, 100.0):
        for b in (1, 2):
            fobj = gmadl_fobj_builder(Rtr, a, b)
            bst = lgb.train({"learning_rate": 0.03, "num_leaves": 127, "min_child_samples": 300, "subsample": 0.8,
                             "bagging_freq": 1, "colsample_bytree": 0.5, "reg_lambda": 20, "verbosity": -1, "num_threads": 20},
                            dtr, num_boost_round=600, fobj=fobj)
            rawva = bst.predict(Xva, raw_score=True)
            row = {"a": a, "b": b}
            for side, nm in ((1, "UP"), (0, "DOWN")):
                fr = worst_val_half_cover(rawva, yva, tsv, nyv, side)
                cthr = fr[0] if fr else None
                yr = {}
                for w, lab in (("test24", "2024"), ("test25", "2025"), ("oos", "2026")):
                    X, y, ts, ny = years[w]
                    raw = bst.predict(X, raw_score=True)
                    yr[lab] = side_year(raw, y, ts, ny, cthr, side)
                row[nm] = {"val_cover": (None if not fr else {"thr": fr[0], "worsthalf": round(fr[1], 4), "cov": fr[2]}), "years": yr}
            grid.append(row)
            u25 = row["UP"]["years"]["2025"]; d25 = row["DOWN"]["years"]["2025"]
            print(f"  a={a} b={b}: UP25 {u25.get('win')}(n{u25.get('n')}) DOWN25 {d25.get('win')}(n{d25.get('n')}) {time.time()-t0:.0f}s", flush=True)

    # verdict: any (a,b) beats incumbent on binding 2025 either side at healthy n
    def beats(row):
        u = row["UP"]["years"]["2025"]; d = row["DOWN"]["years"]["2025"]
        up_ok = u.get("win") is not None and u["win"] >= INC["UP_2025"] and u.get("n", 0) >= 30
        dn_ok = d.get("win") is not None and d["win"] >= INC["DOWN_2025"] and d.get("n", 0) >= 30
        return up_ok or dn_ok
    survivor = next((r for r in grid if beats(r)), None)
    stub.update({"status": "COMPLETE", "grid": grid,
                 "VERDICT": {"SURVIVES": bool(survivor),
                             "best": ({"a": survivor["a"], "b": survivor["b"]} if survivor else None),
                             "statement": ("GMADL sign-coupled loss @10m " + ("SURVIVES — beats incumbent on 2025." if survivor
                                           else "KILLED — no (a,b) beats incumbent on binding 2025; loss-reopt family exhausted at 10m (cf. magweight + N19 KILLs)."))}})
    json.dump(stub, open(OUT, "w"), indent=1)
    print(f"[gmadl10] {stub['VERDICT']['statement']} ({time.time()-t0:.0f}s) -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
