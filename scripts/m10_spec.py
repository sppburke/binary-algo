"""(c) PURPOSE-BUILT SIDE SPECIALIST @10m — separately-trained per-side cross-pair primary via ASYMMETRIC
class weighting (distinct from magweight's |return| weighting and from N19's relabel). DOWN-specialist upweights
the DOWN class (boundary shifts toward higher DOWN precision); UP-specialist upweights UP. Tests the canonical
A8c question: does a purpose-built side model beat the SYMMETRIC book's side-split? (5m Tier-1: specialists are
WORSE — subset/asym training kills ranking; run once at 10m with a fast-KILL per the coverage rule.)

Eval: same NY x compression(5m_bb_width) x conf-cover gate as the certified book, worst-VAL-half cover select,
per-year side-split, nonoverlap_chrono(600), ties LOSE. INCUMBENT (certified EURUSD.m10xp.v1 symmetric side-split):
DOWN refit p10 .5683 / fwd-2025 .574 ; UP refit p10 .5863 / fwd-2025 .605.
PRE-REGISTERED FALSIFIER: KILL the specialist (symmetric side-split confirmed best) unless a side-specialist beats
the incumbent's SAME side on binding 2025 (DOWN>=.574 OR UP>=.605) at healthy n with up-rate tripwire ok.
    python m10_spec.py
"""
import os
os.environ["MX_HOR"] = "10"
import json, time, numpy as np
import lightgbm as lgb
import m5_xpair as MX

MODE = "xpof"; SPL = MX.SPL; GAP = 600; BE = 0.541
GATE = "5m_bb_width"
OUT = "/home/sean/git/binary-algo/m10_spec_result.json"
INC = {"UP_2025": 0.605, "DOWN_2025": 0.574}


def boot(c, nb=3000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    return tuple(float(x) for x in np.percentile([c[rng.integers(0, n, n)].mean() for _ in range(nb)], [2.5, 97.5]))


def worst_val_half(pr, y, ts, bbw, ny, bthr, side):
    """choose (cov) maximizing worse VAL-half side win, gate = bb<=bthr & NY & predicted-side & conf-cover."""
    conf = np.abs(pr - 0.5); g = (bbw <= bthr) & ny & ((pr > 0.5) == (side == 1))
    mid = np.median(ts); h1 = ts < mid; best = None
    for cov in (0.10, 0.05):
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
    stub = {"experiment": "m10_spec (asym class-weight side specialist)", "incumbent": INC, "breakeven": BE,
            "PRE_REGISTERED_FALSIFIER": "KILL specialist unless a side-specialist beats incumbent SAME side on 2025 "
            "(DOWN>=.574 OR UP>=.605) at healthy n, up-rate in [.47,.53].", "status": "PRE-REGISTERED", "sides": {}}
    json.dump(stub, open(OUT, "w"), indent=1)
    TR = MX.augment(MX.build_xp(SPL["train"], 4), SPL["train"], MODE)
    VA = MX.augment(MX.build_xp(SPL["val"]), SPL["val"], MODE)
    xpc = MX.xp_cols(TR); cols = MX.feat_cols(MODE, TR, xpc)
    ytr = TR["_y"].astype(int).values; Xtr = TR[cols].astype("float32").values
    bbtr = np.nanpercentile(TR[GATE].values.astype(float), 20)  # q20 bb threshold (matches m10 deployable gate spirit)
    Xva = VA[cols].astype("float32").values; yva = VA["_y"].astype(int).values
    tsv = VA["_ts"].values.astype("int64"); bbv = VA[GATE].values.astype(float); nyv = VA["sess_ny"].values > 0.5
    years = {}
    for w, lab in (("test24", "2024"), ("test25", "2025"), ("oos", "2026")):
        D = MX.augment(MX.build_xp(SPL[w]), SPL[w], MODE)
        years[lab] = (D[cols].astype("float32").values, D["_y"].astype(int).values, D["_ts"].values.astype("int64"),
                      D[GATE].values.astype(float), D["sess_ny"].values > 0.5)
        del D
    print(f"[spec10] train={len(TR):,} val={len(VA):,} feats={len(cols)} bb_q20={bbtr:.2e} {time.time()-t0:.0f}s", flush=True)

    for side, nm, cw in ((1, "UP", {0: 1.0, 1: 1.6}), (0, "DOWN", {0: 1.6, 1: 1.0})):
        P = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=127,
            min_child_samples=400, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=20,
            n_estimators=1500, n_jobs=20, verbosity=-1, class_weight=cw)
        P.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc", callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)])
        prv = P.predict_proba(Xva)[:, 1]
        fr = worst_val_half(prv, yva, tsv, bbv, nyv, bbtr, side)
        cthr = fr[0] if fr else None; cov = fr[2] if fr else None
        yr = {}
        for lab in ("2024", "2025", "2026"):
            X, y, ts, bbw, ny = years[lab]
            pr = P.predict_proba(X)[:, 1]; conf = np.abs(pr - 0.5)
            g = (bbw <= bbtr) & ny & ((pr > 0.5) == (side == 1)) & (conf >= (cthr if cthr is not None else 0))
            sel = MX.nonoverlap_chrono(ts, g, GAP)
            if len(sel) < 10: yr[lab] = {"n": len(sel), "win": None}; continue
            cc = ((1 if side == 1 else 0) == y[sel]).astype(float); lo, hi = boot(cc)
            yr[lab] = {"n": len(sel), "win": round(float(cc.mean()), 4), "ci": [round(lo, 4), round(hi, 4)],
                       "out_up_rate": round(float(y[sel].mean()), 4), "all_up_rate": round(float(y.mean()), 4)}
        stub["sides"][nm] = {"class_weight": cw, "val_cover": (None if not fr else {"thr": fr[0], "worsthalf": round(fr[1], 4), "cov": cov}), "years": yr}
        w25 = yr["2025"]
        print(f"  [{nm}-spec cw={cw}] 2025 win={w25.get('win')}(n{w25.get('n')}) vs inc {INC[nm+'_2025']} {time.time()-t0:.0f}s", flush=True)

    def beats(nm):
        w = stub["sides"][nm]["years"]["2025"]
        return w.get("win") is not None and w["win"] >= INC[nm + "_2025"] and w.get("n", 0) >= 30
    surv = beats("UP") or beats("DOWN")
    stub.update({"status": "COMPLETE",
                 "VERDICT": {"SURVIVES": bool(surv),
                             "statement": ("Side specialist @10m " + ("SURVIVES — beats symmetric side-split on 2025." if surv
                                           else "KILLED — symmetric side-split (certified book) remains best; asym-weight specialist does not beat it on 2025 (consistent with 5m: subset/asym training kills ranking)."))}})
    json.dump(stub, open(OUT, "w"), indent=1)
    print(f"[spec10] {stub['VERDICT']['statement']} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
