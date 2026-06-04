"""ROUND-2 DISCOVER (novel combo) — CROSS-HORIZON BLEND of the certified 10m + 15m cross-pair books, for the
10m direction label. MECHANISM (reasoned before building): the 15m book is the SAME cross-pair USD-common-factor
mechanism one horizon out; its P(up) is a longer-horizon directional view that MIGHT carry complementary regime
information for the 10m sign (decorrelated errors → blend firms the binding 2025). RISK: same mechanism/features
→ predictions likely highly correlated → blend ≈ no help (cf. 5m seed-ens corr .694, no lift). So we MEASURE
corr(p10,p15) FIRST, then test mean-blend + learned-weight blend.

Both books have IDENTICAL 331-feat sets (verified) → predict both on one xpof frame @ MX_HOR=10 (10m label/gate).
Gate = 10m book's 5m_bb_width<=thr × NY (regime) × blend-confidence cover (cov10, worst-VAL-half). Side-split.
INCUMBENT = p10-alone (certified m10xp): forward cov10 2025 UP .605 / DOWN .574 ; refit p10 UP .5863 / DOWN .5683.
PRE-REGISTERED FALSIFIER: KILL the cross-horizon blend unless it beats p10-alone on the BINDING 2025 year on >=1
side at >= the same coverage AND corr(p10,p15) is low enough to be a genuine decorrelated gain (else it's just p10).
    python m10_xhorizon_blend.py
"""
import os
os.environ["MX_HOR"] = "10"
import json, time, numpy as np
import lightgbm as lgb
import m5_xpair as MX

GAP = 600; BE = 0.541; R = 0.85
B10 = "/media/sean/CORSAIR/binary-algo/books/EURUSD.m10xp.v1"
B15 = "/media/sean/CORSAIR/binary-algo/books/EURUSD.m15xp.v1"
OUT = "/media/sean/CORSAIR/binary-algo/m10_xhorizon_blend_result.json"
INC = {"UP_2025": 0.605, "DOWN_2025": 0.574}


def boot(c, nb=3000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    return tuple(float(x) for x in np.percentile([c[rng.integers(0, n, n)].mean() for _ in range(nb)], [2.5, 97.5]))


def main():
    t0 = time.time()
    s10 = json.load(open(f"{B10}/m10xp_EURUSD_strategy.json")); c10 = s10["primary_feats"]
    s15 = json.load(open(f"{B15}/m15xp_EURUSD_strategy.json")); c15 = s15["primary_feats"]
    GATEF = s10.get("gate_feat", "5m_bb_width"); bthr = s10["bb_width_thr"]
    P10 = lgb.Booster(model_file=f"{B10}/m10xp_EURUSD_primary_lgb.txt")
    P15 = lgb.Booster(model_file=f"{B15}/m15xp_EURUSD_primary_lgb.txt")

    frames = {}
    for w, lab in (("val", "VAL"), ("test24", "2024"), ("test25", "2025"), ("oos", "2026")):
        D = MX.augment(MX.build_xp(MX.SPL[w]), MX.SPL[w], "xpof")
        p10 = P10.predict(D[c10].astype("float32").values); p15 = P15.predict(D[c15].astype("float32").values)
        frames[lab] = dict(p10=p10, p15=p15, y=D["_y"].astype(int).values, ts=D["_ts"].values.astype("int64"),
                           bb=D[GATEF].values.astype(float), ny=D["sess_ny"].values > 0.5)
        del D
    # MEASURE correlation on VAL gated regime
    v = frames["VAL"]; gm = (v["bb"] <= bthr) & v["ny"]
    corr = float(np.corrcoef(v["p10"][gm], v["p15"][gm])[0, 1])
    print(f"[xhblend] corr(p10,p15) on VAL gated regime = {corr:.3f}  (n={int(gm.sum())}) {time.time()-t0:.0f}s", flush=True)

    # learned weight on VAL worst-half (logistic of [p10,p15] -> y), simple grid on w in [0,1] for mean-blend variants
    def blend_score(p10, p15, w): return w * p10 + (1 - w) * p15

    # choose blend weight + cov on VAL worst-half (side-agnostic, maximize worse-half combined gated acc)
    ts = v["ts"]; mid = np.median(ts); h1 = ts < mid
    best = None
    for w in (1.0, 0.7, 0.5):       # w=1.0 == p10-alone (baseline within the same harness)
        sb = blend_score(v["p10"], v["p15"], w); conf = np.abs(sb - 0.5)
        for cov in (0.10, 0.05):
            g = gm.copy()
            if g.sum() < 300: continue
            cthr = float(np.quantile(conf[g], 1 - cov)); accs = []; ok = True
            for hm in (h1, ~h1):
                m = g & hm & (conf >= cthr); sel = MX.nonoverlap_chrono(ts, m, GAP)
                if len(sel) < 60: ok = False; break
                accs.append(((sb[sel] > 0.5).astype(int) == v["y"][sel]).mean())
            if not ok: continue
            wh = min(accs)
            if best is None or wh > best[0]: best = (wh, w, cov, cthr)
    wh, W, COV, _ = best
    print(f"[xhblend] worst-VAL-half pick: w={W} cov{COV:.0%} worsthalf={wh:.4f}", flush=True)

    res = {"corr_p10_p15": round(corr, 3), "blend_w": W, "cov": COV, "per_year": {}}
    for lab in ("2024", "2025", "2026"):
        f = frames[lab]; sb = blend_score(f["p10"], f["p15"], W); conf = np.abs(sb - 0.5)
        cthr = float(np.quantile(conf[(f["bb"] <= bthr) & f["ny"]], 1 - COV))
        m = (f["bb"] <= bthr) & f["ny"] & (conf >= cthr); sel = MX.nonoverlap_chrono(f["ts"], m, GAP)
        pred = (sb[sel] > 0.5).astype(int); yy = f["y"][sel]
        d = {}
        for side, nm in ((1, "UP"), (0, "DOWN"), (None, "ALL")):
            ss = (pred == side) if side is not None else np.ones(len(pred), bool)
            if ss.sum() < 5: d[nm] = {"n": int(ss.sum()), "win": None}; continue
            win = float((pred[ss] == yy[ss]).mean()); lo, hi = boot((pred[ss] == yy[ss]).astype(float))
            d[nm] = {"n": int(ss.sum()), "win": round(win, 4), "ci": [round(lo, 4), round(hi, 4)], "ev": round(win * R - (1 - win), 4)}
        res["per_year"][lab] = d
        print(f"  {lab}: UP {d['UP'].get('win')}(n{d['UP']['n']}) DOWN {d['DOWN'].get('win')}(n{d['DOWN']['n']})", flush=True)

    u25 = res["per_year"]["2025"]["UP"]; d25 = res["per_year"]["2025"]["DOWN"]
    beats = ((u25.get("win") or 0) >= INC["UP_2025"] and u25.get("n", 0) >= 30) or \
            ((d25.get("win") or 0) >= INC["DOWN_2025"] and d25.get("n", 0) >= 30)
    genuine = corr < 0.9 and W < 1.0
    res.update({"incumbent": INC, "breakeven": BE,
                "PRE_REGISTERED_FALSIFIER": "KILL unless blend beats p10-alone on binding 2025 (UP>=.605 OR DOWN>=.574) at >=same cov AND corr<0.9 (genuine decorrelated gain)",
                "VERDICT": {"SURVIVES": bool(beats and genuine),
                            "note": (f"corr={corr:.3f} (>=0.9 => same signal, blend==p10); chosen w={W} "
                                     + ("(==p10-alone; no decorrelated gain)" if W >= 1.0 else "")
                                     + (" SURVIVES" if (beats and genuine) else " KILLED — cross-horizon blend does not beat p10-alone on 2025"))}})
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"[xhblend] {res['VERDICT']['note']} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
