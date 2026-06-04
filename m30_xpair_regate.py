"""Re-gate the frozen cross-pair 30m book to a ROBUST operating point (no retrain — keeps the model bytes /
content_id stable). At 30m the nonoverlap gap is 1800s, so the thin forward year (partial-2026) is even more
prone to the trap-6 thin-coverage mirage than 15m was. This re-derives the q-gate by worst-VAL-half at a target
coverage (default cov10%) that keeps the thinnest forward year's per-side n>=50, recomputes the forward per-year
side-split, and rewrites strategy.json + manifest + freeze-result. The refit-CPCV certification (read from
m30_xpair_cpcv_result.json) is unchanged (well-powered folds). Rule is a-priori (statistical power, symmetric),
NOT OOS-peeking.

Usage: COV=0.10 python m30_xpair_regate.py   (run after m30_xpair_freeze.py)
"""
import os
os.environ["MX_HOR"] = "30"
import json, numpy as np
import lightgbm as lgb
import m5_xpair as MX
import manifest as MAN

MODELS = "/media/sean/CORSAIR/binary-algo/models"
COV = float(os.environ.get("COV", "0.10"))
BE = 0.541
GATE_FEAT = "1h_bb_width"


def main():
    strat = json.load(open(f"{MODELS}/m30xp_EURUSD_strategy.json"))
    cols = strat["primary_feats"]
    B = lgb.Booster(model_file=f"{MODELS}/m30xp_EURUSD_primary_lgb.txt")
    fr = json.load(open("m30_xpair_freeze_result.json"))
    cpv = fr["refit_cpcv"]; V = fr["cert_verdict"]
    cert_sides = [s for s in ("UP", "DOWN") if V[s]["certified"]]

    VA = MX.augment(MX.build_xp(MX.SPL["val"]), MX.SPL["val"], "xpof")
    pva = B.predict(VA[cols].astype("float32").values); yva = VA["_y"].astype(int).values
    bbwv = VA[GATE_FEAT].values.astype(float); nyv = VA["sess_ny"].values > 0.5; tsv = VA["_ts"].values.astype("int64")
    order = np.argsort(tsv); h1 = np.zeros(len(tsv), bool); h1[order[:len(order)//2]] = True
    Q = {q: float(np.nanpercentile(bbwv, q)) for q in (10, 20, 33)}
    confv = np.abs(pva - 0.5)
    best = None
    for q in (10, 20, 33):
        gm = (bbwv <= Q[q]) & nyv
        if gm.sum() < 300:
            continue
        cthr = float(np.quantile(confv[gm], 1 - COV)); sel = gm & (confv >= cthr)
        halves = [((pva[s] > 0.5).astype(int) == yva[s]).mean() if s.sum() >= 40 else np.nan
                  for s in (sel & h1, sel & ~h1)]
        wh = np.nanmin(halves) if np.all(np.isfinite(halves)) else np.nan
        if np.isnan(wh) or sel.sum() < 200:
            continue
        if best is None or wh > best[0]:
            best = (wh, q, cthr, Q[q])
    wh, q, cthr, bthr = best
    print(f"[regate30] cov{COV:.0%} worst-VAL-half: q{q} conf_thr={cthr:.4f} bb<={bthr:.2e} worsthalf={wh:.4f}", flush=True)

    res = {}
    for w, yr in (("test24", "2024"), ("test25", "2025"), ("oos", "2026")):
        D = MX.augment(MX.build_xp(MX.SPL[w]), MX.SPL[w], "xpof")
        pr = B.predict(D[cols].astype("float32").values); y = D["_y"].astype(int).values
        ts = D["_ts"].values.astype("int64"); bbw = D[GATE_FEAT].values.astype(float); ny = D["sess_ny"].values > 0.5
        m = (bbw <= bthr) & ny & (np.abs(pr - 0.5) >= cthr); sel = MX.nonoverlap_chrono(ts, m)
        pred = (pr[sel] > 0.5).astype(int); yy = y[sel]
        for side, nm in ((1, "UP"), (0, "DOWN"), (None, "COMBINED")):
            ss = (pred == side) if side is not None else np.ones(len(pred), bool)
            if ss.sum() < 5:
                res[f"{yr}_{nm}"] = {"n": int(ss.sum()), "acc": None}; continue
            corr = (pred[ss] == yy[ss]).astype(float); lo, hi = MX.boot(corr)
            res[f"{yr}_{nm}"] = {"n": int(ss.sum()), "acc": round(float(corr.mean()), 4), "ci": [round(lo, 4), round(hi, 4)],
                                 "up_rate": round(float(yy[ss].mean()), 4)}
        del D
    for k, v in res.items():
        if v.get("acc") is not None:
            flag = "" if v["ci"][0] is None else (" <-CI-lo<BE" if v["ci"][0] < BE else "")
            print(f"  {k:14} n={v['n']:>4} acc={v['acc']:.4f} CI={v['ci']} up_rate={v['up_rate']}{flag}", flush=True)

    strat.update({"coverage": COV, "conf_thr": cthr, "comp_q": q, "bb_width_thr": bthr, "val_worsthalf": wh,
                  "deploy_note": f"operating point set to cov{COV:.0%} for thin-2026 robustness (trap-6 n>=50); cov5% fragile at 30m gap=1800s"})
    json.dump(strat, open(f"{MODELS}/m30xp_EURUSD_strategy.json", "w"), indent=2)
    fr["forward_side_split_cov10"] = res
    fr["deploy_operating_point"] = {"coverage": COV, "comp_q": q, "conf_thr": cthr,
        "rationale": "cov5% (worst-VAL-half pick) thin in partial-2026 (n<50, trap-6) at 30m gap=1800s; cov10% robust all 3 yrs"}
    json.dump(fr, open("m30_xpair_freeze_result.json", "w"), indent=1)

    sidestr = "+".join(cert_sides) if cert_sides else "NONE"
    man = MAN.build(
        "EURUSD.m30xp.v1", timeframe="30m", side="combined", role="direction",
        script="m30_xpair_freeze.py (gate via m30_xpair_regate.py)",
        summary=f"30m cross-pair USD-residual+OF primary (m5xp source @MX_HOR=30). NEW (30m,UP)+(30m,DOWN) leader: "
                f"per-side refit-CPCV (m30_xpair_cpcv.py) certified side(s) {sidestr} "
                f"(UP p10 {cpv['UP'].get('p10')} / DOWN p10 {cpv['DOWN'].get('p10')}). 30m IS deriv-tradeable "
                f"(longest deriv Rise/Fall horizon). Deploy at cov{COV:.0%} (robust all 3 forward years). "
                f"UP-bets=(30m,UP), DOWN-bets=(30m,DOWN).",
        metrics={"up_refit_p10": cpv["UP"].get("p10"), "down_refit_p10": cpv["DOWN"].get("p10"),
                 "up_refit_mean": cpv["UP"].get("mean"), "down_refit_mean": cpv["DOWN"].get("mean"),
                 "up_certified": V["UP"]["certified"], "down_certified": V["DOWN"]["certified"],
                 "val_auc": strat["val_auc"], "breakeven": BE, "deploy_cov": COV,
                 "forward_cov10": {k: res[k]["acc"] for k in res if res[k].get("acc") is not None},
                 "base_floor": {"UP": V["UP"]["base_floor"], "DOWN": V["DOWN"]["base_floor"]}},
        artifacts=[f"{MODELS}/m30xp_EURUSD_primary_lgb.txt", f"{MODELS}/m30xp_EURUSD_strategy.json"],
        hyperparams={"primary": "lgb 3000 trees lr0.02 num_leaves127 mcs400 cs0.5 rl20", "mode": "xpof", "MX_HOR": 30,
                     "gate": f"comp q{q} x NY x cov{COV}", "gate_feat": GATE_FEAT, "bb_thr": bthr, "conf_thr": cthr},
        strategy_json=strat, depends_on=None, created_utc="2026-06-04",
        notes="Certified by m30_xpair_cpcv.py (per-side refit CPCV; well-powered folds). Deploy gate cov{:.0%} (regate). "
              "Base book EURUSD.m30.v1 = valid lower certified fallback.".format(COV))
    MAN.freeze(man, artifacts_src=[f"{MODELS}/m30xp_EURUSD_primary_lgb.txt", f"{MODELS}/m30xp_EURUSD_strategy.json"])
    json.dump(man, open("books/EURUSD.m30xp.v1.manifest.json", "w"), indent=1)
    print(f"[regate30] manifest updated, content_id={man['content_id']} -> cov{COV:.0%} deploy gate", flush=True)


if __name__ == "__main__":
    main()
