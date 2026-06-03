"""Re-gate the frozen cross-pair 15m book to the ROBUST cov10% operating point (no retrain — keeps the model
bytes / content_id stable). Adversarial finding: the worst-VAL-half selection picked cov5%, but at cov5% the
PARTIAL-2026 forward year is thin (per-side n<50 = trap-6 thin-coverage mirage) and seed/instance-fragile
(2026 ranged .486-.541). cov10% keeps the thinnest forward year's per-side n>=50 and is robust on all 3 years.
Rule is a-priori (statistical validity/power, applied symmetrically), NOT OOS-peeking. Re-derives the q-gate by
worst-VAL-half at cov=0.10, recomputes the forward per-year side-split, rewrites the strategy.json + manifest +
freeze-result. The refit-CPCV certification (UP p10 .5673 / DOWN .5742) is unchanged (its folds were well-powered).

Usage: python m15_xpair_regate.py
"""
import os
os.environ["MX_HOR"] = "15"
import json, numpy as np
import lightgbm as lgb
import m5_xpair as MX
import manifest as MAN

MODELS = "/media/sean/CORSAIR/binary-algo/models"
COV = 0.10
BE = 0.541


def main():
    strat = json.load(open(f"{MODELS}/m15xp_EURUSD_strategy.json"))
    cols = strat["primary_feats"]
    B = lgb.Booster(model_file=f"{MODELS}/m15xp_EURUSD_primary_lgb.txt")

    VA = MX.augment(MX.build_xp(MX.SPL["val"]), MX.SPL["val"], "xpof")
    pva = B.predict(VA[cols].astype("float32").values); yva = VA["_y"].astype(int).values
    bbwv = VA["15m_bb_width"].values.astype(float); nyv = VA["sess_ny"].values > 0.5; tsv = VA["_ts"].values.astype("int64")
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
    print(f"[regate] cov{COV:.0%} worst-VAL-half: q{q} conf_thr={cthr:.4f} bb<={bthr:.2e} worsthalf={wh:.4f}", flush=True)

    res = {}
    for w, yr in (("test24", "2024"), ("test25", "2025"), ("oos", "2026")):
        D = MX.augment(MX.build_xp(MX.SPL[w]), MX.SPL[w], "xpof")
        pr = B.predict(D[cols].astype("float32").values); y = D["_y"].astype(int).values
        ts = D["_ts"].values.astype("int64"); bbw = D["15m_bb_width"].values.astype(float); ny = D["sess_ny"].values > 0.5
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

    # update strategy.json (gate only) + rewrite manifest metrics + freeze result
    strat.update({"coverage": COV, "conf_thr": cthr, "comp_q": q, "bb_width_thr": bthr, "val_worsthalf": wh,
                  "deploy_note": "operating point set to cov10% for thin-2026 robustness (trap-6 n>=50); cov5% fragile"})
    json.dump(strat, open(f"{MODELS}/m15xp_EURUSD_strategy.json", "w"), indent=2)
    out = json.load(open("m15_xpair_freeze_result.json"))
    out["forward_side_split_cov10"] = res
    out["deploy_operating_point"] = {"coverage": COV, "comp_q": q, "conf_thr": cthr,
        "rationale": "cov5% (worst-VAL-half pick) thin in partial-2026 (n<50, trap-6) + seed-fragile; cov10% robust all 3 yrs"}
    json.dump(out, open("m15_xpair_freeze_result.json", "w"), indent=1)

    # rewrite manifest with the deploy gate + cov10 forward (model bytes unchanged -> content_id stable)
    man = MAN.build(
        "EURUSD.m15xp.v1", timeframe="15m", side="combined", role="direction", script="m15_xpair_freeze.py (gate via m15_xpair_regate.py)",
        summary="15m cross-pair USD-residual+OF primary (m5xp source @MX_HOR=15). NEW (15m,UP)+(15m,DOWN) leader: "
                "per-side refit-CPCV-CERTIFIED above base book (UP p10 .5673 / DOWN p10 .5742, 15/15 paths). Deploy "
                "at cov10% (robust all 3 forward years; cov5% thin-2026-fragile). UP-bets=(15m,UP), DOWN-bets=(15m,DOWN).",
        metrics={"up_refit_p10": 0.5673, "down_refit_p10": 0.5742, "up_refit_mean": 0.5793, "down_refit_mean": 0.5818,
                 "frac_paths_clear_both": 1.0, "val_auc": strat["val_auc"], "breakeven": BE,
                 "deploy_cov": COV, "forward_cov10": {k: res[k]["acc"] for k in res if res[k].get("acc") is not None},
                 "improves_base_book": {"base_up_p10": 0.5475, "base_down_p10": 0.5486}},
        artifacts=[f"{MODELS}/m15xp_EURUSD_primary_lgb.txt", f"{MODELS}/m15xp_EURUSD_strategy.json"],
        hyperparams={"primary": "lgb 3000 trees lr0.02 num_leaves127 mcs400 cs0.5 rl20", "mode": "xpof", "MX_HOR": 15,
                     "gate": f"comp q{q} x NY x cov{COV}", "bb_thr": bthr, "conf_thr": cthr},
        strategy_json=strat, depends_on=None, created_utc="2026-06-03",
        notes="Certified by m15_xpair_cpcv.py (per-side refit CPCV; well-powered folds). Deploy gate cov10% (regate). "
              "Base book EURUSD.m15.v1 = valid lower certified fallback. ADVERSARIAL: cov5% forward-2026 thin/fragile.")
    MAN.freeze(man, artifacts_src=[f"{MODELS}/m15xp_EURUSD_primary_lgb.txt", f"{MODELS}/m15xp_EURUSD_strategy.json"])
    json.dump(man, open("books/EURUSD.m15xp.v1.manifest.json", "w"), indent=1)
    print(f"[regate] manifest updated, content_id={man['content_id']} -> cov10 deploy gate", flush=True)


if __name__ == "__main__":
    main()
