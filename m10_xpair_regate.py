"""Re-gate the frozen cross-pair 10m book (EURUSD.m10xp.v1) to the ROBUST cov10% operating point (no retrain —
model bytes / content_id stable). Retarget of m15_xpair_regate.py to MX_HOR=10 / gate 5m_bb_width. The freeze
picked cov5% by worst-VAL-half, but at cov5% the PARTIAL-2026 forward year is thin (per-side n=8/10 < 50 =
trap-6) — the same fragility that triggered the 15m regate. cov10% keeps the thinnest forward year's per-side n
larger and is the a-priori-valid (power, not OOS-peeking) deploy point. Re-derives the q-gate by worst-VAL-half
at cov0.10, recomputes the forward per-year side-split, rewrites strategy.json + freeze-result + manifest. The
refit-CPCV certification (UP p10 .5863 / DOWN p10 .5683, 15/15 paths, well-powered) is UNCHANGED — that is the
durable number; the forward side-split is descriptive. Breakeven 0.541.

Usage: python m10_xpair_regate.py
"""
import os
os.environ["MX_HOR"] = "10"
import json, numpy as np
import lightgbm as lgb
import m5_xpair as MX
import manifest as MAN

MODELS = "/media/sean/CORSAIR/binary-algo/models"
COV = 0.10
BE = 0.541


def main():
    strat = json.load(open(f"{MODELS}/m10xp_EURUSD_strategy.json"))
    cols = strat["primary_feats"]; GATE_FEAT = strat.get("gate_feat", "5m_bb_width")
    B = lgb.Booster(model_file=f"{MODELS}/m10xp_EURUSD_primary_lgb.txt")
    cp = json.load(open("m10_xpair_cpcv_result.json"))

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
    print(f"[regate10] cov{COV:.0%} worst-VAL-half: q{q} conf_thr={cthr:.4f} bb<={bthr:.2e} worsthalf={wh:.4f}", flush=True)

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
                  "deploy_note": "operating point set to cov10% for thin-2026 robustness (trap-6); cov5% fragile (n=8/10)"})
    json.dump(strat, open(f"{MODELS}/m10xp_EURUSD_strategy.json", "w"), indent=2)
    out = json.load(open("m10_xpair_freeze_result.json"))
    out["forward_side_split_cov10"] = res
    out["deploy_operating_point"] = {"coverage": COV, "comp_q": q, "conf_thr": cthr,
        "rationale": "cov5% (worst-VAL-half pick) thin in partial-2026 (n=8/10, trap-6); cov10% more robust. Durable number = refit-CPCV p10 (UP .5863 / DOWN .5683), NOT the overstated frozen-forward."}
    json.dump(out, open("m10_xpair_freeze_result.json", "w"), indent=1)

    man = MAN.build(
        "EURUSD.m10xp.v1", timeframe="10m", side="combined", role="direction", script="m10_xpair_freeze.py (gate via m10_xpair_regate.py)",
        summary="10m cross-pair USD-residual+OF primary (m5xp source @MX_HOR=10). NEW (10m,UP)+(10m,DOWN) leader: "
                "per-side refit-CPCV-CERTIFIED above base book (UP p10 .5863 / DOWN p10 .5683, 15/15 paths each, "
                "improves base floors .561/.552). Deploy at cov10% (cov5% thin-2026). 10m<deriv 15m min = research "
                "horizon; deployable sibling EURUSD.m15xp.v1. UP-bets=(10m,UP), DOWN-bets=(10m,DOWN).",
        metrics={"up_refit_p10": cp["UP"]["p10"], "down_refit_p10": cp["DOWN"]["p10"],
                 "up_refit_mean": cp["UP"]["mean"], "down_refit_mean": cp["DOWN"]["mean"],
                 "frac_paths_clear_both": 1.0, "val_auc": strat["val_auc"], "breakeven": BE,
                 "deploy_cov": COV, "forward_cov10": {k: res[k]["acc"] for k in res if res[k].get("acc") is not None},
                 "up_certified": True, "down_certified": True,
                 "improves_base_book": {"base_up_p10": 0.5611, "base_down_p10": 0.5519}},
        artifacts=[f"{MODELS}/m10xp_EURUSD_primary_lgb.txt", f"{MODELS}/m10xp_EURUSD_strategy.json"],
        hyperparams={"primary": "lgb 3000 trees lr0.02 num_leaves127 mcs400 cs0.5 rl20", "mode": "xpof", "MX_HOR": 10,
                     "gate": f"comp q{q} x NY x cov{COV}", "gate_feat": GATE_FEAT, "bb_thr": bthr, "conf_thr": cthr},
        strategy_json=strat, depends_on=None, created_utc="2026-06-04",
        notes="Certified by m10_xpair_cpcv.py (per-side refit CPCV; well-powered folds). Deploy gate cov10% (regate). "
              "Base book EURUSD.m10.v1 (combined) = lower fallback. Durable number = refit p10; frozen-forward overstates.")
    MAN.freeze(man, artifacts_src=[f"{MODELS}/m10xp_EURUSD_primary_lgb.txt", f"{MODELS}/m10xp_EURUSD_strategy.json"])
    json.dump(man, open("books/EURUSD.m10xp.v1.manifest.json", "w"), indent=1)
    print(f"[regate10] manifest updated, content_id={man['content_id']} -> cov10 deploy gate", flush=True)


if __name__ == "__main__":
    main()
