"""FREEZE the certified cross-pair 15m book (A6f) -> EURUSD.m15xp.v1, the new (15m,UP)+(15m,DOWN) leader.

Trains the cross-pair USD-residual+OF PRIMARY at full fidelity (m5xp config, MX_HOR=15) on train 2012-2021,
early-stop on VAL; derives the comp(15m_bb_width q)xNY x conf-coverage gate by WORST-VAL-HALF stability (never
VAL-acc-max). Saves model + strategy.json, records the forward side-split (2024/25/26), then manifest.build +
freeze + git tag book/EURUSD.m15xp.v1. The per-side refit-CPCV certification (UP p10 .5673 / DOWN .5742, 15/15
paths) is in m15_xpair_cpcv_result.json; this is the deployable frozen artifact. Breakeven 0.541.

Usage: python m15_xpair_freeze.py
"""
import os
os.environ["MX_HOR"] = "15"
import json, time, subprocess, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import m5_xpair as MX
import manifest as MAN

MODELS = "/media/sean/CORSAIR/binary-algo/models"
BE = 0.541
YEARS = (("test24", "2024"), ("test25", "2025"), ("oos", "2026"))


def main():
    t0 = time.time()
    os.makedirs(MODELS, exist_ok=True)
    TR = MX.build_xp(MX.SPL["train"], 4); VA = MX.build_xp(MX.SPL["val"])
    xpc = MX.xp_cols(TR)
    TR = MX.augment(TR, MX.SPL["train"], "xpof"); VA = MX.augment(VA, MX.SPL["val"], "xpof")
    cols = MX.feat_cols("xpof", TR, xpc)
    ytr = TR["_y"].astype(int).values; yva = VA["_y"].astype(int).values
    P = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=127, min_child_samples=400,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=20, n_estimators=3000, n_jobs=20, verbosity=-1)
    P.fit(TR[cols].astype("float32"), ytr, eval_set=[(VA[cols].astype("float32"), yva)], eval_metric="auc",
          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    pva = P.predict_proba(VA[cols].astype("float32"))[:, 1]
    val_auc = float(roc_auc_score(yva, pva))
    print(f"[freeze] primary VAL AUC={val_auc:.4f} {time.time()-t0:.0f}s", flush=True)

    # gate: worst-VAL-half over (bb q{10,20,33} x cov{5,10%})
    bbwv = VA["15m_bb_width"].values.astype(float); nyv = VA["sess_ny"].values > 0.5; tsv = VA["_ts"].values.astype("int64")
    order = np.argsort(tsv); h1 = np.zeros(len(tsv), bool); h1[order[:len(order)//2]] = True
    Q = {q: float(np.nanpercentile(bbwv, q)) for q in (10, 20, 33)}
    best = None
    confv = np.abs(pva - 0.5)
    for q in (10, 20, 33):
        gm = (bbwv <= Q[q]) & nyv
        if gm.sum() < 300:
            continue
        for cov in (0.05, 0.10):
            cthr = float(np.quantile(confv[gm], 1 - cov)); sel = gm & (confv >= cthr)
            halves = []
            for hm in (h1, ~h1):
                s = sel & hm
                halves.append(((pva[s] > 0.5).astype(int) == yva[s]).mean() if s.sum() >= 40 else np.nan)
            wh = np.nanmin(halves) if np.all(np.isfinite(halves)) else np.nan
            if np.isnan(wh) or sel.sum() < 150:
                continue
            if best is None or wh > best[0]:
                best = (wh, q, cov, cthr, Q[q])
    wh, q, cov, cthr, bthr = best
    print(f"[freeze] gate worst-VAL-half: q{q} cov{cov:.0%} conf_thr={cthr:.4f} bb<={bthr:.2e} worsthalf={wh:.4f}", flush=True)

    # save artifacts
    model_path = f"{MODELS}/m15xp_EURUSD_primary_lgb.txt"
    P.booster_.save_model(model_path)
    strat = {"pair": "EURUSD", "mode": "xpof", "MX_HOR": 15, "primary_feats": cols, "bb_width_thr": bthr,
             "comp_q": q, "coverage": cov, "conf_thr": cthr, "gate": "15m_bb_width<=thr AND sess_ny>0.5 AND |p-0.5|>=conf_thr",
             "val_auc": val_auc, "val_worsthalf": wh, "settlement": "15m wall-clock mid-to-mid, ties excluded, nonoverlap_chrono 900s",
             "splits": {"train": "2012-2021", "val": "2022-2023", "test": "2024-2025", "oos": "2026"}}
    strat_path = f"{MODELS}/m15xp_EURUSD_strategy.json"
    json.dump(strat, open(strat_path, "w"), indent=2)

    # forward side-split per year at the frozen gate
    res = {}
    for w, yr in YEARS:
        D = MX.augment(MX.build_xp(MX.SPL[w]), MX.SPL[w], "xpof")
        pr = P.predict_proba(D[cols].astype("float32"))[:, 1]; y = D["_y"].astype(int).values
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
    out = {"book": "EURUSD.m15xp.v1", "val_auc": val_auc, "gate": strat["gate"],
           "refit_cpcv_cert": {"UP_p10": 0.5673, "DOWN_p10": 0.5742, "frac_clear_both": 1.0, "src": "m15_xpair_cpcv_result.json"},
           "forward_side_split": res, "breakeven": BE}
    json.dump(out, open("m15_xpair_freeze_result.json", "w"), indent=1)
    for k, v in res.items():
        if v.get("acc") is not None:
            print(f"  {k:14} n={v['n']:>4} acc={v['acc']:.4f} CI={v['ci']} up_rate={v['up_rate']}", flush=True)

    # manifest + freeze + git tag
    man = MAN.build(
        "EURUSD.m15xp.v1", timeframe="15m", side="combined", role="direction", script="m15_xpair_freeze.py",
        summary="15m cross-pair USD-residual+OF primary (m5xp source @MX_HOR=15). The NEW (15m,UP)+(15m,DOWN) "
                "leader: per-side refit-CPCV-CERTIFIED above the base book (UP p10 .5673 / DOWN p10 .5742, 15/15 "
                "purged paths clear both sides). Deploy: UP-bets = (15m,UP), DOWN-bets = (15m,DOWN). Deriv-tradeable.",
        metrics={"up_refit_p10": 0.5673, "down_refit_p10": 0.5742, "up_refit_mean": 0.5793, "down_refit_mean": 0.5818,
                 "frac_paths_clear_both": 1.0, "val_auc": val_auc, "breakeven": BE,
                 "improves_base_book": {"base_up_p10": 0.5475, "base_down_p10": 0.5486}},
        artifacts=[model_path, strat_path],
        hyperparams={"primary": "lgb 3000 trees lr0.02 num_leaves127 mcs400 cs0.5 rl20", "mode": "xpof", "MX_HOR": 15,
                     "gate": f"comp q{q} x NY x cov{cov}", "bb_thr": bthr, "conf_thr": cthr},
        strategy_json=strat, depends_on=None, created_utc="2026-06-03",
        notes="Certified by m15_xpair_cpcv.py (per-side refit CPCV). Forward side-split in m15_xpair_freeze_result.json. "
              "Base book EURUSD.m15.v1 remains a valid lower certified fallback.")
    path = MAN.freeze(man, artifacts_src=[model_path, strat_path])
    json.dump(man, open(f"books/EURUSD.m15xp.v1.manifest.json", "w"), indent=1)
    print(f"[freeze] froze -> {path} content_id={man['content_id']}", flush=True)
    try:
        subprocess.run(["git", "tag", "book/EURUSD.m15xp.v1"], check=False)
    except Exception as e:
        print("git tag note:", e)
    print(f"[freeze] DONE {time.time()-t0:.0f}s -> EURUSD.m15xp.v1 + m15_xpair_freeze_result.json", flush=True)


if __name__ == "__main__":
    main()
