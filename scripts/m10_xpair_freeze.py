"""FREEZE the cross-pair 10m book -> EURUSD.m10xp.v1 (retarget of m15_xpair_freeze.py to MX_HOR=10).

Trains the cross-pair USD-residual+OF PRIMARY at full fidelity (m5xp config, MX_HOR=10) on train 2012-2021,
early-stop on VAL; derives the comp(5m_bb_width q)xNY x conf-coverage gate by WORST-VAL-HALF stability (never
VAL-acc-max). Saves model + strategy.json, records the forward side-split (2024/25/26), then manifest.build +
freeze + git tag. The per-side refit-CPCV certification verdict is READ from m10_xpair_cpcv_result.json and the
base floors from m10_cpcv_side_result.json. Only meaningfully deployable for SIDES that certified (flagged per
side in metrics). NOTE: 10m < deriv 15m forex minimum -> research horizon; deployable sibling = EURUSD.m15xp.v1.
Breakeven 0.541.

Usage: python m10_xpair_freeze.py   (run only after m10_xpair_cpcv.py; freezes iff >=1 side certified)
"""
import os
os.environ["MX_HOR"] = "10"
import json, time, subprocess, sys, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import m5_xpair as MX
import manifest as MAN

MODELS = "/home/sean/git/binary-algo/models"
BE = 0.541
GATE_FEAT = "5m_bb_width"
YEARS = (("test24", "2024"), ("test25", "2025"), ("oos", "2026"))


def cert_verdict():
    cp = json.load(open("m10_xpair_cpcv_result.json"))
    base = json.load(open("m10_cpcv_side_result.json"))
    out = {}
    for s in ("UP", "DOWN"):
        r = cp.get(s, {}); bf = base.get(s, {}).get("p10", BE) if base.get(s, {}).get("n_paths", 0) >= 5 else BE
        certified = r.get("n_paths", 0) >= 5 and r.get("p10", 0) >= BE and r.get("frac_clear_BE", 0) >= 0.80
        improves = r.get("p10", 0) > bf
        out[s] = {"p10": r.get("p10"), "frac": r.get("frac_clear_BE"), "n_paths": r.get("n_paths"),
                  "base_floor": bf, "certified": bool(certified), "improves_base": bool(improves and certified)}
    return cp, base, out


def main():
    t0 = time.time()
    cp, base, V = cert_verdict()
    cert_sides = [s for s in ("UP", "DOWN") if V[s]["certified"]]
    print(f"[freeze10] A6c verdict: " + " ".join(f"{s}{'CERT' if V[s]['certified'] else 'no'}(p10={V[s]['p10']},base={V[s]['base_floor']})" for s in ("UP", "DOWN")), flush=True)
    if not cert_sides and "--force" not in sys.argv:
        print("[freeze10] NO side certified under refit-CPCV -> NOT freezing a cross-pair book (base EURUSD.m10.v1 stays incumbent). Use --force to freeze anyway.", flush=True)
        return
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
    print(f"[freeze10] primary VAL AUC={val_auc:.4f} {time.time()-t0:.0f}s", flush=True)

    bbwv = VA[GATE_FEAT].values.astype(float); nyv = VA["sess_ny"].values > 0.5; tsv = VA["_ts"].values.astype("int64")
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
    print(f"[freeze10] gate worst-VAL-half: q{q} cov{cov:.0%} conf_thr={cthr:.4f} bb<={bthr:.2e} worsthalf={wh:.4f}", flush=True)

    model_path = f"{MODELS}/m10xp_EURUSD_primary_lgb.txt"
    P.booster_.save_model(model_path)
    strat = {"pair": "EURUSD", "mode": "xpof", "MX_HOR": 10, "primary_feats": cols, "bb_width_thr": bthr,
             "comp_q": q, "coverage": cov, "conf_thr": cthr, "gate_feat": GATE_FEAT,
             "gate": f"{GATE_FEAT}<=thr AND sess_ny>0.5 AND |p-0.5|>=conf_thr",
             "val_auc": val_auc, "val_worsthalf": wh, "settlement": "10m wall-clock mid-to-mid, ties excluded, nonoverlap_chrono 600s",
             "splits": {"train": "2012-2021", "val": "2022-2023", "test": "2024-2025", "oos": "2026"},
             "cert_sides": cert_sides}
    strat_path = f"{MODELS}/m10xp_EURUSD_strategy.json"
    json.dump(strat, open(strat_path, "w"), indent=2)

    res = {}
    for w, yr in YEARS:
        D = MX.augment(MX.build_xp(MX.SPL[w]), MX.SPL[w], "xpof")
        pr = P.predict_proba(D[cols].astype("float32"))[:, 1]; y = D["_y"].astype(int).values
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
    out = {"book": "EURUSD.m10xp.v1", "val_auc": val_auc, "gate": strat["gate"], "cert_verdict": V,
           "refit_cpcv": {s: {"p10": cp[s].get("p10"), "frac": cp[s].get("frac_clear_BE"), "mean": cp[s].get("mean")} for s in ("UP", "DOWN")},
           "forward_side_split": res, "breakeven": BE}
    json.dump(out, open("m10_xpair_freeze_result.json", "w"), indent=1)
    for k, v in res.items():
        if v.get("acc") is not None:
            print(f"  {k:14} n={v['n']:>4} acc={v['acc']:.4f} CI={v['ci']} up_rate={v['up_rate']}", flush=True)

    sidestr = "+".join(cert_sides) if cert_sides else "NONE(forced)"
    man = MAN.build(
        "EURUSD.m10xp.v1", timeframe="10m", side="combined", role="direction", script="m10_xpair_freeze.py",
        summary=f"10m cross-pair USD-residual+OF primary (m5xp source @MX_HOR=10). Certified side(s): {sidestr} "
                f"(per-side refit-CPCV m10_xpair_cpcv.py: UP p10 {V['UP']['p10']} / DOWN p10 {V['DOWN']['p10']}; "
                f"base floors UP {V['UP']['base_floor']} / DOWN {V['DOWN']['base_floor']}). 10m < deriv 15m min = "
                f"research horizon; deployable sibling EURUSD.m15xp.v1.",
        metrics={"up_refit_p10": cp["UP"].get("p10"), "down_refit_p10": cp["DOWN"].get("p10"),
                 "up_certified": V["UP"]["certified"], "down_certified": V["DOWN"]["certified"],
                 "val_auc": val_auc, "breakeven": BE,
                 "base_floor": {"UP": V["UP"]["base_floor"], "DOWN": V["DOWN"]["base_floor"]}},
        artifacts=[model_path, strat_path],
        hyperparams={"primary": "lgb 3000 trees lr0.02 num_leaves127 mcs400 cs0.5 rl20", "mode": "xpof", "MX_HOR": 10,
                     "gate": f"comp q{q} x NY x cov{cov}", "gate_feat": GATE_FEAT, "bb_thr": bthr, "conf_thr": cthr},
        strategy_json=strat, depends_on=None, created_utc="2026-06-04",
        notes="Certified by m10_xpair_cpcv.py (per-side refit CPCV). Forward side-split in m10_xpair_freeze_result.json. "
              "Base book EURUSD.m10.v1 (combined) remains the lower fallback.")
    path = MAN.freeze(man, artifacts_src=[model_path, strat_path])
    json.dump(man, open("books/EURUSD.m10xp.v1.manifest.json", "w"), indent=1)
    print(f"[freeze10] froze -> {path} content_id={man['content_id']} cert_sides={cert_sides}", flush=True)
    try:
        subprocess.run(["git", "tag", "book/EURUSD.m10xp.v1"], check=False)
    except Exception as e:
        print("git tag note:", e)
    print(f"[freeze10] DONE {time.time()-t0:.0f}s -> EURUSD.m10xp.v1 + m10_xpair_freeze_result.json", flush=True)


if __name__ == "__main__":
    main()
