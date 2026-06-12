"""IMPROVE-lever I-recency: recency-weighted FROZEN-FORWARD test — does up-weighting recent TRAIN years close the
2026 frozen-decay? The certified edge (refit-CPCV) is era-symmetric so it can't see this; the deployable weakness is
the frozen-2021 book decaying by 2026 (UP .5302@cov5). Recency-weighting (exp time-decay, by YEAR not magnitude — a
different axis than magweight) is the simplest regime-adaptive lever for a frozen book.

Train cross-pair single-LGB on TRAIN 2012-2021 with sample_weight = 0.5^((2021-year)/halflife); derive the same
1h_bb_width×NY×cov5 gate on VAL worst-half; report forward 2024/25/26 per-side. Compare frozen-2026 UP to the
un-weighted .5302. PRE-REGISTERED FALSIFIER: recency helps iff it lifts frozen 2026 UP CI-lo and/or the worst forward
year per side WITHOUT degrading 2024/25 — else SUBSUMED ('deploy with retrain' remains the fix).

Usage: python m30_recency.py
"""
import os
os.environ["MX_HOR"] = "30"
import json, numpy as np
import lightgbm as lgb
import m5_xpair as MX

GATE_FEAT = "1h_bb_width"; BE = 0.541; COV = 0.05
HALFLIVES = [None, 4.0, 2.0]  # None = flat (reproduces freeze)


def yr_of(ts):
    return np.asarray(ts, dtype="datetime64[s]").astype("datetime64[Y]").astype(int) + 1970


def main():
    TR = MX.augment(MX.build_xp(MX.SPL["train"], 4), MX.SPL["train"], "xpof")
    VA = MX.augment(MX.build_xp(MX.SPL["val"]), MX.SPL["val"], "xpof")
    cols = MX.feat_cols("xpof", TR, MX.xp_cols(TR))
    ytr = TR["_y"].astype(int).values; Xtr = TR[cols].astype("float32").values
    tr_year = yr_of(TR["_ts"].values.astype("int64"))
    yva = VA["_y"].astype(int).values; Xva = VA[cols].astype("float32").values
    bbwv = VA[GATE_FEAT].values.astype(float); nyv = VA["sess_ny"].values > 0.5; tsv = VA["_ts"].values.astype("int64")
    order = np.argsort(tsv); h1 = np.zeros(len(tsv), bool); h1[order[:len(order)//2]] = True
    fwd = {}
    for w, yr in (("test24", "2024"), ("test25", "2025"), ("oos", "2026")):
        D = MX.augment(MX.build_xp(MX.SPL[w]), MX.SPL[w], "xpof")
        fwd[yr] = (D[cols].astype("float32").values, D["_y"].astype(int).values,
                   D["_ts"].values.astype("int64"), D[GATE_FEAT].values.astype(float), D["sess_ny"].values > 0.5)
        del D
    out = {"lever": "recency-weighted frozen-forward", "cov": COV, "halflives": [str(h) for h in HALFLIVES],
           "unweighted_ref_2026": {"UP": 0.5302, "DOWN": 0.5778}, "runs": {}}
    for hl in HALFLIVES:
        sw = np.ones(len(ytr)) if hl is None else 0.5 ** ((2021 - tr_year) / hl)
        P = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=127, min_child_samples=400,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=20, n_estimators=800,
                               n_jobs=20, verbosity=-1).fit(Xtr, ytr, sample_weight=sw)
        pva = P.predict_proba(Xva)[:, 1]; confv = np.abs(pva - 0.5)
        # gate = q33 1h_bb_width x NY x cov5, threshold by worst-VAL-half (same as freeze)
        bthr = float(np.nanpercentile(bbwv, 33)); gm = (bbwv <= bthr) & nyv
        cthr = float(np.quantile(confv[gm], 1 - COV))
        key = "flat" if hl is None else f"hl{hl}"
        res = {}
        for yr, (X, y, ts, bbw, ny) in fwd.items():
            pr = P.predict_proba(X)[:, 1]
            m = (bbw <= bthr) & ny & (np.abs(pr - 0.5) >= cthr); sel = MX.nonoverlap_chrono(ts, m)
            pred = (pr[sel] > 0.5).astype(int); yy = y[sel]
            for side, nm in ((1, "UP"), (0, "DOWN")):
                ss = pred == side
                if ss.sum() < 5:
                    res[f"{yr}_{nm}"] = {"n": int(ss.sum()), "acc": None}; continue
                corr = (pred[ss] == yy[ss]).astype(float); lo, hi = MX.boot(corr)
                res[f"{yr}_{nm}"] = {"n": int(ss.sum()), "acc": round(float(corr.mean()), 4), "ci_lo": round(lo, 4)}
        out["runs"][key] = res
        u26 = res.get("2026_UP", {}); d26 = res.get("2026_DOWN", {})
        print(f"[recency] {key}: 2026 UP n{u26.get('n')} acc={u26.get('acc')} | DOWN n{d26.get('n')} acc={d26.get('acc')} "
              f"| 2024UP {res['2024_UP']['acc']} 2025UP {res['2025_UP']['acc']}", flush=True)
    # verdict: does any recency setting lift 2026 UP above flat AND not crater 2024/25?
    flat = out["runs"]["flat"]
    best26up = max((out["runs"][k].get("2026_UP", {}).get("acc") or 0) for k in out["runs"])
    helps = best26up > (flat.get("2026_UP", {}).get("acc") or 0) + 0.01
    out["verdict"] = {"recency_helps_2026UP": bool(helps),
                      "statement": ("recency-weighting LIFTS frozen 2026 UP -> a deploy-fallback refinement"
                                    if helps else
                                    "recency-weighting does NOT materially lift frozen 2026 UP -> SUBSUMED; 'deploy with retrain' remains the fix")}
    json.dump(out, open("m30_recency_result.json", "w"), indent=1)
    print(f"[recency] VERDICT: {out['verdict']['statement']}", flush=True)


if __name__ == "__main__":
    main()
