"""10-MIN probe #1: does the ALREADY-TRAINED 15m ensemble's DIRECTION transfer to the 10-min outcome?
Rationale: 10-min is CLOSER to 15 than 5-min was, so dir15 should predict the 10-min sign BETTER than the
5-min sign (which it did at 0.597 standalone -> stack 0.648). Reuses models/m15_EURUSD_* (no retraining); the
10-min label comes for free from m5_xpair.build_xp with MX_HOR=10. Honest: NY & compression-NY gates, selective
by conf15, chronological non-overlap (gap=600s), per-window {test24,test25,oos} + combined, CI95 bootstrap.
"""
import os
os.environ["MX_HOR"] = "10"
import json, numpy as np
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import harness as H
import m5_xpair as MX

MODELS = "/media/sean/CORSAIR/binary-algo/models"; PAIR = "EURUSD"
base = list(H.feature_cols("EURUSD"))
SPL = {"val": ["2022", "2023"], "test24": ["2024"], "test25": ["2025"], "oos": ["2026"]}
GAP = 600  # 10-min independence window

def a15(n): return f"{MODELS}/m15_{PAIR}_{n}"
def load15():
    p = json.load(open(a15("strategy.json")))
    L = lgb.Booster(model_file=a15("direction_lgb.txt"))
    G = xgb.XGBClassifier(); G.load_model(a15("direction_xgb.json"))
    C = CatBoostClassifier(); C.load_model(a15("direction_cat.cbm"))
    return p, L, G, C
def p15f(L, G, C, X): return (L.predict(X.values) + G.predict_proba(X)[:, 1] + C.predict_proba(X.fillna(-999))[:, 1]) / 3.0
def boot(c, nb=5000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
def selacc(dirp, y, ts, gate, conf, thr, gap=GAP):
    m = gate & (conf >= thr); sel = MX.nonoverlap_chrono(ts, m, gap)
    if len(sel) == 0: return (0, float("nan"), np.array([]))
    corr = (dirp[sel] == y[sel]).astype(float)
    return (len(sel), float(corr.mean()), corr)

def main():
    p15s, L, G, C = load15(); bbw_thr = p15s["bb_width_thr"]
    print(f"[m10probe] MX.HOR={MX.HOR} GAP={GAP} 15m bb_width_thr={bbw_thr:.3e}", flush=True)
    W = {}
    for w in SPL:
        D = MX.build_xp(SPL[w]); D = MX.augment(D, SPL[w], "xpof")
        p15 = p15f(L, G, C, D[base].astype("float32"))
        W[w] = dict(p15=p15, dir15=(p15 > 0.5).astype(int), conf15=np.abs(p15 - 0.5),
                    y=D["_y"].astype(int).values, ts=D["_ts"].values.astype("int64"),
                    ny=D["sess_ny"].values > 0.5, bbw=D["15m_bb_width"].values.astype(float))
        a = roc_auc_score(W[w]["y"], p15)
        full = (W[w]["dir15"] == W[w]["y"]).mean()
        print(f"  [{w}] n={len(W[w]['y']):,} AUC(p15 vs 10m)={a:.4f} dir15 full-acc={full:.4f} NYfrac={W[w]['ny'].mean():.2f}", flush=True)

    for gname, gfn in [("NY", lambda d: d["ny"]),
                       ("compxNY", lambda d: d["ny"] & (d["bbw"] <= bbw_thr))]:
        print(f"\n===== GATE={gname} : dir15 on 10-min outcome, selective by conf15, non-overlap {GAP}s =====", flush=True)
        # pick conf thresholds from VAL gated distribution (cross-window comparable -> fixed conf)
        vg = gfn(W["val"]); vconf = W["val"]["conf15"][vg]
        thrs = [float(np.quantile(vconf, q)) for q in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95)]
        print(f"{'thr':>7} {'VALn':>6} {'VALacc':>7}  {'t24':>13} {'t25':>13} {'oos':>13}  {'FLOOR':>6} {'COMBINED':>22}", flush=True)
        for thr in thrs:
            nv, av, _ = selacc(W["val"]["dir15"], W["val"]["y"], W["val"]["ts"], vg, W["val"]["conf15"], thr)
            res = {w: selacc(W[w]["dir15"], W[w]["y"], W[w]["ts"], gfn(W[w]), W[w]["conf15"], thr) for w in ("test24", "test25", "oos")}
            A = np.concatenate([res[w][2] for w in res]) if all(len(res[w][2]) for w in res) else np.array([])
            if len(A) == 0: continue
            lo, hi = boot(A); fl = min(res[w][1] for w in res)
            print(f"{thr:7.4f} {nv:6d} {av:7.3f}  {res['test24'][1]:.3f}(n{res['test24'][0]:>4}) "
                  f"{res['test25'][1]:.3f}(n{res['test25'][0]:>4}) {res['oos'][1]:.3f}(n{res['oos'][0]:>4})  "
                  f"{fl:6.3f} {A.mean():.3f}[{lo:.3f},{hi:.3f}]n{len(A)}", flush=True)

if __name__ == "__main__":
    main()
