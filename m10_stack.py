"""10-MIN cross-horizon stack + meta-labeler (the method that lifted the 5-min book to 0.613/0.648).

Targets the 10-MIN outcome (MX_HOR=10). Three primaries, each gated by a learned META-LABELER that predicts
P(primary correct on 10-min) from ORTHOGONAL axes (cross-pair agreement/dispersion, order-flow, confidences,
session/compression). Threshold chosen by WORST-VAL-YEAR-HALF stability (NOT VAL-acc-max → avoids the corr=-0.54 trap).
Reported held-out vs the 10-min label across {test24,test25,oos} + combined, CI95, chronological non-overlap 600s.

  primary=dir15  : the trained 15m ensemble direction (reuses models/m15_EURUSD_*)
  primary=dir10  : the trained native-10 ensemble direction (models/m10_EURUSD_*, if present)
  primary=agree  : trade only when dir15==dir10 (consensus), meta on the agreeing subset

No leakage: 15m & 10m primaries trained 2012-21, meta trained on VAL (2022-23), report 2024/25/26.
Usage: python m10_stack.py [dir15|dir10|agree]
"""
import os
os.environ["MX_HOR"] = "10"
import sys, json, time, numpy as np
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
import harness as H, m5_xpair as MX
MODELS = "/home/sean/git/binary-algo/models"; PAIR = "EURUSD"
base = list(H.feature_cols("EURUSD"))
SPL = {"val": ["2022", "2023"], "test24": ["2024"], "test25": ["2025"], "oos": ["2026"]}
GAP = 600

def _a(pfx, n): return f"{MODELS}/{pfx}_{PAIR}_{n}"
def load_ens(pfx):
    """Load a {pfx}_EURUSD_direction_{lgb,xgb,cat} ensemble; return (lgb,xgb,cat) or None if absent."""
    lp = _a(pfx, "direction_lgb.txt")
    if not os.path.exists(lp): return None
    L = lgb.Booster(model_file=lp)
    G = xgb.XGBClassifier(); G.load_model(_a(pfx, "direction_xgb.json"))
    C = CatBoostClassifier(); C.load_model(_a(pfx, "direction_cat.cbm"))
    return L, G, C
def pens(ens, X): L, G, C = ens; return (L.predict(X.values) + G.predict_proba(X)[:, 1] + C.predict_proba(X.fillna(-999))[:, 1]) / 3.0
def boot(c, nb=4000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
def yr(ts): return (np.asarray(ts, dtype="datetime64[s]").astype("datetime64[Y]").astype(int) + 1970)

def features(w, e15, e10):
    D = MX.build_xp(SPL[w]); D = MX.augment(D, SPL[w], "xpof")
    Xb = D[base].astype("float32")
    p15 = pens(e15, Xb); p10 = pens(e10, Xb) if e10 is not None else p15
    y = D["_y"].astype(int).values; ts = D["_ts"].values.astype("int64"); ny = D["sess_ny"].values > 0.5
    conf15 = np.abs(p15 - 0.5).astype("float32"); conf10 = np.abs(p10 - 0.5).astype("float32")
    agree = (np.sign(p15 - 0.5) == np.sign(p10 - 0.5)).astype("float32")
    xpm = [c for c in D.columns if c.startswith("agree") or c.startswith("disp") or c == "comp60" or c.startswith("OF_")]
    extra = {"conf15": conf15, "p15": p15.astype("float32"), "conf10": conf10, "p10": p10.astype("float32"),
             "agree1510": agree, "bbw15": D["15m_bb_width"].values.astype("float32"),
             "sess_ny": ny.astype("float32"), "sess_ln": D["sess_ln"].values.astype("float32")}
    mcols = list(extra.keys()) + xpm
    Xm = np.column_stack([extra[k] for k in extra] + [D[c].values.astype("float32") for c in xpm])
    return dict(p15=p15, p10=p10, dir15=(p15 > 0.5).astype(int), dir10=(p10 > 0.5).astype(int),
                agree=agree.astype(bool), y=y, ts=ts, ny=ny, Xm=Xm, mcols=mcols, bbw=D["15m_bb_width"].values.astype(float))

def main(primary="dir15"):
    t0 = time.time(); e15 = load_ens("m15"); e10 = load_ens("m10")
    if e15 is None: print("[m10stack] FATAL: m15 ensemble missing"); return
    if e10 is None and primary in ("dir10", "agree"):
        print(f"[m10stack] native-10 ensemble (m10_*) not found — required for primary={primary}. Train m10_production.py first."); return
    print(f"[m10stack] primary={primary} | e10={'yes' if e10 else 'no'} | target=10-min outcome; meta=P(primary correct); {time.time()-t0:.0f}s", flush=True)
    W = {w: features(w, e15, e10) for w in SPL}
    print(f"[m10stack] features built {time.time()-t0:.0f}s; meta cols={len(W['val']['mcols'])}", flush=True)
    va = W["val"]; vny = va["ny"]
    dirkey = {"dir15": "dir15", "dir10": "dir10", "agree": "dir15"}[primary]
    def base_mask(d):  # which rows are eligible before meta/conf selection
        return d["ny"] & d["agree"] if primary == "agree" else d["ny"]
    vmask0 = base_mask(va)
    ycorr = (va[dirkey] == va["y"]).astype(int)
    M = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=15, min_child_samples=800,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=20, n_estimators=400, n_jobs=20, verbosity=-1)
    M.fit(va["Xm"][vmask0], ycorr[vmask0])
    S = {w: M.predict_proba(W[w]["Xm"])[:, 1] for w in SPL}
    print(f"[m10stack] meta trained on n={int(vmask0.sum())}; VAL primary-correct rate={ycorr[vmask0].mean():.3f}", flush=True)
    vts = va["ts"]; vyr = yr(vts)
    def winsel(w, thr):
        d = W[w]; m = base_mask(d) & (S[w] >= thr); sel = MX.nonoverlap_chrono(d["ts"], m, GAP)
        if len(sel) == 0: return (0, float("nan"), np.array([]))
        corr = (d[dirkey][sel] == d["y"][sel]).astype(float); return (len(sel), float(corr.mean()), corr)
    def halfmin(thr):
        accs = []
        for yy in sorted(set(vyr.tolist())):
            m = vmask0 & (vyr == yy) & (S["val"] >= thr); sel = MX.nonoverlap_chrono(vts, m, GAP)
            if len(sel) < 30: return float("nan")
            accs.append((va[dirkey][sel] == va["y"][sel]).mean())
        return min(accs)
    grid = [float(np.quantile(S["val"][vmask0], q)) for q in (0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.97, 0.98)]
    print(f"\n{'thr':>7} {'VALn':>6} {'hmin':>6}  {'t24':>13} {'t25':>13} {'oos':>13}  {'FLOOR':>6} {'COMB':>22}", flush=True)
    rows = []
    for thr in grid:
        nv, av, _ = winsel("val", thr)
        if nv < 150: continue
        hm = halfmin(thr); res = {w: winsel(w, thr) for w in ("test24", "test25", "oos")}
        if any(res[w][0] < 10 for w in res): continue
        A = np.concatenate([res[w][2] for w in res]); lo, hi = boot(A); fl = min(res[w][1] for w in res)
        rows.append((thr, nv, hm, res, fl, (len(A), A.mean(), lo, hi)))
        print(f"{thr:7.3f} {nv:6d} {hm:6.3f}  {res['test24'][1]:.3f}(n{res['test24'][0]:>4}) {res['test25'][1]:.3f}(n{res['test25'][0]:>4}) "
              f"{res['oos'][1]:.3f}(n{res['oos'][0]:>4})  {fl:6.3f} {A.mean():.3f}[{lo:.3f},{hi:.3f}]n{len(A)}", flush=True)
    elig = [r for r in rows if not np.isnan(r[2]) and r[2] >= 0.56]
    if elig:
        b = max(elig, key=lambda r: r[2])
        print(f"\n[HONEST worst-half-stable] thr={b[0]:.3f} VALhmin={b[2]:.3f} -> t24 {b[3]['test24'][1]:.3f}/t25 {b[3]['test25'][1]:.3f}/"
              f"oos {b[3]['oos'][1]:.3f} FLOOR={b[4]:.3f} COMBINED {b[5][1]:.3f} CI[{b[5][2]:.3f},{b[5][3]:.3f}] n{b[5][0]}", flush=True)
    if rows:
        o = max(rows, key=lambda r: r[4])
        print(f"[ORACLE max-floor] thr={o[0]:.3f} -> t24 {o[3]['test24'][1]:.3f}/t25 {o[3]['test25'][1]:.3f}/oos {o[3]['oos'][1]:.3f}(n{o[3]['oos'][0]}) FLOOR={o[4]:.3f}", flush=True)
        ver = [r for r in rows if r[3]['oos'][0] >= 80]
        if ver:
            bv = max(ver, key=lambda r: r[4])
            print(f"[BEST VERIFIABLE oos n>=80] thr={bv[0]:.3f} -> t24 {bv[3]['test24'][1]:.3f}/t25 {bv[3]['test25'][1]:.3f}/"
                  f"oos {bv[3]['oos'][1]:.3f}(n{bv[3]['oos'][0]}) FLOOR={bv[4]:.3f} COMBINED {bv[5][1]:.3f} CI[{bv[5][2]:.3f},{bv[5][3]:.3f}]", flush=True)
    print(f"[m10stack] DONE {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dir15")
