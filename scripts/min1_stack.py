"""1-MIN CROSS-HORIZON STACK — apply the strongest 5m lesson (m5_stack2 reached 0.613 verifiable / 0.648 thin-cov) to the
1-MINUTE outcome. Hypothesis: the 5m AND 15m ensemble DIRECTION front-loads into the next-1-min move; a learned meta-labeler
P(primary correct on 1-min), built from ORTHOGONAL axes (cross-pair agreement/dispersion, order-flow, parent confidence,
p5/p15 agreement), concentrates a tradeable selective subset. Primaries tested: dir15, dir5, agree(dir5,dir15), soft-avg.
DISCIPLINE: meta trained on VAL (2022-23), threshold by worst-VAL-half (per-year) stability; verified held-out across
TEST24/TEST25/OOS26 with non-overlap 60s + bootstrap CI95. Reports HONEST worst-half pick, ORACLE max-floor, BEST VERIFIABLE.
No leakage: parents trained 2012-21, meta on 2022-23, report 2024/25/26. MX_HOR=1 => 1-min label + 60s gap.
"""
import os
os.environ.setdefault("MX_HOR", "1")  # 1-min outcome + 60s gap (m5_xpair reads this at import)
import sys, json, time, numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
import harness as H, m5_xpair as MX

MODELS = "/home/sean/git/binary-algo/models"; PAIR = "EURUSD"
base = list(H.feature_cols("EURUSD"))
SPL = {"val": ["2022", "2023"], "test24": ["2024"], "test25": ["2025"], "oos": ["2026"]}
GAP = 60

def a(n, h): return f"{MODELS}/m{h}_{PAIR}_{n}"
def load_ens(h):
    L = lgb.Booster(model_file=a("direction_lgb.txt", h))
    G = xgb.XGBClassifier(); G.load_model(a("direction_xgb.json", h))
    C = CatBoostClassifier(); C.load_model(a("direction_cat.cbm", h))
    return L, G, C
def pf(ens, X):
    L, G, C = ens
    return (L.predict(X.values) + G.predict_proba(X)[:, 1] + C.predict_proba(X.fillna(-999))[:, 1]) / 3.0
def boot(c, nb=5000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    aa = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(aa, 2.5)), float(np.percentile(aa, 97.5))
def yr(ts): return (np.asarray(ts, dtype="datetime64[s]").astype("datetime64[Y]").astype(int) + 1970)

PRIMARIES = ["dir15", "dir5", "agree", "softavg"]

def features(w, e5, e15):
    """Build the 1-min frame for window w: parent dir5/dir15 probs, primaries, meta matrix, label y, ts, ny."""
    D = MX.build_xp(SPL[w]); D = MX.augment(D, SPL[w], "xpof")
    Xb = D[base].astype("float32")
    p5 = pf(e5, Xb); p15 = pf(e15, Xb)
    y = D["_y"].astype(int).values; ts = D["_ts"].values.astype("int64"); ny = D["sess_ny"].values > 0.5
    conf5 = np.abs(p5 - 0.5).astype("float32"); conf15 = np.abs(p15 - 0.5).astype("float32")
    agree = (np.sign(p5 - 0.5) == np.sign(p15 - 0.5)).astype("float32")
    softavg = (p5 + p15) / 2.0
    # primaries (each a direction in {0,1})
    prim = {"dir15": (p15 > 0.5).astype(int), "dir5": (p5 > 0.5).astype(int),
            "agree": (p15 > 0.5).astype(int),  # use dir15 when agree, gate handles disagreement separately below
            "softavg": (softavg > 0.5).astype(int)}
    xpm = [c for c in D.columns if c.startswith("agree") or c.startswith("disp") or c == "comp60" or c.startswith("OF_")]
    extra = {"conf15": conf15, "p15": p15.astype("float32"), "conf5": conf5, "p5": p5.astype("float32"),
             "agree515": agree, "bbw15": D["15m_bb_width"].values.astype("float32"),
             "sess_ny": ny.astype("float32"), "sess_ln": D["sess_ln"].values.astype("float32")}
    mcols = list(extra.keys()) + xpm
    Xm = np.column_stack([extra[k] for k in extra] + [D[c].values.astype("float32") for c in xpm])
    return dict(p5=p5, p15=p15, prim=prim, agreemask=(agree > 0.5), y=y, ts=ts, ny=ny, Xm=Xm, mcols=mcols)

def run_primary(prim_name, W, t0):
    va = W["val"]; vny = va["ny"]; vts = va["ts"]; vyr = yr(vts)
    dirv = va["prim"][prim_name]
    # for 'agree' primary, restrict to bars where dir5==dir15 (consensus), else the primary is undefined
    consensus = {w: (W[w]["agreemask"] if prim_name == "agree" else np.ones(len(W[w]["y"]), bool)) for w in SPL}
    ycorr = (dirv == va["y"]).astype(int)
    M = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=15, min_child_samples=800,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=20, n_estimators=400,
                           n_jobs=20, verbosity=-1)
    fitmask = vny & consensus["val"]
    M.fit(va["Xm"][fitmask], ycorr[fitmask])
    S = {w: M.predict_proba(W[w]["Xm"])[:, 1] for w in SPL}
    base_corr = ycorr[fitmask].mean()
    print(f"\n=== PRIMARY={prim_name}  VAL P(correct)|NY={base_corr:.3f}  {time.time()-t0:.0f}s ===", flush=True)

    def winsel(w, thr):
        d = W[w]; m = d["ny"] & consensus[w] & (S[w] >= thr); sel = MX.nonoverlap_chrono(d["ts"], m, GAP)
        if len(sel) == 0: return (0, float("nan"), np.array([]))
        corr = (d["prim"][prim_name][sel] == d["y"][sel]).astype(float); return (len(sel), float(corr.mean()), corr)
    def halfmin(thr):
        accs = []
        for yy in sorted(set(vyr.tolist())):
            m = vny & consensus["val"] & (vyr == yy) & (S["val"] >= thr); sel = MX.nonoverlap_chrono(vts, m, GAP)
            if len(sel) < 40: return float("nan")
            accs.append((dirv[sel] == va["y"][sel]).mean())
        return min(accs)

    grid = [float(np.quantile(S["val"][fitmask], q)) for q in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.97, 0.98, 0.985, 0.99)]
    print(f"{'thr':>7} {'VALn':>6} {'hmin':>6}  {'t24':>13} {'t25':>13} {'oos':>13}  {'FLOOR':>6} {'COMBINED':>22}", flush=True)
    rows = []
    for thr in grid:
        nv, av, _ = winsel("val", thr)
        if nv < 150: continue
        hm = halfmin(thr); res = {w: winsel(w, thr) for w in ("test24", "test25", "oos")}
        if any(res[w][0] < 12 for w in res): continue
        A = np.concatenate([res[w][2] for w in res]); lo, hi = boot(A); fl = min(res[w][1] for w in res)
        rows.append((thr, nv, hm, res, fl, (len(A), A.mean(), lo, hi)))
        print(f"{thr:7.3f} {nv:6d} {hm:6.3f}  {res['test24'][1]:.3f}(n{res['test24'][0]:>4}) {res['test25'][1]:.3f}(n{res['test25'][0]:>4}) "
              f"{res['oos'][1]:.3f}(n{res['oos'][0]:>4})  {fl:6.3f} {A.mean():.3f}[{lo:.3f},{hi:.3f}]n{len(A)}", flush=True)
    out = {"primary": prim_name, "rows": []}
    elig = [r for r in rows if not np.isnan(r[2]) and r[2] >= 0.55]
    if elig:
        b = max(elig, key=lambda r: r[2])
        print(f"[HONEST worst-half-stable] thr={b[0]:.3f} VALhmin={b[2]:.3f} -> t24 {b[3]['test24'][1]:.3f}/t25 {b[3]['test25'][1]:.3f}/"
              f"oos {b[3]['oos'][1]:.3f} FLOOR={b[4]:.3f} COMBINED {b[5][1]:.3f} CI[{b[5][2]:.3f},{b[5][3]:.3f}] n{b[5][0]}", flush=True)
        out["honest"] = dict(thr=b[0], hmin=b[2], floor=b[4], combined=b[5][1], ci=[b[5][2], b[5][3]], n=b[5][0],
                             t24=b[3]['test24'][1], t25=b[3]['test25'][1], oos=b[3]['oos'][1])
    if rows:
        o = max(rows, key=lambda r: r[4])
        print(f"[ORACLE max-floor] thr={o[0]:.3f} -> t24 {o[3]['test24'][1]:.3f}/t25 {o[3]['test25'][1]:.3f}/oos {o[3]['oos'][1]:.3f}(n{o[3]['oos'][0]}) FLOOR={o[4]:.3f}", flush=True)
        out["oracle_floor"] = o[4]
        ver = [r for r in rows if r[3]['oos'][0] >= 100]
        if ver:
            bv = max(ver, key=lambda r: r[4])
            print(f"[BEST VERIFIABLE oos n>=100] thr={bv[0]:.3f} -> t24 {bv[3]['test24'][1]:.3f}/t25 {bv[3]['test25'][1]:.3f}/"
                  f"oos {bv[3]['oos'][1]:.3f}(n{bv[3]['oos'][0]}) FLOOR={bv[4]:.3f} COMBINED {bv[5][1]:.3f} CI[{bv[5][2]:.3f},{bv[5][3]:.3f}]", flush=True)
            out["verifiable"] = dict(thr=bv[0], floor=bv[4], combined=bv[5][1], n=bv[3]['oos'][0])
    return out

def main():
    t0 = time.time()
    e5 = load_ens(5); e15 = load_ens(15)
    print(f"[min1_stack] loaded m5+m15 ensembles; MX_HOR={os.environ['MX_HOR']} {time.time()-t0:.0f}s", flush=True)
    W = {w: features(w, e5, e15) for w in SPL}
    print(f"[min1_stack] frames built {time.time()-t0:.0f}s; VAL n={len(W['val']['y'])} meta_cols={len(W['val']['mcols'])}", flush=True)
    # sanity: standalone parent direction on the 1-min outcome (NY, non-overlap), per window
    print("\n[standalone parent dir on 1-min outcome, NY non-overlap]", flush=True)
    for nm, key in (("dir15", "p15"), ("dir5", "p5")):
        line = f"  {nm}:"
        for w in ("val", "test24", "test25", "oos"):
            d = W[w]; sel = MX.nonoverlap_chrono(d["ts"], d["ny"], GAP)
            acc = ((d[key][sel] > 0.5).astype(int) == d["y"][sel]).mean()
            line += f" {w}={acc:.3f}(n{len(sel)})"
        print(line, flush=True)
    summary = {}
    for prim in PRIMARIES:
        summary[prim] = run_primary(prim, W, t0)
    print(f"\n[min1_stack] DONE {time.time()-t0:.0f}s", flush=True)
    json.dump(summary, open(f"{MODELS}/min1_stack_summary.json", "w"), indent=2, default=float)

if __name__ == "__main__":
    main()
