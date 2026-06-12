"""IMPROVE battery @15m (round 2): test whether model-changing levers beat the certified frozen-book floor
(UP refit-p10 .5475 / DOWN .5486). Fair, fast, matched comparison — lgb-only architecture for ALL variants
(a winner is escalated to full 3-model + CPCV before any freeze):
  V0  raw-binary lgb                         (control / baseline at this fidelity)
  V1  3-class up/flat/down deadband relabel  (N16; gamma=tertile; trade sign(p_up-p_dn) by margin) — 15m trend hypo
  V2  seed-ensemble (K=3 lgb seeds, avg)     (I2; variance reduction for the thin confidence-driven edge)

Same comp(bb_width q33)xNY regime, same 15m settlement (_y=ret>0 ties-excluded, nonoverlap_chrono 900s).
Confidence threshold selected on VAL at FIXED coverage {0.05,0.10} (adequate n; avoids the thin-cov2% mirage
that made the magweight control noise-dominated). Forward side-split per year vs the control + frozen-forward.
A variant must beat the control on the binding-2025 side AND not collapse 2026, up-rate in [.47,.53]. Breakeven .541.

Usage: python m15_improve.py
"""
import json, time, os, numpy as np, pandas as pd
import lightgbm as lgb
import harness as H
import m15_production as M15

HOR = 15
base = list(H.feature_cols("EURUSD"))
SUB_FIT = 150_000
BE = 0.541
COVS = (0.05, 0.10)


def load_fwd(years, stride=1):
    parts = []
    for y in years:
        p = f"{H.FEAT_DIR}/EURUSD_{y}.parquet"
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p, columns=base + H.META_COLS); df = df[~df.index.duplicated(keep="last")]
        c = df["close"].values; n = len(c)
        secs = df.index.values.astype("datetime64[s]").astype("int64")
        contig = np.zeros(n, bool)
        if n > HOR:
            contig[:n - HOR] = (secs[HOR:] - secs[:-HOR]) == HOR * 60
        fwd = np.full(n, np.nan); fwd[:n - HOR] = c[HOR:]; ret = fwd / c - 1.0
        valid = contig & np.isfinite(ret) & (ret != 0)
        d = df.loc[valid, base].copy()
        d["_y"] = (ret[valid] > 0).astype(int); d["_ret"] = ret[valid]; d["_ts"] = secs[valid]
        parts.append(d.iloc[::stride] if stride > 1 else d)
    return pd.concat(parts)


def mk_lgb(seed=7, objective="binary", num_class=None):
    kw = dict(learning_rate=0.03, num_leaves=255, min_child_samples=200, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.5, reg_lambda=10, n_estimators=600, n_jobs=20, verbosity=-1, random_state=seed)
    if objective == "binary":
        return lgb.LGBMClassifier(objective="binary", metric="auc", **kw)
    return lgb.LGBMClassifier(objective="multiclass", num_class=num_class, **kw)


def side_split(conf_dir_p, y, ts, bbw, ny, bthr, cthr):
    """conf_dir_p: P(up)-like score in (0,1) where >0.5 = UP bet, confidence=|p-0.5|."""
    m = (bbw <= bthr) & ny & (np.abs(conf_dir_p - 0.5) >= cthr)
    sel = M15.nonoverlap_chrono(ts, m)
    pred = (conf_dir_p[sel] > 0.5).astype(int); yy = y[sel]
    out = {}
    for side, nm in ((1, "UP"), (0, "DOWN"), (None, "COMBINED")):
        ss = (pred == side) if side is not None else np.ones(len(pred), bool)
        if ss.sum() < 5:
            out[nm] = {"n": int(ss.sum()), "acc": None}; continue
        corr = (pred[ss] == yy[ss]).astype(float); lo, hi = M15.boot(corr)
        out[nm] = {"n": int(ss.sum()), "acc": round(float(corr.mean()), 4), "ci": [round(lo, 4), round(hi, 4)],
                   "up_rate": round(float(yy[ss].mean()), 4)}
    return out


def main():
    t0 = time.time()
    TR = load_fwd(H.SPLITS["train"], stride=3)
    VA = load_fwd(H.SPLITS["val"])
    if len(TR) > SUB_FIT:
        TR = TR.iloc[np.linspace(0, len(TR) - 1, SUB_FIT).astype(int)]
    Xtr = TR[base].astype("float32").values; ytr = TR["_y"].values; rtr = TR["_ret"].values
    Xva = VA[base].astype("float32").values; yva = VA["_y"].values
    bbwv = VA["15m_bb_width"].values.astype(float); nyv = VA["sess_ny"].values.astype(float) > 0.5
    bthr = float(np.nanpercentile(TR["15m_bb_width"].values.astype(float), 33))
    print(f"[improve] train={len(TR):,} val={len(VA):,} bthr(q33)={bthr:.2e} {time.time()-t0:.0f}s", flush=True)

    # --- fit the three variants ---
    variants = {}
    # V0 raw binary
    L0 = mk_lgb(7, "binary").fit(Xtr, ytr); variants["V0_raw"] = L0.predict_proba(Xva)[:, 1]
    print(f"[improve] V0 raw fit {time.time()-t0:.0f}s", flush=True)
    # V1 3-class deadband: gamma = mean(|Q33|,|Q66|) of |ret| on train
    a = np.abs(rtr); g = float((np.quantile(a, 0.33) + np.quantile(a, 0.66)) / 2.0)
    cls = np.where(rtr > g, 2, np.where(rtr < -g, 0, 1)).astype(int)  # 0=down,1=flat,2=up
    L1 = mk_lgb(7, "multi", num_class=3).fit(Xtr, cls)
    pm = L1.predict_proba(Xva); variants["V1_3class"] = pm[:, 2] - pm[:, 0] + 0.5  # P(up)-P(down) recentred to ~0.5
    print(f"[improve] V1 3class fit (gamma={g:.2e}, flat-frac={np.mean(cls==1):.2f}) {time.time()-t0:.0f}s", flush=True)
    # V2 seed-ensemble K=3
    ps = np.mean([mk_lgb(s, "binary").fit(Xtr, ytr).predict_proba(Xva)[:, 1] for s in (7, 17, 29)], axis=0)
    variants["V2_seedens"] = ps
    print(f"[improve] V2 seedens fit {time.time()-t0:.0f}s", flush=True)

    # --- per-variant: choose conf thr on VAL at fixed cov, forward side-split each year ---
    test = {yr: load_fwd([yr]) for yr in ("2024", "2025", "2026")}
    res = {"frozen_forward_ref": {"UP": "2024 .725/2025 .607/2026 .674", "DOWN": ".653/.559/.650"},
           "certified_floor": {"UP_refit_p10": 0.5475, "DOWN_refit_p10": 0.5486}, "breakeven": BE, "variants": {}}
    for vn, pva in variants.items():
        vres = {}
        gmask = (bbwv <= bthr) & nyv; confv = np.abs(pva - 0.5)
        for cov in COVS:
            cthr = float(np.quantile(confv[gmask], 1 - cov))
            # recompute each test year's score with the SAME variant model
            yrres = {}
            for yr in ("2024", "2025", "2026"):
                D = test[yr]; X = D[base].astype("float32").values
                if vn == "V0_raw":
                    p = L0.predict_proba(X)[:, 1]
                elif vn == "V1_3class":
                    pm2 = L1.predict_proba(X); p = pm2[:, 2] - pm2[:, 0] + 0.5
                else:
                    p = np.mean([m.predict_proba(X)[:, 1] for m in (Ls7, Ls17, Ls29)], axis=0)
                y = D["_y"].values; ts = D["_ts"].values
                bbw = D["15m_bb_width"].values.astype(float); ny = D["sess_ny"].values.astype(float) > 0.5
                yrres[yr] = side_split(p, y, ts, bbw, ny, bthr, cthr)
            vres[f"cov{cov}"] = yrres
        res["variants"][vn] = vres
    json.dump(res, open("m15_improve_result.json", "w"), indent=1)

    # --- print binding-2025 + 2024/2026 per side at cov0.10 ---
    print("\n=== forward side-split @cov0.10 (binding year = 2025) ===", flush=True)
    print(f"{'variant':12} | {'2024 UP/DN':16} | {'2025 UP/DN(bind)':18} | {'2026 UP/DN':16}", flush=True)
    for vn in variants:
        r = res["variants"][vn]["cov0.1"]
        def cell(yr):
            u = r[yr]["UP"]; d = r[yr]["DOWN"]
            us = f"{u['acc']:.3f}" if u.get("acc") else " NA "
            ds = f"{d['acc']:.3f}" if d.get("acc") else " NA "
            return f"{us}/{ds}(n{u.get('n',0)}/{d.get('n',0)})"
        print(f"{vn:12} | {cell('2024'):16} | {cell('2025'):18} | {cell('2026'):16}", flush=True)
    print(f"\n[improve] frozen-forward UP .725/.607/.674 DOWN .653/.559/.650 | certified floors UP .5475 DOWN .5486", flush=True)
    print(f"[improve] -> m15_improve_result.json ({time.time()-t0:.0f}s)", flush=True)


# seed models need module scope for the test-year recompute closure
Ls7 = Ls17 = Ls29 = None
if __name__ == "__main__":
    # rebind seed models so the V2 closure can re-score test years
    import m15_improve as _self

    _orig = main

    def main2():
        global Ls7, Ls17, Ls29
        t0 = time.time()
        TR = load_fwd(H.SPLITS["train"], stride=3); VA = load_fwd(H.SPLITS["val"])
        if len(TR) > SUB_FIT:
            TR = TR.iloc[np.linspace(0, len(TR) - 1, SUB_FIT).astype(int)]
        Xtr = TR[base].astype("float32").values; ytr = TR["_y"].values; rtr = TR["_ret"].values
        Xva = VA[base].astype("float32").values
        bbwv = VA["15m_bb_width"].values.astype(float); nyv = VA["sess_ny"].values.astype(float) > 0.5
        bthr = float(np.nanpercentile(TR["15m_bb_width"].values.astype(float), 33))
        print(f"[improve] train={len(TR):,} val={len(VA):,} bthr(q33)={bthr:.2e} {time.time()-t0:.0f}s", flush=True)
        L0 = mk_lgb(7, "binary").fit(Xtr, ytr)
        a = np.abs(rtr); g = float((np.quantile(a, 0.33) + np.quantile(a, 0.66)) / 2.0)
        cls = np.where(rtr > g, 2, np.where(rtr < -g, 0, 1)).astype(int)
        L1 = mk_lgb(7, "multi", num_class=3).fit(Xtr, cls)
        print(f"[improve] V1 gamma={g:.2e} flat-frac={np.mean(cls==1):.2f}", flush=True)
        Ls7 = mk_lgb(7, "binary").fit(Xtr, ytr); Ls17 = mk_lgb(17, "binary").fit(Xtr, ytr); Ls29 = mk_lgb(29, "binary").fit(Xtr, ytr)
        print(f"[improve] all fit {time.time()-t0:.0f}s", flush=True)
        scorers = {
            "V0_raw": lambda X: L0.predict_proba(X)[:, 1],
            "V1_3class": lambda X: (lambda pm: pm[:, 2] - pm[:, 0] + 0.5)(L1.predict_proba(X)),
            "V2_seedens": lambda X: np.mean([m.predict_proba(X)[:, 1] for m in (Ls7, Ls17, Ls29)], axis=0),
        }
        test = {yr: load_fwd([yr]) for yr in ("2024", "2025", "2026")}
        res = {"certified_floor": {"UP_refit_p10": 0.5475, "DOWN_refit_p10": 0.5486},
               "frozen_forward_ref": {"UP": ".725/.607/.674", "DOWN": ".653/.559/.650"}, "breakeven": BE, "variants": {}}
        print("\n=== forward side-split @cov0.05 & 0.10 (binding=2025) ===", flush=True)
        for vn, sc in scorers.items():
            pva = sc(Xva); gmask = (bbwv <= bthr) & nyv; confv = np.abs(pva - 0.5); vres = {}
            for cov in COVS:
                cthr = float(np.quantile(confv[gmask], 1 - cov)); yrres = {}
                for yr in ("2024", "2025", "2026"):
                    D = test[yr]; p = sc(D[base].astype("float32").values)
                    yrres[yr] = side_split(p, D["_y"].values, D["_ts"].values,
                                           D["15m_bb_width"].values.astype(float), D["sess_ny"].values.astype(float) > 0.5, bthr, cthr)
                vres[f"cov{cov}"] = yrres
            res["variants"][vn] = vres
            for cov in COVS:
                r = vres[f"cov{cov}"]
                def c(yr, s):
                    v = r[yr][s]; return f"{v['acc']:.3f}(n{v['n']})" if v.get("acc") else "NA"
                print(f"  {vn:11} cov{cov}: 2024 UP {c('2024','UP')} DN {c('2024','DOWN')} | "
                      f"2025 UP {c('2025','UP')} DN {c('2025','DOWN')} | 2026 UP {c('2026','UP')} DN {c('2026','DOWN')}", flush=True)
        json.dump(res, open("m15_improve_result.json", "w"), indent=1)
        print(f"\n[improve] frozen-fwd UP .725/.607/.674 DOWN .653/.559/.650 | floors UP .5475 DOWN .5486", flush=True)
        print(f"[improve] -> m15_improve_result.json ({time.time()-t0:.0f}s)", flush=True)

    main2()
