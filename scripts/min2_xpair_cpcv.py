"""PER-SIDE full-refit CPCV of the CROSS-PAIR primary @MX_HOR=2 (the keystone lever, escalated from the probe
which showed 2024 .6126 / 2025 .5603 selective acc with balanced up-rate but AUC ~0.51 and a coverage-starved
2026). Decides whether the 2m cross-pair USD-residual+OF signal CERTIFIES — i.e. survives per-fold gate re-tuning
across purged-combinatorial time blocks, not just the one VAL-frozen pocket.

Mirrors m15_xpair_cpcv.py (6 groups, C(6,2)=15 purged paths, purge+embargo=1 horizon, per-fold REFIT, gate tuned
on a within-fold VAL split by max-VAL-acc, test trades split by predicted side) — with TWO upgrades for 2m honesty:
  * TIES-STRICT: build keeps exact-zero 2-bar windows; a bet on a flat bar is charged a LOSS (deriv settlement).
    (The 2m tie-fraction is 3.1% — non-negligible, unlike 15m where the harness could drop them.)
  * Reports per-path AUC + test-block moved up-rate so a high-acc/low-AUC mirage or an up-rate-drift mirage shows.

Gate = comp(1m_bb_width q) [x NY] x conf-coverage. CERTIFY a side iff per-side p10 >= 0.541 AND >= ~80% paths
clear 0.541. IMPROVEMENT over the (uncertified) 2m incumbent iff p10 > 0.5445 (UP) / > breakeven (DOWN dead).
Memory-safe: per-year build, stride-2 pooled, float32, subsample fit. Single lgb n_est=800. Breakeven 0.541.
Usage: python min2_xpair_cpcv.py
"""
import os
os.environ["MX_HOR"] = "2"
import json, time, numpy as np, pandas as pd
from itertools import combinations
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import m5_xpair as MX

YEARS = list(range(2012, 2027))
N_GROUPS, K_TEST = 6, 2
SUB_FIT = 150_000
HOR = 2
GAP_S = HOR * 60
BE = 0.541
MIN_SIDE_N = 25
GATE_FEAT = "1m_bb_width"
INCUMBENT = {"UP": 0.5445, "DOWN": 0.541}   # 2m UP uncertified frozen-book CPCV p10; DOWN dead -> use breakeven


def build_keepties(years, stride):
    """Cross-pair xpof frame KEEPING exact-zero forward returns (ties), so deriv settlement can charge them.
    Returns per-year list of (X[cols], fwd_raw, ts, bb, ny). Mirrors MX.build_xp feature logic minus the !=0 drop."""
    PAIRS, NONEU, LB = MX.PAIRS, MX.NONEU, MX.LB
    ref = MX.augment(MX.build_xp([2020], stride=8), [2020], "xpof")
    cols = MX.feat_cols("xpof", ref, MX.xp_cols(ref))
    assert GATE_FEAT in ref.columns and "sess_ny" in ref.columns, "gate cols missing"
    del ref
    out = []
    for y in years:
        cl = {}; ok = True
        for p in PAIRS:
            fp = f"{MX.FEAT}/{p}_{y}.parquet"
            if not os.path.exists(fp):
                ok = False; break
            d = pd.read_parquet(fp, columns=["close"]); d = d[~d.index.duplicated(keep="last")]
            cl[p] = d["close"]
        if not ok:
            continue
        df = pd.DataFrame(cl).dropna()
        if len(df) < 100:
            continue
        idx = df.index; secs = idx.values.astype("datetime64[s]").astype("int64"); n = len(df)
        lr = {p: np.log(df[p].values) for p in PAIRS}
        feats = {}
        rets = {p: {k: np.concatenate([[np.nan] * k, lr[p][k:] - lr[p][:-k]]) for k in LB} for p in PAIRS}
        for k in LB:
            eu_r = rets["EURUSD"][k]
            basket = np.nanmean(np.vstack([MX.eu_equiv_sign(p) * rets[p][k] for p in NONEU]), axis=0)
            disp = np.nanstd(np.vstack([MX.eu_equiv_sign(p) * rets[p][k] for p in NONEU]), axis=0)
            agree = np.nanmean(np.vstack([(np.sign(MX.eu_equiv_sign(p) * rets[p][k]) == np.sign(basket)).astype(float) for p in NONEU]), axis=0)
            feats[f"eu_r{k}"] = eu_r; feats[f"usdbask{k}"] = basket; feats[f"catchup{k}"] = basket - eu_r
            feats[f"eurresid{k}"] = eu_r - basket; feats[f"disp{k}"] = disp; feats[f"agree{k}"] = agree
            for p in NONEU:
                feats[f"ll_{p}{k}"] = MX.eu_equiv_sign(p) * rets[p][k] - eu_r
        hours = idx.hour.values + idx.minute.values / 60.0
        feats["sess_ny"] = ((hours >= 13.0) & (hours < 22.0)).astype(float)
        feats["sess_ln"] = ((hours >= 7.0) & (hours < 16.0)).astype(float)
        feats["hour"] = hours
        feats["comp60"] = pd.Series(rets["EURUSD"][1]).rolling(60, min_periods=20).std().values
        fwd = np.full(n, np.nan)
        if n > HOR:
            contig = (secs[HOR:] - secs[:-HOR]) == HOR * 60
            fr = lr["EURUSD"][HOR:] - lr["EURUSD"][:-HOR]
            fwd[:n - HOR] = np.where(contig, fr, np.nan)
        F = pd.DataFrame(feats, index=idx); F["_ts"] = secs; F["_fwd"] = fwd
        F = F.loc[np.isfinite(fwd)]                      # KEEP ties (fwd==0); drop only non-contiguous/edge
        if stride > 1:
            F = F.iloc[::stride]
        B = MX._read_years(MX.FEAT, "EURUSD", [str(y)], list(__import__("harness").feature_cols("EURUSD")))
        if B is not None:
            F = F.join(B[[c for c in B.columns if c not in F.columns]], how="left")
        O = MX._read_years(MX.OFDIR, "EURUSD", [str(y)], MX.OF_COLS)
        if O is not None:
            F = F.join(O[[c for c in O.columns if c not in F.columns]], how="left")
        for c in cols:
            if c not in F.columns:
                F[c] = np.nan
        out.append((F[cols].values.astype("float32"), F["_fwd"].values.astype("float64"),
                    F["_ts"].values.astype("int64"), F[GATE_FEAT].values.astype("float32"),
                    (F["sess_ny"].values.astype("float32") > 0.5)))
        del F
    return out, cols


def build_pooled():
    parts, cols = build_keepties([str(y) for y in YEARS], stride=2)
    X = np.concatenate([p[0] for p in parts]); fwd = np.concatenate([p[1] for p in parts])
    ts = np.concatenate([p[2] for p in parts]); bb = np.concatenate([p[3] for p in parts])
    ny = np.concatenate([p[4] for p in parts])
    o = np.argsort(ts, kind="mergesort")
    return X[o], fwd[o], ts[o], bb[o], ny[o], cols


def summ(a):
    a = np.array(a, float)
    if len(a) == 0:
        return {"n_paths": 0}
    return {"n_paths": int(len(a)), "mean": round(float(a.mean()), 4), "p10": round(float(np.percentile(a, 10)), 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
            "frac_clear_BE": round(float((a >= BE).mean()), 3)}


def main():
    t0 = time.time()
    X, fwd, ts, bb, ny, cols = build_pooled()
    moved = fwd != 0.0
    yfull = (fwd > 0).astype("int8")          # label; ties (fwd==0) are class 0 but only fit on moved rows
    n = len(fwd)
    print(f"[xpcpcv2m] pooled n={n:,} feats={len(cols)} moved-up-rate={yfull[moved].mean():.4f} "
          f"tie-frac={1 - moved.mean():.4f} load {time.time()-t0:.0f}s", flush=True)
    edges = np.linspace(0, n, N_GROUPS + 1).astype(int)
    grp = [(edges[i], edges[i + 1]) for i in range(N_GROUPS)]
    embargo = HOR * 60; purge = HOR * 60 + embargo
    accs_c, accs_u, accs_d, aucs, uprates = [], [], [], [], []
    for pi, testg in enumerate(combinations(range(N_GROUPS), K_TEST)):
        te = np.zeros(n, bool)
        for gi in testg:
            te[grp[gi][0]:grp[gi][1]] = True
        te_lo = min(ts[grp[gi][0]] for gi in testg); te_hi = max(ts[grp[gi][1] - 1] for gi in testg)
        tr = ~te
        for gi in testg:
            a, b = ts[grp[gi][0]], ts[grp[gi][1] - 1]
            tr &= ~((ts >= a - purge) & (ts <= b + embargo))
        tri = np.where(tr & moved)[0]            # FIT on moved bars only (5m/15m convention)
        if len(tri) < 5000:
            continue
        cut = tri[int(len(tri) * 0.8)]; ct = ts[cut]
        fit = tri[ts[tri] < ct]; val = tri[ts[tri] >= ct]
        if len(fit) > SUB_FIT:
            fit = fit[np.linspace(0, len(fit) - 1, SUB_FIT).astype(int)]
        P = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.03, num_leaves=127,
                               min_child_samples=400, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                               reg_lambda=20, n_estimators=800, n_jobs=20, verbosity=-1).fit(X[fit], yfull[fit])
        pv = P.predict_proba(X[val])[:, 1]
        # within-fold gate tune by max-VAL-acc (ties-strict); cross-path p10 is the real guard
        best = None
        for useny in (True, False):
            nyv = ny[val] if useny else np.ones(val.shape[0], bool)
            for q in (20, 33, 100):
                bthr = np.nanpercentile(bb[fit], q)
                gmask = (bb[val] <= bthr) & nyv
                if gmask.sum() < 200:
                    continue
                for cov in (0.05, 0.10, 0.20):
                    cthr = np.quantile(np.abs(pv[gmask] - 0.5), 1 - cov); sel = gmask & (np.abs(pv - 0.5) >= cthr)
                    if sel.sum() < 100:
                        continue
                    pred = (pv[sel] > 0.5).astype(int); fv = fwd[val][sel]
                    win = ((pred == 1) & (fv > 0)) | ((pred == 0) & (fv < 0))   # ties-strict
                    acc = win.mean()
                    if best is None or acc > best[0]:
                        best = (acc, bthr, cthr, q, cov, useny)
        if best is None:
            continue
        _, bthr, cthr, q, cov, useny = best
        pt = P.predict_proba(X[te])[:, 1]; tst = ts[te]; fte = fwd[te]
        nyt = ny[te] if useny else np.ones(te.sum(), bool)
        gt = (bb[te] <= bthr) & nyt & (np.abs(pt - 0.5) >= cthr)
        sel = MX.nonoverlap_chrono(tst, gt, GAP_S)
        if len(sel) < 10:
            continue
        pred = (pt[sel] > 0.5).astype(int); fv = fte[sel]
        win = (((pred == 1) & (fv > 0)) | ((pred == 0) & (fv < 0))).astype(float)   # ties-strict (flat bet = loss)
        accs_c.append(win.mean())
        # diagnostics: AUC on moved test bars + moved up-rate (mirage checks)
        tem = te.copy(); mv_te = fte != 0
        try:
            auc = roc_auc_score((fte[mv_te] > 0).astype(int), pt[mv_te]); aucs.append(auc)
        except Exception:
            auc = float("nan")
        upr = float((fv[fv != 0] > 0).mean()) if (fv != 0).any() else float("nan"); uprates.append(upr)
        su = pred == 1; sd = pred == 0
        au = ad = None
        if su.sum() >= MIN_SIDE_N:
            au = float(win[su].mean()); accs_u.append(au)
        if sd.sum() >= MIN_SIDE_N:
            ad = float(win[sd].mean()); accs_d.append(ad)
        yl = int(str(np.datetime64(int(te_lo), "s"))[:4]); yh = int(str(np.datetime64(int(te_hi), "s"))[:4])
        print(f"  path{pi:>2} era={yl}-{yh} ny{int(useny)} q{q} cov{cov:.0%} n={len(sel)} comb={win.mean():.4f} "
              f"AUC={auc:.4f} upr={upr:.3f} UP={('%.4f'%au) if au else '  -  '}(n{int(su.sum())}) "
              f"DOWN={('%.4f'%ad) if ad else '  -  '}(n{int(sd.sum())}) {time.time()-t0:.0f}s", flush=True)
    res = {"model": "cross-pair xpof primary @MX_HOR=2, TIES-STRICT refit-CPCV", "breakeven": BE,
           "combined": summ(accs_c), "UP": summ(accs_u), "DOWN": summ(accs_d),
           "moved_auc": {"mean": round(float(np.nanmean(aucs)), 4) if aucs else None,
                         "min": round(float(np.nanmin(aucs)), 4) if aucs else None},
           "test_block_up_rate": {"mean": round(float(np.nanmean(uprates)), 4) if uprates else None,
                                  "min": round(float(np.nanmin(uprates)), 4) if uprates else None,
                                  "max": round(float(np.nanmax(uprates)), 4) if uprates else None},
           "incumbent_floor": INCUMBENT,
           "CERT_RULE": "side CERTIFIED iff p10>=.541 & frac_clear>=.80; IMPROVES incumbent iff p10>incumbent_floor"}
    for s in ("combined", "UP", "DOWN"):
        r = res[s]; tag = ""
        if s in ("UP", "DOWN") and r.get("n_paths", 0) >= 5:
            cert = r["p10"] >= BE and r["frac_clear_BE"] >= 0.80
            impr = r["p10"] > INCUMBENT[s]
            tag = f" ==> {'CERTIFIED' if cert else 'NOT certified'}{' + IMPROVES incumbent' if (cert and impr) else ''}"
        print(f"\n[{s}] {r}{tag}", flush=True)
    print(f"[moved_auc] {res['moved_auc']}  [test_block_up_rate] {res['test_block_up_rate']}", flush=True)
    json.dump(res, open("min2_xpair_cpcv_result.json", "w"), indent=1)
    print(f"\n[xpcpcv2m] DONE {time.time()-t0:.0f}s -> min2_xpair_cpcv_result.json", flush=True)


if __name__ == "__main__":
    main()
