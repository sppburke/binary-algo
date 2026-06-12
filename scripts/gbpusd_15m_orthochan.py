"""GBPUSD 15m NY — isolate the GBP-specific ORTHOGONAL own-pair channels (famonly confirm-or-kill).

SCOPE: GBPUSD · 15m · NY. Fork of audusd_15m_orthochan.py retargeted to PAIR/TARGET=GBPUSD.
The xpair screen (gbpusd_15m_xpair.py) put cross-pair + EURGBP-RV + euro-bloc + risk-bloc features
into one combined model. This runs a FAMONLY isolation test of four mechanistically-distinct GBP-
specific channels (the base 239 contain NO cross-pair/EURGBP info, so these are genuinely orthogonal
candidates):

  1. EURGBP synthetic cross block (family "eurgbp"):
       eurgbp_r{k} = gbp_r{k} - eur_r{k}   (both USD-quote raw log-returns; USD leg algebraically
       cancelled → immune to USD-factor sign inversion; RBA→RBA-RBNZ analogue = BoE-vs-ECB mean
       reversion channel). eurgbp_dev{60,240} = rolling deviation of log(EURGBP) from its rolling-
       mean anchor (same windows/min_periods convention as AUDNZD dev in the AUDUSD template).
       NOTE: market EURGBP is NOT on disk. This synthetic cross's net-new content beyond the xpair
       pool's linear span is the nonlinear rolling-window statistics (mean-reversion deviation at
       two horizon windows) — the linear return eurgbp_r{k} = gbp_r{k} - eur_r{k} is a linear
       combination of panel members but the rolling-deviation nonlinearity is not spanned.

  2. Commodity-vs-safe-haven RISK factor (family "risk"):
       COMMOD=["AUDUSD","NZDUSD","USDCAD"], SAFEHAVEN=["USDJPY","USDCHF"]; risk{k}=commod-haven.
       GBP loads mildly risk-on. Augmented with:
         eurobloc{k} = mean(eur_r{k}, -usdchf_r{k})  (European-bloc USD-weakness factor: EUR+CHF)
         gbpbloc_resid{k} = gbp_r{k} - eurobloc{k}   (GBP beyond the European bloc; UK-idiosyncratic)
       All three sub-blocks folded into family "risk" by column prefix.

  3. Signed realized-semivariance asymmetry (family "semi", backlog #11, PAIR-generic):
       rsasym{k} = (RS+ - RS-) / (RS+ + RS+),  k in {15,30,60},  from GBPUSD 1m returns.

  4. Cross-sectional carry-rank (family "carryrank", discovery N50):
       At each 1m bar, rank GBPUSD's trailing k-bar USD-weakness-signed return among all 7 majors:
         equiv_sign(p)*r_p  (equiv_sign = -1 for USDJPY/USDCHF/USDCAD, else +1).
       Feature = rank/7 - 0.5 (zero-centred, range [-0.5, +0.5]).
       Also: sign_agree{k} = fraction of top-3 ranked majors (by USD-weakness) whose sign agrees
       with gbp_r{k} — measures GBP co-leadership in the USD-weakness bloc at lag k.
       Pure panel features, no external data.

NY-restricted, single-fit, VAL worst-half gate. Reports each block ALONE (famonly), base-only, and
base+block per-year cov2% per side. A channel ADDS only if base+block VAL-AUC > base + 0.005 AND
2026 cov2% COMB > base 2026 cov2 (same famonly ΔVAL-AUC screen as the AUDUSD template).
Falsifier pre-registered in result JSON.

Incumbent context (from gbpusd_15m_cpcv_session_ny_multicov_result.json — reference only; screen
metric stays famonly ΔVAL-AUC):
  NY incumbent p10 cov2% UP .6039 / DOWN .5937
  (both sides CERTIFIED; edge is NY-concentrated)

Lineage: fork of audusd_15m_orthochan.py (AUDUSD 15m NY) with:
  - AUDNZD residual-difference → EURGBP synthetic cross (family "eurgbp")
  - COMMOD/SAFEHAVEN adapted to GBP context; EUR-bloc factor added (family "risk")
  - rsasym retargeted to GBPUSD 1m returns (family "semi")
  - NEW carryrank family (N50 discovery)

Usage: ~/binary-algo-venv/bin/python gbpusd_15m_orthochan.py
"""
import os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from sessions import session_mask
from gbpusd_15m_xpair import build_xp_gbp, augment, SPL, BE
from gbpusd_15m_base import side_eval, boot

SESSION = "ny"; RESULT = "gbpusd_15m_orthochan_result.json"
FEAT = "/home/sean/git/binary-algo/features"
PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"]
USD_BASE = {"USDJPY", "USDCHF", "USDCAD"}
TARGET = "GBPUSD"
LB = [1, 3, 5, 10, 15, 30]    # must match gbpusd_15m_xpair.LB

def equiv_sign(p): return -1.0 if p in USD_BASE else +1.0


# ---------------------------------------------------------------------------
# Semivariance block (family "semi") — GBPUSD 1m returns
# ---------------------------------------------------------------------------
def add_semivar(F, years):
    """Signed realized-semivariance asymmetry from GBPUSD 1m returns: rsasym_k=(RS+-RS-)/(RS++RS-)."""
    parts = []
    for y in years:
        p = f"{FEAT}/GBPUSD_{y}.parquet"
        if not os.path.exists(p): continue
        d = pd.read_parquet(p, columns=["close"]); d = d[~d.index.duplicated(keep="last")]
        parts.append(d)
    if not parts: return F
    F = F[~F.index.duplicated(keep="last")]
    c = pd.concat(parts); c = c[~c.index.duplicated(keep="last")]
    lr = np.log(c["close"].values); r1 = np.concatenate([[np.nan], np.diff(lr)])
    s = pd.Series(r1, index=c.index)
    feats = {}
    for k in (15, 30, 60):
        rp = (s.clip(lower=0) ** 2).rolling(k, min_periods=k // 2).sum()
        rm = (s.clip(upper=0) ** 2).rolling(k, min_periods=k // 2).sum()
        feats[f"rsasym{k}"] = ((rp - rm) / (rp + rm + 1e-12)).values
    SV = pd.DataFrame(feats, index=c.index)
    return F.join(SV[[col for col in SV.columns if col not in F.columns]], how="left")


# ---------------------------------------------------------------------------
# Carry-rank block (family "carryrank") — cross-sectional panel, N50 discovery
# ---------------------------------------------------------------------------
def add_carryrank(F, years):
    """Cross-sectional carry-rank of GBPUSD among 7 majors at each bar.

    For each lookback k in LB:
      carryrank{k}  = rank_of(equiv_sign(GBP)*gbp_r{k}) among all 7 / 7 − 0.5
      sign_agree{k} = fraction of top-3 USD-weakness ranked majors whose
                      equiv_sign-adjusted sign agrees with gbp_r{k}
    Pure panel features; no external data required.
    """
    parts = {p: [] for p in PAIRS}
    present_years = []
    for y in years:
        ok = True
        for p in PAIRS:
            fp = f"{FEAT}/{p}_{y}.parquet"
            if not os.path.exists(fp): ok = False; break
        if not ok: continue
        present_years.append(y)
        for p in PAIRS:
            fp = f"{FEAT}/{p}_{y}.parquet"
            d = pd.read_parquet(fp, columns=["close"]); d = d[~d.index.duplicated(keep="last")]
            parts[p].append(d)
    if not present_years: return F
    F = F[~F.index.duplicated(keep="last")]
    cl = {}
    for p in PAIRS:
        s = pd.concat(parts[p])["close"]
        cl[p] = s[~s.index.duplicated(keep="last")]
    df = pd.DataFrame(cl).dropna()
    idx = df.index
    lr = {p: np.log(df[p].values) for p in PAIRS}
    n = len(df)
    feats = {}
    for k in LB:
        # USD-weakness equiv returns for each pair, shape (7, n)
        uw = np.vstack([
            np.concatenate([[np.nan]*k, equiv_sign(p) * (lr[p][k:] - lr[p][:-k])])
            for p in PAIRS
        ])   # (7, n)
        gbp_idx = PAIRS.index("GBPUSD")
        # rank GBPUSD among all 7 (0-based → convert to 0-centred)
        # scipy.stats.rankdata per column is expensive; do it vectorised
        ranks = np.sum(uw < uw[gbp_idx, :], axis=0).astype(float)   # 0 = lowest, 6 = highest
        feats[f"carryrank{k}"] = ranks / 7.0 - 0.5                  # centred, range [-0.5, +0.5]
        # sign_agree: fraction of top-3 USD-weakness majors (excl GBP) that agree with GBP direction
        gbp_sign = np.sign(uw[gbp_idx, :])
        # top-3 among the 6 non-GBP pairs
        nongbp_uw = np.delete(uw, gbp_idx, axis=0)    # (6, n)
        # sort descending per column; top3 rows
        sorted_idx = np.argsort(-nongbp_uw, axis=0)   # (6, n) descending ranks
        agree = np.zeros(n)
        for rank_pos in range(3):
            row_sel = sorted_idx[rank_pos, :]          # shape (n,)
            top3_val = nongbp_uw[row_sel, np.arange(n)]
            agree += (np.sign(top3_val) == gbp_sign).astype(float)
        feats[f"sign_agree{k}"] = agree / 3.0
    CR = pd.DataFrame(feats, index=idx)
    return F.join(CR[[col for col in CR.columns if col not in F.columns]], how="left")


# ---------------------------------------------------------------------------
# Feature-family selector
# ---------------------------------------------------------------------------
def cols_of(df, kind):
    xp = [c for c in df.columns if c not in ("_y", "_ts", "_fwd", "hour")]
    base = [c for c in H.feature_cols("GBPUSD") if c in df.columns]
    eurgbp = [c for c in xp if c.startswith("eurgbp")]
    risk = [c for c in xp if c.startswith("risk") or c.startswith("eurobloc") or c.startswith("gbpbloc_resid")]
    semi = [c for c in df.columns if c.startswith("rsasym")]
    carryrank = [c for c in df.columns if c.startswith("carryrank") or c.startswith("sign_agree")]
    return {"base": base, "eurgbp": eurgbp, "risk": risk, "semi": semi, "carryrank": carryrank}[kind]


# ---------------------------------------------------------------------------
# Single-fit evaluator
# ---------------------------------------------------------------------------
def fit_eval(TR, VA, YR, cols, tag):
    ytr = TR["_y"].astype(int).values; yva = VA["_y"].astype(int).values
    Xtr = TR[cols].astype("float32").values
    if len(Xtr) > 200_000:                       # subsample fit (memory; matches cpcv SUB_FIT)
        rng = np.random.default_rng(7); sel = rng.choice(len(Xtr), 200_000, replace=False)
        Xtr = Xtr[sel]; ytr = ytr[sel]
    L = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=127,
        min_child_samples=400, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
        reg_lambda=20, n_estimators=2000, n_jobs=20, verbosity=-1)
    L.fit(Xtr, ytr, eval_set=[(VA[cols].astype("float32").values, yva)], eval_metric="auc",
          callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)])
    pva = L.predict_proba(VA[cols].astype("float32"))[:, 1]; aucv = float(roc_auc_score(yva, pva))
    confv = np.abs(pva - 0.5); THR = float(np.quantile(confv, 1 - 0.02))
    out = {"val_auc": round(aucv, 4), "n_feats": len(cols), "years": {}}
    for w in YR:
        D = YR[w]; pr = L.predict_proba(D[cols].astype("float32"))[:, 1]
        y = D["_y"].astype(int).values; fwd = D["_fwd"].values; ts = D["_ts"].values.astype("int64")
        mv = fwd != 0; moved = mv
        g = side_eval(pr, y, moved, ts, THR)
        out["years"][w] = {"auc": round(float(roc_auc_score(y[mv], pr[mv])), 4),
            "cov2": {k: {"n": g[k]["n"], "wr": round(g[k]["wr"], 4)} for k in ("COMBINED", "UP", "DOWN")} if g else None}
    c = out["years"]
    print(f"  [orthochan GBPUSD {tag}] feats={len(cols)} VAL_AUC={aucv:.4f} | 2026 cov2 COMB "
          f"{c['oos']['cov2']['COMBINED'] if c['oos']['cov2'] else None}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("[orthochan GBPUSD] building NY-restricted panel...", flush=True)

    def prep(years, stride):
        F = build_xp_gbp(years, stride); F = augment(F, years, "xpbase")
        F = add_semivar(F, years); F = add_carryrank(F, years)
        m = session_mask(F["_ts"].values.astype("int64"), SESSION)
        return F.loc[m]

    TR = prep(SPL["train"], 6); VA = prep(SPL["val"], 2)
    YR = {w: prep(SPL[w], 1) for w in ("test24", "test25", "oos")}
    print(f"[orthochan GBPUSD] train_NY={len(TR):,} val_NY={len(VA):,} build={time.time()-t0:.0f}s", flush=True)

    base      = cols_of(TR, "base")
    eurgbp    = cols_of(TR, "eurgbp")
    risk      = cols_of(TR, "risk")
    semi      = cols_of(TR, "semi")
    carryrank = cols_of(TR, "carryrank")
    print(f"[orthochan GBPUSD] |base|={len(base)} |eurgbp|={len(eurgbp)} "
          f"|risk|={len(risk)} |semi|={len(semi)} |carryrank|={len(carryrank)}", flush=True)

    res = {
        "key": "GBPUSD.15m.NY",
        "breakeven": BE,
        "falsifier": {
            "registered": "pre-OOS",
            "ADDS_if": "base+block VAL-AUC > base_VAL_AUC+0.005 AND 2026 cov2 COMB > base 2026 cov2",
            "KILLS_if": "delta_auc <= 0.005 OR 2026 cov2 COMB <= base OR base+block val_auc not reproducible at refit-CPCV",
            "note": "falsifier registered before any held-out evaluation; screen metric = famonly ΔVAL-AUC"
        },
        "arms": {}
    }
    json.dump(res, open(RESULT, "w"), indent=2)

    res["arms"]["base"] = fit_eval(TR, VA, YR, base, "base")

    for nm, blk in (("eurgbp", eurgbp), ("risk", risk), ("semi", semi), ("carryrank", carryrank)):
        if blk: res["arms"][f"{nm}_only"] = fit_eval(TR, VA, YR, blk, f"{nm}_only")
        res["arms"][f"base+{nm}"] = fit_eval(TR, VA, YR, base + blk, f"base+{nm}")

    b = res["arms"]["base"]; b_auc = b["val_auc"]
    b26 = b["years"]["oos"]["cov2"]["COMBINED"]["wr"] if b["years"]["oos"]["cov2"] else float("nan")
    verdict = {}
    for nm in ("eurgbp", "risk", "semi", "carryrank"):
        a = res["arms"].get(f"base+{nm}")
        if not a: continue
        a26 = a["years"]["oos"]["cov2"]["COMBINED"]["wr"] if a["years"]["oos"]["cov2"] else float("nan")
        verdict[nm] = {
            "base+block_val_auc": a["val_auc"],
            "delta_auc": round(a["val_auc"] - b_auc, 4),
            "block_2026_cov2": a26,
            "ADDS": bool(a["val_auc"] > b_auc + 0.005 and np.isfinite(a26) and a26 > b26)
        }
    res["verdict"] = verdict; res["base_val_auc"] = b_auc; res["base_2026_cov2"] = b26
    json.dump(res, open(RESULT, "w"), indent=2)

    print(f"\n[orthochan GBPUSD] base VAL_AUC={b_auc} 2026cov2={b26}", flush=True)
    for nm, v in verdict.items():
        print(f"  {nm}: base+block AUC {v['base+block_val_auc']} (Δ{v['delta_auc']}) "
              f"2026cov2 {v['block_2026_cov2']} -> ADDS={v['ADDS']}", flush=True)
    print(f"[orthochan GBPUSD] done {time.time()-t0:.0f}s -> {RESULT}", flush=True)


if __name__ == "__main__":
    main()
