"""FAST single-fit PROBE: cross-pair USD-residual/order-flow primary @MX_HOR=2 (the keystone lever NEVER run at 2m).

Decides go/no-go for a full per-fold-refit CPCV. Cross-pair feature-row POOLING (m5_xpair build) is the only lever
that certified at 5m (UP .577) and 15m (BOTH .567/.574); it is mechanistically DISTINCT from the killed cross-LEG-sign
(min2_legsign) and triangular-residual (min2_triangular) channels — those use another pair's *sign* as a signal; this
trains one EURUSD-sign model on the multi-pair feature ROWS (USD common factor removed, lead-lag residuals, dispersion).
The online-ARF efficiency keystone tested single-pair ADAPTATION, not cross-pair DECORRELATION — so it does not subsume this.

Discipline: gate frozen by WORST-VAL-HALF stability (skill rule), NOT VAL-acc-max. Side-split by predicted side.
Per-year 2024/2025/2026 combined + UP-split + DOWN-split acc with bootstrap CI95, nonoverlap_chrono, moved bars
(ties dropped per the 5m/15m convention) PLUS the measured 2m tie-fraction so the entry-lag/tie optimism is quantified.

Pre-registered falsifier (written to JSON BEFORE held-out is inspected):
  KILL_cross_pair_2m if VAL dirAUC <= 0.515  (no in-sample cross-pair direction signal at 2m)
  ELSE ESCALATE to refit-CPCV iff some side's point acc >= 0.545 in >= 2 of {2024,2025,2026} held-out years
  ELSE KILL (cross-pair direction dead at 2m on-disk; redirect to discovery).
Usage: python min2_xpair_probe.py
"""
import os
os.environ["MX_HOR"] = "2"
import json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import m5_xpair as MX

HOR = 2; GAP_S = HOR * 60; BE = 0.541
GATE_FEAT = "1m_bb_width"   # short-horizon compression gate (analog of 15m's 15m_bb_width)
FEAT = MX.FEAT


def yr(ts):
    return (np.asarray(ts, dtype="datetime64[s]").astype("datetime64[Y]").astype(int) + 1970)


def tie_fraction(years):
    """Fraction of contiguous 2-bar (120s) EURUSD windows with EXACT-zero forward log-return (would be a deriv tie=loss)."""
    nz = ntot = 0
    for y in years:
        fp = f"{FEAT}/EURUSD_{y}.parquet"
        if not os.path.exists(fp):
            continue
        d = pd.read_parquet(fp, columns=["close"]); d = d[~d.index.duplicated(keep="last")]
        secs = d.index.values.astype("datetime64[s]").astype("int64"); lr = np.log(d["close"].values); n = len(lr)
        if n <= HOR:
            continue
        contig = (secs[HOR:] - secs[:-HOR]) == HOR * 60
        fr = lr[HOR:] - lr[:-HOR]
        fr = fr[contig]
        nz += int((fr == 0).sum()); ntot += int(len(fr))
    return (nz / ntot) if ntot else float("nan"), ntot


def boot(corr, nb=5000, seed=7):
    corr = np.asarray(corr, float)
    if len(corr) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(corr)
    a = np.array([corr[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    t0 = time.time()
    TR = MX.build_xp(MX.SPL["train"], stride=4); VA = MX.build_xp(MX.SPL["val"])
    xpc = MX.xp_cols(TR)
    TR = MX.augment(TR, MX.SPL["train"], "xpof"); VA = MX.augment(VA, MX.SPL["val"], "xpof")
    cols = MX.feat_cols("xpof", TR, xpc)
    assert GATE_FEAT in VA.columns, f"{GATE_FEAT} missing"
    print(f"[probe] train={len(TR):,} val={len(VA):,} feats={len(cols)} build {time.time()-t0:.0f}s", flush=True)
    ytr = TR["_y"].astype(int).values; yva = VA["_y"].astype(int).values
    L = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=127, min_child_samples=400,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=20, n_estimators=3000,
                           n_jobs=20, verbosity=-1)
    L.fit(TR[cols].astype("float32"), ytr, eval_set=[(VA[cols].astype("float32"), yva)], eval_metric="auc",
          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    pva = L.predict_proba(VA[cols].astype("float32"))[:, 1]
    valauc = float(roc_auc_score(yva, pva))
    print(f"[probe] best_iter={L.best_iteration_} VAL AUC={valauc:.4f} {time.time()-t0:.0f}s", flush=True)

    # ---- pre-register falsifier BEFORE any held-out inspection ----
    res = {"test": "cross-pair xpof primary @MX_HOR=2 single-fit side-split probe", "breakeven": BE,
           "gate_feat": GATE_FEAT, "val_auc": round(valauc, 4),
           "falsifier": {"KILL_if_val_auc<=": 0.515,
                         "ESCALATE_if": "some side point-acc >= 0.545 in >=2 of 3 held-out years",
                         "ELSE": "KILL: cross-pair direction dead at 2m on-disk"}}
    json.dump(res, open("min2_xpair_probe_result.json", "w"), indent=1)
    if valauc <= 0.515:
        res["verdict"] = {"KILLED": True, "reason": f"VAL dirAUC {valauc:.4f} <= 0.515: no cross-pair 2m direction signal."}
        json.dump(res, open("min2_xpair_probe_result.json", "w"), indent=1)
        print(f"[probe] VERDICT KILLED (VAL AUC {valauc:.4f} <= 0.515)", flush=True)
        return

    # ---- freeze gate by WORST-VAL-HALF stability (min over VAL years), grid over {NY?, bb-q, cov} ----
    tsv = VA["_ts"].values.astype("int64"); vyr = yr(tsv); bbv = VA[GATE_FEAT].values.astype("float32")
    fin = np.isfinite(bbv); nyv = VA["sess_ny"].values > 0.5; confv = np.abs(pva - 0.5)
    vyears = sorted(set(vyr.tolist()))
    best = None
    for useny in (False, True):
        for bq in (10, 20, 33, 100):           # 100 = no compression gate
            bthr = np.nanpercentile(bbv[fin], bq)
            base_all = fin & (bbv <= bthr) & (nyv if useny else np.ones(len(bbv), bool))
            if base_all.sum() < 200:
                continue
            for cov in (0.05, 0.10, 0.20):
                cthr = float(np.quantile(confv[base_all], 1 - cov))
                accs, ntot = [], 0
                for Y in vyears:
                    m = base_all & (vyr == Y) & (confv >= cthr); sel = MX.nonoverlap_chrono(tsv, m, GAP_S)
                    if len(sel) < 40:
                        accs = None; break
                    accs.append(((pva[sel] > 0.5).astype(int) == yva[sel]).mean()); ntot += len(sel)
                if accs is None or ntot < 150:
                    continue
                wh = min(accs)
                if best is None or wh > best[0]:
                    best = (wh, useny, float(bthr), cthr, bq, cov, ntot)
    if best is None:
        res["verdict"] = {"KILLED": True, "reason": "no VAL gate reached n>=150 with worst-half defined; cross-pair too thin at 2m."}
        json.dump(res, open("min2_xpair_probe_result.json", "w"), indent=1)
        print("[probe] VERDICT KILLED (no stable VAL gate)", flush=True)
        return
    wh, USENY, BTHR, CTHR, BQ, COV, nV = best
    print(f"[probe] FROZEN gate: NY={USENY} {GATE_FEAT}<=q{BQ}({BTHR:.3e}) cov{COV:.0%} conf>={CTHR:.4f} | VAL worst-half={wh:.4f} n={nV}", flush=True)
    res["frozen_gate"] = {"use_ny": USENY, "bb_q": BQ, "bb_thr": BTHR, "cov": COV, "conf_thr": CTHR,
                          "val_worst_half": round(wh, 4), "val_n": nV}

    # ---- held-out per year: combined + side-split, CI95, nonoverlap, moved bars ----
    per_year = {}
    for w, ylabel in (("test24", 2024), ("test25", 2025), ("oos", 2026)):
        D = MX.build_xp(MX.SPL[w]); D = MX.augment(D, MX.SPL[w], "xpof")
        for c in cols:
            if c not in D.columns:
                D[c] = np.nan
        X = D[cols].astype("float32"); pr = L.predict_proba(X)[:, 1]; y = D["_y"].astype(int).values
        ts = D["_ts"].values.astype("int64"); bb = D[GATE_FEAT].values.astype("float32")
        base = np.isfinite(bb) & (bb <= BTHR)
        if USENY:
            base = base & (D["sess_ny"].values > 0.5)
        conf = np.abs(pr - 0.5); cand = base & (conf >= CTHR)
        sel = MX.nonoverlap_chrono(ts, cand, GAP_S)
        rec = {"n_sel": int(len(sel)), "auc_full": round(float(roc_auc_score(y, pr)), 4)}
        if len(sel):
            pred = (pr[sel] > 0.5).astype(int); yt = y[sel]
            corr = (pred == yt).astype(float); lo, hi = boot(corr)
            rec["combined"] = {"acc": round(float(corr.mean()), 4), "ci": [round(lo, 4), round(hi, 4)], "n": int(len(sel)),
                               "up_rate_moved": round(float(yt.mean()), 4)}
            su = pred == 1; sd = pred == 0
            if su.sum() >= 25:
                c = (pred[su] == yt[su]).astype(float); lo2, hi2 = boot(c)
                rec["UP"] = {"acc": round(float(c.mean()), 4), "ci": [round(lo2, 4), round(hi2, 4)], "n": int(su.sum())}
            if sd.sum() >= 25:
                c = (pred[sd] == yt[sd]).astype(float); lo2, hi2 = boot(c)
                rec["DOWN"] = {"acc": round(float(c.mean()), 4), "ci": [round(lo2, 4), round(hi2, 4)], "n": int(sd.sum())}
        per_year[str(ylabel)] = rec
        u = rec.get("UP", {}); dn = rec.get("DOWN", {})
        print(f"=== {w} ({ylabel}) === AUC={rec['auc_full']:.4f} n={rec['n_sel']} "
              f"comb={rec.get('combined',{}).get('acc','-')} "
              f"UP={u.get('acc','-')}(n{u.get('n','-')}) DOWN={dn.get('acc','-')}(n{dn.get('n','-')}) {time.time()-t0:.0f}s", flush=True)
    res["per_year"] = per_year

    tf, ntf = tie_fraction(["2024", "2025", "2026"])
    res["tie_fraction_heldout"] = {"frac_exact_zero_2bar": round(tf, 5), "n": ntf,
                                   "note": "moved-bar accs above DROP these; deriv charges them as losses (~this much optimism upper bound)"}
    print(f"[probe] 2m tie-fraction (held-out) = {tf:.5f} over n={ntf:,}", flush=True)

    # ---- apply pre-registered escalation rule ----
    def side_clears(side):
        return sum(1 for Y in ("2024", "2025", "2026")
                   if per_year[Y].get(side, {}).get("acc", 0) >= 0.545)
    up_yrs, dn_yrs = side_clears("UP"), side_clears("DOWN")
    escalate = (up_yrs >= 2) or (dn_yrs >= 2)
    res["verdict"] = {"ESCALATE": bool(escalate), "up_years_ge_0.545": up_yrs, "down_years_ge_0.545": dn_yrs,
                      "reason": ("ESCALATE to refit-CPCV: a side point-clears 0.545 in >=2 held-out years"
                                 if escalate else
                                 "KILL: cross-pair direction dead at 2m on-disk (no side clears 0.545 in >=2 years)")}
    json.dump(res, open("min2_xpair_probe_result.json", "w"), indent=1)
    print(f"[probe] VERDICT: {'ESCALATE' if escalate else 'KILL'} (UP {up_yrs}/3 yrs, DOWN {dn_yrs}/3 yrs >= 0.545) -> min2_xpair_probe_result.json", flush=True)


if __name__ == "__main__":
    main()
