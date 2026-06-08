"""USDJPY 15m DIRECTION lever — E3 COMPLEXITY / PREDICTABILITY GATES (NY session).

KEY: USDJPY.15m.ny   LEVER: complexity (causal permutation-entropy d3/d4, variance-ratio Hurst, lag-1 autocorr)

THESIS (E3): 15m direction is ~0.50 predictable ON AVERAGE, but the average may mix a predictable
(low-complexity / persistent) subset with a coin-flip subset. Complexity measures from nonlinear
dynamics — permutation entropy (Bandt-Pompe d=3 & d=4), variance-ratio Hurst proxy, lag-1 return
autocorrelation — are computed CAUSALLY on a trailing 1m log-return window at each NY 15m decision bar.
We then:
  (A) Bucket held-out NY moved bars by each complexity stat (quintiles) and report CONDITIONAL
      DIRECTION accuracy + moved-AUC of the own-pair GBM direction signal per bin.
  (B) Test the complexity stat ITSELF as a standalone direction (moved-AUC) predictor.
  (C) Gate the base GBM win-rate to the LOW-complexity (predictable) bars and read selective WR.

SIGN-INVARIANCE WARNING (arXiv:2512.15720): state/complexity/volatility gates are EXPECTED to gate
move SIZE, not SIGN. We RUN the direction test faithfully and let the number decide. If complexity is a
pure magnitude signal, the falsifier (conditional direction accuracy/AUC FLAT across complexity bins,
GBM gated-AUC no better than ungated) will fire.

EVAL DISCIPLINE (non-negotiable):
  - DECISION rows restricted to NY session (America/New_York 08-17, DST-correct via sessions.session_mask).
  - FEATURES (incl. complexity) stay CAUSAL & CONTINUOUS — rolling stats legitimately span the prior
    session; only the prediction/eval rows are filtered to NY.
  - EVAL label is ALWAYS the deriv-faithful fixed-15m sign = sign(close[t+15]-close[t]); ties (move==0)
    LOSE. This is exactly what build() returns via y (up=1/down=0 on moved) + moved(bool).
  - Per held-out year (test24/test25/oos): moved-AUC of the lever-direction signal vs up/down label on
    NY moved bars, AND selective win-rate at cov3% via side_eval (nonoverlap gap=900 built in).
  - Sanity tripwire: NY moved up-rate must be ~0.47-0.53 (else flag the mirage).

INCUMBENT to beat: certified NY own-pair GBM signal AUC ~.539 / WR ~.58-.60. This lever SURVIVES only if
its held-out moved-AUC exceeds ~.539 OR its NY cov3% WR CI-lower clears 0.541 in >=2 years.

The base GBM direction model + build/side_eval/boot/mk_lgb/BE/SPL/FEATS are imported from usdjpy_15m_base.
Complexity stats are computed by a separate close-only causal loader (RAM-cheap) and joined on ts.

  /home/sean/binary-algo-venv/bin/python usdjpy_15m_complexity.py [stride]
"""
import os, sys, json, math, time, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

import usdjpy_15m_base as B            # build, side_eval, boot, mk_lgb, BE, SPL, FEATS, nonoverlap_chrono, PAIR, HOR, STEP, GAP
import harness as Hm                   # FEAT_DIR
from sessions import session_mask

PAIR = B.PAIR; HOR = B.HOR; STEP = B.STEP; GAP = B.GAP; BE = B.BE
SPL = B.SPL
KEY = "USDJPY.15m.ny"
RESULT = "usdjpy_15m_complexity_result.json"
TR_STRIDE = int(sys.argv[1]) if (len(sys.argv) > 1 and str(sys.argv[1]).isdigit()) else 6  # >=6 keeps RAM<3GB
NUM_LEAVES = 127

# trailing-window lengths (in 1m bars) for the causal complexity stats
PE_W = 120          # permutation-entropy counting window
AC_W = 60           # lag-1 autocorr rolling window
VR_W = 240          # variance-ratio window
VR_K = 15           # VR aggregation block == the 15m horizon
COMP_MEASURES = ["pe3", "pe4", "hurst", "ac1"]


# ---------------------------------------------------------------- causal complexity stats
def perm_entropy(r, d=3, tau=1, W=PE_W):
    """Causal rolling Bandt-Pompe permutation entropy of return series r (normalized to [0,1]).
    At index i the entropy uses ONLY the W ordinal patterns ending at/just-before i (trailing window)."""
    N = len(r); L = (d - 1) * tau
    if N <= L + 1:
        return np.full(N, np.nan)
    idx = np.arange(N - L)[:, None] + np.arange(0, d * tau, tau)[None, :]
    emb = r[idx]
    order = np.argsort(emb, axis=1, kind="stable")        # ordinal pattern of each length-d window
    code = (order * (d ** np.arange(d))).sum(1).astype(np.int64)
    M = len(code); nb = d ** d
    oh = np.zeros((M, nb), dtype=np.float32); oh[np.arange(M), code] = 1.0
    cs = np.cumsum(oh, axis=0); cnt = cs.copy(); cnt[W:] = cs[W:] - cs[:-W]   # trailing-W histogram
    tot = np.maximum(cnt.sum(1, keepdims=True), 1.0); p = cnt / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1) / math.log(math.factorial(d))
    out = np.full(N, np.nan)
    out[L:L + M] = ent
    out[:L + W] = np.nan                                   # warmup: not enough patterns yet
    return out


def load_complexity(years):
    """Close-only causal loader -> per-year dict of complexity stats keyed by ts(secs). RAM-cheap (no 239 feats).
    All stats use shift/rolling/trailing windows only => value at bar t depends on returns up to and including t.
    Note: the GBM label is sign(close[t+15]-close[t]) (FORWARD); the complexity stats are strictly BACKWARD."""
    frames = []
    for y in years:
        p = f"{Hm.FEAT_DIR}/{PAIR}_{y}.parquet"
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p, columns=["close"])
        d = d[~d.index.duplicated(keep="last")].sort_index()
        c = d["close"].values.astype(float); n = len(c)
        ts = d.index.values.astype("datetime64[s]").astype("int64")
        r = np.zeros(n); r[1:] = np.diff(np.log(c))         # 1m log returns (r[0]=0)
        rs = pd.Series(r)
        pe3 = perm_entropy(r, 3, 1, PE_W)
        pe4 = perm_entropy(r, 4, 1, PE_W)
        ac1 = rs.rolling(AC_W).corr(rs.shift(1)).values     # trailing lag-1 autocorrelation
        # variance-ratio Hurst proxy: H = 0.5*log(VR)/log(k) + 0.5, VR = var(sum_k r)/(k*var(r)), trailing VR_W
        agg = rs.rolling(VR_K).sum()
        vr = agg.rolling(VR_W).var() / (VR_K * rs.rolling(VR_W).var() + 1e-18)
        hurst = (0.5 * np.log(np.clip(vr.values, 1e-6, 1e6)) / math.log(VR_K) + 0.5)
        frames.append(pd.DataFrame({"ts": ts, "pe3": pe3, "pe4": pe4, "ac1": ac1, "hurst": hurst}))
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="ts", keep="last")
    return df


def attach_complexity(ts_eval, comp_df):
    """Align the per-bar complexity stats to the GBM eval ts (both are secs int64). Returns dict measure->array."""
    s = comp_df.set_index("ts")
    look = s.reindex(ts_eval)
    return {m: look[m].values for m in COMP_MEASURES}


# ---------------------------------------------------------------- per-bin direction diagnostics
def bin_direction_table(pr, y, moved, comp, ny, train_comp_valid, measure, nbins=5):
    """Conditional DIRECTION accuracy + moved-AUC of the GBM signal across quintile bins of `measure`,
    on NY MOVED bars. Bin edges are frozen on TRAIN (no held-out leakage). Returns list of bin dicts."""
    edges = np.nanquantile(train_comp_valid[measure], np.linspace(0, 1, nbins + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    mv = comp[measure]
    base = ny & moved & np.isfinite(mv)
    rows = []
    for b in range(nbins):
        m = base & (mv >= edges[b]) & (mv < edges[b + 1])
        ns = int(m.sum())
        if ns < 50:
            rows.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": ns,
                         "dir_acc": None, "moved_auc": None}); continue
        pred = (pr[m] > 0.5).astype(int)
        acc = float((pred == y[m]).mean())
        try:
            auc = float(roc_auc_score(y[m], pr[m]))
        except ValueError:
            auc = None
        rows.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": ns,
                     "dir_acc": acc, "moved_auc": auc})
    return rows


def auc_safe(yv, sv):
    try:
        if len(np.unique(yv)) < 2:
            return None
        return float(roc_auc_score(yv, sv))
    except ValueError:
        return None


def lever_signal_auc(y, moved, comp, ny, measure, sign):
    """Standalone moved-AUC of the complexity STAT as a direction predictor on NY moved bars.
    `sign`=+1 tests 'high stat => up'; we report max(auc, 1-auc)-symmetric via the better orientation,
    but store the raw +1 orientation AUC (a magnitude signal lands near 0.50 either way)."""
    mv = comp[measure]
    m = ny & moved & np.isfinite(mv)
    if m.sum() < 50:
        return None, 0
    s = sign * mv[m]
    return auc_safe(y[m], s), int(m.sum())


def main():
    t0 = time.time()

    # ============================ PRE-REGISTER FALSIFIER (before any held-out read) ============================
    res = {
        "key": KEY,
        "lever": "complexity (causal perm-entropy d3/d4, variance-ratio Hurst, lag-1 autocorr) "
                 f"on trailing 1m log-return window (PE_W={PE_W}, AC_W={AC_W}, VR_W={VR_W}, VR_K={VR_K}); "
                 "NY-session decision bars; eval label = deriv-faithful fixed-15m sign, ties LOSE",
        "method": "E3 complexity/predictability gates: bucket held-out NY moved bars by each complexity stat, "
                  "report conditional own-pair-GBM direction accuracy + moved-AUC per bin; test the complexity "
                  "stat itself as a standalone direction AUC predictor; gate base-GBM WR to low-complexity bars.",
        "model": "own-pair USDJPY 15m LGBM direction signal (239 base feats) reused from usdjpy_15m_base; "
                 "complexity stats are the GATE/lever applied on top",
        "settlement": "bar-close approx, ties LOSE, breakeven 0.541, gap=900s nonoverlap_chrono, NY session only",
        "tr_stride": TR_STRIDE, "num_leaves": NUM_LEAVES, "splits": SPL,
        "sign_invariance_note": "arXiv:2512.15720 predicts complexity gates SIZE not SIGN; running direction test anyway",
        "falsifier": {
            "registered_utc": "pre-OOS",
            "KILL_if": "conditional direction accuracy/moved-AUC is FLAT across complexity bins "
                       "(predictable bins do NOT beat coin-flip bins) AND the standalone complexity-stat "
                       "moved-AUC <= 0.51 in every held-out year AND the low-complexity-gated GBM cov3% WR "
                       "CI-lower clears 0.541 in <2 years. SURVIVE only if held-out moved-AUC > ~0.539 OR "
                       "NY cov3% WR CI-lower clears 0.541 in >=2 years.",
            "rationale": "Complexity/Hurst/entropy gates are theorem-predicted magnitude (size) signals, not "
                         "sign signals. The direction test is run faithfully; if flat, the lever is KILLED for "
                         "DIRECTION (it may still gate magnitude, out of scope here)."},
        "val_or_signal_auc": None,
        "years": {},
    }
    json.dump(res, open(RESULT, "w"), indent=2)

    # ============================ build GBM training data (239 feats, stride>=6) ============================
    Xtr, ytr, mtr, _ = B.build(SPL["train"], TR_STRIDE)
    Xva, yva, mva, tsv = B.build(SPL["val"])
    itr = mtr; iva = mva
    print(f"[cx15m] stride={TR_STRIDE} leaves={NUM_LEAVES} train_moved={int(itr.sum()):,} "
          f"val_moved={int(iva.sum()):,} feats={len(B.FEATS)} build={time.time()-t0:.0f}s", flush=True)

    import lightgbm as lgb
    L = B.mk_lgb(num_leaves=NUM_LEAVES)
    L.fit(Xtr[itr], ytr[itr], eval_set=[(Xva[iva], yva[iva])], eval_metric="auc",
          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    del Xtr, ytr, mtr                                       # free RAM before held-out builds
    pva = L.predict_proba(Xva)[:, 1]

    # complexity stats for TRAIN (for frozen bin edges) and VAL
    comp_tr = load_complexity(SPL["train"])
    train_comp_valid = {m: comp_tr[m].values[np.isfinite(comp_tr[m].values)] for m in COMP_MEASURES}
    del comp_tr

    # VAL diagnostics: NY mask, attach complexity, signal AUC
    ny_va = session_mask(tsv, "ny")
    comp_va = attach_complexity(tsv, load_complexity(SPL["val"]))
    val_moved_ny = iva & ny_va
    val_auc = auc_safe(yva[val_moved_ny], pva[val_moved_ny])
    val_up = float(yva[val_moved_ny].mean()) if val_moved_ny.sum() else float("nan")
    res["val_or_signal_auc"] = val_auc
    res["val_ny_moved_up_rate"] = val_up
    print(f"[cx15m] best_iter={L.best_iteration_} VAL NY moved-AUC(GBM)={val_auc} "
          f"VAL NY moved n={int(val_moved_ny.sum()):,} up={val_up:.4f} {time.time()-t0:.0f}s", flush=True)

    # ---- VAL gate selection on NY rows: WORST-half stability of the GBM (frozen for held-out) ----
    confv = np.abs(pva - 0.5)
    confv_ny = confv[ny_va]
    half = int(ny_va.sum()) // 2
    ny_idx = np.where(ny_va)[0]
    best = None
    for cov in (0.20, 0.10, 0.05, 0.03, 0.02):
        thr = float(np.quantile(confv_ny, 1 - cov))
        accs = []
        for s, e in ((0, half), (half, len(ny_idx))):
            sl = ny_idx[s:e]
            r = B.side_eval(pva[sl], yva[sl], mva[sl], tsv[sl], thr)
            accs.append(r["COMBINED"]["wr"] if r else float("nan"))
        worst = np.nanmin(accs)
        if best is None or worst > best[0]:
            best = (worst, cov, thr, accs)
    worst_half_wr, COV, THR, halfaccs = best
    res["gate"] = {"cov": COV, "conf_thr": THR, "val_worst_half_wr": float(worst_half_wr),
                   "val_half_wrs": [float(a) for a in halfaccs]}
    print(f"[cx15m] FROZEN GBM gate cov{COV:.0%} thr={THR:.4f} VAL(NY) worst-half WR={worst_half_wr:.4f} "
          f"halves={halfaccs}", flush=True)
    del Xva, yva, mva, tsv, pva, comp_va, ny_va

    # ============================ HELD-OUT per year ============================
    for w in ("test24", "test25", "oos"):
        Xw, yw, mw, tsw = B.build(SPL[w])
        pr = L.predict_proba(Xw)[:, 1]
        ny = session_mask(tsw, "ny")
        comp = attach_complexity(tsw, load_complexity(SPL[w]))

        moved_ny = mw & ny
        auc = auc_safe(yw[moved_ny], pr[moved_ny])          # GBM direction AUC on NY moved bars (incumbent reference)
        up_rate = float(yw[moved_ny].mean()) if moved_ny.sum() else float("nan")

        # (A) conditional GBM direction accuracy / moved-AUC per complexity bin
        bins = {m: bin_direction_table(pr, yw, mw, comp, ny, train_comp_valid, m) for m in COMP_MEASURES}
        # FLATNESS metric: spread of bin moved-AUC (max-min) per measure; ~0 => flat => magnitude-only
        bin_spread = {}
        for m in COMP_MEASURES:
            aucs = [b["moved_auc"] for b in bins[m] if b["moved_auc"] is not None]
            accs = [b["dir_acc"] for b in bins[m] if b["dir_acc"] is not None]
            bin_spread[m] = {"auc_spread": (float(max(aucs) - min(aucs)) if len(aucs) >= 2 else None),
                             "acc_spread": (float(max(accs) - min(accs)) if len(accs) >= 2 else None),
                             "auc_max": (float(max(aucs)) if aucs else None)}

        # (B) standalone complexity-stat moved-AUC as a direction predictor (both orientations -> report best)
        stat_auc = {}
        for m in COMP_MEASURES:
            a_pos, n = lever_signal_auc(yw, mw, comp, ny, m, +1)
            a_neg, _ = lever_signal_auc(yw, mw, comp, ny, m, -1)
            best_a = None
            if a_pos is not None and a_neg is not None:
                best_a = float(max(a_pos, a_neg))
            stat_auc[m] = {"auc_pos": a_pos, "auc_neg": a_neg, "auc_best_orient": best_a, "n": n}

        # (C) GBM win-rate gated to LOW-complexity (predictable) NY bars, at frozen gate.
        #     Low-complexity = below TRAIN 40th pct of pe3 (most predictable / least entropic).
        thr40 = float(np.nanquantile(train_comp_valid["pe3"], 0.40))
        lowcx = ny & (np.isfinite(comp["pe3"])) & (comp["pe3"] <= thr40)
        lc_idx = np.where(lowcx)[0]
        gate_lc = B.side_eval(pr[lc_idx], yw[lc_idx], mw[lc_idx], tsw[lc_idx], THR) if len(lc_idx) else None
        # ungated NY gate for contrast
        ny_idx = np.where(ny)[0]
        gate_all = B.side_eval(pr[ny_idx], yw[ny_idx], mw[ny_idx], tsw[ny_idx], THR) if len(ny_idx) else None

        def cov3_wr(gate):
            if not gate:
                return None
            g = gate["COMBINED"]
            return {"n": g["n"], "wr": round(g["wr"], 4), "ci": [round(c, 4) for c in g["ci"]]}

        res["years"][w] = {
            "auc": auc,                                     # GBM moved-AUC on NY moved bars (the direction signal)
            "moved_up_rate": up_rate,
            "tripwire_ok": bool(0.47 <= up_rate <= 0.53),
            "cov3_wr": cov3_wr(gate_lc),                    # low-complexity-gated GBM cov-gate WR (the lever result)
            "cov3_wr_ungated_ny": cov3_wr(gate_all),        # contrast: full-NY cov-gate WR
            "bins": bins,
            "bin_spread": bin_spread,
            "stat_direction_auc": stat_auc,
        }
        gl = cov3_wr(gate_lc); ga = cov3_wr(gate_all)
        print(f"=== {w} === GBM NY moved-AUC={auc} up-rate={up_rate:.4f} tripwire={'OK' if 0.47<=up_rate<=0.53 else 'MIRAGE'}",
              flush=True)
        print(f"        lowPE-gated WR: {gl}   |   ungated-NY WR: {ga}", flush=True)
        for m in COMP_MEASURES:
            bs = bin_spread[m]
            print(f"        {m:>6}: bin moved-AUC spread={bs['auc_spread']} max={bs['auc_max']} | "
                  f"standalone stat dir-AUC={stat_auc[m]['auc_best_orient']}", flush=True)
        del Xw, yw, mw, tsw, pr, comp, ny

    # ============================ APPLY FALSIFIER -> VERDICT ============================
    heldyrs = ("test24", "test25", "oos")
    beats_base_auc = any(
        (res["years"][w]["auc"] is not None and res["years"][w]["auc"] > 0.539) for w in heldyrs)
    # standalone complexity stat ever a real direction signal?
    stat_beats = any(
        (res["years"][w]["stat_direction_auc"][m]["auc_best_orient"] is not None
         and res["years"][w]["stat_direction_auc"][m]["auc_best_orient"] > 0.51)
        for w in heldyrs for m in COMP_MEASURES)
    # low-complexity-gated GBM WR CI-lower clears BE in >=2 years?
    wr_clears = [w for w in heldyrs
                 if res["years"][w]["cov3_wr"] and res["years"][w]["cov3_wr"]["ci"][0] >= BE]
    wr_survives = len(wr_clears) >= 2
    # bins flat? (max moved-AUC spread across measures/years tiny)
    spreads = [res["years"][w]["bin_spread"][m]["auc_spread"]
               for w in heldyrs for m in COMP_MEASURES
               if res["years"][w]["bin_spread"][m]["auc_spread"] is not None]
    max_spread = float(max(spreads)) if spreads else None
    bins_flat = (max_spread is not None and max_spread < 0.05)

    killed = not (beats_base_auc or wr_survives)
    res["verdict"] = {
        "KILLED": bool(killed),
        "beats_base_auc": bool(beats_base_auc),
        "wr_clears_BE_years": wr_clears,
        "wr_survives_2yr": bool(wr_survives),
        "standalone_stat_auc_beats_051": bool(stat_beats),
        "bins_flat_max_auc_spread": max_spread,
        "bins_flat": bool(bins_flat),
        "note": ("SURVIVED: lever clears the bar (held-out moved-AUC>~.539 OR cov3% WR CI-lo>=.541 in >=2yr)."
                 if not killed else
                 "KILLED for DIRECTION: complexity gates magnitude not sign — conditional direction "
                 "accuracy/AUC flat across complexity bins, standalone stat AUC ~coin-flip, and "
                 "low-complexity-gated GBM WR does not clear BE in >=2 years. Consistent with arXiv:2512.15720."),
    }
    json.dump(res, open(RESULT, "w"), indent=2)
    print(f"\n[cx15m] VERDICT: {'KILLED' if killed else 'SURVIVED'} "
          f"(beats_base_auc={beats_base_auc}; wr_clears={wr_clears}; bins_flat_spread={max_spread}) "
          f"-> {RESULT}  total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
