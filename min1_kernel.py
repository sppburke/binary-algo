"""EXPERIMENT B — SCHOLARLY KERNEL METHOD (Fletcher 2010, MLMI/NIPS: MKL/SVM over financially-motivated
features on EBS EUR/USD at the SECONDS scale). This is the one genuinely-new MODEL CLASS not yet tried on
our 1s microstructure: an RBF kernel classifier (Nystroem feature map + SGD log-loss, scalable surrogate
for SVC) over Fletcher-style financially-motivated features PLUS genuine SIGNED per-side order flow built
from the raw EURUSD tick archive (bid-vol/ask-vol per side, tick-rule signed volume, flow imbalance over
5/15/30/60s windows).

DISCIPLINE (deriv-faithful, reuses min1_production UNCHANGED):
  label = sign(close[t+H]-close[t]); H in {15,30,60}s. entry = next tick after order (1s lag); exit = last
  tick <= entry+H; ties (ret==0) LOSE. Non-overlapping trades via M.nonoverlap_chrono(gap=H+TOL). bootstrap
  CI95 over non-overlap trades. Model + confidence threshold selected on VAL ONLY (worst-VAL-half stability),
  then judged on EACH of 2024 / 2025 / 2026 SEPARATELY. A book "wins" only if OOS n>=25 AND CI95 lower bound
  > the bar on all three windows.

MEMORY SAFETY: process ONE split at a time; order-flow features built day-by-day and CACHED to parquet under
flow_cache/<split>_flow.parquet; train subsample <=80k rows. Never hold all four feature frames at once.

Usage:
  python min1_kernel.py flow [split]   # build+cache 1s signed order-flow features for a split (or all)
  python min1_kernel.py run            # build (if needed), fit kernel @ H=15/30/60, eval per-window, report
"""
import sys, os, json, time, glob, gc
import numpy as np, pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import min1_production as M

ROOT = "/home/sean/git/binary-algo"
RAW  = "/media/sean/CORSAIR/tick_data/raw/EURUSD"
FLOWDIR = f"{ROOT}/flow_cache"
TOL_S = M.TOL_S            # 10s settlement tolerance (matches production)
LAG_S = M.ENTRY_LAG_S      # 1s entry lag (deriv next-tick)
HORIZONS = [15, 30, 60]    # Fletcher's edge is strongest at the short end
SUB_TRAIN = 80_000         # kernel train subsample cap
N_COMP = 300               # Nystroem components
COV = 0.10                 # selective coverage target on the gate (top-confidence fraction)
SEED = 0

# financially-motivated microstructure features from M.feats (Fletcher-style: microprice dev, imbalance + EMAs,
# spread, short returns / realized vol). Kept compact so the RBF kernel is well-conditioned.
MICRO_FEATS = [
    "micro_dev", "micro_dev_ema5", "micro_dev_ema15", "micro_dev_ema30",
    "imb", "imb_ema3", "imb_ema5", "imb_ema10", "imb_ema20", "imb_acc",
    "spread", "spread_ema30", "nt", "tsz",
    "ret5", "ret15", "ret30", "ret60", "ret300",
    "rv30", "rv60", "rv300",
]
# genuine signed order-flow features (built here from raw per-side volumes)
FLOW_FEATS = [
    "flow_imb5", "flow_imb15", "flow_imb30", "flow_imb60",   # tick-rule signed vol / total vol over window
    "netba5", "netba15", "netba30", "netba60",               # (bid-vol - ask-vol) signed depth flow, normalized
    "sv_ema15", "sv_ema60",                                   # EMA of per-1s signed volume
]

# ----------------------------- order-flow feature builder -----------------------------
def _day_flow(day):
    """Aggregate one UTC day of raw ticks to a 1s-grid signed-flow primitive frame. Returns DataFrame indexed
    by integer epoch-second with columns [sv, nba, vol]. sv=tick-rule signed volume, nba=sum(bid-vol-ask-vol)."""
    pat = f"{RAW}/EURUSD_{day.strftime('%Y-%m-%d')}_*.parquet"
    fs = sorted(glob.glob(pat))
    if not fs:
        return None
    parts = []
    for f in fs:
        try:
            parts.append(pd.read_parquet(f, columns=["timestamp_utc", "bid", "ask", "bid-vol", "ask-vol"]))
        except Exception:
            pass
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True).sort_values("timestamp_utc")
    del parts
    ts = df["timestamp_utc"].values.astype(float)
    mid = ((df["bid"].values + df["ask"].values) * 0.5).astype(float)
    bv = df["bid-vol"].values.astype(float); av = df["ask-vol"].values.astype(float)
    vol = bv + av
    # tick rule: sign of mid change, carry last non-zero sign forward (causal)
    dmid = np.diff(mid, prepend=mid[0])
    s = np.sign(dmid)
    # vectorized carry-forward of last non-zero sign
    nz = s != 0
    idx = np.where(nz, np.arange(len(s)), 0)
    np.maximum.accumulate(idx, out=idx)
    s = s[idx]
    s[0] = 0 if not nz[0] else s[0]
    sv = s * vol
    nba = bv - av
    sec = np.floor(ts).astype("int64")
    g = pd.DataFrame({"sec": sec, "sv": sv, "nba": nba, "vol": vol})
    agg = g.groupby("sec", sort=True).agg(sv=("sv", "sum"), nba=("nba", "sum"), vol=("vol", "sum"))
    del df, g
    return agg

def build_flow(sp):
    """Build & cache 1s-grid signed order-flow features for split sp, aligned to the existing 1s tick cache
    timestamps. Rolling 5/15/30/60s flow features are computed on a CONTIGUOUS per-second axis (forward-filled
    over empty seconds with zero flow) so the windows are wall-clock-faithful, then sampled at the grid bars."""
    os.makedirs(FLOWDIR, exist_ok=True)
    out = f"{FLOWDIR}/{sp}_flow.parquet"
    if os.path.exists(out):
        print(f"[flow] {sp}: cached -> {out}", flush=True)
        return out
    t0 = time.time()
    b = M.load_split(sp)
    grid_sec = b.index.values.astype("datetime64[s]").astype("int64")
    gidx = b.index
    days = pd.Series(b.index.normalize().unique()).sort_values()
    sec_lo = int(grid_sec.min()) - 70; sec_hi = int(grid_sec.max()) + 1
    span = sec_hi - sec_lo + 1
    # dense per-second axis: position of epoch-second s is (s - sec_lo). float32 to bound memory.
    sv = np.zeros(span, "float64"); nba = np.zeros(span, "float64"); vol = np.zeros(span, "float64")
    nd = 0
    for day in days:
        agg = _day_flow(day)
        if agg is None:
            continue
        si = agg.index.values
        keep = (si >= sec_lo) & (si <= sec_hi)
        si = si[keep]
        ii = (si - sec_lo).astype("int64")
        sv[ii]  += agg["sv"].values[keep]
        nba[ii] += agg["nba"].values[keep]
        vol[ii] += agg["vol"].values[keep]
        nd += 1
        del agg
        if nd % 20 == 0:
            print(f"[flow] {sp}: {nd}/{len(days)} days {time.time()-t0:.0f}s", flush=True)
    # trailing wall-clock rolling sums on the dense per-second axis: out[i]=sum(a[max(0,i+1-w):i+1])
    def rs(a, w):
        c = np.concatenate([[0.0], np.cumsum(a)])      # c[i]=sum(a[:i])
        n = len(a)
        lo = c[np.maximum(np.arange(1, n + 1) - w, 0)]
        return c[1:] - lo
    feats = {}
    for w in (5, 15, 30, 60):
        sw = rs(sv, w); vw = rs(vol, w); nbw = rs(nba, w)
        feats[f"flow_imb{w}"] = (sw / (vw + 1e-9)).astype("float32")
        feats[f"netba{w}"]   = (nbw / (vw + 1e-9)).astype("float32")
        del sw, vw, nbw
    # EMAs of per-1s signed volume normalized by per-1s vol (flow direction strength)
    sv_norm = sv / (vol + 1e-9)
    sv_s = pd.Series(sv_norm)
    feats["sv_ema15"] = sv_s.ewm(span=15).mean().values.astype("float32")
    feats["sv_ema60"] = sv_s.ewm(span=60).mean().values.astype("float32")
    del sv, nba, vol, sv_norm, sv_s
    gc.collect()
    # sample at grid bars (position = grid_sec - sec_lo, dense axis is contiguous)
    gi = (grid_sec - sec_lo).astype("int64")
    G = pd.DataFrame({k: v[gi] for k, v in feats.items()}, index=gidx)
    del feats
    G.to_parquet(out)
    print(f"[flow] {sp}: built {len(G)} grid rows from {nd} days, {time.time()-t0:.0f}s -> {out}", flush=True)
    del b, G
    gc.collect()
    return out

# ----------------------------- feature assembly -----------------------------
def build_Xyt(sp, horizon):
    """Return (Xdf, y, mag, valid, ts) for split sp at horizon H. Combines Fletcher micro feats (from M.feats)
    + cached signed order-flow feats. Labels via M.wc_ret at this horizon (deriv-faithful)."""
    b = M.load_split(sp)
    Xmic = M.feats(b)[MICRO_FEATS]
    flow_path = f"{FLOWDIR}/{sp}_flow.parquet"
    Fl = pd.read_parquet(flow_path)
    # align flow onto the same index (identical grid by construction)
    Fl = Fl.reindex(Xmic.index)
    X = pd.concat([Xmic, Fl[FLOW_FEATS]], axis=1)
    mid = b["mid"].values.astype(float)
    ts = b.index.values.astype("datetime64[s]").astype("int64")
    ret, valid = M.wc_ret(ts, mid, horizon_s=horizon, tol_s=TOL_S, lag_s=LAG_S)
    y = (ret > 0).astype(int); mag = np.abs(ret)
    year = b.index.year.values
    del b
    return X, y, mag, valid, ts, year

def boot(c, nb=4000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def winacc(ts, valid, y, mag, p_up, cand, gap):
    """non-overlap chronological selection; return (correct[], sel_idx)."""
    sel = M.nonoverlap_chrono(ts, cand, gap=gap)
    if len(sel) == 0:
        return np.array([]), sel
    pred = (p_up[sel] > 0.5).astype(int)
    correct = ((pred == y[sel]) & (mag[sel] > 0)).astype(float)
    return correct, sel

# ----------------------------- main experiment -----------------------------
def run():
    t0 = time.time()
    np.random.seed(SEED)
    results = {}
    BAR = 0.541  # deriv breakeven; we also report vs the honest 0.55/0.60 frontier in the summary
    # ensure flow caches exist (build one split at a time)
    for sp in ("train", "val", "test", "oos"):
        build_flow(sp)
        gc.collect()

    for H in HORIZONS:
        gap = H + TOL_S
        print(f"\n######## HORIZON H={H}s (gap={gap}s) ########", flush=True)
        # ---- TRAIN: fit scaler+Nystroem+SGD on <=SUB_TRAIN subsample ----
        Xtr, ytr, mtr, vtr, tstr, _ = build_Xyt("train", H)
        feat_names = list(Xtr.columns)
        itr = np.where(vtr & (mtr > 0))[0]                       # drop ties from fit
        if len(itr) > SUB_TRAIN:
            rng = np.random.default_rng(SEED)
            itr = np.sort(rng.choice(itr, SUB_TRAIN, replace=False))
        Xtr_s = Xtr.iloc[itr].values.astype("float32")
        Xtr_s = np.nan_to_num(Xtr_s, nan=0.0, posinf=0.0, neginf=0.0)
        ytr_s = ytr[itr]
        scaler = StandardScaler().fit(Xtr_s)
        Ztr = scaler.transform(Xtr_s)
        nys = Nystroem(kernel="rbf", gamma=1.0 / Xtr_s.shape[1], n_components=N_COMP, random_state=SEED)
        Ftr = nys.fit_transform(Ztr)
        clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=50, tol=1e-4,
                            random_state=SEED, n_jobs=20)
        clf.fit(Ftr, ytr_s)
        del Xtr, Xtr_s, Ztr, Ftr, ytr, mtr, vtr, tstr
        gc.collect()
        print(f"[H{H}] kernel fit on n={len(itr)} subsample {time.time()-t0:.0f}s", flush=True)

        def predict(Xdf):
            Z = scaler.transform(np.nan_to_num(Xdf.values.astype("float32"), nan=0.0, posinf=0.0, neginf=0.0))
            return clf.predict_proba(nys.transform(Z))[:, 1]

        # ---- VAL: select confidence threshold by worst-VAL-half stability ----
        Xva, yva, mva, vva, tsva, _ = build_Xyt("val", H)
        pva = predict(Xva)
        del Xva
        gc.collect()
        vord = np.argsort(tsva); h = len(vord) // 2
        vh1 = np.zeros(len(tsva), bool); vh1[vord[:h]] = True; vh2 = ~vh1
        confva = np.abs(pva - 0.5)
        base = vva & (mva >= 0)                                  # gate = valid bars; selectivity via confidence
        # candidate thresholds = confidence quantiles on VAL valid bars
        qs = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
        grid = [float(np.quantile(confva[base], q)) for q in qs]
        best = None
        for thr in grid:
            cand = base & (confva >= thr)
            c1, _ = winacc(tsva, vva, yva, mva, pva, cand & vh1, gap)
            c2, _ = winacc(tsva, vva, yva, mva, pva, cand & vh2, gap)
            if len(c1) < 15 or len(c2) < 15:
                continue
            wh = min(c1.mean(), c2.mean())
            if best is None or wh > best[0]:
                best = (wh, thr)
        if best is None:
            # fall back to median confidence threshold
            best = (float("nan"), grid[0])
        vhmin, thr = best
        # full-VAL accuracy at chosen thr
        cva, sva = winacc(tsva, vva, yva, mva, pva, base & (confva >= thr), gap)
        print(f"[H{H}] VAL thr={thr:.4f} worst-half={vhmin:.3f} | VAL acc={cva.mean() if len(cva) else float('nan'):.3f} n={len(cva)}", flush=True)
        del yva, mva, vva, tsva, pva
        gc.collect()

        # ---- TEST (split 2024/2025) + OOS (2026): judge per window ----
        Xte, yte, mte, vte, tste, yrte = build_Xyt("test", H)
        pte = predict(Xte); confte = np.abs(pte - 0.5)
        del Xte; gc.collect()
        Xoo, yoo, moo, voo, tsoo, yroo = build_Xyt("oos", H)
        poo = predict(Xoo); confoo = np.abs(poo - 0.5)
        del Xoo; gc.collect()

        def winwin(ts, valid, y, mag, p, conf, yr, mask=None):
            cand = valid & (mag >= 0) & (conf >= thr)
            if mask is not None:
                cand = cand & mask
            return winacc(ts, valid, y, mag, p, cand, gap)

        c24, _ = winwin(tste, vte, yte, mte, pte, confte, yrte, yrte == 2024)
        c25, _ = winwin(tste, vte, yte, mte, pte, confte, yrte, yrte == 2025)
        c26, _ = winwin(tsoo, voo, yoo, moo, poo, confoo, yroo)
        a = lambda c: (float(c.mean()) if len(c) else float("nan"))
        lo26, hi26 = boot(c26)
        lo24, hi24 = boot(c24); lo25, hi25 = boot(c25)
        floor = min(a(c24), a(c25), a(c26))
        clears = (len(c26) >= 25 and lo24 > 0.65 and lo25 > 0.65 and lo26 > 0.65)
        beats060 = (len(c26) >= 25 and lo24 > 0.60 and lo25 > 0.60 and lo26 > 0.60)
        res = dict(thr=thr, vhmin=vhmin,
                   a24=a(c24), a25=a(c25), a26=a(c26),
                   n24=len(c24), n25=len(c25), n26=len(c26),
                   ci24=[lo24, hi24], ci25=[lo25, hi25], ci26=[lo26, hi26],
                   floor=floor, clears065=bool(clears), beats060=bool(beats060))
        results[f"H{H}"] = res
        print(f"[H{H}] 2024 {a(c24):.3f} n{len(c24)} CI[{lo24:.3f},{hi24:.3f}] | "
              f"2025 {a(c25):.3f} n{len(c25)} CI[{lo25:.3f},{hi25:.3f}] | "
              f"2026 {a(c26):.3f} n{len(c26)} CI[{lo26:.3f},{hi26:.3f}] | FLOOR={floor:.3f}", flush=True)
        print(f"[H{H}] clears>0.65 all-windows={clears} ; beats 0.60 all-windows={beats060}", flush=True)
        del yte, mte, vte, tste, yrte, pte, yoo, moo, voo, tsoo, yroo, poo
        del clf, nys, scaler
        gc.collect()

    os.makedirs(f"{ROOT}/models", exist_ok=True)
    json.dump(results, open(f"{ROOT}/models/min1_kernel_summary.json", "w"), indent=2, default=float)
    print(f"\n[kernel] DONE {time.time()-t0:.0f}s -> models/min1_kernel_summary.json", flush=True)
    # pick best H by FLOOR (the binding-window discipline)
    bestH = max(results, key=lambda k: (results[k]["floor"] if not np.isnan(results[k]["floor"]) else -1))
    print(f"[kernel] BEST by floor: {bestH} floor={results[bestH]['floor']:.3f}", flush=True)
    return results

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "flow":
        sps = [sys.argv[2]] if len(sys.argv) > 2 else ["train", "val", "test", "oos"]
        for sp in sps:
            build_flow(sp); gc.collect()
    else:
        run()
