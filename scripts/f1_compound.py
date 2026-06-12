"""F1 — Is the verified 3s edge compoundable into a 15m directional bet, net of cost?

Three tests on the saved 3s ensemble probs (models/probs_tickens_H3.npz) aligned to the
1s-bar parquet (features_tick/{split}_1s.parquet, EURUSD):

  T1 DECAY     accuracy of the fast signal sign(p-0.5) at predicting the h-seconds-ahead
               direction, h in {3..900}. Binary framing: does a single fast call survive to 15m?
  T2 AGG-BINARY  does an AGGREGATE of fast signals over a trailing window W predict the
               15m-ENDPOINT sign better than ~0.51? (the actual 15m binary question)
  T3 HARVEST   continuous + Garleanu-Pedersen-smoothed position; gross vs NET-of-spread P&L,
               Sharpe, turnover, and fraction of non-overlapping 15m windows with net P&L>0
               (compare to break-even ~57% @0.75 payout). Trading framing: is the edge harvestable?

Gate (kickoff): integrated fast edge clears break-even (~57%) at 15m -> primary thesis; else closed.
Causal throughout; OOS-2026 is the held-out judge, TEST shown for robustness. No retraining.
"""
import sys, numpy as np, pandas as pd

NPZ = "models/probs_tickens_H3.npz"
TICK = "features_tick"
HORIZONS = [3, 5, 10, 30, 60, 120, 300, 600, 900]   # seconds
GAP = 10            # >GAP sec between bars => segment break (no return/carry across)
WIN = 900           # 15 minutes
BREAKEVEN = 0.5714  # 1/1.75  (0.75 payout)
d = np.load(NPZ)

def load(split):
    p = d[f"p{split[:2]}"] ; y = d[f"y{split[:2]}"]
    o = pd.read_parquet(f"{TICK}/{split}_1s.parquet")
    mid = o["mid"]; ret3 = mid.shift(-3)/mid - 1
    m = ((ret3 > 0).astype(float).where(ret3 != 0)).notna()      # exact mask used at save time
    assert int(m.sum()) == len(p), (int(m.sum()), len(p))
    b = o.loc[m, ["mid", "spread"]].copy()
    b["p"] = p ; b["y"] = y
    b["sig"] = np.sign(b["p"].values - 0.5)
    return o, b   # o = full series (for honest forward-price lookups), b = signal bars

def fwd_ret_at(full_mid, ts_index, h):
    """mid(t+h)/mid(t)-1 using nearest actual bar within tolerance (handles session gaps)."""
    left = pd.DataFrame({"t": ts_index + pd.Timedelta(seconds=h)}).sort_values("t")
    right = pd.DataFrame({"t": full_mid.index, "mfut": full_mid.values})
    tol = pd.Timedelta(seconds=max(2, int(h*0.15)))
    mg = pd.merge_asof(left, right, on="t", direction="nearest", tolerance=tol)
    out = pd.Series(mg["mfut"].values, index=ts_index[np.argsort(ts_index + pd.Timedelta(seconds=h))])
    return out.reindex(ts_index)

def acc(sig, fr):
    v = np.isfinite(fr) & (fr != 0) & np.isfinite(sig) & (sig != 0)
    return ( (np.sign(fr[v]) == sig[v]).mean(), int(v.sum()) )

def run(split):
    full, b = load(split)
    fmid = full["mid"]
    ts = b.index
    sig = b["sig"].values
    print(f"\n================  {split.upper()}  n_signal_bars={len(b):,}  "
          f"span {ts.min().date()}->{ts.max().date()}  ================")
    sp = b["spread"].values; mid = b["mid"].values
    hs_ret = (sp/2.0)/mid                              # half-spread in return units
    print(f"spread: median={np.nanmedian(sp):.6g}  half-spread_ret median={np.nanmedian(hs_ret):.2e}  "
          f"1s-ret std={np.nanstd(np.diff(mid)/mid[:-1]):.2e}")

    # ---- T1 DECAY ----
    print("\n[T1 decay] acc of sign(p-.5) vs h-sec-ahead direction (binary, single fast call):")
    print(f"  {'h(s)':>6} {'acc':>8} {'n':>10}")
    for h in HORIZONS:
        mh = fwd_ret_at(fmid, ts, h).values          # mid(t+h)
        r = mh/mid - 1.0
        a, n = acc(sig, r)
        print(f"  {h:>6} {a:>8.4f} {n:>10,}")

    # ---- T2 AGG-BINARY ----  predict 15m endpoint from trailing aggregate of fast signals
    m900 = fwd_ret_at(fmid, ts, WIN).values
    r900 = m900/mid - 1.0
    valid900 = np.isfinite(r900) & (r900 != 0)
    s = pd.Series(b["p"].values - 0.5, index=ts)
    print("\n[T2 agg-binary] sign(trailing-mean(p-.5) over W sec) vs 15m-endpoint direction:")
    print(f"  {'W(s)':>6} {'acc15m':>8} {'cov':>8} {'n':>10}")
    base = (np.sign(r900[valid900])>0).mean()
    print(f"  {'base':>6} {max(base,1-base):>8.4f} {'-':>8} {int(valid900.sum()):>10,}  (always-up/down)")
    for W in [3, 30, 120, 300, 900]:
        agg = s.rolling(f"{W}s").mean().values
        pred = np.sign(agg)
        v = valid900 & np.isfinite(pred) & (pred != 0)
        a = (np.sign(r900[v]) == pred[v]).mean()
        print(f"  {W:>6} {a:>8.4f} {v.mean():>8.3f} {int(v.sum()):>10,}")
    # confident-tail: how many high-conf (|p-.5|) signals per 15m window?
    for cov in (0.0005, 0.001, 0.002):
        thr = np.quantile(np.abs(b["p"].values-0.5), 1-cov)
        nconf = int((np.abs(b["p"].values-0.5) >= thr).sum())
        span_s = (ts.max()-ts.min()).total_seconds()
        print(f"  conf@{cov:.2%}: thr={thr:.4f}  n_conf={nconf:,}  ~{nconf/(span_s/WIN):.3f} signals per 15m window")

    # ---- T3 HARVEST ----  continuous + GP-smoothed position, gross vs net of spread
    dts = np.diff(ts.view('i8'))/1e9
    same = np.concatenate([[False], dts <= GAP])          # bar i in same segment as i-1
    r1 = np.concatenate([mid[1:]/mid[:-1]-1, [0.0]])       # return from bar i to i+1
    r1[~np.concatenate([same[1:], [False]])] = 0.0         # zero if next bar is a new segment
    print("\n[T3 harvest] position from fast signal; cost = |dpos|*half-spread; per-second:")
    print(f"  {'rule':>14} {'a':>5} {'gross/yr':>9} {'net/yr':>9} {'Sharpe_net':>10} {'turnover/day':>12} {'win15m%':>8}")
    wins = pd.Series(np.zeros(len(ts)), index=ts)          # for window aggregation
    def sim(aim, a, label):
        pos = np.empty(len(ts)); pos[0] = 0.0
        for i in range(1, len(ts)):
            pos[i] = pos[i-1] if not same[i] else (1-a)*pos[i-1] + a*aim[i]
        dpos = np.abs(np.diff(pos, prepend=0.0))
        pnl = pos*r1 - dpos*np.nan_to_num(hs_ret)         # net per-second
        gross = pos*r1
        yrs = (ts.max()-ts.min()).total_seconds()/ (365.25*86400)
        net_yr = pnl.sum()/yrs ; gr_yr = gross.sum()/yrs
        sh = pnl.mean()/ (pnl.std()+1e-12) * np.sqrt(252*6.5*3600)  # ~per-second annualized
        turn = dpos.sum()/ ((ts.max()-ts.min()).total_seconds()/86400)
        w = pd.Series(pnl, index=ts).resample(f"{WIN}s").sum()
        win15 = (w[w!=0] > 0).mean()
        print(f"  {label:>14} {a:>5.2f} {gr_yr:>9.4f} {net_yr:>9.4f} {sh:>10.2f} {turn:>12.2f} {win15:>8.3f}")
        return net_yr, win15
    aim_sign = sig.copy()
    aim_conf = np.clip((b["p"].values-0.5)/0.05, -1, 1)   # confidence-scaled aim
    sim(aim_sign, 1.00, "sign full")                       # flip every bar (max turnover)
    for a in (0.30, 0.10, 0.03, 0.01):
        sim(aim_sign, a, "GP sign")
    for a in (0.10, 0.03):
        sim(aim_conf, a, "GP conf")
    # gross-only positive control at full sign, no cost already shown via gross/yr

if __name__ == "__main__":
    for split in (sys.argv[1:] or ["oos", "test"]):
        run(split)
    print("\nGATE: T2 acc15m > %.3f at usable coverage OR T3 net/yr>0 with win15m%% > %.3f  =>  F1 PASS"
          % (BREAKEVEN, BREAKEVEN))
