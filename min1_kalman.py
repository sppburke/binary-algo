"""1-MIN KALMAN experiments (user idea) — causal FILTER only (NO smoother; the smoother peeks at the future = leakage).

Three signals on the 1-minute EURUSD bar grid (label = sign(close[t+1min]-close[t]), contiguous 60s, ties excluded):
  (1) KALMAN-CHANNEL REVERSION: local-linear-trend Kalman gives a filtered "fair value" level; bet AGAINST the deviation
      (close - level) — the adaptive-band version of our compression x reversion lever.
  (2) KALMAN VELOCITY (trend): bet in the direction of the filtered slope — smoothed momentum (expected weak/regime-flipping).
  (3) KALMAN time-varying CROSS-PAIR beta: recursively estimate beta of EURUSD on the USD-basket return; residual reversion.
DISCIPLINE: select (process/measurement noise ratio, signal, threshold) on VAL=2022-2023 by worst-VAL-half stability; judge on
EACH of 2024/2025/2026 separately; non-overlapping next-bar trades (naturally disjoint 60s windows); bootstrap CI95; ties LOSE.
CAUSALITY: forward Kalman recursion only — state at t uses observations <= t. Never the RTS smoother.
"""
import os, numpy as np, pandas as pd, time
import harness as H

FEAT = "/media/sean/CORSAIR/binary-algo/features"
PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"]
USD_BASE = {"USDJPY", "USDCHF", "USDCAD"}
NONEU = [p for p in PAIRS if p != "EURUSD"]
SPL = {"val": ["2022", "2023"], "2024": ["2024"], "2025": ["2025"], "2026": ["2026"]}

def boot(c, nb=4000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def kalman_llt(y, q_level, q_slope, r):
    """Local-linear-trend Kalman FILTER (forward only). Returns filtered (level, slope) per step, causal. y = log price."""
    n = len(y)
    F = np.array([[1.0, 1.0], [0.0, 1.0]]); H_ = np.array([1.0, 0.0])
    Q = np.array([[q_level, 0.0], [0.0, q_slope]]); R = r
    x = np.array([y[0], 0.0]); P = np.eye(2) * 1e-4
    lev = np.empty(n); slp = np.empty(n)
    for t in range(n):
        # predict
        x = F @ x; P = F @ P @ F.T + Q
        # update with obs y[t]
        S = H_ @ P @ H_ + R; K = (P @ H_) / S
        x = x + K * (y[t] - H_ @ x); P = P - np.outer(K, H_ @ P)
        lev[t] = x[0]; slp[t] = x[1]
    return lev, slp

def kalman_beta(x, y, q, r):
    """Time-varying regression beta_t of y on x (both returns), forward Kalman. Returns filtered beta + residual y - beta*x (causal)."""
    n = len(x); beta = 0.0; P = 1.0; bser = np.empty(n)
    for t in range(n):
        P = P + q                      # random-walk beta
        S = x[t] * P * x[t] + r; K = P * x[t] / S
        beta = beta + K * (y[t] - beta * x[t]); P = (1 - K * x[t]) * P
        bser[t] = beta
    return bser

def load_year(y):
    p = f"{FEAT}/EURUSD_{y}.parquet"
    cols = list(dict.fromkeys(["close"] + list(H.META_COLS)))   # META_COLS already includes 'close' -> dedupe
    d = pd.read_parquet(p, columns=cols); d = d[~d.index.duplicated(keep="last")]
    idx = d.index; c = d["close"].values; n = len(c)
    secs = idx.values.astype("datetime64[s]").astype("int64")
    contig = np.zeros(n, bool); contig[:n-1] = (secs[1:] - secs[:-1]) == 60
    fwd = np.full(n, np.nan); fwd[:n-1] = c[1:]; ret = fwd / c - 1.0
    valid = contig & np.isfinite(ret) & (ret != 0)
    return c, secs, (ret > 0).astype(int), valid

def basket_year(y):
    """USD-weakness basket 1-min return aligned to EURUSD bars (eu-equivalent)."""
    cl = {}
    for p in PAIRS:
        fp = f"{FEAT}/{p}_{y}.parquet"
        if not os.path.exists(fp): return None, None
        d = pd.read_parquet(fp, columns=["close"]); d = d[~d.index.duplicated(keep="last")]; cl[p] = d["close"]
    df = pd.DataFrame(cl).dropna()
    lr = {p: np.log(df[p].values) for p in PAIRS}
    eu = np.concatenate([[0.0], lr["EURUSD"][1:] - lr["EURUSD"][:-1]])
    bk = []
    for p in NONEU:
        r = np.concatenate([[0.0], lr[p][1:] - lr[p][:-1]]); bk.append((-1.0 if p in USD_BASE else 1.0) * r)
    basket = np.nanmean(np.vstack(bk), axis=0)
    return eu, basket, df.index

def evalwin(sig, y, valid, thr):
    """selective: trade when |sig|>=thr, bet sign(sig); next-bar labels are disjoint -> no de-overlap needed. ties already dropped."""
    m = valid & (np.abs(sig) >= thr) & np.isfinite(sig)
    if m.sum() < 20: return (0, float("nan"), np.array([]))
    pred_up = (sig[m] > 0).astype(int)
    corr = (pred_up == y[m]).astype(float)
    return (int(m.sum()), float(corr.mean()), corr)

def main():
    t0 = time.time()
    # precompute per-window signals for the two single-series Kalman models (channel-revert, velocity)
    Q_GRID = [(1e-8, 1e-10), (1e-7, 1e-9), (1e-6, 1e-8)]; R = 1e-6
    data = {}
    for w, yrs in SPL.items():
        cs, ss, ys, vs, lev_rev, lev_trd = [], [], [], [], {}, {}
        cc = []; yy = []; vv = []
        sig_rev = {i: [] for i in range(len(Q_GRID))}; sig_trd = {i: [] for i in range(len(Q_GRID))}
        for y in yrs:
            c, secs, yb, valid = load_year(y); lp = np.log(c)
            cc.append(c); yy.append(yb); vv.append(valid)
            for i, (ql, qs) in enumerate(Q_GRID):
                lev, slp = kalman_llt(lp, ql, qs, R)
                sig_rev[i].append(-(lp - lev))   # revert: bet against deviation from fair value
                sig_trd[i].append(slp)           # trend: bet with filtered slope
        data[w] = dict(y=np.concatenate(yy), v=np.concatenate(vv),
                       rev=[np.concatenate(sig_rev[i]) for i in range(len(Q_GRID))],
                       trd=[np.concatenate(sig_trd[i]) for i in range(len(Q_GRID))])
        print(f"[kalman] window {w} built n={len(data[w]['y'])} {time.time()-t0:.0f}s", flush=True)

    def report(name, getter):
        # select (Q-index, thr) on VAL by worst-half (split val by index halves), then judge 2024/25/26
        V = data["val"]; vh = len(V["y"]) // 2
        best = None
        for i in range(len(Q_GRID)):
            sig = getter(V, i)
            for q in (0.5, 0.6, 0.7, 0.8, 0.9):
                thr = float(np.nanquantile(np.abs(sig[np.isfinite(sig)]), q))
                m = V["v"] & (np.abs(sig) >= thr) & np.isfinite(sig)
                idx = np.where(m)[0]
                if len(idx) < 40: continue
                h1 = idx[idx < vh]; h2 = idx[idx >= vh]
                if len(h1) < 20 or len(h2) < 20: continue
                a1 = ((sig[h1] > 0).astype(int) == V["y"][h1]).mean(); a2 = ((sig[h2] > 0).astype(int) == V["y"][h2]).mean()
                hm = min(a1, a2)
                if best is None or hm > best[0]: best = (hm, i, thr)
        if best is None: print(f"  {name}: no eligible VAL config"); return
        hm, i, thr = best
        line = f"  {name} (Qidx={i} thr={thr:.2e} VALhalf={hm:.3f}):"
        accs = []
        for w in ("2024", "2025", "2026"):
            D = data[w]; n, a, c = evalwin(getter(D, i), D["y"], D["v"], thr)
            if n < 20: line += f" {w}:thin"; accs.append(np.nan); continue
            lo, hi = boot(c); accs.append(a); line += f" {w}:{a:.3f}(n{n},CI[{lo:.3f},{hi:.3f}])"
        line += f"  FLOOR={np.nanmin(accs):.3f}"
        print(line, flush=True)

    print("\n[kalman] CAUSAL FILTER signals (no smoother), VAL-selected, judged 2024/2025/2026:", flush=True)
    report("channel-REVERT", lambda D, i: D["rev"][i])
    report("velocity-TREND ", lambda D, i: D["trd"][i])

    # (3) time-varying cross-pair beta residual reversion
    print("\n[kalman] time-varying cross-pair beta residual:", flush=True)
    bdata = {}
    for w, yrs in SPL.items():
        res = []; yy = []; vv = []
        for y in yrs:
            eu, basket, bidx = basket_year(y)
            if eu is None: continue
            beta = kalman_beta(basket, eu, 1e-6, 1e-6)
            resid = eu - beta * basket            # EURUSD move beyond its basket-implied move
            # label = next-bar EURUSD direction on this aligned index
            c = np.exp(np.cumsum(np.concatenate([[np.log(1.0)], eu[1:]])))  # not needed; use eu sign of next bar
            secs = bidx.values.astype("datetime64[s]").astype("int64"); n = len(eu)
            contig = np.zeros(n, bool); contig[:n-1] = (secs[1:] - secs[:-1]) == 60
            ynext = np.zeros(n, int); ynext[:n-1] = (eu[1:] > 0).astype(int)
            valid = contig & (np.concatenate([eu[1:], [0]]) != 0)
            res.append(-resid)  # revert: bet against the idiosyncratic deviation
            yy.append(ynext); vv.append(valid)
        if res:
            bdata[w] = dict(sig=np.concatenate(res), y=np.concatenate(yy), v=np.concatenate(vv))
    if "val" in bdata:
        V = bdata["val"]; vh = len(V["y"]) // 2; best = None
        sig = V["sig"]
        for q in (0.5, 0.6, 0.7, 0.8, 0.9):
            thr = float(np.nanquantile(np.abs(sig[np.isfinite(sig)]), q))
            idx = np.where(V["v"] & (np.abs(sig) >= thr) & np.isfinite(sig))[0]
            if len(idx) < 40: continue
            h1 = idx[idx < vh]; h2 = idx[idx >= vh]
            if len(h1) < 20 or len(h2) < 20: continue
            a1 = ((sig[h1] > 0).astype(int) == V["y"][h1]).mean(); a2 = ((sig[h2] > 0).astype(int) == V["y"][h2]).mean()
            hm = min(a1, a2)
            if best is None or hm > best[0]: best = (hm, thr)
        if best is not None:
            hm, thr = best; line = f"  xpair-beta-REVERT (thr={thr:.2e} VALhalf={hm:.3f}):"; accs = []
            for w in ("2024", "2025", "2026"):
                D = bdata.get(w)
                if D is None: line += f" {w}:NA"; continue
                n, a, c = evalwin(D["sig"], D["y"], D["v"], thr)
                if n < 20: line += f" {w}:thin"; accs.append(np.nan); continue
                lo, hi = boot(c); accs.append(a); line += f" {w}:{a:.3f}(n{n},CI[{lo:.3f},{hi:.3f}])"
            line += f"  FLOOR={np.nanmin(accs):.3f}"; print(line, flush=True)
    print(f"\n[kalman] DONE {time.time()-t0:.0f}s  (need n>=25 & CI95-lo>0.65 on all 3 windows to clear the goal)", flush=True)

if __name__ == "__main__":
    main()
