#!/usr/bin/env python3
"""SWEEP_MATRIX family feature blocks on the Deriv feed (all causal), keyed by bar_start_ts.

    deriv_features_family.py <out dir> <first day> <last day> [PAIR ...]

One block per non-closed SWEEP_MATRIX family whose inputs exist on Deriv's 1-second mid feed (no book, no volume).
Rows and bars are built exactly as in tools/deriv_dataset.py (one row per minute bar in which the pair ticked, index =
minute start, true UTC; decision at the bar close = start + 60 s). Every value of a row uses only ticks strictly before
the decision instant (the bar's own ticks and earlier); other pairs enter through their last minute close at or before
the decision (as-of backward, as deriv_dataset.cross_features). Windows count the pair's own bars (as v0.1's rolling
statistics) unless stated otherwise.

  tf_  B5  tick-rule signed flow: (up-ticks - down-ticks) / (up + down + 1) and (up - down) / sqrt(up + down + 1) over
           5/15/60/240 bars (a tick = a change of the 1-s mid; size ignored, unlike a return)
  sg_  B7  path signature, level 2: Levy area of (bar index, log price) over 15/60 bars (where in the window the move
           happened), scaled by w * sqrt(quadratic variation); Levy area of (log price, USD-basket index) over 15/60/240
           bars (lead-lag orientation), scaled by sqrt(QV_price * QV_basket)
  kf_  C2  Kalman forward filters: local-linear-trend slope and innovation at two fixed noise ratios (slow, fast), scaled by
           the 60-bar return std; a random-walk beta of the pair's 1-bar return on the USD-basket return (observation
           variance = trailing 1,440-bar variance) and the cumulative one-step-ahead residual over 15/60 bars
  rm_  C3  random-matrix eigen-residual: on the union minute grid, the 7 pairs' 1-minute USD-oriented returns are
           standardized and their correlation estimated over the previous 2,880 grid minutes, re-estimated every 60 grid
           minutes from strictly earlier data; modes above the Marchenko-Pastur edge are projected out; the pair's
           standardized residual is summed over 15/60/240 grid minutes (/ sqrt(w)), quote-oriented
  fd_  C6  fractional differencing of log close (d = 0.3, 0.5; fixed 1,440-lag window), z-scored over 1,440 bars
  cx_  E3  complexity / regime: permutation entropy (order 3, 4) over 60/240 bars, lag-1 autocorrelation over 60/240,
           variance ratio VR(5) over 240, and a two-state (calm / turbulent) forward-filtered volatility-regime
           probability whose state scales are trailing 7,200-bar quantiles of |return| (strictly earlier bars)
  tk_  E4  tick clock (information bars): log return over the last 100/500/2,000 changes of the mid and log1p of the
           seconds they took
  pa_  F2  price action: RSI(2) and RSI(14) on 1-bar closes, RSI(14) on wall-clock 15-minute steps, Bollinger %b over 20/60
           bars, NR7 on 15-bar ranges (flag and ratio to their mean), NY trading day (starts 17:00 ET): return since the
           day's open, position in the day's range so far, distance to the previous day's high / low / close (scaled by
           the 1,440-bar return std * sqrt(1,440))

The F4 family (residualized TARGET) is a training label, not a feature: tools/deriv_labels_resid.py.
"""
import json
import math
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deriv_dataset as dd  # noqa: E402

FAMILY_VERSION = 'deriv_features_family/0.1'
SIGN = {p: (-1.0 if p in dd.USD_BASE else 1.0) for p in dd.PAIRS}   # + = the non-USD currency gains vs USD


def roll_sum(x, w):
    return pd.Series(x).rolling(w, min_periods=max(1, w // 3)).sum().to_numpy()


def tick_flow(ts, px, bars):
    d = np.r_[0.0, np.diff(px)]
    m = (ts // 60) * 60
    up = pd.Series((d > 0).astype(np.int32)).groupby(m).sum().reindex(bars.index).fillna(0).to_numpy(np.float64)
    dn = pd.Series((d < 0).astype(np.int32)).groupby(m).sum().reindex(bars.index).fillna(0).to_numpy(np.float64)
    out = {}
    for w in (5, 15, 60, 240):
        su = pd.Series(up).rolling(w, min_periods=1).sum().to_numpy()
        sd = pd.Series(dn).rolling(w, min_periods=1).sum().to_numpy()
        out[f'tf_imb_{w}m'] = (su - sd) / (su + sd + 1.0)
        out[f'tf_net_{w}m'] = (su - sd) / np.sqrt(su + sd + 1.0)
    return out


def tick_clock(ts, px, dec):
    d = np.r_[0.0, np.diff(px)]
    ci = np.where(d != 0)[0]
    tc, pc = ts[ci], np.log(px[ci])
    k = np.searchsorted(tc, dec, side='left') - 1          # last change strictly before the decision instant
    out = {}
    for n in (100, 500, 2000):
        ok = (k - n) >= 0
        j = np.clip(k - n, 0, None)
        kk = np.clip(k, 0, None)
        out[f'tk_ret_{n}'] = np.where(ok, pc[kk] - pc[j], np.nan)
        out[f'tk_lsecs_{n}'] = np.where(ok, np.log1p(np.maximum(dec - tc[j], 0)), np.nan)
    return out


def levy_area(X, Y, w):
    """Levy area of the 2-D path (X, Y) over the last w increments ending at each row (NaN before w rows)."""
    n = len(X)
    dX = np.r_[0.0, np.diff(X)]
    dY = np.r_[0.0, np.diff(Y)]
    Xp = np.r_[X[0], X[:-1]]
    Yp = np.r_[Y[0], Y[:-1]]
    P = np.cumsum(Xp * dY)
    Q = np.cumsum(Yp * dX)
    A = np.full(n, np.nan)
    k = np.arange(w, n)
    s = k - w
    A[k] = 0.5 * ((P[k] - P[s]) - X[s] * (Y[k] - Y[s]) - (Q[k] - Q[s]) + Y[s] * (X[k] - X[s]))
    qx = pd.Series(dX ** 2).rolling(w, min_periods=w).sum().to_numpy()
    qy = pd.Series(dY ** 2).rolling(w, min_periods=w).sum().to_numpy()
    return A, qx, qy


def basket_at(dec, closes, pair):
    """USD-basket index (mean USD-oriented log-return path of the other pairs) at the target's decision instants."""
    incs = []
    for p, s in closes.items():
        if p == pair:
            continue
        lp = pd.Series(np.log(s.values), index=s.index).reindex(dec, method='ffill').to_numpy()
        incs.append(SIGN[p] * np.r_[np.nan, np.diff(lp)])
    inc = np.nanmean(np.vstack(incs), axis=0)
    inc[~np.isfinite(inc)] = 0.0
    return np.cumsum(inc), inc


def signatures(lc, B):
    out = {}
    T = np.arange(len(lc), dtype=np.float64)
    for w in (15, 60):
        A, _, qp = levy_area(T, lc, w)                       # qp = quadratic variation of the price leg
        out[f'sg_area_tx_{w}m'] = A / (w * np.sqrt(qp) + 1e-12)
    for w in (15, 60, 240):
        A, qx, qy = levy_area(lc, B, w)
        out[f'sg_area_xb_{w}m'] = A / (np.sqrt(qx * qy) + 1e-18)
    return out


def llt_gains(ql, qs, r=1.0, iters=5000):
    """Steady-state Kalman gain of the local-linear-trend model (level, slope) with noise ratios ql, qs (r = 1)."""
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Qm = np.diag([ql, qs])
    P = np.eye(2)
    for _ in range(iters):
        Pp = F @ P @ F.T + Qm
        S = (H @ Pp @ H.T)[0, 0] + r
        K = (Pp @ H.T / S).ravel()
        P = (np.eye(2) - np.outer(K, H)) @ Pp
    return K


def kalman(lc, r1, b_inc):
    out = {}
    sd60 = pd.Series(r1).rolling(60, min_periods=20).std().to_numpy() + 1e-12
    for name, (ql, qs) in {'slow': (1e-2, 1e-5), 'fast': (1e-1, 1e-3)}.items():
        k1, k2 = (float(v) for v in llt_gains(ql, qs))
        lev, slo = float(lc[0]), 0.0
        slope = np.empty(len(lc)); innov = np.empty(len(lc))
        for i, y in enumerate(lc.tolist()):
            pl = lev + slo
            e = y - pl
            lev = pl + k1 * e
            slo = slo + k2 * e
            slope[i] = slo; innov[i] = e
        out[f'kf_slope_{name}'] = slope / sd60
        out[f'kf_innov_{name}'] = innov / sd60
    # random-walk beta of the pair's 1-bar return on the basket's 1-bar return (one-step-ahead residual)
    R = pd.Series(r1).rolling(1440, min_periods=120).var().shift(1).to_numpy()
    beta, P, q = 0.0, 1.0, 1e-4
    betas = np.full(len(lc), np.nan); res = np.full(len(lc), np.nan)
    for i, (y, x, Ri) in enumerate(zip(r1.tolist(), b_inc.tolist(), R.tolist())):
        P += q
        if not (math.isfinite(y) and math.isfinite(x) and math.isfinite(Ri) and Ri > 0):
            betas[i] = beta
            continue
        res[i] = y - beta * x                         # residual against the beta known before this bar
        S = x * x * P + Ri
        K = P * x / S
        beta += K * (y - beta * x)
        P = (1.0 - K * x) * P
        betas[i] = beta
    out['kf_beta_bask'] = betas
    rz = np.where(np.isfinite(res), res, 0.0)
    for w in (15, 60):
        out[f'kf_resid_{w}m'] = roll_sum(rz, w) / (sd60 * np.sqrt(w))
    return out


def rmt_residuals(closes, w_est=2880, step=60):
    """Grid-level standardized eigen-residuals of all pairs (columns in dd.PAIRS order), indexed by close time."""
    grid = np.unique(np.concatenate([s.index.values for s in closes.values()]))
    R = np.column_stack([SIGN[p] * np.r_[np.nan, np.diff(np.log(closes[p].reindex(grid, method='ffill').to_numpy()))]
                         for p in dd.PAIRS])
    R[~np.isfinite(R)] = 0.0
    n, N = R.shape
    lam_plus = (1.0 + np.sqrt(N / w_est)) ** 2
    resid = np.full((n, N), np.nan)
    modes = np.full(n, np.nan)
    for a in range(w_est, n, step):
        W = R[a - w_est:a]                                  # strictly earlier grid minutes
        sd = W.std(axis=0) + 1e-12
        Z = W / sd
        C = np.corrcoef(Z, rowvar=False)
        if not np.all(np.isfinite(C)):
            continue
        ev, V = np.linalg.eigh(C)
        Vs = V[:, ev > lam_plus]
        b = min(a + step, n)
        Zb = R[a:b] / sd
        resid[a:b] = Zb - (Zb @ Vs) @ Vs.T if Vs.shape[1] else Zb
        modes[a:b] = Vs.shape[1]
    return grid, resid, modes


def fracdiff(lc, d, L=1440):
    w = np.empty(L); w[0] = 1.0
    for k in range(1, L):
        w[k] = -w[k - 1] * (d - k + 1) / k
    y = fftconvolve(lc - lc[0], w)[:len(lc)]
    y[:L - 1] = np.nan
    s = pd.Series(y)
    return ((s - s.rolling(1440, min_periods=480).mean()) / (s.rolling(1440, min_periods=480).std() + 1e-12)).to_numpy()


def perm_entropy(r, m, w):
    n = len(r)
    S = np.column_stack([np.r_[np.full(j, np.nan), r[:n - j]] for j in range(m - 1, -1, -1)])   # oldest .. newest
    ok = np.all(np.isfinite(S), axis=1)
    ranks = np.argsort(np.argsort(np.where(np.isfinite(S), S, 0.0), axis=1, kind='stable'), axis=1, kind='stable')
    code = (ranks * (m ** np.arange(m))).sum(axis=1)
    uniq = np.unique(code[ok])
    idx = np.searchsorted(uniq, code)
    onehot = np.zeros((n, len(uniq)))
    onehot[np.arange(n)[ok], idx[ok]] = 1.0
    cnt = pd.DataFrame(onehot).rolling(w, min_periods=w // 2).sum().to_numpy()
    tot = cnt.sum(axis=1, keepdims=True)
    p = cnt / np.where(tot > 0, tot, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    h = h / np.log(math.factorial(m))
    h[~np.isfinite(tot.ravel()) | (tot.ravel() == 0)] = np.nan
    return h


def complexity(r1):
    out = {}
    for m in (3, 4):
        for w in (60, 240):
            out[f'cx_pe{m}_{w}m'] = perm_entropy(r1, m, w)
    s = pd.Series(r1)
    for w in (60, 240):
        out[f'cx_ac1_{w}m'] = s.rolling(w, min_periods=w // 2).corr(s.shift(1)).to_numpy()
    r5 = s.rolling(5, min_periods=5).sum()
    out['cx_vr5_240m'] = (r5.rolling(240, min_periods=120).var() / (5 * s.rolling(240, min_periods=120).var() + 1e-30)).to_numpy()
    a = s.abs()
    lo = (a.rolling(7200, min_periods=1440).quantile(0.25) * np.sqrt(np.pi / 2)).shift(1).to_numpy()
    hi = (a.rolling(7200, min_periods=1440).quantile(0.75) * np.sqrt(np.pi / 2)).shift(1).to_numpy()
    p_stay = 0.98
    post = 0.5
    ph = np.full(len(r1), np.nan)
    for i, (x, sl, sh) in enumerate(zip(r1.tolist(), lo.tolist(), hi.tolist())):
        if not (math.isfinite(x) and math.isfinite(sl) and math.isfinite(sh) and sl > 0 and sh > sl):
            continue
        prior = p_stay * post + (1 - p_stay) * (1 - post)
        l_hi = math.exp(-0.5 * (x / sh) ** 2) / sh
        l_lo = math.exp(-0.5 * (x / sl) ** 2) / sl
        num = prior * l_hi
        den = num + (1 - prior) * l_lo
        post = num / den if den > 0 else prior
        ph[i] = post
    out['cx_hmm_phigh'] = ph
    return out


def rsi(delta, n):
    g = pd.Series(np.clip(delta, 0, None)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    l = pd.Series(np.clip(-delta, 0, None)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return (100.0 - 100.0 / (1.0 + g / (l + 1e-18))).to_numpy()


def price_action(bars, lc, r1, dec):
    out = {}
    c = bars['close'].to_numpy(np.float64)
    out['pa_rsi2'] = rsi(np.nan_to_num(r1), 2)
    out['pa_rsi14'] = rsi(np.nan_to_num(r1), 14)
    known = pd.Series(lc, index=dec)
    steps = [known.reindex(dec - 900 * k, method='ffill').to_numpy() for k in range(15)]
    d15 = np.column_stack([steps[k] - steps[k + 1] for k in range(14)])
    gain = np.nanmean(np.clip(d15, 0, None), axis=1); loss = np.nanmean(np.clip(-d15, 0, None), axis=1)
    out['pa_rsi14_15m'] = 100.0 - 100.0 / (1.0 + gain / (loss + 1e-18))
    cs = pd.Series(c)
    for w in (20, 60):
        ma = cs.rolling(w, min_periods=w).mean(); sd = cs.rolling(w, min_periods=w).std()
        out[f'pa_bbpctb_{w}m'] = ((cs - (ma - 2 * sd)) / (4 * sd + 1e-18)).to_numpy()
    hi15 = bars['high'].rolling(15, min_periods=15).max(); lo15 = bars['low'].rolling(15, min_periods=15).min()
    rng = (hi15 - lo15).to_numpy()
    lagged = np.column_stack([np.r_[np.full(15 * k, np.nan), rng[:len(rng) - 15 * k]] for k in range(7)])
    out['pa_nr7_15m'] = np.where(np.all(np.isfinite(lagged), axis=1), (lagged[:, 0] <= np.min(lagged, axis=1)).astype(float), np.nan)
    out['pa_rng_ratio_15m'] = lagged[:, 0] / (np.nanmean(lagged, axis=1) + 1e-18)
    # NY trading day = 17:00 ET to 17:00 ET (the FX rollover); decision instant decides the day
    t = pd.to_datetime(dec, unit='s', utc=True).tz_convert('America/New_York')
    day = (t - pd.Timedelta(hours=17)).date
    df = pd.DataFrame({'day': day, 'c': c, 'h': bars['high'].to_numpy(np.float64), 'l': bars['low'].to_numpy(np.float64)})
    g = df.groupby('day', sort=False)
    dopen = g['c'].transform('first').to_numpy()
    dhi = g['h'].cummax().to_numpy(); dlo = g['l'].cummin().to_numpy()
    daily = g.agg(hi=('h', 'max'), lo=('l', 'min'), cl=('c', 'last'))
    prev = daily.shift(1).reindex(df['day'])
    scale = (pd.Series(r1).rolling(1440, min_periods=480).std() * np.sqrt(1440)).to_numpy() + 1e-12
    out['pa_ret_dayopen'] = (np.log(c) - np.log(dopen)) / scale
    out['pa_dayrange_pos'] = (c - dlo) / (dhi - dlo + 1e-18)
    for k, col in (('prevhi', 'hi'), ('prevlo', 'lo'), ('prevcl', 'cl')):
        out[f'pa_dist_{k}'] = (np.log(c) - np.log(prev[col].to_numpy())) / scale
    return out


def main():
    out = Path(sys.argv[1]); first = date.fromisoformat(sys.argv[2]); last = date.fromisoformat(sys.argv[3])
    pairs = sys.argv[4:] or dd.PAIRS
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    man = {'builder': FAMILY_VERSION, 'first_day': first.isoformat(), 'last_day': last.isoformat(), 'pairs': {}, 'inputs': {}}
    closes, files_used = {}, {}
    for p in dd.PAIRS:                                       # every pair's minute closes (basket, RMT)
        files_used[p] = dd.day_files(p, first, last)
        ts, px = dd.load_ticks(files_used[p])
        b = dd.minute_bars(ts, px)
        closes[p] = pd.Series(b['close'].to_numpy(np.float64), index=b.index.values.astype(np.int64) + 60)
        del ts, px, b
    print('closes loaded', f'{time.time() - t0:.0f}s', flush=True)
    grid, gres, gmodes = rmt_residuals(closes)
    print('rmt grid', len(grid), 'modes', np.nanmean(gmodes), f'{time.time() - t0:.0f}s', flush=True)
    for pair in pairs:
        ts, px = dd.load_ticks(files_used[pair])
        bars = dd.minute_bars(ts, px)
        start = bars.index.values.astype(np.int64)
        dec = start + 60
        lc = np.log(bars['close'].to_numpy(np.float64))
        r1 = np.r_[np.nan, np.diff(lc)]
        cols = {'bar_start_ts': start}
        cols.update(tick_flow(ts, px, bars))
        cols.update(tick_clock(ts, px, dec))
        del ts, px
        B, b_inc = basket_at(dec, closes, pair)
        cols.update(signatures(lc, B))
        cols.update(kalman(lc, r1, b_inc))
        j = dd.PAIRS.index(pair)
        zres = np.where(np.isfinite(gres[:, j]), gres[:, j], 0.0)
        for w in (15, 60, 240):
            cum = pd.Series(zres).rolling(w, min_periods=w // 3).sum().to_numpy() / np.sqrt(w)
            cum[~np.isfinite(gres[:, j])] = np.nan
            cols[f'rm_resid_{w}m'] = SIGN[pair] * pd.Series(cum, index=grid).reindex(dec).to_numpy()
        cols['rm_modes'] = pd.Series(gmodes, index=grid).reindex(dec).to_numpy()
        for d in (0.3, 0.5):
            cols[f'fd_z_{d}'] = fracdiff(lc, d)
        cols.update(complexity(r1))
        cols.update(price_action(bars, lc, r1, dec))
        df = pd.DataFrame(cols)
        feat = [c for c in df.columns if c != 'bar_start_ts']
        df[feat] = df[feat].astype('float32')
        df.to_parquet(out / f'{pair}.parquet', index=False)
        man['pairs'][pair] = {'rows': int(len(df)), 'features': feat,
                              'finite_share': {c: round(float(np.isfinite(df[c]).mean()), 4) for c in feat}}
        print(pair, len(df), len(feat), f'{time.time() - t0:.0f}s', flush=True)
    for p in dd.PAIRS:
        man['inputs'][p] = [{'file': f.name, 'bytes': f.stat().st_size} for f in files_used[p]]
    man['seconds'] = round(time.time() - t0, 1)
    (out / 'manifest.json').write_text(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
