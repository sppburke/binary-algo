#!/usr/bin/env python3
"""SWEEP_MATRIX C4 convergent cross-mapping (CCM) coupling block on the Deriv feed (causal), keyed by bar_start_ts.

    deriv_features_ccm.py <out dir> <first day> <last day> [PAIR ...]

Rows and bars exactly as tools/deriv_dataset.py (the target pair's own minute bars; decision at the bar close). For the
target's 1-bar log returns y and each other pair's log-close increments x_d between the target's consecutive decision
instants (as-of backward), every 15 bars the target's shadow manifold is embedded from the last 240 bars (E = 3, lag 1,
points ending at the current bar), each point's 4 nearest neighbours (itself excluded) give simplex weights
exp(-d / d_min), and the cross-mapped estimate of every driver is correlated with the driver over the library:
rho_d = corr(x_hat_d, x_d) (Sugihara et al. 2012: a driver's influence on the target is recoverable from the target's
manifold). Values are carried forward to the next 14 bars (computed only from bars at or before each row).
  cm_rho_<DRIVER>   CCM skill driver -> target for the six other pairs
  cm_rho_max, cm_rho_mean
  cm_push_15m       sum over drivers of max(rho_d, 0) * sign(corr(y, x_d) over the library) * the driver's last 15-bar
                    log change: the coupled drivers' recent push on the target
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deriv_dataset as dd  # noqa: E402

CCM_VERSION = 'deriv_features_ccm/0.1'
E, L, STEP, K = 3, 240, 15, 4


def ccm_block(y, X):
    """y: (n,) target returns; X: (n, D) driver increments. Returns rho (n, D) and sign of corr(y, x_d) (n, D) at the
    window-end rows (every STEP rows), NaN elsewhere."""
    n, D = X.shape
    rho = np.full((n, D), np.nan)
    sgn = np.full((n, D), np.nan)
    for end in range(L + E, n, STEP):
        idx = np.arange(end - L + 1, end + 1)                      # library points (inclusive of the current bar)
        M = np.column_stack([y[idx - j] for j in range(E)])         # shadow manifold of the target
        if not np.all(np.isfinite(M)):
            continue
        d2 = ((M[:, None, :] - M[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        nb = np.argpartition(d2, K, axis=1)[:, :K]
        dist = np.sqrt(np.take_along_axis(d2, nb, axis=1))
        dmin = np.maximum(dist[:, :1], 1e-15)
        w = np.exp(-dist / dmin)
        w /= w.sum(axis=1, keepdims=True)
        Xl = X[idx]
        if not np.all(np.isfinite(Xl)):
            continue
        xhat = np.einsum('ik,ikd->id', w, Xl[nb])
        for d in range(D):
            a, b = xhat[:, d], Xl[:, d]
            if a.std() > 0 and b.std() > 0:
                rho[end, d] = np.corrcoef(a, b)[0, 1]
            yy = y[idx]
            if yy.std() > 0 and b.std() > 0:
                sgn[end, d] = np.sign(np.corrcoef(yy, b)[0, 1])
    return rho, sgn


def main():
    out = Path(sys.argv[1]); first = date.fromisoformat(sys.argv[2]); last = date.fromisoformat(sys.argv[3])
    pairs = sys.argv[4:] or dd.PAIRS
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    closes, bars_by = {}, {}
    for p in dd.PAIRS:
        ts, px = dd.load_ticks(dd.day_files(p, first, last))
        b = dd.minute_bars(ts, px)
        closes[p] = pd.Series(np.log(b['close'].to_numpy(np.float64)), index=b.index.values.astype(np.int64) + 60)
        bars_by[p] = b.index.values.astype(np.int64)
        del ts, px, b
    man = {'builder': CCM_VERSION, 'first_day': first.isoformat(), 'last_day': last.isoformat(),
           'params': {'E': E, 'library': L, 'step': STEP, 'neighbours': K}, 'pairs': {}}
    for pair in pairs:
        start = bars_by[pair]
        dec = start + 60
        y = np.r_[np.nan, np.diff(closes[pair].to_numpy())]
        drivers = [p for p in dd.PAIRS if p != pair]
        lvl = np.column_stack([closes[p].reindex(dec, method='ffill').to_numpy() for p in drivers])
        Xd = np.vstack([np.full(len(drivers), np.nan), np.diff(lvl, axis=0)])
        Xd[~np.isfinite(Xd)] = 0.0
        rho, sgn = ccm_block(y, Xd)
        rho_f = pd.DataFrame(rho).ffill(limit=STEP - 1).to_numpy()
        sgn_f = pd.DataFrame(sgn).ffill(limit=STEP - 1).to_numpy()
        cols = {'bar_start_ts': start}
        for j, p in enumerate(drivers):
            cols[f'cm_rho_{p}'] = rho_f[:, j]
        cols['cm_rho_max'] = np.nanmax(rho_f, axis=1)
        cols['cm_rho_mean'] = np.nanmean(rho_f, axis=1)
        push15 = lvl - np.vstack([np.full((15, len(drivers)), np.nan), lvl[:-15]])
        cols['cm_push_15m'] = np.nansum(np.clip(rho_f, 0, None) * sgn_f * push15, axis=1)
        cols['cm_push_15m'][~np.isfinite(cols['cm_rho_max'])] = np.nan
        df = pd.DataFrame(cols)
        feat = [c for c in df.columns if c != 'bar_start_ts']
        df[feat] = df[feat].astype('float32')
        df.to_parquet(out / f'{pair}.parquet', index=False)
        man['pairs'][pair] = {'rows': int(len(df)), 'features': feat,
                              'finite_share': {c: round(float(np.isfinite(df[c]).mean()), 4) for c in feat},
                              'rho_median': {c: round(float(np.nanmedian(df[c])), 4) for c in feat if c.startswith('cm_rho_')}}
        print(pair, len(df), f'{time.time() - t0:.0f}s', man['pairs'][pair]['rho_median'], flush=True)
    man['seconds'] = round(time.time() - t0, 1)
    (out / 'manifest.json').write_text(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
