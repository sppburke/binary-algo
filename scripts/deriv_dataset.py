#!/usr/bin/env python3
"""Deriv-only dataset: minute decision rows with causal features and Deriv-faithful 15m/30m labels.

    deriv_dataset.py <out dir> <first day> <last day> [PAIR ...]

Inputs: ONLY /mnt/data/deriv/ticks (Deriv's own 1-second mid ticks, true UTC). Read-only.
One row per minute bar (indexed at the minute START, true UTC) in which the pair ticked. The
decision is taken at the bar CLOSE (start + 60 s). Labels use the repo's settlement rule
(scripts/min1_production.wc_ret semantics): entry = first tick at/after close + 1 s (within TOL),
exit = last tick at/before entry target + H (within TOL), ret = exit/entry - 1, ties stay valid
and LOSE. Features use only ticks at or before the bar close of the target pair; other pairs enter
through their own last price at or before that instant (as-of join backward, never forward), with
a staleness column, so no fake-flat target bars are created.

Writes <out dir>/<PAIR>.parquet and <out dir>/manifest.json (inputs with sizes and checksums,
parameters, row counts, label sanity per pair).
"""
import hashlib
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ARCHIVE = Path('/mnt/data/deriv/ticks')
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD']
USD_BASE = {'USDJPY', 'USDCHF', 'USDCAD'}          # USD is the base currency: invert for a "USD strength" view
HORIZONS = {15: (900, 30), 30: (1800, 60)}       # minutes -> (seconds, tolerance seconds)
LAG_S = 1
BUILDER_VERSION = 'deriv_dataset/0.1'


def day_files(pair, first, last):
    d, out = first, []
    while d <= last:
        f = ARCHIVE / pair / f'{pair}_{d.isoformat()}_ticks.parquet'
        if f.exists():
            out.append(f)
        d += timedelta(days=1)
    return out


def load_ticks(files):
    parts = []
    for f in files:
        t = pq.read_table(f, columns=['datetime_utc', 'price'])
        parts.append((t.column(0).cast('int64').to_numpy() // 1_000_000_000, t.column(1).to_numpy()))
    ts = np.concatenate([p[0] for p in parts]); px = np.concatenate([p[1] for p in parts]).astype(np.float64)
    order = np.argsort(ts, kind='stable')
    return ts[order], px[order]


def minute_bars(ts, px):
    m = (ts // 60) * 60
    df = pd.DataFrame({'m': m, 'px': px})
    g = df.groupby('m', sort=True)['px']
    bars = pd.DataFrame({'open': g.first(), 'high': g.max(), 'low': g.min(), 'close': g.last(), 'n': g.size()})
    chg = np.r_[False, np.diff(px) != 0]
    bars['nchg'] = pd.Series(chg).groupby(m).sum().reindex(bars.index).to_numpy()
    return bars                                    # index = minute start (epoch seconds)


def settle(dec_t, ts, px, horizon_s, tol_s):
    """Deriv settlement for decisions at dec_t (epoch s): wc_ret semantics with the entry target dec_t + LAG."""
    n = len(ts); entry_t = dec_t + LAG_S
    ei = np.searchsorted(ts, entry_t, side='left'); exit_t = entry_t + horizon_s
    xi = np.searchsorted(ts, exit_t, side='right') - 1
    eic = np.clip(ei, 0, n - 1); xic = np.clip(xi, 0, n - 1)
    valid = (ei < n) & (xi > ei) & ((ts[eic] - entry_t) <= tol_s) & ((exit_t - ts[xic]) <= tol_s)
    ret = np.where(valid, px[xic] / px[eic] - 1.0, np.nan)
    valid &= np.isfinite(ret)                      # as wc_ret: a non-finite return is not a valid label
    return ret, valid


def own_features(bars):
    c = bars['close']; lc = np.log(c); r1 = lc.diff()
    X = pd.DataFrame(index=bars.index)
    close_t = bars.index.values.astype(np.int64) + 60          # each bar's close time = its decision time
    known = pd.Series(lc.values, index=close_t)
    for w in (1, 2, 5, 10, 15, 30, 60, 120, 240, 480, 1440):
        then = known.reindex(close_t - 60 * w, method='ffill').values   # last close at or before t - w minutes
        X[f'ret_{w}m'] = lc.values - then           # wall-clock lookback, not a count of bars
    for w in (15, 30, 60, 240, 1440):
        X[f'rv_{w}m'] = r1.rolling(w, min_periods=max(5, w // 3)).std()
    for w in (15, 60, 240, 1440):
        hi = bars['high'].rolling(w, min_periods=max(5, w // 3)).max(); lo = bars['low'].rolling(w, min_periods=max(5, w // 3)).min()
        X[f'rangepos_{w}m'] = (c - lo) / (hi - lo + 1e-12)
    for w in (15, 60, 240):
        ema = c.ewm(span=w, adjust=False).mean()
        X[f'emadist_{w}m'] = c / ema - 1
        X[f'stretch_{w}m'] = X[f'emadist_{w}m'] / (r1.rolling(w, min_periods=5).std() * np.sqrt(w) + 1e-12)
    X['volratio_15_240'] = X['rv_15m'] / (X['rv_240m'] + 1e-12)
    X['volratio_60_1440'] = X['rv_60m'] / (X['rv_1440m'] + 1e-12)
    X['skew_240m'] = r1.rolling(240, min_periods=60).skew()
    X['kurt_240m'] = r1.rolling(240, min_periods=60).kurt()
    X['bar_range'] = (bars['high'] - bars['low']) / c
    X['bar_body'] = (bars['close'] - bars['open']) / c
    for w in (1, 5, 15, 60):
        X[f'activity_{w}m'] = bars['nchg'].rolling(w, min_periods=1).sum() / (bars['n'].rolling(w, min_periods=1).sum() + 1e-9)
    X['ticks_per_bar_15m'] = bars['n'].rolling(15, min_periods=1).mean()
    gap = pd.Series(bars.index, index=bars.index).diff().fillna(60)
    X['bar_gap_s'] = gap.clip(upper=7 * 86400)  # seconds since the previous bar (session breaks)
    t = pd.to_datetime(bars.index + 60, unit='s', utc=True)   # decision time (bar close)
    X['hour_sin'] = np.sin(2 * np.pi * (t.hour + t.minute / 60) / 24); X['hour_cos'] = np.cos(2 * np.pi * (t.hour + t.minute / 60) / 24)
    X['dow'] = t.dayofweek.astype('float32')
    return X


def cross_features(target_close_t, closes):
    """For each other pair: log returns over 5/15/60 minutes from its own last price at/before the decision."""
    X = pd.DataFrame(index=target_close_t.index)
    usd = []
    for pair, s in closes.items():                  # s: close price indexed by bar CLOSE time (start + 60)
        lp = np.log(s)
        for w in (5, 15, 60):
            past = pd.Series(lp.values, index=s.index - 0)  # value known at its close time
            now = past.reindex(target_close_t.values, method='ffill')       # as-of, backward only
            then = past.reindex(target_close_t.values - 60 * w, method='ffill')
            X[f'x_{pair}_ret_{w}m'] = (now.values - then.values)
            if w == 15:
                sign = -1.0 if pair in USD_BASE else 1.0   # +: the currency gains vs USD
                usd.append(sign * (now.values - then.values))
        last_t = pd.Series(s.index.values, index=s.index).reindex(target_close_t.values, method='ffill')
        X[f'x_{pair}_stale_s'] = target_close_t.values - last_t.values
    if usd:
        X['x_usd_weak_15m'] = np.nanmean(np.vstack(usd), axis=0)   # average non-USD strength vs USD
    return X


def file_sha(f):
    h = hashlib.sha256()
    with open(f, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    out = Path(sys.argv[1]); first = date.fromisoformat(sys.argv[2]); last = date.fromisoformat(sys.argv[3])
    pairs = sys.argv[4:] or PAIRS
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ticks, files_used = {}, {}
    for pair in PAIRS:                              # every pair is needed for the cross-pair features
        files = day_files(pair, first, last)
        files_used[pair] = files
        ticks[pair] = load_ticks(files)
    bars = {p: minute_bars(*ticks[p]) for p in PAIRS}
    closes = {p: pd.Series(b['close'].values, index=b.index.values + 60) for p, b in bars.items()}
    manifest = {'builder': BUILDER_VERSION, 'first_day': first.isoformat(), 'last_day': last.isoformat(),
                'lag_s': LAG_S, 'horizons': {str(k): {'seconds': v[0], 'tol_s': v[1]} for k, v in HORIZONS.items()},
                'decision': 'bar close = minute start + 60 s, true UTC', 'pairs': {}, 'inputs': {}}
    for pair in pairs:
        b = bars[pair]
        X = own_features(b)
        dec = b.index.values.astype(np.int64) + 60
        target_close_t = pd.Series(dec, index=b.index)
        others = {p: s for p, s in closes.items() if p != pair}
        X = X.join(cross_features(target_close_t, others).set_index(b.index))
        ts, px = ticks[pair]
        sanity = {}
        for hmin, (hs, tol) in HORIZONS.items():
            ret, valid = settle(dec, ts, px, hs, tol)
            X[f'ret_fwd_{hmin}m'] = ret; X[f'valid_{hmin}m'] = valid
            moved = valid & (ret != 0)
            sanity[f'{hmin}m'] = {'rows': int(len(ret)), 'valid': int(valid.sum()), 'ties': int((valid & (ret == 0)).sum()),
                                  'moved': int(moved.sum()), 'moved_up_rate': round(float((ret[moved] > 0).mean()), 4)}
        X['decision_ts'] = dec
        X = X.astype({c: 'float32' for c in X.columns if X[c].dtype == np.float64 and not c.startswith('ret_fwd')})
        X.reset_index(names='bar_start_ts').to_parquet(out / f'{pair}.parquet', index=False)
        manifest['pairs'][pair] = {'rows': int(len(X)), 'columns': int(X.shape[1]), 'label_sanity': sanity}
        print(pair, manifest['pairs'][pair], flush=True)
    for pair in PAIRS:
        manifest['inputs'][pair] = [{'file': f.name, 'bytes': f.stat().st_size, 'sha256': file_sha(f)} for f in files_used[pair]]
    manifest['seconds'] = round(time.time() - t0, 1)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print('done in', manifest['seconds'], 's')


if __name__ == '__main__':
    main()
