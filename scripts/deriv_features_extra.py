#!/usr/bin/env python3
"""Extra Deriv-native feature families from the literature fan-out (all causal), keyed by bar_start_ts.

    deriv_features_extra.py <out dir> <first day> <last day> [PAIR ...]

Families (minute bars and Deriv-settled 15m returns built exactly as in tools/deriv_dataset.py):
  slot_*   same-time-of-day seasonality: for the decision's America/New_York minute-of-day slot, the mean (and up share)
           of the Deriv-settled 15m return at that slot over the previous 5 / 20 trading days that have it (strictly
           earlier local dates; a slot's return is realized 15 minutes after its own decision, so it is known by the next
           day). Mechanism (literature): recurring session flows such as fixings create repeating signed pressure.
  rn_*     round-number geometry: pips to the big figure (100 pips) and half figure (50 pips) above and below, scaled by
           recent 1-minute volatility in pips; number of 50-pip grid crossings in the last 60 minutes; signed approach
           (last 15-minute return times proximity). Mechanism: take-profit clusters oppose moves at a level, stop clusters
           extend moves through it.
  sv_*     realized semivariances of 1-minute log returns over 60 and 240 minutes and their signed asymmetry
           (pos - neg) / (pos + neg). Mechanism: asymmetric recent moves summarize signed pressure.
Pip size: 0.01 for USDJPY, 0.0001 otherwise.
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

EXTRA_VERSION = 'deriv_features_extra/0.1'


def slot_features(start, ret15, valid):
    t = pd.to_datetime(start + 60, unit='s', utc=True).tz_convert('America/New_York')   # decision time, NY local
    df = pd.DataFrame({'d': t.date, 'slot': t.hour * 60 + t.minute, 'r': np.where(valid, ret15, np.nan)})
    piv = df.pivot_table(index='d', columns='slot', values='r', aggfunc='first')          # one decision per slot per day
    out = {}
    for n in (5, 20):
        mean = piv.shift(1).rolling(n, min_periods=max(2, n // 2)).mean()                   # strictly earlier days
        up = (piv > 0).where(piv.notna()).astype(float).shift(1).rolling(n, min_periods=max(2, n // 2)).mean()
        key = pd.MultiIndex.from_arrays([df['d'], df['slot']])
        out[f'slot_ret_{n}d'] = mean.stack(future_stack=True).reindex(key).to_numpy()
        out[f'slot_up_{n}d'] = up.stack(future_stack=True).reindex(key).to_numpy()
    return out


def round_number_features(bars, pip):
    c = bars['close'].to_numpy(np.float64)
    p = c / pip
    lr = np.r_[np.nan, np.diff(np.log(c))]
    vol_pips = pd.Series(np.r_[np.nan, np.diff(p)]).rolling(60, min_periods=20).std().to_numpy() * np.sqrt(15) + 1e-9
    out = {}
    for L in (100, 50):
        pos = np.mod(p, L)
        out[f'rn_up{L}_z'] = (L - pos) / vol_pips
        out[f'rn_dn{L}_z'] = pos / vol_pips
    grid = np.floor(p / 50)
    cross = np.r_[0, (np.diff(grid) != 0).astype(float)]
    out['rn_cross50_60m'] = pd.Series(cross).rolling(60, min_periods=1).sum().to_numpy()
    r15 = pd.Series(lr).rolling(15, min_periods=5).sum().to_numpy()
    near = np.minimum(out['rn_up50_z'], out['rn_dn50_z'])
    out['rn_approach'] = np.sign(r15) / (1.0 + near)
    return out


def semivariance_features(bars):
    c = bars['close'].to_numpy(np.float64)
    lr = pd.Series(np.r_[np.nan, np.diff(np.log(c))])
    out = {}
    for w in (60, 240):
        pos = (lr.clip(lower=0) ** 2).rolling(w, min_periods=w // 3).sum()
        neg = (lr.clip(upper=0) ** 2).rolling(w, min_periods=w // 3).sum()
        out[f'sv_pos_{w}m'] = pos.to_numpy()
        out[f'sv_neg_{w}m'] = neg.to_numpy()
        out[f'sv_asym_{w}m'] = ((pos - neg) / (pos + neg + 1e-18)).to_numpy()
    return out


def main():
    out = Path(sys.argv[1]); first = date.fromisoformat(sys.argv[2]); last = date.fromisoformat(sys.argv[3])
    pairs = sys.argv[4:] or dd.PAIRS
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    man = {'builder': EXTRA_VERSION, 'first_day': first.isoformat(), 'last_day': last.isoformat(), 'pairs': {}}
    for pair in pairs:
        ts, px = dd.load_ticks(dd.day_files(pair, first, last))
        bars = dd.minute_bars(ts, px)
        start = bars.index.values.astype(np.int64)
        ret15, valid15 = dd.settle(start + 60, ts, px, *dd.HORIZONS[15])
        cols = {'bar_start_ts': start}
        cols.update(slot_features(start, ret15, valid15))
        cols.update(round_number_features(bars, 0.01 if pair == 'USDJPY' else 0.0001))
        cols.update(semivariance_features(bars))
        df = pd.DataFrame(cols)
        feat = [c for c in df.columns if c != 'bar_start_ts']
        df[feat] = df[feat].astype('float32')
        df.to_parquet(out / f'{pair}.parquet', index=False)
        man['pairs'][pair] = {'rows': int(len(df)), 'features': feat,
                              'finite_share': {c: round(float(np.isfinite(df[c]).mean()), 4) for c in feat}}
        print(pair, len(df), f'{time.time() - t0:.0f}s', flush=True)
    man['seconds'] = round(time.time() - t0, 1)
    (out / 'manifest.json').write_text(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
