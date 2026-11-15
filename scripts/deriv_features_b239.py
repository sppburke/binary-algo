#!/usr/bin/env python3
"""Deriv-only rebuild of the incumbent books' feature schemas, with Deriv-settled labels.

    deriv_features_b239.py <out dir> <first day> <last day> [PAIR ...]

Inputs: ONLY /mnt/data/deriv/ticks (Deriv 1-second mid ticks, true UTC). Read-only.
Minute bars come from tools/deriv_dataset.py (index = minute START, true UTC; the decision is taken at
the bar close = start + 60 s), shaped like scripts/pipeline.resample_1m output: open/high/low/close,
volume = number of Deriv ticks in the minute (the live path's tick_count proxy), gap_prev = minutes
since the previous available bar (first bar 1.0).
Features come from the repository's recipe code, unchanged:
  - pipeline.build_features(m1)                       -> the 239-column base schema (B239);
  - live_features.build_cross_pair_features(pair, closes, []) for EURUSD, GBPUSD and USDCHF on the joint
    seven-pair minute clock (closes.dropna()), joined like LiveFeatureBuilder._joined_row_frame
    (cross-pair frame first, then the base columns it does not already carry, left join).
EURUSD's 18 order-flow columns (OF_*) need a volume feed and cannot be built from mid ticks: absent.
Labels: deriv_dataset.settle at 15 and 30 minutes (entry first tick >= decision + 1 s, exit last tick
<= entry target + H, ties valid). Writes <out>/<PAIR>.parquet and <out>/manifest.json.
"""
import hashlib
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

TOOLS = Path(__file__).resolve().parent
REPO = Path('/home/sean/git/binary-algo')
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(REPO / 'scripts'))
import deriv_dataset as dd  # noqa: E402
import pipeline  # noqa: E402
import live_features  # noqa: E402

BUILDER_VERSION = 'deriv_features_b239/0.1'
XPAIR = ('EURUSD', 'GBPUSD', 'USDCHF')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def m1_frame(bars):
    idx = pd.to_datetime(bars.index.values.astype(np.int64), unit='s', utc=True)
    m = pd.DataFrame({'open': bars['open'].values, 'high': bars['high'].values, 'low': bars['low'].values,
                      'close': bars['close'].values, 'volume': bars['n'].values.astype(float)}, index=idx)
    m.index.name = 'datetime_utc'
    dt_min = m.index.to_series().diff().dt.total_seconds().div(60.0)
    m['gap_prev'] = dt_min.fillna(1.0).values
    return m


def main():
    out = Path(sys.argv[1]); first = date.fromisoformat(sys.argv[2]); last = date.fromisoformat(sys.argv[3])
    pairs = sys.argv[4:] or dd.PAIRS
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ticks, files_used, m1 = {}, {}, {}
    for pair in dd.PAIRS:                       # all seven are needed for the cross-pair blocks
        files_used[pair] = dd.day_files(pair, first, last)
        ticks[pair] = dd.load_ticks(files_used[pair])
        m1[pair] = m1_frame(dd.minute_bars(*ticks[pair]))
    closes = pd.DataFrame({p: m1[p]['close'] for p in live_features.PAIRS}).dropna()
    manifest = {'builder': BUILDER_VERSION, 'first_day': first.isoformat(), 'last_day': last.isoformat(),
                'lag_s': dd.LAG_S, 'horizons': {str(k): {'seconds': v[0], 'tol_s': v[1]} for k, v in dd.HORIZONS.items()},
                'decision': 'bar close = minute start + 60 s, true UTC', 'volume': 'ticks per minute (Deriv tick count)',
                'recipe_code': {'pipeline.py': sha256(REPO / 'scripts' / 'pipeline.py'),
                                'live_features.py': sha256(REPO / 'scripts' / 'live_features.py'),
                                'deriv_dataset.py': sha256(TOOLS / 'deriv_dataset.py')},
                'joint_clock_rows': int(len(closes)), 'pairs': {}, 'inputs': {}}
    for pair in pairs:
        base = pipeline.build_features(m1[pair])
        if pair in XPAIR:
            xp = live_features.build_cross_pair_features(pair, closes, [])
            X = xp.join(base[[c for c in base.columns if c not in xp.columns]], how='left')
            kind = 'cross-pair block on the joint seven-pair clock + B239 base'
        else:
            X = base
            kind = 'B239 base on the pair clock'
        X = X.astype('float32')
        start = X.index.as_unit('s').asi8           # epoch seconds whatever the index resolution (pandas 3 infers it)
        dec = start + 60
        ts, px = ticks[pair]
        sanity = {}
        for hmin, (hs, tol) in dd.HORIZONS.items():
            ret, valid = dd.settle(dec, ts, px, hs, tol)
            X[f'ret_fwd_{hmin}m'] = ret
            X[f'valid_{hmin}m'] = valid
            moved = valid & (ret != 0)
            sanity[f'{hmin}m'] = {'rows': int(len(ret)), 'valid': int(valid.sum()), 'ties': int((valid & (ret == 0)).sum()),
                                  'moved': int(moved.sum()), 'moved_up_rate': round(float((ret[moved] > 0).mean()), 4)}
        X['decision_ts'] = dec
        X.insert(0, 'bar_start_ts', start)
        X.reset_index(drop=True).to_parquet(out / f'{pair}.parquet', index=False)
        feats = [c for c in X.columns if c not in ('bar_start_ts', 'decision_ts') and not c.startswith(('ret_fwd_', 'valid_'))]
        manifest['pairs'][pair] = {'rows': int(len(X)), 'features': len(feats), 'kind': kind, 'label_sanity': sanity}
        print(pair, manifest['pairs'][pair], f'{time.time() - t0:.0f}s', flush=True)
    for pair in dd.PAIRS:
        manifest['inputs'][pair] = [{'file': f.name, 'bytes': f.stat().st_size, 'sha256': dd.file_sha(f)} for f in files_used[pair]]
    manifest['seconds'] = round(time.time() - t0, 1)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print('done in', manifest['seconds'], 's')


if __name__ == '__main__':
    main()
