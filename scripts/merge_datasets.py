#!/usr/bin/env python3
"""Union of two Deriv-only datasets on the same minute decisions.

    merge_datasets.py <out dir> <primary dataset dir> <secondary dataset dir> <prefix> [PAIR ...]

Keeps every column of the primary dataset (features, labels, decision_ts, bar_start_ts) and adds the
secondary dataset's FEATURE columns renamed <prefix><name>, inner-joined on bar_start_ts (rows present in
both). Labels are taken from the primary only; the secondary's labels must agree where both are valid
(checked and reported). Writes <out>/<PAIR>.parquet and <out>/manifest.json.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD']
MERGER_VERSION = 'merge_datasets/0.1'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    out, prim, sec, prefix = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    pairs = sys.argv[5:] or PAIRS
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    man = {'merger': MERGER_VERSION, 'primary': {'dir': str(prim), 'manifest_sha256': sha256(prim / 'manifest.json')},
           'secondary': {'dir': str(sec), 'manifest_sha256': sha256(sec / 'manifest.json'), 'prefix': prefix}, 'pairs': {}}
    for pair in pairs:
        a = pd.read_parquet(prim / f'{pair}.parquet')
        b = pd.read_parquet(sec / f'{pair}.parquet')
        labels = [c for c in b.columns if c.startswith(('ret_fwd_', 'valid_'))]
        feats = [c for c in b.columns if c not in labels and c not in ('bar_start_ts', 'decision_ts')]
        bb = b[['bar_start_ts'] + feats + labels].rename(columns={c: prefix + c for c in feats + labels})
        m = a.merge(bb, on='bar_start_ts', how='inner', validate='one_to_one')
        agree = {}
        for c in labels:
            both = m[c.replace('ret_fwd_', 'valid_')].to_numpy(bool) & m[prefix + c.replace('ret_fwd_', 'valid_')].to_numpy(bool) \
                if c.startswith('ret_fwd_') else None
            if both is not None:
                x, y = m[c].to_numpy()[both], m[prefix + c].to_numpy()[both]
                agree[c] = {'both_valid': int(both.sum()), 'max_abs_diff': float(np.nanmax(np.abs(x - y))) if both.any() else None}
        m = m.drop(columns=[prefix + c for c in labels])
        m.to_parquet(out / f'{pair}.parquet', index=False)
        man['pairs'][pair] = {'rows_primary': int(len(a)), 'rows_secondary': int(len(b)), 'rows': int(len(m)),
                              'added_features': len(feats), 'label_agreement': agree}
        print(pair, man['pairs'][pair], f'{time.time() - t0:.0f}s', flush=True)
    man['seconds'] = round(time.time() - t0, 1)
    (out / 'manifest.json').write_text(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
