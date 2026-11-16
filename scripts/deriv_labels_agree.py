#!/usr/bin/env python3
"""Entry-delay agreement training labels on the Deriv minute grid (discovery D1-02; training target only).

    deriv_labels_agree.py <out dir> <first day> <last day> [PAIR ...]

For each decision (minute close, true UTC; as tools/deriv_dataset.py) and horizon H, the Deriv contract is settled
with the dataset's semantics (entry = first tick at or after decision + d, exit = last tick at or before entry target
+ H, the same tolerances) for entry delays d in (1, 5, 15) seconds. agree3_{H}m = 1 if all three settle strictly UP,
0 if all three settle strictly DOWN, NaN otherwise (a disagreement, any tie or any invalid settlement). Validation,
thresholds and every reported number keep using the +1 s settlement sign; this file only changes what a model is fit
on. The d = 1 s settlement is asserted equal to deriv_dataset.settle. The manifest records, per pair and horizon, the
share of moved +1 s rows without an agreed label (the pre-registered pre-fit KILL is < 1%).
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

AGREE_VERSION = 'deriv_labels_agree/0.1'
DELAYS_S = (1, 5, 15)


def settle_lag(dec_t, ts, px, horizon_s, tol_s, lag):
    """deriv_dataset.settle with the entry delay as a parameter (identical arithmetic)."""
    n = len(ts)
    entry_t = dec_t + lag
    ei = np.searchsorted(ts, entry_t, side='left')
    exit_t = entry_t + horizon_s
    xi = np.searchsorted(ts, exit_t, side='right') - 1
    eic = np.clip(ei, 0, n - 1)
    xic = np.clip(xi, 0, n - 1)
    valid = (ei < n) & (xi > ei) & ((ts[eic] - entry_t) <= tol_s) & ((exit_t - ts[xic]) <= tol_s)
    ret = np.where(valid, px[xic] / px[eic] - 1.0, np.nan)
    valid &= np.isfinite(ret)
    return ret, valid


def main():
    out = Path(sys.argv[1]); first = date.fromisoformat(sys.argv[2]); last = date.fromisoformat(sys.argv[3])
    pairs = sys.argv[4:] or dd.PAIRS
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    assert dd.LAG_S == DELAYS_S[0]
    man = {'builder': AGREE_VERSION, 'first_day': first.isoformat(), 'last_day': last.isoformat(),
           'delays_s': list(DELAYS_S), 'pairs': {}}
    for pair in pairs:
        ts, px = dd.load_ticks(dd.day_files(pair, first, last))
        bars = dd.minute_bars(ts, px)
        start = bars.index.values.astype(np.int64)
        dec = start + 60
        cols = {'bar_start_ts': start}
        stats = {}
        for H, (hs, tol) in dd.HORIZONS.items():
            settled = [settle_lag(dec, ts, px, hs, tol, d) for d in DELAYS_S]
            r1, v1 = dd.settle(dec, ts, px, hs, tol)
            assert np.array_equal(v1, settled[0][1]) and np.array_equal(r1[v1], settled[0][0][v1]), 'd=1 s differs'
            allv = np.logical_and.reduce([v for _, v in settled])
            up = allv & np.logical_and.reduce([np.nan_to_num(r) > 0 for r, _ in settled])
            dn = allv & np.logical_and.reduce([np.nan_to_num(r) < 0 for r, _ in settled])
            y = np.full(len(dec), np.nan, np.float32)
            y[up] = 1.0
            y[dn] = 0.0
            cols[f'agree3_{H}m'] = y
            moved = v1 & (r1 != 0)
            stats[f'{H}m'] = {'moved_rows': int(moved.sum()), 'agreed': int(np.isfinite(y).sum()),
                              'moved_without_agreed_label': round(float((~np.isfinite(y[moved])).mean()), 4),
                              'agreed_up_rate': round(float(np.nanmean(y)), 4),
                              'agreed_matches_1s_sign': round(float((y[moved & np.isfinite(y)] == (r1[moved & np.isfinite(y)] > 0)).mean()), 6)}
        pd.DataFrame(cols).to_parquet(out / f'{pair}.parquet', index=False)
        man['pairs'][pair] = {'rows': int(len(start)), 'labels': stats}
        print(pair, man['pairs'][pair], f'{time.time() - t0:.0f}s', flush=True)
    man['skipped_incomplete_days'] = [list(x) for x in dd.SKIPPED_INCOMPLETE]
    man['seconds'] = round(time.time() - t0, 1)
    (out / 'manifest.json').write_text(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
