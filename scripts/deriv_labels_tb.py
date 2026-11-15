#!/usr/bin/env python3
"""Triple-barrier FIRST-TOUCH training labels on the Deriv minute grid (training target only).

    deriv_labels_tb.py <out dir> <first day> <last day> [PAIR ...]

Repository definition (scripts/usdjpy_15m_tbfirsttouch.py), applied to Deriv minute closes built exactly as in
tools/deriv_dataset.py (index = minute start, true UTC; decision at the bar close):
  sigma_t = trailing std of 1-minute simple returns over the last 900 bars (min 60), causal;
  barriers = close_t * (1 +/- k * sigma_t);
  window   = the next H minute closes, required to be the H consecutive wall-clock minutes (no gaps);
  label    = 1 if the upper barrier is touched strictly first, 0 if the lower is, else the sign of the Deriv
             settlement return (ret_fwd_H > 0; ties -> 0); NaN where the window is not contiguous, sigma is
             undefined, or the settlement label is invalid.
Columns: bar_start_ts, tb{k}_{H}m for k in (1, 2) and H in (15, 30). Validation, thresholds and every
reported number keep using the Deriv settlement sign; this file only changes what the model is fit on.
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

TB_VERSION = 'deriv_labels_tb/0.1'
VOL_WIN, VOL_MINP = 900, 60
KS = (1, 2)


def first_touch(c, contig, sig, end_up, H, k):
    n = len(c)
    cpad = np.concatenate([c, np.full(H, np.nan)])
    fwd = np.lib.stride_tricks.sliding_window_view(cpad, H)[1:n + 1]
    c0 = c[:, None]
    cmax = np.fmax.accumulate(np.nan_to_num(fwd, nan=-np.inf), axis=1)
    cmin = np.fmin.accumulate(np.nan_to_num(fwd, nan=np.inf), axis=1)
    up_hit = cmax >= c0 * (1.0 + k * sig[:, None])
    dn_hit = cmin <= c0 * (1.0 - k * sig[:, None])
    tu = np.where(up_hit.any(axis=1), up_hit.argmax(axis=1), H)
    td = np.where(dn_hit.any(axis=1), dn_hit.argmax(axis=1), H)
    y = end_up.astype(float)
    y[tu < td] = 1.0
    y[td < tu] = 0.0
    y[~contig | ~np.isfinite(sig) | (sig <= 0) | ~np.isfinite(end_up)] = np.nan
    return y


def main():
    out = Path(sys.argv[1]); first = date.fromisoformat(sys.argv[2]); last = date.fromisoformat(sys.argv[3])
    pairs = sys.argv[4:] or dd.PAIRS
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    man = {'builder': TB_VERSION, 'first_day': first.isoformat(), 'last_day': last.isoformat(), 'vol_win': VOL_WIN,
           'vol_minp': VOL_MINP, 'ks': list(KS), 'pairs': {}}
    for pair in pairs:
        ts, px = dd.load_ticks(dd.day_files(pair, first, last))
        bars = dd.minute_bars(ts, px)
        start = bars.index.values.astype(np.int64)
        c = bars['close'].to_numpy(np.float64)
        r1 = np.full(len(c), np.nan)
        r1[1:] = c[1:] / c[:-1] - 1.0
        r1[1:][np.diff(start) != 60] = np.nan                      # no return across a gap
        sig = pd.Series(r1).rolling(VOL_WIN, min_periods=VOL_MINP).std().to_numpy()
        dec = start + 60
        cols = {'bar_start_ts': start}
        stats = {}
        for H, (hs, tol) in dd.HORIZONS.items():
            ret, valid = dd.settle(dec, ts, px, hs, tol)
            end_up = np.where(valid, (ret > 0).astype(float), np.nan)
            # contiguous: the bar H minutes later exists exactly H*60 s after this one, with every minute between
            pos = np.searchsorted(start, start + 60 * H)
            ok = pos < len(start)
            contig = np.zeros(len(start), bool)
            contig[ok] = (start[pos[ok]] == start[ok] + 60 * H) & (pos[ok] - np.arange(len(start))[ok] == H)
            for k in KS:
                y = first_touch(c, contig, sig, end_up, H, k)
                cols[f'tb{k}_{H}m'] = y.astype(np.float32)
                fin = np.isfinite(y)
                agree = (y[fin & valid & (ret != 0)] == (ret[fin & valid & (ret != 0)] > 0)).mean()
                stats[f'tb{k}_{H}m'] = {'labelled': int(fin.sum()), 'up_rate': round(float(np.nanmean(y)), 4),
                                         'agrees_with_settlement_sign': round(float(agree), 4)}
        pd.DataFrame(cols).to_parquet(out / f'{pair}.parquet', index=False)
        man['pairs'][pair] = {'rows': int(len(start)), 'labels': stats}
        print(pair, man['pairs'][pair], f'{time.time() - t0:.0f}s', flush=True)
    man['seconds'] = round(time.time() - t0, 1)
    (out / 'manifest.json').write_text(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
