#!/usr/bin/env python3
"""Hysteretic play-operator feature blocks on the Deriv minute grid (discovery D1-03; all causal).

    deriv_features_play.py <out dir> <first day> <last day> [PAIR ...]

For each decision (minute close i, true UTC; minute bars as tools/deriv_dataset.py) and replay length R:
  sigma_15 = sqrt(900/3600 * sum of the 60 squared one-minute log returns before the replay starts);
  widths a_j = k_j * sigma_15, k_j in (0.5, 1, 2);
  play operator over the replay's log closes x_u: p(start) = x_start, p(u) = max(x_u - a, min(x_u + a, p(u-1)));
  g_j = (x_i - p_j(i)) / sigma_15                      signed displacement inside the hysteresis band;
  h_j = (p_j(i) - p_j(i - 15 min)) / sigma_15          centre movement over the last 15 minutes (reset block: since
                                                       the replay start);
  s_j = -g_j * [p_j unchanged over the last 5 updates] stalled-centre displacement (relaxation hypothesis sign).
Block hy_ replays the last 240 minutes (return-point memory); block hr_ is the 15-minute-reset control. A decision
whose sigma window and replay are not contiguous wall-clock minutes, or whose sigma_15 is 0, gets NaN.
Columns: bar_start_ts, hy_{g,h,s}{05,1,2}, hr_{g,h,s}{05,1,2}.
"""
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deriv_dataset as dd  # noqa: E402

PLAY_VERSION = 'deriv_features_play/0.1'
KS = (0.5, 1.0, 2.0)
KNAME = ('05', '1', '2')
SIG_N = 60          # one-minute returns in the sigma window
STALL = 5           # updates without a centre change
LOOK_H = 15         # minutes for the centre movement


def block(x, start, R, prefix):
    n = len(x)
    i = np.arange(n)
    span = SIG_N + R                                   # bars i-span .. i must be contiguous minutes
    ok = i >= span
    ok[ok] = (start[i[ok]] - start[i[ok] - span]) == 60 * span
    lo = np.where(ok, i - span, 0)
    r = np.diff(x, prepend=np.nan)                     # r[m] = x[m] - x[m-1]
    rv = np.zeros(n)
    for m in range(1, SIG_N + 1):                      # returns at bars lo+1 .. lo+60 (before the replay start lo+61)
        rv += r[np.clip(lo + m, 0, n - 1)] ** 2
    sig = np.sqrt(0.25 * rv)
    ok &= np.isfinite(sig) & (sig > 0)
    a = np.array(KS)[:, None] * sig[None, :]          # (3, n)
    s0 = i - R + 1                                     # replay start bar
    p = np.repeat(x[np.clip(s0, 0, n - 1)][None, :], len(KS), axis=0)
    p_h = p.copy()                                     # centre LOOK_H minutes before i (or at the replay start)
    changed = np.zeros_like(p, dtype=bool)
    for u in range(1, R):
        xu = x[np.clip(s0 + u, 0, n - 1)][None, :]
        new = np.maximum(xu - a, np.minimum(xu + a, p))
        if u >= R - STALL:
            changed |= new != p
        p = new
        if u == R - 1 - LOOK_H:
            p_h = p.copy()
    g = (x[None, :] - p) / sig[None, :]
    h = (p - p_h) / sig[None, :]
    s = -g * (~changed)
    out = {}
    for j, kn in enumerate(KNAME):
        for name, v in (('g', g[j]), ('h', h[j]), ('s', s[j])):
            out[f'{prefix}{name}{kn}'] = np.where(ok, v, np.nan).astype(np.float32)
    return out


def main():
    out = Path(sys.argv[1]); first = date.fromisoformat(sys.argv[2]); last = date.fromisoformat(sys.argv[3])
    pairs = sys.argv[4:] or dd.PAIRS
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    man = {'builder': PLAY_VERSION, 'first_day': first.isoformat(), 'last_day': last.isoformat(), 'ks': list(KS),
           'blocks': {'hy_': 240, 'hr_': 15}, 'pairs': {}}
    for pair in pairs:
        ts, px = dd.load_ticks(dd.day_files(pair, first, last))
        bars = dd.minute_bars(ts, px)
        start = bars.index.values.astype(np.int64)
        x = np.log(bars['close'].to_numpy(np.float64))
        cols = {'bar_start_ts': start}
        cols.update(block(x, start, 240, 'hy_'))
        cols.update(block(x, start, 15, 'hr_'))
        df = pd.DataFrame(cols)
        df.to_parquet(out / f'{pair}.parquet', index=False)
        feat = [c for c in df.columns if c != 'bar_start_ts']
        man['pairs'][pair] = {'rows': int(len(df)), 'features': feat,
                              'finite_share': {c: round(float(np.isfinite(df[c]).mean()), 4) for c in feat}}
        print(pair, len(df), {c: man['pairs'][pair]['finite_share'][c] for c in ('hy_g1', 'hr_g1')}, f'{time.time() - t0:.0f}s',
              flush=True)
    man['skipped_incomplete_days'] = [list(x) for x in dd.SKIPPED_INCOMPLETE]
    man['allow_incomplete'] = os.environ.get('DERIV_ALLOW_INCOMPLETE') == '1'
    man['seconds'] = round(time.time() - t0, 1)
    (out / 'manifest.json').write_text(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
