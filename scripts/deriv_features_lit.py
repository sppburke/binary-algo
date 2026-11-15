#!/usr/bin/env python3
"""Literature-ranked feature blocks on the Deriv feed (all causal), keyed by bar_start_ts.

    deriv_features_lit.py <out dir> <first day> <last day> [PAIR ...]

From the literature consolidation (consolidated.jsonl ranks 1-3); rows and bars as tools/deriv_dataset.py, every value
uses ticks strictly before the decision (bar close = minute start + 60 s).
  fx_  fixing-window response (rank 1): for the London 16:00 fix (Europe/London wall clock, DST-correct) and the ECB
       14:15 Frankfurt reference rate (Europe/Berlin): minutes relative to the fix in [-180, 180] (NaN outside), triangular
       bases pre = (60 + m) / 60 on [-60, 0] and post = (60 - m) / 60 on (0, 60], their products with the USD-oriented
       15- and 60-minute returns (+ for USDJPY/USDCHF/USDCAD, - for the others), and for post-fix minutes the USD-oriented
       return over the hour before the fix (pre-registered hypotheses: pre-fix USD strength, post-fix reversal).
  ck_  compact return kernels (rank 2): wall-clock cumulative log returns over 5/15/30/60/240 min and exponentially
       weighted 1-bar return sums with half-lives 5/15/60 bars, each scaled by the lagged 1,440-bar return std times
       sqrt(window); fast-minus-slow differences (5, 30) and (15, 60) of the scaled cumulative returns; the lagged scale.
  rl_  round levels (rank 3): the deriv_features_extra round-number geometry (50 / 100-pip grids) on the real grid.
  rs_  the same geometry on a half-grid SHIFTED placebo grid (25 / 50 pips off): the falsifier compares rl_ with rs_.
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
from deriv_features_extra import round_number_features  # noqa: E402

LIT_VERSION = 'deriv_features_lit/0.1'
USD_SIGN = {p: (1.0 if p in dd.USD_BASE else -1.0) for p in dd.PAIRS}       # + = USD strengthens


def wallclock_returns(lc, dec, windows):
    known = pd.Series(lc, index=dec)
    return {w: lc - known.reindex(dec - 60 * w, method='ffill').to_numpy() for w in windows}


def fixing(dec, lc, pair):
    out = {}
    rets = wallclock_returns(lc, dec, (15, 60))
    u15, u60 = USD_SIGN[pair] * rets[15], USD_SIGN[pair] * rets[60]
    utc = pd.to_datetime(dec, unit='s', utc=True)
    known = pd.Series(lc, index=dec)
    for name, tz, hh, mm in (('ldn', 'Europe/London', 16, 0), ('ecb', 'Europe/Berlin', 14, 15)):
        loc = utc.tz_convert(tz)
        fix_local = loc.normalize() + pd.Timedelta(hours=hh, minutes=mm)          # same local day's fix
        fix_utc = fix_local.tz_convert('UTC')
        m = ((utc - fix_utc).total_seconds() / 60.0).to_numpy()
        rel = np.where(np.abs(m) <= 180, m, np.nan)
        pre = np.where((m >= -60) & (m <= 0), (60 + m) / 60.0, 0.0)
        post = np.where((m > 0) & (m <= 60), (60 - m) / 60.0, 0.0)
        out[f'fx_{name}_rel_min'] = rel
        out[f'fx_{name}_pre'] = pre
        out[f'fx_{name}_post'] = post
        out[f'fx_{name}_pre_x_u15'] = pre * u15
        out[f'fx_{name}_post_x_u15'] = post * u15
        out[f'fx_{name}_pre_x_u60'] = pre * u60
        out[f'fx_{name}_post_x_u60'] = post * u60
        fs = fix_utc.asi8 // 10**9 if hasattr(fix_utc, 'asi8') else np.asarray(fix_utc.view('int64')) // 10**9
        fs = np.asarray(pd.DatetimeIndex(fix_utc).as_unit('s').asi8)
        at_fix = known.reindex(fs, method='ffill').to_numpy()                    # last close at or before the fix
        before = known.reindex(fs - 3600, method='ffill').to_numpy()             # ... and one hour before it
        prefix_move = USD_SIGN[pair] * (at_fix - before)
        out[f'fx_{name}_prefix_u60'] = np.where((m > 0) & (m <= 60), prefix_move, np.nan)
    return out


def kernels(lc, dec):
    out = {}
    r1 = np.r_[np.nan, np.diff(lc)]
    scale = pd.Series(r1).rolling(1440, min_periods=480).std().shift(1).to_numpy() + 1e-12
    cums = wallclock_returns(lc, dec, (5, 15, 30, 60, 240))
    z = {}
    for w, v in cums.items():
        z[w] = v / (scale * np.sqrt(w))
        out[f'ck_cum_{w}m'] = z[w]
    r0 = np.nan_to_num(r1)
    for hl in (5, 15, 60):
        ew = pd.Series(r0).ewm(halflife=hl, adjust=False).mean().to_numpy() * (1.0 / (1 - 0.5 ** (1 / hl)))
        out[f'ck_ew_{hl}'] = ew / (scale * np.sqrt(hl))
    out['ck_fs_5_30'] = z[5] - z[30]
    out['ck_fs_15_60'] = z[15] - z[60]
    out['ck_scale'] = scale
    return out


def rounds(bars, pair):
    pip = 0.01 if pair == 'USDJPY' else 0.0001
    real = round_number_features(bars, pip)
    shifted_bars = bars.copy()
    for col in ('open', 'high', 'low', 'close'):
        shifted_bars[col] = bars[col] + 25 * pip                                # 50-grid shifted by half, 100-grid by a quarter
    sh = round_number_features(shifted_bars, pip)
    out = {('rl_' + k[3:]): v for k, v in real.items()}
    out.update({('rs_' + k[3:]): v for k, v in sh.items()})
    return out


def main():
    out = Path(sys.argv[1]); first = date.fromisoformat(sys.argv[2]); last = date.fromisoformat(sys.argv[3])
    pairs = sys.argv[4:] or dd.PAIRS
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    man = {'builder': LIT_VERSION, 'first_day': first.isoformat(), 'last_day': last.isoformat(), 'pairs': {},
           'allow_incomplete': __import__('os').environ.get('DERIV_ALLOW_INCOMPLETE') == '1'}
    for pair in pairs:
        ts, px = dd.load_ticks(dd.day_files(pair, first, last))
        bars = dd.minute_bars(ts, px)
        del ts, px
        start = bars.index.values.astype(np.int64)
        dec = start + 60
        lc = np.log(bars['close'].to_numpy(np.float64))
        cols = {'bar_start_ts': start}
        cols.update(fixing(dec, lc, pair))
        cols.update(kernels(lc, dec))
        cols.update(rounds(bars, pair))
        df = pd.DataFrame(cols)
        feat = [c for c in df.columns if c != 'bar_start_ts']
        df[feat] = df[feat].astype('float32')
        df.to_parquet(out / f'{pair}.parquet', index=False)
        man['pairs'][pair] = {'rows': int(len(df)), 'features': feat,
                              'finite_share': {c: round(float(np.isfinite(df[c]).mean()), 4) for c in feat}}
        print(pair, len(df), len(feat), f'{time.time() - t0:.0f}s', flush=True)
    man['skipped_incomplete_days'] = [list(x) for x in dd.SKIPPED_INCOMPLETE]
    man['seconds'] = round(time.time() - t0, 1)
    (out / 'manifest.json').write_text(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
