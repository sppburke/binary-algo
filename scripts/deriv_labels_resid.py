#!/usr/bin/env python3
"""Residualized TRAINING target (SWEEP_MATRIX F4) on the Deriv minute grid (training target only).

    deriv_labels_resid.py <out dir> <dataset dir> <first day> <last day> [PAIR ...]

For each decision of the dataset (bar close of the pair's own minute bar, true UTC):
  own   = the Deriv-settled H-minute return of the dataset (ret_fwd_Hm where valid_Hm) - the label every reported
          number keeps using;
  bask  = the USD-basket return over the same interval: mean over the other six pairs of the USD-oriented change of
          their log minute close, as-of (decision + 1 s + H) minus as-of (decision + 1 s);
  beta  = OLS slope of the pair's bar-to-bar log return on the basket's increment over the previous 7,200 bars (strictly
          earlier bars only);
  resid_Hm = 1 if own - beta * bask > 0, 0 if < 0, NaN where own is invalid, beta is undefined or the residual is 0.
Mechanism (SWEEP_MATRIX F4): the common USD factor is hard to forecast and dominates the label; fitting the pair's
idiosyncratic part may expose pair-specific structure. Columns: bar_start_ts, resid_15m, resid_30m.
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
from deriv_features_family import SIGN, basket_at  # noqa: E402

RESID_VERSION = 'deriv_labels_resid/0.1'


def main():
    out, ds = Path(sys.argv[1]), Path(sys.argv[2])
    first, last = date.fromisoformat(sys.argv[3]), date.fromisoformat(sys.argv[4])
    pairs = sys.argv[5:] or dd.PAIRS
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    closes = {}
    for p in dd.PAIRS:
        ts, px = dd.load_ticks(dd.day_files(p, first, last))
        b = dd.minute_bars(ts, px)
        closes[p] = pd.Series(b['close'].to_numpy(np.float64), index=b.index.values.astype(np.int64) + 60)
        del ts, px, b
    man = {'builder': RESID_VERSION, 'dataset': str(ds), 'first_day': first.isoformat(), 'last_day': last.isoformat(), 'pairs': {}}
    for pair in pairs:
        d = pd.read_parquet(ds / f'{pair}.parquet', columns=['bar_start_ts', 'decision_ts', 'ret_fwd_15m', 'valid_15m',
                                                              'ret_fwd_30m', 'valid_30m'])
        own_c = closes[pair]
        dec = own_c.index.values
        _, b_inc = basket_at(dec, closes, pair)
        r_own = np.r_[np.nan, np.diff(np.log(own_c.to_numpy()))]
        s_y, s_x = pd.Series(r_own), pd.Series(b_inc)
        cov = s_y.rolling(7200, min_periods=1440).cov(s_x).shift(1)
        var = s_x.rolling(7200, min_periods=1440).var().shift(1)
        beta = pd.Series((cov / var.where(var > 0)).to_numpy(), index=dec)
        cols = {'bar_start_ts': d['bar_start_ts'].to_numpy(np.int64)}
        dts = d['decision_ts'].to_numpy(np.int64)
        bt = beta.reindex(dts).to_numpy()
        info = {'rows': int(len(d)), 'beta_finite': int(np.isfinite(bt).sum()), 'beta_median': float(np.nanmedian(bt))}
        for H in (15, 30):
            fwd = []
            for p, s in closes.items():
                if p == pair:
                    continue
                lp = pd.Series(np.log(s.to_numpy()), index=s.index)
                a = lp.reindex(dts + 1, method='ffill').to_numpy()
                b = lp.reindex(dts + 1 + 60 * H, method='ffill').to_numpy()
                fwd.append(SIGN[p] * (b - a))
            bask = np.nanmean(np.vstack(fwd), axis=0)
            own = np.where(d[f'valid_{H}m'].to_numpy(bool), d[f'ret_fwd_{H}m'].to_numpy(np.float64), np.nan)
            resid = own - bt * bask
            y = np.where(np.isfinite(resid) & (resid != 0), (resid > 0).astype(float), np.nan)
            cols[f'resid_{H}m'] = y.astype(np.float32)
            ok = np.isfinite(y) & np.isfinite(own) & (own != 0)
            info[f'{H}m'] = {'labelled': int(np.isfinite(y).sum()), 'up_share': float(np.nanmean(y)),
                             'agree_with_settlement_sign': float(((own[ok] > 0) == (y[ok] > 0.5)).mean())}
        pd.DataFrame(cols).to_parquet(out / f'{pair}.parquet', index=False)
        man['pairs'][pair] = info
        print(pair, info, f'{time.time() - t0:.0f}s', flush=True)
    man['seconds'] = round(time.time() - t0, 1)
    (out / 'manifest.json').write_text(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
