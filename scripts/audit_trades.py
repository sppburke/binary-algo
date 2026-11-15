#!/usr/bin/env python3
"""Per-trade artifact audit of frozen books scored on a Deriv-only dataset (as in score_frozen_books.py).

    audit_trades.py --dataset DIR --books ID,ID,... --session ny|nylate [--h 15] --out audit.json [--trades trades.parquet]

For each book: the same rows, gate and first-come de-overlap as score_frozen_books.py, then per trade: decision time,
side, probability, Deriv-settled return, win (ties lose), tick count in the settlement window, entry/exit tick lags,
share of seconds with a price change in the window, move size in pips. Reports win rate (iid bootstrap CI95) for:
  all trades; TRADEABLE trades only (decision outside Deriv's 15m FX purchase blackout: a contract whose expiry lands in
  16:50-18:00 New York is rejected; last accepted start 16:34:40 NY, first rejected 16:35:10 NY - docs/DERIV_DEMO_EXECUTOR.md
  (g) - so decisions in [16:35, 18:00) NY are excluded); by New York hour; by weekday; by month; by move-size quintile;
and the same breakdowns for tick density (stale-quote check).
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

TOOLS = Path(__file__).resolve().parent
REPO = Path('/home/sean/git/binary-algo')
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(REPO / 'scripts'))
import book_runtime as br  # noqa: E402
import deriv_dataset as dd  # noqa: E402
import score_frozen_books as sfb  # noqa: E402
from sessions import session_mask  # noqa: E402

PIP = {'USDJPY': 0.01}


def tradeable(ts):
    t = pd.to_datetime(ts, unit='s', utc=True).tz_convert('America/New_York')
    m = t.hour.values * 60 + t.minute.values
    return ~((m >= 16 * 60 + 35) & (m < 18 * 60))


def ci(win):
    lo, hi = sfb.boot(win)
    return [int(len(win)), round(float(np.mean(win)), 4) if len(win) else None, round(lo, 4), round(hi, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--books', required=True)
    ap.add_argument('--session', default='nylate', choices=('ny', 'nylate'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--trades', default=None)
    a = ap.parse_args()
    index = {e['id']: e for e in json.loads((REPO / 'books' / 'INDEX.json').read_text())['books']}
    res, all_trades, ticks_cache = {'dataset': a.dataset, 'session': a.session, 'books': {}}, [], {}
    for bid in a.books.split(','):
        e = index[bid]
        pair, h = e['currency'], int(e['timeframe'].rstrip('m'))
        gap = 60 * h
        strategy = json.loads(br.load_registered_strategy_path(bid).read_text())
        boosters = [lgb.Booster(model_file=str(p)) for p in br._load_registered_model_paths(bid)]
        cols, _ = sfb.feature_names(strategy, boosters)
        metrics = e.get('metrics') if isinstance(e.get('metrics'), dict) else {}
        cov, _ = br._resolve_coverage(bid, strategy, metrics)
        thr = br._conf_thr(strategy, cov)
        bb_col, bb_thr = sfb.compression_gate(strategy)
        df = pd.read_parquet(Path(a.dataset) / f'{pair}.parquet')
        ts = df['decision_ts'].to_numpy(np.int64)
        fwd = df[f'ret_fwd_{h}m'].to_numpy(np.float64)
        valid = df[f'valid_{h}m'].to_numpy(bool) & np.isfinite(fwd)
        Xdf = df[cols].astype('float64')
        finite = np.isfinite(Xdf.to_numpy()).all(axis=1)
        if a.session == 'ny':
            sess = session_mask(ts, 'ny')
        else:
            hh = pd.to_datetime(ts, unit='s', utc=True).tz_convert('America/New_York').hour.values
            sess = (hh >= 13) & (hh < 22)
        rows = np.where(sess & valid & finite)[0]
        rows = rows[np.argsort(ts[rows], kind='stable')]
        p = np.mean([b.predict(pd.DataFrame(Xdf.to_numpy()[rows], columns=cols)) for b in boosters], axis=0)
        f, t = fwd[rows], ts[rows]
        gate = ~(np.abs(p - 0.5) < thr)
        if bb_col:
            gate &= ~(df[bb_col].to_numpy(np.float64)[rows] > bb_thr)
        if pair not in ticks_cache:
            first = pd.to_datetime(ts.min(), unit='s', utc=True).date()
            last = pd.to_datetime(ts.max(), unit='s', utc=True).date()
            ticks_cache[pair] = dd.load_ticks(dd.day_files(pair, first, last))
        tk, px = ticks_cache[pair]
        out = {'pair': pair, 'conf_thr': thr, 'coverage': cov}
        for side in ('UP', 'DOWN'):
            tr, win = sfb.stream(p, f, t, gate, side, gap)
            dec = t[tr]
            entry_t = dec + 1
            ei = np.searchsorted(tk, entry_t, side='left')
            xi = np.searchsorted(tk, entry_t + gap, side='right') - 1
            n_ticks = xi - ei + 1
            chg = np.array([int((np.diff(px[i:j + 1]) != 0).sum()) for i, j in zip(ei, xi)])
            pips = np.abs(f[tr]) * px[np.clip(ei, 0, len(px) - 1)] / PIP.get(pair, 0.0001)
            nyt = pd.to_datetime(dec, unit='s', utc=True).tz_convert('America/New_York')
            trade = pd.DataFrame({'book': bid, 'pair': pair, 'side': side, 'decision_ts': dec, 'p': p[tr], 'ret': f[tr], 'win': win,
                                  'ny_hour': nyt.hour, 'ny_weekday': nyt.dayofweek, 'month': nyt.strftime('%Y-%m'),
                                  'tradeable': tradeable(dec), 'n_ticks': n_ticks, 'n_changes': chg,
                                  'change_share': chg / np.maximum(n_ticks - 1, 1), 'entry_lag_s': tk[ei] - entry_t,
                                  'exit_lag_s': (entry_t + gap) - tk[xi], 'move_pips': pips})
            all_trades.append(trade)
            T = trade[trade.tradeable]
            q = pd.qcut(T.move_pips.rank(method='first'), 5, labels=False) if len(T) >= 5 else pd.Series(dtype=float)
            out[side] = {'all': ci(trade.win.to_numpy()), 'tradeable': ci(T.win.to_numpy()),
                         'untradeable_blackout': ci(trade[~trade.tradeable].win.to_numpy()),
                         'by_ny_hour': {int(k): ci(g.win.to_numpy()) for k, g in T.groupby('ny_hour')},
                         'by_weekday': {int(k): ci(g.win.to_numpy()) for k, g in T.groupby('ny_weekday')},
                         'by_month': {k: ci(g.win.to_numpy()) for k, g in T.groupby('month')},
                         'by_move_quintile': {int(k): [ci(g.win.to_numpy()), round(float(g.move_pips.median()), 2)] for k, g in T.groupby(q)} if len(T) >= 5 else {},
                         'tradeable_afternoon_13_1635': ci(T[T.ny_hour < 17].win.to_numpy()),
                         'tradeable_evening_18_22': ci(T[T.ny_hour >= 18].win.to_numpy()),
                         'median_ticks_in_window': float(np.median(n_ticks)) if len(n_ticks) else None,
                         'median_change_share': round(float(np.median(trade.change_share)), 3) if len(trade) else None,
                         'win_vs_change_share': {'wins': round(float(trade[trade.win == 1].change_share.median()), 3) if (trade.win == 1).any() else None,
                                                 'losses': round(float(trade[trade.win == 0].change_share.median()), 3) if (trade.win == 0).any() else None},
                         'ties': int((trade.ret == 0).sum())}
        res['books'][bid] = out
        print(bid, {s: (out[s]['all'], out[s]['tradeable'], out[s]['tradeable_afternoon_13_1635'], out[s]['tradeable_evening_18_22'])
                    for s in ('UP', 'DOWN')}, flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    if a.trades:
        pd.concat(all_trades).to_parquet(a.trades, index=False)


if __name__ == '__main__':
    main()
