#!/usr/bin/env python3
"""Payout-aware reading of a CPCV side (analysis; reading pre-registered in the output before the dump is opened).

    analyze_payout_aware.py <dump.npz> <pair> <quote_model_dir> <out.json> [--cov 0.03] [--result result.json]
                            [--original recorded_result.json] [--quote-dir DIR]

The dump comes from cpcv_deriv_x.py --dump-preds (15-minute key only: the quote model prices 15-minute contracts).
The model's trades are FROZEN exactly as cpcv_deriv.side_wr selects them (side filter, confidence >= the path's
validation threshold, first-come de-overlap with gap H). Each trade is then priced with the payout Deriv would have
quoted for its side at the decision, reconstructed by venue_quote_model from the pair's own ticks at or before the
decision (an ASSUMPTION outside 2026-07-02..2026-07-31: see that tool). A trade whose weekday x clock bucket held no
quote in the archive is NOT PRICED (no contract was on offer at that clock time in July) and is left out of every
payout statistic; its share is reported. Per path and side: win rate (ties lose), mean payout, mean breakeven faced,
expected value per unit staked = mean(win x payout) - 1, the same split by lower-paid (favoured) and higher-paid
(unfavoured) side and by payout floor, and the always-side baseline on the path's de-overlapped decisions.
--result checks that the frozen trades reproduce the harness's own per-path counts and win rates; --original does the
same against the candidate's recorded result. With --quote-dir, trades whose decision has a fresh matched quote
within 20 s are also priced with the ACTUAL quote (July 2026 only) and the reconstruction error is reported.

PRE-REGISTERED READING (per side, New York clock): PAYOUT-CLEARED iff at least 12 of the 15 CPCV paths have >= 25
priced trades, the expected value is above 0 on at least 12 of them and the 10th percentile of their expected values
is at least 0; otherwise NOT CLEARED AT THE QUOTED PAYOUT. ADVERSELY PRICED iff on at least 12 counted paths the
trades' mean payout is at least 0.02 below the always-side mean payout; FAVOURABLY PRICED iff at least 0.02 above on
at least 12; NEUTRAL otherwise. The fixed UTC-4 clock (winter sensitivity), the favoured/unfavoured split and the
payout floors are descriptive: they change no verdict and select nothing.
"""
import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpcv_deriv as cd  # noqa: E402
import deriv_dataset as dd  # noqa: E402
import venue_quote_model as vqm  # noqa: E402
import venue_quote_study as vqs  # noqa: E402

VERSION = 'analyze_payout_aware/0.1'
READING = ('per side, New York clock: PAYOUT-CLEARED iff >= 12 of 15 CPCV paths have >= 25 priced trades, expected value per '
           'unit staked at the reconstructed payout is above 0 on >= 12 of them and the p10 of their expected values is >= 0, '
           'else NOT CLEARED AT THE QUOTED PAYOUT; ADVERSELY PRICED iff on >= 12 counted paths the trades\' mean payout is '
           '>= 0.02 below the always-side mean payout, FAVOURABLY PRICED iff >= 0.02 above on >= 12, else NEUTRAL; the fixed '
           'UTC-4 clock, the favoured/unfavoured split and the payout floors are descriptive and change no verdict')
FLOORS = (1.75, 1.80, 1.85)


def stats(win, own, opp):
    n = len(win)
    if n == 0:
        return {'n': 0}
    return {'n': int(n), 'wins': int(win.sum()), 'wr': float(win.mean()), 'payout': float(own.mean()),
            'breakeven': float((1.0 / own).mean()), 'ev': float((win * own).mean() - 1.0),
            'favoured_share': float((own < opp).mean()), 'unfavoured_share': float((own > opp).mean())}


def pooled(rows, key):
    """Sum over the CPCV paths (a test group sits in five paths, so trades repeat across paths: descriptive only)."""
    rs = [r[key] for r in rows if r[key]['n'] > 0]
    n = sum(x['n'] for x in rs)
    if n == 0:
        return {'n': 0}
    w = sum(x['wins'] for x in rs)
    pay = sum(x['payout'] * x['n'] for x in rs) / n
    ev = sum(x['ev'] * x['n'] for x in rs) / n
    return {'n': int(n), 'wr': float(w / n), 'payout': float(pay), 'breakeven_of_mean_payout': float(1.0 / pay), 'ev': float(ev)}


def summarise(rows):
    counted = [r for r in rows if r['trades']['n'] >= cd.MIN_TRADES and r['always']['n'] > 0]
    ev = [r['trades']['ev'] for r in counted]; wr = [r['trades']['wr'] for r in counted]
    gapv = [r['trades']['payout'] - r['always']['payout'] for r in counted]
    mean = lambda k, part='trades': float(np.mean([r[part][k] for r in counted])) if counted else None   # noqa: E731
    return {'paths': len(rows), 'paths_counted': len(counted), 'paths_clear_0.541': int(sum(x >= cd.BE for x in wr)),
            'p10_win_rate': float(np.percentile(wr, 10)) if wr else None, 'paths_ev_positive': int(sum(x > 0 for x in ev)),
            'p10_ev': float(np.percentile(ev, 10)) if ev else None, 'mean_ev': float(np.mean(ev)) if ev else None,
            'min_ev': float(np.min(ev)) if ev else None, 'max_ev': float(np.max(ev)) if ev else None,
            'mean_win_rate': mean('wr'), 'mean_payout': mean('payout'), 'mean_breakeven_faced': mean('breakeven'),
            'mean_favoured_share': mean('favoured_share'), 'mean_always_payout': mean('payout', 'always'),
            'mean_always_ev': mean('ev', 'always'), 'mean_always_win_rate': mean('wr', 'always'),
            'paths_payout_0.02_below_always': int(sum(x <= -0.02 for x in gapv)),
            'paths_payout_0.02_above_always': int(sum(x >= 0.02 for x in gapv)),
            'trades_frozen': int(sum(r['frozen'] for r in rows)), 'trades_not_priced': int(sum(r['not_priced'] for r in rows))}


def reproduces(res, labels_rows, cov, side):
    """Do the frozen trades (before pricing) match a harness result's per-path [n, win rate] of this side?"""
    by = {str(x['test']): x for x in res['paths']}
    ok, seen, worst = True, 0, 0.0
    for lab, n, wr in labels_rows:
        if lab not in by or cov not in by[lab]['bycov']:
            continue
        seen += 1
        n_ref, wr_ref = by[lab]['bycov'][cov][side]
        same = (n_ref == n) and (n_ref == 0 or abs(wr_ref - wr) < 1e-9)
        ok &= bool(same)
        if n_ref and n:
            worst = max(worst, abs(wr_ref - wr))
    return {'paths_compared': seen, 'identical': bool(ok and seen > 0), 'max_abs_win_rate_difference': float(worst)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dump'); ap.add_argument('pair'); ap.add_argument('model_dir'); ap.add_argument('out')
    ap.add_argument('--cov', default='0.03'); ap.add_argument('--horizon', type=int, default=15)
    ap.add_argument('--result'); ap.add_argument('--original'); ap.add_argument('--quote-dir')
    ap.add_argument('--candidate', default=None, help='tag of the recorded candidate whose trades are priced (label only)')
    ap.add_argument('--first', default='2025-08-11'); ap.add_argument('--last', default='2026-07-31')
    a = ap.parse_args()
    assert a.horizon == 15, 'the quote model prices 15-minute contracts only'
    gap = a.horizon * 60
    out = {'analysis': VERSION, 'pair': a.pair, 'horizon_min': a.horizon, 'cov': a.cov, 'candidate': a.candidate,
           'registered_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'pre_registered_reading': READING, 'sides': {}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    model = vqm.load(a.model_dir); mm = model['meta']
    out['quote_model'] = {'dir': a.model_dir, 'tool': mm['tool'], 'sha256': mm['sha256'], 'rows': mm['rows'],
                          'pair_has_quote_lane': a.pair in mm['pairs'], 'assumptions': mm['assumptions']}
    if a.pair not in mm['pairs']:
        out['quote_model']['leave_one_pair_out_r2_skew'] = {p: v['r2_skew'] for p, v in mm['leave_one_pair_out'].items()}
    ts_k, px_k = dd.load_ticks(dd.day_files(a.pair, date.fromisoformat(a.first), date.fromisoformat(a.last)))
    assert ts_k.max() < vqs.CUTOFF, 'ticks outside development would be read'
    d = np.load(a.dump)
    labels = [str(x) for x in d['labels']]
    keep = [i for i, lab in enumerate(labels) if not lab.startswith('forward')]
    all_ts = np.unique(np.concatenate([d[f'ts_{i}'] for i in keep]))
    assert all_ts.max() < vqs.CUTOFF, 'a decision outside development would be priced'
    priced = {c: vqm.payouts(model, ts_k, px_k, all_ts.astype(float), c) for c in vqm.CLOCKS}
    pu0, pd0, ok0 = priced['ny']; pu1, pd1, ok1 = priced['utc4']
    both = ok0 & ok1
    out['decisions_priced'] = {
        'unique': int(len(all_ts)), 'priced_new_york_clock': int(ok0.sum()), 'priced_utc4_clock': int(ok1.sum()),
        'rise_payout_p5_p50_p95': [float(x) for x in np.percentile(pu0[ok0], (5, 50, 95))] if ok0.any() else None,
        'fall_payout_p5_p50_p95': [float(x) for x in np.percentile(pd0[ok0], (5, 50, 95))] if ok0.any() else None,
        'mean_breakeven_rise': float((1 / pu0[ok0]).mean()) if ok0.any() else None,
        'mean_breakeven_fall': float((1 / pd0[ok0]).mean()) if ok0.any() else None,
        'share_clocks_differ_by_more_than_0.02': float(((np.abs(pu0 - pu1) > 0.02) | (np.abs(pd0 - pd1) > 0.02))[both].mean()) if both.any() else None}
    res = json.load(open(a.result)) if a.result else None
    orig = json.load(open(a.original)) if a.original else None
    actual = None
    if a.quote_dir:                                   # the pair's own July quotes, if it has a lane
        lanes, _ = vqs.load_quotes(a.quote_dir)
        if (a.pair, 'UP') in lanes and (a.pair, 'DOWN') in lanes:
            actual = vqs.match_lanes(lanes[(a.pair, 'UP')], lanes[(a.pair, 'DOWN')])
    for side in ('UP', 'DOWN'):
        rows = {c: [] for c in vqm.CLOCKS}; frozen = []
        act = {'own': [], 'opp': [], 'hat': [], 'win': [], 'ts': []}
        for i in keep:
            ts, p, fwd = d[f'ts_{i}'], d[f'p_{i}'], d[f'fwd_{i}']
            thr = json.loads(str(d['thr'][i]))[a.cov]
            j = np.searchsorted(all_ts, ts)
            conf = np.abs(p - 0.5); want = (p > 0.5) if side == 'UP' else (p < 0.5)
            tr = cd.nonoverlap_chrono(ts, want & (conf >= thr) & np.isfinite(fwd), gap)
            al = cd.nonoverlap_chrono(ts, np.isfinite(fwd), gap)
            win = ((fwd > 0) if side == 'UP' else (fwd < 0)).astype(float)
            frozen.append((labels[i], int(len(tr)), float(win[tr].mean()) if len(tr) else float('nan')))
            for c in vqm.CLOCKS:
                pu, pd_, ok = (x[j] for x in priced[c])
                own, opp = (pu, pd_) if side == 'UP' else (pd_, pu)
                t_ok, a_ok = tr[ok[tr]], al[ok[al]]
                row = {'test': labels[i], 'frozen': int(len(tr)), 'not_priced': int(len(tr) - len(t_ok)),
                       'trades': stats(win[t_ok], own[t_ok], opp[t_ok]), 'always': stats(win[a_ok], own[a_ok], opp[a_ok])}
                if c == 'ny':
                    fav = t_ok[own[t_ok] < opp[t_ok]]; unf = t_ok[own[t_ok] > opp[t_ok]]
                    row['favoured'] = stats(win[fav], own[fav], opp[fav]); row['unfavoured'] = stats(win[unf], own[unf], opp[unf])
                    for fl in FLOORS:
                        k = t_ok[own[t_ok] >= fl - 1e-9]
                        row[f'payout_at_least_{fl:.2f}'] = stats(win[k], own[k], opp[k])
                    if actual is not None and len(t_ok):
                        tq = actual['t_dec'].to_numpy(); t = ts[t_ok].astype(float) + 1.0
                        k = np.clip(np.searchsorted(tq, t), 0, len(tq) - 1); k0 = np.clip(k - 1, 0, len(tq) - 1)
                        k = np.where(np.abs(tq[k] - t) <= np.abs(tq[k0] - t), k, k0); m = np.abs(tq[k] - t) <= 20.0
                        au, ad = actual['pu'].to_numpy()[k][m], actual['pd'].to_numpy()[k][m]
                        act['own'] += list(au if side == 'UP' else ad); act['opp'] += list(ad if side == 'UP' else au)
                        act['hat'] += list(own[t_ok][m]); act['win'] += list(win[t_ok][m]); act['ts'] += list(ts[t_ok][m])
                rows[c].append(row)
        summ = summarise(rows['ny']); alt = summarise(rows['utc4'])
        cleared = (summ['paths_counted'] >= 12 and summ['paths_ev_positive'] >= 12 and summ['p10_ev'] is not None and summ['p10_ev'] >= 0)
        pricing = ('ADVERSELY PRICED' if summ['paths_payout_0.02_below_always'] >= 12 else
                   'FAVOURABLY PRICED' if summ['paths_payout_0.02_above_always'] >= 12 else 'NEUTRAL')
        side_out = {'paths': rows['ny'], 'summary': summ, 'verdict': 'PAYOUT-CLEARED' if cleared else 'NOT CLEARED AT THE QUOTED PAYOUT',
                    'pricing': pricing,
                    'descriptive_pooled_over_paths': {k: pooled(rows['ny'], k) for k in ['trades', 'favoured', 'unfavoured']
                                                      + [f'payout_at_least_{fl:.2f}' for fl in FLOORS]},
                    'sensitivity_utc4_clock': {k: alt[k] for k in ('paths_counted', 'paths_ev_positive', 'p10_ev', 'mean_ev', 'mean_payout',
                                                                    'mean_win_rate', 'trades_not_priced')},
                    'harness_trades_reproduced': reproduces(res, frozen, a.cov, side) if res is not None else None,
                    'recorded_candidate_reproduced': reproduces(orig, frozen, a.cov, side) if orig is not None else None}
        if actual is not None:
            own_a, hat, w = np.array(act['own']), np.array(act['hat']), np.array(act['win'])
            side_out['actual_quotes'] = {'trade_rows': int(len(own_a)), 'unique_decisions': int(len(set(act['ts'])))}
            if len(own_a):
                side_out['actual_quotes'].update({
                    'mean_actual_payout': float(own_a.mean()), 'mean_reconstructed_payout': float(hat.mean()),
                    'mean_abs_error': float(np.abs(own_a - hat).mean()), 'share_within_0.02': float((np.abs(own_a - hat) <= 0.0201).mean()),
                    'favoured_share_actual': float((own_a < np.array(act['opp'])).mean()), 'win_rate': float(w.mean()),
                    'ev_actual': float((w * own_a).mean() - 1.0), 'ev_reconstructed': float((w * hat).mean() - 1.0)})
        out['sides'][side] = side_out
        show = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summ.items()}
        print(side, side_out['verdict'], '|', pricing, '|', show, '| utc4:', {k: (round(v, 4) if isinstance(v, float) else v)
              for k, v in side_out['sensitivity_utc4_clock'].items()}, '| harness reproduced:', side_out['harness_trades_reproduced'],
              '| recorded reproduced:', side_out['recorded_candidate_reproduced'], flush=True)
    out['code_sha256'] = cd.sha256(__file__)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
