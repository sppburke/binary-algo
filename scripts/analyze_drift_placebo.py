#!/usr/bin/env python3
"""Drift-neutral check of a CPCV side's selection skill (analysis; reading pre-registered before the dump is opened).

    analyze_drift_placebo.py <dump.npz> <dataset dir> <pair> <out.json> [--cov 0.03] [--side UP] [--horizon 15]

The dump comes from cpcv_deriv_x.py --dump-preds (every path's test decisions: ts, p, fwd, validation thresholds).
For each of the 15 CPCV paths the model's side trades at the coverage are rebuilt exactly as cpcv_deriv.side_wr (side
filter, confidence >= the path's validation threshold, first-come de-overlap with gap H, win = right sign AND moved,
ties lose) and compared with baselines from the SAME path's test decisions:
  always      : the side's win rate over all of the path's decisions (always-side, ties lose);
  hour        : for each trade, the side's win rate of the path's decisions in the same New York local hour,
                averaged over the trades (removes timing of intraday drift);
  hour_vol    : the same within (NY hour, rv_60m tercile of the path's decisions) cells (also removes volatility timing).
PRE-REGISTERED READING: the side's selection carries skill beyond timing the drift by hour and volatility iff the
hour_vol excess is > 0 on at least 12 of the counted paths (>= 25 trades) AND its mean over counted paths is >= 0.02.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpcv_deriv as cd  # noqa: E402

VERSION = 'analyze_drift_placebo/0.1'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dump'); ap.add_argument('dataset'); ap.add_argument('pair'); ap.add_argument('out')
    ap.add_argument('--cov', default='0.03')
    ap.add_argument('--horizon', type=int, default=15)
    ap.add_argument('--result', default=None, help='the run\'s CPCV result JSON: rebuilt trades must match its path values')
    a = ap.parse_args()
    gap = a.horizon * 60
    reading = ('per side: skill beyond timing the drift iff the hour_vol excess is > 0 on >= 12 counted paths (>= 25 '
               'trades) AND its mean over counted paths >= 0.02')
    out = {'analysis': VERSION, 'pair': a.pair, 'horizon_min': a.horizon, 'cov': a.cov,
           'dump': a.dump, 'pre_registered_reading': reading,
           'registered_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'sides': {}}
    Path(a.out).write_text(json.dumps(out, indent=1))          # reading written before the dump is opened
    d = np.load(a.dump)
    labels = list(d['labels'])
    rv = pd.read_parquet(Path(a.dataset) / f'{a.pair}.parquet', columns=['decision_ts', 'rv_60m']).set_index('decision_ts')['rv_60m']
    ref = json.loads(Path(a.result).read_text()) if a.result else None
    for side in ('UP', 'DOWN'):
        out['sides'][side] = one_side(a, d, labels, rv, side, gap, ref)
    out['code_sha256'] = cd.sha256(__file__)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({s: {'summary': v['summary'], 'verdict': v['verdict'], 'rebuild_matches_run': v['rebuild_matches_run']}
                      for s, v in out['sides'].items()}, indent=1))


def one_side(a, d, labels, rv, side, gap, ref):
    win_side = (lambda f: f > 0) if side == 'UP' else (lambda f: f < 0)
    paths = []
    match = True
    for i, lab in enumerate(labels):
        ts, p, fwd = d[f'ts_{i}'], d[f'p_{i}'], d[f'fwd_{i}']
        thr = json.loads(str(d['thr'][i]))[a.cov]
        conf = np.abs(p - 0.5)
        want = (p > 0.5) if side == 'UP' else (p < 0.5)
        tr = cd.nonoverlap_chrono(ts, want & (conf >= thr) & np.isfinite(fwd), gap)
        n = len(tr)
        if ref is not None and i < len(ref['paths']):
            rn, rw = ref['paths'][i]['bycov'][a.cov][side]
            mine = float((win_side(fwd[tr]) & (fwd[tr] != 0)).mean()) if n else float('nan')
            match &= (rn == n) and (n == 0 or abs(rw - mine) < 1e-12)
        hour = pd.to_datetime(ts, unit='s', utc=True).tz_convert('America/New_York').hour.values
        v = rv.reindex(ts).to_numpy(np.float64)
        q1, q2 = np.nanquantile(v, [1 / 3, 2 / 3])
        terc = np.where(np.isnan(v), -1, np.digitize(v, [q1, q2]))
        w_all = win_side(fwd).astype(float)
        df = pd.DataFrame({'h': hour, 't': terc, 'w': w_all})
        cell_h = df.groupby('h')['w'].mean()
        cell_hv = df.groupby(['h', 't'])['w'].mean()
        row = {'test': lab, 'n_trades': n}
        if n:
            wr = float((win_side(fwd[tr]) & (fwd[tr] != 0)).mean())
            b_all = float(w_all.mean())
            b_h = float(cell_h.reindex(hour[tr]).mean())
            b_hv = float(cell_hv.reindex(list(zip(hour[tr], terc[tr]))).mean())
            row.update({'wr_model': wr, 'base_always': b_all, 'base_hour': b_h, 'base_hour_vol': b_hv,
                        'excess_always': wr - b_all, 'excess_hour': wr - b_h, 'excess_hour_vol': wr - b_hv})
        paths.append(row)
    cnt = [r for r in paths if r['n_trades'] >= cd.MIN_TRADES]
    summ = {}
    for k in ('excess_always', 'excess_hour', 'excess_hour_vol'):
        x = [r[k] for r in cnt]
        summ[k] = {'mean': float(np.mean(x)) if x else None, 'n_positive': int(sum(e > 0 for e in x)), 'n_counted': len(x)}
    hv = summ['excess_hour_vol']
    return {'paths': paths, 'summary': summ, 'rebuild_matches_run': bool(match) if ref is not None else None,
            'verdict': ('SKILL BEYOND DRIFT TIMING' if hv['n_counted'] and hv['n_positive'] >= 12 and hv['mean'] >= 0.02
                        else 'EXPLAINED BY DRIFT TIMING (hour x volatility) or too thin')}


if __name__ == '__main__':
    main()
