#!/usr/bin/env python3
"""Dependence of a CPCV side's trade outcomes (DEV only; reading pre-registered before the dump is opened).

    analyze_dependence.py <dump.npz> <pair> <out.json> [--cov 0.03] [--side UP] [--horizon 15] [--result RUN.json]

The dump comes from cpcv_deriv_x.py --dump-preds. Each of the 15 CPCV paths' side trades is rebuilt exactly as
analyze_drift_placebo / cpcv_deriv.side_wr (confidence >= the path's validation threshold, first-come de-overlap with
gap H, win = right sign AND moved, ties lose). Per path:
  vif_day  = day-clustered variance of the mean win (clusters = America/New_York calendar dates, G/(G-1) correction)
             divided by the iid variance p(1-p)/n;
  vif_week = the same with ISO-week clusters;
  acf1     = lag-1 autocorrelation of the win indicator in trade order.
PRE-REGISTERED READING: the holdout test's conditional-independence assumption is SUPPORTED iff the median vif_day and
the median vif_week over the counted paths (>= 25 trades) are both <= 1.15; otherwise OVERDISPERSED (the owner is told
before any holdout look, with the estimated z inflation sqrt(median vif)).
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

VERSION = 'analyze_dependence/0.1'


def clustered_vif(w, keys):
    n = len(w)
    p = w.mean()
    v_iid = p * (1 - p) / n
    if n < 2 or v_iid <= 0:
        return float('nan'), 0
    s = pd.Series(w - p).groupby(keys).sum().to_numpy()
    g = len(s)
    v_cl = (g / (g - 1)) * float((s ** 2).sum()) / n ** 2 if g > 1 else float('nan')
    return v_cl / v_iid, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dump'); ap.add_argument('pair'); ap.add_argument('out')
    ap.add_argument('--cov', default='0.03')
    ap.add_argument('--side', default='UP', choices=('UP', 'DOWN'))
    ap.add_argument('--horizon', type=int, default=15)
    ap.add_argument('--result', default=None)
    a = ap.parse_args()
    reading = ('SUPPORTED iff the median vif_day and the median vif_week over counted paths (>= 25 trades) are both '
               '<= 1.15, else OVERDISPERSED')
    out = {'analysis': VERSION, 'pair': a.pair, 'horizon_min': a.horizon, 'cov': a.cov, 'side': a.side, 'dump': a.dump,
           'pre_registered_reading': reading, 'registered_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    Path(a.out).write_text(json.dumps(out, indent=1))          # reading written before the dump is opened
    d = np.load(a.dump)
    labels = list(d['labels'])
    ref = json.loads(Path(a.result).read_text()) if a.result else None
    gap = a.horizon * 60
    win_side = (lambda f: f > 0) if a.side == 'UP' else (lambda f: f < 0)
    rows, match = [], True
    for i, lab in enumerate(labels):
        ts, p, fwd = d[f'ts_{i}'], d[f'p_{i}'], d[f'fwd_{i}']
        thr = json.loads(str(d['thr'][i]))[a.cov]
        conf = np.abs(p - 0.5)
        want = (p > 0.5) if a.side == 'UP' else (p < 0.5)
        tr = cd.nonoverlap_chrono(ts, want & (conf >= thr) & np.isfinite(fwd), gap)
        w = (win_side(fwd[tr]) & (fwd[tr] != 0)).astype(float)
        if ref is not None and i < len(ref['paths']):
            rn, rw = ref['paths'][i]['bycov'][a.cov][a.side]
            match &= (rn == len(tr)) and (len(tr) == 0 or abs(rw - float(w.mean())) < 1e-12)
        row = {'test': lab, 'n_trades': int(len(tr))}
        if len(tr) >= 2:
            ny = pd.to_datetime(ts[tr], unit='s', utc=True).tz_convert('America/New_York')
            vd, gd = clustered_vif(w, ny.strftime('%Y-%m-%d').to_numpy())
            iso = ny.isocalendar()
            vw, gw = clustered_vif(w, (iso['year'].astype(str) + '-' + iso['week'].astype(str)).to_numpy())
            acf1 = float(np.corrcoef(w[:-1], w[1:])[0, 1]) if w[:-1].std() > 0 and w[1:].std() > 0 else float('nan')
            row.update({'win_rate': float(w.mean()), 'vif_day': vd, 'days': gd, 'vif_week': vw, 'weeks': gw, 'acf1': acf1,
                        'trades_per_day': len(tr) / gd})
        rows.append(row)
    cpcv = [r for r in rows if not str(r['test']).startswith('forward') and r['n_trades'] >= cd.MIN_TRADES]
    med = lambda k: float(np.nanmedian([r[k] for r in cpcv])) if cpcv else float('nan')
    summ = {'paths_counted': len(cpcv), 'median_vif_day': med('vif_day'), 'median_vif_week': med('vif_week'),
            'median_acf1': med('acf1'), 'median_trades_per_day': med('trades_per_day'),
            'z_inflation_estimate': float(np.sqrt(max(med('vif_day'), med('vif_week'), 1.0)))}
    out.update({'paths': rows, 'summary': summ, 'rebuild_matches_run': bool(match) if ref is not None else None,
                'verdict': 'SUPPORTED' if summ['median_vif_day'] <= 1.15 and summ['median_vif_week'] <= 1.15 else 'OVERDISPERSED',
                'code_sha256': cd.sha256(__file__)})
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({'summary': summ, 'verdict': out['verdict'], 'rebuild_matches_run': out['rebuild_matches_run']}, indent=1))


if __name__ == '__main__':
    main()
