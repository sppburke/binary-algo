#!/usr/bin/env python3
"""Zero-parameter RULE scored on the CPCV path structure (nothing is fitted, so nothing can be refit per fold).

    cpcv_rule.py --pair P --horizon 15 --dataset DIR --signal COL [--direction 1|-1] [--usd-oriented]
                 [--session ny] [--out result.json] [--tag TEXT]

Side at a decision = sign(direction x signal), with the signal first converted from USD orientation to the pair's quote
orientation when --usd-oriented (USD strength is + for USDJPY/USDCHF/USDCAD and - for the other pairs' quotes). Every
in-session tradeable decision with a finite, non-zero signal is a candidate; first-come de-overlap (gap H), win = right
sign AND moved (ties lose). Groups, the 15 two-group paths, the always-side baselines, validity and both verdicts are
exactly tools/cpcv_deriv.py's (aggregate()); there is no validation set and no threshold, so the operating coverage is
"all signalled decisions" (recorded as covs [1.0]). The falsifier is written before any decision is scored.
"""
import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpcv_deriv as cd  # noqa: E402

RULE_VERSION = 'cpcv_rule/0.1'
USD_BASE = {'USDJPY', 'USDCHF', 'USDCAD'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', required=True)
    ap.add_argument('--horizon', type=int, default=15, choices=(15, 30))
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--signal', required=True)
    ap.add_argument('--direction', type=int, default=1, choices=(1, -1))
    ap.add_argument('--usd-oriented', action='store_true')
    ap.add_argument('--session', default='ny')
    ap.add_argument('--out', default=None)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    gap = a.horizon * 60
    out = Path(a.out or f'{a.pair}_{a.horizon}m_rule_result.json')
    t0 = time.time()
    # two always-finite companions keep every decision row (cd.load drops rows with >= 50% missing features), so the
    # groups and the validity check use the same in-session decisions as every other run
    X, fwd, ts, feats = cd.load(a.dataset, a.pair, a.horizon, 'all', 1, [a.signal, 'hour_sin', 'hour_cos'])
    sig = X[:, 0].astype(np.float64)
    if a.usd_oriented:
        sig = sig * (1.0 if a.pair in USD_BASE else -1.0)          # USD strength -> pair quote direction
    sig = a.direction * sig
    sess = cd.session_mask(ts, a.session) & cd.tradeable(ts, a.horizon)
    moved = np.isfinite(fwd) & (fwd != 0)
    tgs = ts[sess]
    bnds = [tgs[int(k * len(tgs) / cd.N_GROUPS)] for k in range(cd.N_GROUPS)] + [tgs[-1] + 1]
    groups = [(int(bnds[g]), int(bnds[g + 1])) for g in range(cd.N_GROUPS)]
    day = lambda s: pd.to_datetime(s, unit='s', utc=True).strftime('%Y-%m-%d')
    data_uprate = float((fwd[sess & moved] > 0).mean())
    res = {'harness': RULE_VERSION, 'base_harness': cd.HARNESS_VERSION, 'tag': a.tag,
           'model': f'zero-parameter rule: side = sign({a.direction} x {a.signal}{" (USD-oriented -> quote)" if a.usd_oriented else ""})',
           'key': f'{a.pair}.{a.horizon}m', 'data': 'Deriv 1-second mid ticks only (/mnt/data/deriv), no Dukascopy',
           'dataset': {'dir': a.dataset, 'manifest_sha256': cd.sha256(Path(a.dataset) / 'manifest.json')},
           'breakeven': cd.BE, 'session': a.session, 'tradeable_only': True, 'blackout_rule': cd.BLACKOUT_RULE,
           'horizon_min': a.horizon, 'gap_s': gap, 'covs': [1.0], 'operating_cov': 1.0, 'nseed': 0,
           'feature_set': f'rule:{a.signal}', 'n_features': 1, 'features': [a.signal], 'n_groups': cd.N_GROUPS,
           'k_test': cd.K_TEST, 'min_trades_per_path': cd.MIN_TRADES, 'rows_in_session': int(sess.sum()),
           'signalled_in_session': int((sess & np.isfinite(sig) & (sig != 0)).sum()),
           'in_session_moved_up_rate': data_uprate, 'groups': [(day(lo), day(hi - 1)) for lo, hi in groups],
           'CERT_RULE': 'side CERTIFIED iff VALID AND p10>=0.541 AND frac_clear_BE>=0.80 (paths with >= 25 trades); goal rule in verdict_goal',
           'falsifier': {'version': 3, 'registered_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                         'when': 'before any decision is scored',
                         'KILL_if': 'each side p10<0.541 OR frac_clear_BE<0.80 over the 15 paths (all signalled decisions)',
                         'INVALID_if': 'in-session moved up-rate outside [0.47,0.53], or any path moved up-rate outside [0.45,0.55]'},
           'code': {'rule_sha256': cd.sha256(__file__), 'base_harness_sha256': cd.sha256(cd.__file__)}}
    out.write_text(json.dumps(res, indent=1))
    p = np.where(np.isfinite(sig) & (sig != 0), 0.5 + 0.25 * np.sign(sig), 0.5)
    pathinfo = []
    for testg in combinations(range(cd.N_GROUPS), cd.K_TEST):
        tin = np.zeros(len(ts), bool)
        for g in testg:
            lo, hi = groups[g]
            tin |= (ts >= lo) & (ts < hi)
        m = tin & sess & np.isfinite(fwd)
        f, t, pp = fwd[m], ts[m], p[m]
        mv = f != 0
        sel = mv & (pp != 0.5)
        row = {'test': list(testg), 'n_test': int(m.sum()),
               'auc': float(cd.roc_auc_score((f[sel] > 0).astype(int), pp[sel])) if sel.sum() > 20 and len(set(pp[sel])) > 1 else float('nan'),
               'up_rate': float((f[mv] > 0).mean()) if mv.sum() else float('nan'), 'always': {}, 'bycov': {}}
        res_row = cd.score_row(row, pp, f, t, np.array([1e-9, 1e-9]), [1.0], gap)   # threshold 1e-9: unsignalled rows never trade
        pathinfo.append(res_row)
    cd.aggregate(res, pathinfo, [], [1.0], data_uprate, np.ones(1), [a.signal])
    res.pop('top_gain_features', None)
    res['seconds'] = round(time.time() - t0, 1)
    out.write_text(json.dumps(res, indent=1))
    cd.print_summary(res, [1.0], f'{a.pair} {a.horizon}m rule {a.signal}', t0, out)


if __name__ == '__main__':
    main()
