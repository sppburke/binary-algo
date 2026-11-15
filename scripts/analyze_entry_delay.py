#!/usr/bin/env python3
"""Entry-delay stress and feed-staleness audit of a CPCV side (analysis; reading pre-registered before the dump is opened).

    analyze_entry_delay.py <dump.npz> <pair> <out.json> [--cov 0.03] [--horizon 15] [--delays 0,2,4,9,29]

The dump comes from cpcv_deriv_x.py --dump-preds. The model's trades are FROZEN exactly as cpcv_deriv.side_wr selects
them on the baseline labels (side filter, confidence >= the path's validation threshold, first-come de-overlap with gap
H). Each trade is then re-settled from the raw Deriv ticks with the contract started d extra seconds later (entry =
first tick at or after decision + 1 s + d, exit = last tick at or before that entry target + H, the same tolerances;
ties lose; a delay whose entry or exit tick is missing within tolerance is counted as a LOSS). d = 0 reproduces the
baseline labels (checked). Per path and delay: win rate of the frozen trades, and the always-side baseline re-settled
the same way on the path's de-overlapped decisions.
Staleness audit: for every frozen trade, the age of the last mid change before the decision (seconds); win rates and
excess over always-side by tercile of that age within the 15 CPCV paths.
PRE-REGISTERED READING (per side): OPERATIONALLY ROBUST iff with d = 4 s (entry about 5 s after the decision) the side
still has >= 12 of the 15 CPCV paths with >= 25 trades clearing 0.541 and their p10 >= 0.541; otherwise FRAGILE.
Staleness: the skill is NOT a stale-feed artifact iff the excess over always-side is positive in each age tercile.
"""
import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpcv_deriv as cd  # noqa: E402
import deriv_dataset as dd  # noqa: E402

VERSION = 'analyze_entry_delay/0.2'   # 0.2: rank-based staleness terciles (0.1 value quantiles tied)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dump'); ap.add_argument('pair'); ap.add_argument('out')
    ap.add_argument('--cov', default='0.03'); ap.add_argument('--horizon', type=int, default=15)
    ap.add_argument('--delays', default='0,2,4,9,29')
    ap.add_argument('--first', default='2025-08-11'); ap.add_argument('--last', default='2026-07-31')
    a = ap.parse_args()
    H = a.horizon
    gap = H * 60
    hs, tol = dd.HORIZONS[H]
    delays = [int(x) for x in a.delays.split(',')]
    out = {'analysis': VERSION, 'pair': a.pair, 'horizon_min': H, 'cov': a.cov, 'delays_extra_s': delays,
           'registered_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'pre_registered_reading': 'per side: OPERATIONALLY ROBUST iff at d = 4 s >= 12 of 15 CPCV paths (>= 25 trades) '
                                     'clear 0.541 and their p10 >= 0.541, else FRAGILE; staleness: not a stale-feed '
                                     'artifact iff the excess over always-side is positive in every age tercile',
           'sides': {}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    ts_k, px_k = dd.load_ticks(dd.day_files(a.pair, date.fromisoformat(a.first), date.fromisoformat(a.last)))
    chg_t = ts_k[np.r_[True, np.diff(px_k) != 0]]                  # times of mid changes (first tick counts)
    d = np.load(a.dump)
    labels = [str(x) for x in d['labels']]
    settled = {}
    for side in ('UP', 'DOWN'):
        win = (lambda f: f > 0) if side == 'UP' else (lambda f: f < 0)
        rows, base_ok, trade_age, trade_ex = [], True, [], []
        for i, lab in enumerate(labels):
            if lab.startswith('forward'):
                continue
            ts, p, fwd = d[f'ts_{i}'], d[f'p_{i}'], d[f'fwd_{i}']
            thr = json.loads(str(d['thr'][i]))[a.cov]
            conf = np.abs(p - 0.5)
            want = (p > 0.5) if side == 'UP' else (p < 0.5)
            tr = cd.nonoverlap_chrono(ts, want & (conf >= thr) & np.isfinite(fwd), gap)
            al = cd.nonoverlap_chrono(ts, np.isfinite(fwd), gap)
            row = {'test': lab, 'n_trades': int(len(tr)), 'by_delay': {}}
            for dl in delays:
                key = (i, dl)
                if key not in settled:
                    r, v = dd.settle(ts + dl, ts_k, px_k, hs, tol)
                    settled[key] = np.where(v, r, np.nan)
                r = settled[key]
                if dl == 0:
                    base_ok &= bool(np.allclose(np.nan_to_num(r[np.isfinite(fwd)]), np.nan_to_num(fwd[np.isfinite(fwd)]), atol=0))
                wr = float(np.mean([bool(np.isfinite(x) and win(x)) for x in r[tr]])) if len(tr) else float('nan')
                wa = float(np.mean([bool(np.isfinite(x) and win(x)) for x in r[al]])) if len(al) else float('nan')
                row['by_delay'][str(dl)] = {'wr': wr, 'always': wa, 'missing': int((~np.isfinite(r[tr])).sum())}
            k = np.searchsorted(chg_t, ts[tr], side='left') - 1            # last change strictly before the decision
            age = np.where(k >= 0, ts[tr] - chg_t[np.clip(k, 0, None)], np.nan)
            wa0 = row['by_delay']['0']['always']
            trade_age += list(age)
            trade_ex += [float(win(x)) - wa0 for x in fwd[tr]]
            rows.append(row)
        summ = {}
        for dl in delays:
            w = [r['by_delay'][str(dl)]['wr'] for r in rows if r['n_trades'] >= cd.MIN_TRADES]
            ex = [r['by_delay'][str(dl)]['wr'] - r['by_delay'][str(dl)]['always'] for r in rows if r['n_trades'] >= cd.MIN_TRADES]
            summ[str(dl)] = {'paths_counted': len(w), 'paths_clear': int(sum(x >= cd.BE for x in w)),
                             'p10': float(np.percentile(w, 10)) if w else None, 'mean': float(np.mean(w)) if w else None,
                             'excess_vs_always_mean': float(np.mean(ex)) if ex else None}
        s4 = summ.get('4')
        robust = bool(s4 and s4['paths_counted'] >= 12 and s4['paths_clear'] >= 12 and s4['p10'] is not None and s4['p10'] >= cd.BE)
        age = np.asarray(trade_age, float); exv = np.asarray(trade_ex, float)
        ok = np.isfinite(age)
        terc = {}
        if ok.sum() >= 30:                          # rank-based terciles (ages tie heavily at 0-2 s; 0.1 used value quantiles)
            r = np.full(len(age), np.nan)
            r[ok] = np.argsort(np.argsort(age[ok], kind='stable'), kind='stable') / ok.sum()
            bins = np.where(ok, np.minimum(np.floor(np.nan_to_num(r) * 3), 2), -1).astype(int)
            for b, name in enumerate(('fresh', 'middle', 'stale')):
                m = ok & (bins == b)
                terc[name] = {'n': int(m.sum()), 'age_s_max': float(np.nanmax(age[m])) if m.any() else None,
                              'excess_vs_path_always_mean': float(exv[m].mean()) if m.any() else None}
        stale_ok = bool(terc) and all(v['excess_vs_path_always_mean'] is not None and v['excess_vs_path_always_mean'] > 0
                                      for v in terc.values())
        out['sides'][side] = {'paths': rows, 'summary_by_delay': summ, 'baseline_labels_reproduced_at_d0': base_ok,
                              'verdict_delay': 'OPERATIONALLY ROBUST' if robust else 'FRAGILE',
                              'staleness_terciles': terc,
                              'verdict_staleness': 'NOT A STALE-FEED ARTIFACT' if stale_ok else 'EXCESS NOT POSITIVE IN EVERY AGE TERCILE'}
        print(side, out['sides'][side]['verdict_delay'], {k: (v['paths_clear'], v['paths_counted'], v['p10'] and round(v['p10'], 4)) for k, v in summ.items()},
              out['sides'][side]['verdict_staleness'], {k: (v['n'], v['excess_vs_path_always_mean'] and round(v['excess_vs_path_always_mean'], 4)) for k, v in terc.items()},
              'd0 reproduces labels:', base_ok, flush=True)
    out['code_sha256'] = cd.sha256(__file__)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
