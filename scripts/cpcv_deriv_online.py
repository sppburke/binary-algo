#!/usr/bin/env python3
"""SWEEP_MATRIX C5 online learner (river Adaptive Random Forest, ADWIN drift detection per tree) on the Deriv feed,
scored prequentially on the CPCV group structure.

    cpcv_deriv_online.py --pair P --horizon 15 [--session ny] [--covs 0.03,0.02,0.05] [--n-models 10]
                         [--window 20000] [--dataset DIR] [--out result.json] [--tag TEXT]

An online learner is never refit per fold: it learns every in-session tradeable moved decision in time order, each one
only after its label is settled (decision + 1 s + H; the settlement rule's exit target), and predicts each decision
before learning anything that settles after it. The confidence threshold at a decision is the (1 - coverage) quantile of
the |p - 0.5| it produced over the previous --window in-session decisions (strictly earlier; no trade before 2,000 exist).
Rows, sessions, the purchase-blackout filter, the 6 equal-count groups, the 15 two-group test paths, the side filter,
first-come de-overlap (gap H), ties-lose wins, >= 25 trades per path and the validity rule are those of
tools/cpcv_deriv.py; a path is the union of its two groups' prequential decisions. Every path is out of sample in time
(the model at any decision has learned only earlier settled decisions), so there is no separate forward walk. Missing
feature values enter river as 0. The falsifier KILLs exactly as the CPCV rule.
"""
import argparse
import sys
import time
from collections import deque
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpcv_deriv as cd  # noqa: E402

ONLINE_VERSION = 'cpcv_deriv_online/0.1.1'   # 0.1.1: the venue's window (owner decision 14) and the guards of E-003,
#   both through cpcv_deriv 0.4.2


def main():
    from river import __version__ as river_version
    from river import forest
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', required=True)
    ap.add_argument('--horizon', type=int, default=15, choices=(15, 30))
    ap.add_argument('--session', default='ny', choices=('ny', 'ldn', 'asia', 'overlap', 'all'))
    ap.add_argument('--covs', default='0.03,0.02,0.05')
    ap.add_argument('--n-models', type=int, default=10)
    ap.add_argument('--window', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--features', default='all', choices=('all', 'noclock', 'own', 'own_noclock'))
    ap.add_argument('--dataset', default=str(cd.LAB / 'cache' / 'dataset_dev_v0.1'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--july-filter', action='store_true',
                    help='the July 2026 tradeability filter, without the venue window of owner decision 14 '
                         '(regression against stored results only; such a result cannot be promoted)')
    a = ap.parse_args()
    covs = cd.parse_covs(a.covs)
    if a.n_models < 1 or not 2000 <= a.window <= 2 ** 63 - 1:   # test E-003: before anything is loaded or written
        raise SystemExit(f'--n-models {a.n_models} / --window {a.window}: at least 1 model, and a window of 2000 '
                         'decisions (the rolling threshold needs 2000 earlier ones) to 2^63 - 1 (pandas\' 64-bit '
                         'window)')
    gap = a.horizon * 60
    out = Path(a.out or f'{a.pair}_{a.horizon}m_{a.session}_online_result.json')
    t0 = time.time()
    X, fwd, ts, feats = cd.load(a.dataset, a.pair, a.horizon, a.features, 1, None)
    sess = cd.session_mask(ts, a.session) & cd.tradeable(ts, a.horizon, venue_window=not a.july_filter)
    moved = np.isfinite(fwd) & (fwd != 0.0)
    tgs = ts[sess]
    bnds = [tgs[int(k * len(tgs) / cd.N_GROUPS)] for k in range(cd.N_GROUPS)] + [tgs[-1] + 1]
    groups = [(int(bnds[g]), int(bnds[g + 1])) for g in range(cd.N_GROUPS)]
    day = lambda s: pd.to_datetime(s, unit='s', utc=True).strftime('%Y-%m-%d')
    data_uprate = float((fwd[sess & moved] > 0).mean())
    manifest = Path(a.dataset) / 'manifest.json'
    res = {
        'harness': ONLINE_VERSION, 'base_harness': cd.HARNESS_VERSION, 'tag': a.tag,
        'model': f'river ARFClassifier (n_models {a.n_models}, ADWIN drift detection, seed {a.seed}) learned prequentially '
                 f'@{a.horizon}m on the Deriv feed, session {a.session}',
        'key': f'{a.pair}.{a.horizon}m', 'data': 'Deriv 1-second mid ticks only (/mnt/data/deriv), no Dukascopy',
        'dataset': {'dir': a.dataset, 'manifest_sha256': cd.sha256(manifest)},
        'window': {'first_decision': day(ts[0]), 'last_decision': day(ts[-1])},
        'breakeven': cd.BE, 'session': a.session, 'tradeable_only': True, 'blackout_rule': cd.BLACKOUT_RULE,
        'venue_window_rule': None if a.july_filter else cd.VENUE_WINDOW_RULE,
        'regression_only': bool(a.july_filter),     # True: never a candidate result
        'run_status': 'running',                    # 'complete' when written at the end
        'horizon_min': a.horizon, 'gap_s': gap, 'covs': covs, 'operating_cov': covs[0], 'nseed': 1,
        'feature_set': a.features, 'n_features': len(feats), 'features': feats, 'n_groups': cd.N_GROUPS, 'k_test': cd.K_TEST,
        'threshold_window': a.window, 'min_trades_per_path': cd.MIN_TRADES,
        'rows': int(len(ts)), 'rows_in_session': int(sess.sum()), 'in_session_moved_up_rate': data_uprate,
        'groups': [(day(lo), day(hi - 1)) for lo, hi in groups],
        'evaluation': 'prequential (learn each decision only after it settles; predict before learning anything settling '
                      'later); paths = unions of two groups; no per-fold refit (an online learner has no folds)',
        'CERT_RULE': ('never certified: the online learner is a continuing policy, a forward diagnostic only (owner '
                      'decision 13; cpcv_deriv.adaptive_policy); its figures follow the shared rule (unrounded path '
                      'win rates; paths with >= 25 trades)'),
        'falsifier': {'version': 3, 'registered_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      'when': 'before any decision is predicted',
                      'KILL_if': f'at the operating coverage {covs[0]}: each side p10<0.541 OR frac_clear_BE<0.80',
                      'INVALID_if': 'in-session moved up-rate outside [0.47,0.53], or any path moved up-rate outside '
                                    '[0.45,0.55]; an INVALID run certifies nothing in any field'},
        'code': {'harness_sha256': cd.sha256(__file__), 'base_harness_sha256': cd.sha256(cd.__file__), 'river': river_version},
    }
    out.write_text(cd.result_json(res))
    print(f'[{a.pair} {a.horizon}m online] rows {len(ts):,} in-session {int(sess.sum()):,} features {len(feats)}', flush=True)

    model = forest.ARFClassifier(n_models=a.n_models, seed=a.seed)
    Xs = np.where(np.isfinite(X), X, 0.0).astype(np.float64)
    settle = ts + 1 + gap
    idx = np.where(sess & np.isfinite(fwd))[0]                 # chronological (cd.load sorts by decision time)
    p = np.full(len(ts), np.nan)
    pending = deque()
    learned = 0
    for n, i in enumerate(idx):
        t = ts[i]
        while pending and settle[pending[0]] <= t:
            j = pending.popleft()
            if moved[j]:
                model.learn_one(dict(zip(feats, Xs[j].tolist())), bool(fwd[j] > 0))
                learned += 1
        pr = model.predict_proba_one(dict(zip(feats, Xs[i].tolist())))
        p[i] = float(pr.get(True, 0.5)) if pr else 0.5
        pending.append(i)
        if n % 20000 == 0:
            print(f'  {n:,}/{len(idx):,} predicted, {learned:,} learned ({time.time() - t0:.0f}s)', flush=True)
    conf = np.abs(p - 0.5)
    thr = {}
    for c in covs:
        q = pd.Series(conf[idx]).shift(1).rolling(a.window, min_periods=2000).quantile(1 - c).to_numpy()
        full = np.full(len(ts), np.inf)
        full[idx] = np.where(np.isfinite(q), q, np.inf)
        thr[c] = full

    def score(mask, label):
        f, t_, pr = fwd[mask], ts[mask], p[mask]
        mv = f != 0
        mvf = mv & np.isfinite(pr)                    # test E-003: the diagnostic reads finite scores only
        row = {'test': label, 'n_test': int(mask.sum()),
               'auc': float(cd.roc_auc_score((f[mvf] > 0).astype(int), pr[mvf])) if mvf.sum() > 20 else float('nan'),
               'up_rate': float((f[mv] > 0).mean()) if mv.sum() else float('nan'), 'always': {}, 'bycov': {}}
        for s in ('UP', 'DOWN'):
            row['always'][s] = list(cd.always(f, t_, s, gap))
        for c in covs:
            ex = conf[mask] - thr[c][mask]
            pr2 = np.where(ex > 0, 0.5 + np.sign(pr - 0.5) * ex, 0.5)     # trade iff conf exceeds the causal threshold
            row['bycov'][f'{c}'] = {'thr_median': float(np.nanmedian(np.where(np.isfinite(thr[c][mask]), thr[c][mask], np.nan)))}
            for s in cd.SIDES:
                row['bycov'][f'{c}'][s] = list(cd.side_wr(pr2, f, t_, 1e-15, s, gap))
        return row

    base = sess & np.isfinite(fwd)
    gmask = [base & (ts >= lo) & (ts < hi) for lo, hi in groups]
    res['groups_prequential'] = [score(m, f'g{g}') for g, m in enumerate(gmask)]
    pathinfo = []
    for fi, testg in enumerate(combinations(range(cd.N_GROUPS), cd.K_TEST)):
        row = score(gmask[testg[0]] | gmask[testg[1]], list(testg))
        pathinfo.append(row)
        print(f'  path {fi + 1:2d}/15 g{list(testg)} AUC={row["auc"]:.4f} | cov{covs[0]}: {cd.fmt_side(row["bycov"][f"{covs[0]}"])}',
              flush=True)
    cd.aggregate(res, pathinfo, [], covs, data_uprate, np.ones(len(feats)), feats)
    res.pop('top_gain_features', None)
    res['learned'] = learned
    res['seconds'] = round(time.time() - t0, 1)
    res['run_status'] = 'complete'
    out.write_text(cd.result_json(res))
    cd.print_summary(res, covs, f'{a.pair} {a.horizon}m online', t0, out)


if __name__ == '__main__':
    main()
