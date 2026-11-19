#!/usr/bin/env python3
"""POOLED-training variant of tools/cpcv_deriv.py (same CPCV, same scoring, same falsifier).

    cpcv_deriv_pool.py --pair EURUSD --horizon 15 [--pool EURUSD,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD]
                       [--session ny] [--covs 0.03,0.02,0.05] [--nseed 1] [--features own|own_noclock]
                       [--dataset DIR] [--out result.json] [--tag TEXT] [--no-forward] [--max-paths N]

Repository precedent (scripts/usdjpy_15m_xpair.py "pool"): training rows come from every pooled pair, each
labeled by its OWN Deriv-settled sign, on pair-agnostic features; validation (early stopping and the
threshold quantile), test paths and every reported number use the TARGET pair only. Cross-pair feature
columns are named after the other pairs, so only the own-pair feature set is pair-agnostic (default own).
Per CPCV path, every pooled pair's rows inside [test start - H, test end + H) are purged (shared clock).
Validation = random 15% of the TARGET's training rows (rng 13, as in the single-pair harness); the fit pool
= the other 85% of the target's rows + all other pooled pairs' training rows, subsampled to 150,000.
Repository closure respected by the caller: no pooling for the USDJPY/USDCAD/AUDUSD/NZDUSD family targets.
"""
import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpcv_deriv as cd  # noqa: E402  (load, mk_lgb, side_wr, always, summ, session_mask, constants)

POOL_VERSION = 'cpcv_deriv_pool/0.3.1'   # 0.3.1: the venue's window (owner decision 14) and the guards of E-003, both
#   through cpcv_deriv 0.4.2; 0.3: shared unrounded aggregation, no fit-pool sort; 0.2: blackout excluded
CLOSED_TARGETS = {'USDJPY', 'USDCAD', 'AUDUSD', 'NZDUSD'}


class PoolFitter(cd.Fitter):
    def __init__(self, X, fwd, ts, a, gap, covs, Xo, yo):
        super().__init__(X, fwd, ts, a, gap, covs)
        self.Xo, self.yo = Xo, yo                     # other pairs' candidate training rows (masked per path)

    def split_pool(self, tr_idx, other_keep, test_mask, rng, label):
        X, fwd, a = self.X, self.fwd, self.a
        valsel = rng.random(len(tr_idx)) < cd.VAL_FRAC
        val_idx = tr_idx[valsel]
        own_fit = tr_idx[~valsel]
        Xfit = np.vstack([X[own_fit], self.Xo[other_keep]])
        yfit = np.concatenate([(fwd[own_fit] > 0).astype(int), self.yo[other_keep]])
        if len(yfit) > cd.SUB_FIT:
            pick = rng.choice(len(yfit), cd.SUB_FIT, replace=False)    # sampled order kept (no sort), as the reference
            Xfit, yfit = Xfit[pick], yfit[pick]
        return self._fit_score(Xfit, yfit, val_idx, test_mask, label, n_own=int(len(own_fit)), n_other=int(other_keep.sum()))

    def _fit_score(self, Xfit, yfit, val_idx, test_mask, label, n_own, n_other):
        X, fwd, ts, a, gap = self.X, self.fwd, self.ts, self.a, self.gap
        if len(yfit) == 0 or len(val_idx) == 0:         # test E-003: nothing to fit or to set a threshold on
            return cd.unsupported_row(label, len(yfit), len(val_idx), fwd[test_mask], ts[test_mask], self.covs, gap)
        yval = (fwd[val_idx] > 0).astype(int)
        why = ('insufficient support: no test row' if not test_mask.any() else
               'insufficient support: fewer than two fit rows (LightGBM needs two)' if len(yfit) < 2 else
               'insufficient support: a validation label is absent from the fit labels'
               if not np.isin(yval, yfit).all() else None)     # LightGBM's early stopping refuses such a label
        if why:                                         # test E-003
            return cd.unsupported_row(label, len(yfit), len(val_idx), fwd[test_mask], ts[test_mask], self.covs, gap,
                                      why)
        pval = np.zeros(len(val_idx))
        ptst = np.zeros(int(test_mask.sum()))
        iters = []
        for sd in range(a.nseed):
            m = cd.mk_lgb(sd, a.num_leaves, a.n_jobs)
            m.fit(Xfit, yfit, eval_set=[(X[val_idx], yval)], eval_metric='auc',
                  callbacks=[cd.lgb.early_stopping(80, verbose=False), cd.lgb.log_evaluation(0)])
            pval += m.predict_proba(X[val_idx])[:, 1]
            ptst += m.predict_proba(X[test_mask])[:, 1]
            self.gain += m.booster_.feature_importance(importance_type='gain')
            iters.append(int(m.best_iteration_ or 0))
        pval /= a.nseed
        ptst /= a.nseed
        ftst = fwd[test_mask]
        ttst = ts[test_mask]
        mv = ftst != 0
        mvf = mv & np.isfinite(ptst)                  # test E-003: the diagnostic reads finite scores only
        auc = float(cd.roc_auc_score((ftst[mvf] > 0).astype(int), ptst[mvf])) if mvf.sum() > 20 else float('nan')
        uprate = float((ftst[mv] > 0).mean()) if mv.sum() else float('nan')
        row = {'test': label, 'n_fit': int(len(yfit)), 'n_fit_target': n_own, 'n_fit_other_available': n_other,
               'n_val': int(len(val_idx)), 'n_test': int(test_mask.sum()), 'best_iter': iters, 'auc': auc,
               'up_rate': uprate, 'always': {}, 'bycov': {}}
        return cd.score_row(row, ptst, ftst, ttst, np.abs(pval - 0.5), self.covs, gap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', required=True)
    ap.add_argument('--horizon', type=int, default=15, choices=(15, 30))
    ap.add_argument('--pool', default='EURUSD,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD')
    ap.add_argument('--session', default='ny', choices=('ny', 'ldn', 'asia', 'overlap', 'all'))
    ap.add_argument('--covs', default='0.03,0.02,0.05')
    ap.add_argument('--nseed', type=int, default=1)
    ap.add_argument('--features', default='own', choices=('own', 'own_noclock'))
    ap.add_argument('--num-leaves', type=int, default=255)
    ap.add_argument('--dataset', default=str(cd.LAB / 'cache' / 'dataset_dev_v0.1'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--n-jobs', type=int, default=16)
    ap.add_argument('--no-forward', action='store_true')
    ap.add_argument('--max-paths', type=int, default=15, help='smoke tests only; a certification run uses all 15')
    ap.add_argument('--july-filter', action='store_true',
                    help='the July 2026 tradeability filter, without the venue window of owner decision 14 '
                         '(regression against stored results only; such a result cannot be promoted)')
    a = ap.parse_args()
    if a.pair in CLOSED_TARGETS:
        raise SystemExit(f'{a.pair}: cross-pair pooling is closed for this target by the repository (AGENTS.md)')
    pool = [p for p in a.pool.split(',') if p != a.pair]
    covs = cd.parse_covs(a.covs)
    if a.nseed < 1 or a.max_paths < 1 or not 2 <= a.num_leaves <= 131072:
        raise SystemExit(f'--nseed {a.nseed} / --max-paths {a.max_paths} / --num-leaves {a.num_leaves}: at least 1 '
                         'seed and 1 path, and 2 to 131072 leaves (LightGBM)')
    gap = a.horizon * 60
    out = Path(a.out or f'{a.pair}_{a.horizon}m_{a.session}_pool_cpcv_result.json')
    t0 = time.time()

    X, fwd, ts, feats = cd.load(a.dataset, a.pair, a.horizon, a.features, 1)
    # Deriv purchase blackout (and, unless --july-filter, the venue's window) excluded
    sess = cd.session_mask(ts, a.session) & cd.tradeable(ts, a.horizon, venue_window=not a.july_filter)
    moved = np.isfinite(fwd) & (fwd != 0.0)
    others = {}
    for p in pool:
        Xp, fp, tp, fe = cd.load(a.dataset, p, a.horizon, a.features, 1)
        if fe != feats:
            raise SystemExit(f'{p}: feature columns differ from the target')
        keep = (cd.session_mask(tp, a.session) & cd.tradeable(tp, a.horizon, venue_window=not a.july_filter)
                & np.isfinite(fp) & (fp != 0.0))
        others[p] = (Xp[keep], (fp[keep] > 0).astype(int), tp[keep])
    Xo = np.vstack([v[0] for v in others.values()])
    yo = np.concatenate([v[1] for v in others.values()])
    to = np.concatenate([v[2] for v in others.values()])
    tgs = ts[sess]
    bnds = [tgs[int(k * len(tgs) / cd.N_GROUPS)] for k in range(cd.N_GROUPS)] + [tgs[-1] + 1]
    groups = [(int(bnds[g]), int(bnds[g + 1])) for g in range(cd.N_GROUPS)]
    day = lambda s: pd.to_datetime(s, unit='s', utc=True).strftime('%Y-%m-%d')
    data_uprate = float((fwd[sess & moved] > 0).mean())
    manifest = Path(a.dataset) / 'manifest.json'
    res = {
        'harness': POOL_VERSION, 'base_harness': cd.HARNESS_VERSION, 'tag': a.tag,
        'model': f'POOLED LightGBM (repository mk_lgb) @{a.horizon}m on the Deriv feed, target {a.pair}, pool {pool}, '
                 f'session {a.session}, ties-strict per-fold-refit CPCV',
        'key': f'{a.pair}.{a.horizon}m', 'data': 'Deriv 1-second mid ticks only (/mnt/data/deriv), no Dukascopy',
        'dataset': {'dir': a.dataset, 'manifest_sha256': cd.sha256(manifest)},
        'window': {'first_decision': day(ts[0]), 'last_decision': day(ts[-1])},
        'breakeven': cd.BE, 'session': a.session, 'tradeable_only': True, 'blackout_rule': cd.BLACKOUT_RULE,
        'venue_window_rule': None if a.july_filter else cd.VENUE_WINDOW_RULE,
        'regression_only': bool(a.july_filter),     # True: never a candidate result
        'run_status': 'running',                    # 'complete' when written at the end
        'horizon_min': a.horizon, 'gap_s': gap, 'covs': covs,
        'operating_cov': covs[0], 'nseed': a.nseed, 'num_leaves': a.num_leaves, 'feature_set': a.features,
        'n_features': len(feats), 'features': feats, 'pool': pool,
        'pool_rows_available': {p: int(len(v[1])) for p, v in others.items()},
        'n_groups': cd.N_GROUPS, 'k_test': cd.K_TEST, 'purge_s': gap, 'embargo_s': gap, 'sub_fit': cd.SUB_FIT,
        'val_frac': cd.VAL_FRAC, 'min_trades_per_path': cd.MIN_TRADES, 'max_paths': a.max_paths,
        'rows': int(len(ts)), 'rows_in_session': int(sess.sum()), 'in_session_moved_up_rate': data_uprate,
        'groups': [(day(lo), day(hi - 1)) for lo, hi in groups],
        'CERT_RULE': 'side CERTIFIED iff VALID AND p10>=0.541 AND frac_clear_BE>=0.80 (unrounded path win rates; paths with >= 25 trades)',
        'falsifier': {'version': 3, 'registered_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      'when': 'before any path is fitted',
                      'KILL_if': f'at the operating coverage {covs[0]}: each side p10<0.541 OR frac_clear_BE<0.80',
                      'INVALID_if': 'in-session moved up-rate outside [0.47,0.53], or any path moved up-rate outside [0.45,0.55]'},
        'code': {'pool_harness_sha256': cd.sha256(__file__), 'base_harness_sha256': cd.sha256(cd.__file__),
                 'sessions_py_sha256': cd.sha256(cd.REPO / 'scripts' / 'sessions.py'), 'lightgbm': cd.lgb.__version__},
    }
    out.write_text(cd.result_json(res))
    print(f'[{a.pair} pooled {a.horizon}m {a.session}] target rows {len(ts):,} in-session {int(sess.sum()):,}; '
          f'pool rows {len(yo):,}; features {len(feats)}', flush=True)

    fitter = PoolFitter(X, fwd, ts, a, gap, covs, Xo, yo)
    rng = np.random.default_rng(13)
    pathinfo = []
    for fi, testg in enumerate(combinations(range(cd.N_GROUPS), cd.K_TEST)):
        if fi >= a.max_paths:
            break
        tin = np.zeros(len(ts), bool)
        purged = np.zeros(len(ts), bool)
        opurged = np.zeros(len(to), bool)
        for g in testg:
            lo, hi = groups[g]
            tin |= (ts >= lo) & (ts < hi)
            purged |= (ts >= lo - gap) & (ts < hi + gap)
            opurged |= (to >= lo - gap) & (to < hi + gap)
        row = fitter.split_pool(np.where((~purged) & moved & sess)[0], ~opurged, tin & sess & np.isfinite(fwd), rng, list(testg))
        pathinfo.append(row)
        print(f'  path {fi + 1:2d}/15 g{list(testg)} AUC={row["auc"]:.4f} up={row["up_rate"]:.4f} n_fit={row["n_fit"]} | '
              f'cov{covs[0]}: {cd.fmt_side(row["bycov"][f"{covs[0]}"])} ({time.time() - t0:.0f}s)', flush=True)
    forward = []
    if not a.no_forward:
        frng = np.random.default_rng(17)
        for g in range(2, cd.N_GROUPS):
            lo, hi = groups[g]
            tr = np.where((ts < lo - gap) & moved & sess)[0]
            row = fitter.split_pool(tr, to < lo - gap, (ts >= lo) & (ts < hi) & sess & np.isfinite(fwd), frng, f'forward g{g}')
            forward.append(row)
            print(f'  forward g{g} AUC={row["auc"]:.4f} | cov{covs[0]}: {cd.fmt_side(row["bycov"][f"{covs[0]}"])} '
                  f'({time.time() - t0:.0f}s)', flush=True)
    cd.aggregate(res, pathinfo, forward, covs, data_uprate, fitter.gain, feats)
    if a.max_paths < 15:                         # smoke runs never certify
        for k in res['bycov']:
            for s in cd.SIDES:
                res['bycov'][k]['verdict'][s]['CERTIFIED'] = False
        res['verdict'] = {s: f'SMOKE ({a.max_paths} paths)' for s in ('UP', 'DOWN')}
    res['seconds'] = round(time.time() - t0, 1)
    res['run_status'] = 'complete'
    out.write_text(cd.result_json(res))
    cd.print_summary(res, covs, f'{a.pair} {a.horizon}m pooled', t0, out)


if __name__ == '__main__':
    main()
