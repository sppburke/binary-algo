#!/usr/bin/env python3
"""Variant harness on top of tools/cpcv_deriv.py: same CPCV, scoring, validity rule and falsifier, plus
knobs the base mirror does not expose.

    cpcv_deriv_x.py --pair P --horizon 15 [--session ny] [--train-session all] [--covs 0.03,0.02,0.05]
                    [--nseed 1] [--features all|noclock|own|own_noclock] [--feature-list cols.json]
                    [--num-leaves 255] [--min-child 400] [--lr 0.02] [--n-estimators 800] [--colsample 0.5]
                    [--reg-lambda 20] [--dataset DIR] [--out result.json] [--tag TEXT] [--no-forward] [--max-paths N]

--train-session widens (or changes) the rows the model is FIT on; groups, validation rows (early stopping and
the threshold quantile), test paths and every reported number stay on the evaluation --session. Purge and
embargo apply to all rows. --val-mode block replaces the repository's random 15% validation rows (label-
overlapping neighbours of fit rows: overconfident thresholds, early stopping that never fires) with the last
15% of the training rows in time, purged one horizon from the fit rows.
Defaults reproduce cpcv_deriv.py exactly (same rng stream and parameters; checked on E1 USDJPY path 1).
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

X_VERSION = 'cpcv_deriv_x/0.3'   # 0.3: shared unrounded aggregation (cpcv_deriv 0.4); 0.2: blackout excluded
# Extra local-time windows (America/New_York, [start, end) hours) beyond sessions.py. 'nylate' = 13:00-22:00 ET: the
# TRUE hours selected by session_mask('ny') on the Dukascopy-derived stores before 2025-08-22, whose timestamps were
# true UTC - 5 h (see docs/DERIV_SOTA_PROGRAM.md) - i.e. where the incumbents' "NY" certifications actually lived.
LOCAL_WINDOWS = {'nylate': (13, 22), 'nyam': (8, 13), 'nypm': (13, 17), 'nyeve': (17, 22)}


def session(ts, name):
    if name in LOCAL_WINDOWS:
        lo, hi = LOCAL_WINDOWS[name]
        h = pd.to_datetime(np.asarray(ts), unit='s', utc=True).tz_convert('America/New_York').hour.values
        return (h >= lo) & (h < hi)
    return cd.session_mask(ts, name)


def make_model(a, seed):
    return cd.lgb.LGBMClassifier(objective='binary', metric='auc', learning_rate=a.lr, num_leaves=a.num_leaves,
                                 min_child_samples=a.min_child, subsample=0.8, subsample_freq=1, colsample_bytree=a.colsample,
                                 reg_lambda=a.reg_lambda, n_estimators=a.n_estimators, n_jobs=a.n_jobs, verbosity=-1,
                                 random_state=seed, bagging_seed=seed, feature_fraction_seed=seed)


class XFitter(cd.Fitter):
    def split2(self, tr_idx, val_pool, test_mask, rng, label):
        """Fit on tr_idx; validation = random VAL_FRAC of the evaluation-session rows among tr_idx (val_pool mask)."""
        X, fwd, ts, a, gap = self.X, self.fwd, self.ts, self.a, self.gap
        if a.val_mode == 'block':
            # validation = the chronologically last VAL_FRAC of the evaluation-session training rows; fit rows end one
            # horizon before it (purge), so early stopping and the threshold see genuinely unseen, later data
            cand = tr_idx[val_pool[tr_idx]]
            val_idx = cand[-max(1, int(round(len(cand) * cd.VAL_FRAC))):]
            fit_pool = tr_idx[ts[tr_idx] < ts[val_idx[0]] - gap]
        else:
            r = rng.random(len(tr_idx))
            valsel = (r < cd.VAL_FRAC) & val_pool[tr_idx]
            val_idx = tr_idx[valsel]
            fit_pool = tr_idx[~valsel]
        if self.ytrain is not None:                 # alternative TRAINING target (validation keeps the settlement sign)
            fit_pool = fit_pool[np.isfinite(self.ytrain[fit_pool])]
        if len(fit_pool) > cd.SUB_FIT:
            fit_pool = rng.choice(fit_pool, cd.SUB_FIT, replace=False)     # sampled order kept, as the reference
        ytr = (fwd[fit_pool] > 0).astype(int) if self.ytrain is None else self.ytrain[fit_pool].astype(int)
        yval = (fwd[val_idx] > 0).astype(int)
        pval = np.zeros(len(val_idx))
        ptst = np.zeros(int(test_mask.sum()))
        iters = []
        for sd in range(a.nseed):
            m = make_model(a, sd)
            if a.no_early_stop:                     # validation rows only set the threshold
                m.fit(X[fit_pool], ytr)
            else:
                m.fit(X[fit_pool], ytr, eval_set=[(X[val_idx], yval)], eval_metric='auc',
                      callbacks=[cd.lgb.early_stopping(80, verbose=False), cd.lgb.log_evaluation(0)])
            pval += m.predict_proba(X[val_idx])[:, 1]
            ptst += m.predict_proba(X[test_mask])[:, 1]
            self.gain += m.booster_.feature_importance(importance_type='gain')
            iters.append(int(m.best_iteration_ or 0))
        pval /= a.nseed
        ptst /= a.nseed
        vconf = np.abs(pval - 0.5)
        ftst = fwd[test_mask]
        ttst = ts[test_mask]
        mv = ftst != 0
        auc = float(cd.roc_auc_score((ftst[mv] > 0).astype(int), ptst[mv])) if mv.sum() > 20 else float('nan')
        uprate = float((ftst[mv] > 0).mean()) if mv.sum() else float('nan')
        row = {'test': label, 'n_fit': int(len(fit_pool)), 'n_val': int(len(val_idx)), 'n_test': int(test_mask.sum()),
               'best_iter': iters, 'auc': auc, 'up_rate': uprate, 'always': {}, 'bycov': {}}
        vgate = np.ones(len(val_idx), bool)
        if a.mag_keep is not None:
            # MAGNITUDE GATE (sign-invariant): a second model ranks "big move" (|ret| >= the fit pool's median |ret|);
            # keep only the top mag_keep share (threshold = validation quantile); the direction threshold is then the
            # coverage quantile among validation rows that pass the gate, so coverage is relative to gated rows.
            med = float(np.median(np.abs(fwd[fit_pool])))
            mm = make_model(a, 0)
            mm.fit(X[fit_pool], (np.abs(fwd[fit_pool]) >= med).astype(int))
            mval = mm.predict_proba(X[val_idx])[:, 1]
            mtst = mm.predict_proba(X[test_mask])[:, 1]
            gthr = float(np.quantile(mval, 1 - a.mag_keep))
            vgate = mval >= gthr
            tgate = mtst >= gthr
            big = (np.abs(ftst) >= med).astype(int)
            row['mag_gate'] = {'keep': a.mag_keep, 'median_abs_ret': med, 'gate_thr': gthr,
                               'test_pass_rate': float(tgate.mean()),
                               'mag_auc_test': float(cd.roc_auc_score(big, mtst)) if 0 < big.sum() < len(big) else None,
                               'tie_rate_pass': float((ftst[tgate] == 0).mean()) if tgate.any() else None,
                               'tie_rate_fail': float((ftst[~tgate] == 0).mean()) if (~tgate).any() else None}
            ptst = np.where(tgate, ptst, 0.5)        # gated-out rows get zero confidence: never selected
        return cd.score_row(row, ptst, ftst, ttst, vconf[vgate], self.covs, gap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', required=True)
    ap.add_argument('--horizon', type=int, default=15, choices=(15, 30))
    sessions = ('ny', 'ldn', 'asia', 'overlap', 'all') + tuple(LOCAL_WINDOWS)
    ap.add_argument('--session', default='ny', choices=sessions)
    ap.add_argument('--train-session', default=None, choices=sessions)
    ap.add_argument('--covs', default='0.03,0.02,0.05')
    ap.add_argument('--nseed', type=int, default=1)
    ap.add_argument('--features', default='all', choices=('all', 'noclock', 'own', 'own_noclock'))
    ap.add_argument('--feature-list', default=None)
    ap.add_argument('--num-leaves', type=int, default=255)
    ap.add_argument('--min-child', type=int, default=400)
    ap.add_argument('--lr', type=float, default=0.02)
    ap.add_argument('--n-estimators', type=int, default=800)
    ap.add_argument('--colsample', type=float, default=0.5)
    ap.add_argument('--reg-lambda', type=float, default=20.0)
    ap.add_argument('--val-mode', default='random', choices=('random', 'block'),
                    help='random = repository protocol (15%% random rows); block = last 15%% of training rows in time, purged')
    ap.add_argument('--no-early-stop', action='store_true', help='fit all n_estimators; validation only sets the threshold')
    ap.add_argument('--train-label', default='sign', choices=('sign', 'tb1', 'tb2'),
                    help='fit target: Deriv settlement sign (default) or triple-barrier first touch k=1/2 (validation, '
                         'thresholds and scoring always use the settlement sign)')
    ap.add_argument('--label-dir', default=str(cd.LAB / 'cache' / 'labels_tb_dev_v0.1'))
    ap.add_argument('--include-blackout', action='store_true', help='keep Deriv purchase-blackout decisions (not Deriv-faithful)')
    ap.add_argument('--mag-keep', type=float, default=None,
                    help='magnitude gate: keep only the top share (e.g. 0.5) of rows by predicted big-move probability')
    ap.add_argument('--dataset', default=str(cd.LAB / 'cache' / 'dataset_dev_v0.1'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--n-jobs', type=int, default=16)
    ap.add_argument('--no-forward', action='store_true')
    ap.add_argument('--max-paths', type=int, default=15)
    a = ap.parse_args()
    a.train_session = a.train_session or a.session
    covs = [float(c) for c in a.covs.split(',')]
    gap = a.horizon * 60
    out = Path(a.out or f'{a.pair}_{a.horizon}m_{a.session}_x_cpcv_result.json')
    t0 = time.time()
    listed = json.loads(Path(a.feature_list).read_text()) if a.feature_list else None
    X, fwd, ts, feats = cd.load(a.dataset, a.pair, a.horizon, a.features, 1, listed)
    sess = session(ts, a.session)
    tsess = session(ts, a.train_session)
    if not a.include_blackout:                   # Deriv purchase blackout: untradeable decisions out of every role
        tr_ok = cd.tradeable(ts, a.horizon)
        sess &= tr_ok
        tsess &= tr_ok
    moved = np.isfinite(fwd) & (fwd != 0.0)
    tgs = ts[sess]
    bnds = [tgs[int(k * len(tgs) / cd.N_GROUPS)] for k in range(cd.N_GROUPS)] + [tgs[-1] + 1]
    groups = [(int(bnds[g]), int(bnds[g + 1])) for g in range(cd.N_GROUPS)]
    day = lambda s: pd.to_datetime(s, unit='s', utc=True).strftime('%Y-%m-%d')
    data_uprate = float((fwd[sess & moved] > 0).mean())
    manifest = Path(a.dataset) / 'manifest.json'
    knobs = {k: getattr(a, k) for k in ('train_session', 'num_leaves', 'min_child', 'lr', 'n_estimators', 'colsample', 'reg_lambda',
                                        'val_mode', 'no_early_stop', 'train_label', 'mag_keep', 'include_blackout')}
    res = {
        'harness': X_VERSION, 'base_harness': cd.HARNESS_VERSION, 'tag': a.tag,
        'model': f'LightGBM @{a.horizon}m on the Deriv feed, eval session {a.session}, fit session {a.train_session}, knobs {knobs}',
        'key': f'{a.pair}.{a.horizon}m', 'data': 'Deriv 1-second mid ticks only (/mnt/data/deriv), no Dukascopy',
        'dataset': {'dir': a.dataset, 'manifest_sha256': cd.sha256(manifest)},
        'window': {'first_decision': day(ts[0]), 'last_decision': day(ts[-1])},
        'breakeven': cd.BE, 'session': a.session, 'tradeable_only': not a.include_blackout,
        'blackout_rule': None if a.include_blackout else cd.BLACKOUT_RULE,
        'knobs': knobs, 'horizon_min': a.horizon, 'gap_s': gap, 'covs': covs,
        'operating_cov': covs[0], 'nseed': a.nseed, 'num_leaves': a.num_leaves,
        'feature_set': f'list:{Path(a.feature_list).name}' if a.feature_list else a.features, 'n_features': len(feats), 'features': feats,
        'n_groups': cd.N_GROUPS, 'k_test': cd.K_TEST, 'purge_s': gap, 'embargo_s': gap, 'sub_fit': cd.SUB_FIT,
        'val_frac': cd.VAL_FRAC, 'min_trades_per_path': cd.MIN_TRADES, 'max_paths': a.max_paths,
        'rows': int(len(ts)), 'rows_in_session': int(sess.sum()), 'rows_fit_session': int(tsess.sum()),
        'in_session_moved_up_rate': data_uprate, 'groups': [(day(lo), day(hi - 1)) for lo, hi in groups],
        'CERT_RULE': 'side CERTIFIED iff VALID AND p10>=0.541 AND frac_clear_BE>=0.80 (unrounded path win rates; paths with >= 25 trades)',
        'falsifier': {'version': 3, 'registered_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'when': 'before any path is fitted',
                      'KILL_if': f'at the operating coverage {covs[0]}: each side p10<0.541 OR frac_clear_BE<0.80',
                      'INVALID_if': 'in-session moved up-rate outside [0.47,0.53], or any path moved up-rate outside [0.45,0.55]; an INVALID run certifies nothing in any field'},
        'code': {'x_harness_sha256': cd.sha256(__file__), 'base_harness_sha256': cd.sha256(cd.__file__),
                 'sessions_py_sha256': cd.sha256(cd.REPO / 'scripts' / 'sessions.py'), 'lightgbm': cd.lgb.__version__},
    }
    out.write_text(json.dumps(res, indent=1))
    print(f'[{a.pair} {a.horizon}m eval {a.session} fit {a.train_session}] rows {len(ts):,} eval {int(sess.sum()):,} '
          f'fit-session {int(tsess.sum()):,} features {len(feats)} knobs {knobs}', flush=True)
    fitter = XFitter(X, fwd, ts, a, gap, covs)
    fitter.ytrain = None
    if a.train_label != 'sign':
        lab = pd.read_parquet(Path(a.label_dir) / f'{a.pair}.parquet').set_index('bar_start_ts')[f'{a.train_label}_{a.horizon}m']
        fitter.ytrain = lab.reindex(ts - 60).to_numpy(np.float64)      # decision = bar start + 60 s
        res['train_label'] = {'name': a.train_label, 'dir': a.label_dir,
                              'manifest_sha256': cd.sha256(Path(a.label_dir) / 'manifest.json'),
                              'aligned_finite': int(np.isfinite(fitter.ytrain).sum())}
        out.write_text(json.dumps(res, indent=1))
    rng = np.random.default_rng(13)
    pathinfo = []
    for fi, testg in enumerate(combinations(range(cd.N_GROUPS), cd.K_TEST)):
        if fi >= a.max_paths:
            break
        tin = np.zeros(len(ts), bool)
        purged = np.zeros(len(ts), bool)
        for g in testg:
            lo, hi = groups[g]
            tin |= (ts >= lo) & (ts < hi)
            purged |= (ts >= lo - gap) & (ts < hi + gap)
        row = fitter.split2(np.where((~purged) & moved & tsess)[0], sess, tin & sess & np.isfinite(fwd), rng, list(testg))
        pathinfo.append(row)
        print(f'  path {fi + 1:2d}/15 g{list(testg)} AUC={row["auc"]:.4f} n_fit={row["n_fit"]} iters={row["best_iter"]} | '
              f'cov{covs[0]}: {cd.fmt_side(row["bycov"][f"{covs[0]}"])} ({time.time() - t0:.0f}s)', flush=True)
    forward = []
    if not a.no_forward:
        frng = np.random.default_rng(17)
        for g in range(2, cd.N_GROUPS):
            lo, hi = groups[g]
            row = fitter.split2(np.where((ts < lo - gap) & moved & tsess)[0], sess, (ts >= lo) & (ts < hi) & sess & np.isfinite(fwd),
                                frng, f'forward g{g}')
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
    out.write_text(json.dumps(res, indent=1))
    cd.print_summary(res, covs, f'{a.pair} {a.horizon}m x', t0, out)


if __name__ == '__main__':
    main()
