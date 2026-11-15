#!/usr/bin/env python3
"""Deriv-only per-fold-refit CPCV certification harness, plus a forward-only walk diagnostic.

    cpcv_deriv.py --pair AUDUSD --horizon 15 [--session ny] [--covs 0.03,0.02,0.05] [--nseed 1]
                  [--stride 1] [--features all|noclock|own|own_noclock] [--feature-list cols.json]
                  [--num-leaves 255] [--dataset DIR] [--out result.json] [--tag TEXT] [--no-forward]

Mirror of the repository's 15m session certification (scripts/audusd_15m_cpcv_session.py), applied
to a Deriv-only dataset (tools/deriv_dataset.py or tools/deriv_features_b239.py), with one deliberate
difference: labels are Deriv-settled from ticks (entry = first tick at/after decision + 1 s, exit =
last tick at/before entry target + H, ties valid and LOSE) instead of bar close to bar close.

CPCV protocol (unchanged from the repository script):
  - session gate on the decision time with sessions.session_mask (DST-correct IANA time zones);
  - 6 equal-count groups of in-session decisions; every C(6,2) = 15 pair of groups is a test "path";
  - purge and embargo of H minutes around each test block; train on moved, in-session, unpurged rows;
  - a random 15% of the training rows is the validation set (early stopping 80 rounds, threshold
    source); the fit pool is subsampled to at most 150,000 rows; rng seed 13;
  - LightGBM with the repository's mk_lgb parameters, averaged over NSEED seeds;
  - threshold = quantile(|p_val - 0.5|, 1 - coverage); trades = side filter and confidence >= threshold,
    de-overlapped first come first served (nonoverlap_chrono, gap H minutes); win = right sign AND moved;
  - a path counts when it has >= 25 trades; a side is CERTIFIED iff p10 >= 0.541 and >= 80% of the
    counted paths are >= 0.541. The falsifier is written to the result file before any path is fitted.
  - validity (as the repository applies it): the moved up-rate of all in-session decisions must lie in
    [0.47, 0.53], and every path's in [0.45, 0.55] (the certifier's tripwire).
Diagnostics (not part of the CERTIFIED rule): per-path AUC on moved rows; the no-skill baselines
"always UP" / "always DOWN" on the same de-overlapped test decisions and each side's excess over them;
a FORWARD walk (train only on groups strictly before group g, purge H before it, test on group g, for
g = 2..5, rng seed 17) that guards against flanked-fold memorization of slow features (trap 9).
"""
import argparse
import hashlib
import json
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path('/home/sean/git/binary-algo')
sys.path.insert(0, str(REPO / 'scripts'))
from sessions import session_mask  # noqa: E402  (the repository's single source of truth for sessions)

warnings.filterwarnings('ignore', message='X does not have valid feature names')
HARNESS_VERSION = 'cpcv_deriv/0.2.1'   # 0.2.1: writable feature matrix copy (no change in results)
LAB = Path('/mnt/data/deriv-sota-lab')
BE = 0.541
N_GROUPS, K_TEST = 6, 2
SUB_FIT = 150_000
MIN_TRADES = 25
VAL_FRAC = 0.15
CLOCK = ('hour_sin', 'hour_cos', 'dow')
SIDES = ('UP', 'DOWN', 'COMBINED')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def feature_list(columns, which):
    feats = [c for c in columns if c not in ('bar_start_ts', 'decision_ts') and not c.startswith(('ret_fwd_', 'valid_'))]
    if which in ('noclock', 'own_noclock'):
        feats = [c for c in feats if c not in CLOCK]
    if which in ('own', 'own_noclock'):
        feats = [c for c in feats if not c.startswith('x_')]
    return feats


def load(dataset, pair, horizon, which, stride, listed=None):
    df = pd.read_parquet(Path(dataset) / f'{pair}.parquet')
    feats = feature_list(df.columns, which)
    if listed is not None:                      # an explicit schema (for example a frozen book's columns)
        missing = [c for c in listed if c not in df.columns]
        if missing:
            raise SystemExit(f'dataset lacks {len(missing)} listed features, e.g. {missing[:8]}')
        feats = list(listed)
    ts = df['decision_ts'].to_numpy(np.int64)
    fwd = df[f'ret_fwd_{horizon}m'].to_numpy(np.float64).copy()
    fwd[~df[f'valid_{horizon}m'].to_numpy(bool)] = np.nan
    X = df[feats].to_numpy(np.float32, copy=True)   # pandas 3 copy-on-write can hand back a read-only view
    X[~np.isfinite(X)] = np.nan
    keep = np.isfinite(fwd) & (np.isnan(X).mean(axis=1) < 0.5)
    idx = np.where(keep)[0]
    if stride > 1:
        idx = idx[::stride]
    idx = idx[np.argsort(ts[idx], kind='stable')]
    return X[idx], fwd[idx], ts[idx], feats


def mk_lgb(seed, num_leaves, n_jobs):
    return lgb.LGBMClassifier(objective='binary', metric='auc', learning_rate=0.02, num_leaves=num_leaves,
                              min_child_samples=400, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                              reg_lambda=20, n_estimators=800, n_jobs=n_jobs, verbosity=-1, random_state=seed,
                              bagging_seed=seed, feature_fraction_seed=seed)


def nonoverlap_chrono(ts, mask, gap):
    take, block = [], -1
    for i in np.where(mask)[0]:
        if ts[i] < block:
            continue
        take.append(i)
        block = int(ts[i]) + gap
    return np.array(take, dtype=int)


def side_wr(pr, fwd, ts, thr, side, gap):
    conf = np.abs(pr - 0.5)
    want = (pr > 0.5) if side == 'UP' else (pr < 0.5) if side == 'DOWN' else np.ones(len(pr), bool)
    tr = nonoverlap_chrono(ts, want & (conf >= thr) & np.isfinite(fwd), gap)
    if len(tr) == 0:
        return 0, float('nan')
    pred = (pr[tr] > 0.5).astype(int)
    ylab = (fwd[tr] > 0).astype(int)
    moved = fwd[tr] != 0
    return len(tr), float(((pred == ylab) & moved).mean())


def always(fwd, ts, side, gap):
    tr = nonoverlap_chrono(ts, np.isfinite(fwd), gap)
    if len(tr) == 0:
        return 0, float('nan')
    win = (fwd[tr] > 0) if side == 'UP' else (fwd[tr] < 0)     # a tie loses either way
    return len(tr), float(win.mean())


def summ(a):
    a = np.asarray([x for x in a if np.isfinite(x)], float)
    if len(a) == 0:
        return {'n_paths': 0}
    return {'n_paths': int(len(a)), 'mean': round(float(a.mean()), 4), 'p10': round(float(np.percentile(a, 10)), 4),
            'p50': round(float(np.percentile(a, 50)), 4), 'min': round(float(a.min()), 4),
            'frac_clear_BE': round(float((a >= BE).mean()), 3)}


class Fitter:
    def __init__(self, X, fwd, ts, a, gap, covs):
        self.X, self.fwd, self.ts, self.a, self.gap, self.covs = X, fwd, ts, a, gap, covs
        self.gain = np.zeros(X.shape[1])

    def split(self, tr_idx, test_mask, rng, label):
        """Fit on tr_idx (moved rows), threshold from a random validation subset, score test_mask."""
        X, fwd, ts, a, gap = self.X, self.fwd, self.ts, self.a, self.gap
        valsel = rng.random(len(tr_idx)) < VAL_FRAC
        val_idx = tr_idx[valsel]
        fit_pool = tr_idx[~valsel]
        if len(fit_pool) > SUB_FIT:
            fit_pool = np.sort(rng.choice(fit_pool, SUB_FIT, replace=False))
        ytr = (fwd[fit_pool] > 0).astype(int)
        yval = (fwd[val_idx] > 0).astype(int)
        pval = np.zeros(len(val_idx))
        ptst = np.zeros(int(test_mask.sum()))
        iters = []
        for sd in range(a.nseed):
            m = mk_lgb(sd, a.num_leaves, a.n_jobs)
            m.fit(X[fit_pool], ytr, eval_set=[(X[val_idx], yval)], eval_metric='auc',
                  callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
            pval += m.predict_proba(X[val_idx])[:, 1]
            ptst += m.predict_proba(X[test_mask])[:, 1]
            self.gain += m.booster_.feature_importance(importance_type='gain')
            iters.append(int(m.best_iteration_ or 0))
        pval /= a.nseed
        ptst /= a.nseed
        vconf = np.abs(pval - 0.5)
        ftst = fwd[test_mask]
        ttst = ts[test_mask]                      # chronological (rows are sorted by decision time)
        mv = ftst != 0
        auc = float(roc_auc_score((ftst[mv] > 0).astype(int), ptst[mv])) if mv.sum() > 20 else float('nan')
        uprate = float((ftst[mv] > 0).mean()) if mv.sum() else float('nan')
        row = {'test': label, 'n_fit': int(len(fit_pool)), 'n_val': int(len(val_idx)), 'n_test': int(test_mask.sum()),
               'best_iter': iters, 'auc': round(auc, 4), 'up_rate': round(uprate, 4), 'always': {}, 'bycov': {}}
        for s in ('UP', 'DOWN'):
            n, w = always(ftst, ttst, s, gap)
            row['always'][s] = [n, round(w, 4)]
        for c in self.covs:
            thr = float(np.quantile(vconf, 1 - c))
            row['bycov'][f'{c}'] = {'thr': round(thr, 5)}
            for s in SIDES:
                n, w = side_wr(ptst, ftst, ttst, thr, s, gap)
                row['bycov'][f'{c}'][s] = [n, round(w, 4)]
        return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', required=True)
    ap.add_argument('--horizon', type=int, default=15, choices=(15, 30))
    ap.add_argument('--session', default='ny', choices=('ny', 'ldn', 'asia', 'overlap', 'all'))
    ap.add_argument('--covs', default='0.03,0.02,0.05')
    ap.add_argument('--nseed', type=int, default=1)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--features', default='all', choices=('all', 'noclock', 'own', 'own_noclock'))
    ap.add_argument('--feature-list', default=None, help='JSON file with an explicit list of feature columns')
    ap.add_argument('--num-leaves', type=int, default=255)
    ap.add_argument('--dataset', default=str(LAB / 'cache' / 'dataset_dev_v0.1'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--n-jobs', type=int, default=16)
    ap.add_argument('--no-forward', action='store_true')
    a = ap.parse_args()
    covs = [float(c) for c in a.covs.split(',')]
    gap = a.horizon * 60
    out = Path(a.out or f'{a.pair}_{a.horizon}m_{a.session}_cpcv_result.json')
    t0 = time.time()

    listed = json.loads(Path(a.feature_list).read_text()) if a.feature_list else None
    X, fwd, ts, feats = load(a.dataset, a.pair, a.horizon, a.features, a.stride, listed)
    if listed is not None:
        a.features = f'list:{Path(a.feature_list).name}'
    sess = session_mask(ts, a.session)
    moved = np.isfinite(fwd) & (fwd != 0.0)
    tgs = ts[sess]
    bnds = [tgs[int(k * len(tgs) / N_GROUPS)] for k in range(N_GROUPS)] + [tgs[-1] + 1]
    groups = [(int(bnds[g]), int(bnds[g + 1])) for g in range(N_GROUPS)]
    day = lambda s: pd.to_datetime(s, unit='s', utc=True).strftime('%Y-%m-%d')
    sm_ = sess & moved
    data_uprate = float((fwd[sm_] > 0).mean())
    manifest = Path(a.dataset) / 'manifest.json'
    res = {
        'harness': HARNESS_VERSION, 'tag': a.tag,
        'model': f'LightGBM (repository mk_lgb) @{a.horizon}m on the Deriv feed, session {a.session}, '
                 f'ties-strict per-fold-refit CPCV',
        'key': f'{a.pair}.{a.horizon}m', 'data': 'Deriv 1-second mid ticks only (/mnt/data/deriv), no Dukascopy',
        'dataset': {'dir': a.dataset, 'manifest_sha256': sha256(manifest) if manifest.exists() else None,
                    'builder': json.loads(manifest.read_text()).get('builder') if manifest.exists() else None},
        'window': {'first_decision': day(ts[0]), 'last_decision': day(ts[-1])},
        'label': f'Deriv settlement over {a.horizon} min: entry first tick >= decision + 1 s, exit last tick <= '
                 f'entry target + {a.horizon} min, ties valid and lose',
        'breakeven': BE, 'session': a.session, 'horizon_min': a.horizon, 'gap_s': gap, 'stride': a.stride,
        'covs': covs, 'operating_cov': covs[0], 'nseed': a.nseed, 'num_leaves': a.num_leaves,
        'feature_set': a.features, 'n_features': len(feats), 'features': feats,
        'n_groups': N_GROUPS, 'k_test': K_TEST, 'purge_s': gap, 'embargo_s': gap, 'sub_fit': SUB_FIT,
        'val_frac': VAL_FRAC, 'min_trades_per_path': MIN_TRADES,
        'rows': int(len(ts)), 'rows_in_session': int(sess.sum()), 'in_session_moved_up_rate': round(data_uprate, 4),
        'groups': [(day(lo), day(hi - 1)) for lo, hi in groups],
        'CERT_RULE': 'side CERTIFIED iff p10>=0.541 AND frac_clear_BE>=0.80 (paths with >= 25 trades)',
        'falsifier': {'version': 2, 'registered_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      'when': 'before any path is fitted',
                      'KILL_if': f'at the operating coverage {covs[0]}: each side p10<0.541 OR frac_clear_BE<0.80',
                      'INVALID_if': 'in-session moved up-rate outside [0.47,0.53], or any path moved up-rate outside '
                                    '[0.45,0.55] (the repository certifier tripwire)',
                      'note': 'version 1 (smoke test only) applied the [0.47,0.53] band per path; USDJPY 15m tripped it '
                              'through genuine 2026 drift (path up-rate 0.541) on tick-settled labels with no forward '
                              'fill; version 2 applies the band exactly as the repository certifier does'},
        'code': {'harness_sha256': sha256(__file__), 'sessions_py_sha256': sha256(REPO / 'scripts' / 'sessions.py'),
                 'lightgbm': lgb.__version__, 'numpy': np.__version__, 'pandas': pd.__version__},
    }
    out.write_text(json.dumps(res, indent=1))
    print(f'[{a.pair} {a.horizon}m {a.session}] rows {len(ts):,} in-session {int(sess.sum()):,} features {len(feats)} '
          f'up-rate {data_uprate:.4f} groups {res["groups"]}', flush=True)

    fitter = Fitter(X, fwd, ts, a, gap, covs)
    rng = np.random.default_rng(13)
    pathinfo = []
    for fi, testg in enumerate(combinations(range(N_GROUPS), K_TEST)):
        tin = np.zeros(len(ts), bool)
        purged = np.zeros(len(ts), bool)
        for g in testg:
            lo, hi = groups[g]
            tin |= (ts >= lo) & (ts < hi)
            purged |= (ts >= lo - gap) & (ts < hi + gap)
        row = fitter.split(np.where((~purged) & moved & sess)[0], tin & sess & np.isfinite(fwd), rng, list(testg))
        pathinfo.append(row)
        b = row['bycov'][f'{covs[0]}']
        print(f'  path {fi + 1:2d}/15 g{list(testg)} AUC={row["auc"]:.4f} up={row["up_rate"]:.4f} iters={row["best_iter"]} | '
              f'cov{covs[0]}: UP {b["UP"]} DOWN {b["DOWN"]} COMB {b["COMBINED"]} | always UP {row["always"]["UP"]} '
              f'({time.time() - t0:.0f}s)', flush=True)
    forward = []
    if not a.no_forward:
        frng = np.random.default_rng(17)
        for g in range(2, N_GROUPS):
            lo, hi = groups[g]
            tr = np.where((ts < lo - gap) & moved & sess)[0]
            row = fitter.split(tr, (ts >= lo) & (ts < hi) & sess & np.isfinite(fwd), frng, f'forward g{g}')
            forward.append(row)
            b = row['bycov'][f'{covs[0]}']
            print(f'  forward g{g} (train groups 0..{g - 1}) AUC={row["auc"]:.4f} up={row["up_rate"]:.4f} | cov{covs[0]}: '
                  f'UP {b["UP"]} DOWN {b["DOWN"]} COMB {b["COMBINED"]} ({time.time() - t0:.0f}s)', flush=True)

    res['paths'] = pathinfo
    res['forward'] = forward
    aucs = [p['auc'] for p in pathinfo]
    ups = [p['up_rate'] for p in pathinfo]
    res['auc_summary'] = {'mean': round(float(np.nanmean(aucs)), 4), 'min': round(float(np.nanmin(aucs)), 4),
                          'max': round(float(np.nanmax(aucs)), 4)}
    res['validity'] = {'in_session_up_rate_in_band': bool(0.47 <= data_uprate <= 0.53),
                       'paths_up_rate_in_tripwire': bool(all(0.45 <= u <= 0.55 for u in ups if np.isfinite(u))),
                       'paths_up_rate_in_0.47_0.53': bool(all(0.47 <= u <= 0.53 for u in ups if np.isfinite(u))),
                       'path_up_rate_range': [round(float(np.nanmin(ups)), 4), round(float(np.nanmax(ups)), 4)]}
    res['baseline_always'] = {s: summ([p['always'][s][1] for p in pathinfo]) for s in ('UP', 'DOWN')}
    res['bycov'] = {}
    for c in covs:
        k = f'{c}'
        sm, vd = {}, {}
        for s in SIDES:
            kept = [p['bycov'][k][s] for p in pathinfo if p['bycov'][k][s][0] >= MIN_TRADES]
            sm[s] = summ([w for _, w in kept])
            p10 = sm[s].get('p10', float('nan'))
            fc = sm[s].get('frac_clear_BE', 0.0)
            vd[s] = {'p10': p10, 'frac_clear_BE': fc, 'med_n_per_path': int(np.median([n for n, _ in kept])) if kept else 0,
                     'CERTIFIED': bool(np.isfinite(p10) and p10 >= BE and fc >= 0.80 and sm[s]['n_paths'] >= 1)}
            if s != 'COMBINED':
                ex = [p['bycov'][k][s][1] - p['always'][s][1] for p in pathinfo if p['bycov'][k][s][0] >= MIN_TRADES]
                vd[s]['excess_vs_always'] = {'mean': round(float(np.mean(ex)), 4) if ex else None,
                                             'frac_paths_positive': round(float(np.mean([e > 0 for e in ex])), 3) if ex else None}
        fw = {}
        for s in SIDES:
            vals = [(r['bycov'][k][s][0], r['bycov'][k][s][1]) for r in forward]
            ok = [w for n, w in vals if n >= MIN_TRADES]
            fw[s] = {'paths': vals, 'mean': round(float(np.mean(ok)), 4) if ok else None,
                     'min': round(float(np.min(ok)), 4) if ok else None,
                     'n_clear_BE': int(sum(w >= BE for w in ok)), 'n_counted': len(ok)}
        res['bycov'][k] = {'summary': sm, 'verdict': vd, 'forward': fw}
    op = res['bycov'][f'{covs[0]}']['verdict']
    res['verdict'] = {s: ('CERTIFIED' if op[s]['CERTIFIED'] else 'KILLED') for s in ('UP', 'DOWN')}
    if not (res['validity']['in_session_up_rate_in_band'] and res['validity']['paths_up_rate_in_tripwire']):
        res['verdict'] = {s: 'INVALID (moved up-rate validity check failed)' for s in ('UP', 'DOWN')}
    order = np.argsort(-fitter.gain)[:20]
    res['top_gain_features'] = [[feats[i], round(float(fitter.gain[i] / fitter.gain.sum()), 4)] for i in order]
    res['seconds'] = round(time.time() - t0, 1)
    out.write_text(json.dumps(res, indent=1))
    print(f'[{a.pair} {a.horizon}m] AUC {res["auc_summary"]} validity {res["validity"]} '
          f'always UP {res["baseline_always"]["UP"].get("mean")} DOWN {res["baseline_always"]["DOWN"].get("mean")}', flush=True)
    for c in covs:
        for s in SIDES:
            v = res['bycov'][f'{c}']['verdict'][s]
            f = res['bycov'][f'{c}']['forward'][s]
            print(f'   cov{c} {s:8s} p10={v["p10"]} mean={res["bycov"][f"{c}"]["summary"][s].get("mean")} '
                  f'frac={v["frac_clear_BE"]} med_n={v["med_n_per_path"]} CERT={v["CERTIFIED"]} | forward mean={f["mean"]} '
                  f'min={f["min"]} clear={f["n_clear_BE"]}/{f["n_counted"]}', flush=True)
    print(f'verdict {res["verdict"]} in {res["seconds"]} s -> {out}', flush=True)


if __name__ == '__main__':
    main()
