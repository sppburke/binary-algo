#!/usr/bin/env python3
"""One-time sealed-HOLDOUT evaluation of a finalized candidate (opened once per candidate).

    holdout_eval.py --pair P --horizon 15 --dataset FULL_DIR [--feature-list cols.json | --features all|noclock|own|own_noclock]
                    [--pool PAIR,PAIR,...] [--nseed 3] [--num-leaves 255] [--session ny] --cov 0.03
                    [--holdout-start 2026-08-01] [--holdout-end 2026-11-14] --out result.json [--tag TEXT]

FULL_DIR is a Deriv-only dataset covering DEV and the holdout (features are causal, so the DEV rows equal
the DEV-only build). Two pre-registered readings, both on holdout decisions only (NY DST session, valid
Deriv-settled labels, side filter, threshold, first-come de-overlap gap H, ties lose):
  PRIMARY  walk-forward monthly refit (the deployable form of refit-dependent edges): for each calendar month
           of the holdout, train on every in-session moved row whose label ends before that month starts
           (purge gap H + 2 min), random 15% validation (rng 13 + month index) for early stopping and the
           threshold quantile at --cov, score that month.
  CONTEXT  one model frozen at the end of DEV, scored over the whole holdout.
Optional pooling (--pool) follows tools/cpcv_deriv_pool.py: other pairs' rows in the fit pool only.
Per side: n, win rate, CI95 (iid bootstrap 5000, seed 7), by month; moved-row AUC; always-UP/DOWN baselines.
Falsifier (written before the holdout is scored): a side is SOTA-ELIGIBLE iff the PRIMARY reading has
n >= 50 and CI95-lower >= 0.541; comparison with the incumbents' holdout numbers is done by the caller
(score_frozen_books.py on the same window) and must show a higher point win rate.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))   # cpcv_deriv puts the repository scripts first; the lab harness wins
from cpcv_deriv_x import LOCAL_WINDOWS, session as x_session  # noqa: E402
if Path(sys.modules['cpcv_deriv_x'].__file__).resolve().parent != Path(__file__).resolve().parent:
    raise SystemExit(f'holdout_eval: cpcv_deriv_x resolved to {sys.modules["cpcv_deriv_x"].__file__}, not the harness next to this file')

HOLDOUT_VERSION = 'holdout_eval/0.6.1'  # 0.6.1: falsifier text names owner decision 6 (two-look seq_decision); numbers unchanged
# 0.6: --weight absret (as cpcv_deriv_x); drift test unavailable without volatility cut
#                                        points; Bonferroni flag renamed to context (binding = caller's Holm); harness provenance
#                                        enforced (next to this file)
#                                        0.5: drift_matched = NY hour x rv_60m tercile (binding, owner decision 1) with
#                                        'passes' (CI95 lower > 0); the hour-only baseline kept as drift_matched_hour
#                                        0.4: one-sided bootstrap p (Holm across a key's finalists) and the NY-hour
#                                        drift-matched always-side baseline per side (paired excess + CI; context);
#                                        0.3: --model (cpcv_deriv_x model classes); 0.2: blackout excluded; local NY windows


def boot(corr, nb=5000, seed=7, alpha=0.05):
    corr = np.asarray(corr, dtype=float)
    if len(corr) < 5:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    n = len(corr)
    a = np.array([corr[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 100 * alpha / 2)), float(np.percentile(a, 100 * (1 - alpha / 2)))


def boot_p(corr, thr, nb=5000, seed=7):
    """One-sided bootstrap p-value of mean <= thr (same resamples as boot); the input of a Holm step-down."""
    corr = np.asarray(corr, dtype=float)
    if len(corr) < 5:
        return float('nan')
    rng = np.random.default_rng(seed)
    n = len(corr)
    a = np.array([corr[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float((a <= thr).mean())


def fit_predict(X, fwd, train_idx, score_mask, a, rng, others=None):
    valsel = rng.random(len(train_idx)) < cd.VAL_FRAC
    val_idx = train_idx[valsel]
    own_fit = train_idx[~valsel]
    Xfit, yfit = X[own_fit], (fwd[own_fit] > 0).astype(int)
    afit = np.abs(fwd[own_fit])                                       # |settled return| of the fit rows (moved rows only)
    if others is not None:
        Xo, yo = others
        Xfit, yfit = np.vstack([Xfit, Xo]), np.concatenate([yfit, yo])
    if len(yfit) > cd.SUB_FIT:
        pick = rng.choice(len(yfit), cd.SUB_FIT, replace=False)        # sampled order kept (no sort)
        Xfit, yfit = Xfit[pick], yfit[pick]
        afit = afit[pick] if others is None else afit
    wfit = None if a.weight == 'none' else afit / (afit.mean() + 1e-18)   # as cpcv_deriv_x --weight absret (mean 1)
    yval = (fwd[val_idx] > 0).astype(int)
    pval = np.zeros(len(val_idx))
    ps = np.zeros(int(score_mask.sum()))
    if a.model != 'lgb':                    # the CPCV variant harness's own model code (cpcv_deriv_x.XFitter.fit_one)
        import cpcv_deriv_x as cx
        from types import SimpleNamespace
        ns = SimpleNamespace(model=a.model, nseed=a.nseed, n_jobs=a.n_jobs, lr=0.02, num_leaves=a.num_leaves, min_child=400,
                             colsample=0.5, reg_lambda=20.0, n_estimators=800, no_early_stop=False, cat_depth=a.cat_depth,
                             logit_c=a.logit_c, nys_gamma_mult=a.nys_gamma_mult, nys_components=a.nys_components)
        fx = cx.XFitter(X, fwd, np.zeros(len(fwd), dtype=np.int64), ns, 0, [a.cov])
        fx.wfit = wfit
        for sd in range(a.nseed):
            pv, pt, _ = fx.fit_one(sd, Xfit, yfit, X[val_idx], yval, X[score_mask])
            pval += pv
            ps += pt
    else:
        for sd in range(a.nseed):
            m = cd.mk_lgb(sd, a.num_leaves, a.n_jobs)
            m.fit(Xfit, yfit, eval_set=[(X[val_idx], yval)], eval_metric='auc',
                  callbacks=[cd.lgb.early_stopping(80, verbose=False), cd.lgb.log_evaluation(0)],
                  **({} if wfit is None else {'sample_weight': wfit}))
            pval += m.predict_proba(X[val_idx])[:, 1]
            ps += m.predict_proba(X[score_mask])[:, 1]
    pval /= a.nseed
    ps /= a.nseed
    return ps, float(np.quantile(np.abs(pval - 0.5), 1 - a.cov)), int(len(yfit))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', required=True)
    ap.add_argument('--horizon', type=int, default=15, choices=(15, 30))
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--features', default='all', choices=('all', 'noclock', 'own', 'own_noclock'))
    ap.add_argument('--feature-list', default=None)
    ap.add_argument('--pool', default=None)
    ap.add_argument('--nseed', type=int, default=3)
    ap.add_argument('--num-leaves', type=int, default=255)
    ap.add_argument('--weight', default='none', choices=('none', 'absret'),
                    help='training weights as cpcv_deriv_x --weight: absret = |settled return| of each fit row, mean 1')
    ap.add_argument('--model', default='lgb', choices=('lgb', 'ens3', 'logit', 'nystroem', 'forest', 'pool'),
                    help='lgb = repository mk_lgb (default); otherwise cpcv_deriv_x.py --model, same code')
    ap.add_argument('--nys-gamma-mult', type=float, default=1.0)
    ap.add_argument('--nys-components', type=int, default=500)
    ap.add_argument('--logit-c', type=float, default=0.05)
    ap.add_argument('--cat-depth', type=int, default=8)
    ap.add_argument('--session', default='ny')
    ap.add_argument('--cov', type=float, required=True)
    ap.add_argument('--holdout-start', default='2026-08-01')
    ap.add_argument('--holdout-end', default='2026-11-14')
    ap.add_argument('--n-jobs', type=int, default=16)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='')
    ap.add_argument('--n-candidates', type=int, required=True,
                    help='finalized candidates opened on the holdout for this key (Bonferroni: CI at 1 - 0.05/n)')
    a = ap.parse_args()
    if a.weight != 'none' and a.pool:
        raise SystemExit('--weight needs the own-pair fit rows; it is not defined with --pool')
    gap = a.horizon * 60
    t0 = time.time()
    hs = int(pd.Timestamp(a.holdout_start, tz='UTC').timestamp())
    he = int((pd.Timestamp(a.holdout_end, tz='UTC') + pd.Timedelta(days=1)).timestamp())
    listed = json.loads(Path(a.feature_list).read_text()) if a.feature_list else None
    X, fwd, ts, feats = cd.load(a.dataset, a.pair, a.horizon, a.features, 1, listed)
    sess = (x_session(ts, a.session) if a.session in LOCAL_WINDOWS else cd.session_mask(ts, a.session)) & cd.tradeable(ts, a.horizon)
    moved = np.isfinite(fwd) & (fwd != 0)
    hold = (ts >= hs) & (ts < he) & sess & np.isfinite(fwd)
    others = None
    if a.pool:
        others = {}
        for p in a.pool.split(','):
            Xp, fp, tp, fe = cd.load(a.dataset, p, a.horizon, a.features, 1, listed)
            keep = cd.session_mask(tp, a.session) & np.isfinite(fp) & (fp != 0)
            others[p] = (Xp[keep], (fp[keep] > 0).astype(int), tp[keep])
    res = {'holdout_eval': HOLDOUT_VERSION, 'tag': a.tag, 'key': f'{a.pair}.{a.horizon}m',
           'data': 'Deriv 1-second mid ticks only (/mnt/data/deriv), no Dukascopy',
           'dataset': a.dataset, 'features': feats, 'n_features': len(feats), 'feature_set': a.feature_list or a.features,
           'pool': a.pool.split(',') if a.pool else None, 'nseed': a.nseed, 'num_leaves': a.num_leaves,
           'model': {'model': a.model, 'nys_gamma_mult': a.nys_gamma_mult, 'nys_components': a.nys_components,
                     'logit_c': a.logit_c, 'cat_depth': a.cat_depth, 'weight': a.weight},
           'session': a.session, 'cov': a.cov, 'holdout': [a.holdout_start, a.holdout_end], 'breakeven': cd.BE,
           'n_candidates_opened_for_key': a.n_candidates,
           'falsifier': {'registered_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'when': 'before the holdout is scored',
                         'binding_decision': 'made by holdout/seq_decision.py per holdout/REGISTER.md (owner decision 6: '
                                             'two-look group-sequential test, one finalist per key, one-sided family alpha '
                                             '0.025): exact binomial boundaries on the cumulative PRIMARY wins with floors '
                                             'z >= 2.80 (look 1) and z >= 1.98 (look 2), plus PRIMARY n >= 50, '
                                             'drift_matched.passes (owner decision 1) and every incumbent book beaten on the '
                                             'same window; the earlier Holm and Bonferroni readings are superseded',
                         'bonferroni_context_iff': 'PRIMARY n >= 50 AND the Bonferroni (1 - 0.05/n_candidates) bootstrap '
                                                   'lower bound >= 0.541 (context only; superseded by Holm)'},
           'code': {'holdout_sha256': cd.sha256(__file__), 'base_harness_sha256': cd.sha256(cd.__file__),
                    'x_harness': {'file': sys.modules['cpcv_deriv_x'].__file__,
                                  'sha256': cd.sha256(Path(sys.modules['cpcv_deriv_x'].__file__))}}}
    Path(a.out).write_text(json.dumps(res, indent=1))

    def pool_rows(cut):
        if others is None:
            return None
        Xo = np.vstack([v[0][v[2] < cut] for v in others.values()])
        yo = np.concatenate([v[1][v[2] < cut] for v in others.values()])
        return Xo, yo

    readings = {}
    # PRIMARY: monthly walk-forward refit
    months = pd.date_range(pd.Timestamp(a.holdout_start, tz='UTC').normalize().replace(day=1), pd.Timestamp(a.holdout_end, tz='UTC'), freq='MS')
    p_all = np.full(len(ts), np.nan)
    thr_all = np.full(len(ts), np.nan)
    fit_info = []
    for mi, m0 in enumerate(months):
        m1 = m0 + pd.offsets.MonthBegin(1)
        lo, hi = max(int(m0.timestamp()), hs), min(int(m1.timestamp()), he)
        cut = lo - gap - 120
        train_idx = np.where((ts < cut) & moved & sess)[0]
        mask = hold & (ts >= lo) & (ts < hi)
        if not mask.any():
            continue
        ps, thr, nfit = fit_predict(X, fwd, train_idx, mask, a, np.random.default_rng(13 + mi), pool_rows(cut))
        p_all[mask] = ps
        thr_all[mask] = thr
        fit_info.append({'month': m0.strftime('%Y-%m'), 'train_rows': int(len(train_idx)), 'n_fit': nfit,
                         'thr': round(thr, 5), 'scored': int(mask.sum())})
        print(f'  refit {m0:%Y-%m}: train {len(train_idx):,} fit {nfit:,} thr {thr:.4f} scored {int(mask.sum()):,} '
              f'({time.time() - t0:.0f}s)', flush=True)
    # CONTEXT: frozen at DEV end
    cut = hs - gap - 120
    ps, thr0, nfit0 = fit_predict(X, fwd, np.where((ts < cut) & moved & sess)[0], hold, a, np.random.default_rng(13), pool_rows(cut))
    p_frozen = np.full(len(ts), np.nan)
    p_frozen[hold] = ps
    idx = np.where(hold)[0]
    f, t = fwd[idx], ts[idx]
    mv = f != 0
    mlabel = pd.to_datetime(t, unit='s', utc=True).strftime('%Y-%m').to_numpy()
    # drift-matched baselines: each side's always-side win rate (ties lose) over every holdout row, per America/New_York
    # hour x rv_60m tercile (BINDING under owner decision 1; terciles over the holdout rows, as analyze_drift_placebo does
    # within a path) and per hour only (context)
    nyh = pd.to_datetime(t, unit='s', utc=True).tz_convert('America/New_York').hour.to_numpy()
    hour_rate = {s: {int(h): float(((f[nyh == h] > 0) if s == 'UP' else (f[nyh == h] < 0)).mean()) for h in np.unique(nyh)}
                 for s in ('UP', 'DOWN')}
    rv = pd.read_parquet(Path(a.dataset) / f'{a.pair}.parquet', columns=['decision_ts', 'rv_60m']).set_index(
        'decision_ts')['rv_60m'].reindex(t).to_numpy(np.float64)
    q1, q2 = np.nanquantile(rv, [1 / 3, 2 / 3]) if np.isfinite(rv).any() else (np.nan, np.nan)
    vol_ok = bool(np.isfinite(q1) and np.isfinite(q2) and q1 < q2)   # else the binding test is unavailable (never passes)
    terc = np.where(np.isnan(rv), -1, np.digitize(rv, [q1, q2])) if vol_ok else np.full(len(rv), -1)
    hv_rate = {s: pd.Series(((f > 0) if s == 'UP' else (f < 0)).astype(float)).groupby([nyh, terc]).mean().to_dict()
               for s in ('UP', 'DOWN')}
    for name, p, thr in (('PRIMARY_monthly_refit', p_all[idx], thr_all[idx]), ('CONTEXT_frozen_dev_end', p_frozen[idx], np.full(len(idx), thr0))):
        r = {'auc_moved': round(float(cd.roc_auc_score((f[mv] > 0).astype(int), p[mv])), 4), 'sides': {}}
        conf = np.abs(p - 0.5)
        for side in ('UP', 'DOWN', 'COMBINED'):
            want = (p > 0.5) if side == 'UP' else (p < 0.5) if side == 'DOWN' else np.ones(len(p), bool)
            tr = cd.nonoverlap_chrono(t, want & (conf >= thr), gap)
            win = np.where(p[tr] > 0.5, f[tr] > 0, f[tr] < 0).astype(float)
            lo_, hi_ = boot(win)
            alo, ahi = boot(win, alpha=0.05 / a.n_candidates)
            bym = {m: [int((mlabel[tr] == m).sum()), round(float(win[mlabel[tr] == m].mean()), 4) if (mlabel[tr] == m).any() else None]
                   for m in sorted(set(mlabel))}
            r['sides'][side] = {'n': int(len(win)), 'win_rate': round(float(win.mean()), 4) if len(win) else None,
                                'ci95': [round(lo_, 4), round(hi_, 4)],
                                'ci_bonferroni': {'level': round(1 - 0.05 / a.n_candidates, 4), 'interval': [round(alo, 4), round(ahi, 4)]},
                                'by_month': bym}
            if side != 'COMBINED':
                r['sides'][side]['bonferroni_context_eligible'] = bool(name.startswith('PRIMARY') and len(win) >= 50
                                                                       and np.isfinite(alo) and alo >= cd.BE)
                r['sides'][side]['p_boot_le_breakeven'] = round(boot_p(win, cd.BE), 5)
                if len(win):
                    hbv = np.array([hv_rate[side][(int(h), int(k))] for h, k in zip(nyh[tr], terc[tr])])
                    elo, ehi = boot(win - hbv)
                    r['sides'][side]['drift_matched'] = {'match': 'NY hour x rv_60m tercile (binding, owner decision 1)',
                                                         'available': vol_ok, 'rv_60m_finite_share': round(float(np.isfinite(rv).mean()), 4),
                                                         'baseline_mean': round(float(hbv.mean()), 4),
                                                         'excess': round(float((win - hbv).mean()), 4),
                                                         'excess_ci95': [round(elo, 4), round(ehi, 4)],
                                                         'passes': bool(vol_ok and np.isfinite(elo) and elo > 0)}
                    hb = np.array([hour_rate[side][int(h)] for h in nyh[tr]])
                    hlo, hhi = boot(win - hb)
                    r['sides'][side]['drift_matched_hour'] = {'match': 'NY hour only (context)',
                                                              'baseline_mean': round(float(hb.mean()), 4),
                                                              'excess': round(float((win - hb).mean()), 4),
                                                              'excess_ci95': [round(hlo, 4), round(hhi, 4)]}
        readings[name] = r
        s = r['sides']
        print(f'{name}: AUC {r["auc_moved"]} | UP {s["UP"]["n"]} {s["UP"]["win_rate"]} {s["UP"]["ci95"]} | '
              f'DOWN {s["DOWN"]["n"]} {s["DOWN"]["win_rate"]} {s["DOWN"]["ci95"]} | COMB {s["COMBINED"]["n"]} {s["COMBINED"]["win_rate"]}', flush=True)
    base = {}
    for side in ('UP', 'DOWN'):
        tr = cd.nonoverlap_chrono(t, np.ones(len(t), bool), gap)
        base[side] = round(float(((f[tr] > 0) if side == 'UP' else (f[tr] < 0)).mean()), 4)
    res.update(readings=readings, monthly_fits=fit_info, frozen_fit={'n_fit': nfit0, 'thr': round(thr0, 5)},
               holdout_rows=int(hold.sum()), holdout_moved_up_rate=round(float((f[mv] > 0).mean()), 4),
               baseline_always=base, seconds=round(time.time() - t0, 1))
    Path(a.out).write_text(json.dumps(res, indent=1))
    print('done', res['seconds'], 's ->', a.out)


if __name__ == '__main__':
    main()
