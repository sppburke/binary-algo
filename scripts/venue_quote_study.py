#!/usr/bin/env python3
"""Venue quote study on DEVELOPMENT data only (quotes and ticks dated before 2026-08-01).

    venue_quote_study.py <quote_audit_dir> <out_json> [--executor-dir DIR] [--boot N] [--n-jobs N]

Inputs (read-only): a quote-audit archive written by scripts/deriv_quote_workers.py --audit (one `quote_snapshot`
row per pair and side every 30 s: 15-minute Rise/Fall proposal at stake 1.00) and Deriv's own ticks under
/mnt/data/deriv/ticks. An archive directory (quotes, executor logs) is read only if every *.jsonl* file in it is a
regular file (not a link) named YYYY-MM-DD.jsonl or YYYY-MM-DD.jsonl.N for a calendar day up to 2026-07-31;
otherwise the run stops before a file is opened. Every non-empty file of the quote archive is then opened and
hashed. The name is all that is checked before a file is opened, and a rotated file can hold rows of later days than
its name, so rows are also dropped by their own time: a quote row dated 2026-08-01T00:00:00Z or later that carries a
payout is skipped and counted (`dated_after_development`), never matched or settled. The archive must therefore be a
development export; the tool cannot prove that from the names. No tick file dated after 2026-07-31 is opened, so the
settlement routine sees development ticks only; a quote whose contract would expire on or after
2026-08-01T00:00:00Z is flagged not settled and enters no outcome statistic.

Part A  calibration: is the payout skew between Rise and Fall a direction forecast, is it calibrated, and does any
        policy that uses only the quote pay? Settlement = the programme's rule (deriv_dataset.settle: entry = first
        tick at or after decision + 1 s, exit = last tick at or before entry target + 900 s, ties lose).
Part B  reconstruction: can the quoted payouts be rebuilt from the price path (ticks at or before the quote)?
Part C  (optional, no outcomes) the demo executor's selected side against the skew at the same moment.

A venue characterisation, not a strategy: it certifies nothing.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deriv_dataset import LAG_S, day_files, load_ticks, settle   # noqa: E402  (the programme's own settlement)

TOOL = 'venue_quote_study/0.2'     # 0.2: archive files checked by name; deterministic fits; z_* key names; wording
NY = ZoneInfo('America/New_York')
CUTOFF = datetime(2026, 8, 1, tzinfo=timezone.utc).timestamp()     # nothing at or after this instant is read
FIRST_TICK_DAY, LAST_TICK_DAY = date(2026, 7, 1), date(2026, 7, 31)
FRESH_MS = 5000.0
MATCH_TOL_S = 2.0
H_S, TOL_S = 900, 30
SESSION = (8 * 60, 16 * 60 + 35)        # weekday New York minutes: the programme's tradeable 15m window
RET_WINDOWS = (1, 2, 3, 5, 10, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400)
VOL_WINDOWS = (60, 300, 900, 3600, 14400)
PATH_HORIZONS = (1, 2, 5, 10, 30, 60, 300, 900)
PAST_WINDOWS = (1, 5, 30, 60, 300, 900, 3600)

REGISTERED = {
    'K1_skew_carries_direction': 'session quotes pooled over the pairs: AUC of the normalised implied probability of '
                                 'Rise for a strict rise among moved outcomes; SUPPORTED iff the date-cluster bootstrap '
                                 'CI95 lower bound is above 0.50, otherwise KILLED',
    'K2_no_quote_only_policy_pays': 'session quotes pooled: expected value per unit staked of always buying the '
                                    'favoured (lower-paid) side, the unfavoured side, Rise, Fall; SUPPORTED iff every '
                                    'CI95 upper bound is below 0; a policy whose CI95 lower bound is above 0 is FLAGGED '
                                    '(a lead to re-test, not a result)',
    'K3_calibration': 'session quotes pooled: realised win rate of the favoured side minus its normalised implied '
                      'probability; CALIBRATED iff the absolute difference is at most 0.01 and the CI95 covers 0',
    'B_reconstruction': 'RECONSTRUCTIBLE iff the out-of-fold R2 of the skew (five date-blocked folds, session quotes) '
                        'is at least 0.80 and the median absolute payout error is at most 0.02 on each side and both '
                        'half-splits (forward, backward) reach R2 0.70; NOT RECONSTRUCTIBLE iff the out-of-fold R2 is '
                        'below 0.50; PARTIAL otherwise',
    'prior_exposure': 'before registration the primary agent had seen outcome-free payout tables of this archive and '
                      'the owner\'s pre-declared July test T92 (side paying 1.92 or more won 681 of 1,498; the '
                      'opposite side 794 of 1,498)',
}


def dev_files(src):
    """The *.jsonl* files of an archive directory, refused as a whole unless every one of them is a regular file (not
    a link) whose whole name is YYYY-MM-DD.jsonl or YYYY-MM-DD.jsonl.N for a calendar day up to LAST_TICK_DAY: a file
    named for a later day, or for none, is never opened."""
    files = sorted(glob.glob(os.path.join(src, '*.jsonl*')))

    def development(f):
        m = re.fullmatch(r'(\d{4}-\d{2}-\d{2})\.jsonl(\.\d+)?', os.path.basename(f))
        try:
            return bool(m) and date.fromisoformat(m.group(1)) <= LAST_TICK_DAY and os.path.isfile(f) and not os.path.islink(f)
        except ValueError:                                # the ten characters are not a calendar day
            return False
    late = [os.path.basename(f) for f in files if not development(f)]
    if late:
        raise ValueError(f'venue_quote_study: {src} holds {len(late)} file(s) that are not regular files named '
                         f'YYYY-MM-DD.jsonl[.N] for a day up to {LAST_TICK_DAY} (first: {late[0]}): outside development, '
                         'nothing was opened')
    return files


def load_quotes(src):
    lanes, counts = {}, Counter()
    files = [f for f in dev_files(src) if os.path.getsize(f) > 0]
    h = hashlib.sha256()
    total = 0
    for f in files:
        total += os.path.getsize(f)
        with open(f, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                h.update(chunk)
        with open(f, errors='replace') as fh:
            for line in fh:
                if '"quote_snapshot"' not in line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    counts['bad_json'] += 1
                    continue
                counts['snapshots'] += 1
                if d.get('invalid_reason') is not None or d.get('payout') is None or d.get('quote_age_ms') is None:
                    counts['invalid'] += 1
                    continue
                t = datetime.fromisoformat(d['timestamp_utc']).timestamp()
                if t >= CUTOFF:
                    counts['dated_after_development'] += 1
                    continue
                age = float(d['quote_age_ms'])
                if age >= FRESH_MS:
                    counts['stale'] += 1
                    continue
                if abs(float(d['ask']) - 1.0) > 1e-9:
                    counts['ask_not_one'] += 1
                    continue
                counts['fresh_valid'] += 1
                lanes.setdefault((d['pair'], d['side']), []).append((t, float(d['payout']), age))
    meta = {'dir': src, 'files': len(files), 'bytes': total, 'sha256_of_concatenated_files': h.hexdigest(),
            'counts': dict(counts)}
    return lanes, meta


def match_lanes(up, dn):
    """One row per moment at which a fresh Rise and a fresh Fall quote of one pair were sampled within MATCH_TOL_S."""
    up = np.array(sorted(up)); dn = np.array(sorted(dn))
    tu, td = up[:, 0], dn[:, 0]
    j = np.clip(np.searchsorted(td, tu), 0, len(td) - 1); j0 = np.clip(j - 1, 0, len(td) - 1)
    pick = np.where(np.abs(td[j] - tu) <= np.abs(td[j0] - tu), j, j0)
    ok = np.abs(td[pick] - tu) <= MATCH_TOL_S
    _, first = np.unique(pick[ok], return_index=True)                     # each Fall row used once
    iu = np.flatnonzero(ok)[first]; idn = pick[ok][first]
    o = np.argsort(tu[iu]); iu, idn = iu[o], idn[o]
    t_dec = np.maximum(tu[iu], td[idn])                                   # both quotes are known by then
    t_quote = np.maximum(tu[iu] - up[iu, 2] / 1000.0, td[idn] - dn[idn, 2] / 1000.0)
    return pd.DataFrame({'t_dec': t_dec, 't_quote': t_quote, 'pu': up[iu, 1], 'pd': dn[idn, 1]})


def local_fields(t):
    loc = [datetime.fromtimestamp(x, NY) for x in t]
    return (np.array([l.strftime('%Y-%m-%d') for l in loc]), np.array([l.hour * 60 + l.minute + l.second / 60.0 for l in loc]),
            np.array([l.weekday() for l in loc]))


def path_returns(ts, px, dec_t):
    """Returns from the entry tick to the last tick at or before entry target + h (programme entry rule)."""
    n = len(ts); entry_t = dec_t + LAG_S
    ei = np.searchsorted(ts, entry_t, side='left'); eic = np.clip(ei, 0, n - 1)
    ve = (ei < n) & ((ts[eic] - entry_t) <= TOL_S)
    out = {}
    for h in PATH_HORIZONS:
        xt = entry_t + h
        xi = np.searchsorted(ts, xt, side='right') - 1; xic = np.clip(xi, 0, n - 1)
        v = ve & (xi >= ei) & ((xt - ts[xic]) <= TOL_S)
        out[h] = np.where(v, px[xic] / px[eic] - 1.0, np.nan)
    x60 = np.clip(np.searchsorted(ts, entry_t + 60, side='right') - 1, 0, n - 1)
    x900 = np.clip(np.searchsorted(ts, entry_t + 900, side='right') - 1, 0, n - 1)
    v = ve & (x900 > x60) & ((entry_t + 900 - ts[x900]) <= TOL_S) & ((entry_t + 60 - ts[x60]) <= TOL_S)
    out['60_to_900'] = np.where(v, px[x900] / px[x60] - 1.0, np.nan)
    return out


def price_features(ts, px, t):
    lp = np.log(px); n = len(ts)
    i0 = np.searchsorted(ts, t, side='right') - 1
    ok = i0 >= 1
    i0c = np.clip(i0, 0, n - 1)
    r1 = np.diff(lp, prepend=lp[0]); gap = np.diff(ts, prepend=ts[0]); r1[gap > 60] = 0.0     # no return across a gap
    csq = np.cumsum(r1 * r1); cchg = np.cumsum(r1 != 0)
    f = {'tick_age_s': t - ts[i0c]}
    for w in RET_WINDOWS:
        iw = np.clip(np.searchsorted(ts, t - w, side='right') - 1, 0, n - 1)
        f[f'ret_{w}s'] = (lp[i0c] - lp[iw]) * 1e4
    for w in VOL_WINDOWS:
        iw = np.clip(np.searchsorted(ts, t - w, side='right') - 1, 0, n - 1)
        f[f'rv_{w}s'] = np.sqrt(np.maximum(csq[i0c] - csq[iw], 0.0)) * 1e4
        f[f'nchg_{w}s'] = (cchg[i0c] - cchg[iw]).astype(float)
        f[f'n_{w}s'] = (i0c - iw).astype(float)
    for w in RET_WINDOWS:
        f[f'z_{w}s'] = f[f'ret_{w}s'] / (f['rv_900s'] * np.sqrt(w / 900.0) + 1e-9)
    s = pd.Series(lp, index=pd.to_datetime(ts, unit='s'))
    for w in (900, 3600):
        hi = s.rolling(f'{w}s').max().to_numpy(); lo = s.rolling(f'{w}s').min().to_numpy()
        f[f'pos_{w}s'] = (lp[i0c] - lo[i0c]) / (hi[i0c] - lo[i0c] + 1e-12)
    out = pd.DataFrame(f)
    out['price_ok'] = ok
    return out


def auc(score, y):
    y = np.asarray(y, bool); n1 = int(y.sum()); n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = pd.Series(score).rank().to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def cluster_ci(dates, num, den, boot, rng, shift=0.0):
    """Date-cluster bootstrap CI95 of sum(num)/sum(den) + shift."""
    u, inv = np.unique(dates, return_inverse=True)
    ns = np.bincount(inv, weights=num, minlength=len(u)); ds = np.bincount(inv, weights=den, minlength=len(u))
    idx = rng.integers(0, len(u), size=(boot, len(u)))
    vals = ns[idx].sum(1) / np.maximum(ds[idx].sum(1), 1e-12) + shift
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def policy(df, side_up, boot, rng, mask=None):
    """Buy Rise where side_up is True, Fall where False (rows in mask). Ties lose."""
    m = np.ones(len(df), bool) if mask is None else mask
    d = df[m]; su = np.asarray(side_up)[m]
    win = np.where(su, d['ret'].to_numpy() > 0, d['ret'].to_numpy() < 0)
    pay = np.where(su, d['pu'].to_numpy(), d['pd'].to_numpy())
    n = len(d)
    if n == 0:
        return {'n': 0}
    ones = np.ones(n)
    return {'n': int(n), 'dates': int(d['nydate'].nunique()), 'win_rate': float(win.mean()),
            'win_rate_ci95': cluster_ci(d['nydate'].to_numpy(), win.astype(float), ones, boot, rng),
            'mean_payout': float(pay.mean()), 'mean_breakeven': float((1.0 / pay).mean()),
            'ev_per_unit': float((win * pay).mean() - 1.0),
            'ev_ci95': cluster_ci(d['nydate'].to_numpy(), win * pay, ones, boot, rng, shift=-1.0)}


def part_a(df, boot, rng):
    res = {}
    ret = df['ret'].to_numpy(); moved = ret != 0
    q_up = df['q_up'].to_numpy(); up = ret > 0
    res['n'] = int(len(df)); res['dates'] = int(df['nydate'].nunique()); res['tie_share'] = float((~moved).mean())
    res['moved_up_rate'] = float(up[moved].mean())
    a = auc(q_up[moved], up[moved]); res['auc_q_up_moved'] = a
    u, inv = np.unique(df['nydate'].to_numpy()[moved], return_inverse=True)
    groups = [np.flatnonzero(inv == k) for k in range(len(u))]
    qm, um = q_up[moved], up[moved]; vals = []
    for _ in range(min(boot, 400)):
        pick = np.concatenate([groups[k] for k in rng.integers(0, len(u), len(u))])
        vals.append(auc(qm[pick], um[pick]))
    res['auc_ci95'] = [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]
    fav_up = df['pu'].to_numpy() < df['pd'].to_numpy(); skewed = df['pu'].to_numpy() != df['pd'].to_numpy()
    res['share_skewed'] = float(skewed.mean())
    res['policies'] = {'favoured': policy(df, fav_up, boot, rng, skewed), 'unfavoured': policy(df, ~fav_up, boot, rng, skewed),
                       'rise': policy(df, np.ones(len(df), bool), boot, rng), 'fall': policy(df, np.zeros(len(df), bool), boot, rng)}
    # calibration of the favoured side
    d = df[skewed]; fu = fav_up[skewed]
    q_f = np.where(fu, d['q_up'].to_numpy(), 1.0 - d['q_up'].to_numpy())
    win_f = np.where(fu, d['ret'].to_numpy() > 0, d['ret'].to_numpy() < 0)
    diff = win_f.astype(float) - q_f
    res['calibration'] = {'n': int(len(d)), 'mean_implied_favoured': float(q_f.mean()), 'win_rate_favoured': float(win_f.mean()),
                          'difference': float(diff.mean()),
                          'difference_ci95': cluster_ci(d['nydate'].to_numpy(), diff, np.ones(len(d)), boot, rng)}
    # by size of the payout gap
    gap = np.abs(df['pu'].to_numpy() - df['pd'].to_numpy()); bins = [(0.005, 0.03), (0.03, 0.08), (0.08, 0.13), (0.13, 0.18), (0.18, 9.0)]
    res['by_gap'] = {}
    for lo, hi in bins:
        m = (gap >= lo) & (gap < hi)
        if m.sum() < 50:
            continue
        pf = policy(df, fav_up, boot, rng, m); pu_ = policy(df, ~fav_up, boot, rng, m)
        qf = np.where(fav_up[m], q_up[m], 1.0 - q_up[m])
        res['by_gap'][f'{lo:.3f}-{hi:.2f}'] = {'n': int(m.sum()), 'mean_implied_favoured': float(qf.mean()), 'favoured': pf, 'unfavoured': pu_}
    # deciles of the implied probability of Rise
    edges = np.quantile(q_up, np.linspace(0, 1, 11)); res['by_q_up_decile'] = []
    for k in range(10):
        m = (q_up >= edges[k]) & ((q_up < edges[k + 1]) if k < 9 else (q_up <= edges[k + 1]))
        if m.sum() == 0:
            continue
        res['by_q_up_decile'].append({'n': int(m.sum()), 'mean_q_up': float(q_up[m].mean()), 'rise_rate': float((ret[m] > 0).mean()),
                                      'fall_rate': float((ret[m] < 0).mean()), 'mean_pu': float(df['pu'].to_numpy()[m].mean()),
                                      'mean_pd': float(df['pd'].to_numpy()[m].mean()),
                                      'ev_rise': float(((ret[m] > 0) * df['pu'].to_numpy()[m]).mean() - 1.0),
                                      'ev_fall': float(((ret[m] < 0) * df['pd'].to_numpy()[m]).mean() - 1.0)})
    # where along the 15 minutes does the favoured side win?
    res['favoured_win_by_horizon'] = {}
    for h in list(PATH_HORIZONS) + ['60_to_900']:
        r = df[f'path_{h}'].to_numpy()[skewed]; v = np.isfinite(r)
        w = np.where(fu[v], r[v] > 0, r[v] < 0); l = np.where(fu[v], r[v] < 0, r[v] > 0)
        res['favoured_win_by_horizon'][str(h)] = {'n': int(v.sum()), 'win': float(w.mean()), 'lose': float(l.mean()), 'tie': float((r[v] == 0).mean())}
    # does the favoured side follow or oppose the move just before the quote?
    res['favoured_vs_past_move'] = {}
    for w_ in PAST_WINDOWS:
        r = df[f'ret_{w_}s'].to_numpy()[skewed]; nz = r != 0
        res['favoured_vs_past_move'][f'{w_}s'] = {'n_moved': int(nz.sum()), 'share_same_direction': float((fu[nz] == (r[nz] > 0)).mean())}
    # de-overlapped count for reference (one bet per pair every 900 s)
    keep = 0
    for _, g in df.groupby('pair'):
        last = -1e18
        for t in g['t_dec'].to_numpy():
            if t >= last + H_S:
                keep += 1; last = t
    res['non_overlapping_decisions'] = int(keep)
    return res


def fit_predict(Xtr, ytr, Xte, n_jobs):
    import lightgbm as lgb
    # deterministic + force_row_wise since 0.2 (the settings of venue_quote_model): 0.1 ran without them, and a re-run
    # of its fits differed from the stored result in two fitted numbers at the 7th decimal (lab test x-regress-venue-02)
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=n_jobs, verbose=-1, random_state=0,
                          deterministic=True, force_row_wise=True)
    m.fit(Xtr, ytr)
    return m.predict(Xte), m


def r2(y, p):
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def recon_stats(df, skew_hat, over_hat):
    inv_u = (0.5 + skew_hat) * (1.0 + over_hat); inv_d = (0.5 - skew_hat) * (1.0 + over_hat)
    pu_hat = np.round(1.0 / np.clip(inv_u, 0.3, 0.99), 2); pd_hat = np.round(1.0 / np.clip(inv_d, 0.3, 0.99), 2)
    out = {}
    for name, hat, act in (('rise', pu_hat, df['pu'].to_numpy()), ('fall', pd_hat, df['pd'].to_numpy())):
        e = np.abs(hat - act)
        out[name] = {'median_abs_error': float(np.median(e)), 'p90_abs_error': float(np.percentile(e, 90)),
                     'share_within_0.01': float((e <= 0.0101).mean()), 'share_within_0.02': float((e <= 0.0201).mean()),
                     'share_within_0.05': float((e <= 0.0501).mean())}
    act_fav_up = df['pu'].to_numpy() < df['pd'].to_numpy(); sk = df['pu'].to_numpy() != df['pd'].to_numpy()
    out['favoured_side_accuracy'] = float(((skew_hat > 0) == act_fav_up)[sk].mean())
    return out


def part_b(df, feats, n_jobs):
    X = df[feats]; ys = df['skew'].to_numpy(); yo = df['over'].to_numpy()
    dates = np.array(sorted(df['nydate'].unique())); res = {'n': int(len(df)), 'dates': int(len(dates)), 'features': len(feats)}
    fold_of = {d: k for k, block in enumerate(np.array_split(dates, 5)) for d in block}
    fold = df['nydate'].map(fold_of).to_numpy()
    ps = np.full(len(df), np.nan); po = np.full(len(df), np.nan); gain = np.zeros(len(feats))
    for k in range(5):
        tr, te = fold != k, fold == k
        ps[te], m = fit_predict(X[tr], ys[tr], X[te], n_jobs); gain += m.booster_.feature_importance('gain')
        po[te], _ = fit_predict(X[tr], yo[tr], X[te], n_jobs)
    res['oof'] = {'r2_skew': r2(ys, ps), 'r2_overround': r2(yo, po), 'corr_skew': float(np.corrcoef(ys, ps)[0, 1]),
                  'mae_skew': float(np.abs(ys - ps).mean()), 'sd_skew': float(ys.std()), 'payout': recon_stats(df, ps, po)}
    res['oof_by_pair'] = {}
    for p, g in df.groupby('pair'):
        i = g.index.to_numpy()
        res['oof_by_pair'][p] = {'n': int(len(i)), 'r2_skew': r2(ys[i], ps[i]), 'r2_overround': r2(yo[i], po[i])}
    half = len(dates) // 2; first = df['nydate'].isin(dates[:half]).to_numpy()
    for name, tr, te in (('forward', first, ~first), ('backward', ~first, first)):
        a, _ = fit_predict(X[tr], ys[tr], X[te], n_jobs); b, _ = fit_predict(X[tr], yo[tr], X[te], n_jobs)
        d = df[te].reset_index(drop=True)
        res[name] = {'n_train': int(tr.sum()), 'n_test': int(te.sum()), 'r2_skew': r2(ys[te], a), 'r2_overround': r2(yo[te], b),
                     'payout': recon_stats(d, a, b)}
    # linear baseline on the same folds
    from sklearn.linear_model import Ridge
    Z = X.to_numpy(float); Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0); pl = np.full(len(df), np.nan)
    for k in range(5):
        tr, te = fold != k, fold == k
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
        A = np.clip((Z[tr] - mu) / sd, -5, 5); B = np.clip((Z[te] - mu) / sd, -5, 5)
        pl[te] = Ridge(alpha=10.0).fit(A, ys[tr]).predict(B)
    res['oof_linear_r2_skew'] = r2(ys, pl)
    order = np.argsort(-gain)[:15]
    res['top_gain_features_skew'] = [[feats[i], float(gain[i] / gain.sum())] for i in order]
    # volatility-scaled past returns (the z_* features); 0.1 stored the same values under ret_* names
    res['corr_skew_with_returns'] = {f'z_{w}s': float(np.corrcoef(ys, df[f'z_{w}s'].to_numpy())[0, 1]) for w in RET_WINDOWS}
    return res


def part_c(exec_dir, quotes):
    seen, rows = set(), []
    for f in dev_files(exec_dir):
        with open(f, errors='replace') as fh:
            for line in fh:
                if '"proposal_selected"' not in line:
                    continue
                d = json.loads(line)
                if d.get('event') != 'proposal_selected' or d.get('signal_id') in seen:
                    continue
                t = datetime.fromisoformat(d['timestamp_utc']).timestamp()
                if t >= CUTOFF:
                    continue
                seen.add(d.get('signal_id')); rows.append((d['pair'], d['side'], t, float(d['payout'])))
    out = {'signals': len(rows)}; own, opp, sel = [], [], []
    for pair, side, t, p in rows:
        q = quotes.get(pair)
        if q is None:
            continue
        tq = q['t_dec'].to_numpy(); j = np.clip(np.searchsorted(tq, t), 0, len(tq) - 1); j0 = max(j - 1, 0)
        k = j if abs(tq[j] - t) <= abs(tq[j0] - t) else j0
        if abs(tq[k] - t) > 30.0:
            continue
        pu, pd_ = q['pu'].to_numpy()[k], q['pd'].to_numpy()[k]
        own.append(pu if side == 'UP' else pd_); opp.append(pd_ if side == 'UP' else pu); sel.append(p)
    own, opp, sel = np.array(own), np.array(opp), np.array(sel)
    out.update({'matched_within_30s': int(len(own)), 'mean_selected_payout_executor': float(sel.mean()) if len(sel) else None,
                'mean_own_side_payout_audit': float(own.mean()) if len(own) else None,
                'mean_opposite_side_payout_audit': float(opp.mean()) if len(opp) else None,
                'share_selected_side_lower_paid': float((own < opp).mean()) if len(own) else None,
                'share_equal': float((own == opp).mean()) if len(own) else None,
                'share_selected_side_higher_paid': float((own > opp).mean()) if len(own) else None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('quote_dir'); ap.add_argument('out_json')
    ap.add_argument('--executor-dir'); ap.add_argument('--boot', type=int, default=1000); ap.add_argument('--n-jobs', type=int, default=8)
    a = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(20261117)
    lanes, qmeta = load_quotes(a.quote_dir)
    pairs = sorted({p for p, _ in lanes})
    print('quotes', json.dumps(qmeta['counts']), '| pairs', pairs, flush=True)
    frames, tick_meta, quotes_by_pair = [], {}, {}
    for pair in pairs:
        if (pair, 'UP') not in lanes or (pair, 'DOWN') not in lanes:
            continue
        q = match_lanes(lanes[(pair, 'UP')], lanes[(pair, 'DOWN')]); quotes_by_pair[pair] = q
        files = day_files(pair, FIRST_TICK_DAY, LAST_TICK_DAY)
        assert all(f.name.split('_')[1] < '2026-08-01' for f in files), 'a tick file outside development would be read'
        ts, px = load_ticks(files)
        assert ts.max() < CUTOFF
        tick_meta[pair] = {'files': len(files), 'first': files[0].name, 'last': files[-1].name, 'ticks': int(len(ts))}
        ret, valid = settle(q['t_dec'].to_numpy(), ts, px, H_S, TOL_S)
        settles_in_dev = (q['t_dec'].to_numpy() + LAG_S + H_S + TOL_S) < CUTOFF
        paths = path_returns(ts, px, q['t_dec'].to_numpy())
        f = price_features(ts, px, q['t_quote'].to_numpy())
        d = pd.concat([q, f], axis=1); d['pair'] = pair; d['ret'] = ret
        for h, r in paths.items():
            d[f'path_{h}'] = r
        nyd, mod, wd = local_fields(d['t_dec'].to_numpy())
        d['nydate'], d['ny_minute'], d['weekday'] = nyd, mod, wd
        d['minutes_to_17'] = 17 * 60 - mod
        d['q_up'] = (1.0 / d['pu']) / (1.0 / d['pu'] + 1.0 / d['pd']); d['skew'] = d['q_up'] - 0.5
        d['over'] = 1.0 / d['pu'] + 1.0 / d['pd'] - 1.0
        d['settled'] = valid & settles_in_dev
        frames.append(d)
        print(f'{pair}: matched quotes {len(q)}, settled {int(d["settled"].sum())}, ticks {len(ts)} ({time.time() - t0:.0f}s)', flush=True)
    df = pd.concat(frames, ignore_index=True)
    df['pair_code'] = df['pair'].astype('category').cat.codes
    wk = df['weekday'] < 5
    sess = wk & (df['ny_minute'] >= SESSION[0]) & (df['ny_minute'] < SESSION[1])
    out = {'tool': TOOL, 'registered': REGISTERED, 'quotes': qmeta, 'ticks': tick_meta, 'pairs': pairs,
           'settlement': {'rule': 'deriv_dataset.settle', 'lag_s': LAG_S, 'horizon_s': H_S, 'tol_s': TOL_S},
           'matched_quotes': int(len(df)), 'matched_session': int(sess.sum()), 'session': 'weekday New York 08:00-16:35'}
    # Part A
    A = {}
    for name, m in (('session', sess & df['settled']), ('all_weekday_hours', wk & df['settled'])):
        d = df[m].reset_index(drop=True); A[name] = part_a(d, a.boot, rng)
        A[name]['by_pair'] = {}
        for p, g in d.groupby('pair'):
            g = g.reset_index(drop=True); fu = g['pu'].to_numpy() < g['pd'].to_numpy(); sk = g['pu'].to_numpy() != g['pd'].to_numpy()
            mv = g['ret'].to_numpy() != 0
            A[name]['by_pair'][p] = {'n': int(len(g)), 'auc_q_up_moved': auc(g['q_up'].to_numpy()[mv], (g['ret'].to_numpy() > 0)[mv]),
                                     'favoured': policy(g, fu, a.boot, rng, sk), 'unfavoured': policy(g, ~fu, a.boot, rng, sk)}
        print(f'part A {name} done ({time.time() - t0:.0f}s)', flush=True)
    out['A'] = A
    s = A['session']; pol = s['policies']
    k1 = s['auc_ci95'][0] > 0.50
    k2 = all(pol[k]['ev_ci95'][1] < 0 for k in pol); flagged = [k for k in pol if pol[k]['ev_ci95'][0] > 0]
    c = s['calibration']; k3 = abs(c['difference']) <= 0.01 and c['difference_ci95'][0] <= 0 <= c['difference_ci95'][1]
    # Part B
    feats = [c_ for c_ in df.columns if c_.startswith(('ret_', 'rv_', 'nchg_', 'n_', 'z_', 'pos_'))] + ['tick_age_s', 'ny_minute', 'weekday', 'minutes_to_17', 'pair_code']
    B = {}
    for name, m in (('session', sess & df['price_ok']), ('all_weekday_hours', wk & df['price_ok'])):
        B[name] = part_b(df[m].reset_index(drop=True), feats, a.n_jobs)
        print(f'part B {name} done ({time.time() - t0:.0f}s)', flush=True)
    out['B'] = B
    b = B['session']; med = max(b['oof']['payout']['rise']['median_abs_error'], b['oof']['payout']['fall']['median_abs_error'])
    if b['oof']['r2_skew'] >= 0.80 and med <= 0.02 and min(b['forward']['r2_skew'], b['backward']['r2_skew']) >= 0.70:
        vb = 'RECONSTRUCTIBLE'
    elif b['oof']['r2_skew'] < 0.50:
        vb = 'NOT RECONSTRUCTIBLE'
    else:
        vb = 'PARTIAL'
    if a.executor_dir:
        out['C'] = part_c(a.executor_dir, quotes_by_pair)
    out['verdicts'] = {'K1_skew_carries_direction': 'SUPPORTED' if k1 else 'KILLED',
                       'K2_no_quote_only_policy_pays': 'SUPPORTED' if k2 else 'NOT SUPPORTED', 'K2_flagged_policies': flagged,
                       'K3_calibration': 'CALIBRATED' if k3 else 'NOT CALIBRATED', 'B_reconstruction': vb}
    out['seconds'] = round(time.time() - t0, 1)
    with open(a.out_json, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('verdicts', json.dumps(out['verdicts']))
    print('wrote', a.out_json, f'({out["seconds"]}s)')


if __name__ == '__main__':
    main()
