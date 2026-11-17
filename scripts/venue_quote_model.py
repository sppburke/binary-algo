#!/usr/bin/env python3
"""Deriv 15-minute Rise/Fall quote model: the quoted payouts as a function of the price path (development data only).

    venue_quote_model.py build <quote_audit_dir> <out_dir> [--n-jobs N]

Fits two LightGBM regressors on every fresh matched Rise/Fall quote of the archive (weekdays, all hours): the skew
(normalised implied probability of Rise minus 0.5) and the overround, from features of the pair's own Deriv ticks at
or before the quote and the clock. No pair identity is used, so the model can be applied to a pair without a quote
lane; the build reports leave-one-pair-out and date-blocked out-of-fold quality for that use.
Writes <out_dir>/skew.txt, <out_dir>/overround.txt and <out_dir>/meta.json. `load()` and `payouts()` apply it.

What it is: the pricing rule Deriv used in the archive's dates (2026-07-02..2026-07-31), measured, then applied to
other dates as an ASSUMPTION that the rule was the same there (no older quotes exist on disk to check it). The archive
has summer dates only, so it cannot say whether the daily pattern (lower payouts in the last New York hour, no offer
17:00-20:00 New York) follows the New York clock or a fixed UTC clock in winter: `payouts()` takes clock='ny' (the
pattern follows New York time, the default) or clock='utc4' (the pattern stays at the UTC times of July, i.e. a fixed
UTC-4 clock); the two agree whenever New York is on daylight time. `offered` marks the weekday x 5-minute clock
buckets in which the archive holds quotes: a decision outside them could not have been priced in July.
Nothing dated 2026-08-01 or later is read.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import venue_quote_study as vqs   # noqa: E402
from deriv_dataset import day_files, load_ticks   # noqa: E402

TOOL = 'venue_quote_model/0.1'
FEATURES = (['tick_age_s'] + [f'ret_{w}s' for w in vqs.RET_WINDOWS]
            + [f'{k}_{w}s' for w in vqs.VOL_WINDOWS for k in ('rv', 'nchg', 'n')]
            + [f'z_{w}s' for w in vqs.RET_WINDOWS] + ['pos_900s', 'pos_3600s', 'ny_minute', 'weekday', 'minutes_to_17'])
PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=0, deterministic=True, force_row_wise=True)
BUCKET_MIN = 5            # clock bucket of the `offered` table, minutes
OFFER_MIN_ROWS = 30       # a weekday x bucket cell counts as offered with at least this many fresh matched quotes
CLOCKS = ('ny', 'utc4')


def clock_fields(t, clock='ny'):
    """Vectorised clock fields of epoch seconds t: (date string, minute of day, weekday) on the New York clock
    (clock='ny') or on a fixed UTC-4 clock (clock='utc4': New York's summer offset all year)."""
    assert clock in CLOCKS
    idx = pd.to_datetime(np.asarray(t, float), unit='s', utc=True)
    loc = idx.tz_convert('America/New_York') if clock == 'ny' else idx - pd.Timedelta(hours=4)
    mod = loc.hour.to_numpy() * 60 + loc.minute.to_numpy() + loc.second.to_numpy() / 60.0
    return np.asarray(loc.strftime('%Y-%m-%d')), mod, loc.weekday.to_numpy()


def feature_frame(ts, px, t, clock='ny'):
    f = vqs.price_features(ts, px, np.asarray(t, float))
    _, mod, wd = clock_fields(t, clock)
    f['ny_minute'], f['weekday'], f['minutes_to_17'] = mod, wd, 17 * 60 - mod
    return f[FEATURES], f['price_ok'].to_numpy(), (wd, mod)


def bucket(wd, mod):
    return (np.asarray(wd, int) * (1440 // BUCKET_MIN) + (np.asarray(mod) // BUCKET_MIN).astype(int))


def to_payouts(skew, over):
    inv_u = (0.5 + skew) * (1.0 + over); inv_d = (0.5 - skew) * (1.0 + over)
    return np.round(1.0 / np.clip(inv_u, 0.3, 0.99), 2), np.round(1.0 / np.clip(inv_d, 0.3, 0.99), 2)


def fit(X, y, n_jobs):
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_jobs=n_jobs, **PARAMS).fit(X, y)


def quality(df, skew_hat, over_hat):
    out = {'n': int(len(df)), 'r2_skew': vqs.r2(df['skew'].to_numpy(), skew_hat), 'r2_overround': vqs.r2(df['over'].to_numpy(), over_hat)}
    out.update(vqs.recon_stats(df, skew_hat, over_hat))
    pu, pd_ = to_payouts(skew_hat, over_hat)
    out['mean_payout_actual'] = {'rise': float(df['pu'].mean()), 'fall': float(df['pd'].mean())}
    out['mean_payout_reconstructed'] = {'rise': float(pu.mean()), 'fall': float(pd_.mean())}
    return out


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def build(quote_dir, out_dir, n_jobs):
    t0 = time.time()
    lanes, qmeta = vqs.load_quotes(quote_dir)
    frames = []
    for pair in sorted({p for p, _ in lanes}):
        q = vqs.match_lanes(lanes[(pair, 'UP')], lanes[(pair, 'DOWN')])
        files = day_files(pair, vqs.FIRST_TICK_DAY, vqs.LAST_TICK_DAY)
        assert all(f.name.split('_')[1] < '2026-08-01' for f in files), 'a tick file outside development would be read'
        ts, px = load_ticks(files)
        assert ts.max() < vqs.CUTOFF
        X, ok, (wd, mod) = feature_frame(ts, px, q['t_quote'].to_numpy())
        nyd, _, _ = clock_fields(q['t_quote'].to_numpy())
        a, b, c = vqs.local_fields(q['t_quote'].to_numpy()[:2000])               # self-check of the vectorised clock
        assert (a == nyd[:2000]).all() and np.allclose(b, mod[:2000]) and (c == wd[:2000]).all()
        d = pd.concat([q, X], axis=1); d['pair'] = pair; d['nydate'] = nyd; d['ok'] = ok & (wd < 5)
        d['q_up'] = (1.0 / d['pu']) / (1.0 / d['pu'] + 1.0 / d['pd']); d['skew'] = d['q_up'] - 0.5
        d['over'] = 1.0 / d['pu'] + 1.0 / d['pd'] - 1.0
        frames.append(d)
        print(f'{pair}: matched quotes {len(q)}, usable weekday rows {int(d["ok"].sum())} ({time.time() - t0:.0f}s)', flush=True)
    df = pd.concat(frames, ignore_index=True); df = df[df['ok']].reset_index(drop=True)
    cells = np.bincount(bucket(df['weekday'].to_numpy(), df['ny_minute'].to_numpy()), minlength=5 * 1440 // BUCKET_MIN)
    offered = cells >= OFFER_MIN_ROWS
    per_day = offered.reshape(5, -1)
    gaps = {}
    for w in range(5):                                # the clock ranges without an offer, per weekday (for the record)
        off = np.flatnonzero(~per_day[w]); runs = []
        if len(off):
            start = prev = off[0]
            for k in off[1:]:
                if k != prev + 1:
                    runs.append((start, prev)); start = k
                prev = k
            runs.append((start, prev))
        gaps[str(w)] = ['%02d:%02d-%02d:%02d' % (s * BUCKET_MIN // 60, s * BUCKET_MIN % 60, (e + 1) * BUCKET_MIN // 60, (e + 1) * BUCKET_MIN % 60)
                        for s, e in runs]
    meta = {'tool': TOOL, 'features': FEATURES, 'params': PARAMS, 'quotes': qmeta, 'rows': int(len(df)),
            'dates': sorted(df['nydate'].unique().tolist()), 'pairs': sorted(df['pair'].unique().tolist()),
            'bucket_min': BUCKET_MIN, 'offer_min_rows': OFFER_MIN_ROWS, 'offered': offered.astype(int).tolist(),
            'not_offered_new_york_clock_by_weekday': gaps,
            'assumptions': ['the pricing rule of 2026-07-02..2026-07-31 is applied to other dates unchanged (not checkable on disk)',
                            'clock=ny: the daily pattern follows the New York clock all year; clock=utc4: it stays at the UTC '
                            'times of July (the archive has summer dates only, so it cannot tell the two apart)',
                            'a pair without a quote lane is priced by the rule of the six quoted pairs (see leave_one_pair_out)']}
    # leave one pair out
    lopo = {}
    for pair in meta['pairs']:
        tr, te = (df['pair'] != pair).to_numpy(), (df['pair'] == pair).to_numpy()
        ms, mo = fit(df.loc[tr, FEATURES], df.loc[tr, 'skew'], n_jobs), fit(df.loc[tr, FEATURES], df.loc[tr, 'over'], n_jobs)
        lopo[pair] = quality(df[te].reset_index(drop=True), ms.predict(df.loc[te, FEATURES]), mo.predict(df.loc[te, FEATURES]))
        print('leave out', pair, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in lopo[pair].items()
                                             if k in ('n', 'r2_skew', 'r2_overround', 'favoured_side_accuracy')}),
              '| within 0.02: rise', round(lopo[pair]['rise']['share_within_0.02'], 3), 'fall', round(lopo[pair]['fall']['share_within_0.02'], 3),
              f'({time.time() - t0:.0f}s)', flush=True)
    meta['leave_one_pair_out'] = lopo
    # date-blocked out of fold
    dates = np.array(meta['dates']); fold_of = {d: k for k, block in enumerate(np.array_split(dates, 5)) for d in block}
    fold = df['nydate'].map(fold_of).to_numpy(); ps = np.full(len(df), np.nan); po = np.full(len(df), np.nan)
    for k in range(5):
        tr, te = fold != k, fold == k
        ps[te] = fit(df.loc[tr, FEATURES], df.loc[tr, 'skew'], n_jobs).predict(df.loc[te, FEATURES])
        po[te] = fit(df.loc[tr, FEATURES], df.loc[tr, 'over'], n_jobs).predict(df.loc[te, FEATURES])
    meta['date_blocked_out_of_fold'] = quality(df, ps, po)
    sess = ((df['ny_minute'] >= vqs.SESSION[0]) & (df['ny_minute'] < vqs.SESSION[1])).to_numpy()
    meta['date_blocked_out_of_fold_session'] = quality(df[sess].reset_index(drop=True), ps[sess], po[sess])
    hour = (df['ny_minute'].to_numpy() // 60).astype(int); meta['date_blocked_out_of_fold_by_new_york_hour'] = {}
    for h in sorted(set(hour.tolist())):
        m = hour == h
        if m.sum() >= 200:
            meta['date_blocked_out_of_fold_by_new_york_hour'][str(h)] = quality(df[m].reset_index(drop=True), ps[m], po[m])
    print('out of fold', json.dumps({k: v for k, v in meta['date_blocked_out_of_fold'].items() if k in ('n', 'r2_skew', 'r2_overround')}),
          '| session', json.dumps({k: v for k, v in meta['date_blocked_out_of_fold_session'].items() if k in ('n', 'r2_skew', 'r2_overround')}),
          f'({time.time() - t0:.0f}s)', flush=True)
    # final models on every row
    ms, mo = fit(df[FEATURES], df['skew'], n_jobs), fit(df[FEATURES], df['over'], n_jobs)
    os.makedirs(out_dir, exist_ok=True)
    ms.booster_.save_model(os.path.join(out_dir, 'skew.txt')); mo.booster_.save_model(os.path.join(out_dir, 'overround.txt'))
    meta['in_sample'] = quality(df, ms.predict(df[FEATURES]), mo.predict(df[FEATURES]))
    meta['sha256'] = {'skew.txt': sha(os.path.join(out_dir, 'skew.txt')), 'overround.txt': sha(os.path.join(out_dir, 'overround.txt')),
                      'venue_quote_model.py': sha(os.path.abspath(__file__)), 'venue_quote_study.py': sha(vqs.__file__)}
    import lightgbm
    meta['lightgbm'] = lightgbm.__version__
    with open(os.path.join(out_dir, 'meta.json'), 'w') as fh:           # written before the reload check: load() reads it
        json.dump(meta, fh, indent=1)
    again = load(out_dir); head = df[FEATURES].head(5000)
    meta['reload_max_abs_difference'] = float(max(np.abs(again['skew'].predict(head) - ms.predict(head)).max(),
                                                  np.abs(again['over'].predict(head) - mo.predict(head)).max()))
    assert meta['reload_max_abs_difference'] < 1e-9, 'the stored model does not reproduce the fitted one'
    meta['seconds'] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)
    print('not offered (New York clock, by weekday):', json.dumps(gaps))
    print('wrote', out_dir, f"({meta['seconds']}s)")


def load(model_dir):
    import lightgbm as lgb
    meta = json.load(open(os.path.join(model_dir, 'meta.json')))
    assert meta['features'] == FEATURES, 'feature list of the stored model differs from this code'
    assert meta['bucket_min'] == BUCKET_MIN
    for name in ('skew.txt', 'overround.txt'):
        assert sha(os.path.join(model_dir, name)) == meta['sha256'][name], f'{name} does not match its recorded checksum'
    return {'skew': lgb.Booster(model_file=os.path.join(model_dir, 'skew.txt')),
            'over': lgb.Booster(model_file=os.path.join(model_dir, 'overround.txt')),
            'offered': np.asarray(meta['offered'], bool), 'meta': meta}


def payouts(model, ts, px, t, clock='ny'):
    """Reconstructed (Rise payout, Fall payout, usable) for decisions at epoch seconds t on ticks (ts, px). usable =
    a tick exists at or before the decision and the decision's weekday x clock bucket had quotes in the archive."""
    X, ok, (wd, mod) = feature_frame(ts, px, t, clock)
    pu, pd_ = to_payouts(model['skew'].predict(X), model['over'].predict(X))
    wk = np.asarray(wd) < 5
    off = np.zeros(len(wk), bool); off[wk] = model['offered'][bucket(np.asarray(wd)[wk], np.asarray(mod)[wk])]
    return pu, pd_, ok & off


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build']); ap.add_argument('quote_dir'); ap.add_argument('out_dir')
    ap.add_argument('--n-jobs', type=int, default=8)
    a = ap.parse_args()
    build(a.quote_dir, a.out_dir, a.n_jobs)
