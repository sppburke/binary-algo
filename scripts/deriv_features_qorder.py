#!/usr/bin/env python3
"""Discovery D3-05: signed common-shock fingerprints of cross-pair quote-update ORDER, plus a permuted-order control.

    deriv_features_qorder.py <out dir> <first day> <last day>

Per UTC day, on the 1-second grid of the Deriv mid ticks (a quote is fresh for 2 s; nothing is filled beyond that):
  z_j = a_j * log(mid_j), a_j = +1 for XXX/USD and -1 for USD/XXX (USD-weakness orientation);
  x_j(s) = (z_j(s) - z_j(s-5)) / sd_j, sd_j = sd of the pair's 5-second changes over the 3600 s ending at s-5 (>= 600);
  for a TARGET pair i: B_i(s) = median of x_j over the six peers j != i.
Seeds: first crossings of |B_i| above theta_i(D) = 95th percentile of |B_i| over the NY-session (08:00-17:00 America/
New_York) seconds of the 20 PRECEDING trading days (causal; no seed until 20 days exist), 60 s refractory, S = sign(B).
A seed is a COMPLETE burst iff every pair is fresh at s, changed at least once in [s-30, s] and changes again in
(s, s+30]; tau_j = first changed second in (s, s+30]. Orders O_jk = sign(tau_k - tau_j) for the 21 pairs j < k.
Feature at decision t (bar start + 60 s), over complete bursts with s in [t-900, t-90] (every order window closed >= 60 s
before t), w = exp(-(t - s)/300):
  qo_<j>_<k> = a_i * sum(w S O_jk) / sum(w),   qo_n = number of such bursts (0 -> every qo_ = 0);
  qp_<j>_<k>, qp_n = the same with tau PERMUTED across the seven pairs inside each burst (fixed seed per target):
  shock marks, counts and the multiset of update delays kept, the pair-to-rank assignment broken (the control).
NaN before 20 days of history and where the day's windows are incomplete (first 4510 s of a UTC day).
Columns: bar_start_ts, qo_* (22), qp_* (22); rows = every bar of the target pair.
"""
import json
import os
import sys
import time
import warnings
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deriv_dataset as dd  # noqa: E402

QO_VERSION = 'deriv_features_qorder/0.1'
warnings.filterwarnings('ignore', message='All-NaN slice')
PAIRS = list(dd.PAIRS)
A = {p: (-1.0 if p.startswith('USD') else 1.0) for p in PAIRS}
JK = list(combinations(range(len(PAIRS)), 2))
NAMES = [f'{PAIRS[j]}_{PAIRS[k]}' for j, k in JK]
HIST_DAYS, WIN_LO, WIN_HI, TAU_W, REFRACT, HALF = 20, 900, 90, 30, 60, 300.0


def day_arrays(d):
    """1-second arrays for UTC day d: z, fresh, changed per pair (None if any pair has no file)."""
    t0 = int(pd.Timestamp(d.isoformat(), tz='UTC').timestamp())
    n = 86400
    Z, FR, CH = {}, {}, {}
    for p in PAIRS:
        files = dd.day_files(p, d, d)
        if not files:
            return None
        ts, px = dd.load_ticks(files)
        k = ts - t0
        ok = (k >= 0) & (k < n)
        k, px = k[ok], px[ok]
        _, last = np.unique(k[::-1], return_index=True)               # last tick of each second
        sec = np.unique(k)
        val = px[::-1][last]
        obs = np.full(n, np.nan)
        obs[sec] = val
        s = pd.Series(obs)
        lastobs = pd.Series(np.where(np.isfinite(obs), np.arange(n), np.nan)).ffill().to_numpy()
        FR[p] = (np.arange(n) - lastobs) <= 2
        filled = s.ffill().to_numpy()
        Z[p] = A[p] * np.log(filled)
        prev = np.r_[np.nan, filled[:-1]]
        CH[p] = np.isfinite(obs) & np.isfinite(prev) & (obs != prev)
    return t0, Z, FR, CH


def main():
    out = Path(sys.argv[1]); first = date.fromisoformat(sys.argv[2]); last = date.fromisoformat(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    days = [first + timedelta(i) for i in range((last - first).days + 1)]
    hist_absB = {p: [] for p in PAIRS}        # per target: NY-session |B| values of the last HIST_DAYS trading days
    feats = {p: [] for p in PAIRS}            # per target: list of DataFrames (decision_t, qo_*, qp_*)
    rngs = {p: np.random.default_rng(3050 + i) for i, p in enumerate(PAIRS)}
    burst_stats = {p: {'days_with_seeds': 0, 'seeds': 0, 'complete': 0, 'ties': 0, 'orders': 0} for p in PAIRS}
    secs = np.arange(86400)
    for d in days:
        arr = day_arrays(d)
        if arr is None:
            continue
        t0, Z, FR, CH = arr
        hour = pd.to_datetime(t0 + secs, unit='s', utc=True).tz_convert('America/New_York').hour.to_numpy()
        nymask = (hour >= 8) & (hour < 17)
        xs = []
        for p in PAIRS:
            dz = pd.Series(Z[p]) - pd.Series(Z[p]).shift(5)
            sd = dz.shift(5).rolling(3600, min_periods=600).std()
            xs.append((dz / sd).to_numpy())
        xs = np.vstack(xs)
        chg_any30 = {p: pd.Series(CH[p].astype(float)).rolling(31, min_periods=1).sum().to_numpy() > 0 for p in PAIRS}
        nxt = {}
        for p in PAIRS:
            cs = np.flatnonzero(CH[p])
            idx = np.searchsorted(cs, secs, side='right')
            nx = np.full(86400, -1)
            okk = idx < len(cs)
            nx[okk] = cs[idx[okk]]
            nxt[p] = nx
        fresh_all = np.all(np.vstack([FR[p] for p in PAIRS]), axis=0)
        recent_all = np.all(np.vstack([chg_any30[p] for p in PAIRS]), axis=0)
        for i, tgt in enumerate(PAIRS):
            peers = [j for j in range(len(PAIRS)) if j != i]
            with np.errstate(all='ignore'):
                B = np.nanmedian(xs[peers], axis=0)
            prev = hist_absB[tgt][-HIST_DAYS:]
            today = np.abs(B[nymask & np.isfinite(B)])
            if today.size >= 3600:                                   # a trading day (>= 1 h of NY-session values)
                hist_absB[tgt] = prev + [today]
            if len(prev) < HIST_DAYS:
                continue
            theta = float(np.quantile(np.concatenate(prev), 0.95))
            a = np.abs(B)
            cross = np.r_[False, (a[1:] > theta) & (a[:-1] <= theta)]
            seeds, lastseed = [], -10**9
            for s in np.flatnonzero(cross):
                if s - lastseed >= REFRACT:
                    seeds.append(s)
                    lastseed = s
            st = burst_stats[tgt]
            st['days_with_seeds'] += 1
            st['seeds'] += len(seeds)
            es, eS, eO, eP = [], [], [], []
            for s in seeds:
                if s + TAU_W >= 86400 or not (fresh_all[s] and recent_all[s]):
                    continue
                tau = np.array([nxt[p][s] - s for p in PAIRS])
                if np.any(tau < 1) or np.any(tau > TAU_W):
                    continue
                o = np.sign(tau[[k for _, k in JK]] - tau[[j for j, _ in JK]])
                tp = rngs[tgt].permutation(tau)
                op = np.sign(tp[[k for _, k in JK]] - tp[[j for j, _ in JK]])
                es.append(s); eS.append(np.sign(B[s])); eO.append(o); eP.append(op)
                st['ties'] += int((o == 0).sum()); st['orders'] += len(o)
            st['complete'] += len(es)
            dec = np.arange(0, 86400, 60) + 60                       # decision seconds (bar start + 60) of this day
            bar_start = t0 + dec - 60
            qo = np.full((len(dec), len(JK) + 1), np.nan)
            qp = np.full((len(dec), len(JK) + 1), np.nan)
            valid = dec >= 3600 + 5 + WIN_LO + 5                     # sd window + feature window complete inside the day
            if es:
                es_ = np.array(es); SO = np.array(eS)[:, None] * np.array(eO); SP = np.array(eS)[:, None] * np.array(eP)
                lag = dec[:, None] - es_[None, :]
                W = np.where((lag >= WIN_HI) & (lag <= WIN_LO), np.exp(-lag / HALF), 0.0)
                cnt = ((lag >= WIN_HI) & (lag <= WIN_LO)).sum(axis=1)
                den = W.sum(axis=1)
                with np.errstate(invalid='ignore', divide='ignore'):
                    qo[:, :-1] = np.where(den[:, None] > 0, A[tgt] * (W @ SO) / den[:, None], 0.0)
                    qp[:, :-1] = np.where(den[:, None] > 0, A[tgt] * (W @ SP) / den[:, None], 0.0)
                qo[:, -1] = cnt; qp[:, -1] = cnt
            else:
                qo[:] = 0.0; qp[:] = 0.0
            qo[~valid] = np.nan; qp[~valid] = np.nan
            df = pd.DataFrame(np.hstack([qo, qp]).astype(np.float32),
                              columns=[f'qo_{n}' for n in NAMES] + ['qo_n'] + [f'qp_{n}' for n in NAMES] + ['qp_n'])
            df.insert(0, 'bar_start_ts', bar_start.astype(np.int64))
            feats[tgt].append(df)
        print(d, f'{time.time() - t_start:.0f}s', flush=True)
    man = {'builder': QO_VERSION, 'first_day': first.isoformat(), 'last_day': last.isoformat(), 'pairs': {},
           'definition': __doc__}
    for p in PAIRS:
        ts_, px_ = dd.load_ticks(dd.day_files(p, first, last))
        starts = dd.minute_bars(ts_, px_).index.values.astype(np.int64)
        allf = pd.concat(feats[p], ignore_index=True) if feats[p] else pd.DataFrame(columns=['bar_start_ts'])
        allf = allf.drop_duplicates('bar_start_ts').set_index('bar_start_ts').reindex(starts)
        allf.index.name = 'bar_start_ts'
        allf.reset_index().to_parquet(out / f'{p}.parquet', index=False)
        st = burst_stats[p]
        man['pairs'][p] = {'rows': int(len(allf)), 'finite_share_qo_n': round(float(np.isfinite(allf['qo_n']).mean()), 4)
                           if 'qo_n' in allf else 0.0, **st,
                           'tie_share': round(st['ties'] / st['orders'], 4) if st['orders'] else None}
        print(p, man['pairs'][p], flush=True)
    man['skipped_incomplete_days'] = [list(x) for x in dd.SKIPPED_INCOMPLETE]
    man['allow_incomplete'] = os.environ.get('DERIV_ALLOW_INCOMPLETE') == '1'
    man['seconds'] = round(time.time() - t_start, 1)
    (out / 'manifest.json').write_text(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
