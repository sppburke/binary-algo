#!/usr/bin/env python3
"""Deriv-only per-fold-refit CPCV certification harness, plus a forward-only walk diagnostic.

    cpcv_deriv.py --pair AUDUSD --horizon 15 [--session ny] [--covs 0.03,0.02,0.05] [--nseed 1]
                  [--stride 1] [--features all|noclock|own|own_noclock] [--feature-list cols.json]
                  [--num-leaves 255] [--dataset DIR] [--out result.json] [--tag TEXT] [--no-forward]
                  [--include-blackout]

Mirror of the repository's 15m session certification (scripts/audusd_15m_cpcv_session.py), applied to a
Deriv-only dataset (tools/deriv_dataset.py or tools/deriv_features_b239.py), with these declared differences:
  - labels are Deriv-settled from ticks (entry = first tick at/after decision + 1 s, exit = last tick at/before
    entry target + H, ties valid and LOSE) instead of bar close to bar close;
  - decisions inside Deriv's FX purchase blackout are untradeable and excluded from every role (BLACKOUT_RULE)
    unless --include-blackout;
  - the session gate is applied to the decision time (bar close); default stride 1 (the reference uses 2, per UTC
    year; --stride k > 1 reproduces that per-year sampling);
  - validity (strategy-eval section 2): the moved up-rate of all in-session decisions must lie in [0.47, 0.53] and
    every path's in [0.45, 0.55] (the reference's tripwire); an invalid run certifies nothing, in every field.
CPCV protocol otherwise as the reference: 6 equal-count groups of in-session decisions, all C(6,2) = 15 test paths,
purge and embargo of H around each test block, training on moved in-session unpurged rows, a random 15% validation
subset (early stopping 80 rounds, threshold source), fit pool subsampled to <= 150,000 rows in rng order (rng 13),
LightGBM with the repository's mk_lgb parameters averaged over NSEED seeds, threshold = quantile(|p_val - 0.5|,
1 - coverage), side filter then first-come de-overlap (gap H), win = right sign AND moved, a path counts with >= 25
trades, a side is CERTIFIED iff p10 >= 0.541 and >= 80% of counted paths >= 0.541 (computed on unrounded path win
rates). The falsifier is written to the result file before any path is fitted.
Diagnostics (not part of the rule): per-path AUC on moved rows; always-UP / always-DOWN baselines on their own
de-overlapped schedule; a FORWARD walk (train only on groups before group g, test group g, g = 2..5, rng 17).
"""
import argparse
import copy
import hashlib
import json
import re
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
from fork_jobs import in_order, run_forked  # noqa: E402  (next to this file, in tools/ and in the repository mirror)

warnings.filterwarnings('ignore', message='X does not have valid feature names')
# 0.6: preserve noncertifying reasons and veto payout survival decisions.
HARNESS_VERSION = 'cpcv_deriv/0.6'   # 0.5: --path-jobs N (test E-020: paths and forward splits in forked workers,
#   tools/fork_jobs.py; identical path rows; with --nseed > 1 the summed feature gain can differ in its last bits)
#   and a recorded 'compute' block; defaults unchanged.
#   0.4.2: + the venue's window 21:00-23:59:59 UTC (owner decision 14) and the
#                                       guards of E-003; 0.4.1: + verdict_goal field; 0.4: review fixes
LAB = Path('/mnt/data/deriv-sota-lab')
BE = 0.541
N_GROUPS, K_TEST = 6, 2
SUB_FIT = 150_000
MIN_TRADES = 25
VAL_FRAC = 0.15
SIDES = ('UP', 'DOWN', 'COMBINED')
# Feature families across the supported schemas (v0.1 names; B239 + cross-pair blocks; b_/z_ prefixed union columns)
CLOCK_BASE = {'hour_sin', 'hour_cos', 'dow', 'hour', 'sess_ny', 'sess_london', 'sess_overlap', 'sess_ln', 'sess_as'}
XPAIR_RE = re.compile(r'^(x_.*|eu_r\d+|gbp_r\d+|tgt_r\d+|usdbask\d+|catchup\d+|eurresid\d+|gbpresid\d+|tgtresid\d+|disp\d+|'
                      r'agree\d+|ll_[A-Z]{6}\d+|eurgbp_r\d+|eurgbp_dev\d+|eurobloc\d+|gbpbloc_resid\d+|risk\d+|eurbloc\d+|'
                      r'eurcatch\d+|chf_eurresid\d+)$')
PREFIXES = ('b_', 'z_')
BLACKOUT_RULE = ('Deriv rejects an FX Rise/Fall purchase whose expiry lands in 16:50-18:00 New York (daily close/rollover; '
                 '15m: last accepted start 16:34:40 NY, first rejected 16:35:10 NY - docs/DERIV_DEMO_EXECUTOR.md (g)); '
                 'decisions with start in [16:50 - H, 18:00) NY are untradeable and excluded from training, validation and '
                 'test (starts inside the rollover itself, 17:45-18:00 for 15m, are excluded conservatively)')
VENUE_WINDOW_RULE = ('Deriv also refuses every FX contract under five hours bought from 21:00:00 to 23:59:59 UTC '
                     '("Trading on forex contracts with duration less than 5 hours is not available from 21:00:00 to '
                     '23:59:59", seen on 2026-11-17 with New York on standard time; owner decision 14 of 2026-11-18): '
                     'decisions with start in that window are untradeable as well (with New York on daylight time the '
                     'window lies after the 17:00 New York end of the session)')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def venue_open(ts, horizon_min):
    """VENUE_WINDOW_RULE alone: False for a decision under five hours starting 21:00:00-23:59:59 UTC."""
    if horizon_min >= 300:
        return np.ones(len(np.asarray(ts)), bool)
    return pd.to_datetime(np.asarray(ts), unit='s', utc=True).hour.values < 21


def tradeable(ts, horizon_min, venue_window=True):
    """Decisions Deriv would sell: outside BLACKOUT_RULE and, unless venue_window is False (the July 2026 filter, for
    regression against stored results only), outside VENUE_WINDOW_RULE for contracts under five hours."""
    utc = pd.to_datetime(np.asarray(ts), unit='s', utc=True)
    t = utc.tz_convert('America/New_York')
    m = t.hour.values * 60 + t.minute.values
    ok = ~((m >= 16 * 60 + 50 - horizon_min) & (m < 18 * 60))
    if venue_window:
        ok &= venue_open(ts, horizon_min)
    return ok


def base_name(c):
    for p in PREFIXES:
        if c.startswith(p):
            return c[len(p):]
    return c


def feature_list(columns, which):
    feats = [c for c in columns if c not in ('bar_start_ts', 'decision_ts') and not c.startswith(('ret_fwd_', 'valid_'))]
    if which in ('noclock', 'own_noclock'):
        feats = [c for c in feats if base_name(c) not in CLOCK_BASE]
    if which in ('own', 'own_noclock'):
        feats = [c for c in feats if not XPAIR_RE.match(base_name(c))]
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
    idx = idx[np.argsort(ts[idx], kind='stable')]
    if stride > 1:                              # the reference samples every stride-th eligible row within each UTC year
        yrs = pd.to_datetime(ts[idx], unit='s', utc=True).year.values
        idx = np.concatenate([idx[yrs == y][::stride] for y in np.unique(yrs)])
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


def side_wr(pr, fwd, ts, thr, side, gap, keep=None):
    """First-come trades on one side at threshold thr (ties lose). keep (test E-003): False on a row a gate
    excluded, which then trades on no side, COMBINED included, whatever the threshold; None: no gate, every row is
    eligible as before. A row without a finite probability never trades."""
    conf = np.abs(pr - 0.5)
    ok = np.isfinite(pr) if keep is None else np.isfinite(pr) & keep
    want = (ok & (pr > 0.5)) if side == 'UP' else (ok & (pr < 0.5)) if side == 'DOWN' else ok
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
    """Path summary on UNROUNDED win rates (the reference aggregates raw values); a missing rate (NaN, or null in a
    reread file) is skipped."""
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], float)
    if len(a) == 0:
        return {'n_paths': 0}
    return {'n_paths': int(len(a)), 'mean': float(a.mean()), 'p10': float(np.percentile(a, 10)),
            'p50': float(np.percentile(a, 50)), 'min': float(a.min()), 'frac_clear_BE': float((a >= BE).mean())}


class Unsupported(Exception):
    """A split whose local support is missing (test E-003): an empty partition or model subset, an empty out-of-fold
    half, no blend selection, no gated fit row, labels the selected model class cannot fit. The harness gives the
    split the explicit zero-trade row with this message as the reason; anything fitted before is discarded."""


def num_or_nan(x):
    """A number from a result row; a missing one (None: how a reread result file shows NaN) becomes NaN."""
    return float('nan') if x is None else x


def unsupported(vconf, vkeep=None):
    """Why a path's validation population cannot set a threshold, or None (the support rule of test E-003): the
    population (the validation rows through the gates that define the coverage) is empty; a confidence in it is not
    a finite number; or no row in it was kept by every gate (vkeep False: a gate excluded the row but left it in the
    population at confidence 0, the overall-coverage gates)."""
    v = np.asarray(vconf, float)
    if v.size == 0:
        return 'insufficient validation support: no validation row passes the gates'
    if not np.all(np.isfinite(v)):
        return 'insufficient validation support: a validation confidence is not a finite number'
    if vkeep is not None and not np.any(vkeep):
        return 'insufficient validation support: a gate excluded every validation row of the population'
    return None


def no_trades(row, covs, why):
    """A path without validation support: no threshold and zero strategy trades on every side at every coverage (no
    threshold is taken from another population), the reason in insufficient_support."""
    row['insufficient_support'] = why
    for c in covs:
        row['bycov'][f'{c}'] = {'thr': None, **{s: [0, float('nan')] for s in SIDES}}
    return row


def unsupported_row(label, n_fit, n_val, ftst, ttst, covs, gap,
                    why='insufficient support: an empty fit or validation partition'):
    """A split that cannot be fitted or scored (test E-003): no strategy trade (anything fitted before the failure
    is discarded); the always-side baselines are kept."""
    mv = ftst != 0
    row = {'test': label, 'n_fit': int(n_fit), 'n_val': int(n_val), 'n_test': int(len(ftst)), 'best_iter': [],
           'auc': float('nan'), 'up_rate': float((ftst[mv] > 0).mean()) if mv.sum() else float('nan'),
           'always': {s: list(always(ftst, ttst, s, gap)) for s in ('UP', 'DOWN')}, 'bycov': {}}
    return no_trades(row, covs, why)


def parse_covs(text):
    """--covs as numbers in their given order (the first is the operating coverage): each finite and inside (0, 1],
    none given twice; otherwise the run stops before anything is loaded, fitted or written."""
    try:
        covs = [float(c) for c in str(text).split(',')]
    except ValueError:
        raise SystemExit(f'--covs {text!r}: not a comma-separated list of numbers') from None
    if not all(np.isfinite(c) and 0 < c <= 1 for c in covs):
        raise SystemExit(f'--covs {text!r}: every coverage must be a number in (0, 1]')
    if len({f'{c}' for c in covs}) != len(covs):
        raise SystemExit(f'--covs {text!r}: a coverage is given twice')
    return covs


def json_safe(x):
    """A copy of a result for standard JSON: a number that is not finite (NaN, an infinity) becomes null (test
    E-003: no NaN or Infinity token is written); this module's readers (summ, aggregate, print_summary, fmt_side)
    take null as missing."""
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def result_json(res):
    """A result file's text: standard JSON; a number that is missing is null. Test E-003: a continuing policy's
    result is written as a forward diagnostic (veto_adaptive on the copy: its certificates false, its verdict
    replaced, its reason recorded), since json_safe writes a non-finite ACI gamma as null; a reread file then stays
    noncertifying whether or not the result was aggregated, or certified by an earlier build, before it was
    written."""
    out = json_safe(res)
    why = adaptive_policy(res)
    if why:
        veto_adaptive(out, why)
    return json.dumps(out, indent=1, allow_nan=False)


def score_row(row, ptst, ftst, ttst, vconf, covs, gap, vkeep=None, tkeep=None):
    """Fill a path row (always baselines and per-coverage side results) with UNROUNDED values. vconf: the
    threshold population's confidences; vkeep / tkeep (test E-003): the rows of that population and of the test
    rows that every gate kept (None: no gate). A path without validation support (unsupported) has no threshold
    and no strategy trade."""
    for s in ('UP', 'DOWN'):
        row['always'][s] = list(always(ftst, ttst, s, gap))
    why = unsupported(vconf, vkeep)
    if why:
        return no_trades(row, covs, why)
    for c in covs:
        thr = float(np.quantile(vconf, 1 - c))
        row['bycov'][f'{c}'] = {'thr': thr}
        for s in SIDES:
            row['bycov'][f'{c}'][s] = list(side_wr(ptst, ftst, ttst, thr, s, gap, tkeep))
    return row


ONLINE_POLICY_REASON = 'a continuing policy: the online learner updates on settled outcomes during each test path'
ACI_POLICY_REASON = 'an adaptive threshold: ACI moves it with settled test outcomes'

def adaptive_policy(res):
    """Why a result is a continuing (adaptive) policy, or None (test E-003; owner decision 13: such policies count
    only as forward diagnostics): the online harness learns from settled outcomes as it goes; an ACI threshold
    moves with settled test outcomes (stored results of earlier builds record it in their configuration). A result
    marked noncertifying (by aggregate, or by result_json in the file it writes) stays so: result_json writes a
    non-finite ACI gamma as null, which a reread configuration would read as ACI off."""
    if res.get('noncertifying'):                    # the reason aggregate recorded (a string: it survives JSON)
        return str(res['noncertifying'])
    if str(res.get('harness', '')).startswith('cpcv_deriv_online'):
        return ONLINE_POLICY_REASON
    for key in ('knobs', 'config'):
        conf = res.get(key)
        if isinstance(conf, dict) and conf.get('aci_gamma') is not None:
            return ACI_POLICY_REASON
    return None


def veto_adaptive(res, why):
    """Test E-003, owner decision 13: a continuing policy is a forward diagnostic only. Every certificate the result
    holds is set false (each coverage's per-side verdicts and the strict verdict_goal), a verdict it holds is
    replaced, and the reason is recorded. aggregate applies it to the result, result_json to the copy it writes,
    print_summary to a copy it prints. Other recorded noncertifying reasons keep their own verdict text;
    payout survival decisions are vetoed too."""
    for v in (res.get('bycov') or {}).values():
        for s in SIDES:
            cert = (v.get('verdict') or {}).get(s) if isinstance(v, dict) else None
            if isinstance(cert, dict) and 'CERTIFIED' in cert:
                cert['CERTIFIED'] = False
    for g in (res.get('verdict_goal') or {}).values():
        if isinstance(g, dict) and 'CERTIFIED' in g:
            g['CERTIFIED'] = False
    for key in ('payout_aware', 'payout_aware_forward', 'payout_aware_confidence'):
        reading = res.get(key)
        sides = reading.get('sides') if isinstance(reading, dict) else None
        if isinstance(sides, dict):
            for side in sides.values():
                if isinstance(side, dict) and 'survives' in side:
                    side['survives'] = False
    if 'verdict' in res:
        continuing = why in (
            ONLINE_POLICY_REASON,
            ACI_POLICY_REASON,
        )
        verdict = 'FORWARD DIAGNOSTIC (owner decision 13): never certified' if continuing else why
        res['verdict'] = {s: verdict for s in ('UP', 'DOWN')}
    res['noncertifying'] = why


def aggregate(res, pathinfo, forward, covs, data_uprate, gain, feats):
    """Verdicts for every harness (base, variant, pooled), one consistent rule set. A continuing (adaptive) policy
    (adaptive_policy) is never certified, whatever its figures, also when a stored result is aggregated again."""
    res['paths'] = pathinfo
    res['forward'] = forward
    aucs = [num_or_nan(p['auc']) for p in pathinfo]           # a reread file shows a missing number as null
    ups = [num_or_nan(p['up_rate']) for p in pathinfo]
    res['auc_summary'] = {'mean': float(np.nanmean(aucs)), 'min': float(np.nanmin(aucs)), 'max': float(np.nanmax(aucs))}
    data_uprate = num_or_nan(data_uprate)                   # null in a reread file (an all-tie run): not in band
    band = bool(0.47 <= data_uprate <= 0.53)
    trip = bool(all(0.45 <= u <= 0.55 for u in ups if np.isfinite(u)))
    valid = band and trip
    res['validity'] = {'VALID': valid, 'in_session_up_rate_in_band': band, 'paths_up_rate_in_tripwire': trip,
                       'paths_up_rate_in_0.47_0.53': bool(all(0.47 <= u <= 0.53 for u in ups if np.isfinite(u))),
                       'path_up_rate_range': [float(np.nanmin(ups)), float(np.nanmax(ups))]}
    res['baseline_always'] = {s: summ([p['always'][s][1] for p in pathinfo]) for s in ('UP', 'DOWN')}
    res['bycov'] = {}
    for c in covs:
        k = f'{c}'
        sm, vd, fw = {}, {}, {}
        for s in SIDES:
            kept = [p['bycov'][k][s] for p in pathinfo if p['bycov'][k][s][0] >= MIN_TRADES]
            sm[s] = summ([w for _, w in kept])
            p10 = sm[s].get('p10', float('nan'))
            fc = sm[s].get('frac_clear_BE', 0.0)
            vd[s] = {'p10': p10, 'frac_clear_BE': fc, 'n_paths_counted': sm[s]['n_paths'],
                     'med_n_per_path': int(np.median([n for n, _ in kept])) if kept else 0, 'VALID': valid,
                     'CERTIFIED': bool(valid and np.isfinite(p10) and p10 >= BE and fc >= 0.80 and sm[s]['n_paths'] >= 1)}
            if s != 'COMBINED':
                ex = [num_or_nan(p['bycov'][k][s][1]) - num_or_nan(p['always'][s][1]) for p in pathinfo
                  if p['bycov'][k][s][0] >= MIN_TRADES]
                vd[s]['excess_vs_always'] = {'mean': float(np.mean(ex)) if ex else None,
                                             'frac_paths_positive': float(np.mean([e > 0 for e in ex])) if ex else None}
            vals = [(r['bycov'][k][s][0], r['bycov'][k][s][1]) for r in forward]
            ok = [num_or_nan(w) for n, w in vals if n >= MIN_TRADES]
            fw[s] = {'paths': vals, 'mean': float(np.mean(ok)) if ok else None, 'min': float(np.min(ok)) if ok else None,
                     'n_clear_BE': int(sum(w >= BE for w in ok)), 'n_counted': len(ok)}
        res['bycov'][k] = {'summary': sm, 'verdict': vd, 'forward': fw}
    op = res['bycov'][f'{covs[0]}']['verdict']
    res['verdict'] = {s: ('CERTIFIED' if op[s]['CERTIFIED'] else 'KILLED' if valid else
                          'INVALID (moved up-rate validity check failed)') for s in ('UP', 'DOWN')}
    # The /goal's reading of "15 purged paths, p10 >= .541, >= 80% of paths clear": at least 12 of the 15 paths must be
    # counted (>= 25 trades) AND clear .541 - the repository rule above can certify from as few as one counted path.
    k0 = f'{covs[0]}'
    gv = {}
    for s in ('UP', 'DOWN'):
        w = [num_or_nan(p['bycov'][k0][s][1]) for p in pathinfo if p['bycov'][k0][s][0] >= MIN_TRADES]
        clear = int(sum(x >= BE for x in w))
        p10 = float(np.percentile(w, 10)) if w else float('nan')
        gv[s] = {'CERTIFIED': bool(valid and len(pathinfo) == 15 and len(w) >= 12 and clear >= 12 and p10 >= BE),
                 'paths_counted': len(w), 'paths_clear': clear, 'p10': p10}
    res['verdict_goal'] = gv
    why = adaptive_policy(res)
    if why:                                         # test E-003, owner decision 13: a forward diagnostic only
        veto_adaptive(res, why)
    res['paths_insufficient_support'] = [p['test'] for p in pathinfo if p.get('insufficient_support')]
    res['forward_insufficient_support'] = [p['test'] for p in forward if p.get('insufficient_support')]
    total = float(np.sum(gain))
    if np.isfinite(total) and total > 0:
        order = np.argsort(-gain)[:20]
        res['top_gain_features'] = [[feats[i], float(gain[i] / gain.sum())] for i in order]
    else:                                           # no tree gain: a logistic, kernel or forest model
        res['top_gain_features'] = None
        res['gain_report'] = 'not applicable: the fitted models report no tree gain (total gain 0 or not finite)'


def print_summary(res, covs, tag, t0, out):
    """The run's summary lines. Test E-003: a continuing policy (adaptive_policy) prints as a forward diagnostic
    whatever the result holds - veto_adaptive on a display copy; the result itself is not changed - so a result
    that an earlier build certified prints no certificate, held in memory or read back."""
    why = adaptive_policy(res)
    if why:
        import copy
        res = copy.deepcopy(res)
        veto_adaptive(res, why)
    print(f'[{tag}] AUC mean {num_or_nan(res["auc_summary"]["mean"]):.4f} '
          f'min {num_or_nan(res["auc_summary"]["min"]):.4f} '
          f'valid {res["validity"]["VALID"]} always UP {res["baseline_always"]["UP"].get("mean")} '
          f'DOWN {res["baseline_always"]["DOWN"].get("mean")}', flush=True)
    for c in covs:
        for s in SIDES:
            v = res['bycov'][f'{c}']['verdict'][s]
            f = res['bycov'][f'{c}']['forward'][s]
            print(f'   cov{c} {s:8s} p10={v["p10"]:.4f} frac={v["frac_clear_BE"]:.3f} paths={v["n_paths_counted"]} '
                  f'med_n={v["med_n_per_path"]} legacy CERT={v["CERTIFIED"]} | forward mean={num_or_nan(f["mean"])} '
                  f'clear={f["n_clear_BE"]}/'
                  f'{f["n_counted"]}' if np.isfinite(num_or_nan(v['p10'])) else f'   cov{c} {s:8s} no counted path',
                  flush=True)
    strict = {s: (res.get('verdict_goal') or {}).get(s, {}).get('CERTIFIED') for s in ('UP', 'DOWN')}
    print(f'strict certificate (verdict_goal) CERTIFIED {strict} | legacy verdict {res["verdict"]} '
          f'in {time.time() - t0:.1f} s -> {out}', flush=True)


class Fitter:
    def __init__(self, X, fwd, ts, a, gap, covs):
        self.X, self.fwd, self.ts, self.a, self.gap, self.covs = X, fwd, ts, a, gap, covs
        self.gain = np.zeros(X.shape[1])

    def draw_split(self, tr_idx, rng):
        """Validation rows and fit pool of a split: every draw a split takes from the shared generator happens here and
        nowhere else, so --path-jobs can advance the generator past a split without fitting it."""
        valsel = rng.random(len(tr_idx)) < VAL_FRAC
        val_idx = tr_idx[valsel]
        fit_pool = tr_idx[~valsel]
        if len(fit_pool) > SUB_FIT:
            fit_pool = rng.choice(fit_pool, SUB_FIT, replace=False)    # reference: sampled order kept (no sort)
        return val_idx, fit_pool

    def split(self, tr_idx, test_mask, rng, label):
        """Fit on tr_idx (moved rows), threshold from a random validation subset, score test_mask."""
        X, fwd, ts, a, gap = self.X, self.fwd, self.ts, self.a, self.gap
        val_idx, fit_pool = self.draw_split(tr_idx, rng)
        if len(fit_pool) == 0 or len(val_idx) == 0:     # test E-003: nothing to fit or to set a threshold on
            return unsupported_row(label, len(fit_pool), len(val_idx), fwd[test_mask], ts[test_mask], self.covs, gap)
        if not test_mask.any():                         # test E-003: nothing to score
            return unsupported_row(label, len(fit_pool), len(val_idx), fwd[test_mask], ts[test_mask], self.covs, gap,
                                   'insufficient support: no test row')
        if len(fit_pool) < 2:                           # test E-003: LightGBM needs at least two fit rows
            return unsupported_row(label, len(fit_pool), len(val_idx), fwd[test_mask], ts[test_mask], self.covs, gap,
                                   'insufficient support: fewer than two fit rows (LightGBM needs two)')
        ytr = (fwd[fit_pool] > 0).astype(int)
        yval = (fwd[val_idx] > 0).astype(int)
        if not np.isin(yval, ytr).all():    # test E-003: LightGBM's early stopping refuses a label the fit lacks
            return unsupported_row(label, len(fit_pool), len(val_idx), fwd[test_mask], ts[test_mask], self.covs, gap,
                                   'insufficient support: a validation label is absent from the fit labels')
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
        ftst = fwd[test_mask]
        ttst = ts[test_mask]                      # chronological (rows are sorted by decision time)
        mv = ftst != 0
        mvf = mv & np.isfinite(ptst)                  # test E-003: the diagnostic reads finite scores only
        auc = float(roc_auc_score((ftst[mvf] > 0).astype(int), ptst[mvf])) if mvf.sum() > 20 else float('nan')
        uprate = float((ftst[mv] > 0).mean()) if mv.sum() else float('nan')
        row = {'test': label, 'n_fit': int(len(fit_pool)), 'n_val': int(len(val_idx)), 'n_test': int(test_mask.sum()),
               'best_iter': iters, 'auc': auc, 'up_rate': uprate, 'always': {}, 'bycov': {}}
        return score_row(row, ptst, ftst, ttst, np.abs(pval - 0.5), self.covs, gap)


_PJ = {}        # --path-jobs: the fitter, set once before the first worker is forked


def _path_job(spec):
    """One split in a forked worker, from the generator state the sequential loop has when it reaches this split."""
    label, tr_idx, test_mask, state = spec
    f = _PJ['fitter']
    rng = np.random.default_rng()
    rng.bit_generator.state = state
    f.gain = np.zeros_like(f.gain)
    return f.split(tr_idx, test_mask, rng, label), f.gain


def fmt_side(b):
    """One coverage's side results for the log; a missing win rate (NaN, or null in a reread file) prints nan."""
    return ' '.join(f'{lab} [{b[s][0]}, {num_or_nan(b[s][1]):.4f}]'
                    for lab, s in (('UP', 'UP'), ('DOWN', 'DOWN'), ('COMB', 'COMBINED')))


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
    ap.add_argument('--n-jobs', type=int, default=16, help='threads of each model fit')
    ap.add_argument('--path-jobs', type=int, default=1,
                    help='splits (CPCV paths, forward walks) fitted at once, each in a forked worker with --n-jobs threads')
    ap.add_argument('--no-forward', action='store_true')
    ap.add_argument('--include-blackout', action='store_true',
                    help='keep decisions inside the Deriv purchase blackout (not Deriv-faithful; <= 0.2.1 behavior)')
    ap.add_argument('--july-filter', action='store_true',
                    help='the July 2026 tradeability filter, without the venue window of owner decision 14 '
                         '(regression against stored results only; such a result cannot be promoted)')
    a = ap.parse_args()
    if a.path_jobs < 1 or (a.path_jobs > 1 and a.n_jobs < 1):    # a sequential run takes any --n-jobs
        raise SystemExit('--path-jobs must be >= 1; parallel paths (--path-jobs > 1) need --n-jobs >= 1 per worker')
    covs = parse_covs(a.covs)
    for name, value, ok in (('--nseed', a.nseed, a.nseed >= 1), ('--stride', a.stride, a.stride >= 1),
                            ('--num-leaves', a.num_leaves, 2 <= a.num_leaves <= 131072)):
        if not ok:                                  # test E-003: before anything is loaded or written
            raise SystemExit(f'{name} {value}: out of its range (seeds and stride >= 1; LightGBM takes 2 to 131072 '
                             'leaves)')
    gap = a.horizon * 60
    out = Path(a.out or f'{a.pair}_{a.horizon}m_{a.session}_cpcv_result.json')
    t0 = time.time()

    listed = json.loads(Path(a.feature_list).read_text()) if a.feature_list else None
    X, fwd, ts, feats = load(a.dataset, a.pair, a.horizon, a.features, a.stride, listed)
    if listed is not None:
        a.features = f'list:{Path(a.feature_list).name}'
    sess = session_mask(ts, a.session)
    if not a.include_blackout:
        sess &= tradeable(ts, a.horizon, venue_window=not a.july_filter)
    elif not a.july_filter:                         # owner decision 14 holds with --include-blackout too
        sess &= venue_open(ts, a.horizon)
    moved = np.isfinite(fwd) & (fwd != 0.0)
    tgs = ts[sess]
    bnds = [tgs[int(k * len(tgs) / N_GROUPS)] for k in range(N_GROUPS)] + [tgs[-1] + 1]
    groups = [(int(bnds[g]), int(bnds[g + 1])) for g in range(N_GROUPS)]
    day = lambda s: pd.to_datetime(s, unit='s', utc=True).strftime('%Y-%m-%d')
    data_uprate = float((fwd[sess & moved] > 0).mean())
    manifest = Path(a.dataset) / 'manifest.json'
    res = {
        'harness': HARNESS_VERSION, 'tag': a.tag, 'run_status': 'running',   # 'complete' when written at the end
        'model': f'LightGBM (repository mk_lgb) @{a.horizon}m on the Deriv feed, session {a.session}, '
                 f'ties-strict per-fold-refit CPCV',
        'key': f'{a.pair}.{a.horizon}m', 'data': 'Deriv 1-second mid ticks only (/mnt/data/deriv), no Dukascopy',
        'dataset': {'dir': a.dataset, 'manifest_sha256': sha256(manifest) if manifest.exists() else None,
                    'builder': json.loads(manifest.read_text()).get('builder') if manifest.exists() else None},
        'window': {'first_decision': day(ts[0]), 'last_decision': day(ts[-1])},
        'label': f'Deriv settlement over {a.horizon} min: entry first tick >= decision + 1 s, exit last tick <= '
                 f'entry target + {a.horizon} min, ties valid and lose',
        'breakeven': BE, 'session': a.session, 'tradeable_only': not a.include_blackout,
        'blackout_rule': None if a.include_blackout else BLACKOUT_RULE,
        'venue_window_rule': None if a.july_filter else VENUE_WINDOW_RULE,
        'regression_only': bool(a.july_filter),     # True: a regression against stored results, never a candidate
        'horizon_min': a.horizon, 'gap_s': gap, 'stride': a.stride,
        'covs': covs, 'operating_cov': covs[0], 'nseed': a.nseed, 'num_leaves': a.num_leaves,
        'feature_set': a.features, 'n_features': len(feats), 'features': feats,
        'n_groups': N_GROUPS, 'k_test': K_TEST, 'purge_s': gap, 'embargo_s': gap, 'sub_fit': SUB_FIT,
        'val_frac': VAL_FRAC, 'min_trades_per_path': MIN_TRADES,
        'rows': int(len(ts)), 'rows_in_session': int(sess.sum()), 'in_session_moved_up_rate': data_uprate,
        'groups': [(day(lo), day(hi - 1)) for lo, hi in groups],
        'CERT_RULE': 'side CERTIFIED iff VALID AND p10>=0.541 AND frac_clear_BE>=0.80 (unrounded path win rates; paths with >= 25 trades)',
        'falsifier': {'version': 3, 'registered_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      'when': 'before any path is fitted',
                      'KILL_if': f'at the operating coverage {covs[0]}: each side p10<0.541 OR frac_clear_BE<0.80',
                      'INVALID_if': 'in-session moved up-rate outside [0.47,0.53], or any path moved up-rate outside '
                                    '[0.45,0.55]; an INVALID run certifies nothing in any field'},
        'code': {'harness_sha256': sha256(__file__), 'sessions_py_sha256': sha256(REPO / 'scripts' / 'sessions.py'),
                 'lightgbm': lgb.__version__, 'numpy': np.__version__, 'pandas': pd.__version__},
        'compute': {'n_jobs': a.n_jobs, 'path_jobs': a.path_jobs},
    }
    out.write_text(result_json(res))
    print(f'[{a.pair} {a.horizon}m {a.session}] rows {len(ts):,} in-session {int(sess.sum()):,} features {len(feats)} '
          f'up-rate {data_uprate:.4f} groups {res["groups"]}', flush=True)

    fitter = Fitter(X, fwd, ts, a, gap, covs)

    def path_split(testg):                      # training rows, test rows
        tin = np.zeros(len(ts), bool)
        purged = np.zeros(len(ts), bool)
        for g in testg:
            lo, hi = groups[g]
            tin |= (ts >= lo) & (ts < hi)
            purged |= (ts >= lo - gap) & (ts < hi + gap)
        return np.where((~purged) & moved & sess)[0], tin & sess & np.isfinite(fwd)

    def forward_split(g):
        lo, hi = groups[g]
        return np.where((ts < lo - gap) & moved & sess)[0], (ts >= lo) & (ts < hi) & sess & np.isfinite(fwd)

    def say_path(fi, testg, row):
        print(f'  path {fi + 1:2d}/15 g{list(testg)} AUC={row["auc"]:.4f} up={row["up_rate"]:.4f} iters={row["best_iter"]} | '
              f'cov{covs[0]}: {fmt_side(row["bycov"][f"{covs[0]}"])} ({time.time() - t0:.0f}s)', flush=True)

    def say_forward(g, row):
        print(f'  forward g{g} AUC={row["auc"]:.4f} | cov{covs[0]}: {fmt_side(row["bycov"][f"{covs[0]}"])} '
              f'({time.time() - t0:.0f}s)', flush=True)

    path_groups = list(combinations(range(N_GROUPS), K_TEST))
    forward_groups = [] if a.no_forward else list(range(2, N_GROUPS))
    pathinfo, forward = [], []
    if a.path_jobs == 1:
        rng = np.random.default_rng(13)
        for fi, testg in enumerate(path_groups):
            row = fitter.split(*path_split(testg), rng, list(testg))
            pathinfo.append(row)
            say_path(fi, testg, row)
        frng = np.random.default_rng(17)
        for g in forward_groups:
            row = fitter.split(*forward_split(g), frng, f'forward g{g}')
            forward.append(row)
            say_forward(g, row)
    else:
        # Each split is handed the generator state the sequential loop has when it reaches that split: the parent takes
        # the split's draws (draw_split) without fitting it, in the sequential order, on the same two streams.
        specs = []
        rng = np.random.default_rng(13)
        for testg in path_groups:
            tr_idx, test_mask = path_split(testg)
            specs.append((list(testg), tr_idx, test_mask, copy.deepcopy(rng.bit_generator.state)))
            fitter.draw_split(tr_idx, rng)
        frng = np.random.default_rng(17)
        for g in forward_groups:
            tr_idx, test_mask = forward_split(g)
            specs.append((f'forward g{g}', tr_idx, test_mask, copy.deepcopy(frng.bit_generator.state)))
            fitter.draw_split(tr_idx, frng)
        _PJ.update(fitter=fitter)
        done = []
        for i, got in in_order(run_forked(_path_job, specs, a.path_jobs)):
            done.append(got)
            if i < len(path_groups):
                say_path(i, path_groups[i], got[0])
            else:
                say_forward(forward_groups[i - len(path_groups)], got[0])
        for row, gain in done:                      # merged in the sequential order
            fitter.gain += gain
        pathinfo = [d[0] for d in done[:len(path_groups)]]
        forward = [d[0] for d in done[len(path_groups):]]
    aggregate(res, pathinfo, forward, covs, data_uprate, fitter.gain, feats)
    res['seconds'] = round(time.time() - t0, 1)
    res['run_status'] = 'complete'
    out.write_text(result_json(res))
    print_summary(res, covs, f'{a.pair} {a.horizon}m', t0, out)


if __name__ == '__main__':
    main()
