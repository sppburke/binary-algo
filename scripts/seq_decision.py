#!/usr/bin/env python3
"""Two-look group-sequential holdout decision (owner decision 6, 2026-11-15; holdout/REGISTER.md).

    seq_decision.py --look 1 --result LOOK1.json [--incumbents DEPLOYED.json TRAINED.json]
    seq_decision.py --look 2 --result LOOK2.json --look1-result LOOK1.json [--incumbents DEPLOYED.json TRAINED.json]

LOOKn.json: holdout_eval.py output for ONE registered finalist (REGISTERED below; k = 1 per key). Incumbents: the two
score_frozen_books.py outputs on the same window (as deployed: session ny on dataset_full_b239_*; as trained: nylate on
the shift-5 schema) - required for a key with frozen books, refused for a key without. The binding side is the
registered side (UP for both finalists). Pure arithmetic on JSON files; reads no market data.

Null model (stated assumption, the same one the program's single-look CI95 rule makes): conditional on the decision
process - refits and trade selection use only data before each trade - the nonoverlapping PRIMARY trade outcomes are
independent Bernoulli(p) with p <= 0.541 (ties lose). Under it the cumulative wins are exactly binomial at the boundary
p = 0.541, so the boundaries below are EXACT integer win counts:

  look 1 (n1 trades): certify-cross iff W1 >= w1 = max(smallest w with P0(Bin(n1) >= w) <= ALPHA1,
                       ceil(0.541 n1 + 2.80 sqrt(0.541 x 0.459 x n1)))           ALPHA1 = 1 - Phi(2.797) = 0.00258
  look 2 (n2 >= n1):   certify-cross iff W2 >= w2 = max(smallest w with
                       P0(W1 >= w1) + P0(W1 < w1, W1 + Bin(n2 - n1) >= w) <= 0.025,
                       ceil(0.541 n2 + 1.98 sqrt(0.541 x 0.459 x n2)))
So the family-wise one-sided error is <= 0.025 per key exactly under the null model, and the owner's quoted thresholds
(z >= 2.80 at look 1, z >= 1.98 at look 2) are floors that are never relaxed.
CERTIFIED at a look iff it crosses AND at that look: PRIMARY n >= 50; drift_matched.available and .passes; the exact
win fraction W/n strictly above every registered incumbent book's UP win fraction (both readings). A crossing with a
failed condition does not certify and does not stop the test; no crossing at look 2 = not certified (final).
Provenance (1.2): both looks run the frozen code in holdout/frozen_v1 (hashes in CODE below, plus the repository's
sessions.py / book_runtime.py); wins are counted from the evaluator's PRIMARY trade list, and at look 2 the trades
dated inside the look-1 window must equal the look-1 trades exactly (timestamps and outcomes). An incumbent book
without UP trades on the window cannot be compared: the tool refuses (owner decision needed).
"""
import argparse
import json
import math
import sys
from pathlib import Path

from scipy.stats import binom, norm

VERSION = 'seq_decision/1.2'
P0, ALPHA = 0.541, 0.025
ALPHA1 = float(1 - norm.cdf(2.797))
Z1_FLOOR, Z2_FLOOR = 2.80, 1.98
INDEX = Path('/home/sean/git/binary-algo/books/INDEX.json')
FROZEN = '/mnt/data/deriv-sota-lab/holdout/frozen_v1/'
CODE = {'holdout_eval': 'ad6b133fceb3977b8f2b350d58830e955c737fd7c2787e35ae02963128f57926',      # 0.6.2
        'cpcv_deriv': '9118fffdc0b4860109e9ea3caa5d12f77dda845f3a48f074b856ef2d985898d4',        # 0.4.1
        'cpcv_deriv_x': 'f0798bd33e65ce3c128b7b742497e4211ed70d13074d5c2b8f4cb4a8ca97e15a',      # 0.7.2
        'sessions': '2a20b7a57e3e6e5884256c640b10b2bec9cfd2c21325617d4ed6aacf823d1493',          # repository scripts
        'score_frozen_books': 'b1c9cd6e30f0f5bd6a717c7ab36a80e8a13986d24f872006fcc036228087f84b',  # 0.4
        'book_runtime': 'fb299d8adc8b585e0c19294a362c32811912eadf5d87f5a92397c7aba77e916d'}       # repository scripts
LOOK_WINDOWS = {1: ['2026-08-01', '2026-11-14'], 2: ['2026-08-01', '2027-02-26']}
LOOK_DATASET = {1: 'dataset_full_v0.1', 2: 'dataset_full_v0.2'}
LOOK_BOOK_DATASETS = {1: {'ny': 'dataset_full_b239_v0.1', 'nylate': 'dataset_full_b239_shift-5_v0.1'},
                      2: {'ny': 'dataset_full_b239_v0.2', 'nylate': 'dataset_full_b239_shift-5_v0.2'}}
KERNEL = {'model': 'nystroem', 'nys_gamma_mult': 1.0, 'nys_components': 500, 'weight': 'none'}
REGISTERED = {
    'USDJPY.15m': {'finalist': 'e30nystroem', 'side': 'UP', 'model': KERNEL, 'nseed': 1, 'cov': 0.03, 'session': 'ny',
                   'features': ('all', 66),
                   'incumbents': ['USDJPY.m15ny.v1', 'USDJPY.m15ny_seedens.v1', 'USDJPY.m15ny_tbft_seedens.v1']},
    'USDJPY.30m': {'finalist': 'e49nys30', 'side': 'UP', 'model': KERNEL, 'nseed': 1, 'cov': 0.03, 'session': 'ny',
                   'features': ('all', 66), 'incumbents': []},
}


class Refused(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise Refused(msg)


def floor_count(n, z):
    return math.ceil(P0 * n + z * math.sqrt(P0 * (1 - P0) * n) - 1e-9)


def look1_threshold(n1):
    w = int(binom.isf(ALPHA1, n1, P0)) + 1          # smallest w with P(Bin >= w) <= ALPHA1 ...
    while w > 0 and binom.sf(w - 2, n1, P0) <= ALPHA1:
        w -= 1                                       # ... made exact against isf rounding
    while binom.sf(w - 1, n1, P0) > ALPHA1:
        w += 1
    return max(w, floor_count(n1, Z1_FLOOR))


def look2_threshold(n1, n2, w1):
    spent1 = float(binom.sf(w1 - 1, n1, P0))
    m = n2 - n1
    pmf1 = binom.pmf(range(w1), n1, P0)              # W1 = 0 .. w1 - 1 (no look-1 crossing)

    def total(w):                                    # P0(cross at 1) + P0(no cross at 1, W2 >= w)
        return spent1 + float(sum(pmf1[k] * binom.sf(w - k - 1, m, P0) for k in range(w1)))
    lo, hi = 0, n2 + 1                               # total() is non-increasing in w; total(n2 + 1) = spent1 <= ALPHA
    while lo < hi:
        mid = (lo + hi) // 2
        if total(mid) <= ALPHA:
            hi = mid
        else:
            lo = mid + 1
    return max(lo, floor_count(n2, Z2_FLOOR)), spent1, total(max(lo, floor_count(n2, Z2_FLOOR)))


def zscore(w, n):
    return (w - P0 * n) / math.sqrt(P0 * (1 - P0) * n) if n else float('-inf')


def load_finalist(path, look):
    r = json.load(open(path))
    key = r.get('key')
    need(key in REGISTERED, f'{path}: key {key!r} is not a registered finalist key')
    reg = REGISTERED[key]
    need(r.get('holdout_eval') == 'holdout_eval/0.6.2', f'{path}: evaluator {r.get("holdout_eval")!r} is not the frozen 0.6.2')
    code = r.get('code') or {}
    need(code.get('holdout_sha256') == CODE['holdout_eval'], f'{path}: holdout_eval hash differs from the frozen code')
    need(code.get('base_harness_sha256') == CODE['cpcv_deriv'], f'{path}: cpcv_deriv hash differs from the frozen code')
    need((code.get('x_harness') or {}).get('sha256') == CODE['cpcv_deriv_x'], f'{path}: cpcv_deriv_x hash differs from the frozen code')
    need(str((code.get('x_harness') or {}).get('file', '')).startswith(FROZEN), f'{path}: the model code did not run from {FROZEN}')
    need((code.get('sessions') or {}).get('sha256') == CODE['sessions'], f'{path}: sessions.py hash differs from the registration')
    need(r.get('holdout') == LOOK_WINDOWS[look], f'{path}: window {r.get("holdout")} is not the look-{look} window {LOOK_WINDOWS[look]}')
    need(Path(str(r.get('dataset', ''))).name == LOOK_DATASET[look], f'{path}: dataset {r.get("dataset")!r} is not {LOOK_DATASET[look]}')
    need(r.get('n_candidates_opened_for_key') == 1, f'{path}: --n-candidates must be 1 (registered k = 1)')
    need(r.get('pool') is None, f'{path}: a pooled fit is not the registered configuration')
    need(r.get('nseed') == reg['nseed'] and abs(float(r.get('cov', -1)) - reg['cov']) < 1e-12 and r.get('session') == reg['session'],
         f'{path}: nseed / cov / session {r.get("nseed")} / {r.get("cov")} / {r.get("session")} differ from the registration')
    for k, v in reg['model'].items():
        need(r.get('model', {}).get(k) == v, f'{path}: model.{k} = {r.get("model", {}).get(k)!r}, registered {v!r}')
    need((r.get('feature_set'), r.get('n_features')) == reg['features'], f'{path}: features {r.get("feature_set")} / {r.get("n_features")} differ')
    s = r['readings']['PRIMARY_monthly_refit']['sides'][reg['side']]
    trades = s.get('trades')
    need(isinstance(trades, list), f'{path}: no PRIMARY trade list (evaluator older than 0.6.2?)')
    lo, hi = window_seconds(LOOK_WINDOWS[look])
    prev = None
    for tr in trades:
        need(isinstance(tr, list) and len(tr) == 2 and all(isinstance(v, int) for v in tr) and tr[1] in (0, 1),
             f'{path}: malformed trade {tr!r}')
        need(lo <= tr[0] < hi, f'{path}: trade at {tr[0]} outside the look-{look} window')
        need(prev is None or tr[0] > prev, f'{path}: trades not strictly increasing in time')
        prev = tr[0]
    n, w = len(trades), sum(tr[1] for tr in trades)
    need(isinstance(s.get('n'), int) and s['n'] == n, f'{path}: n {s.get("n")} differs from the {n} listed trades')
    need(n == 0 or (s.get('win_rate') is not None and abs(s['win_rate'] - round(w / n, 4)) < 1e-9),
         f'{path}: win_rate {s.get("win_rate")} differs from the listed trades ({w}/{n})')
    need(sum(v[0] for v in s.get('by_month', {}).values()) == n, f'{path}: by_month counts do not sum to n')
    return r, reg, s, n, w, trades


def window_seconds(win):
    import datetime as dt
    a = dt.datetime.fromisoformat(win[0]).replace(tzinfo=dt.timezone.utc)
    b = dt.datetime.fromisoformat(win[1]).replace(tzinfo=dt.timezone.utc) + dt.timedelta(days=1)
    return int(a.timestamp()), int(b.timestamp())


def check_prefix(trades1, trades2):
    """The look-2 trades dated inside the look-1 window must be exactly the look-1 trades (cumulative windows)."""
    _, end1 = window_seconds(LOOK_WINDOWS[1])
    prefix = [tr for tr in trades2 if tr[0] < end1]
    need(prefix == trades1, f'look-2 trades inside the look-1 window ({len(prefix)}) are not the look-1 trades '
                            f'({len(trades1)}): the looks are not cumulative - stop and investigate')


def incumbents(paths, reg, look, cand_w, cand_n):
    if not reg['incumbents']:
        need(not paths, 'this key has no registered incumbent; do not pass --incumbents')
        index = json.loads(INDEX.read_text())['books']
        need(not any(e.get('currency') == 'USDJPY' and e.get('timeframe') == '30m' for e in index),
             'books/INDEX.json now lists a USDJPY 30m book; the registration says none - stop and re-register')
        return True, []
    need(paths and len(paths) == 2, 'pass exactly two incumbent files (as deployed: ny; as trained: nylate)')
    seen, rows = set(), []
    for p in paths:
        d = json.load(open(p))
        sess = 'ny' if str(d.get('session', '')).startswith('ny (') else 'nylate' if str(d.get('session', '')).startswith('nylate') else None
        need(sess is not None and sess not in seen, f'{p}: session {d.get("session")!r} missing or duplicated')
        seen.add(sess)
        need(d.get('window_filter') == LOOK_WINDOWS[look], f'{p}: window {d.get("window_filter")} is not the look-{look} window')
        need(Path(d['dataset']['dir']).name == LOOK_BOOK_DATASETS[look][sess], f'{p}: dataset {d["dataset"]["dir"]} is not {LOOK_BOOK_DATASETS[look][sess]}')
        need(d.get('tradeable_only') is True, f'{p}: incumbents must be scored on tradeable decisions only')
        dc = d.get('code') or {}
        need(dc.get('scorer_sha256') == CODE['score_frozen_books'], f'{p}: score_frozen_books hash differs from the frozen code')
        need(dc.get('book_runtime_sha256') == CODE['book_runtime'] and dc.get('sessions_sha256') == CODE['sessions'],
             f'{p}: book_runtime / sessions hash differs from the registration')
        books = {b['book_id']: b for b in d.get('books', [])}
        for bid in reg['incumbents']:
            need(bid in books and books[bid].get('status') == 'SCORED', f'{p}: incumbent {bid} missing or not SCORED')
            u = books[bid]['sides']['UP']
            need(isinstance(u.get('n'), int) and u['n'] >= 0, f'{p}: {bid} UP n {u.get("n")!r} is not a count')
            nb = u['n']
            need(nb > 0, f'{p}: {bid} has no UP trades on the window - no point win rate to compare (owner decision needed)')
            need(u.get('win_rate') is not None and 0 <= u['win_rate'] <= 1, f'{p}: {bid} UP win_rate {u.get("win_rate")!r} invalid')
            wb = int(round(u['win_rate'] * nb))
            need(abs(wb / nb - u['win_rate']) < 6e-5, f'{p}: {bid} win count not recoverable from its win_rate')
            beat = cand_w * nb > wb * cand_n
            rows.append({'book_id': bid, 'reading': sess, 'n': nb, 'wins': wb, 'win_rate': u['win_rate'], 'beaten': beat})
    return all(x['beaten'] for x in rows), rows


def decide(look, result, look1_result=None, incumbent_paths=None):
    r, reg, s, n, w, trades = load_finalist(result, look)
    side = reg['side']
    n1, w1_obs = n, w
    if look == 1:
        thr = look1_threshold(n) if n else None
        spent = float(binom.sf(thr - 1, n, P0)) if n else 0.0
        total = spent
    else:
        need(look1_result is not None, '--look 2 needs --look1-result')
        r1, reg1, s1, n1, w1_obs, trades1 = load_finalist(look1_result, 1)
        need(r1['key'] == r['key'], 'look-1 and look-2 files are for different keys')
        need(n >= n1, f'look 2 has fewer trades ({n}) than look 1 ({n1})')
        check_prefix(trades1, trades)
        t1 = look1_threshold(n1) if n1 else n1 + 1   # no look-1 trades: nothing could cross at look 1
        thr, spent, total = look2_threshold(n1, n, t1) if n else (None, 0.0, 0.0)
    crossed = bool(n and w >= thr)
    dm = s.get('drift_matched') or {}
    inc_ok, inc_rows = incumbents(incumbent_paths or [], reg, look, w, n)
    conds = {'n_ge_50': n >= 50, 'drift_matched_available': bool(dm.get('available')),
             'drift_matched_passes': bool(dm.get('passes')), 'beats_every_incumbent': bool(inc_ok)}
    cert = bool(crossed and all(conds.values()))
    return {'tool': VERSION, 'look': look, 'key': r['key'], 'finalist': reg['finalist'], 'side': side, 'window': r['holdout'],
            'result': str(result), 'look1_result': None if look1_result is None else str(look1_result),
            'n': n, 'wins': w, 'win_rate': s['win_rate'], 'z_score': round(zscore(w, n), 4),
            'threshold_wins': thr, 'threshold_z': None if thr is None else round(zscore(thr, n), 4),
            'n_look1': n1, 'wins_look1': w1_obs,
            'alpha_spent_look1_exact': round(spent if look == 1 else spent, 6), 'alpha_total_exact': round(total, 6),
            'crossed': crossed, 'conditions': conds, 'incumbents': inc_rows,
            'drift_matched': dm, 'CERTIFIED_AT_THIS_LOOK': cert,
            'next': 'stop: certified' if cert else ('final: not certified' if look == 2 else 'continue to look 2')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--look', type=int, choices=(1, 2), required=True)
    ap.add_argument('--result', required=True)
    ap.add_argument('--look1-result')
    ap.add_argument('--incumbents', nargs='*', default=[])
    a = ap.parse_args()
    try:
        out = decide(a.look, a.result, a.look1_result, a.incumbents)
    except Refused as e:
        sys.exit(f'REFUSED: {e}')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
