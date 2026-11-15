#!/usr/bin/env python3
"""Forward test of the frozen 15m/30m books on the Deriv feed, as deployed.

    score_frozen_books.py --dataset DIR [--books ID,ID,...] [--out result.json]

DIR is a Deriv-only feature dataset written by tools/deriv_features_b239.py (the books' own feature
recipes rebuilt from Deriv 1-second mid ticks, Deriv-settled labels). For every registered book with a
15m or 30m timeframe (books/INDEX.json):
  - the exact LightGBM models, feature columns, deploy coverage and confidence threshold are read
    through scripts/book_runtime (no lifecycle check: this scores, it never deploys);
  - rows: every decision in the dataset window inside the DST-correct NY session (08:00-17:00
    America/New_York, the live executor's external gate) whose label is valid and whose book features
    are all finite (the live path fails closed otherwise);
  - gate: |p - 0.5| >= threshold AND, when the strategy has a compression gate, <tf>_bb_width <= its
    threshold (the column the strategy names; the runtime always reads 15m_bb_width, which differs
    from the strategy only for the 30m EURUSD books, noted per book);
  - trades: side filter, gate, first-come de-overlap with gap H (nonoverlap_chrono); a win needs the
    right sign and a move (ties lose). CI95: the repository's iid bootstrap (5000, seed 7).
All books were trained on 2012-2021 and gated on 2022-2023, so any window from 2025-08-11 on is
strictly out of sample for them.
Falsifier, registered before scoring (per book and side): the book keeps a deployable edge on the Deriv
feed iff n >= 50 and CI95-lower >= 0.541; otherwise NOT SUPPORTED.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path('/home/sean/git/binary-algo')
sys.path.insert(0, str(REPO / 'scripts'))
import book_runtime as br  # noqa: E402
from sessions import session_mask  # noqa: E402

SCORER_VERSION = 'score_frozen_books/0.4'   # 0.4: runtime compression-gate column by default (--gate-source)   # 0.3: Deriv purchase-blackout decisions excluded by default; 0.2 window filter


def tradeable_mask(ts, horizon_min):
    """Deriv rejects an FX Rise/Fall purchase whose expiry lands in 16:50-18:00 New York (docs/DERIV_DEMO_EXECUTOR.md (g)):
    decisions with start in [16:50 - H, 18:00) NY are untradeable."""
    t = pd.to_datetime(np.asarray(ts), unit='s', utc=True).tz_convert('America/New_York')
    m = t.hour.values * 60 + t.minute.values
    return ~((m >= 16 * 60 + 50 - horizon_min) & (m < 18 * 60))
BE = 0.541


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def boot(corr, nb=5000, seed=7):
    corr = np.asarray(corr, dtype=float)
    if len(corr) < 5:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    n = len(corr)
    a = np.array([corr[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def nonoverlap_chrono(ts, mask, gap):
    take, block = [], -1
    for i in np.where(mask)[0]:
        if ts[i] < block:
            continue
        take.append(i)
        block = int(ts[i]) + gap
    return np.array(take, dtype=int)


def stream(p, fwd, ts, gate, side, gap):
    want = (p > 0.5) if side == 'UP' else (p < 0.5) if side == 'DOWN' else np.ones(len(p), bool)
    tr = nonoverlap_chrono(ts, gate & want, gap)
    if len(tr) == 0:
        return tr, np.zeros(0)
    pred_up = p[tr] > 0.5
    win = np.where(pred_up, fwd[tr] > 0, fwd[tr] < 0).astype(float)     # ties (fwd == 0) lose
    return tr, win


def feature_names(strategy, boosters):
    for key in ('feature_cols', 'primary_feats', 'feature_names', 'features'):
        vals = strategy.get(key)
        if isinstance(vals, list) and vals:
            return [str(v) for v in vals], f'strategy.{key}'
    names = boosters[0].feature_name()
    if names and not all(n.startswith('Column_') for n in names):
        return list(names), 'booster feature_names'
    raise RuntimeError('no feature names in the strategy or the booster')


def compression_gate(strategy, source='runtime'):
    """runtime (default, as deployed): book_runtime.LoadedBook applies any bb_width threshold to 15m_bb_width;
    strategy: the column the strategy JSON names (differs only for the 30m EURUSD books)."""
    thr = None
    for key in ('bb_width_thr', 'bbw_thr'):
        if isinstance(strategy.get(key), (int, float)):
            thr = float(strategy[key])
    gate = strategy.get('gate')
    if thr is None and isinstance(gate, dict):
        for key in ('bb_width_thr', 'bbw_thr'):
            if isinstance(gate.get(key), (int, float)):
                thr = float(gate[key])
    if thr is None:
        return None, None
    if source == 'runtime':
        return '15m_bb_width', thr
    col = strategy.get('gate_feat') or (f'{strategy["comp_tf"]}_bb_width' if strategy.get('comp_tf') else '15m_bb_width')
    return col, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--books', default=None)
    ap.add_argument('--out', default='frozen_books_deriv_result.json')
    ap.add_argument('--session', default='ny', choices=('ny', 'nylate'),
                    help="ny = America/New_York 08-17 (live executor); nylate = 13-22 ET, the true hours the books' Dukascopy "
                         "'NY' gate selected before 2025-08-22 (stored = true UTC - 5 h)")
    ap.add_argument('--gate-source', default='runtime', choices=('runtime', 'strategy'),
                    help='compression-gate column: runtime = 15m_bb_width as book_runtime applies it (as deployed); '
                         'strategy = the column the strategy JSON names')
    ap.add_argument('--include-blackout', action='store_true',
                    help='keep decisions inside the Deriv purchase blackout (0.1/0.2 behavior; not tradeable)')
    ap.add_argument('--from-day', default=None, help='first decision day (UTC) to score, e.g. 2026-08-01 for the holdout')
    ap.add_argument('--to-day', default=None, help='last decision day (UTC) to score, inclusive')
    a = ap.parse_args()
    lo_ts = int(pd.Timestamp(a.from_day, tz='UTC').timestamp()) if a.from_day else -1
    hi_ts = int((pd.Timestamp(a.to_day, tz='UTC') + pd.Timedelta(days=1)).timestamp()) if a.to_day else 1 << 62
    t0 = time.time()
    index = json.loads((REPO / 'books' / 'INDEX.json').read_text())['books']
    wanted = set(a.books.split(',')) if a.books else None
    books = [e for e in index if e.get('timeframe') in ('15m', '30m') and (wanted is None or e['id'] in wanted)]
    manifest = Path(a.dataset) / 'manifest.json'
    res = {'scorer': SCORER_VERSION, 'dataset': {'dir': a.dataset, 'manifest_sha256': sha256(manifest),
                                                  'builder': json.loads(manifest.read_text()).get('builder')},
           'data': 'Deriv 1-second mid ticks only (/mnt/data/deriv), no Dukascopy data; frozen book artifacts are scored, not refit',
           'session': ('ny (America/New_York 08:00-17:00, DST-correct, sessions.session_mask)' if a.session == 'ny'
                       else 'nylate (America/New_York 13:00-22:00, DST-correct)') + ' on the decision time',
           'window_filter': [a.from_day, a.to_day], 'tradeable_only': not a.include_blackout, 'gate_source': a.gate_source,
           'breakeven': BE,
           'falsifier': {'registered_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'when': 'before scoring',
                         'SUPPORTED_iff': 'per book and side: n >= 50 AND CI95-lower >= 0.541 (else NOT SUPPORTED)'},
           'code': {'scorer_sha256': sha256(__file__), 'book_runtime_sha256': sha256(REPO / 'scripts' / 'book_runtime.py'),
                    'sessions_sha256': sha256(REPO / 'scripts' / 'sessions.py'), 'lightgbm': lgb.__version__},
           'books': []}
    Path(a.out).write_text(json.dumps(res, indent=1))
    cache = {}
    for e in books:
        bid, pair, tf = e['id'], e['currency'], e['timeframe']
        h = int(tf.rstrip('m'))
        gap = 60 * h
        out = {'book_id': bid, 'pair': pair, 'timeframe': tf}
        try:
            strategy = json.loads(br.load_registered_strategy_path(bid).read_text())
            boosters = [lgb.Booster(model_file=str(p)) for p in br._load_registered_model_paths(bid)]
            cols, cols_src = feature_names(strategy, boosters)
            if {b.num_feature() for b in boosters} != {len(cols)}:
                raise RuntimeError('model feature count differs from the column list')
            metrics = e.get('metrics') if isinstance(e.get('metrics'), dict) else {}
            cov, cov_src = br._resolve_coverage(bid, strategy, metrics)
            thr = br._conf_thr(strategy, cov)
        except Exception as exc:                       # noqa: BLE001  (recorded per book)
            out.update(status='LOAD_FAILED', reason=str(exc))
            res['books'].append(out)
            print(bid, 'LOAD_FAILED', exc, flush=True)
            continue
        if pair not in cache:
            cache[pair] = pd.read_parquet(Path(a.dataset) / f'{pair}.parquet')
        df = cache[pair]
        missing = [c for c in cols if c not in df.columns]
        bb_col, bb_thr = compression_gate(strategy, a.gate_source)
        out.update(n_models=len(boosters), n_features=len(cols), features_from=cols_src, coverage=cov,
                   coverage_source=cov_src, conf_thr=thr, compression_gate=[bb_col, bb_thr] if bb_col else None)
        intended, _ = compression_gate(strategy, 'strategy')
        if bb_col and intended != bb_col:
            out['gate_note'] = f'gate column {bb_col} ({a.gate_source}); the strategy JSON names {intended}'
        if missing:
            out.update(status='SCHEMA_UNAVAILABLE', missing=len(missing), missing_examples=missing[:20],
                       reason='columns not computable from Deriv mid ticks (EURUSD order-flow needs a volume feed)')
            res['books'].append(out)
            print(bid, 'SCHEMA_UNAVAILABLE', len(missing), missing[:4], flush=True)
            continue
        ts = df['decision_ts'].to_numpy(np.int64)
        fwd = df[f'ret_fwd_{h}m'].to_numpy(np.float64)
        valid = df[f'valid_{h}m'].to_numpy(bool) & np.isfinite(fwd)
        Xdf = df[cols].astype('float64')
        finite = np.isfinite(Xdf.to_numpy()).all(axis=1)
        if a.session == 'ny':
            sess = session_mask(ts, 'ny')
        else:
            hh = pd.to_datetime(ts, unit='s', utc=True).tz_convert('America/New_York').hour.values
            sess = (hh >= 13) & (hh < 22)
        if not a.include_blackout:
            sess = sess & tradeable_mask(ts, h)
        sess = sess & (ts >= lo_ts) & (ts < hi_ts)
        rows = np.where(sess & valid & finite)[0]
        rows = rows[np.argsort(ts[rows], kind='stable')]
        frame = pd.DataFrame(Xdf.to_numpy()[rows], columns=cols)
        p = np.mean([b.predict(frame) for b in boosters], axis=0)
        f = fwd[rows]
        t = ts[rows]
        conf = np.abs(p - 0.5)
        gate = ~(conf < thr)
        if bb_col:
            gate &= ~(df[bb_col].to_numpy(np.float64)[rows] > bb_thr)
        mv = f != 0
        out.update(status='SCORED', rows_session=int(sess.sum()), rows_scored=int(len(rows)),
                   rows_dropped_nonfinite=int((sess & valid & ~finite).sum()),
                   window=[pd.to_datetime(t[0], unit='s', utc=True).isoformat(), pd.to_datetime(t[-1], unit='s', utc=True).isoformat()],
                   auc_moved=round(float(roc_auc_score((f[mv] > 0).astype(int), p[mv])), 4),
                   moved_up_rate=round(float((f[mv] > 0).mean()), 4),
                   realized_gate_rate=round(float(gate.mean()), 4),
                   mean_p=round(float(p.mean()), 4), sides={})
        quarters = pd.to_datetime(t, unit='s', utc=True).tz_convert(None).to_period('Q').astype(str)
        for side in ('UP', 'DOWN', 'COMBINED'):
            tr, win = stream(p, f, t, gate, side, gap)
            lo, hi = boot(win)
            byq = {}
            for q in sorted(set(quarters[tr])) if len(tr) else []:
                w = win[quarters[tr] == q]
                byq[q] = [int(len(w)), round(float(w.mean()), 4)]
            verdict = 'SUPPORTED' if (len(win) >= 50 and np.isfinite(lo) and lo >= BE) else 'NOT SUPPORTED'
            out['sides'][side] = {'n': int(len(win)), 'win_rate': round(float(win.mean()), 4) if len(win) else None,
                                  'ci95': [round(lo, 4), round(hi, 4)], 'by_quarter': byq,
                                  'verdict': verdict if side != 'COMBINED' else 'context'}
        res['books'].append(out)
        s = out['sides']
        print(f'{bid}: rows {len(rows):,} AUC {out["auc_moved"]} gate {out["realized_gate_rate"]} | '
              f'UP {s["UP"]["n"]} {s["UP"]["win_rate"]} {s["UP"]["ci95"]} {s["UP"]["verdict"]} | '
              f'DOWN {s["DOWN"]["n"]} {s["DOWN"]["win_rate"]} {s["DOWN"]["ci95"]} {s["DOWN"]["verdict"]} | '
              f'COMB {s["COMBINED"]["n"]} {s["COMBINED"]["win_rate"]} ({time.time() - t0:.0f}s)', flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    res['seconds'] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print('done', res['seconds'], 's ->', a.out)


if __name__ == '__main__':
    main()
