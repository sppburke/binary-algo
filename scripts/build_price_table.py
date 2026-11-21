#!/usr/bin/env python3
"""Payout tests V-04 to V-09, shared prerequisite: the price table of every development decision.

    build_price_table.py <out dir> <dataset dir> [<dataset dir> ...] --pairs PAIR[,PAIR...]

For each pair: the union of the decision times of <dataset dir>/<PAIR>.parquet over the given development datasets
(every session: a training session can differ from the evaluation session), priced by the frozen quote model
(cache/venue_quote_model_v1) on both clocks (venue_quote_model.CLOCKS): the raw skew and overround outputs, the Rise
and Fall payouts (venue_quote_model.to_payouts of those outputs) and the priced flag (a tick at or before the
decision and the weekday x clock bucket offered). Refuses a dataset decision or a tick at or after the development
cutoff, and a decision whose 15-minute contract would expire at or after it. Checks that every row's payouts and
flag equal venue_quote_model.payouts (the call analyze_payout_aware.py makes) and that every dataset's decision
times are unique and all present in the table. Writes <out dir>/<PAIR>_15m.npz (decision_ts, and per clock c:
skew_c, over_c, pu_c, pd_c, priced_c) and <out dir>/manifest.json (this file's SHA-256, the quote model's files,
every dataset's manifest and parquet, every tick file, with SHA-256; counts per pair). The table is written into
<out dir>.partial and renamed to <out dir> only when every pair is done; an existing output or partial build is never
replaced.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

TOOLS = Path('/mnt/data/deriv-sota-lab/tools')
sys.path.insert(0, str(TOOLS))
import deriv_dataset as dd  # noqa: E402
import venue_quote_model as vqm  # noqa: E402
import venue_quote_study as vqs  # noqa: E402

VERSION = 'build_price_table/0.3'
MODEL_DIR = Path('/mnt/data/deriv-sota-lab/cache/venue_quote_model_v1')
FIRST_TICK_DAY = date(2025, 8, 11)                  # the archive's first development day (analyze_payout_aware's default)
HORIZON_S, TOL_S = dd.HORIZONS[15]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def need(ok: bool, why: str) -> None:
    if not ok:
        raise SystemExit(f'build_price_table: {why}')


def validate_raw_ticks(files, pair):
    """The archive reader divides integer timestamps by 1e9: accept its ns schema only.
    Check the Arrow values before that conversion can truncate or hide a cutoff tick.
    """
    for path in files:
        raw = pq.read_table(path, columns=['datetime_utc']).column(0)
        need(pa.types.is_timestamp(raw.type) and raw.type.unit == 'ns',
             f'{pair} {path}: archive datetime_utc must be timestamp[ns]')
        need(raw.null_count == 0, f'{pair} {path}: a raw tick timestamp is null')
        cutoff = pa.scalar(datetime.fromtimestamp(vqs.CUTOFF, timezone.utc), type=raw.type)
        need(len(raw) == 0 or pc.all(pc.less(raw, cutoff)).as_py(),
             f'{pair}: ticks outside development were loaded (raw timestamp at or after cutoff)')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('datasets', nargs='+')
    ap.add_argument('--pairs', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    part = out.with_name(out.name + '.partial')
    need(not out.exists() and not part.exists(), f'{out} or {part} exists: a table is never replaced')
    pairs = a.pairs.split(',')
    need(all(p in dd.PAIRS for p in pairs), f'--pairs {a.pairs}: unknown pair')
    need(vqs.LAST_TICK_DAY < datetime.fromtimestamp(vqs.CUTOFF, timezone.utc).date(), 'cutoff constants disagree')
    # Preflight every dataset and every tick vector, for ALL pairs, before model loading or output creation.
    # Keep originals for exact join checks: casting must never hide a non-finite or fractional decision.
    datasets, prepared = {}, {}
    for ds in a.datasets:
        dsp = Path(ds)
        need((dsp / 'manifest.json').exists(), f'{ds}: no manifest.json')
        datasets[ds] = {'manifest_sha256': sha(dsp / 'manifest.json'),
                        'parquet_sha256': {p: sha(dsp / f'{p}.parquet') for p in pairs}}
    for pair in pairs:
        per_ds = {}
        for ds in a.datasets:
            raw = pq.read_table(Path(ds) / f'{pair}.parquet', columns=['decision_ts']).column(0).to_numpy()
            need(raw.ndim == 1 and raw.dtype.kind in 'iuf' and np.isfinite(raw).all(),
                 f'{ds} {pair}: decision time is not finite')
            need(np.all(raw == np.floor(raw)), f'{ds} {pair}: decision time is not whole seconds')
            need(np.all(raw < vqs.CUTOFF), f'{pair}: a decision at or after the development cutoff')
            need(np.all(raw < vqs.CUTOFF - dd.LAG_S - HORIZON_S - TOL_S),
                 f'{pair}: a decision whose 15-minute contract would expire at or after the development cutoff')
            # Epoch seconds are passed to the quote model as float64: require exact round trips there as well.
            need(np.all(raw >= -(2**53)), f'{ds} {pair}: decision time is outside exact quote-model seconds')
            t = raw.astype(np.int64)
            need(np.array_equal(t, raw) and np.array_equal(t.astype(float), raw),
                 f'{ds} {pair}: decision identity changed in conversion')
            need(len(np.unique(raw)) == len(raw), f'{ds} {pair}: decision times are not unique')
            per_ds[ds] = raw.copy()
        dec = np.unique(np.concatenate(list(per_ds.values()))).astype(np.int64)
        need(len(dec) > 0, f'{pair}: no finite decision time')
        for ds, raw in per_ds.items():
            j = np.searchsorted(dec, raw)
            need(np.all(j < len(dec)) and np.array_equal(dec[np.minimum(j, len(dec)-1)], raw),
                 f'{ds} {pair}: an original decision is missing from the table')
        files = dd.day_files(pair, FIRST_TICK_DAY, vqs.LAST_TICK_DAY)
        validate_raw_ticks(files, pair)
        ts_k, px_k = dd.load_ticks(files)
        need(len(ts_k) > 0 and np.isfinite(ts_k).all() and ts_k.max() < vqs.CUTOFF,
             f'{pair}: ticks outside development were loaded')
        prepared[pair] = (per_ds, dec, files, ts_k, px_k)
    model = vqm.load(str(MODEL_DIR))
    manifest = {'tool': VERSION, 'tool_sha256': sha(Path(__file__)), 'built_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                time.gmtime()), 'cutoff_utc': datetime.fromtimestamp(vqs.CUTOFF, timezone.utc).isoformat(),
                'quote_model': {'dir': str(MODEL_DIR), 'tool': model['meta']['tool'], 'sha256': model['meta']['sha256'],
                                'meta_sha256': sha(MODEL_DIR / 'meta.json')},
                'clocks': list(vqm.CLOCKS), 'horizon_min': 15, 'datasets': datasets, 'pairs': {}}
    part.mkdir(parents=True)
    for pair in pairs:
        t0 = time.time()
        per_ds, dec, files, ts_k, px_k = prepared.pop(pair)
        arrays = {'decision_ts': dec}
        tf = dec.astype(float)
        for c in vqm.CLOCKS:
            X, ok, (wd, mod) = vqm.feature_frame(ts_k, px_k, tf, c)
            skew, over = model['skew'].predict(X), model['over'].predict(X)
            pu, pdn = vqm.to_payouts(skew, over)
            wk = np.asarray(wd) < 5
            off = np.zeros(len(wk), bool)
            off[wk] = model['offered'][vqm.bucket(np.asarray(wd)[wk], np.asarray(mod)[wk])]
            priced = np.asarray(ok, bool) & off
            ref_u, ref_d, ref_ok = vqm.payouts(model, ts_k, px_k, tf, c)
            need(np.array_equal(pu, ref_u, equal_nan=True) and np.array_equal(pdn, ref_d, equal_nan=True)
                 and np.array_equal(priced, np.asarray(ref_ok, bool)),
                 f'{pair} {c}: the table differs from venue_quote_model.payouts')
            arrays.update({f'skew_{c}': np.asarray(skew, np.float64), f'over_{c}': np.asarray(over, np.float64),
                           f'pu_{c}': np.asarray(pu, np.float64), f'pd_{c}': np.asarray(pdn, np.float64),
                           f'priced_{c}': priced})
        for ds, t in per_ds.items():
            j = np.searchsorted(arrays['decision_ts'], t)
            need(np.all(j < len(dec)) and np.array_equal(arrays['decision_ts'][np.minimum(j, len(dec)-1)], t),
                 f'{ds} {pair}: an original decision is missing from the table')
        np.savez_compressed(part / f'{pair}_15m.npz', **arrays)
        manifest['pairs'][pair] = {
            'rows': int(len(dec)), 'priced_ny': int(arrays['priced_ny'].sum()), 'priced_utc4': int(arrays['priced_utc4'].sum()),
            'first_decision_utc': datetime.fromtimestamp(int(dec[0]), timezone.utc).isoformat(),
            'last_decision_utc': datetime.fromtimestamp(int(dec[-1]), timezone.utc).isoformat(),
            'tick_files': {str(f): dd.file_sha(f) for f in files}, 'npz_sha256': sha(part / f'{pair}_15m.npz'),
            'seconds': round(time.time() - t0, 1)}
        print(f'{pair}: {len(dec)} decisions, priced NY {manifest["pairs"][pair]["priced_ny"]}, '
              f'UTC-4 {manifest["pairs"][pair]["priced_utc4"]}, {time.time() - t0:.0f} s', flush=True)
        del ts_k, px_k, arrays
    (part / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    os.replace(part, out)
    print('written', out)


if __name__ == '__main__':
    main()
