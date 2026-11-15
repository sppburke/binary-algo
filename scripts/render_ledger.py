#!/usr/bin/env python3
"""Render the Deriv-feed program's per-pair ledger tables straight from the result JSONs (no hand copying).

    render_ledger.py <PAIR> [--results DIR]

Reads results/json/deriv_r1_frozen_books_*_result.json (frozen books, as deployed) and every
results/json/deriv_<exp>_<PAIR>_<H>m_result.json (CPCV runs) and prints two Markdown tables for the pair.
"""
import argparse
import json
from pathlib import Path

DEFAULT = Path('/home/sean/git/binary-algo/results/json')


def fmt(x, nd=4):
    return '—' if x is None else (f'{x:.{nd}f}'.lstrip('0') if isinstance(x, float) and 0 <= x < 1 else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pair')
    ap.add_argument('--results', default=str(DEFAULT))
    a = ap.parse_args()
    R = Path(a.results)
    out = []
    for f in sorted(R.glob('deriv_r1*_frozen_books_*_result.json')):
        r = json.loads(f.read_text())
        rows = [b for b in r['books'] if b['pair'] == a.pair]
        if not rows:
            continue
        clock = 'training clock (true UTC - 5 h)' if 'shift-5' in r['dataset']['dir'] else 'true clock'
        trd = 'tradeable decisions only (Deriv purchase blackout excluded)' if r.get('tradeable_only') else 'ALL decisions incl. the untradeable blackout'
        out.append(f'**Frozen books on the Deriv feed — {clock}, {r["session"].split(" on the")[0]}, {trd}** (`{f.name}`; '
                   f'strategy gate, tick-settled, ties lose; SUPPORTED iff n >= 50 and CI95-lo >= .541):\n')
        out.append('| Book | AUC (moved) | gate rate | UP n · WR [CI95] | DOWN n · WR [CI95] | COMB n · WR | verdict UP / DOWN |')
        out.append('|---|---|---|---|---|---|---|')
        for b in rows:
            if b['status'] != 'SCORED':
                out.append(f'| `{b["book_id"]}` | — | — | — | — | — | {b["status"]}: {b.get("reason", "")} |')
                continue
            s = b['sides']
            cell = lambda x: f'{x["n"]} · {fmt(x["win_rate"])} [{fmt(x["ci95"][0])}, {fmt(x["ci95"][1])}]'
            out.append(f'| `{b["book_id"]}` | {fmt(b["auc_moved"])} | {fmt(b["realized_gate_rate"])} | {cell(s["UP"])} | '
                       f'{cell(s["DOWN"])} | {s["COMBINED"]["n"]} · {fmt(s["COMBINED"]["win_rate"])} | '
                       f'{s["UP"]["verdict"]} / {s["DOWN"]["verdict"]} |')
        out.append('')
    runs = []
    for f in sorted(R.glob(f'deriv_*_{a.pair}_*m_result.json')):
        r = json.loads(f.read_text())
        if 'bycov' not in r:
            continue
        runs.append((f, r))
    if runs:
        out.append('**Per-fold-refit CPCV on the Deriv feed** (DEV 2025-08-11..2026-07-31, NY DST session, 15 paths; '
                   'p10/frac = path win-rate 10th percentile / share of paths >= .541; fwd = forward-walk paths clearing .541 '
                   'and their mean):\n')
        out.append('| Run | Model / features | H | cov | AUC mean (min) | UP p10 · frac · mean | UP fwd | DOWN p10 · frac · mean | '
                   'DOWN fwd | always UP / DOWN | verdict UP / DOWN | frac @.565 UP / DOWN | result JSON |')
        out.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for f, r in runs:
            c = str(r['operating_cov'])
            b = r['bycov'][c]
            v, sm, fw = b['verdict'], b['summary'], b['forward']
            def frac565(s):                         # payout sensitivity: one recorded venue trade paid 1.77 (BE 1/1.77)
                w = [p['bycov'][c][s][1] for p in r['paths'] if p['bycov'][c][s][0] >= 25]
                return f'{sum(x >= 1 / 1.77 for x in w) / len(w):.2f}' if w else '—'
            model = f'{r.get("tag", "")}: {r["feature_set"]} ({r["n_features"]}), nseed {r["nseed"]}'
            if 'pool' in r:
                model += f', pooled +{len(r["pool"])} pairs'
            side = lambda s: f'{fmt(v[s]["p10"])} · {fmt(v[s]["frac_clear_BE"], 3)} · {fmt(sm[s].get("mean"))}'
            fwds = lambda s: f'{fw[s]["n_clear_BE"]}/{fw[s]["n_counted"]} · {fmt(fw[s]["mean"])}'
            out.append(f'| {r.get("tag", "")} | {model} | {r["horizon_min"]}m | {c} | {fmt(r["auc_summary"]["mean"])} '
                       f'({fmt(r["auc_summary"]["min"])}) | {side("UP")} | {fwds("UP")} | {side("DOWN")} | {fwds("DOWN")} | '
                       f'{fmt(r["baseline_always"]["UP"].get("mean"))} / {fmt(r["baseline_always"]["DOWN"].get("mean"))} | '
                       f'{r["verdict"]["UP"]} / {r["verdict"]["DOWN"]} | {frac565("UP")} / {frac565("DOWN")} | `{f.name}` |')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
