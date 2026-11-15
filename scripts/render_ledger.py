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


def path_test(r, c, s, be=0.541):
    """The /goal's path test from the stored per-path data of any harness version: of the 15 purged paths, at least 12
    must have >= 25 trades AND a win rate >= .541, and the p10 of the counted paths must be >= .541.
    Returns (passes, counted, clear, p10)."""
    import numpy as np
    w = [p['bycov'][c][s][1] for p in r['paths'] if p['bycov'][c][s][0] >= 25]
    clear = sum(x >= be for x in w)
    p10 = float(np.percentile(w, 10)) if w else float('nan')
    return bool(len(r['paths']) == 15 and len(w) >= 12 and clear >= 12 and p10 >= be), len(w), clear, p10


def goal_verdict(r, c, s, be=0.541):
    """The /goal's certification reading: the path test on a VALID run; an invalid run (moved up-rate checks)
    certifies nothing here (see amended_verdict for the owner's drift amendment)."""
    ok, n, clear, _ = path_test(r, c, s, be)
    valid = r.get('validity', {}).get('VALID', r.get('validity', {}).get('in_session_up_rate_in_band', True) and
                                      r.get('validity', {}).get('paths_up_rate_in_tripwire', True))
    return f'{"CERTIFIED" if ok and valid else "no"} ({n}/{len(r["paths"])}, {clear} clear)'


# Owner decision 1 (docs/DERIV_SOTA_PROGRAM.md "Owner decisions", 2026-11-15): DRIFT AMENDMENT. A run INVALID through the
# moved up-rate checks (in-session band and per-path tripwire) is DEV-certified iff the path test holds, a drift placebo
# of exactly that configuration - same tag, pair, horizon and operating coverage, per the lab registry's commands -
# reads SKILL BEYOND DRIFT TIMING for the side, and the pair passes the mirage check. Final certification additionally
# needs the holdout drift-matched excess (hour x volatility) CI95 lower bound > 0 and the holdout conditions.
PLACEBO_OF = {('e1t', 'USDJPY', 15, '0.03'): 'a1placebo', ('e30nystroem', 'USDJPY', 15, '0.03'): 'a3placebo',
              ('c2nysabs', 'USDJPY', 15, '0.03'): 'a6placebo', ('e49nys30', 'USDJPY', 30, '0.03'): 'a8placebo'}
MIRAGE_OK = {   # (pair, horizon) -> mirage-check evidence (dataset_dev_v0.1, NY in-session tradeable, valid labels)
    ('USDJPY', 15): '15m tie share 0.0119 (1,542 / 129,284), the other six pairs 0.0109-0.0182; tick-settled labels, no forward fill',
    ('USDJPY', 30): '30m tie share 0.0081 (1,016 / 125,528), the other six pairs 0.0073-0.0131; tick-settled labels, no forward fill',
}


def amended_verdict(r, c, s, pair, results_dir=DEFAULT):
    """'' unless the run is DEV-certified under the drift amendment; then a short status naming the placebo."""
    if r.get('validity', {}).get('VALID', True) or (pair, int(r['horizon_min'])) not in MIRAGE_OK:
        return ''
    pl = PLACEBO_OF.get((r.get('tag'), pair, int(r['horizon_min']), c))
    if pl is None or not path_test(r, c, s)[0]:
        return ''
    pf = Path(results_dir) / f'deriv_{pl}_{pair}_{int(r["horizon_min"])}m_result.json'
    if not pf.exists():
        return ''
    pj = json.loads(pf.read_text())
    if str(pj.get('cov')) != c or int(pj.get('horizon_min', -1)) != int(r['horizon_min']):
        return ''
    return f'DEV-CERT (drift amendment, placebo `{pl}`)' if pj['sides'][s]['verdict'] == 'SKILL BEYOND DRIFT TIMING' else ''


def effective(original, amended):
    """One status string that names the rule behind each part."""
    return original if not amended else f'original rule: {original}; effective (owner decision 1): {amended}'


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
                   'DOWN fwd | always UP / DOWN | repo-rule verdict UP / DOWN | GOAL rule UP / DOWN (paths counted) | '
                   'frac @.565 UP / DOWN | result JSON |')
        out.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for f, r in runs:
            c = str(r['operating_cov'])
            b = r['bycov'][c]
            v, sm, fw = b['verdict'], b['summary'], b['forward']
            def frac565(s):                         # payout sensitivity: one recorded venue trade paid 1.77 (BE 1/1.77)
                w = [p['bycov'][c][s][1] for p in r['paths'] if p['bycov'][c][s][0] >= 25]
                return f'{sum(x >= 1 / 1.77 for x in w) / len(w):.2f}' if w else '—'

            def goal(s):                            # goal-faithful: >= 12 of the 15 paths counted AND clearing .541, p10 >= .541
                return effective(goal_verdict(r, c, s), amended_verdict(r, c, s, a.pair, R))
            model = f'{r.get("tag", "")}: {r["feature_set"]} ({r["n_features"]}), nseed {r["nseed"]}'
            if 'pool' in r:
                model += f', pooled +{len(r["pool"])} pairs'
            side = lambda s: f'{fmt(v[s]["p10"])} · {fmt(v[s]["frac_clear_BE"], 3)} · {fmt(sm[s].get("mean"))}'
            fwds = lambda s: f'{fw[s]["n_clear_BE"]}/{fw[s]["n_counted"]} · {fmt(fw[s]["mean"])}'
            out.append(f'| {r.get("tag", "")} | {model} | {r["horizon_min"]}m | {c} | {fmt(r["auc_summary"]["mean"])} '
                       f'({fmt(r["auc_summary"]["min"])}) | {side("UP")} | {fwds("UP")} | {side("DOWN")} | {fwds("DOWN")} | '
                       f'{fmt(r["baseline_always"]["UP"].get("mean"))} / {fmt(r["baseline_always"]["DOWN"].get("mean"))} | '
                       f'{r["verdict"]["UP"]} / {r["verdict"]["DOWN"]} | {goal("UP")} / {goal("DOWN")} | '
                       f'{frac565("UP")} / {frac565("DOWN")} | `{f.name}` |')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
