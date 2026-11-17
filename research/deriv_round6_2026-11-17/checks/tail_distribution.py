#!/usr/bin/env python3
"""Round 6: distribution of confident-tail win rates over every stored run, and the trade counts behind them.

From the repository root:

    python3 -I research/deriv_round6_2026-11-17/checks/tail_distribution.py results/json

A "side" is one (result file, coverage, UP or DOWN). A path is counted when it holds at least 25 trades. Sides with at
least 12 counted paths are summarized by the mean of their counted paths' win rates. Reads result JSONs only, so the
output depends on the result files present when it runs; the first line printed counts them. The summaries describe
the outcomes of the search at one horizon (15 minutes); 30-minute results are counted separately and never mixed in.
They are not a performance estimate for any key: the record of truth for a key is results/<PAIR>_RESULTS.md.
"""
import json
import statistics as st
import sys
from pathlib import Path

J = Path(sys.argv[1])
rows, files, valid_files = [], 0, 0
for f in sorted(J.glob('deriv_*_result.json')):
    try:
        r = json.loads(f.read_text())
    except Exception:
        continue
    if not isinstance(r, dict) or not isinstance(r.get('paths'), list) or not r['paths'] or 'bycov' not in r['paths'][0]:
        continue
    files += 1
    valid = bool((r.get('validity') or {}).get('VALID'))
    valid_files += valid
    pair = r.get('pair') or f.name.split('_')[2]
    hor = f.name.rsplit('_', 2)[1]                      # '15m' or '30m'
    for cov in r['paths'][0]['bycov']:
        for side in ('UP', 'DOWN'):
            cells = [p['bycov'][cov][side] for p in r['paths'] if cov in p.get('bycov', {}) and side in p['bycov'][cov]]
            counted = [(n, w) for n, w in cells if n is not None and w is not None and n >= 25]
            if len(counted) >= 12:
                rows.append({'file': f.name, 'pair': pair, 'hor': hor, 'valid': valid, 'cov': cov, 'side': side, 'counted': len(counted),
                             'op': str(cov) == str(r.get('operating_cov')),
                             'mean': st.mean(w for _, w in counted), 'med_n': st.median(n for n, _ in counted),
                             'clear': sum(w >= 0.541 for _, w in counted), 'wr': [w for _, w in counted],
                             'spread': max(n for n, _ in counted) / min(n for n, _ in counted)})


def q(xs, p):
    xs = sorted(xs)
    i = p * (len(xs) - 1)
    lo = int(i)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (i - lo)


print(f'result files with paths: {files} (VALID {valid_files})')
print('sides with >= 12 counted paths by horizon:', dict(sorted(__import__('collections').Counter(x['hor'] for x in rows).items())))
rows30 = [x for x in rows if x['hor'] != '15m']
rows = [x for x in rows if x['hor'] == '15m']
print('everything below: 15-minute results only, read from the result JSONs; the record of truth for a key is results/<PAIR>_RESULTS.md')
for name, sel in (('VALID, not USDJPY, operating coverage only', [x for x in rows if x['valid'] and x['pair'] != 'USDJPY' and x['op']]),
                  ('VALID, not USDJPY, every coverage', [x for x in rows if x['valid'] and x['pair'] != 'USDJPY']),
                  ('[USDJPY·15m] (results/USDJPY_RESULTS.md), valid flag ignored (the stored flag predates the drift amendment), operating coverage only',
                   [x for x in rows if x['pair'] == 'USDJPY' and x['op']])):
    m = [x['mean'] for x in sel]
    n = [x['med_n'] for x in sel]
    print(f'\n{name}: {len(sel)} sides with >= 12 counted paths')
    print('  mean path win rate: median %.4f p75 %.4f p90 %.4f p95 %.4f p99 %.4f max %.4f' %
          (q(m, .5), q(m, .75), q(m, .9), q(m, .95), q(m, .99), max(m)))
    print('  share with mean >= 0.541: %.1f%% | >= 0.55: %.1f%% | >= 0.56: %.1f%% | >= 0.58: %.1f%%' %
          tuple(100 * sum(v >= t for v in m) / len(m) for t in (0.541, 0.55, 0.56, 0.58)))
    print('  median trades per counted path: median %.0f p10 %.0f p90 %.0f' % (q(n, .5), q(n, .1), q(n, .9)))
    print('  counted paths per side:', dict(sorted(__import__('collections').Counter(x['counted'] for x in sel).items())))
    print('  largest / smallest trade count among a side\'s counted paths: median %.2f p90 %.2f' %
          (q([x['spread'] for x in sel], .5), q([x['spread'] for x in sel], .9)))
    print('  sides with >= 12 paths clear: %d' % sum(x['clear'] >= 12 for x in sel), '| by key:',
          dict(sorted(__import__('collections').Counter(f"[{x['pair']}·{x['hor']}] {x['side']}" for x in sel if x['clear'] >= 12).items())))
    full = [x for x in sel if x['clear'] >= 12 and q(x['wr'], .10) >= 0.541]      # q(., .10) = numpy.percentile(., 10)
    print('  sides meeting the goal path test (>= 12 clear and 10th percentile >= 0.541): %d' % len(full), '| by key:',
          dict(sorted(__import__('collections').Counter(f"[{x['pair']}·{x['hor']}] {x['side']}" for x in full).items())))
top = sorted((x for x in rows if x['valid'] and x['pair'] != 'USDJPY'), key=lambda x: -x['mean'])[:8]
print('\ntop VALID non-USDJPY sides by mean path win rate (each line read from its result JSON; the record of truth for a '
      'key is results/<PAIR>_RESULTS.md):')
for x in top:
    print(f"  [{x['pair']}·{x['hor']}] {x['file'][6:-12]:34s} cov {x['cov']} {x['side']:4s} mean {x['mean']:.4f} "
          f"counted {x['counted']} clear {x['clear']} trades/path {x['med_n']:.0f}")
print('\nthe win rate the goal rule needs at a given number of trades per path, when all 15 paths are counted: '
      'pass_probability.py, beside this file')
