#!/usr/bin/env python3
"""Round 6: the table of configurations already run, without any result number, and its tallies by family.

From the repository root, the tallies of the committed table (this reproduces tested_configs.log):

    python3 -I research/deriv_round6_2026-11-17/checks/tested_configs.py research/deriv_round6_2026-11-17/tested_configs.tsv

The table itself was written in the lab with a second argument:

    python3 -I tested_configs.py <experiments_inventory.md> <out.tsv>

where the first argument is the inventory the round-6 lenses read (one block per experiment tag, written from
results/json on 2026-11-17 at 02:30 UTC; it carries result numbers and stays in the lab). One line per tag with its
configuration: horizon, session, operating coverage, seeds, feature set, dataset, harness, number of pairs run, and the
knobs that differ from the default recipe. The default recipe is LightGBM with 255 leaves, min_child 400, learning rate
0.02, 800 trees, colsample 0.5, lambda 20, 15% random validation rows, the settlement-sign label, no gate and no weights.
The model of a tag is its `model` knob, except for the three harnesses that have none: cpcv_rule (a fixed rule),
cpcv_deriv_online (river's adaptive random forest, learned prequentially) and cpcv_deriv_pool (the default LightGBM
fitted on the rows of several pairs).
"""
import collections
import csv
import json
import re
import sys

src = sys.argv[1]
cols = ('tag', 'horizon_min', 'session', 'operating_cov', 'nseed', 'features', 'n_features', 'dataset', 'harness', 'pairs_run')
rows = []
if len(sys.argv) == 2:                                  # the committed table
    for r in csv.DictReader(open(src), delimiter='\t'):
        rows.append(dict({c: r[c] for c in cols}, knobs=json.loads(r['knobs_not_default']), extra=json.loads(r['extra'])))
else:                                                   # the lab inventory -> table
    head = re.compile(r'^## (\S+) \| h=(\S+) sess=(\S+) cov=(\S+) nseed=(\S+) feat=(\S+) nf=(\S+) ds=(\S+) harness=(\S+) sec_med=(\S+) n_json=(\S+)')
    knob = re.compile(r'^\s+knobs!=default: (\{.*\}) extra: (\{.*\})\s*$')
    cur = None
    for ln in open(src):
        m = head.match(ln)
        if m:
            cur = dict(zip(('tag', 'horizon_min', 'session', 'operating_cov', 'nseed', 'features', 'n_features', 'dataset',
                            'harness', 'sec_med', 'pairs_run'), m.groups()))
            continue
        m = knob.match(ln)
        if m and cur is not None:
            cur['knobs'], cur['extra'] = json.loads(m.group(1)), json.loads(m.group(2))
            rows.append(cur)
            cur = None
    with open(sys.argv[2], 'w') as fh:
        fh.write('\t'.join(cols + ('knobs_not_default', 'extra')) + '\n')
        for r in rows:
            fh.write('\t'.join(str(r[c]) for c in cols) + '\t' + json.dumps(r['knobs'], sort_keys=True) + '\t'
                     + json.dumps(r['extra'], sort_keys=True) + '\n')


def model(r):
    if r['harness'].startswith('cpcv_rule'):
        return 'rule (no model)'
    if r['harness'].startswith('cpcv_deriv_online'):
        return 'online adaptive random forest (river ARF)'
    if r['harness'].startswith('cpcv_deriv_pool'):
        return 'lgb (default), fitted on several pairs'
    return r['knobs'].get('model', 'lgb (default)')


C = collections.Counter
print(f'{len(rows)} tags, {sum(int(r["pairs_run"]) for r in rows)} results')
print('model:', dict(C(model(r) for r in rows).most_common()))
print('harness:', dict(C(r['harness'].split('/')[0] for r in rows).most_common()))
print('horizon:', dict(C(r['horizon_min'] for r in rows)), '| session:', dict(C(r['session'] for r in rows).most_common()))
print('seeds:', dict(C(r['nseed'] for r in rows).most_common()))
print('feature set:', dict(C(re.sub(r'^list:.*', 'list:<file>', r['features']) for r in rows).most_common()))
print('feature families added (--families):', dict(C(r['knobs']['families'] for r in rows if 'families' in r['knobs']).most_common()))
for name, key in (('compression gate', 'comp_gate'), ('magnitude gate', 'mag_keep'), ('meta-label gate', 'meta_keep'),
                  ('adaptive-conformal gate', 'aci_gamma'), ('bootstrap gate', 'boot_gate'), ('rule gate', 'rule_gate'),
                  ('training label', 'train_label'), ('weights', 'weight'), ('fit session', 'train_session'),
                  ('validation mode', 'val_mode'), ('drift fixed effect', 'drift_fe'), ('reflection', 'reflect'),
                  ('specialist', 'specialist'), ('signed-only inputs', 'signed_only'), ('tail auxiliary loss', 'tail_aux'),
                  ('multi-horizon rows', 'multi_horizon'), ('monotone constraints', 'monotone'), ('blend', 'blend_absret')):
    hit = [r for r in rows if key in r['knobs']]
    print(f'{name}: {len(hit)} tags', dict(C(json.dumps(r['knobs'][key]) for r in hit)))
hp = [r['tag'] for r in rows if set(r['knobs']) & {'num_leaves', 'min_child', 'colsample', 'n_estimators', 'nys_components',
                                                   'nys_gamma_mult'}]
print('hyperparameter one-offs:', len(hp), hp)
