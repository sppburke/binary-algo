#!/usr/bin/env bash
# Sign-null replay of one CPCV configuration (literature: full-search null replay / reality check), as one recorded
# lab test: the reading is written first, then cpcv_deriv_x.py runs with --null-shift-weeks W for W = 3, 6, ..., 48
# (labels replaced by the label W weeks later: drift and intraday structure kept, feature-label link broken) and the
# real configuration's result is compared with the null distribution -> results/json/deriv_<exp>_<PAIR>_15m_result.json.
#     run_nullreplay.sh <PAIR> <experiment id> <real result JSON> <side> [extra cpcv_deriv_x.py arguments ...]
set -u
LAB=/mnt/data/deriv-sota-lab
T=$LAB/tools/lab_test.sh
PY=/home/sean/binary-algo-venv/bin/python
pair=$1; exp=$2; real=$3; side=$4; shift 4
id="$exp-$pair-15m"
out=/home/sean/git/binary-algo/results/json/deriv_${exp}_${pair}_15m_result.json
$T start "$id" "$exp: sign-null replay x16 (labels shifted 3..48 weeks) of cpcv_deriv_x.py $pair [$*], side $side, vs $(basename $real)" > /dev/null || exit 1
dir=$LAB/tests/$id
$PY -I - "$dir/reading.json" "$real" "$side" "$*" <<'EOF'
import json, sys, time
json.dump({'analysis': 'run_nullreplay/0.1', 'registered_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'real_result': sys.argv[2], 'side': sys.argv[3], 'harness_args': sys.argv[4], 'null_shifts_weeks': list(range(3, 49, 3)),
           'pre_registered_reading': 'SIGNIFICANT iff the real p10 at the operating coverage exceeds every null p10 except at '
                                     'most one (rank >= 16 of 17) AND at most 1 of the 16 nulls meets the goal-rule path test '
                                     '(>= 12 of 15 counted paths clearing 0.541 with p10 >= 0.541)'},
          open(sys.argv[1], 'w'), indent=1)
EOF
$T run "$id" 7200 14 -- bash -c "for w in \$(seq 3 3 48); do $PY -B $LAB/tools/cpcv_deriv_x.py --pair $pair --horizon 15 --tag $exp-w\$w --no-forward --null-shift-weeks \$w --out null_w\$w.json $* > null_w\$w.log 2>&1 || echo FAIL \$w; done; echo done" > /dev/null
rc=$?
note=$($PY -I - "$dir" "$real" "$side" "$out" <<'EOF'
import glob, json, sys
import numpy as np
d, real, side, out = sys.argv[1:5]
reading = json.load(open(f'{d}/reading.json'))
def stat(r):
    c = str(r['operating_cov'])
    w = [p['bycov'][c][side][1] for p in r['paths'] if p['bycov'][c][side][0] >= 25]
    p10 = float(np.percentile(w, 10)) if w else float('nan')
    return {'p10': p10, 'paths_counted': len(w), 'paths_clear': int(sum(x >= 0.541 for x in w)),
            'goal_path_test': bool(len(w) >= 12 and sum(x >= 0.541 for x in w) >= 12 and p10 >= 0.541),
            'auc_mean': r['auc_summary']['mean'], 'up_rate': r['in_session_moved_up_rate']}
R = stat(json.load(open(real)))
nulls = {}
for f in sorted(glob.glob(f'{d}/null_w*.json')):
    r = json.load(open(f))
    if 'paths' in r and len(r['paths']) == 15:
        nulls[f.split('null_w')[1].split('.json')[0]] = stat(r)
p10s = [v['p10'] for v in nulls.values() if v['p10'] == v['p10']]
rank = 1 + sum(R['p10'] > x for x in p10s)
n_goal = sum(v['goal_path_test'] for v in nulls.values())
sig = len(nulls) == 16 and rank >= 16 and n_goal <= 1
reading.update(real=R, nulls=nulls, n_nulls=len(nulls), real_p10_rank_of=f'{rank} of {len(p10s) + 1}',
               nulls_meeting_goal_path_test=n_goal, null_p10_quantiles={q: float(np.percentile(p10s, q)) for q in (50, 90, 95)} if p10s else None,
               verdict='SIGNIFICANT vs sign-null replay' if sig else 'NOT SIGNIFICANT vs sign-null replay (or incomplete)')
json.dump(reading, open(out, 'w'), indent=1)
print(f"{reading['verdict']}: real p10 {R['p10']:.4f} ({R['paths_clear']}/{R['paths_counted']} clear) rank {rank}/{len(p10s) + 1}; "
      f"null p10 median {np.median(p10s):.4f} max {max(p10s):.4f}; nulls meeting the goal path test {n_goal}/{len(nulls)}")
EOF
)
if [[ -s $out ]]; then $T finish "$id" done "$note" "$out"; else $T finish "$id" failed "exit $rc; $note"; fi
echo "$id: $note"
$T clean "$id" > /dev/null
