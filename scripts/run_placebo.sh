#!/usr/bin/env bash
# Drift-neutral placebo analysis of the E1t model for one pair, as one recorded lab test:
# cpcv_deriv_x.py (E1t defaults, NY tradeable) with --dump-preds -> analyze_drift_placebo.py (reading pre-registered in
# its output before the dump is opened) -> results/json/deriv_a1placebo_<PAIR>_15m_result.json -> finish -> clean.
#     run_placebo.sh <PAIR> [<experiment id> [extra cpcv_deriv_x.py arguments ...]]   (default id a1placebo, E1t defaults)
set -u
LAB=/mnt/data/deriv-sota-lab
T=$LAB/tools/lab_test.sh
PY=/home/sean/binary-algo-venv/bin/python
pair=$1
exp=${2:-a1placebo}
shift; [[ $# -gt 0 ]] && shift
id="$exp-$pair-15m"
out=/home/sean/git/binary-algo/results/json/deriv_${exp}_${pair}_15m_result.json
$T start "$id" "$exp: cpcv_deriv_x.py $pair [$*] 15m NY with --dump-preds, then hour / hour x volatility matched always-side baselines per CPCV path" > /dev/null || exit 1
$T run "$id" 5400 14 -- bash -c "$PY -B $LAB/tools/cpcv_deriv_x.py --pair $pair --horizon 15 --tag $exp --out result.json --dump-preds dump.npz $* > x.log 2>&1 && $PY -B $LAB/tools/analyze_drift_placebo.py dump.npz $LAB/cache/dataset_dev_v0.1 $pair placebo.json --result result.json" > /dev/null
rc=$?
if [[ $rc -eq 0 && -s $LAB/tests/$id/placebo.json ]]; then
  cp "$LAB/tests/$id/placebo.json" "$out"
  note=$($PY -I -c '
import json, sys
o = json.load(open(sys.argv[1]))
print(" | ".join("%s %s (hour_vol excess mean %s, %s/%s paths > 0; hour %s; always %s; rebuild matches run %s)" % (
    s, v["verdict"], None if v["summary"]["excess_hour_vol"]["mean"] is None else round(v["summary"]["excess_hour_vol"]["mean"], 4),
    v["summary"]["excess_hour_vol"]["n_positive"], v["summary"]["excess_hour_vol"]["n_counted"],
    None if v["summary"]["excess_hour"]["mean"] is None else round(v["summary"]["excess_hour"]["mean"], 4),
    None if v["summary"]["excess_always"]["mean"] is None else round(v["summary"]["excess_always"]["mean"], 4),
    v["rebuild_matches_run"]) for s, v in o["sides"].items()))' "$out")
  $T finish "$id" done "$note" "$out"
  echo "$id: $note"
else
  $T finish "$id" failed "exit $rc; see run.log tail: $(tail -n 3 "$LAB/tests/$id/run.log" | tr '\n' ' ' | cut -c1-400)"
  echo "$id: FAILED exit $rc"
fi
$T clean "$id" > /dev/null
