#!/usr/bin/env bash
# Entry-delay stress + staleness audit of the E1t model for one pair, as one recorded lab test:
# cpcv_deriv_x.py (E1t defaults, NY tradeable) with --dump-preds -> analyze_entry_delay.py (reading pre-registered in its
# output before the dump is opened; ticks selected as the DEV build did, DERIV_ALLOW_INCOMPLETE=1, so d = 0 reproduces
# the dataset labels) -> results/json/deriv_a2delay_<PAIR>_15m_result.json -> finish -> clean.
#     run_entrydelay.sh <PAIR>
set -u
LAB=/mnt/data/deriv-sota-lab
T=$LAB/tools/lab_test.sh
PY=/home/sean/binary-algo-venv/bin/python
pair=$1
id="a2delay-$pair-15m"
out=/home/sean/git/binary-algo/results/json/deriv_a2delay_${pair}_15m_result.json
$T start "$id" "a2delay: E1t $pair 15m NY with --dump-preds, frozen trades re-settled with +0/2/4/9/29 s extra entry delay; staleness terciles" > /dev/null || exit 1
$T run "$id" 5400 14 -- bash -c "$PY -B $LAB/tools/cpcv_deriv_x.py --pair $pair --horizon 15 --tag a2delay --out result.json --dump-preds dump.npz > x.log 2>&1 && DERIV_ALLOW_INCOMPLETE=1 $PY -B $LAB/tools/analyze_entry_delay.py dump.npz $pair delay.json" > /dev/null
rc=$?
if [[ $rc -eq 0 && -s $LAB/tests/$id/delay.json ]]; then
  cp "$LAB/tests/$id/delay.json" "$out"
  note=$($PY -I -c '
import json, sys
o = json.load(open(sys.argv[1]))
parts = []
for s, v in o["sides"].items():
    sm = v["summary_by_delay"]
    parts.append("%s %s / %s; clear-of-counted by delay %s; d0 reproduces labels %s" % (s, v["verdict_delay"], v["verdict_staleness"],
        {k: "%s/%s p10 %s" % (x["paths_clear"], x["paths_counted"], None if x["p10"] is None else round(x["p10"], 3)) for k, x in sm.items()},
        v["baseline_labels_reproduced_at_d0"]))
print(" | ".join(parts))' "$out")
  $T finish "$id" done "$note" "$out"
  echo "$id: $note"
else
  $T finish "$id" failed "exit $rc; see run.log tail: $(tail -n 3 "$LAB/tests/$id/run.log" | tr '\n' ' ' | cut -c1-400)"
  echo "$id: FAILED exit $rc"
fi
$T clean "$id" > /dev/null
