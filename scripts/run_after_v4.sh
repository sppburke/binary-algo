#!/bin/bash
cd /home/sean/git/binary-algo
PY=~/binary-algo-venv/bin/python
until grep -q "V4 DONE" /tmp/v4.log 2>/dev/null; do sleep 15; done
echo "### V4 done -> running V5 (event specialist) ###"
$PY exp_v5.py > /tmp/v5.log 2>&1
echo "### V5 done -> running per-pair verification ###"
$PY exp_allpairs.py > /tmp/allpairs.log 2>&1
echo "### ALL POST-V4 DONE ###"
