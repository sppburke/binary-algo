#!/bin/bash
cd /home/sean/git/binary-algo
until grep -q "ALL ABLATIONS DONE" /tmp/v3_ablations.log 2>/dev/null; do sleep 10; done
echo "ablations done; launching V4 ensemble..."
~/binary-algo-venv/bin/python exp_v4.py 4
echo "V4 DONE"
