#!/bin/bash
cd /home/sean/git/binary-algo
PY=~/binary-algo-venv/bin/python
for H in 5 10 15 30 60; do
  echo "############ HORIZON ${H}m ############"
  $PY exp_horizon.py $H
done
echo "############ HORIZON SWEEP DONE ############"
