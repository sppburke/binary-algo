#!/bin/bash
cd /home/sean/git/binary-algo
PY=~/binary-algo-venv/bin/python
$PY kronos_bars.py 2 test,oos       # native 2m  -> EURUSD_2m
$PY kronos_bars.py 2 test,oos 1     # fine 1m->2m -> EURUSD_g1_h2
$PY kronos_bars.py 5 test,oos 1     # fine 1m->5m -> EURUSD_g1_h5
$PY kronos_bars.py 10 test,oos 1    # fine 1m->10m -> EURUSD_g1_h10
$PY kronos_bars.py 10 test,oos 5    # fine 5m->10m -> EURUSD_g5_h10
echo "KRONOS CACHES BUILT"
