#!/bin/bash
# V3 signal-attribution ablations. Each isolates one signal block.
cd /home/sean/git/binary-algo
PY=~/binary-algo-venv/bin/python
echo "waiting for base feature gen..."
until [ -f features/USDJPY_2026.parquet ]; do sleep 8; done
echo "base ready."
echo "######## ABLATION A: base only ########"
$PY exp_v3.py --no-cross --no-of
echo "######## ABLATION B: base + cross-pair ########"
$PY exp_v3.py --no-of
echo "######## ABLATION C: base + cross + order-flow (FULL) ########"
$PY exp_v3.py
echo "waiting for order-flow gen (peer-OF)..."
until [ -f features_of/USDJPY_2026.parquet ]; do sleep 8; done
echo "######## ABLATION D: + peer order-flow ########"
$PY exp_v3.py --peer-of
echo "######## ALL ABLATIONS DONE ########"
