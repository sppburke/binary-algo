#!/bin/bash
# Fast tail of GPU lane 2 after killing the slow fine1_10m: small-N fine runs + multi-TF ensembles + per-session bar-CNN.
cd /home/sean/git/binary-algo
PY=~/binary-algo-venv/bin/python
echo "== $(date +%H:%M:%S) fine1_10m (fast N=800) =="; $PY kronos_mtf.py 10 all 1 256 20 800 NeoQuasar/Kronos-small mtf_zs_fine1_10m
echo "== $(date +%H:%M:%S) fine5_10m =="; $PY kronos_mtf.py 10 all 5 256 20 3000 NeoQuasar/Kronos-small mtf_zs_fine5_10m
echo "== $(date +%H:%M:%S) ensembles =="
$PY kronos_ensemble.py 5m_multi  mtf_zs_fine1_5m  mtf_zs_5m_all
$PY kronos_ensemble.py 10m_multi mtf_zs_fine1_10m mtf_zs_fine5_10m mtf_zs_10m_all
echo "== $(date +%H:%M:%S) bar-CNN per session =="
for S in ny ldn asia; do echo "-- $S --"; $PY barcnn_run.py hist 30 120000 25 $S > barcnn_hist_${S}.log 2>&1; tail -1 barcnn_hist_${S}.log; done
echo "TAIL DONE"
