#!/bin/bash
cd /home/sean/git/binary-algo
PY=~/binary-algo-venv/bin/python
run(){ echo "===== $(date +%H:%M:%S) kronos_mtf $* ====="; $PY kronos_mtf.py "$@"; }
# NATIVE (2m new; re-run 5m/10m so they save per-bar npz for the ensemble)
run 2  all 2  256 20 3000 NeoQuasar/Kronos-small mtf_zs_2m_all
run 5  all 5  256 20 3000 NeoQuasar/Kronos-small mtf_zs_5m_all
run 10 all 10 256 20 3000 NeoQuasar/Kronos-small mtf_zs_10m_all
# FINE "up the chain": 1m context -> predict N steps
run 2  all 1  256 20 3000 NeoQuasar/Kronos-small mtf_zs_fine1_2m
run 5  all 1  256 20 3000 NeoQuasar/Kronos-small mtf_zs_fine1_5m
run 10 all 1  256 20 3000 NeoQuasar/Kronos-small mtf_zs_fine1_10m
run 10 all 5  256 20 3000 NeoQuasar/Kronos-small mtf_zs_fine5_10m
# MULTI-TF ENSEMBLES (the user's idea: combine TFs for one horizon)
echo "===== $(date +%H:%M:%S) ensembles ====="
$PY kronos_ensemble.py 5m_multi  mtf_zs_fine1_5m  mtf_zs_5m_all
$PY kronos_ensemble.py 10m_multi mtf_zs_fine1_10m mtf_zs_fine5_10m mtf_zs_10m_all
# BAR-IMAGE CNN direction per session (cell 1m-E)
echo "===== $(date +%H:%M:%S) bar-CNN per session ====="
for S in ny ldn asia; do echo "-- barcnn $S --"; $PY barcnn_run.py hist 30 120000 25 $S > barcnn_hist_${S}.log 2>&1; tail -1 barcnn_hist_${S}.log; done
echo "ALL KRONOS2+CNN DONE"
