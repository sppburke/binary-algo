#!/bin/bash
cd /media/sean/CORSAIR/binary-algo
for H in 5 10 15 30; do
  echo "===== build EURUSD_${H}m test,oos ====="
  ~/binary-algo-venv/bin/python kronos_bars.py $H test,oos
done
echo "ALL CACHES BUILT"
