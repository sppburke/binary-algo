#!/bin/bash
cd /media/sean/CORSAIR/binary-algo
for S in ny ldn asia; do
  echo "===== $(date +%H:%M:%S) START FT $S ====="
  ~/binary-algo-venv/bin/python kronos_ft.py $S 12 24 256 20 3000 > kronos_ft_$S.log 2>&1
  echo "===== $(date +%H:%M:%S) DONE FT $S (exit $?) ====="
done
echo "ALL FT DONE"
