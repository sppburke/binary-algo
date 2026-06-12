#!/bin/bash
cd /home/sean/git/binary-algo
for H in 5 2; do
  echo "===== $(date +%H:%M:%S) xpair $H ====="
  ~/binary-algo-venv/bin/python session_xpair.py $H > session_xpair_${H}.log 2>&1
done
echo "LOWCHAIN XPAIR DONE"
