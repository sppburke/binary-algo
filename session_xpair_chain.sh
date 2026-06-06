#!/bin/bash
cd /media/sean/CORSAIR/binary-algo
# wait for HOR=10 to finish all 3 sessions
while [ ! -f session_xpair_10m_asia_result.json ]; do sleep 30; done
sleep 5
for H in 15 30; do
  echo "===== $(date +%H:%M:%S) START xpair $H ====="
  ~/binary-algo-venv/bin/python session_xpair.py $H > session_xpair_$H.log 2>&1
  echo "===== $(date +%H:%M:%S) DONE xpair $H ====="
done
echo "ALL XPAIR DONE"
