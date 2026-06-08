#!/bin/bash
# Serial execution of the remaining 5m must-run rows + tight-cov UP refit. ONE heavy job at a time (OOM safety).
cd /home/sean/git/binary-algo
PY=~/binary-algo-venv/bin/python
echo "=== QUEUE START $(date +%H:%M:%S) ==="
echo "### B3a CKS-OFI @300s ###"; $PY m5_cksofi300.py            > q_b3a.log 2>&1; echo "B3a exit=$?"
echo "### N12/N13 irreversibility ###"; $PY m5_irrev.py          > q_irrev.log 2>&1; echo "irrev exit=$?"
echo "### D3 USD-strength DOWN ###"; MX_HOR=5 $PY m5_downcond.py  > q_d3.log 2>&1; echo "D3 exit=$?"
echo "### B5a per-side raw flow @300s ###"; MX_HOR=5 $PY m5_perside_flow.py > q_b5a.log 2>&1; echo "B5a exit=$?"
echo "### tight-cov UP refit ###"; $PY m5_refit_tightcov.py      > q_tightcov.log 2>&1; echo "tightcov exit=$?"
echo "=== QUEUE DONE $(date +%H:%M:%S) ==="
