#!/bin/bash
# Corrected Kronos direction eval sweep (kronos_mtf.py). Waits for the FT chain (GPU free) + caches, then runs
# the prioritized run plan: P0 corrected-1m-all -> P1 strict per-session zero-shot -> P2 re-eval the 3 FT models
# -> P4 native higher-TF zero-shot. All on GPU, serial (one model on the 8GB card at a time).
cd /media/sean/CORSAIR/binary-algo
PY=~/binary-algo-venv/bin/python
echo "$(date +%H:%M:%S) waiting for FT chain (GPU free)..."
while ! grep -q "ALL FT DONE" kronos_ft_orch.log 2>/dev/null; do sleep 30; done
echo "$(date +%H:%M:%S) GPU free — starting corrected eval sweep"

run() { echo "===== $(date +%H:%M:%S) $* ====="; $PY kronos_mtf.py "$@"; }

# P0: corrected 1m all-sessions zero-shot (disambiguate the confounded null; per-session breakdown)
run 1 all 1 256 20 3000 NeoQuasar/Kronos-small mtf_zs_1m_all
# P1: 1m strict session-only-input zero-shot
run 1 ny   1 256 20 3000 NeoQuasar/Kronos-small mtf_zs_1m_ny
run 1 ldn  1 256 20 3000 NeoQuasar/Kronos-small mtf_zs_1m_ldn
run 1 asia 1 256 20 3000 NeoQuasar/Kronos-small mtf_zs_1m_asia
# P2: re-eval the 3 fine-tuned session models (strict session-only-input) — corrected harness
run 1 ny   1 256 20 3000 models/kronos_ft_ny   ftmtf_1m_ny
run 1 ldn  1 256 20 3000 models/kronos_ft_ldn  ftmtf_1m_ldn
run 1 asia 1 256 20 3000 models/kronos_ft_asia ftmtf_1m_asia
# P4: native higher-TF zero-shot, all-sessions (per-session breakdown). Needs caches from kronos_bars_chain.
for H in 5 10 15 30; do
  while [ ! -f ohlc_cache/EURUSD_${H}m_test.parquet ]; do echo "waiting cache ${H}m..."; sleep 20; done
  run $H all $H 256 20 3000 NeoQuasar/Kronos-small mtf_zs_${H}m_all
done
echo "ALL KRONOS-MTF EVAL DONE"
