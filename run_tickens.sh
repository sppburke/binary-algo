#!/bin/bash
cd /home/sean/git/binary-algo
until grep -q "CACHE DONE" /tmp/tickcache.log 2>/dev/null; do sleep 20; done
echo "cache done -> running ensemble"
~/binary-algo-venv/bin/python tick_ensemble.py > /tmp/tickens.log 2>&1
echo "ENSEMBLE RUNNER DONE"
