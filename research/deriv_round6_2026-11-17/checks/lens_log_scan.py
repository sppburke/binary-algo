#!/usr/bin/env python3
"""Round 6: did any lens name a holdout path in a shell command?  Reads the lenses' command logs, which stay in the lab.

From the repository root:

    python3 -I research/deriv_round6_2026-11-17/checks/lens_log_scan.py /mnt/data/deriv-sota-lab/discovery/round6/logs

Each log is the output of one read-only Codex lens; a shell command appears as a line that starts with a shell
invocation. The scan counts the commands and lists those that name a path the prompts forbid (the holdout folder, the
full datasets, files named look1, look2 or holdout, and tick or bar files dated 2026-08-01 or later). A command that
names such a path only inside an exclusion pattern ('!...') does not read it. Limits: a command that reached a
forbidden file without naming it (through a wildcard or a script) would not be caught, and the read-only sandbox is a
write restriction, not a read restriction.
"""
import re
import sys
from pathlib import Path

forbidden = re.compile(r'deriv-sota-lab/holdout|/holdout|holdout|dataset_full_|look1|look2')
late = re.compile(r'20(26-(0[89]|1[0-2])|27-\d\d)-\d\d')                 # a date from 2026-08-01 on
shell = re.compile(r'^\s*(/bin/)?(ba)?sh -lc ')
excl = re.compile(r"-g\s+'\"'\"'!|-g\s+'!|--glob\s+'!|-g\s+\"!|--exclude")
logs = sorted(p for p in Path(sys.argv[1]).iterdir() if p.is_file())
n_cmd, named, only_excluded = 0, [], 0
for f in logs:
    for ln in f.read_text(errors='replace').splitlines():
        if not shell.match(ln):
            continue
        n_cmd += 1
        if forbidden.search(ln) or (re.search(r'ticks|bars|sidecar', ln) and late.search(ln)):
            stripped = re.sub(r"'\"'\"'![^']*'\"'\"'|'![^']*'|\"![^\"]*\"", '', ln)     # drop exclusion patterns
            if forbidden.search(stripped) or (re.search(r'ticks|bars|sidecar', stripped) and late.search(stripped)):
                named.append((f.name, ln.strip()[:200]))
            else:
                only_excluded += 1
print(f'{len(logs)} logs, {n_cmd} shell command lines')
print(f'command lines that name a forbidden path only to exclude it from a listing: {only_excluded}')
print(f'command lines that name a forbidden path otherwise: {len(named)}')
for name, ln in named:
    print('  ', name, ln)
