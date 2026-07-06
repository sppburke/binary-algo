#!/usr/bin/env python3
"""Phase-5 shifted-lane audit: quote-coverage readiness notifier (issue #4).

Fully hands-off. Fires weekday 17:30 NY on the dev host (quantum) via
deriv-phase5-readiness.timer, AFTER the NY session close so the day's coverage is
complete. It ssh's to the trading VPS (64.177.80.63, agent-less via id_personal),
counts FULL-CLEAN NY-session quote_snapshot coverage per own-pair lane, and:

  READY  — every own-pair lane (USDJPY/USDCAD/AUDUSD/NZDUSD x UP/DOWN) has
           >= 3 full-clean NY sessions -> posts a one-shot gh comment on issue #4
           and self-disables its own timer (job done). exit 0.
  NOT-READY — logs current per-lane counts, stays armed for the next weekday. exit 1.
  ERROR  — ssh/parse failure -> logs, stays armed. exit 2.

A "full-clean" session = a weekday NY date whose lane rows (event=quote_snapshot)
span >= 5 distinct hours within 08:00-17:00 NY (excludes partial/holiday-thin
days). Reads only counts from the VPS — no secrets, no trading, no writes there.
Verdict -> logs/phase5_readiness.log (gitignored). This is the enable_bar's
>=3-NY-session precondition; a first-pass audit before it is met would rest on a
thin quote median.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "logs" / "phase5_readiness.log"
SENTINEL = REPO / "logs" / ".phase5_ready_notified"
NY = ZoneInfo("America/New_York")
VPS = "64.177.80.63"
KEY = str(Path.home() / ".ssh" / "id_personal")
SSH = [
    "ssh", "-T", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
    "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20",
    "-i", KEY, VPS, "python3", "-",
]
OWN_PAIRS = ("USDJPY", "USDCAD", "AUDUSD", "NZDUSD")
SIDES = ("UP", "DOWN")
MIN_SESSIONS = 3
MIN_HOURS = 5
VERIFY_TIMER = "deriv-phase5-readiness.timer"
ISSUE, SLUG = "4", "sppburke/binary-algo"

# Runs ON the VPS via ssh stdin; emits {lane: [full_clean_dates]} as one JSON line.
REMOTE = r'''
import json, glob, gzip, sys
from datetime import datetime
from zoneinfo import ZoneInfo
NY = ZoneInfo("America/New_York")
QDIR = "/home/sean/binary-algo/logs/paper_trades/quote_audit"
cov = {}
for fp in glob.glob(QDIR + "/*.jsonl*"):
    opener = gzip.open if fp.endswith(".gz") else open
    try:
        with opener(fp, "rt") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("event") != "quote_snapshot":
                    continue
                pair, side, ts = r.get("pair"), r.get("side"), r.get("timestamp_utc")
                if not (pair and side and ts):
                    continue
                try:
                    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(NY)
                except Exception:
                    continue
                if dt.weekday() >= 5 or not (8 <= dt.hour < 17):
                    continue
                cov.setdefault(pair + ":" + side, {}).setdefault(dt.date().isoformat(), set()).add(dt.hour)
    except Exception:
        continue
out = {lane: sorted(d for d, hrs in dates.items() if len(hrs) >= 5) for lane, dates in cov.items()}
print(json.dumps(out))
'''


def emit(verdict: str, msg: str) -> None:
    stamp = datetime.now(NY).strftime("%Y-%m-%d %H:%M %Z")
    line = f"[{stamp}] {verdict}: {msg}"
    print(line, file=(sys.stderr if verdict == "ERROR" else sys.stdout), flush=True)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a") as fh:
            fh.write(line + "\n")
    except Exception:  # noqa: BLE001
        pass


def gh_comment_once(body: str) -> str:
    if SENTINEL.exists():
        return "(sentinel set; comment skipped)"
    try:
        tok = subprocess.run(["gh", "auth", "token", "--user", "sppburke"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        if not tok:
            return "(no gh token; verdict logged only)"
        rc = subprocess.run(["gh", "issue", "comment", ISSUE, "-R", SLUG, "--body", body],
                            capture_output=True, text=True, timeout=60,
                            env={**os.environ, "GH_TOKEN": tok}).returncode
        SENTINEL.write_text("notified\n")
        return f"(gh comment rc={rc}; sentinel set)"
    except Exception as exc:  # noqa: BLE001
        return f"(gh comment failed: {exc})"


def main() -> int:
    try:
        p = subprocess.run(SSH, input=REMOTE, capture_output=True, text=True, timeout=90)
    except Exception as exc:  # noqa: BLE001
        emit("ERROR", f"ssh to VPS failed: {exc}")
        return 2
    if p.returncode != 0:
        emit("ERROR", f"remote coverage check rc={p.returncode}: {(p.stderr or '').strip()[:200]}")
        return 2
    try:
        cov = json.loads((p.stdout or "").strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001
        emit("ERROR", f"bad remote output: {exc}; stdout={p.stdout[:200]!r}")
        return 2

    lanes = [f"{pr}:{sd}" for pr in OWN_PAIRS for sd in SIDES]
    counts = {ln: len(cov.get(ln, [])) for ln in lanes}
    summary = ", ".join(f"{ln}={counts[ln]}" for ln in lanes)
    if all(counts[ln] >= MIN_SESSIONS for ln in lanes):
        emit("READY", f"all {len(lanes)} own-pair lanes >= {MIN_SESSIONS} full-clean NY sessions ({summary})")
        note = gh_comment_once(
            f"✅ **Phase-5 quote coverage READY** — every own-pair lane now has "
            f">= {MIN_SESSIONS} full-clean NY sessions (>= {MIN_HOURS}h within 08:00-17:00 NY): {summary}. "
            "Next step: assemble + run the shifted-lane offset audit as a strategy-eval session "
            "(`deriv_offset_audit.py` machinery over the `/home/sean/git/raw/<PAIR>/` tick archive + the "
            "`quote_audit` medians). This readiness notifier has self-disabled."
        )
        emit("READY", note)
        subprocess.run(["systemctl", "--user", "disable", "--now", VERIFY_TIMER], capture_output=True)
        return 0
    emit("NOT-READY", f"waiting; full-clean NY sessions per own-pair lane: {summary} (need >= {MIN_SESSIONS} each)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
