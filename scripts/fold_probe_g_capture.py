"""Fold a scheduled probe g near-close capture into the tracked result JSON,
then commit + push — unattended (issue #4).

The systemd timer `ops/deriv-probe-g-nearclose.timer` fires `deriv_api_probe.py
--probes g` into a gitignored capture file. This script is the ExecStartPost
half: when that capture holds a CLEAN accepted->rejected transition
(n_accepted>0 AND n_rejected>0), it merges the `g_near_close` record (and its
derived `decisions.last_start_cutoff_ny`) into
`results/json/deriv_api_probe_result.json` and pushes to main. On a non-clean
capture (holiday/closed market) it is a no-op (exit 1) so the timer keeps
retrying the next weekday.

Fail-safe git discipline: commit, then `fetch` + `rebase origin/main` (abort +
undo the commit on any conflict), then `push` (retried); never force-push. On
any git failure it exits non-zero leaving the gitignored capture intact for the
next weekday's retry.

Exit codes: 0 = folded+pushed (or already up to date) -> timer may self-disable;
1 = nothing to do / not clean / git failure -> keep the timer armed.

Run from repo root:
    ~/binary-algo-venv/bin/python scripts/fold_probe_g_capture.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_PATH = REPO_ROOT / "logs" / "probe_g_scheduled_capture.json"
RESULT_PATH = REPO_ROOT / "results" / "json" / "deriv_api_probe_result.json"
PUSH_RETRIES = 3
PUSH_RETRY_SLEEP_S = 10


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                          capture_output=True, text=True, check=check)


def _log(msg: str) -> None:
    print(f"[fold_probe_g] {msg}")


def is_clean_transition(capture: dict) -> bool:
    obs = (capture.get("runs", {}).get("g_near_close", {}) or {}).get("observations", {}) or {}
    return obs.get("n_accepted", 0) > 0 and obs.get("n_rejected", 0) > 0


def merge(capture: dict, result: dict) -> dict:
    """Scoped merge: only g_near_close, the derived last_start_cutoff_ny, and
    updated_utc are touched — every other probe record is left untouched."""
    result.setdefault("runs", {})["g_near_close"] = capture["runs"]["g_near_close"]
    cutoff = (capture.get("decisions") or {}).get("last_start_cutoff_ny")
    if cutoff is not None:
        result.setdefault("decisions", {})["last_start_cutoff_ny"] = cutoff
    result["updated_utc"] = capture.get("updated_utc") or datetime.now(timezone.utc).isoformat()
    return result


def push_with_retries() -> bool:
    for attempt in range(1, PUSH_RETRIES + 1):
        if _git("push", check=False).returncode == 0:
            return True
        _log(f"push attempt {attempt}/{PUSH_RETRIES} failed")
        if attempt < PUSH_RETRIES:
            time.sleep(PUSH_RETRY_SLEEP_S)
    return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="merge + show the diff, skip commit/push")
    p.add_argument("--capture-path", type=Path, default=CAPTURE_PATH)
    p.add_argument("--result-path", type=Path, default=RESULT_PATH)
    args = p.parse_args()

    if not args.capture_path.exists():
        _log(f"no capture at {args.capture_path}; nothing to fold")
        return 1
    capture = json.loads(args.capture_path.read_text())
    if not is_clean_transition(capture):
        obs = (capture.get("runs", {}).get("g_near_close", {}) or {}).get("observations", {}) or {}
        _log(f"capture not a clean transition (n_accepted={obs.get('n_accepted')}, "
             f"n_rejected={obs.get('n_rejected')}); leaving timer armed for retry")
        return 1

    try:
        rel = str(args.result_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        _log(f"result path {args.result_path} is outside the repo; refusing to fold")
        return 1
    result = json.loads(args.result_path.read_text())
    merged = merge(capture, result)
    args.result_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")

    if args.dry_run:
        print(_git("diff", "--", rel, check=False).stdout)
        _log("dry-run: merged in working tree, no commit/push")
        return 0

    _git("add", rel)
    if _git("diff", "--cached", "--quiet", check=False).returncode == 0:
        # no staged change: either already folded, or recover an earlier
        # committed-but-unpushed fold
        _git("fetch", "origin", "main", check=False)
        ahead = _git("rev-list", "origin/main..HEAD", "--", rel, check=False).stdout.strip()
        if ahead:
            _log("prior fold committed but unpushed; pushing")
            return 0 if push_with_retries() else 1
        _log("result JSON already up to date; nothing to commit")
        return 0

    verdict = (capture["runs"]["g_near_close"].get("observations") or {}).get("cutoff_verdict", "")
    _git("commit", "-m",
         "Deriv: fold scheduled probe g near-close capture (issue #4)\n\n"
         f"Automated fold-in of a clean accepted->rejected transition: {verdict}\n\n"
         "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>")

    _git("fetch", "origin", "main", check=False)
    if _git("rebase", "origin/main", check=False).returncode != 0:
        _git("rebase", "--abort", check=False)
        _git("reset", "--soft", "HEAD~1", check=False)  # undo the commit, keep the merge staged-free for retry
        _log("rebase conflict; aborted and undid the commit — timer stays armed")
        return 1

    if not push_with_retries():
        _log("push failed after retries; local commit retained, will re-push next run")
        return 1
    _log(f"folded + pushed: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
