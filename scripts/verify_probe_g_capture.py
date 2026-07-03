#!/usr/bin/env python3
"""Verify the scheduled probe g near-close capture actually landed (issue #4).

Fired by deriv-probe-g-verify.timer each weekday at 17:30 NY — AFTER the capture
window (16:20->17:01 NY) and its fold+commit. It exists to catch the SILENT
failure mode: a clean accepted->rejected transition IS captured, but the fold /
commit / push does not complete, so the capture timer stays armed and the pinned
near-close cutoff never reaches the tree. 2026-07-03 already proved a silent
no-op is possible (an off-by-one-minute timer fire), and 2026-07-06 is the FIRST
live run of the crash-safe fold+commit path (commit 0bed351) — worth an
independent check.

Read-only w.r.t. Deriv (no network, no account, no trading): it only inspects
local systemd state, the gitignored capture JSON, and git history.

Verdict -> logs/probe_g_verify.log (gitignored) + stdout/stderr (journal) + exit:
  PASS (0)  clean transition captured AND a fold-commit landed today AND the
            capture timer self-disabled -> self-disables THIS verifier timer.
  NOOP (1)  no clean transition today (pending / thin / holiday / all-rejected);
            capture re-armed for next weekday -> verifier stays armed. Treated as
            success by the unit (SuccessExitStatus=0 1).
  FAIL (2)  clean transition captured but NOT shipped (uncommitted / timer still
            armed), OR capture timer disabled with no transition on record, OR no
            capture attempt recorded today -> loud log + one-shot gh comment on
            issue #4 (sentinel-guarded) -> verifier stays armed. Exit 2 surfaces
            in `systemctl --user status`.
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
CAPTURE = REPO / "logs" / "probe_g_scheduled_capture.json"
VERIFY_LOG = REPO / "logs" / "probe_g_verify.log"
ALERT_SENTINEL = REPO / "logs" / ".probe_g_verify_fail_alerted"
FOLD_TARGET = "results/json/deriv_api_probe_result.json"  # what fold_probe_g_capture.py commits
CAPTURE_TIMER = "deriv-probe-g-nearclose.timer"
VERIFY_TIMER = "deriv-probe-g-verify.timer"
NY = ZoneInfo("America/New_York")
ISSUE = "4"
REPO_SLUG = "sppburke/binary-algo"


def sh(*args: str, env: dict | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=60, cwd=REPO, env=env)
        return p.returncode, (p.stdout or "").strip()
    except Exception as exc:  # noqa: BLE001 — a verifier must never crash the unit
        return 1, f"<error: {exc}>"


def capture_timer_enabled() -> bool:
    _, out = sh("systemctl", "--user", "is-enabled", CAPTURE_TIMER)
    return out.strip() == "enabled"


def load_capture() -> dict | None:
    try:
        return json.loads(CAPTURE.read_text())["runs"]["g_near_close"]
    except Exception:  # noqa: BLE001 — missing/partial JSON is a real state, not a crash
        return None


def fold_committed_today() -> bool:
    # A fresh fold-commit touches the fold target since local midnight.
    rc, out = sh("git", "log", "--since=00:00", "--format=%H", "--", FOLD_TARGET)
    return rc == 0 and bool(out.strip())


def emit(verdict: str, msg: str) -> None:
    stamp = datetime.now(NY).strftime("%Y-%m-%d %H:%M:%S %Z")
    line = f"[{stamp}] {verdict}: {msg}"
    print(line, file=(sys.stderr if verdict == "FAIL" else sys.stdout), flush=True)
    try:
        VERIFY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with VERIFY_LOG.open("a") as fh:
            fh.write(line + "\n")
    except Exception:  # noqa: BLE001
        pass


def gh_comment_once(body: str) -> None:
    """Best-effort one-shot gh comment on FAIL; sentinel-guarded to avoid daily spam."""
    if ALERT_SENTINEL.exists():
        emit("FAIL", "(gh alert suppressed — sentinel already set; verdict logged only)")
        return
    rc, tok = sh("gh", "auth", "token", "--user", "sppburke")
    if rc != 0 or not tok or tok.startswith("<error"):
        emit("FAIL", "(gh token unavailable in service env; verdict logged only)")
        return
    crc, _ = sh(
        "gh", "issue", "comment", ISSUE, "-R", REPO_SLUG, "--body", body,
        env={**os.environ, "GH_TOKEN": tok},
    )
    try:
        ALERT_SENTINEL.write_text("alerted\n")
    except Exception:  # noqa: BLE001
        pass
    emit("FAIL", f"(gh comment attempt rc={crc}; sentinel set to prevent daily repeats)")


def fail(msg: str) -> int:
    emit("FAIL", msg)
    stamp = datetime.now(NY).strftime("%Y-%m-%d %H:%M NY")
    gh_comment_once(
        f"⚠️ **probe g capture verifier FAIL** ({stamp}, host quantum) — issue #4\n\n"
        f"{msg}\n\n"
        f"The pinned near-close cutoff has NOT reached the tree. Check the capture "
        f"timer state (`systemctl --user status {CAPTURE_TIMER}`), "
        f"`logs/probe_g_scheduled_capture.json`, and `logs/probe_g_verify.log` on quantum."
    )
    return 2


def main() -> int:
    cap = load_capture()
    cap_enabled = capture_timer_enabled()
    today = datetime.now(NY).date().isoformat()
    ny_time = str((cap or {}).get("context", {}).get("ny_time", ""))
    cap_today = ny_time.startswith(today)
    status = (cap or {}).get("status")
    obs = (cap or {}).get("observations", {}) or {}
    n_acc = obs.get("n_accepted", 0) or 0
    n_rej = obs.get("n_rejected", 0) or 0
    clean = bool(cap and status == "done" and n_acc > 0 and n_rej > 0)
    committed = fold_committed_today()
    verdict = obs.get("cutoff_verdict", "?")

    if clean and committed and not cap_enabled:
        emit("PASS", f"clean transition captured ({verdict}); fold-commit landed today; capture timer self-disabled")
        sh("systemctl", "--user", "disable", "--now", VERIFY_TIMER)
        return 0
    if clean and (not committed or cap_enabled):
        return fail(
            f"clean transition captured ({verdict}) but NOT shipped — "
            f"fold_committed_today={committed}, capture_timer_still_enabled={cap_enabled}. "
            f"The fold/commit/push likely failed; the captured data sits in gitignored "
            f"logs/probe_g_scheduled_capture.json and needs a manual fold via dev-cycle."
        )
    if not cap_enabled:
        return fail(
            "capture timer is DISABLED but no clean transition is on record "
            f"(status={status}, n_accepted={n_acc}, n_rejected={n_rej}) — inconsistent state, investigate"
        )
    if cap_today:
        emit("NOOP", f"no clean transition today (status={status}, n_accepted={n_acc}, n_rejected={n_rej}); capture re-armed for next weekday")
        return 1
    return fail(
        f"no probe g capture attempt recorded for today ({today}) — capture timer may not have fired "
        f"(capture JSON present={cap is not None}, last ny_time={ny_time or 'n/a'})"
    )


if __name__ == "__main__":
    sys.exit(main())
