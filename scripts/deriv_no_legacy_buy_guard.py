"""Refuse to start the issue-#6 executor while a legacy buyer is running."""

from __future__ import annotations

import os
import sys
from pathlib import Path


BUY_FLAGS = {"--demo-buy", "--auth-smoke"}


def cmdline(pid_dir: Path) -> list[str]:
    try:
        raw = (pid_dir / "cmdline").read_bytes()
    except OSError:
        return []
    return [p.decode("utf-8", errors="replace") for p in raw.split(b"\0") if p]


def offender(parts: list[str]) -> bool:
    if not parts:
        return False
    joined = " ".join(parts)
    if "deriv_runtime_supervisor.py" in joined and "--demo-buy" in parts:
        return True
    if "deriv_demo_executor.py" in joined and any(flag in parts for flag in BUY_FLAGS):
        return True
    return False


def main() -> int:
    self_pid = os.getpid()
    offenders: list[tuple[int, list[str]]] = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        pid = int(item.name)
        if pid == self_pid:
            continue
        parts = cmdline(item)
        if offender(parts):
            offenders.append((pid, parts))
    if offenders:
        for pid, parts in offenders:
            print(f"legacy buy-capable process still running pid={pid}: {' '.join(parts)}", file=sys.stderr)
        return 2
    print("PASS: no legacy deriv demo buyer command lines found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
