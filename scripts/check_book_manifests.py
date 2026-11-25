#!/usr/bin/env python3
"""Check every frozen book manifest: `currency` must equal the pair that starts its book id.

A book id is `<PAIR>.<book>.v<N>`. `manifest.build` writes currency EURUSD whatever the id, so a freeze
script that does not set it mislabels the book. `manifest.py` is pinned by frozen evidence objects, so the
check lives here instead of inside it. Read-only; run it after a freeze and before the commit:

    ~/binary-algo-venv/bin/python scripts/check_book_manifests.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def manifest_paths(root: Path = ROOT) -> list[Path]:
    """Legacy manifests sit beside their book directory, candidate manifests inside it."""
    books = root / "books"
    return sorted([*books.glob("*.manifest.json"), *books.glob("*/*.manifest.json")])


def check(paths: list[Path]) -> list[str]:
    failures: list[str] = []
    for path in paths:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            failures.append(f"{path}: cannot read: {exc}")
            continue
        book_id = manifest.get("id") if isinstance(manifest, dict) else None
        currency = manifest.get("currency") if isinstance(manifest, dict) else None
        pair = book_id.split(".", 1)[0] if isinstance(book_id, str) else ""
        if not pair or not isinstance(currency, str) or not currency:
            failures.append(f"{path}: needs a string id '<PAIR>.<book>.v<N>' and a currency")
        elif currency != pair:
            failures.append(f"{path}: currency {currency!r} does not match book id {book_id!r}")
    return failures


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: ~/binary-algo-venv/bin/python scripts/check_book_manifests.py", file=sys.stderr)
        return 2
    paths = manifest_paths()
    failures = check(paths)
    if failures:
        print("ERROR: book manifest check failed", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"OK: {len(paths)} book manifests; every currency matches its book id")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
