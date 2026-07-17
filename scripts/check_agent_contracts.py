#!/usr/bin/env python3
"""Validate binary-algo's shared agent instructions and skill projections."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / ".codex" / "skills"
CLAUDE_SKILLS = ROOT / ".claude" / "skills"
NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
MAX_DESCRIPTION = 1024

PLATFORM_TOKENS = (
    "Claude",
    "Codex",
    "Cursor",
    "Opus",
    "Sonnet",
    "Haiku",
    "TodoWrite",
    "codex review",
    ".cursor/skills",
    ".agents/skills",
    "Bugbot",
    "Greptile",
    "Railway",
    "agent-coordination:v1",
)
PLATFORM_PATH_ALLOWLIST = ("CLAUDE.md", ".claude/skills/", ".codex/skills/")

REQUIRED_BODY_TEXT = {
    "feature-dev": (
        "Classify one primary plan kind:",
        "**experiment**",
        "**infrastructure**",
        "**roadmap**",
        "Stop with a no-change conclusion",
        "Ask only about a user-owned choice",
        "**surface budget**",
        "`DO NOT SHIP`",
        "`why_valuable`",
        "fresh read-only `plan-review` reviewer",
        "Print the final title and body exactly as filed",
        "plan-review-receipt/v1",
    ),
    "plan-review": (
        "`experiment`, `infrastructure`, `roadmap`",
        "**material-claim ledger**",
        "**named-surface/citation inventory**",
        "`supported`",
        "`contradicted`",
        "`evidence-gap`",
        "`N = S + C + G`",
        "strongest simpler repository-native design",
        "operation: add | replace | delete | move",
        "any Blocking finding or contradicted material claim -> `reject`",
        "otherwise any exhausted material evidence gap -> `blocked`",
        "`wc_ret`",
        "ties LOSE",
        "VAL worst-half",
        "2026` strict OOS",
        "`nonoverlap_chrono`",
        "`boot`",
        "`0.541`",
        "`p10",
        "`80%`",
    ),
    "dev-cycle": (
        "implement -> test -> clean/archive -> review/fix -> final gate -> "
        "commit -> sync/re-gate -> push",
        "ships directly to `main`",
        "pull-request, worktree, or CI workflow",
        "**agent-instructions/skills**",
        "`plan-review` coverage",
        "**surface budget**",
        "focused negative/adversarial tests",
        "**Reuse / precedent**",
        "**Simpler counterfactual**",
        "**Complexity**",
        "**Cleanliness**",
        "review only the unreviewed delta",
        "scripts/check_agent_contracts.py",
        "Never force-push `main`",
        "Shortcuts / hacks taken: none",
    ),
    "strategy-eval": (
        "`results/<PAIR>_RESULTS.md`",
        "`wc_ret`",
        "`nonoverlap_chrono`",
        "`boot`",
        "Pooled-CPCV",
        "One heavy job at a time",
    ),
}

AGENTS_ANCHORS = (
    "`AGENTS.md` is the canonical shared instruction file",
    "`.codex/skills/` is the canonical physical skill catalog",
    "Tracked-file mutation requires `dev-cycle`",
    "`docs/_EVIDENCE-FIRST.md` is the binding",
    "Run scripts from the repository root",
    "Never pool across timeframes",
    "Certification requires 15 refit-CPCV purged paths",
)


def add(failures: list[str], path: Path, message: str) -> None:
    try:
        label = path.relative_to(ROOT)
    except ValueError:
        label = path
    failures.append(f"{label}: {message}")


def find_platform_tokens(body: str) -> list[str]:
    scan = body
    for allowed in PLATFORM_PATH_ALLOWLIST:
        scan = scan.replace(allowed, "")
    folded = scan.casefold()
    return [token for token in PLATFORM_TOKENS if token.casefold() in folded]


def read_text(path: Path, failures: list[str]) -> str | None:
    try:
        data = path.read_bytes()
    except OSError as exc:
        add(failures, path, f"cannot read: {exc}")
        return None
    if data.startswith(b"\xef\xbb\xbf"):
        add(failures, path, "UTF-8 BOM is not allowed")
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        add(failures, path, f"not valid UTF-8: {exc}")
        return None
    if "\t" in text:
        add(failures, path, "tabs are not allowed")
    bad = next(
        (char for char in text if ord(char) < 0x20 and char not in "\n\r"),
        None,
    )
    if bad is not None:
        add(failures, path, f"unsupported control character U+{ord(bad):04X}")
    return text.replace("\r\n", "\n")


def parse_skill(
    path: Path, expected_name: str, failures: list[str]
) -> tuple[str, str] | None:
    text = read_text(path, failures)
    if text is None:
        return None
    lines = text.split("\n")
    if not lines or lines[0] != "---":
        add(failures, path, "must start with an exact frontmatter delimiter")
        return None
    try:
        closing = lines.index("---", 1)
    except ValueError:
        add(failures, path, "frontmatter closing delimiter is missing")
        return None

    frontmatter = lines[1:closing]
    values: dict[str, str] = {}
    index = 0
    while index < len(frontmatter):
        line = frontmatter[index]
        if ":" not in line or line.startswith(" "):
            add(failures, path, f"invalid frontmatter line: {line!r}")
            index += 1
            continue
        key, raw = line.split(":", 1)
        if key in values:
            add(failures, path, f"duplicate frontmatter key {key!r}")
        if key not in {"name", "description"}:
            add(failures, path, f"unsupported frontmatter key {key!r}")
        raw = raw.strip()
        if key == "description" and raw in {">", ">-", "|", "|-"}:
            payload: list[str] = []
            index += 1
            while index < len(frontmatter) and frontmatter[index].startswith("  "):
                payload.append(frontmatter[index][2:].strip())
                index += 1
            values[key] = " ".join(payload).strip()
            continue
        values[key] = raw
        index += 1

    if set(values) != {"name", "description"}:
        add(failures, path, "frontmatter must contain exactly name and description")
        return None
    if values["name"] != expected_name:
        add(
            failures,
            path,
            f"frontmatter name {values['name']!r} does not match {expected_name!r}",
        )
    description = values["description"]
    if not description:
        add(failures, path, "description is empty")
    elif len(description) > MAX_DESCRIPTION:
        add(
            failures,
            path,
            f"description length {len(description)} exceeds {MAX_DESCRIPTION}",
        )
    return "\n".join(lines[closing + 1 :]), description


def validate_interface(path: Path, skill: str, failures: list[str]) -> None:
    text = read_text(path, failures)
    if text is None:
        return
    lines = [line for line in text.splitlines() if line]
    if not lines or lines[0] != "interface:":
        add(failures, path, "must start with exact interface root")
        return
    values: dict[str, str] = {}
    pattern = re.compile(r'  ([a-z_]+): ("(?:[^"\\]|\\.)*")\Z')
    for line in lines[1:]:
        match = pattern.fullmatch(line)
        if match is None:
            add(failures, path, f"unsupported interface line: {line!r}")
            continue
        key, encoded = match.groups()
        if key in values:
            add(failures, path, f"duplicate interface key {key!r}")
            continue
        try:
            values[key] = json.loads(encoded)
        except json.JSONDecodeError as exc:
            add(failures, path, f"invalid quoted value for {key}: {exc}")
    expected = {"display_name", "short_description", "default_prompt"}
    if set(values) != expected:
        add(failures, path, f"interface keys must be exactly {sorted(expected)}")
        return
    if not values["display_name"] or len(values["display_name"]) > 64:
        add(failures, path, "display_name must contain 1-64 characters")
    short = values["short_description"]
    if not 25 <= len(short) <= 64:
        add(failures, path, "short_description must contain 25-64 characters")
    prompt = values["default_prompt"]
    token = f"${skill}"
    if not prompt.startswith(f"Use {token}") or prompt.count(token) != 1:
        add(failures, path, f"default_prompt must start with 'Use {token}' exactly once")
    if len(prompt) > 256:
        add(failures, path, "default_prompt exceeds 256 characters")


def validate() -> list[str]:
    failures: list[str] = []
    agents = ROOT / "AGENTS.md"
    claude = ROOT / "CLAUDE.md"

    if agents.is_symlink() or not agents.is_file():
        add(failures, agents, "must be the canonical regular file")
    agents_text = read_text(agents, failures) if agents.is_file() else None
    if agents_text is not None:
        for anchor in AGENTS_ANCHORS:
            if anchor not in agents_text:
                add(failures, agents, f"missing shared invariant {anchor!r}")

    if not claude.is_symlink():
        add(failures, claude, "must be a relative symlink to AGENTS.md")
    else:
        target = claude.readlink()
        if str(target) != "AGENTS.md":
            add(failures, claude, f"expected link target 'AGENTS.md', found {target!s}")
        if claude.resolve() != agents.resolve():
            add(failures, claude, "does not resolve to canonical AGENTS.md")

    if CANONICAL.is_symlink() or not CANONICAL.is_dir():
        add(failures, CANONICAL, "must be a physical canonical directory")
        return failures
    if CLAUDE_SKILLS.is_symlink() or not CLAUDE_SKILLS.is_dir():
        add(failures, CLAUDE_SKILLS, "must be a physical projection directory")
        return failures

    canonical_entries = sorted(CANONICAL.iterdir(), key=lambda path: path.name)
    names: list[str] = []
    for directory in canonical_entries:
        name = directory.name
        if not NAME_RE.fullmatch(name):
            add(failures, directory, "skill name must be safe lowercase kebab-case")
            continue
        if directory.is_symlink() or not directory.is_dir():
            add(failures, directory, "canonical skill entry must be a physical directory")
            continue
        names.append(name)
        for child in directory.rglob("*"):
            if child.is_symlink():
                add(failures, child, "canonical skill trees must not contain symlinks")

        skill_path = directory / "SKILL.md"
        interface_path = directory / "agents" / "openai.yaml"
        if skill_path.is_symlink() or not skill_path.is_file():
            add(failures, skill_path, "must be a regular file")
            continue
        parsed = parse_skill(skill_path, name, failures)
        if parsed is not None:
            body, _description = parsed
            body_flat = " ".join(body.split())
            for token in find_platform_tokens(body):
                add(
                    failures,
                    skill_path,
                    f"platform/dashboard-specific token remains: {token!r}",
                )
            for required in REQUIRED_BODY_TEXT.get(name, ()):
                if " ".join(required.split()) not in body_flat:
                    add(failures, skill_path, f"missing workflow invariant {required!r}")
        if interface_path.is_symlink() or not interface_path.is_file():
            add(failures, interface_path, "must be a regular file")
        else:
            validate_interface(interface_path, name, failures)

    if not names:
        add(failures, CANONICAL, "contains no valid skills")

    projected_entries = sorted(CLAUDE_SKILLS.iterdir(), key=lambda path: path.name)
    projected_names = [path.name for path in projected_entries]
    if projected_names != names:
        add(
            failures,
            CLAUDE_SKILLS,
            f"projection names {projected_names!r} do not match canonical {names!r}",
        )
    for name in names:
        projection = CLAUDE_SKILLS / name
        canonical = CANONICAL / name
        expected = f"../../.codex/skills/{name}"
        if not projection.is_symlink():
            add(failures, projection, "must be a relative directory symlink")
            continue
        target = projection.readlink()
        if str(target) != expected:
            add(failures, projection, f"expected link target {expected!r}, found {target!s}")
        if projection.resolve() != canonical.resolve():
            add(failures, projection, "does not resolve to the canonical skill directory")
            continue
        try:
            canonical_bytes = (canonical / "SKILL.md").read_bytes()
            projected_bytes = (projection / "SKILL.md").read_bytes()
        except OSError as exc:
            add(failures, projection, f"cannot read resolved SKILL.md: {exc}")
        else:
            if projected_bytes != canonical_bytes:
                add(failures, projection, "resolved SKILL.md bytes differ from canonical")

    return failures


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(
            "usage: ~/binary-algo-venv/bin/python scripts/check_agent_contracts.py",
            file=sys.stderr,
        )
        return 2
    failures = validate()
    if failures:
        print("ERROR: agent contract validation failed", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(
        "OK: canonical AGENTS, runtime-neutral skills, metadata, and Claude projections"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
