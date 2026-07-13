"""Registry-driven active coverage, side-floor, and settlement-haircut resolver.

Resolves, for each Deriv-enabled pair, the exact deploy operating point the
executor must gate payouts against: active coverage, confidence threshold,
side-specific refit-CPCV p10 floors, and the settlement haircut with its
evidentiary source. Everything fails closed: a missing, ambiguous, or
conflicting source raises instead of defaulting.

Floors are refit-CPCV per-era floors: the frozen vintages the executor loads
decay forward, so the resolution carries the book freeze/vintage date and the
gate certifies the documented floor under the periodic-retrain deploy policy,
not the frozen artifact's current forward win rate.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from book_runtime import (
    REPO_ROOT,
    TARGET_BOOKS,
    BookRuntimeError,
    _conf_thr,
    _resolve_coverage,
    load_index,
    load_registered_manifest,
    load_registered_strategy_path,
    require_active_book_lifecycle,
)
from deriv_backfill import DEFAULT_ENABLED_PAIRS


TICKSETTLE_DIR = REPO_ROOT / "results" / "json"
FLOOR_TOL = 1e-9

# Hand-mapped cov-suffix table (issue #3): suffix encodings are inconsistent
# across the registry ("cov1"=0.01 but "cov005"=0.005, "cov5"=0.05), so a
# numeric regex derivation would mis-read them. Unknown suffixes fail closed.
COV_SUFFIX = {
    "cov005": 0.005,
    "cov1": 0.01,
    "cov02": 0.02,
    "cov2": 0.02,
    "cov3": 0.03,
    "cov5": 0.05,
    "cov10": 0.10,
}

HAIRCUT_KEYS = ("tick_haircut", "barclose_haircut", "ticksettle_mean_delta")

# Rule-3 documented trusted proxies: only pairs whose Tier-2 ledger explicitly
# says the proxy is trusted for that pair. Values are signed as recorded.
DOCUMENTED_PROXY_HAIRCUTS = {
    "USDCAD": {
        "raw": -0.0035,
        "detail": (
            "results/USDCAD_RESULTS.md: bar-close proxy haircut -0.0035 validated on "
            "USDJPY/AUDUSD tick settlement, documented trusted for USDCAD"
        ),
    },
}

# Vintage fallback for manifests with created_utc null (issue #3): ledger freeze
# dates of record.
LEDGER_FREEZE_DATES = {
    "GBPUSD.m15ny_xpair_seedens8.v1": (
        "2026-06-10",
        "sweeps/GBPUSD_15m.md 2026-06-10 'Book GBPUSD.m15ny_xpair_seedens8.v1 FROZEN'",
    ),
    "USDCHF.m15ny_xpair_seedens.v1": (
        "2026-06-11",
        "results/USDCHF_RESULTS.md 'seed-ens supersede 2026-06-11' (SWEEP CLOSED line)",
    ),
}


class FloorResolutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class PairFloorResolution:
    pair: str
    book_id: str
    strategy_path: str
    coverage: float
    coverage_source: str
    confidence_threshold: float
    up_floor: float
    down_floor: float
    floor_source: str
    haircut_raw: float
    haircut_abs: float
    haircut_source: str
    haircut_source_detail: str
    book_created_utc: str
    vintage_source: str
    registry_content_id: str | None

    def side_floor(self, direction: str) -> float:
        if direction == "UP":
            return self.up_floor
        if direction == "DOWN":
            return self.down_floor
        raise FloorResolutionError(f"{self.pair}: unknown direction {direction!r}")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _floor_cells(
    source_name: str,
    metrics: dict[str, Any],
    default_cov: float | None,
) -> list[tuple[float, str, float, str]]:
    """Yield (cov, side, value, detail) floor cells from one metrics dict."""
    cells: list[tuple[float, str, float, str]] = []
    for key, val in metrics.items():
        m = re.fullmatch(r"(up|down)_refit_cpcv_p10(?:_(\w+))?", key)
        m2 = re.fullmatch(r"refit_cpcv_p10_(\w+)_(UP|DOWN)", key)
        if not m and not m2:
            continue
        if not isinstance(val, (int, float)):
            raise FloorResolutionError(f"{source_name}: floor key {key} is not numeric: {val!r}")
        if m:
            side = m.group(1).upper()
            suffix = m.group(2)
        else:
            side = m2.group(2)
            suffix = m2.group(1)
        if suffix is None:
            if default_cov is None:
                raise FloorResolutionError(
                    f"{source_name}: unsuffixed floor key {key} without a scalar cov context"
                )
            cov = float(default_cov)
        else:
            if suffix not in COV_SUFFIX:
                raise FloorResolutionError(f"{source_name}: unknown cov suffix in floor key {key}")
            cov = COV_SUFFIX[suffix]
        cells.append((cov, side, float(val), f"{source_name}[{key}]"))
    return cells


def _deploy_floor_cells(source_name: str, deploy: dict[str, Any]) -> list[tuple[float, str, float, str]]:
    cells: list[tuple[float, str, float, str]] = []
    for key, val in deploy.items():
        m = re.fullmatch(r"refit_floor_p10_(\w+)", key)
        if not m:
            continue
        suffix = m.group(1)
        if suffix not in COV_SUFFIX:
            raise FloorResolutionError(f"{source_name}: unknown cov suffix in deploy key {key}")
        if not isinstance(val, dict):
            raise FloorResolutionError(f"{source_name}: deploy key {key} is not a side dict: {val!r}")
        for side in ("UP", "DOWN"):
            if side in val:
                if not isinstance(val[side], (int, float)):
                    raise FloorResolutionError(f"{source_name}: {key}.{side} is not numeric")
                cells.append((COV_SUFFIX[suffix], side, float(val[side]), f"{source_name}[{key}.{side}]"))
    return cells


def normalize_floors(
    book_id: str,
    index_metrics: dict[str, Any],
    manifest_metrics: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[tuple[float, str], dict[str, Any]]:
    """Normalize all known floor key forms into (coverage, side) -> {value, sources}.

    Sources: INDEX metrics, manifest metrics, and the strategy deploy block. The
    same cell appearing in multiple sources must agree exactly; a conflict fails
    closed. Prose certification strings are never parsed.
    """
    cells: list[tuple[float, str, float, str]] = []
    cells += _floor_cells(f"books/INDEX.json[{book_id}].metrics", index_metrics,
                          index_metrics.get("cov") if isinstance(index_metrics.get("cov"), (int, float)) else None)
    cells += _floor_cells(f"{book_id}.manifest metrics", manifest_metrics,
                          manifest_metrics.get("cov") if isinstance(manifest_metrics.get("cov"), (int, float)) else None)
    deploy = strategy.get("deploy")
    if isinstance(deploy, dict):
        cells += _deploy_floor_cells(f"{book_id} strategy.deploy", deploy)

    out: dict[tuple[float, str], dict[str, Any]] = {}
    for cov, side, value, detail in cells:
        cell = out.setdefault((cov, side), {"value": value, "sources": []})
        if abs(cell["value"] - value) > FLOOR_TOL:
            raise FloorResolutionError(
                f"{book_id}: conflicting floors for (cov={cov}, {side}): "
                f"{cell['value']} from {cell['sources']} vs {value} from {detail}"
            )
        cell["sources"].append(detail)
    return out


def _structured_haircuts(source_name: str, obj: dict[str, Any]) -> list[tuple[float, str]]:
    found = []
    for key in HAIRCUT_KEYS:
        val = obj.get(key)
        if isinstance(val, (int, float)):
            found.append((float(val), f"{source_name}[{key}]"))
    return found


def _open_real_directory(path: Path) -> tuple[Path, int]:
    """Open a directory by walking every component with O_NOFOLLOW."""

    raw = Path(path)
    if ".." in raw.parts:
        raise FloorResolutionError(f"authority directory contains '..': {path}")
    absolute = raw if raw.is_absolute() else Path.cwd() / raw
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        fd = os.open(absolute.anchor, flags)
    except OSError as exc:
        raise FloorResolutionError(
            f"cannot open authority filesystem root for {absolute}: {exc}"
        ) from exc
    try:
        for component in absolute.parts[1:]:
            try:
                next_fd = os.open(component, flags, dir_fd=fd)
            except OSError as exc:
                raise FloorResolutionError(
                    f"authority directory chain contains a symlink or non-directory at "
                    f"{absolute}: {exc}"
                ) from exc
            os.close(fd)
            fd = next_fd
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise FloorResolutionError(f"authority path is not a directory: {absolute}")
        return absolute, fd
    except Exception:
        os.close(fd)
        raise


def _load_ticksettle_at(directory_fd: int, name: str, display_path: Path) -> tuple[float, bool]:
    if Path(name).name != name or not name:
        raise FloorResolutionError(f"invalid tick-settlement authority filename: {name!r}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise FloorResolutionError(
            f"tick-settlement authority is missing, symlinked, or unreadable: {display_path}: {exc}"
        ) from exc
    try:
        descriptor_stat = os.fstat(fd)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise FloorResolutionError(
                f"tick-settlement authority is not a regular file: {display_path}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        try:
            path_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as exc:
            raise FloorResolutionError(
                f"tick-settlement authority changed while reading: {display_path}"
            ) from exc
        if (
            path_stat.st_dev != descriptor_stat.st_dev
            or path_stat.st_ino != descriptor_stat.st_ino
        ):
            raise FloorResolutionError(
                f"tick-settlement authority changed while reading: {display_path}"
            )
    finally:
        os.close(fd)
    try:
        data = json.loads(b"".join(chunks))
    except json.JSONDecodeError as exc:
        raise FloorResolutionError(
            f"tick-settlement authority is not valid JSON: {display_path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise FloorResolutionError(
            f"tick-settlement authority must contain an object: {display_path}"
        )
    val = data.get("mean_tick_minus_bar_COMB")
    if (
        not isinstance(val, (int, float))
        or isinstance(val, bool)
        or not math.isfinite(float(val))
    ):
        raise FloorResolutionError(
            f"{display_path}: mean_tick_minus_bar_COMB must be a finite non-boolean number"
        )
    verdict = data.get("verdict")
    validated = bool(
        isinstance(verdict, dict)
        and verdict.get("tick_settlement_preserves_edge") is True
    )
    return float(val), validated


def _load_ticksettle(path: Path) -> tuple[float, bool]:
    raw = Path(path)
    if ".." in raw.parts:
        raise FloorResolutionError(f"tick-settlement authority contains '..': {path}")
    absolute = raw if raw.is_absolute() else Path.cwd() / raw
    directory, directory_fd = _open_real_directory(absolute.parent)
    try:
        return _load_ticksettle_at(directory_fd, absolute.name, directory / absolute.name)
    finally:
        os.close(directory_fd)


def recompute_fallback_max(ticksettle_dir: Path = TICKSETTLE_DIR) -> tuple[float, str, str]:
    """Rule-4 fallback: max |mean_tick_minus_bar_COMB| over validated on-disk
    ticksettle result JSONs, recomputed at resolution time — never a frozen constant."""
    best: tuple[float, str, str] | None = None
    directory, directory_fd = _open_real_directory(Path(ticksettle_dir))
    try:
        names = sorted(
            name
            for name in os.listdir(directory_fd)
            if name.endswith("_15m_ticksettle_result.json")
        )
        for name in names:
            path = directory / name
            raw, validated = _load_ticksettle_at(directory_fd, name, path)
            if not validated:
                continue
            pair = name.split("_15m_ticksettle_result.json")[0].upper()
            if best is None or abs(raw) > best[0]:
                source = (
                    str(path.relative_to(REPO_ROOT))
                    if path.is_relative_to(REPO_ROOT)
                    else str(path)
                )
                best = (abs(raw), pair, source)
    finally:
        os.close(directory_fd)
    if best is None:
        raise FloorResolutionError(f"no validated ticksettle result JSONs under {ticksettle_dir}")
    return best


def resolve_haircut(
    pair: str,
    book_id: str,
    index_metrics: dict[str, Any],
    manifest_metrics: dict[str, Any],
    strategy: dict[str, Any],
    *,
    ticksettle_dir: Path = TICKSETTLE_DIR,
    proxy_table: dict[str, dict[str, Any]] | None = None,
) -> tuple[float, float, str, str]:
    """Return (raw_signed, abs_applied, source_rule, source_detail) per precedence.

    1. structured — haircut keys in INDEX metrics / manifest metrics / strategy deploy
    2. direct_ticksettle_json — the pair's own validated ticksettle result JSON
    3. documented_proxy — ledger-documented trusted proxy for that pair
    4. fallback_max_validated_fx_ticksettle — recomputed max over validated JSONs
    """
    proxies = DOCUMENTED_PROXY_HAIRCUTS if proxy_table is None else proxy_table

    found = _structured_haircuts(f"books/INDEX.json[{book_id}].metrics", index_metrics)
    found += _structured_haircuts(f"{book_id}.manifest metrics", manifest_metrics)
    deploy = strategy.get("deploy")
    if isinstance(deploy, dict):
        found += _structured_haircuts(f"{book_id} strategy.deploy", deploy)
    if found:
        values = {round(v, 12) for v, _ in found}
        if len(values) != 1:
            raise FloorResolutionError(f"{pair}: conflicting structured haircuts: {found}")
        raw = found[0][0]
        return raw, abs(raw), "structured", "; ".join(d for _, d in found)

    direct = ticksettle_dir / f"{pair.lower()}_15m_ticksettle_result.json"
    if direct.is_symlink() or direct.exists():
        raw, validated = _load_ticksettle(direct)
        if not validated:
            raise FloorResolutionError(f"{pair}: {direct} verdict does not preserve edge; refusing haircut")
        rel = str(direct.relative_to(REPO_ROOT)) if direct.is_relative_to(REPO_ROOT) else str(direct)
        return raw, abs(raw), "direct_ticksettle_json", f"{rel}[mean_tick_minus_bar_COMB]"

    if pair in proxies:
        raw = float(proxies[pair]["raw"])
        return raw, abs(raw), "documented_proxy", str(proxies[pair]["detail"])

    max_abs, contributor, path = recompute_fallback_max(ticksettle_dir)
    return -max_abs, max_abs, "fallback_max_validated_fx_ticksettle", (
        f"recomputed at resolution time over {ticksettle_dir.name}/*_15m_ticksettle_result.json; "
        f"contributor {contributor} ({path})"
    )


def resolve_vintage(book_id: str, manifest: dict[str, Any]) -> tuple[str, str]:
    created = manifest.get("created_utc")
    if isinstance(created, str) and created:
        return created, "manifest created_utc"
    if book_id in LEDGER_FREEZE_DATES:
        date, detail = LEDGER_FREEZE_DATES[book_id]
        return date, detail
    return "vintage_date_unavailable", "manifest created_utc null and no ledger freeze date recorded"


def resolve_book(
    pair: str,
    book_id: str,
    index_entry: dict[str, Any],
    manifest: dict[str, Any],
    strategy: dict[str, Any],
    strategy_path: str,
    *,
    ticksettle_dir: Path = TICKSETTLE_DIR,
    proxy_table: dict[str, dict[str, Any]] | None = None,
) -> PairFloorResolution:
    """Pure resolution from already-loaded registry dicts (fixture-testable)."""
    try:
        require_active_book_lifecycle(book_id, index_entry, manifest)
    except BookRuntimeError as exc:
        raise FloorResolutionError(str(exc)) from exc
    index_metrics = index_entry.get("metrics")
    if not isinstance(index_metrics, dict):
        raise FloorResolutionError(f"{book_id}: books/INDEX.json entry has no metrics dict")
    manifest_metrics = manifest.get("metrics")
    if not isinstance(manifest_metrics, dict):
        raise FloorResolutionError(f"{book_id}: manifest has no metrics dict")

    try:
        coverage, coverage_source = _resolve_coverage(book_id, strategy, index_metrics)
        if coverage is None:
            raise FloorResolutionError(f"{book_id}: no deploy coverage in strategy or registry")
        conf_thr = _conf_thr(strategy, coverage)
    except BookRuntimeError as exc:
        raise FloorResolutionError(str(exc)) from exc

    floors = normalize_floors(book_id, index_metrics, manifest_metrics, strategy)
    missing = [side for side in ("UP", "DOWN") if (coverage, side) not in floors]
    if missing:
        raise FloorResolutionError(
            f"{book_id}: no refit-CPCV p10 floor for sides {missing} at active coverage {coverage}; "
            f"available cells: {sorted(floors)}"
        )
    up = floors[(coverage, "UP")]
    down = floors[(coverage, "DOWN")]

    raw, abs_applied, rule, detail = resolve_haircut(
        pair, book_id, index_metrics, manifest_metrics, strategy,
        ticksettle_dir=ticksettle_dir, proxy_table=proxy_table,
    )
    vintage, vintage_source = resolve_vintage(book_id, manifest)
    content_id = index_metrics.get("content_id") or manifest.get("content_id")
    return PairFloorResolution(
        pair=pair,
        book_id=book_id,
        strategy_path=strategy_path,
        coverage=float(coverage),
        coverage_source=str(coverage_source),
        confidence_threshold=float(conf_thr),
        up_floor=up["value"],
        down_floor=down["value"],
        floor_source="; ".join(up["sources"] + down["sources"]),
        haircut_raw=raw,
        haircut_abs=abs_applied,
        haircut_source=rule,
        haircut_source_detail=detail,
        book_created_utc=vintage,
        vintage_source=vintage_source,
        registry_content_id=str(content_id) if content_id is not None else None,
    )


def resolve_enabled_pairs(pairs: list[str] | None = None) -> dict[str, PairFloorResolution]:
    """Resolve floors for Deriv-enabled pairs. Domain = DEFAULT_ENABLED_PAIRS;
    EURUSD is excluded (executor hard-raises on it) and never blocks startup."""
    selected = list(pairs) if pairs is not None else list(DEFAULT_ENABLED_PAIRS)
    unknown = [p for p in selected if p not in DEFAULT_ENABLED_PAIRS]
    if unknown:
        raise FloorResolutionError(f"pairs outside the enabled Deriv domain: {unknown}")
    index = load_index()
    out: dict[str, PairFloorResolution] = {}
    for pair in selected:
        book_id = TARGET_BOOKS[pair]
        if book_id not in index:
            raise FloorResolutionError(f"{book_id} is not registered in books/INDEX.json")
        entry = index[book_id]
        try:
            _, manifest = load_registered_manifest(book_id, entry)
            require_active_book_lifecycle(book_id, entry, manifest)
            strategy_path = load_registered_strategy_path(book_id)
        except BookRuntimeError as exc:
            raise FloorResolutionError(str(exc)) from exc
        # Manifests' strategy_json entries record a stale pre-reorg models/ path;
        # the strategy of record is the single *strategy*.json inside books/<id>/.
        try:
            strategy = json.loads(strategy_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise FloorResolutionError(
                f"{book_id}: cannot read strategy JSON: {exc}"
            ) from exc
        if not isinstance(strategy, dict):
            raise FloorResolutionError(f"{book_id}: strategy JSON must contain an object")
        out[pair] = resolve_book(
            pair, book_id, entry, manifest, strategy,
            str(strategy_path.relative_to(REPO_ROOT)),
        )
    return out


def inventory_rows(pairs: list[str] | None = None) -> list[dict[str, Any]]:
    rows = [r.as_dict() for r in resolve_enabled_pairs(pairs).values()]
    max_abs, contributor, path = recompute_fallback_max()
    rows.append(
        {
            "rule4_fallback_max_abs": max_abs,
            "rule4_contributor": contributor,
            "rule4_source": path,
            "note": "rule-4 fallback recomputed from on-disk validated ticksettle JSONs at resolution time",
        }
    )
    return rows


if __name__ == "__main__":
    print(json.dumps(inventory_rows(), indent=2))
