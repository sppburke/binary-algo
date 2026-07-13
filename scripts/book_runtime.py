"""Frozen 15m book discovery, schema normalization, and scoring."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

import harness


REPO_ROOT = Path(__file__).resolve().parents[1]
BOOKS_DIR = REPO_ROOT / "books"
INDEX_PATH = BOOKS_DIR / "INDEX.json"

TARGET_BOOKS = {
    "EURUSD": "EURUSD.m15xp.v1",
    "USDCHF": "USDCHF.m15ny_xpair_seedens.v1",
    "GBPUSD": "GBPUSD.m15ny_xpair_seedens8.v1",
    "USDJPY": "USDJPY.m15ny_seedens.v1",
    "USDCAD": "USDCAD.m15ny_seedens.v1",
    "AUDUSD": "AUDUSD.m15ny_seedens.v1",
    "NZDUSD": "NZDUSD.m15ny_seedens.v1",
}

# Compatibility is deliberately tied to these seven historical byte bundles,
# not to TARGET_BOOKS (which is an active mapping and may change later).  The
# old manifests/index rows predate lifecycle metadata; no prefix or version
# inference may silently grant the same exception to another book.
LEGACY_ACTIVE_BOOK_IDS = frozenset(
    {
        "EURUSD.m15xp.v1",
        "USDCHF.m15ny_xpair_seedens.v1",
        "GBPUSD.m15ny_xpair_seedens8.v1",
        "USDJPY.m15ny_seedens.v1",
        "USDCAD.m15ny_seedens.v1",
        "AUDUSD.m15ny_seedens.v1",
        "NZDUSD.m15ny_seedens.v1",
    }
)
ACTIVE_LIFECYCLE = "active"
INACTIVE_CANDIDATE_LIFECYCLE = "inactive_shadow_candidate"
CANDIDATE_IDENTITY_FIELDS = frozenset({"bundle_id", "prereg_id", "run_id"})


class BookRuntimeError(RuntimeError):
    pass


def _checked_books_dir() -> Path:
    if BOOKS_DIR.is_symlink() or not BOOKS_DIR.is_dir():
        raise BookRuntimeError(f"books directory is missing or symlinked: {BOOKS_DIR}")
    try:
        resolved = BOOKS_DIR.resolve(strict=True)
        expected = REPO_ROOT.resolve(strict=True) / "books"
    except OSError as exc:
        raise BookRuntimeError(f"cannot resolve books directory: {exc}") from exc
    if resolved != expected:
        raise BookRuntimeError(f"books directory escapes repository root: {BOOKS_DIR}")
    return resolved


def _checked_book_id(book_id: str) -> str:
    if (
        not isinstance(book_id, str)
        or not book_id
        or Path(book_id).name != book_id
        or book_id in {".", ".."}
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", book_id) is None
    ):
        raise BookRuntimeError(f"invalid registered book id {book_id!r}")
    return book_id


def _checked_book_dir(book_id: str) -> Path:
    bid = _checked_book_id(book_id)
    books = _checked_books_dir()
    candidate = BOOKS_DIR / bid
    if candidate.is_symlink() or not candidate.is_dir():
        raise BookRuntimeError(f"{bid}: book directory is missing or symlinked: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise BookRuntimeError(f"{bid}: cannot resolve book directory: {exc}") from exc
    if resolved.parent != books or resolved.name != bid:
        raise BookRuntimeError(f"{bid}: book directory escapes the exact books parent")
    return resolved


def _checked_book_file(book_dir: Path, path: Path, *, kind: str) -> Path:
    if path.parent != book_dir or path.is_symlink() or not path.is_file():
        raise BookRuntimeError(f"{book_dir.name}: {kind} is not a direct regular file: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise BookRuntimeError(f"{book_dir.name}: cannot resolve {kind}: {path}") from exc
    if resolved.parent != book_dir:
        raise BookRuntimeError(f"{book_dir.name}: {kind} escapes the exact book directory: {path}")
    return resolved


def load_registered_strategy_path(book_id: str) -> Path:
    """Resolve the sole strategy through the same strict path guard as active loading."""

    book_dir = _checked_book_dir(book_id)
    candidates = sorted(book_dir.glob("*strategy*.json"))
    if len(candidates) != 1:
        raise BookRuntimeError(
            f"{book_id}: expected exactly one strategy JSON, found {len(candidates)}"
        )
    return _checked_book_file(book_dir, candidates[0], kind="strategy JSON")


def _load_registered_model_paths(book_id: str) -> list[Path]:
    book_dir = _checked_book_dir(book_id)
    candidates = sorted(book_dir.glob("*.txt"))
    if not candidates:
        raise BookRuntimeError(f"{book_id}: no LightGBM text models found")
    return [
        _checked_book_file(book_dir, path, kind="LightGBM model")
        for path in candidates
    ]


def _valid_bundle_id(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _require_matching_candidate_identity(
    book_id: str,
    index_entry: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    if "bundle_id" not in index_entry or "bundle_id" not in manifest:
        raise BookRuntimeError(
            f"{book_id}: candidate bundle_id is present on only one registry surface"
        )
    idx_bundle = index_entry["bundle_id"]
    man_bundle = manifest["bundle_id"]
    if not _valid_bundle_id(idx_bundle) or not _valid_bundle_id(man_bundle):
        raise BookRuntimeError(
            f"{book_id}: candidate bundle_id must be 64 lowercase hex characters"
        )
    if idx_bundle != man_bundle:
        raise BookRuntimeError(f"{book_id}: candidate bundle_id mismatch")
    for name in ("prereg_id", "run_id"):
        idx_value = index_entry.get(name)
        man_value = manifest.get(name)
        if not _valid_bundle_id(idx_value) or not _valid_bundle_id(man_value):
            raise BookRuntimeError(
                f"{book_id}: candidate {name} must be 64 lowercase hex characters"
            )
        if idx_value != man_value:
            raise BookRuntimeError(f"{book_id}: candidate {name} mismatch")


def require_active_book_lifecycle(
    book_id: str,
    index_entry: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    """Validate registry/manifest lifecycle before any active policy is read.

    The function is pure so both runtime loaders and fixture tests exercise one
    truth table.  Missing and explicit null are intentionally different.
    Returns ``legacy-active`` or ``active`` only for an allowed active book;
    every inactive, ambiguous, malformed, or unknown state raises.
    """
    if not isinstance(index_entry, dict) or not isinstance(manifest, dict):
        raise BookRuntimeError(f"{book_id}: index entry and manifest must be objects")
    if index_entry.get("id") != book_id:
        raise BookRuntimeError(f"{book_id}: INDEX id does not match requested book")
    if manifest.get("id") != book_id:
        raise BookRuntimeError(f"{book_id}: manifest id does not match requested book")

    idx_has_lifecycle = "lifecycle_status" in index_entry
    man_has_lifecycle = "lifecycle_status" in manifest
    active_identity_fields = sorted(
        (CANDIDATE_IDENTITY_FIELDS & set(index_entry))
        | (CANDIDATE_IDENTITY_FIELDS & set(manifest))
    )

    if not idx_has_lifecycle and not man_has_lifecycle:
        if book_id not in LEGACY_ACTIVE_BOOK_IDS:
            raise BookRuntimeError(
                f"{book_id}: lifecycle absent for a non-whitelisted legacy book"
            )
        if active_identity_fields:
            raise BookRuntimeError(
                f"{book_id}: legacy-active lifecycle cannot carry candidate identity fields "
                f"{active_identity_fields}"
            )
        if manifest.get("schema") != "book-manifest/v1":
            raise BookRuntimeError(
                f"{book_id}: whitelisted legacy manifest schema is malformed"
            )
        return "legacy-active"

    if idx_has_lifecycle != man_has_lifecycle:
        raise BookRuntimeError(f"{book_id}: lifecycle is present on only one registry surface")
    idx_lifecycle = index_entry["lifecycle_status"]
    man_lifecycle = manifest["lifecycle_status"]
    if not isinstance(idx_lifecycle, str) or not isinstance(man_lifecycle, str):
        raise BookRuntimeError(f"{book_id}: lifecycle must be a non-null string")
    if idx_lifecycle != man_lifecycle:
        raise BookRuntimeError(
            f"{book_id}: lifecycle mismatch: INDEX {idx_lifecycle!r} != manifest {man_lifecycle!r}"
        )

    manifest_schema = manifest.get("schema")
    if idx_lifecycle == ACTIVE_LIFECYCLE:
        if manifest_schema == "book-manifest/v1":
            if active_identity_fields:
                raise BookRuntimeError(
                    f"{book_id}: active book-manifest/v1 cannot carry candidate "
                    f"identity fields {active_identity_fields}"
                )
        elif manifest_schema == "book-manifest/candidate-v1":
            _require_matching_candidate_identity(book_id, index_entry, manifest)
        else:
            raise BookRuntimeError(
                f"{book_id}: explicit active lifecycle manifest schema is malformed"
            )
        return ACTIVE_LIFECYCLE

    if idx_lifecycle == INACTIVE_CANDIDATE_LIFECYCLE:
        if manifest_schema != "book-manifest/candidate-v1":
            raise BookRuntimeError(
                f"{book_id}: inactive candidate manifest schema is malformed"
            )
        _require_matching_candidate_identity(book_id, index_entry, manifest)
        raise BookRuntimeError(f"{book_id}: inactive shadow candidate cannot be loaded as active")
    raise BookRuntimeError(f"{book_id}: unknown lifecycle {idx_lifecycle!r}")


@dataclass(frozen=True)
class BookScore:
    pair: str
    book_id: str
    proba: float
    confidence: float
    threshold: float
    direction: str
    gate_passed: bool
    gate_reasons: list[str]


@dataclass(frozen=True)
class BookGateComponents:
    """Row-aligned model-owned gate decisions for a loaded book.

    Confidence and structural decisions stay separate so parity callers cannot
    hide one component crossing behind a still-false combined mask.
    """

    confidence_passed: np.ndarray
    structural_passed: np.ndarray

    def __post_init__(self) -> None:
        confidence = np.asarray(self.confidence_passed)
        structural = np.asarray(self.structural_passed)
        if (
            confidence.ndim != 1
            or structural.ndim != 1
            or confidence.dtype != np.bool_
            or structural.dtype != np.bool_
            or confidence.shape != structural.shape
        ):
            raise BookRuntimeError("gate components must be aligned boolean vectors")
        confidence = np.array(confidence, dtype=bool, copy=True)
        structural = np.array(structural, dtype=bool, copy=True)
        confidence.setflags(write=False)
        structural.setflags(write=False)
        object.__setattr__(self, "confidence_passed", confidence)
        object.__setattr__(self, "structural_passed", structural)

    @property
    def gate_passed(self) -> np.ndarray:
        combined = self.confidence_passed & self.structural_passed
        combined.setflags(write=False)
        return combined


@dataclass
class LoadedBook:
    pair: str
    book_id: str
    book_dir: Path
    strategy_path: Path
    model_paths: list[Path]
    strategy: dict[str, Any]
    feature_cols: list[str]
    boosters: list[lgb.Booster]
    conf_thr: float
    content_id: str
    coverage: float | None = None
    coverage_source: str | None = None

    def _ordered_feature_frame(self, feature_rows: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(feature_rows, pd.DataFrame):
            raise BookRuntimeError(
                f"{self.book_id}: batch scoring requires a pandas DataFrame"
            )
        if feature_rows.columns.has_duplicates:
            raise BookRuntimeError(f"{self.book_id}: duplicate feature columns")
        missing = [column for column in self.feature_cols if column not in feature_rows.columns]
        if missing:
            raise BookRuntimeError(
                f"{self.book_id}: missing feature columns: {missing[:12]}"
            )
        try:
            ordered = feature_rows.loc[:, self.feature_cols].astype(
                "float64", copy=False
            )
        except (TypeError, ValueError) as exc:
            raise BookRuntimeError(
                f"{self.book_id}: feature values are not numeric"
            ) from exc
        finite = np.isfinite(ordered.to_numpy(dtype="float64", copy=False))
        if not finite.all():
            bad_columns = ordered.columns[np.any(~finite, axis=0)].tolist()
            raise BookRuntimeError(
                f"{self.book_id}: non-finite feature values: {bad_columns[:12]}"
            )
        # Reconstruct the exact float64, column-ordered DataFrame supplied by
        # the historical scalar scorer instead of exposing a caller's dtype or
        # extra-column layout to LightGBM.
        return pd.DataFrame(
            ordered.to_numpy(dtype="float64", copy=False),
            index=ordered.index,
            columns=self.feature_cols,
            copy=False,
        )

    def predict_probabilities(self, feature_rows: pd.DataFrame) -> np.ndarray:
        """Score ordered rows through the canonical loaded-book ensemble path."""

        frame = self._ordered_feature_frame(feature_rows)
        if not self.boosters:
            raise BookRuntimeError(f"{self.book_id}: no loaded boosters")
        if frame.empty:
            probabilities = np.empty(0, dtype="float64")
        else:
            predictions = np.asarray(
                [
                    np.asarray(booster.predict(frame), dtype="float64")
                    for booster in self.boosters
                ],
                dtype="float64",
            )
            if predictions.shape != (len(self.boosters), len(frame)):
                raise BookRuntimeError(
                    f"{self.book_id}: booster prediction shape is not row-aligned"
                )
            probabilities = predictions.mean(axis=0)
        if probabilities.shape != (len(frame),) or not np.isfinite(probabilities).all():
            raise BookRuntimeError(
                f"{self.book_id}: ensemble returned non-finite or misaligned probabilities"
            )
        probabilities = np.asarray(probabilities, dtype="float64")
        probabilities.setflags(write=False)
        return probabilities

    def gate_components(
        self,
        feature_rows: pd.DataFrame,
        probabilities: np.ndarray | list[float],
    ) -> BookGateComponents:
        """Return separate confidence and structural masks in input row order."""

        if not isinstance(feature_rows, pd.DataFrame):
            raise BookRuntimeError(
                f"{self.book_id}: batch gates require a pandas DataFrame"
            )
        if feature_rows.columns.has_duplicates:
            raise BookRuntimeError(f"{self.book_id}: duplicate feature columns")
        p = np.asarray(probabilities)
        if (
            p.ndim != 1
            or len(p) != len(feature_rows)
            or not np.issubdtype(p.dtype, np.floating)
            or not np.isfinite(p).all()
        ):
            raise BookRuntimeError(
                f"{self.book_id}: gate probabilities must be aligned finite floats"
            )
        confidence = np.abs(p.astype("float64", copy=False) - 0.5)
        return self._gate_components_from_confidence(feature_rows, confidence)

    def _gate_components_from_confidence(
        self,
        feature_rows: pd.DataFrame,
        confidence: np.ndarray | list[float],
    ) -> BookGateComponents:
        if not isinstance(feature_rows, pd.DataFrame):
            raise BookRuntimeError(
                f"{self.book_id}: batch gates require a pandas DataFrame"
            )
        if feature_rows.columns.has_duplicates:
            raise BookRuntimeError(f"{self.book_id}: duplicate feature columns")
        conf = np.asarray(confidence)
        if (
            conf.ndim != 1
            or len(conf) != len(feature_rows)
            or not np.issubdtype(conf.dtype, np.floating)
        ):
            raise BookRuntimeError(
                f"{self.book_id}: gate confidence must be an aligned floating vector"
            )
        # This is deliberately the negation of the historical failure
        # predicate, rather than ``>=``: the private scalar gate's behavior for
        # unusual NaN confidence inputs remains unchanged.
        confidence_passed = ~(conf.astype("float64", copy=False) < float(self.conf_thr))
        structural_passed = np.ones(len(feature_rows), dtype=bool)
        bb_thr = _first_number(self.strategy, ("bb_width_thr", "bbw_thr"))
        gate = self.strategy.get("gate")
        if bb_thr is None and isinstance(gate, dict):
            bb_thr = _first_number(gate, ("bb_width_thr", "bbw_thr"))
        if bb_thr is not None:
            if "15m_bb_width" not in feature_rows.columns:
                structural_passed[:] = False
            else:
                try:
                    width = feature_rows["15m_bb_width"].astype("float64").to_numpy()
                except (TypeError, ValueError) as exc:
                    raise BookRuntimeError(
                        f"{self.book_id}: 15m_bb_width is not numeric"
                    ) from exc
                # Preserve the historical failure predicate exactly.  In
                # particular, an extra (non-model) NaN width did not satisfy
                # ``width > threshold`` and therefore was not rejected.
                structural_passed = ~(width > float(bb_thr))
        return BookGateComponents(confidence_passed, structural_passed)

    def score(self, feature_row: pd.Series | dict[str, Any]) -> BookScore:
        row = pd.Series(feature_row, dtype="float64")
        missing = [c for c in self.feature_cols if c not in row.index]
        if missing:
            raise BookRuntimeError(f"{self.book_id}: missing feature columns: {missing[:12]}")
        x = row[self.feature_cols].astype("float64")
        bad = x.index[~np.isfinite(x.values)].tolist()
        if bad:
            raise BookRuntimeError(f"{self.book_id}: non-finite feature values: {bad[:12]}")
        frame = pd.DataFrame([x.values], columns=self.feature_cols)
        proba = float(self.predict_probabilities(frame)[0])
        conf = abs(proba - 0.5)
        direction = "UP" if proba >= 0.5 else "DOWN"
        passed, reasons = self._gate(row, conf)
        return BookScore(
            pair=self.pair,
            book_id=self.book_id,
            proba=proba,
            confidence=conf,
            threshold=self.conf_thr,
            direction=direction,
            gate_passed=passed,
            gate_reasons=reasons,
        )

    def _gate(self, row: pd.Series, conf: float) -> tuple[bool, list[str]]:
        frame = pd.DataFrame([row.to_numpy(copy=True)], columns=row.index)
        # Delegate both decisions while preserving the private method's
        # caller-provided confidence semantics exactly.
        components = self._gate_components_from_confidence(
            frame, np.asarray([float(conf)], dtype="float64")
        )
        reasons: list[str] = []
        if not components.confidence_passed[0]:
            reasons.append(f"confidence {conf:.8f} < threshold {self.conf_thr:.8f}")
        bb_thr = _first_number(self.strategy, ("bb_width_thr", "bbw_thr"))
        gate = self.strategy.get("gate")
        if bb_thr is None and isinstance(gate, dict):
            bb_thr = _first_number(gate, ("bb_width_thr", "bbw_thr"))
        if bb_thr is not None:
            if "15m_bb_width" not in row.index:
                reasons.append("missing 15m_bb_width for strategy compression gate")
            elif not components.structural_passed[0]:
                reasons.append(f"15m_bb_width {float(row['15m_bb_width']):.8f} > threshold {bb_thr:.8f}")
        return (len(reasons) == 0), reasons


def load_index() -> dict[str, dict[str, Any]]:
    _checked_books_dir()
    if INDEX_PATH.is_symlink() or not INDEX_PATH.is_file():
        raise BookRuntimeError(f"books/INDEX.json is missing or symlinked: {INDEX_PATH}")
    try:
        index_path = INDEX_PATH.resolve(strict=True)
    except OSError as exc:
        raise BookRuntimeError(f"cannot resolve books/INDEX.json: {exc}") from exc
    if index_path.parent != BOOKS_DIR.resolve(strict=True):
        raise BookRuntimeError("books/INDEX.json escapes the exact books directory")
    try:
        raw = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BookRuntimeError(f"cannot read books/INDEX.json: {exc}") from exc
    books = raw.get("books")
    if raw.get("schema") != "book-index/v1" or not isinstance(books, list):
        raise BookRuntimeError("books/INDEX.json must contain a top-level books list")
    out: dict[str, dict[str, Any]] = {}
    for pos, entry in enumerate(books):
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise BookRuntimeError(f"books/INDEX.json books[{pos}] must have a string id")
        book_id = entry["id"]
        if book_id in out:
            raise BookRuntimeError(f"books/INDEX.json contains duplicate id {book_id}")
        out[book_id] = entry
    return out


def load_registered_manifest(book_id: str, index_entry: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Read only the manifest named by INDEX, constrained to this repository."""
    _checked_book_id(book_id)
    _checked_books_dir()
    manifest_rel = index_entry.get("manifest")
    if not isinstance(manifest_rel, str) or not manifest_rel:
        raise BookRuntimeError(f"{book_id}: INDEX entry has no manifest path")
    rel = Path(manifest_rel)
    if rel.is_absolute() or ".." in rel.parts:
        raise BookRuntimeError(f"{book_id}: manifest path must be repo-relative without '..'")
    if "lifecycle_status" in index_entry:
        expected_rel = Path("books") / book_id / f"{book_id}.manifest.json"
        _checked_book_dir(book_id)
    else:
        expected_rel = Path("books") / f"{book_id}.manifest.json"
    if rel != expected_rel:
        raise BookRuntimeError(
            f"{book_id}: manifest path {manifest_rel!r} != required {expected_rel.as_posix()!r}"
        )
    manifest_path = REPO_ROOT / rel
    try:
        resolved = manifest_path.resolve(strict=True)
        resolved.relative_to(REPO_ROOT.resolve())
    except (OSError, ValueError) as exc:
        raise BookRuntimeError(f"{book_id}: invalid manifest path {manifest_rel!r}") from exc
    if manifest_path.is_symlink() or not resolved.is_file():
        raise BookRuntimeError(f"{book_id}: manifest is not a regular file: {manifest_rel}")
    try:
        manifest = json.loads(resolved.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BookRuntimeError(f"{book_id}: cannot read manifest {manifest_rel}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise BookRuntimeError(f"{book_id}: manifest must contain a JSON object")
    return resolved, manifest


def load_target_books(pairs: list[str] | None = None) -> dict[str, LoadedBook]:
    selected = pairs or list(TARGET_BOOKS)
    return {pair: load_book(pair, TARGET_BOOKS[pair]) for pair in selected}


def load_book(pair: str, book_id: str) -> LoadedBook:
    index = load_index()
    if book_id not in index:
        raise BookRuntimeError(f"{book_id} is not registered in books/INDEX.json")
    entry = index[book_id]
    _, manifest = load_registered_manifest(book_id, entry)
    require_active_book_lifecycle(book_id, entry, manifest)
    book_dir = _checked_book_dir(book_id)
    strategy_path = load_registered_strategy_path(book_id)
    model_paths = _load_registered_model_paths(book_id)
    try:
        strategy = json.loads(strategy_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BookRuntimeError(f"{book_id}: cannot read strategy JSON: {exc}") from exc
    if not isinstance(strategy, dict):
        raise BookRuntimeError(f"{book_id}: strategy JSON must contain an object")
    boosters = [lgb.Booster(model_file=str(p)) for p in model_paths]
    feature_cols = _feature_cols(pair, strategy, boosters)
    counts = {b.num_feature() for b in boosters}
    if counts != {len(feature_cols)}:
        raise BookRuntimeError(f"{book_id}: model feature counts {sorted(counts)} != strategy columns {len(feature_cols)}")
    metrics = entry.get("metrics")
    coverage, coverage_source = _resolve_coverage(book_id, strategy, metrics if isinstance(metrics, dict) else {})
    conf_thr = _conf_thr(strategy, coverage)
    return LoadedBook(
        pair=pair,
        book_id=book_id,
        book_dir=book_dir,
        strategy_path=strategy_path,
        model_paths=model_paths,
        strategy=strategy,
        feature_cols=feature_cols,
        boosters=boosters,
        conf_thr=conf_thr,
        content_id=_content_id([strategy_path, *model_paths]),
        coverage=coverage,
        coverage_source=coverage_source,
    )


def _resolve_coverage(
    book_id: str,
    strategy: dict[str, Any],
    index_metrics: dict[str, Any],
) -> tuple[float | None, str | None]:
    """Resolve the deploy coverage for a book, cross-checking strategy vs registry.

    Frozen strategy JSONs must not be edited (that changes content_id), so registry
    deploy coverage lives in books/INDEX.json metrics (deploy_cov preferred, cov
    fallback). When both the strategy and the registry state a coverage they must
    agree exactly; a mismatch fails closed.
    """
    idx_cov = _first_number(index_metrics, ("deploy_cov", "cov"))
    strat_cov = _first_number(strategy, ("coverage", "cov"))
    strat_src = "strategy"
    for block_name in ("gate", "deploy"):
        block = strategy.get(block_name)
        if strat_cov is None and isinstance(block, dict):
            strat_cov = _first_number(block, ("coverage", "cov"))
            strat_src = f"strategy.{block_name}"
    if idx_cov is not None and strat_cov is not None and abs(idx_cov - strat_cov) > 1e-12:
        raise BookRuntimeError(
            f"{book_id}: coverage mismatch: books/INDEX.json {idx_cov} != {strat_src} {strat_cov}"
        )
    if strat_cov is not None:
        return float(strat_cov), strat_src
    if idx_cov is not None:
        return float(idx_cov), "books/INDEX.json metrics"
    return None, None


def _feature_cols(pair: str, strategy: dict[str, Any], boosters: list[lgb.Booster]) -> list[str]:
    for key in ("feature_cols", "primary_feats", "feature_names", "features"):
        vals = strategy.get(key)
        if isinstance(vals, list) and vals:
            return [str(v) for v in vals]
    base = harness.feature_cols(pair)
    counts = {b.num_feature() for b in boosters}
    if counts == {len(base)}:
        return base
    n_features = strategy.get("n_features")
    raise BookRuntimeError(f"cannot derive feature columns for {strategy.get('pair', pair)}: n_features={n_features}, model_counts={counts}")


def _conf_thr(strategy: dict[str, Any], coverage: float | None = None) -> float:
    val = _first_number(strategy, ("conf_thr", "confidence_threshold", "threshold", "thr"))
    gate = strategy.get("gate")
    if val is None and isinstance(gate, dict):
        val = _first_number(gate, ("conf_thr", "confidence_threshold", "threshold", "thr"))
    gates = strategy.get("gates")
    if val is None and isinstance(gates, dict) and gates:
        if coverage is not None:
            # Resolved deploy coverage must select its exact gates entry; the old
            # silent "0.02" fallback loaded cov2 thresholds for cov1 deploy books.
            matches = [k for k in gates if _looks_float(k) and float(k) == float(coverage)]
            if len(matches) != 1:
                raise BookRuntimeError(
                    f"no unique gates entry for deploy coverage {coverage}: keys={sorted(gates)}"
                )
            gate_obj = gates[matches[0]]
        else:
            gate_obj = gates.get("0.02")
            if gate_obj is None:
                numeric_keys = sorted((float(k), k) for k in gates if _looks_float(k))
                gate_obj = gates[numeric_keys[0][1]] if numeric_keys else next(iter(gates.values()))
        if isinstance(gate_obj, dict):
            val = _first_number(gate_obj, ("conf_thr", "confidence_threshold", "threshold", "thr"))
        elif isinstance(gate_obj, (int, float)):
            val = float(gate_obj)
    if val is None:
        raise BookRuntimeError("strategy has no confidence threshold")
    return float(val)


def _first_number(obj: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        val = obj.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _looks_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _content_id(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def book_inventory(pairs: list[str] | None = None) -> list[dict[str, Any]]:
    rows = []
    for book in load_target_books(pairs).values():
        rows.append(
            {
                "pair": book.pair,
                "book_id": book.book_id,
                "strategy": str(book.strategy_path.relative_to(REPO_ROOT)),
                "models": [str(p.relative_to(REPO_ROOT)) for p in book.model_paths],
                "n_features": len(book.feature_cols),
                "model_counts": [b.num_feature() for b in book.boosters],
                "conf_thr": book.conf_thr,
                "coverage": book.coverage,
                "coverage_source": book.coverage_source,
                "content_id": book.content_id,
            }
        )
    return rows


if __name__ == "__main__":
    print(json.dumps(book_inventory(), indent=2))
