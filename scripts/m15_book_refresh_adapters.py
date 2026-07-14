"""Thin, fail-closed adapters over the seven-pair feature-source universe.

The adapters normalize the existing research builders into one fitting and
calibration contract.  They intentionally do not reproduce any feature logic:

* EURUSD delegates to :mod:`m5_xpair` in ``MX_HOR=15``/``xpof`` mode.
* GBPUSD and USDCHF delegate to their import-safe frozen xpair ``build_mat``
  helpers.
* The four own-pair books delegate to their existing ``*_15m_base.build``
  helpers.

No result JSON, replay outcome, or command-line argument is read at import.
Timestamp defects are rejected before invoking legacy builders that otherwise
deduplicate inputs, and normalized rows are never sorted or deduplicated here.

Blind arm scoring uses :class:`ScoreRows` and :func:`build_score_rows`.  That
path projects only ordered covariates (plus the close series needed by the
public live cross-pair builder); it never requests or derives a label, forward
return, validity flag, or settlement outcome.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

import harness
from sessions import session_mask


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_DIR = REPO_ROOT / "features"
DEFAULT_ORDERFLOW_DIR = REPO_ROOT / "features_of"
HORIZON_SECONDS = 15 * 60
NS_PER_SECOND = 1_000_000_000
FEATURE_DECISION_SHIFT_SECONDS = 60
FIT_CAP = 150_000
DEFAULT_NUM_THREADS = 16
FEATURE_SOURCE_PAIR_ORDER = (
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD",
)
# Backward-compatible name for builder code.  This is a source-provider order,
# not the issue-#9 target-book order; USDCAD remains a required xpair covariate.
ALL_PAIRS = FEATURE_SOURCE_PAIR_ORDER
OWN_PAIR_MODULES = {
    "USDJPY": "usdjpy_15m_base",
    "USDCAD": "usdcad_15m_base",
    "AUDUSD": "audusd_15m_base",
    "NZDUSD": "nzdusd_15m_base",
}
FORBIDDEN_SCORE_COLUMNS = frozenset(
    {
        "y",
        "fwd_ret",
        "valid",
        "_fwd",
        "label",
        "moved",
        "outcome",
        "wc_ret",
        "win",
        "close",
    }
)


class AdapterError(ValueError):
    """The existing builder cannot satisfy the sealed refresh contract."""


@dataclass(frozen=True)
class AdapterContract:
    pair: str
    family: str
    n_features: int
    fit_stride: int
    seeds: tuple[int, ...]
    target_coverage: float
    num_leaves: int
    max_rounds: int
    early_stopping_rounds: int
    fit_session: str
    calibration_session: str
    stride_phase: int = 0
    fit_ties: str = "drop"
    early_stopping_ties: str = "drop"
    feature_dtype: str = "float32"
    replay_session: str = "dst_correct_ny"
    structural_column: str | None = None
    structural_quantile: float | None = None


CONTRACTS: dict[str, AdapterContract] = {
    "EURUSD": AdapterContract(
        "EURUSD", "eur_xpair_of", 331, 4, (0,), 0.10, 127, 3000, 150,
        "all", "legacy_fixed_utc_13:00_22:00",
        structural_column="15m_bb_width", structural_quantile=0.33,
    ),
    "GBPUSD": AdapterContract(
        "GBPUSD", "gbpchf_xpair", 340, 3, tuple(range(8)), 0.01, 255, 800, 80,
        "dst_correct_ny", "dst_correct_ny", early_stopping_ties="include_as_down",
    ),
    "USDCHF": AdapterContract(
        "USDCHF", "gbpchf_xpair", 337, 3, tuple(range(3)), 0.01, 255, 800, 80,
        "dst_correct_ny", "dst_correct_ny", early_stopping_ties="include_as_down",
    ),
    "USDJPY": AdapterContract(
        "USDJPY", "own_pair", 239, 6, tuple(range(3)), 0.02, 127, 3000, 150,
        "dst_correct_ny", "dst_correct_ny",
    ),
    "USDCAD": AdapterContract(
        "USDCAD", "own_pair", 239, 6, tuple(range(3)), 0.02, 127, 3000, 150,
        "dst_correct_ny", "dst_correct_ny",
    ),
    "AUDUSD": AdapterContract(
        "AUDUSD", "own_pair", 239, 6, tuple(range(3)), 0.03, 127, 3000, 150,
        "dst_correct_ny", "dst_correct_ny",
    ),
    "NZDUSD": AdapterContract(
        "NZDUSD", "own_pair", 239, 6, tuple(range(3)), 0.02, 127, 3000, 150,
        "dst_correct_ny", "dst_correct_ny",
    ),
}


@dataclass(frozen=True)
class AdapterRows:
    """Ordered rows returned by one incumbent-family builder.

    ``fit_row_id`` is the causal completed-minute decision timestamp: the
    left-labeled feature source index plus 60 seconds, encoded as signed int64
    Unix nanoseconds.  ``X`` is a C-contiguous float32 matrix in ``feature_cols``
    order.  Builders only return rows with a valid 15-minute label interval;
    ``moved`` distinguishes ties, which lose at evaluation and are excluded
    from model fitting for every supported target contract.
    """

    pair: str
    purpose: str
    feature_cols: tuple[str, ...]
    X: np.ndarray
    y: np.ndarray
    moved: np.ndarray
    fit_row_id: np.ndarray
    label_exit_ns: np.ndarray
    fit_session_mask: np.ndarray
    calibration_session_mask: np.ndarray
    replay_session_mask: np.ndarray
    source_feature_ns: np.ndarray | None = None
    builder_stride: int | None = None
    sealed_control_label_permutation: bool = False

    def __post_init__(self) -> None:
        pair = str(self.pair)
        if pair != pair.upper():
            raise AdapterError(f"adapter pair must be canonical uppercase: {pair!r}")
        contract = contract_for(pair)
        if self.purpose not in {"fit", "calibration", "replay", "parity"}:
            raise AdapterError(f"{pair}: unknown adapter purpose {self.purpose!r}")
        _validate_feature_contract(contract, self.feature_cols)
        n = len(self.fit_row_id)
        if self.X.shape != (n, len(self.feature_cols)):
            raise AdapterError(
                f"{self.pair}: matrix shape {self.X.shape} does not match "
                f"{n} rows x {len(self.feature_cols)} columns"
            )
        if self.X.dtype != np.dtype("float32") or not self.X.flags.c_contiguous:
            raise AdapterError(f"{self.pair}: adapter matrix must be C-contiguous float32")
        if len(set(self.feature_cols)) != len(self.feature_cols):
            raise AdapterError(f"{self.pair}: duplicate feature names")
        for name, values in (
            ("y", self.y),
            ("moved", self.moved),
            ("label_exit_ns", self.label_exit_ns),
            ("fit_session_mask", self.fit_session_mask),
            ("calibration_session_mask", self.calibration_session_mask),
            ("replay_session_mask", self.replay_session_mask),
        ):
            if len(values) != n:
                raise AdapterError(f"{self.pair}: {name} is not row-aligned")
        _validate_row_ids(self.fit_row_id, name=f"{self.pair} fit_row_id")
        shift_ns = np.int64(FEATURE_DECISION_SHIFT_SECONDS * NS_PER_SECOND)
        if self.source_feature_ns is None:
            raise AdapterError(f"{self.pair}: explicit source_feature_ns is required")
        source_feature_ns = np.asarray(self.source_feature_ns)
        _validate_row_ids(source_feature_ns, name=f"{self.pair} source_feature_ns")
        if not np.array_equal(self.fit_row_id, source_feature_ns + shift_ns):
            raise AdapterError(
                f"{self.pair}: decision timestamps are not source feature timestamps +60s"
            )
        object.__setattr__(self, "source_feature_ns", source_feature_ns)
        if self.label_exit_ns.dtype != np.dtype("int64"):
            raise AdapterError(f"{self.pair}: label_exit_ns must be signed int64")
        horizon_ns = np.int64(HORIZON_SECONDS * NS_PER_SECOND)
        if np.any(self.fit_row_id > np.iinfo(np.int64).max - horizon_ns):
            raise AdapterError(f"{self.pair}: label exit overflows signed int64 nanoseconds")
        if not np.array_equal(
            self.label_exit_ns,
            self.fit_row_id + horizon_ns,
        ):
            raise AdapterError(f"{self.pair}: label exits are not entry + 900 seconds")
        if self.y.dtype != np.dtype("uint8") or np.any((self.y != 0) & (self.y != 1)):
            raise AdapterError(f"{self.pair}: labels must be binary uint8")
        if self.moved.dtype != np.dtype("bool"):
            raise AdapterError(f"{self.pair}: moved mask must be boolean")
        if type(self.sealed_control_label_permutation) is not bool:
            raise AdapterError(
                f"{self.pair}: sealed_control_label_permutation must be boolean"
            )
        if self.sealed_control_label_permutation and self.purpose not in {
            "fit",
            "calibration",
        }:
            raise AdapterError(
                f"{self.pair}: sealed control label permutation is fit-only"
            )
        if (
            not self.sealed_control_label_permutation
            and contract.early_stopping_ties == "include_as_down"
            and np.any(self.y[~self.moved] != 0)
        ):
            raise AdapterError(
                f"{self.pair}: early-stopping ties must carry the DOWN label"
            )
        if (
            self.builder_stride is None
            or isinstance(self.builder_stride, bool)
            or not isinstance(self.builder_stride, (int, np.integer))
            or int(self.builder_stride) <= 0
        ):
            raise AdapterError(f"{self.pair}: explicit positive builder_stride is required")
        object.__setattr__(self, "builder_stride", int(self.builder_stride))
        for name in ("fit_session_mask", "calibration_session_mask", "replay_session_mask"):
            if getattr(self, name).dtype != np.dtype("bool"):
                raise AdapterError(f"{self.pair}: {name} must be boolean")

    @property
    def entry_ns(self) -> np.ndarray:
        return self.fit_row_id

    @property
    def fit_mask(self) -> np.ndarray:
        return self.fit_session_mask & self.moved

    @property
    def early_stopping_mask(self) -> np.ndarray:
        contract = contract_for(self.pair)
        if contract.early_stopping_ties == "include_as_down":
            return self.fit_session_mask.copy()
        if contract.early_stopping_ties == "drop":
            return self.fit_session_mask & self.moved
        raise AdapterError(
            f"{self.pair}: unknown early-stopping tie rule {contract.early_stopping_ties!r}"
        )

    @property
    def calibration_mask(self) -> np.ndarray:
        # Historical GBP/CHF/own-pair calibration includes valid-label ties;
        # EURUSD's builder has already removed ties.
        return self.calibration_session_mask.copy()


@dataclass(frozen=True)
class ScoreRows:
    """Full-resolution, outcome-free rows used to score a sealed book.

    ``entry_ns`` is the causal completed-minute close clock (the left-labeled
    source feature index plus 60 seconds) in signed-int64 UTC nanoseconds.
    The matrix retains warm-up/non-finite covariates so callers can apply the
    same explicit scoreability rule on both sides of a parity comparison.  No
    label, forward-return, validity, or settlement field is represented.
    """

    pair: str
    feature_cols: tuple[str, ...]
    X: np.ndarray
    entry_ns: np.ndarray
    source_feature_ns: np.ndarray | None = None

    def __post_init__(self) -> None:
        pair = str(self.pair)
        if pair != pair.upper():
            raise AdapterError(f"score pair must be canonical uppercase: {pair!r}")
        contract = contract_for(pair)
        _validate_feature_contract(contract, self.feature_cols)
        _validate_score_feature_names(self.feature_cols, pair=pair)
        n = len(self.entry_ns)
        if self.X.shape != (n, len(self.feature_cols)):
            raise AdapterError(
                f"{pair}: score matrix shape {self.X.shape} does not match "
                f"{n} rows x {len(self.feature_cols)} columns"
            )
        if self.X.dtype != np.dtype("float32") or not self.X.flags.c_contiguous:
            raise AdapterError(f"{pair}: score matrix must be C-contiguous float32")
        _validate_row_ids(self.entry_ns, name=f"{pair} score entry_ns")
        shift_ns = np.int64(FEATURE_DECISION_SHIFT_SECONDS * NS_PER_SECOND)
        if self.source_feature_ns is None:
            raise AdapterError(f"{pair}: explicit score source_feature_ns is required")
        source_feature_ns = np.asarray(self.source_feature_ns)
        _validate_row_ids(source_feature_ns, name=f"{pair} score source_feature_ns")
        if not np.array_equal(self.entry_ns, source_feature_ns + shift_ns):
            raise AdapterError(f"{pair}: score decision timestamps are not source +60s")
        object.__setattr__(self, "source_feature_ns", source_feature_ns)

    def __len__(self) -> int:
        return len(self.entry_ns)


@dataclass(frozen=True)
class SeedCheckpoint:
    pair: str
    seed: int
    best_iteration: int
    model_path: Path
    sha256: str


@dataclass(frozen=True)
class PolicyCalibration:
    pair: str
    target_coverage: float
    confidence_threshold: float
    calibration_rows: int
    confidence_rows: int
    structural_rows: int | None = None
    structural_column: str | None = None
    structural_quantile: float | None = None
    structural_threshold: float | None = None


def contract_for(pair: str) -> AdapterContract:
    try:
        return CONTRACTS[str(pair).upper()]
    except KeyError as exc:
        raise AdapterError(f"unsupported 15m refresh pair: {pair}") from exc


def build_rows(
    pair: str,
    years: Sequence[str | int],
    *,
    purpose: str = "replay",
    feature_dir: str | Path = DEFAULT_FEATURE_DIR,
    orderflow_dir: str | Path = DEFAULT_ORDERFLOW_DIR,
) -> AdapterRows:
    """Build one family without copying its feature or label implementation.

    ``purpose='fit'`` invokes the incumbent's historical training stride.
    Calibration, replay, and parity are always full resolution.  Exact UTC
    split boundaries are applied separately with :func:`window_mask`.
    """

    contract = contract_for(pair)
    normalized_years = _validate_years(years)
    if purpose not in {"fit", "calibration", "replay", "parity"}:
        raise AdapterError(f"unknown adapter purpose: {purpose}")
    stride = contract.fit_stride if purpose == "fit" else 1
    feature_path = Path(feature_dir).resolve()
    orderflow_path = Path(orderflow_dir).resolve()

    if contract.family == "eur_xpair_of":
        return build_eurusd_xpof_rows(
            normalized_years,
            purpose=purpose,
            stride=stride,
            feature_dir=feature_path,
            orderflow_dir=orderflow_path,
        )
    if contract.family == "gbpchf_xpair":
        return build_xpair_rows(
            contract.pair,
            normalized_years,
            purpose=purpose,
            stride=stride,
            feature_dir=feature_path,
        )
    return build_own_pair_rows(
        contract.pair,
        normalized_years,
        purpose=purpose,
        stride=stride,
        feature_dir=feature_path,
    )


def build_fit_rows(pair: str, years: Sequence[str | int], **kwargs: Any) -> AdapterRows:
    return build_rows(pair, years, purpose="fit", **kwargs)


def build_calibration_rows(pair: str, years: Sequence[str | int], **kwargs: Any) -> AdapterRows:
    return build_rows(pair, years, purpose="calibration", **kwargs)


def build_replay_rows(pair: str, years: Sequence[str | int], **kwargs: Any) -> AdapterRows:
    return build_rows(pair, years, purpose="replay", **kwargs)


def build_score_rows(
    pair: str,
    years: Sequence[str | int],
    feature_cols: Sequence[str],
    feature_dir: str | Path = DEFAULT_FEATURE_DIR,
    orderflow_dir: str | Path = DEFAULT_ORDERFLOW_DIR,
) -> ScoreRows:
    """Build full-resolution book covariates without outcome access.

    Own-pair books are projected directly from their feature parquet in the
    supplied book-column order.  Cross-pair books read only synchronized close
    series and delegate their derived block to the public
    :func:`live_features.build_cross_pair_features` recipe.  EURUSD additionally
    joins only the requested ``OF_*`` columns.  The path never requests
    ``y``, ``fwd_ret``, ``valid``, ``_fwd``, or any settlement field, and it
    deliberately does not trim rows based on future label availability.
    """

    pair = str(pair).upper()
    contract = contract_for(pair)
    years = _validate_years(years)
    cols = tuple(str(col) for col in feature_cols)
    _validate_feature_contract(contract, cols)
    _validate_score_feature_names(cols, pair=pair)
    feature_path = Path(feature_dir).resolve()
    orderflow_path = Path(orderflow_dir).resolve()

    # This import is intentionally local: adapter import remains free of live
    # client setup, while the scoring path is visibly bound to the public recipe.
    from live_features import build_cross_pair_features, is_base_feature

    base_cols = tuple(col for col in cols if is_base_feature(col))
    non_base_cols = tuple(col for col in cols if not is_base_feature(col))

    if contract.family == "own_pair":
        if non_base_cols:
            raise AdapterError(
                f"{pair}: own-pair score schema contains non-base columns "
                f"{list(non_base_cols[:12])}"
            )
        base = _read_covariate_years(feature_path, pair, years, cols)
        return _score_rows_from_frame(pair, cols, base)

    if contract.family not in {"eur_xpair_of", "gbpchf_xpair"}:
        raise AdapterError(f"{pair}: unsupported score adapter family {contract.family}")

    closes: dict[str, pd.Series] = {}
    for source_pair in ALL_PAIRS:
        close_frame = _read_covariate_years(
            feature_path, source_pair, years, ("close",), allow_close=True
        )
        close_values = close_frame["close"].to_numpy()
        try:
            finite_close = np.isfinite(close_values)
        except TypeError as exc:
            raise AdapterError(f"{source_pair}: close source is not numeric") from exc
        if not np.all(finite_close):
            raise AdapterError(f"{source_pair}: close source contains non-finite values")
        closes[source_pair] = close_frame["close"]

    # Source clocks were checked before this synchronized inner clock is formed.
    # Missing timestamps across instruments are normal market-data alignment,
    # whereas defects within any individual source fail above.
    close_matrix = pd.DataFrame(closes).dropna(how="any")
    _datetime_index_ns(close_matrix.index, name=f"{pair} synchronized close clock")
    try:
        cross = build_cross_pair_features(pair, close_matrix, list(cols))
    except Exception as exc:
        raise AdapterError(f"{pair}: public live cross-pair builder failed: {exc}") from exc
    cross_clock = _datetime_index_ns(cross.index, name=f"{pair} live cross-pair clock")
    close_clock = _datetime_index_ns(
        close_matrix.index, name=f"{pair} synchronized close clock"
    )
    if not np.array_equal(cross_clock, close_clock):
        raise AdapterError(f"{pair}: public live cross-pair builder changed the source clock")

    base = _read_covariate_years(feature_path, pair, years, base_cols)
    frame = cross.join(
        base.loc[:, [col for col in base_cols if col not in cross.columns]],
        how="left",
    )

    of_cols = tuple(col for col in cols if col.startswith("OF_"))
    if pair == "EURUSD":
        if not of_cols:
            raise AdapterError("EURUSD: xpof score schema contains no OF_* columns")
        orderflow = _read_covariate_years(orderflow_path, pair, years, of_cols)
        frame = frame.join(
            orderflow.loc[:, [col for col in of_cols if col not in frame.columns]],
            how="left",
        )
    elif of_cols:
        raise AdapterError(f"{pair}: unexpected OF_* columns in non-EUR score schema")

    missing = [col for col in cols if col not in frame.columns]
    if missing:
        raise AdapterError(f"{pair}: score builder is missing columns {missing[:12]}")
    if not frame.index.equals(cross.index):
        raise AdapterError(f"{pair}: covariate joins changed the synchronized score clock")
    return _score_rows_from_frame(pair, cols, frame.loc[:, list(cols)])


def build_eurusd_xpof_rows(
    years: Sequence[str],
    *,
    purpose: str = "replay",
    stride: int | None = None,
    feature_dir: str | Path = DEFAULT_FEATURE_DIR,
    orderflow_dir: str | Path = DEFAULT_ORDERFLOW_DIR,
) -> AdapterRows:
    years = _validate_years(years)
    feature_dir = Path(feature_dir).resolve()
    orderflow_dir = Path(orderflow_dir).resolve()
    stride = _resolve_stride(CONTRACTS["EURUSD"], purpose, stride)
    _validate_source_files(feature_dir, ALL_PAIRS, years)
    _validate_source_files(orderflow_dir, ("EURUSD",), years)
    xp = _import_m15_eur_builder()
    with _temporary_attrs(xp, FEAT=str(feature_dir), OFDIR=str(orderflow_dir)):
        frame, cols = _build_eurusd_xpof_provenance_units(xp, years, stride)
    contract = CONTRACTS["EURUSD"]
    _validate_feature_contract(contract, cols)
    ts_s = _frame_seconds(frame, "EURUSD")
    fwd = np.asarray(frame["_fwd"], dtype="float64")
    if np.any(~np.isfinite(fwd)) or np.any(fwd == 0.0):
        raise AdapterError("EURUSD: xpof builder must return finite, non-tie labels")
    fixed_utc_cal = np.asarray(frame["sess_ny"] > 0.5, dtype=bool)
    return _make_rows(
        contract,
        purpose,
        frame.loc[:, list(cols)].to_numpy(dtype="float32", copy=True),
        cols,
        fwd > 0.0,
        np.ones(len(frame), dtype=bool),
        ts_s,
        np.ones(len(frame), dtype=bool),
        fixed_utc_cal,
        stride,
    )


def _build_eurusd_xpof_provenance_units(
    builder: Any,
    years: Sequence[str | int],
    stride: int,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Build EURUSD as ordered, indivisible nominal-file provenance units.

    The legacy builder already constructs cross-pair labels one nominal year at
    a time.  Its multi-year ``augment`` path instead concatenates base and
    order-flow inputs before joining, which can create a many-to-many product
    where adjacent nominal files share a boundary clock.  Keep each label,
    base, and order-flow recipe in the same nominal provenance unit, then
    concatenate completed units without sorting or deduplicating.  The strict
    final clock check deliberately rejects any remaining overlap or reversal.
    """

    normalized_years = _validate_years(years)
    if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)) or stride <= 0:
        raise AdapterError("EURUSD: provenance-unit stride must be a positive integer")
    stride = int(stride)
    units: list[pd.DataFrame] = []
    expected_frame_columns: tuple[str, ...] | None = None
    expected_xpair_columns: tuple[str, ...] | None = None
    expected_feature_columns: tuple[str, ...] | None = None
    required_meta = {"_ts", "_fwd", "sess_ny"}
    for year in normalized_years:
        unit = builder.build_xp([year], stride)
        if not isinstance(unit, pd.DataFrame):
            raise AdapterError(f"EURUSD {year}: build_xp did not return a DataFrame")
        raw_xpair_columns = builder.xp_cols(unit)
        try:
            xpair_columns = tuple(str(value) for value in raw_xpair_columns)
        except TypeError as exc:
            raise AdapterError(f"EURUSD {year}: xpair columns are not iterable") from exc
        unit = builder.augment(unit, [year], "xpof")
        if not isinstance(unit, pd.DataFrame):
            raise AdapterError(f"EURUSD {year}: augment did not return a DataFrame")
        if unit.columns.has_duplicates:
            raise AdapterError(f"EURUSD {year}: augmented frame has duplicate columns")
        frame_columns = tuple(str(value) for value in unit.columns)
        missing_meta = sorted(required_meta.difference(frame_columns))
        if missing_meta:
            raise AdapterError(
                f"EURUSD {year}: augmented frame is missing metadata {missing_meta}"
            )
        raw_feature_columns = builder.feat_cols("xpof", unit, list(xpair_columns))
        try:
            feature_columns = tuple(str(value) for value in raw_feature_columns)
        except TypeError as exc:
            raise AdapterError(f"EURUSD {year}: feature columns are not iterable") from exc
        if any(column not in unit.columns for column in feature_columns):
            raise AdapterError(f"EURUSD {year}: feature schema is absent from augmented frame")
        _frame_seconds(unit, "EURUSD")
        if expected_frame_columns is None:
            expected_frame_columns = frame_columns
            expected_xpair_columns = xpair_columns
            expected_feature_columns = feature_columns
        elif (
            frame_columns != expected_frame_columns
            or xpair_columns != expected_xpair_columns
            or feature_columns != expected_feature_columns
        ):
            raise AdapterError(
                f"EURUSD {year}: nominal provenance-unit schema/order differs"
            )
        units.append(unit)
    if not units or expected_feature_columns is None:
        raise AdapterError("EURUSD: no nominal provenance units were constructed")
    frame = pd.concat(units, axis=0, copy=False)
    if tuple(str(value) for value in frame.columns) != expected_frame_columns:
        raise AdapterError("EURUSD: provenance-unit concatenation changed frame order")
    _frame_seconds(frame, "EURUSD")
    return frame, expected_feature_columns


def build_xpair_rows(
    pair: str,
    years: Sequence[str],
    *,
    purpose: str = "replay",
    stride: int | None = None,
    feature_dir: str | Path = DEFAULT_FEATURE_DIR,
) -> AdapterRows:
    pair = str(pair).upper()
    years = _validate_years(years)
    feature_dir = Path(feature_dir).resolve()
    contract = contract_for(pair)
    stride = _resolve_stride(contract, purpose, stride)
    _validate_source_files(feature_dir, ALL_PAIRS, years)
    if pair == "GBPUSD":
        frozen = importlib.import_module("gbpusd_15m_xpair_frozen")
    elif pair == "USDCHF":
        frozen = importlib.import_module("usdchf_15m_xpair_frozen")
    else:
        raise AdapterError(f"{pair}: not an xpair adapter")
    with _temporary_attrs(frozen.XP, FEAT=str(feature_dir)):
        X, fwd, ts_s, raw_cols = frozen.build_mat(list(years), stride)
    cols = tuple(str(c) for c in raw_cols)
    _validate_feature_contract(contract, cols)
    fwd = np.asarray(fwd, dtype="float64")
    if np.any(~np.isfinite(fwd)):
        raise AdapterError(f"{pair}: xpair builder returned a non-finite label")
    ts_s = _validate_epoch_seconds(ts_s, name=f"{pair} builder timestamps")
    moved = fwd != 0.0
    ny = np.asarray(session_mask(ts_s, "ny"), dtype=bool)
    return _make_rows(
        contract, purpose, X, cols, fwd > 0.0, moved, ts_s, ny, ny, stride
    )


def build_own_pair_rows(
    pair: str,
    years: Sequence[str],
    *,
    purpose: str = "replay",
    stride: int | None = None,
    feature_dir: str | Path = DEFAULT_FEATURE_DIR,
) -> AdapterRows:
    pair = str(pair).upper()
    years = _validate_years(years)
    feature_dir = Path(feature_dir).resolve()
    contract = contract_for(pair)
    stride = _resolve_stride(contract, purpose, stride)
    _validate_source_files(feature_dir, (pair,), years)
    try:
        base = importlib.import_module(OWN_PAIR_MODULES[pair])
    except KeyError as exc:
        raise AdapterError(f"{pair}: not an own-pair adapter") from exc
    with _temporary_attrs(base, FEAT=str(feature_dir)):
        frame, y, moved, ts_s = base.build(list(years), stride)
    cols = tuple(str(c) for c in harness.feature_cols(pair))
    _validate_feature_contract(contract, cols)
    ts_s = _validate_epoch_seconds(ts_s, name=f"{pair} builder timestamps")
    index_ns = _datetime_index_ns(frame.index, name=f"{pair} builder index")
    if not np.array_equal(index_ns, _seconds_to_ns(ts_s, name=f"{pair} builder timestamps")):
        raise AdapterError(f"{pair}: builder frame index and timestamp vector are not row-aligned")
    y = np.asarray(y)
    if np.any((y != 0) & (y != 1)):
        raise AdapterError(f"{pair}: own-pair builder returned non-binary labels")
    moved = np.asarray(moved, dtype=bool)
    ny = np.asarray(session_mask(ts_s, "ny"), dtype=bool)
    return _make_rows(
        contract,
        purpose,
        frame.loc[:, list(cols)].to_numpy(dtype="float32", copy=True),
        cols,
        y.astype(bool),
        moved,
        ts_s,
        ny,
        ny,
        stride,
    )


def direct_incumbent_builder_parity_report(
    adapter_rows: AdapterRows,
    years: Sequence[str | int],
    *,
    purpose: str,
    stride: int | None = None,
    feature_dir: str | Path = DEFAULT_FEATURE_DIR,
    orderflow_dir: str | Path = DEFAULT_ORDERFLOW_DIR,
    entry_at_or_after: str | pd.Timestamp | np.datetime64 | int | None = None,
    label_exit_before: str | pd.Timestamp | np.datetime64 | int | None = None,
    cap: int = FIT_CAP,
) -> dict[str, Any]:
    """Compare normalized rows with a fresh direct incumbent-builder call.

    This is intentionally separate from :func:`build_rows`: it invokes the
    family authority again and independently reconstructs its source clock,
    causal decision clock, ordered float32 matrix, labels/ties, legacy
    fitting/calibration masks, standardized replay mask, and capped fitting
    row IDs.  Optional UTC bounds apply only to the capped-row comparison.
    The function reports mismatches; malformed direct-builder output raises
    :class:`AdapterError` rather than being coerced or repaired.
    """

    if not isinstance(adapter_rows, AdapterRows):
        raise AdapterError("direct incumbent parity requires AdapterRows")
    pair = adapter_rows.pair
    contract = contract_for(pair)
    normalized_years = _validate_years(years)
    direct_stride = _resolve_stride(contract, purpose, stride)
    feature_path = Path(feature_dir).resolve()
    orderflow_path = Path(orderflow_dir).resolve()

    if contract.family == "eur_xpair_of":
        _validate_source_files(feature_path, ALL_PAIRS, normalized_years)
        _validate_source_files(orderflow_path, (pair,), normalized_years)
        builder = _import_m15_eur_builder()
        with _temporary_attrs(
            builder, FEAT=str(feature_path), OFDIR=str(orderflow_path)
        ):
            frame, direct_cols = _build_eurusd_xpof_provenance_units(
                builder, normalized_years, direct_stride
            )
        source_s = _frame_seconds(frame, pair)
        fwd = np.asarray(frame["_fwd"], dtype="float64")
        if np.any(~np.isfinite(fwd)) or np.any(fwd == 0.0):
            raise AdapterError(
                "EURUSD: direct xpof builder must return finite, non-tie labels"
            )
        direct_X = frame.loc[:, list(direct_cols)].to_numpy(
            dtype="float32", copy=True
        )
        direct_y = np.asarray(fwd > 0.0, dtype="uint8")
        direct_moved = np.ones(len(frame), dtype=bool)
        direct_fit_session = np.ones(len(frame), dtype=bool)
        direct_calibration_session = np.asarray(
            frame["sess_ny"] > 0.5, dtype=bool
        )
    elif contract.family == "gbpchf_xpair":
        _validate_source_files(feature_path, ALL_PAIRS, normalized_years)
        module_name = (
            "gbpusd_15m_xpair_frozen"
            if pair == "GBPUSD"
            else "usdchf_15m_xpair_frozen"
        )
        builder = importlib.import_module(module_name)
        with _temporary_attrs(builder.XP, FEAT=str(feature_path)):
            direct_X, fwd, source_s, raw_cols = builder.build_mat(
                list(normalized_years), direct_stride
            )
        direct_cols = tuple(str(value) for value in raw_cols)
        source_s = _validate_epoch_seconds(
            source_s, name=f"{pair} direct builder timestamps"
        )
        fwd = np.asarray(fwd, dtype="float64")
        if np.any(~np.isfinite(fwd)):
            raise AdapterError(f"{pair}: direct xpair builder returned a non-finite label")
        direct_X = np.ascontiguousarray(direct_X, dtype="float32")
        direct_y = np.asarray(fwd > 0.0, dtype="uint8")
        direct_moved = np.asarray(fwd != 0.0, dtype=bool)
        legacy_ny = np.asarray(session_mask(source_s, "ny"), dtype=bool)
        direct_fit_session = legacy_ny
        direct_calibration_session = legacy_ny.copy()
    elif contract.family == "own_pair":
        _validate_source_files(feature_path, (pair,), normalized_years)
        try:
            builder = importlib.import_module(OWN_PAIR_MODULES[pair])
        except KeyError as exc:  # pragma: no cover - guarded by contract table
            raise AdapterError(f"{pair}: no direct own-pair builder") from exc
        with _temporary_attrs(builder, FEAT=str(feature_path)):
            frame, raw_y, raw_moved, source_s = builder.build(
                list(normalized_years), direct_stride
            )
        direct_cols = tuple(str(value) for value in harness.feature_cols(pair))
        source_s = _validate_epoch_seconds(
            source_s, name=f"{pair} direct builder timestamps"
        )
        source_index_ns = _datetime_index_ns(
            frame.index, name=f"{pair} direct builder index"
        )
        if not np.array_equal(
            source_index_ns,
            _seconds_to_ns(source_s, name=f"{pair} direct builder timestamps"),
        ):
            raise AdapterError(
                f"{pair}: direct builder frame index and timestamps are not aligned"
            )
        raw_y = np.asarray(raw_y)
        if np.any((raw_y != 0) & (raw_y != 1)):
            raise AdapterError(f"{pair}: direct own-pair labels are not binary")
        direct_X = frame.loc[:, list(direct_cols)].to_numpy(
            dtype="float32", copy=True
        )
        direct_y = np.asarray(raw_y, dtype="uint8")
        direct_moved = np.asarray(raw_moved, dtype=bool)
        legacy_ny = np.asarray(session_mask(source_s, "ny"), dtype=bool)
        direct_fit_session = legacy_ny
        direct_calibration_session = legacy_ny.copy()
    else:  # pragma: no cover - closed contract table
        raise AdapterError(f"{pair}: unsupported direct-builder family {contract.family}")

    _validate_feature_contract(contract, direct_cols)
    direct_X = np.ascontiguousarray(direct_X, dtype="float32")
    row_count = len(source_s)
    if direct_X.shape != (row_count, len(direct_cols)):
        raise AdapterError(f"{pair}: direct builder matrix is not row-aligned")
    for name, values in (
        ("labels", direct_y),
        ("moved", direct_moved),
        ("fit session", direct_fit_session),
        ("calibration session", direct_calibration_session),
    ):
        if np.asarray(values).ndim != 1 or len(values) != row_count:
            raise AdapterError(f"{pair}: direct builder {name} is not row-aligned")
    if direct_y.dtype != np.dtype("uint8"):
        raise AdapterError(f"{pair}: direct builder labels are not uint8")
    if direct_moved.dtype != np.dtype("bool"):
        raise AdapterError(f"{pair}: direct builder moved mask is not boolean")
    if (
        contract.early_stopping_ties == "include_as_down"
        and np.any(direct_y[~direct_moved] != 0)
    ):
        raise AdapterError(f"{pair}: direct early-stopping ties are not DOWN")

    source_ns = _seconds_to_ns(source_s, name=f"{pair} direct source timestamps")
    decision_s = source_s + np.int64(FEATURE_DECISION_SHIFT_SECONDS)
    decision_ns = _seconds_to_ns(
        decision_s, name=f"{pair} direct decision timestamps"
    )
    shift_ns = np.int64(FEATURE_DECISION_SHIFT_SECONDS * NS_PER_SECOND)
    horizon_ns = np.int64(HORIZON_SECONDS * NS_PER_SECOND)
    if np.any(decision_ns > np.iinfo(np.int64).max - horizon_ns):
        raise AdapterError(f"{pair}: direct label-exit clock overflows")
    label_exit_ns = decision_ns + horizon_ns
    direct_replay_session = np.asarray(session_mask(decision_s, "ny"), dtype=bool)
    direct_fit_mask = direct_fit_session & direct_moved
    if contract.early_stopping_ties == "include_as_down":
        direct_early_stopping = direct_fit_session.copy()
    elif contract.early_stopping_ties == "drop":
        direct_early_stopping = direct_fit_mask.copy()
    else:  # pragma: no cover - validated sealed contract
        raise AdapterError(
            f"{pair}: unknown direct early-stopping tie rule "
            f"{contract.early_stopping_ties!r}"
        )

    if isinstance(cap, bool) or not isinstance(cap, (int, np.integer)) or cap <= 0:
        raise AdapterError(f"cap must be a positive integer, got {cap!r}")
    cap = int(cap)

    def exact_fit_window(
        row_ids: np.ndarray,
        exits: np.ndarray,
        fit_mask: np.ndarray,
    ) -> np.ndarray:
        admitted = np.asarray(fit_mask, dtype=bool).copy()
        if entry_at_or_after is not None:
            admitted &= row_ids >= _utc_ns(entry_at_or_after)
        if label_exit_before is not None:
            admitted &= exits < _utc_ns(label_exit_before)
        return np.flatnonzero(admitted)

    direct_positions = exact_fit_window(
        decision_ns, label_exit_ns, direct_fit_mask
    )
    if len(direct_positions) <= cap:
        direct_ranks = np.arange(len(direct_positions), dtype="int64")
    else:
        direct_ranks = np.floor(
            np.linspace(0, len(direct_positions) - 1, cap)
        ).astype("int64")
    direct_selected_ids = decision_ns[direct_positions[direct_ranks]]

    adapter_positions = exact_fit_window(
        adapter_rows.fit_row_id,
        adapter_rows.label_exit_ns,
        adapter_rows.fit_mask,
    )
    adapter_ranks = capped_fit_ranks(len(adapter_positions), cap=cap)
    adapter_selected_ids = adapter_rows.fit_row_id[
        adapter_positions[adapter_ranks]
    ]

    def i64_digest(values: np.ndarray) -> str:
        return hashlib.sha256(
            np.asarray(values, dtype="<i8").tobytes(order="C")
        ).hexdigest()

    feature_shape_equal = adapter_rows.X.shape == direct_X.shape
    feature_dtype_equal = (
        adapter_rows.X.dtype == direct_X.dtype == np.dtype("float32")
        and adapter_rows.X.flags.c_contiguous
        and direct_X.flags.c_contiguous
    )
    report: dict[str, Any] = {
        "schema": "m15-direct-incumbent-builder-parity/v1",
        "pair": pair,
        "family": contract.family,
        "years": list(normalized_years),
        "purpose": purpose,
        "direct_stride": direct_stride,
        "adapter_builder_stride": adapter_rows.builder_stride,
        "purpose_equal": adapter_rows.purpose == purpose,
        "stride_equal": adapter_rows.builder_stride == direct_stride,
        "row_count_equal": len(adapter_rows.fit_row_id) == row_count,
        "source_feature_clock_equal": bool(
            np.array_equal(adapter_rows.source_feature_ns, source_ns)
        ),
        "decision_clock_equal": bool(
            np.array_equal(adapter_rows.fit_row_id, decision_ns)
        ),
        "source_to_decision_shift_exact": bool(
            np.array_equal(decision_ns, source_ns + shift_ns)
            and np.array_equal(
                adapter_rows.fit_row_id,
                adapter_rows.source_feature_ns + shift_ns,
            )
        ),
        "label_exit_clock_equal": bool(
            np.array_equal(adapter_rows.label_exit_ns, label_exit_ns)
        ),
        "feature_order_equal": adapter_rows.feature_cols == direct_cols,
        "feature_shape_equal": feature_shape_equal,
        "feature_dtype_equal": bool(feature_dtype_equal),
        "feature_bytes_equal": bool(
            feature_shape_equal
            and feature_dtype_equal
            and adapter_rows.X.tobytes(order="C") == direct_X.tobytes(order="C")
        ),
        "labels_equal": bool(np.array_equal(adapter_rows.y, direct_y)),
        "moved_equal": bool(np.array_equal(adapter_rows.moved, direct_moved)),
        "tie_mask_equal": bool(
            np.array_equal(~adapter_rows.moved, ~direct_moved)
        ),
        "legacy_fit_session_mask_equal": bool(
            np.array_equal(adapter_rows.fit_session_mask, direct_fit_session)
        ),
        "legacy_calibration_session_mask_equal": bool(
            np.array_equal(
                adapter_rows.calibration_session_mask,
                direct_calibration_session,
            )
        ),
        "replay_session_mask_equal": bool(
            np.array_equal(adapter_rows.replay_session_mask, direct_replay_session)
        ),
        "fit_mask_equal": bool(
            np.array_equal(adapter_rows.fit_mask, direct_fit_mask)
        ),
        "early_stopping_mask_equal": bool(
            np.array_equal(
                adapter_rows.early_stopping_mask,
                direct_early_stopping,
            )
        ),
        "capped_selected_row_ids_equal": bool(
            np.array_equal(adapter_selected_ids, direct_selected_ids)
        ),
        "adapter_capped_selected_rows": int(len(adapter_selected_ids)),
        "direct_capped_selected_rows": int(len(direct_selected_ids)),
        "adapter_capped_selected_row_ids_sha256": i64_digest(
            adapter_selected_ids
        ),
        "direct_capped_selected_row_ids_sha256": i64_digest(
            direct_selected_ids
        ),
    }
    required = (
        "purpose_equal",
        "stride_equal",
        "row_count_equal",
        "source_feature_clock_equal",
        "decision_clock_equal",
        "source_to_decision_shift_exact",
        "label_exit_clock_equal",
        "feature_order_equal",
        "feature_shape_equal",
        "feature_dtype_equal",
        "feature_bytes_equal",
        "labels_equal",
        "moved_equal",
        "tie_mask_equal",
        "legacy_fit_session_mask_equal",
        "legacy_calibration_session_mask_equal",
        "replay_session_mask_equal",
        "fit_mask_equal",
        "early_stopping_mask_equal",
        "capped_selected_row_ids_equal",
    )
    report["all_equal"] = all(report[name] is True for name in required)
    return report


def score_window_mask(
    rows: ScoreRows,
    *,
    entry_at_or_after: str | pd.Timestamp | np.datetime64 | int | None = None,
    entry_before: str | pd.Timestamp | np.datetime64 | int | None = None,
) -> np.ndarray:
    """Select an exact UTC covariate window without consulting outcomes."""

    mask = np.ones(len(rows.entry_ns), dtype=bool)
    if entry_at_or_after is not None:
        mask &= rows.entry_ns >= _utc_ns(entry_at_or_after)
    if entry_before is not None:
        mask &= rows.entry_ns < _utc_ns(entry_before)
    return mask


def slice_score_rows(
    rows: ScoreRows,
    *,
    entry_at_or_after: str | pd.Timestamp | np.datetime64 | int | None = None,
    entry_before: str | pd.Timestamp | np.datetime64 | int | None = None,
) -> ScoreRows:
    """Return an outcome-free exact-clock slice of full-resolution rows."""

    keep = score_window_mask(
        rows,
        entry_at_or_after=entry_at_or_after,
        entry_before=entry_before,
    )
    return ScoreRows(
        pair=rows.pair,
        feature_cols=rows.feature_cols,
        X=np.ascontiguousarray(rows.X[keep], dtype="float32"),
        entry_ns=np.asarray(rows.entry_ns[keep], dtype="int64"),
        source_feature_ns=np.asarray(rows.source_feature_ns[keep], dtype="int64"),
    )


def score_rows_parity_report(
    reference: ScoreRows,
    rebuilt: ScoreRows,
    *,
    reference_probabilities: Sequence[float] | np.ndarray | None = None,
    rebuilt_probabilities: Sequence[float] | np.ndarray | None = None,
    confidence_threshold: float | None = None,
    structural_column: str | None = None,
    structural_threshold: float | None = None,
) -> dict[str, Any]:
    """Report exact, outcome-free feature and policy parity.

    Feature deltas are diagnostic: decoded float32 values may differ while the
    hard behavior contract remains unchanged.  When predictions are supplied,
    probability parity is byte-for-byte (including dtype), followed by exact
    direction, confidence, structural-gate, and ordered pre-schedule decision
    comparisons.  No label or settlement input is accepted.
    """

    if (reference_probabilities is None) != (rebuilt_probabilities is None):
        raise AdapterError("parity requires both probability vectors or neither")
    if (structural_column is None) != (structural_threshold is None):
        raise AdapterError("structural parity requires both column and threshold")
    if confidence_threshold is not None:
        confidence_threshold = _finite_scalar(
            confidence_threshold, name="confidence threshold"
        )
        if confidence_threshold < 0.0:
            raise AdapterError("confidence threshold must be non-negative")
    if structural_threshold is not None:
        structural_threshold = _finite_scalar(
            structural_threshold, name="structural threshold"
        )

    pair_equal = reference.pair == rebuilt.pair
    schema_equal = reference.feature_cols == rebuilt.feature_cols
    source_clock_equal = np.array_equal(
        reference.source_feature_ns, rebuilt.source_feature_ns
    )
    clock_equal = np.array_equal(reference.entry_ns, rebuilt.entry_ns)
    decision_shift_exact = bool(
        np.array_equal(
            reference.entry_ns,
            reference.source_feature_ns
            + np.int64(FEATURE_DECISION_SHIFT_SECONDS * NS_PER_SECOND),
        )
        and np.array_equal(
            rebuilt.entry_ns,
            rebuilt.source_feature_ns
            + np.int64(FEATURE_DECISION_SHIFT_SECONDS * NS_PER_SECOND),
        )
    )
    shape_equal = reference.X.shape == rebuilt.X.shape
    dtype_equal = reference.X.dtype == rebuilt.X.dtype == np.dtype("float32")
    comparable = (
        pair_equal
        and schema_equal
        and source_clock_equal
        and clock_equal
        and decision_shift_exact
        and shape_equal
    )

    report: dict[str, Any] = {
        "pair_equal": pair_equal,
        "schema_equal": schema_equal,
        "source_clock_equal": source_clock_equal,
        "clock_equal": clock_equal,
        "decision_shift_exact": decision_shift_exact,
        "shape_equal": shape_equal,
        "feature_dtype_equal": dtype_equal,
        "rows": int(len(reference.entry_ns)) if clock_equal else None,
        "feature_bytes_equal": False,
        "feature_finite_mask_equal": False,
        "scoreability_equal": False,
        "mutually_scoreable_rows": None,
        "has_mutually_scoreable_rows": False,
        "finite_value_mismatches": None,
        "max_abs_finite_delta": None,
        "per_column_diagnostics": None,
        "probability_bitwise_equal": None,
        "probability_value_mismatches": None,
        "direction_equal": None,
        "confidence_gate_equal": None,
        "structural_gate_equal": None,
        "ordered_pre_schedule_decisions_equal": None,
        "reference_selected": None,
        "rebuilt_selected": None,
    }
    if comparable:
        left_finite = np.isfinite(reference.X)
        right_finite = np.isfinite(rebuilt.X)
        both_finite = left_finite & right_finite
        finite_deltas = np.abs(
            reference.X[both_finite].astype("float64")
            - rebuilt.X[both_finite].astype("float64")
        )
        column_diagnostics: dict[str, dict[str, int | float]] = {}
        for column, name in enumerate(reference.feature_cols):
            left_col = reference.X[:, column]
            right_col = rebuilt.X[:, column]
            left_col_finite = np.isfinite(left_col)
            right_col_finite = np.isfinite(right_col)
            common_finite = left_col_finite & right_col_finite
            finite_xor = int(np.count_nonzero(left_col_finite != right_col_finite))
            left_common = left_col[common_finite]
            right_common = right_col[common_finite]
            mismatch = int(np.count_nonzero(left_common != right_common))
            if finite_xor or mismatch:
                abs_delta = np.abs(
                    left_common.astype("float64") - right_common.astype("float64")
                )
                ulp_delta = _float32_ulp_distance(left_common, right_common)
                column_diagnostics[name] = {
                    "finite_xor": finite_xor,
                    "value_mismatches": mismatch,
                    "max_abs_delta": float(np.max(abs_delta)) if len(abs_delta) else 0.0,
                    "max_ulp": int(np.max(ulp_delta)) if len(ulp_delta) else 0,
                }
        report.update(
            feature_bytes_equal=(
                dtype_equal and reference.X.tobytes(order="C") == rebuilt.X.tobytes(order="C")
            ),
            feature_finite_mask_equal=bool(np.array_equal(left_finite, right_finite)),
            scoreability_equal=bool(
                np.array_equal(left_finite.all(axis=1), right_finite.all(axis=1))
            ),
            mutually_scoreable_rows=int(
                np.count_nonzero(left_finite.all(axis=1) & right_finite.all(axis=1))
            ),
            has_mutually_scoreable_rows=bool(
                np.any(left_finite.all(axis=1) & right_finite.all(axis=1))
            ),
            finite_value_mismatches=int(np.count_nonzero(finite_deltas != 0.0)),
            max_abs_finite_delta=(
                float(np.max(finite_deltas)) if len(finite_deltas) else 0.0
            ),
            per_column_diagnostics=column_diagnostics,
        )

    if reference_probabilities is None:
        return report

    left_probability = _validate_probability_vector(
        reference_probabilities, len(reference.entry_ns), "reference probabilities"
    )
    right_probability = _validate_probability_vector(
        rebuilt_probabilities, len(rebuilt.entry_ns), "rebuilt probabilities"
    )
    if not comparable:
        return report

    left_direction = left_probability >= 0.5
    right_direction = right_probability >= 0.5
    left_scoreable = np.isfinite(reference.X).all(axis=1)
    right_scoreable = np.isfinite(rebuilt.X).all(axis=1)
    mutually_scoreable = left_scoreable & right_scoreable
    left_scored_probability = np.ascontiguousarray(left_probability[mutually_scoreable])
    right_scored_probability = np.ascontiguousarray(right_probability[mutually_scoreable])
    probability_bytes_equal = (
        left_scored_probability.dtype == right_scored_probability.dtype
        and left_scored_probability.tobytes(order="C")
        == right_scored_probability.tobytes(order="C")
    )
    left_confidence = np.ones(len(left_probability), dtype=bool)
    right_confidence = np.ones(len(right_probability), dtype=bool)
    confidence_equal: bool | None = None
    if confidence_threshold is not None:
        left_confidence = np.abs(left_probability - 0.5) >= confidence_threshold
        right_confidence = np.abs(right_probability - 0.5) >= confidence_threshold
        confidence_equal = bool(
            np.array_equal(
                left_confidence[mutually_scoreable],
                right_confidence[mutually_scoreable],
            )
        )

    left_structural = np.ones(len(left_probability), dtype=bool)
    right_structural = np.ones(len(right_probability), dtype=bool)
    structural_equal: bool | None = None
    if structural_column is not None:
        if structural_column not in reference.feature_cols:
            raise AdapterError(f"parity structural column is absent: {structural_column}")
        col = reference.feature_cols.index(structural_column)
        assert structural_threshold is not None
        left_structural = reference.X[:, col] <= structural_threshold
        right_structural = rebuilt.X[:, col] <= structural_threshold
        structural_equal = bool(
            np.array_equal(
                left_structural[mutually_scoreable],
                right_structural[mutually_scoreable],
            )
        )

    left_selected = left_scoreable & left_confidence & left_structural
    right_selected = right_scoreable & right_confidence & right_structural
    report.update(
        probability_bitwise_equal=bool(probability_bytes_equal),
        probability_value_mismatches=int(
            np.count_nonzero(left_scored_probability != right_scored_probability)
        ),
        direction_equal=bool(
            np.array_equal(
                left_direction[mutually_scoreable],
                right_direction[mutually_scoreable],
            )
        ),
        confidence_gate_equal=confidence_equal,
        structural_gate_equal=structural_equal,
        ordered_pre_schedule_decisions_equal=bool(
            np.array_equal(reference.entry_ns[left_selected], rebuilt.entry_ns[right_selected])
        ),
        reference_selected=int(np.count_nonzero(left_selected)),
        rebuilt_selected=int(np.count_nonzero(right_selected)),
    )
    return report


def assert_score_rows_parity(
    reference: ScoreRows,
    rebuilt: ScoreRows,
    **kwargs: Any,
) -> dict[str, Any]:
    """Fail closed unless every supplied behavior-parity check is exact."""

    report = score_rows_parity_report(reference, rebuilt, **kwargs)
    required = [
        "pair_equal",
        "schema_equal",
        "source_clock_equal",
        "clock_equal",
        "decision_shift_exact",
        "shape_equal",
        "feature_dtype_equal",
        "scoreability_equal",
        "has_mutually_scoreable_rows",
    ]
    if kwargs.get("reference_probabilities") is not None:
        required.extend(
            [
                "probability_bitwise_equal",
                "direction_equal",
                "ordered_pre_schedule_decisions_equal",
            ]
        )
        if kwargs.get("confidence_threshold") is not None:
            required.append("confidence_gate_equal")
        if kwargs.get("structural_column") is not None:
            required.append("structural_gate_equal")
    failures = [name for name in required if report.get(name) is not True]
    if failures:
        raise AdapterError(f"blind score parity failed: {failures}; report={report}")
    return report


def assert_pre_april_score_parity(
    reference: ScoreRows,
    rebuilt: ScoreRows,
    *,
    replay_entry_at_or_after: str | pd.Timestamp | np.datetime64 | int = "2026-04-01T00:00:00Z",
    **kwargs: Any,
) -> dict[str, Any]:
    """Outcome-free parity gate that refuses any replay-period covariate row."""

    boundary = _utc_ns(replay_entry_at_or_after)
    if not len(reference.entry_ns) or not len(rebuilt.entry_ns):
        raise AdapterError("pre-April score parity requires non-empty covariate rows")
    if np.any(reference.entry_ns >= boundary) or np.any(rebuilt.entry_ns >= boundary):
        raise AdapterError("pre-April score parity received a replay-period timestamp")
    return assert_score_rows_parity(reference, rebuilt, **kwargs)


def window_mask(
    rows: AdapterRows,
    *,
    entry_at_or_after: str | pd.Timestamp | np.datetime64 | int | None = None,
    entry_before: str | pd.Timestamp | np.datetime64 | int | None = None,
    label_exit_before: str | pd.Timestamp | np.datetime64 | int | None = None,
) -> np.ndarray:
    """Return an exact UTC boundary mask without consulting nominal file years.

    Integer boundaries are interpreted as signed-int64 Unix nanoseconds.
    String/datetime boundaries must be timezone-aware.
    """

    mask = np.ones(len(rows.fit_row_id), dtype=bool)
    if entry_at_or_after is not None:
        mask &= rows.fit_row_id >= _utc_ns(entry_at_or_after)
    if entry_before is not None:
        mask &= rows.fit_row_id < _utc_ns(entry_before)
    if label_exit_before is not None:
        mask &= rows.label_exit_ns < _utc_ns(label_exit_before)
    return mask


def fitting_indices(
    rows: AdapterRows,
    *,
    entry_at_or_after: str | pd.Timestamp | np.datetime64 | int,
    label_exit_before: str | pd.Timestamp | np.datetime64 | int,
    cap: int = FIT_CAP,
) -> np.ndarray:
    """Apply split, family fit/tie/session filters, then the sealed cap."""

    if rows.purpose != "fit":
        raise AdapterError("fitting_indices requires rows built with purpose='fit'")
    admitted = rows.fit_mask & window_mask(
        rows,
        entry_at_or_after=entry_at_or_after,
        label_exit_before=label_exit_before,
    )
    source = np.flatnonzero(admitted)
    ranks = capped_fit_ranks(len(source), cap=cap)
    selected = source[ranks]
    _validate_row_ids(rows.fit_row_id[selected], name=f"{rows.pair} selected fit_row_id")
    return selected


def early_stopping_indices(
    rows: AdapterRows,
    *,
    entry_at_or_after: str | pd.Timestamp | np.datetime64 | int,
    label_exit_before: str | pd.Timestamp | np.datetime64 | int,
) -> np.ndarray:
    admitted = rows.early_stopping_mask & window_mask(
        rows,
        entry_at_or_after=entry_at_or_after,
        label_exit_before=label_exit_before,
    )
    return np.flatnonzero(admitted)


def calibration_indices(
    rows: AdapterRows,
    *,
    entry_at_or_after: str | pd.Timestamp | np.datetime64 | int,
    label_exit_before: str | pd.Timestamp | np.datetime64 | int,
) -> np.ndarray:
    admitted = rows.calibration_mask & window_mask(
        rows,
        entry_at_or_after=entry_at_or_after,
        label_exit_before=label_exit_before,
    )
    return np.flatnonzero(admitted)


def capped_fit_ranks(n: int, cap: int = FIT_CAP) -> np.ndarray:
    """Exact ``floor(linspace(0, n-1, cap))`` timestamp-only selector."""

    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
        raise AdapterError(f"row count must be a non-negative integer, got {n!r}")
    if isinstance(cap, bool) or not isinstance(cap, (int, np.integer)) or cap <= 0:
        raise AdapterError(f"cap must be a positive integer, got {cap!r}")
    n = int(n)
    cap = int(cap)
    if n <= cap:
        return np.arange(n, dtype="int64")
    ranks = np.floor(np.linspace(0, n - 1, cap)).astype("int64")
    if len(ranks) != cap or ranks[0] != 0 or ranks[-1] != n - 1:
        raise AdapterError("capped selector endpoint invariant failed")
    if np.any(ranks[1:] <= ranks[:-1]):
        raise AdapterError("capped selector did not produce unique ordered ranks")
    return ranks


# Public aliases used by parity/boundary tests and the campaign runner.
deterministic_cap_indices = capped_fit_ranks
timestamp_cap_selector = capped_fit_ranks


def calibration_threshold(
    probabilities: Sequence[float] | np.ndarray,
    admitted_mask: Sequence[bool] | np.ndarray,
    target_coverage: float,
) -> float:
    """Construct the frozen confidence threshold with NumPy's linear quantile."""

    if not 0.0 < float(target_coverage) < 1.0:
        raise AdapterError(f"target coverage must be in (0, 1), got {target_coverage}")
    p = np.asarray(probabilities, dtype="float64")
    admitted = np.asarray(admitted_mask, dtype=bool).copy()
    if p.ndim != 1 or admitted.ndim != 1 or len(p) != len(admitted):
        raise AdapterError("probability and calibration masks must be aligned 1-D vectors")
    selected = p[admitted & np.isfinite(p)]
    if len(selected) == 0:
        raise AdapterError("empty finite calibration probability vector")
    confidence = np.abs(selected - 0.5)
    if np.any(~np.isfinite(confidence)):
        raise AdapterError("non-finite calibration confidence vector")
    threshold = float(
        np.quantile(confidence, 1.0 - float(target_coverage), method="linear")
    )
    if not np.isfinite(threshold):
        raise AdapterError("non-finite calibration threshold")
    return threshold


def calibrate_policy(
    rows: AdapterRows,
    probabilities: Sequence[float] | np.ndarray,
    admitted_mask: Sequence[bool] | np.ndarray,
    *,
    target_coverage: float | None = None,
    structural_scope_mask: Sequence[bool] | np.ndarray | None = None,
) -> PolicyCalibration:
    """Freeze one family's structural and confidence thresholds.

    EURUSD first computes the fixed q33 ``15m_bb_width`` threshold on every
    label-valid row in the exact calibration window, before the legacy
    fixed-UTC session mask.  Its confidence quantile is then constructed on
    fixed-UTC rows inside that structural gate, matching
    ``m15_xpair_freeze.py``.  The other six families have no model-owned
    structural gate.
    """

    contract = contract_for(rows.pair)
    p = np.asarray(probabilities, dtype="float64")
    admitted = np.asarray(admitted_mask, dtype=bool).copy()
    if p.ndim != 1 or admitted.ndim != 1 or len(p) != len(rows.fit_row_id) or len(admitted) != len(p):
        raise AdapterError("probabilities and admitted_mask must align with adapter rows")
    admitted &= rows.calibration_mask
    coverage = contract.target_coverage if target_coverage is None else float(target_coverage)
    structural_column: str | None = None
    structural_quantile: float | None = None
    structural_threshold: float | None = None
    structural_rows: int | None = None
    confidence_mask = admitted.copy()
    if contract.structural_column is not None:
        structural_column = contract.structural_column
        structural_quantile = contract.structural_quantile
        if structural_quantile is None:
            raise AdapterError(f"{contract.pair}: structural quantile is not configured")
        try:
            column = rows.feature_cols.index(structural_column)
        except ValueError as exc:
            raise AdapterError(
                f"{contract.pair}: missing {structural_column} structural feature"
            ) from exc
        if structural_scope_mask is None:
            raise AdapterError(
                f"{contract.pair}: structural calibration requires an explicit full-window mask"
            )
        structural_scope = np.asarray(structural_scope_mask, dtype=bool)
        if structural_scope.ndim != 1 or len(structural_scope) != len(rows.fit_row_id):
            raise AdapterError("structural_scope_mask must align with calibration rows")
        values = rows.X[:, column].astype("float64")
        finite = structural_scope & np.isfinite(values)
        if not finite.any():
            raise AdapterError(f"{contract.pair}: empty finite structural calibration vector")
        structural_rows = int(finite.sum())
        structural_threshold = float(
            np.quantile(values[finite], structural_quantile, method="linear")
        )
        if not np.isfinite(structural_threshold):
            raise AdapterError(f"{contract.pair}: non-finite structural threshold")
        confidence_mask &= np.isfinite(values) & (values <= structural_threshold)
    threshold = calibration_threshold(p, confidence_mask, coverage)
    confidence_rows = int((confidence_mask & np.isfinite(p)).sum())
    return PolicyCalibration(
        pair=contract.pair,
        target_coverage=coverage,
        confidence_threshold=threshold,
        calibration_rows=int(admitted.sum()),
        confidence_rows=confidence_rows,
        structural_rows=structural_rows,
        structural_column=structural_column,
        structural_quantile=structural_quantile,
        structural_threshold=structural_threshold,
    )


def policy_selection_mask(
    rows: AdapterRows,
    probabilities: Sequence[float] | np.ndarray,
    calibration: PolicyCalibration,
) -> np.ndarray:
    """Apply only model-owned gates; the common DST evaluator remains external."""

    if calibration.pair != rows.pair:
        raise AdapterError("policy calibration pair does not match adapter rows")
    p = np.asarray(probabilities, dtype="float64")
    if p.ndim != 1 or len(p) != len(rows.fit_row_id):
        raise AdapterError("probabilities must align with adapter rows")
    selected = np.isfinite(p) & (np.abs(p - 0.5) >= calibration.confidence_threshold)
    if calibration.structural_column is not None:
        if calibration.structural_threshold is None:
            raise AdapterError("structural calibration is missing its threshold")
        try:
            column = rows.feature_cols.index(calibration.structural_column)
        except ValueError as exc:
            raise AdapterError(
                f"{rows.pair}: missing structural feature {calibration.structural_column}"
            ) from exc
        values = rows.X[:, column]
        selected &= np.isfinite(values) & (values <= calibration.structural_threshold)
    return selected


def model_parameters(
    pair: str, seed: int, *, n_jobs: int = DEFAULT_NUM_THREADS
) -> dict[str, Any]:
    """Return the sealed deterministic LightGBM settings for one seed."""

    contract = contract_for(pair)
    if seed not in contract.seeds:
        raise AdapterError(f"{contract.pair}: seed {seed} is outside declared order {contract.seeds}")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs <= 0:
        raise AdapterError("n_jobs must be a positive integer")
    return {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.02,
        "num_leaves": contract.num_leaves,
        "min_child_samples": 400,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.5,
        "reg_lambda": 20,
        "n_estimators": contract.max_rounds,
        "n_jobs": n_jobs,
        "verbosity": -1,
        "random_state": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
        "deterministic": True,
        "force_col_wise": True,
    }


def make_model(
    pair: str, seed: int, *, n_jobs: int = DEFAULT_NUM_THREADS
) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(**model_parameters(pair, seed, n_jobs=n_jobs))


def fit_seed_checkpoint(
    pair: str,
    seed: int,
    fit_rows: AdapterRows,
    fit_indices: Sequence[int] | np.ndarray,
    validation_rows: AdapterRows,
    validation_indices: Sequence[int] | np.ndarray,
    checkpoint_path: str | Path,
    *,
    n_jobs: int = DEFAULT_NUM_THREADS,
) -> SeedCheckpoint:
    """Fit exactly one seed and atomically checkpoint its model bytes.

    The destination must not exist.  Callers run this serially in declared seed
    order; :func:`fit_seed_checkpoints` provides that loop.
    """

    contract = contract_for(pair)
    if fit_rows.pair != contract.pair or validation_rows.pair != contract.pair:
        raise AdapterError(f"{contract.pair}: row pair does not match model contract")
    if fit_rows.feature_cols != validation_rows.feature_cols:
        raise AdapterError(f"{contract.pair}: fit/validation feature order mismatch")
    fit_idx = _validate_positions(fit_indices, len(fit_rows.fit_row_id), "fit indices")
    val_idx = _validate_positions(
        validation_indices, len(validation_rows.fit_row_id), "validation indices"
    )
    if len(fit_idx) == 0 or len(val_idx) == 0:
        raise AdapterError(f"{contract.pair}: empty fit or early-stopping rows")
    if np.any(~fit_rows.fit_mask[fit_idx]):
        raise AdapterError(f"{contract.pair}: fit indices include a tie or wrong-session row")
    if np.any(~validation_rows.early_stopping_mask[val_idx]):
        raise AdapterError(
            f"{contract.pair}: validation indices include a tie or wrong-session row"
        )

    destination = Path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"checkpoint already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary checkpoint already exists: {temporary}")

    model = make_model(contract.pair, seed, n_jobs=n_jobs)
    model.fit(
        fit_rows.X[fit_idx],
        fit_rows.y[fit_idx],
        eval_set=[(validation_rows.X[val_idx], validation_rows.y[val_idx])],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(contract.early_stopping_rounds),
            lgb.log_evaluation(0),
        ],
    )
    model.booster_.save_model(str(temporary))
    _fsync_file(temporary)
    try:
        # Hard-link publication is destination-must-not-exist, unlike
        # os.replace(), which could overwrite a concurrently sealed attempt.
        os.link(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    _fsync_dir(destination.parent)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return SeedCheckpoint(
        pair=contract.pair,
        seed=seed,
        best_iteration=int(model.best_iteration_ or contract.max_rounds),
        model_path=destination,
        sha256=digest,
    )


def fit_seed_checkpoints(
    pair: str,
    fit_rows: AdapterRows,
    fit_indices: Sequence[int] | np.ndarray,
    validation_rows: AdapterRows,
    validation_indices: Sequence[int] | np.ndarray,
    checkpoint_dir: str | Path,
    *,
    filename_template: str = "seed_{seed}.txt",
    n_jobs: int = DEFAULT_NUM_THREADS,
) -> list[SeedCheckpoint]:
    """Fit/checkpoint declared seeds serially, never retaining an ensemble in RAM."""

    contract = contract_for(pair)
    destination = Path(checkpoint_dir)
    checkpoints: list[SeedCheckpoint] = []
    for seed in contract.seeds:
        name = filename_template.format(pair=contract.pair, seed=seed)
        checkpoints.append(
            fit_seed_checkpoint(
                contract.pair,
                seed,
                fit_rows,
                fit_indices,
                validation_rows,
                validation_indices,
                destination / name,
                n_jobs=n_jobs,
            )
        )
    return checkpoints


def predict_checkpoint_mean(
    checkpoints: Sequence[SeedCheckpoint | str | Path],
    X: np.ndarray,
) -> np.ndarray:
    """Average seed probabilities in the caller-provided declared order."""

    matrix = np.ascontiguousarray(X, dtype="float32")
    if matrix.ndim != 2:
        raise AdapterError("prediction matrix must be 2-D")
    if not checkpoints:
        raise AdapterError("at least one checkpoint is required")
    total = np.zeros(len(matrix), dtype="float64")
    for checkpoint in checkpoints:
        path = checkpoint.model_path if isinstance(checkpoint, SeedCheckpoint) else Path(checkpoint)
        booster = lgb.Booster(model_file=str(path))
        total += np.asarray(booster.predict(matrix), dtype="float64")
        del booster
    return total / float(len(checkpoints))


def _make_rows(
    contract: AdapterContract,
    purpose: str,
    X: np.ndarray,
    cols: tuple[str, ...],
    y: np.ndarray,
    moved: np.ndarray,
    ts_s: Sequence[int] | np.ndarray,
    fit_session: np.ndarray,
    calibration_session: np.ndarray,
    builder_stride: int,
) -> AdapterRows:
    ts_s = _validate_epoch_seconds(ts_s, name=f"{contract.pair} builder timestamps")
    if np.any(ts_s > np.iinfo(np.int64).max - FEATURE_DECISION_SHIFT_SECONDS):
        raise AdapterError(f"{contract.pair}: completed-minute decision clock overflows")
    decision_s = ts_s + np.int64(FEATURE_DECISION_SHIFT_SECONDS)
    row_id = _seconds_to_ns(decision_s, name=f"{contract.pair} decision timestamps")
    _validate_row_ids(row_id, name=f"{contract.pair} fit_row_id")
    horizon_ns = np.int64(HORIZON_SECONDS * NS_PER_SECOND)
    if np.any(row_id > np.iinfo(np.int64).max - horizon_ns):
        raise AdapterError(f"{contract.pair}: label exit overflows signed int64 nanoseconds")
    # Fitting/calibration masks above deliberately retain each legacy builder's
    # source-label session recipe.  The standardized replay session owns the
    # causal completed-minute decision timestamp.
    replay_ny = np.asarray(session_mask(decision_s, "ny"), dtype=bool)
    return AdapterRows(
        pair=contract.pair,
        purpose=purpose,
        feature_cols=cols,
        X=np.ascontiguousarray(X, dtype="float32"),
        y=np.asarray(y, dtype="uint8"),
        moved=np.asarray(moved, dtype=bool),
        fit_row_id=row_id,
        label_exit_ns=row_id + horizon_ns,
        fit_session_mask=np.asarray(fit_session, dtype=bool),
        calibration_session_mask=np.asarray(calibration_session, dtype=bool),
        replay_session_mask=replay_ny,
        source_feature_ns=_seconds_to_ns(
            ts_s, name=f"{contract.pair} source feature timestamps"
        ),
        builder_stride=builder_stride,
    )


def _import_m15_eur_builder() -> Any:
    module = sys.modules.get("m5_xpair")
    if module is not None:
        if getattr(module, "HOR", None) != 15:
            raise AdapterError("m5_xpair was already imported without MX_HOR=15")
        return module
    previous = os.environ.get("MX_HOR")
    os.environ["MX_HOR"] = "15"
    try:
        module = importlib.import_module("m5_xpair")
    finally:
        if previous is None:
            os.environ.pop("MX_HOR", None)
        else:
            os.environ["MX_HOR"] = previous
    if getattr(module, "HOR", None) != 15:
        raise AdapterError("m5_xpair did not initialize at the 15-minute horizon")
    return module


@contextlib.contextmanager
def _temporary_attrs(module: Any, **updates: Any) -> Iterator[None]:
    previous = {name: getattr(module, name) for name in updates}
    for name, value in updates.items():
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(module, name, value)


def _validate_years(years: Sequence[str | int]) -> tuple[str, ...]:
    normalized = tuple(str(y) for y in years)
    if not normalized:
        raise AdapterError("at least one nominal source year is required")
    if any(not value.isdigit() or len(value) != 4 for value in normalized):
        raise AdapterError(f"invalid nominal source years: {normalized}")
    numeric = np.asarray([int(value) for value in normalized], dtype="int64")
    if len(numeric) > 1 and np.any(numeric[1:] <= numeric[:-1]):
        raise AdapterError("nominal source years must be strictly increasing and unique")
    return normalized


def _resolve_stride(
    contract: AdapterContract, purpose: str, stride: int | None
) -> int:
    if purpose not in {"fit", "calibration", "replay", "parity"}:
        raise AdapterError(f"unknown adapter purpose: {purpose}")
    if stride is None:
        return contract.fit_stride if purpose == "fit" else 1
    if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)) or stride <= 0:
        raise AdapterError(f"{contract.pair}: stride must be a positive integer")
    return int(stride)


def _validate_score_feature_names(cols: tuple[str, ...], *, pair: str) -> None:
    if not cols or any(not isinstance(col, str) or not col for col in cols):
        raise AdapterError(f"{pair}: score feature names must be non-empty strings")
    forbidden = [col for col in cols if col in FORBIDDEN_SCORE_COLUMNS]
    if forbidden:
        raise AdapterError(f"{pair}: outcome/meta columns are forbidden in ScoreRows: {forbidden}")


def _read_covariate_years(
    directory: Path,
    pair: str,
    years: Sequence[str],
    cols: Sequence[str],
    *,
    allow_close: bool = False,
) -> pd.DataFrame:
    """Project only named covariates and preserve their unfiltered source clock."""

    ordered_cols = tuple(str(col) for col in cols)
    if len(set(ordered_cols)) != len(ordered_cols):
        raise AdapterError(f"{pair}: duplicate requested score-source columns")
    if allow_close:
        if ordered_cols != ("close",):
            raise AdapterError(f"{pair}: close projection must request only close")
    else:
        _validate_score_feature_names(ordered_cols, pair=pair)
    _validate_source_files(directory, (pair,), years)
    parts: list[pd.DataFrame] = []
    for year in years:
        path = directory / f"{pair}_{year}.parquet"
        try:
            frame = pd.read_parquet(path, columns=list(ordered_cols))
        except Exception as exc:
            raise AdapterError(
                f"{pair}: cannot project score covariates {list(ordered_cols[:12])} "
                f"from {path}: {exc}"
            ) from exc
        _datetime_index_ns(frame.index, name=f"{path} projected score clock")
        if tuple(str(col) for col in frame.columns) != ordered_cols:
            raise AdapterError(
                f"{pair}: parquet projection changed requested covariate order in {path}"
            )
        parts.append(frame)
    combined = pd.concat(parts, axis=0, copy=False)
    _datetime_index_ns(combined.index, name=f"{pair} combined score-source clock")
    return combined


def _score_rows_from_frame(
    pair: str,
    cols: tuple[str, ...],
    frame: pd.DataFrame,
) -> ScoreRows:
    if tuple(str(col) for col in frame.columns) != cols:
        raise AdapterError(f"{pair}: score frame does not preserve ordered book columns")
    source_ns = _datetime_index_ns(frame.index, name=f"{pair} score frame clock")
    shift_ns = np.int64(FEATURE_DECISION_SHIFT_SECONDS * NS_PER_SECOND)
    if np.any(source_ns > np.iinfo(np.int64).max - shift_ns):
        raise AdapterError(f"{pair}: completed-minute score clock overflows")
    entry_ns = source_ns + shift_ns
    try:
        matrix = frame.to_numpy(dtype="float32", copy=True)
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"{pair}: score covariates are not numeric") from exc
    return ScoreRows(
        pair=pair,
        feature_cols=cols,
        X=np.ascontiguousarray(matrix, dtype="float32"),
        entry_ns=np.asarray(entry_ns, dtype="int64"),
        source_feature_ns=np.asarray(source_ns, dtype="int64"),
    )


def _validate_source_files(directory: Path, pairs: Iterable[str], years: Sequence[str]) -> None:
    for year in years:
        for pair in pairs:
            path = directory / f"{pair}_{year}.parquet"
            if not path.is_file():
                raise AdapterError(f"missing builder source: {path}")
            index = pd.read_parquet(path, columns=[]).index
            _datetime_index_ns(index, name=str(path))


def _finite_scalar(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise AdapterError(f"{name} must be a finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"{name} must be a finite number") from exc
    if not np.isfinite(converted):
        raise AdapterError(f"{name} must be a finite number")
    return converted


def _validate_probability_vector(
    values: Sequence[float] | np.ndarray | None,
    n: int,
    name: str,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) != n or not np.issubdtype(raw.dtype, np.floating):
        raise AdapterError(f"{name} must be a row-aligned floating-point vector")
    if not raw.flags.c_contiguous:
        raw = np.ascontiguousarray(raw)
    if np.any(~np.isfinite(raw)) or np.any((raw < 0.0) | (raw > 1.0)):
        raise AdapterError(f"{name} must contain finite probabilities in [0, 1]")
    return raw


def _float32_ulp_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return representational ULP distance for aligned finite float32 values."""

    left = np.ascontiguousarray(left, dtype="float32")
    right = np.ascontiguousarray(right, dtype="float32")
    if left.shape != right.shape:
        raise AdapterError("ULP comparison requires aligned float32 arrays")
    sign = np.uint32(0x80000000)

    def ordered(values: np.ndarray) -> np.ndarray:
        bits = values.view("uint32")
        return np.where((bits & sign) != 0, ~bits, bits ^ sign).astype("uint64")

    left_ordered = ordered(left)
    right_ordered = ordered(right)
    return np.maximum(left_ordered, right_ordered) - np.minimum(
        left_ordered, right_ordered
    )


def _datetime_index_ns(index: pd.Index, *, name: str) -> np.ndarray:
    if not isinstance(index, pd.DatetimeIndex):
        raise AdapterError(f"{name}: expected a DatetimeIndex")
    if index.tz is None or str(index.tz) != "UTC":
        raise AdapterError(f"{name}: timestamps must be timezone-aware UTC")
    values = np.asarray(index.as_unit("ns").asi8, dtype="int64")
    if len(values) == 0:
        raise AdapterError(f"{name}: empty timestamp index")
    if np.any(values == np.iinfo(np.int64).min):
        raise AdapterError(f"{name}: non-finite timestamp")
    if len(values) > 1 and np.any(values[1:] <= values[:-1]):
        raise AdapterError(f"{name}: timestamps must be strictly increasing and unique")
    return values


def _frame_seconds(frame: pd.DataFrame, pair: str) -> np.ndarray:
    index_ns = _datetime_index_ns(frame.index, name=f"{pair} builder index")
    ts_s = _validate_epoch_seconds(frame["_ts"].to_numpy(), name=f"{pair} _ts")
    if not np.array_equal(index_ns, _seconds_to_ns(ts_s, name=f"{pair} _ts")):
        raise AdapterError(f"{pair}: builder index and _ts are not row-aligned")
    return ts_s


def _validate_epoch_seconds(values: Sequence[int] | np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise AdapterError(f"{name}: expected a 1-D integer Unix-seconds vector")
    lo = -9_223_372_036
    hi = 9_223_372_036
    if np.issubdtype(raw.dtype, np.unsignedinteger):
        if np.any(raw > hi):
            raise AdapterError(f"{name}: timestamp is outside signed-int64 nanosecond range")
    elif np.any(raw < lo) or np.any(raw > hi):
        raise AdapterError(f"{name}: timestamp is outside signed-int64 nanosecond range")
    seconds = raw.astype("int64", copy=False)
    if len(seconds) > 1 and np.any(seconds[1:] <= seconds[:-1]):
        raise AdapterError(f"{name}: timestamps must be strictly increasing and unique")
    return seconds


def _seconds_to_ns(seconds: np.ndarray, *, name: str) -> np.ndarray:
    seconds = _validate_epoch_seconds(seconds, name=name)
    return seconds * np.int64(NS_PER_SECOND)


def _validate_row_ids(values: np.ndarray, *, name: str) -> None:
    if not isinstance(values, np.ndarray) or values.dtype != np.dtype("int64") or values.ndim != 1:
        raise AdapterError(f"{name}: must be a 1-D signed-int64 vector")
    if np.any(values == np.iinfo(np.int64).min):
        raise AdapterError(f"{name}: contains the non-finite NaT sentinel")
    if len(values) > 1 and np.any(values[1:] <= values[:-1]):
        raise AdapterError(f"{name}: must be strictly increasing and unique")


def _validate_feature_contract(contract: AdapterContract, cols: tuple[str, ...]) -> None:
    if len(cols) != contract.n_features:
        raise AdapterError(
            f"{contract.pair}: expected {contract.n_features} ordered features, got {len(cols)}"
        )
    if len(set(cols)) != len(cols):
        raise AdapterError(f"{contract.pair}: duplicate feature names")


def _utc_ns(value: str | pd.Timestamp | np.datetime64 | int) -> np.int64:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        raw = int(value)
        if raw <= np.iinfo(np.int64).min or raw > np.iinfo(np.int64).max:
            raise AdapterError(f"UTC nanosecond boundary is out of range: {value}")
        return np.int64(raw)
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise AdapterError(f"invalid UTC timestamp boundary: {value!r}") from exc
    if timestamp.tzinfo is None:
        raise AdapterError(f"timestamp boundary must be timezone-aware: {value!r}")
    try:
        return np.int64(timestamp.tz_convert("UTC").value)
    except (OverflowError, ValueError) as exc:
        raise AdapterError(f"UTC timestamp boundary is out of range: {value!r}") from exc


def _validate_positions(values: Sequence[int] | np.ndarray, n: int, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise AdapterError(f"{name} must be a 1-D integer vector")
    positions = raw.astype("int64", copy=False)
    if np.any(positions < 0) or np.any(positions >= n):
        raise AdapterError(f"{name} contains an out-of-range row")
    if len(positions) > 1 and np.any(positions[1:] <= positions[:-1]):
        raise AdapterError(f"{name} must be strictly increasing and unique")
    return positions


def _fsync_file(path: Path) -> None:
    with path.open("rb") as fh:
        os.fsync(fh.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "AdapterContract",
    "AdapterError",
    "AdapterRows",
    "CONTRACTS",
    "DEFAULT_NUM_THREADS",
    "FEATURE_DECISION_SHIFT_SECONDS",
    "FIT_CAP",
    "PolicyCalibration",
    "ScoreRows",
    "SeedCheckpoint",
    "assert_pre_april_score_parity",
    "assert_score_rows_parity",
    "build_calibration_rows",
    "build_eurusd_xpof_rows",
    "build_fit_rows",
    "build_own_pair_rows",
    "build_replay_rows",
    "build_rows",
    "build_score_rows",
    "build_xpair_rows",
    "calibration_indices",
    "calibration_threshold",
    "calibrate_policy",
    "capped_fit_ranks",
    "contract_for",
    "deterministic_cap_indices",
    "direct_incumbent_builder_parity_report",
    "early_stopping_indices",
    "fit_seed_checkpoint",
    "fit_seed_checkpoints",
    "fitting_indices",
    "make_model",
    "model_parameters",
    "predict_checkpoint_mean",
    "policy_selection_mask",
    "score_rows_parity_report",
    "score_window_mask",
    "slice_score_rows",
    "timestamp_cap_selector",
    "window_mask",
]
