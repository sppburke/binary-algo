"""Bounded statistics and canonical evaluator wrappers for issue #9.

This module deliberately owns only the measurement surface that does not
already have a repository authority:

* strict preconditions around the canonical settlement/session/scheduler
  helpers;
* date-cluster numerator/denominator endpoint representation;
* synchronized represented-date moving-block max-T inference; and
* pure retrospective/shadow status decisions.

It does not load books, build features, fit models, inspect replay files, or
write evidence.  The campaign runner is responsible for access ordering and
for passing all 24/9/3k preregistered slots, including non-computable ones.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.special import ndtr
from scipy.stats import beta, norm

from deriv_runtime_core import after_last_start_cutoff, is_ny_session
from min1_production import nonoverlap_chrono as _canonical_nonoverlap_chrono
from min1_production import wc_ret as _canonical_wc_ret
from sessions import session_mask as _canonical_session_mask


NS_PER_SECOND = 1_000_000_000
NY_TZ = "America/New_York"
MAIN_FAMILY_SIZE = 24
CONTROL_FAMILY_SIZE = 9
PRODUCTION_BLOCK_LENGTHS = (5, 10, 20)
PRODUCTION_BOOTSTRAP_REPLICATES = 10_000
MAIN_BOOTSTRAP_SEED = 9001
CONTROL_BOOTSTRAP_SEED = 9002
SHADOW_BOOTSTRAP_SEED = 9003
NULL_TEST_SEED = 9005
NULL_CAMPAIGNS_PER_CELL = 400
NULL_BOOTSTRAP_REPLICATES = 1_000
NULL_BLOCK_LENGTHS = (1, 2, 3)
NULL_FALSE_POSITIVE_LIMIT = 32
NULL_SYNTHETIC_NY_DATES = 60
NULL_ROWS_PER_PAIR_DATE = 480
NULL_NY_START_HOUR = 8
NULL_ROW_MINUTES = "08:00_through_15:59_America/New_York"
NULL_RHO_ARM = 0.80
DEFERRED_SHADOW_PHASE_CODE = 1
DEFERRED_SHADOW_SEED_DOMAIN = (
    "m15-book-refresh/deferred-shadow-null-seed-identity/v1"
)

PAIR_ORDER = (
    "EURUSD",
    "USDJPY",
    "GBPUSD",
    "USDCHF",
    "AUDUSD",
    "NZDUSD",
)
FEATURE_SOURCE_PAIR_ORDER = (
    "EURUSD",
    "GBPUSD",
    "AUDUSD",
    "NZDUSD",
    "USDJPY",
    "USDCHF",
    "USDCAD",
)
PAIR_COVERAGE = {
    "EURUSD": 0.10,
    "USDJPY": 0.02,
    "GBPUSD": 0.01,
    "USDCHF": 0.01,
    "AUDUSD": 0.03,
    "NZDUSD": 0.02,
}
_NULL_MECHANICS_COVERAGE = {**PAIR_COVERAGE, "USDCAD": 0.02}

TAIL_TWO_SIDED = "two_sided"
TAIL_LOWER_ONLY = "lower_only"


class StatisticsInputError(ValueError):
    """Malformed evaluator/statistics input (an integrity defect, not sparsity)."""


def _as_one_dimensional(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise StatisticsInputError(f"{name} must be one-dimensional, got shape {arr.shape}")
    return arr


def _checked_integer_array(values: Any, *, name: str) -> np.ndarray:
    arr = _as_one_dimensional(values, name=name)
    if np.issubdtype(arr.dtype, np.bool_):
        raise StatisticsInputError(f"{name} must contain timestamps, not booleans")
    if np.issubdtype(arr.dtype, np.integer):
        if np.issubdtype(arr.dtype, np.unsignedinteger) and np.any(
            arr > np.uint64(np.iinfo(np.int64).max)
        ):
            raise StatisticsInputError(f"{name} contains an out-of-range timestamp")
        out = arr.astype(np.int64, copy=False)
    elif np.issubdtype(arr.dtype, np.floating):
        if not np.isfinite(arr).all():
            raise StatisticsInputError(f"{name} contains a non-finite timestamp")
        if not np.equal(arr, np.floor(arr)).all():
            raise StatisticsInputError(f"{name} contains a fractional timestamp")
        # Use an exclusive upper bound.  ``int64.max`` rounds to 2**63 when
        # coerced to float64, so comparing against it inclusively would admit
        # 2**63 and wrap it to ``int64.min`` during the cast.
        lo = float(-(2**63))
        hi_exclusive = float(2**63)
        if np.any(arr < lo) or np.any(arr >= hi_exclusive):
            raise StatisticsInputError(f"{name} contains an out-of-range timestamp")
        out = arr.astype(np.int64)
    else:
        raise StatisticsInputError(f"{name} must be an integer timestamp vector, got {arr.dtype}")
    return np.array(out, dtype=np.int64, copy=True)


def validate_epoch_seconds(
    timestamps: Any,
    *,
    name: str = "timestamps",
    allow_empty: bool = False,
) -> np.ndarray:
    """Return a checked copy of a strict, unique, increasing epoch-second clock."""
    ts = _checked_integer_array(timestamps, name=name)
    if not allow_empty and ts.size == 0:
        raise StatisticsInputError(f"{name} is empty")
    if ts.size > 1:
        if np.any(ts[1:] == ts[:-1]):
            raise StatisticsInputError(f"{name} contains duplicate timestamps")
        if np.any(ts[1:] < ts[:-1]):
            raise StatisticsInputError(f"{name} is not strictly increasing")
    # pandas and the runtime helpers are the downstream timestamp authorities.
    # This also rejects integer values outside their representable UTC range.
    if ts.size:
        try:
            converted = pd.to_datetime(ts[[0, -1]], unit="s", utc=True)
        except Exception as exc:  # pragma: no cover - pandas exception varies by version
            raise StatisticsInputError(f"{name} is outside the supported UTC range") from exc
        if converted.isna().any():
            raise StatisticsInputError(f"{name} contains an undefined UTC timestamp")
    ts.setflags(write=False)
    return ts


def validate_epoch_nanoseconds(
    timestamps: Any,
    *,
    name: str = "timestamps_ns",
    allow_empty: bool = False,
) -> np.ndarray:
    """Return a checked copy of a strict signed-int64 Unix-nanosecond clock."""
    ts = _checked_integer_array(timestamps, name=name)
    if not allow_empty and ts.size == 0:
        raise StatisticsInputError(f"{name} is empty")
    if ts.size > 1:
        if np.any(ts[1:] == ts[:-1]):
            raise StatisticsInputError(f"{name} contains duplicate timestamps")
        if np.any(ts[1:] < ts[:-1]):
            raise StatisticsInputError(f"{name} is not strictly increasing")
    if ts.size:
        try:
            converted = pd.to_datetime(ts[[0, -1]], unit="ns", utc=True)
        except Exception as exc:  # pragma: no cover - pandas exception varies by version
            raise StatisticsInputError(f"{name} is outside the supported UTC range") from exc
        if converted.isna().any():
            raise StatisticsInputError(f"{name} contains an undefined UTC timestamp")
    ts.setflags(write=False)
    return ts


def epoch_seconds_to_nanoseconds(timestamps_s: Any, *, name: str = "timestamps_s") -> np.ndarray:
    """Convert seconds to signed-int64 nanoseconds, rejecting overflow."""
    ts = validate_epoch_seconds(timestamps_s, name=name, allow_empty=True)
    int64_info = np.iinfo(np.int64)
    lo = -((-int(int64_info.min)) // NS_PER_SECOND)
    hi = int(int64_info.max) // NS_PER_SECOND
    if np.any(ts < lo) or np.any(ts > hi):
        raise StatisticsInputError(f"{name} overflows signed-int64 nanoseconds")
    out = ts.astype(np.int64, copy=True) * NS_PER_SECOND
    out.setflags(write=False)
    return out


def epoch_nanoseconds_to_seconds(
    timestamps_ns: Any,
    *,
    name: str = "timestamps_ns",
    require_exact: bool = True,
) -> np.ndarray:
    """Convert nanoseconds to seconds without silently truncating sub-second IDs."""
    ts = validate_epoch_nanoseconds(timestamps_ns, name=name, allow_empty=True)
    if require_exact and np.any(np.remainder(ts, NS_PER_SECOND) != 0):
        raise StatisticsInputError(f"{name} is not exactly representable in epoch seconds")
    out = np.floor_divide(ts, NS_PER_SECOND).astype(np.int64)
    return validate_epoch_seconds(out, name=f"{name}->seconds", allow_empty=True)


def _aligned_float_vector(values: Any, n: int, *, name: str, positive: bool = False) -> np.ndarray:
    arr = _as_one_dimensional(values, name=name).astype(float, copy=False)
    if len(arr) != n:
        raise StatisticsInputError(f"{name} length {len(arr)} != timestamp length {n}")
    if not np.isfinite(arr).all():
        raise StatisticsInputError(f"{name} contains non-finite values")
    if positive and np.any(arr <= 0):
        raise StatisticsInputError(f"{name} must be strictly positive")
    return np.array(arr, dtype=float, copy=True)


def _aligned_bool_vector(values: Any, n: int, *, name: str) -> np.ndarray:
    arr = _as_one_dimensional(values, name=name)
    if len(arr) != n:
        raise StatisticsInputError(f"{name} length {len(arr)} != timestamp length {n}")
    if arr.dtype != np.bool_:
        if not np.isin(arr, (0, 1)).all():
            raise StatisticsInputError(f"{name} must be boolean")
        arr = arr.astype(bool)
    return np.array(arr, dtype=bool, copy=True)


def canonical_wc_ret(
    observation_timestamps_s: Any,
    closes: Any,
    *,
    horizon_s: int = 900,
    tol_s: int = 10,
    lag_s: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate an observation clock, then call ``min1_production.wc_ret`` exactly."""
    ts = validate_epoch_seconds(observation_timestamps_s, name="observation_timestamps_s")
    mid = _aligned_float_vector(closes, len(ts), name="closes", positive=True)
    for key, value in (("horizon_s", horizon_s), ("tol_s", tol_s), ("lag_s", lag_s)):
        if not isinstance(value, (int, np.integer)) or int(value) < 0:
            raise StatisticsInputError(f"{key} must be a nonnegative integer")
    ret, valid = _canonical_wc_ret(ts, mid, int(horizon_s), int(tol_s), int(lag_s))
    ret = np.asarray(ret, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if ret.shape != ts.shape or valid.shape != ts.shape:
        raise StatisticsInputError("canonical wc_ret returned a row-misaligned result")
    return ret, valid


def canonical_settlement_with_exit_on_decision_grid(
    decision_timestamps_s: Any,
    observation_timestamps_s: Any,
    closes: Any,
    *,
    horizon_s: int = 900,
    tol_s: int = 10,
    lag_s: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Settle an exact decision subset and expose canonical exit observations.

    ``wc_ret`` has no separate decision-clock argument.  The issue-#9 adapter
    therefore has to prove that every completed-minute decision timestamp is an
    exact member of the sealed 10-second observation clock; interpolation or a
    nearest-time repair is forbidden.  The returned exit timestamp is derived
    with the canonical helper's exact rule -- the last observation at or before
    ``decision + lag_s + horizon_s`` -- so callers can enforce a boundary on
    the observation actually used instead of on that nominal target.
    """
    decisions = validate_epoch_seconds(decision_timestamps_s, name="decision_timestamps_s")
    if np.any(np.remainder(decisions, 60) != 0):
        raise StatisticsInputError(
            "decision_timestamps_s must identify completed-minute boundaries"
        )
    observations = validate_epoch_seconds(observation_timestamps_s, name="observation_timestamps_s")
    positions = np.searchsorted(observations, decisions, side="left")
    in_range = positions < len(observations)
    if not in_range.all() or not np.array_equal(observations[positions], decisions):
        raise StatisticsInputError("decision grid is not an exact subset of the observation clock")
    for key, value in (("horizon_s", horizon_s), ("tol_s", tol_s), ("lag_s", lag_s)):
        if not isinstance(value, (int, np.integer)) or int(value) < 0:
            raise StatisticsInputError(f"{key} must be a nonnegative integer")
    offset_s = int(lag_s) + int(horizon_s)
    if (
        offset_s > np.iinfo(np.int64).max
        or offset_s > np.iinfo(np.int64).max - int(observations[-1])
    ):
        raise StatisticsInputError("canonical settlement target overflows signed int64")
    ret, valid = canonical_wc_ret(
        observations,
        closes,
        horizon_s=horizon_s,
        tol_s=tol_s,
        lag_s=lag_s,
    )
    exit_targets = decisions + np.int64(offset_s)
    exit_positions = np.searchsorted(observations, exit_targets, side="right") - 1
    # Decisions are exact observation members and both offsets are nonnegative,
    # so every canonical exit index is at or after its decision index.  Keep an
    # explicit assertion here: clipping would conceal a clock-contract defect.
    if np.any(exit_positions < positions) or np.any(exit_positions >= len(observations)):
        raise StatisticsInputError("canonical settlement produced an invalid exit index")
    exit_timestamps = np.asarray(observations[exit_positions], dtype="int64")
    exit_timestamps.setflags(write=False)
    return ret[positions], valid[positions], exit_timestamps


def canonical_settlement_on_decision_grid(
    decision_timestamps_s: Any,
    observation_timestamps_s: Any,
    closes: Any,
    *,
    horizon_s: int = 900,
    tol_s: int = 10,
    lag_s: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Settle an exact subset of the observation clock through canonical ``wc_ret``."""

    ret, valid, _exit_timestamps = canonical_settlement_with_exit_on_decision_grid(
        decision_timestamps_s,
        observation_timestamps_s,
        closes,
        horizon_s=horizon_s,
        tol_s=tol_s,
        lag_s=lag_s,
    )
    return ret, valid


def canonical_nonoverlap_chrono(
    timestamps_s: Any,
    mask: Any,
    *,
    gap_s: int = 900,
) -> np.ndarray:
    """Validate rows/mask, then call the repository chronological scheduler."""
    ts = validate_epoch_seconds(timestamps_s, name="schedule_timestamps_s", allow_empty=True)
    selected = _aligned_bool_vector(mask, len(ts), name="schedule_mask")
    if not isinstance(gap_s, (int, np.integer)) or int(gap_s) <= 0:
        raise StatisticsInputError("gap_s must be a positive integer")
    take = np.asarray(_canonical_nonoverlap_chrono(ts, selected, int(gap_s)), dtype=np.int64)
    if take.ndim != 1 or np.any(take < 0) or np.any(take >= len(ts)):
        raise StatisticsInputError("canonical nonoverlap_chrono returned invalid row indices")
    if take.size and (not selected[take].all() or np.any(np.diff(take) <= 0)):
        raise StatisticsInputError("canonical nonoverlap_chrono returned misordered/unselected rows")
    if take.size > 1 and np.any(np.diff(ts[take]) < int(gap_s)):
        raise StatisticsInputError("canonical nonoverlap_chrono violated the requested gap")
    take.setflags(write=False)
    return take


def canonical_runtime_ny_mask(timestamps_s: Any) -> np.ndarray:
    """Return the pinned weekday NY-session and 16:35-last-start mask.

    ``sessions.session_mask`` supplies the vectorized DST-correct 08:00-17:00
    session.  Runtime ``is_ny_session`` is additionally authoritative for the
    weekday rule, and ``after_last_start_cutoff`` owns the 16:35 rejection.
    """
    ts = validate_epoch_seconds(timestamps_s, name="runtime_timestamps_s", allow_empty=True)
    session = np.asarray(_canonical_session_mask(ts, "ny"), dtype=bool)
    if session.shape != ts.shape:
        raise StatisticsInputError("canonical session_mask returned a row-misaligned result")
    runtime = np.empty(len(ts), dtype=bool)
    cutoff = np.empty(len(ts), dtype=bool)
    for i, value in enumerate(ts):
        dt = datetime.fromtimestamp(int(value), tz=timezone.utc)
        runtime[i] = bool(is_ny_session(dt))
        cutoff[i] = bool(after_last_start_cutoff(dt))
    # On weekdays the two session authorities must agree.  session_mask does
    # not itself exclude weekends, so runtime is intentionally stricter there.
    weekday = pd.to_datetime(ts, unit="s", utc=True).tz_convert(NY_TZ).weekday < 5
    if not np.array_equal(session[weekday], runtime[weekday]):
        raise StatisticsInputError("sessions.session_mask and runtime NY-session behavior disagree")
    return runtime & ~cutoff


def ny_date_keys(timestamps_s: Any) -> np.ndarray:
    """Encode America/New_York calendar dates as ordered integer YYYYMMDD keys."""
    ts = validate_epoch_seconds(timestamps_s, name="date_timestamps_s", allow_empty=True)
    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(NY_TZ)
    out = (idx.year.to_numpy(dtype=np.int64) * 10_000 + idx.month.to_numpy(dtype=np.int64) * 100
           + idx.day.to_numpy(dtype=np.int64))
    out.setflags(write=False)
    return out


def calibration_threshold(probabilities: Any, target_coverage: float) -> float:
    """Issue-#9 linear confidence quantile, with no tie-count repair or jitter."""
    p = _as_one_dimensional(probabilities, name="calibration_probabilities").astype(float, copy=False)
    if p.size == 0 or not np.isfinite(p).all():
        raise StatisticsInputError("calibration probabilities are empty or non-finite")
    if not isinstance(target_coverage, (int, float, np.integer, np.floating)):
        raise StatisticsInputError("target_coverage must be numeric")
    coverage = float(target_coverage)
    if not 0.0 < coverage <= 1.0:
        raise StatisticsInputError("target_coverage must lie in (0,1]")
    return float(np.quantile(np.abs(p - 0.5), 1.0 - coverage, method="linear"))


@dataclass(frozen=True)
class ScheduledArm:
    """One arm's combined chronological schedule on a common eligible grid."""

    selected_indices: np.ndarray
    selected_mask: np.ndarray
    direction_up: np.ndarray
    correct: np.ndarray
    utility: np.ndarray


def scheduled_arm(
    timestamps_s: Any,
    probabilities: Any,
    returns: Any,
    *,
    threshold: float,
    structural_gate: Any | None = None,
    gap_s: int = 900,
) -> ScheduledArm:
    """Apply an arm gate, one combined schedule, and ties-lose correctness.

    ``utility[t]`` is issue #9's ``u[a,t]``: zero for abstention and +1/-1
    for a scheduled correct/incorrect decision.  UP is ``p>=0.5``.
    """
    ts = validate_epoch_seconds(timestamps_s, name="arm_timestamps_s", allow_empty=True)
    p = _aligned_float_vector(probabilities, len(ts), name="arm_probabilities")
    ret = _aligned_float_vector(returns, len(ts), name="arm_returns")
    if not np.isfinite(float(threshold)) or float(threshold) < 0:
        raise StatisticsInputError("threshold must be finite and nonnegative")
    structural = (np.ones(len(ts), dtype=bool) if structural_gate is None
                  else _aligned_bool_vector(structural_gate, len(ts), name="structural_gate"))
    pre_schedule = structural & (np.abs(p - 0.5) >= float(threshold))
    take = canonical_nonoverlap_chrono(ts, pre_schedule, gap_s=gap_s)
    selected = np.zeros(len(ts), dtype=bool)
    selected[take] = True
    direction_up = p >= 0.5
    # A zero return is a loss whichever direction was selected.
    correct = ((direction_up & (ret > 0)) | (~direction_up & (ret < 0))) & selected
    utility = np.zeros(len(ts), dtype=float)
    utility[selected] = np.where(correct[selected], 1.0, -1.0)
    for value in (take, selected, direction_up, correct, utility):
        value.setflags(write=False)
    return ScheduledArm(take, selected, direction_up, correct, utility)


def _validate_calendar(values: Any, *, name: str, allow_empty: bool = False) -> np.ndarray:
    cal = _checked_integer_array(values, name=name)
    if not allow_empty and cal.size == 0:
        raise StatisticsInputError(f"{name} is empty")
    if cal.size > 1:
        if np.any(cal[1:] == cal[:-1]):
            raise StatisticsInputError(f"{name} contains duplicate dates")
        if np.any(cal[1:] < cal[:-1]):
            raise StatisticsInputError(f"{name} is not strictly increasing")
    cal.setflags(write=False)
    return cal


@dataclass(frozen=True)
class DateRatioEndpoint:
    """A ratio statistic represented by finite per-NY-date totals."""

    name: str
    date_keys: np.ndarray
    numerators: np.ndarray
    denominators: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise StatisticsInputError("endpoint name must be a non-empty string")
        dates = _validate_calendar(self.date_keys, name=f"{self.name}.date_keys", allow_empty=True)
        num = _aligned_float_vector(self.numerators, len(dates), name=f"{self.name}.numerators")
        den = _aligned_float_vector(self.denominators, len(dates), name=f"{self.name}.denominators")
        if np.any(den < 0):
            raise StatisticsInputError(f"{self.name}.denominators contains a negative value")
        num.setflags(write=False)
        den.setflags(write=False)
        object.__setattr__(self, "date_keys", dates)
        object.__setattr__(self, "numerators", num)
        object.__setattr__(self, "denominators", den)

    @property
    def observed_numerator(self) -> float:
        return float(self.numerators.sum())

    @property
    def observed_denominator(self) -> float:
        return float(self.denominators.sum())

    @property
    def estimate(self) -> float | None:
        den = self.observed_denominator
        return self.observed_numerator / den if den > 0 and np.isfinite(den) else None


def endpoint_from_date_totals(
    name: str,
    date_keys: Any,
    numerators: Any,
    denominators: Any,
) -> DateRatioEndpoint:
    return DateRatioEndpoint(name, np.asarray(date_keys), np.asarray(numerators), np.asarray(denominators))


def endpoint_from_rows(
    name: str,
    timestamps_s: Any,
    row_numerators: Any,
    row_denominators: Any,
) -> DateRatioEndpoint:
    """Aggregate an ordered row statistic into finite NY-date totals."""
    ts = validate_epoch_seconds(timestamps_s, name=f"{name}.timestamps_s", allow_empty=True)
    num = _aligned_float_vector(row_numerators, len(ts), name=f"{name}.row_numerators")
    den = _aligned_float_vector(row_denominators, len(ts), name=f"{name}.row_denominators")
    if np.any(den < 0):
        raise StatisticsInputError(f"{name}.row_denominators contains a negative value")
    dates = ny_date_keys(ts)
    if not len(dates):
        return DateRatioEndpoint(name, np.array([], dtype=np.int64), np.array([]), np.array([]))
    starts = np.r_[0, np.flatnonzero(np.diff(dates) != 0) + 1]
    unique_dates = dates[starts]
    return DateRatioEndpoint(
        name,
        unique_dates,
        np.add.reduceat(num, starts),
        np.add.reduceat(den, starts),
    )


def master_calendar_from_endpoints(endpoints: Sequence[DateRatioEndpoint]) -> np.ndarray:
    if not endpoints:
        raise StatisticsInputError("at least one endpoint is required")
    nonempty = [e.date_keys for e in endpoints if len(e.date_keys)]
    if not nonempty:
        raise StatisticsInputError("cannot infer a master calendar from empty endpoints")
    out = np.unique(np.concatenate(nonempty)).astype(np.int64)
    return _validate_calendar(out, name="master_calendar")


@dataclass(frozen=True)
class LengthInterval:
    block_length: int
    standard_error: float | None
    critical_value: float | None
    lower: float | None
    upper: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "block_length": self.block_length,
            "standard_error": self.standard_error,
            "critical_value": self.critical_value,
            "lower": self.lower,
            "upper": self.upper,
        }


@dataclass(frozen=True)
class EndpointInference:
    name: str
    estimate: float | None
    observed_denominator: float
    computable: bool
    noncomputable_reasons: tuple[str, ...]
    lower: float | None
    upper: float | None
    by_length: tuple[LengthInterval, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "estimate": self.estimate,
            "observed_denominator": self.observed_denominator,
            "computable": self.computable,
            "noncomputable_reasons": list(self.noncomputable_reasons),
            "lower": self.lower,
            "upper": self.upper,
            "by_length": [row.as_dict() for row in self.by_length],
        }


@dataclass(frozen=True)
class FamilyInference:
    family: str
    tail: str
    family_size: int
    alpha: float
    bootstrap_replicates: int
    bootstrap_seed: int
    block_lengths: tuple[int, ...]
    bonferroni_floor: float
    master_calendar: tuple[int, ...]
    endpoints: tuple[EndpointInference, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "m15-book-refresh-family-inference/v1",
            "family": self.family,
            "tail": self.tail,
            "family_size": self.family_size,
            "alpha": self.alpha,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "block_lengths": list(self.block_lengths),
            "bonferroni_floor": self.bonferroni_floor,
            "master_calendar": list(self.master_calendar),
            "endpoints": [row.as_dict() for row in self.endpoints],
        }


def _align_endpoints(
    endpoints: Sequence[DateRatioEndpoint],
    master_calendar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_endpoint, n_date = len(endpoints), len(master_calendar)
    numerator = np.zeros((n_endpoint, n_date), dtype=float)
    denominator = np.zeros((n_endpoint, n_date), dtype=float)
    for row, endpoint in enumerate(endpoints):
        if not len(endpoint.date_keys):
            continue
        positions = np.searchsorted(master_calendar, endpoint.date_keys)
        if (np.any(positions >= n_date)
                or not np.array_equal(master_calendar[positions], endpoint.date_keys)):
            raise StatisticsInputError(f"{endpoint.name}: endpoint date is outside the master calendar")
        numerator[row, positions] = endpoint.numerators
        denominator[row, positions] = endpoint.denominators
    return numerator, denominator


def _sampled_date_indices(
    rng: np.random.Generator,
    *,
    n_dates: int,
    block_length: int,
    replicates: int,
) -> np.ndarray:
    if block_length > n_dates:
        raise StatisticsInputError(
            f"block length {block_length} exceeds master-calendar length {n_dates}"
        )
    n_blocks = int(math.ceil(n_dates / block_length))
    starts = rng.integers(0, n_dates - block_length + 1, size=(replicates, n_blocks))
    sampled = (starts[:, :, None] + np.arange(block_length, dtype=np.int64)[None, None, :])
    return sampled.reshape(replicates, -1)[:, :n_dates]


def _bootstrap_ratios(
    numerator_by_date: np.ndarray,
    denominator_by_date: np.ndarray,
    sampled_indices: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_endpoint = numerator_by_date.shape[0]
    replicates = sampled_indices.shape[0]
    estimates = np.full((replicates, n_endpoint), np.nan, dtype=float)
    zero_or_bad_denominator = np.zeros(n_endpoint, dtype=bool)
    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        idx = sampled_indices[start:stop]
        # Shape while indexing is (endpoint, chunk, sampled date).  Chunking
        # bounds peak memory while preserving one synchronized calendar draw.
        num = numerator_by_date[:, idx].sum(axis=2).T
        den = denominator_by_date[:, idx].sum(axis=2).T
        bad = (~np.isfinite(den)) | (den <= 0)
        zero_or_bad_denominator |= bad.any(axis=0)
        np.divide(num, den, out=estimates[start:stop], where=~bad)
    return estimates, zero_or_bad_denominator


def infer_family(
    endpoints: Sequence[DateRatioEndpoint],
    *,
    master_calendar: Any,
    family: str,
    family_size: int,
    tail: str,
    bootstrap_seed: int,
    block_lengths: Sequence[int] = PRODUCTION_BLOCK_LENGTHS,
    bootstrap_replicates: int = PRODUCTION_BOOTSTRAP_REPLICATES,
    alpha: float = 0.05,
    bootstrap_chunk_size: int = 512,
    _seed_prefix: Sequence[int] | None = None,
    _seed_suffix: Sequence[int] = (),
) -> FamilyInference:
    """Run the synchronized fixed-family represented-date max-T procedure.

    ``_seed_prefix/_seed_suffix`` are intentionally private calibration hooks.
    Production calls create exactly ``SeedSequence([bootstrap_seed, L])``.
    The null-calibration API uses the issue's longer domain-separated seed tuple
    without maintaining a second statistics implementation.
    """
    rows = tuple(endpoints)
    if not isinstance(family_size, int) or family_size <= 0:
        raise StatisticsInputError("family_size must be a positive integer")
    if len(rows) != family_size:
        raise StatisticsInputError(
            f"{family}: received {len(rows)} endpoints for fixed family_size={family_size}"
        )
    names = [row.name for row in rows]
    if len(set(names)) != len(names):
        raise StatisticsInputError(f"{family}: endpoint names must be unique")
    if tail not in (TAIL_TWO_SIDED, TAIL_LOWER_ONLY):
        raise StatisticsInputError(f"unknown tail {tail!r}")
    if not isinstance(bootstrap_replicates, int) or bootstrap_replicates < 2:
        raise StatisticsInputError("bootstrap_replicates must be an integer >=2")
    if not isinstance(bootstrap_chunk_size, int) or bootstrap_chunk_size <= 0:
        raise StatisticsInputError("bootstrap_chunk_size must be positive")
    if not 0.0 < float(alpha) < 1.0:
        raise StatisticsInputError("alpha must lie in (0,1)")
    lengths = tuple(int(v) for v in block_lengths)
    if not lengths or any(v <= 0 for v in lengths) or len(set(lengths)) != len(lengths):
        raise StatisticsInputError("block_lengths must be unique positive integers")
    calendar = _validate_calendar(master_calendar, name="master_calendar")
    numerator, denominator = _align_endpoints(rows, calendar)
    observed_denominator = denominator.sum(axis=1)
    observed_numerator = numerator.sum(axis=1)
    observed_ok = np.isfinite(observed_denominator) & (observed_denominator > 0)
    theta = np.full(family_size, np.nan, dtype=float)
    np.divide(observed_numerator, observed_denominator, out=theta, where=observed_ok)

    stars: dict[int, np.ndarray] = {}
    ses: dict[int, np.ndarray] = {}
    bad_resample_den: dict[int, np.ndarray] = {}
    bad_se: dict[int, np.ndarray] = {}
    seed_prefix = tuple(int(v) for v in (_seed_prefix if _seed_prefix is not None else (bootstrap_seed,)))
    seed_suffix = tuple(int(v) for v in _seed_suffix)
    for length in lengths:
        material = [*seed_prefix, int(length), *seed_suffix]
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(material)))
        sampled = _sampled_date_indices(
            rng,
            n_dates=len(calendar),
            block_length=length,
            replicates=bootstrap_replicates,
        )
        estimates, bad_den = _bootstrap_ratios(
            numerator,
            denominator,
            sampled,
            chunk_size=bootstrap_chunk_size,
        )
        se = np.std(estimates, axis=0, ddof=1)
        bad_sd = (~np.isfinite(se)) | (se == 0)
        stars[length] = estimates
        ses[length] = se
        bad_resample_den[length] = bad_den
        bad_se[length] = bad_sd

    computable = observed_ok.copy()
    for length in lengths:
        computable &= ~bad_resample_den[length] & ~bad_se[length]

    floor_probability = 1.0 - float(alpha) / (
        (2 * family_size) if tail == TAIL_TWO_SIDED else family_size
    )
    bonferroni_floor = float(norm.ppf(floor_probability))
    critical: dict[int, float | None] = {}
    lower_by_length: dict[int, np.ndarray] = {}
    upper_by_length: dict[int, np.ndarray] = {}
    for length in lengths:
        lower = np.full(family_size, np.nan, dtype=float)
        upper = np.full(family_size, np.nan, dtype=float)
        if computable.any():
            standardized = (
                (stars[length][:, computable] - theta[computable][None, :])
                / ses[length][computable][None, :]
            )
            maxima = (np.max(np.abs(standardized), axis=1) if tail == TAIL_TWO_SIDED
                      else np.max(standardized, axis=1))
            empirical = float(np.quantile(maxima, 1.0 - float(alpha), method="linear"))
            crit = max(empirical, bonferroni_floor)
            critical[length] = crit
            lower[computable] = theta[computable] - crit * ses[length][computable]
            if tail == TAIL_TWO_SIDED:
                upper[computable] = theta[computable] + crit * ses[length][computable]
        else:
            critical[length] = None
        lower_by_length[length] = lower
        upper_by_length[length] = upper

    endpoint_results: list[EndpointInference] = []
    for index, endpoint in enumerate(rows):
        reasons: list[str] = []
        if not observed_ok[index]:
            reasons.append("zero_or_undefined_observed_denominator")
        for length in lengths:
            if bad_resample_den[length][index]:
                reasons.append(f"zero_or_undefined_resampled_denominator:L={length}")
            if bad_se[length][index]:
                reasons.append(f"zero_or_nonfinite_standard_error:L={length}")
        by_length: list[LengthInterval] = []
        for length in lengths:
            by_length.append(
                LengthInterval(
                    block_length=length,
                    standard_error=(float(ses[length][index])
                                    if np.isfinite(ses[length][index]) else None),
                    critical_value=critical[length] if computable[index] else None,
                    lower=(float(lower_by_length[length][index])
                           if np.isfinite(lower_by_length[length][index]) else None),
                    upper=(float(upper_by_length[length][index])
                           if np.isfinite(upper_by_length[length][index]) else None),
                )
            )
        if computable[index]:
            lower_envelope = float(min(row.lower for row in by_length if row.lower is not None))
            upper_envelope = (
                float(max(row.upper for row in by_length if row.upper is not None))
                if tail == TAIL_TWO_SIDED else None
            )
            estimate = float(theta[index])
        else:
            lower_envelope = None
            upper_envelope = None
            estimate = float(theta[index]) if np.isfinite(theta[index]) else None
        endpoint_results.append(
            EndpointInference(
                name=endpoint.name,
                estimate=estimate,
                observed_denominator=float(observed_denominator[index]),
                computable=bool(computable[index]),
                noncomputable_reasons=tuple(dict.fromkeys(reasons)),
                lower=lower_envelope,
                upper=upper_envelope,
                by_length=tuple(by_length),
            )
        )

    return FamilyInference(
        family=family,
        tail=tail,
        family_size=family_size,
        alpha=float(alpha),
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=int(bootstrap_seed),
        block_lengths=lengths,
        bonferroni_floor=bonferroni_floor,
        master_calendar=tuple(int(v) for v in calendar),
        endpoints=tuple(endpoint_results),
    )


def infer_main_family(
    endpoints: Sequence[DateRatioEndpoint],
    *,
    master_calendar: Any,
    bootstrap_seed: int = MAIN_BOOTSTRAP_SEED,
    block_lengths: Sequence[int] = PRODUCTION_BLOCK_LENGTHS,
    bootstrap_replicates: int = PRODUCTION_BOOTSTRAP_REPLICATES,
    **kwargs: Any,
) -> FamilyInference:
    return infer_family(
        endpoints,
        master_calendar=master_calendar,
        family="main_24",
        family_size=MAIN_FAMILY_SIZE,
        tail=TAIL_TWO_SIDED,
        bootstrap_seed=bootstrap_seed,
        block_lengths=block_lengths,
        bootstrap_replicates=bootstrap_replicates,
        **kwargs,
    )


def infer_control_family(
    endpoints: Sequence[DateRatioEndpoint],
    *,
    master_calendar: Any,
    bootstrap_seed: int = CONTROL_BOOTSTRAP_SEED,
    block_lengths: Sequence[int] = PRODUCTION_BLOCK_LENGTHS,
    bootstrap_replicates: int = PRODUCTION_BOOTSTRAP_REPLICATES,
    **kwargs: Any,
) -> FamilyInference:
    return infer_family(
        endpoints,
        master_calendar=master_calendar,
        family="negative_control_9",
        family_size=CONTROL_FAMILY_SIZE,
        tail=TAIL_LOWER_ONLY,
        bootstrap_seed=bootstrap_seed,
        block_lengths=block_lengths,
        bootstrap_replicates=bootstrap_replicates,
        **kwargs,
    )


def infer_shadow_family(
    endpoints: Sequence[DateRatioEndpoint],
    *,
    master_calendar: Any,
    k_shadow: int,
    bootstrap_seed: int = SHADOW_BOOTSTRAP_SEED,
    block_lengths: Sequence[int] = PRODUCTION_BLOCK_LENGTHS,
    bootstrap_replicates: int = PRODUCTION_BOOTSTRAP_REPLICATES,
    **kwargs: Any,
) -> FamilyInference:
    if not isinstance(k_shadow, int) or not 1 <= k_shadow <= len(FEATURE_SOURCE_PAIR_ORDER):
        raise StatisticsInputError("k_shadow must be an integer in [1,7]")
    return infer_family(
        endpoints,
        master_calendar=master_calendar,
        family=f"prospective_shadow_{3 * k_shadow}",
        family_size=3 * k_shadow,
        tail=TAIL_LOWER_ONLY,
        bootstrap_seed=bootstrap_seed,
        block_lengths=block_lengths,
        bootstrap_replicates=bootstrap_replicates,
        **kwargs,
    )


@dataclass(frozen=True)
class StatusDecision:
    status: str
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "reasons": list(self.reasons)}


def _require_nested_counts(
    counts: Mapping[str, Mapping[str, Any]],
    *,
    arms: Iterable[str],
    name: str,
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for arm in arms:
        if arm not in counts or not isinstance(counts[arm], Mapping):
            raise StatisticsInputError(f"{name} is missing arm {arm}")
        out[arm] = {}
        for side in ("UP", "DOWN"):
            value = counts[arm].get(side)
            if not isinstance(value, (int, np.integer)) or int(value) < 0:
                raise StatisticsInputError(f"{name}[{arm}][{side}] must be a nonnegative integer")
            out[arm][side] = int(value)
    return out


def _require_finite_mapping(values: Mapping[str, Any], keys: Sequence[str], *, name: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        value = values.get(key)
        if not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
            raise StatisticsInputError(f"{name}[{key}] must be finite")
        out[key] = float(value)
    return out


def assign_terminal_status(
    *,
    invalid_reasons: Sequence[str] = (),
    statistics_computable: bool,
    control_estimable: bool,
    side_trade_counts: Mapping[str, Mapping[str, Any]],
    combined_trade_counts: Mapping[str, Any],
    promotion_lower_bounds: Mapping[str, Any],
    retention_upper_bounds: Mapping[str, Any],
    monthly_c_side_counts: Mapping[str, Mapping[str, Any]],
    monthly_c_whole_book_yield: Mapping[str, Any],
    whole_book_pass: bool = True,
) -> StatusDecision:
    """Apply issue #9's whole-book terminal-state precedence exactly."""
    defects = tuple(str(v) for v in invalid_reasons if str(v))
    if defects:
        return StatusDecision("INVALID", defects)
    if not isinstance(statistics_computable, (bool, np.bool_)):
        raise StatisticsInputError("statistics_computable must be boolean")
    if not isinstance(control_estimable, (bool, np.bool_)):
        raise StatisticsInputError("control_estimable must be boolean")
    if not isinstance(whole_book_pass, (bool, np.bool_)):
        raise StatisticsInputError("whole_book_pass must be boolean")
    if not statistics_computable:
        return StatusDecision("INCONCLUSIVE", ("required_endpoint_not_estimable",))
    if not control_estimable:
        return StatusDecision("INCONCLUSIVE", ("control_not_estimable",))

    counts = _require_nested_counts(side_trade_counts, arms=("A", "B", "C"), name="side_trade_counts")
    combined: dict[str, int] = {}
    for arm in ("A", "B", "C"):
        value = combined_trade_counts.get(arm)
        if not isinstance(value, (int, np.integer)) or int(value) < 0:
            raise StatisticsInputError(f"combined_trade_counts[{arm}] must be a nonnegative integer")
        combined[arm] = int(value)
    lower = _require_finite_mapping(
        promotion_lower_bounds,
        ("C_minus_B_yield", "C_minus_A_yield", "C_UP_accuracy_minus_0_5", "C_DOWN_accuracy_minus_0_5"),
        name="promotion_lower_bounds",
    )
    upper = _require_finite_mapping(
        retention_upper_bounds,
        ("C_minus_A_yield", "C_UP_accuracy_minus_0_5", "C_DOWN_accuracy_minus_0_5"),
        name="retention_upper_bounds",
    )
    monthly_counts = _require_nested_counts(
        monthly_c_side_counts,
        arms=("April", "May"),
        name="monthly_c_side_counts",
    )
    monthly_yield: dict[str, float | None] = {}
    for month in ("April", "May"):
        value = monthly_c_whole_book_yield.get(month)
        if value is None:
            monthly_yield[month] = None
        elif isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
            monthly_yield[month] = float(value)
        else:
            raise StatisticsInputError(
                f"monthly_c_whole_book_yield[{month}] must be finite or null"
            )

    promotion_failures: list[str] = []
    if counts["C"]["UP"] < 50 or counts["C"]["DOWN"] < 50:
        promotion_failures.append("C_side_count_screen")
    if not all(value > 0 for value in lower.values()):
        promotion_failures.append("promotion_simultaneous_lower_bounds")
    if combined["C"] < 0.8 * combined["B"] or combined["C"] < 0.8 * combined["A"]:
        promotion_failures.append("promotion_activity_screen")
    if any(monthly_counts[month][side] < 1 for month in ("April", "May") for side in ("UP", "DOWN")):
        promotion_failures.append("promotion_month_side_presence")
    if any(monthly_yield[month] is None or monthly_yield[month] < 0 for month in ("April", "May")):
        promotion_failures.append("promotion_month_yield_guardrail")
    if not bool(whole_book_pass):
        promotion_failures.append("whole_book_failed")
    if not promotion_failures:
        return StatusDecision("PROMOTE_TO_SHADOW", ("all_promotion_conditions_passed",))

    adequately_sampled = all(counts[arm][side] >= 50 for arm in ("A", "B", "C") for side in ("UP", "DOWN"))
    if adequately_sampled:
        retain_reasons: list[str] = []
        if upper["C_minus_A_yield"] < 0:
            retain_reasons.append("C_minus_A_upper_below_zero")
        if upper["C_UP_accuracy_minus_0_5"] <= 0:
            retain_reasons.append("C_UP_accuracy_upper_at_or_below_zero")
        if upper["C_DOWN_accuracy_minus_0_5"] <= 0:
            retain_reasons.append("C_DOWN_accuracy_upper_at_or_below_zero")
        if retain_reasons:
            return StatusDecision("RETAIN_V1", tuple(retain_reasons))

    reasons = list(promotion_failures)
    if not adequately_sampled:
        reasons.append("A_B_C_not_adequately_sampled")
    if adequately_sampled:
        reasons.append("retention_intervals_do_not_clear")
    return StatusDecision("INCONCLUSIVE", tuple(dict.fromkeys(reasons)))


def shadow_empty_set_status(
    survivors: Sequence[str],
    shadow_eligible: Sequence[str],
    *,
    exclusions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Construct only the two zero-eligible shadow handoff states.

    A non-empty eligible set needs the full prospective contract and therefore
    cannot use this bounded helper.
    """
    survivor_ids = tuple(str(v) for v in survivors)
    eligible_ids = tuple(str(v) for v in shadow_eligible)
    if len(set(survivor_ids)) != len(survivor_ids) or len(set(eligible_ids)) != len(eligible_ids):
        raise StatisticsInputError("survivor and shadow-eligible IDs must be unique")
    if any(value not in survivor_ids for value in eligible_ids):
        raise StatisticsInputError("S_shadow must be an ordered subset of S")
    expected_order = tuple(value for value in survivor_ids if value in set(eligible_ids))
    if eligible_ids != expected_order:
        raise StatisticsInputError("S_shadow does not preserve S order")
    if eligible_ids:
        raise StatisticsInputError("shadow_empty_set_status requires k_shadow=0")
    exclusion_map = dict(exclusions or {})
    if not survivor_ids:
        if exclusion_map:
            raise StatisticsInputError("no-candidate status cannot contain exclusions")
        status = "no_candidates"
    else:
        missing = [value for value in survivor_ids if value not in exclusion_map]
        unknown = [value for value in exclusion_map if value not in survivor_ids]
        if missing or unknown or any(not str(exclusion_map[key]) for key in exclusion_map):
            raise StatisticsInputError(
                f"exclusions must give one non-empty reason per survivor; missing={missing}, unknown={unknown}"
            )
        status = "no_shadow_eligible_candidates"
    # Deliberately omit T0 and any max-T/CPCV family.
    return {
        "schema": "m15-book-refresh-shadow-empty/v1",
        "status": status,
        "S": list(survivor_ids),
        "S_shadow": [],
        "k_shadow": 0,
        "exclusions": exclusion_map,
    }


@dataclass(frozen=True)
class NullFixture:
    name: str
    rho_cross: float
    phi_label: float
    phi_score: float
    missing_pair_dates: float


NULL_FIXTURES = (
    NullFixture("F0", 0.0, 0.0, 0.0, 0.0),
    NullFixture("F1", 0.60, 0.0, 0.0, 0.0),
    NullFixture("F2", 0.60, 0.60, 0.60, 0.0),
    NullFixture("F3", 0.60, 0.60, 0.60, 0.20),
)
NULL_HELPER_CODES = {"main": 1, "control": 2, "shadow": 3}
NULL_STREAM_CODES = {
    "labels": 1,
    "policy_shared": 2,
    "policy_left": 3,
    "policy_right": 4,
    "policy_reference": 5,
    "missing_dates": 6,
    "bootstrap": 7,
}


@lru_cache(maxsize=1)
def _synthetic_null_clock() -> tuple[np.ndarray, np.ndarray]:
    # Sixty represented NY business dates; this range crosses the 2020 US DST
    # transition and every date contributes exactly 08:00..15:59 (480 rows).
    days = pd.bdate_range("2020-01-06", periods=NULL_SYNTHETIC_NY_DATES)
    chunks = []
    for day in days:
        local_start = pd.Timestamp(day.date(), tz=NY_TZ) + pd.Timedelta(
            hours=NULL_NY_START_HOUR
        )
        chunks.append(
            pd.date_range(
                local_start, periods=NULL_ROWS_PER_PAIR_DATE, freq="min"
            ).tz_convert("UTC")
        )
    idx = chunks[0].append(chunks[1:])
    ts = (idx.as_unit("ns").asi8 // NS_PER_SECOND).astype(np.int64)
    ts = validate_epoch_seconds(ts, name="synthetic_null_clock")
    runtime = canonical_runtime_ny_mask(ts)
    if not runtime.all():
        raise StatisticsInputError("sealed null clock contains a row outside the canonical runtime window")
    dates = ny_date_keys(ts)
    return ts, dates


def _null_rng(
    fixture_index: int,
    helper_code: int,
    k_shadow: int,
    campaign_index: int,
    *,
    length: int = 0,
    stream_code: int,
    seed_identity_material: Sequence[int] = (),
) -> np.random.Generator:
    material = [
        NULL_TEST_SEED,
        *(int(value) for value in seed_identity_material),
        int(fixture_index),
        int(helper_code),
        int(k_shadow),
        int(campaign_index),
        int(length),
        int(stream_code),
    ]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(material)))


def _stationary_ar1(rng: np.random.Generator, n_series: int, n: int, phi: float) -> np.ndarray:
    if not 0.0 <= phi < 1.0:
        raise StatisticsInputError("sealed AR(1) phi must lie in [0,1)")
    if phi == 0:
        return rng.standard_normal((n_series, n))
    initial = rng.standard_normal((n_series, 1))
    innovations = rng.standard_normal((n_series, n - 1))
    scale = math.sqrt(1.0 - phi * phi)
    rest, _ = lfilter(
        [scale],
        [1.0, -phi],
        innovations,
        axis=1,
        zi=phi * initial,
    )
    return np.concatenate([initial, rest], axis=1)


def _cross_pair_field(
    rng: np.random.Generator,
    *,
    n_pairs: int,
    n_rows: int,
    rho_cross: float,
    phi: float,
) -> np.ndarray:
    if not 0.0 <= rho_cross <= 1.0:
        raise StatisticsInputError("sealed rho_cross must lie in [0,1]")
    common = _stationary_ar1(rng, 1, n_rows, phi)
    idiosyncratic = _stationary_ar1(rng, n_pairs, n_rows, phi)
    return math.sqrt(rho_cross) * common + math.sqrt(1.0 - rho_cross) * idiosyncratic


def _correlated_policy_fields(
    fixture: NullFixture,
    *,
    fixture_index: int,
    helper_code: int,
    k_shadow: int,
    campaign_index: int,
    n_pairs: int,
    n_rows: int,
    seed_identity_material: Sequence[int] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shared = _cross_pair_field(
        _null_rng(fixture_index, helper_code, k_shadow, campaign_index,
                  stream_code=NULL_STREAM_CODES["policy_shared"],
                  seed_identity_material=seed_identity_material),
        n_pairs=n_pairs, n_rows=n_rows, rho_cross=fixture.rho_cross, phi=fixture.phi_score,
    )
    left_noise = _cross_pair_field(
        _null_rng(fixture_index, helper_code, k_shadow, campaign_index,
                  stream_code=NULL_STREAM_CODES["policy_left"],
                  seed_identity_material=seed_identity_material),
        n_pairs=n_pairs, n_rows=n_rows, rho_cross=fixture.rho_cross, phi=fixture.phi_score,
    )
    right_noise = _cross_pair_field(
        _null_rng(fixture_index, helper_code, k_shadow, campaign_index,
                  stream_code=NULL_STREAM_CODES["policy_right"],
                  seed_identity_material=seed_identity_material),
        n_pairs=n_pairs, n_rows=n_rows, rho_cross=fixture.rho_cross, phi=fixture.phi_score,
    )
    rho_arm = NULL_RHO_ARM
    left = math.sqrt(rho_arm) * shared + math.sqrt(1.0 - rho_arm) * left_noise
    right = math.sqrt(rho_arm) * shared + math.sqrt(1.0 - rho_arm) * right_noise
    reference = _cross_pair_field(
        _null_rng(fixture_index, helper_code, k_shadow, campaign_index,
                  stream_code=NULL_STREAM_CODES["policy_reference"],
                  seed_identity_material=seed_identity_material),
        n_pairs=n_pairs, n_rows=n_rows, rho_cross=fixture.rho_cross, phi=fixture.phi_score,
    )
    return left, right, reference


def _null_availability(
    fixture: NullFixture,
    *,
    fixture_index: int,
    helper_code: int,
    k_shadow: int,
    campaign_index: int,
    n_pairs: int,
    seed_identity_material: Sequence[int] = (),
) -> np.ndarray:
    if fixture.missing_pair_dates == 0:
        return np.ones((n_pairs, NULL_SYNTHETIC_NY_DATES), dtype=bool)
    rng = _null_rng(
        fixture_index,
        helper_code,
        k_shadow,
        campaign_index,
        stream_code=NULL_STREAM_CODES["missing_dates"],
        seed_identity_material=seed_identity_material,
    )
    return (
        rng.random((n_pairs, NULL_SYNTHETIC_NY_DATES))
        >= fixture.missing_pair_dates
    )


def _null_scheduled_arrays(
    timestamps: np.ndarray,
    labels: np.ndarray,
    latents: np.ndarray,
    availability_by_date: np.ndarray,
    pairs: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_pairs, n_rows = latents.shape
    probability = ndtr(latents)
    utility = np.zeros((n_pairs, n_rows), dtype=float)
    selected = np.zeros((n_pairs, n_rows), dtype=bool)
    direction = probability >= 0.5
    correct = np.zeros((n_pairs, n_rows), dtype=bool)
    row_available = np.repeat(
        availability_by_date, NULL_ROWS_PER_PAIR_DATE, axis=1
    )
    for pair_index, pair in enumerate(pairs):
        available = row_available[pair_index]
        threshold = calibration_threshold(
            probability[pair_index, available], _NULL_MECHANICS_COVERAGE[pair]
        )
        take = canonical_nonoverlap_chrono(
            timestamps,
            available & (np.abs(probability[pair_index] - 0.5) >= threshold),
            gap_s=900,
        )
        selected[pair_index, take] = True
        corr = direction[pair_index] == labels[pair_index]
        correct[pair_index, take] = corr[take]
        utility[pair_index, take] = np.where(corr[take], 1.0, -1.0)
    return utility, selected, direction, correct


def _date_sum(values: np.ndarray) -> np.ndarray:
    return values.reshape(
        values.shape[0], NULL_SYNTHETIC_NY_DATES, NULL_ROWS_PER_PAIR_DATE
    ).sum(axis=2)


def _null_endpoints(
    fixture: NullFixture,
    *,
    fixture_index: int,
    helper: str,
    k_shadow: int,
    campaign_index: int,
    ordered_shadow_pairs: Sequence[str] = (),
    shadow_accuracy_null: float = 0.5,
    seed_identity_material: Sequence[int] = (),
) -> tuple[tuple[DateRatioEndpoint, ...], np.ndarray]:
    if helper not in NULL_HELPER_CODES:
        raise StatisticsInputError(f"unknown null helper {helper!r}")
    if helper != "shadow" and k_shadow != 0:
        raise StatisticsInputError("k_shadow is only valid for the shadow helper")
    shadow_pairs = tuple(str(pair) for pair in ordered_shadow_pairs)
    if helper == "shadow":
        if not 1 <= k_shadow <= len(FEATURE_SOURCE_PAIR_ORDER):
            raise StatisticsInputError("shadow null cells require k_shadow in [1,7]")
        if len(shadow_pairs) != k_shadow or len(set(shadow_pairs)) != k_shadow:
            raise StatisticsInputError(
                "shadow null cells require exactly k_shadow unique ordered pairs"
            )
        if not 0.0 < float(shadow_accuracy_null) < 1.0:
            raise StatisticsInputError("shadow_accuracy_null must lie strictly inside (0,1)")
    elif shadow_pairs:
        raise StatisticsInputError("ordered_shadow_pairs is only valid for the shadow helper")
    helper_code = NULL_HELPER_CODES[helper]
    n_pairs = len(PAIR_ORDER) if helper == "main" else (3 if helper == "control" else k_shadow)
    pairs = (("EURUSD", "GBPUSD", "USDJPY") if helper == "control"
             else (shadow_pairs if helper == "shadow" else PAIR_ORDER))
    timestamps, all_date_rows = _synthetic_null_clock()
    calendar = np.unique(all_date_rows)
    n_rows = len(timestamps)
    labels_latent = _cross_pair_field(
        _null_rng(fixture_index, helper_code, k_shadow, campaign_index,
                  stream_code=NULL_STREAM_CODES["labels"],
                  seed_identity_material=seed_identity_material),
        n_pairs=n_pairs,
        n_rows=n_rows,
        rho_cross=fixture.rho_cross,
        phi=fixture.phi_label,
    )
    left, right, reference = _correlated_policy_fields(
        fixture,
        fixture_index=fixture_index,
        helper_code=helper_code,
        k_shadow=k_shadow,
        campaign_index=campaign_index,
        n_pairs=n_pairs,
        seed_identity_material=seed_identity_material,
        n_rows=n_rows,
    )
    if helper == "shadow":
        # Shadow-boundary fixtures need all three endpoints centered at their
        # registered nulls.  Both policies therefore share a direction while
        # retaining distinct correlated confidence magnitudes.  A separate
        # AR/cross-pair field determines correctness at the requested boundary,
        # independently of score magnitude and scheduling.
        shared_sign = np.where(right >= 0.0, 1.0, -1.0)
        right = shared_sign * np.abs(right)
        left = shared_sign * np.abs(left)
        shared_direction = right >= 0.0
        correctness_boundary = float(norm.ppf(1.0 - shadow_accuracy_null))
        correct_at_null = labels_latent >= correctness_boundary
        labels = np.where(correct_at_null, shared_direction, ~shared_direction)
    else:
        labels = labels_latent >= 0
    availability = _null_availability(
        fixture,
        fixture_index=fixture_index,
        helper_code=helper_code,
        k_shadow=k_shadow,
        campaign_index=campaign_index,
        n_pairs=n_pairs,
        seed_identity_material=seed_identity_material,
    )
    left_u, left_sel, left_dir, left_corr = _null_scheduled_arrays(
        timestamps, labels, left, availability, pairs
    )
    right_u, right_sel, right_dir, right_corr = _null_scheduled_arrays(
        timestamps, labels, right, availability, pairs
    )
    reference_u = None
    if helper == "main":
        reference_u, _, _, _ = _null_scheduled_arrays(
            timestamps, labels, reference, availability, pairs
        )
    eligible_den = availability.astype(float) * float(NULL_ROWS_PER_PAIR_DATE)
    endpoints: list[DateRatioEndpoint] = []

    if helper == "main":
        assert reference_u is not None  # established by the helper branch above
        for pair_index, pair in enumerate(pairs):
            endpoints.append(DateRatioEndpoint(
                f"{pair}:C_minus_B_yield", calendar,
                _date_sum((right_u - left_u)[pair_index:pair_index + 1])[0],
                eligible_den[pair_index],
            ))
        for pair_index, pair in enumerate(pairs):
            endpoints.append(DateRatioEndpoint(
                f"{pair}:C_minus_A_yield", calendar,
                _date_sum((right_u - reference_u)[pair_index:pair_index + 1])[0],
                eligible_den[pair_index],
            ))
        for side, direction_up in (("UP", True), ("DOWN", False)):
            for pair_index, pair in enumerate(pairs):
                side_mask = right_sel[pair_index] & (right_dir[pair_index] == direction_up)
                numerator = np.where(side_mask, right_corr[pair_index].astype(float) - 0.5, 0.0)
                endpoints.append(DateRatioEndpoint(
                    f"{pair}:C_{side}_accuracy_minus_0_5", calendar,
                    _date_sum(numerator[None, :])[0],
                    _date_sum(side_mask.astype(float)[None, :])[0],
                ))
    elif helper == "control":
        # One representative for each of the three adapter families.
        for pair_index, pair in enumerate(pairs):
            endpoints.append(DateRatioEndpoint(
                f"{pair}:C_perm_minus_B_perm_yield", calendar,
                _date_sum((right_u - left_u)[pair_index:pair_index + 1])[0],
                eligible_den[pair_index],
            ))
            for side, direction_up in (("UP", True), ("DOWN", False)):
                side_mask = right_sel[pair_index] & (right_dir[pair_index] == direction_up)
                numerator = np.where(side_mask, right_corr[pair_index].astype(float) - 0.5, 0.0)
                endpoints.append(DateRatioEndpoint(
                    f"{pair}:C_perm_{side}_accuracy_minus_0_5", calendar,
                    _date_sum(numerator[None, :])[0],
                    _date_sum(side_mask.astype(float)[None, :])[0],
                ))
    else:
        for pair_index, pair in enumerate(pairs):
            endpoints.append(DateRatioEndpoint(
                f"{pair}:candidate_minus_v1_yield", calendar,
                _date_sum((right_u - left_u)[pair_index:pair_index + 1])[0],
                eligible_den[pair_index],
            ))
            for side, direction_up in (("UP", True), ("DOWN", False)):
                side_mask = right_sel[pair_index] & (right_dir[pair_index] == direction_up)
                numerator = np.where(
                    side_mask,
                    right_corr[pair_index].astype(float) - float(shadow_accuracy_null),
                    0.0,
                )
                endpoints.append(DateRatioEndpoint(
                    (
                        f"{pair}:candidate_{side}_accuracy_minus_"
                        f"{str(shadow_accuracy_null).replace('.', '_')}"
                    ),
                    calendar,
                    _date_sum(numerator[None, :])[0],
                    _date_sum(side_mask.astype(float)[None, :])[0],
                ))
    return tuple(endpoints), calendar


def _null_inference(
    fixture: NullFixture,
    *,
    fixture_index: int,
    helper: str,
    k_shadow: int,
    campaign_index: int,
    bootstrap_replicates: int,
    block_lengths: Sequence[int],
    ordered_shadow_pairs: Sequence[str] = (),
    shadow_accuracy_null: float = 0.5,
    seed_identity_material: Sequence[int] = (),
) -> FamilyInference:
    endpoints, calendar = _null_endpoints(
        fixture,
        fixture_index=fixture_index,
        helper=helper,
        k_shadow=k_shadow,
        campaign_index=campaign_index,
        ordered_shadow_pairs=ordered_shadow_pairs,
        shadow_accuracy_null=shadow_accuracy_null,
        seed_identity_material=seed_identity_material,
    )
    seed_prefix = (
        NULL_TEST_SEED,
        *(int(value) for value in seed_identity_material),
        fixture_index,
        NULL_HELPER_CODES[helper],
        k_shadow,
        campaign_index,
    )
    kwargs = {
        "master_calendar": calendar,
        "bootstrap_seed": NULL_TEST_SEED,
        "block_lengths": block_lengths,
        "bootstrap_replicates": bootstrap_replicates,
        "_seed_prefix": seed_prefix,
        "_seed_suffix": (NULL_STREAM_CODES["bootstrap"],),
    }
    if helper == "main":
        return infer_main_family(endpoints, **kwargs)
    if helper == "control":
        return infer_control_family(endpoints, **kwargs)
    return infer_shadow_family(endpoints, k_shadow=k_shadow, **kwargs)


def _is_null_false_positive(result: FamilyInference) -> bool:
    if result.tail == TAIL_TWO_SIDED:
        return any(
            endpoint.computable
            and endpoint.lower is not None
            and endpoint.upper is not None
            and (endpoint.lower > 0 or endpoint.upper < 0)
            for endpoint in result.endpoints
        )
    return any(
        endpoint.computable and endpoint.lower is not None and endpoint.lower > 0
        for endpoint in result.endpoints
    )


def _result_digest(result: FamilyInference) -> str:
    payload = json.dumps(
        result.as_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class NullCalibrationCellResult:
    fixture: str
    helper: str
    k_shadow: int
    ordered_shadow_pairs: tuple[str, ...]
    accuracy_null: float
    binding: bool
    calibration_role: str
    campaign_count: int
    bootstrap_replicates: int
    block_lengths: tuple[int, ...]
    false_positive_count: int
    false_positive_limit: int
    clopper_pearson_upper_95: float
    cell_passed: bool
    accepted: bool
    deterministic_repeat: bool
    campaign_digest: str
    seed_binding: Mapping[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        value = {
            "schema": "m15-book-refresh-null-calibration-cell/v1",
            "fixture": self.fixture,
            "helper": self.helper,
            "k_shadow": self.k_shadow,
            "ordered_shadow_pairs": list(self.ordered_shadow_pairs),
            "accuracy_null": self.accuracy_null,
            "binding": self.binding,
            "calibration_role": self.calibration_role,
            "campaign_count": self.campaign_count,
            "bootstrap_replicates": self.bootstrap_replicates,
            "block_lengths": list(self.block_lengths),
            "false_positive_count": self.false_positive_count,
            "false_positive_limit": self.false_positive_limit,
            "clopper_pearson_upper_95": self.clopper_pearson_upper_95,
            "cell_passed": self.cell_passed,
            "accepted": self.accepted,
            "deterministic_repeat": self.deterministic_repeat,
            "campaign_digest": self.campaign_digest,
        }
        if self.seed_binding is not None:
            value["seed_binding"] = json.loads(json.dumps(self.seed_binding))
        return value


def _run_null_calibration_cell_once(
    fixture: NullFixture,
    *,
    fixture_index: int,
    helper: str,
    k_shadow: int,
    campaign_count: int,
    bootstrap_replicates: int,
    block_lengths: Sequence[int],
    ordered_shadow_pairs: Sequence[str],
    shadow_accuracy_null: float,
    seed_identity_material: Sequence[int],
) -> tuple[int, str]:
    false_positive_count = 0
    digest = hashlib.sha256()
    for campaign_index in range(campaign_count):
        result = _null_inference(
            fixture,
            fixture_index=fixture_index,
            helper=helper,
            k_shadow=k_shadow,
            campaign_index=campaign_index,
            bootstrap_replicates=bootstrap_replicates,
            block_lengths=block_lengths,
            ordered_shadow_pairs=ordered_shadow_pairs,
            shadow_accuracy_null=shadow_accuracy_null,
            seed_identity_material=seed_identity_material,
        )
        false_positive_count += int(_is_null_false_positive(result))
        digest.update(bytes.fromhex(_result_digest(result)))
    return false_positive_count, digest.hexdigest()


def _resolve_null_fixture(fixture: str | NullFixture) -> tuple[int, NullFixture]:
    if isinstance(fixture, str):
        matches = [(i, row) for i, row in enumerate(NULL_FIXTURES) if row.name == fixture]
        if len(matches) != 1:
            raise StatisticsInputError(f"unknown null fixture {fixture!r}")
        return matches[0]
    if isinstance(fixture, NullFixture):
        matches = [(i, row) for i, row in enumerate(NULL_FIXTURES) if row == fixture]
        if len(matches) != 1:
            raise StatisticsInputError("null fixture is not one of the four sealed fixtures")
        return matches[0]
    raise StatisticsInputError("fixture must be a fixture name or NullFixture")


def _ordered_subset(
    values: Sequence[str],
    *,
    universe: Sequence[str],
    name: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise StatisticsInputError(f"{name} must be a sequence of pair names")
    pairs = tuple(str(value) for value in values)
    if not pairs:
        raise StatisticsInputError(f"{name} must be nonempty")
    if len(set(pairs)) != len(pairs):
        raise StatisticsInputError(f"{name} contains duplicate pairs")
    unknown = sorted(set(pairs).difference(universe))
    if unknown:
        raise StatisticsInputError(f"{name} contains unknown pairs {unknown}")
    expected = tuple(pair for pair in universe if pair in set(pairs))
    if pairs != expected:
        raise StatisticsInputError(f"{name} does not preserve the sealed pair order")
    return pairs


def deferred_shadow_seed_binding(
    *,
    ordered_shadow_pairs: Sequence[str],
    ordered_shadow_bundle_ids: Sequence[str],
) -> dict[str, Any]:
    """Derive domain-separated SeedSequence material from exact C bundles."""

    pairs = _ordered_subset(
        ordered_shadow_pairs,
        universe=PAIR_ORDER,
        name="ordered_shadow_pairs",
    )
    if isinstance(ordered_shadow_bundle_ids, (str, bytes)):
        raise StatisticsInputError(
            "ordered_shadow_bundle_ids must be a sequence of bundle IDs"
        )
    bundle_ids = tuple(str(value) for value in ordered_shadow_bundle_ids)
    if len(bundle_ids) != len(pairs) or len(set(bundle_ids)) != len(bundle_ids):
        raise StatisticsInputError(
            "ordered shadow bundle IDs must be unique and pair-aligned"
        )
    if any(
        len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
        for value in bundle_ids
    ):
        raise StatisticsInputError(
            "ordered shadow bundle IDs must be lowercase SHA-256 identities"
        )
    identity_payload = [
        {"pair": pair, "candidate_C_bundle_id": bundle_id}
        for pair, bundle_id in zip(pairs, bundle_ids)
    ]
    encoded = json.dumps(
        identity_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    domain = DEFERRED_SHADOW_SEED_DOMAIN.encode("utf-8")
    digest.update(len(domain).to_bytes(8, "big"))
    digest.update(domain)
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    digest_bytes = digest.digest()
    words = [
        int.from_bytes(digest_bytes[offset:offset + 4], "big")
        for offset in range(0, len(digest_bytes), 4)
    ]
    return {
        "domain": DEFERRED_SHADOW_SEED_DOMAIN,
        "shadow_phase_code": DEFERRED_SHADOW_PHASE_CODE,
        "identity_payload": identity_payload,
        "identity_sha256": digest.hexdigest(),
        "identity_digest_u32be": words,
        "seed_sequence_layout": [
            "stats_null_test_seed",
            "shadow_phase_code",
            "identity_digest_u32be[0..7]",
            "fixture_index",
            "helper_code",
            "k_shadow",
            "campaign_index",
            "L_or_0",
            "stream_code",
        ],
    }


def _run_configured_null_calibration_cell(
    fixture: str | NullFixture,
    *,
    helper: str,
    ordered_shadow_pairs: Sequence[str],
    shadow_accuracy_null: float,
    binding: bool,
    calibration_role: str,
    seed_binding: Mapping[str, Any] | None = None,
    campaign_count: int = NULL_CAMPAIGNS_PER_CELL,
    bootstrap_replicates: int = NULL_BOOTSTRAP_REPLICATES,
    block_lengths: Sequence[int] = NULL_BLOCK_LENGTHS,
    verify_repeat: bool = True,
) -> NullCalibrationCellResult:
    fixture_index, fixture_obj = _resolve_null_fixture(fixture)
    if helper not in NULL_HELPER_CODES:
        raise StatisticsInputError(f"unknown null helper {helper!r}")
    shadow_pairs = tuple(str(pair) for pair in ordered_shadow_pairs)
    if helper == "shadow":
        if not shadow_pairs:
            raise StatisticsInputError("shadow calibration requires ordered shadow pairs")
    elif shadow_pairs:
        raise StatisticsInputError("non-shadow calibration forbids ordered shadow pairs")
    k_shadow = len(shadow_pairs)
    if type(binding) is not bool:
        raise StatisticsInputError("binding must be boolean")
    roles = {
        "binding_current": (True, {"main", "control"}),
        "nonbinding_shadow_mechanics": (False, {"shadow"}),
        "binding_deferred_shadow": (True, {"shadow"}),
    }
    expected = roles.get(calibration_role)
    if expected is None or (binding, helper in expected[1]) != (expected[0], True):
        raise StatisticsInputError("null calibration role/helper/binding combination is invalid")
    if calibration_role == "binding_deferred_shadow":
        if not isinstance(seed_binding, Mapping):
            raise StatisticsInputError(
                "deferred shadow calibration requires bundle seed binding"
            )
        words = seed_binding.get("identity_digest_u32be")
        if (
            seed_binding.get("shadow_phase_code") != DEFERRED_SHADOW_PHASE_CODE
            or not isinstance(words, list)
            or len(words) != 8
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 0xFFFFFFFF
                for value in words
            )
        ):
            raise StatisticsInputError("deferred shadow seed binding is malformed")
        seed_identity_material = (
            DEFERRED_SHADOW_PHASE_CODE,
            *(int(value) for value in words),
        )
    else:
        if seed_binding is not None:
            raise StatisticsInputError(
                "current/mechanics null cells forbid deferred bundle seed binding"
            )
        seed_identity_material = ()
    if not isinstance(campaign_count, int) or campaign_count <= 0:
        raise StatisticsInputError("campaign_count must be a positive integer")
    if not isinstance(bootstrap_replicates, int) or bootstrap_replicates < 2:
        raise StatisticsInputError("bootstrap_replicates must be >=2")
    lengths = tuple(int(v) for v in block_lengths)
    first_count, first_digest = _run_null_calibration_cell_once(
        fixture_obj,
        fixture_index=fixture_index,
        helper=helper,
        k_shadow=k_shadow,
        campaign_count=campaign_count,
        bootstrap_replicates=bootstrap_replicates,
        block_lengths=lengths,
        ordered_shadow_pairs=shadow_pairs,
        shadow_accuracy_null=shadow_accuracy_null,
        seed_identity_material=seed_identity_material,
    )
    # Absence of a repeat is not evidence of determinism.  Bounded diagnostics
    # may opt out of the second pass, but they must then remain ineligible for
    # use as an accepted calibration artifact.
    repeatable = False
    if verify_repeat:
        second_count, second_digest = _run_null_calibration_cell_once(
            fixture_obj,
            fixture_index=fixture_index,
            helper=helper,
            k_shadow=k_shadow,
            campaign_count=campaign_count,
            bootstrap_replicates=bootstrap_replicates,
            block_lengths=lengths,
            ordered_shadow_pairs=shadow_pairs,
            shadow_accuracy_null=shadow_accuracy_null,
            seed_identity_material=seed_identity_material,
        )
        repeatable = first_count == second_count and first_digest == second_digest
    cp_upper = (
        1.0 if first_count == campaign_count
        else float(beta.ppf(0.95, first_count + 1, campaign_count - first_count))
    )
    return NullCalibrationCellResult(
        fixture=fixture_obj.name,
        helper=helper,
        k_shadow=k_shadow,
        ordered_shadow_pairs=shadow_pairs,
        accuracy_null=float(shadow_accuracy_null),
        binding=binding,
        calibration_role=calibration_role,
        campaign_count=campaign_count,
        bootstrap_replicates=bootstrap_replicates,
        block_lengths=lengths,
        false_positive_count=first_count,
        false_positive_limit=NULL_FALSE_POSITIVE_LIMIT,
        clopper_pearson_upper_95=cp_upper,
        cell_passed=bool(first_count <= NULL_FALSE_POSITIVE_LIMIT and repeatable),
        accepted=bool(binding and first_count <= NULL_FALSE_POSITIVE_LIMIT and repeatable),
        deterministic_repeat=repeatable,
        campaign_digest=first_digest,
        seed_binding=seed_binding,
    )


def run_null_calibration_cell(
    fixture: str | NullFixture,
    *,
    helper: str,
    k_shadow: int = 0,
    campaign_count: int = NULL_CAMPAIGNS_PER_CELL,
    bootstrap_replicates: int = NULL_BOOTSTRAP_REPLICATES,
    block_lengths: Sequence[int] = NULL_BLOCK_LENGTHS,
    verify_repeat: bool = True,
) -> NullCalibrationCellResult:
    """Exercise one current binding main/control null cell.

    Shadow cells deliberately have separate APIs: prefix mechanics are
    nonbinding, while boundary calibration requires the exact post-replay
    ``S_shadow`` order.  Bounded counts are diagnostics and are not eligible to
    become campaign acceptance artifacts even if this pure cell result passes.
    """
    if helper == "shadow":
        raise StatisticsInputError(
            "shadow calibration requires a dedicated mechanics or deferred-S_shadow API"
        )
    if helper not in {"main", "control"}:
        raise StatisticsInputError(f"unknown current binding null helper {helper!r}")
    if k_shadow != 0:
        raise StatisticsInputError("current binding null calibration requires k_shadow=0")
    return _run_configured_null_calibration_cell(
        fixture,
        helper=helper,
        ordered_shadow_pairs=(),
        shadow_accuracy_null=0.5,
        binding=True,
        calibration_role="binding_current",
        campaign_count=campaign_count,
        bootstrap_replicates=bootstrap_replicates,
        block_lengths=block_lengths,
        verify_repeat=verify_repeat,
    )


def run_null_calibration_suite(
    *,
    campaign_count: int = NULL_CAMPAIGNS_PER_CELL,
    bootstrap_replicates: int = NULL_BOOTSTRAP_REPLICATES,
    block_lengths: Sequence[int] = NULL_BLOCK_LENGTHS,
    verify_repeat: bool = True,
) -> tuple[NullCalibrationCellResult, ...]:
    """Run only the eight current binding F0..F3 × {main, control} cells."""
    cells: list[NullCalibrationCellResult] = []
    for fixture in NULL_FIXTURES:
        cells.append(run_null_calibration_cell(
            fixture,
            helper="main",
            campaign_count=campaign_count,
            bootstrap_replicates=bootstrap_replicates,
            block_lengths=block_lengths,
            verify_repeat=verify_repeat,
        ))
        cells.append(run_null_calibration_cell(
            fixture,
            helper="control",
            campaign_count=campaign_count,
            bootstrap_replicates=bootstrap_replicates,
            block_lengths=block_lengths,
            verify_repeat=verify_repeat,
        ))
    return tuple(cells)


def run_shadow_null_mechanics_suite(
    *,
    campaign_count: int = NULL_CAMPAIGNS_PER_CELL,
    bootstrap_replicates: int = NULL_BOOTSTRAP_REPLICATES,
    block_lengths: Sequence[int] = NULL_BLOCK_LENGTHS,
    verify_repeat: bool = True,
) -> tuple[NullCalibrationCellResult, ...]:
    """Run 28 nonbinding prefix/.5 cells for shadow mechanics only.

    These source-pair prefixes are not candidate sets, every result has
    ``binding=False`` and ``accepted=False``, and this suite is excluded from
    the current acceptance artifact.
    """
    cells: list[NullCalibrationCellResult] = []
    for fixture in NULL_FIXTURES:
        for k_shadow in range(1, len(FEATURE_SOURCE_PAIR_ORDER) + 1):
            cells.append(_run_configured_null_calibration_cell(
                fixture,
                helper="shadow",
                ordered_shadow_pairs=FEATURE_SOURCE_PAIR_ORDER[:k_shadow],
                shadow_accuracy_null=0.5,
                binding=False,
                calibration_role="nonbinding_shadow_mechanics",
                campaign_count=campaign_count,
                bootstrap_replicates=bootstrap_replicates,
                block_lengths=block_lengths,
                verify_repeat=verify_repeat,
            ))
    return tuple(cells)


def run_deferred_shadow_null_calibration_suite(
    *,
    ordered_shadow_pairs: Sequence[str],
    ordered_shadow_bundle_ids: Sequence[str],
    campaign_count: int = NULL_CAMPAIGNS_PER_CELL,
    bootstrap_replicates: int = NULL_BOOTSTRAP_REPLICATES,
    block_lengths: Sequence[int] = NULL_BLOCK_LENGTHS,
    verify_repeat: bool = True,
) -> tuple[NullCalibrationCellResult, ...]:
    """Run four binding .541 cells for an exact sealed ``S_shadow`` subset.

    The explicit nonempty, duplicate-free, order-preserving target subset and
    aligned candidate-C bundle identities prevent a pre-replay prefix from
    masquerading as the eventual survivor family.  The campaign runner must
    call this only after sealing ``S_shadow``.
    """
    pairs = _ordered_subset(
        ordered_shadow_pairs,
        universe=PAIR_ORDER,
        name="ordered_shadow_pairs",
    )
    seed_binding = deferred_shadow_seed_binding(
        ordered_shadow_pairs=pairs,
        ordered_shadow_bundle_ids=ordered_shadow_bundle_ids,
    )
    return tuple(
        _run_configured_null_calibration_cell(
            fixture,
            helper="shadow",
            ordered_shadow_pairs=pairs,
            shadow_accuracy_null=0.541,
            binding=True,
            calibration_role="binding_deferred_shadow",
            seed_binding=seed_binding,
            campaign_count=campaign_count,
            bootstrap_replicates=bootstrap_replicates,
            block_lengths=block_lengths,
            verify_repeat=verify_repeat,
        )
        for fixture in NULL_FIXTURES
    )


__all__ = [
    "CONTROL_BOOTSTRAP_SEED",
    "CONTROL_FAMILY_SIZE",
    "DEFERRED_SHADOW_PHASE_CODE",
    "DEFERRED_SHADOW_SEED_DOMAIN",
    "DateRatioEndpoint",
    "EndpointInference",
    "FEATURE_SOURCE_PAIR_ORDER",
    "FamilyInference",
    "MAIN_BOOTSTRAP_SEED",
    "MAIN_FAMILY_SIZE",
    "NULL_BLOCK_LENGTHS",
    "NULL_BOOTSTRAP_REPLICATES",
    "NULL_CAMPAIGNS_PER_CELL",
    "NULL_FALSE_POSITIVE_LIMIT",
    "NULL_FIXTURES",
    "NULL_HELPER_CODES",
    "NULL_RHO_ARM",
    "NULL_NY_START_HOUR",
    "NULL_ROW_MINUTES",
    "NULL_ROWS_PER_PAIR_DATE",
    "NULL_STREAM_CODES",
    "NULL_SYNTHETIC_NY_DATES",
    "NULL_TEST_SEED",
    "NullCalibrationCellResult",
    "NullFixture",
    "PAIR_COVERAGE",
    "PAIR_ORDER",
    "PRODUCTION_BLOCK_LENGTHS",
    "PRODUCTION_BOOTSTRAP_REPLICATES",
    "SHADOW_BOOTSTRAP_SEED",
    "ScheduledArm",
    "StatisticsInputError",
    "StatusDecision",
    "assign_terminal_status",
    "calibration_threshold",
    "canonical_nonoverlap_chrono",
    "canonical_runtime_ny_mask",
    "canonical_settlement_on_decision_grid",
    "canonical_settlement_with_exit_on_decision_grid",
    "canonical_wc_ret",
    "deferred_shadow_seed_binding",
    "endpoint_from_date_totals",
    "endpoint_from_rows",
    "epoch_nanoseconds_to_seconds",
    "epoch_seconds_to_nanoseconds",
    "infer_control_family",
    "infer_family",
    "infer_main_family",
    "infer_shadow_family",
    "master_calendar_from_endpoints",
    "ny_date_keys",
    "run_deferred_shadow_null_calibration_suite",
    "run_null_calibration_cell",
    "run_null_calibration_suite",
    "run_shadow_null_mechanics_suite",
    "scheduled_arm",
    "shadow_empty_set_status",
    "validate_epoch_nanoseconds",
    "validate_epoch_seconds",
]
