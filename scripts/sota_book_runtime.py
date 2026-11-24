"""Deterministic SOTA multi-model book loader and scorer (sota-book-manifest/v1).

Sibling to scripts/book_runtime.py (which stays byte-frozen for test_protected_evaluation.py
and USDJPY Look 2 seq_decision.py). Supports pure-NumPy linear/kernel models (`logit`,
`logit-hourly`, `nystroem` — zero pickle execution) and LightGBM (`lgb`), per-book
horizons (`15m`/`30m`), explicit `allowed_sides`, Platt ties-lose side calibration, and
mandatory 6-gate certification provenance verification.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from deriv_runtime_core_v2 import (
    ALL_BARRIER_TYPES,
    SUPPORTED_CONTRACT_FAMILIES,
    SUPPORTED_HORIZONS_MINUTES,
    TWO_BARRIER_TYPES,
    finite_positive_number,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOKS_DIR = REPO_ROOT / "books"
SOTA_INDEX_PATH = BOOKS_DIR / "SOTA_INDEX.json"
SOTA_MANIFEST_SCHEMA = "sota-book-manifest/v1"
SOTA_INDEX_SCHEMA = "sota-book-index/v1"
ACTIVE_LIFECYCLE = "active"
DEMO_CANARY_LIFECYCLE = "demo_venue_settled_canary"
INACTIVE_LIFECYCLE = "inactive_candidate"
NY_TZ = ZoneInfo("America/New_York")

REQUIRED_DEV_GATES = ("G1_cpcv", "G2_placebo", "G3_entry_delay", "G4_sign_null", "G5_payout")


class SotaBookError(RuntimeError):
    pass


def _expit(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return np.where(np.isfinite(z), out, np.nan)


def _check_book_id(book_id: str) -> str:
    if (
        not isinstance(book_id, str)
        or not book_id
        or Path(book_id).name != book_id
        or book_id in {".", ".."}
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", book_id) is None
    ):
        raise SotaBookError(f"invalid SOTA book id {book_id!r}")
    return book_id


def _require_real_file(path: Path, root: Path = REPO_ROOT) -> Path:
    if path.is_symlink():
        raise SotaBookError(f"refusing symlinked book artifact: {path}")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise SotaBookError(f"invalid or out-of-repo path {path}: {exc}") from exc
    st = os.lstat(path)
    if not stat.S_ISREG(st.st_mode):
        raise SotaBookError(f"not a regular file: {path}")
    return resolved


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_sota_content_id(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda item: item.name):
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _rbf_nystroem_member_prob(
    Z: np.ndarray,
    z_sq: np.ndarray,
    comp: np.ndarray,
    norm: np.ndarray,
    gamma: float,
    coef: np.ndarray,
    intercept: float,
) -> np.ndarray:
    comp64 = np.asarray(comp, dtype=np.float64)
    norm64 = np.asarray(norm, dtype=np.float64)
    coef64 = np.asarray(coef, dtype=np.float64)
    sq_dists = z_sq + np.sum(comp64 * comp64, axis=1)[None, :] - 2.0 * (Z @ comp64.T)
    K = np.exp(-gamma * np.maximum(sq_dists, 0.0))
    Fk = K @ norm64.T
    return _expit(Fk @ coef64 + intercept)


def _validate_weights_npz(
    book_id: str,
    model_kind: str,
    weights_npz: dict[str, np.ndarray],
    n_features: int,
) -> None:
    if not isinstance(weights_npz, dict) or not weights_npz:
        raise SotaBookError(f"{book_id}: empty or invalid weights_npz")
    int_keys = {"hours", "n_seeds"}
    for k, arr in weights_npz.items():
        if not isinstance(arr, np.ndarray):
            raise SotaBookError(f"{book_id}: non-array in weights[{k!r}]")
        if k in int_keys:
            if arr.dtype.kind not in {"i", "u"}:
                raise SotaBookError(f"{book_id}: integer array required for weights[{k!r}]")
        elif arr.dtype.kind != "f":
            raise SotaBookError(f"{book_id}: floating-point array required for weights[{k!r}], got {arr.dtype}")
        if arr.size == 0 or not np.isfinite(arr).all():
            raise SotaBookError(f"{book_id}: empty or non-finite values in weights[{k!r}]")

    def _check_scaler(prefix: str = "") -> None:
        for s_key in (f"{prefix}med", f"{prefix}mu", f"{prefix}sd"):
            if s_key not in weights_npz or weights_npz[s_key].shape != (n_features,):
                raise SotaBookError(f"{book_id}: weights[{s_key!r}] must have shape ({n_features},)")
        if not np.all(weights_npz[f"{prefix}sd"] > 0.0):
            raise SotaBookError(f"{book_id}: weights['{prefix}sd'] must be strictly positive")

    def _check_nys_member(prefix: str = "") -> None:
        c_key = f"{prefix}nys_components"
        n_key = f"{prefix}nys_normalization"
        g_key = f"{prefix}nys_gamma"
        w_key = f"{prefix}sgd_coef"
        b_key = f"{prefix}sgd_intercept"
        for req_k in (c_key, n_key, g_key, w_key, b_key):
            if req_k not in weights_npz:
                raise SotaBookError(f"{book_id}: missing weights[{req_k!r}]")
        comp = weights_npz[c_key]
        if comp.ndim != 2 or comp.shape[0] < 1 or comp.shape[1] != n_features:
            raise SotaBookError(f"{book_id}: invalid {c_key} shape {comp.shape}")
        m_dim = int(comp.shape[0])
        if weights_npz[n_key].shape != (m_dim, m_dim):
            raise SotaBookError(f"{book_id}: invalid {n_key} shape {weights_npz[n_key].shape}")
        if weights_npz[g_key].size != 1 or float(weights_npz[g_key].reshape(-1)[0]) <= 0.0:
            raise SotaBookError(f"{book_id}: {g_key} must be a positive scalar")
        if weights_npz[w_key].shape != (m_dim,):
            raise SotaBookError(f"{book_id}: invalid {w_key} shape {weights_npz[w_key].shape}")
        if weights_npz[b_key].size != 1:
            raise SotaBookError(f"{book_id}: {b_key} must be scalar")

    if model_kind == "logit":
        expected_keys = {"med", "mu", "sd", "coef", "intercept"}
        if set(weights_npz.keys()) != expected_keys:
            raise SotaBookError(f"{book_id}: unexpected logit weight keys {sorted(weights_npz.keys())}")
        _check_scaler("")
        if weights_npz["coef"].shape != (n_features,) or weights_npz["intercept"].size != 1:
            raise SotaBookError(f"{book_id}: invalid logit coef/intercept shape")
    elif model_kind == "logit-hourly":
        if "hours" not in weights_npz or weights_npz["hours"].ndim != 1 or weights_npz["hours"].dtype.kind not in {"i", "u"}:
            raise SotaBookError(f"{book_id}: logit-hourly requires 1D integer 'hours' array")
        hours = [int(h) for h in weights_npz["hours"].tolist()]
        if not hours or any(h < 0 or h > 23 for h in hours):
            raise SotaBookError(f"{book_id}: invalid hours {hours} in logit-hourly")
        expected_keys = {"hours"} | {
            f"h{h}_{sfx}" for h in hours for sfx in ("med", "mu", "sd", "coef", "intercept")
        }
        if set(weights_npz.keys()) != expected_keys:
            raise SotaBookError(f"{book_id}: unexpected logit-hourly weight keys")
        for h in hours:
            _check_scaler(f"h{h}_")
            if weights_npz[f"h{h}_coef"].shape != (n_features,) or weights_npz[f"h{h}_intercept"].size != 1:
                raise SotaBookError(f"{book_id}: invalid h{h}_coef/intercept shape")
    elif model_kind == "nystroem":
        expected_keys = {
            "med", "mu", "sd",
            "nys_components", "nys_normalization", "nys_gamma", "sgd_coef", "sgd_intercept",
        }
        if set(weights_npz.keys()) != expected_keys:
            raise SotaBookError(f"{book_id}: unexpected nystroem weight keys {sorted(weights_npz.keys())}")
        _check_scaler("")
        _check_nys_member("")
    elif model_kind == "nystroem-ens":
        if "n_seeds" not in weights_npz or weights_npz["n_seeds"].size != 1 or weights_npz["n_seeds"].dtype.kind not in {"i", "u"}:
            raise SotaBookError(f"{book_id}: nystroem-ens requires scalar integer 'n_seeds'")
        n_seeds = int(weights_npz["n_seeds"].reshape(-1)[0])
        if n_seeds < 1 or n_seeds > 32:
            raise SotaBookError(f"{book_id}: nystroem-ens n_seeds={n_seeds} outside [1, 32]")
        expected_keys = {"med", "mu", "sd", "n_seeds"} | {
            f"s{i}_{sfx}"
            for i in range(n_seeds)
            for sfx in ("nys_components", "nys_normalization", "nys_gamma", "sgd_coef", "sgd_intercept")
        }
        if set(weights_npz.keys()) != expected_keys:
            raise SotaBookError(f"{book_id}: nystroem-ens key mismatch vs n_seeds={n_seeds}")
        _check_scaler("")
        for i in range(n_seeds):
            _check_nys_member(f"s{i}_")
    else:
        raise SotaBookError(f"{book_id}: unsupported npz model_kind {model_kind!r}")


@dataclass(frozen=True)
class SotaScore:
    book_id: str
    pair: str
    horizon_minutes: int
    proba_up_raw: float
    confidence: float
    threshold: float
    direction: str
    calibrated_win_prob: float | None
    effective_floor: float
    drop_ny_16h: bool
    contract_family: str
    gate_passed: bool
    gate_reasons: list[str]
    expected_ev: float | None = None
    multiplier: int | None = None
    barrier_contract_type: str | None = None
    barrier_rel_half_width: float | None = None


@dataclass
class LoadedSotaBook:
    book_id: str
    pair: str
    horizon_minutes: int
    allowed_sides: tuple[str, ...]
    lifecycle_status: str
    model_kind: str
    contract_family: str
    feature_cols: list[str]
    coverage: float
    conf_thr: float
    up_floor: float | None
    down_floor: float | None
    payout_edge_margin: float
    min_payout_ratio: float | None
    drop_ny_16h: bool
    platt_calibration: dict[str, dict[str, float]]
    content_id: str
    manifest: dict[str, Any]
    weights_npz: dict[str, np.ndarray] | None = None
    boosters: list[lgb.Booster] | None = None
    neutral_floor: float | None = None
    ev_floors: dict[str, float] | None = None
    multiplier: int | None = None
    certified_commission_frac: float = 0.0
    max_commission_frac: float = 0.20
    min_stop_out_rel_dist: float = 0.0008
    barrier_contract_type: str | None = None
    barrier_rel_half_width: float | None = None

    def side_floor(self, side: str) -> float:
        s = side.upper()
        if s not in self.allowed_sides:
            raise SotaBookError(f"{self.book_id}: side {s!r} not in allowed_sides {self.allowed_sides}")
        if s == "NEUTRAL":
            val = self.neutral_floor
        else:
            val = self.up_floor if s == "UP" else self.down_floor
        if val is None or not math.isfinite(val):
            raise SotaBookError(f"{self.book_id}: missing finite refit floor for {s}")
        return float(val)

    def effective_floor(self, side: str, extra_margin: float = 0.0) -> float:
        # SOTA books are certified directly on Deriv 1s ticks (settlement_substrate == deriv_1s_ticks)
        return self.side_floor(side) - float(self.payout_edge_margin) - float(extra_margin)

    def _transform_std(self, X: np.ndarray, prefix: str = "") -> np.ndarray:
        assert self.weights_npz is not None
        med = self.weights_npz[f"{prefix}med"]
        mu = self.weights_npz[f"{prefix}mu"]
        sd = self.weights_npz[f"{prefix}sd"]
        Z = np.where(np.isfinite(X), X, med).astype(np.float64)
        return np.clip((Z - mu) / sd, -5.0, 5.0)

    def predict_probabilities(
        self,
        feature_rows: pd.DataFrame,
        decision_epochs: np.ndarray | list[int] | None = None,
    ) -> np.ndarray:
        if not isinstance(feature_rows, pd.DataFrame):
            raise SotaBookError(f"{self.book_id}: feature_rows must be a DataFrame")
        missing = [c for c in self.feature_cols if c not in feature_rows.columns]
        if missing:
            raise SotaBookError(f"{self.book_id}: missing feature columns {missing[:10]}")
        X = feature_rows.loc[:, self.feature_cols].to_numpy(dtype=np.float64, copy=True)
        if len(X) == 0:
            return np.empty(0, dtype=np.float64)
        if np.isinf(X).any():
            bad = [self.feature_cols[i] for i in np.flatnonzero(np.any(np.isinf(X), axis=0))]
            raise SotaBookError(f"{self.book_id}: infinite feature values {bad[:10]}")
        # Match cpcv_deriv_x.py / build_certified_sota_books.py: reject >= 50% NaN rows, impute < 50% NaN via med
        if np.any(np.mean(np.isnan(X), axis=1) >= 0.50):
            raise SotaBookError(f"{self.book_id}: feature row has >= 50% missing values")

        if self.model_kind == "lgb":
            if not self.boosters:
                raise SotaBookError(f"{self.book_id}: no loaded LightGBM boosters")
            df = pd.DataFrame(X, columns=self.feature_cols)
            preds = np.mean([ np.asarray(b.predict(df), dtype=np.float64) for b in self.boosters ], axis=0)
        elif self.model_kind == "logit":
            assert self.weights_npz is not None
            Z = self._transform_std(X)
            coef = self.weights_npz["coef"]
            intercept = float(self.weights_npz["intercept"].reshape(-1)[0])
            preds = _expit(Z @ coef + intercept)
        elif self.model_kind == "logit-hourly":
            assert self.weights_npz is not None
            if decision_epochs is None:
                raise SotaBookError(f"{self.book_id}: logit-hourly requires decision_epochs")
            epochs = np.asarray(decision_epochs, dtype=np.int64)
            if epochs.shape != (len(X),):
                raise SotaBookError(f"{self.book_id}: decision_epochs length mismatch")
            ny_hours = pd.to_datetime(epochs, unit="s", utc=True).tz_convert(NY_TZ).hour.to_numpy()
            preds = np.full(len(X), np.nan, dtype=np.float64)
            supported_hours = set(int(h) for h in self.weights_npz["hours"].tolist())
            for h in sorted(set(int(v) for v in ny_hours.tolist())):
                if h not in supported_hours:
                    raise SotaBookError(f"{self.book_id}: unsupported NY hour {h} for logit-hourly")
                m = ny_hours == h
                Z = self._transform_std(X[m], prefix=f"h{h}_")
                coef = self.weights_npz[f"h{h}_coef"]
                intercept = float(self.weights_npz[f"h{h}_intercept"].reshape(-1)[0])
                preds[m] = _expit(Z @ coef + intercept)
        elif self.model_kind in {"nystroem", "nystroem-ens"}:
            assert self.weights_npz is not None
            Z = self._transform_std(X)
            z_sq = np.sum(Z * Z, axis=1, keepdims=True)
            if self.model_kind == "nystroem":
                preds = _rbf_nystroem_member_prob(
                    Z,
                    z_sq,
                    self.weights_npz["nys_components"],
                    self.weights_npz["nys_normalization"],
                    float(self.weights_npz["nys_gamma"].reshape(-1)[0]),
                    self.weights_npz["sgd_coef"],
                    float(self.weights_npz["sgd_intercept"].reshape(-1)[0]),
                )
            else:
                n_seeds = int(self.weights_npz["n_seeds"].reshape(-1)[0])
                acc = np.zeros(len(X), dtype=np.float64)
                for s_i in range(n_seeds):
                    acc += _rbf_nystroem_member_prob(
                        Z,
                        z_sq,
                        self.weights_npz[f"s{s_i}_nys_components"],
                        self.weights_npz[f"s{s_i}_nys_normalization"],
                        float(self.weights_npz[f"s{s_i}_nys_gamma"].reshape(-1)[0]),
                        self.weights_npz[f"s{s_i}_sgd_coef"],
                        float(self.weights_npz[f"s{s_i}_sgd_intercept"].reshape(-1)[0]),
                    )
                preds = acc / float(n_seeds)
        else:
            raise SotaBookError(f"{self.book_id}: unsupported model_kind {self.model_kind!r}")

        if preds.shape != (len(X),) or not np.isfinite(preds).all():
            raise SotaBookError(f"{self.book_id}: non-finite predicted probabilities")
        return preds

    def calibrate_side_prob(self, proba_up_raw: float, side: str) -> float | None:
        cal = self.platt_calibration.get(side.upper())
        if not cal:
            return None
        p = float(np.clip(proba_up_raw, 1e-6, 1.0 - 1e-6))
        logit = float(np.clip(math.log(p / (1.0 - p)), -15.0, 15.0))
        z = float(cal["coef"]) * logit + float(cal["intercept"])
        return float(_expit(np.array([z]))[0])

    def score_row(
        self,
        feature_row: pd.Series | dict[str, Any],
        decision_epoch: int,
        *,
        extra_edge_margin: float = 0.0,
    ) -> SotaScore:
        s = pd.Series(feature_row, dtype="float64")
        frame = pd.DataFrame([s[self.feature_cols].to_numpy(dtype=np.float64)], columns=self.feature_cols)
        proba = float(self.predict_probabilities(frame, decision_epochs=[int(decision_epoch)])[0])
        if self.allowed_sides == ("NEUTRAL",):
            conf = proba
            direction = "NEUTRAL"
        else:
            conf = abs(proba - 0.5)
            direction = "UP" if proba > 0.5 else ("DOWN" if proba < 0.5 else "FLAT")
        reasons: list[str] = []

        if direction == "FLAT":
            reasons.append("exact_0.5_tie_abstain")
        elif direction not in self.allowed_sides:
            reasons.append(f"direction_{direction}_not_in_allowed_sides_{','.join(self.allowed_sides)}")

        if conf < self.conf_thr:
            reasons.append(f"confidence {conf:.6f} < threshold {self.conf_thr:.6f}")

        if self.drop_ny_16h:
            dt_ny = datetime.fromtimestamp(int(decision_epoch), tz=timezone.utc).astimezone(NY_TZ)
            mod = dt_ny.hour * 60 + dt_ny.minute
            if 960 <= mod < 995:
                reasons.append("excluded_ny_16h_payout_haircut_window")

        eff_floor = (
            self.effective_floor(direction, extra_margin=extra_edge_margin)
            if (self.contract_family != "multiplier" and direction in self.allowed_sides)
            else (0.0 if self.contract_family == "multiplier" else 1.0)
        )
        cal_prob = (
            self.calibrate_side_prob(proba, direction)
            if (self.contract_family != "multiplier" and direction in self.allowed_sides)
            else None
        )
        exp_ev = (
            float(self.ev_floors[direction]) - float(self.payout_edge_margin) - float(extra_edge_margin)
            if (self.contract_family == "multiplier" and self.ev_floors and direction in self.ev_floors)
            else None
        )
        return SotaScore(
            book_id=self.book_id,
            pair=self.pair,
            horizon_minutes=self.horizon_minutes,
            proba_up_raw=proba,
            confidence=conf,
            threshold=self.conf_thr,
            direction=direction,
            calibrated_win_prob=cal_prob,
            effective_floor=eff_floor,
            drop_ny_16h=self.drop_ny_16h,
            contract_family=self.contract_family,
            gate_passed=(len(reasons) == 0),
            gate_reasons=reasons,
            expected_ev=exp_ev,
            multiplier=self.multiplier,
            barrier_contract_type=self.barrier_contract_type,
            barrier_rel_half_width=self.barrier_rel_half_width,
        )


def load_sota_index(repo_root: Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    idx_path = _require_real_file(repo_root / "books" / "SOTA_INDEX.json", root=repo_root)
    raw = json.loads(idx_path.read_text(encoding="utf-8"))
    if raw.get("schema") != SOTA_INDEX_SCHEMA or not isinstance(raw.get("books"), list):
        raise SotaBookError(f"malformed SOTA_INDEX.json schema at {idx_path}")
    out: dict[str, dict[str, Any]] = {}
    for entry in raw["books"]:
        bid = _check_book_id(entry.get("id", ""))
        if bid in out:
            raise SotaBookError(f"duplicate SOTA book id {bid}")
        out[bid] = entry
    return out


def _validate_certification_gates(book_id: str, lifecycle: str, manifest: dict[str, Any], allow_canary: bool) -> None:
    if lifecycle == INACTIVE_LIFECYCLE:
        raise SotaBookError(f"{book_id}: inactive_candidate cannot be loaded for execution")
    if lifecycle == DEMO_CANARY_LIFECYCLE and not allow_canary:
        raise SotaBookError(
            f"{book_id}: lifecycle is {DEMO_CANARY_LIFECYCLE!r} (Demo VENUE_SETTLED only; live CR refused)"
        )
    if lifecycle not in {ACTIVE_LIFECYCLE, DEMO_CANARY_LIFECYCLE}:
        raise SotaBookError(f"{book_id}: unknown lifecycle_status {lifecycle!r}")

    certs = manifest.get("gate_certificates")
    if not isinstance(certs, dict):
        raise SotaBookError(f"{book_id}: missing gate_certificates dict")
    for g in REQUIRED_DEV_GATES:
        g_obj = certs.get(g)
        if not isinstance(g_obj, dict) or g_obj.get("passed") is not True:
            raise SotaBookError(f"{book_id}: mandatory certification gate {g} is missing or not passed")
    if lifecycle == ACTIVE_LIFECYCLE:
        g6 = certs.get("G6_holdout_or_prospective")
        if not isinstance(g6, dict) or g6.get("passed") is not True:
            raise SotaBookError(f"{book_id}: active lifecycle requires passed G6_holdout_or_prospective certificate")


def load_sota_book(
    book_id: str,
    *,
    repo_root: Path = REPO_ROOT,
    allow_demo_canary: bool = True,
) -> LoadedSotaBook:
    _check_book_id(book_id)
    index = load_sota_index(repo_root=repo_root)
    if book_id not in index:
        raise SotaBookError(f"{book_id}: not registered in books/SOTA_INDEX.json")
    entry = index[book_id]
    book_dir = repo_root / "books" / book_id
    if book_dir.is_symlink() or not book_dir.is_dir():
        raise SotaBookError(f"{book_id}: missing or symlinked book directory {book_dir}")
    manifest_path = _require_real_file(book_dir / f"{book_id}.manifest.json", root=repo_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SOTA_MANIFEST_SCHEMA:
        raise SotaBookError(f"{book_id}: invalid manifest schema {manifest.get('schema')!r}")
    if manifest.get("id") != book_id:
        raise SotaBookError(f"{book_id}: manifest id mismatch")

    lifecycle = str(entry.get("lifecycle_status", ""))
    if lifecycle != manifest.get("lifecycle_status"):
        raise SotaBookError(f"{book_id}: lifecycle mismatch between SOTA_INDEX.json and manifest")
    _validate_certification_gates(book_id, lifecycle, manifest, allow_canary=allow_demo_canary)

    if manifest.get("settlement_substrate") != "deriv_1s_ticks":
        raise SotaBookError(f"{book_id}: settlement_substrate must be 'deriv_1s_ticks'")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise SotaBookError(f"{book_id}: manifest has no artifacts")
    art_paths: list[Path] = []
    for art in artifacts:
        ap = _require_real_file(book_dir / str(art["file"]), root=repo_root)
        if file_sha256(ap) != art.get("sha256") or ap.stat().st_size != art.get("bytes"):
            raise SotaBookError(f"{book_id}: artifact {ap.name} failed SHA-256 or byte-length verification")
        art_paths.append(ap)

    cid = compute_sota_content_id(art_paths)
    if cid != manifest.get("content_id") or cid != entry.get("content_id"):
        raise SotaBookError(f"{book_id}: content_id mismatch ({cid})")

    strat_file = next((p for p in art_paths if p.name.endswith("_strategy.json")), None)
    if strat_file is None:
        raise SotaBookError(f"{book_id}: missing *_strategy.json artifact")
    strategy = json.loads(strat_file.read_text(encoding="utf-8"))

    feature_cols = [str(c) for c in strategy["feature_cols"]]
    if not feature_cols:
        raise SotaBookError(f"{book_id}: empty feature_cols")

    horizon_minutes = int(manifest["horizon_minutes"])
    if horizon_minutes not in SUPPORTED_HORIZONS_MINUTES or int(strategy.get("horizon_minutes", horizon_minutes)) != horizon_minutes:
        raise SotaBookError(f"{book_id}: unsupported or mismatched horizon_minutes {horizon_minutes}")

    contract_family = str(strategy.get("contract_family", "risefall"))
    if contract_family not in SUPPORTED_CONTRACT_FAMILIES:
        raise SotaBookError(f"{book_id}: unsupported contract_family {contract_family!r}")

    model_kind = str(manifest["model_kind"])
    weights_npz: dict[str, np.ndarray] | None = None
    boosters: list[lgb.Booster] | None = None
    if model_kind in {"logit", "logit-hourly", "nystroem", "nystroem-ens"}:
        npz_file = next((p for p in art_paths if p.suffix == ".npz"), None)
        if npz_file is None:
            raise SotaBookError(f"{book_id}: missing .npz weights artifact for {model_kind}")
        with np.load(npz_file, allow_pickle=False) as z:
            weights_npz = {k: z[k].copy() for k in z.files}
        _validate_weights_npz(book_id, model_kind, weights_npz, len(feature_cols))
    elif model_kind == "lgb":
        txt_files = [p for p in art_paths if p.suffix == ".txt"]
        if not txt_files:
            raise SotaBookError(f"{book_id}: missing .txt LightGBM artifacts")
        boosters = [lgb.Booster(model_file=str(p)) for p in txt_files]
    else:
        raise SotaBookError(f"{book_id}: unknown model_kind {model_kind!r}")

    allowed_sides = tuple(str(s).upper() for s in strategy["allowed_sides"])
    bct_opt: str | None = None
    brw_opt: float | None = None
    mult_opt: int | None = None
    cert_comm_frac = float(strategy.get("certified_commission_frac", 0.0))
    max_comm_frac = float(strategy.get("max_commission_frac", 0.20))
    min_so_dist = float(strategy.get("min_stop_out_rel_dist", 0.0008))

    if contract_family == "barrier_1d":
        if horizon_minutes != 1440:
            raise SotaBookError(f"{book_id}: barrier_1d requires horizon_minutes=1440, got {horizon_minutes}")
        bct_opt = str(strategy.get("barrier_contract_type") or "").strip().upper()
        if bct_opt not in ALL_BARRIER_TYPES:
            raise SotaBookError(f"{book_id}: invalid barrier_contract_type {bct_opt!r}")
        brw_opt = finite_positive_number(strategy.get("barrier_rel_half_width"))
        if brw_opt is None or brw_opt >= 0.25:
            raise SotaBookError(f"{book_id}: invalid barrier_rel_half_width {strategy.get('barrier_rel_half_width')!r}")
        if bct_opt in TWO_BARRIER_TYPES:
            if allowed_sides != ("NEUTRAL",):
                raise SotaBookError(f"{book_id}: two-barrier {bct_opt} requires allowed_sides=['NEUTRAL']")
        else:
            if not allowed_sides or any(s not in {"UP", "DOWN"} for s in allowed_sides):
                raise SotaBookError(f"{book_id}: single-barrier {bct_opt} requires UP/DOWN allowed_sides")
    else:
        if not allowed_sides or any(s not in {"UP", "DOWN"} for s in allowed_sides):
            raise SotaBookError(f"{book_id}: invalid allowed_sides {allowed_sides!r}")

    floors = strategy.get("refit_cpcv_p10_floors", {})
    ev_floors_raw = strategy.get("refit_cpcv_p10_ev") or {}
    ev_floors: dict[str, float] = {}
    platt_cal = dict(strategy.get("platt_calibration") or {})

    if contract_family == "multiplier":
        raw_mult = strategy.get("multiplier")
        if isinstance(raw_mult, bool) or not isinstance(raw_mult, int) or raw_mult < 1:
            raise SotaBookError(f"{book_id}: multiplier book requires positive integer 'multiplier'")
        mult_opt = int(raw_mult)
        for s in allowed_sides:
            v_ev = ev_floors_raw.get(s)
            if isinstance(v_ev, bool) or not isinstance(v_ev, (int, float)) or not math.isfinite(float(v_ev)):
                raise SotaBookError(f"{book_id}: multiplier book missing finite refit_cpcv_p10_ev[{s!r}]")
            ev_floors[s] = float(v_ev)
    else:
        for s in allowed_sides:
            if s not in floors or isinstance(floors[s], bool) or not isinstance(floors[s], (int, float)) or not math.isfinite(float(floors[s])):
                raise SotaBookError(f"{book_id}: missing refit_cpcv_p10_floors[{s!r}]")
            cal_s = platt_cal.get(s)
            if (
                not isinstance(cal_s, dict)
                or isinstance(cal_s.get("coef"), bool)
                or isinstance(cal_s.get("intercept"), bool)
                or not isinstance(cal_s.get("coef"), (int, float))
                or not isinstance(cal_s.get("intercept"), (int, float))
                or not math.isfinite(float(cal_s["coef"]))
                or not math.isfinite(float(cal_s["intercept"]))
            ):
                raise SotaBookError(f"{book_id}: missing or non-finite platt_calibration[{s!r}]")
            if s in {"UP", "NEUTRAL"} and float(cal_s["coef"]) <= 0.0:
                raise SotaBookError(f"{book_id}: platt_calibration[{s!r}]['coef'] must be > 0, got {cal_s['coef']}")
            if s == "DOWN" and float(cal_s["coef"]) >= 0.0:
                raise SotaBookError(f"{book_id}: platt_calibration['DOWN']['coef'] must be < 0, got {cal_s['coef']}")

    return LoadedSotaBook(
        book_id=book_id,
        pair=str(manifest["pair"]).upper(),
        horizon_minutes=horizon_minutes,
        allowed_sides=allowed_sides,
        lifecycle_status=lifecycle,
        model_kind=model_kind,
        contract_family=contract_family,
        feature_cols=feature_cols,
        coverage=float(strategy["coverage"]),
        conf_thr=float(strategy["confidence_threshold"]),
        up_floor=float(floors["UP"]) if "UP" in floors else None,
        down_floor=float(floors["DOWN"]) if "DOWN" in floors else None,
        payout_edge_margin=float(strategy.get("payout_edge_margin", 0.005)),
        min_payout_ratio=float(strategy["min_payout_ratio"]) if strategy.get("min_payout_ratio") is not None else None,
        drop_ny_16h=bool(strategy.get("drop_ny_16h", False)),
        platt_calibration=platt_cal,
        content_id=cid,
        manifest=manifest,
        weights_npz=weights_npz,
        boosters=boosters,
        neutral_floor=float(floors["NEUTRAL"]) if "NEUTRAL" in floors else None,
        ev_floors=ev_floors if ev_floors else None,
        multiplier=mult_opt,
        certified_commission_frac=cert_comm_frac,
        max_commission_frac=max_comm_frac,
        min_stop_out_rel_dist=min_so_dist,
        barrier_contract_type=bct_opt,
        barrier_rel_half_width=brw_opt,
    )


def parse_book_ids_spec(raw: str | None) -> list[str] | None:
    if raw is None or str(raw).strip() == "":
        return None
    parts = [p.strip() for p in str(raw).split(",")]
    if any(not p for p in parts):
        raise SotaBookError(f"empty book_id token in book_ids specification: {raw!r}")
    if len(set(parts)) != len(parts):
        raise SotaBookError(f"duplicate book_id in book_ids specification: {parts!r}")
    return parts


def load_enabled_sota_books(
    *,
    repo_root: Path = REPO_ROOT,
    account_mode: str = "demo",
    book_ids: list[str] | None = None,
) -> list[LoadedSotaBook]:
    allow_canary = str(account_mode).lower() == "demo"
    index = load_sota_index(repo_root=repo_root)
    if book_ids is not None:
        if len(set(book_ids)) != len(book_ids):
            raise SotaBookError(f"duplicate book_id in book_ids: {book_ids!r}")
        wanted = list(book_ids)
    else:
        eligible = {ACTIVE_LIFECYCLE, DEMO_CANARY_LIFECYCLE} if allow_canary else {ACTIVE_LIFECYCLE}
        cand_items = [(bid, ent) for bid, ent in index.items() if ent.get("lifecycle_status") in eligible]
        cand_items.sort(
            key=lambda kv: (
                float((kv[1].get("metrics") or {}).get("annual_floor_log_growth_p10") or 0.0),
                float((kv[1].get("metrics") or {}).get("cpcv_p10_ev") or 0.0),
            ),
            reverse=True,
        )
        wanted = [bid for bid, _ in cand_items]
    return [load_sota_book(bid, repo_root=repo_root, allow_demo_canary=allow_canary) for bid in wanted]
