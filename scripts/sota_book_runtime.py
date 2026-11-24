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

    def side_floor(self, side: str) -> float:
        s = side.upper()
        if s not in self.allowed_sides:
            raise SotaBookError(f"{self.book_id}: side {s!r} not in allowed_sides {self.allowed_sides}")
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
        elif self.model_kind == "nystroem":
            assert self.weights_npz is not None
            Z = self._transform_std(X)
            comp = self.weights_npz["nys_components"]  # (M, F)
            norm = self.weights_npz["nys_normalization"]  # (M, M)
            gamma = float(self.weights_npz["nys_gamma"].reshape(-1)[0])
            coef = self.weights_npz["sgd_coef"]
            intercept = float(self.weights_npz["sgd_intercept"].reshape(-1)[0])
            # Exact RBF kernel K(Z, comp) = exp(-gamma * ||Z - comp||^2)
            sq_dists = (
                np.sum(Z * Z, axis=1, keepdims=True)
                + np.sum(comp * comp, axis=1)[None, :]
                - 2.0 * (Z @ comp.T)
            )
            K = np.exp(-gamma * np.maximum(sq_dists, 0.0))
            Fk = K @ norm.T
            preds = _expit(Fk @ coef + intercept)
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
            if direction in self.allowed_sides
            else 1.0
        )
        cal_prob = self.calibrate_side_prob(proba, direction) if direction in {"UP", "DOWN"} else None
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

    model_kind = str(manifest["model_kind"])
    weights_npz: dict[str, np.ndarray] | None = None
    boosters: list[lgb.Booster] | None = None
    if model_kind in {"logit", "logit-hourly", "nystroem"}:
        npz_file = next((p for p in art_paths if p.suffix == ".npz"), None)
        if npz_file is None:
            raise SotaBookError(f"{book_id}: missing .npz weights artifact for {model_kind}")
        with np.load(npz_file, allow_pickle=False) as z:
            weights_npz = {k: z[k].copy() for k in z.files}
    elif model_kind == "lgb":
        txt_files = [p for p in art_paths if p.suffix == ".txt"]
        if not txt_files:
            raise SotaBookError(f"{book_id}: missing .txt LightGBM artifacts")
        boosters = [lgb.Booster(model_file=str(p)) for p in txt_files]
    else:
        raise SotaBookError(f"{book_id}: unknown model_kind {model_kind!r}")

    allowed_sides = tuple(str(s).upper() for s in strategy["allowed_sides"])
    if not allowed_sides or any(s not in {"UP", "DOWN"} for s in allowed_sides):
        raise SotaBookError(f"{book_id}: invalid allowed_sides {allowed_sides!r}")

    floors = strategy.get("refit_cpcv_p10_floors", {})
    platt_cal = dict(strategy.get("platt_calibration") or {})
    for s in allowed_sides:
        if s not in floors or not isinstance(floors[s], (int, float)) or not math.isfinite(float(floors[s])):
            raise SotaBookError(f"{book_id}: missing refit_cpcv_p10_floors[{s!r}]")
        cal_s = platt_cal.get(s)
        if (
            not isinstance(cal_s, dict)
            or not isinstance(cal_s.get("coef"), (int, float))
            or not isinstance(cal_s.get("intercept"), (int, float))
            or not math.isfinite(float(cal_s["coef"]))
            or not math.isfinite(float(cal_s["intercept"]))
        ):
            raise SotaBookError(f"{book_id}: missing or non-finite platt_calibration[{s!r}]")
        # logit_input is logit(proba_up_raw): UP requires coef > 0; DOWN requires coef < 0
        if s == "UP" and float(cal_s["coef"]) <= 0.0:
            raise SotaBookError(f"{book_id}: platt_calibration['UP']['coef'] must be > 0, got {cal_s['coef']}")
        if s == "DOWN" and float(cal_s["coef"]) >= 0.0:
            raise SotaBookError(f"{book_id}: platt_calibration['DOWN']['coef'] must be < 0, got {cal_s['coef']}")

    return LoadedSotaBook(
        book_id=book_id,
        pair=str(manifest["pair"]).upper(),
        horizon_minutes=int(manifest["horizon_minutes"]),
        allowed_sides=allowed_sides,
        lifecycle_status=lifecycle,
        model_kind=model_kind,
        contract_family=str(strategy.get("contract_family", "risefall")),
        feature_cols=[str(c) for c in strategy["feature_cols"]],
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
    )


def load_enabled_sota_books(
    *,
    repo_root: Path = REPO_ROOT,
    account_mode: str = "demo",
    book_ids: list[str] | None = None,
) -> list[LoadedSotaBook]:
    allow_canary = str(account_mode).lower() == "demo"
    index = load_sota_index(repo_root=repo_root)
    if book_ids is not None:
        wanted = book_ids
    else:
        eligible = {ACTIVE_LIFECYCLE, DEMO_CANARY_LIFECYCLE} if allow_canary else {ACTIVE_LIFECYCLE}
        wanted = [bid for bid, ent in index.items() if ent.get("lifecycle_status") in eligible]
    return [load_sota_book(bid, repo_root=repo_root, allow_demo_canary=allow_canary) for bid in wanted]
