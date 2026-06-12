"""Frozen 15m book discovery, schema normalization, and scoring."""

from __future__ import annotations

import hashlib
import json
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


class BookRuntimeError(RuntimeError):
    pass


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
        preds = np.array([float(b.predict(frame)[0]) for b in self.boosters], dtype=float)
        proba = float(preds.mean())
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
        reasons: list[str] = []
        if conf < self.conf_thr:
            reasons.append(f"confidence {conf:.8f} < threshold {self.conf_thr:.8f}")
        bb_thr = _first_number(self.strategy, ("bb_width_thr", "bbw_thr"))
        gate = self.strategy.get("gate")
        if bb_thr is None and isinstance(gate, dict):
            bb_thr = _first_number(gate, ("bb_width_thr", "bbw_thr"))
        if bb_thr is not None:
            if "15m_bb_width" not in row.index:
                reasons.append("missing 15m_bb_width for strategy compression gate")
            elif float(row["15m_bb_width"]) > bb_thr:
                reasons.append(f"15m_bb_width {float(row['15m_bb_width']):.8f} > threshold {bb_thr:.8f}")
        return (len(reasons) == 0), reasons


def load_index() -> dict[str, dict[str, Any]]:
    raw = json.loads(INDEX_PATH.read_text())
    books = raw.get("books")
    if not isinstance(books, list):
        raise BookRuntimeError("books/INDEX.json must contain a top-level books list")
    return {b["id"]: b for b in books}


def load_target_books(pairs: list[str] | None = None) -> dict[str, LoadedBook]:
    selected = pairs or list(TARGET_BOOKS)
    return {pair: load_book(pair, TARGET_BOOKS[pair]) for pair in selected}


def load_book(pair: str, book_id: str) -> LoadedBook:
    index = load_index()
    if book_id not in index:
        raise BookRuntimeError(f"{book_id} is not registered in books/INDEX.json")
    book_dir = BOOKS_DIR / book_id
    if not book_dir.is_dir():
        raise BookRuntimeError(f"{book_id}: missing book directory {book_dir}")
    strategy_files = sorted(book_dir.glob("*strategy*.json"))
    if len(strategy_files) != 1:
        raise BookRuntimeError(f"{book_id}: expected exactly one strategy JSON, found {len(strategy_files)}")
    model_paths = sorted(p for p in book_dir.glob("*.txt") if p.is_file())
    if not model_paths:
        raise BookRuntimeError(f"{book_id}: no LightGBM text models found")
    strategy = json.loads(strategy_files[0].read_text())
    boosters = [lgb.Booster(model_file=str(p)) for p in model_paths]
    feature_cols = _feature_cols(pair, strategy, boosters)
    counts = {b.num_feature() for b in boosters}
    if counts != {len(feature_cols)}:
        raise BookRuntimeError(f"{book_id}: model feature counts {sorted(counts)} != strategy columns {len(feature_cols)}")
    conf_thr = _conf_thr(strategy)
    return LoadedBook(
        pair=pair,
        book_id=book_id,
        book_dir=book_dir,
        strategy_path=strategy_files[0],
        model_paths=model_paths,
        strategy=strategy,
        feature_cols=feature_cols,
        boosters=boosters,
        conf_thr=conf_thr,
        content_id=_content_id([strategy_files[0], *model_paths]),
    )


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


def _conf_thr(strategy: dict[str, Any]) -> float:
    val = _first_number(strategy, ("conf_thr", "confidence_threshold", "threshold", "thr"))
    gate = strategy.get("gate")
    if val is None and isinstance(gate, dict):
        val = _first_number(gate, ("conf_thr", "confidence_threshold", "threshold", "thr"))
    gates = strategy.get("gates")
    if val is None and isinstance(gates, dict) and gates:
        preferred = strategy.get("coverage") or strategy.get("cov")
        deploy = strategy.get("deploy")
        if preferred is None and isinstance(deploy, dict):
            preferred = deploy.get("coverage") or deploy.get("cov")
        gate_obj = None
        for key in (str(preferred) if preferred is not None else None, "0.02"):
            if key and key in gates:
                gate_obj = gates[key]
                break
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
                "content_id": book.content_id,
            }
        )
    return rows


if __name__ == "__main__":
    print(json.dumps(book_inventory(), indent=2))
