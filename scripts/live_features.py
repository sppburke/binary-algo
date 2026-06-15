"""Live/replay feature rows for the Deriv demo executor.

Live rows reuse the production `pipeline.build_features` base feature recipe.
Cross-pair rows are rebuilt from synchronized close series and then joined with
the target pair's base columns. Any missing/non-finite requested model column is
a hard failure.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import harness
import pipeline
from deriv_client import DerivOptionsClient


PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"]
DERIV_SYMBOLS = {pair: f"frx{pair}" for pair in PAIRS}
USD_BASE = {"USDJPY", "USDCHF", "USDCAD"}
LB = [1, 3, 5, 10, 15, 30]
BASE_PREFIXES = ("1m_", "5m_", "15m_", "30m_", "1h_", "4h_", "mtf_", "hour_sin", "hour_cos", "dow", "sess_ny", "sess_london", "sess_overlap", "vol_z", "zero_vol", "gap_prev")


class LiveFeatureError(RuntimeError):
    pass


@dataclass(frozen=True)
class FeatureRow:
    pair: str
    timestamp: pd.Timestamp
    row: pd.Series
    source: str
    warnings: list[str]


def is_base_feature(col: str) -> bool:
    return col.startswith(BASE_PREFIXES)


def pair_symbol(pair: str) -> str:
    if pair not in DERIV_SYMBOLS:
        raise LiveFeatureError(f"unsupported pair {pair}")
    return DERIV_SYMBOLS[pair]


def candles_to_m1(candles: list[dict[str, Any]]) -> pd.DataFrame:
    if not candles:
        raise LiveFeatureError("Deriv returned no candles")
    frame = pd.DataFrame(candles)
    required = {"epoch", "open", "high", "low", "close"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise LiveFeatureError(f"Deriv candle payload missing columns {missing}")
    epochs = pd.to_numeric(frame["epoch"], errors="raise").astype("int64").to_numpy()
    idx = pd.to_datetime(epochs, unit="s", utc=True)
    volume_warning = ""
    if "volume" in frame.columns:
        volume = pd.to_numeric(frame["volume"], errors="coerce").to_numpy()
    elif "tick_count" in frame.columns:
        volume = pd.to_numeric(frame["tick_count"], errors="coerce").to_numpy()
    else:
        volume = np.full(len(frame), 1.0, dtype="float64")
        volume_warning = "synthetic_unit_volume_no_deriv_tick_count"
    out = pd.DataFrame(
        {
            "open": pd.to_numeric(frame["open"], errors="coerce").to_numpy(),
            "high": pd.to_numeric(frame["high"], errors="coerce").to_numpy(),
            "low": pd.to_numeric(frame["low"], errors="coerce").to_numpy(),
            "close": pd.to_numeric(frame["close"], errors="coerce").to_numpy(),
            "volume": pd.Series(volume).astype("float64").fillna(1.0).to_numpy(),
        },
        index=idx,
    ).dropna(subset=["open", "high", "low", "close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    now_floor = pd.Timestamp.utcnow().floor("min") - pd.Timedelta(minutes=1)
    out = out.loc[out.index <= now_floor]
    if len(out) < 500:
        raise LiveFeatureError(f"not enough completed 1m candles: {len(out)}")
    dt_min = out.index.to_series().diff().dt.total_seconds().div(60.0)
    out["gap_prev"] = dt_min.fillna(1.0).values
    if volume_warning:
        out.attrs["volume_warning"] = volume_warning
    return out


def ticks_to_volume(ticks_payload: dict[str, Any]) -> pd.Series:
    hist = ticks_payload.get("history", {})
    times = hist.get("times") or hist.get("epoch") or []
    if not times:
        return pd.Series(dtype="float64")
    idx = pd.to_datetime(pd.Series(times, dtype="int64"), unit="s", utc=True).dt.floor("min")
    return idx.value_counts().sort_index().astype("float64")


def apply_tick_volume(m1: pd.DataFrame, volume: pd.Series) -> pd.DataFrame:
    if volume.empty:
        return m1
    out = m1.copy()
    overlap = out.index.intersection(volume.index)
    if len(overlap):
        out.loc[overlap, "volume"] = volume.loc[overlap].values
    return out


def build_base_features(m1: pd.DataFrame) -> pd.DataFrame:
    x = pipeline.build_features(m1)
    return x.replace([np.inf, -np.inf], np.nan)


def latest_replay_row(pair: str, expected_cols: list[str]) -> FeatureRow:
    path = f"{harness.FEAT_DIR}/{pair}_2026.parquet"
    df = pd.read_parquet(path)
    if _needs_cross_pair(expected_cols):
        closes = pd.DataFrame(
            {
                p: pd.read_parquet(f"{harness.FEAT_DIR}/{p}_2026.parquet", columns=["close"])["close"]
                for p in PAIRS
            }
        ).dropna()
        xp = build_cross_pair_features(pair, closes, expected_cols)
        if pair == "EURUSD":
            of_path = f"{harness.FEAT_DIR.replace('/features', '/features_of')}/EURUSD_2026.parquet"
            of = pd.read_parquet(of_path)
            xp = xp.join(of[[c for c in of.columns if c not in xp.columns]], how="left")
        df = xp.join(df[[c for c in df.columns if c not in xp.columns]], how="left")
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise LiveFeatureError(f"replay {pair}: missing columns {missing[:12]}")
    clean = df.dropna(subset=expected_cols)
    if clean.empty:
        raise LiveFeatureError(f"replay {pair}: no finite row for expected columns")
    row = clean.iloc[-1][expected_cols]
    return FeatureRow(pair=pair, timestamp=pd.Timestamp(clean.index[-1]), row=row, source=path, warnings=[])


class LiveFeatureBuilder:
    def __init__(
        self,
        client: DerivOptionsClient,
        history_minutes: int = 14000,
        tick_volume_count: int = 5000,
        store_dir: str | Path | None = None,
        stale_seconds: int = 180,
    ):
        self.client = client
        self.history_minutes = int(history_minutes)
        self.tick_volume_count = int(tick_volume_count)
        self.store_dir = Path(store_dir) if store_dir is not None else None
        self.stale_seconds = int(stale_seconds)
        self._m1: dict[str, pd.DataFrame] = {}
        self._base: dict[str, pd.DataFrame] = {}
        self._warnings: dict[str, list[str]] = {}

    def feature_row(self, pair: str, expected_cols: list[str]) -> FeatureRow:
        pair = pair.upper()
        needed = list(PAIRS) if _needs_cross_pair(expected_cols) else [pair]
        self._ensure_pairs(needed)
        full = self._joined_row_frame(pair, expected_cols)
        missing = [c for c in expected_cols if c not in full.columns]
        nan_cols = [c for c in expected_cols if c in full.columns and not np.isfinite(full[c]).any()]
        if missing or nan_cols:
            raise LiveFeatureError(f"{pair}: cannot build requested feature row; missing={missing[:12]} nan={nan_cols[:12]}")
        clean = full.dropna(subset=expected_cols)
        if clean.empty:
            raise LiveFeatureError(f"{pair}: no complete live feature row for requested schema")
        ts = pd.Timestamp(clean.index[-1])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        stale_s = time.time() - int(ts.timestamp())
        if stale_s > self.stale_seconds:
            raise LiveFeatureError(f"{pair}: latest feature row is stale: {stale_s:.0f}s old")
        row = clean.iloc[-1][expected_cols]
        bad = row.index[~np.isfinite(row.astype("float64").values)].tolist()
        if bad:
            raise LiveFeatureError(f"{pair}: non-finite live feature values: {bad[:12]}")
        warnings = [w for p in needed for w in self._warnings.get(p, [])]
        source = f"deriv_store:{self.store_dir}" if self.store_dir is not None else "deriv_ticks_history"
        return FeatureRow(pair=pair, timestamp=ts, row=row, source=source, warnings=warnings)

    def _ensure_pairs(self, pairs: list[str]) -> None:
        for pair in pairs:
            pair = pair.upper()
            if pair in self._base:
                continue
            if self.store_dir is None:
                m1, warnings = self._fetch_pair_m1(pair)
            else:
                m1, warnings = self._load_store_pair_m1(pair)
            self._m1[pair] = m1
            self._base[pair] = build_base_features(m1)
            self._warnings[pair] = warnings

    def _fetch_pair_m1(self, pair: str) -> tuple[pd.DataFrame, list[str]]:
        symbol = pair_symbol(pair)
        warnings: list[str] = []
        resp = self.client.ticks_history(symbol, style="candles", count=self.history_minutes, granularity=60)
        candles = resp.get("candles") or resp.get("history", {}).get("candles") or []
        m1 = candles_to_m1(candles)
        if m1.attrs.get("volume_warning"):
            warnings.append(str(m1.attrs["volume_warning"]))
        if self.tick_volume_count > 0:
            volume = ticks_to_volume(self.client.ticks_history(symbol, style="ticks", count=self.tick_volume_count, granularity=None))
            if volume.empty:
                warnings.append("tick_volume_unavailable")
            else:
                m1 = apply_tick_volume(m1, volume)
        return m1, warnings

    def _load_store_pair_m1(self, pair: str) -> tuple[pd.DataFrame, list[str]]:
        assert self.store_dir is not None
        path = self.store_dir / f"{pair}.parquet"
        if not path.exists():
            raise LiveFeatureError(f"{pair}: rolling store file missing: {path}")
        frame = pd.read_parquet(path)
        if "timestamp" in frame.columns:
            idx = pd.to_datetime(frame["timestamp"], utc=True)
        else:
            idx = pd.to_datetime(frame.index, utc=True)
        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise LiveFeatureError(f"{pair}: rolling store missing columns {missing}")
        if "volume" in frame.columns:
            volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(1.0).to_numpy()
            warnings: list[str] = []
        else:
            volume = np.full(len(frame), 1.0, dtype="float64")
            warnings = ["synthetic_unit_volume_no_deriv_tick_count"]
        out = pd.DataFrame(
            {
                "open": pd.to_numeric(frame["open"], errors="coerce").to_numpy(),
                "high": pd.to_numeric(frame["high"], errors="coerce").to_numpy(),
                "low": pd.to_numeric(frame["low"], errors="coerce").to_numpy(),
                "close": pd.to_numeric(frame["close"], errors="coerce").to_numpy(),
                "volume": volume,
            },
            index=idx,
        ).dropna(subset=required)
        out = out[~out.index.duplicated(keep="last")].sort_index().tail(self.history_minutes)
        if len(out) < 500:
            raise LiveFeatureError(f"{pair}: rolling store has too few completed rows: {len(out)}")
        dt_min = out.index.to_series().diff().dt.total_seconds().div(60.0)
        out["gap_prev"] = dt_min.fillna(1.0).values
        if not out.index.is_monotonic_increasing:
            raise LiveFeatureError(f"{pair}: rolling store index is not monotonic")
        if out.index.duplicated().any():
            raise LiveFeatureError(f"{pair}: rolling store has duplicate timestamps")
        return out, warnings

    def _joined_row_frame(self, pair: str, expected_cols: list[str]) -> pd.DataFrame:
        base = self._base[pair]
        if not _needs_cross_pair(expected_cols):
            return base
        closes = pd.DataFrame({p: self._m1[p]["close"] for p in PAIRS}).dropna()
        xp = build_cross_pair_features(pair, closes, expected_cols)
        base_only = base[[c for c in base.columns if c not in xp.columns]]
        return xp.join(base_only, how="left")


def _needs_cross_pair(cols: list[str]) -> bool:
    return any(not is_base_feature(c) for c in cols)


def build_cross_pair_features(pair: str, closes: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    if pair == "EURUSD":
        xp = _eurusd_xp(closes)
    elif pair == "USDCHF":
        xp = _usdchf_xp(closes)
    elif pair == "GBPUSD":
        xp = _gbpusd_xp(closes)
    else:
        raise LiveFeatureError(f"{pair}: no cross-pair feature recipe")
    missing = [c for c in expected_cols if not is_base_feature(c) and c not in xp.columns]
    if pair == "EURUSD":
        missing = [c for c in missing if not c.startswith("OF_")]
    if missing:
        raise LiveFeatureError(f"{pair}: cross-pair recipe missing columns {missing[:16]}")
    return xp


def _equiv_sign(pair: str) -> float:
    return -1.0 if pair in USD_BASE else 1.0


def _aligned_sign(target: str, pair: str) -> float:
    return _equiv_sign(target) * _equiv_sign(pair)


def _log_returns(closes: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    lr = {p: np.log(closes[p].astype(float).values) for p in PAIRS}
    rets = {p: {k: np.concatenate([np.full(k, np.nan), lr[p][k:] - lr[p][:-k]]) for k in LB} for p in PAIRS}
    return lr, rets


def _session_cols(idx: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    hours = idx.hour.values + idx.minute.values / 60.0
    return {
        "sess_ny": ((hours >= 13.0) & (hours < 22.0)).astype(float),
        "sess_ln": ((hours >= 7.0) & (hours < 16.0)).astype(float),
        "hour": hours,
    }


def _eurusd_xp(closes: pd.DataFrame) -> pd.DataFrame:
    idx = closes.index
    _, rets = _log_returns(closes)
    feats: dict[str, Any] = {}
    others = [p for p in PAIRS if p != "EURUSD"]
    for k in LB:
        eu_r = rets["EURUSD"][k]
        aligned = np.vstack([_equiv_sign(p) * rets[p][k] for p in others])
        basket = np.nanmean(aligned, axis=0)
        feats[f"eu_r{k}"] = eu_r
        feats[f"usdbask{k}"] = basket
        feats[f"catchup{k}"] = basket - eu_r
        feats[f"eurresid{k}"] = eu_r - basket
        feats[f"disp{k}"] = np.nanstd(aligned, axis=0)
        feats[f"agree{k}"] = np.nanmean((np.sign(aligned) == np.sign(basket)).astype(float), axis=0)
        for p in others:
            feats[f"ll_{p}{k}"] = _equiv_sign(p) * rets[p][k] - eu_r
    feats.update(_session_cols(idx))
    r1 = rets["EURUSD"][1]
    feats["comp60"] = pd.Series(r1).rolling(60, min_periods=20).std().values
    return pd.DataFrame(feats, index=idx).astype("float32")


def _usdchf_xp(closes: pd.DataFrame) -> pd.DataFrame:
    idx = closes.index
    _, rets = _log_returns(closes)
    target = "USDCHF"
    others = [p for p in PAIRS if p != target]
    eur_bloc = ["EURUSD", "GBPUSD"]
    safe_haven = ["USDJPY"]
    feats: dict[str, Any] = {}
    for k in LB:
        tgt_r = rets[target][k]
        aligned = np.vstack([_aligned_sign(target, p) * rets[p][k] for p in others])
        basket = np.nanmean(aligned, axis=0)
        feats[f"tgt_r{k}"] = tgt_r
        feats[f"usdbask{k}"] = basket
        feats[f"catchup{k}"] = basket - tgt_r
        feats[f"tgtresid{k}"] = tgt_r - basket
        feats[f"disp{k}"] = np.nanstd(aligned, axis=0)
        feats[f"agree{k}"] = np.nanmean((np.sign(aligned) == np.sign(basket)).astype(float), axis=0)
        for p in others:
            feats[f"ll_{p}{k}"] = _aligned_sign(target, p) * rets[p][k] - tgt_r
        eurbloc = np.nanmean(np.vstack([_aligned_sign(target, p) * rets[p][k] for p in eur_bloc]), axis=0)
        safeh = np.nanmean(np.vstack([_aligned_sign(target, p) * rets[p][k] for p in safe_haven]), axis=0)
        feats[f"eurbloc{k}"] = eurbloc
        feats[f"eurcatch{k}"] = eurbloc - tgt_r
        feats[f"risk{k}"] = eurbloc - safeh
        feats[f"chf_eurresid{k}"] = tgt_r - eurbloc
    feats.update(_session_cols(idx))
    feats["comp60"] = pd.Series(rets[target][1]).rolling(60, min_periods=20).std().values
    return pd.DataFrame(feats, index=idx).astype("float32")


def _gbpusd_xp(closes: pd.DataFrame) -> pd.DataFrame:
    idx = closes.index
    lr, rets = _log_returns(closes)
    target = "GBPUSD"
    others = [p for p in PAIRS if p != target]
    commod = ["AUDUSD", "NZDUSD", "USDCAD"]
    safe_haven = ["USDJPY", "USDCHF"]
    feats: dict[str, Any] = {}
    for k in LB:
        gbp_r = rets[target][k]
        aligned = np.vstack([_equiv_sign(p) * rets[p][k] for p in others])
        basket = np.nanmean(aligned, axis=0)
        feats[f"gbp_r{k}"] = gbp_r
        feats[f"usdbask{k}"] = basket
        feats[f"catchup{k}"] = basket - gbp_r
        feats[f"gbpresid{k}"] = gbp_r - basket
        feats[f"disp{k}"] = np.nanstd(aligned, axis=0)
        feats[f"agree{k}"] = np.nanmean((np.sign(aligned) == np.sign(basket)).astype(float), axis=0)
        for p in others:
            feats[f"ll_{p}{k}"] = _equiv_sign(p) * rets[p][k] - gbp_r
        eur_r = rets["EURUSD"][k]
        feats[f"eurgbp_r{k}"] = gbp_r - eur_r
        eurobloc = np.nanmean(np.vstack([rets["EURUSD"][k], _equiv_sign("USDCHF") * rets["USDCHF"][k]]), axis=0)
        feats[f"eurobloc{k}"] = eurobloc
        feats[f"gbpbloc_resid{k}"] = gbp_r - eurobloc
        commod_factor = np.nanmean(np.vstack([_equiv_sign(p) * rets[p][k] for p in commod]), axis=0)
        safeh = np.nanmean(np.vstack([_equiv_sign(p) * rets[p][k] for p in safe_haven]), axis=0)
        feats[f"risk{k}"] = commod_factor - safeh
    eurgbp = lr["EURUSD"] - lr[target]
    for win in (60, 240):
        anchor = pd.Series(eurgbp).rolling(win, min_periods=win // 2).mean().values
        feats[f"eurgbp_dev{win}"] = eurgbp - anchor
    feats.update(_session_cols(idx))
    feats["sess_as"] = ((idx.hour.values >= 0.0) & (idx.hour.values < 9.0)).astype(float)
    feats["comp60"] = pd.Series(rets[target][1]).rolling(60, min_periods=20).std().values
    return pd.DataFrame(feats, index=idx).astype("float32")
