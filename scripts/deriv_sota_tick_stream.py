"""Deriv SOTA 1-second tick stream & zero-drift live feature builder.

Sibling to scripts/deriv_market_stream.py and scripts/live_features.py (both kept
byte-frozen for evidence/objects and PARITY_TRACKED_PATHS seals).

Calls scripts/deriv_dataset.py (`minute_bars`, `own_features`, `cross_features`)
directly on 1-second mid ticks so live feature vectors (`n`, `nchg`, `activity_*`,
`ticks_per_bar_15m`, `ret_*`, `rv_*`, `rangepos_*`, `x_<PAIR>_ret_*`, `x_<PAIR>_stale_s`,
`x_usd_weak_15m`) are float32-identical to the CPCV certification dataset (`dataset_dev_v0.1`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np
import pandas as pd

import deriv_dataset as dd
from deriv_async_client import DerivAsyncClient
from deriv_client import PUBLIC_WS_URL, DerivAPIError
from deriv_runtime_core_v2 import SOTA_DERIV_SYMBOLS, JsonlLogger, assert_non_us_deriv_host

HistoryRequestFn = Callable[[dict[str, Any], float], Awaitable[dict[str, Any]]]

MAJORS: tuple[str, ...] = tuple(dd.PAIRS)
DEFAULT_RING_SECONDS = 345_600  # 96h of 1s ticks (~2.8 MB/pair) spans 48h FX weekends + 1440m lookbacks
DEFAULT_WARMUP_MIN_BARS = 1_600
DEFAULT_SOTA_TICK_STORE = Path("deriv_data/sota_ticks_1s")


class SotaFeatureError(RuntimeError):
    pass


@dataclass(frozen=True)
class LiveSotaFeatureRow:
    pair: str
    decision_epoch: int
    decision_utc: str
    row: pd.Series
    tick_age_s: float
    source: str
    spot_ref: float = 0.0


def build_sota_feature_frame_from_ticks(
    target_pair: str,
    ticks_by_pair: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Pure, deterministic feature builder calling deriv_dataset.{minute_bars,own_features,cross_features}."""
    target = target_pair.upper()
    missing_majors = [p for p in MAJORS if p not in ticks_by_pair]
    if missing_majors:
        raise SotaFeatureError(f"{target}: missing major tick series for cross features: {missing_majors}")
    if target not in ticks_by_pair:
        raise SotaFeatureError(f"{target}: target tick series missing")

    needed_pairs = list(MAJORS) if target in MAJORS else list(MAJORS) + [target]
    bars: dict[str, pd.DataFrame] = {}
    closes: dict[str, pd.Series] = {}
    for p in needed_pairs:
        ts_p, px_p = ticks_by_pair[p]
        if len(ts_p) == 0:
            raise SotaFeatureError(f"{p}: empty tick series")
        b = dd.minute_bars(np.asarray(ts_p, dtype=np.int64), np.asarray(px_p, dtype=np.float64))
        bars[p] = b
        closes[p] = pd.Series(b["close"].values, index=b.index.values.astype(np.int64) + 60)

    b_tgt = bars[target]
    X = dd.own_features(b_tgt)
    dec = b_tgt.index.values.astype(np.int64) + 60
    target_close_t = pd.Series(dec, index=b_tgt.index)
    others = {p: closes[p] for p in needed_pairs if p != target}
    X = X.join(dd.cross_features(target_close_t, others).set_index(b_tgt.index))
    X["decision_ts"] = dec
    X = X.astype({c: "float32" for c in X.columns if X[c].dtype == np.float64})
    X = X.reset_index(names="bar_start_ts")
    if feature_cols is not None:
        absent = [c for c in feature_cols if c not in X.columns]
        if absent:
            raise SotaFeatureError(f"{target}: missing requested columns {absent[:10]}")
    return X


def extract_decision_feature_row(
    target_pair: str,
    ticks_by_pair: dict[str, tuple[np.ndarray, np.ndarray]],
    decision_epoch: int,
    feature_cols: list[str],
    *,
    max_tick_staleness_s: float = 120.0,
    source: str = "live_1s_ticks",
) -> LiveSotaFeatureRow:
    """Build causal features strictly from ticks < decision_epoch and return the exact bar-close row."""
    dec_t = int(decision_epoch)
    if dec_t % 60 != 0:
        raise SotaFeatureError(f"{target_pair}: decision_epoch {dec_t} is not on a minute boundary")

    target_u = target_pair.upper()
    if target_u not in ticks_by_pair:
        raise SotaFeatureError(f"{target_u}: target tick series missing")
    missing_majors = [p for p in MAJORS if p not in ticks_by_pair]
    if missing_majors:
        raise SotaFeatureError(f"{target_u}: missing major tick series for cross features: {missing_majors}")

    needed_pairs = list(MAJORS) if target_u in MAJORS else list(MAJORS) + [target_u]
    clipped: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for p in needed_pairs:
        ts_p, px_p = ticks_by_pair[p]
        # Minute bar [m, m+60) uses ticks with ts < dec_t: zero ticks at or after decision_epoch enter features.
        cut = int(np.searchsorted(ts_p, dec_t, side="left"))
        if cut == 0:
            raise SotaFeatureError(f"{p}: no ticks strictly before decision_epoch {dec_t}")
        clipped[p] = (ts_p[:cut], px_p[:cut])

    tgt_last_ts = int(clipped[target_u][0][-1])
    tgt_spot_ref = float(clipped[target_u][1][-1])
    tick_age_s = float(dec_t - tgt_last_ts)
    if tick_age_s > max_tick_staleness_s:
        raise SotaFeatureError(
            f"{target_pair}: target tick feed stale at decision {dec_t} (age={tick_age_s:.1f}s > {max_tick_staleness_s}s)"
        )

    frame = build_sota_feature_frame_from_ticks(target_pair, clipped, feature_cols=feature_cols)
    matched = frame.loc[frame["decision_ts"] == dec_t]
    if matched.empty:
        raise SotaFeatureError(
            f"{target_pair}: no Real-Tick minute bar closed at {dec_t} (never forward-filled)"
        )
    row = matched.iloc[-1][feature_cols].astype("float64")
    vals = row.to_numpy(dtype=np.float64)
    if np.isinf(vals).any():
        bad_inf = row.index[np.isinf(vals)].tolist()
        raise SotaFeatureError(f"{target_pair}: infinite live features at {dec_t}: {bad_inf[:10]}")
    # Match cpcv_deriv_x.py / build_certified_sota_books.py: reject >= 50% missing, allow < 50% NaN for median imputation
    if float(np.mean(np.isnan(vals))) >= 0.50:
        bad_nan = row.index[np.isnan(vals)].tolist()
        raise SotaFeatureError(f"{target_pair}: >=50% missing live features at {dec_t}: {bad_nan[:10]}")

    return LiveSotaFeatureRow(
        pair=target_u,
        decision_epoch=dec_t,
        decision_utc=datetime.fromtimestamp(dec_t, tz=timezone.utc).isoformat(),
        row=row,
        tick_age_s=tick_age_s,
        source=source,
        spot_ref=tgt_spot_ref,
    )


def build_pair_decision_snapshot(
    target_pair: str,
    ticks_by_pair: dict[str, tuple[np.ndarray, np.ndarray]],
    decision_epoch: int,
    feature_cols: list[str],
    *,
    min_active_bars: int = DEFAULT_WARMUP_MIN_BARS,
    max_tick_staleness_s: float = 120.0,
    source: str = "live_1s_ticks",
) -> LiveSotaFeatureRow:
    """Build ONE causal pair snapshot enforcing min_active_bars=1600 strictly before decision_epoch (spec §3.3)."""
    dec_t = int(decision_epoch)
    target_u = target_pair.upper()
    if target_u not in ticks_by_pair:
        raise SotaFeatureError(f"{target_u}: target tick series missing")
    ts_tgt, _ = ticks_by_pair[target_u]
    cut = int(np.searchsorted(ts_tgt, dec_t, side="left"))
    if cut == 0:
        raise SotaFeatureError(f"{target_u}: no ticks strictly before decision_epoch {dec_t}")
    active_bars = int(len(np.unique(ts_tgt[:cut] // 60)))
    if active_bars < int(min_active_bars):
        raise SotaFeatureError(
            f"{target_u}: insufficient warmup active bars before {dec_t} ({active_bars} < {min_active_bars})"
        )
    return extract_decision_feature_row(
        target_u,
        ticks_by_pair,
        dec_t,
        feature_cols,
        max_tick_staleness_s=max_tick_staleness_s,
        source=source,
    )


def read_tick_store_watermark(store_dir: Path) -> dict[str, Any]:
    wm = Path(store_dir) / "watermark.json"
    if not wm.exists():
        return {}
    try:
        return json.loads(wm.read_text(encoding="utf-8"))
    except Exception:
        return {}


class LiveTickBuffer:
    """In-memory + atomic-parquet 1-second mid-tick ring buffer with minute-boundary watermark."""

    def __init__(self, pairs: list[str], store_dir: Path = DEFAULT_SOTA_TICK_STORE, ring_seconds: int = DEFAULT_RING_SECONDS):
        self.pairs = sorted(set(list(MAJORS) + [p.upper() for p in pairs]))
        for p in self.pairs:
            if p not in SOTA_DERIV_SYMBOLS:
                raise SotaFeatureError(f"unsupported Deriv FX symbol {p!r}")
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.ring_seconds = int(ring_seconds)
        self._epochs: dict[str, deque[int]] = {p: deque(maxlen=self.ring_seconds) for p in self.pairs}
        self._prices: dict[str, deque[float]] = {p: deque(maxlen=self.ring_seconds) for p in self.pairs}

    def ingest_ticks(self, pair: str, epochs: list[int] | np.ndarray, prices: list[float] | np.ndarray) -> int:
        p = pair.upper()
        de, dp = self._epochs[p], self._prices[p]
        added = 0
        for e_raw, px_raw in zip(epochs, prices):
            e = int(e_raw)
            px = float(px_raw)
            if not math.isfinite(px) or px <= 0.0:
                continue
            if de and e <= de[-1]:
                if e == de[-1]:
                    dp[-1] = px
                continue
            de.append(e)
            dp.append(px)
            added += 1
        return added

    def merge_sorted_ticks(self, pair: str, epochs: np.ndarray, prices: np.ndarray) -> None:
        """Merge arbitrary historical + new tick arrays in strictly ascending epoch order."""
        p = pair.upper()
        cur_e = np.asarray(self._epochs[p], dtype=np.int64)
        cur_p = np.asarray(self._prices[p], dtype=np.float64)
        all_e = np.concatenate([cur_e, np.asarray(epochs, dtype=np.int64)])
        all_p = np.concatenate([cur_p, np.asarray(prices, dtype=np.float64)])
        fin = np.isfinite(all_p) & (all_p > 0.0)
        all_e, all_p = all_e[fin], all_p[fin]
        if len(all_e) == 0:
            return
        order = np.argsort(all_e, kind="stable")
        all_e, all_p = all_e[order], all_p[order]
        # Keep last quote per second
        keep_last = np.empty(len(all_e), dtype=bool)
        keep_last[:-1] = all_e[:-1] != all_e[1:]
        keep_last[-1] = True
        all_e, all_p = all_e[keep_last][-self.ring_seconds :], all_p[keep_last][-self.ring_seconds :]
        self._epochs[p] = deque(all_e.tolist(), maxlen=self.ring_seconds)
        self._prices[p] = deque(all_p.tolist(), maxlen=self.ring_seconds)

    def distinct_minute_count(self, pair: str) -> int:
        arr = np.asarray(self._epochs[pair.upper()], dtype=np.int64)
        return int(len(np.unique(arr // 60))) if len(arr) else 0

    def snapshot_ticks(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {
            p: (np.asarray(self._epochs[p], dtype=np.int64), np.asarray(self._prices[p], dtype=np.float64))
            for p in self.pairs
        }

    def persist_snapshot(self, *, minute_boundary_epoch: int | None = None) -> None:
        max_by_pair: dict[str, int] = {}
        for p in self.pairs:
            if not self._epochs[p]:
                continue
            max_by_pair[p] = int(self._epochs[p][-1])
            df = pd.DataFrame({
                "epoch": np.asarray(self._epochs[p], dtype=np.int64),
                "price": np.asarray(self._prices[p], dtype=np.float64),
            })
            out = self.store_dir / f"{p}_ticks.parquet"
            tmp = out.with_suffix(".parquet.tmp")
            df.to_parquet(tmp, index=False)
            tmp.replace(out)

        wm_payload = {
            "flushed_utc": datetime.now(timezone.utc).isoformat(),
            "flushed_epoch": int(time.time()),
            "minute_boundary_epoch": int(minute_boundary_epoch) if minute_boundary_epoch is not None else None,
            "max_tick_epoch_by_pair": max_by_pair,
        }
        wm_out = self.store_dir / "watermark.json"
        wm_tmp = wm_out.with_suffix(".json.tmp")
        wm_tmp.write_text(json.dumps(wm_payload, sort_keys=True) + "\n", encoding="utf-8")
        wm_tmp.replace(wm_out)

    def load_persisted(self) -> None:
        for p in self.pairs:
            path = self.store_dir / f"{p}_ticks.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                self.merge_sorted_ticks(p, df["epoch"].to_numpy(dtype=np.int64), df["price"].to_numpy(dtype=np.float64))

    async def _request_history_paced(
        self,
        client: Any,
        sym: str,
        w_start: int,
        w_end: int,
        *,
        request_spacing_s: float,
        history_request_fn: HistoryRequestFn | None = None,
    ) -> tuple[list[int], list[float]]:
        """Fetch one <=1000s window with explicit {start, end} via injected history_request_fn (spec §3.3)."""
        req_fn: Any = history_request_fn if history_request_fn is not None else client.request
        backoff = 5.0
        for attempt in range(4):
            try:
                payload = {
                    "ticks_history": sym,
                    "style": "ticks",
                    "start": int(w_start),
                    "end": int(w_end),
                    "count": 1000,
                }
                try:
                    resp = await req_fn(payload, timeout=20.0)
                except TypeError:
                    resp = await req_fn(payload, 20.0)
                if request_spacing_s > 0.0:
                    await asyncio.sleep(request_spacing_s)
                hist = resp.get("history") or {}
                raw_t = [int(x) for x in (hist.get("times") or [])]
                raw_p = [float(x) for x in (hist.get("prices") or [])]
                # Guard Provenance #3: keep only ticks strictly inside [w_start, w_end]
                times = [t for t, p in zip(raw_t, raw_p) if w_start <= t <= w_end and math.isfinite(p) and p > 0.0]
                prices = [p for t, p in zip(raw_t, raw_p) if w_start <= t <= w_end and math.isfinite(p) and p > 0.0]
                return times, prices
            except DerivAPIError as exc:
                if (exc.code == "RateLimit" or "RateLimit" in str(exc)) and attempt < 3:
                    await asyncio.sleep(backoff)
                    backoff *= 2.0
                    continue
                raise
        return [], []

    async def warmup_from_public_ws(
        self,
        client: Any,
        *,
        lookback_hours: int = 36,
        min_bars: int = DEFAULT_WARMUP_MIN_BARS,
        request_spacing_s: float = 2.6,
        history_request_fn: HistoryRequestFn | None = None,
        now_epoch: int | None = None,
    ) -> None:
        """Rate-limit-safe (< 24 req/min), weekend-aware incremental warmup using 1000s {start, end} windows."""
        now_s = int(time.time()) if now_epoch is None else int(now_epoch)
        max_scan_floor = now_s - 5 * 86_400  # 5 calendar days spans any 48h weekend + holiday + 1440m lookback
        for p in self.pairs:
            sym = SOTA_DERIV_SYMBOLS[p]
            existing_max = int(self._epochs[p][-1]) if self._epochs[p] else None
            existing_min = int(self._epochs[p][0]) if self._epochs[p] else None

            # 1. Fill forward gap from `now_s` down to `existing_max`
            cursor = now_s
            while cursor > max_scan_floor:
                if existing_max is not None and cursor <= existing_max:
                    break
                if existing_max is None and self.distinct_minute_count(p) >= min_bars and cursor <= (now_s - lookback_hours * 3600):
                    break
                w_start = max(max_scan_floor, (existing_max + 1) if existing_max is not None else (cursor - 999), cursor - 999)
                times, prices = await self._request_history_paced(
                    client,
                    sym,
                    w_start,
                    cursor,
                    request_spacing_s=request_spacing_s,
                    history_request_fn=history_request_fn,
                )
                if times:
                    self.merge_sorted_ticks(
                        p,
                        np.asarray(times, dtype=np.int64),
                        np.asarray(prices, dtype=np.float64),
                    )
                cursor = w_start - 1

            # 2. If persisted + recent ticks still have < min_bars distinct minutes (e.g. Monday morning after 48h weekend),
            # page backwards prior to `existing_min` in 1000s {start, end} windows until >= min_bars active minutes are loaded.
            if self.distinct_minute_count(p) < min_bars and existing_min is not None:
                cursor_old = existing_min - 1
                while cursor_old > max_scan_floor and self.distinct_minute_count(p) < min_bars:
                    w_start = max(max_scan_floor, cursor_old - 999)
                    times, prices = await self._request_history_paced(
                        client,
                        sym,
                        w_start,
                        cursor_old,
                        request_spacing_s=request_spacing_s,
                        history_request_fn=history_request_fn,
                    )
                    if times:
                        self.merge_sorted_ticks(
                            p,
                            np.asarray(times, dtype=np.int64),
                            np.asarray(prices, dtype=np.float64),
                        )
                    cursor_old = w_start - 1

        self.persist_snapshot(minute_boundary_epoch=(now_s // 60) * 60)


def run_dataset_parity_selfcheck(
    dataset_dir: Path = Path("/mnt/data/deriv-sota-lab/cache/dataset_dev_v0.1"),
    ticks_dir: Path = Path("/mnt/data/deriv/ticks"),
    pair: str = "USDJPY",
) -> dict[str, Any]:
    """Prove exact Decimal 0.0 float32 drift via O_NOFOLLOW DEV loader across all 7 input lanes (spec §3.3)."""
    from certify_sota_stack05 import (
        load_pinned_dev_manifest,
        load_pinned_dev_parquet,
        load_pinned_dev_ticks,
    )

    dev_manifest = load_pinned_dev_manifest(dataset_dir)
    # Load all 7 pinned input lanes including holiday files 2025-12-24 & 2025-12-31 via O_NOFOLLOW!
    ticks_by_pair: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    files_by_pair: dict[str, int] = {}
    holiday_files_verified: list[str] = []
    for p in MAJORS:
        ts_p, px_p, fnames = load_pinned_dev_ticks(
            p,
            dev_dir=dataset_dir,
            ticks_root=ticks_dir,
            manifest=dev_manifest,
            first_day="2025-08-11",
            last_day="2026-07-31",
        )
        ticks_by_pair[p] = (ts_p, px_p)
        files_by_pair[p] = len(fnames)
        for hf in (f"{p}_2025-12-24_ticks.parquet", f"{p}_2025-12-31_ticks.parquet"):
            if hf not in fnames:
                raise SotaFeatureError(f"missing expected holiday file {hf} in DEV manifest inputs")
            holiday_files_verified.append(hf)

    ref = load_pinned_dev_parquet(pair, dev_dir=dataset_dir)
    fcols = [c for c in ref.columns if c not in ("bar_start_ts", "decision_ts") and not c.startswith(("ret_fwd_", "valid_"))]
    rebuilt = build_sota_feature_frame_from_ticks(pair, ticks_by_pair, feature_cols=fcols)

    # Compare across nonempty multi-week window spanning both early DEV and Dec holiday files
    t_lo = int(datetime(2025, 12, 18, 0, 0, tzinfo=timezone.utc).timestamp())
    t_hi = int(datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc).timestamp())
    ref_sub = ref[(ref["decision_ts"] >= t_lo) & (ref["decision_ts"] < t_hi)].set_index("decision_ts")[fcols]
    reb_sub = rebuilt[(rebuilt["decision_ts"] >= t_lo) & (rebuilt["decision_ts"] < t_hi)].set_index("decision_ts")[fcols]
    if ref_sub.empty:
        raise SotaFeatureError("empty USDJPY comparison window")
    if not ref_sub.index.equals(reb_sub.index):
        raise SotaFeatureError(f"decision_ts index mismatch: ref={len(ref_sub)} vs rebuilt={len(reb_sub)}")

    a = ref_sub.to_numpy(dtype=np.float32)
    b = reb_sub.to_numpy(dtype=np.float32)
    nan_match = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    fin = np.isfinite(a) & np.isfinite(b)
    max_diff = float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else 0.0
    exact_share = float(np.mean(a[fin] == b[fin])) if fin.any() else 1.0
    if not nan_match or Decimal(str(max_diff)) != Decimal("0"):
        raise SotaFeatureError(f"feature parity failed: nan_match={nan_match}, max_diff={max_diff}")
    return {
        "pair": pair,
        "input_pair_lanes": list(MAJORS),
        "daily_files_verified_by_pair": files_by_pair,
        "holiday_files_verified_count": len(holiday_files_verified),
        "window_utc": [t_lo, t_hi],
        "rows_checked": int(len(ref_sub)),
        "features_checked": int(len(fcols)),
        "nan_mask_identical": nan_match,
        "float32_exact_share": round(exact_share, 6),
        "max_abs_diff": max_diff,
        "passed": True,
    }


async def run_stream_daemon(args: argparse.Namespace) -> int:
    assert_non_us_deriv_host()
    logger = JsonlLogger(Path(args.log_dir) / "sota_tick_stream")
    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    buf = LiveTickBuffer(pairs=pairs, store_dir=Path(args.store_dir))
    buf.load_persisted()
    async with DerivAsyncClient(PUBLIC_WS_URL, silence_timeout_s=45.0) as client:
        logger.write("warmup_start", pairs=buf.pairs, lookback_hours=args.warmup_hours)
        await buf.warmup_from_public_ws(client, lookback_hours=args.warmup_hours)
        logger.write(
            "warmup_complete",
            counts={p: len(buf._epochs[p]) for p in buf.pairs},
            distinct_minutes={p: buf.distinct_minute_count(p) for p in buf.pairs},
        )
        subs = {}
        for p in buf.pairs:
            subs[p] = await client.subscribe({"ticks": SOTA_DERIV_SYMBOLS[p], "subscribe": 1})
        last_flush = time.monotonic()
        last_flushed_minute = (int(time.time()) // 60) * 60
        deadline = (time.monotonic() + args.duration_seconds) if args.duration_seconds else None
        while deadline is None or time.monotonic() < deadline:
            if client._background_failed.is_set():
                raise SotaFeatureError(f"public WS background failure: {client._background_error}")
            for p, sub in subs.items():
                if sub.error is not None:
                    raise SotaFeatureError(f"tick subscription error on {p}: {sub.error}")
                while not sub.queue.empty():
                    frame = sub.queue.get_nowait()
                    tk = frame.get("tick")
                    if isinstance(tk, dict) and "epoch" in tk and "quote" in tk:
                        buf.ingest_ticks(p, [int(tk["epoch"])], [float(tk["quote"])])

            now_wall = time.time()
            cur_min = (int(now_wall) // 60) * 60
            all_past_boundary = all(
                bool(buf._epochs[p]) and int(buf._epochs[p][-1]) >= cur_min for p in buf.pairs
            )
            if cur_min > last_flushed_minute and (all_past_boundary or (now_wall - cur_min) >= 1.2):
                buf.persist_snapshot(minute_boundary_epoch=cur_min)
                last_flushed_minute = cur_min
                last_flush = time.monotonic()
            elif time.monotonic() - last_flush >= 15.0:
                buf.persist_snapshot(minute_boundary_epoch=last_flushed_minute)
                last_flush = time.monotonic()
            await asyncio.sleep(0.15)
        buf.persist_snapshot(minute_boundary_epoch=(int(time.time()) // 60) * 60)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity-selfcheck", action="store_true")
    ap.add_argument("--pairs", default="EURUSD,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD")
    ap.add_argument("--store-dir", default=str(DEFAULT_SOTA_TICK_STORE))
    ap.add_argument("--log-dir", default="logs/sota_runtime")
    ap.add_argument("--warmup-hours", type=int, default=36)
    ap.add_argument("--duration-seconds", type=float, default=0.0)
    args = ap.parse_args()
    if args.parity_selfcheck:
        res = run_dataset_parity_selfcheck()
        print(json.dumps(res, indent=2))
        return 0
    return asyncio.run(run_stream_daemon(args))


if __name__ == "__main__":
    raise SystemExit(main())
