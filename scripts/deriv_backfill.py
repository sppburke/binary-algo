"""Backfill Deriv-native FX candle history into an isolated local store.

The store is intentionally outside the existing training data and model
locations. It writes under ``deriv_data/`` by default, which is gitignored.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from deriv_client import DerivAPIError, DerivOptionsClient


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "deriv_data" / "candles_1m"

DERIV_LEGACY_WS_URL = "wss://ws.derivws.com/websockets/v3?app_id=1089"
DERIV_OPTIONS_WS_URL = "wss://api.derivws.com/trading/v1/options/ws/public"
ENDPOINTS = {
    "legacy": DERIV_LEGACY_WS_URL,
    "options": DERIV_OPTIONS_WS_URL,
}
DERIV_DATA_ROOT = REPO_ROOT / "deriv_data"

PAIR_TO_SYMBOL = {
    "EURUSD": "frxEURUSD",
    "USDCHF": "frxUSDCHF",
    "GBPUSD": "frxGBPUSD",
    "USDJPY": "frxUSDJPY",
    "USDCAD": "frxUSDCAD",
    "AUDUSD": "frxAUDUSD",
    "NZDUSD": "frxNZDUSD",
}


@dataclass(frozen=True)
class PageResult:
    payload: dict[str, Any]
    row_count: int
    oldest_epoch: int | None
    newest_epoch: int | None


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def epoch_iso(epoch: int | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(int(epoch), UTC).replace(microsecond=0).isoformat()


def completed_cutoff_epoch(granularity: int) -> int:
    now = int(time.time())
    return (now // granularity) * granularity - granularity


def parse_pairs(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(PAIR_TO_SYMBOL)
    out = []
    for part in raw.split(","):
        pair = part.strip().upper()
        if not pair:
            continue
        if pair not in PAIR_TO_SYMBOL:
            raise SystemExit(f"unknown pair {pair!r}; valid: {','.join(PAIR_TO_SYMBOL)}")
        out.append(pair)
    if not out:
        raise SystemExit("no pairs selected")
    return out


def ensure_deriv_data_path(path: Path, *, purpose: str) -> Path:
    resolved = path.expanduser().resolve()
    root = DERIV_DATA_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise SystemExit(f"{purpose} must be under {root}; got {resolved}")
    return resolved


def request_candle_page(
    client: DerivOptionsClient,
    *,
    symbol: str,
    granularity: int,
    count: int,
    end: str,
    timeout: float,
) -> PageResult:
    payload = {
        "ticks_history": symbol,
        "style": "candles",
        "granularity": int(granularity),
        "count": int(count),
        "end": end,
        "adjust_start_time": 1,
    }
    resp = client.request(payload, timeout=max(timeout, 20.0))
    candles = resp.get("candles") or []
    epochs = [int(c["epoch"]) for c in candles if "epoch" in c]
    return PageResult(
        payload=resp,
        row_count=len(candles),
        oldest_epoch=min(epochs) if epochs else None,
        newest_epoch=max(epochs) if epochs else None,
    )


def candles_frame(
    *,
    pair: str,
    symbol: str,
    page: PageResult,
    granularity: int,
    endpoint: str,
    drop_open_candle: bool,
) -> pd.DataFrame:
    rows = page.payload.get("candles") or []
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    required = {"epoch", "open", "high", "low", "close"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DerivAPIError(f"{pair}: candle payload missing columns {missing}")

    frame["epoch"] = pd.to_numeric(frame["epoch"], errors="raise").astype("int64")
    for col in ("open", "high", "low", "close"):
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype("float64")

    if drop_open_candle:
        frame = frame.loc[frame["epoch"] <= completed_cutoff_epoch(granularity)]

    frame["timestamp"] = pd.to_datetime(frame["epoch"], unit="s", utc=True)
    frame["pair"] = pair
    frame["symbol"] = symbol
    frame["granularity_s"] = int(granularity)
    frame["source_endpoint"] = endpoint
    frame = frame.set_index("timestamp", drop=False)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame[
        [
            "timestamp",
            "epoch",
            "pair",
            "symbol",
            "granularity_s",
            "source_endpoint",
            "open",
            "high",
            "low",
            "close",
        ]
    ]


def merge_existing(path: Path, frame: pd.DataFrame, replace: bool) -> pd.DataFrame:
    if replace or not path.exists():
        return frame
    old = pd.read_parquet(path)
    if "timestamp" in old.columns:
        old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True)
        old = old.set_index("timestamp", drop=False)
    merged = pd.concat([old, frame], axis=0)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged


def write_metadata(
    *,
    out_dir: Path,
    pair: str,
    metadata: dict[str, Any],
) -> None:
    meta_path = out_dir / f"{pair}_metadata.json"
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(meta_path)


def pair_page_dir(out_dir: Path, pair: str) -> Path:
    return out_dir / "_pages" / pair


def pair_progress_path(out_dir: Path, pair: str) -> Path:
    return out_dir / f"{pair}_progress.json"


def reset_pair_artifacts(out_dir: Path, pair: str) -> None:
    for path in (
        out_dir / f"{pair}.parquet",
        out_dir / f"{pair}_metadata.json",
        pair_progress_path(out_dir, pair),
    ):
        if path.exists():
            path.unlink()
    page_dir = pair_page_dir(out_dir, pair)
    if page_dir.exists():
        shutil.rmtree(page_dir)


def write_progress(out_dir: Path, pair: str, progress: dict[str, Any]) -> None:
    path = pair_progress_path(out_dir, pair)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_progress(out_dir: Path, pair: str) -> dict[str, Any] | None:
    path = pair_progress_path(out_dir, pair)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_page_shard(out_dir: Path, pair: str, pages: int, frame: pd.DataFrame) -> Path:
    page_dir = pair_page_dir(out_dir, pair)
    page_dir.mkdir(parents=True, exist_ok=True)
    oldest = int(frame["epoch"].min())
    newest = int(frame["epoch"].max())
    path = page_dir / f"{pages:06d}_{oldest}_{newest}.parquet"
    tmp = path.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=True)
    tmp.replace(path)
    return path


def load_page_shards(out_dir: Path, pair: str) -> pd.DataFrame:
    page_dir = pair_page_dir(out_dir, pair)
    if not page_dir.exists():
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in sorted(page_dir.glob("*.parquet"))]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, axis=0)
    if "timestamp" in combined.columns:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
        combined = combined.set_index("timestamp", drop=False)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined


def backfill_pair(
    *,
    pair: str,
    out_dir: Path,
    endpoint: str,
    granularity: int,
    count: int,
    max_pages: int,
    sleep_s: float,
    timeout: float,
    replace: bool,
    include_open_candle: bool,
) -> dict[str, Any]:
    symbol = PAIR_TO_SYMBOL[pair]
    url = ENDPOINTS[endpoint]
    errors: list[str] = []
    if replace:
        reset_pair_artifacts(out_dir, pair)

    progress = load_progress(out_dir, pair)
    if progress and progress.get("status") == "running":
        pages = int(progress.get("pages_requested", 0))
        rows_seen = int(progress.get("rows_seen_before_dedupe", 0))
        oldest_seen = progress.get("oldest_page_epoch")
        newest_seen = progress.get("newest_page_epoch")
        end = str(progress.get("next_end", "latest"))
    elif progress and progress.get("status") == "stopped" and progress.get("next_end"):
        pages = int(progress.get("pages_requested", 0))
        rows_seen = int(progress.get("rows_seen_before_dedupe", 0))
        oldest_seen = progress.get("oldest_page_epoch")
        newest_seen = progress.get("newest_page_epoch")
        end = str(progress["next_end"])
    else:
        pages = 0
        rows_seen = 0
        oldest_seen = None
        newest_seen = None
        end = "latest"
    next_end_cursor = end
    stopped_reason = "max_pages"

    with DerivOptionsClient(url, timeout=timeout) as client:
        while True:
            if max_pages and pages >= max_pages:
                stopped_reason = "max_pages"
                break
            try:
                page = request_candle_page(
                    client,
                    symbol=symbol,
                    granularity=granularity,
                    count=count,
                    end=end,
                    timeout=timeout,
                )
            except DerivAPIError as exc:
                errors.append(str(exc))
                stopped_reason = "api_error"
                break

            pages += 1
            rows_seen += page.row_count
            if page.row_count == 0 or page.oldest_epoch is None:
                stopped_reason = "empty_page"
                break

            if oldest_seen is not None and page.oldest_epoch >= oldest_seen:
                stopped_reason = "no_backward_progress"
                break

            frame = candles_frame(
                pair=pair,
                symbol=symbol,
                page=page,
                granularity=granularity,
                endpoint=endpoint,
                drop_open_candle=not include_open_candle,
            )
            if not frame.empty:
                write_page_shard(out_dir, pair, pages, frame)
                oldest_seen = page.oldest_epoch
                newest_seen = max(newest_seen or page.newest_epoch or 0, page.newest_epoch or 0)

            next_end = str(int(page.oldest_epoch) - int(granularity))
            next_end_cursor = next_end
            write_progress(
                out_dir,
                pair,
                {
                    "status": "running",
                    "pair": pair,
                    "symbol": symbol,
                    "endpoint": endpoint,
                    "granularity_s": int(granularity),
                    "pages_requested": int(pages),
                    "rows_seen_before_dedupe": int(rows_seen),
                    "oldest_page_epoch": oldest_seen,
                    "oldest_page_utc": epoch_iso(oldest_seen),
                    "newest_page_epoch": newest_seen,
                    "newest_page_utc": epoch_iso(newest_seen),
                    "next_end": next_end,
                    "updated_at_utc": utc_now_iso(),
                },
            )
            if next_end == end:
                stopped_reason = "stalled_end_cursor"
                break
            end = next_end
            if sleep_s > 0:
                time.sleep(sleep_s)

    pair_path = out_dir / f"{pair}.parquet"
    shard_frame = load_page_shards(out_dir, pair)
    if not shard_frame.empty:
        combined = shard_frame
        combined = merge_existing(pair_path, combined, replace=replace)
        tmp_path = pair_path.with_suffix(".parquet.tmp")
        combined.to_parquet(tmp_path, index=True)
        tmp_path.replace(pair_path)
    elif pair_path.exists() and not replace:
        combined = pd.read_parquet(pair_path)
        if "timestamp" in combined.columns:
            combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
            combined = combined.set_index("timestamp", drop=False)
    else:
        combined = pd.DataFrame()

    if not combined.empty:
        epochs = pd.to_numeric(combined["epoch"], errors="coerce")
        first_epoch = int(epochs.min())
        last_epoch = int(epochs.max())
        rows_written = int(len(combined))
    else:
        first_epoch = None
        last_epoch = None
        rows_written = 0

    metadata = {
        "schema": "deriv_candles.v1",
        "pair": pair,
        "symbol": symbol,
        "endpoint": endpoint,
        "endpoint_url": url,
        "granularity_s": int(granularity),
        "count_per_request": int(count),
        "fetched_at_utc": utc_now_iso(),
        "pages_requested": int(pages),
        "rows_seen_before_dedupe": int(rows_seen),
        "rows_written": rows_written,
        "first_epoch": first_epoch,
        "first_utc": epoch_iso(first_epoch),
        "last_epoch": last_epoch,
        "last_utc": epoch_iso(last_epoch),
        "oldest_page_epoch": oldest_seen,
        "oldest_page_utc": epoch_iso(oldest_seen),
        "newest_page_epoch": newest_seen,
        "newest_page_utc": epoch_iso(newest_seen),
        "stopped_reason": stopped_reason,
        "errors": errors,
        "artifact": str(pair_path.relative_to(REPO_ROOT)),
        "page_shards": str(pair_page_dir(out_dir, pair).relative_to(REPO_ROOT)),
        "include_open_candle": bool(include_open_candle),
        "next_end": next_end_cursor,
    }
    write_metadata(out_dir=out_dir, pair=pair, metadata=metadata)
    write_progress(
        out_dir,
        pair,
        {
            **metadata,
            "status": "complete" if stopped_reason != "api_error" else "stopped",
        },
    )
    return metadata


def write_run_summary(out_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    summary = {
        "schema": "deriv_backfill_run.v1",
        "created_at_utc": utc_now_iso(),
        "args": {
            "pairs": args.pairs,
            "endpoint": args.endpoint,
            "granularity": args.granularity,
            "count": args.count,
            "max_pages": args.max_pages,
            "out_dir": str(Path(args.out_dir).resolve()),
        },
        "pairs": rows,
    }
    path = out_dir / "_last_run_summary.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", default="all", help="comma-separated pairs, or all")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="gitignored output directory")
    p.add_argument("--endpoint", choices=sorted(ENDPOINTS), default="legacy")
    p.add_argument("--granularity", type=int, default=60, help="candle size in seconds; 60 is Deriv's smallest candle bar")
    p.add_argument("--count", type=int, default=5000, help="requested candles per page")
    p.add_argument("--max-pages", type=int, default=0, help="0 means page backward until Deriv stops making progress")
    p.add_argument("--sleep-s", type=float, default=0.2)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--replace", action="store_true", help="replace pair parquet instead of merging with existing rows")
    p.add_argument("--include-open-candle", action="store_true", help="keep Deriv's current incomplete candle if returned")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    pairs = parse_pairs(args.pairs)
    out_dir = ensure_deriv_data_path(Path(args.out_dir), purpose="Deriv backfill output directory")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for pair in pairs:
        meta = backfill_pair(
            pair=pair,
            out_dir=out_dir,
            endpoint=args.endpoint,
            granularity=args.granularity,
            count=args.count,
            max_pages=args.max_pages,
            sleep_s=args.sleep_s,
            timeout=args.timeout,
            replace=args.replace,
            include_open_candle=args.include_open_candle,
        )
        rows.append(meta)
        print(
            " ".join(
                [
                    "PAIR",
                    pair,
                    f"rows={meta['rows_written']}",
                    f"first={meta['first_utc']}",
                    f"last={meta['last_utc']}",
                    f"stop={meta['stopped_reason']}",
                ]
            )
        )

    summary_path = write_run_summary(out_dir, rows, args)
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
