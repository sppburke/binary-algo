"""Train deriv-labeled experimental models from the isolated Deriv candle store.

Artifacts are deliberately written under ``deriv_data/models/`` and are not
registered as certified books. The latest two calendar months of Deriv data are
held out by default and are never used for fitting or threshold selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import roc_auc_score

import pipeline
from sessions import session_mask


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "deriv_data" / "candles_1m"
DEFAULT_OUT_DIR = REPO_ROOT / "deriv_data" / "models"
DERIV_DATA_ROOT = REPO_ROOT / "deriv_data"
PAIRS = ["EURUSD", "USDCHF", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "NZDUSD"]


@dataclass(frozen=True)
class DerivDataset:
    data: pd.DataFrame
    latest_raw_ts: pd.Timestamp
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    oos: pd.DataFrame
    latest_ts: pd.Timestamp
    oos_cutoff: pd.Timestamp
    train_cutoff: pd.Timestamp


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def parse_pairs(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(PAIRS)
    pairs = []
    for part in raw.split(","):
        pair = part.strip().upper()
        if not pair:
            continue
        if pair not in PAIRS:
            raise SystemExit(f"unknown pair {pair!r}; valid: {','.join(PAIRS)}")
        pairs.append(pair)
    if not pairs:
        raise SystemExit("no pairs selected")
    return pairs


def ensure_deriv_data_path(path: Path, *, purpose: str) -> Path:
    resolved = path.expanduser().resolve()
    root = DERIV_DATA_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise SystemExit(f"{purpose} must be under {root}; got {resolved}")
    return resolved


def validate_backfill_metadata(
    data_dir: Path,
    pair: str,
    *,
    min_rows: int,
    allow_api_error_boundary: bool,
) -> dict[str, Any]:
    meta_path = data_dir / f"{pair}_metadata.json"
    progress_path = data_dir / f"{pair}_progress.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing Deriv metadata: {meta_path}")
    if not progress_path.exists():
        raise FileNotFoundError(f"missing Deriv progress file: {progress_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("status") == "running":
        raise ValueError(f"{pair}: backfill is still running; refusing to train")
    if metadata.get("include_open_candle"):
        raise ValueError(f"{pair}: backfill included an open candle; rerun without --include-open-candle")
    if int(metadata.get("rows_written") or 0) < min_rows:
        raise ValueError(f"{pair}: rows_written below min_rows={min_rows}: {metadata.get('rows_written')}")
    stopped_reason = metadata.get("stopped_reason")
    if stopped_reason == "max_pages":
        raise ValueError(f"{pair}: backfill was capped by --max-pages; rerun uncapped before training")
    if stopped_reason == "api_error" and not allow_api_error_boundary:
        raise ValueError(f"{pair}: backfill stopped on API error; pass --allow-api-error-boundary only after review")
    return metadata


def load_m1(data_dir: Path, pair: str) -> pd.DataFrame:
    path = data_dir / f"{pair}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing Deriv candle parquet: {path}")
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        idx = pd.to_datetime(df["timestamp"], utc=True)
    else:
        idx = pd.to_datetime(df.index, utc=True)
    m1 = pd.DataFrame(
        {
            "open": pd.to_numeric(df["open"], errors="raise"),
            "high": pd.to_numeric(df["high"], errors="raise"),
            "low": pd.to_numeric(df["low"], errors="raise"),
            "close": pd.to_numeric(df["close"], errors="raise"),
            "volume": 1.0,
        },
        index=idx,
    )
    m1 = m1.dropna(subset=["open", "high", "low", "close"])
    m1 = m1[~m1.index.duplicated(keep="last")].sort_index()
    dt_min = m1.index.to_series().diff().dt.total_seconds().div(60.0)
    m1["gap_prev"] = dt_min.fillna(1.0).values
    return m1


def build_deriv_dataset(
    data_dir: Path,
    pair: str,
    horizon: int,
    *,
    min_rows: int,
    allow_api_error_boundary: bool,
) -> DerivDataset:
    metadata = validate_backfill_metadata(
        data_dir,
        pair,
        min_rows=min_rows,
        allow_api_error_boundary=allow_api_error_boundary,
    )
    m1 = load_m1(data_dir, pair)
    latest_raw_ts = pd.Timestamp(m1.index.max()).tz_convert("UTC")
    x = pipeline.build_features(m1).replace([np.inf, -np.inf], np.nan)
    close = m1["close"]
    fwd = close.shift(-horizon)
    ret = fwd / close - 1.0
    gap_ok = (m1["gap_prev"] <= 1.5).astype(int)
    contig = gap_ok.shift(-1)
    for k in range(2, horizon + 1):
        contig = contig * gap_ok.shift(-k)
    valid = (contig.fillna(0) > 0) & fwd.notna()
    data = x.copy()
    data["y_train"] = (ret > 0).astype("float32")
    data["fwd_ret"] = ret
    data["is_tie"] = ret == 0
    data["valid"] = valid.astype(bool)
    data["close"] = m1["close"]
    data = data.loc[data["valid"] & data["fwd_ret"].notna()].copy()
    epochs = (data.index.view("int64") // 1_000_000_000).astype("int64")
    data["session_ny"] = session_mask(epochs, "ny")
    data = data.loc[data["session_ny"]].copy()
    return DerivDataset(data=data, latest_raw_ts=latest_raw_ts, metadata=metadata)


def split_two_month_oos(
    data: pd.DataFrame,
    latest_raw_ts: pd.Timestamp,
    horizon: int,
    holdout_months: int,
    val_fraction: float,
) -> SplitFrames:
    if data.empty:
        raise ValueError("empty Deriv dataset after valid/session filters")
    latest_ts = pd.Timestamp(latest_raw_ts).tz_convert("UTC")
    oos_cutoff = latest_ts - DateOffset(months=holdout_months)
    expiry_ts = data.index + pd.Timedelta(minutes=horizon)
    pre_oos = data.loc[expiry_ts < oos_cutoff]
    oos = data.loc[data.index >= oos_cutoff]
    if pre_oos.empty or oos.empty:
        raise ValueError(
            f"insufficient data for {holdout_months}-month OOS holdout: "
            f"pre_oos={len(pre_oos)} oos={len(oos)}"
        )
    train_cutoff = pre_oos.index.min() + (pre_oos.index.max() - pre_oos.index.min()) * (1.0 - val_fraction)
    pre_oos_expiry = pre_oos.index + pd.Timedelta(minutes=horizon)
    train = pre_oos.loc[pre_oos_expiry < train_cutoff]
    val = pre_oos.loc[pre_oos.index >= train_cutoff]
    if train.empty or val.empty:
        raise ValueError(f"insufficient pre-OOS data for train/val split: train={len(train)} val={len(val)}")
    return SplitFrames(
        train=train,
        val=val,
        oos=oos,
        latest_ts=latest_ts,
        oos_cutoff=pd.Timestamp(oos_cutoff).tz_convert("UTC"),
        train_cutoff=pd.Timestamp(train_cutoff).tz_convert("UTC"),
    )


def feature_columns(data: pd.DataFrame) -> list[str]:
    banned = {"y_train", "fwd_ret", "is_tie", "valid", "close", "session_ny"}
    cols = [c for c in data.columns if c not in banned]
    return [c for c in cols if pd.api.types.is_numeric_dtype(data[c])]


def fit_lgb(train: pd.DataFrame, val: pd.DataFrame, cols: list[str], seed: int) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )
    model.fit(
        train[cols].astype("float32"),
        train["y_train"].astype("int8"),
        eval_set=[(val[cols].astype("float32"), val["y_train"].astype("int8"))],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def wilson_ci(wins: int, n: int, z: float = 1.96) -> list[float] | None:
    if n <= 0:
        return None
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt((p * (1.0 - p) + z * z / (4 * n)) / n) / denom
    return [float(center - half), float(center + half)]


def nonoverlap_mask(index: pd.Index, selected: np.ndarray, horizon: int) -> np.ndarray:
    keep = np.zeros(len(index), dtype=bool)
    next_allowed: pd.Timestamp | None = None
    gap = pd.Timedelta(minutes=horizon)
    for i, ts in enumerate(index):
        if not selected[i]:
            continue
        stamp = pd.Timestamp(ts)
        if next_allowed is None or stamp >= next_allowed:
            keep[i] = True
            next_allowed = stamp + gap
    return keep


def eval_split(frame: pd.DataFrame, proba: np.ndarray, threshold: float, horizon: int) -> dict[str, Any]:
    ret = frame["fwd_ret"].to_numpy()
    y_moved = (ret > 0).astype(int)
    pred_up = proba > 0.5
    moved = ret != 0
    tie = ret == 0
    sel = np.abs(proba - 0.5) >= threshold
    correct = np.where(pred_up, ret > 0, ret < 0)
    sel_moved = sel & moved
    sel_nonoverlap = nonoverlap_mask(frame.index, sel, horizon)
    sel_wins = int(correct[sel].sum()) if sel.any() else 0
    sel_nonoverlap_wins = int(correct[sel_nonoverlap].sum()) if sel_nonoverlap.any() else 0
    out: dict[str, Any] = {
        "rows": int(len(frame)),
        "moved_rows": int(moved.sum()),
        "tie_rows": int(tie.sum()),
        "moved_up_rate": float(np.mean(y_moved[moved])) if moved.any() else None,
        "auc_moved_only": float(roc_auc_score(y_moved[moved], proba[moved])) if len(np.unique(y_moved[moved])) == 2 else None,
        "all_accuracy_ties_lose": float(np.mean(correct)) if len(correct) else None,
        "selected_rows": int(sel.sum()),
        "selected_coverage": float(sel.sum() / len(frame)) if len(frame) else None,
        "selected_accuracy_ties_lose": float(np.mean(correct[sel])) if sel.any() else None,
        "selected_wilson_ci95": wilson_ci(sel_wins, int(sel.sum())),
        "selected_moved_rows": int(sel_moved.sum()),
        "selected_moved_accuracy": float(np.mean(correct[sel_moved])) if sel_moved.any() else None,
        "selected_nonoverlap_rows": int(sel_nonoverlap.sum()),
        "selected_nonoverlap_accuracy_ties_lose": float(np.mean(correct[sel_nonoverlap])) if sel_nonoverlap.any() else None,
        "selected_nonoverlap_wilson_ci95": wilson_ci(sel_nonoverlap_wins, int(sel_nonoverlap.sum())),
    }
    return out


def train_pair(
    *,
    pair: str,
    data_dir: Path,
    out_dir: Path,
    horizon: int,
    holdout_months: int,
    val_fraction: float,
    target_coverage: float,
    seed: int,
    seed_count: int,
    min_rows: int,
    allow_api_error_boundary: bool,
    run_id: str,
) -> dict[str, Any]:
    dataset = build_deriv_dataset(
        data_dir,
        pair,
        horizon,
        min_rows=min_rows,
        allow_api_error_boundary=allow_api_error_boundary,
    )
    data = dataset.data
    splits = split_two_month_oos(data, dataset.latest_raw_ts, horizon, holdout_months, val_fraction)
    cols = feature_columns(data)
    clean = {
        name: frame.dropna(subset=cols + ["y_train", "fwd_ret"]).copy()
        for name, frame in (("train", splits.train), ("val", splits.val), ("oos", splits.oos))
    }
    if min(len(v) for v in clean.values()) < 100:
        raise ValueError({name: len(frame) for name, frame in clean.items()})

    train_fit = clean["train"].loc[~clean["train"]["is_tie"]].copy()
    val_fit = clean["val"].loc[~clean["val"]["is_tie"]].copy()
    if min(len(train_fit), len(val_fit)) < 100:
        raise ValueError({"train_fit": len(train_fit), "val_fit": len(val_fit)})

    run_out = out_dir / run_id
    pair_out = run_out / f"deriv_{pair}_m{horizon}_lgb"
    if pair_out.exists():
        raise FileExistsError(f"refusing to overwrite existing Deriv model directory: {pair_out}")
    pair_out.mkdir(parents=True, exist_ok=True)

    seeds = [int(seed + i) for i in range(seed_count)]
    models = [fit_lgb(train_fit, val_fit, cols, s) for s in seeds]

    def blend(frame: pd.DataFrame) -> np.ndarray:
        x = frame[cols].astype("float32")
        return np.mean([m.predict_proba(x)[:, 1] for m in models], axis=0)

    p_val = blend(clean["val"])
    conf_val = np.abs(p_val - 0.5)
    threshold = float(np.quantile(conf_val, 1.0 - target_coverage))
    p_train = blend(clean["train"])
    p_oos = blend(clean["oos"])

    model_paths = []
    for model, s in zip(models, seeds, strict=True):
        model_path = pair_out / f"deriv_{pair}_m{horizon}_seed{s}_lgb.txt"
        model.booster_.save_model(model_path)
        model_paths.append(str(model_path.relative_to(REPO_ROOT)))
    manifest = {
        "schema": "deriv_experimental_model.v1",
        "status": "EXPERIMENTAL_NOT_CERTIFIED",
        "run_id": run_id,
        "created_at_utc": utc_now_iso(),
        "pair": pair,
        "horizon_minutes": int(horizon),
        "source_data_dir": str(data_dir.relative_to(REPO_ROOT)),
        "source_backfill_metadata": dataset.metadata,
        "artifact_dir": str(pair_out.relative_to(REPO_ROOT)),
        "model_paths": model_paths,
        "feature_count": len(cols),
        "feature_cols": cols,
        "seeds": seeds,
        "seed_count": int(seed_count),
        "target_coverage": float(target_coverage),
        "conf_threshold_selected_on_val": threshold,
        "holdout_months": int(holdout_months),
        "latest_deriv_timestamp_utc": splits.latest_ts.isoformat(),
        "oos_cutoff_utc": splits.oos_cutoff.isoformat(),
        "train_val_cutoff_utc": splits.train_cutoff.isoformat(),
        "splits": {
            "train": {
                "first": clean["train"].index.min().isoformat(),
                "last": clean["train"].index.max().isoformat(),
                "rows": int(len(clean["train"])),
                "fit_rows_ex_ties": int(len(train_fit)),
            },
            "val": {
                "first": clean["val"].index.min().isoformat(),
                "last": clean["val"].index.max().isoformat(),
                "rows": int(len(clean["val"])),
                "fit_rows_ex_ties": int(len(val_fit)),
            },
            "oos": {
                "first": clean["oos"].index.min().isoformat(),
                "last": clean["oos"].index.max().isoformat(),
                "rows": int(len(clean["oos"])),
            },
        },
        "metrics": {
            "train": eval_split(clean["train"], p_train, threshold, horizon),
            "val": eval_split(clean["val"], p_val, threshold, horizon),
            "oos_heldout_latest_two_months": eval_split(clean["oos"], p_oos, threshold, horizon),
        },
    }
    manifest_path = pair_out / f"deriv_{pair}_m{horizon}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", default="all")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--holdout-months", type=int, default=2)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--target-coverage", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=20260614)
    p.add_argument("--seed-count", type=int, default=5)
    p.add_argument("--min-rows", type=int, default=50_000)
    p.add_argument("--allow-api-error-boundary", action="store_true")
    p.add_argument("--run-id", default=None, help="output run id; default is UTC timestamp")
    p.add_argument("--check-only", action="store_true", help="build datasets and split, but do not fit models")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pairs = parse_pairs(args.pairs)
    data_dir = ensure_deriv_data_path(Path(args.data_dir), purpose="Deriv training data directory")
    out_dir = ensure_deriv_data_path(Path(args.out_dir), purpose="Deriv model output directory")
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if not args.check_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        run_dir = out_dir / run_id
        if run_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing Deriv training run: {run_dir}")

    results = []
    for pair in pairs:
        if args.check_only:
            dataset = build_deriv_dataset(
                data_dir,
                pair,
                args.horizon,
                min_rows=args.min_rows,
                allow_api_error_boundary=args.allow_api_error_boundary,
            )
            data = dataset.data
            splits = split_two_month_oos(
                data,
                dataset.latest_raw_ts,
                args.horizon,
                args.holdout_months,
                args.val_fraction,
            )
            print(
                " ".join(
                    [
                        "CHECK",
                        pair,
                        f"rows={len(data)}",
                        f"train={len(splits.train)}",
                        f"val={len(splits.val)}",
                        f"oos={len(splits.oos)}",
                        f"oos_cutoff={splits.oos_cutoff.isoformat()}",
                    ]
                )
            )
            continue
        manifest = train_pair(
            pair=pair,
            data_dir=data_dir,
            out_dir=out_dir,
            horizon=args.horizon,
            holdout_months=args.holdout_months,
            val_fraction=args.val_fraction,
            target_coverage=args.target_coverage,
            seed=args.seed,
            seed_count=args.seed_count,
            min_rows=args.min_rows,
            allow_api_error_boundary=args.allow_api_error_boundary,
            run_id=run_id,
        )
        results.append(manifest)
        oos = manifest["metrics"]["oos_heldout_latest_two_months"]
        print(
            " ".join(
                [
                    "DERIV_MODEL",
                    pair,
                    f"oos_rows={oos['rows']}",
                    f"oos_sel={oos['selected_rows']}",
                    f"oos_acc_ties_lose={oos['selected_accuracy_ties_lose']}",
                    f"oos_no_acc_ties_lose={oos['selected_nonoverlap_accuracy_ties_lose']}",
                    f"oos_no_n={oos['selected_nonoverlap_rows']}",
                    f"oos_cutoff={manifest['oos_cutoff_utc']}",
                    f"artifact={manifest['artifact_dir']}",
                ]
            )
        )

    if results:
        summary = {
            "schema": "deriv_experimental_training_run.v1",
            "created_at_utc": utc_now_iso(),
            "status": "EXPERIMENTAL_NOT_CERTIFIED",
            "run_id": run_id,
            "holdout_months": int(args.holdout_months),
            "pairs": results,
        }
        path = out_dir / run_id / f"deriv_m{args.horizon}_run_summary.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"summary={path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
