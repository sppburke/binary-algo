"""Deriv-native replication attempt for the certified 15m book schemas.

This is a certification attempt, not a certification shortcut. It trains against
the incumbent book feature columns where Deriv 1m candles can reproduce them,
uses the latest two months as OOS, counts ties as losses, and reports
first-come non-overlapping selected trades.
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

import deriv_train
from book_runtime import load_target_books
from live_features import LiveFeatureError, build_cross_pair_features


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "deriv_data" / "candles_1m"
DEFAULT_MODEL_DIR = REPO_ROOT / "deriv_data" / "models"
DEFAULT_RESULT_DIR = REPO_ROOT / "results" / "json"
PAIRS = ["EURUSD", "USDCHF", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "NZDUSD"]
BREAKEVEN = 0.541
SETTLEMENT_MODE = "deriv_candle_close_proxy"
CERTIFICATION_ELIGIBLE = False


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def parse_pairs(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(PAIRS)
    pairs = [p.strip().upper() for p in raw.split(",") if p.strip()]
    bad = [p for p in pairs if p not in PAIRS]
    if bad:
        raise SystemExit(f"unknown pairs {bad}; valid: {','.join(PAIRS)}")
    if not pairs:
        raise SystemExit("no pairs selected")
    return pairs


def load_all_m1(data_dir: Path, pairs: list[str], min_rows: int, allow_api_error_boundary: bool) -> dict[str, pd.DataFrame]:
    out = {}
    for pair in pairs:
        deriv_train.validate_backfill_metadata(
            data_dir,
            pair,
            min_rows=min_rows,
            allow_api_error_boundary=allow_api_error_boundary,
        )
        out[pair] = deriv_train.load_m1(data_dir, pair)
    return out


def add_schema_features(
    *,
    pair: str,
    expected_cols: list[str],
    base: deriv_train.DerivDataset,
    closes: pd.DataFrame,
) -> tuple[pd.DataFrame | None, list[str], str | None]:
    data = base.data.copy()
    missing = [c for c in expected_cols if c not in data.columns]
    if missing:
        try:
            xp = build_cross_pair_features(pair, closes, expected_cols)
        except LiveFeatureError as exc:
            return None, missing, str(exc)
        overlap_cols = [c for c in xp.columns if c in data.columns]
        if overlap_cols:
            xp = xp.drop(columns=overlap_cols)
        data = data.join(xp, how="left")
    missing = [c for c in expected_cols if c not in data.columns]
    if missing:
        return None, missing, "schema columns unavailable from Deriv candles"
    return data, [], None


def assert_feature_schema_safe(cols: list[str]) -> None:
    banned_exact = {"y", "target", "label", "fwd_ret", "valid", "is_tie", "close", "future_close"}
    banned_fragments = ("fwd", "future", "target", "label", "is_tie")
    bad = [
        c
        for c in cols
        if c in banned_exact or any(fragment in c.lower() for fragment in banned_fragments)
    ]
    if bad:
        raise ValueError(f"feature schema contains forbidden target-like columns: {bad[:20]}")


def split_frames(
    data: pd.DataFrame,
    latest_raw_ts: pd.Timestamp,
    horizon: int,
    holdout_months: int,
    val_fraction: float,
    cols: list[str],
) -> tuple[deriv_train.SplitFrames, dict[str, pd.DataFrame], dict[str, int]]:
    splits = deriv_train.split_two_month_oos(data, latest_raw_ts, horizon, holdout_months, val_fraction)
    clean = {
        name: frame.dropna(subset=cols + ["y_train", "fwd_ret"]).copy()
        for name, frame in (("train", splits.train), ("val", splits.val), ("oos", splits.oos))
    }
    fit_counts = {
        "train_fit_ex_ties": int((~clean["train"]["is_tie"]).sum()),
        "val_fit_ex_ties": int((~clean["val"]["is_tie"]).sum()),
    }
    return splits, clean, fit_counts


def fit_seed_ensemble(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cols: list[str],
    seeds: list[int],
) -> list[lgb.LGBMClassifier]:
    train_fit = train.loc[~train["is_tie"]].copy()
    val_fit = val.loc[~val["is_tie"]].copy()
    if min(len(train_fit), len(val_fit)) < 100:
        raise ValueError({"train_fit": len(train_fit), "val_fit": len(val_fit)})
    return [deriv_train.fit_lgb(train_fit, val_fit, cols, seed) for seed in seeds]


def predict_ensemble(models: list[lgb.LGBMClassifier], frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = frame[cols].astype("float32")
    return np.mean([m.predict_proba(x)[:, 1] for m in models], axis=0)


def select_threshold_by_worst_half(
    val: pd.DataFrame,
    p_val: np.ndarray,
    *,
    horizon: int,
    coverages: list[float],
) -> dict[str, Any]:
    if val.empty:
        raise ValueError("empty validation split")
    midpoint = val.index.min() + (val.index.max() - val.index.min()) / 2
    halves = [val.index < midpoint, val.index >= midpoint]
    best: dict[str, Any] | None = None
    for cov in coverages:
        conf = np.abs(p_val - 0.5)
        threshold = float(np.quantile(conf, max(0.0, 1.0 - cov)))
        half_metrics = []
        for mask in halves:
            frame = val.loc[mask]
            proba = p_val[mask]
            metrics = deriv_train.eval_split(frame, proba, threshold, horizon)
            half_metrics.append(metrics)
        if any(m["selected_nonoverlap_rows"] < 25 for m in half_metrics):
            continue
        worst = min(m["selected_nonoverlap_accuracy_ties_lose"] for m in half_metrics)
        candidate = {
            "target_coverage": float(cov),
            "conf_threshold": threshold,
            "worst_half_nonoverlap_accuracy": float(worst),
            "half_metrics": half_metrics,
        }
        if best is None or candidate["worst_half_nonoverlap_accuracy"] > best["worst_half_nonoverlap_accuracy"]:
            best = candidate
    if best is None:
        conf = np.abs(p_val - 0.5)
        threshold = float(np.quantile(conf, 0.90))
        best = {
            "target_coverage": 0.10,
            "conf_threshold": threshold,
            "worst_half_nonoverlap_accuracy": None,
            "half_metrics": [],
            "fallback": "no coverage candidate had >=25 nonoverlap trades in both halves",
        }
    return best


def write_models(
    *,
    models: list[lgb.LGBMClassifier],
    model_dir: Path,
    run_id: str,
    pair: str,
    book_id: str,
    horizon: int,
    seeds: list[int],
) -> list[str]:
    pair_dir = model_dir / run_id / f"book_deriv_{pair}_m{horizon}_{book_id.replace('.', '_')}"
    if pair_dir.exists():
        raise FileExistsError(f"refusing to overwrite {pair_dir}")
    pair_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for model, seed in zip(models, seeds, strict=True):
        path = pair_dir / f"book_deriv_{pair}_m{horizon}_seed{seed}_lgb.txt"
        model.booster_.save_model(path)
        paths.append(str(path.relative_to(REPO_ROOT)))
    return paths


def cpcv_paths(
    data: pd.DataFrame,
    cols: list[str],
    *,
    horizon: int,
    seed_base: int,
    seed_count: int,
    n_folds: int,
    test_folds: int,
    coverages: list[float],
) -> dict[str, Any]:
    ordered = data.sort_index()
    fold_ids = np.array_split(np.arange(len(ordered)), n_folds)
    fold_intervals = [(ordered.index[idx[0]], ordered.index[idx[-1]]) for idx in fold_ids if len(idx)]
    path_rows = []
    path_no = 0
    for combo in itertools.combinations(range(len(fold_ids)), test_folds):
        path_no += 1
        test_idx = np.concatenate([fold_ids[i] for i in combo])
        test_mask = np.zeros(len(ordered), dtype=bool)
        test_mask[test_idx] = True
        train_mask = ~test_mask
        expiry = ordered.index + pd.Timedelta(minutes=horizon)
        for i in combo:
            start, end = fold_intervals[i]
            train_mask &= ~((ordered.index < start) & (expiry >= start))
            train_mask &= ~((ordered.index >= start) & (ordered.index <= end))
        trainval = ordered.loc[train_mask].copy()
        test = ordered.loc[test_mask].copy()
        if len(trainval) < 1000 or len(test) < 100:
            path_rows.append({"path": path_no, "folds": list(combo), "status": "skipped_thin"})
            continue
        train_cutoff = trainval.index.min() + (trainval.index.max() - trainval.index.min()) * 0.75
        train = trainval.loc[trainval.index < train_cutoff].copy()
        val = trainval.loc[trainval.index >= train_cutoff].copy()
        train = train.dropna(subset=cols + ["y_train", "fwd_ret"])
        val = val.dropna(subset=cols + ["y_train", "fwd_ret"])
        test = test.dropna(subset=cols + ["y_train", "fwd_ret"])
        if min(len(train), len(val), len(test)) < 100:
            path_rows.append({"path": path_no, "folds": list(combo), "status": "skipped_after_dropna"})
            continue
        seeds = [seed_base + path_no * 100 + i for i in range(seed_count)]
        models = fit_seed_ensemble(train, val, cols, seeds)
        p_val = predict_ensemble(models, val, cols)
        selected = select_threshold_by_worst_half(val, p_val, horizon=horizon, coverages=coverages)
        p_test = predict_ensemble(models, test, cols)
        metrics = deriv_train.eval_split(test, p_test, selected["conf_threshold"], horizon)
        path_rows.append(
            {
                "path": path_no,
                "folds": list(combo),
                "status": "ok",
                "train_rows": int(len(train)),
                "val_rows": int(len(val)),
                "test_rows": int(len(test)),
                "threshold": selected["conf_threshold"],
                "target_coverage": selected["target_coverage"],
                "metrics": metrics,
            }
        )
    accs = [
        row["metrics"]["selected_nonoverlap_accuracy_ties_lose"]
        for row in path_rows
        if row.get("status") == "ok" and row["metrics"]["selected_nonoverlap_accuracy_ties_lose"] is not None
    ]
    ns = [
        row["metrics"]["selected_nonoverlap_rows"]
        for row in path_rows
        if row.get("status") == "ok"
    ]
    return {
        "schema": "book_deriv_cpcv.v1",
        "n_folds": int(n_folds),
        "test_folds": int(test_folds),
        "path_count": len(path_rows),
        "ok_path_count": int(len(accs)),
        "p10_selected_nonoverlap_accuracy": float(np.quantile(accs, 0.10)) if accs else None,
        "mean_selected_nonoverlap_accuracy": float(np.mean(accs)) if accs else None,
        "clear_rate_breakeven": float(np.mean(np.array(accs) >= BREAKEVEN)) if accs else None,
        "min_selected_nonoverlap_rows": int(min(ns)) if ns else 0,
        "paths": path_rows,
    }


def replicate_pair(
    *,
    pair: str,
    book: Any,
    base: deriv_train.DerivDataset,
    closes: pd.DataFrame,
    model_dir: Path,
    run_id: str,
    horizon: int,
    holdout_months: int,
    val_fraction: float,
    coverages: list[float],
    seed_base: int,
    run_cpcv: bool,
    cpcv_seed_count: int,
) -> dict[str, Any]:
    expected_cols = book.feature_cols
    assert_feature_schema_safe(expected_cols)
    data, missing, reason = add_schema_features(pair=pair, expected_cols=expected_cols, base=base, closes=closes)
    if data is None:
        return {
            "pair": pair,
            "book_id": book.book_id,
            "status": "SCHEMA_UNAVAILABLE",
            "missing_columns": missing,
            "reason": reason,
            "of_missing_count": sum(c.startswith("OF_") for c in missing),
        }
    splits, clean, fit_counts = split_frames(data, base.latest_raw_ts, horizon, holdout_months, val_fraction, expected_cols)
    seed_count = max(1, len(book.model_paths))
    seeds = [seed_base + i for i in range(seed_count)]
    models = fit_seed_ensemble(clean["train"], clean["val"], expected_cols, seeds)
    p_val = predict_ensemble(models, clean["val"], expected_cols)
    selected = select_threshold_by_worst_half(clean["val"], p_val, horizon=horizon, coverages=coverages)
    p_train = predict_ensemble(models, clean["train"], expected_cols)
    p_oos = predict_ensemble(models, clean["oos"], expected_cols)
    model_paths = write_models(
        models=models,
        model_dir=model_dir,
        run_id=run_id,
        pair=pair,
        book_id=book.book_id,
        horizon=horizon,
        seeds=seeds,
    )
    metrics = {
        "train": deriv_train.eval_split(clean["train"], p_train, selected["conf_threshold"], horizon),
        "val": deriv_train.eval_split(clean["val"], p_val, selected["conf_threshold"], horizon),
        "oos_latest_two_months": deriv_train.eval_split(clean["oos"], p_oos, selected["conf_threshold"], horizon),
    }
    cpcv = None
    if run_cpcv:
        pre_oos = pd.concat([clean["train"], clean["val"]], axis=0).sort_index()
        cpcv = cpcv_paths(
            pre_oos,
            expected_cols,
            horizon=horizon,
            seed_base=seed_base + 10_000,
            seed_count=cpcv_seed_count,
            n_folds=6,
            test_folds=2,
            coverages=coverages,
        )
    gates = {
        "breakeven": BREAKEVEN,
        "settlement_mode": SETTLEMENT_MODE,
        "eligible_for_certification": CERTIFICATION_ELIGIBLE,
        "certification_blocker": (
            "Deriv public candle closes are not Deriv tick-settlement labels; "
            "official certification needs tick-level entry/expiry settlement evidence."
        ),
        "oos_nonoverlap_clears": bool(
            (metrics["oos_latest_two_months"]["selected_nonoverlap_accuracy_ties_lose"] or 0.0) >= BREAKEVEN
        ),
        "cpcv_p10_clears": bool(cpcv and (cpcv["p10_selected_nonoverlap_accuracy"] or 0.0) >= BREAKEVEN),
        "cpcv_clear_rate_clears": bool(cpcv and (cpcv["clear_rate_breakeven"] or 0.0) >= 0.80),
    }
    gates["would_clear_numeric_gates"] = bool(
        gates["oos_nonoverlap_clears"] and gates["cpcv_p10_clears"] and gates["cpcv_clear_rate_clears"]
    )
    gates["verdict"] = "NOT_CERTIFIED"
    return {
        "pair": pair,
        "book_id": book.book_id,
        "status": "OK",
        "verdict": gates["verdict"],
        "horizon_minutes": horizon,
        "feature_count": len(expected_cols),
        "model_count": seed_count,
        "model_paths": model_paths,
        "latest_deriv_timestamp_utc": splits.latest_ts.isoformat(),
        "oos_cutoff_utc": splits.oos_cutoff.isoformat(),
        "train_val_cutoff_utc": splits.train_cutoff.isoformat(),
        "split_rows": {
            "train": int(len(clean["train"])),
            "val": int(len(clean["val"])),
            "oos": int(len(clean["oos"])),
            **fit_counts,
        },
        "selection": selected,
        "metrics": metrics,
        "cpcv": cpcv,
        "certification": gates,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", default="all")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    p.add_argument("--run-id", default=None)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--holdout-months", type=int, default=2)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--coverages", default="0.02,0.05,0.10,0.15")
    p.add_argument("--min-rows", type=int, default=50_000)
    p.add_argument("--allow-api-error-boundary", action="store_true")
    p.add_argument("--seed-base", type=int, default=20260614)
    p.add_argument("--cpcv", choices=["none", "candidates", "all"], default="candidates")
    p.add_argument("--cpcv-seed-count", type=int, default=1)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    pairs = parse_pairs(args.pairs)
    coverages = [float(x) for x in args.coverages.split(",") if x.strip()]
    data_dir = deriv_train.ensure_deriv_data_path(Path(args.data_dir), purpose="Deriv candle input directory")
    model_dir = deriv_train.ensure_deriv_data_path(Path(args.model_dir), purpose="Deriv model output directory")
    result_dir = Path(args.result_dir).expanduser().resolve()
    if result_dir != (REPO_ROOT / "results" / "json").resolve():
        raise SystemExit("result-dir must be results/json for tracked evidence")
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.allow_api_error_boundary:
        raise SystemExit("--allow-api-error-boundary is disabled for book_deriv replication")

    all_m1 = load_all_m1(data_dir, PAIRS, args.min_rows, False)
    common_latest = min(pd.Timestamp(frame.index.max()).tz_convert("UTC") for frame in all_m1.values())
    all_m1 = {pair: frame.loc[frame.index <= common_latest].copy() for pair, frame in all_m1.items()}
    closes = pd.DataFrame({pair: all_m1[pair]["close"] for pair in PAIRS}).dropna()
    books = load_target_books()
    bases = {}
    for pair in pairs:
        base = deriv_train.build_deriv_dataset(
            data_dir,
            pair,
            args.horizon,
            min_rows=args.min_rows,
            allow_api_error_boundary=False,
        )
        base = deriv_train.DerivDataset(
            data=base.data.loc[base.data.index <= common_latest].copy(),
            latest_raw_ts=common_latest,
            metadata=base.metadata,
        )
        bases[pair] = base

    first_pass = []
    for i, pair in enumerate(pairs):
        book = books[pair]
        result = replicate_pair(
            pair=pair,
            book=book,
            base=bases[pair],
            closes=closes,
            model_dir=model_dir,
            run_id=run_id,
            horizon=args.horizon,
            holdout_months=args.holdout_months,
            val_fraction=args.val_fraction,
            coverages=coverages,
            seed_base=args.seed_base + i * 1000,
            run_cpcv=False,
            cpcv_seed_count=args.cpcv_seed_count,
        )
        first_pass.append(result)
        if result["status"] == "OK":
            oos = result["metrics"]["oos_latest_two_months"]
            print(
                " ".join(
                    [
                        "BOOK_DERIV",
                        pair,
                        f"book={book.book_id}",
                        f"verdict_pre={result['verdict']}",
                        f"oos_no_acc={oos['selected_nonoverlap_accuracy_ties_lose']}",
                        f"oos_no_n={oos['selected_nonoverlap_rows']}",
                    ]
                )
            )
        else:
            print(f"BOOK_DERIV {pair} status={result['status']} reason={result.get('reason')}")

    cpcv_pairs: set[str]
    if args.cpcv == "all":
        cpcv_pairs = {r["pair"] for r in first_pass if r["status"] == "OK"}
    elif args.cpcv == "candidates":
        cpcv_pairs = {
            r["pair"]
            for r in first_pass
            if r["status"] == "OK"
            and (r["metrics"]["oos_latest_two_months"]["selected_nonoverlap_accuracy_ties_lose"] or 0.0) >= BREAKEVEN
            and r["metrics"]["oos_latest_two_months"]["selected_nonoverlap_rows"] >= 50
        }
    else:
        cpcv_pairs = set()

    final = []
    for i, result in enumerate(first_pass):
        if result["pair"] not in cpcv_pairs or result["status"] != "OK":
            final.append(result)
            continue
        pair = result["pair"]
        print(f"BOOK_DERIV_CPCV {pair} start")
        cpcv_result = replicate_pair(
            pair=pair,
            book=books[pair],
            base=bases[pair],
            closes=closes,
            model_dir=model_dir,
            run_id=f"{run_id}_cpcv",
            horizon=args.horizon,
            holdout_months=args.holdout_months,
            val_fraction=args.val_fraction,
            coverages=coverages,
            seed_base=args.seed_base + 50_000 + i * 1000,
            run_cpcv=True,
            cpcv_seed_count=args.cpcv_seed_count,
        )
        final.append(cpcv_result)
        cpcv = cpcv_result["cpcv"]
        print(
            " ".join(
                [
                    "BOOK_DERIV_CPCV",
                    pair,
                    f"verdict={cpcv_result['verdict']}",
                    f"p10={cpcv['p10_selected_nonoverlap_accuracy']}",
                    f"clear_rate={cpcv['clear_rate_breakeven']}",
                ]
            )
        )

    summary = {
        "schema": "book_deriv_replicate_result.v1",
        "created_at_utc": utc_now_iso(),
        "run_id": run_id,
        "status": "CERTIFICATION_ATTEMPT",
        "settlement_mode": SETTLEMENT_MODE,
        "eligible_for_certification": CERTIFICATION_ELIGIBLE,
        "certification_blocker": (
            "Deriv public candle closes are not Deriv tick-settlement labels; "
            "numeric gates are diagnostic until tick-level settlement labels are added."
        ),
        "breakeven": BREAKEVEN,
        "data_dir": str(data_dir.relative_to(REPO_ROOT)),
        "model_dir": str((model_dir / run_id).relative_to(REPO_ROOT)),
        "holdout_months": int(args.holdout_months),
        "horizon_minutes": int(args.horizon),
        "pairs": final,
        "common_latest_deriv_timestamp_utc": common_latest.isoformat(),
        "verdict": "NOT_CERTIFIED",
    }
    result_path = result_dir / f"book_deriv_replicate_{run_id}_result.json"
    result_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"result_json={result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
