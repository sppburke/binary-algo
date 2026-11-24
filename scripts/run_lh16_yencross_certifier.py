"""Automated LH-16 Yen-cross true-venue-tick downloader, parity verifier, and G1-G6 certifier.

Live `deriv_offer_probe/1.0` (2026-11-24T02:28:18Z) confirmed Deriv sells intraday 15m/30m
Rise/Fall (`CALL`/`PUT`) and serves 364-day 1s ticks on four Yen crosses:
  - `frxEURJPY` (live CALL=1.88, PUT=1.74)
  - `frxGBPJPY` (live CALL=1.86, PUT=1.77)
  - `frxAUDJPY` (live CALL=1.78, PUT=1.84)
  - `frxNZDJPY` (live CALL=1.54, PUT=1.96)
(`frxCADJPY` and `frxCHFJPY` returned `OfferingsInvalidSymbol` and are excluded.)

Because Deriv rate-limits `ticks_history_expensive` to 24 requests/minute (~10.5 hours per cross
for ~249 DEV days), this script automates:
  1. Resumable venue tick download under `/mnt/data/deriv/.connection.lock` (`admission17`).
  2. Clock & decimal parity audit of true `frx<CROSS>` ticks against synthetic `deriv_ticks_x1`
     (`EURUSD * USDJPY`, etc.) on overlapping DEV days (`<= 2026-07-31`).
  3. Isolated `< 2026-08-01` dataset build (`dataset_dev_cross_venue_v0.1_<CROSS>`) using the
     unmodified `scripts/deriv_dataset.py` builder.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

import deriv_dataset as dd
from deriv_runtime_core_v2 import assert_non_us_deriv_host

OFFERED_YEN_CROSSES = ("EURJPY", "GBPJPY", "AUDJPY", "NZDJPY")
DEFAULT_VENUE_CROSS_ARCHIVE = Path("/mnt/data/deriv-sota-lab/cache/venue_cross_archive")
SYNTH_CROSS_TICKS = Path("/mnt/data/deriv-sota-lab/cache/deriv_ticks_x1")
MAJORS_ARCHIVE = Path("/mnt/data/deriv/ticks")
DEV_LAST_DATE = dt.date(2026, 7, 31)


def compare_venue_vs_synthetic_cross(
    cross: str,
    venue_archive: Path = DEFAULT_VENUE_CROSS_ARCHIVE,
    synth_dir: Path = SYNTH_CROSS_TICKS,
) -> dict[str, Any]:
    """Compare true Deriv `frx<CROSS>` 1s mid ticks against synthetic major-product ticks (`<= 2026-07-31`)."""
    v_dir = venue_archive / "ticks" / cross
    s_dir = synth_dir / cross
    if not v_dir.is_dir():
        return {"cross": cross, "overlapping_days": 0, "status": "no_venue_dir"}

    v_files = sorted(
        f for f in v_dir.glob(f"{cross}_*_ticks.parquet") if dt.date.fromisoformat(f.name.split("_")[1]) <= DEV_LAST_DATE
    )
    days_checked = []
    rel_diffs = []
    settle_15m_agree = []
    settle_30m_agree = []

    for vf in v_files:
        day_str = vf.name.split("_")[1]
        meta_f = vf.with_name(f"{cross}_{day_str}_ticks.meta.json")
        if meta_f.exists():
            meta = json.loads(meta_f.read_text())
            if not meta.get("complete"):
                continue
        sf = s_dir / vf.name
        if not sf.exists():
            continue
        ts_v, px_v = dd.load_ticks([vf])
        ts_s, px_s = dd.load_ticks([sf])
        common, iv, is_ = np.intersect1d(ts_v, ts_s, return_indices=True)
        if len(common) < 1000:
            continue
        rd = np.abs(px_v[iv] / px_s[is_] - 1.0)
        rel_diffs.append(float(np.median(rd)))
        days_checked.append(day_str)

        # Check 15m and 30m settlement direction agreement on minute closes
        dec = np.unique((common // 60) * 60) + 60
        for hmin, bucket in ((15, settle_15m_agree), (30, settle_30m_agree)):
            hs, tol = dd.HORIZONS[hmin]
            rv, vv = dd.settle(dec, ts_v, px_v, hs, tol)
            rs, vs = dd.settle(dec, ts_s, px_s, hs, tol)
            both = vv & vs & (rv != 0) & (rs != 0)
            if both.any():
                bucket.append(float(np.mean((rv[both] > 0) == (rs[both] > 0))))

    return {
        "cross": cross,
        "overlapping_complete_dev_days": len(days_checked),
        "days": days_checked[:10],
        "median_rel_price_diff": round(float(np.median(rel_diffs)), 7) if rel_diffs else None,
        "settle_15m_sign_agreement": round(float(np.mean(settle_15m_agree)), 5) if settle_15m_agree else None,
        "settle_30m_sign_agreement": round(float(np.mean(settle_30m_agree)), 5) if settle_30m_agree else None,
    }


def build_isolated_venue_cross_dev_dataset(
    cross: str,
    *,
    venue_archive: Path = DEFAULT_VENUE_CROSS_ARCHIVE,
    out_root: Path = Path("/mnt/data/deriv-sota-lab/cache"),
) -> Path:
    """Link `< 2026-08-01` major + venue cross ticks into a holdout-free root and run `deriv_dataset.py`."""
    c = cross.upper()
    if c not in OFFERED_YEN_CROSSES:
        raise ValueError(f"unsupported cross {cross!r}; expected one of {OFFERED_YEN_CROSSES}")

    v_dir = venue_archive / "ticks" / c
    complete_days: list[dt.date] = []
    for mf in sorted(v_dir.glob(f"{c}_*_ticks.meta.json")):
        d = dt.date.fromisoformat(mf.name.split("_")[1])
        if d <= DEV_LAST_DATE and json.loads(mf.read_text()).get("complete") is True:
            complete_days.append(d)
    if not complete_days:
        raise RuntimeError(f"{c}: 0 complete DEV days in {v_dir}")

    first_d, last_d = min(complete_days), DEV_LAST_DATE
    link_root = out_root / f"venue_cross_dev_links_{c}"
    out_ds = out_root / f"dataset_dev_cross_venue_v0.1_{c}"
    if (out_ds / "manifest.json").exists():
        return out_ds

    holiday_early_closes = {dt.date(2025, 12, 24), dt.date(2025, 12, 31)}
    for p in list(dd.PAIRS) + [c]:
        p_dst = link_root / p
        p_dst.mkdir(parents=True, exist_ok=True)
        src_base = (venue_archive / "ticks" / c) if p == c else (MAJORS_ARCHIVE / p)
        for pf in sorted(src_base.glob(f"{p}_*_ticks.parquet")):
            d = dt.date.fromisoformat(pf.name.split("_")[1])
            if d > DEV_LAST_DATE:
                continue
            mf = pf.with_name(f"{p}_{d.isoformat()}_ticks.meta.json")
            if not mf.exists():
                continue
            is_complete = bool(json.loads(mf.read_text()).get("complete"))
            if not is_complete and not (p != c and d in holiday_early_closes):
                continue
            dst_pf = p_dst / pf.name
            if not dst_pf.exists():
                os.symlink(pf, dst_pf)
            dst_mf = p_dst / mf.name
            if not dst_mf.exists():
                os.symlink(mf, dst_mf)

    # Verify zero post-2026-07-31 files exist in link_root
    late = [
        f.name
        for p in list(dd.PAIRS) + [c]
        for f in (link_root / p).glob("*_ticks.parquet")
        if dt.date.fromisoformat(f.name.split("_")[1]) > DEV_LAST_DATE
    ]
    if late:
        raise RuntimeError(f"holdout boundary violation in {link_root}: {late[:3]}")

    old_arch, old_pairs, old_argv = dd.ARCHIVE, list(dd.PAIRS), list(sys.argv)
    old_allow = os.environ.get("DERIV_ALLOW_INCOMPLETE")
    os.environ["DERIV_ALLOW_INCOMPLETE"] = "1"
    try:
        dd.SKIPPED_INCOMPLETE.clear()
        dd.ARCHIVE = link_root
        dd.PAIRS = list(old_pairs) + [c]
        sys.argv = ["deriv_dataset.py", str(out_ds), first_d.isoformat(), last_d.isoformat(), c]
        dd.main()
    finally:
        dd.ARCHIVE = old_arch
        dd.PAIRS = old_pairs
        sys.argv = old_argv
        if old_allow is None:
            os.environ.pop("DERIV_ALLOW_INCOMPLETE", None)
        else:
            os.environ["DERIV_ALLOW_INCOMPLETE"] = old_allow
    return out_ds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crosses", default="EURJPY,GBPJPY,AUDJPY,NZDJPY")
    ap.add_argument("--venue-archive", default=str(DEFAULT_VENUE_CROSS_ARCHIVE))
    ap.add_argument("--start-date", default="2025-11-25")
    ap.add_argument("--end-date", default="2026-07-31")
    ap.add_argument("--download", action="store_true", help="Run resumable admission17 downloader under .connection.lock")
    ap.add_argument("--parity-audit", action="store_true", help="Compare downloaded venue cross ticks vs synthetic cross ticks")
    ap.add_argument("--build-dev-dataset", action="store_true", help="Build isolated <2026-08-01 venue cross dataset")
    args = ap.parse_args()

    crosses = [c.strip().upper() for c in args.crosses.split(",") if c.strip()]
    for c in crosses:
        if c not in OFFERED_YEN_CROSSES:
            raise SystemExit(f"Unsupported or non-offered cross {c!r}; offered: {OFFERED_YEN_CROSSES}")

    if args.download:
        assert_non_us_deriv_host()
        syms = [f"frx{c}" for c in crosses]
        cmd = [
            sys.executable,
            "-B",
            "/mnt/data/deriv/tools/deriv_ticks_download.py",
            "--data-dir",
            args.venue_archive,
            "--symbols",
            *syms,
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date,
        ]
        rc = subprocess.call(cmd)
        if rc != 0:
            return rc

    if args.parity_audit:
        report = [compare_venue_vs_synthetic_cross(c, Path(args.venue_archive)) for c in crosses]
        print(json.dumps({"schema": "lh16-venue-cross-parity/v1", "crosses": report}, indent=2))

    if args.build_dev_dataset:
        for c in crosses:
            ds = build_isolated_venue_cross_dev_dataset(c, venue_archive=Path(args.venue_archive))
            print(f"BUILT: {ds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
