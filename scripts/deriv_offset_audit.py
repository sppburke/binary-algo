"""Shifted-offset offline audit (issue #4 Phase 5) — pre-registered protocol.

Falsifier-FIRST: `--write-falsifier` records the pre-registered constants to
`deriv_offset_audit_falsifier.json` BEFORE any outcome is computed; the audit
run refuses to start unless that file already exists. The audited stream is
de-overlapped BY CONSTRUCTION: chronological first-come-first-served through
the per-pair 15-minute cooldown across offsets and sides (the
`nonoverlap_chrono` discipline; precedent `scripts/nzdusd_15m_ticksettle.py`).
`max_open` / `max_trades_day` / KILL / lock are runtime risk budgets — not
simulated, per the issue-#4 r3 protocol.

Settlement is deriv-faithful, computed directly from ticks per offset:
entry = first tick >= signal_close + 1s, exit = last tick <= entry + 900s,
mid-to-mid, ties LOSE. No wall-clock haircut is applied anywhere in this
lane. CI95 via `min1_production.boot` on the de-overlapped win vector,
per (pair, side); enable bar = CI95-lower > max(0.541, quote median bar),
where the quote median bar is the MAXIMUM of per-source medians of
live_breakeven over quote_snapshot rows (executor rows carry no `source`
field; the deriv_quote_workers --audit sampler stamps `quote_audit`), each
source admitted only with >= 3 NY sessions of lane coverage — no admitted
source DEFERS the lane (see the FALSIFIER constant, amended pre-outcome
per issue #5). Per-offset breakdowns are diagnostic only — never
per-offset enablement.

Feature substrate: shifted 60s bars (deriv_market_stream.shifted_bar) at
offset k, translated to the wall grid (index = close - k) so
pipeline.build_features computes the certified feature cascade unchanged;
decision close = translated_index + 60s + k. SCOPE (v1): own-pair books
(USDJPY/USDCAD/AUDUSD/NZDUSD); the xpair batch join (USDCHF/GBPUSD) fails
loudly and is resolved in the strategy-eval audit run.

The full audit RUN produces measured numbers -> it is governed by the
strategy-eval discipline and runs as its own session. This module ships the
machinery + a deterministic smoke.

Run from repo root:
    ~/binary-algo-venv/bin/python scripts/deriv_offset_audit.py --write-falsifier
    ~/binary-algo-venv/bin/python scripts/deriv_offset_audit.py --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deriv_market_stream import shifted_bar
from deriv_runtime_core import after_last_start_cutoff, is_ny_session

FALSIFIER_PATH = Path("deriv_offset_audit_falsifier.json")
RESULT_PATH = Path("deriv_offset_audit_result.json")
OFFSETS = tuple(range(0, 60, 5))
OWN_PAIR_BOOKS = ("USDJPY", "USDCAD", "AUDUSD", "NZDUSD")
XPAIR_BOOKS = ("USDCHF", "GBPUSD")
HORIZON_S = 900
ENTRY_LAG_S = 1
ENTRY_TOL_S = 30.0  # nzdusd_15m_ticksettle.py precedent: TOL=30s at a 900s horizon
# Cooldown must exceed horizon + entry tolerance: entry can land up to
# close+31s, so 900s close-to-close spacing could overlap settlement windows
# by ~30s at the tolerance edge, violating boot()'s independence assumption.
COOLDOWN_S = HORIZON_S + int(ENTRY_TOL_S) + 1
CERT_BREAKEVEN = 0.541
N_MIN = 400

FALSIFIER = {
    "falsifier": "issue#4 Phase 5 shifted-offset audit — pre-registered before any outcome data is examined",
    "offsets_s": list(OFFSETS),
    # Amended pre-outcome 2026-07-06 (issue #4/#6): last buy start corrected
    # 16:44:59 -> 16:34:59 NY to MATCH live. deoverlap_stream now calls the shared
    # deriv_runtime_core.after_last_start_cutoff (reject >= 16:35:00 NY), the
    # probe-g-pinned cutoff (last accepted 16:34:40, first rejection 16:35:10;
    # commit ba0a3b5). The old literal 16:45:00 admitted ~10 min/day of near-close
    # trades the live runtime rejects, overstating realizable edge. Legitimate
    # pre-outcome (no audit result examined); it TIGHTENS the tradeable window
    # (fewer trades) — fail-closed, consistent with the MAX-rule discipline.
    "session": "NY 08:00-17:00 America/New_York DST-correct; last buy start 16:34:59 NY (shared deriv_runtime_core.after_last_start_cutoff, reject >= 16:35:00 NY)",
    "trade_stream": (
        "chronological first-come-first-served through the per-pair cooldown across offsets and sides "
        f"(nonoverlap_chrono discipline; spacing {COOLDOWN_S}s = horizon 900s + entry tolerance 30s + 1s, "
        "so settlement windows are strictly non-overlapping even at the tolerance edge); "
        "max_open/max_trades_day/KILL/lock are runtime risk budgets, NOT simulated — they bind live trades "
        "when configured (defaults are unconstrained on demo per the r4 amendment)"
    ),
    "settlement": (
        "entry = first tick >= signal_close + 1s (tol 30s, else invalid); exit = last tick <= entry + 900s "
        "(exit must lie within 30s of entry+900s, else invalid — fail-closed both ends); mid-to-mid; ties LOSE"
    ),
    "haircuts": "no wall-clock haircut applied anywhere in this lane",
    "n_min_deoverlapped_per_pair_side": N_MIN,
    "ci": "min1_production.boot nb=5000 CI95 on the de-overlapped win vector, per (pair, side)",
    # Amended pre-outcome 2026-07-02 (issue #5): the old executor timer is
    # replaced by the deriv_quote_workers --audit sampler as the quote_snapshot
    # source. Substantive amendment (evidence-source change), legitimate solely
    # because no outcome has been examined; written fail-closed (MAX rule —
    # a lower median would WEAKEN the bar that gates real demo buys).
    "enable_bar": (
        "CI95-lower > max(0.541, quote median bar for the (pair, side)); quote median bar = the MAXIMUM of "
        "per-source medians of live_breakeven over quote_snapshot rows, where source is distinguished by the "
        "`source` field (absent ⇒ production executor; 'quote_audit' ⇒ the deriv_quote_workers --audit "
        "sampler) and a source is admitted for a lane only with >= 3 NY sessions of coverage for that lane; "
        "where both sources are admitted the per-lane median deltas are recorded in the audit result; no "
        "admitted source ⇒ sparse quote coverage DEFERS the lane. The audit-run median loader must glob "
        "rotated files too (`quote_audit/*.jsonl*` including `.1`/`.gz`; rows carry timestamp_utc — "
        "filenames are not authoritative)."
    ),
    "per_offset_breakdowns": "diagnostic only — never per-offset enablement",
    "failure": "lane stays no-buy permanently absent new evidence",
    "scope_v1": "own-pair books only; xpair batch join resolved in the strategy-eval run",
}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_falsifier(path: Path) -> None:
    if path.exists():
        print(f"falsifier already registered: {path}")
        return
    doc = {**FALSIFIER, "registered_utc": now_utc_iso(), "outcomes_examined_at_registration": False}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    print(f"falsifier registered -> {path}")


def require_falsifier(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"REFUSED: falsifier not registered; run --write-falsifier first ({path})")
    return json.loads(path.read_text())


# ---------------------------------------------------------------- shifted features


def translated_m1(ticks: list[tuple[int, float, float, float]], offset_s: int) -> pd.DataFrame:
    """Shifted 60s bars at offset k, translated to the wall grid (index =
    close - 60 - k), so pipeline.build_features runs the certified cascade
    unchanged. Decision close (true) = index + 60s + k.

    Vectorized bulk path over searchsorted windows; semantics identical to
    `deriv_market_stream.shifted_bar` (the smoke asserts equality on sampled
    windows). Precondition: ticks deduped by epoch, ascending (the tick store
    writer and compactor guarantee this)."""
    if not ticks:
        return pd.DataFrame()
    ep = np.asarray([t[0] for t in ticks], dtype="int64")
    mid = np.asarray([t[1] for t in ticks], dtype="float64")
    first, last = int(ep[0]), int(ep[-1])
    start = ((first // 60) + 2) * 60 + offset_s
    ends = np.arange(start, last + 1, 60, dtype="int64")
    if not len(ends):
        return pd.DataFrame()
    i0 = np.searchsorted(ep, ends - 59, side="left")
    i1 = np.searchsorted(ep, ends, side="right")
    valid = (i1 - i0) == 60  # exactly 60 one-second ticks in (end-60, end]; never ffilled
    rows = []
    for end, a, b in zip(ends[valid], i0[valid], i1[valid]):
        w = mid[a:b]
        rows.append({"end": int(end), "open": float(w[0]), "high": float(w.max()),
                     "low": float(w.min()), "close": float(w[-1]), "volume": 60.0})
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame.index = pd.to_datetime(frame.pop("end") - 60 - offset_s, unit="s", utc=True)
    # pipeline.resample_1m contract: gap (minutes) to the previous available bar
    dt_min = frame.index.to_series().diff().dt.total_seconds().div(60.0)
    frame["gap_prev"] = dt_min.fillna(1.0).values
    return frame


def batch_signals(m1: pd.DataFrame, book: Any, offset_s: int, conf_thr: float | None = None) -> pd.DataFrame:
    """Score every completed row of the translated frame with the frozen book
    (seed-averaged boosters, same gate formula as LoadedBook.score)."""
    import pipeline

    X = pipeline.build_features(m1)
    cols = book.feature_cols
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise SystemExit(f"{book.book_id}: batch features missing columns {missing[:8]} "
                         f"(xpair books are out of v1 scope)")
    Xv = X[cols].astype("float64")
    finite = np.isfinite(Xv.values).all(axis=1)
    Xv = Xv.loc[finite]
    preds = np.mean([b.predict(Xv) for b in book.boosters], axis=0)
    conf = np.abs(preds - 0.5)
    thr = book.conf_thr if conf_thr is None else conf_thr
    out = pd.DataFrame({
        "proba": preds, "confidence": conf,
        "side": np.where(preds >= 0.5, "UP", "DOWN"),
    }, index=Xv.index)
    out = out[out["confidence"] >= thr]
    out["offset_id"] = offset_s
    out["close_epoch"] = (out.index.astype("int64") // 10**9) + 60 + offset_s
    return out


# ---------------------------------------------------------------- stream simulation + settlement


def deoverlap_stream(signals: pd.DataFrame) -> pd.DataFrame:
    """Chronological FCFS through the per-pair cooldown across offsets/sides,
    inside NY session and before the last-start cutoff."""
    taken = []
    last_close = -10**12
    for _, sig in signals.sort_values("close_epoch").iterrows():
        close = int(sig["close_epoch"])
        ts = datetime.fromtimestamp(close, timezone.utc)
        if not is_ny_session(ts):
            continue
        if after_last_start_cutoff(ts):  # shared runtime cutoff (reject >= 16:35:00 NY); replaces a stale literal 16:45:00 that desynced the backtest from live
            continue
        if close - last_close < COOLDOWN_S:
            continue
        last_close = close
        taken.append(sig)
    return pd.DataFrame(taken)


def settle(trades: pd.DataFrame, tick_epochs: np.ndarray, tick_mids: np.ndarray) -> pd.DataFrame:
    """Deriv-faithful tick settlement: +1s entry lag, 900s horizon, ties LOSE."""
    outcomes = []
    for _, tr in trades.iterrows():
        close = int(tr["close_epoch"])
        i = int(np.searchsorted(tick_epochs, close + ENTRY_LAG_S, side="left"))
        if i >= len(tick_epochs) or tick_epochs[i] - (close + ENTRY_LAG_S) > ENTRY_TOL_S:
            outcomes.append({**tr, "valid": False, "win": None})
            continue
        entry_epoch, entry = int(tick_epochs[i]), float(tick_mids[i])
        j = int(np.searchsorted(tick_epochs, entry_epoch + HORIZON_S, side="right")) - 1
        if j <= i or (entry_epoch + HORIZON_S) - tick_epochs[j] > ENTRY_TOL_S:
            outcomes.append({**tr, "valid": False, "win": None})
            continue
        exit_ = float(tick_mids[j])
        win = exit_ > entry if tr["side"] == "UP" else exit_ < entry  # tie (==) LOSES
        outcomes.append({**tr, "valid": True, "win": bool(win),
                         "entry_epoch": entry_epoch, "entry": entry, "exit": exit_})
    return pd.DataFrame(outcomes)


def per_lane_verdicts(settled: pd.DataFrame, breakeven_medians: dict[str, dict[str, float]],
                      pair: str) -> dict[str, Any]:
    from min1_production import boot

    lanes: dict[str, Any] = {}
    valid = settled[settled["valid"] == True]  # noqa: E712
    for side in ("UP", "DOWN"):
        wins = valid.loc[valid["side"] == side, "win"].astype(bool).to_numpy()
        n = len(wins)
        lane: dict[str, Any] = {"n_deoverlapped": n}
        if n:
            lo, hi = boot(wins.astype(float))
            med = (breakeven_medians.get(pair) or {}).get(side)
            bar = max(CERT_BREAKEVEN, med) if med is not None else None
            lane.update({
                "win_rate": round(float(wins.mean()), 4),
                "ci95": [round(float(lo), 4), round(float(hi), 4)],
                "quote_median_live_breakeven": med,
                "enable_bar": bar,
                "verdict": (
                    "deferred_n_below_min" if n < N_MIN
                    else "deferred_sparse_quote_coverage" if bar is None
                    else "passed" if lo > bar else "failed"
                ),
            })
        else:
            lane["verdict"] = "deferred_no_trades"
        lanes[side] = lane
    return lanes


# ---------------------------------------------------------------- smoke


def run_smoke() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    # 0. falsifier-first guard
    try:
        require_falsifier(Path("nonexistent_falsifier.json"))
        check("audit refuses without registered falsifier", False)
    except SystemExit:
        check("audit refuses without registered falsifier", True)

    # synthetic 1s tick series: seeded random walk, Tuesday 2026-06-30 (in-session hours included)
    rng = np.random.default_rng(7)
    t0 = 1781956800  # 2026-06-20 14:40 UTC; 10-day span (4h_x_48 features need ~8 days of warmup)
    n = 240 * 3600
    epochs = np.arange(t0, t0 + n)
    mids = 150.0 + np.cumsum(rng.normal(0, 0.002, n))
    ticks = [(int(e), float(m), float(m - 1e-4), float(m + 1e-4)) for e, m in zip(epochs, mids)]

    # 1. translated frame: shape + index translation invariant
    for k in (0, 25):
        m1 = translated_m1(ticks, k)
        idx_epochs = m1.index.astype("int64") // 10**9
        check(f"offset {k}: translated index on wall minutes", bool(len(m1)) and bool(((idx_epochs % 60) == 0).all()),
              f"rows={len(m1)}")
        first_close = int(idx_epochs[0]) + 60 + k
        bar0, _ = shifted_bar(ticks, first_close)
        check(f"offset {k}: first row equals shifted_bar at true close",
              bar0 is not None and abs(m1["close"].iloc[0] - bar0["close"]) < 1e-12)

    # 2. batch scoring == LoadedBook.score on the same row (parity of the audit scorer)
    from book_runtime import load_target_books
    book = load_target_books(["USDJPY"])["USDJPY"]
    m1 = translated_m1(ticks, 25)
    sigs = batch_signals(m1, book, 25, conf_thr=0.0)
    import pipeline
    X = pipeline.build_features(m1)
    probe_ts = sigs.index[len(sigs) // 2]
    single = book.score(X.loc[probe_ts, book.feature_cols])
    check("batch proba == single-row book.score proba",
          abs(float(sigs.loc[probe_ts, 'proba']) - single.proba) < 1e-9,
          f"{float(sigs.loc[probe_ts, 'proba']):.9f} vs {single.proba:.9f}")

    # 3. de-overlap: per-pair closes >= 931s apart (horizon + tolerance + 1),
    # session + cutoff respected
    stream = deoverlap_stream(sigs)
    gaps = np.diff(stream["close_epoch"].to_numpy())
    check(f"de-overlapped closes >= {COOLDOWN_S}s apart", bool(len(stream) > 1) and bool((gaps >= COOLDOWN_S).all()),
          f"n={len(stream)} min_gap={gaps.min() if len(gaps) else None}")

    # 4. settlement: ties LOSE (flat series), +1s entry honored
    flat_epochs = np.arange(t0, t0 + 4000)
    flat_mids = np.full(4000, 100.0)
    tr = pd.DataFrame([{"side": "UP", "close_epoch": t0 + 100, "proba": 0.6, "confidence": 0.1, "offset_id": 0},
                       {"side": "DOWN", "close_epoch": t0 + 1100, "proba": 0.4, "confidence": 0.1, "offset_id": 5}])
    st = settle(tr, flat_epochs, flat_mids)
    check("flat market: ties LOSE both sides", bool((st["valid"] == True).all()) and bool((st["win"] == False).all()))  # noqa: E712
    check("entry honors +1s lag", int(st["entry_epoch"].iloc[0]) == t0 + 101)

    # 5. settlement wins on a known move
    up_mids = np.concatenate([np.full(2000, 100.0), np.full(2000, 101.0)])
    # close 1500 -> entry 1501 (100.0), exit = last tick <= 2401 (101.0): the move lands inside the window
    st2 = settle(pd.DataFrame([{"side": "UP", "close_epoch": t0 + 1500, "proba": 0.6, "confidence": 0.1, "offset_id": 0}]),
                 flat_epochs, up_mids)
    check("known up-move settles as UP win", bool(st2["win"].iloc[0]) is True)

    # 6. lane verdicts: N gate + boot CI sanity
    fake = pd.DataFrame({"side": ["UP"] * 500, "valid": [True] * 500,
                         "win": [True] * 300 + [False] * 200})
    lanes = per_lane_verdicts(fake, {"USDJPY": {"UP": 0.55}}, "USDJPY")
    up = lanes["UP"]
    check("boot CI95 sane and verdict computed", up["n_deoverlapped"] == 500 and up["ci95"][0] < 0.6 < up["ci95"][1]
          and up["verdict"] in ("passed", "failed"), json.dumps(up))
    fake_small = fake.head(100)
    check("n < 400 defers", per_lane_verdicts(fake_small, {}, "USDJPY")["UP"]["verdict"] == "deferred_sparse_quote_coverage"
          or per_lane_verdicts(fake_small, {}, "USDJPY")["UP"]["verdict"] == "deferred_n_below_min")

    print(f"\n{'SMOKE ALL PASS' if not failures else f'SMOKE {len(failures)} FAILURES: {failures}'}")
    return 1 if failures else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--write-falsifier", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--falsifier-path", type=Path, default=FALSIFIER_PATH)
    args = p.parse_args()
    if args.write_falsifier:
        write_falsifier(args.falsifier_path)
        return 0
    if args.smoke:
        return run_smoke()
    p.error("the full audit run is a strategy-eval session; here pass --write-falsifier or --smoke")
    return 2


if __name__ == "__main__":
    sys.exit(main())
