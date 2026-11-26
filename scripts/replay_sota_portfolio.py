#!/usr/bin/env python3
"""Deterministic 72-scenario multi-book DEV portfolio sensitivity replay (spec §3.7).

Schema: `sota-portfolio-pareto-replay/v1`.
Sensitivity ONLY: does not confer execution authority for m60, Kelly sizing, or Decision 5.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import stat
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "sota-portfolio-pareto-replay/v1"
PATH_YEARS = 10_224_000 / 31_557_600
INITIAL_CASH = 1000.0
EXPECTED_TEST_GROUPS = [[i, j] for i in range(6) for j in range(i + 1, 6)]

# Ordered pinned books (priority: ens5_m30 -> stack05_m30 -> ens5_m15 -> ens5_m60)
BOOK_SPECS: tuple[dict[str, Any], ...] = (
    {
        "short_id": "ens5_m30",
        "book_id": "USDJPY.sota_ens5_m30_up.v1",
        "dossier_rel": "results/json/deriv_5gate_USDJPY_sota_ens5_m30_up_v1_result.json",
        "pinned_sha256": "182505e2155cc8d88bba5a653275daa8e407cc1fcd7db1a6d2170e5c995f6922",
        "horizon_s": 1800,
        "payout_ratio": 1.82,
        "role": "certified_incumbent",
    },
    {
        "short_id": "stack05_m30",
        "book_id": "USDJPY.sota_stack05_m30_up.v1",
        "dossier_rel": "results/json/deriv_5gate_USDJPY_sota_stack05_m30_up_v1_result.json",
        "pinned_sha256": None,  # Bound dynamically from manifest.evidence_binding.dossier_sha256
        "horizon_s": 1800,
        "payout_ratio": 1.82,
        "role": "certified_phase1_candidate",
    },
    {
        "short_id": "ens5_m15",
        "book_id": "USDJPY.sota_ens5_m15_up.v1",
        "dossier_rel": "results/json/deriv_5gate_USDJPY_sota_ens5_m15_up_v1_result.json",
        "pinned_sha256": "c7961663210ff7d9263c4ac9a709b302c5a96f063af7626c978df935ae5127cb",
        "horizon_s": 900,
        "payout_ratio": 1.776447,
        "role": "certified_incumbent",
    },
    {
        "short_id": "ens5_m60",
        "book_id": "USDJPY.sota_ens5_m60_up.v1",
        "dossier_rel": "results/json/deriv_5gate_USDJPY_sota_ens5_m60_up_v1_result.json",
        "pinned_sha256": "07ce706bb4a12d689d3da9ee4dcc50e3a7e130632c1aeb597f011591006745b0",
        "horizon_s": 3600,
        "payout_ratio": 1.82,
        "role": "exploratory_m60_sensitivity",
    },
)

PAYOUT_POLICY: tuple[tuple[str, float], ...] = tuple((b["short_id"], float(b["payout_ratio"])) for b in BOOK_SPECS)

PORTFOLIOS: dict[str, tuple[str, ...]] = {
    "1b_ens5_m30": ("ens5_m30",),
    "2b_incumbent": ("ens5_m30", "ens5_m15"),
    "3b_phase1": ("ens5_m30", "stack05_m30", "ens5_m15"),
    "3b_with_m60": ("ens5_m30", "stack05_m30", "ens5_m60"),
    "4b_with_m60": ("ens5_m30", "stack05_m30", "ens5_m15", "ens5_m60"),
}

# Exact 18 cells in canonical order (portfolio_cap, path_set)
CANONICAL_CELLS: tuple[tuple[str, int, str, str], ...] = (
    # 12 certified_phase1 cells
    ("1b_ens5_m30", 2, "all_15_exported_paths", "certified_phase1"),
    ("1b_ens5_m30", 2, "certified_3book_common_supported_paths", "certified_phase1"),
    ("1b_ens5_m30", 2, "m60_common_supported_paths", "certified_phase1"),
    ("2b_incumbent", 2, "all_15_exported_paths", "certified_phase1"),
    ("2b_incumbent", 2, "certified_3book_common_supported_paths", "certified_phase1"),
    ("2b_incumbent", 2, "m60_common_supported_paths", "certified_phase1"),
    ("3b_phase1", 2, "all_15_exported_paths", "certified_phase1"),
    ("3b_phase1", 2, "certified_3book_common_supported_paths", "certified_phase1"),
    ("3b_phase1", 2, "m60_common_supported_paths", "certified_phase1"),
    ("3b_phase1", 3, "all_15_exported_paths", "certified_phase1"),
    ("3b_phase1", 3, "certified_3book_common_supported_paths", "certified_phase1"),
    ("3b_phase1", 3, "m60_common_supported_paths", "certified_phase1"),
    # 6 exploratory_m60_sensitivity cells
    ("3b_with_m60", 3, "all_15_exported_paths", "exploratory_m60_sensitivity"),
    ("3b_with_m60", 3, "m60_common_supported_paths", "exploratory_m60_sensitivity"),
    ("4b_with_m60", 3, "all_15_exported_paths", "exploratory_m60_sensitivity"),
    ("4b_with_m60", 3, "m60_common_supported_paths", "exploratory_m60_sensitivity"),
    ("4b_with_m60", 4, "all_15_exported_paths", "exploratory_m60_sensitivity"),
    ("4b_with_m60", 4, "m60_common_supported_paths", "exploratory_m60_sensitivity"),
)

SIZING_CONFIGS: tuple[dict[str, Any], ...] = (
    {"name": "flat_1usd", "sizing_family": "flat", "kelly_mult": None},
    {"name": "kelly_0.15", "sizing_family": "kelly", "kelly_mult": 0.15},
    {"name": "kelly_0.20", "sizing_family": "kelly", "kelly_mult": 0.20},
    {"name": "kelly_0.25", "sizing_family": "kelly", "kelly_mult": 0.25},
)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def simulate_path_events(
    events: list[tuple[int, int, int, int, float]],
    *,
    max_open_cap: int,
    sizing_family: str,
    book_kelly_fractions: dict[int, float] | None = None,
    initial_cash: float = INITIAL_CASH,
    path_years: float = PATH_YEARS,
) -> dict[str, Any]:
    """Simulate one CPCV path in exact (decision_ts, book_priority_idx) order (spec §3.7 + B8).

    Each event is `(decision_ts, book_priority_idx, horizon_s, win_01, payout_ratio)`.
    """
    ordered = sorted(events, key=lambda e: (int(e[0]), int(e[1])))
    cash = float(initial_cash)
    peak = float(initial_cash)
    max_dd = 0.0
    # Min-heap of (t_exp, seq, stake, win_01, payout_ratio)
    open_heap: list[tuple[int, int, float, int, float]] = []
    open_stake_sum = 0.0
    seq = 0
    trades = 0
    wins = 0
    skipped_cap = 0
    skipped_cash = 0

    def _mark_equity() -> float:
        nonlocal peak, max_dd
        eq = cash + open_stake_sum
        if eq > peak:
            peak = eq
        dd = ((peak - eq) / peak) if peak > 0.0 else 1.0
        if dd > max_dd:
            max_dd = dd
        return eq

    def _settle_up_to(t_limit: int | None) -> None:
        nonlocal cash, open_stake_sum
        while open_heap and (t_limit is None or open_heap[0][0] <= t_limit):
            grp_t = open_heap[0][0]
            credit = 0.0
            while open_heap and open_heap[0][0] == grp_t:
                _, _, st_i, win_i, r_i = heapq.heappop(open_heap)
                open_stake_sum -= st_i
                credit += st_i * r_i * float(win_i)
            # Guard tiny float drift when heap empties
            if not open_heap:
                open_stake_sum = 0.0
            cash += credit
            _mark_equity()

    for dec_ts, prio_idx, hor_s, win_01, r_b in ordered:
        # 1. Settle all distinct t_exp <= decision_ts ascending BEFORE admitting same-second entries
        _settle_up_to(int(dec_ts))
        # 2. Enforce max_open_cap
        if len(open_heap) >= int(max_open_cap):
            skipped_cap += 1
            continue
        # 3. Compute stake
        eq_now = cash + open_stake_sum
        if sizing_family == "flat":
            st = 1.0 if cash >= 1.0 else 0.0
        elif sizing_family == "kelly":
            f_b = float((book_kelly_fractions or {}).get(int(prio_idx), 0.0))
            st = min(cash, eq_now * f_b) if (cash > 0.0 and eq_now > 0.0 and f_b > 0.0) else 0.0
        else:
            raise ValueError(f"unsupported sizing_family {sizing_family!r}")
        if not math.isfinite(st) or st <= 0.0:
            skipped_cash += 1
            continue
        # 4. Debit cash -= st at admission, push open position, update equity/peak/max_dd
        cash -= st
        open_stake_sum += st
        seq += 1
        heapq.heappush(open_heap, (int(dec_ts) + int(hor_s), seq, st, int(win_01), float(r_b)))
        trades += 1
        wins += int(win_01)
        _mark_equity()

    # 5. Drain remaining open positions by distinct ascending t_exp
    _settle_up_to(None)

    final_cash = float(cash)
    if trades == 0:
        return {
            "trades": 0,
            "wins": 0,
            "skipped_cap": skipped_cap,
            "skipped_cash": skipped_cash,
            "final_cash": final_cash,
            "ruined": False,
            "path_max_dd": 0.0,
            "path_log_growth": 0.0,
        }
    ruined = bool(final_cash <= 0.0)
    if ruined:
        return {
            "trades": trades,
            "wins": wins,
            "skipped_cap": skipped_cap,
            "skipped_cash": skipped_cash,
            "final_cash": final_cash,
            "ruined": True,
            "path_max_dd": 1.0,
            "path_log_growth": None,
        }
    return {
        "trades": trades,
        "wins": wins,
        "skipped_cap": skipped_cap,
        "skipped_cash": skipped_cash,
        "final_cash": final_cash,
        "ruined": False,
        "path_max_dd": float(max_dd),
        "path_log_growth": float(math.log(final_cash / float(initial_cash)) / float(path_years)),
    }


def reduce_scenario_paths(path_ids: list[str], path_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Exact scenario reduction per spec §3.7 + B8 (linear p10_g / p90_dd, any_ruined=False on empty)."""
    if len(path_ids) == 0:
        return {
            "status": "empty_intersection",
            "any_ruined": False,
            "n_paths": 0,
            "p10_g": None,
            "mean_g": None,
            "p90_dd": None,
            "worst_path_max_dd": None,
            "p10_g_over_p90_dd": None,
            "pareto_efficient": False,
        }
    dds = [float(r["path_max_dd"]) for r in path_results]
    p90_dd = float(np.percentile(dds, 90, method="linear"))
    worst_dd = float(max(dds))
    if any(bool(r["ruined"]) for r in path_results):
        return {
            "status": "ruined",
            "any_ruined": True,
            "n_paths": len(path_ids),
            "p10_g": None,
            "mean_g": None,
            "p90_dd": p90_dd,
            "worst_path_max_dd": worst_dd,
            "p10_g_over_p90_dd": None,
            "pareto_efficient": False,
        }
    gs = [float(r["path_log_growth"]) for r in path_results]
    p10_g = float(np.percentile(gs, 10, method="linear"))
    mean_g = float(np.mean(gs))
    ratio = None if p90_dd == 0.0 else float(p10_g / p90_dd)
    return {
        "status": "ok",
        "any_ruined": False,
        "n_paths": len(path_ids),
        "p10_g": p10_g,
        "mean_g": mean_g,
        "p90_dd": p90_dd,
        "worst_path_max_dd": worst_dd,
        "p10_g_over_p90_dd": ratio,
        "pareto_efficient": False,
    }


def compute_pareto_flags(scenarios: list[dict[str, Any]]) -> list[bool]:
    """Compute Pareto efficiency maximizing p10_g and minimizing p90_dd within exact comparison groups (B8)."""
    flags = [False] * len(scenarios)
    groups: dict[tuple[Any, ...], list[int]] = {}
    for idx, sc in enumerate(scenarios):
        if sc.get("status") != "ok":
            continue
        gkey = (
            tuple(sc["path_ids"]),
            tuple((str(k), float(v)) for k, v in sc["payout_policy"]),
            str(sc["sizing_family"]),
            str(sc["scope"]),
        )
        groups.setdefault(gkey, []).append(idx)

    for idxs in groups.values():
        for i in idxs:
            si = scenarios[i]
            pi, di = float(si["p10_g"]), float(si["p90_dd"])
            dominated = False
            for j in idxs:
                if i == j:
                    continue
                sj = scenarios[j]
                pj, dj = float(sj["p10_g"]), float(sj["p90_dd"])
                if pj >= pi and dj <= di and (pj > pi or dj < di):
                    dominated = True
                    break
            flags[i] = not dominated
    return flags


ALLOWED_STACK05_DOSSIER_RELS: frozenset[str] = frozenset(
    {
        "results/json/deriv_5gate_USDJPY_sota_stack05_m30_up_v1_result.json",
        "results/json/deriv_5gate_USDJPY_sota_stack05_m30_up_v2_result.json",
    }
)


def _read_repo_file_nofollow(repo_root: Path, rel_path: str) -> bytes:
    """Read a repo-relative regular file with component-by-component O_NOFOLLOW."""
    p_rel = Path(rel_path)
    if p_rel.is_absolute() or not p_rel.parts or any(part in ("", ".", "..") for part in p_rel.parts):
        raise RuntimeError(f"refusing non-canonical replay relative path: {rel_path!r}")
    root_fd = os.open(str(repo_root), os.O_RDONLY | os.O_DIRECTORY)
    cur_fd = root_fd
    try:
        for part in p_rel.parts[:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=cur_fd)
            if cur_fd != root_fd:
                os.close(cur_fd)
            cur_fd = nxt
        leaf_fd = os.open(p_rel.parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=cur_fd)
        try:
            st = os.fstat(leaf_fd)
            if not stat.S_ISREG(st.st_mode):
                raise RuntimeError(f"replay input is not a regular file: {rel_path!r}")
            with os.fdopen(leaf_fd, "rb") as fh:
                leaf_fd = -1
                return fh.read()
        finally:
            if leaf_fd >= 0:
                os.close(leaf_fd)
    except OSError as exc:
        raise RuntimeError(f"failed to safely read replay input {rel_path!r}: {exc}") from exc
    finally:
        if cur_fd != root_fd:
            os.close(cur_fd)
        os.close(root_fd)


def _load_and_validate_dossiers(repo_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, str]]:
    """Load & authenticate all 4 book dossiers (stack05 pinned to manifest.evidence_binding.dossier_sha256)."""
    # Read stack05 expected dossier SHA-256 from its frozen manifest
    stack_manifest_rel = "books/USDJPY.sota_stack05_m30_up.v1/USDJPY.sota_stack05_m30_up.v1.manifest.json"
    stack_manifest_raw = _read_repo_file_nofollow(repo_root, stack_manifest_rel)
    stack_manifest = json.loads(stack_manifest_raw.decode("utf-8"))
    stack_expected_sha = str((stack_manifest.get("evidence_binding") or {}).get("dossier_sha256") or "")
    stack_dossier_rel = str((stack_manifest.get("evidence_binding") or {}).get("dossier_path") or "")
    if len(stack_expected_sha) != 64:
        raise RuntimeError("stack05 manifest missing valid evidence_binding.dossier_sha256")
    if stack_dossier_rel and stack_dossier_rel not in ALLOWED_STACK05_DOSSIER_RELS:
        raise RuntimeError(f"refusing unapproved stack05 manifest dossier_path: {stack_dossier_rel!r}")

    dossiers: dict[str, dict[str, Any]] = {}
    shas: dict[str, str] = {}
    rels: dict[str, str] = {}
    for spec in BOOK_SPECS:
        sid = spec["short_id"]
        d_rel = stack_dossier_rel if (sid == "stack05_m30" and stack_dossier_rel) else str(spec["dossier_rel"])
        raw = _read_repo_file_nofollow(repo_root, d_rel)
        got_sha = _sha256_bytes(raw)
        exp_sha = stack_expected_sha if sid == "stack05_m30" else str(spec["pinned_sha256"])
        if got_sha != exp_sha:
            raise RuntimeError(f"dossier SHA-256 mismatch for {sid}: {got_sha} != {exp_sha}")
        dos = json.loads(raw.decode("utf-8"))
        g1 = dos.get("G1_cpcv") or {}
        paths = g1.get("paths") or []
        if len(paths) != 15:
            raise RuntimeError(f"{sid}: expected exact 15 exported paths, got {len(paths)}")
        got_groups = [p.get("test_groups") for p in paths]
        if got_groups != EXPECTED_TEST_GROUPS:
            raise RuntimeError(f"{sid}: exported test_groups do not match lexicographic 15-path order")
        for p in paths:
            n = int(p["n"])
            ts_l = p.get("trade_ts") or []
            w_l = p.get("trade_wins") or []
            if len(ts_l) != n or len(w_l) != n:
                raise RuntimeError(f"{sid}: trade_ts/trade_wins length mismatch vs n={n}")
        dossiers[sid] = dos
        shas[sid] = got_sha
        rels[sid] = d_rel
    return dossiers, shas, rels


def build_portfolio_pareto_report(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    dossiers, shas, rels = _load_and_validate_dossiers(repo_root)
    spec_by_short = {b["short_id"]: b for b in BOOK_SPECS}
    prio_by_short = {b["short_id"]: idx for idx, b in enumerate(BOOK_SPECS)}

    all_path_ids = [f"{a}_{b}" for a, b in EXPECTED_TEST_GROUPS]
    # Per-book path map & full authenticated G1 p10_wr
    book_paths: dict[str, dict[str, dict[str, Any]]] = {}
    book_p10_wr: dict[str, float] = {}
    supported_sets: dict[str, set[str]] = {}
    for b in BOOK_SPECS:
        sid = b["short_id"]
        g1 = dossiers[sid]["G1_cpcv"]
        book_p10_wr[sid] = float(g1["p10_wr"])
        pmap: dict[str, dict[str, Any]] = {}
        sup: set[str] = set()
        for p in g1["paths"]:
            ga, gb = p["test_groups"]
            pid = f"{ga}_{gb}"
            pmap[pid] = p
            if bool(p.get("supported")):
                sup.add(pid)
        book_paths[sid] = pmap
        supported_sets[sid] = sup

    path_set_map: dict[str, list[str]] = {
        "all_15_exported_paths": list(all_path_ids),
        "certified_3book_common_supported_paths": [
            pid for pid in all_path_ids if pid in (supported_sets["ens5_m30"] & supported_sets["stack05_m30"] & supported_sets["ens5_m15"])
        ],
        "m60_common_supported_paths": [
            pid
            for pid in all_path_ids
            if pid in (supported_sets["ens5_m30"] & supported_sets["stack05_m30"] & supported_sets["ens5_m15"] & supported_sets["ens5_m60"])
        ],
    }

    scenarios: list[dict[str, Any]] = []
    # Exact canonical order: (portfolio_cap, path_set, sizing) -> 18 x 4 = 72 scenarios
    for port_name, cap, ps_name, scope in CANONICAL_CELLS:
        p_books = PORTFOLIOS[port_name]
        pids = path_set_map[ps_name]
        for sz in SIZING_CONFIGS:
            sfam = str(sz["sizing_family"])
            kmult = sz["kelly_mult"]
            kelly_by_short: dict[str, float] = {}
            kelly_by_prio: dict[int, float] = {}
            if sfam == "kelly":
                assert kmult is not None
                for b_short in p_books:
                    r_b = float(spec_by_short[b_short]["payout_ratio"])
                    p10_b = book_p10_wr[b_short]
                    f_b = min(0.08, float(kmult) * max(0.0, (p10_b * r_b - 1.0) / (r_b - 1.0)))
                    kelly_by_short[b_short] = f_b
                    kelly_by_prio[prio_by_short[b_short]] = f_b

            path_results: list[dict[str, Any]] = []
            total_trades = 0
            total_wins = 0
            total_skip_cap = 0
            total_skip_cash = 0
            for pid in pids:
                events: list[tuple[int, int, int, int, float]] = []
                for b_short in p_books:
                    p_obj = book_paths[b_short][pid]
                    prio = prio_by_short[b_short]
                    hor_s = int(spec_by_short[b_short]["horizon_s"])
                    r_b = float(spec_by_short[b_short]["payout_ratio"])
                    for t_i, w_i in zip(p_obj["trade_ts"], p_obj["trade_wins"]):
                        events.append((int(t_i), prio, hor_s, int(w_i), r_b))
                pres = simulate_path_events(
                    events,
                    max_open_cap=cap,
                    sizing_family=sfam,
                    book_kelly_fractions=kelly_by_prio if sfam == "kelly" else None,
                    initial_cash=INITIAL_CASH,
                    path_years=PATH_YEARS,
                )
                path_results.append(pres)
                total_trades += pres["trades"]
                total_wins += pres["wins"]
                total_skip_cap += pres["skipped_cap"]
                total_skip_cash += pres["skipped_cash"]

            red = reduce_scenario_paths(pids, path_results)
            scenarios.append(
                {
                    "scenario_id": f"{port_name}__cap{cap}__{ps_name}__{sz['name']}",
                    "portfolio": port_name,
                    "books": [spec_by_short[s]["book_id"] for s in p_books],
                    "max_open_cap": cap,
                    "path_set": ps_name,
                    "path_ids": list(pids),
                    "scope": scope,
                    "sizing": sz["name"],
                    "sizing_family": sfam,
                    "kelly_multiplier": kmult,
                    "book_kelly_fractions": kelly_by_short,
                    "payout_policy": [[k, v] for k, v in PAYOUT_POLICY],
                    "total_trades": total_trades,
                    "total_wins": total_wins,
                    "total_skipped_cap": total_skip_cap,
                    "total_skipped_cash": total_skip_cash,
                    **red,
                }
            )

    assert len(scenarios) == 72
    pflags = compute_pareto_flags(scenarios)
    for sc, fl in zip(scenarios, pflags):
        sc["pareto_efficient"] = bool(fl)

    return {
        "schema": SCHEMA,
        "classification": "dev_portfolio_sensitivity_only_no_m60_or_kelly_execution_authority",
        "path_years": PATH_YEARS,
        "initial_cash": INITIAL_CASH,
        "payout_policy": [[k, v] for k, v in PAYOUT_POLICY],
        "inputs": [
            {
                "short_id": b["short_id"],
                "book_id": b["book_id"],
                "dossier_rel": rels[b["short_id"]],
                "dossier_sha256": shas[b["short_id"]],
                "horizon_s": b["horizon_s"],
                "payout_ratio": b["payout_ratio"],
                "role": b["role"],
                "g1_p10_wr": book_p10_wr[b["short_id"]],
                "supported_paths_count": len(supported_sets[b["short_id"]]),
            }
            for b in BOOK_SPECS
        ],
        "path_sets": path_set_map,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(REPO_ROOT))
    ap.add_argument("--out", default="results/json/deriv_4book_portfolio_pareto_result.json")
    ap.add_argument("--stdout", action="store_true", help="Print canonical JSON to stdout without writing any file")
    args = ap.parse_args(argv)
    rroot = Path(args.repo_root).resolve()
    report = build_portfolio_pareto_report(repo_root=rroot)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.stdout:
        print(payload, end="")
        return 0
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = rroot / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload, encoding="utf-8")
    print(f"WROTE {out_path} ({report['scenario_count']} scenarios)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
