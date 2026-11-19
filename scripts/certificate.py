"""The strict certificate of a CPCV result (test E-002), in one place for every report and every promotion.

    python -I -B certificate.py selftest                 # the kill rule's fixtures
    python -I -B certificate.py reread [RESULTS_DIR]     # every stored result read again through the function

`strict_certificate(result, side)` reads a result of the harness (`cpcv_deriv*`: `paths`, `bycov`, `validity`,
`verdict_goal`) and returns the certificate of the goal with its reasons, from the stored per-path figures only:
no probability, threshold, trade or path statistic is recomputed (the 10th percentile is numpy's, as the harness
takes it). The certificate: all 15 purged paths evaluated; at least 12 paths with 25 or more trades and a win rate
of 0.541 or more at the operating coverage (the run's `operating_cov`, by the key the harness writes for it, else the
one key numerically equal to it, else to the first of its `covs`); no counted path without a finite win rate; the 10th
percentile of the counted paths' win rates at 0.541 or more; a run whose validity record says it is valid (the moved
up-rate band and the per-path tripwire; a result without a validity record is not taken as valid); and the harness's
own strict field (`verdict_goal`), which must agree. A noncertifying marker overrides the status to "not certified"
with its reason, retaining the path statistics; the reread expects a false strict field for these results.
The reread compares that stored field with the reading of the
stored figures, never with a status the field itself decided. Path success
(`path_success`), validity (`valid`: True, False or None when there is no record) and the drift placebo
(`drift_placebo`, as the caller looked it up, e.g. `render_ledger.amended_verdict`; None when not looked up) are three
separate fields: the placebo never changes the status. Validity, the stored strict field and the legacy field are
read before the per-path figures, so a result of any status carries them. The status is one of:

    certified       every condition holds
    not certified   a condition fails; `reasons` names each
    incomplete      fewer than 15 paths were evaluated
    invalid         the validity record says the run failed its checks
    unreadable      the result has no per-path figures at the operating coverage

`side_text` (sweep ledgers, batch notes) and `cell_text` (results tables) are the one wording of a side: each names
invalidity whenever the record says so, and why a side whose path test passes is not certified; neither prints the
placebo or the legacy field. The harness's older field (`verdict`, which certifies from as few as one counted path)
stays in the result files and is printed only as "legacy".
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

# 0.5: enforce noncertifying markers independently, including on reread.
TOOL = "certificate/0.5"
BE = 0.541            # Rise/Fall breakeven, ties lose
MIN_TRADES = 25       # a path counts with this many trades or more
PATHS = 15            # purged CPCV paths of the goal
MIN_CLEAR = 12        # paths that must count and clear BE
SIDES = ("UP", "DOWN")
NOTES = (             # (words of a reason, long note, short note): why a side whose path test passes is not certified
    ("no strict field", "the result has no strict field", "no strict field"),
    ("stored strict field reads", "the stored strict field disagrees", "the strict field disagrees"),
    ("no validity record", "the result has no validity record", "no validity record"),
)


def _p10(values: list[float]) -> float:
    """The harness's own reading: numpy.percentile(values, 10)."""
    return float(np.percentile(values, 10)) if values else float("nan")


def _missing(x: object) -> bool:
    """A stored number that is not there: JSON null, or NaN."""
    return x is None or (isinstance(x, float) and math.isnan(x))


def _num(x: object) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def operating_coverage(result: dict) -> str | None:
    """The run's operating coverage as its key in the per-path figures. The coverage is `operating_cov` (the harness
    writes the first of `--covs` there), or, when the result has none, the first of `covs`. Its key: the key the
    harness writes for it (its stored text, or `f'{c}'` of the number); else the one key numerically equal to it, and
    None when no key or more than one is. A result with neither field: the first key (the harness writes the
    coverages in the order of `--covs`)."""
    keys = result.get("bycov")
    if not isinstance(keys, dict) or not keys:
        paths = result.get("paths")
        keys = paths[0].get("bycov") if isinstance(paths, list) and paths and isinstance(paths[0], dict) else None
    keys = list(keys) if isinstance(keys, dict) else []
    want = result.get("operating_cov")
    if want is None and isinstance(result.get("covs"), list) and result["covs"]:
        want = result["covs"][0]
    if want is None:
        return keys[0] if keys else None
    for exact in (str(want), f"{_num(want)}"):     # the key as the harness writes it (f'{c}' of the float) first
        if exact in keys:
            return exact
    match = [k for k in keys if _num(k) is not None and _num(k) == _num(want)]
    return match[0] if len(match) == 1 else None


def validity(result: dict) -> bool | None:
    """True or False as the result's validity record says, with booleans (`VALID`, else both component checks); None
    when it has no record that says it so (none, or a flag that is not a boolean)."""
    v = result.get("validity")
    if not isinstance(v, dict):
        return None
    if "VALID" in v:
        return v["VALID"] if isinstance(v["VALID"], bool) else None
    band, trip = v.get("in_session_up_rate_in_band"), v.get("paths_up_rate_in_tripwire")
    return band and trip if isinstance(band, bool) and isinstance(trip, bool) else None


def _noncertifying_reason(result: dict) -> str | None:
    """Independent light-reader equivalent of cpcv_deriv.adaptive_policy (owner decision 13).
    A recorded reason wins; every non-None ACI gamma, including zero and nonfinite values, marks a policy.
    """
    if result.get('noncertifying'):
        return str(result['noncertifying'])
    if str(result.get('harness', '')).startswith('cpcv_deriv_online'):
        return 'a continuing policy: the online learner updates on settled outcomes during each test path'
    for key in ('knobs', 'config'):
        conf = result.get(key)
        if isinstance(conf, dict) and conf.get('aci_gamma') is not None:
            return 'an adaptive threshold: ACI moves it with settled test outcomes'
    return None


def strict_certificate(result: dict, side: str, cov: str | None = None, be: float = BE,
                       placebo: str | None = None) -> dict:
    """The strict certificate of one side of a result, with its reasons (see the module's docstring)."""
    cov = cov if cov is not None else operating_coverage(result)
    legacy, goal = result.get("verdict"), result.get("verdict_goal")
    side_goal = goal.get(side) if isinstance(goal, dict) else None
    stored = side_goal.get("CERTIFIED") if isinstance(side_goal, dict) else None
    valid = validity(result)
    why = _noncertifying_reason(result)
    out = {"tool": TOOL, "side": side, "operating_coverage": cov, "status": "unreadable", "reasons": [],
           "path_success": False, "valid": valid, "drift_placebo": placebo, "paths_evaluated": 0,
           "paths_counted": 0, "paths_clear": 0, "nonfinite_counted": 0, "p10": float("nan"), "stored_strict": stored,
           "legacy": legacy.get(side) if isinstance(legacy, dict) else legacy}
    if why:
        out["status"] = "not certified"
        out["reasons"].append(f"never certified: {why}")
    try:
        rates = [(p["bycov"][cov][side][0], p["bycov"][cov][side][1]) for p in result["paths"]]
        counted = [float("nan") if _missing(w) else float(w) for n, w in rates if n is not None and n >= MIN_TRADES]
    except (TypeError, KeyError, IndexError, ValueError):
        out["reasons"].append(f"no per-path figures for {side} at the operating coverage {cov}")
        return out
    unknown = sum(1 for w in counted if not math.isfinite(w))
    clear = sum(1 for w in counted if math.isfinite(w) and w >= be)     # a rate that is not finite never clears
    p10 = _p10(counted)
    path_success = bool(len(rates) == PATHS and len(counted) >= MIN_CLEAR and clear >= MIN_CLEAR and not unknown
                        and math.isfinite(p10) and p10 >= be)
    out.update(path_success=path_success, paths_evaluated=len(rates), paths_counted=len(counted),
               paths_clear=clear, p10=p10, nonfinite_counted=unknown)
    reasons = out["reasons"]
    if len(rates) != PATHS:
        reasons.append(f"{len(rates)} of {PATHS} paths evaluated")
    if len(counted) < MIN_CLEAR:
        reasons.append(f"{len(counted)} paths with {MIN_TRADES} or more trades ({MIN_CLEAR} needed)")
    if clear < MIN_CLEAR:
        reasons.append(f"{clear} paths clear {be} ({MIN_CLEAR} needed)")
    if unknown:
        reasons.append(f"{unknown} counted paths without a finite win rate")
    if not counted:
        reasons.append("no counted path, no p10")
    elif not math.isfinite(p10):
        reasons.append("the p10 is not a finite number")
    elif p10 < be:
        reasons.append(f"p10 {p10:.4f} below {be}")
    if valid is False:
        reasons.append("invalid: the moved up-rate checks failed")
    elif valid is None:
        reasons.append("the result has no validity record")
    if stored is None:
        reasons.append("the result carries no strict field (verdict_goal)")
    elif not isinstance(stored, bool):
        reasons.append(f"the stored strict field reads {stored!r}, not a boolean")
    elif not why and stored != (path_success and valid is True):
        figures = path_success and valid is True
        reasons.append(f"the stored strict field reads {stored}, the stored figures {figures}")
    if why:
        out["status"] = "not certified"
    elif len(rates) != PATHS:
        out["status"] = "incomplete"
    elif valid is False:
        out["status"] = "invalid"
    elif reasons:
        out["status"] = "not certified"
    else:
        out["status"] = "certified"
    return out


def certified(result: dict, side: str, cov: str | None = None) -> bool:
    return strict_certificate(result, side, cov)["status"] == "certified"


def _unreadable(cert: dict) -> bool:
    """Figure availability survives the noncertifying status override."""
    return (cert["status"] == "unreadable"
            or any(why.startswith("no per-path figures for ") for why in cert["reasons"]))


def _blocking(cert: dict, long: bool) -> str | None:
    """The policy reason regardless of path success; otherwise why a passing path test is not certified."""
    if cert["status"] != "not certified":
        return None
    policy = next((why for why in cert["reasons"] if why.startswith("never certified: ")), None)
    if policy:
        return policy
    if not cert["path_success"]:
        return None
    for words, long_note, short_note in NOTES:
        if any(words in why for why in cert["reasons"]):
            return long_note if long else short_note
    return "; ".join(cert["reasons"]) or None


def side_text(cert: dict) -> str:
    """One side of a sweep ledger or a batch note: the strict status with its counts; never the legacy field, never
    the placebo."""
    counts = f'{cert["paths_counted"]} of {PATHS} paths counted, {cert["paths_clear"]} clear'
    if cert.get("nonfinite_counted"):
        counts += f'; {cert["nonfinite_counted"]} counted paths without a finite win rate'
    bad = "; invalid: moved up-rate checks" if cert["valid"] is False else ""
    if cert["status"] == "certified":
        return f"CERTIFIED ({counts})"
    if cert["status"] == "unreadable":
        return f"unreadable (no per-path figures{bad})"
    if cert["status"] == "incomplete":
        return f'incomplete ({cert["paths_evaluated"]} of {PATHS} paths evaluated; {counts}{bad})'
    if cert["status"] == "invalid":
        test = "path test passed" if cert["path_success"] else "path test not passed"
        return f"invalid (moved up-rate checks; {test}: {counts})"
    why = _blocking(cert, long=True)
    if _unreadable(cert):
        counts = "unreadable: no per-path figures"
    elif cert["paths_evaluated"] != PATHS:
        counts = f'incomplete: {cert["paths_evaluated"]} of {PATHS} paths evaluated; {counts}'
    return f"not certified ({counts}{bad}; {why})" if why else f"not certified ({counts}{bad})"


def cell_text(cert: dict) -> str:
    """One side of a results-table cell: CERTIFIED or no, the counted / evaluated paths and the clearing ones, then
    every flag the counts do not show (incomplete, invalid, a counted win rate that is not finite, and why a passing
    path test is not certified)."""
    unreadable = _unreadable(cert)
    flags = ((["incomplete"] if cert["status"] == "incomplete" else
              [f'incomplete: {cert["paths_evaluated"]} of {PATHS} paths evaluated']
              if not unreadable and cert["paths_evaluated"] != PATHS else [])
             + (["invalid"] if cert["valid"] is False else [])
             + (["a win rate not finite"] if cert.get("nonfinite_counted") else []))
    why = _blocking(cert, long=False)
    tail = "".join(f"; {f}" for f in flags + ([why] if why else []))
    if cert["status"] == "unreadable":
        return f"unreadable (no per-path figures{tail})"
    if unreadable:
        return f"no (unreadable: no per-path figures{tail})"
    head = "CERTIFIED" if cert["status"] == "certified" else "no"
    return f'{head} ({cert["paths_counted"]}/{cert["paths_evaluated"]}, {cert["paths_clear"]} clear{tail})'


def note(result: dict) -> str:
    """The batch note's verdict part: the strict certificate per side, then the legacy field, labelled."""
    certs = {s: strict_certificate(result, s) for s in SIDES}
    legacy = result.get("verdict")
    return ("strict " + " | ".join(f"{s} {side_text(certs[s])}" for s in SIDES)
            + f" | legacy verdict {json.dumps(legacy, sort_keys=True)}")


def same_number(a: object, b: object) -> bool:
    """Two stored numbers are the same: both missing (null or NaN), or both numbers with the same float bits."""
    if _missing(a) or _missing(b):
        return _missing(a) and _missing(b)
    return (isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool)
            and not isinstance(b, bool) and float(a).hex() == float(b).hex())


# ---------------------------------------------------------------- selftest (the kill rule of E-002)

def _fixture(clear_rates: list[float], low_rates: list[float] = (), thin: int = 0, zero: int = 0,
             valid: bool | None = True, with_goal: bool = True, trades: int = 60) -> dict:
    """A result whose UP side has paths at `clear_rates` and `low_rates` (with `trades` trades each), `thin` paths
    with 10 trades and `zero` paths without a trade; DOWN has no counted path. verdict_goal is filled as the harness
    fills it (cpcv_deriv.aggregate); `valid=None` leaves the validity record out."""
    paths = []
    for w in list(clear_rates) + list(low_rates):
        paths.append({"bycov": {"0.03": {"UP": [trades, w], "DOWN": [3, 0.6]}}})
    for _ in range(thin):
        paths.append({"bycov": {"0.03": {"UP": [10, 0.9], "DOWN": [3, 0.6]}}})
    for _ in range(zero):
        paths.append({"bycov": {"0.03": {"UP": [0, float("nan")], "DOWN": [0, float("nan")]}}})
    r = {"operating_cov": 0.03, "paths": paths, "verdict": {"UP": "CERTIFIED", "DOWN": "KILLED"}}
    if valid is not None:
        r["validity"] = {"VALID": valid}
    if with_goal:
        g = {}
        for s in SIDES:
            w = [p["bycov"]["0.03"][s][1] for p in paths if p["bycov"]["0.03"][s][0] >= MIN_TRADES]
            c = int(sum(x >= BE for x in w))
            p10 = float(np.percentile(w, 10)) if w else float("nan")
            g[s] = {"CERTIFIED": bool(bool(valid) and len(paths) == 15 and len(w) >= 12 and c >= 12 and p10 >= BE),
                    "paths_counted": len(w), "paths_clear": c, "p10": p10}
        r["verdict_goal"] = g
    return r


def _reread_result(name: str, r: dict) -> tuple[int, list[str], list[str], list[str]]:
    """One stored result that carries `verdict_goal`, read again: (sides equal to the stored field, sides that differ,
    sides without per-path figures or a stored side, legacy-only sides). The stored field must be a boolean equal to
    the reading of the stored figures (path test and validity), never to a status the field itself decided."""
    exact, differ, unreadable, legacy_only = 0, [], [], []
    for s in SIDES:
        c = strict_certificate(r, s)
        g = r["verdict_goal"].get(s)
        if _unreadable(c) or not isinstance(g, dict):
            unreadable.append(f"{name} {s}: {'; '.join(c['reasons']) or 'no stored strict side'}")
            continue
        figures = bool(not _noncertifying_reason(r) and c["path_success"] and c["valid"] is True)
        same = (isinstance(g.get("CERTIFIED"), bool) and g["CERTIFIED"] == figures
                and g.get("paths_counted") == c["paths_counted"] and g.get("paths_clear") == c["paths_clear"]
                and same_number(g.get("p10"), c["p10"]))
        if same:
            exact += 1
        else:
            differ.append(f"{name} {s}: stored {g}, function {c['status']} {c['paths_counted']} {c['paths_clear']} "
                          f"{c['p10']!r}")
        legacy = r.get("verdict", {}).get(s) if isinstance(r.get("verdict"), dict) else None
        if legacy == "CERTIFIED" and c["status"] != "certified":
            legacy_only.append(f"{name} {s}: {side_text(c)}")
    return exact, differ, unreadable, legacy_only


def selftest() -> int:
    failures = []

    def expect(name: str, holds: bool, shown: object = "") -> None:
        print(f"{'PASS' if holds else 'FAIL'}: {name}" + (f" - {shown}" if shown != "" else ""))
        if not holds:
            failures.append(name)

    never = [
        ("1 clearing path of 1 counted (14 with 10 trades)", _fixture([0.60], thin=14)),
        ("11 clearing paths of 15 counted", _fixture([0.60] * 11, [0.50] * 4)),
        ("12 clearing of 15 counted with p10 below 0.541", _fixture([0.60] * 12, [0.30] * 3)),
        ("all clearing with fewer than 15 paths evaluated (14)", _fixture([0.60] * 14)),
        ("12 clearing of 12 counted in an invalid run", _fixture([0.60] * 12, thin=3, valid=False)),
        ("a run without trades (15 paths, 0 trades each)", _fixture([], zero=15)),
        ("a result without the strict field", _fixture([0.60] * 15, with_goal=False)),
        ("a result without a validity record", _fixture([0.541] * 12, thin=3, valid=None)),
    ]
    for name, r in never:
        c = strict_certificate(r, "UP")
        expect(f"never certified: {name}", c["status"] != "certified", (c["status"], c["reasons"]))
    for name, r, counted, clear in (
            ("12 of 15 counted and clearing (3 with 10 trades), p10 at or above 0.541",
             _fixture([0.60] * 12, thin=3), 12, 12),
            ("15 of 15 clearing", _fixture([0.60] * 15), 15, 15),
            ("12 counted at exactly 0.541 (3 with 10 trades)", _fixture([0.541] * 12, thin=3), 12, 12)):
        c = strict_certificate(r, "UP")
        expect(f"certified: {name}", c["status"] == "certified" and (c["paths_counted"], c["paths_clear"]) ==
               (counted, clear), (c["status"], c["paths_counted"], c["paths_clear"], c["reasons"]))
    c = strict_certificate(_fixture([0.60] * 14), "UP")
    expect("fewer than 15 paths reads 'incomplete'", c["status"] == "incomplete", c["status"])
    c = strict_certificate(_fixture([0.60] * 12, thin=3, valid=False), "UP", placebo="SKILL BEYOND DRIFT TIMING")
    expect("an invalid run reads 'invalid', keeps path success, validity and the placebo as three fields, and the "
           "placebo does not certify",
           c["status"] == "invalid" and c["path_success"] is True and c["valid"] is False
           and c["drift_placebo"] == "SKILL BEYOND DRIFT TIMING", (c["status"], c["path_success"], c["valid"]))
    c = strict_certificate({"verdict": {"UP": "CERTIFIED"}}, "UP")
    expect("a result without per-path figures reads 'unreadable'", c["status"] == "unreadable", c["status"])
    r = _fixture([0.60] * 15)
    r["verdict_goal"]["UP"]["CERTIFIED"] = False
    c = strict_certificate(r, "UP")
    expect("a stored strict field that disagrees with the stored figures does not certify", c["status"] != "certified",
           c["reasons"])
    n = note(_fixture([0.60] * 11, [0.50] * 4))
    expect("the batch note leads with the strict reading and labels the older field legacy",
           n.startswith("strict UP not certified") and "legacy verdict" in n and "CERTIFIED (" not in n, n)
    # second review: validity is shown apart from path success in every wording
    c = strict_certificate(_fixture([0.60] * 11, [0.50] * 4, valid=False), "UP")
    expect("an invalid run whose path test fails reads invalid in the cell and the side text",
           c["status"] == "invalid" and cell_text(c) == "no (15/15, 11 clear; invalid)"
           and side_text(c).startswith("invalid ("), (cell_text(c), side_text(c)))
    c = strict_certificate(_fixture([0.60] * 14, valid=False), "UP")
    expect("an incomplete invalid run names both", c["status"] == "incomplete" and cell_text(c) ==
           "no (14/14, 14 clear; incomplete; invalid)" and "invalid" in side_text(c), (cell_text(c), side_text(c)))
    r = _fixture([0.60] * 12, thin=3)
    r["validity"] = {"in_session_up_rate_in_band": True, "paths_up_rate_in_tripwire": True}
    c1 = strict_certificate(r, "UP")
    r["validity"] = {"in_session_up_rate_in_band": True, "paths_up_rate_in_tripwire": False}
    c2 = strict_certificate(r, "UP")
    expect("a validity record of the two component checks is read (both true: valid; one false: invalid)",
           c1["status"] == "certified" and c2["status"] == "invalid", (c1["status"], c2["status"]))
    r = _fixture([0.60] * 12, thin=3, valid=False)
    del r["paths"][0]["bycov"]["0.03"]
    c = strict_certificate(r, "UP")
    expect("an unreadable result keeps its validity and its stored strict field",
           c["status"] == "unreadable" and c["valid"] is False and c["stored_strict"] is False
           and cell_text(c) == "unreadable (no per-path figures; invalid)",
           (c["valid"], c["stored_strict"], cell_text(c)))
    for missing in (None, float("nan")):
        r = _fixture([0.60] * 15)
        r["paths"][0]["bycov"]["0.03"]["UP"][1] = missing
        c = strict_certificate(r, "UP")
        expect(f"a counted path whose win rate is {missing!r} never certifies and is named",
               c["status"] != "certified" and any("without a finite win rate" in x for x in c["reasons"])
               and any("p10 is not a finite number" in x for x in c["reasons"]), c["reasons"])
    r = _fixture([0.60] * 15)
    r["operating_cov"] = "0.030"
    c1 = strict_certificate(r, "UP")
    r = _fixture([0.60] * 15)
    del r["operating_cov"]
    r["covs"] = [0.03, 0.02]
    for p in r["paths"]:
        p["bycov"] = {"0.02": {"UP": [60, 0.40], "DOWN": [3, 0.6]}, "0.03": p["bycov"]["0.03"]}
    c2 = strict_certificate(r, "UP")
    expect("the operating coverage is the key numerically equal to operating_cov, else to covs[0], whatever its order",
           c1["status"] == "certified" and c1["operating_coverage"] == "0.03" and c2["operating_coverage"] == "0.03",
           (c1["operating_coverage"], c2["operating_coverage"]))
    expect("stored numbers compare exactly: NaN with null, not NaN with infinity, not 0.1 + 0.2 with 0.3",
           same_number(float("nan"), None) and not same_number(float("nan"), float("inf"))
           and not same_number(0.1 + 0.2, 0.3) and same_number(0.5625, 0.5625) and not same_number(True, 1.0))
    r = {"verdict": {"UP": "CERTIFIED", "DOWN": "KILLED"},
         "verdict_goal": {s: {"CERTIFIED": True, "paths_counted": 15, "paths_clear": 15, "p10": 0.6} for s in SIDES}}
    exact, differ, unreadable, _ = _reread_result("pathless", r)
    expect("the reread counts a stored strict field without per-path figures as unreadable (a failure)",
           exact == 0 and not differ and len(unreadable) == 2, unreadable)
    # third review
    r = _fixture([0.60] * 15)
    for p in r["paths"]:
        p["bycov"] = {"0.0": p["bycov"]["0.03"], "-0.0": p["bycov"]["0.03"]}
    r["operating_cov"] = 0.0
    c = strict_certificate(r, "UP")
    expect("the key the harness writes for the operating coverage wins over a numerically equal one ('0.0' / '-0.0')",
           c["operating_coverage"] == "0.0" and c["status"] == "certified", (c["operating_coverage"], c["status"]))
    r = _fixture([0.60] * 15)
    r["verdict_goal"]["UP"]["CERTIFIED"] = False
    exact, differ, _, _ = _reread_result("flipped", r)
    r = _fixture([], zero=15)
    del r["verdict_goal"]["UP"]["CERTIFIED"]
    exact2, differ2, _, _ = _reread_result("no CERTIFIED", r)
    expect("the reread fails a stored strict field that disagrees with the stored figures, or is not a boolean",
           len(differ) == 1 and exact == 1 and len(differ2) == 1 and exact2 == 1, (differ, differ2))
    r = _fixture([0.60] * 15)
    r["paths"][0]["bycov"]["0.03"]["UP"][1] = float("inf")
    c = strict_certificate(r, "UP")
    expect("a counted win rate that is not finite (infinity) never clears, fails the path test and is named in both "
           "wordings", c["status"] != "certified" and not c["path_success"] and c["paths_clear"] == 14
           and "a win rate not finite" in cell_text(c) and "without a finite win rate" in side_text(c),
           (cell_text(c), side_text(c)))
    # fourth review
    r = _fixture([0.60] * 15)
    r["verdict_goal"]["UP"]["CERTIFIED"] = "false"
    c1 = strict_certificate(r, "UP")
    r = _fixture([0.60] * 15)
    r["validity"] = {"VALID": "false"}
    c2 = strict_certificate(r, "UP")
    r["validity"] = {"in_session_up_rate_in_band": "true", "paths_up_rate_in_tripwire": True}
    c3 = strict_certificate(r, "UP")
    expect("a flag that is not a boolean (a stored CERTIFIED or VALID of 'false', a component check of 'true') never "
           "certifies", c1["status"] != "certified" and c2["status"] != "certified" and c2["valid"] is None
           and c3["valid"] is None, (c1["reasons"], c2["reasons"], c3["valid"]))
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        r = _fixture([0.60] * 15)
        r["verdict_goal"] = None
        (Path(tmp) / "malformed_result.json").write_text(json.dumps(r))
        code = reread(Path(tmp))
    expect("the reread fails a result whose strict field is there but is not a mapping", code == 1, code)
    print(f"\nCERTIFICATE SELFTEST: {'PASS' if not failures else 'FAIL'} ({len(failures)} failed)")
    return 0 if not failures else 1


# ---------------------------------------------------------------- reread (the stored results through the function)

def reread(results_dir: Path) -> int:
    """Every stored result that carries `verdict_goal`, read again: the function's strict reading must equal the
    stored field on every side, its counts the stored counts, and its p10 the stored p10 exactly (both missing, or
    the same float bits); a stored strict field without per-path figures fails the reread. Lists the results whose
    legacy verdict says CERTIFIED while the strict certificate does not."""
    read, exact, differ, unreadable, legacy_only = 0, 0, [], [], []
    for path in sorted(results_dir.glob("*.json")):
        try:
            r = json.loads(path.read_text())
        except ValueError:
            continue
        if not isinstance(r, dict) or "verdict_goal" not in r:
            continue
        read += 1
        if not isinstance(r["verdict_goal"], dict):              # a strict field that is there but not a mapping
            unreadable += [f"{path.name} {s}: the stored strict field is not a mapping" for s in SIDES]
            continue
        e, d, u, lo = _reread_result(path.name, r)
        exact, differ, unreadable, legacy_only = exact + e, differ + d, unreadable + u, legacy_only + lo
    print(f"results read: {read}; sides equal to the stored strict field exactly (status, counts, p10 to the bit): "
          f"{exact} of {2 * read}; sides that differ: {len(differ)}; sides without per-path figures: {len(unreadable)}")
    for d in differ:
        print("DIFFERS:", d)
    for u in unreadable:
        print("UNREADABLE:", u)
    print(f"legacy CERTIFIED while the strict certificate is not: {len(legacy_only)}")
    for item in legacy_only:
        print("  legacy only:", item)
    return 0 if not differ and not unreadable else 1


def main(argv: list[str]) -> int:
    if argv[:1] == ["selftest"]:
        return selftest()
    if argv[:1] == ["reread"]:
        return reread(Path(argv[1]) if len(argv) > 1 else Path("/home/sean/git/binary-algo/results/json"))
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
