# REPO MAP — file-organization convention (GENERIC ↔ KEY-SPECIFIC)

**Read this first if you are an agent maintaining this repo's records.** It defines where everything goes so
the bookkeeping stays maintainable as we add currencies and timeframes. The `strategy-eval` skill enforces it.

This repo forecasts deriv.com binary **DIRECTION** (`sign(close(t+H)−close(t))`) and **MAGNITUDE** (`|ret_H|≥Q`)
for FX pairs across timeframes. A **key** = `(PAIR, timeframe[, side])`, side ∈ {UP, DOWN}. **5m EURUSD is ONE
key among many** (we will run GBPUSD, 10m, 30m, …). Every document is exactly one of two tiers:

---

## TIER 1 — GENERIC (currency/timeframe-AGNOSTIC)

Reusable across all keys. **MUST NOT bake in any single key's incumbent numbers, scripts, or verdicts.** If a
generic doc must cite a result as an example, it **tags it inline `[PAIR·tf]`** and points to the Tier-2 file of
record. Methods are written to retarget via env `MX_HOR=<minutes>` (bar models) / `HS`,`HSEC` (tick models).

| File | Role (generic) |
|---|---|
| `METHODS_CATALOG.md` | Reusable methodologies — one entry per technique: what · how-to-retarget · why · leakage traps · status-**pointer** (no per-key numbers). |
| `SWEEP_MATRIX.md` | The permutation **menu** (Tier A–F fixed + Tier-N discovered), with retarget knobs. The "what to try" for any key. |
| `IDEAS_LOG.md` | Generic **idea→experiment backlog**: each idea = mechanism + sign-invariance note + falsifier **template** + `tested-on-keys:` pointers. NO per-key incumbents/numbers. |
| `THEORY.md` | Cross-key facts: sign-invariance theorem, direction-ceiling shape by horizon, deriv settlement & breakeven (0.541 @ R≈1.85). |
| `GOAL_PROMPT.md` | `/goal` kickoff templates (parameterized by `<X>`,`<CURRENCY>`). |
| `research/**` | Literature corpus + synthesis (generic). |
| `README.md`, `REPO_MAP.md` | Repo overview + this convention. |
| `CORRECTNESS_AUDIT.md` | Record-of-truth for evaluation integrity: FM-A…FM-G failure-mode taxonomy, repo-wide audit verdict (FM-F look-forward bug isolated to Kronos), the 4 clean-substrate proofs, flagged-scripts table + remediation. |

## TIER 2 — KEY-SPECIFIC (named/labeled by the key — ALL incumbents, numbers, results, backlogs live here)

| File pattern | Role (key-specific) |
|---|---|
| `<PAIR>_RESULTS.md` | Per-**currency** results across tf×side: MASTER KEY TABLE + per-tf experiment tables + the UP/DOWN leaderboard. (e.g. `EURUSD_RESULTS.md`) |
| `sweeps/<PAIR>_<tf>.md` | Per-(currency,tf) **SWEEP LEDGER** — the resumable state of the exhaustive sweep (row-by-row status). |
| `sweeps/<PAIR>_<tf>_backlog.md` | Per-key **EXECUTABLE BACKLOG** — the ranked TOP-N experiments, FIRST-TO-RUN queue, "incumbents to beat", and key-specific discovery rounds. (Moved out of the generic `IDEAS_LOG.md`.) |
| `<PAIR>_<tf>_research_log.md` | Per-key narrative research log (the running prose journal for that key). |
| `MAGNITUDE_FINDINGS.md` | Magnitude is sign-invariant → no UP/DOWN key, but still **per-currency**; label each result by `PAIR`. |
| `MODEL_REGISTRY.md` + `books/` | Multi-currency book **registry**; every entry keyed `<PAIR>.<book>.v1` with a provenance manifest. |

---

### Legacy key-specific files (named generically, but EURUSD-scoped — labeled by banner, NOT renamed)

These predate the convention and are densely cross-referenced (some by production `.py` scripts), so renaming
was rejected as high-churn/low-gain. Each instead carries a `SCOPE: EURUSD[·tf]` banner that is the label:
`research_log.md` (cross-tf), `m5_research_log.md` `[·5m]`, `m10_` `[·10m]`, `m30_` `[·30m]`, `min1_` `[·60s]`,
`EXPERIMENT_LEDGER.md`, `EXPERIMENT_BACKLOG.md`, `FINDINGS.md`, `DIRECTION_FINDINGS.md`, `CCM_DESIGN.md` `[·60s]`.
For a NEW currency, prefer the key-named patterns above (`<PAIR>_...`) rather than copying these legacy names.

### Novel-methods campaign files (2026-06-07) — `execute ALL of NOVEL_METHODS_RESEARCH §7`

Registered here so future agents can find them. Verified-facts index for the whole campaign:
**`CAMPAIGN_2026-06-07_FACTS.md`** — single source of truth, every number from a committed result JSON / captured run
(use ONLY its facts; campaign certified NOTHING new — all levers KILLED or REAL-but-SUB-BAR; the one positive is the
certified MAGNITUDE edge RE-VALIDATED on a clean forward holdout, no decay).

**TIER 1 — GENERIC reusable infra / methods**

| File | Role (generic) |
|---|---|
| `NOVEL_METHODS_RESEARCH.md` + `novel_methods_candidates.json` | The campaign's idea slate (§0 gates, §7 method families T1–T5/D1–D8/G1–G3/M4–M6) + candidate registry. |
| `build_panel.py` → `features/panel_<year>.parquet` (gitignored) | Clean 7-pair USD return panel (5.3M bars 2012–2026): `r_<PAIR>`/`fac`/`e_<PAIR>`/`c_eur`. Substrate for ALL path-based cross-sectional methods. |
| `panel_faithcheck.py` + `panel_faithcheck_result.json`, `build_panel_validate_15m.json` | Faithfulness A/B of the panel vs certified `build_xp` (continuous channels bit-identical) → panel CONSTRUCTION-FAITHFUL. |
| `fwd_holdout.py` | **MANDATORY gate** — frozen-past forward-holdout (train≤2023 → per-year 2024/25/26), mag-AUC/lift + dir cov-selacc modes. NOVEL §0 #1. |
| `surrogate_null.py` | **MANDATORY gate** — phase-randomize / IAAFT surrogate-null (separates spectral re-encoding from genuine nonlinear structure). NOVEL §0 #2. |
| `frac_diff.py` | Hand-rolled fixed-width fractional differentiation (FFD weights + ADF d*-selection): "stationarity with memory". Generic transform. |
| `xsec_direction.py` + `xsec_direction_{sig,havok,semivar}_{15,30}m_result.json` (6) | Unified cross-sectional DIRECTION harness on the panel — three dependency-free families, each with a mechanism-specificity SHUFFLE control, run via forward holdout (NY cov0.10 selacc) at 15m & 30m: **D1 sig** (depth-2 lead-lag signature / Lévy area), **D6 havok** (frozen-basis Hankel-Koopman forcing), **D7 semivar** (Patton-Sheppard signed-semivariance). ALL KILLED forward; D1/D6 fail the shuffle null (not genuine), D7 PASSES its sign-flip null (real signed-vol direction content) but is sub-breakeven & decaying & non-additive to the base book. Per-key results in `EURUSD_RESULTS.md`. |

`fwd_holdout.py` + `surrogate_null.py` are the two MANDATORY reusable gates every new lever must pass.

**TIER 2 — KEY-SPECIFIC results (EURUSD)**

| File | Role (key-specific) |
|---|---|
| `mag_har.py` + `mag_har_result.json` | EURUSD MAGNITUDE HAR/RV-canon arms (har/jump/semivar/harq/all) — recorded in `MAGNITUDE_FINDINGS.md §6g`; KILLED as upgrade. |
| `frac_direction.py` + `frac_direction_15m_result.json`, `frac_direction_30m_result.json` | EURUSD FFD-into-direction-book at 15m & 30m — both KILLED (decays in recent years). |

## THE RULE OF THUMB

- "Does this transfer to **GBPUSD or 10m unchanged**?" (a method, idea, template, theorem) → **TIER 1**.
- "Is this a **number / incumbent / verdict / backlog about one key**?" → **TIER 2** (the key's file).
- A Tier-1 doc that must reference a key result → **tag inline `[PAIR·tf]`** and cite the Tier-2 file.

## BOOTSTRAPPING A NEW KEY

1. Confirm data exists (`features/<PAIR>_<year>.parquet` / `features_tick_<PAIR>/`).
2. `<PAIR>_RESULTS.md`: copy `EURUSD_RESULTS.md` structure, reset all keys to UNTESTED.
3. `sweeps/<PAIR>_<tf>.md` (+ `_backlog.md`): instantiate `SWEEP_MATRIX.md` for the key.
4. Retarget Tier-1 methods via `MX_HOR`/`HS`; record results ONLY in the Tier-2 files; freeze survivors as `<PAIR>.<book>.v1`.

## MAINTENANCE INVARIANTS (do not regress)

- Never put a per-key number in a Tier-1 file without a `[PAIR·tf]` tag + Tier-2 citation.
- Never pool across timeframes or copy one key's number into another key.
- Every Tier-1 file opens with a 1-line **SCOPE: GENERIC** banner; every Tier-2 file opens with **SCOPE: <PAIR>[·<tf>]**.
- Evidence-first: every number traces to an on-disk result JSON / ledger line, never prose.
