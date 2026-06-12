---
name: plan-review
description: Review a binary-algo experiment plan, method design, sweep plan, new-key bootstrap, infra spec, or GitHub issue before execution. Use for plan soundness, leakage checks, repo fit, and minimum viable scope. If invoked with `#222`, review issue 222 from `sppburke/binary-algo`.
---

# plan-review

Review a plan, spec, or hand-off against the repo it will be run in. Output a verdict and findings; never run the
experiment, fit a model, freeze a book, or rewrite the plan into a new plan.

## GitHub Issue Inputs

- A bare issue reference like `#222` means `https://github.com/sppburke/binary-algo/issues/222`.
- Fetch it before normalizing the plan: `gh issue view 222 -R sppburke/binary-algo --json number,title,body,url,state,labels`.
- Use the issue title/body/comments supplied by the user as the plan artifact. Do not ask the user to paste the
  issue unless `gh` cannot access it.
- Use the locally configured `sppburke` GitHub token/account. If `gh` is on the wrong active account or repo lookup
  fails, run `gh auth switch -u sppburke` and verify with
  `gh repo view sppburke/binary-algo --json nameWithOwner,url`. Never print, extract, or paste the token.
- The review output should name the issue number and URL. Do not edit, close, or comment on the issue unless the
  user explicitly asks you to post the review.

## Required Inputs (and what to do if missing)

| Missing | Action |
|---------|--------|
| Plan / design / backlog-row body | Block — nothing to review |
| Intent / the edge question + key | Infer from artifact; block only if unrecoverable |
| Acceptance criteria / KILL conditions | A **pre-registered falsifier is mandatory**; if absent, Should-fix at minimum and often Blocking — an experiment with no KILL conditions is untrustworthy by construction |
| Non-goals, constraints | Should-fix; do not block |
| Touchpoints (scripts/features/ledgers) | Discover via repo scan; do not ask |
| Existing result / partial code | Optional; if present → drift check (see Partial) |

Ask the user only when ambiguity changes the measured result, the key, the settlement, the held-out window, or
the deliverable (number vs reusable method vs frozen book). Otherwise pick the repo-default and mark the assumption.

## User Guidelines (verbatim)

Each label is the user's exact wording; the gloss is operational for an experiment/infra plan in this repo.

1. **accurate to the intent of the change** — plan answers the stated edge question for the stated key; the
   evaluation target (direction win-rate vs magnitude AUC) traces to intent.
2. **internally consistent** — label, settlement (`wc_ret`, ties-lose), splits, falsifier, evaluation, and
   recording all align; no step contradicts another.
3. **the minimum viable code change required for deployment** — the smallest run/diff that answers the question;
   optional scope → non-goal. Don't plan a 12-arm sweep when one falsifiable arm decides it.
4. **re-uses as much as possible of the current code base** — import `harness.py`, `min1_production.py`
   (`wc_ret`/`boot`/`nonoverlap_chrono`/`mk_lgb`), the per-horizon production templates, `crosspair.py`,
   `cpcv_certify.py`, `manifest.py`; retarget an existing `METHODS_CATALOG` method via `MX_HOR`/`HS` rather than
   writing new.
5. **leverages repo precedence when re-use is not possible** — match the deriv-faithful discipline, the
   GENERIC↔KEY-SPECIFIC filing, per-pair artifact naming, the result-JSON + falsifier shape.
6. **is as close to a global optimum as possible while being the minimum viable code change** — right mechanism
   and complexity *for this key on this data*; not speculative, not underfit. (Honest prior: ≤5m direction is
   near-efficient; magnitude is the certified edge.)
7. **simplicity is important and essential** — fewer arms, one coherent evaluation path, one falsifier.
8. **code elegance is the highest form of beauty** — clean reuse of the harness, minimal new surface, repo-native
   shape; no clever novelty where a retarget suffices.

**Conflict ladder** (earlier wins unless later is Blocking for correctness, leakage, an invalid number, an
unfalsifiable claim, or a discipline breach):

```
1 intent  >  2 consistency  >  3 MVCC  >  4 reuse  >  5 precedent  >  6 optimum  >  7 simplicity  >  8 elegance
```

Scope expansion past MVCC requires cited Blocking evidence; record the trade.

## Method

Tier annotations follow the **evidence-first standard** (`docs/_EVIDENCE-FIRST.md` — Primary Sources tiers: T1
repo/data ground truth, T2 runtime evidence captured now, T3 specs/docs read this session, T4 indirect/fallible).

1. **Normalize [T0]** — extract intent (the edge question + key), AC / KILL conditions, non-goals, constraints,
   touchpoints, the evaluation target; tag each as explicit / inferred / missing / contradicted.
2. **Claim map [T0]** — convert plan statements into testable assertions (mechanism, label/settlement, splits,
   features, leakage exposure, incumbent comparison, certification, filing). A claim is **non-trivial** if it
   affects the measured number, the validity of the evaluation, the deliverable, or the recording. Build it as an
   explicit **ledger**, one row per claim. You **must** enumerate a row for each of: every named script /
   function / feature-set the plan cites, every key `(pair, tf, side)` and `MX_HOR`/`HS` value, every number it
   claims (incumbent, target, prior), every result-JSON / ledger / book id it references, and every KILL
   condition. Each row carries its own falsifier (step 3). The sweep is complete only when every row is resolved
   (`checked` or `Blocked`) — **a verdict-determining finding does not license leaving rows unresolved.** Carry
   the enumerated/resolved counts into the Output `## Coverage` line.
3. **Falsify [T1 → T2]** — per non-trivial claim: pre-commit a falsifier; check Tier 1 (the scripts, `harness.py`,
   `METHODS_CATALOG`, the `results/<PAIR>_RESULTS.md` / `sweeps/` ledgers, `books/INDEX.json`, on-disk
   `features/`); descend to Tier 2 (a targeted run, a quick `wc_ret`/up-rate check, a row count on a parquet
   slice) only when T1 is silent; else **Blocked** with the exact next check.
   - **3b. Citation sweep [T1]** — resolve every script/function/feature/ledger-line/book-id the plan cites
     against the current tree in one pass. A citation that doesn't resolve (a renamed script, a number not in any
     result JSON, a book id not in `INDEX.json`) is a Should-fix (drift) at minimum; batch them.
4. **Trace [T1]** — label → settlement (`wc_ret`) → splits (TRAIN/VAL/held-out per year) → selection (VAL
   worst-half) → evaluation (moved-bars, `nonoverlap_chrono`, `boot` CI95) → recording (Tier-2 ledger + result
   JSON). Enumerate where the plan would pool across timeframes, copy a number across keys, or skip a stage.
5. **Score** — apply Guidelines + the Plan Audit matrix to the *predicted run*.
6. **Drift** — if a partial result/script exists, classify each plan step `done | partial | missing | extra |
   changed-shape` (see Partial).

## Plan Audit (merged matrix)

For each area: what the plan must contain, the smell that signals failure, and the evidence that resolves it.

| Area | Required | Smell (Blocking unless justified) | Evidence |
|------|----------|------------------------------------|----------|
| Intent | The edge question, the key `(pair, tf, side)`, the evaluation target | Restated as "try method X" with no stated question or key | Read intent vs the falsifier |
| Mechanism / sign-invariance | A plausible reason the lever carries *sign*, not just size | Entropy/OF/complexity/Hurst/HMM "direction" idea — sign-invariance theorem says these gate SIZE | `docs/THEORY.md`; reframe as magnitude `|ret|≥Q` |
| Falsifier | Explicit pre-registered KILL conditions, written into the result JSON before OOS is seen | "See if it works"; no kill threshold; tuned-until-passes | The falsifier text; `strategy-eval` §2 |
| Settlement | Deriv-faithful `wc_ret` (mid-to-mid, +1s lag, **ties LOSE**, breakeven 0.541) | `mid.shift(-N)` on gap-dropped bars; ties counted as wins | `min1_production.py:wc_ret`; trap #3/#7 |
| Splits / held-out | Per-year 2024 / 2025 / 2026 separately with `boot` CI95; 2026 strict OOS | A single pooled number; no per-year breakdown | `harness.py:SPLITS`; §2 |
| Selection | Threshold/gate chosen on **VAL worst-half** | VAL-acc-max selection | `corr(VAL_acc,OOS_acc)=−0.54`; trap #5 |
| Moved-bars | Evaluate on `|ret|>0`; verify moved up-rate ∈ [0.47, 0.53] | All-bars accuracy; no up-rate check | trap #2 (ffill-flat mirage) |
| Leakage | Names which §3 traps it avoids and how | Any of: bar-shift mislabel, ffill-flat, greedy de-overlap, thin-coverage, future-peek smoother, forecast-window misalign | `strategy-eval` §3 |
| Reuse | Retarget an existing method / import the blocks | New script/feature without a `METHODS_CATALOG`/`scripts/` grep first | grep before accepting a new surface |
| Precedent | Matches the discipline, filing, per-pair naming, result-JSON shape | Ad-hoc evaluation outside the §2 protocol | `REPO_MAP.md`; an existing sibling script |
| Incumbent | Beats the **best combination** for the key on its binding (worst held-out) year, CI95-lower clearing it | Compares against the old base book, or a pooled number | `MODEL_REGISTRY.md` / `books/INDEX.json` |
| Certification (deliverable book) | Refit-CPCV 15 purged paths, per-fold refit, **p10 ≥ 0.541 AND ≥ 80% paths clear**, NY-session gate | Frozen-vintage or pooled-CPCV-only claim of a deployable edge | `cpcv_certify.py`; trap #9 |
| Non-stationary features | Calendar/seasonal/slow features confirmed on a per-year frozen-past forward holdout | Pooled-CPCV edge leaning on calendar/seasonal features, no forward check | trap #9 (worked example: time-of-day +.014 → −.054) |
| Data | On-disk substrate named (`features/<PAIR>_<year>.parquet`, `features_tick_*`) vs acquisition | "Use the tick data" when raw ticks are dormant on disk | `README.md` data section; `ls features*/` |
| Filing | Where the result is recorded — Tier-2 ledger + result JSON; generic idea logged once | Plan to write a per-key number into a generic doc, or duplicate it across files | `REPO_MAP.md` GENERIC↔KEY-SPECIFIC |
| Compute | One heavy fit at a time; TRAIN subsample ≤100–150k; idempotent/resumable | Parallel heavy fits; whole-history in memory | §2 memory/compute; OOM history |
| Verification / evidence | Every claimed number traces to a result JSON or ledger line (Tier-1) | Numbers from prose/memory; "probably ~0.58" | `docs/_EVIDENCE-FIRST.md` |
| Granularity | Steps describe outcome + key script ownership | Over-specified line-edits OR vague "then evaluate" | Compare to a prior backlog row |

## Severity

| Tag | Meaning | Example |
|-----|---------|---------|
| **Blocking** | Invalid number, leakage trap, unfalsifiable claim, non-deriv-faithful settlement, ties-as-wins, pooled-across-tf, sign-invariant idea sold as direction, deployable claim without the certification bar, duplicates a measured key | `mid.shift(-N)` settlement on gap-dropped bars |
| **Should fix** | Missing non-goals, weak/absent falsifier detail, ambiguous deliverable, filing into the wrong tier, citation drift | Falsifier omits the per-year CI95-lower condition |
| **Consider** | Tradeoff, alternate mechanism, optional cheaper arm | A combination worth trying before a novel lever |
| **Blocked** | Review evidence missing | Tier 1 silent on whether the incumbent number is in any result JSON; need to grep `results/json/` |

Any **Blocking** ⇒ verdict cannot be approve. Any unverified material claim ⇒ **Blocked**, not speculation.

## Partial Implementation

If plan + a partial run/script both exist:

1. Review plan quality first.
2. Classify each plan step: `done | partial | missing | extra | changed-shape`.
3. `changed-shape` without a plan update → Blocking unless harmless and documented.
4. `extra` must satisfy MVCC or move to non-goals.
5. Number-correctness findings (is the recorded value right?) → defer to `strategy-eval` recording; here flag
   plan/run contract drift only (the run measured a different key/horizon than the plan stated).

## Multi-phase sweeps

If the plan spans multiple tiers/rounds (a `SWEEP_MATRIX` sweep):

- Each tier/arm gets its own falsifier (sized to its prior — low-prior arms die fast).
- The `sweeps/<PAIR>_<tf>.md` ledger IS the resumable state; the plan must not restart a ledger or re-run a
  `done` row.
- Discovery rounds must have a stop rule (K consecutive dry rounds), not open-ended.

## Worked Example (binary-algo)

```text
Plan claim: "Add time-of-day + day-of-week features to the EURUSD 15m magnitude book; expect +AUC since vol is
seasonal. Validate with pooled CPCV (15 paths); ship if ΔAUC > 0.”

Assertions:
  A1 the book lacks calendar features today
  A2 calendar features lift pooled-CPCV magnitude AUC
  A3 pooled-CPCV lift implies a deployable edge

Falsifiers:
  A1 — grep the feature builder / METHODS_CATALOG for existing time-of-day features
  A2 — read mag_har / deseason result JSONs for prior calendar-feature runs
  A3 — locate any per-year frozen-past forward-holdout result for calendar features

Finding (Blocking):
  A3 is the pooled-CPCV non-stationary-feature memorization trap (#9). Calendar/seasonal features memorize
  era-local structure across flanked CPCV folds and score high with NO forward transfer; the program already
  measured exactly this — raw time-of-day gave +0.014 pooled-CPCV ΔAUC that decayed 2024 +.02 → 2026 −.054 on a
  frozen-past holdout and was KILLED for deployment (`docs/MAGNITUDE_FINDINGS.md §6f`; result JSON cited there).
  Plan delta: replace A3's pooled-CPCV ship criterion with "beat the forward-robust base rv30/rv120 model on a
  per-year frozen-past holdout (train≤Y → test Y+1, Y+2), CI95-lower clearing the incumbent in every year", or
  reclassify as a non-deployable diagnostic. As written, the plan would re-derive a known false edge.
```

## Output

```text
## Coverage
claims enumerated: N | falsified: N | blocked: M | citations checked: C
# N must equal (falsified + blocked). If not, the sweep is incomplete — do not emit a verdict.

## Verdict
approve | approve with revisions | reject — one sentence.

## Handoff-Ready
yes | yes-after-revisions | no — if not yes, list the minimum missing items (usually: a written falsifier).

## Blocking
- [finding] — [plan §X | script:fn | ledger line | result JSON | run output] → [exact plan delta].

## Should Fix
- [finding] — [evidence] → [delta].

## Consider
- [tradeoff] — [evidence] → [delta if any].

## Guideline Fails
- [n] [name]: [one-line reason]   # only list fails; passes implied

## Global Optima
One paragraph: where the plan sits on the complexity/mechanism curve; what moves it toward the optimum *for this
key on this data* (given the honest prior).

## Blocked
- Checked: [...]   Showed: [...]   Unknown: [...]   Needed: [exact run/grep/file].
```

Omit empty severity sections. Inline plan deltas in each finding — no separate deltas block.

## Self-Check

Before emitting:

1. Every factual claim has a Tier-1/2 citation or is Blocked. The claim-map ledger is fully resolved — every
   enumerated row is `checked` or `Blocked` — **regardless of whether the verdict is already determined.** The
   `## Coverage` tally must balance (`enumerated == falsified + blocked`).
2. Every Blocking has an exact plan delta.
3. MVCC honored — no speculative arms demanded.
4. Reuse falsifier run before accepting any new script/feature/method (grep `METHODS_CATALOG` + `scripts/` first).
5. The falsifier check: did you confirm the plan pre-registers KILL conditions written into the result JSON before
   OOS? An experiment plan without that is Blocking by default.
6. Every delta you propose is itself a plan claim. Run the reuse + precedent + leakage falsifier on each delta
   before writing it — a recommendation that introduces a new arm/feature/settlement without that check is the
   same defect you'd flag, and becomes the next round's finding.

## Do Not

Run the experiment, fit a model, or freeze a book. Rewrite the plan into a new plan. Approve while a Blocking
exists. Approve an experiment with no pre-registered falsifier. Accept a number that doesn't trace to a result
JSON / ledger line. Accept `mid.shift(-N)` settlement, ties-as-wins, VAL-acc-max selection, an all-bars number,
or a pooled-across-tf claim. Sell a sign-invariant statistic as a direction edge. Treat a pooled-CPCV lift as a
deployable edge. Use stale prose/memory for "already tried" — grep the ledgers. Accept "TBD". Stop the claim-map
sweep early because the verdict is already decided.
