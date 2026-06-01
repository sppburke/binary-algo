# /goal prompt templates — exhaustive + generative + edge-IMPROVING strategy search

Fill `<X>` (timeframe, e.g. `5`) and `<CURRENCY>` (e.g. `EURUSD`). Paste after `/goal`.

---

## ATTITUDE — read first, this governs everything

**We are hunting an edge and improving the edges we already have. We do NOT use research, null
results, or "the literature says it's hard" as permission to give up.**

- **Negative results describe other people's failures, not our ceiling.** When a paper reports "deep learning
  gets 50% on FX" or "X is unpredictable," that describes researchers who found *nothing*. We have, on this
  data, found *something* (e.g. a certified ~0.55–0.57 5m UP edge where those papers found 0.50). Their nulls
  **do not cap us.** Treat every "it can't be done" as "they couldn't do it," then ask what *we* have that they
  didn't (cross-pair structure, a strong magnitude edge, a disciplined gate).
- **A literature review's real value is a TOOLKIT of edge-IMPROVING levers, not a verdict to stop.** When a
  scholarly search returns mostly nulls, mine it for the *constructive* techniques buried inside —
  seed-ensembling, |return|-weighted / GMADL losses, calibration + adaptive conformal, decorrelated DL+GBM
  stacking, cross-pair pooling, regime adaptation — and GO RUN THEM against our existing edge. Do not let the
  null headline pull you into a defeatist frame; that framing is a bug, correct it.
- **Default to PURSUING, not confirming.** A null on one lever is a *redirect to the next lever*, never a reason
  to halt the search. "This specific method didn't improve it" ≠ "stop improving." Keep a running list of
  untried improvement levers and work it down. The search ends only when the genuine improvement levers are
  exhausted AND the new-method/discovery loop has gone dry for K rounds — not because a paper was pessimistic.
- **Rigor is for HONESTY about what we found, not for dismissing it.** The deriv-faithful discipline (below)
  exists so we never fool ourselves into a fake edge — and equally so we never *undersell* a real one. State
  the verdict precisely; if a harder test changes it, correct the record (in either direction). An edge that is
  "marginal/regime-dependent" is still an edge to improve, not to discard.
- **The binding constraint is usually DATA, and that is a TODO, not a wall.** If every model converges to the
  same ceiling, the lever is new *information* (external data: rate differentials, implied-vol/risk-reversal,
  triangular legs) or a better *use* of what we have (magnitude→direction bridge, cross-pair pooling) — pursue
  those, don't conclude "impossible."

## Reference: the academic-papers corpus

`/home/sean/git/academic-papers/` — a shared folder of downloaded papers and review reports. **Use it and grow
it:** when you research, download the actual PDFs here from ANY source (arXiv, OpenAlex/Unpaywall, author
homepages, SSRN via cookie) — OpenAlex's API (`api.openalex.org/works?...` → `best_oa_location.pdf_url`) and
Unpaywall (DOI→OA) are the most reliable, no-auth fetchers; arXiv `arxiv.org/pdf/<id>` never blocks. Write a
short review report (`_*.md`) summarizing the evidence + the edge-improving levers it surfaced, and cite the
saved PDFs by filename. (SSRN is Cloudflare-gated; a logged-in browser cookie passed by the user unblocks it for
targeted fetches.)

---

## Full prompt (recommended)
```
/goal Find the best <X>-minute UP predictor AND the best <X>-minute DOWN predictor for <CURRENCY> binary
direction, OOS-verified — TWO independent deliverables, each certified to the same standard and each shipped
with a deployable spec. This is your goal; do not stop until both sides are certified-or-honestly-exhausted AND
the improvement + discovery loops have gone dry. NULL literature is not permission to give up (see ATTITUDE in
GOAL_PROMPT.md): we improve the edges we have and hunt for more.

1. EXHAUSTIVE SWEEP — BOTH SIDES, SYMMETRICALLY. Follow the `strategy-eval` skill in EXHAUSTIVE SWEEP MODE. Work
SWEEP_MATRIX.md variant by variant (Tier A→F), retargeting EVERY method/ensemble/RL/DQN/state-space/deep/
microstructure/magnitude permutation to <X> minutes via the catalog knobs (MX_HOR/HS). Reuse the frozen books
in MODEL_REGISTRY.md / books/ as parents/baselines. Score COMBINED, UP, and DOWN as THREE SEPARATE questions. A
SIDE IS NOT "DONE" until it has had the FULL pipeline applied to IT: (a) side-split; (b) a side-specific
gate/threshold selected on worst-VAL-half; (c) a purpose-built side SPECIALIST; (d) a confidence/coverage curve;
(e) — for any operating point whose binding (worst) held-out year clears breakeven on CI-lower — the full-refit
CPCV. Declaring a side dead requires running (a)-(d) and SHOWING the wall, not inferring it from the other side.

2. DISCOVER + INVENT (continuously, mechanism-first, for BOTH directions). Scour arXiv (q-fin.TR/ST, stat.ML),
SSRN, cross-disciplinary fields (econophysics, information theory, point processes, causal discovery), and novel
COMBINATIONS. Download every paper you review to /home/sean/git/academic-papers (arXiv/OpenAlex/Unpaywall/author
pages). When a side is failing, run a MECHANISM-first research pass (WHY is it hard here?) before cycling more
methods. Vet each candidate (genuinely new; plausible DIRECTION mechanism surviving sign-invariance — say which
SIDE's sign it carries; data on disk); append to SWEEP_MATRIX.md Tier-N + IDEAS_LOG.md; test it. Use sub-agents.
Loop discovery until K dry rounds.

3. IMPROVE WHAT YOU FIND (do NOT stop at the first certified book). For every edge found, run the edge-IMPROVING
levers from the literature toolkit and measure each against the incumbent: seed-ensembling (stabilize the tail /
lift the CPCV path-clear-rate), |return|-weighted / GMADL loss (bridge the magnitude edge into direction),
calibration (temperature/Venn-Abers) + ADAPTIVE CONFORMAL gate (regime robustness), DL-as-decorrelated-stack-
member (not replacement), cross-pair POOLING (weight-shared net across pairs), AdamW + tuned LR + Optuna
TPE/Hyperband (n_trials capped & logged, select on worst-VAL-half). Each improvement gets a pre-registered
falsifier: it must beat the incumbent on the binding (worst) held-out year and/or raise the CPCV path-clear-rate.

4. EVALUATION & CERTIFICATION DISCIPLINE (non-negotiable). Deriv-faithful settlement (wc_ret/contig ties-LOSE);
nonoverlap_chrono; per-year 2024/2025/2026 bootstrap CI95; selection on worst-VAL-half NEVER VAL-acc-max
(corr(VAL,OOS)=-0.54 → VAL-maximal pockets ANTI-transfer; this trap bites SPECIALISTS hardest — watch
VAL-great/OOS-dead); moved-bars-only with up-rate∈[0.47,0.53] tripwire; pre-register a falsifier in the result
JSON BEFORE looking at OOS. CERTIFICATION STANDARD: a frozen-trade CPCV (resampling a fixed rule) does NOT refit
→ blind to selection overfitting → OVER-states; a loose-coverage refit UNDER-states. The certifying test is the
FULL per-fold-refit CPCV evaluated AT THE BOOK'S OPERATING GATE: certify only if refit p10 ≥ breakeven AND ≥~80%
of purged folds clear. Use precise vocabulary ("forward-positive" ≠ "refit-certified" ≠ "regime-dependent" ≠
"dead"); CORRECT THE RECORD immediately if a harder test changes the verdict (either direction). ADVERSARIALLY
VERIFY every positive (panel: leakage, selection/refit-overfit, drift-not-skill, settlement, coverage).

5. DEPLOYMENT SPEC for every survivor (each side). Operating gate + coverage (trades/session); confidence/
coverage curve; FRACTIONAL-Kelly sizing on the ROBUST FLOOR (refit p10), not the optimistic single-split;
equity path with max drawdown + losing-streak; honest risk note (any kill-switch must be TESTED not assumed).
VERIFY the actual tradeable venue/min-duration from a live/authoritative source before calling anything "the
best algo to trade" — do not rely on a stale assumption about the floor.

6. SUBSUMPTION AUDIT + RECORD-KEEPING. Mark a row "subsumed" only with a Tier-1 rationale citing the
script/result-JSON/theorem that fixes its answer; periodically AUDIT the subsumed set (an adversarial pass) to
separate "truly determined" from "must actually run" — run, don't argue. Record COMBINED+UP+DOWN into
<CURRENCY>_RESULTS.md (magnitude → MAGNITUDE_FINDINGS.md); maintain sweeps/<CURRENCY>_<X>m.md + the two-sided
leaderboard; freeze every survivor as <CURRENCY>.<book>.v1 via manifest.py + git-tag. ONE heavy job at a time
(OOM); commit ledger+results frequently (flaky external drive); EVIDENCE-FIRST — every number traces to a
result JSON, never prose; flag thin-coverage (n<~50) and VAL-acc-max as non-robust. Think before each
permutation. Be thorough; don't over-claim; and DON'T GIVE UP — a null is a redirect, not a stop.
```

## Short prompt
```
/goal Best <X>-minute UP predictor AND best DOWN predictor for <CURRENCY>, OOS-verified — two deliverables, full
symmetric pipeline each (side-split → side-gate → side-specialist → confidence curve → full-refit CPCV at the
operating gate). Use the strategy-eval skill in sweep mode: run every SWEEP_MATRIX.md permutation at <X>m AND
research/invent new ones (download papers to /home/sean/git/academic-papers) AND run the edge-IMPROVING levers
(seed-ensemble, |ret|-weighted/GMADL loss, calibration+adaptive-conformal, DL+GBM stacking, cross-pair pooling)
on whatever you find. Certify = full-refit CPCV at the operating gate; ship a deployment spec (gate, fractional-
Kelly on the refit floor, equity/drawdown). Record combined+up+down; resumable ledger + two-sided leaderboard;
freeze survivors. Evidence-first; one heavy job at a time; null literature ≠ give up — improve the edge.
```

## Notes
- The skill (`.claude/skills/strategy-eval/SKILL.md`) is auto-discovered; the prompt sets the objective + scope.
- For a NEW currency with no data on disk, the first step is data acquisition (a prerequisite the skill flags).
- To resume an interrupted sweep, the same prompt works — the ledger is the state; pick up the first `pending`
  row, and the improvement/discovery loops continue.
- **The two failure modes to avoid are symmetric:** fooling yourself into a fake edge (cured by the discipline),
  and talking yourself out of a real one (cured by the ATTITUDE). Hold both.
```
