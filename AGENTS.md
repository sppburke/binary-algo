# Repository Agent Standard — binary-algo

## Scope and precedence

`AGENTS.md` is the canonical shared instruction file. `CLAUDE.md` must be a symlink to
`AGENTS.md`, so supported agent runtimes receive the same project standard.

These are repository defaults. An applicable repository skill may specialize execution,
authorization, mutation boundaries, verification, stopping conditions, and output. Reconcile
compatible instructions and report any unresolved conflict. The user's request defines the
desired outcome; it does not authorize unrelated changes or external side effects.

## Repository workflows

`.codex/skills/` is the canonical physical skill catalog. `.claude/skills/` exposes the same skills
through relative directory symlinks. Never maintain separate skill bodies. Run
`~/binary-algo-venv/bin/python scripts/check_agent_contracts.py` after changing instructions,
skills, their metadata, or projections.

For every task:

1. Determine whether a current skill applies from its frontmatter; do not rely on remembered
   triggers.
2. Use the smallest set that fully covers the request. Use an explicitly named available skill.
3. Read each selected `SKILL.md` completely and every reference it marks as required for the task.
4. Let the selected skill own its workflow while independently verifying its application.

Tracked-file mutation requires `dev-cycle` from the start. Strategy evaluation, retargeting,
sweeping, freezing, or result-ledger work requires `strategy-eval`; infrastructure and
documentation changes remain in `dev-cycle`. A batch committed by the research lab's reviewed
recorder, or by the reviewed scheduled probe fold, runs under that tool's own contract; changing
such a tool is `dev-cycle`.

`feature-dev` writes a plan and hands it to its authority; `plan-review` reviews one and returns
proposed text. A request to review, improve, or expand an existing plan asks for proposed text;
publish it only when asked.

Helper skills:

- `git-workflow` holds the Git and GitHub mechanics every workflow uses.
- `deriv-lab-ops` operates the Deriv-feed research lab, which lives outside the repository.
- `deriv-runtime-ops` inspects and operates the Deriv demo runtime.
- `repo-audit` is a read-only consistency audit.

## Core standard

Use evidence-first verification. Accuracy is the success metric—not agreement, speed, verbosity,
or approval. Test the premise and its strongest material counterargument. State negative results
directly. Never present an inference, generated example, fixture, assumption, or helper-agent
claim as an observed fact.

`docs/_EVIDENCE-FIRST.md` is the binding claim, source-tier, falsifier, conflict, and
`Checked / Showed / Unknown / Needed` standard. Point to it instead of duplicating its rules in
skills or other instructions.

## Read by task

Always read `docs/_EVIDENCE-FIRST.md` before material factual investigation or mutation. Then
load only what the task requires:

- Repository layout, filing, or movement: relevant `README.md` and `REPO_MAP.md` sections.
- Strategy or measured-result work: `README.md`, `REPO_MAP.md`, `MODEL_REGISTRY.md`,
  `books/INDEX.json`, the applicable Tier-2 ledger/sweep, and `strategy-eval`.
- Code or infrastructure: owning implementation, call sites, focused tests, and relevant contract
  docs; load model registries only when the change affects them.
- Deriv-feed program work: `docs/DERIV_SOTA_PROGRAM.md`, then the newest applicable owner decision
  in the lab's log, through `deriv-lab-ops`.
- Deriv runtime code, `ops/` units, or the trading host: `docs/DERIV_DEMO_EXECUTOR.md`, through
  `deriv-runtime-ops`.
- Roadmap-linked work: the authoritative parent/child issue or owner-approved program record and
  the machine projection it names.

Do not load unrelated registries or findings merely because they exist.

## Repository map

| What | Where |
|---|---|
| Frozen deployable models | `books/<PAIR>.<book>.v1/` |
| Per-pair records of truth | `results/<PAIR>_RESULTS.md` |
| Raw experiment JSON | `results/json/` |
| Python and shell scripts | `scripts/` |
| Methodology and contracts | `docs/` |
| Sweep ledgers and backlogs | `sweeps/` |
| Literature corpus | `research/` |
| Logs and checkpoints | `logs/` |
| Runtime unit templates | `ops/` |
| Research lab and Deriv tick archive, outside the repository | `/mnt/data/deriv-sota-lab/`, `/mnt/data/deriv/` |
| Feature data and `macro_calendar.parquet` | repository root; do not move |

## Engineering decision standard

Apply these principles together:

1. Preserve the requested intent and explicit non-goals.
2. Require evidence-backed internal consistency and define falsifiers for material premises.
3. Make the minimum complete change—not merely the smallest diff.
4. Extend the correct existing semantic owner when its contract fits.
5. Follow current healthy repository precedent when reuse is impossible.
6. Compare non-obvious choices with the strongest simpler repository-native alternative.
7. Minimize durable surfaces, states, dependencies, and control paths.
8. Define elegance as clear ownership, explicit failure semantics, and a minimal exposed API.
9. Reject speculative over-engineering and correctness-reducing shortcuts equally.

Scale safeguards to actual risk. Security, protected-data access, identity, immutability,
multiplicity, settlement, activation authority, recovery, and verification are constraints when
the change touches them; they are not boilerplate for unrelated work.

## Working method

- Inspect the semantic owner, consumers, nearest precedent, failure behavior, and verification
  path before editing.
- Trace material behavior end to end rather than reasoning from one file or the happy path.
- Preserve user work and unrelated changes; do not use destructive recovery or broad cleanup.
- Keep external-system investigation read-only unless the user or selected workflow authorizes the
  mutation.
- Use native subagents selectively for bounded independent research or adversarial review. The
  primary agent must verify, reconcile, and disposition every material finding.
- Run focused verification first and broaden only when the blast radius is unclear or a binding
  workflow requires it.
- Inspect the final diff and repository state before completion. Report shortcuts, skipped checks,
  temporary workarounds, declined review findings, and unresolved concerns; state `none` when
  applicable.

## Operational rules

- **Deriv traffic comes only from the Mexico VPS.** Nothing on the US workstation opens a connection
  to `*.derivws.com` or `*.deriv.com`, public or authenticated: no tick or candle downloads, quote
  surveys, offer or near-close probes, or ad-hoc calls (owner decision of 2026-11-25). Archived data
  may be synced here for offline work.
- Run scripts from the repository root: `~/binary-algo-venv/bin/python scripts/<name>.py`. Never
  run them from `scripts/`.
- Scripts may write `*_result.json` to the current directory. Archive finished results under
  `results/json/` before the final gate and commit.
- Do not move `features*/`, `syn_data/`, or `macro_calendar.parquet`; scripts contain absolute and
  relative assumptions about those locations.
- Use `~/binary-algo-venv`; pinned dependencies live in `ENVIRONMENT_libs.txt`.
- Heavy fits run one at a time because of prior OOM failures, unless an owner decision in the
  governing program record authorizes bounded concurrency.
- Other processes write to this checkout and to `main`: the lab's result recorder and the scheduled
  probe fold. Check for them before editing a tracked file or leaving an untracked one;
  `git-workflow` has the procedure.
- Never read or print the private runtime environment file, or a Deriv account ID or token. A code
  change never authorizes a service action on the trading host or anything on a real account.
- For `sppburke/binary-algo` GitHub access, use the current `gh` account first. If it cannot resolve
  the repository, run the command with an ephemeral token from `gh auth token --user sppburke`.
  Never print, persist, or pass that token through helper-agent messages.

## Research discipline

- Every measured number traces to an on-disk result JSON or the applicable Tier-2 ledger.
- Generic docs never own one key's number. Tag generic citations `[PAIR·tf]` and point to the
  Tier-2 record.
- Never pool across timeframes or transfer a number between keys.
- Deriv Rise/Fall settlement is ties-lose with breakeven `0.541`.
- Certification requires 15 refit-CPCV purged paths with per-fold refit, `p10 >= 0.541`, and at
  least 80% of paths clearing breakeven, gated to `America/New_York` 08:00–17:00 with DST handled
  correctly.
- Frozen books are refit-dependent vintages, not permanent evidence of current edge.
- For Deriv-feed results the certificate of record is `strict_certificate` in
  `scripts/certificate.py`, read at the registered operating coverage.

All seven 15-minute on-disk sweeps on the Dukascopy-derived stores are closed. Do not re-mine the
neural/spectral, image, GRU/ESN, TSFM, or Optuna nulls; cross-pair pooling for the
USDJPY/USDCAD/AUDUSD/NZDUSD own-pair family; or Deriv synthetic indices, which are recorded as
IID-Gaussian/memoryless in `docs/SYNTHETIC_RNG_FINDINGS.md`. See `scripts/README.md` and the
per-pair ledgers. New edge on those stores requires separately authorized external
macro/rates/flow data, not more on-disk feature churn. A closure is reopened only by a recorded
owner decision, and only for the substrate, keys, and family it names; the Deriv-feed program's
bounded reopenings are in `docs/DERIV_SOTA_PROGRAM.md`.

## Completion and communication

Lead with the verified conclusion, product/research effect, and governing logic. Distinguish fact,
inference, judgment, and unknowns. Keep detail proportional to material decisions and omit filler.
Put a decision that belongs to the owner as a self-contained question with options and a marked
recommendation, never as a list of pending items in a report. Do in the same turn what you said you
would do, and arrange to be told when queued work ends before leaving it.

Record measured results through `strategy-eval`: Tier-2 ledger plus sweep state, result JSON, and—
for a certified survivor whose deliverable is a book—the frozen book, `MODEL_REGISTRY.md`, and
`books/INDEX.json`. The lab's recorder records the Deriv-feed result families it covers. Record
generic methods in `docs/METHODS_CATALOG.md`, deferred ideas in `docs/IDEAS_LOG.md`, and cross-key
sweep status in `SWEEP_MATRIX.md`.
