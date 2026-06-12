---
name: dev-cycle
description: Full development iteration for binary-algo — analyse, plan, implement, run the local evidence gate (smoke-run + evaluation discipline), self-review, then commit and push to main, update/close the driving GitHub issue when present, and archive artifacts. Use for "next phase", "ship change X", "implement Y", "add the harness/module/script", "do the reorg", or any engineering/infra/tooling/doc work that ends in a push to main. If invoked with `#<issue-number>`, treat it as issue <issue-number> in `sppburke/binary-algo`. For evaluating or sweeping a STRATEGY at a (currency, timeframe, side), use the `strategy-eval` skill instead — this skill is for code/infra/doc changes; it defers the evaluation protocol to `strategy-eval`.
---

# Dev Cycle

Full iteration: analyse → decide → plan → implement → verify → **use-case verify** → review → ship → cleanup.
Target branch is `main` (this repo ships **direct to main**, linear history — no PRs, no worktrees, no CI).

## Communication

Terse, high information density. Final summary lists every shortcut, hack, or skipped check. Default is `none`.

## GitHub Issue Workflow

- A bare issue reference like `#<issue-number>` means `https://github.com/sppburke/binary-algo/issues/<issue-number>`.
- If the request includes an issue reference, fetch it during analysis with
  `GH_TOKEN="$(gh auth token --user sppburke)" gh issue view <issue-number> -R sppburke/binary-algo --json number,title,body,url,state,labels,comments` and treat the
  issue body as the task spec. Do not ask the user to paste the issue unless the token-prefixed `gh` command cannot access it.
- Use the locally configured `sppburke` GitHub token/account. Prefer commands with `-R sppburke/binary-algo`. If GitHub access needs an explicit account token, use `GH_TOKEN="$(gh auth token --user sppburke)" gh <your command>` and verify with
  `GH_TOKEN="$(gh auth token --user sppburke)" gh repo view sppburke/binary-algo --json nameWithOwner,url`. Never print, extract, or paste the token.
- After the change is pushed to `main`, comment on the driving issue with what shipped, the commit SHA, the checks
  run, and any skipped checks/shortcuts. Then close the issue once the commit is on `main`.
- This repo ships direct to `main`; if a PR exists for the issue, mention it in the closeout comment, but do not
  invent a PR workflow.

## Model selection

Heavy research/design reasoning (analysis, planning, reviewing a result's validity) benefits from the strongest available reasoning model.
Mechanical edits (renames, path fixes, doc churn, moving files) are fine on any model. State the active session
model in your first status update; do not block on it. There is no review-subagent model split in this repo —
if you dispatch the `code-review` skill (optional, step 6), it runs through whatever subagent mechanism Codex exposes.

## Subagent policy

Delegate **research and discovery** to subagents — codebase exploration, "where is X / how does Y work", paper
mining, prior-result lookup across `results/` and `docs/`. Use available Codex subagent/multi-agent tools so
the main session's context stays clean. Heavy model fits run **one at a time** (OOM history) on the main session;
do not fan out parallel training jobs.

## Mandatory reading before touching code

Per `CLAUDE.md`:

1. `docs/_EVIDENCE-FIRST.md` — the binding investigation standard (tiered sources, pre-committed falsifiers,
   forbidden hedging, Blocked format). Every factual claim in your work traces to it.
2. `README.md` — certified leaderboard, directory structure, how to run scripts, how to load a book.
3. `REPO_MAP.md` — the GENERIC ↔ KEY-SPECIFIC filing convention. Follow it when you add or move anything.
4. `MODEL_REGISTRY.md` + `books/INDEX.json` — frozen books and their provenance.
5. The doc relevant to the task (`docs/METHODS_CATALOG.md`, `docs/DIRECTION_FINDINGS.md`, the per-key
   `results/<PAIR>_RESULTS.md` / `sweeps/<PAIR>_<tf>.md`).

**Doc authority on conflict:** `docs/_EVIDENCE-FIRST.md` (discipline) > `REPO_MAP.md` (filing) > `README.md` >
the per-key Tier-2 ledger > a generic findings doc. Fix the lower-priority doc; never duplicate a number across
files (REPO_MAP rule: a per-key number lives in exactly one Tier-2 file; generic docs tag `[PAIR·tf]` + cite it).

## Task scope classification

Before any work, classify the task by what files the diff will touch. Tag with one or more:

- **code** — diff touches `scripts/*.py` (a shared module, a new/edited experiment or production script, a harness).
- **books/results** — diff freezes or edits a `books/<id>/`, a `results/<PAIR>_RESULTS.md`, or a result JSON.
- **docs** — diff touches only `docs/`, `*.md`, `README.md`, `REPO_MAP.md`, `CLAUDE.md`.
- **tooling** — diff touches only `.gitignore`, `.claude/`, `ENVIRONMENT_libs.txt`, shell runners.

The tag controls:

- **Step 4 discipline** — `books/results` scope (anything producing or changing a measured number) is governed by
  the `strategy-eval` §2 evaluation discipline and the certification bar; this skill does not relax it.
- **Step 5 gate subset** — `docs`/`tooling` scopes skip the smoke-run; a path/link check is the gate.
- **Step 5b use-case verification** — see its skip rule.

Mixed-scope diffs take the union of requirements. State the tag in the commit message body.

## 1. Analyse

State the active session model in your first status update.

```bash
git fetch origin main
git log --oneline -12                  # what shipped recently; carry forward new conventions
git status --short                     # is the tree clean / what's untracked
git log --oneline -3 -- results/ books/ docs/   # recent record-of-truth changes
```

If an issue reference was provided, fetch the issue with `GH_TOKEN="$(gh auth token --user sppburke)" gh issue view <n> -R sppburke/binary-algo ...` before
deciding scope, and include its number/title in the plan.

For recent commits that touch the area you're about to change, read the diff (`git show <sha>`). Identify new
shared utilities, naming, and conventions that landed; carry them forward. If a recent commit makes the task
obsolete or contradicts a ledger, note it in step 2.

## 2. Decide

Pick a task that is:

- Not already done — check the relevant `results/<PAIR>_RESULTS.md` / `sweeps/<PAIR>_<tf>.md` (don't redo a
  measured key; that's strategy-eval §1) and `git log` (don't re-ship a landed change).
- Self-contained — one coherent change you can verify and push in one iteration.
- Honest about the deliverable: an experiment/sweep that produces a number → route through `strategy-eval`; a
  reusable module / refactor / bugfix / tooling / doc change → continue here.

If the task is a multi-step research sweep, the executable backlog is the state: `sweeps/<PAIR>_<tf>_backlog.md`.
Work its FIRST-TO-RUN queue; this skill handles the per-change engineering, strategy-eval handles the evaluation.

## 3. Plan

Present plan, open questions, and design choices — then **answer the open questions yourself with your
recommended choice and proceed**. Do not pause for approval; the user has standing authorization and will
interrupt if they disagree.

Format so the user can scan and intervene:

- **Plan:** numbered steps.
- **Open questions:** each followed by `→ Recommendation: <choice>` and a one-line rationale.
- **Proceeding** with the recommendations above unless interrupted.

Then move straight to implementation. This repo uses a **single checkout, direct to main** — no worktree. Keep
the working tree clean before you start (`git status`); if heavy/regenerable artifacts are lying around, that's a
step-11 cleanup, not a new branch.

## 4. Implement

For `books/results` scope (anything that produces or changes a measured number), the `strategy-eval` skill's §2
evaluation discipline and §3 leakage traps are **non-negotiable and owned there** — invoke that skill for the
evaluation; this step covers only the surrounding code. Do not re-derive or relax that discipline here.

### Style and structure
- **Run everything from the repo root**: `~/binary-algo-venv/bin/python scripts/<name>.py`. This is what makes
  `import harness`, `from sessions import ...` resolve (`sys.path[0]` = `scripts/`) and what lines up the
  hardcoded `features/` absolute paths. Never `cd scripts/`.
- Follow neighbouring file style. **Import, don't reinvent** the building blocks: `harness.py`
  (`feature_cols`, `SPLITS`, the 239 multi-TF features), `min1_production.py` (`wc_ret`, `boot`,
  `nonoverlap_chrono`, `mk_lgb`), the per-horizon `m{5,10,15,30}_production.py` templates, `manifest.py`
  (freeze a book). Check `docs/METHODS_CATALOG.md` before writing a "new" method — most are already implemented
  and horizon-parameterized via `MX_HOR` / `HS`.
- Scripts write `*_result.json` to the **current working directory** (repo root). That is expected; step 11
  archives them to `results/json/`.
- Update `CLAUDE.md` / `REPO_MAP.md` only when shipped work changes a stated rule. Prefer the shortest tweak.
- New shared module? Other scripts import it by bare name (`import foo`) — it must live flat in `scripts/`.

### Data and compute safety
- **One heavy fit at a time** (OOM history). Subsample TRAIN ≤100–150k; build tick features month-by-month.
- Builds should be **idempotent/resumable** — a long job may be interrupted; re-running must not corrupt state.
- Do **not** move root data dirs (`features*/`, `syn_data/`) or `macro_calendar.parquet` — absolute paths are
  hardcoded in scripts. Source 10s OHLCV bars live at `/home/sean/git/processed/<PAIR>/`.

### Honest sentinels and preconditions
- A deferred/stub field uses the value that **fails safely** if downstream code gates on it: `0`, `False`,
  `None`, empty. A stub that *passes* a downstream eligibility check is a logic error, not a deferral.
- Document non-obvious preconditions on any function whose output depends on call order, accumulated state, or a
  clock — one `# Precondition:` line is enough.

## 5. Verify (local evidence gate)

There is no CI and no automated test suite in this repo — **the local evidence gate is authoritative.** By task scope:

- **code**: smoke-run the affected script(s) from repo root on a small slice; confirm imports resolve, no
  exception, and the output shape is sane. For a shared-module change, run one downstream script that imports it.
- **books/results**: the `strategy-eval` §2 discipline IS the gate — deriv-faithful `wc_ret` settlement (ties
  LOSE, breakeven 0.541), independent trades (`nonoverlap_chrono`), per-year held-out 2024/25/26 with `boot`
  CI95, selection on VAL worst-half (never VAL-acc-max), moved-bars only with up-rate ∈ [0.47, 0.53], and a
  **pre-registered falsifier** written into the result JSON before you look at OOS. A deliverable book must clear
  the **certification bar**: refit-CPCV 15 purged paths, per-fold refit, **p10 ≥ 0.541 AND ≥ 80% of paths clear
  breakeven**, NY-session gate. Pooled CPCV is necessary, not sufficient — calendar/seasonal/slow features must
  also pass a per-year frozen-past forward holdout (trap #9).
- **docs/tooling**: skip the smoke-run. Gate = every path/link you touched resolves
  (`for p in <paths>; do [ -e "$p" ] || echo MISS $p; done`), and no mid-token corruption from bulk edits
  (no `_docs` segment, no doubled `docs` segment, and no doubled `results` segment in changed paths).

**The gate must be the LAST thing before `git commit` — no edits in between.** If you change any file after the
gate ran, RE-RUN the gate. A passing smoke-run from five minutes ago is worthless if the file changed since.

### Leakage pre-flight (books/results scope only)
Before believing any number, scan for the recurring fake-edge classes (`strategy-eval` §3): bar-shift horizon
mislabel (use `wc_ret`, not `mid.shift(-N)`), ffill-flat mirage (keep the target pair's own clock; verify moved
up-rate band), greedy-by-confidence de-overlap (use `nonoverlap_chrono`), thin-coverage pocket (n<25–50 = noise),
sign-invariance (entropy/OF/complexity/Hurst gate SIZE not SIGN — test as magnitude). If the diff introduces any
of these, the number is invalid regardless of how good it looks.

## 5b. Use-case verification

Step 5 proves it runs and the number is discipline-clean. This step proves the change **does the right thing
end-to-end** and that every claim traces to evidence.

### When to skip
Skip if the diff has no observable behavioural change: pure doc edits, path-only fixes, dependency-list bumps,
tooling tweaks. State `Use-case verification: N/A — <reason>` in the commit body.

### Determine scope
Cover the changed surface: every function in the diff that takes external input or produces a recorded
output/number/artifact needs one concrete exercise unless a sibling already covers the same path. Ask *"what
would break silently if I wired this up wrong?"* and *"what will the next agent actually do with this on day
one?"* — not *"what does an existing run already cover?"*

Two anchor examples:
- a new shared helper (`feature_cols`-style): feed it one real pair/year slice → assert the returned columns /
  shape match what the production script expects.
- a freeze (`manifest.py`): freeze the book → reload it via the README loading snippet (glob the
  `*strategy.json`, average the `*_lgb.txt` seeds) → assert predictions reproduce the recorded number.

### Exercise and assert
Write the PASS criterion **before** running. One criterion per check, printed to stdout:

```
Check: reload frozen book NZDUSD.m15ny_seedens.v1
PASS: seed-averaged proba on the VAL NY slice reproduces strategy.json's recorded cov2 win-rate ±tolerance
FAIL: file not found, shape mismatch, or number drifts beyond tolerance
```

Determinism: fixed seeds, no wall-clock dependence in the check, recorded slices only. A check that passes once
and fails on re-run is a worse signal than none — fix the leak before re-running.

### On failure
A failure is a **blocker** — fix in step 4, re-run step 5, re-run this step. Never loosen the assertion to pass.

## 6. Review (self-audit; optional code-review skill)

Commit so the diff is stable, then audit it against `docs/_EVIDENCE-FIRST.md`:

- Every factual claim in the change (comments, commit body, ledger rows) has a Tier-1/2 citation or is marked
  Blocked. No forbidden hedging ("should work", "probably reuses") standing in for a check.
- Every number written to a `results/` ledger or a `*strategy.json` traces to an on-disk result JSON — never an
  agent's prose summary.
- Reuse falsifier: did you grep for an existing helper before adding a new one? (`docs/METHODS_CATALOG.md` +
  `scripts/` first.)

For a substantial `code` change you may dispatch the `code-review` skill via an available Codex subagent for a second pass; locate
each finding in your actual diff (`git diff --stat`) before acting — discard findings that cite code not in the
diff. Fix only confirmed findings, then **re-run step 5** (review fixes can break the smoke-run).

## 7. Ship (commit + push to main)

This repo commits **directly to `main`** (Tier-1 practice: the git history is linear; strategy-eval §8.f ships
the ledger+results per step so progress survives interruption). There is no PR, no CI wait, no merge step.

```bash
git fetch origin main
git status --short            # confirm only intended paths are staged-worthy
git add <paths>
git commit -m "<imperative summary>"   # body: scope tag, why-not-what, verdict + Tier-1 evidence (result JSON keys)
```

Commit messages: imperative mood, ≤72-char summary, optional scoped prefix (`NZDUSD 15m:`, `scripts:`,
`docs:`). For a `books/results` change, the body states the **verdict** (CERTIFIED / KILLED / SUBSUMED) and the
Tier-1 evidence (the result-JSON keys the numbers came from), exactly as the existing history reads.

**Pre-push sync.** If `git fetch` shows `origin/main` moved (another machine pushed), `git merge origin/main` (or
`git pull --rebase`) and re-run the step 5 gate against the merged tree before pushing.

```bash
git push
git status --short && git log --oneline -1    # confirm clean tree + your commit is HEAD
```

If push is rejected (non-fast-forward), STOP, sync, re-gate, retry. Do not force-push `main`.

If this work came from a GitHub issue, update the issue after the successful push:

```bash
GH_TOKEN="$(gh auth token --user sppburke)" gh issue comment <n> -R sppburke/binary-algo --body "<what shipped, commit SHA, checks, skipped checks>"
GH_TOKEN="$(gh auth token --user sppburke)" gh issue close <n> -R sppburke/binary-algo
```

Only close after `git push` succeeds and the shipped commit is confirmed on `main`.

## 8. Cleanup

Mandatory after a successful push:

- **Archive result JSONs**: scripts wrote `*_result.json` to the repo root → `mv *_result.json results/json/`
  (then commit the move, or stage it with the work in step 7). The root must not accumulate result JSONs.
- **No stray artifacts at root**: logs/checkpoints/`.npz`/`.pt` belong in `logs/` (and are gitignored). Confirm
  `git status --short` shows nothing unexpected.
- **Ledgers current**: if this change produced/updated a measured key, the `results/<PAIR>_RESULTS.md` row, the
  `sweeps/<PAIR>_<tf>.md` ledger line, and (if a book was frozen) `MODEL_REGISTRY.md` + `books/INDEX.json` are
  all updated — that recording is owned by `strategy-eval` §5–7; confirm it happened.
- **Issue closed**: if the work came from a GitHub issue, confirm the issue has a delivery comment and is closed.

## 9. Summarise

Return to the user:

- The pushed commit (`git log --oneline -1`) and what shipped (1–2 lines).
- Scope tag; for `books/results`, the verdict and the binding-year number with its result-JSON citation.
- `Shortcuts / hacks taken: <list or "none">`.
- Any check skipped and why (e.g. `Use-case verification: N/A — doc-only`).
- Confirmation the working tree is clean and result JSONs were archived.
- If issue-driven, the GitHub issue number/URL and confirmation it was updated and closed.

## Failure modes

- **`import harness` / `ModuleNotFoundError`:** you ran from `scripts/` or via a path that didn't put `scripts/`
  on `sys.path[0]`. Run `~/binary-algo-venv/bin/python scripts/<name>.py` from the repo root.
- **A number looks too good (AUC ~0.7 that should be ~0.55):** almost always a leakage trap (ffill-flat mirage,
  bar-shift mislabel, thin-coverage pocket). Run the step-5 leakage pre-flight before believing it.
- **A frozen book won't reload:** the `strategy.json` filename differs per book — glob `*strategy*.json`, don't
  hardcode (this was a real README bug). Average all `*_lgb.txt` seeds before the confidence gate.
- **Result JSONs piling up at root:** step 8 not done — `mv *_result.json results/json/`.
- **Push rejected (non-fast-forward):** another machine pushed; `git pull --rebase`, re-gate, re-push. Never
  force-push `main`.
- **A CPCV edge that decays on a frozen-past holdout:** pooled-CPCV non-stationary-feature memorization
  (trap #9). Pooled CPCV is necessary, not sufficient; confirm calendar/seasonal/slow features per-year.

## Discipline to honour during implementation

Step 4/5 must respect the canonical rules in `CLAUDE.md` (run-from-root, no cross-tf pooling, Tier-1 vs Tier-2
filing, certification bar, the already-exhausted list) and the full evaluation protocol in the `strategy-eval`
skill. Do not duplicate those rules here — reference the canonical home.
