---
name: feature-dev
description: Guided planning workflow for a new experiment, method, sweep, new-key bootstrap, or infra feature in binary-algo — understand the repo via parallel agents, ask clarifying questions, design 2-3 candidate approaches (each with a mechanism + sign-invariance note + pre-registered falsifier), finalize the chosen one, and file it as a durable GitHub issue hand-off artifact. Does NOT run or implement (that happens in a separate `strategy-eval` run for experiments, or `dev-cycle` for infra). Use for "plan an experiment", "design a method before running it", "spec out X", "what should we try for <pair>/<tf>", "add this idea to the backlog", or any request that ends at a hand-off artifact rather than at a measured result or shipped code.
---

# Feature / Experiment Planning

You are helping plan a new experiment, method, sweep, new-currency bootstrap, or infra change, and capture the
agreed plan as a durable hand-off artifact. Follow a systematic approach: understand the repo deeply, identify
and ask about all underspecified details, design well-mechanised candidate approaches, then file the chosen plan.
**Running and implementation are out of scope** — this skill ends at the filed artifact, which is picked up by
`strategy-eval` (to run an experiment under the evaluation discipline) or `dev-cycle` (to build infra).

## Core Principles

- **Ask clarifying questions**: Identify all ambiguities, edge cases, and underspecified behaviours. Ask
  specific, concrete questions rather than assuming. Wait for answers before designing.
- **Understand before acting**: Read existing patterns first. Most methods are already implemented and
  horizon-parameterized — check `docs/METHODS_CATALOG.md` and the per-key `results/<PAIR>_RESULTS.md` before
  proposing anything "new". **Don't plan to redo a measured key.**
- **Read files identified by agents**: When launching agents, ask them to return the 5–10 most important files;
  read those after the agents return to build real context.
- **Mechanism first**: A candidate is only worth filing if there is a plausible mechanism by which it carries
  *direction* (sign), not just *magnitude* (size). The sign-invariance theorem kills most "direction" ideas that
  are really vol/complexity statistics — flag it at design time, not after a wasted run.
- **Evidence-first**: every factual premise about the repo (what exists, what was already tried, what a number
  was) traces to a Tier-1 source per `docs/_EVIDENCE-FIRST.md`. No "probably already tried" — grep it.
- **Use TodoWrite**: Track progress through the phases.
- **GitHub issue hand-off**: The durable output of this skill is a GitHub issue in `sppburke/binary-algo`, no
  questions asked. If the request references an existing issue like `#<issue-number>`, treat it as
  `https://github.com/sppburke/binary-algo/issues/<issue-number>`, read it with `gh issue view <issue-number> -R sppburke/binary-algo`,
  and update/comment that issue instead of creating a duplicate. Otherwise create a new issue with the finalized
  plan.
- **GitHub auth**: Use the locally configured `sppburke` GitHub token/account. Prefer commands with
  `-R sppburke/binary-algo`. If `gh` is on the wrong active account or repo lookup fails, run
  `gh auth switch -u sppburke` and verify with `gh repo view sppburke/binary-algo --json nameWithOwner,url`.
  Never print, extract, or paste the token.

---

## Phase 0: Model Preflight (HARD GATE)

**Goal**: Ensure this skill runs on Claude Opus — planning and mechanism-vetting quality degrade on smaller models.

**This is a hard gate. Do not run any other phase, tool call, agent dispatch, or clarifying question until it passes.**

**Actions**:
1. Inspect the current model from the system environment (the "powered by the model named …" line). Treat this as
   Opus-only — any model whose name does not start with "Opus" (e.g. Sonnet, Haiku) fails the gate.
2. If the current model is **not** Opus:
   - Print exactly this message and then **STOP** — emit no further text, no tool calls, no agent dispatches,
     no TodoWrite, nothing:

     > ⛔ `feature-dev` requires Claude Opus, but the current model is **<detected model>**.
     >
     > Please switch to an Opus model with `/model` and then re-state the request so the `feature-dev` skill
     > re-triggers. I will wait — I am not going to proceed on a non-Opus model.

   - Substitute `<detected model>` with the actual model name from the environment.
   - Do **not** continue automatically when the user replies. The user must re-state the request after switching
     so the skill re-triggers; a follow-up message alone is not sufficient.
3. If the current model **is** Opus, record it in the todo list ("Phase 0: Opus confirmed — <model name>") and
   proceed to Phase 1.

---

## Phase 1: Discovery

**Goal**: Understand what needs to be planned and pin the unique key.

Initial request: the user's description from the invoking message. If they only said "use feature-dev to plan X"
without specifics, jump to the clarifying questions in step 2.

If the invoking message contains a GitHub issue reference like `#<issue-number>`, fetch that issue first and use its title/body
as the initial request. The issue number always refers to `sppburke/binary-algo`.

**Actions**:
1. Create the todo list with all phases.
2. Pin the scope. For an experiment/method/sweep, that means the unique key(s) `(currency, timeframe, side)` and
   the horizon knobs (`MX_HOR` minutes for bar models; `HS`/`HSEC` for tick models). For an infra feature, the
   module/script surface it touches. If unclear, ask the user:
   - What problem / which key? What would "it works" look like (a number? a reusable module? a frozen book?)?
   - Any constraints (data on disk vs acquisition, compute, deriv-faithfulness)?
3. Summarize understanding and confirm with the user.

---

## Phase 2: Repo Exploration

**Goal**: Understand the relevant existing code, prior results, and conventions at both high and low levels.

**Actions**:
1. Launch 2–3 explorer agents in parallel (`Explore` / `general-purpose`). Each targets a different aspect and
   returns 5–10 key files to read. Example prompts, retargeted to this repo:
   - "Find prior experiments related to [idea] for [pair]: search `results/<PAIR>_RESULTS.md`, `sweeps/`,
     `docs/METHODS_CATALOG.md`, `docs/IDEAS_LOG.md`, `docs/DIRECTION_FINDINGS.md` — what was tried, the verdict,
     and WHY (diagnosed cause: common-factor / regime / info-bound / overfit). Return the exact ledger lines."
   - "Map the reusable building blocks relevant to [idea]: trace `harness.py`, `min1_production.py`
     (`wc_ret`/`boot`/`nonoverlap_chrono`/`mk_lgb`), the per-horizon `m{5,10,15,30}_production.py`,
     `crosspair.py`, `cpcv_certify.py`, `manifest.py`. What can be imported vs must be written?"
   - "Identify the leakage traps and discipline that apply to [idea] from the `strategy-eval` skill §2/§3 and
     `docs/_EVIDENCE-FIRST.md`. Which traps does this idea risk?"
2. Read every file the agents flag. Build deep context — especially the **incumbent** for the key (best
   combination in `MODEL_REGISTRY.md` / `books/INDEX.json`) and its binding (worst held-out) year.
3. Present a comprehensive summary: what already exists, what's already been tried-and-killed (with the why), the
   incumbent to beat, and which building blocks are reusable.

---

## Phase 3: Clarifying Questions

**Goal**: Fill gaps and resolve all ambiguities before designing.

**CRITICAL**: One of the most important phases. DO NOT SKIP.

**Actions**:
1. Review the Phase 2 findings against the original request.
2. Identify underspecified aspects: the exact key(s) and sides, the label/settlement (deriv-faithful `wc_ret`?
   ties-lose?), the evaluation target (direction win-rate vs magnitude AUC), the data substrate (on-disk
   `features/` vs tick vs acquisition), the incumbent to beat and on which year, the KILL conditions, and whether
   the deliverable is a number, a reusable method, or a frozen book.
3. **Present all questions to the user in a clear, organized list.**
4. **Wait for answers before designing.**

If the user says "whatever you think is best", provide your recommendation (grounded in the Phase 2 evidence) and
get explicit confirmation.

---

## Phase 4: Approach Design

**Goal**: Design 2–3 candidate approaches with different mechanisms / trade-offs.

**Actions**:
1. Launch 2–3 design agents in parallel with different focuses, e.g.: **minimal** (smallest retarget of an
   existing method — reuse `METHODS_CATALOG` entry + `MX_HOR`, maximum reuse), **novel-mechanism** (a genuinely
   new lever or cross-disciplinary transfer — physics / info-theory / point-process / causal-discovery — with an
   explicit direction mechanism), and **combination** (gate / blend / stack / regime-route / cross-horizon of
   existing certified edges — often the cheapest novelty).
2. For **each** candidate, the design must state:
   - **Mechanism** — why this could carry *sign*, and how it survives (or doesn't) the sign-invariance theorem.
     If it's really a magnitude/gate idea, say so and reframe it as `|ret|≥Q` magnitude.
   - **Data + reuse** — on-disk substrate vs acquisition; which building blocks it imports.
   - **Pre-registered falsifier** — exact KILL conditions (e.g. "KILL if VAL dirAUC ≤ 0.515, OR no held-out year's
     moved-acc CI95-lower clears breakeven 0.541"), sized to the candidate's prior so low-prior ideas die fast.
   - **Leakage risk** — which §3 traps it must avoid, and the discipline it must run under (§2).
   - **Incumbent comparison** — what it must beat (the best combination for the key) and on which binding year.
3. Review all candidates; form your opinion on which fits best **for this key on this data** (honest prior:
   ≤5m direction is near-efficient; 15m direction ~0.58 robust; magnitude is the one certified edge). Present:
   brief summary of each, trade-offs, **your recommendation with reasoning**, and the concrete differences.
4. **Ask the user which approach they prefer.**

---

## Phase 5: Finalize and File

**Goal**: Capture the agreed plan as a durable hand-off artifact — printed to screen AND filed as a GitHub issue
in `sppburke/binary-algo`. **Do not write or run any experiment code in this skill.**

**Prerequisite**: Phase 4 ended with the user choosing one approach. If approval isn't explicit, ask once and wait.

**Actions**:

1. **Compose the finalized plan** from prior phases. Pull only what was agreed; invent no new scope. Sections,
   in order:
   - **Title** — short imperative summary (e.g. "GBPUSD 15m: test Hawkes branching-ratio gate on the xpair book").
   - **Key(s)** — `(currency, timeframe, side)` + `MX_HOR`/`HS`.
   - **Context** — 1–3 sentences: the problem, the incumbent to beat (number + binding year + book id), why now.
   - **Repo findings** — bulletised Phase 2 highlights: prior-tried verdicts (with the diagnosed why), reusable
     blocks, the relevant ledger/results files.
   - **Resolved questions** — each Phase 3 question with its final answer.
   - **Chosen approach** — name + 1-paragraph description of the picked candidate, including its **mechanism** and
     **sign-invariance note**. List rejected candidates one line each with the reason.
   - **Pre-registered falsifier** — the exact KILL conditions, verbatim, to be written into the result JSON stub.
   - **Implementation outline** — numbered concrete steps (script to retarget/write, settlement, splits,
     evaluation, the certification check if it's a deliverable book).
   - **Files expected to change** — script(s), the result JSON, the `results/<PAIR>_RESULTS.md` row, the sweep
     ledger line; one-line purpose each.
   - **Leakage traps to avoid** — the specific §3 traps this approach risks.
   - **Out of scope** — anything explicitly deferred.
   - **Open risks / follow-ups** — flagged uncertainties or future work.

2. **Print the full plan to screen** as a single self-contained markdown block — a reader who didn't sit through
   phases 1–4 should be able to act on it.

3. **File it as a GitHub issue**:
   - Existing issue referenced (`#<issue-number>`): post the finalized plan as a comment or update the issue body when the
     user asked for an update. Use `gh issue comment <issue-number> -R sppburke/binary-algo --body-file <file>`.
   - No issue referenced: create one with `gh issue create -R sppburke/binary-algo --title "<title>" --body-file <file>`.
   - Include enough detail for `strategy-eval` or `dev-cycle` to pick it up without conversation context.
   - Do **not** ask where to file the plan; GitHub issue filing is the default for this repo.
   - Do **not** append repo backlog rows, `docs/IDEAS_LOG.md`, or `SWEEP_MATRIX.md` unless the user explicitly asks
     for those legacy ledgers in addition to the issue. If a later run produces measured results, `strategy-eval`
     remains responsible for the Tier-2 result ledger.

4. **Report what was filed** — the GitHub issue number + URL. This is the hand-off artifact: running happens in
   `strategy-eval`, infra build in `dev-cycle`.

5. **Mark all todos complete and stop.** The skill ends here. Do not run the experiment, fit a model, write a
   result JSON, or freeze a book — those happen in a separate `strategy-eval` invocation that picks up the backlog
   row.
