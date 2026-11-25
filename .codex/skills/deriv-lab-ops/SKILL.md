---
name: deriv-lab-ops
description: >-
  Operate the Deriv-feed research program's lab, which lives outside the
  repository: establish its current state, register and queue a run, run it
  under the lab's limits, recover an interrupted lane, get finished results
  recorded in binary-algo, stop and resume the result recorder, file a
  reviewed specification or registration, and carry out an owner-authorized
  holdout look, staged install, or tick-archive operation. Use for "queue the
  next arm", "what is running in the lab", "resume the lab", "record the
  finished runs", "pull the latest Deriv ticks", or taking the program over
  from an earlier session. Evaluation discipline stays in strategy-eval.
---

# Deriv lab operations

The program's rules live in documents that change by dated owner decision. This skill says where
they are, the order to read them in, and the rules that do not change. It never restates a number
that an owner decision can move: look caps, windows, and releases up; do not recall them.

**Sealed data stays sealed.** Development work stays inside the governing development window and
never opens `holdout/`, `cache/dataset_full_*`, ticks from the start of the sealed window onward, or
quote-survey rows. Outside an authorized look, only the operational checks the governing decision
permits may touch them, and they report names and counts, never values. Give a reviewer the same
exclusions in writing.

| What | Where |
|---|---|
| Program contract: goal, settlement, certification, holdout design, owner decisions | `docs/DERIV_SOTA_PROGRAM.md` |
| Lab rules, layout, commands | `/mnt/data/deriv-sota-lab/README.md` |
| Log of record: registrations, dated owner decisions, current state, open questions | `/mnt/data/deriv-sota-lab/PROGRESS.md` |
| Latest operator handover | `/mnt/data/deriv-sota-lab/handover/` |
| Holdout register (binding) and runbook; read only for an authorized look | `/mnt/data/deriv-sota-lab/holdout/` |
| Staged builds: approvals, hashes, install procedures | `/mnt/data/deriv-sota-lab/staged/README.md` |
| Deriv tick archive, its tools and connection rules | `/mnt/data/deriv/README.md` |
| Records rendered from lab results | `results/json/`, `results/<PAIR>_RESULTS.md`, `sweeps/<PAIR>_15m.md` |

Terms: an **arm** is one registered test variant; a **lane** is a queue or driver together with the
workers that take its jobs.

## 1. Establish the current state first

1. Read the log by heading (`grep -n '^## '`), not end to end: the `Owner decisions (<date>)`
   sections from the newest backwards until the question is covered, the registration section for
   the work at hand and everything recorded after it, the end of the log, and
   `Open questions for the owner`. Owner directions are also written inline in the newest
   sections, such as a pause or the decisions quoted at the head of a registration; search for
   them.
2. **The newest dated owner decision wins** for the rule it changes. Earlier text is not rewritten:
   the lab README, the program document's opening, and older handover notes still state rules that
   later decisions amended. Quote the decision that governs and name the rule it replaces.
3. The log, a handover note, a memory, and this skill describe the past, and work may have been
   done since the log's last entry. Verify the present: `systemctl --user list-units 'deriv-*' --all`
   and `systemctl --user show <unit> -p ActiveState -p SubState -p FreezerState`, the queue files,
   `tools/lab_test.sh status`, installed tool versions and file times against the log's last entry,
   and `git --no-optional-locks status --short --branch` in the repository. Report what the log does
   not record.
4. **Honor a pause.** A unit can be frozen by an owner pause or by the health guard under memory or
   temperature pressure; check the pause record, the guard's events in `queue/health_events.log`,
   and whether the guard itself is running, since a running guard can thaw a lane. When the cause is
   unclear, treat it as a pause. Under an owner pause do not thaw, restart, re-queue, or take a lock
   until the owner lifts it. The pause entry lists its own resume steps and what the pause costs;
   report costs that are coming due. Asked to queue work during a pause, prepare the registration
   and the queue line in the response and change nothing.
5. **Deriv traffic comes only from the Mexico VPS** (`AGENTS.md`, owner decision of 2026-11-25). A
   download, survey, or probe that older lab notes plan for this machine does not run here: not the
   quote-survey restart in the pause entry, not the probe or cross-history downloads in
   `staged/README.md`. Run such work from the VPS under its own rules, or take it to the owner.

## 2. Rules for every run

- **Runs live in the lab.** One directory per test under `tests/<test_id>/`, driven by
  `tools/lab_test.sh start | run | finish | clean | status`. Never in the repository, never in a
  temporary directory, never inside the tick archive, which experiments only read. A run is
  offline: no network, no trading, inputs read-only, writes only in its own directory and its
  result handoff.
- **Data roles.** Every fold and forward reading used as Deriv-feed evidence, and every certificate
  and holdout score, is on Deriv-feed rows. Other history may fill only the roles the arm's
  registration names (fit, stopping, calibration, pretraining, or a separately registered
  diagnostic), and only as the latest owner decision allows. That a tool for it exists is not
  permission.
- **Register before the first run.** Each arm is entered in the log first, with its test, tag,
  tool, exact arguments, dataset manifest checksum, pairs, horizon, sides, seeds, all coverages and
  the one operating coverage, and how its result will be read: the falsifier, the terminal rule,
  and the budget. The active campaign names its own registration tool; for the frozen
  round-6 campaign, from the lab root:
  `~/binary-algo-venv/bin/python -I -B tools/campaign_check.py add <addition.json>` (dry run),
  again with `--write`, then `check`, and `check --complete` at closure. A choice made after
  outer-fold outcomes were seen is a new arm and counts. Nothing leaves the registry: a failed,
  stopped, or abandoned run keeps its status, and every registered arm ends as a result, a
  recorded failure, or a numbered `not run` status.
- **Compute.** One heavy CPU job at a time unless an owner decision says otherwise. Path workers,
  GPU lanes, the light lane, and every fit and GPU-hour cap are owner decisions recorded in the
  log. Before starting heavy work check the running processes and job scopes yourself; a runner
  only knows its own jobs. Charge recorded spend, jobs already running or admitted, and the
  proposed job against each cap before admitting it; a cap guard holds pending work but does not
  stop running jobs. Keep held and exclusive queue lines as they are, and ask the owner before a
  cap would be exceeded.
- **Shared tools.** Build and check a changed tool in a staged copy, get an independent review when
  it touches shared evaluation code, and swap it in with `mv`. Never edit a tool in place while a
  queue runs on it.
- **Deletion.** Once a test's outcome is recorded and its processes have exited, clean it with
  `tools/lab_test.sh clean <test_id>`. Any manual removal names one literal absolute path and is
  never re-issued in a different form after a refusal.
- **One Deriv connection at a time.** Only a connector that implements the archive's shared
  admission state may connect, under `/mnt/data/deriv/.connection.lock`, and only from a host the
  owner's rule allows (section 1, step 5). Verify that the connector takes the lock itself and
  attaches the admission state before it connects; a command that only wraps `flock` around an
  older script does not. Follow `/mnt/data/deriv/README.md`, including which tools it says must not
  be started, and skip the older connection commands still printed there (the quote survey and
  `reply_shape_probe.py`). A connector that finds the lock held sends nothing and ends with status
  75. No new data acquisition without the owner's go-ahead.
- Write every time in UTC from `date -u`, never from memory.

## 3. Queue and run

- Read a lane's scripts and units before touching it: whether it takes jobs from a pending file or
  runs a fixed list, its workers, its locks (some are shared between lanes), its stop mechanism, its
  logs, and its result destination. The main heavy lane is `queue/`:
  `pending.txt` holds one shell command per line, the runner takes the first line under
  `queue/.lock`, and `touch queue/STOP` lets it exit between jobs. Change `pending.txt` only while
  holding that lock.
- Queue lines are built from the registered arguments with the lane's batch driver, such as
  `tools/run_batch.sh` (`<experiment id> <horizon> "<PAIR ...>" [arguments]`, harness chosen by
  `HARNESS`), which runs each pair under the timeout and memory cap, copies the result JSON to
  `results/json/`, records the outcome, and cleans. A `done` record does not prove the copy
  succeeded: check that each result file exists and is complete.
- Do not leave queued work unwatched. Arrange to be told when each job ends, and act then.

## 4. Recover an interrupted lane

1. Check the processes and units first; a note that says something is running may be stale.
2. Once the runner and its job scopes have exited, reconcile the arms released to that lane with its
   pending, held, running, done, and finish records and the results at its real destination. A job
   taken from the queue can be lost before any record of it is written. A failed finish may be a
   health-guard interruption: use the exit, the guard's events, and the registered retry rule, and
   treat a recorded failure as final unless that rule allows a retry.
3. Before removing or re-queueing anything, check the destination and any result left in the test
   directory: a partial file at the destination makes some lane jobs skip, and the test directory
   may hold the only complete copy. Then put each pair that may be retried back once, through the
   lane's own mechanism, keeping any held or exclusive status, and remove the dead test directory by
   the deletion rule before the same tag and pair runs again.
4. Restart lanes as durable user units so they outlive the session, and record what was restarted.

## 5. Get results recorded

- The lab's recorder is an automated writer to `main`. When result files arrive it checks their
  names and contents, compacts large ones and archives the originals, regenerates the ledger blocks
  with `scripts/update_records.py`, requires that nothing else changed, makes one commit for the
  batch listing a reading per result, and pushes. Its acceptance does not prove an arm was
  registered; the campaign record does.
- **Stopping and resuming it.** Its unit's `ExecStart` names its script, and the script's loop
  condition names its stop file. Frozen under an owner pause, leave it alone: it cannot write.
  Running, create the stop file, wait until the loop has exited, and do the tracked work.
  Afterwards, with no pause ordered meanwhile and the checkout clean and equal to a freshly fetched
  `origin/main`, remove only a stop file you created, start it again, and confirm it stays running.
  Restore only what you stopped.
- When it refuses or stops, read its log. A failed run can leave archive copies, staged changes, or
  a local commit behind: establish what each is, verify whether that exact commit is on the remote
  before any retry (`git-workflow`), and do not hand-commit around it.
  A change to what it accepts or to how ledgers render is a tracked change for `dev-cycle`.
- Results the recorder does not cover are recorded through `strategy-eval`.
- A pilot never certifies. Report the strict status that `strict_certificate` in
  `scripts/certificate.py` returns for the side at its registered operating coverage; name a
  drift-amended reading or a holdout decision separately, as such. A development certificate is
  necessary, not sufficient: promotion and deployment also need the campaign's gates and the
  holdout decision. State negative results directly.
- Afterwards add a log bullet for every completed arm and give the owner a per-key scoreboard.

## 6. File a reviewed specification or registration

`feature-dev` hands over the reviewed text as one file, with its SHA-256 and its destination. For a
specification, write exactly those bytes to the named new file and confirm the hash. For a
registration, enter exactly that text in the named log section before the arm's first run, check
the inserted bytes, and record its UTC time. In either case record the path or section, the hash,
the repository revision it was reviewed on, the review verdict, and the owner approval that covers
it. If the destination or the approval does not exist yet, change nothing and ask.

When `dev-cycle` ships a change governed by a lab specification, append its completion to the log:
the specification's path and hash, the approval, the landed commit, the checks and review outcome,
shortcuts, and any operator action still owed.

## 7. Holdout looks

A look is opened only at the owner's explicit direction for that named look. The keys, the number
of looks, the windows, and the decision rule come from `holdout/REGISTER.md`, which binds, and the
steps from `holdout/RUNBOOK.md`. Preparation that reads the full window is part of the look, not
ordinary preparation. Use only the frozen, hash-registered evaluator copies. A refusal stops the
procedure: investigate, and never edit an input to make it pass. Keep every earlier look's
outputs; the next look verifies against them.

## 8. Staged installs and archive work

`staged/README.md` lists each approved build with its preconditions, install order, and expected
check counts, and its opening lines can lag reality. Compare the installed files' hashes with the
approval record first, to learn whether an install already happened. Carry out only the build that
was authorized: stop before another build, a venue probe, a download, or a registration unless the
current authority names it. Confirm every precondition now, back up what will be replaced, and run
each listed check from the place and with the exact invocation the approval records. Tick-archive
work uses only the tools the archive README names.

## 9. Owner decisions

Stage releases and reopenings, cap increases, new data sources, protected looks, venue access, and
any change to the certification standard are the owner's. Put each to the owner as a
self-contained question with options and a marked recommendation, then record the chosen label
verbatim in the log under a dated `Owner decisions` heading.

## Report

Lead with the per-key scoreboard and what changed. Then: what ran and finished, what is queued or
paused, budget used against each cap from the ledgers, anything left unrecorded, and anything the
log does not yet say. Ask open owner decisions as questions; do not list them.
