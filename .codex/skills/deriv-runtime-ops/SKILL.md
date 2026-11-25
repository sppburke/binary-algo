---
name: deriv-runtime-ops
description: >-
  Inspect, diagnose, halt, deploy to, or resume the binary-algo Deriv demo
  runtime: its systemd user units and timers, the signal queue, the single
  credentialed buyer, the account lock, public observers, candle stores, and
  the parity gate. Use for "is the executor running", "why was this signal
  skipped", "restart", "stop buying", "deploy to the trading host", "rotate
  the token", probe or timer status, or to state what a tracked change means
  for the running services. Changing runtime code is dev-cycle; this skill
  never authorizes real-money trading.
---

# Deriv demo runtime operations

The runbook of record is `docs/DERIV_DEMO_EXECUTOR.md`; unit templates live in `ops/`; store
ownership is in `docs/DERIV_DATA_STORE.md`. This skill routes to them, carries the rules that
prevent a wrong or duplicate buy, and separates looking from acting. Where a document and the code
disagree, the unit directive and the code win; report the stale passage. A unit, script, or runbook
this skill does not name is outside it: read its own runbook and get the owner's direction first.

## Authority

- **Demo only.** The runbook states that this tooling does not enable real-money execution, and
  the client refuses a buy-capable connection whose one-time address lacks `/demo`. Nothing here,
  and no test result, authorizes a real account.
- **Reading is free; acting is separate.** A request to change code does not authorize a service
  start, stop, or restart, a timer change, setting or clearing a `KILL` file, fetching or moving a
  host's checkout, a private configuration edit, or a demo buy. Each needs the owner's go-ahead for
  that action on that host. In an incident, such as a non-finite quote on the buy path, recommend
  the halt at once instead of waiting for a report.
- **Where traffic may come from.** `AGENTS.md` allows Deriv connections, public endpoints included,
  only from the Mexico VPS. The near-close probe unit's development-host deployment predates that
  rule: do not arm it on the workstation. Take any other exception to the owner.
- **Secrets.** Never read, print, copy, or log the private environment file
  `~/.config/binary-algo/deriv-demo-executor.env`, an account ID, a Deriv token, or a raw one-time
  connection address. Variable names come from `ops/deriv-demo-executor.env.example` and the code.
  Report unit states, process IDs, event names, and reasons; do not paste raw authentication
  errors or full command lines. Secrets, runtime logs, and `deriv_data/` stay out of Git.
- Tracked changes go through `dev-cycle`; a measured audit of shifted lanes is `strategy-eval`; Git
  and host-checkout mechanics are `git-workflow`.

## Know what each unit can do

| Unit | Role | Gets the Deriv token and account ID | Can buy |
|---|---|---|---|
| `deriv-market-stream.service` | tick store, shifted bars, daemon candle store | no | no |
| `deriv-runtime-supervisor.service` | producer: scores books, writes selected-side signals to the queue | no | no |
| `deriv-trade-executor.service` | the buyer: claims signals, final payout gate, buys, reconciles | yes | yes |
| `deriv-quote-audit.service` | public quote sampler, one CALL and one PUT lane per pair | no | no |
| `deriv-production-candle-refresh.timer` and its service | designated writer of `deriv_data/candles_1m/` | no | no |
| `deriv-usdchf-prospective.service` | prospective trial collector: reads the stores, writes trial state and sealed evidence | no | no |
| `deriv-probe-g-nearclose.timer` | near-close probe; its post-step commits and pushes a result | no | no |
| `deriv-probe-g-verify.timer` | checks for a same-day local commit on the result path (not its content, not the remote); can post an issue comment | no | no |
| `deriv-phase5-readiness.timer` | coverage notifier; connects to the trading host and can post an issue comment | no | no |
| `deriv-demo-executor.service` and `.timer` | deprecated one-shot, cannot buy, still writes the candle store; starting it stops the buyer (`Conflicts=`); keep both disabled | no | no |

The table describes the checked-in templates. The non-buyer units that load the private file drop
the token and account ID with `UnsetEnvironment`; the prospective collector and the probe,
verifier, and notifier units load no private file. Keep each unit's boundary when editing it.
Before interpreting evidence, name the host (`hostname`) and read what is really
there: `systemctl --user list-unit-files 'deriv-*'`, `systemctl --user list-units 'deriv-*' --all`,
`systemctl --user show <unit> -p ActiveState -p SubState -p UnitFileState`, and
`systemctl --user cat <unit>` (the file on disk, which the manager may not have reloaded).
`list-units` without `--all` omits inactive units. Per the unit comments, the buyer and its
producer are deployed on the trading host, and the notifier and probe units on the development host.

## Look without touching

Safe: the `systemctl` reads above, `systemctl --user list-timers --all`, `pgrep -f deriv_`,
selected events and fields from `journalctl --user -u <unit>` and the JSONL logs (filter before
displaying; error text can carry raw responses), and the signal queue through a read-only
connection (Python `sqlite3.connect("file:<db>?mode=ro", uri=True)`, `SELECT` only; default
`deriv_data/runtime/trade_queue.sqlite`). For a skipped signal, find its row in `signals` by
`(pair, lane, offset_id, signal_close_utc)` and read `status`, `terminal_reason`, `terminal_code`,
and `terminal_source`, then the log events for that signal ID. A signal the producer rejected has
no row; read the producer's skip events.

Not read-only, despite their names:

- `scripts/deriv_trade_executor.py --run --duration-seconds 1`, which the runbook lists as an auth
  health check, authenticates, takes the account lock, connects, and starts the live allocator
  before its one-second timer. Run by hand it skips the unit's pre-start guard and reads `KILL`
  from `logs/deriv_trade_executor/`, not from the service's directory. It is a buyer start.
- `scripts/deriv_trade_queue.py --assert-pragmas` opens a writing connection and runs schema
  migration.
- `scripts/deriv_backfill.py health` creates its output directory before checking it. Point it only
  at an existing store, with that store's documented freshness limit.
- `scripts/fold_probe_g_capture.py --dry-run` still rewrites the tracked result file when there is
  a capture to fold.
- Gates, smokes, soaks, and replays can write result files, rebuild a store, or open connections;
  read a command's outputs and network use before running it. The allocator replay defaults to a
  tracked file under `results/json/`, and the market-stream smoke deletes and rebuilds its smoke
  store. Run them where a new file cannot disturb another writer of the checkout, and report
  fixture results separately from live evidence.
- The probe, verifier, and notifier units write logs, can commit, push, or post issue comments, and
  each tries to disable its own timer when it succeeds; check the timer itself.

## Rules that hold everywhere

- **One buyer.** `scripts/deriv_trade_executor.py --run` is the only credentialed buyer the runbook
  authorizes. Never run a legacy buy mode: the one-shot's `--demo-buy` and `--auth-smoke` are
  disabled in code, but `scripts/deriv_runtime_supervisor.py --demo-buy` outside queue mode can
  still buy, under a lock of its own that the account lock does not see. The buyer unit's
  pre-start guard only scans local command lines at start; it does not prove exclusivity.
- **Account lock.** The buyer holds one lock file per account under `deriv_data/runtime/locks`
  (`--lock-root`, then `DERIV_RUNTIME_LOCK_ROOT`, override it; every launch must resolve to the
  same path). It refuses to start when the file exists and never checks whether the holder is
  alive. The file records the holder's process ID and when the lock was taken. Establish who holds
  it before the owner authorizes removing a stale lock, and never remove one whose holder is live
  or unknown.
- **Session and cutoff.** New buys are admitted on weekdays from 08:00 to before 17:00
  America/New_York and refused from the last-start cutoff, `LAST_START_NY` in
  `scripts/deriv_runtime_core.py`. The cutoff closes admission only: a candidate already past the
  gate can still be bought afterwards. The producer may enqueue after the cutoff; the buyer rejects
  those signals.
- **Uncertain buys stay uncertain.** A signal in `buy_intent` or `buy_unknown` may have reached the
  venue. Never re-submit it, and keep the queue database and its write-ahead files intact. A
  restart re-claims only `queued` signals and `claimed` ones whose lease is missing or ran out.
  Automatic reconciliation follows recorded contracts of `bought` and `buy_unknown` signals only.
  Any `buy_intent`, even with a contract recorded, and any `buy_unknown` without one, is resolved by
  the owner against the venue.
- **Parity re-arm.** A change to the registry, a book, `live_features.py`, or
  `deriv_runtime_core.py` re-arms the KILL #0 re-check before the next demo-buy session. Shipping
  or deploying the change does not satisfy it, and the buyer does not enforce it at start: keep
  buying halted until the runbook's full-session capture and replay pass on that revision. Never
  overwrite the historical parity result.
- **Stores.** One writer per store. The deprecated one-shot and the production candle refresh write
  the same temporary files: follow the runbook's cutover order exactly, and keep manual backfills
  away from a store whose writer is active. The market stream must be healthy before the supervisor
  starts.
- **Prospective trial.** The trial collector is bound to one exact revision and needs the checkout's
  `HEAD` to equal `origin/main`. While a trial is open, even a fetch that moves `origin/main`, like
  a checkout move, a worktree edit, or a restart, can end it as invalid for good; after its logical
  seal the running collector can finish publication, and an interrupted one is recovered only by a
  restart on the exact revision. Read
  `docs/USDCHF_M15_PROSPECTIVE.md` and its outcome-blind status first, and take an open or unclear
  trial to the owner.
- The runbook lists the known open weaknesses of the shared payout gate.

## Stop buying

1. `touch <log-dir>/KILL` in the buyer's own log directory: the one passed to it as `--log-dir`,
   which the checked-in unit sets to `${DERIV_DEMO_LOG_DIR}/trade_executor`. The supervisor's
   `KILL` file does not stop queue-mode enqueueing, and `logs/paper_trades/KILL` belongs to the
   deprecated one-shot.
2. The switch is read before a candidate's quote is requested. A candidate already past that point
   can still be bought. Signals refused while it is set are closed for good and are not replayed.
3. For a hard stop, also stop `deriv-trade-executor.service` with the owner's go-ahead. Stopping
   the buyer stops its reconciliation too.
4. Confirm the buyer's own state, not only a quiet log or queue: record the halt time, look for any
   `buy_intent_utc` after it in every status, follow it and every intent unresolved at the halt to
   its contract, and list every open or uncertain contract for the owner. Work already past the
   switch can appear after a quiet look.
5. Once `KILL` is cleared, a running buyer can admit any eligible queued signal at its next check.
   Clearing needs the owner's go-ahead, after the queue, the open exposure, and the parity status
   have been checked.

## Authorized host actions

- **Deploy.** A push is not a deployment. Before the checkout moves, confirm the host, the approved
  commit, a clean worktree, no open prospective trial, and the owner's go-ahead for each service
  action. Work out which units are affected from their unit files, start commands, environment,
  books, and loaded code. For a change that re-arms parity, set the buyer's halt first, let
  in-flight buys resolve, and follow the runbook's re-arm procedure, whose capture needs the
  supervisor, buyer, and prospective units inactive; keep buying halted until it passes. Move the
  checkout
  (`git-workflow`), reload the manager when a unit file changed, follow the runbook's order for the
  named change, and restart only the affected units. `restart` also starts a stopped unit: leave an
  intentionally stopped one stopped.
- **Restart.** Outside a documented procedure, a restart needs the owner's go-ahead. The clock does
  not prove the buyer is idle: while buys can be admitted, set the halt first, then confirm that
  in-flight work has drained and list open and uncertain contracts. Take every uncertain one to the
  owner before restarting. Afterwards verify, for this start, that the unit stays active, the
  pre-start guard printed its pass line, `auth_health` reports `ok`, and `exec_client_connected`
  appears. `executor_startup` is written before the lock and the main connection, so it does not
  prove the buyer is running.
- **Credential rotation.** Follow the runbook's rotation item with the owner: stop
  `deriv-trade-executor.service`, have the owner re-mint and update the token in the private file,
  run the auth-health or fake gate, then restart. The auth-health command is the one-second buyer
  start above, and the fake gate (`--fake-gate`) checks fixtures only; agree which one with the
  owner, and confirm the new credential from the buyer's own `auth_health` event after the
  authorized start.
- **Timers that publish.** Enabling the probe, the verifier, or the notifier authorizes their
  stated side effects only when the request covers them.

## Close out

Report the host, the revision running there, each unit's state before and after, the commands run
with their results, any open or uncertain contract, whether the parity re-check is now pending,
and every stale runbook passage found. If a needed procedure is not written down, say so and ask
the owner rather than inventing one.
