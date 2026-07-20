# USDCHF 15m prospective-shadow contract

> **SCOPE: USDCHF·15m.** Fixed prospective procedure only. This document records no
> current-edge, realized-economics, profitability, publication, routing, or activation claim.

## Purpose and claim limit

Issue #19 compares the exact inactive candidate
`USDCHF.m15ny_xpair_seedens.r202605.v1` with the exact active comparison book
`USDCHF.m15ny_xpair_seedens.v1` on fresh post-T0 observations. It reports UP and
DOWN separately; COMBINED is context only, and neither side inherits the other
side's accuracy.

The strongest permitted positive result is prospective public-Deriv,
tick-settled directional shadow support at the fixed `0.541` breakeven proxy,
evidence class `OUTCOME_ONLY_TICK_PROXY`. Monetary fields remain null. This is
not quote-conditioned, fill-conditioned, demo-settled, or venue-settled
evidence, and it grants no book publication, runtime route, credential, account,
proposal, purchase, or real-money authority.

## Operator

The versioned machine contract is
`scripts/usdchf_m15_prospective_v1_spec.json`; the operator authenticates its
exact bytes before using any bound identity, path, clock, or threshold.

Run from the repository root with the pinned environment. These are the complete
public command surface:

```bash
~/binary-algo-venv/bin/python scripts/usdchf_m15_prospective_v1.py preflight --session-date YYYY-MM-DD [--output <repo-relative-path>]
~/binary-algo-venv/bin/python scripts/usdchf_m15_prospective_v1.py preregister --t0-utc <ISO-8601-Z> --t0-ny <ISO-8601-NY-offset> --preflight <repo-relative-path> --user-approved-t0
~/binary-algo-venv/bin/python scripts/usdchf_m15_prospective_v1.py collect [--prereg-id <64-hex>]
~/binary-algo-venv/bin/python scripts/usdchf_m15_prospective_v1.py status [--prereg-id <64-hex>]
~/binary-algo-venv/bin/python scripts/usdchf_m15_prospective_v1.py analyze [--prereg-id <64-hex>]
~/binary-algo-venv/bin/python scripts/usdchf_m15_prospective_v1.py verify [--prereg-id <64-hex>]
```

After the implementation H0 is committed and pushed, `preflight` uses the newest
prior New-York weekday and requires its full spent session to prove
candidate/incumbent feature, score, gate, clock, and live-tick provenance parity
without computing or displaying outcomes. Its default output is the ignored file
`deriv_data/prospective/preflight/usdchf_m15_prospective_v1_preflight.json`.
`preregister` embeds that complete canonical payload and its SHA-256; the
preflight is neither tracked nor an envelope artifact. The command requires an
operator-proposed, user-approved future T0 at Monday 08:00:00
`America/New_York`, records both the exact UTC instant and DST-correct New-York
offset, and creates the preregistration plus dependent collection access
receipt. Registration must occur within 24 hours of the spent-session preflight.
The receipt binds the VPS host identity and both pinned dependency authorities
and installed versions. The preflight must actually observe and reject at least
one recovered or malformed quote row, not merely assert that the filter exists.
H1 has H0 as its sole parent and changes only those two evidence objects; T0 is
after the H1 push.

The service invokes bare `collect`. Omission of `--prereg-id` is deliberate: the
operator discovers exactly one launchable preregistration and fails when there
are zero or multiple candidates. `status` is outcome-blind. `analyze` is allowed
only at the fixed cutoff and enforces collection receipt -> standalone sealed
source -> strict-new analysis receipt -> reducer -> terminal ordering. The one
exception is a durable pre-seal protocol failure: collection receipt -> durable
INVALID record -> no-source result -> terminal, with no sealed source, analysis
receipt, or source read. A completed `verify` authenticates the terminal without
reopening its source packet. There are no manual backfill, supply, source-seal,
T0-reset, threshold change, or optional-relook commands.

Contract failures use exit status 2, which the unit does not restart;
unexpected process failures remain restartable. A first launch at or after T0
is rejected rather than converted into catch-up observations.

## Capture and storage contract

The observer reads only the existing local stores:

- `deriv_data/candles_1m_daemon/` for the seven-pair feature inputs;
- `deriv_data/ticks_1s/_pages/USDCHF/` plus
  `deriv_data/ticks_1s/USDCHF_progress.json` for persisted live ticks.

It never compacts, backfills, repairs, or requests history. Only finite,
positive quote/bid/ask rows with strict increasing, duplicate-free clocks are
admitted; recovered or otherwise non-finite bid/ask rows are rejected. A
missing persisted observation at its deadline is `UNSETTLED`, not an invitation
to fetch or reconstruct it. Each tick read snapshots the provider's committed
page inventory and progress identity, rejects missing, duplicate, malformed, or
mutated committed pages, and allows only genuinely later append pages on a later
read. It selects only page ranges that can contain registered settlement
windows, so the six-month cap does not require retaining the full public stream
in memory.

Rolling candle-store absence, staleness, and incomplete live availability are
transient `UNSCORED` conditions. A concurrent atomic store replacement receives
one clean retry. A stable malformed schema, clock, or non-finite/non-positive
price source is a protocol failure, not missingness.

Each preregistration owns one ignored SQLite database:

`deriv_data/prospective/USDCHF.m15ny_xpair_seedens.r202605.v1/<prereg-id>/trial.sqlite`

The live database stores outcome-blind identities, feature hashes, scores,
gates, directions, clocks, exclusions, and lifecycle state only. It never
stores settlement prices, correctness, returns, payouts, or money. At the fixed
cutoff, the writer stops and the collection-authorized supplier creates one
strict-new standalone snapshot, selects only the preregistered entry/expiry tick
rows, validates the schema/counts/integrity and absence of WAL/SHM sidecars,
then closes and hashes the packet before analysis access exists.
The logical seal also binds a canonical SHA-256 over the exact selected
settlement rows, including quote, bid, and ask values; recovery must reproduce
that digest before adopting or publishing a packet.

The decision grid is weekday completed minutes from 08:00 through 16:34
`America/New_York`. Entry targets `decision close + 1s`; expiry targets that
fixed entry target plus 900 seconds, with the registered ten-second tolerances.
Entry delay never shifts expiry. Candidate and incumbent use separate
first-come, 900-second chronological schedules on one common settled grid.
Ties lose for either direction.

## Fixed look and lifecycle

There is one outcome look: the first complete Friday 17:00 New-York week end at
or after eight complete weeks with at least 100 settled scheduled candidate
decisions in each side, capped at 26 complete weeks. Counts, elapsed weeks,
freshness, missingness, clock defects, scheduled-side counts, and settlement-row
availability may be shown before analysis access; prices, correctness,
accuracy, yield, or confidence bounds may not.

Before a logical seal exists, post-T0 identity, code, spec, package, provider,
clock, access, or no-buy drift is durably terminal `INVALID`; it never resets T0.
`status` can expose that fixed failure under drift, but exact H1 bytes must be
restored solely to publish or verify its no-source terminal. Restoration cannot
resume collection or create a new look. A launch missed before T0 is abandoned
and requires an explicitly linked successor with a later future T0.

After the logical seal is committed, an interruption or worktree drift cannot
replace it with `INVALID`. The process exits retryably; an exact-H1 restart must
finish the same digest-bound packet and evidence publication without admitting
new observations. Terminal statuses are `SHADOW_SURVIVOR_INACTIVE`,
`SHADOW_REJECTED`, `INCONCLUSIVE`, or `INVALID`, and every terminal retains
`activation=false`.

## Buy-incapable service boundary

`ops/deriv-usdchf-prospective.service` has no `EnvironmentFile`, unsets every
credential and buyer-mode variable rejected by the operator, clears ambient
`MX_HOR`, makes the private binary-algo configuration directory inaccessible,
and permits only Unix-domain socket creation. It reads the candle/tick stores,
writes this candidate family's ignored prospective-state root plus the fixed
sealed-source evidence object at cutoff, and lets
`deriv-market-stream.service` own the public network connection. It neither
starts, stops, nor reconfigures that service and has no dependency on the supervisor,
queue, demo executor, or credentialed trade executor.

Before installing the unit, create its narrow writable root so the systemd
sandbox can bind it:

```bash
install -d -m 0700 /home/sean/git/binary-algo/deriv_data/prospective/USDCHF.m15ny_xpair_seedens.r202605.v1
install -D -m 0644 ops/deriv-usdchf-prospective.service ~/.config/systemd/user/deriv-usdchf-prospective.service
systemctl --user daemon-reload
systemd-analyze --user verify deriv-usdchf-prospective.service
```

Enable or start the service only from the exact clean H1 and only after all
pre-T0 gates pass. Inspect the loaded fragment, command line, sandbox, writable
paths, and environment variable names (never secret values) before T0.

## Verification and protected surfaces

The implementation gate is:

```bash
~/binary-algo-venv/bin/python -m py_compile scripts/usdchf_m15_prospective_v1.py scripts/test_usdchf_m15_prospective_v1.py
~/binary-algo-venv/bin/python scripts/test_usdchf_m15_prospective_v1.py
~/binary-algo-venv/bin/python scripts/test_m15_book_refresh.py --skip-null
~/binary-algo-venv/bin/python scripts/test_evidence_store.py
~/binary-algo-venv/bin/python scripts/evidence_store.py verify-store
~/binary-algo-venv/bin/python scripts/check_agent_contracts.py
git diff --check
```

This procedure does not modify or register `books/**`, `books/INDEX.json`,
`MODEL_REGISTRY.md`, or `TARGET_BOOKS`; it does not modify issue-#18 or issue-#9
artifacts, active runtime code/units/databases, credentials, or any real-money
surface. The still-deferred venue-settled demo trial, publication, cadence,
alpha-compiler, and real-money-canary capabilities remain recorded in
`docs/IDEAS_LOG.md`; none enters this trial's acceptance path.
