> **SCOPE: GENERIC** (one fixed demo Rise/Fall economics reference; no strategy result).

# Deriv demo Rise/Fall economics ledger (ROS-3A)

ROS-3A is the minimum independent reference ledger for one retained 15-minute
Deriv FX Rise/Fall settlement. It distinguishes actual linked venue economics
from outcome accuracy, a tick-direction proxy, a constant-payout stress case,
a quote without a fill, or a framework simulation. It grants no strategy,
model, activation, purchase, or real-account authority.

## Fixed slice and authority

The acquisition is fixed before the record supplier reads terminal source
fields: New York date `2026-07-06`, `USDJPY`, `UP`/`CALL`, 15 minutes, USD
`basis=stake`, demo/virtual funds, and the earliest complete chain ordered by
`(contracts.created_utc, contract_id)`. The acquisition specification binds
the exact exporter, evidence-store, and canonical-JSON implementation bytes;
source paths, source-row key sets, packet schema, selector, equations,
provenance rules, claim limits, and falsifiers are part of its immutable
payload. The accepted exporter is `scripts/deriv_economics_ledger_v2.py`;
`scripts/deriv_economics_ledger.py` is the byte-frozen failed predecessor.

The operational source is read only on the trading VPS. The exporter opens
only the queue through SQLite `mode=ro` plus `query_only=ON` in one snapshot,
and only the exact fixed executor JSONL. It imports no Deriv client, reads no
credential or account environment variable, makes no network request, takes
no buyer lock, and changes no runtime service or source file. A bounded copied
verifier bundle authenticates the acquisition object and its exact artifact
bindings before either source opens.

## Evidence chain and access order

The accepted four stages are canonical `research-evidence-envelope/v1` objects:

| Stage | Kind | Binding |
|---|---|---|
| Acquisition specification | `research.deriv_economics.acquisition_spec/v1` | Outcome-independent supplier authority and exact executing bytes. |
| Sealed source | `research.deriv_economics.sealed_source/v1` | Externally declared packet `ArtifactRef` and non-outcome descriptor; depends on the specification. |
| Source-access receipt | `research.deriv_economics.source_access_receipt/v1` | `spent_may_have_been_read`, published strict-new before the reducing actor reads packet content. |
| Normalized receipt | `research.deriv_economics.normalized_receipt/v1` | Self-contained observation, exact derivation, provenance gaps, and `activation=false`; depends on the access receipt. |

The ignored source packet stays at
`deriv_data/venue_evidence/<sha256>.json`. Its `ArtifactRef` is payload data,
not an envelope artifact, so a fresh clone can verify the tracked store and a
completed resume does not reread the ignored source. A receipt without a
normalized descendant is ambiguously spent and blocks. An exact completed
lineage is recomputed from the normalized receipt with zero packet reads.

The first acquisition specification,
`a0ba3d8dfdb187b2abc9952fa897c8bdf4f9f2815d825757dd19665740c532d4`,
failed closed before packet creation because the exact log contained two
previously unregistered source-key variants. It remains immutable and binds
the frozen v1 exporter at SHA-256
`7163a502ccda0cdcbf6b4ec9eee3665a0e503bd51cbd5be0fd4382a10a00de1e`.
The reviewed v2 successor admits exactly the two observed startup key sets and
the two observed proposal key sets; a third variant still blocks. V2 binds
itself, frozen v1, the evidence store, and the canonical JSON owner, and
depends on the failed specification. No v1 global is rebound or modified.

## Economics boundary

The reducer recognizes four non-interchangeable evidence classes:

- `OUTCOME_ONLY_TICK_PROXY`
- `CONSTANT_PAYOUT_STRESS`
- `QUOTE_CONDITIONED_NO_FILL`
- `VENUE_SETTLED`

Only `VENUE_SETTLED` with the admitted proposal, buy, and terminal chain may
carry realized money. The other classes require `realized_profit=null`.
Money enters the packet as canonical two-decimal strings after
`Decimal(str(SQLite_REAL))` exact-cent validation; the reducer accepts no JSON
numeric money, rounding, exponent form, or non-finite value.

```text
quoted_win_profit     = proposal_payout - proposal_ask
quoted_loss_profit    = -proposal_ask
terminal_implied_cost = terminal_sell_price - terminal_recorded_profit
realized_profit       = terminal_sell_price - terminal_implied_cost
breakeven_fraction    = proposal_ask / proposal_payout
```

The breakeven fraction is retained as exact numerator and denominator.
Proposal and terminal payout fields remain separate. The admitted chain must
satisfy implied cost = recorded buy-price projection = stake = proposal ask,
and recomputed realized profit = recorded profit. A win requires positive
profit and sell price equal to terminal payout; a loss requires zero sell
price and profit equal to negative cost.

The pure reference helper uses `CALL: exit > entry`, `PUT: exit < entry`, and
ties lose. The actual retained chain has no entry or exit tick, so its direction
is explicitly `unverified_not_retained`; the helper's synthetic goldens do not
manufacture venue evidence.

## Commands and verification

Run from the repository root in the pinned environment:

```bash
~/binary-algo-venv/bin/python scripts/deriv_economics_ledger_v2.py seal-acquisition
~/binary-algo-venv/bin/python scripts/deriv_economics_ledger_v2.py export-one --acquisition-spec-id <id>
~/binary-algo-venv/bin/python scripts/deriv_economics_ledger_v2.py reduce \
  --acquisition-spec-id <id> \
  --source-path deriv_data/venue_evidence/<sha256>.json \
  --source-bytes <bytes> --source-sha256 <sha256> \
  --descriptor-json '<non-outcome descriptor>'
~/binary-algo-venv/bin/python scripts/test_deriv_economics_ledger.py
~/binary-algo-venv/bin/python scripts/evidence_store.py verify-store
```

`export-one` is authorized only in the bounded VPS bundle described above;
the repository command is documented for exact invocation, not for local or
real-money acquisition.

## Accepted evidence

The one accepted operational run produced this four-object lineage:

| Stage | Object ID |
|---|---|
| Acquisition specification | `c98d10c42d960940bac542a964ac52465d6efdff7a3df245d106677467be1644` |
| Sealed source | `25389c3a62870718d531fb86f8111d2882cba585c2f02a6d1a9cec44fa272046` |
| Source-access receipt | `f6f0652dfbee3255b999473fed7e1113e352a6020aa3182c25ebabb81554e34f` |
| Normalized receipt | `e2512a4a15626de6007db155aee1de4f9b58a8cc0b68ed1b4f2559ab1684b3a0` |

The ignored sanitized packet is bound by SHA-256
`4f070d540c1d1bd75aa1511e1a7f483db7b5687c6a1ea2b411b7e5ea61b0ba76`.
The full store verifies without that packet. The reducer's completed resume
returned the same normalized ID without a packet read, and independent
calculations from the self-contained receipt confirmed all five registered
equations, the exact money chain, inactive state, venue-settled class, and
unverified tick boundary. Both trading services remained active; no proposal,
buy, API call, credential read, service action, or real-money action occurred.

## Deliberate exclusions

ROS-3A adds no new order, API capture, raw-response hook, multi-chain corpus,
other product or duration, early-sale engine, payout model, staking,
backtester adapter, Rust reducer, database projection, service, scheduler, UI,
strategy evaluation, activation, or generic workflow framework. Original
request/response bytes, trade-time code revision, buy-response values, and
settlement ticks were not retained and are not inferred.
