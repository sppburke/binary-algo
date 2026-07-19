# USDCHF 15m current-refit contract

> **SCOPE: USDCHF·15m.** Procedure and package contract only. This document records no
> current-edge, profitability, or side-specific efficacy claim.

## Purpose and authority

Issue #18 creates one deterministic `side="combined"` refit of the existing
USDCHF 15m New-York cross-pair seed-ensemble policy. The refit is latest-on-disk:
its information cutoff is `2026-05-09T00:00:00Z`, not the wall-clock date when
the operator runs.

The output is an inactive, unpublished research package. It is not a book, is
not registered in `books/INDEX.json` or `MODEL_REGISTRY.md`, cannot be loaded by
`book_runtime`, and grants no demo-buy, real-money, deployment, or activation
authority. A future prospective issue must bind the exact main commit and
package hashes, then seal its T0 before observing eligible outcomes. That trial
must report UP and DOWN as separate keys; neither side may inherit a combined
statistic.

## Operator

Run from the repository root with the pinned environment:

```bash
~/binary-algo-venv/bin/python scripts/usdchf_m15_current_refit_v1.py seal
~/binary-algo-venv/bin/python scripts/usdchf_m15_current_refit_v1.py fit --seal-id <id>
~/binary-algo-venv/bin/python scripts/usdchf_m15_current_refit_v1.py verify --seal-id <id>
```

`seal` publishes the tracked source manifest and immutable execution seal before
importing a label-bearing builder. Commit and push those two artifacts as H1;
H1 must have the clean H0 as its sole parent and no other changed path. `fit`
then authenticates H1, the sealed interpreter/dependency versions, shared code,
and every source identity before doing work; source identity is checked again
after matrix construction.
`verify` is read-only and performs zero fits.

The domain evidence chain is exactly:

1. `research.usdchf_current_refit.execution_seal/v1`; then
2. `research.usdchf_current_refit.terminal/v1`, directly dependent on the seal.

`scripts/evidence_store.py` owns object identity, immutability, ancestry, and
ordered `ArtifactRef` verification. The domain operator owns the refit-specific
payload validation, lock/recovery rules, and package transaction. It reuses the
repository-wide nonblocking research lock and never changes
`scripts/research_campaign_v1.py` or adds a general artifact API.

## Fixed data and model contract

- Fit decisions begin at `2012-01-01T00:00:00Z`; every fit label exits strictly
  before `2026-04-01T00:00:00Z`.
- Early-stopping and policy-calibration decisions begin at
  `2026-04-01T00:00:00Z`; every label exits strictly before
  `2026-05-09T00:00:00Z`.
- The April–May rows are spent procedure inputs only. They may affect fitting,
  early stopping, and fixed-threshold construction, but carry zero efficacy
  weight.
- Matrix construction reaches `usdchf_15m_xpair_frozen.build_mat` only through
  the existing `build_fit_rows` and `build_calibration_rows` adapters.
- Ambient `MX_HOR` must be absent or exactly `15`; the public and frozen
  USDCHF builders must resolve to the 15-minute/900-second label contract.
- The decision clock, 900-second label exit, New-York/DST masks, moved/tie
  semantics, 150,000-row timestamp cap, seeds `[0, 1, 2]`, serial fitting,
  arithmetic seed mean, 1% coverage, linear-quantile calibration, and
  `p>=0.5_is_UP` direction rule are fixed.

No feature, model, parameter, seed, coverage, side, pair, split, or
outcome-based search is authorized. The operator must not open issue-#9
replay/result/status files, VPS data, credentials, APIs, settlements, payouts,
or P&L.

## Package and lifecycle

Successful fitting installs exactly one tracked package at:

`results/artifacts/USDCHF.m15ny_xpair_seedens.r202605.v1/`

The completion order is fixed:

1. `m15xpny_USDCHF_r202605_s0.txt`
2. `m15xpny_USDCHF_r202605_s1.txt`
3. `m15xpny_USDCHF_r202605_s2.txt`
4. `m15xpny_USDCHF_r202605_strategy.json`
5. `USDCHF.m15ny_xpair_seedens.r202605.v1.resolved_refit_spec.json`
6. `USDCHF.m15ny_xpair_seedens.r202605.v1.manifest.json`

The manifest is written last with atomic no-replace semantics. Canonical and
independent-audit fits must produce byte-identical model and strategy output.
The terminal result is
`results/json/usdchf_m15_current_refit_v1_result.json`; it and every successful
package file are bound in fixed order by the terminal evidence object.

The only terminal decisions are `INACTIVE_UNTESTED_PROSPECTIVE` and
`NO_CANDIDATE`. The latter is authorized only when complete authenticated
canonical and audit outputs differ semantically or byte-for-byte, and it
installs no package. Operational, identity, protected-access, source, or
publication failures block without manufacturing a terminal decision.

## Verification

```bash
~/binary-algo-venv/bin/python -m py_compile scripts/usdchf_m15_current_refit_v1.py scripts/test_usdchf_m15_current_refit_v1.py
~/binary-algo-venv/bin/python scripts/test_usdchf_m15_current_refit_v1.py
~/binary-algo-venv/bin/python scripts/test_m15_book_refresh.py --skip-null
~/binary-algo-venv/bin/python scripts/test_evidence_store.py
~/binary-algo-venv/bin/python scripts/evidence_store.py verify-store
git diff --check
```

The focused test uses synthetic rows and fake seed artifacts; it does not fit a
real model. The six real seed fits—three canonical and three audit—run serially
only in the separately governed strategy-evaluation phase.

## Deferred follow-ons

The five retained follow-ons, their unlock triggers, and their owners are
recorded once in `docs/IDEAS_LOG.md` under `REFIT-PROSPECTIVE-1`,
`REFIT-PUBLISH-1`, `REFIT-CADENCE-1`, `ALPHA-COMPILER-1`, and
`REAL-MONEY-CANARY-1`. None is authorized by this contract.
