# VERIFICATION — Adversarial fact-check of the load-bearing claims

Each claim that *drives a prioritize / de-prioritize decision* in `EXPERIMENT-BACKLOG.md` / `SYNTHESIS.md`
was handed to an independent verifier instructed to **try to refute it** against primary sources (wrong
number / asset class / horizon / sign / over-generalized scope). Verdicts: CONFIRMED = source supports as
stated · PARTIAL = core right, something overstated (correction given) · REFUTED · UNVERIFIABLE.

**Result: 13 CONFIRMED, 5 PARTIAL, 0 REFUTED, 0 UNVERIFIABLE (all high-confidence).**
No decision in the backlog is overturned. Five claims are sharpened, and **one wrong author attribution was
fixed in the corpus** (C3). The 3 spine citations were pre-verified separately (below).

## Spine (pre-verified, this session)
| Source | Verdict | Note |
|---|---|---|
| Petrova–Vilhelmsson–Nordén, *IJF* 2026 — FX LOB, low predictability (EMH) | ✅ | Actual title *"Assessing Cross-Currency Predictability in Forex Markets: Insights from Limit Order Book Data"* |
| Krohn–Mueller–Whelan, *JoF* 79 (2024) — W-shaped USD-into-fixes drift | ✅ | 21y, all-G10; inventory-risk mechanism |
| Gârleanu–Pedersen, *JoF* 68 (2013) — multi-alpha aggregation by decay speed | ✅ | "aim in front of the target" |

## Verdict table (18 decision-driving claims)
| ID | Claim (abbrev) | Verdict | Primary source confirmed |
|---|---|---|---|
| C1 | "79% is equities-5s with order-flow peek, decays by minutes" | ✅ CONFIRMED | Aït-Sahalia, Fan, Xue, Zhou, NBER w30366 — 68.3% no-peek→79.0% with peek @5s; "coin toss in 5 min" |
| C2 | Integrated multi-level OFI ~87% R² but **contemporaneous** | ✅ CONFIRMED | Cont, Cucuringu & Zhang, *Quant. Finance* 23(10) — 71.16%/87.14% in-sample, forward decays "several min" |
| C3 | Order-flow long memory in FX spot → metaorder detector | ⚠️ PARTIAL | **Citation fixed** (see below); premise holds via Gould–Porter–Howison 2016 (H≈0.7) |
| C4 | Gamma→intraday-momentum insignificant in currencies | ⚠️ PARTIAL | Baltussen, Da, Lammers & Martens, *JFE* 2021 — FX "positive but insignificant"; reports already correct |
| C5 | Deriv synthetic-index direction = CSPRNG martingale | ⚠️ PARTIAL | Holds for symmetric Volatility family; scope-note added (Crash/Boom + bias products excluded) |
| C6 | Jane Street 2024: online learning >> feature lift | ✅ CONFIRMED | Volkova Kaggle writeup — online/walk-forward update is the dominant lift |
| C7 | Breedon–Ranaldo local-hours depreciation drift | ✅ CONFIRMED | Breedon & Ranaldo, SNB WP 2011-4 — flow-driven, signed, local-hours |
| C8 | ABDV: FX reacts to **signed** macro surprise within min | ✅ CONFIRMED | Andersen, Bollerslev, Diebold & Vega, *AER* 2003 |
| C9 | Pre-FOMC drift real but **decayed post-2015** | ✅ CONFIRMED | Lucca & Moench *JoF* 2015 + post-2015 weakening |
| C10 | EURUSD dominated by common dollar factor | ✅ CONFIRMED | Verdelhan, *JoF* 73 (2018) "Share of Systematic Variation" |
| C11 | Learning-to-rank across currencies (arXiv:2105.10019) | ✅ CONFIRMED | Poh, Lim, Zohren, Roberts — context-aware LTR |
| C12 | Signed-path-dependence predictive model | ✅ CONFIRMED | Dias & Peters, *Computational Economics* 2020 |
| C13 | DLinear beats Transformers incl. Exchange-Rate | ✅ CONFIRMED | Zeng et al., AAAI 2023 (arXiv:2205.13504) |
| C14 | Zero-shot TSFMs ≈ coin-flip for return direction | ⚠️ PARTIAL | True for off-the-shelf Chronos/TimesFM on **daily equity**; not FX-specific; pretrain-from-scratch differs |
| C15 | Monotone calibration can't improve risk-coverage | ⚠️ PARTIAL | Holds; refinement: isotonic is *weakly* monotone (ties only degrade ranking) |
| C16 | ACI / DtACI hold coverage under distribution shift | ✅ CONFIRMED | Gibbs & Candès, NeurIPS 2021 + DtACI 2022 |
| C17 | Honest daily FX ceiling ~58.5% | ✅ CONFIRMED | Guyard & Deriaz, arXiv:2409.04471 — EUR/USD daily ML direction |
| C18 | Microprice predicts short-horizon future mid | ✅ CONFIRMED | Stoikov, *Quant. Finance* 2018 |

## The 5 PARTIAL corrections (what to change in your thinking)

**C3 — metaorder long memory in FX (affects E7).** The cited *paper* (the Oxford `porterm/long-memory` PDF)
is correct, but the corpus mis-attributed its **authors** as "Lallouache & Abergel." The real authors are
**Gould, Porter & Howison (2016)** (H≈0.7 in FX spot). The *actual* Lallouache–Abergel (2014, EBS) paper says
the **opposite** for FX aggressive flow: market-order sign ACF "rapidly decaying… null after about 2 minutes."
**Net:** E7's premise survives, but reframe the usable horizon — aggressive **market-order** sign persistence in
FX is short (~2 min); limit/cancel-sign runs longer (~5 min). Do not expect an "extended multi-minute" trade-sign
edge. *Citation corrected in `02-*.md` and `EXPERIMENT-BACKLOG.md`.*

**C4 — gamma in currencies (affects E12).** Author is **Martens** (not "Van den Assem"). Plain intraday momentum
*is* present in FX; what's weak/insignificant is the **gamma-distinctive** signature (rest-of-day > last-30-min,
opposite-sign edge, next-day reversal). De-prioritization of an options-gamma 15m **trigger** holds — and it's
daily-close futures, not 15m spot, so don't over-read it as "no gamma effect exists in FX intraday."

**C5 — Deriv (affects the DO-NOT).** The CSPRNG-martingale conclusion is sound for the **symmetric Volatility
indices** only. Deriv also sells **Crash/Boom and directional-bias** products with intentional asymmetric drift —
so scope the DO-NOT to symmetric Vol indices; don't claim *all* synthetics are unpredictable.

**C14 — zero-shot TSFMs (affects the DO-NOT).** Keep the rule, soften the rationale: evidence is **daily equity**,
not FX; and the failure is **off-the-shelf zero-shot** (Chronos/TimesFM/Moirai). Finance-native *pretrain-from-
scratch* is a different, not-ruled-out path (already E13's note).

**C15 — calibration (affects the DO-NOT/E3).** "Cannot improve the risk-coverage curve" holds for all three
methods. Refinement: **isotonic** is only weakly monotone (collapses scores to ties → can *only degrade* ranking);
Platt/temperature are strictly order-preserving. Prefer Platt/temperature if you need probabilities without
touching the selective ordering. The push toward **conformal/DtACI + feature-aware scoring** (not rescaling) stands.

## Net
The backlog's prioritization is sound: every "do-NOT" survives (with C5/C14 scoped more precisely), every new-
direction driver (C1/C3/C6/C7/C8/C11/C12) is primary-source-backed, and the skepticism calls (C9 pre-FOMC decay,
C17 ceiling) are confirmed. The only corpus error was the C3 author attribution, now fixed.
