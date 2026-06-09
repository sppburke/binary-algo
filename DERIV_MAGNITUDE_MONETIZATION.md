<!-- Generated 2026-06-09 by a multi-agent deep-dive workflow (deriv-magnitude-monetization).
     Tier-1 anchor: deriv_frxEURUSD_contracts_for.json (committed 47744f6, snapshot 2026-06-03).
     Independently re-verified this session: FX exposes ONLY daily (1d-365d) touch/range/ends binaries
     + 15m-floor sign-only Rise/Fall; lookbacks/vanillas/accumulators/turbos/digits are non_available on FX. -->

# Monetizing Magnitude Predictability on Deriv.com

> Status: research deliverable, evidence-first. Every load-bearing fact below is tagged by tier. **T1** = repo file at current commit (`deriv_frxEURUSD_contracts_for.json`, `cpcv_certify_result.json`, `MAGNITUDE_FINDINGS.md`, etc.) read this session. **T3** = repo docs / Deriv product pages described in the supplied research fragments. **T4** = Deriv API behaviour from the supplied fragments not independently re-verified against a live socket this session. Claims that depend on a Deriv fact we could **not** confirm at T1 are flagged inline.
>
> One verification caveat up front: the on-disk `deriv_frxEURUSD_contracts_for.json` is a **single FX-major snapshot** (`echo_req = {contracts_for: "frxEURUSD", currency: "USD"}`, `spot: 1.15991`, file mtime 2026-06-03). It is genuine Tier-1 for *what FX exposes*, but it is one symbol, one fetch. I did **not** open a live socket this session; all API-mechanics and synthetic-index (R_100) claims are T3/T4 from the supplied fragments and are flagged as such.

---

## 1. The edge we are monetizing

We can predict **how far price moves, not which way.** Concretely, the certified target is a **sign-invariant binary classifier**: `label = 1 iff |ret_H| >= train-Q` (Q ∈ {Q67, Q75, Q90}), where `ret_H` is the H-horizon log return. No up/down. The backbone is a LightGBM on realized-volatility-clustering features (`rv30`, `rv120` = rolling std of log-returns); permutation entropy is inert on FX.

**Strength (T1, verbatim from `cpcv_certify_result.json` → `magnitude_30m`):**
- EURUSD 30m CPCV: `auc_mean = 0.7439`, `auc_p10 = 0.7177`, `auc_min = 0.7054`, `auc_max = 0.7819`, over 28 purged CPCV paths.
- Deflated expectation `0.7124` (`deflation.magnitude_auc`) — clears the 0.55 bar comfortably.
- **Top-vs-bottom-decile realized-|ret| lift: `lift_mean = 5.013×`, `lift_p10 = 3.956×`** (`magnitude_30m.lift_mean/lift_p10`). This is the economically load-bearing number: rank bars by predicted magnitude, the top decile realizes ~5× the |move| of the bottom decile.

**Strength at other horizons (T1 result files / `MAGNITUDE_FINDINGS.md` table, lines 162–172):**
- 60s EURUSD: magAUC **0.787** vs dirAUC **0.5096** on identical data (`magnitude_verified.json`; the cleanest sign-invariance proof).
- USDJPY 1m: magAUC Q67 0.716 / Q75 0.730 / Q90 **0.790**; **top-decile |ret| lift only ~2.1–2.5×** (`usdjpy_1m_magnitude_result.json`).
- USDJPY 2m: magAUC Q75 0.722 / Q90 0.785; **top-decile |ret| lift ~2.0–2.5×** (`usdjpy_2m_magnitude_result.json`).

**Direction is dead.** EURUSD 15m direction CPCV `auc_mean = 0.5198` (`cpcv_certify_result.json → direction_15m`); the selective direction book deflates `0.647 → 0.5455` and FAILS the 0.541 breakeven. USDJPY/AUDUSD direction is own-pair-specific and near-efficient at ≥5m.

**The implication.** A sign-invariant magnitude forecast monetizes on **payoffs that key off |move|, not sign**: buy volatility when we predict a large move (touch / range-out / lookback / straddle), sell volatility when we predict a small move (range-in / no-touch / accumulator). Our **best** axis (magnitude) aligns exactly with **volatility-structured** payoffs and is orthogonal to the directional binaries (Rise/Fall) that dominate Deriv's FX retail menu.

> **Honest caveat that recurs throughout:** every certified number is at **60s–30m**. The decile lift quoted as "5×" is the **30m** figure; at 1m/2m it is only **~2.0–2.5×** (T1, `MAGNITUDE_FINDINGS.md` L171–172). This matters because the cleanest FX magnitude products turn out to be **daily-only** (§2), a horizon at which our edge is **literally unmeasured**.

---

## 2. Deriv product landscape (magnitude lens)

The decisive constraint is **what forex majors actually expose**, and at what duration. This is settled at Tier-1 from `deriv_frxEURUSD_contracts_for.json` (frxEURUSD, snapshot 2026-06-03). I enumerated every `(contract_type, expiry_type, min_duration, max_duration, barrier_category)` tuple in the `available` bucket.

**FX majors expose exactly 12 contract types (T1, `contracts_for.available`, 22 rows / 12 distinct codes):**

| Code | Display | Magnitude payoff? | expiry_type | min–max duration | barrier_category |
|---|---|---|---|---|---|
| CALL / PUT | Rise / Fall (ATM) | **no** (pure sign) | intraday **and** daily | **15m**–1d (intraday); 1d–365d (daily) | euro_atm |
| CALL / PUT | Higher / Lower (barrier) | low | daily only | 1d–365d | euro_non_atm |
| CALLE / PUTE | Rise/Fall Equals | no | intraday + daily | 15m / 1d | euro_atm |
| **ONETOUCH** | Touches | **high** (win-prob ∝ \|move\|) | **daily only** | **1d**–365d | american |
| **NOTOUCH** | Does Not Touch | **high** (small-move) | **daily only** | **1d**–365d | american |
| **EXPIRYMISS** | Ends Outside | **high, two-sided** | **daily only** | **1d**–365d | euro_non_atm |
| **EXPIRYRANGE** | Ends Between | **high, two-sided** | **daily only** | **1d**–365d | euro_non_atm |
| **RANGE** | Stays Between | **high, two-sided** (path) | **daily only** | **1d**–365d | american |
| **UPORDOWN** | Goes Outside | **high, two-sided** (path) | **daily only** | **1d**–365d | american |
| **MULTUP / MULTDOWN** | Multipliers | high (continuous, **directional**) | no_expiry (stop-out) | open-ended | american |

**Products NOT available on forex majors (T1, `contracts_for.non_available` — 22 codes, each carrying only `contract_category` + display name, NO `expiry_type`, NO duration, NO barrier fields → not priceable/tradeable on FX):**

> `ACCU` (Accumulators), `LBFLOATCALL`, `LBFLOATPUT`, `LBHIGHLOW` (Lookbacks), `VANILLALONGCALL`, `VANILLALONGPUT` (Vanillas), `TURBOSLONG`, `TURBOSSHORT` (Turbos), `RESETCALL`, `RESETPUT`, `TICKHIGH`, `TICKLOW`, `RUNHIGH`, `RUNLOW`, `ASIANU`, `ASIAND`, all six `DIGIT*`.

This **corrects** the consolidated catalog, which labelled lookbacks/vanillas "disputed (T3 yes / T1 absent)." They are **not absent** — they are **present-but-`non_available`** on frxEURUSD, which is a *firmer* confirmed-no for FX than "absent": Deriv lists the product category and then refuses to surface durations/barriers for it on this symbol. The T3 product page that says "Forex" for lookbacks/vanillas is **outranked** by this T1 `non_available` evidence.

**The two facts that make or break monetization:**

1. **Every two-sided / touch magnitude binary on FX is DAILY-ONLY.** ONETOUCH, NOTOUCH, EXPIRYMISS, EXPIRYRANGE, RANGE, UPORDOWN each have **exactly one** row: `expiry_type = daily, min = 1d, max = 365d`. There is **no intraday touch/range/ends product on FX**. (Contrast: CALL/PUT *do* have intraday rows at a **15m** floor — proving the daily-only restriction is product-specific, not a data gap.)

2. **The only intraday FX products are ATM CALL/PUT/CALLE/PUTE at a 15m floor — and they are magnitude-irrelevant (pure sign).** The only continuous-payout instrument confirmed on FX is **MULTUP/MULTDOWN**, which is **directional and open-ended (stop-out)**, not a fixed-horizon two-sided |move| bet. `MULTUP` exposes `multiplier_range = [100,200,300,500,800]` and an **empty** `cancellation_range` (T1) — cancellation/deal-cancellation is not offered on FX multipliers in this snapshot.

**The structural mismatch:** our edge is strongest at **60s–30m**; on FX the only products whose *payout* keys off |move| both ways are **daily-only**; and the only intraday FX product is sign-only. So the **payoff-perfect** magnitude vehicles (Lookback High-Low, Accumulator, Vanilla straddle) live on **synthetic indices (R_100)**, not FX majors. This is the central honest finding of this deliverable.

---

## 3. Ranked monetization paths

Ranking combines payoff-fit (sign-invariant > one-sided > directional), the adversarial verdicts (a path that fails the FX/duration/EV gate ranks low or is marked blocked), and availability honesty. **No path below is currently a ready FX deployment.** That is the truth, not a hedge.

### Rank 1 — Lookback High-Low (LBHIGHLOW) — payoff-perfect, **synthetic-only / FX-blocked**

- **Fit:** highest possible. Payoff = `multiplier × (High_path − Low_path)` = the **realized range**, always ≥ 0, fully sign-invariant, continuous. The payout *is* the quantity our model forecasts. On synthetics the duration is ≤30m intraday (T3), which **matches our 60s–30m edge exactly** — the rare clean horizon fit.
- **Signal → trade rule:** score `P(|ret_H| >= Q90)`; **enter only the top decile** (≈ cov5% operating point). No barrier choice — buy the excursion. Size by quarter-Kelly on `P(realized_range > R*)` where `R* = stake/multiplier` is the broker's break-even range, capped 1–2% bankroll.
- **EV logic:** Deriv's edge is baked into the multiplier (no separate markup field) so `R*` sits above the *unconditional* median range. Our classifier identifies the conditional sub-population whose range is materially larger. **Adversarial verdict: SURVIVES, narrowly and conditionally.** The dominant unrefuted risk: if Deriv reprices the multiplier intraday off the same `rv30/rv120` clustering we use, our edge collapses to the thin incremental gain (rv .73→.74). Unmeasured.
- **FX/duration:** **confirmed-no on FX** (T1, `non_available`). Viable only via an R_100 retraining program — and **no synthetic-index data exists on disk** (T1: a `find` over the repo/processed data returns no R_100 OHLC). Edge transfer from FX to a GBM-style synthetic is plausible (mechanism = generic vol-clustering) but **untested**.
- **Risks:** (a) availability — FX-blocked, requires a new synthetic research program; (b) broker prices the same rv signal; (c) tick-settlement vs bar-training gap (lookback settles on tick extremes, model trained on bar OHLC ranges — tick range ≥ bar range, shifting R* unfavourably); (d) AUC ranks, EV needs calibration.

### Rank 2 — Ends Between / Ends Outside (EXPIRYRANGE / EXPIRYMISS) — best FX-native two-sided fit, but **daily-only**

- **Fit (catalog fit_score 5):** textbook two-sided magnitude binary. EXPIRYMISS wins if exit lands **outside** a symmetric band (large |move|, either way); EXPIRYRANGE wins **inside** (small |move|). At-expiry settlement (`euro_non_atm`), not path-dependent. **FX-confirmed at T1** (both in `available`; the EXPIRYMISS row shows `high_barrier = 1.16279`, `low_barrier = 1.15705`, `barriers = 2`).
- **Signal → trade rule:** large leg → buy EXPIRYMISS in the top score decile with band half-width `w = k·σ̂·√τ`, k swept via repeated `proposal` calls to maximize `(our_P_out − implied_q_out)`; small leg → buy EXPIRYRANGE in the bottom decile. Trade only top/bottom deciles, skip the middle 8. Daily contracts require **absolute** barrier strings (`entry ± w`), not relative offsets.
- **EV logic:** `EV = P_model·payout − ask`; positive iff `P_model > q = ask/payout`. The implicit binary haircut is real (T1 `THEORY.md` L97: breakeven win-rate ≈ 0.541 at payout R ≈ 1.85, ~4.1pp over a coin).
- **Adversarial verdict: DOES NOT SURVIVE as scoped.** Fails the duration gate: daily-only on FX, while every certified horizon is 60s–30m. EV is **doubtful** — there is **no 1d magnitude model anywhere on disk** (verified by grep across all `*result*.json`; the only "1d" string is a Deriv product spec), and `rv30/rv120` autocorrelation mean-reverts over a day, so the 5× lift very likely attenuates. At-expiry settlement also means a large intraday move that reverts by daily close *loses* — so terminal `|ret_1d|` is a **different label** than the intraday `|ret_H|` we certified.
- **Path to viability:** retrain + CPCV-certify a `|ret_1d| >= Q` classifier (does not exist), then live-probe `proposal` to confirm `P_model − q ≥ margin`.

### Rank 3 — Stays Between / Goes Outside (RANGE / UPORDOWN) — two-sided, path-dependent, **daily-only**

- **Fit (fit_score 5):** same two-sided logic as Rank 2 but **american/path-dependent** (touch any time in life), so win-prob is *even more* sensitive to realized |move|. FX-confirmed at T1 (`american`, `barriers = 2`).
- **Verdict: DOES NOT SURVIVE as scoped** — identical kill to Rank 2 (daily-only on FX vs 60s–30m edge; EV doubtful, conditional on an untested 1d retrain). Path-dependence adds intra-day-path settlement risk and requires a live tick feed to monitor the open contract.

### Rank 4 — Touch / No-Touch (ONETOUCH / NOTOUCH) — clean one-sided fit, **daily-only**

- **Fit (fit_score 4, realistically 1–2 pending retrain):** win-prob is a monotone function of |move| vs barrier distance. ONETOUCH = large-move leg, NOTOUCH = small-move leg; pair two opposite ONETOUCHes to approximate a two-sided large-move bet. FX-confirmed at T1 (ONETOUCH `barrier = 1.16279` absolute, `american`).
- **Verdict: DOES NOT SURVIVE as scoped.** Daily-only on FX (1d–365d); intraday rejected. EV **unverifiable** — needs a first-passage/touch calibration layer that does not exist on disk; adverse-selection risk (Deriv prices touch barriers off a vol model that also sees recent realized vol, raising `ask` in lock-step on exactly the rows we flag); path/American settlement can trigger on a single intra-bar wick the OHLC model never saw. Fixed payout leaves most of the 5× magnitude lift on the table vs a continuous-payout product.

### Rank 5 — Vanilla straddle (long VANILLALONGCALL + long VANILLALONGPUT) — ideal shape, **FX-blocked**

- **Fit (fit_score 6):** same-strike same-expiry straddle isolates |move| (delta-neutral). A pure realized-vs-implied vol bet — exactly what our rv-forecaster is.
- **Verdict: DOES NOT SURVIVE on FX.** **Confirmed-no on FX** at T1 (`non_available`). The T&C §2.1.1.11 mention of "Selling Vanilla Options on FX within 24h prior to expiration" is a **sell-side mechanics clause**, not proof of buy-side intraday availability, and is a lower tier than the T1 `non_available` probe. EV is **doubtful and overstated** in the catalog: it leans on the **5× (30m)** lift, but the straddle Q90 gate targets 60s/1m/2m where the lift is only **~2.0–2.5×** (T1), and the correct comparison is realized-vs-**implied** (implied already conditions on the same public rv state) — not realized-vs-unconditional. EV is also **un-backtestable on disk**: no intraday implied-vol feed exists (T1 `MAGNITUDE_FINDINGS.md` L322–323). Pursue only on synthetics, re-certified.

### Rank 6 — Accumulators (ACCU) — the inverse (small-move) bet, **FX-blocked**

- **Fit:** monetizes confident **LOW-|move|** (the sell-vol leg of the duality): stake compounds while spot stays in a dynamic per-tick range, knocks out to 0 on a large move. Natural complement to LBHIGHLOW on synthetics.
- **Verdict: synthetic-only.** **Confirmed-no on FX** at T1 (`non_available`). Pursue on R_100 alongside Rank 1 if a synthetic program is launched.

### Rank 7 — Multipliers (MULTUP / MULTDOWN) — only continuous payout confirmed on FX, but **directional**

- **Fit (fit_score B-tier):** the *only* continuous-payout magnitude instrument confirmed on FX (T1). P/L is linear in |move| — but **directional** (must pick a side) and **open-ended** (no_expiry, stop-out), so it is a poor structural fit for a sign-invariant *fixed-horizon* classifier. Usable only if paired with a separate (dead) direction signal, or by sizing the stop-out to harvest realized vol. Exposes a `commission` field (unlike binaries) and `multiplier_range = [100..800]`.
- **Verdict: weak fit, not a clean monetizer for this edge.** Direction being dead (AUC ~0.51) means we cannot pick the side; this is the wrong payoff geometry for a size-only edge.

**Bottom line:** the **payoff-perfect** products (LBHIGHLOW, ACCU, vanilla straddle) are **confirmed-no on FX** and live only on synthetic indices, where our FX-certified edge must be **re-mined from scratch** (different generating process, no synthetic data on disk). The **FX-native two-sided magnitude binaries** (Ends/Stays Between/Outside, Touch) are **daily-only**, a horizon where our edge is **unmeasured**. The only intraday FX product is **sign-only** Rise/Fall. **No path is a ready deployment today.** The two live options are (A) build & certify a **1-day** FX magnitude model to unlock Ends-Outside/Touch, or (B) stand up a **synthetic-index (R_100)** program to unlock LBHIGHLOW/ACCU at native intraday horizons.

---

## 4. The house-edge problem (critical analysis)

**Where Deriv's edge sits.** Deriv binaries carry **no separate markup field** — the haircut is baked into the `payout/stake` ratio. The broker-implied win-probability is `q = ask_price / payout`. For a true-probability-p outcome the fair payout is `1/p`; Deriv quotes `payout < 1/q`, so `q > p_fair`. T1 anchor: `THEORY.md` L97 — Rise/Fall breakeven win-rate **≈ 0.541 at payout R ≈ 1.85**, i.e. a **~4.1 percentage-point** haircut over a fair coin. Touch/range binaries carry a comparable implicit margin folded into `ask_price/payout`. Multipliers/ACCU instead expose an explicit `commission` (%).

**The EV condition.** For any fixed-payout magnitude binary:

```
EV = P_model · payout − ask_price          (per unit stake, payout in payout-units)
   > 0   ⟺   P_model > q = ask_price / payout
```

So the model must beat the **broker's implied probability** by more than the haircut + slippage + deflation. A practical gate: trade only when `P_model − q ≥ 0.06`.

**Can a size-only edge clear it?** The mechanism is sound. The broker bands its barrier near the **unconditional** touch/ends-outside rate. Our classifier, with `lift_mean = 5.013×` (30m, T1), identifies a conditional sub-population whose realized |move| is ~5× the bottom decile. On a barrier placed at `k·σ̂·√τ` with `k≈1`, a top-decile move crosses with materially higher probability than the unconditional rate the broker prices. So `P_model − q` can plausibly exceed a 4–6pp haircut **on the extreme deciles only**.

**Three reasons this is NOT established by AUC alone (all unrefuted):**

1. **AUC measures ranking, not calibration.** EV needs a *calibrated absolute* `P(touch | barrier, H)` or `P(ends-outside | band, H)`. The repo stores AUC + decile lift only — **no calibration for magnitude band-crossing exists on disk** (verified: `cpcv_certify_result.json` and `mag_har_result.json` carry no brier/reliability keys; `brier_audit_result.json` exists but audits the **direction** books, not magnitude). A high-AUC ranker can still be miscalibrated and lose money at a *specific* barrier.

2. **Adverse selection / "the edge is in the price."** Vol-clustering is **public** information. Deriv prices barriers off a vol model that also sees recent realized vol, so on exactly the rows we flag most confidently, `ask_price` rises in lock-step — compressing the gap precisely when we are most confident. Our *true* exploitable edge is only the **increment** our model has over Deriv's vol model, not the full lift. T1 corroboration that this increment is thin: forward-vol features (Kronos dispersion) correlate 0.46–0.83 with `rv30` and add **nothing** orthogonal (`MAGNITUDE_FINDINGS.md` §6c, KILLED at every horizon).

3. **Horizon mismatch compounds it.** The 5× lift is at **30m**; FX two-sided binaries are **1d**; at 1m/2m the lift is only ~2.0–2.5×. At the *tradeable* FX horizon (1d) the edge is **unmeasured**, and rv-autocorrelation mean-reverts over a day.

**What we must measure live before risking capital:** for a chosen symbol + barrier + duration, call `proposal`, read `q = ask_price/payout`, and compare against the **calibrated** `P_model` on a held-out replay. EV>0 is **only** established by `(P_model − q) ≥ margin` holding out-of-sample on live (or recorded) quotes — never by AUC. The continuous-payout products (LBHIGHLOW) are structurally better here because the haircut is a **single-digit-% multiplier margin** rather than the ~4–6pp binary implied-prob haircut, and the payout **scales** with how far the move travels, capturing more of the lift.

---

## 5. Recommended modeling approach

The existing classifier ranks bars by `P(|ret_H| >= Q)`. To trade, we need **calibrated absolute probabilities of the contract-specific event**, plus deriv-faithful settlement. Three new label families, all buildable from on-disk OHLC (and, where settlement is path/tick-based, from on-disk raw ticks 2012–2026, T1 `AUDUSD_RESULTS.md` L12):

1. **Touch label (for ONETOUCH/NOTOUCH, and as a building block for path products):**
   `label = 1 iff path crosses entry ± k·σ̂·√τ within H`. Build by walking the **tick** path (not the bar close — touch is path/American). Fit a calibrated `P_touch(k, σ̂, τ)` via a first-passage/reflection anchor `≈ 2·Φ(−D/(σ̂√τ))`, then **isotonic-correct** on a held-out window against historical realized touch frequencies. Calibration, not AUC, is the deliverable.

2. **Realized-range label (for LBHIGHLOW):**
   `label = realized (High − Low) over H` (a regression / CDF target, not binary). From bars for a first pass, then **re-derive from ticks** because the contract settles on tick extremes and tick range ≥ bar range. The trading quantity is `P(realized_range > R*)` where `R* = stake/multiplier` read live from `proposal`.

3. **Ends-outside / band label (for EXPIRYMISS/EXPIRYRANGE, at-expiry):**
   `label = 1 iff |ret_H| > w` at the **terminal** bar (not path). For FX this must be built at **H = 1d** (the only FX duration), which is a **new horizon never certified** — re-run `cpcv_certify.py` to confirm `AUC ≥ 0.65` and recompute deflation/decile-lift at 1d **before** any EV claim.

**Calibration > AUC.** For every product, the trading signal is a calibrated probability compared to `q = ask/payout`. Fit isotonic/Platt on the magnitude score → event probability, validate with Brier/reliability (the repo has the machinery in `brier_audit` — currently pointed at direction books; repoint it at magnitude bands).

**Deriv-faithful settlement discipline (already used in the repo, T1 `AUDUSD_RESULTS.md` L12):** bar-close approx, **mid-to-mid**, **next-tick entry +1s**, **ties LOSE**, breakeven 0.541 at R≈1.85, strict OOS splits (train 2012–21 / val 2022–23 / test 2024 / 2025 / oos 2026), selection on VAL worst-half (never VAL-acc-max; `corr(VAL,OOS) = −0.54`), CPCV with purge+embargo, per-fold refit. Touch/range labels must be **path-aware** (tick-level), since a bar-close label systematically understates American-barrier touch frequency. Note also the **refit-decay** finding (T1 `AUDUSD_RESULTS.md` L20–21): frozen-2021 books decay forward (.61→.58→.53 cov5 by 2026) — any deployed magnitude→trade model must be **periodically retrained and sized on the refit floor**, never frozen.

---

## 6. API integration sketch

> T3/T4 from the supplied fragments (classic v3 WebSocket) — not re-verified against a live socket this session. The `barrier`-offset syntax and the `q = ask/payout` reading are the load-bearing pieces.

```
# 1. Connect (classic v3 — this is the API that exposes FX barrier products)
wss://ws.derivws.com/websockets/v3?app_id=1089     # 1089 = public test id; register own for prod
# JSON over one socket; pair req/resp via req_id; 2-min idle timeout -> send {ping:1} to keep alive.

# 2. Authorize (only needed for buy/sell/portfolio; proposal & contracts_for need NO auth)
{ "authorize": "<API_TOKEN>" }
# scopes: read | trade | trading_information | payments | admin
#   - proposal / contracts_for / active_symbols / ticks  -> no auth
#   - buy / sell / sell_expired                           -> trade scope
#   - portfolio / proposal_open_contract                 -> read scope
# demo vs real = which token: loginid VRTC... = virtual, CR.../MF... = real (same endpoint/code).

# 3. Discover barriers & durations for the symbol  (THIS is how we settled §2)
{ "contracts_for": "frxEURUSD", "currency": "USD" }
# -> available[] gives contract_type, expiry_type, min/max_contract_duration,
#    barrier_category, barriers, and for daily contracts the absolute high_barrier/low_barrier.
#    On FX, every Touch/Range/Ends row is expiry_type=daily, min=1d (verified T1).

# 4. Price a SPECIFIC barrier (read the implied probability from the quote)
{ "proposal": 1, "contract_type": "ONETOUCH", "symbol": "frxEURUSD",
  "currency": "USD", "amount": 100, "basis": "payout",   # price the cost of a target payout
  "duration": 1, "duration_unit": "d",                   # FX touch is daily-only
  "barrier": "1.16279",                                  # FX >=24h: ABSOLUTE level (string)
  "subscribe": 1 }                                       # re-quote per tick
# Response: ask_price (= display_value, what you pay), payout, spot, longcode, id, commission,
#           validation_params{stake{min,max}, payout{max}}.
# IMPLIED PROBABILITY:  q = ask_price / payout.   Trade only if  P_model_calibrated - q >= margin.

# barrier-offset syntax (load-bearing):
#   string, regex ^[+-]?[0-9]+\.?[0-9]*$
#   FX < 24h  -> RELATIVE signed offset added to entry spot, e.g. "+0.0010" (~10 pips on EURUSD).
#               (NB: no FX magnitude binary is < 24h, so in practice you will use absolute.)
#   FX >= 24h -> ABSOLUTE level, e.g. "1.16279"  (verified T1: daily contracts carry absolute barriers).
#   two-barrier: barrier="1.16279", barrier2="1.15705"  (EXPIRYMISS/RANGE on FX, daily).

# 5. Buy with slippage cap
{ "buy": "<proposal_id>", "price": <max_ask> }
# -> contract_id, buy_price, payout, balance_after, shortcode

# 6. Monitor open contract
{ "proposal_open_contract": 1, "contract_id": <id>, "subscribe": 1 }
# -> bid_price, profit, is_valid_to_sell, is_sold, is_expired, status(open/sold/won/lost)

# 7. Sell early (only while is_valid_to_sell)
{ "sell": <contract_id>, "price": <min_acceptable | 0=market> }

# Rate limits: read at runtime from website_status.api_call_limits (do NOT hardcode):
#   max_requests_pricing (proposal/ticks), max_requests_outcome (buy/sell), max_proposal_subscription.
#   Use forget / forget_all to free subscription slots.
```

> Migration note (T3): Deriv is moving to a newer Options API (`https://api.derivws.com`, OAuth Bearer, param renamed `underlying_symbol`). Classic v3 above is still live and is what exposes FX barrier-offset pricing — target v3 for now.

---

## 7. Corrections to the repo's existing productization notes

`MAGNITUDE_FINDINGS.md` §6 was written from general knowledge / MEMORY before the live `contracts_for` snapshot existed. Specific corrections and confirmations, now that the T1 FX probe is on disk:

1. **CONFIRMED — Rise/Fall is the wrong shape and 15m is the FX intraday floor.** §6 L306–307 says "forex Rise/Fall minimum expiry = 15 minutes and is DIRECTIONAL." T1 confirms exactly: CALL/PUT/CALLE/PUTE intraday rows show `min = 15m, euro_atm` (sign-only). Magnitude-irrelevant, as §6 states.

2. **CORRECTION — "Touch/No-Touch and Range/Boundary are buildable … with the existing magnitude model" understates the duration problem.** §6 L313–314 + L328 imply Touch/Range are near-ready. **T1 reality: on FX they are DAILY-ONLY (1d–365d).** Our certified edge is 60s–30m. They are **not** buildable with the existing model — they require a **new 1-day magnitude model + CPCV cert** that does not exist on disk. §6 should add: "FX Touch/Range/Ends are daily-only; the 60s–30m edge does not transfer to a 1d horizon without re-certification."

3. **CORRECTION — Straddle/VRP are not merely "needs an IV feed"; the vanilla product itself is confirmed-no on FX.** §6 L316/L329 frames the straddle blocker as the missing implied-vol feed. T1 adds a prior blocker: `VANILLALONGCALL/PUT` are in `non_available` on frxEURUSD — **you cannot trade a vanilla straddle on FX majors at all**, IV feed or not. The straddle is a **synthetic-index** play.

4. **NEW (not in §6) — the payoff-perfect product is Lookback High-Low, and it is also FX-blocked.** §6's product table omits Lookbacks. LBHIGHLOW (`payoff = mult × (High − Low)` = realized range) is the single best payoff match for a sign-invariant edge, **but** it is `non_available` on FX (T1). Add it to §6 as the top synthetic-index candidate.

5. **NEW — Accumulators are the small-move (sell-vol) leg, also synthetic-only.** §6 has no entry for the LOW-|move| tail beyond "No-Touch / In." ACCU is the natural sell-vol instrument; T1 confirms it is `non_available` on FX. Add as the synthetic complement to LBHIGHLOW.

6. **CONFIRMED — no intraday implied-vol feed on disk.** §6 L322–323 is correct (T1). This blocks VRP/straddle EV backtesting regardless of venue.

7. **CONFIRMED — MULTUP/MULTDOWN is the only continuous-payout instrument on FX, and it is directional.** Not mentioned in §6. T1 confirms it (`available`, `no_expiry`, `multiplier_range [100..800]`, empty `cancellation_range`). Worth noting as the only FX continuous payout, with the caveat that direction is dead so its fit is weak.

8. **METHODOLOGY NOTE — cite the on-disk file, not a "live 2026-06-09 probe."** The consolidated catalog header claims a live fetch dated 2026-06-09. The actual Tier-1 artifact is `deriv_frxEURUSD_contracts_for.json` (snapshot **2026-06-03**, frxEURUSD only). Conclusions are unaffected, but §6 / any catalog should cite the on-disk file and note it is a single-symbol snapshot, not a live multi-symbol probe.

---

## 8. Next steps (prioritized, concrete)

Ordered by EV-per-effort, each tied to whether the product is actually FX-available.

1. **Live-probe the FX-available daily binaries first (cheapest, no model work).** Open a v3 socket, call `proposal` for **ONETOUCH** and **EXPIRYMISS** on `frxEURUSD` at `duration=1, duration_unit="d"` with absolute barriers near `entry ± 1σ_daily`. Record `ask_price`, `payout`, compute `q = ask/payout`. **Goal:** measure the actual house haircut and barrier-acceptance range on the products that *are* FX-available. *(FX-available: ONETOUCH/EXPIRYMISS confirmed T1.)*

2. **Build the 1-day EURUSD magnitude model and CPCV-certify it.** Re-label EURUSD daily data as `|ret_1d| >= train-Q{75,90}`, run `cpcv_certify.py`, report `auc_mean / p10 / deflated / decile-lift`. **Kill gate:** if 1d `AUC < 0.65` or decile-lift collapses, **all FX two-sided/touch monetization is dead** and we pivot entirely to synthetics. *(This is the single decision that determines whether FX is viable at all.)*

3. **If step 2 survives: calibrate touch & ends-outside probabilities and replay against step-1 quotes.** Fit isotonic `P_touch(k,σ̂,τ)` and `P_ends_outside(w,1d)` from tick paths; replay held-out 2026 against recorded `proposal` quotes; confirm `(P_model − q) ≥ 0.06` out-of-sample. **This is the EV>0 proof — no capital before it passes.**

4. **In parallel, pull R_100 synthetic-index history and re-mine the magnitude edge there.** *(LBHIGHLOW / ACCU are confirmed-no on FX, intraday on synthetics.)* Pull R_100 OHLC + ticks (none on disk today), re-run `cpcv_certify.py` for `|ret_H|>=Q` at 60s–30m on R_100. **Kill gate:** `AUC ≥ 0.65`. Only then probe the LBHIGHLOW multiplier vs `σ̂` correlation to test whether the broker already prices our rv signal.

5. **Measure the adverse-selection coefficient directly.** On both FX (step 1) and synthetics (step 4), regress the broker's implied `q` against our `σ̂` / `rv30` across many quotes. If `q` tracks `rv30` tightly, "the edge is in the price" and our exploitable increment is the residual — quantify it before sizing. *(Applies to every product; this is the make-or-break of §4.)*

6. **Only after 3 or 4+5 pass: paper-trade on a virtual (VRTC) account** through the same socket, with deriv-faithful settlement and refit-floor sizing (quarter-Kelly, ≤2% bankroll, periodic retrain per the AUDUSD decay finding), before any real-money `CR/MF` account.

> Honest disposition: **no product is deploy-ready today.** Steps 2 and 4 are the two forks — a 1-day FX magnitude model (unlocks Ends-Outside/Touch on FX) or a synthetic-index program (unlocks LBHIGHLOW/ACCU at native horizons). Both are **new research programs**, not deployments of the existing certified edge. Every EV claim remains **unverified** until a calibrated model probability beats a live `ask/payout` quote out-of-sample.