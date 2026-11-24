> **SCOPE: GENERIC registry** (multi-currency SOTA Deriv-tick certified books). Every entry is keyed `<PAIR>.<book>.v1` with a 6-gate provenance manifest. See `REPO_MAP.md`.

# SOTA Certified Binary-Options Model Registry (`sota-book-index/v1`)

Sibling registry to `MODEL_REGISTRY.md` and `books/INDEX.json` (both kept byte-frozen so existing `evidence/objects/*.json`, `test_protected_evaluation.py`, and `USDJPY` Look 2 `seq_decision.py` cryptographic seals remain 100% intact).

All deployable SOTA books are registered in [SOTA_INDEX.json](books/SOTA_INDEX.json), loaded with zero-pickle pure-NumPy inference via [sota_book_runtime.py](scripts/sota_book_runtime.py), scored on 1-second mid-tick features (`float32` bit-exact with `dataset_dev_v0.1`, `max_abs_diff = 0.0` across `9,895` bars via [deriv_sota_tick_stream.py](scripts/deriv_sota_tick_stream.py)), and executed with dual floor + calibrated row-EV gating via [deriv_sota_executor.py](scripts/deriv_sota_executor.py).

---

## 1. Certified SOTA Books (`books/SOTA_INDEX.json`)

| Book ID | Key (`Pair · Horizon · Side`) | Contract | Architecture | G1 Refit-CPCV (`>=12/15`, `p10>=0.541`) | G2 Drift Placebo (`NY hr × rv_60m`) | G3 `+4s` Delay & Staleness | G4 `16x` Sign-Null (`W=3..48`) | G5 Quoted-Payout EV (`PAYOUT-CLEARED`) | Lifecycle & G6 Status |
|---|---|---|---|---|---|---|---|---|---|
| `USDJPY.sota_c2nysabs_m15_up.v1` | `[USDJPY·15m·UP]` | `CALL` (`risefall`) | RBF Nyström (`500`) + `\|ret\|`-weighted Logit (`66` 1s-tick features, zero pickle) | **PASS** (`15/15` clear, `p10 = 0.5760`, `mean = 0.6000`, `3/3` fwd `0.5785`) | **PASS** (`+0.063` excess, `15/15` paths `> 0`) | **PASS** (`15/15` clear at `+4s`, `p10 = 0.575`, all 3 terciles `> 0`) | **PASS** (`rank 17/17`, `null_max = 0.5177`, `0/16` goal passes) | **PASS** (`14/15` `EV > 0`, `p10_ev = +0.0143`, `mean_ev = +0.0621`, streamed `ev_actual = +0.2629`, `NEUTRAL` `1.7764`) | `demo_venue_settled_canary` (Look 2 to `2027-02-26` preserved; promote per-book to `active` / `CR` once Demo `VENUE_SETTLED` reaches `n >= 50`, `EV CI95_lo > 0`) |
| `USDJPY.sota_e49nys30_m30_up.v1` | `[USDJPY·30m·UP]` | `CALL` (`risefall`) | RBF Nyström (`500`) + Logit (`66` 1s-tick features, zero pickle) | **PASS** (`15/15` clear, `p10 = 0.5984`, `mean = 0.6342`, `3/4` fwd `0.5835`) | **PASS** (`+0.0753` excess, `15/15` paths `> 0`) | **PASS** (`15/15` clear at `+4s`, `p10 = 0.5939`, all 3 terciles `> 0`) | **PASS** (`rank 17/17`, `null_max = 0.5274`, `0/16` goal passes) | **PASS** (`15/15` `EV > 0` @ flat `1.82` quote, `p10_ev = +0.0891`, `mean_ev = +0.1543`, `fwd_mean_ev = +0.0619`) | `demo_venue_settled_canary` (Look 2 to `2027-02-26` preserved; promote per-book to `active` / `CR` once Demo `VENUE_SETTLED` reaches `n >= 50`, `EV CI95_lo > 0`) |

---

## 2. Empirical Venue Economics & Contract-Family Proof (`V-10 .. V-12` & `offer_probe`)

1. **`30m` Eliminates Directional Quote Skew (`V-11`, `results/json/deriv_winter_quote_survey_v10_v12_result.json`, `99,664` priced poll sides):**
   - At `15m` (`n = 4,598` NY-session sides), Deriv dynamically skews payouts (`mean = 1.8046`, `p10 = 1.71`, `mean_abs_side_skew = 0.1194`, `mean_overround = 11.04%`, plus `1.58` haircut in `16:00–16:35 EDT`). Unweighted `[USDJPY·15m]` models (`e30nystroem` `mean_payout = 1.7477`, `p10_ev = -0.0275` in `results/json/deriv_a16pay_USDJPY_15m_result.json`; `e1t` `mean_payout = 1.7274`, `p10_ev = -0.0486` in `results/json/deriv_a17pay_USDJPY_15m_result.json`; `lh12-hlogit` drop-16h `mean_payout = 1.7295`, `p10_ev = -0.0616` in `results/json/deriv_lh12_drop16h_audit_USDJPY_15m_result.json`) fail `G5` (`p10_ev < 0`). Weighting `15m` training by `|ret_fwd_15m|` (`c2nysabs`, `results/json/deriv_a24pay_USDJPY_15m_result.json`) neutralizes quote skew (`mean_payout = 1.7764`, `p10_ev = +0.0143`, `mean_ev = +0.0621`).
   - At `30m`, `60m`, and `120m` (`n = 4,409` NY-session `30m` sides across all 7 majors), Deriv quotes **constant flat `1.82` on both `CALL` and `PUT` (`p10 = 1.82`, `median = 1.82`, `mean_abs_side_skew = 0.0000`, `mean_overround = 9.89%`, `breakeven = 0.5495`)**, allowing `[USDJPY·30m·UP]` `e49nys30` (`p10_wr = 0.5984`) to convert directional accuracy directly into `p10_ev = +0.0891` and `mean_ev = +0.1543` per unit stake (`results/json/deriv_a10signnull_e49nys30_USDJPY_30m_result.json`).
2. **`CALL/PUT` (`risefall`) Strictly Dominates `CALLE/PUTE` (`equal`) at `15m` & `30m` (`V-10`, `results/json/deriv_winter_quote_survey_v10_v12_result.json`):**
   - Deriv haircuts `CALLE/PUTE` by `-0.0639` at `15m` (`1.7408` vs `1.8047`) and `-0.0700` at `30m` (`1.7500` vs `1.8200`), exceeding the `0.0081–0.0119` tie share (`results/USDJPY_RESULTS.md`) by more than `3x`.
3. **Live Yen-Cross Venue Availability & Tick Parity (`results/json/deriv_offer_probe_20261124T022818Z_result.json`, `results/json/deriv_lh16_yencross_tick_parity_result.json`):**
   - Live Deriv Options probe confirmed intraday `15m..1d` `CALL/PUT` and 364-day tick retention on **`frxEURJPY`** (`CALL=1.88, PUT=1.74`), **`frxGBPJPY`** (`CALL=1.86, PUT=1.77`), **`frxAUDJPY`** (`CALL=1.78, PUT=1.84`), and **`frxNZDJPY`** (`CALL=1.54, PUT=1.96`).
   - **`frxCADJPY`** and **`frxCHFJPY`** returned `OfferingsInvalidSymbol` and are permanently excluded.
   - True `frxEURJPY`, `frxGBPJPY`, and `frxNZDJPY` 1s mid ticks (`2026-07-29..30`) match synthetic major-product ticks within `0.046–0.106` bps (`99.07%–99.40%` 15m/30m settlement sign agreement), while `0.6%–0.9%` of close-call settlements flip sign—confirming that Yen crosses require complete ~249-day venue histories via `scripts/run_lh16_yencross_certifier.py` (~10.5h/cross at Deriv's 24 req/min cap) before `LH-16` `G1–G6` book freeze.
4. **Negative Results Recorded Directly (`AGENTS.md` Core Standard):**
   - Across all 6 non-JPY FX majors (`EURUSD`, `GBPUSD`, `USDCHF`, `USDCAD`, `AUDUSD`, `NZDUSD`) at `15m` and `30m` (`results/json/deriv_sota_multimajor_c2nysabs_screen_result.json` + 20 archived `results/json/deriv_c2nysabs*_result.json` dossiers), **0** configurations cleared `G1` (`>=12/15` paths `>= 0.541` with `p10 >= 0.541`); zero non-JPY books are frozen.
   - FX touch/range/barrier contracts (`EXPIRYMISS`, `EXPIRYRANGE`, `UPORDOWN`, `ONETOUCH`) are daily-only (`min = 1d`, zero intraday FX offerings in `docs/deriv_frxEURUSD_contracts_for.json` and `docs/DERIV_MAGNITUDE_MONETIZATION.md`) with zero historical 1d barrier payout archive on disk, and unweighted `[USDJPY·15m]` candidates (`e30nystroem`, `e1t`, `e57reflect`, `lh12-hlogit`) failed `G5` (`p10_ev < 0`) and are rejected.
