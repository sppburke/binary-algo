SCOPE: GENERIC construction reference (bar/candlestick-pattern 2-D CNN family) + the EURUSD 60s runnable experiment & verdict. Method goes in METHODS_CATALOG; per-key results live in EURUSD_RESULTS.md / sweeps/EURUSD_1m*.md.

# Bar / candlestick-pattern 2-D CNN — exact construction spec + EURUSD 60s experiment

Answers `/goal` ask (a) "thorough read of Sezer CNN-BI + Kronos → exact construction into a runnable
experiment spec" and ask (b) "build and run the bar-image CNN through the CPCV harness". The one
**non-subsumed** bar/candlestick sub-lever identified in the 2026-06-05 discovery vetting (commit 8bfa7f6,
`SWEEP_MATRIX.md` row N / "BAR / CANDLESTICK PATTERN recognition family"): a 2-D conv over rendered OHLC
**images** learns LOCAL 2-D pattern detectors that a per-bar GBM (239-feat ensemble) and a 1-D GRU/CNN
(`m_cnn.py`, `min1_v14_cnn.py`, SWEEP_MATRIX D1, already null at 60s) cannot represent. Explicit candlestick
*features* (rangepos/atr/gap/bb_width) are already in the GBM and are magnitude-not-sign → subsumed; the
image conv was the only untested piece.

## 1. Sezer & Ozbayoglu CNN-BI — exact construction (Tier-1, read from the PDF)
Source: `OA_Sezer_AlgorithmicFinancialTrading_StockBarChartImageCNN.pdf` = "Financial Trading Model with Stock
Bar Chart Image Time Series with Deep CNNs", arXiv:1903.04610v1 (2019). Verified by reading all pages + figs.

- **Image (Sec.3 "image creation", Figs.2–3).** Slide a **30-trading-day** window over the daily **close**
  series, stride 1. Min-max normalize the 30 closes WITHIN the window. Render a **30×30×1 binary** image as a
  **bottom-anchored vertical-bar histogram**: column i (x = time, i=0..29) is filled black from the bottom up
  to height_i = round(29·(c_i − min)/(max − min)); rest white. **It is NOT an OHLC candlestick** — it is a
  one-bar-per-day close-level histogram (confirmed by Fig.3's SELL/BUY/HOLD samples).
- **Label (Sec.3, Eqs.1–3, Algo.1) — HAS LOOK-AHEAD.** 3-class Buy/Hold/Sell from FUTURE slopes measured from
  the window's last day: slopeRef=(v34−v30)/(34−30), slopeCurrent=(v45−v30)/(45−30); cut at the 2/5 & 3/5
  quantiles of the sorted training slopeRef list. **Uses days +4 and +15 ahead → not usable as-is** (leakage).
- **CNN (Sec.3, Fig.5) — MNIST-class, 8 layers.** Input 30×30×1 → Conv2D 32×(3×3,same)+ReLU → Conv2D
  64×(3×3,same)+ReLU → MaxPool 2×2 → Dropout 0.25 → Flatten → Dense 128+ReLU → Dropout 0.5 → Dense(softmax).
- **Training.** 100 epochs, batch 1028, categorical-CE, Adam (default lr), resample(train) to balance classes,
  per-stock model, **no retraining** forward (tests reliability).
- **Reported (Dow-30 daily equities only; NO FX, NO intraday).** Classification accuracy **44–52%** (3-class,
  random=33%); transaction success 52.4% / 53.4%; beats Buy&Hold in the volatile 2007–12 regime, loses in the
  2012–17 bull. Modest, daily, equity. **Prior for an FX 60s binary: low (~0.50–0.55).**

## 2. GAF / MTF image encoding — exact math (Wang & Oates 2015, arXiv:1506.00327; pyts conventions)
Length-W series X. (1) min-max scale to [−1,1]: x̃_i = ((x_i−max)+(x_i−min))/(max−min). (2) polar:
φ_i = arccos(x̃_i) ∈ [0,π]. (3) **GASF_ij = cos(φ_i+φ_j)** = x̃_i x̃_j − √(1−x̃_i²)√(1−x̃_j²) — **symmetric →
magnitude/shape**. (4) **GADF_ij = sin(φ_j−φ_i)** = √(1−x̃_i²)·x̃_j − x̃_i·√(1−x̃_j²) — **antisymmetric →
the SIGN/ordering-bearing field** (GADF_ji = −GADF_ij). (5) MTF: Q quantile bins → 1st-order Markov transition
matrix W (Q×Q, row-normalized) → MTF_ij = W[bin(x̃_i), bin(x̃_j)]. Stack as a multi-channel W×W image. The
sign-invariance theorem predicts GASF gates magnitude; only GADF can carry 60s direction.

## 3. Kronos — exact construction + feasibility (Tier-3/4: arXiv:2508.02739 + GitHub shiyu-coder/Kronos)
Two-stage K-line foundation model. **Tokenizer** = Transformer autoencoder with a **Binary Spherical
Quantization (BSQ)** layer: each OHLCV bar → continuous latent → projected to a unit hypersphere → binarized
to a **k-bit code (k≈20)**, factorized into **n=2 subspaces → exactly 2 hierarchical tokens/bar** (coarse +
fine, each over a 2^(k/2) sub-vocab; factorizing cuts the vocab head ~99.8%, ≈1.7B→3.4M params). **Stage 2** =
decoder-only causal Transformer, **autoregressive next-token MLE**, coarse-then-fine factorization, pretrained
on >12B K-lines / 45 exchanges. Checkpoints: mini **4.1M** / small 24.7M / base 102.3M (open) / large 499M.
**Use** = zero-shot generative forecasting (sample token trajectories, decode, ensemble). **Reported gains are
RankIC (correlation) +87–93%, vol-MAE, generative fidelity — NOT directional accuracy; no FX, no 60s, no
per-bar sign %.** Runs on CPU (tiny). **Verdict:** a from-scratch Kronos-*mini*-style AR tokenized-OHLCV model
is CPU-feasible on ~400k EURUSD 1-min bars, BUT it optimizes reconstruction + next-token likelihood (magnitude
/path), not sign; arXiv:2511.18578 ("Re(Visiting) TSFMs") finds off-the-shelf TSFMs poor zero-shot AND under
fine-tuning, only from-scratch financial pretraining helps. → magnitude/path probe, NOT a 60s direction lever;
deprioritized (matches the established 60s near-efficiency).

## 4. The runnable EURUSD 60s experiment (as implemented)
- `barcnn_bars.py` → 1-min OHLCV bars from `features_tick/*_1s.parquet` (mid→OHLC, nt→volume), right-closed so
  bar timestamp == decision instant; 60s up/down label via the deriv-faithful `wc_ret` (next-tick entry +1s,
  exit last tick ≤+61s, **ties LOSE**) computed on the same 1s clock. Moved up-rate ∈ [0.497,0.503] every year
  (no fake-flat mirage). Splits: train 2021–23 (418k moved), val 2024-H1 (49.7k), test 2024.09–25.11 (234k),
  oos 2026 (78.5k).
- `barcnn_run.py <variant>` → per-window min-max-normalized images (causal), small 2-D CNN (faithful Sezer
  arch: Conv32→Conv64→MaxPool→Dropout.25→Dense128→Dropout.5→1-logit sigmoid; **BCE not the look-ahead slope
  label**), AdamW lr1e-3 wd1e-4, batch1024, subsample train 80k, VAL-AUC early-stop. Variants: **hist** (Sezer
  close-histogram 1×30×30), **ohlc** (3×30×30: high-low wick / up-body / down-body — adds per-bar sign),
  **gaf** (2×30×30: GASF+GADF). Predicts moved+valid+contiguous-window held-out bars.
- `barcnn_cpcv.py <variant>` → faithful CPCV (mirrors `min1_cpcv.cpcv_side`: 8 groups, C(8,2)=28 purged paths,
  block-bootstrap, ties-strict) on the frozen-CNN selected test+oos trades at cov {100,10,5,2}%. CERTIFY iff
  path_p10 ≥ 0.541 AND block-boot CI95-lo ≥ 0.541. (frozen-model CPCV = optimistic bound → a FAIL is decisive.)

### Pre-registered falsifier (written before viewing OOS)
KILL the variant if **VAL dirAUC ≤ 0.515** OR **no held-out year's moved-acc CI95-lower ≥ 0.541** OR
**frozen-model CPCV path_p10 < 0.541**. (Breakeven 0.541; target 0.65.)

## 5. RESULTS & verdict
(see `barcnn_{variant}_result.json` + `barcnn_cpcv_{variant}_result.json`; filled in §EURUSD_RESULTS.md)

| variant | best VAL dirAUC | test AUC | oos AUC | held-out selective (all yr/cov) | CPCV path_p10 (range over cov) / frac_clear | verdict |
|---|---|---|---|---|---|---|
| hist (Sezer CNN-BI close-hist) | 0.4992 | 0.5029 | 0.5030 | 0.49–0.52, every CI95-lo < 0.519 | 0.486–0.499 / 0.0 (cov0.02 frac 0.036, n2996 noise) | **KILLED** |
| ohlc (3ch wick+up/down body) | 0.5028 (es 0.5072) | 0.5011 | 0.5018 | 0.49–0.52, every CI95-lo < 0.516 | 0.490–0.499 / 0.0 | **KILLED** |
| gaf (GASF+GADF, sign field) | 0.5020 | 0.5006 | 0.4997 | 0.48–0.50, every CI95-lo < 0.515 | 0.484–0.499 / 0.0 | **KILLED** |

All three trip the pre-registered falsifier on ALL THREE conditions (VAL dirAUC ≤ 0.515; no held-out year
CI95-lo ≥ 0.541; CPCV path_p10 < 0.541). Incumbent unbeaten (60s UP 0.613 regime-filter uncertified; DOWN
0.516 dead). >0.65 target not achieved — as the honest prior predicted at ≤5m.

### 5b. "Better than we currently have?" — bar-CNN × incumbent regime gate (`barcnn_regime.py`)
Apples-to-apples vs the 0.613 UP filter: gate the CNN to the compression-release reversion regime (the only 60s
structure) and compare to the pure-reversion baseline (CNN ignored). Result (`barcnn_regime_*_result.json`): the
pure-reversion baseline itself is ~0.48–0.53 / CPCV p10 .43–.47 / 0.0 paths clear on the 1-min-bar regime proxy;
the CNN-gated policy throws scattered high pockets (ohlc 2025 .615, hist 2024 .618) that are **thin-coverage
mirages (n=34–39/yr — leakage trap #6), inconsistent across years (2026 < .50), CPCV p10 .436–.482, frac_clear <
0.54** → none certifies, none beats the incumbent binding-year with CI clearing, none reaches 0.65. The bar
pattern adds nothing to the regime gate.

**Bottom line:** the bar-image CNN does NOT beat the incumbent 60s book and does NOT clear breakeven — VAL
dirAUC ≈ 0.50 and every held-out year's CI95-lower sits far below 0.541. This is the predicted outcome: the
60s EUR/USD direction sign is near-efficient (image geometry of past bars carries magnitude, not next-60s
sign — sign-invariance theorem), while the program's certified edge remains magnitude. The bar/candlestick
2-D-image lever is now **RUN and EXHAUSTED on-disk for EURUSD 60s** (was the last non-subsumed bar sub-lever).
