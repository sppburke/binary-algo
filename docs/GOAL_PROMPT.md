> **SCOPE: GENERIC** (currency/timeframe-agnostic). Per-key numbers, if cited, are tagged [PAIR·tf] examples whose record-of-truth is the Tier-2 file. See REPO_MAP.md.

# /goal prompt templates — exhaustive + generative + edge-IMPROVING strategy search

Fill `<X>` (timeframe, e.g. `5`) and `<CURRENCY>` (e.g. `EURUSD`). Paste after `/goal`.

**FILE CONVENTION (read `REPO_MAP.md` first; the `strategy-eval` skill §0a enforces it).** Keep generic ↔ key-specific
clean: a *discovered generic idea/lever* → `docs/IDEAS_LOG.md` + `SWEEP_MATRIX.md` Tier-N (+ `docs/METHODS_CATALOG.md` if new
method); the *per-key executable queue* (TOP-N experiments, FIRST-TO-RUN, incumbents-to-beat, discovery rounds) →
`sweeps/<CURRENCY>_<X>m_backlog.md`; *results of record* → `results/<CURRENCY>_RESULTS.md` (+ `docs/MAGNITUDE_FINDINGS.md`); *sweep
status* → `sweeps/<CURRENCY>_<X>m.md`; cross-key theory → `docs/THEORY.md`. Never bake per-key numbers into a generic file
(tag `[PAIR·tf]` + cite the Tier-2 file instead).

---

## ATTITUDE — read first, this governs everything

**We are hunting an edge and improving the edges we already have. We do NOT use research, null
results, or "the literature says it's hard" as permission to give up.**

- **Negative results describe other people's failures, not our ceiling.** When a paper reports "deep learning
  gets 50% on FX" or "X is unpredictable," that describes researchers who found *nothing*. We have, on this
  data, found *something* (e.g. a certified ~0.55–0.57 5m UP edge where those papers found 0.50). Their nulls
  **do not cap us.** Treat every "it can't be done" as "they couldn't do it," then ask what *we* have that they
  didn't (cross-pair structure, a strong magnitude edge, a disciplined gate).
- **A literature review's real value is a TOOLKIT of edge-IMPROVING levers, not a verdict to stop.** When a
  scholarly search returns mostly nulls, mine it for the *constructive* techniques buried inside —
  seed-ensembling, |return|-weighted / GMADL losses, calibration + adaptive conformal, decorrelated DL+GBM
  stacking, cross-pair pooling, regime adaptation — and GO RUN THEM against our existing edge. Do not let the
  null headline pull you into a defeatist frame; that framing is a bug, correct it.
- **Default to PURSUING, not confirming.** A null on one lever is a *redirect to the next lever*, never a reason
  to halt the search. "This specific method didn't improve it" ≠ "stop improving." Keep a running list of
  untried improvement levers and work it down. The search ends only when the genuine improvement levers are
  exhausted AND the new-method/discovery loop has gone dry for K rounds — not because a paper was pessimistic.
- **Rigor is for HONESTY about what we found, not for dismissing it.** The deriv-faithful discipline (below)
  exists so we never fool ourselves into a fake edge — and equally so we never *undersell* a real one. State
  the verdict precisely; if a harder test changes it, correct the record (in either direction). An edge that is
  "marginal/regime-dependent" is still an edge to improve, not to discard.
- **The binding constraint is usually DATA, and that is a TODO, not a wall.** If every model converges to the
  same ceiling, the lever is new *information* (external data: rate differentials, implied-vol/risk-reversal,
  triangular legs) or a better *use* of what we have (magnitude→direction bridge, cross-pair pooling) — pursue
  those, don't conclude "impossible."
- **MINE THE PAPERS EXHAUSTIVELY AND EXPERIMENT ON EVERYTHING.** The corpus in `/home/sean/git/academic-papers/`
  is fuel, not decoration. Read EVERY paper THOROUGHLY — every word, equation, diagram, table, AND reference list
  (chase the references that look load-bearing). Extract EVERY piece of logic / math / methodology / loss /
  architecture / gating / labeling / validation / framing that could plausibly raise the edge — no matter how
  small or how skeptical you are. Convert each into a falsifiable experiment with a pre-registered falsifier and
  RUN IT. **Bias hard toward action: it is better to have an idea DISPROVED BY EXPERIMENT than never tried.** Do
  not pre-dismiss an idea because a survey was pessimistic or because it "probably won't work" — that judgment is
  what experiments are for. Maintain the running idea→experiment backlog in `docs/IDEAS_LOG.md`; only retire an idea
  after it has actually been tested (or is provably subsumed by a Tier-1 result), never on a hunch.

## Reference: the academic-papers corpus

`/home/sean/git/academic-papers/` — a shared folder of downloaded papers and review reports.

**START HERE — `_CORPUS_INDEX.md`** distills all 370 papers so you DON'T re-scan the PDFs (two full agent scans
already cost ~10M tokens — do not repeat them). Per paper it has a NEUTRAL summary + general `topics`/`methods`/
`asset_class` tags (cross-task: grep your topic, e.g. `backtest-overfitting`, `conformal-prediction`,
`order-flow-imbalance`, `reinforcement-learning`) PLUS the 5m-FX-direction `levers` + a `fx5m_tier`. A
`tangential` tier means "no 5m-FX-direction lever," NOT "useless" — use the neutral summary/topics for other
tasks. Companions: `_corpus_index.json` (machine-readable), `_extracted_levers.json` (all 2050 levers with
mechanism+how-to-test+falsifier). Rebuild with `/tmp/ssrn_sess/build_corpus_index.py` after adding papers.

**Use it and grow
it:** when you research, download the actual PDFs here from ANY source (arXiv, OpenAlex/Unpaywall, author
homepages, SSRN via cookie) — OpenAlex's API (`api.openalex.org/works?...` → `best_oa_location.pdf_url`) and
Unpaywall (DOI→OA) are the most reliable, no-auth fetchers; arXiv `arxiv.org/pdf/<id>` never blocks. Write a
short review report (`_*.md`) summarizing the evidence + the edge-improving levers it surfaced, and cite the
saved PDFs by filename. (SSRN is Cloudflare-gated; a logged-in browser cookie passed by the user unblocks it for
targeted fetches.)

---

## Full prompt (recommended) — v3 (2026-06-03)
_v3 changes vs prior: §2 renamed DISCOVER+INVENT+COMBINE with an explicit combine-menu + "LEARN FROM PRIOR RESULTS BEFORE BUILDING" (attack the diagnosed failure cause, never re-permute a subsumed combo) + MEASURE-don't-guess + corpus via `_CORPUS_INDEX.md`/`_extracted_levers.json` (every lever → RUN or Tier-1-cited subsumption); §3 = explicit CROSS-PRODUCT model space + evaluate vs the ACTUAL deployment channel (binary AND spot-net-of-spread); §4 frozen-overstatement check (frozen≫refit ⇒ trust refit, baseline control reproduces honest #); §5 DO-NOT-DEPLOY sheet + live venue/channel. Placeholders {X}/{CURRENCY} (instantiated EURUSD·5m on line 1 — change line 1 to re-target). 5,514 chars._

```
{X}=5, {CURRENCY}=EURUSD. GOAL: best {X}m UP AND best {X}m DOWN {CURRENCY} binary-direction predictor, OOS-verified — TWO deliverables, each certified + deployable-spec'd. Don't stop until BOTH sides are certified-or-honestly-exhausted AND improve+discover loops go dry. NULL lit ≠ permission to quit (ATTITUDE): improve our edges, hunt more.

FILE CONVENTION (read REPO_MAP.md; strategy-eval §0a enforces): keep GENERIC (METHODS_CATALOG·SWEEP_MATRIX·IDEAS_LOG·THEORY) separate from KEY-SPECIFIC (results/{CURRENCY}_RESULTS.md · sweeps/{CURRENCY}_{X}m.md ledger · sweeps/{CURRENCY}_{X}m_backlog.md executable backlog · docs/MAGNITUDE_FINDINGS.md · MODEL_REGISTRY/books). Never bake a per-key number into a generic file — tag [PAIR·tf] + cite the Tier-2 file.

1. EXHAUSTIVE SWEEP, BOTH SIDES SYMMETRICALLY (strategy-eval SWEEP MODE). Work SWEEP_MATRIX.md row-by-row (Tier A→F), retarget EVERY permutation to {X}m via MX_HOR/HS. Reuse frozen books as baselines. Score COMBINED+UP+DOWN separately. A side isn't done until the FULL pipeline hits IT: (a) side-split; (b) side gate on worst-VAL-half; (c) purpose-built SPECIALIST; (d) confidence/coverage curve; (e) full-refit CPCV for any op-point whose binding (worst) year clears breakeven on CI-lower. Dead = run (a)-(d) & SHOW the wall, never infer from the other side.

2. DISCOVER + INVENT + COMBINE (continuous, mechanism-first, BOTH dirs). Scour arXiv(q-fin.TR/ST,stat.ML)/SSRN + cross-disciplinary; save every paper to /home/sean/git/academic-papers. MINE THE CORPUS EXHAUSTIVELY: read EVERY paper fully (chase load-bearing refs; use _CORPUS_INDEX.md + _extracted_levers.json), extract EVERY testable lever (math/loss/arch/gating/labeling/validation/framing) — EVERY lever ends as a RUN experiment OR a Tier-1-cited subsumption (none silently skipped). ACTIVELY COMBINE models/signals/levers you judge promising — cascade · gate/filter · blend · k-of-n consensus · stack · regime-route · meta-labeler-on-a-combination · cross-horizon · paper-lever×our-model — wherever a DIRECTION mechanism exists (survives sign-invariance; say which side's sign). LEARN FROM PRIOR RESULTS BEFORE BUILDING: read results/{CURRENCY}_RESULTS.md + ledger/backlog for what combinations already ran, what failed and WHY (diagnosed cause — common-factor/regime/info-bound/overfit); design the NEXT combination to ATTACK that cause and BUILD ON what came closest — never re-permute a subsumed combo. MEASURE assumptions (correlation, coverage); don't guess. Append generic ideas → SWEEP_MATRIX Tier-N + IDEAS_LOG; per-key rows → sweeps/{CURRENCY}_{X}m_backlog.md; test. Sub-agents/fan-out; loop until K dry rounds.

3. IMPROVE WHAT YOU FIND (don't stop at first certified book). The model space is a CROSS-PRODUCT: {base·cross-pair·cross-horizon·pooled·seed-ens·DL-stack} × {fixed·ACI·calibrated gate} × {BCE·|ret|/GMADL loss} × {filter·specialist}. Per edge, vs the BEST incumbent COMBINATION, run: seed-ensembling, |return|-weighted/GMADL loss (magnitude→direction bridge), calibration(temp/Venn-Abers)+ADAPTIVE-CONFORMAL gate, DL-as-decorrelated-stack, cross-pair POOLING, Optuna(capped+logged, worst-VAL-half) — AND novel combinations you believe (from prior results + mechanism) will beat it. Each: pre-registered falsifier — beat the incumbent's binding (worst) year or raise CPCV path-clear. Evaluate against the ACTUAL deployment objective (binary win-rate AND, where that's the channel, spot expectancy net of spread): a lever null for hit-rate can still matter for the deployable channel.

4. EVAL & CERT (non-negotiable) = strategy-eval §2-3: wc_ret/contig ties-LOSE; nonoverlap_chrono; per-year 24/25/26 CI95; worst-VAL-half NOT VAL-acc-max (corr=-0.54, bites SPECIALISTS hardest); moved-bars up-rate∈[.47,.53] tripwire; pre-registered falsifier in the result JSON BEFORE OOS. CERTIFY ONLY via full per-fold-refit CPCV at the OPERATING gate: refit p10≥breakeven AND ≥~80% folds clear. Precise vocab (forward-positive≠refit-certified≠regime-dependent≠dead). ADVERSARIALLY VERIFY every positive (leakage, refit-overfit, drift-not-skill, settlement, coverage): when frozen-book forward ≫ its refit-CPCV it's OVERSTATEMENT — trust the refit; a baseline control on the same harness reproduces the honest number. CORRECT THE RECORD when a harder test flips it.

5. DEPLOYMENT SPEC per survivor (each side; even a non-deployable side gets a documented DO-NOT-DEPLOY sheet w/ the disqualifying metrics): operating gate + coverage (trades/session); confidence/coverage curve; FRACTIONAL-Kelly on the ROBUST FLOOR (refit p10); equity path w/ max-DD + losing-streak; kill-switch TESTED not assumed; VERIFY real tradeable venue/min-duration/settlement from a LIVE source + test the actual channel (binary vs spot net-of-spread) before calling anything "best algo to trade".

6. SUBSUMPTION AUDIT + RECORDS. Mark "subsumed" only w/ Tier-1 rationale (script/JSON/theorem); AUDIT it adversarially — run, don't argue. Record COMBINED+UP+DOWN → results/{CURRENCY}_RESULTS.md (magnitude → docs/MAGNITUDE_FINDINGS.md); maintain ledger + executable backlog + two-sided leaderboard; freeze every survivor (incl. best COMBINATION) via manifest.py + git-tag. INCUMBENT = best COMBINATION in books/INDEX.json, NOT the old base GBM — retarget Tier-I levers + certified combinations to a new key, compare vs ALL. ONE heavy job at a time (OOM); commit often (flaky drive); EVIDENCE-FIRST — every number traces to a result JSON, never prose; flag thin-coverage (n<~50) + VAL-acc-max. DON'T GIVE UP — a null is a redirect, not a stop.
```

## Short prompt
```
/goal Best <X>-minute UP predictor AND best DOWN predictor for <CURRENCY>, OOS-verified — two deliverables, full
symmetric pipeline each (side-split → side-gate → side-specialist → confidence curve → full-refit CPCV at the
operating gate). Use the strategy-eval skill in sweep mode: run every SWEEP_MATRIX.md permutation at <X>m AND
research/invent new ones (download papers to /home/sean/git/academic-papers) AND run the edge-IMPROVING levers
(seed-ensemble, |ret|-weighted/GMADL loss, calibration+adaptive-conformal, DL+GBM stacking, cross-pair pooling)
on whatever you find. Certify = full-refit CPCV at the operating gate; ship a deployment spec (gate, fractional-
Kelly on the refit floor, equity/drawdown). Record combined+up+down in results/<CURRENCY>_RESULTS.md; resumable ledger
(sweeps/<CURRENCY>_<X>m.md) + executable backlog (sweeps/<CURRENCY>_<X>m_backlog.md) + two-sided leaderboard;
generic ideas→docs/IDEAS_LOG.md/SWEEP_MATRIX.md (see REPO_MAP.md); freeze survivors. Evidence-first; one heavy job at
a time; null literature ≠ give up — improve the edge.
```

## Notes
- **File convention:** `REPO_MAP.md` defines GENERIC (methods/menus/ideas/theory — `docs/METHODS_CATALOG.md`,
  `SWEEP_MATRIX.md`, `docs/IDEAS_LOG.md`, `docs/THEORY.md`) vs KEY-SPECIFIC (`results/<CURRENCY>_RESULTS.md`,
  `sweeps/<CURRENCY>_<X>m{,_backlog}.md`). Record per-key numbers ONLY in the key files; tag any example in a
  generic file `[PAIR·tf]`. The skill §0a enforces this.
- The skill (`.claude/skills/strategy-eval/SKILL.md`) is auto-discovered; the prompt sets the objective + scope.
- For a NEW currency with no data on disk, the first step is data acquisition (a prerequisite the skill flags).
- To resume an interrupted sweep, the same prompt works — the ledger is the state; pick up the first `pending`
  row, and the improvement/discovery loops continue.
- **The two failure modes to avoid are symmetric:** fooling yourself into a fake edge (cured by the discipline),
  and talking yourself out of a real one (cured by the ATTITUDE). Hold both.
```
