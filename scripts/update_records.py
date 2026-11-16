#!/usr/bin/env python3
"""Insert or refresh the Deriv-feed program blocks in the repository ledgers, straight from Tier-1 sources.

    update_records.py [PAIR ...]

results/<PAIR>_RESULTS.md : the section '## <PAIR> — DERIV-FEED PROGRAM ...' is appended once; the block between
                            <!-- DERIV-FEED-TABLES:BEGIN/END --> is re-rendered from results/json (render_ledger).
sweeps/<PAIR>_15m.md      : the section '## DERIV-FEED PROGRAM LEDGER ...' is appended once; the block between
                            <!-- DERIV-FEED-LEDGER:BEGIN/END --> is regenerated from the lab registry, the lab queue and
                            the result JSONs (status per experiment).
Hand-written commentary lives outside the markers and is never touched.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path('/home/sean/git/binary-algo')
LAB = Path('/mnt/data/deriv-sota-lab')
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD']
OPENED = '2026-11-15'
EXPERIMENTS = {   # id -> (what, harness)  — generic descriptions; numbers come only from the JSONs
    'r2': ("SUPERSEDED (pre-tradeable-filter, harness 0.2.1): incumbent recipe refit on Deriv data, NY 08-17", 'cpcv_deriv.py'),
    'e1': ('SUPERSEDED (pre-tradeable-filter, harness 0.2/0.2.1): LightGBM on Deriv-native features v0.1, NY 08-17', 'cpcv_deriv.py'),
    'e1t': ('LightGBM on Deriv-native features v0.1 (66: own-pair wall-clock returns, volatility, range, EMA, activity; '
            'clock; other pairs as-of), nseed 1, NY 08:00-16:35 tradeable', 'cpcv_deriv.py'),
    'r2t': ("Incumbent recipe refit on Deriv data (deployed book's exact columns, seeds, deploy coverage), true clock, "
            'NY 08:00-16:35 tradeable', 'cpcv_deriv.py'),
    'e14nylate': ('E1t model in true NY 13:00-16:35 + 18:00-22:00 (the hours the Dukascopy-era NY gate actually selected), '
                  'tradeable', 'cpcv_deriv_x.py'),
    'e15nypm': ('E1t model in NY 13:00-16:35 (tradeable afternoon)', 'cpcv_deriv_x.py'),
    'e16nyeve': ('E1t model in NY 18:00-22:00 (post-rollover evening)', 'cpcv_deriv_x.py'),
    'e17nyam': ('E1t model in NY 08:00-13:00 (morning)', 'cpcv_deriv_x.py'),
    'r3': ("Incumbent recipe refit on Deriv data on the books' TRAINING clock (-5 h features: 4h bars aligned to the 17:00 ET "
           'close), NY 13:00-16:35 + 18:00-22:00 tradeable', 'cpcv_deriv_x.py'),
    'e2noclock': ('E1t without the clock features (hour, weekday): trap #9 ablation', 'cpcv_deriv.py'),
    'e6pool': ('E1t own-pair features, training pooled over all 7 pairs (EURUSD/GBPUSD/USDCHF targets only: the JPY/CAD/AUD/NZD '
               'family pooling closure is respected)', 'cpcv_deriv_pool.py'),
    'e7fitall': ('E1t fit on all tradeable hours, evaluated on NY 08:00-16:35', 'cpcv_deriv_x.py'),
    'e3union': ('E1t features + the incumbent schemas (union)', 'cpcv_deriv.py'),
    'e4seed3': ('E1t with a 3-seed ensemble', 'cpcv_deriv.py'),
    'e5h30': ('E1t at the 30-minute horizon (blackout start window [16:20, 18:00) NY)', 'cpcv_deriv.py'),
    'e8tb2': ('E1t fit on the triple-barrier first-touch label (k=2, repository definition); validation, thresholds and '
              'scoring on the settlement sign', 'cpcv_deriv_x.py'),
    'e9blockval': ('E1t with a time-block validation set (last 15% of training rows, purged) setting the threshold; all '
                   '800 trees', 'cpcv_deriv_x.py'),
    'e10v02': ('E1t + 16 literature-driven causal features (same-slot seasonality, round-number geometry, semivariances): '
               'dataset v0.2', 'cpcv_deriv.py'),
    'e11noslow': ('E1t without the slow features (480/1440-minute windows): trap #9 ablation', 'cpcv_deriv.py'),
    'e12noslowclock': ('E1t without slow and clock features: trap #9 ablation', 'cpcv_deriv.py'),
    'e18lowcap': ('E1t with a low-capacity, strongly regularized LightGBM (31 leaves, min_child 2000, colsample 0.3, 600 trees) '
                  'for the ~15-month Deriv data ceiling', 'cpcv_deriv_x.py'),
    'e13mag50': ('E1t + magnitude gate: a second model ranks big moves (|ret| >= fit-pool median); trade only its top 50% '
                 '(direction coverage relative to gated rows)', 'cpcv_deriv_x.py'),
    # SWEEP_MATRIX family blocks on the Deriv feed (tools/deriv_features_family.py, dataset_dev_v0.3 = v0.1 + blocks)
    'e19tf': ('E1t + SWEEP_MATRIX B5 tick-rule signed flow block (tf_: up/down mid-change imbalance, 5-240 min)', 'cpcv_deriv_x.py'),
    'e20sg': ('E1t + SWEEP_MATRIX B7 path-signature block (sg_: Levy areas time-price and price-USD basket)', 'cpcv_deriv_x.py'),
    'e21kf': ('E1t + SWEEP_MATRIX C2 Kalman block (kf_: local-linear-trend slope/innovation, time-varying basket beta and '
              'residual)', 'cpcv_deriv_x.py'),
    'e22rm': ('E1t + SWEEP_MATRIX C3 random-matrix eigen-residual block (rm_: residual after Marchenko-Pastur modes, 15-240 '
              'min)', 'cpcv_deriv_x.py'),
    'e23fd': ('E1t + SWEEP_MATRIX C6 fractional-differencing block (fd_: d = 0.3, 0.5, z-scored)', 'cpcv_deriv_x.py'),
    'e24cx': ('E1t + SWEEP_MATRIX E3 complexity/regime block (cx_: permutation entropy, autocorrelation, variance ratio, '
              'two-state volatility-regime filter)', 'cpcv_deriv_x.py'),
    'e25tk': ('E1t + SWEEP_MATRIX E4 tick-clock (information-bar) block (tk_: returns over the last 100/500/2,000 mid '
              'changes)', 'cpcv_deriv_x.py'),
    'e26pa': ('E1t + SWEEP_MATRIX F2 price-action block (pa_: RSI, Bollinger %b, NR7, NY-day open/range/previous-day '
              'levels)', 'cpcv_deriv_x.py'),
    'e27famall': ('E1t + all eight SWEEP_MATRIX family blocks together', 'cpcv_deriv_x.py'),
    'e28resid': ('E1t fit on the SWEEP_MATRIX F4 residualized target (own 15m return minus beta x USD-basket return); '
                 'validation, thresholds and scoring on the settlement sign', 'cpcv_deriv_x.py'),
    'e29logit': ('E1t features, L2 logistic regression (C = 0.05, fit-pool median imputation, z-scored, clipped at 5): '
                 'low-capacity linear model for the ~15-month data ceiling', 'cpcv_deriv_x.py'),
    'e30nystroem': ('E1t features, SWEEP_MATRIX B6 kernel model: RBF Nystroem features (500) + logistic SGD', 'cpcv_deriv_x.py'),
    'e31ens3': ('E1t features, SWEEP_MATRIX A1 three-GBM ensemble (LightGBM + XGBoost + CatBoost mean; 400 trees at lr 0.04, '
                'CatBoost depth 6; capped fast-KILL screen without the forward walk)', 'cpcv_deriv_x.py'),
    'e32spec': ('E1t features, SWEEP_MATRIX A8 side specialists (scripts/min1_upspec.py recipe at 15m): in the compression '
                'regime an UP model fit only on dip setups (ret_15m < 0) trades only UP, a DOWN model fit only on rally setups '
                'trades only DOWN', 'cpcv_deriv_x.py'),
    'e33comp33': ('E1t + SWEEP_MATRIX A2 compression gate: trade only when rv_60m <= its fit-pool 33% quantile; operating '
                  'coverage 0.10 of gated validation rows (pre-set so the overall trade share stays near the ungated 0.03)',
                  'cpcv_deriv_x.py'),
    'e34online': ('E1t features, SWEEP_MATRIX C5 online learner: river Adaptive Random Forest (10 trees, ADWIN) learned '
                  'prequentially (each decision only after it settles), causal rolling threshold, scored on the CPCV group '
                  'pairs (forward-only by construction; no per-fold refit)', 'cpcv_deriv_online.py'),
    'e35absret': ('E1t with |return|-weighted training (Tier-I loss lever: LightGBM sample weight |settled return| / mean)',
                  'cpcv_deriv_x.py'),
    'e36aci': ('E1t with an adaptive-conformal confidence gate (Tier-I: threshold += 0.002 x (loss - 0.44) as each taken '
               'trade settles, causal, per test path)', 'cpcv_deriv_x.py'),
    'e37meta': ('E1t + SWEEP_MATRIX A4 meta-labeler gate: out-of-fold primary (two purged chronological halves of the fit '
                'pool), meta model of P(primary side correct), keep the top 50%; operating coverage 0.06 of gated validation '
                'rows (pre-set so the overall trade share stays near 0.03)', 'cpcv_deriv_x.py'),
    'e38release': ('E1t features, SWEEP_MATRIX A3 compression-release x reversion specialists: after a volatility burst '
                   '(volratio_15_240 >= fit-pool 75% quantile) an UP model fit on dips trades only UP, a DOWN model fit on '
                   'rallies trades only DOWN', 'cpcv_deriv_x.py'),
    'e40forest': ('E1t features, shallow bagged extra-trees (300 trees, depth 8, min leaf 200; literature: weak-signal tree '
                  'baseline)', 'cpcv_deriv_x.py'),
    'e41pool': ('E1t features, equal-weight probability pool of LightGBM, L2 logistic, extra-trees and RBF-kernel models '
                '(literature: heterogeneous equal-weight pooling)', 'cpcv_deriv_x.py'),
    # Literature consolidation ranks 1-3, 11 (literature/consolidation/work/consolidated.jsonl; dataset_dev_v0.5)
    'r4fixldn': ('Zero-parameter rule (literature rank 1, pre-registered post-fix reversal): in the hour after the London 16:00 '
                 'fix trade against the pair\'s move over the hour before the fix (cpcv_rule; every signalled decision)', 'cpcv_rule.py'),
    'r5fixecb': ('Zero-parameter rule: post-fix reversal after the ECB 14:15 Frankfurt reference rate (cpcv_rule)', 'cpcv_rule.py'),
    'e51fx': ('E1t + literature rank 1 fixing-window block (fx_: London 16:00 and ECB 14:15 relative time, triangular '
              'pre/post bases and their products with USD-oriented 15/60-min returns, pre-fix move)', 'cpcv_deriv_x.py'),
    'e52ck': ('E1t + literature rank 2 compact return-kernel block (ck_: scaled cumulative and exponentially weighted returns, '
              'fast-minus-slow)', 'cpcv_deriv_x.py'),
    'e53ckcompact': ('Literature rank 2 compact dictionary alone (14 features: ck_ kernels, scale, clock) with LightGBM',
                     'cpcv_deriv_x.py'),
    'e54ckcompactlogit': ('Literature rank 11 compact sign logistic: the 14 compact features with L2 logistic regression',
                          'cpcv_deriv_x.py'),
    'e55rl': ('E1t + literature rank 3 round-level block on the REAL 50/100-pip grids (rl_)', 'cpcv_deriv_x.py'),
    'e56rs': ('E1t + the same round-level block on half-grid SHIFTED placebo levels (rs_); falsifier: e55rl must beat it',
              'cpcv_deriv_x.py'),
    'e57reflect': ('E1t with within-pair sign reflection (literature rank 8): mirrored fit rows, label flipped, half weight '
                   'each - no drift, only sign-symmetric structure', 'cpcv_deriv_x.py'),
    'e58nysreflect': ('e30nystroem with within-pair sign reflection', 'cpcv_deriv_x.py'),
    'e59brier': ('E1t features, LightGBM fit on the Brier score (literature rank 4; custom objective, probabilities from the raw '
                 'score)', 'cpcv_deriv_x.py'),
    # Levers on the best USDJPY 15m UP edge (e30nystroem); USDJPY only
    'e42nysnoclock': ('e30nystroem without the clock features (trap #9 ablation on the best edge)', 'cpcv_deriv_x.py'),
    'e43nysnoslow': ('e30nystroem without the slow 480/1440-minute features (trap #9 ablation on the best edge)', 'cpcv_deriv_x.py'),
    'e44nysseed3': ('e30nystroem with 3 seeds (Nystroem components and SGD seeds averaged)', 'cpcv_deriv_x.py'),
    'e45nysaci': ('e30nystroem with the adaptive-conformal gate (step 0.002, target 0.56)', 'cpcv_deriv_x.py'),
    'e46nysg05': ('e30nystroem with RBF gamma x 0.5 (capped, logged kernel-width grid)', 'cpcv_deriv_x.py'),
    'e47nysg2': ('e30nystroem with RBF gamma x 2 (capped, logged kernel-width grid)', 'cpcv_deriv_x.py'),
    'e48nysc1000': ('e30nystroem with 1,000 Nystroem components', 'cpcv_deriv_x.py'),
    'e49nys30': ('e30nystroem at the 30-minute horizon', 'cpcv_deriv_x.py'),
    'e39cm': ('E1t + SWEEP_MATRIX C4 convergent cross-mapping block (cm_: rolling CCM skill of each other pair onto the '
              'target, E 3, library 240 bars, and the coupled drivers\' 15-minute push; dataset_dev_v0.4)', 'cpcv_deriv_x.py'),
    # Tier-I levers on each key's best DEV edge (pre-registered in the lab PROGRESS.md, 2026-11-15 13:15 UTC); harness 0.5+
    'c1nys3': ('Kernel base (e30nystroem) with 3 seeds; USDCAD (USDJPY = e44nysseed3)', 'cpcv_deriv_x.py'),
    'c2nysabs': ('Kernel base with |return| training weights (mean-1 SGD sample weights, harness 0.5)', 'cpcv_deriv_x.py'),
    'c3nysaci': ('Kernel base with the adaptive-conformal gate (step 0.002, target 0.56); USDCAD (USDJPY = e45nysaci)', 'cpcv_deriv_x.py'),
    'c4nysmeta': ('Kernel base with the meta-label gate (keep 0.5; half-pool primaries are the kernel, harness 0.5)', 'cpcv_deriv_x.py'),
    'c5nysmag': ('Kernel base with the learned magnitude gate (keep 0.5)', 'cpcv_deriv_x.py'),
    'c6nyscomp': ('Kernel base with the compression gate (rv_60m <= fit-pool 0.33 quantile)', 'cpcv_deriv_x.py'),
    'c7comp3': ('Compression-gate base with 3 seeds at coverage 0.03 (not the base 0.10: a coverage variant; see c7comp3b)', 'cpcv_deriv_x.py'),
    'c8compabs': ('Compression-gate base with |return| training weights', 'cpcv_deriv_x.py'),
    'c9compaci': ('Compression-gate base with the adaptive-conformal gate (step 0.002, target 0.56)', 'cpcv_deriv_x.py'),
    'c10compmeta': ('Compression-gate base with the meta-label gate (keep 0.5)', 'cpcv_deriv_x.py'),
    'c11compmag': ('Compression-gate base with the learned magnitude gate (keep 0.5)', 'cpcv_deriv_x.py'),
    'c7comp3b': ('Compression-gate base with 3 seeds at the base coverage 0.10 (c7comp3 ran at 0.03 by mistake)', 'cpcv_deriv_x.py'),
    'c6nyscompb': ('Kernel + compression gate on the gate keys at the base coverage 0.10 (c6 ran at 0.03)', 'cpcv_deriv_x.py'),
    # Literature items, harness 0.6
    'e60stack': ('E1t features, shrunk simplex stack of LightGBM, L2 logistic, extra-trees and RBF kernel (validation '
                 'log-loss weights shrunk 0.5 toward equal; literature rank 6)', 'cpcv_deriv_x.py'),
    'e61ord4': ('E1t features, LightGBM on 4 ordinal signed bins (fit-pool median |return|), P(up) = P(small up) + P(big up) '
                '(literature rank 25)', 'cpcv_deriv_x.py'),
    'e62har': ('E1t + HAR size gate (OLS of log|return| on log realized volatility 15 m / 60 m / 4 h / 1 d and the clock; '
               'keep 0.5; literature ranks 12-15)', 'cpcv_deriv_x.py'),
    'e63bootnys': ('Kernel base with the calibration-uncertainty gate (5 bootstrap replicas, rank by |mean - 0.5| - 1 sd; '
                   'literature rank 16)', 'cpcv_deriv_x.py'),
    'e64boot': ('E1t with the calibration-uncertainty gate (5 bootstrap replicas, rank by |mean - 0.5| - 1 sd; literature '
                'rank 16)', 'cpcv_deriv_x.py'),
    'e65recency': ('E1t with recency training weights (half-life 60 days back from the latest fit row; literature rank 7; '
                   'pre-registered reading: the forward walk only)', 'cpcv_deriv_x.py'),
    'e66mh': ('E1t with shared-horizon partitions: fit rows also enter with the 30-minute label and an indicator column '
              '(30-minute purge), scored at 15 minutes (literature rank 24; harness 0.6.1)', 'cpcv_deriv_x.py'),
    # Discovery round 1 (lab discovery/round1; pre-registered in the lab PROGRESS.md 2026-11-15 13:40 UTC); harness 0.7
    'e67drfe': ('Drift-neutral L2 logistic on the signed coordinates with ISO-week x NY-3-hour x rv tercile fixed effects '
                'excluded from scores (discovery D1-01)', 'cpcv_deriv_x.py'),
    'e67drfenys': ('RBF kernel with the same drift fixed effects excluded from scores (discovery D1-01 kernel variant)', 'cpcv_deriv_x.py'),
    'e68agree': ('E1t fit on the side agreed by settlements at entry delays 1, 5 and 15 s (discovery D1-02)', 'cpcv_deriv_x.py'),
    'e69agreenys': ('RBF kernel fit on the entry-delay agreement label (discovery D1-02 kernel variant)', 'cpcv_deriv_x.py'),
    'e70play': ('E1t + hysteretic play-operator block hy_ (240-minute replays; discovery D1-03)', 'cpcv_deriv_x.py'),
    'e71playreset': ('E1t + play-operator control block hr_ (15-minute replays; discovery D1-03 memory control)', 'cpcv_deriv_x.py'),
    'e72shock': ('E1t + signed shock-response curvature x forecast USD-factor variance (d2_shockvar; discovery D2-01)', 'cpcv_deriv_x.py'),
    'e73mono': ('E1t LightGBM with monotone catch-up constraints on ret_15m and x_usd_weak_15m (discovery D2-02)', 'cpcv_deriv_x.py'),
    'e74logitsig': ('ATTRIBUTION CONTROL for D1-01: L2 logistic on e67drfe\'s signed coordinates and scaling with a global '
                    'intercept, no strata (harness 0.8 --signed-only)', 'cpcv_deriv_x.py'),
    'e75lgbsig': ('LightGBM on e67drfe\'s signed coordinates only (same-input control of D3-01; harness 0.8 --signed-only)', 'cpcv_deriv_x.py'),
    'e76drfelgb': ('LightGBM on the signed coordinates with profiled week x NY-3-hour x rv-tercile offsets excluded from scores '
                   '(discovery D3-01; harness 0.8 --drift-fe)', 'cpcv_deriv_x.py'),
    'e77drfenysabs': ('RBF kernel with |return| weights and the drift fixed effects excluded from scores (discovery D3-02)', 'cpcv_deriv_x.py'),
    'e78nysagree': ('Kernel (|return| weights) proposals kept only where the drift-neutral logistic agrees and its plain '
                    'same-input twin opposes (discovery D3-04; harness 0.8 --model nysagree)', 'cpcv_deriv_x.py'),
    'e79qorder': ('E1t + quote-update order fingerprints after common USD shocks, block qo_ (discovery D3-05)', 'cpcv_deriv_x.py'),
    'e80qperm': ('E1t + the permuted-order control block qp_ (discovery D3-05 control)', 'cpcv_deriv_x.py'),
    'e81tail0': ('Kernel features fitted by full-batch L-BFGS (BCE + ridge), compression gate 0.33: the matched control '
                 'of discovery D4-01 (tail auxiliary weight 0)', 'cpcv_deriv_x.py'),
    'e82tailaux': ('Kernel + compression gate with the fixed-mass tail auxiliary (weight 0.1: soft loss rate of the operating '
                   '3% confidence tail of gated fit rows, mass held fixed; discovery D4-01)', 'cpcv_deriv_x.py'),
    'e83nysrefcomp': ('Kernel with within-pair sign reflection inside the compression gate (discovery D5-02; USDCAD UP)',
                      'cpcv_deriv_x.py'),
    'e84bootcomp': ('Compression-gate base with the calibration-uncertainty gate (5 bootstrap replicas, rank by |mean - 0.5| '
                    '- 1 sd; discovery D5-04; USDCHF UP)', 'cpcv_deriv_x.py'),
    'e85bootcomp0': ('Compression-gate base on the mean of the same 5 bootstrap replicas, no uncertainty penalty: the matched '
                     'control of discovery D5-04', 'cpcv_deriv_x.py'),
    'e86blendsel': ('Compression-gate base: blend of the unweighted and |return|-weighted LightGBM, weight in {0, 0.5, 1} '
                    'chosen on the worse chronological half of the block-validation DOWN trades (discovery D5-01 as '
                    'amended; NZDUSD DOWN)', 'cpcv_deriv_x.py'),
    'e87fixrank': ('London 16:00 fix-reversal rule offers (as r4fixldn, DOWN side) ranked by the compact logistic at 3% '
                   'overall coverage (discovery D5-03; EURUSD DOWN)', 'cpcv_deriv_x.py'),
    'e88comp33blk': ('Compression-gate base with block validation (last 15% of training rows, purged): the matched '
                     'unweighted comparator of the amended D5-01', 'cpcv_deriv_x.py'),
    'e89compabsblk': ('Compression-gate base with |return| weights and block validation: the matched weighted comparator '
                      'of the amended D5-01', 'cpcv_deriv_x.py'),
    'a12b02': ('ROBUSTNESS CHECK, not a candidate: e30nystroem re-run on the deriv_dataset 0.2 DEV build (holdout builder)', 'cpcv_deriv_x.py'),
    'a13b02': ('ROBUSTNESS CHECK, not a candidate: e49nys30 re-run on the deriv_dataset 0.2 DEV build (holdout builder)', 'cpcv_deriv_x.py'),
}
NOT_CANDIDATES = {k for k, v in EXPERIMENTS.items() if v[0].startswith('ROBUSTNESS CHECK')}

R1_FILES = {   # frozen-book readings: file stem -> description
    'deriv_r1_frozen_books_dev_result.json': 'R1: frozen books as deployed (true clock, NY 08-17), ALL decisions incl. the untradeable blackout',
    'deriv_r1b_frozen_books_trainclock_nylate_dev_result.json': 'R1b: frozen books as trained (clock -5 h, 13-22 ET), ALL decisions incl. the blackout',
    'deriv_r1t_frozen_books_dev_tradeable_result.json': 'R1t: frozen books as deployed (true clock, NY 08-17), tradeable decisions only',
    'deriv_r1bt_frozen_books_trainclock_nylate_dev_tradeable_result.json': 'R1bt: frozen books as trained (clock -5 h, 13:00-16:35 + 18:00-22:00 ET), tradeable only',
}


sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_ledger import MIRAGE_OK, amended_verdict, effective, goal_verdict, path_test  # noqa: E402  (shared goal / amendment rules)


def registry():
    rows = [json.loads(l) for l in (LAB / 'registry' / 'tests.jsonl').read_text().splitlines() if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r['test_id'], []).append(r)
    return by


def queued():
    q = set()
    for line in (LAB / 'queue' / 'pending.txt').read_text().splitlines():
        m = re.search(r'run_(?:cpcv_)?batch\.sh (\S+) (\d+) "([^"]+)"', line)
        if m:
            for p in m.group(3).split():
                q.add((m.group(1), p, int(m.group(2))))
    running = (LAB / 'queue' / 'running.txt').read_text() if (LAB / 'queue' / 'running.txt').exists() else ''
    m = re.search(r'run_(?:cpcv_)?batch\.sh (\S+) (\d+) "([^"]+)"', running)
    if m:
        for p in m.group(3).split():
            q.add((m.group(1), p, int(m.group(2))))
    return q


def ledger_block(pair, reg, q):
    lines = ['| id | what | H | status | verdict UP / DOWN (operating cov) | result JSON |', '|---|---|---|---|---|---|']
    for fname, what in R1_FILES.items():
        r1 = REPO / 'results' / 'json' / fname
        if not r1.exists():
            continue
        r = json.loads(r1.read_text())
        books = [b for b in r['books'] if b['pair'] == pair]
        v = '; '.join(f'{b["book_id"]}: ' + (f'{b["sides"]["UP"]["verdict"]} / {b["sides"]["DOWN"]["verdict"]}' if b['status'] == 'SCORED'
                                            else b['status']) for b in books)
        lines.append(f'| {what.split(":")[0]} | {what.split(": ", 1)[1]} (`score_frozen_books.py`) | 15m/30m | done | {v} | `{fname}` |')
    for exp, (what, harness) in EXPERIMENTS.items():
        for h in (15, 30):
            tid = f'{exp}-{pair}-{h}m'
            js = REPO / 'results' / 'json' / f'deriv_{exp}_{pair}_{h}m_result.json'
            ev = reg.get(tid, [])
            if js.exists():
                r = json.loads(js.read_text())
                c = str(r['operating_cov'])
                am = {s: amended_verdict(r, c, s, pair) if 'bycov' in r else '' for s in ('UP', 'DOWN')}
                verdict = effective(f'{r["verdict"]["UP"]} / {r["verdict"]["DOWN"]} (cov {r["operating_cov"]})',
                                    '; '.join(f'{s} {am[s]}' for s in ('UP', 'DOWN') if am[s]))
                status = 'done' if 'CERTIFIED' in (r['verdict']['UP'] + r['verdict']['DOWN']) or any(am.values()) else 'killed'
                lines.append(f'| {exp.upper()} | {what} (`{harness}`) | {h}m | {status} | {verdict} | `{js.name}` |')
            elif ev and not any(e['event'] == 'finish' for e in ev):
                lines.append(f'| {exp.upper()} | {what} (`{harness}`) | {h}m | running | — | — |')
            elif ev:
                fin = [e for e in ev if e['event'] == 'finish'][-1]
                lines.append(f'| {exp.upper()} | {what} (`{harness}`) | {h}m | {fin.get("status")} | {fin.get("note", "")[:160]} | — |')
            elif (exp, pair, h) in q:
                lines.append(f'| {exp.upper()} | {what} (`{harness}`) | {h}m | pending | — | — |')
    return '\n'.join(lines)



def notes_block(pair):
    """Short status for the pair, rendered from the result JSONs (no hand copying)."""
    J = REPO / 'results' / 'json'
    out = []
    for fname, label in (('deriv_r1bt_frozen_books_trainclock_nylate_dev_tradeable_result.json',
                          'Incumbent benchmark on TRADEABLE decisions, books as trained (R1bt, 13:00-16:35 + 18:00-22:00 ET)'),
                         ('deriv_r1t_frozen_books_dev_tradeable_result.json',
                          'Books as deployed today (R1t, true clock, NY 08:00-16:35, tradeable)')):
        f = J / fname
        if not f.exists():
            continue
        books = [b for b in json.loads(f.read_text())['books'] if b['pair'] == pair and b['status'] == 'SCORED']
        if not books:
            out.append(f'- **{label}:** no scorable book for this pair.')
            continue
        parts = []
        for side in ('UP', 'DOWN'):
            b = max(books, key=lambda b: (b['sides'][side]['win_rate'] or 0))
            x = b['sides'][side]
            parts.append(f'{side} best `{b["book_id"]}` {x["win_rate"]:.4f} [{x["ci95"][0]:.4f}, {x["ci95"][1]:.4f}] n={x["n"]} '
                         f'({x["verdict"]})')
        out.append(f'- **{label}:** ' + '; '.join(parts) + '.')
    sys.path.insert(0, str(LAB / 'tools'))
    import numpy as np
    cert, repo_only, amended = [], [], []
    for f in sorted(J.glob(f'deriv_*_{pair}_*m_result.json')):
        r = json.loads(f.read_text())
        if 'bycov' not in r or r.get('tag') in ('e1', 'r2') or r.get('tag') in NOT_CANDIDATES:
            continue
        c = str(r['operating_cov'])
        for side in ('UP', 'DOWN'):
            g = goal_verdict(r, c, side)
            fw = r['bycov'][c]['forward'][side]
            item = f'{r["tag"]} {r["horizon_min"]}m {side} (cov {c}: {g}; forward {fw["n_clear_BE"]}/{fw["n_counted"]})'
            if g.startswith('CERTIFIED'):
                cert.append(item)
            elif r['verdict'][side] == 'CERTIFIED':
                repo_only.append(item)
            am = amended_verdict(r, c, side, pair)
            if am:
                _, _, clear, p10 = path_test(r, c, side)
                amended.append((int(r['horizon_min']), f'{r["tag"]} {r["horizon_min"]}m {side} (cov {c}: {clear}/15 clear, '
                                                       f'p10 {p10:.4f}; {am})'))
    out.append('- **DEV-CPCV certified sides, GOAL rule (>= 12 of 15 paths counted and clear, p10 >= .541; tradeable runs):** '
               + ('; '.join(cert) if cert else 'none yet') +
               '. DEV certification is necessary, not sufficient: the sealed holdout decides (docs/DERIV_SOTA_PROGRAM.md).')
    if amended:
        out.append('- **DEV-CPCV certified under the owner\'s drift amendment** (decision 1, 2026-11-15: for a genuinely '
                   'drifting pair the moved up-rate checks - in-session band and per-path tripwire - are replaced by the hour x '
                   'volatility drift placebo of the same configuration; mirage check: '
                   + '; '.join(MIRAGE_OK[(pair, h)] for h in sorted({h for h, _ in amended})) + '): '
                   + '; '.join(x for _, x in amended) + '. Configurations without a placebo run stay INVALID. Final '
                   'certification also needs the holdout (drift-matched excess CI95 lower bound > 0 and the holdout conditions).')
    if repo_only:
        out.append('- Certified only under the repository rule (too few counted paths for the goal rule): ' + '; '.join(repo_only) + '.')
    # Two-sided DEV leaderboard: the three tradeable runs closest to the goal rule per side and horizon (paths clearing
    # .541 among counted, then p10), with validity and the excess over the always-side baseline.
    import numpy as np
    for h in (15, 30):
        for side in ('UP', 'DOWN'):
            cand, n_runs = [], 0
            for f in sorted(J.glob(f'deriv_*_{pair}_{h}m_result.json')):
                r = json.loads(f.read_text())
                if ('bycov' not in r or r.get('tag') in ('e1', 'r2') or r.get('tag') in NOT_CANDIDATES
                        or not r.get('tradeable_only', True)):
                    continue
                n_runs += 1
                c = str(r['operating_cov'])
                w = [p['bycov'][c][side][1] for p in r['paths'] if p['bycov'][c][side][0] >= 25]
                if len(r['paths']) != 15 or not w:
                    continue
                clear = sum(x >= 0.541 for x in w)
                p10 = float(np.percentile(w, 10))
                ex = (r['bycov'][c]['verdict'][side].get('excess_vs_always') or {}).get('mean')
                valid = r.get('validity', {}).get('VALID')
                am = amended_verdict(r, c, side, pair)
                cand.append((clear, p10, f'`{r["tag"]}` {clear}/{len(w)} clear, p10 {p10:.3f}'
                             + (f', excess {ex:+.3f}' if ex is not None else '')
                             + ('' if valid else ', ' + effective('INVALID (up-rate checks)', am))))
            if cand:
                cand.sort(key=lambda x: (x[0], x[1]), reverse=True)
                out.append(f'- **DEV leaderboard {h}m {side}** (closest to the goal rule at each run\'s operating coverage): '
                           + '; '.join(x[2] for x in cand[:3]) + '.')
            elif n_runs:
                out.append(f'- **DEV leaderboard {h}m {side}:** no CPCV path with 25 or more {side} trades at the operating '
                           f'coverage in any candidate run ({n_runs} run{"s" if n_runs != 1 else ""}).')
    # Sealed-holdout looks (owner decision 6), straight from holdout/seq_decision.py's output, and the deployment status
    # that follows: a key-side is deployable only when its finalist is CERTIFIED at a look.
    latest = {}
    for f in sorted(J.glob(f'deriv_*_look[12]decision_{pair}_*m_result.json')):
        d = json.loads(f.read_text())
        dm = d.get('drift_matched') or {}
        cd_ = d['conditions']
        inc = '; '.join(f'`{x["book_id"]}` ({x["reading"]}) {x["wins"]}/{x["n"]} = {x["win_rate"]:.4f} '
                        f'{"beaten" if x["beaten"] else "NOT beaten"}' for x in d['incumbents']) or 'none registered for this key'
        out.append(f'- **Holdout look {d["look"]}** (`{f.name}`; window {d["window"][0]}..{d["window"][1]}; frozen code; '
                   f'owner decision 6): `{d["finalist"]}` {d["key"]} {d["side"]} won {d["wins"]} of {d["n"]} trades '
                   f'(win rate {d["win_rate"]}; z {d["z_score"]}) against a boundary of {d["threshold_wins"]} wins '
                   f'(z {d["threshold_z"]}): {"crossed" if d["crossed"] else "not crossed"}. Conditions: n >= 50 '
                   f'{cd_["n_ge_50"]}; drift-matched excess (NY hour x rv_60m) {dm.get("excess")} CI95 {dm.get("excess_ci95")}, '
                   f'passes {cd_["drift_matched_passes"]}; incumbents: {inc}. Decision: '
                   f'{"CERTIFIED" if d["CERTIFIED_AT_THIS_LOOK"] else d["next"]}.')
        k = (d['key'], d['side'])
        if k not in latest or d['look'] > latest[k]['look']:
            latest[k] = d
    parts = []
    for (key, side), d in sorted(latest.items()):
        if d['CERTIFIED_AT_THIS_LOOK']:
            parts.append(f'{key} {side} DEPLOYABLE (`{d["finalist"]}`, certified at holdout look {d["look"]})')
        elif d['look'] == 1:
            parts.append(f'{key} {side} DO NOT DEPLOY yet (`{d["finalist"]}` did not certify at look 1; look 2, on the '
                         f'window to 2027-02-26, decides)')
        else:
            parts.append(f'{key} {side} DO NOT DEPLOY (`{d["finalist"]}` not certified at look 2, final)')
    out.append('- **Deployment status (Deriv feed):** ' + ('; '.join(parts) + '; every other key and side: ' if parts else '')
               + 'DO NOT DEPLOY - no candidate has passed both the DEV-CPCV goal rule (or the drift amendment) and a '
               'sealed-holdout look, and deployment needs both.')
    out.append('- Superseded: `e1` and `r2` ran before the venue tradeability filter (they include untradeable blackout '
               'decisions); their tradeable re-runs are `e1t` and `r2t`.')
    checks = sorted(t for t in NOT_CANDIDATES if (J / f'deriv_{t}_{pair}_15m_result.json').exists()
                    or (J / f'deriv_{t}_{pair}_30m_result.json').exists())
    if checks:
        out.append('- Robustness checks, not candidates (excluded from the lists and leaderboards above; tables below): '
                   + ', '.join(f'`{t}`' for t in checks) + '.')
    return '\n'.join(out)

def replace_block(text, tag, body):
    b, e = f'<!-- {tag}:BEGIN -->', f'<!-- {tag}:END -->'
    i, j = text.index(b) + len(b), text.index(e)
    return text[:i] + '\n' + body + '\n' + text[j:]


def main():
    pairs = sys.argv[1:] or PAIRS
    reg, q = registry(), queued()
    for pair in pairs:
        res = REPO / 'results' / f'{pair}_RESULTS.md'
        t = res.read_text()
        if '<!-- DERIV-FEED-TABLES:BEGIN -->' not in t:
            t = t.rstrip('\n') + (
                f'\n\n## {pair} — DERIV-FEED PROGRAM (Deriv-only data; opened {OPENED})\n\n'
                '> **SCOPE:** this key on the Deriv feed only — Deriv 1-second mid ticks (`/mnt/data/deriv`), no Dukascopy data. '
                'Labels settle on Deriv ticks (entry first tick >= decision + 1 s, exit last tick <= entry target + H, ties lose); '
                'DEV window 2025-08-11..2026-07-31 for training, selection and CPCV; HOLDOUT 2026-08-01..2026-11-14 sealed '
                '(opened once per finalized candidate). Protocol and tools: `docs/DERIV_SOTA_PROGRAM.md`. Tables below are '
                'rendered from the result JSONs by `scripts/render_ledger.py` (do not edit by hand).\n\n'
                '<!-- DERIV-FEED-NOTES:BEGIN -->\n<!-- DERIV-FEED-NOTES:END -->\n\n'
                '<!-- DERIV-FEED-TABLES:BEGIN -->\n<!-- DERIV-FEED-TABLES:END -->\n')
        body = subprocess.run([sys.executable, '-B', str(LAB / 'tools' / 'render_ledger.py'), pair],
                              check=True, capture_output=True, text=True).stdout.rstrip('\n')
        t = replace_block(t, 'DERIV-FEED-NOTES', notes_block(pair))
        res.write_text(replace_block(t, 'DERIV-FEED-TABLES', body))
        sw = REPO / 'sweeps' / f'{pair}_15m.md'
        s = sw.read_text()
        if '<!-- DERIV-FEED-LEDGER:BEGIN -->' not in s:
            s = s.rstrip('\n') + (
                f'\n\n## DERIV-FEED PROGRAM LEDGER (Deriv-only data; opened {OPENED})\n\n'
                '> Resumable state of the Deriv-feed sweep for this pair (15m and 30m rows). Status table regenerated from the '
                'lab registry (`/mnt/data/deriv-sota-lab/registry/tests.jsonl`), the lab queue and the result JSONs; results of '
                f'record in `results/{pair}_RESULTS.md` (DERIV-FEED PROGRAM). Protocol: `docs/DERIV_SOTA_PROGRAM.md`.\n\n'
                '<!-- DERIV-FEED-LEDGER:BEGIN -->\n<!-- DERIV-FEED-LEDGER:END -->\n')
        sw.write_text(replace_block(s, 'DERIV-FEED-LEDGER', ledger_block(pair, reg, q)))
        print(pair, 'updated')


if __name__ == '__main__':
    main()
