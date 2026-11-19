#!/usr/bin/env python3
"""Variant harness on top of tools/cpcv_deriv.py: same CPCV, scoring, validity rule and falsifier, plus
knobs the base mirror does not expose.

    cpcv_deriv_x.py --pair P --horizon 15 [--session ny] [--train-session all] [--covs 0.03,0.02,0.05]
                    [--nseed 1] [--features all|noclock|own|own_noclock] [--feature-list cols.json]
                    [--num-leaves 255] [--min-child 400] [--lr 0.02] [--n-estimators 800] [--colsample 0.5]
                    [--reg-lambda 20] [--dataset DIR] [--out result.json] [--tag TEXT] [--no-forward] [--max-paths N]
                    [--n-jobs 16] [--path-jobs 1]

--train-session widens (or changes) the rows the model is FIT on; groups, validation rows (early stopping and
the threshold quantile), test paths and every reported number stay on the evaluation --session. Purge and
embargo apply to all rows. --val-mode block replaces the repository's random 15% validation rows (label-
overlapping neighbours of fit rows: overconfident thresholds, early stopping that never fires) with the last
15% of the training rows in time, purged one horizon from the fit rows.
Defaults reproduce cpcv_deriv.py exactly (same rng stream and parameters; checked on E1 USDJPY path 1).
--path-jobs N > 1 runs the CPCV paths and the forward-walk splits in N forked workers (tools/fork_jobs.py), each
with --n-jobs threads: every split starts from the generator state the sequential loop would have, so each path
row is the sequential one; only the summed feature gain can differ in its last bits when a split fits several
boosters (the per-split sums are added in split order instead of fit by fit).
"""
import argparse
import copy
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpcv_deriv as cd  # noqa: E402
from fork_jobs import in_order, run_forked  # noqa: E402

X_VERSION = 'cpcv_deriv_x/0.11'   # 0.11: --path-jobs N (test E-020: paths and forward splits in forked workers;
#   identical path rows) and a recorded 'compute' block; defaults and every 0.10.1 option unchanged.
#   0.10.1: the venue's window (owner decision 14, cpcv_deriv 0.4.2) and the
#   guards of E-003. 0.10: discovery round 5 - --blend-absret SIDE (D5-01 selected blend of the unweighted
#                                  and |return|-weighted LightGBM; --blend-force-alpha for regression only) and --rule-gate
#                                  COL (D5-03 zero-parameter rule offers ranked by the model); defaults and every 0.9 option
#                                  unchanged
#                                  0.9: discovery round 4 - --model nystail --tail-aux L (D4-01 fixed-mass tail auxiliary on
#                                  the kernel features, full-batch L-BFGS; --tail-aux 0 = its matched control); defaults
#                                  and every 0.8 option unchanged
#                                  0.8: discovery round 3 - --signed-only (logit / lgb on e67drfe's signed coordinates; the
#                                  same-input controls), --drift-fe for --model lgb (D3-01 profiled stratum offsets),
#                                  --model nysagree (D3-04 agreement gate); defaults and every 0.7.2 option unchanged
#                                  0.7.2: fit_one reads drift_fe with getattr (holdout_eval's namespace lacks it)
#                                  0.7.1: --monotone catchup (discovery D2-02); defaults unchanged
#                                  0.7: discovery round 1 - --drift-fe (D1-01), --train-label agree3 (D1-02), play
#                                  families hy_/hr_ (D1-03); defaults unchanged
#                                  0.6.1: --multi-horizon (literature rank 24); --boot-gate seeds from str(label) (0.6
#                                  crashed on forward-walk labels); 0.6: literature items --model stack / lgbord4,
#                                  --mag-model har, --boot-gate, --weight recency (defaults unchanged)
#                                  0.5: logit / forest / nystroem honour training weights (mean-1; before 0.5 they ignored
#                                  --weight, so a uniform reflect weight is unchanged); --meta-keep half-pool primaries use
#                                  the selected model (were always LightGBM, no early stopping); 0.4: --families (SWEEP_MATRIX family
#                                  blocks), --train-label resid, --model; 0.3: shared unrounded aggregation (cpcv_deriv 0.4);
#                                  0.2: blackout excluded
# Column prefixes of the SWEEP_MATRIX family blocks (tools/deriv_features_family.py, merged into dataset_dev_v0.3)
FAMILY_PREFIXES = ('tf_', 'sg_', 'kf_', 'rm_', 'fd_', 'cx_', 'tk_', 'pa_',   # SWEEP_MATRIX blocks (dataset_dev_v0.3+)
                   'cm_',                                                    # CCM block (v0.4+)
                   'fx_', 'ck_', 'rl_', 'rs_',                               # literature blocks (v0.5)
                   'hy_', 'hr_',                                             # discovery D1-03 play blocks (v0.6play)
                   'd2_',                                                    # discovery D2-01 shock block (v0.6d2)
                   'qo_', 'qp_')                                             # discovery D3-05 order block + permuted control (v0.6qo)
BLEND_ALPHAS = (0.0, 0.5, 1.0)                  # D5-01: weight of the |return|-weighted model in the selected blend
RULE_USD_BASE = {'USDJPY', 'USDCHF', 'USDCAD'}  # D5-03 --rule-usd-oriented: as tools/cpcv_rule.py (USD strength = + quote)
RULE_SIGNALS = ('fx_ldn_prefix_u60', 'fx_ecb_prefix_u60')   # D5-03: audited causal rule columns (never an outcome column)
RULE_EPS = 1e-9                                 # an encoded offer stays strictly on its side of 0.5
# Extra local-time windows (America/New_York, [start, end) hours) beyond sessions.py. 'nylate' = 13:00-22:00 ET: the
# TRUE hours selected by session_mask('ny') on the Dukascopy-derived stores before 2025-08-22, whose timestamps were
# true UTC - 5 h (see docs/DERIV_SOTA_PROGRAM.md) - i.e. where the incumbents' "NY" certifications actually lived.
LOCAL_WINDOWS = {'nylate': (13, 22), 'nyam': (8, 13), 'nypm': (13, 17), 'nyeve': (17, 22)}


def session(ts, name):
    if name in LOCAL_WINDOWS:
        lo, hi = LOCAL_WINDOWS[name]
        h = pd.to_datetime(np.asarray(ts), unit='s', utc=True).tz_convert('America/New_York').hour.values
        return (h >= lo) & (h < hi)
    return cd.session_mask(ts, name)


def make_model(a, seed):
    return cd.lgb.LGBMClassifier(objective='binary', metric='auc', learning_rate=a.lr, num_leaves=a.num_leaves,
                                 min_child_samples=a.min_child, subsample=0.8, subsample_freq=1, colsample_bytree=a.colsample,
                                 reg_lambda=a.reg_lambda, n_estimators=a.n_estimators, n_jobs=a.n_jobs, verbosity=-1,
                                 random_state=seed, bagging_seed=seed, feature_fraction_seed=seed)


def standardize(Xf, *others):
    """Fit-pool median imputation, z-scoring and clipping to [-5, 5] (statistics from the fit rows only)."""
    med = np.nanmedian(Xf, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    prep = lambda Z: np.where(np.isnan(Z), med, Z).astype(np.float64)
    F = prep(Xf)
    mu, sdv = F.mean(axis=0), F.std(axis=0) + 1e-12
    z = lambda Z: np.clip((Z - mu) / sdv, -5, 5)
    return (z(F),) + tuple(z(prep(O)) for O in others)


def blend_select(pv0, pv1, vg, tv, fv, cov, side, gap, force=None):
    """D5-01: choose alpha in BLEND_ALPHAS for p = (1 - alpha) pv0 + alpha pv1 on VALIDATION rows only. vg = validation rows
    inside the compression gate (all rows without it), tv / fv = their times and settled returns. Each alpha: threshold =
    the coverage quantile of |p - 0.5| over vg; in each chronological half of vg (split at the median time) the first-come
    win rate on `side` (cpcv_deriv.side_wr: ties lose, gap de-overlap); objective = the smaller half (a half without trades
    scores -inf). The largest objective wins; equal objectives go to the smaller alpha. Returns (alpha, diagnostics)."""
    mid = float(np.median(tv[vg])) if vg.any() else 0.0
    sel = {}
    for al in BLEND_ALPHAS:
        pa_ = (1.0 - al) * pv0 + al * pv1
        thr = float(np.quantile(np.abs(pa_ - 0.5)[vg], 1 - cov)) if vg.any() else None
        halves = [cd.side_wr(pa_[vg & hm], fv[vg & hm], tv[vg & hm], thr, side, gap) if thr is not None else (0, float('nan'))
                  for hm in (tv < mid, tv >= mid)]
        fin = [(int(n), float(wr)) if n > 0 and np.isfinite(wr) else (int(n), None) for n, wr in halves]
        worst = min(wr if wr is not None else -np.inf for _, wr in fin)
        sel[f'{al}'] = {'worst_half_wr': worst if np.isfinite(worst) else None, 'halves': [list(h) for h in fin], 'thr': thr}
    obj = lambda al_: (-np.inf if sel[f'{al_}']['worst_half_wr'] is None else sel[f'{al_}']['worst_half_wr'], -al_)
    return (force if force is not None else max(BLEND_ALPHAS, key=obj)), sel


def rule_encode(p, side):
    """D5-03: encode rule offers for the shared scorer. side -1 (DOWN offer): p' = p_UP / 2, side +1 (UP offer): p' = 0.5 +
    p_UP / 2, each kept strictly on its side of 0.5 (RULE_EPS); side 0 (no offer): 0.5. |p' - 0.5| is half the model
    probability of the offered side, so offers rank by it."""
    return np.where(side < 0, np.minimum(0.5 * p, 0.5 - RULE_EPS),
                    np.where(side > 0, np.maximum(0.5 + 0.5 * p, 0.5 + RULE_EPS), 0.5))


def sigm(f):
    return 1.0 / (1.0 + np.exp(-f))


def prob(f, link=sigm):
    """Test E-003: link(f) where the raw score f is finite, NaN elsewhere. A sigmoid gives 1.0 at +inf and 0.0 at
    -inf, which would make an invalid score a valid probability."""
    return np.where(np.isfinite(f), link(f), np.nan)


def tail_soft_loss(s, y, m, tau):
    """Discovery D4-01 (0.9): the fixed-mass soft loss rate of the two-sided confidence tail and its gradient in s.
    s, y = scores and 0/1 up labels of the eligible fit rows C; m = c |C|. Membership a_i = sigmoid((|s_i| - q) / tau),
    q solved by bisection so that sum a_i = m; A = sum a_i L_i / m with L_i = 1 when the side of s_i loses (s_i > 0 and
    y_i = 0, s_i < 0 and y_i = 1, s_i = 0 counts as a loss). L is piecewise constant (it changes only at s = 0), so
    dA/ds_k = sign(s_k) sigma'_k (L_k - Lbar) / (tau m), Lbar the sigma'-weighted mean of L (implicit dq/ds).
    Returns (A, dA/ds, q, sum a). The bisection runs to the double-precision limit of q (review of 0.9: a width rule
    alone does not bound the mass error); callers record the final mass error."""
    from scipy.special import expit
    u = np.abs(s)
    lo, hi = float(u.min()) - 50.0 * tau, float(u.max()) + 50.0 * tau
    for _ in range(400):
        q = 0.5 * (lo + hi)
        if not lo < q < hi:                  # lo and hi are adjacent doubles: q cannot be resolved further
            break
        if expit((u - q) / tau).sum() > m:
            lo = q
        else:
            hi = q
    q = 0.5 * (lo + hi)
    av = expit((u - q) / tau)
    L = np.where(s > 0, 1.0 - y, np.where(s < 0, y, 1.0))
    sp_ = av * (1.0 - av)
    lbar = float((sp_ * L).sum() / max(float(sp_.sum()), 1e-300))
    return float((av * L).sum() / m), sp_ * (L - lbar) * np.sign(s) / (tau * m), q, float(av.sum())


def nystail_objective(w, P, y, inC, mC, lam, tau, alpha=1e-4):
    """Discovery D4-01 (0.9) objective and gradient in w = (beta, b): mean BCE of s = P beta + b over all rows +
    (alpha / 2) |beta|^2 + lam x tail_soft_loss over the rows inC at mass mC (lam = 0: no auxiliary, tau unused)."""
    from scipy.special import expit
    s = P @ w[:-1] + w[-1]
    gs = (expit(s) - y) / len(y)
    f = float(np.mean(np.logaddexp(0.0, s) - y * s)) + 0.5 * alpha * float(w[:-1] @ w[:-1])
    if lam > 0:
        A, dA, _, _ = tail_soft_loss(s[inC], y[inC], mC, tau)
        f += lam * A
        gs[inC] += lam * dA
    g = np.empty_like(w)
    g[:-1] = P.T @ gs + alpha * w[:-1]
    g[-1] = gs.sum()
    return f, g


ACI_REFUSED = ('ACI is refused: it moves its threshold with settled test outcomes, test E-003 stops a harness '
               'whose thresholds use test outcomes, and owner decision 13 makes adaptive policies forward '
               'diagnostics only, under a protocol of their own (none is registered)')


LGB_MODELS = ('lgb', 'lgbbrier', 'lgbord4', 'ens3', 'pool', 'stack')        # models that fit LightGBM
NYSTROEM_MODELS = ('nystroem', 'pool', 'stack', 'nysagree', 'nystail')     # models that fit a Nystroem kernel
LOGIT_MODELS = ('logit', 'pool', 'stack', 'nysagree')                      # models whose logistic takes --logit-c


INT32_MAX = 2 ** 31 - 1


def null_shift(weeks, span):
    """Test E-003: the null replay's label shift in seconds, reduced modulo span in Python integers ((x + k) % span
    = (x + k % span) % span), so any integer number of weeks works on int64 timestamps."""
    return (int(weeks) * 7 * 86400) % span


def f32_rate(lr):
    """A learning rate as XGBoost and CatBoost hold it (float32; out of range: infinity)."""
    with np.errstate(over='ignore'):
        return np.float32(lr)


def option_domains(a):
    """Test E-003: (option, value, inside its domain) for every option the run's estimators use, at the domain
    the installed library accepts (probed; see patch_e003_round6.py, patch_e003_round7.py and patch_e003_round9.py)
    or the harness's own definition; an option the run does not use is not checked. NaN fails every comparison."""
    m = a.model
    lgb_used = m in LGB_MODELS or (a.mag_keep is not None and a.mag_model == 'lgb')
    nys_used = m in NYSTROEM_MODELS or (a.drift_fe and m == 'nystroem')
    logit_used = m in LOGIT_MODELS or (m == 'logit') or ((a.drift_fe or a.signed_only) and m == 'logit')
    out = []
    if lgb_used:                                    # (CatBoost in ens3 takes the iterations too)
        out += [('--num-leaves', a.num_leaves, 2 <= a.num_leaves <= 131072),
                ('--min-child', a.min_child, 0 <= a.min_child <= INT32_MAX), ('--lr', a.lr, a.lr > 0),
                ('--colsample', a.colsample, 0 < a.colsample <= 1), ('--reg-lambda', a.reg_lambda, a.reg_lambda >= 0),
                ('--n-estimators', a.n_estimators, 1 <= a.n_estimators <= INT32_MAX)]
    if m == 'ens3':                                 # XGBoost and CatBoost too (both hold the rate in float32)
        out += [('--lr', a.lr, bool(np.isfinite(f32_rate(a.lr)) and f32_rate(a.lr) > np.finfo(np.float32).eps)),
                ('--cat-depth', a.cat_depth, 0 <= a.cat_depth <= 16),
                ('--n-jobs', a.n_jobs, a.n_jobs == -1 or 1 <= a.n_jobs <= INT32_MAX)]   # CatBoost's convention
    if m in ('forest', 'pool', 'stack'):            # extra trees: joblib's pre-dispatch '2 * n_jobs' is at most 10^6
        out += [('--n-jobs', a.n_jobs, a.n_jobs != 0 and a.n_jobs <= 500000)]
    if nys_used:
        out += [('--nys-gamma-mult', a.nys_gamma_mult, 0 <= a.nys_gamma_mult < float('inf')),
                ('--nys-components', a.nys_components, a.nys_components >= 1)]
    if logit_used:
        out += [('--logit-c', a.logit_c, a.logit_c > 0)]
    if m == 'stack':
        out += [('--stack-shrink', a.stack_shrink, 0 <= a.stack_shrink <= 1)]
    if a.boot_gate:
        out += [('--boot-z', a.boot_z, a.boot_z >= 0)]
    if a.weight == 'recency':
        out += [('--halflife-days', a.halflife_days, a.halflife_days > 0)]
    return out


class XFitter(cd.Fitter):
    def __init__(self, X, fwd, ts, a, gap, covs):
        if getattr(a, 'aci_gamma', None) is not None:     # test E-003: no path of this module scores ACI
            raise ValueError(ACI_REFUSED)
        super().__init__(X, fwd, ts, a, gap, covs)

    def signed(self, Z):
        """e67drfe's signed coordinates (reflection map: negated columns, rangepos - 0.5), missing = 0."""
        V = Z[:, self.signed_cols].astype(np.float64)
        V[:, self.signed_rng] -= 0.5
        return np.nan_to_num(V, nan=0.0)

    def add_gain(self, imp):
        """Accumulate a booster's gain; a booster fit on the signed coordinates maps back onto their columns."""
        if len(imp) == len(self.gain):
            self.gain += imp
        else:
            self.gain[self.signed_cols] += imp

    def signed_logit(self, Xf, yf, Xv, Xt, skw, fe):
        """L2 logistic on the signed coordinates, scaled by their fit-pool sd without centring and clipped to [-5, 5].
        fe=True (e67drfe): one-hot strata x10 and no intercept, scores WITHOUT the strata; fe=False (the same-input
        plain twin): a global intercept, scores with it. Returns raw validation and test scores and iterations."""
        import scipy.sparse as sp
        from sklearn.linear_model import LogisticRegression
        a = self.a
        F = self.signed(Xf)
        sc = F.std(axis=0) + 1e-12
        z = lambda V: np.clip(V / sc, -5, 5)
        if fe:
            _, inv = np.unique(self.strata_f, return_inverse=True)
            D = sp.csr_matrix((np.full(len(inv), 10.0), (np.arange(len(inv)), inv)))
            m = LogisticRegression(C=a.logit_c, max_iter=3000, fit_intercept=False)
            m.fit(sp.hstack([sp.csr_matrix(z(F)), D]).tocsr(), yf, **skw)
            w = m.coef_[0, :F.shape[1]]
            return z(self.signed(Xv)) @ w, z(self.signed(Xt)) @ w, int(np.max(m.n_iter_))
        m = LogisticRegression(C=a.logit_c, max_iter=3000)
        m.fit(z(F), yf, **skw)
        w, b = m.coef_[0], float(m.intercept_[0])
        return z(self.signed(Xv)) @ w + b, z(self.signed(Xt)) @ w + b, int(np.max(m.n_iter_))

    @staticmethod
    def profile_alpha(inv, y, off, w, alpha=None):
        """Per-stratum ridge offsets: alpha_s = argmin sum_{i in s} w_i BCE(y_i, sigmoid(off_i + alpha_s)) + 0.5 alpha_s^2,
        by safeguarded Newton steps (curvature >= 1, steps clipped to [-2, 2])."""
        S = int(inv.max()) + 1
        al = np.zeros(S) if alpha is None else alpha.astype(np.float64).copy()
        for _ in range(100):
            p = sigm(off + al[inv])
            g = np.bincount(inv, w * (p - y), S) + al
            h = np.bincount(inv, w * p * (1 - p), S) + 1.0
            step = np.clip(g / h, -2.0, 2.0)
            al -= step
            if np.max(np.abs(step)) < 1e-10:
                break
        return al

    def fit_unsupported(self, y, yv):  # noqa: C901 - one rule per model class
        """Test E-003: why the selected model class cannot be fitted on the fit labels y with the validation labels
        yv, or None - each estimator's own requirement. Every model that fits LightGBM needs two fit rows. nystail
        has no class rule (its optimizer takes one class; its own checks stay). lgbord4 refuses a validation bin y
        lacks (LightGBM's early stopping raises) and fit labels whose up bins (2, 3) are not exactly the classes at
        sorted positions 2 and above (the P(up) columns [:, 2:] would then hold a down bin or miss an up bin);
        every other layout, a singleton down bin included, keeps the incumbent result. lgb and lgbbrier
        take one class, but LightGBM's early stopping refuses a validation label y lacks (--no-early-stop lgb has
        no validation set). The other model classes need two classes in y: scikit-learn's logistic and SGD raise,
        extra trees returns one probability column, CatBoost (in ens3) raises (XGBoost fits all-zero labels but
        not all-one). One-class validation labels are fine: every early-stopping metric here and the AUC
        diagnostic take them. A model that fits on the signed coordinates alone (signed-only logistic and LightGBM,
        the drift-neutral LightGBM, the agreement gate's plain logistic) needs at least one signed column; the
        drift-neutral logistic (the agreement gate's other one) fits its strata alone, and the drift-neutral kernel
        model reads every column."""
        a = self.a
        cols = getattr(self, 'signed_cols', None)
        if cols is not None and len(cols) == 0 and (getattr(a, 'signed_only', False) or a.model == 'nysagree'
                                                    or (getattr(a, 'drift_fe', False) and a.model == 'lgb')):
            return 'no signed feature column (this model class fits on the signed coordinates)'
        if a.model == 'nystail':
            return None
        if a.model in ('lgb', 'lgbbrier', 'lgbord4', 'ens3', 'pool', 'stack') and len(y) < 2:
            return 'fewer than two fit rows (LightGBM needs two)'
        if a.model == 'lgbord4':
            c = sorted(set(np.unique(y).tolist()))      # LightGBM encodes them 0..k-1 in this order
            if [b for b in c if b >= 2] != c[2:]:
                return 'the ordinal fit labels make P(up) = [:, 2:] read the wrong bins'
            if not np.isin(yv, y).all():
                return 'a validation bin is absent from the ordinal fit labels (LightGBM early stopping)'
            return None
        if a.model in ('lgb', 'lgbbrier'):
            no_es = a.model == 'lgb' and getattr(a, 'no_early_stop', False) and not getattr(a, 'drift_fe', False)
            if no_es or np.isin(yv, y).all():
                return None
            return 'a validation label is absent from the fit labels (LightGBM early stopping)'
        return None if len(np.unique(y)) >= 2 else 'the fit labels hold one class'

    def fit_one(self, sd, Xf, yf, Xv, yv, Xt):
        """One seed of the selected model class: (validation probabilities, test probabilities, iterations)."""
        a = self.a
        why = self.fit_unsupported(yf, yv)
        if why:                                     # test E-003: the selected model class cannot fit these labels
            raise cd.Unsupported(f'insufficient support: {why}')
        pv, pt, its = [], [], []
        # sklearn models get the training weights rescaled to mean 1 (per-row loss scale unchanged against the penalty)
        skw = {} if self.wfit is None else {'sample_weight': self.wfit / (self.wfit.mean() + 1e-18)}
        if getattr(a, 'drift_fe', False):          # (getattr: holdout_eval passes a reduced namespace)
            return self.fit_drift_fe(sd, Xf, yf, Xv, yv, Xt, skw)
        if a.model == 'nysagree':
            return self.fit_nysagree(sd, Xf, yf, Xv, yv, Xt, skw)
        if a.model == 'nystail':
            return self.fit_nystail(sd, Xf, yf, Xv, Xt)
        if getattr(a, 'signed_only', False):
            if a.model == 'logit':                  # e74logitsig: e67drfe's inputs and scaling, global intercept, no strata
                fv, ft, it = self.signed_logit(Xf, yf, Xv, Xt, skw, fe=False)
                return prob(fv), prob(ft), it         # test E-003: an infinite raw score is invalid
            Xf, Xv, Xt = self.signed(Xf), self.signed(Xv), self.signed(Xt)    # lgb: the usual booster on these columns
        if a.model == 'lgbord4':                    # literature rank 25: ordinal signed bins (4-class), P(up) = P(bin 2) + P(bin 3)
            m = cd.lgb.LGBMClassifier(objective='multiclass', num_class=4, learning_rate=a.lr, num_leaves=a.num_leaves,
                                      min_child_samples=a.min_child, subsample=0.8, subsample_freq=1, colsample_bytree=a.colsample,
                                      reg_lambda=a.reg_lambda, n_estimators=a.n_estimators, n_jobs=a.n_jobs, verbosity=-1,
                                      random_state=sd, bagging_seed=sd, feature_fraction_seed=sd)
            kw = {} if self.wfit is None else {'sample_weight': self.wfit}
            m.fit(Xf, yf, eval_set=[(Xv, yv)], eval_metric='multi_logloss',
                  callbacks=[cd.lgb.early_stopping(80, verbose=False), cd.lgb.log_evaluation(0)], **kw)
            pv.append(m.predict_proba(Xv)[:, 2:].sum(axis=1)); pt.append(m.predict_proba(Xt)[:, 2:].sum(axis=1))
            self.gain += m.booster_.feature_importance(importance_type='gain')
            its.append(int(m.best_iteration_ or 0))
        if a.model in ('lgb', 'ens3', 'pool', 'stack'):
            m = make_model(a, sd)
            if getattr(self, 'mono', None) is not None:     # discovery D2-02 monotone constraints (--monotone)
                m.set_params(monotone_constraints=self.mono)
            kw = {} if self.wfit is None else {'sample_weight': self.wfit}
            if a.no_early_stop:                     # validation rows only set the threshold
                m.fit(Xf, yf, **kw)
            else:
                m.fit(Xf, yf, eval_set=[(Xv, yv)], eval_metric='auc',
                      callbacks=[cd.lgb.early_stopping(80, verbose=False), cd.lgb.log_evaluation(0)], **kw)
            pv.append(m.predict_proba(Xv)[:, 1]); pt.append(m.predict_proba(Xt)[:, 1])
            self.add_gain(m.booster_.feature_importance(importance_type='gain'))
            its.append(int(m.best_iteration_ or 0))
        if a.model == 'lgbbrier':                   # literature rank 4: LightGBM fit on the Brier score (custom objective)
            def brier(y, f):
                p = 1.0 / (1.0 + np.exp(-f))
                g = 2.0 * (p - y) * p * (1 - p)
                h = 2.0 * p * (1 - p) * (p * (1 - p) + (p - y) * (1 - 2 * p))
                return g, np.maximum(h, 1e-6)
            m = cd.lgb.LGBMClassifier(objective=brier, learning_rate=a.lr, num_leaves=a.num_leaves, min_child_samples=a.min_child,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=a.colsample, reg_lambda=a.reg_lambda,
                                      n_estimators=a.n_estimators, n_jobs=a.n_jobs, verbosity=-1, random_state=sd,
                                      bagging_seed=sd, feature_fraction_seed=sd)
            kw = {} if self.wfit is None else {'sample_weight': self.wfit}
            try:
                m.fit(Xf, yf, eval_set=[(Xv, yv)], eval_metric='auc',
                      callbacks=[cd.lgb.early_stopping(80, verbose=False), cd.lgb.log_evaluation(0)], **kw)
            except cd.lgb.basic.LightGBMError as e:       # test E-003: the custom objective with no usable feature
                if 'num_features()' not in str(e):
                    raise
                raise cd.Unsupported('insufficient support: no feature of the fit rows can be split (LightGBM\'s '
                                     'feature pre-filter; the Brier objective needs one)') from None
            # test E-003: prob - an infinite raw score is invalid, not a probability of 0 or 1
            sig = lambda Z: prob(m.booster_.predict(Z, raw_score=True, num_iteration=m.best_iteration_ or None))
            pv.append(sig(Xv)); pt.append(sig(Xt))
            self.gain += m.booster_.feature_importance(importance_type='gain')
            its.append(int(m.best_iteration_ or 0))
        if a.model == 'ens3':                       # SWEEP_MATRIX A1: + XGBoost + CatBoost (scripts/m15_production.py recipe)
            import xgboost as xgb
            from catboost import CatBoostClassifier, CatBoostError
            es = None if a.no_early_stop else 80
            g = xgb.XGBClassifier(n_estimators=a.n_estimators, learning_rate=a.lr, max_depth=8, subsample=0.8,
                                  colsample_bytree=0.5, reg_lambda=10, tree_method='hist', n_jobs=a.n_jobs, eval_metric='auc',
                                  early_stopping_rounds=es, random_state=sd)
            g.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
            pv.append(g.predict_proba(Xv)[:, 1]); pt.append(g.predict_proba(Xt)[:, 1])
            its.append(int(getattr(g, 'best_iteration', a.n_estimators) or 0))
            fill = lambda Z: np.where(np.isnan(Z), -999.0, Z)
            c = CatBoostClassifier(iterations=a.n_estimators, learning_rate=a.lr, depth=a.cat_depth, l2_leaf_reg=10,
                                   eval_metric='AUC', thread_count=a.n_jobs, verbose=False, random_seed=sd,
                                   allow_writing_files=False, early_stopping_rounds=es)
            try:
                c.fit(fill(Xf), yf, eval_set=(fill(Xv), yv))
            except CatBoostError as e:                  # test E-003: no feature of the fit rows varies
                if 'All features are either constant or ignored' not in str(e):
                    raise
                raise cd.Unsupported('insufficient support: no feature of the fit rows varies (CatBoost, in ens3, '
                                     'refuses constant or missing features)') from None
            pv.append(c.predict_proba(fill(Xv))[:, 1]); pt.append(c.predict_proba(fill(Xt))[:, 1])
            its.append(int(c.get_best_iteration() or 0))
        if a.model in ('logit', 'pool', 'stack'):   # low-capacity linear model (L2, fixed C chosen before any run)
            from sklearn.linear_model import LogisticRegression
            F, V, T = standardize(Xf, Xv, Xt)
            m = LogisticRegression(C=a.logit_c, max_iter=3000)
            m.fit(F, yf, **skw)
            pv.append(m.predict_proba(V)[:, 1]); pt.append(m.predict_proba(T)[:, 1])
            its.append(int(np.max(m.n_iter_)))
        if a.model in ('forest', 'pool', 'stack'):  # shallow bagged extremely-randomized trees (literature: weak-signal baseline)
            from sklearn.ensemble import ExtraTreesClassifier
            F, V, T = standardize(Xf, Xv, Xt)
            m = ExtraTreesClassifier(n_estimators=300, max_depth=8, min_samples_leaf=200, max_features=0.3,
                                     bootstrap=True, max_samples=0.5, n_jobs=a.n_jobs, random_state=sd)
            m.fit(F, yf, **skw)
            pv.append(m.predict_proba(V)[:, 1]); pt.append(m.predict_proba(T)[:, 1])
            its.append(300)
        if a.model in ('nystroem', 'pool', 'stack'):  # SWEEP_MATRIX B6: RBF kernel (Nystroem features) + logistic SGD
            from sklearn.kernel_approximation import Nystroem
            from sklearn.linear_model import SGDClassifier
            F, V, T = standardize(Xf, Xv, Xt)
            ny = Nystroem(kernel='rbf', gamma=a.nys_gamma_mult / F.shape[1], n_components=a.nys_components, random_state=sd)
            Fk = ny.fit_transform(F)
            m = SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=30, tol=1e-4, random_state=sd)
            m.fit(Fk, yf, **skw)
            pv.append(m.predict_proba(ny.transform(V))[:, 1]); pt.append(m.predict_proba(ny.transform(T))[:, 1])
            its.append(int(m.n_iter_))
        if a.model == 'stack':
            # literature rank 6, SHRUNK SIMPLEX STACK of lgb, logit, forest, nystroem: simplex weights minimise the validation
            # log-loss (softmax parameters), then shrink toward equal weights by stack_shrink (1 = the equal-weight pool).
            # The validation rows also set early stopping and the threshold: 3 free weights, a declared double use.
            from scipy.optimize import minimize
            if not np.all(np.isfinite(np.column_stack(pv))):   # test E-003: before the clip and the weights
                raise cd.Unsupported('insufficient support: a validation score of a stacked member is not a finite '
                                     'number')
            P = np.clip(np.column_stack(pv), 1e-6, 1 - 1e-6)
            def nll(th):
                w = np.exp(th - th.max()); w /= w.sum()
                q = np.clip(P @ w, 1e-6, 1 - 1e-6)
                return -float(np.mean(yv * np.log(q) + (1 - yv) * np.log(1 - q)))
            th = minimize(nll, np.zeros(P.shape[1]), method='L-BFGS-B').x
            w = np.exp(th - th.max()); w /= w.sum()
            w = (1 - a.stack_shrink) * w + a.stack_shrink / len(w)
            if not hasattr(self, 'stack_w'):
                self.stack_w = []
            self.stack_w.append(w.tolist())
            return P @ w, np.column_stack(pt) @ w, its
        return np.mean(pv, axis=0), np.mean(pt, axis=0), (its[0] if len(its) == 1 else its)

    def fit_drift_fe(self, sd, Xf, yf, Xv, yv, Xt, skw):
        """Discovery D1-01, DRIFT-NEUTRAL fit: one-hot strata (ISO week x NY 3-hour bucket x fit-pool rv_60m tercile,
        columns x10 so the L2 penalty barely touches them, no global intercept) absorb every stratum's direction rate
        during the fit; validation and test scores use the non-stratum coefficients only. logit = the signed
        coordinates (reflection map: negated columns, rangepos - 0.5), scaled without centring, missing = 0;
        nystroem = the usual kernel features (declared approximation: the kernel keeps even functions).
        lgb (0.8, discovery D3-01) = LightGBM on the signed coordinates with the strata as profiled offsets: alpha
        (BCE + 0.5 alpha^2 per stratum) enters as init_score; alpha0 at f = 0, booster 1, alpha1 given booster 1's f,
        booster 2 fit afresh; validation (early stopping) and scores use f alone, centred on the fit rows."""
        import scipy.sparse as sp
        a = self.a
        _, inv = np.unique(self.strata_f, return_inverse=True)
        D = sp.csr_matrix((np.full(len(inv), 10.0), (np.arange(len(inv)), inv)))
        if a.model == 'logit':
            fv, ft, it = self.signed_logit(Xf, yf, Xv, Xt, skw, fe=True)
            return prob(fv), prob(ft), it             # test E-003: an infinite raw score is invalid
        if a.model == 'lgb':
            F, V, T = self.signed(Xf), self.signed(Xv), self.signed(Xt)
            yb = np.asarray(yf, dtype=np.float64)
            w = np.ones(len(yb)) if self.wfit is None else self.wfit / (self.wfit.mean() + 1e-18)
            kw = {} if self.wfit is None else {'sample_weight': self.wfit}
            alpha = self.profile_alpha(inv, yb, np.zeros(len(yb)), w)
            info = {'strata': int(len(alpha)), 'alpha0_sd': float(alpha.std())}
            for k in range(2):
                m = make_model(a, sd)
                m.fit(F, yf, init_score=alpha[inv], eval_set=[(V, yv)], eval_init_score=[np.zeros(len(V))],
                      eval_metric='auc', callbacks=[cd.lgb.early_stopping(80, verbose=False), cd.lgb.log_evaluation(0)], **kw)
                it = int(m.best_iteration_ or 0)
                raw = lambda Z, m=m, it=it: m.booster_.predict(Z, raw_score=True, num_iteration=it or None)
                fF = raw(F)                             # (booster 2's: the centre below, as before)
                if not np.all(np.isfinite(fF)):         # test E-003: they set the next offsets, or the centre
                    raise cd.Unsupported('insufficient support: a fit-row score of the drift-neutral booster is '
                                         'not a finite number (it sets the next offsets or the centre)')
                if k == 0:
                    alpha = self.profile_alpha(inv, yb, fF, w, alpha)
                    info['alpha1_sd'], info['iter1'] = float(alpha.std()), it
            c = float(np.mean(fF))
            info.update(iter2=it, f_centre=c)
            self.lgbfe_info = info
            self.add_gain(m.booster_.feature_importance(importance_type='gain'))
            return prob(raw(V) - c), prob(raw(T) - c), it     # test E-003: an infinite raw score is invalid
        from sklearn.kernel_approximation import Nystroem
        from sklearn.linear_model import SGDClassifier
        F, V, T = standardize(Xf, Xv, Xt)
        ny = Nystroem(kernel='rbf', gamma=a.nys_gamma_mult / F.shape[1], n_components=a.nys_components, random_state=sd)
        Fk = ny.fit_transform(F)
        m = SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=30, tol=1e-4, random_state=sd, fit_intercept=False)
        m.fit(sp.hstack([sp.csr_matrix(Fk), D]).tocsr(), yf, **skw)
        w = m.coef_[0, :Fk.shape[1]]
        return prob(ny.transform(V) @ w), prob(ny.transform(T) @ w), int(m.n_iter_)   # (test E-003: prob)

    def fit_nysagree(self, sd, Xf, yf, Xv, yv, Xt, skw):
        """Discovery D3-04 AGREEMENT GATE (0.8): kernel proposals (the nystroem recipe with the run's training weights)
        are kept only where the drift-neutral logistic (e67drfe, unweighted) takes the kernel's side and its same-input
        plain twin (global intercept, unweighted) takes the other side; every other row gets probability 0.5, so the
        coverage quantile is over ALL validation rows (overall coverage). Exact zero scores abstain. Test E-003: a row
        whose three scores are not all finite is invalid (NaN), whatever the gate."""
        from sklearn.kernel_approximation import Nystroem
        from sklearn.linear_model import SGDClassifier
        a = self.a
        F, V, T = standardize(Xf, Xv, Xt)
        ny = Nystroem(kernel='rbf', gamma=a.nys_gamma_mult / F.shape[1], n_components=a.nys_components, random_state=sd)
        Fk = ny.fit_transform(F)
        m = SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=30, tol=1e-4, random_state=sd)
        m.fit(Fk, yf, **skw)
        kv, kt = m.predict_proba(ny.transform(V))[:, 1], m.predict_proba(ny.transform(T))[:, 1]
        nv, nt, _ = self.signed_logit(Xf, yf, Xv, Xt, {}, fe=True)
        rv, rt, _ = self.signed_logit(Xf, yf, Xv, Xt, {}, fe=False)

        def gate(k, n, r):
            d = np.sign(k - 0.5)
            return (d != 0) & (np.sign(n) == d) & (np.sign(r) == -d)

        def share(b, ok):                           # test E-003: the diagnostics read valid rows only
            return float(b[ok].mean()) if ok.any() else float('nan')
        gv, gt = gate(kv, nv, rv), gate(kt, nt, rt)
        # test E-003: the gate reads a row's three scores, and a sign comparison takes an infinite score as a side, so a
        # row is valid only where all three are finite; an invalid row's probability is NaN
        okv = np.isfinite(kv) & np.isfinite(nv) & np.isfinite(rv)
        okt = np.isfinite(kt) & np.isfinite(nt) & np.isfinite(rt)
        self.agree_info = {'val_share': share(gv, okv), 'test_share': share(gt, okt),
                           'kernel_vs_neutral_agree_val': share(np.sign(kv - 0.5) == np.sign(nv), okv),
                           'neutral_vs_plain_agree_val': share(np.sign(nv) == np.sign(rv), okv)}
        self.agree_keep = (gv, gt)                     # test E-003: the gate's kept rows, carried to the scorer
        return (np.where(okv, np.where(gv, kv, 0.5), np.nan), np.where(okt, np.where(gt, kt, 0.5), np.nan),
                int(m.n_iter_))

    def fit_nystail(self, sd, Xf, yf, Xv, Xt):
        """Discovery D4-01 FIXED-MASS TAIL AUXILIARY (0.9). The nystroem features (fit-only standardization, RBF Nystroem
        gamma nys_gamma_mult / n_features, nys_components, random_state = seed) with a linear score s = phi . beta + b
        fitted by full-batch L-BFGS on the unweighted mean BCE over the fit rows + (1e-4 / 2) |beta|^2 (the SGD recipe's
        alpha; intercept unpenalized) + tail_aux x A(s), A = tail_soft_loss over C = the fit rows inside the compression
        gate (all fit rows without --comp-gate) at mass m = c |C|, c = the operating coverage. Stage 0 (tail_aux = 0, the
        matched control) starts at beta = 0, b = logit(mean y); tau = 0.1 x SD of the stage-0 scores on C, frozen; stage 1
        (tail_aux > 0) starts from the stage-0 solution. At most 500 L-BFGS iterations per stage."""
        from scipy.optimize import minimize
        from scipy.special import expit
        from sklearn.kernel_approximation import Nystroem
        a = self.a
        F, V, T = standardize(Xf, Xv, Xt)
        ny = Nystroem(kernel='rbf', gamma=a.nys_gamma_mult / F.shape[1], n_components=a.nys_components, random_state=sd)
        P = ny.fit_transform(F)
        y = np.asarray(yf, dtype=np.float64)
        n = len(y)
        if a.comp_gate is not None:
            j = self.feats.index('rv_60m')
            inC = Xf[:, j] <= float(np.nanquantile(Xf[:, j], a.comp_gate))     # as split2's gate on these fit rows
        else:
            inC = np.ones(n, bool)
        if not inC.any():                           # test E-003: no fit row inside the compression gate
            raise cd.Unsupported('insufficient support: no fit row inside the compression gate (nystail)')
        c = float(str(a.covs).split(',')[0])
        yC, mC = y[inC], c * float(inC.sum())
        lam = float(a.tail_aux)
        opts = {'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-13}
        obj = lambda w, lam_, tau_: nystail_objective(w, P, y, inC, mC, lam_, tau_)

        def hard_tail(sC):
            top = np.argsort(-np.abs(sC), kind='stable')[:max(1, int(round(mC)))]
            return float(np.where(sC[top] > 0, 1.0 - yC[top], np.where(sC[top] < 0, yC[top], 1.0)).mean())

        def checked(r, f_start, stage):
            # review of 0.9: a numerical failure (non-finite parameters, objective or gradient, or an objective above its
            # start) stops the run; an L-BFGS stop at the iteration budget (status 1) or an abnormal line-search stop with
            # finite values (status 2) keeps the last accepted iterate, and the status is recorded per stage
            if not (np.all(np.isfinite(r.x)) and np.isfinite(r.fun) and np.all(np.isfinite(r.jac)) and r.fun <= f_start):
                raise SystemExit(f'nystail stage {stage}: numerical failure (status {r.status}, {r.message}; '
                                 f'f {r.fun} vs start {f_start})')
            return r

        w0 = np.zeros(P.shape[1] + 1)
        w0[-1] = float(np.log(np.clip(y.mean(), 1e-6, 1 - 1e-6) / np.clip(1 - y.mean(), 1e-6, 1)))
        r0 = checked(minimize(obj, w0, args=(0.0, 1.0), jac=True, method='L-BFGS-B', options=opts), obj(w0, 0.0, 1.0)[0], 0)
        s0 = P @ r0.x[:-1] + r0.x[-1]
        tau = 0.1 * float(np.std(s0[inC]))
        if not np.isfinite(tau) or tau <= 1e-12:     # test E-003: the gated fit rows cannot set a tail scale
            raise cd.Unsupported(f'insufficient support: degenerate tail scale tau = {tau} (nystail: stage-0 '
                                 'scores constant on the gated fit rows)')
        A0 = tail_soft_loss(s0[inC], yC, mC, tau)
        info = {'tail_aux': lam, 'c': c, 'n_fit': int(n), 'n_C': int(inC.sum()), 'tau': tau, 'msg0': str(r0.message)[:60],
                'status0': int(r0.status), 'iters0': int(r0.nit), 'converged0': bool(r0.success),
                'soft_tail_loss0': A0[0], 'hard_tail_loss0': hard_tail(s0[inC]),
                'bce0': float(np.mean(np.logaddexp(0.0, s0) - y * s0)), 'beta_norm0': float(np.linalg.norm(r0.x[:-1]))}
        w, its = r0.x, int(r0.nit)
        if lam > 0:
            r1 = checked(minimize(obj, r0.x, args=(lam, tau), jac=True, method='L-BFGS-B', options=opts),
                         obj(r0.x, lam, tau)[0], 1)
            w, its = r1.x, its + int(r1.nit)
            s1 = P @ w[:-1] + w[-1]
            A1 = tail_soft_loss(s1[inC], yC, mC, tau)
            info.update(iters1=int(r1.nit), status1=int(r1.status), converged1=bool(r1.success), msg1=str(r1.message)[:60],
                        soft_tail_loss1=A1[0],
                        hard_tail_loss1=hard_tail(s1[inC]), mass_err1=abs(A1[3] - mC),
                        bce1=float(np.mean(np.logaddexp(0.0, s1) - y * s1)), beta_norm1=float(np.linalg.norm(w[:-1])))
        else:
            info['mass_err0'] = abs(A0[3] - mC)
        self.tail_info = info
        # test E-003: prob - an infinite raw score is invalid
        return prob(ny.transform(V) @ w[:-1] + w[-1], expit), prob(ny.transform(T) @ w[:-1] + w[-1], expit), its

    def split2(self, tr_idx, val_pool, test_mask, rng, label):
        """One CPCV path or forward split (_split2). Test E-003: when its local support is missing (cd.Unsupported:
        an empty partition, a model subset without fit or validation rows, an empty out-of-fold half, no blend
        selection, no gated fit row) the split gets the explicit zero-trade row with the reason, its feature gain is
        rolled back, and its prediction dump (if any) has no prediction and no kept row. An enabled option's diagnostic
        block (for example meta_gate) is absent from such a row: the reason is in insufficient_support."""
        gain0 = self.gain.copy()
        self._partition = (0, 0)
        try:
            return self._split2(tr_idx, val_pool, test_mask, rng, label)
        except cd.Unsupported as e:
            self.gain = gain0
            ftst, ttst = self.fwd[test_mask], self.ts[test_mask]
            if self.a.dump_preds:
                self.dumps.append({'label': str(label), 'ts': ttst.copy(), 'p': np.full(len(ttst), np.nan),
                                   'fwd': ftst.copy(), 'keep': np.zeros(len(ttst), bool),
                                   'thr': {f'{c}': None for c in self.covs}})
            return cd.unsupported_row(label, *self._partition, ftst, ttst, self.covs, self.gap, str(e))

    def draw_split2(self, tr_idx, val_pool, rng):
        """Validation rows and fit pool of a split. Every draw a split takes from the shared generator happens here and
        nowhere else, so --path-jobs can advance the generator past a split without fitting it."""
        ts, a, gap = self.ts, self.a, self.gap
        if a.val_mode == 'block':
            # validation = the chronologically last VAL_FRAC of the evaluation-session training rows; fit rows end one
            # horizon before it (purge), so early stopping and the threshold see genuinely unseen, later data
            cand = tr_idx[val_pool[tr_idx]]
            val_idx = cand[-max(1, int(round(len(cand) * cd.VAL_FRAC))):]
            fit_pool = tr_idx[ts[tr_idx] < ts[val_idx[0]] - gap] if len(val_idx) else tr_idx[:0]
        else:
            r = rng.random(len(tr_idx))
            valsel = (r < cd.VAL_FRAC) & val_pool[tr_idx]
            val_idx = tr_idx[valsel]
            fit_pool = tr_idx[~valsel]
        if self.ytrain is not None:                 # alternative TRAINING target (validation keeps the settlement sign)
            fit_pool = fit_pool[np.isfinite(self.ytrain[fit_pool])]
        if len(fit_pool) > cd.SUB_FIT:
            fit_pool = rng.choice(fit_pool, cd.SUB_FIT, replace=False)     # sampled order kept, as the reference
        return val_idx, fit_pool

    def _split2(self, tr_idx, val_pool, test_mask, rng, label):
        """Fit on tr_idx; validation = random VAL_FRAC of the evaluation-session rows among tr_idx (val_pool mask)."""
        X, fwd, ts, a, gap = self.X, self.fwd, self.ts, self.a, self.gap
        val_idx, fit_pool = self.draw_split2(tr_idx, val_pool, rng)
        self._partition = (len(fit_pool), len(val_idx))
        if len(fit_pool) == 0 or len(val_idx) == 0:     # test E-003 (draw_split2 took every random draw of the split)
            raise cd.Unsupported('insufficient support: an empty fit or validation partition')
        if not test_mask.any():
            raise cd.Unsupported('insufficient support: no test row')
        ytr = (fwd[fit_pool] > 0).astype(int) if self.ytrain is None else self.ytrain[fit_pool].astype(int)
        yval = (fwd[val_idx] > 0).astype(int)
        yv_fit = yval                               # labels fit_one sees for validation (early stopping, stack weights)
        self.stack_w = []
        if a.model == 'lgbord4':
            # ordinal signed bins by the fit pool's median |return|: 0 big down, 1 small down (and ties), 2 small up, 3 big up
            med = float(np.median(np.abs(fwd[fit_pool])))
            ord4 = lambda f: np.where(f > 0, np.where(f >= med, 3, 2), np.where(f <= -med, 0, 1)).astype(int)
            ytr, yv_fit = ord4(fwd[fit_pool]), ord4(fwd[val_idx])
        pval = np.zeros(len(val_idx))
        ptst = np.zeros(int(test_mask.sum()))
        iters = []
        Xf, Xv, Xt = X[fit_pool], X[val_idx], X[test_mask]
        self.wfit = None
        if a.weight == 'absret':                    # |return|-weighted training (GMADL-style emphasis on large moves)
            w = np.abs(fwd[fit_pool])
            self.wfit = w / (w.mean() + 1e-18)
        if a.weight == 'recency':
            # literature rank 7, RECENCY WEIGHTS: half-life halflife_days back from the latest fit row. In a CPCV path the
            # latest fit rows can postdate the test groups, so the pre-registered reading is the forward walk only.
            age = (ts[fit_pool].max() - ts[fit_pool]) / 86400.0
            w = 0.5 ** (age / a.halflife_days)
            self.wfit = w / (w.mean() + 1e-18)
        if a.drift_fe or a.model == 'nysagree':     # D1-01 strata of the fit rows (fit_one gets Xf in fit_pool order)
            tf_ = pd.to_datetime(ts[fit_pool], unit='s', utc=True)
            iso = tf_.isocalendar()
            wk = iso['year'].to_numpy(np.int64) * 100 + iso['week'].to_numpy(np.int64)
            b3 = tf_.tz_convert('America/New_York').hour.to_numpy() // 3
            jrv = self.feats.index('rv_60m')
            q = np.nanquantile(X[fit_pool, jrv], [1 / 3, 2 / 3])
            # (no finite rv_60m on the fit rows: q is NaN and every fit row falls in tercile 0, as in the incumbent)
            terc = np.digitize(np.nan_to_num(X[fit_pool, jrv], nan=q[0]), q)
            self.strata_f = wk * 100 + b3 * 10 + terc
        gain_mh = None
        if a.multi_horizon:
            # literature rank 24, SHARED-HORIZON PARTITIONS: each fit row outside the 30-minute purge also enters as a
            # 30-minute row (label = 30-minute settlement sign, indicator column 1); validation and test rows are 15-minute
            # rows (indicator 0), so one model shares its partitions across both horizons and is scored at 15 minutes only.
            f30 = self.fwd30[fit_pool]
            d = fit_pool[self.ok30[fit_pool] & np.isfinite(f30) & (f30 != 0)]
            Xf = np.vstack([np.column_stack([Xf, np.zeros(len(Xf), np.float32)]),
                            np.column_stack([X[d], np.ones(len(d), np.float32)])])
            ytr = np.concatenate([ytr, (self.fwd30[d] > 0).astype(int)])
            Xv = np.column_stack([Xv, np.zeros(len(Xv), np.float32)])
            Xt = np.column_stack([Xt, np.zeros(len(Xt), np.float32)])
            gain_mh, self.gain = self.gain, np.zeros(len(self.gain) + 1)
            self.mh_info = {'rows_15m': int(len(fit_pool)), 'rows_30m': int(len(d))}
        if a.reflect:
            # WITHIN-PAIR SIGN REFLECTION (literature rank 8): every fit row gets a mirrored copy - signed features negated,
            # range positions reflected (1 - x), unsigned features kept - with the label flipped; both halves weigh 0.5,
            # so the fit sees no drift and only sign-symmetric structure. Validation and test rows are never mirrored.
            Xr = Xf.copy()
            Xr[:, self.reflect_neg] *= -1.0
            Xr[:, self.reflect_rng] = 1.0 - Xr[:, self.reflect_rng]
            Xf = np.vstack([Xf, Xr])
            ytr = np.concatenate([ytr, 1 - ytr])
            w0 = np.ones(len(fit_pool)) if self.wfit is None else self.wfit
            self.wfit = 0.5 * np.concatenate([w0, w0])
        keep_v, keep_t = np.ones(len(val_idx), bool), np.ones(int(test_mask.sum()), bool)   # test E-003: gate keeps
        if a.specialist:
            # SWEEP_MATRIX A8 side specialists (scripts/min1_upspec.py recipe at 15m): in the compression regime
            # (volratio_15_240 <= its fit-pool median) an UP model is fit only on dip setups (ret_15m < 0) and trades only
            # UP; a DOWN model only on rally setups (ret_15m > 0) and trades only DOWN; every other row is never selected.
            # --specialist-regime release (SWEEP_MATRIX A3 compression-release x reversion): the same specialists trained
            # and traded only right after a volatility burst (volratio_15_240 >= its fit-pool 75% quantile) - fade the move.
            j_r, j_v = self.feats.index('ret_15m'), self.feats.index('volratio_15_240')
            if a.specialist_regime == 'release':
                comp_thr = float(np.nanquantile(Xf[:, j_v], 0.75))
                reg = lambda Z: Z[:, j_v] >= comp_thr
            else:
                comp_thr = float(np.nanmedian(Xf[:, j_v]))
                reg = lambda Z: Z[:, j_v] <= comp_thr
            setups = lambda Z: (reg(Z) & (Z[:, j_r] < 0), reg(Z) & (Z[:, j_r] > 0))
            pval = np.full(len(val_idx), 0.5)
            ptst = np.full(int(test_mask.sum()), 0.5)
            keep_v, keep_t = np.zeros(len(val_idx), bool), np.zeros(int(test_mask.sum()), bool)
            skipped = []
            wall = self.wfit
            try:
                for k, (fm, vm, tm) in enumerate(zip(setups(Xf), setups(Xv), setups(Xt))):
                    if not (fm.any() and vm.any()):
                        # test E-003: a specialist without fit or validation rows is not fitted; its rows stay excluded
                        skipped.append(f'{("UP", "DOWN")[k]}: no {"fit" if not fm.any() else "validation"} row')
                        continue
                    why = self.fit_unsupported(ytr[fm], yv_fit[vm])
                    if why:                             # nor one whose fit labels the model class cannot fit
                        skipped.append(f'{("UP", "DOWN")[k]}: {why}')
                        continue
                    pv_s, pt_s = np.zeros(int(vm.sum())), np.zeros(int(tm.sum()))
                    self.wfit = None if wall is None else wall[fm]
                    Xt_k = Xt[tm] if tm.any() else Xv[vm][:1]     # no test row: a placeholder, prediction dropped
                    for sd in range(a.nseed):
                        pv, pt, it = self.fit_one(sd, Xf[fm], ytr[fm], Xv[vm], yv_fit[vm], Xt_k)
                        pv_s += pv
                        if tm.any():
                            pt_s += pt
                        iters.append(it)
                    clip = np.maximum if k == 0 else np.minimum          # k 0 = UP specialist, 1 = DOWN specialist
                    # test E-003: the clip turns an infinite mean on the other side into 0.5, so a non-finite mean is
                    # invalid (NaN) before it
                    pval[vm] = np.where(np.isfinite(pv_s), clip(pv_s / a.nseed, 0.5), np.nan)
                    ptst[tm] = np.where(np.isfinite(pt_s), clip(pt_s / a.nseed, 0.5), np.nan)
                    # kept: setup rows whose mean probability is on the specialist's side (a clipped row is excluded)
                    keep_v[vm] = (pv_s / a.nseed > 0.5) if k == 0 else (pv_s / a.nseed < 0.5)
                    keep_t[tm] = (pt_s / a.nseed > 0.5) if k == 0 else (pt_s / a.nseed < 0.5)
            finally:                                # test E-003: restored on a failed fit as well
                self.wfit = wall                    # the specialists' subset weights end here (later gates use wall)
            self.spec_info = {'compression_thr': comp_thr, 'fit_up': int(setups(Xf)[0].sum()), 'fit_down': int(setups(Xf)[1].sum())}
            if skipped:
                self.spec_info['skipped'] = skipped
        elif a.blend_absret:
            # DISCOVERY D5-01, SELECTED |RETURN|-WEIGHT BLEND: the unweighted and the |return|-weighted model (weights as
            # --weight absret) are fitted on the same fit rows with seed 0. On the validation rows inside the compression
            # gate (all validation rows without it) each alpha in BLEND_ALPHAS scores p = (1 - alpha) p_unw + alpha p_abs
            # at the operating coverage on side blend_absret: first-come win rate (ties lose) in each chronological half
            # of those rows (split at their median time); alpha maximizes the smaller half (equal values: smaller alpha).
            # The validation threshold and every gate below then run on the chosen blend as usual.
            wabs = np.abs(fwd[fit_pool])
            pv0, pt0, it0 = self.fit_one(0, Xf, ytr, Xv, yv_fit, Xt)
            self.wfit = wabs / (wabs.mean() + 1e-18)
            try:
                pv1, pt1, it1 = self.fit_one(0, Xf, ytr, Xv, yv_fit, Xt)
            finally:                                # test E-003: restored on a failed fit as well
                self.wfit = None
            iters += [it0, it1]
            vg = np.ones(len(val_idx), bool)
            if a.comp_gate is not None:                     # the same mask as the compression gate below
                jg = self.feats.index('rv_60m')
                vg = Xv[:, jg] <= float(np.nanquantile(Xf[:, jg], a.comp_gate))
            if not (np.all(np.isfinite(pv0[vg])) and np.all(np.isfinite(pv1[vg]))):   # test E-003: before selection
                raise cd.Unsupported('insufficient support: a validation probability the blend selection reads is '
                                     'not a finite number')
            alpha, sel = blend_select(pv0, pv1, vg, ts[val_idx], fwd[val_idx], self.covs[0], a.blend_absret, gap,
                                      a.blend_force_alpha)
            if a.blend_force_alpha is None and all(v['worst_half_wr'] is None for v in sel.values()):
                # test E-003: no alpha has a win rate in both validation halves, so no blend can be selected
                raise cd.Unsupported('insufficient support: no blend alpha has validation trades in both halves')
            pval, ptst = (1.0 - alpha) * pv0 + alpha * pv1, (1.0 - alpha) * pt0 + alpha * pt1
            self.blend_info = {'side': a.blend_absret, 'alpha': alpha, 'forced': a.blend_force_alpha is not None,
                               'gated_val_rows': int(vg.sum()), 'by_alpha': sel}
        elif a.boot_gate:
            # literature rank 16, CALIBRATION-UNCERTAINTY GATE: boot_gate bootstrap replicas of the fit pool (own generator per
            # path and replica, so the shared validation/sub-fit stream is unchanged); the score keeps the replica-mean side
            # and ranks by the lower bound |mean - 0.5| - boot_z * sd (floored at 0), for validation and test alike.
            PV, PT = [], []
            wall = self.wfit
            try:
                for b in range(a.boot_gate):
                    rb = np.random.default_rng([1009, b] + [ord(c) for c in str(label)])
                    bi = rb.integers(0, len(Xf), len(Xf))
                    self.wfit = None if wall is None else wall[bi]
                    pv, pt, it = self.fit_one(b, Xf[bi], ytr[bi], Xv, yv_fit, Xt)
                    PV.append(pv); PT.append(pt); iters.append(it)
            finally:                                # test E-003: restored on a failed fit as well
                self.wfit = wall
            lcb = lambda M: 0.5 + np.sign(M.mean(0) - 0.5) * np.maximum(np.abs(M.mean(0) - 0.5) - a.boot_z * M.std(0, ddof=1), 0.0)
            PV, PT = np.array(PV), np.array(PT)
            pval, ptst = lcb(PV), lcb(PT)

            def lcb_keep(M):                        # test E-003: a side and a positive lower bound (else excluded)
                return (M.mean(0) != 0.5) & (np.abs(M.mean(0) - 0.5) - a.boot_z * M.std(0, ddof=1) > 0)
            keep_v, keep_t = lcb_keep(PV), lcb_keep(PT)
            sd_t = PT.std(0, ddof=1)
            sd_ok = np.isfinite(sd_t)                   # test E-003: the diagnostic reads finite rows only
            ok_b = np.isfinite(ptst)                    # (and the zero-bound share valid rows)
            self.boot_info = {'replicas': a.boot_gate, 'z': a.boot_z,
                              'sd_test_median': float(np.median(sd_t[sd_ok])) if sd_ok.any() else None,
                              'zero_lcb_share_test': float((ptst[ok_b] == 0.5).mean()) if ok_b.any() else None}
        else:
            for sd in range(a.nseed):
                pv, pt, it = self.fit_one(sd, Xf, ytr, Xv, yv_fit, Xt)
                pval += pv
                ptst += pt
                iters.append(it)
            pval /= a.nseed
            ptst /= a.nseed
        if a.model == 'nysagree':                   # test E-003: the agreement gate's kept rows
            keep_v, keep_t = self.agree_keep
        if gain_mh is not None:                     # fold the indicator column's gain out of the per-feature report
            gain_mh += self.gain[:-1]
            self.gain = gain_mh
        # test E-003: the validity of the producer's scores, taken before any encoding and carried through it
        ok_v, ok_t = np.isfinite(pval), np.isfinite(ptst)
        if a.rule_gate:
            # DISCOVERY D5-03, RULE-OFFER RANKING: only the zero-parameter rule's offers can trade, on the rule's side, ranked
            # by the model's probability of that side (a DOWN offer gets p' = p_UP / 2, an UP offer 0.5 + p_UP / 2); every
            # other row gets 0.5, i.e. zero confidence, so the coverage threshold is a quantile over ALL validation rows
            # (overall coverage) and a non-offer row is never traded.
            rv_, rt_ = self.rule_side[val_idx], self.rule_side[test_mask]
            pval, ptst = rule_encode(pval, rv_), rule_encode(ptst, rt_)
            keep_v, keep_t = keep_v & (rv_ != 0), keep_t & (rt_ != 0)     # test E-003: only offers can trade
            ok_v, ok_t = ok_v | (rv_ == 0), ok_t | (rt_ == 0)              # (a non-offer row's 0.5 reads no score)
            self.rule_info = {'offer_share_val': float((rv_ != 0).mean()), 'offer_share_test': float((rt_ != 0).mean()),
                              'offers_test': int((rt_ != 0).sum()),
                              'insufficient_support': bool((rv_ != 0).mean() < self.covs[0])}
        # test E-003: an invalid test score never trades and no diagnostic reads it; an invalid validation score in the
        # threshold population leaves the path without support (its confidence is NaN for cd.unsupported)
        keep_t = keep_t & ok_t
        vconf = np.where(ok_v, np.abs(pval - 0.5), np.nan)
        ftst = fwd[test_mask]
        ttst = ts[test_mask]
        mv = ftst != 0
        mvf = mv & ok_t
        auc = float(cd.roc_auc_score((ftst[mvf] > 0).astype(int), ptst[mvf])) if mvf.sum() > 20 else float('nan')
        uprate = float((ftst[mv] > 0).mean()) if mv.sum() else float('nan')
        row = {'test': label, 'n_fit': int(len(fit_pool)), 'n_val': int(len(val_idx)), 'n_test': int(test_mask.sum()),
               'best_iter': iters, 'auc': auc, 'up_rate': uprate, 'always': {}, 'bycov': {}}
        if a.specialist:
            row['specialist'] = self.spec_info              # AUC above is over every test row (non-setup rows at 0.5)
        if a.boot_gate:
            row['boot_gate'] = self.boot_info               # AUC above is over the lower-bound scores
        if a.blend_absret:
            row['blend'] = self.blend_info                  # D5-01 selection (AUC above is over the chosen blend)
        if a.rule_gate:
            row['rule_gate'] = self.rule_info               # D5-03 offers (AUC above is over the offer-ranked scores)
        if self.stack_w:
            row['stack_weights_mean'] = np.mean(self.stack_w, axis=0).round(4).tolist()   # order lgb, logit, forest, nystroem
        if a.multi_horizon:
            row['multi_horizon'] = self.mh_info
        if a.drift_fe and a.model == 'lgb':
            row['lgb_drift_fe'] = self.lgbfe_info
        if a.model == 'nysagree':
            row['agree_gate'] = self.agree_info             # AUC above is over every test row (gated-out rows at 0.5)
        if a.model == 'nystail':
            row['tail_aux'] = self.tail_info                # D4-01 fit diagnostics (fit rows only)
        vgate = np.ones(len(val_idx), bool)
        if a.mag_keep is not None:
            # MAGNITUDE GATE (sign-invariant): a second model ranks "big move" (|ret| >= the fit pool's median |ret|);
            # keep only the top mag_keep share (threshold = validation quantile); the direction threshold is then the
            # coverage quantile among validation rows that pass the gate, so coverage is relative to gated rows.
            med = float(np.median(np.abs(fwd[fit_pool])))
            if a.mag_model == 'har':
                # literature ranks 12-15, HAR SIZE MODEL: OLS of log|return| on log realized volatility at 15 m, 60 m, 4 h and
                # 1 d plus the intraday clock (hour sin/cos), fit-pool medians for missing values; ranks rows by predicted size
                cols = [self.feats.index(c) for c in ('rv_15m', 'rv_60m', 'rv_240m', 'rv_1440m')]
                clk = [self.feats.index(c) for c in ('hour_sin', 'hour_cos') if c in self.feats]
                def design(Z, fill=None):
                    V = np.log(np.maximum(Z[:, cols], 1e-12))
                    fill = np.nanmedian(V, axis=0) if fill is None else fill
                    V = np.where(np.isfinite(V), V, fill)
                    return np.column_stack([np.ones(len(Z)), V] + ([Z[:, clk]] if clk else [])), fill
                D, fill = design(X[fit_pool])
                D = np.where(np.isfinite(D), D, 0.0)
                yl = np.log(np.abs(fwd[fit_pool]) + 1e-12)
                beta = np.linalg.lstsq(D, yl, rcond=None)[0]
                pred = lambda Z: np.where(np.isfinite(design(Z, fill)[0]), design(Z, fill)[0], 0.0) @ beta
                mval, mtst = pred(X[val_idx]), pred(X[test_mask])
            else:
                mm = make_model(a, 0)
                big_f = (np.abs(fwd[fit_pool]) >= med).astype(int)
                if len(fit_pool) < 2:                   # test E-003 (no class check: LightGBM fits one class)
                    raise cd.Unsupported('insufficient support: fewer than two fit rows for the magnitude model '
                                         '(LightGBM needs two)')
                mm.fit(X[fit_pool], big_f)
                mval = mm.predict_proba(X[val_idx])[:, 1]
                mtst = mm.predict_proba(X[test_mask])[:, 1]
            if not np.all(np.isfinite(mval)):           # test E-003: before the gate's quantile
                raise cd.Unsupported('insufficient support: a magnitude-gate validation score is not a finite number')
            gthr = float(np.quantile(mval, 1 - a.mag_keep))
            vgate = mval >= gthr
            tgate = np.isfinite(mtst) & (mtst >= gthr)  # test E-003: a non-finite test score fails the gate
            big = (np.abs(ftst) >= med).astype(int)
            fm = np.isfinite(mtst)                      # test E-003: the diagnostic reads finite scores only
            row['mag_gate'] = {'keep': a.mag_keep, 'median_abs_ret': med, 'gate_thr': gthr,
                               'test_pass_rate': float(tgate.mean()),
                               'mag_auc_test': (float(cd.roc_auc_score(big[fm], mtst[fm]))
                                                if 0 < big[fm].sum() < fm.sum() else None),
                               'tie_rate_pass': float((ftst[tgate] == 0).mean()) if tgate.any() else None,
                               'tie_rate_fail': float((ftst[~tgate] == 0).mean()) if (~tgate).any() else None}
            ptst = np.where(tgate, ptst, 0.5)        # gated-out rows get zero confidence: never selected
            keep_t = keep_t & tgate
        if a.meta_keep is not None:
            # SWEEP_MATRIX A4 META-LABELER (scripts/m15_meta.py idea on the Deriv feed). Out-of-fold primary predictions on
            # the fit pool come from two chronological halves (the first ends one horizon before the split, so no label
            # of one half overlaps the other); a meta model learns P(primary side correct) on moved fit rows from
            # [features, primary confidence, primary side]; validation and test use the full-pool primary (more confident
            # than the half-pool OOF primary: a declared approximation); keep the top meta_keep share by meta score
            # (validation quantile); the direction threshold is the coverage quantile among validation rows in the gate.
            # (0.5: the half-pool primaries are fit by fit_one - the selected model class and recipe, seed 0, early stopping
            # on the validation rows - so a kernel or pooled primary gets a meta model learned on its own errors)
            # test E-003: the meta model reads the primary score of every validation row (the gate's quantile takes
            # them all) and of every test row; a meta score whose primary score is invalid is itself invalid
            if not np.all(ok_v):
                raise cd.Unsupported('insufficient support: a primary validation score the meta gate reads is not a '
                                     'finite number')
            tf_ = ts[fit_pool]
            mid = float(np.median(tf_))
            halves = (tf_ < mid - gap, tf_ >= mid)
            if not (halves[0].any() and halves[1].any()):
                raise cd.Unsupported('insufficient support: an empty out-of-fold half of the meta-labeler')
            oof = np.full(len(fit_pool), np.nan)
            wall, gain0, nstack = self.wfit, self.gain.copy(), len(self.stack_w)
            try:
                for trm, tem in ((halves[0], halves[1]), (halves[1], halves[0])):
                    self.wfit = None if wall is None else wall[trm]
                    _, oof[tem], _ = self.fit_one(0, Xf[trm], ytr[trm], Xv, yv_fit, Xf[tem])
            finally:                                # test E-003: restored on a failed half as well
                self.wfit, self.gain = wall, gain0    # the reported gain and stack weights stay the primary's
                del self.stack_w[nstack:]
            fwf = fwd[fit_pool]
            ok = np.isfinite(oof) & (fwf != 0)
            if ok.sum() < 2:                            # LightGBM needs two fit rows
                raise cd.Unsupported('insufficient support: fewer than two moved out-of-fold rows for the '
                                     'meta-labeler (LightGBM needs two)')
            Zf = np.column_stack([Xf, np.abs(oof - 0.5), (oof > 0.5).astype(np.float32)])
            mm = cd.lgb.LGBMClassifier(objective='binary', learning_rate=0.02, num_leaves=63, min_child_samples=400,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=20,
                                       n_estimators=400, n_jobs=a.n_jobs, verbosity=-1, random_state=0)
            y_meta = ((oof[ok] > 0.5) == (fwf[ok] > 0)).astype(int)
            mm.fit(Zf[ok], y_meta)                      # (LightGBM: no class check, it fits one class)
            mval = mm.predict_proba(np.column_stack([Xv, np.abs(pval - 0.5), (pval > 0.5).astype(np.float32)]))[:, 1]
            mtst = mm.predict_proba(np.column_stack([Xt, np.abs(ptst - 0.5), (ptst > 0.5).astype(np.float32)]))[:, 1]
            if not np.all(np.isfinite(mval)):           # test E-003: before the gate's quantile
                raise cd.Unsupported('insufficient support: a meta-gate validation score is not a finite number')
            gthr = float(np.quantile(mval, 1 - a.meta_keep))
            vgate &= mval >= gthr
            tg = np.isfinite(mtst) & ok_t & (mtst >= gthr)   # test E-003: an invalid test score fails the gate
            corr_t = ((ptst > 0.5) == (ftst > 0)) & (ftst != 0)
            fm = np.isfinite(mtst) & ok_t               # test E-003: the diagnostic reads valid meta scores only
            row['meta_gate'] = {'keep': a.meta_keep, 'gate_thr': gthr, 'oof_rows': int(ok.sum()),
                                'oof_auc': float(cd.roc_auc_score((fwf[ok] > 0).astype(int), oof[ok])) if ok.sum() > 20 else None,
                                'test_pass_rate': float(tg.mean()),
                                'meta_auc_test': (float(cd.roc_auc_score(corr_t[fm].astype(int), mtst[fm]))
                                                  if 0 < corr_t[fm].sum() < fm.sum() else None)}
            ptst = np.where(tg, ptst, 0.5)
            keep_t = keep_t & tg
        if a.comp_gate is not None:
            # SWEEP_MATRIX A2 COMPRESSION GATE: trade only in the low-volatility regime, rv_60m <= its fit-pool quantile
            # comp_gate; the direction threshold is the coverage quantile among validation rows inside the gate.
            j = self.feats.index('rv_60m')
            cthr = float(np.nanquantile(Xf[:, j], a.comp_gate))
            cv, ct = Xv[:, j] <= cthr, Xt[:, j] <= cthr
            vgate &= cv
            row['comp_gate'] = {'q': a.comp_gate, 'rv_60m_thr': cthr, 'test_pass_rate': float(ct.mean()),
                                'tie_rate_pass': float((ftst[ct] == 0).mean()) if ct.any() else None,
                                'tie_rate_fail': float((ftst[~ct] == 0).mean()) if (~ct).any() else None}
            ptst = np.where(ct, ptst, 0.5)
            keep_t = keep_t & ct
        why = cd.unsupported(vconf[vgate], keep_v[vgate])
        if a.dump_preds:                             # analysis only: keep the scored test predictions of every path
            self.dumps.append({'label': str(label), 'ts': ttst.copy(), 'p': ptst.copy(), 'fwd': ftst.copy(),
                               'keep': keep_t.copy(),       # test E-003: rows every gate kept (others never trade)
                               'thr': {f'{c}': None if why else float(np.quantile(vconf[vgate], 1 - c))
                                       for c in self.covs}})
        # test E-003: the threshold comes from the gated validation population as before; a row any gate excluded
        # (keep False: D5-03 non-offers among them) trades on no side, COMBINED included, even at threshold zero
        return cd.score_row(row, ptst, ftst, ttst, vconf[vgate], self.covs, gap, keep_v[vgate], keep_t)


_PJ = {}        # --path-jobs: the fitter and the validation-pool mask, set once before the first worker is forked


def _path_job(spec):
    """One split in a forked worker. The generator is rebuilt in the state the sequential loop has when it reaches this
    split, so the split draws the same validation rows and sub-fit sample; gain and dumps start empty and are returned."""
    label, tr_idx, test_mask, ok30, state = spec
    f = _PJ['fitter']
    rng = np.random.default_rng()
    rng.bit_generator.state = state
    f.gain = np.zeros_like(f.gain)
    f.dumps = []
    f.ok30 = ok30
    row = f.split2(tr_idx, _PJ['val_pool'], test_mask, rng, label)
    return row, f.gain, f.dumps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', required=True)
    ap.add_argument('--horizon', type=int, default=15, choices=(15, 30))
    sessions = ('ny', 'ldn', 'asia', 'overlap', 'all') + tuple(LOCAL_WINDOWS)
    ap.add_argument('--session', default='ny', choices=sessions)
    ap.add_argument('--train-session', default=None, choices=sessions)
    ap.add_argument('--covs', default='0.03,0.02,0.05')
    ap.add_argument('--nseed', type=int, default=1)
    ap.add_argument('--features', default='all', choices=('all', 'noclock', 'own', 'own_noclock'))
    ap.add_argument('--feature-list', default=None)
    ap.add_argument('--num-leaves', type=int, default=255)
    ap.add_argument('--min-child', type=int, default=400)
    ap.add_argument('--lr', type=float, default=0.02)
    ap.add_argument('--n-estimators', type=int, default=800)
    ap.add_argument('--colsample', type=float, default=0.5)
    ap.add_argument('--reg-lambda', type=float, default=20.0)
    ap.add_argument('--model', default='lgb', choices=('lgb', 'ens3', 'logit', 'nystroem', 'forest', 'pool', 'lgbbrier',
                                                       'stack', 'lgbord4', 'nysagree', 'nystail'),
                    help='lgb (default, the mirror); ens3 = LightGBM + XGBoost + CatBoost mean (SWEEP_MATRIX A1; XGB/Cat use '
                         '--n-estimators/--lr); logit = L2 logistic regression on fit-pool-standardized features; '
                         'nystroem = RBF-kernel features + logistic SGD (SWEEP_MATRIX B6); forest = shallow bagged '
                         'extra-trees; pool = equal-weight probability mean of lgb, logit, forest and nystroem; stack = the '
                         'same four with validation-log-loss simplex weights shrunk toward equal (--stack-shrink); lgbord4 = '
                         'LightGBM on 4 ordinal signed bins, P(up) = P(small up) + P(big up)')
    ap.add_argument('--stack-shrink', type=float, default=0.5, help='stack: weight on the equal-weight pool (0 = fitted weights)')
    ap.add_argument('--mag-model', default='lgb', choices=('lgb', 'har'),
                    help='--mag-keep size model: lgb big-move classifier (default) or har (OLS of log|ret| on log rv 15m/60m/4h/1d + clock)')
    ap.add_argument('--boot-gate', type=int, default=None,
                    help='calibration-uncertainty gate: B bootstrap replicas; rank by |mean-0.5| - boot_z*sd (replaces --nseed)')
    ap.add_argument('--boot-z', type=float, default=1.0)
    ap.add_argument('--halflife-days', type=float, default=60.0, help='--weight recency half-life in days')
    ap.add_argument('--multi-horizon', action='store_true',
                    help='literature rank 24: fit rows also enter with the 30-minute label and an indicator (30-minute purge); '
                         'scored at 15 minutes only')
    ap.add_argument('--cat-depth', type=int, default=8)
    ap.add_argument('--nys-gamma-mult', type=float, default=1.0, help='RBF gamma = mult / n_features (nystroem, pool)')
    ap.add_argument('--nys-components', type=int, default=500)
    ap.add_argument('--comp-gate', type=float, default=None,
                    help='SWEEP_MATRIX A2 compression gate: trade only rows with rv_60m <= its fit-pool quantile q (e.g. 0.33)')
    ap.add_argument('--meta-keep', type=float, default=None,
                    help='SWEEP_MATRIX A4 meta-labeler gate: keep the top share (e.g. 0.5) by P(primary side correct)')
    ap.add_argument('--reflect', action='store_true',
                    help='literature rank 8: add sign-mirrored copies of the fit rows (v0.1 features only; label flipped)')
    ap.add_argument('--null-shift-weeks', type=int, default=0,
                    help='analysis only: replace every label by the label W weeks later (sign-null replay); never certifies')
    ap.add_argument('--weight', default='none', choices=('none', 'absret', 'recency'),
                    help='absret: sample weights |settled return| / mean (fit rows only); recency: half-life --halflife-days '
                         'back from the latest fit row (reading: forward walk only)')
    ap.add_argument('--aci-gamma', type=float, default=None,
                    help='REFUSED (test E-003, owner decision 13): the adaptive-conformal gate of earlier builds')
    ap.add_argument('--aci-target', type=float, default=0.56, help='unused: ACI is refused')
    ap.add_argument('--dump-preds', default=None,
                    help='analysis only: write every path\'s test decisions (ts, p, fwd, thresholds) to this .npz')
    ap.add_argument('--specialist', action='store_true',
                    help='SWEEP_MATRIX A8: compression-regime dip-buy UP specialist + rally-sell DOWN specialist')
    ap.add_argument('--specialist-regime', default='compression', choices=('compression', 'release'),
                    help='release = SWEEP_MATRIX A3: specialists only after a volatility burst (volratio_15_240 >= q75)')
    ap.add_argument('--logit-c', type=float, default=0.05)
    ap.add_argument('--val-mode', default='random', choices=('random', 'block'),
                    help='random = repository protocol (15%% random rows); block = last 15%% of training rows in time, purged')
    ap.add_argument('--no-early-stop', action='store_true', help='fit all n_estimators; validation only sets the threshold')
    ap.add_argument('--train-label', default='sign', choices=('sign', 'tb1', 'tb2', 'resid', 'agree3'),
                    help='fit target: Deriv settlement sign (default), triple-barrier first touch k=1/2, the USD-basket '
                         'residualized sign (SWEEP_MATRIX F4), or agree3 (discovery D1-02: the side agreed at entry delays '
                         '1/5/15 s, other rows dropped); validation, thresholds and scoring always use the settlement sign')
    ap.add_argument('--label-dir', default=None, help='default: cache/labels_tb_dev_v0.1 (tb1/tb2), cache/labels_resid_dev_v0.1 '
                                                      '(resid), cache/labels_agree_dev_v0.1 (agree3)')
    ap.add_argument('--monotone', default='none', choices=('none', 'catchup'),
                    help='discovery D2-02 (LightGBM only): catchup = decreasing in own ret_15m; x_usd_weak_15m increasing '
                         'for XXX/USD pairs, decreasing for USD/XXX pairs; every other feature unconstrained')
    ap.add_argument('--drift-fe', action='store_true',
                    help='discovery D1-01: drift-neutral fit with stratum fixed effects (logit on signed coordinates, or nystroem; '
                         '0.8: lgb on signed coordinates with profiled stratum offsets, D3-01); scores exclude every stratum effect')
    ap.add_argument('--tail-aux', type=float, default=None,
                    help='0.9 discovery D4-01 (--model nystail only): weight of the fixed-mass tail auxiliary (0 = the matched '
                         'control: the same L-BFGS kernel fit without it)')
    ap.add_argument('--blend-absret', default=None, choices=('UP', 'DOWN'),
                    help='0.10 discovery D5-01 (--model lgb): fit the unweighted and the |return|-weighted model on the same '
                         'fit rows; choose alpha in {0, 0.5, 1} for p = (1 - alpha) p_unw + alpha p_abs by the smaller SIDE win '
                         'rate over the two chronological halves of the (gated) validation rows at the operating coverage')
    ap.add_argument('--blend-force-alpha', type=float, default=None,
                    help='regression only (never a candidate run): use this alpha (0, 0.5 or 1) instead of selecting it')
    ap.add_argument('--rule-gate', default=None, choices=RULE_SIGNALS,
                    help='0.10 discovery D5-03: offers = rows where this audited rule column (read for the gate only, never a '
                         'model feature) is finite and non-zero, side = sign(direction x signal) as tools/cpcv_rule.py; offers '
                         'ranked by the model probability of that side; other rows never trade')
    ap.add_argument('--rule-direction', type=int, default=None, choices=(1, -1), help='--rule-gate: 1 (default) or -1')
    ap.add_argument('--rule-usd-oriented', action='store_true',
                    help='--rule-gate: the column is USD-oriented (converted to the quote direction as tools/cpcv_rule.py)')
    ap.add_argument('--rule-side', default=None, choices=('UP', 'DOWN'),
                    help='--rule-gate: keep only this side\'s offers (default: both sides)')
    ap.add_argument('--signed-only', action='store_true',
                    help='0.8 same-input controls: logit (e67drfe inputs and scaling, global intercept, no strata) or lgb '
                         '(the usual booster) on the signed coordinates only')
    ap.add_argument('--families', default=None,
                    help='SWEEP_MATRIX family blocks to keep (comma list of tf,sg,kf,rm,fd,cx,tk,pa,cm, or all / none); every '
                         'other block column is dropped from the --features selection')
    ap.add_argument('--include-blackout', action='store_true', help='keep Deriv purchase-blackout decisions (not Deriv-faithful)')
    ap.add_argument('--july-filter', action='store_true',
                    help='the July 2026 tradeability filter, without the venue window of owner decision 14 '
                         '(regression against stored results only; such a result cannot be promoted)')
    ap.add_argument('--mag-keep', type=float, default=None,
                    help='magnitude gate: keep only the top share (e.g. 0.5) of rows by predicted big-move probability')
    ap.add_argument('--dataset', default=str(cd.LAB / 'cache' / 'dataset_dev_v0.1'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--n-jobs', type=int, default=16, help='threads of each model fit')
    ap.add_argument('--path-jobs', type=int, default=1,
                    help='splits (CPCV paths, forward walks) fitted at once, each in a forked worker with --n-jobs threads')
    ap.add_argument('--no-forward', action='store_true')
    ap.add_argument('--max-paths', type=int, default=15)
    a = ap.parse_args()
    if a.path_jobs < 1 or (a.path_jobs > 1 and a.n_jobs < 1):    # a sequential run takes any --n-jobs
        raise SystemExit('--path-jobs must be >= 1; parallel paths (--path-jobs > 1) need --n-jobs >= 1 per worker')
    a.train_session = a.train_session or a.session
    if a.aci_gamma is not None:
        raise SystemExit('--aci-gamma: ' + ACI_REFUSED)
    if a.reflect and a.meta_keep is not None:      # the meta gate's half-pool split indexes the unmirrored fit pool
        raise SystemExit('--reflect and --meta-keep are exclusive')
    if a.boot_gate is not None and (a.boot_gate < 2 or a.nseed != 1 or a.specialist):
        raise SystemExit('--boot-gate needs B >= 2 replicas, --nseed 1 and no --specialist')
    if a.model == 'lgbord4' and (a.reflect or a.train_label != 'sign'):
        raise SystemExit('--model lgbord4 defines its own target: no --reflect, no --train-label')
    if a.drift_fe and (a.model not in ('logit', 'nystroem', 'lgb') or a.reflect or a.meta_keep is not None or a.specialist
                       or a.boot_gate or a.multi_horizon or a.signed_only or (a.model == 'lgb' and (a.no_early_stop
                                                                                                    or a.monotone != 'none'))):
        raise SystemExit('--drift-fe needs --model logit, nystroem or lgb (with early stopping, no monotone) and no reflect / '
                         'meta / specialist / boot / multi-horizon / signed-only')
    if a.signed_only and (a.model not in ('logit', 'lgb') or a.reflect or a.meta_keep is not None or a.specialist
                          or a.boot_gate or a.multi_horizon or a.monotone != 'none' or a.mag_keep is not None):
        raise SystemExit('--signed-only needs --model logit or lgb and no reflect / meta / specialist / boot / multi-horizon / '
                         'monotone / magnitude gate')
    if a.model == 'nysagree' and (a.nseed != 1 or a.reflect or a.meta_keep is not None or a.specialist or a.boot_gate
                                  or a.multi_horizon or a.monotone != 'none' or a.aci_gamma is not None
                                  or a.mag_keep is not None or a.comp_gate is not None):
        raise SystemExit('--model nysagree needs --nseed 1 and no other gate, reflect, meta, specialist, boot, '
                         'multi-horizon, monotone or ACI')
    if (a.model == 'nystail') != (a.tail_aux is not None):
        raise SystemExit('--model nystail and --tail-aux go together')
    if a.model == 'nystail' and (not np.isfinite(a.tail_aux) or a.tail_aux < 0 or a.nseed != 1 or a.weight != 'none'
                                 or a.reflect or a.meta_keep is not None
                                 or a.specialist or a.boot_gate or a.multi_horizon or a.monotone != 'none'
                                 or a.aci_gamma is not None or a.mag_keep is not None or a.drift_fe or a.signed_only
                                 or a.train_label != 'sign'):
        raise SystemExit('--model nystail needs a finite --tail-aux >= 0, --nseed 1, no weights and no reflect / meta / specialist / '
                         'boot / multi-horizon / monotone / ACI / magnitude gate / drift-fe / signed-only / train label')
    if a.multi_horizon and (a.horizon != 15 or a.reflect or a.meta_keep is not None or a.specialist or a.boot_gate
                            or a.model == 'lgbord4' or a.comp_gate is not None or a.weight != 'none'
                            or a.train_label != 'sign' or a.null_shift_weeks):
        raise SystemExit('--multi-horizon runs at --horizon 15 with no reflect / meta / specialist / boot / lgbord4 / comp '
                         'gate / weights / train label / null shift')
    if a.blend_absret and (a.model != 'lgb' or a.weight != 'none' or a.nseed != 1 or a.reflect or a.meta_keep is not None
                           or a.specialist or a.boot_gate or a.multi_horizon or a.aci_gamma is not None
                           or a.mag_keep is not None or a.drift_fe or a.signed_only or a.monotone != 'none'
                           or a.train_label != 'sign' or a.rule_gate is not None):
        raise SystemExit('--blend-absret needs --model lgb, --weight none and --nseed 1, and no reflect / meta / specialist / '
                         'boot / multi-horizon / ACI / magnitude gate / drift-fe / signed-only / monotone / train label / '
                         'rule gate')
    if a.blend_force_alpha is not None and (not a.blend_absret or a.blend_force_alpha not in BLEND_ALPHAS
                                            or a.max_paths != 1 or not a.no_forward):
        raise SystemExit('--blend-force-alpha is regression only: it needs --blend-absret, a value in {0, 0.5, 1}, '
                         '--max-paths 1 and --no-forward')
    if a.rule_gate is None and (a.rule_direction is not None or a.rule_usd_oriented or a.rule_side is not None):
        raise SystemExit('--rule-direction, --rule-usd-oriented and --rule-side need --rule-gate')
    if a.rule_gate is not None and (a.comp_gate is not None or a.meta_keep is not None or a.mag_keep is not None
                                    or a.specialist or a.boot_gate or a.aci_gamma is not None or a.multi_horizon
                                    or a.model in ('nysagree', 'nystail')):
        raise SystemExit('--rule-gate takes no other gate (compression / meta / magnitude / specialist / boot / ACI / '
                         'nysagree / nystail) and no multi-horizon')
    covs = cd.parse_covs(a.covs)
    for name, value, ok in (('--nseed', a.nseed, a.nseed >= 1), ('--max-paths', a.max_paths, a.max_paths >= 1),
                            ('--comp-gate', a.comp_gate, a.comp_gate is None or 0 <= a.comp_gate <= 1),
                            ('--meta-keep', a.meta_keep, a.meta_keep is None or 0 < a.meta_keep <= 1),
                            ('--mag-keep', a.mag_keep, a.mag_keep is None or 0 < a.mag_keep <= 1)):
        if not ok:                                  # NaN fails every comparison above
            raise SystemExit(f'{name} {value}: out of its range (seeds and paths >= 1, gate quantile in [0, 1], '
                             'gate shares in (0, 1])')
    for name, value, ok in option_domains(a):
        if not ok:                                  # test E-003: before anything is loaded or written
            raise SystemExit(f'{name} {value}: outside the domain of the estimator this run fits with it '
                             '(option_domains)')
    gap = a.horizon * 60
    out = Path(a.out or f'{a.pair}_{a.horizon}m_{a.session}_x_cpcv_result.json')
    t0 = time.time()
    listed = json.loads(Path(a.feature_list).read_text()) if a.feature_list else None
    if a.families is not None:
        if listed is not None:
            raise SystemExit('--families and --feature-list are exclusive')
        import pyarrow.parquet as pq
        cols = pq.read_schema(Path(a.dataset) / f'{a.pair}.parquet').names
        want = set(FAMILY_PREFIXES) if a.families == 'all' else set() if a.families == 'none' else \
            {f.strip() + '_' for f in a.families.split(',')}
        unknown = want - set(FAMILY_PREFIXES)
        if unknown:
            raise SystemExit(f'unknown families {sorted(unknown)}')
        listed = [c for c in cd.feature_list(cols, a.features) if not c.startswith(FAMILY_PREFIXES) or c[:3] in want]
        if want and not any(c[:3] in want for c in listed):
            raise SystemExit(f'dataset {a.dataset} has no columns for families {sorted(want)}')
    label_dir = a.label_dir or str(cd.LAB / 'cache' / {'resid': 'labels_resid_dev_v0.1', 'agree3': 'labels_agree_dev_v0.1'}.get(
        a.train_label, 'labels_tb_dev_v0.1'))
    X, fwd, ts, feats = cd.load(a.dataset, a.pair, a.horizon, a.features, 1, listed)
    if a.rule_gate is not None and a.rule_gate in feats:
        raise SystemExit(f'--rule-gate {a.rule_gate} is also a model feature: the gate column must stay outside the model')
    if a.null_shift_weeks:
        # SIGN-NULL REPLAY (literature: full-search null replay / reality check): every decision gets the settled label of
        # the decision exactly W weeks later (circular over the dataset span; same weekday and UTC time of day), which keeps
        # the label's drift and intraday structure and breaks any feature-label link. Missing matches become NaN.
        span = int(ts.max() - ts.min()) + 60
        tgt = ts.min() + ((ts - ts.min() + null_shift(a.null_shift_weeks, span)) % span)
        fwd = pd.Series(fwd, index=ts).reindex(tgt).to_numpy(np.float64)
    sess = session(ts, a.session)
    tsess = session(ts, a.train_session)
    if not a.include_blackout:                   # Deriv purchase blackout: untradeable decisions out of every role
        tr_ok = cd.tradeable(ts, a.horizon, venue_window=not a.july_filter)
        sess &= tr_ok
        tsess &= tr_ok
    elif not a.july_filter:                      # owner decision 14 holds with --include-blackout too
        tr_ok = cd.venue_open(ts, a.horizon)
        sess &= tr_ok
        tsess &= tr_ok
    moved = np.isfinite(fwd) & (fwd != 0.0)
    tgs = ts[sess]
    bnds = [tgs[int(k * len(tgs) / cd.N_GROUPS)] for k in range(cd.N_GROUPS)] + [tgs[-1] + 1]
    groups = [(int(bnds[g]), int(bnds[g + 1])) for g in range(cd.N_GROUPS)]
    day = lambda s: pd.to_datetime(s, unit='s', utc=True).strftime('%Y-%m-%d')
    data_uprate = float((fwd[sess & moved] > 0).mean())
    manifest = Path(a.dataset) / 'manifest.json'
    knobs = {k: getattr(a, k) for k in ('train_session', 'num_leaves', 'min_child', 'lr', 'n_estimators', 'colsample', 'reg_lambda',
                                        'val_mode', 'no_early_stop', 'train_label', 'mag_keep', 'include_blackout', 'families',
                                        'model', 'cat_depth', 'logit_c', 'specialist', 'comp_gate', 'weight',
                                        'aci_gamma', 'aci_target', 'meta_keep', 'specialist_regime', 'nys_gamma_mult',
                                        'nys_components', 'null_shift_weeks', 'reflect', 'stack_shrink', 'mag_model',
                                        'boot_gate', 'boot_z', 'halflife_days', 'multi_horizon', 'drift_fe', 'monotone',
                                        'signed_only', 'tail_aux', 'blend_absret', 'blend_force_alpha', 'rule_gate',
                                        'rule_direction', 'rule_usd_oriented', 'rule_side')}
    res = {
        'harness': X_VERSION, 'base_harness': cd.HARNESS_VERSION, 'tag': a.tag,
        'run_status': 'running',                    # 'complete' when written at the end
        'model': f'LightGBM @{a.horizon}m on the Deriv feed, eval session {a.session}, fit session {a.train_session}, knobs {knobs}',
        'key': f'{a.pair}.{a.horizon}m', 'data': 'Deriv 1-second mid ticks only (/mnt/data/deriv), no Dukascopy',
        'dataset': {'dir': a.dataset, 'manifest_sha256': cd.sha256(manifest)},
        'window': {'first_decision': day(ts[0]), 'last_decision': day(ts[-1])},
        'breakeven': cd.BE, 'session': a.session, 'tradeable_only': not a.include_blackout,
        'blackout_rule': None if a.include_blackout else cd.BLACKOUT_RULE,
        'venue_window_rule': None if a.july_filter else cd.VENUE_WINDOW_RULE,
        'regression_only': bool(a.july_filter),     # True: a regression against stored results, never a candidate
        'knobs': knobs, 'horizon_min': a.horizon, 'gap_s': gap, 'covs': covs,
        'operating_cov': covs[0], 'nseed': a.nseed, 'num_leaves': a.num_leaves,
        'feature_set': f'list:{Path(a.feature_list).name}' if a.feature_list else
                       (f'{a.features}+fam:{a.families}' if a.families is not None else a.features),
        'n_features': len(feats), 'features': feats,
        'n_groups': cd.N_GROUPS, 'k_test': cd.K_TEST, 'purge_s': gap, 'embargo_s': gap, 'sub_fit': cd.SUB_FIT,
        'val_frac': cd.VAL_FRAC, 'min_trades_per_path': cd.MIN_TRADES, 'max_paths': a.max_paths,
        'rows': int(len(ts)), 'rows_in_session': int(sess.sum()), 'rows_fit_session': int(tsess.sum()),
        'in_session_moved_up_rate': data_uprate, 'groups': [(day(lo), day(hi - 1)) for lo, hi in groups],
        'CERT_RULE': 'side CERTIFIED iff VALID AND p10>=0.541 AND frac_clear_BE>=0.80 (unrounded path win rates; paths with >= 25 trades)',
        'falsifier': {'version': 3, 'registered_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'when': 'before any path is fitted',
                      'KILL_if': f'at the operating coverage {covs[0]}: each side p10<0.541 OR frac_clear_BE<0.80',
                      'INVALID_if': 'in-session moved up-rate outside [0.47,0.53], or any path moved up-rate outside [0.45,0.55]; an INVALID run certifies nothing in any field'},
        'code': {'x_harness_sha256': cd.sha256(__file__), 'base_harness_sha256': cd.sha256(cd.__file__),
                 'sessions_py_sha256': cd.sha256(cd.REPO / 'scripts' / 'sessions.py'), 'lightgbm': cd.lgb.__version__},
    }
    out.write_text(cd.result_json(res))
    print(f'[{a.pair} {a.horizon}m eval {a.session} fit {a.train_session}] rows {len(ts):,} eval {int(sess.sum()):,} '
          f'fit-session {int(tsess.sum()):,} features {len(feats)} knobs {knobs}', flush=True)
    fitter = XFitter(X, fwd, ts, a, gap, covs)
    fitter.ytrain = None
    fitter.feats = feats
    fitter.dumps = []
    if a.multi_horizon:
        d30 = pd.read_parquet(Path(a.dataset) / f'{a.pair}.parquet', columns=['decision_ts', 'ret_fwd_30m', 'valid_30m'])
        f30 = d30['ret_fwd_30m'].to_numpy(np.float64).copy()
        f30[~d30['valid_30m'].to_numpy(bool)] = np.nan
        s30 = pd.Series(f30, index=d30['decision_ts'].to_numpy(np.int64))
        if not s30.index.is_unique:
            raise SystemExit('--multi-horizon: duplicate decision_ts in the dataset')
        fitter.fwd30 = s30.reindex(ts).to_numpy(np.float64)
        res['multi_horizon'] = {'label': 'sign of ret_fwd_30m where valid_30m', 'purge_30m_s': 1800,
                                'aligned_finite': int(np.isfinite(fitter.fwd30).sum())}
    if a.rule_gate is not None:
        # D5-03: the rule column is read for the gate only (never a model feature), aligned on decision_ts like
        # --multi-horizon's labels; side = sign(direction x signal) after the USD orientation, exactly tools/cpcv_rule.py
        dr = pd.read_parquet(Path(a.dataset) / f'{a.pair}.parquet', columns=['decision_ts', a.rule_gate])
        sr = pd.Series(dr[a.rule_gate].to_numpy(np.float64), index=dr['decision_ts'].to_numpy(np.int64))
        if not sr.index.is_unique:
            raise SystemExit('--rule-gate: duplicate decision_ts in the dataset')
        sig = sr.reindex(ts).to_numpy(np.float64)
        if a.rule_usd_oriented:
            sig = sig * (1.0 if a.pair in RULE_USD_BASE else -1.0)
        sig = (a.rule_direction or 1) * sig
        side_ = np.where(np.isfinite(sig), np.sign(sig), 0.0)
        if a.rule_side == 'UP':
            side_ = np.where(side_ > 0, side_, 0.0)
        elif a.rule_side == 'DOWN':
            side_ = np.where(side_ < 0, side_, 0.0)
        fitter.rule_side = side_
        res['rule_gate'] = {'column': a.rule_gate, 'direction': a.rule_direction or 1, 'usd_oriented': a.rule_usd_oriented,
                            'side': a.rule_side or 'both',
                            'offers_in_session': int(((side_ != 0) & sess).sum()), 'rows_in_session': int(sess.sum()),
                            'scoring': 'offer p\' = p_UP / 2 (DOWN) or 0.5 + p_UP / 2 (UP); other rows 0.5 (never traded); '
                                       'threshold = coverage quantile over all validation rows (overall coverage)'}
    if a.blend_absret:
        res['blend_absret'] = {'side': a.blend_absret, 'alphas': list(BLEND_ALPHAS),
                               'models': 'the --model lgb recipe, seed 0, unweighted and |return| / mean weighted, same fit rows',
                               'selection': 'max over alpha of the smaller first-come win rate on SIDE in the two chronological '
                                            'halves of the gated validation rows at the operating coverage; ties to smaller alpha',
                               'forced_alpha': a.blend_force_alpha}
    if a.specialist and not {'ret_15m', 'volratio_15_240'} <= set(feats):
        raise SystemExit('--specialist needs the v0.1 columns ret_15m and volratio_15_240')
    if a.comp_gate is not None and 'rv_60m' not in feats:
        raise SystemExit('--comp-gate needs the v0.1 column rv_60m')
    if a.model == 'nystail':
        res['nystail'] = {'features': 'nystroem recipe (fit-only standardization, RBF Nystroem)',
                          'fit': 'full-batch L-BFGS: mean BCE + (1e-4 / 2) |beta|^2 + tail_aux x fixed-mass soft tail loss',
                          'tail': 'C = fit rows in the compression gate (all fit rows without it), mass c |C| at c = the '
                                  'operating coverage, membership sigmoid((|s| - q) / tau), tau = 0.1 x SD(stage-0 scores on C)',
                          'tail_aux': a.tail_aux}
    if a.mag_keep is not None and a.mag_model == 'har' and not {'rv_15m', 'rv_60m', 'rv_240m', 'rv_1440m'} <= set(feats):
        raise SystemExit('--mag-model har needs the v0.1 columns rv_15m, rv_60m, rv_240m, rv_1440m')
    if a.reflect:
        import re as _re
        neg = _re.compile(r'^(ret_\d+m|emadist_\d+m|stretch_\d+m|x_[A-Z]{6}_ret_\d+m|x_usd_weak_15m|bar_body|skew_240m)$')
        rng_ = _re.compile(r'^rangepos_\d+m$')
        keep = _re.compile(r'^(rv_\d+m|volratio_\d+_\d+|kurt_240m|bar_range|activity_\d+m|ticks_per_bar_15m|bar_gap_s|hour_sin|'
                           r'hour_cos|dow|x_[A-Z]{6}_stale_s)$')
        unmapped = [c for c in feats if not (neg.match(c) or rng_.match(c) or keep.match(c))]
        if unmapped:
            raise SystemExit(f'--reflect has no audited reflection map for {unmapped[:6]}')
        fitter.reflect_neg = np.array([i for i, c in enumerate(feats) if neg.match(c)], dtype=int)
        fitter.reflect_rng = np.array([i for i, c in enumerate(feats) if rng_.match(c)], dtype=int)
        res['reflect_map'] = {'negated': [feats[i] for i in fitter.reflect_neg], 'reflected_1_minus_x': [feats[i] for i in fitter.reflect_rng]}
    if a.monotone == 'catchup':
        if a.model != 'lgb' or not {'ret_15m', 'x_usd_weak_15m'} <= set(feats):
            raise SystemExit('--monotone catchup needs --model lgb and the v0.1 columns ret_15m, x_usd_weak_15m')
        usd_side = -1 if a.pair in ('USDJPY', 'USDCHF', 'USDCAD') else 1   # pair UP = USD weakness for XXX/USD
        fitter.mono = [-1 if c == 'ret_15m' else usd_side if c == 'x_usd_weak_15m' else 0 for c in feats]
        res['monotone'] = {'ret_15m': -1, 'x_usd_weak_15m': usd_side, 'others': 0}
    if a.drift_fe or a.signed_only or a.model == 'nysagree':
        import re as _re
        if (a.drift_fe or a.model == 'nysagree') and 'rv_60m' not in feats:
            raise SystemExit('--drift-fe / nysagree need the v0.1 column rv_60m for their strata')
        neg = _re.compile(r'^(ret_\d+m|emadist_\d+m|stretch_\d+m|x_[A-Z]{6}_ret_\d+m|x_usd_weak_15m|bar_body|skew_240m)$')
        rng_ = _re.compile(r'^rangepos_\d+m$')
        cols = [i for i, c in enumerate(feats) if neg.match(c) or rng_.match(c)]
        fitter.signed_cols = np.array(cols, dtype=int)
        fitter.signed_rng = np.array([k for k, i in enumerate(cols) if rng_.match(feats[i])], dtype=int)
        if a.drift_fe:
            res['drift_fe'] = {'strata': 'ISO week x America/New_York 3-hour bucket x fit-pool rv_60m tercile (one-hot x10, no intercept)',
                               'logit_signed_columns': [feats[i] for i in cols]}
            if a.model == 'lgb':
                res['drift_fe'] = {'strata': 'ISO week x America/New_York 3-hour bucket x fit-pool rv_60m tercile',
                                   'offsets': 'per-stratum ridge Newton (BCE + 0.5 alpha^2) as init_score; 2 alternations; '
                                              'scores = booster f alone, centred on the fit rows',
                                   'signed_columns': [feats[i] for i in cols]}
        if a.signed_only:
            res['signed_only'] = {'columns': [feats[i] for i in cols],
                                  'model': 'logit: sd-scaled, clipped, global intercept' if a.model == 'logit' else 'lgb booster'}
        if a.model == 'nysagree':
            res['nysagree'] = {'primary': 'nystroem recipe with the run weights',
                               'gate': 'keep iff drift-neutral logit (e67drfe) side == kernel side AND plain same-input logit '
                                       'side == opposite; others p = 0.5; overall coverage',
                               'signed_columns': [feats[i] for i in cols]}
    if a.train_label != 'sign':
        lab = pd.read_parquet(Path(label_dir) / f'{a.pair}.parquet').set_index('bar_start_ts')[f'{a.train_label}_{a.horizon}m']
        fitter.ytrain = lab.reindex(ts - 60).to_numpy(np.float64)      # decision = bar start + 60 s
        res['train_label'] = {'name': a.train_label, 'dir': label_dir,
                              'manifest_sha256': cd.sha256(Path(label_dir) / 'manifest.json'),
                              'aligned_finite': int(np.isfinite(fitter.ytrain).sum())}
        out.write_text(cd.result_json(res))
    res['compute'] = {'n_jobs': a.n_jobs, 'path_jobs': a.path_jobs}

    def path_split(testg):                      # training rows, test rows, rows usable as 30-minute training rows
        tin = np.zeros(len(ts), bool)
        purged = np.zeros(len(ts), bool)
        purged30 = np.zeros(len(ts), bool)
        for g in testg:
            lo, hi = groups[g]
            tin |= (ts >= lo) & (ts < hi)
            purged |= (ts >= lo - gap) & (ts < hi + gap)
            purged30 |= (ts >= lo - 1800) & (ts < hi + 1800)    # --multi-horizon: 30-minute labels need a 30-minute purge
        return np.where((~purged) & moved & tsess)[0], tin & sess & np.isfinite(fwd), ~purged30

    def forward_split(g):
        lo, hi = groups[g]
        return np.where((ts < lo - gap) & moved & tsess)[0], (ts >= lo) & (ts < hi) & sess & np.isfinite(fwd), ts < lo - 1800

    def say_path(fi, testg, row):
        print(f'  path {fi + 1:2d}/15 g{list(testg)} AUC={row["auc"]:.4f} n_fit={row["n_fit"]} iters={row["best_iter"]} | '
              f'cov{covs[0]}: {cd.fmt_side(row["bycov"][f"{covs[0]}"])} ({time.time() - t0:.0f}s)', flush=True)

    def say_forward(g, row):
        print(f'  forward g{g} AUC={row["auc"]:.4f} | cov{covs[0]}: {cd.fmt_side(row["bycov"][f"{covs[0]}"])} '
              f'({time.time() - t0:.0f}s)', flush=True)

    path_groups = list(combinations(range(cd.N_GROUPS), cd.K_TEST))[:max(a.max_paths, 0)]
    forward_groups = [] if a.no_forward else list(range(2, cd.N_GROUPS))
    pathinfo, forward = [], []
    if a.path_jobs == 1:
        rng = np.random.default_rng(13)
        for fi, testg in enumerate(path_groups):
            tr_idx, test_mask, fitter.ok30 = path_split(testg)
            row = fitter.split2(tr_idx, sess, test_mask, rng, list(testg))
            pathinfo.append(row)
            say_path(fi, testg, row)
        frng = np.random.default_rng(17)
        for g in forward_groups:
            tr_idx, test_mask, fitter.ok30 = forward_split(g)
            row = fitter.split2(tr_idx, sess, test_mask, frng, f'forward g{g}')
            forward.append(row)
            say_forward(g, row)
    else:
        # Each split is handed the generator state the sequential loop has when it reaches that split: the parent takes
        # the split's draws (draw_split) without fitting it, in the sequential order, on the same two streams.
        specs = []
        rng = np.random.default_rng(13)
        for testg in path_groups:
            tr_idx, test_mask, ok30 = path_split(testg)
            specs.append((list(testg), tr_idx, test_mask, ok30, copy.deepcopy(rng.bit_generator.state)))
            fitter.draw_split2(tr_idx, sess, rng)
        frng = np.random.default_rng(17)
        for g in forward_groups:
            tr_idx, test_mask, ok30 = forward_split(g)
            specs.append((f'forward g{g}', tr_idx, test_mask, ok30, copy.deepcopy(frng.bit_generator.state)))
            fitter.draw_split2(tr_idx, sess, frng)
        _PJ.update(fitter=fitter, val_pool=sess)
        done = []
        for i, got in in_order(run_forked(_path_job, specs, a.path_jobs)):
            done.append(got)
            if i < len(path_groups):
                say_path(i, path_groups[i], got[0])
            else:
                say_forward(forward_groups[i - len(path_groups)], got[0])
        for row, gain, dumps in done:               # merged in the sequential order
            fitter.gain += gain
            fitter.dumps.extend(dumps)
        pathinfo = [d[0] for d in done[:len(path_groups)]]
        forward = [d[0] for d in done[len(path_groups):]]
    cd.aggregate(res, pathinfo, forward, covs, data_uprate, fitter.gain, feats)
    if a.max_paths < 15 or a.null_shift_weeks:   # smoke runs and null replays never certify
        for k in res['bycov']:
            for s in cd.SIDES:
                res['bycov'][k]['verdict'][s]['CERTIFIED'] = False
        tag = f'SMOKE ({a.max_paths} paths)' if a.max_paths < 15 else f'NULL REPLAY (labels shifted {a.null_shift_weeks} weeks)'
        res['verdict'] = {s: tag for s in ('UP', 'DOWN')}
        for s in res.get('verdict_goal', {}):
            res['verdict_goal'][s]['CERTIFIED'] = False
    res['seconds'] = round(time.time() - t0, 1)
    res['run_status'] = 'complete'
    out.write_text(cd.result_json(res))
    if a.dump_preds:
        d = fitter.dumps
        np.savez_compressed(a.dump_preds, labels=np.array([x['label'] for x in d]),
                            thr=np.array([json.dumps(x['thr']) for x in d]),
                            **{f'{k}_{i}': x[k] for i, x in enumerate(d) for k in ('ts', 'p', 'fwd', 'keep')})
    cd.print_summary(res, covs, f'{a.pair} {a.horizon}m x', t0, out)


if __name__ == '__main__':
    main()
