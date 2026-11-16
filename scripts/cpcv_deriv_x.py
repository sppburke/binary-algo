#!/usr/bin/env python3
"""Variant harness on top of tools/cpcv_deriv.py: same CPCV, scoring, validity rule and falsifier, plus
knobs the base mirror does not expose.

    cpcv_deriv_x.py --pair P --horizon 15 [--session ny] [--train-session all] [--covs 0.03,0.02,0.05]
                    [--nseed 1] [--features all|noclock|own|own_noclock] [--feature-list cols.json]
                    [--num-leaves 255] [--min-child 400] [--lr 0.02] [--n-estimators 800] [--colsample 0.5]
                    [--reg-lambda 20] [--dataset DIR] [--out result.json] [--tag TEXT] [--no-forward] [--max-paths N]

--train-session widens (or changes) the rows the model is FIT on; groups, validation rows (early stopping and
the threshold quantile), test paths and every reported number stay on the evaluation --session. Purge and
embargo apply to all rows. --val-mode block replaces the repository's random 15% validation rows (label-
overlapping neighbours of fit rows: overconfident thresholds, early stopping that never fires) with the last
15% of the training rows in time, purged one horizon from the fit rows.
Defaults reproduce cpcv_deriv.py exactly (same rng stream and parameters; checked on E1 USDJPY path 1).
"""
import argparse
import json
import sys
import time
from collections import deque
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpcv_deriv as cd  # noqa: E402

X_VERSION = 'cpcv_deriv_x/0.8'    # 0.8: discovery round 3 - --signed-only (logit / lgb on e67drfe's signed coordinates; the
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


def aci_side(pr, fwd, ts, thr0, side, gap, gamma, target):
    """Adaptive-conformal-style confidence gate on one test path, chronological and causal: start at the validation
    threshold; when a taken trade SETTLES (decision + H + 2 s, before any later decision uses it) move the threshold by
    gamma * (loss - (1 - target)) - up after a loss, down after a win, floor 0. Side filter, first-come de-overlap (gap
    H), win = right sign AND moved (ties lose), exactly as cpcv_deriv.side_wr otherwise."""
    conf = np.abs(pr - 0.5)
    want = (pr > 0.5) if side == 'UP' else (pr < 0.5) if side == 'DOWN' else np.ones(len(pr), bool)
    thr, block, taken, wins, pend = thr0, -1, 0, 0, deque()
    for i in range(len(ts)):
        t = ts[i]
        while pend and pend[0][0] <= t:
            thr = max(0.0, thr + gamma * (pend.popleft()[1] - (1.0 - target)))
        if t < block or not want[i] or not np.isfinite(fwd[i]) or conf[i] < thr:
            continue
        win = ((pr[i] > 0.5) == (fwd[i] > 0)) and fwd[i] != 0
        taken += 1
        wins += int(win)
        block = int(t) + gap
        pend.append((int(t) + gap + 2, 0.0 if win else 1.0))
    return taken, (wins / taken if taken else float('nan'))


def aci_score_row(row, ptst, ftst, ttst, vconf, covs, gap, gamma, target):
    for s in ('UP', 'DOWN'):
        row['always'][s] = list(cd.always(ftst, ttst, s, gap))
    for c in covs:
        thr = float(np.quantile(vconf, 1 - c))
        row['bycov'][f'{c}'] = {'thr': thr}
        for s in cd.SIDES:
            row['bycov'][f'{c}'][s] = list(aci_side(ptst, ftst, ttst, thr, s, gap, gamma, target))
    return row


def sigm(f):
    return 1.0 / (1.0 + np.exp(-f))


class XFitter(cd.Fitter):
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

    def fit_one(self, sd, Xf, yf, Xv, yv, Xt):
        """One seed of the selected model class: (validation probabilities, test probabilities, iterations)."""
        a = self.a
        pv, pt, its = [], [], []
        # sklearn models get the training weights rescaled to mean 1 (per-row loss scale unchanged against the penalty)
        skw = {} if self.wfit is None else {'sample_weight': self.wfit / (self.wfit.mean() + 1e-18)}
        if getattr(a, 'drift_fe', False):          # (getattr: holdout_eval passes a reduced namespace)
            return self.fit_drift_fe(sd, Xf, yf, Xv, yv, Xt, skw)
        if a.model == 'nysagree':
            return self.fit_nysagree(sd, Xf, yf, Xv, yv, Xt, skw)
        if getattr(a, 'signed_only', False):
            if a.model == 'logit':                  # e74logitsig: e67drfe's inputs and scaling, global intercept, no strata
                fv, ft, it = self.signed_logit(Xf, yf, Xv, Xt, skw, fe=False)
                return sigm(fv), sigm(ft), it
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
            m.fit(Xf, yf, eval_set=[(Xv, yv)], eval_metric='auc',
                  callbacks=[cd.lgb.early_stopping(80, verbose=False), cd.lgb.log_evaluation(0)], **kw)
            sig = lambda Z: 1.0 / (1.0 + np.exp(-m.booster_.predict(Z, raw_score=True, num_iteration=m.best_iteration_ or None)))
            pv.append(sig(Xv)); pt.append(sig(Xt))
            self.gain += m.booster_.feature_importance(importance_type='gain')
            its.append(int(m.best_iteration_ or 0))
        if a.model == 'ens3':                       # SWEEP_MATRIX A1: + XGBoost + CatBoost (scripts/m15_production.py recipe)
            import xgboost as xgb
            from catboost import CatBoostClassifier
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
            c.fit(fill(Xf), yf, eval_set=(fill(Xv), yv))
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
            return sigm(fv), sigm(ft), it
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
                if k == 0:
                    alpha = self.profile_alpha(inv, yb, raw(F), w, alpha)
                    info['alpha1_sd'], info['iter1'] = float(alpha.std()), it
            c = float(np.mean(raw(F)))
            info.update(iter2=it, f_centre=c)
            self.lgbfe_info = info
            self.add_gain(m.booster_.feature_importance(importance_type='gain'))
            return sigm(raw(V) - c), sigm(raw(T) - c), it
        from sklearn.kernel_approximation import Nystroem
        from sklearn.linear_model import SGDClassifier
        F, V, T = standardize(Xf, Xv, Xt)
        ny = Nystroem(kernel='rbf', gamma=a.nys_gamma_mult / F.shape[1], n_components=a.nys_components, random_state=sd)
        Fk = ny.fit_transform(F)
        m = SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=30, tol=1e-4, random_state=sd, fit_intercept=False)
        m.fit(sp.hstack([sp.csr_matrix(Fk), D]).tocsr(), yf, **skw)
        w = m.coef_[0, :Fk.shape[1]]
        return sigm(ny.transform(V) @ w), sigm(ny.transform(T) @ w), int(m.n_iter_)

    def fit_nysagree(self, sd, Xf, yf, Xv, yv, Xt, skw):
        """Discovery D3-04 AGREEMENT GATE (0.8): kernel proposals (the nystroem recipe with the run's training weights)
        are kept only where the drift-neutral logistic (e67drfe, unweighted) takes the kernel's side and its same-input
        plain twin (global intercept, unweighted) takes the other side; every other row gets probability 0.5, so the
        coverage quantile is over ALL validation rows (overall coverage). Exact zero scores abstain."""
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
        gv, gt = gate(kv, nv, rv), gate(kt, nt, rt)
        self.agree_info = {'val_share': float(gv.mean()), 'test_share': float(gt.mean()),
                           'kernel_vs_neutral_agree_val': float((np.sign(kv - 0.5) == np.sign(nv)).mean()),
                           'neutral_vs_plain_agree_val': float((np.sign(nv) == np.sign(rv)).mean())}
        return np.where(gv, kv, 0.5), np.where(gt, kt, 0.5), int(m.n_iter_)

    def split2(self, tr_idx, val_pool, test_mask, rng, label):
        """Fit on tr_idx; validation = random VAL_FRAC of the evaluation-session rows among tr_idx (val_pool mask)."""
        X, fwd, ts, a, gap = self.X, self.fwd, self.ts, self.a, self.gap
        if a.val_mode == 'block':
            # validation = the chronologically last VAL_FRAC of the evaluation-session training rows; fit rows end one
            # horizon before it (purge), so early stopping and the threshold see genuinely unseen, later data
            cand = tr_idx[val_pool[tr_idx]]
            val_idx = cand[-max(1, int(round(len(cand) * cd.VAL_FRAC))):]
            fit_pool = tr_idx[ts[tr_idx] < ts[val_idx[0]] - gap]
        else:
            r = rng.random(len(tr_idx))
            valsel = (r < cd.VAL_FRAC) & val_pool[tr_idx]
            val_idx = tr_idx[valsel]
            fit_pool = tr_idx[~valsel]
        if self.ytrain is not None:                 # alternative TRAINING target (validation keeps the settlement sign)
            fit_pool = fit_pool[np.isfinite(self.ytrain[fit_pool])]
        if len(fit_pool) > cd.SUB_FIT:
            fit_pool = rng.choice(fit_pool, cd.SUB_FIT, replace=False)     # sampled order kept, as the reference
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
            wall = self.wfit
            for k, (fm, vm, tm) in enumerate(zip(setups(Xf), setups(Xv), setups(Xt))):
                pv_s, pt_s = np.zeros(int(vm.sum())), np.zeros(int(tm.sum()))
                self.wfit = None if wall is None else wall[fm]
                for sd in range(a.nseed):
                    pv, pt, it = self.fit_one(sd, Xf[fm], ytr[fm], Xv[vm], yv_fit[vm], Xt[tm])
                    pv_s += pv
                    pt_s += pt
                    iters.append(it)
                clip = np.maximum if k == 0 else np.minimum          # k 0 = UP specialist, 1 = DOWN specialist
                pval[vm] = clip(pv_s / a.nseed, 0.5)
                ptst[tm] = clip(pt_s / a.nseed, 0.5)
            self.spec_info = {'compression_thr': comp_thr, 'fit_up': int(setups(Xf)[0].sum()), 'fit_down': int(setups(Xf)[1].sum())}
        elif a.boot_gate:
            # literature rank 16, CALIBRATION-UNCERTAINTY GATE: boot_gate bootstrap replicas of the fit pool (own generator per
            # path and replica, so the shared validation/sub-fit stream is unchanged); the score keeps the replica-mean side
            # and ranks by the lower bound |mean - 0.5| - boot_z * sd (floored at 0), for validation and test alike.
            PV, PT = [], []
            wall = self.wfit
            for b in range(a.boot_gate):
                bi = np.random.default_rng([1009, b] + [ord(c) for c in str(label)]).integers(0, len(Xf), len(Xf))
                self.wfit = None if wall is None else wall[bi]
                pv, pt, it = self.fit_one(b, Xf[bi], ytr[bi], Xv, yv_fit, Xt)
                PV.append(pv); PT.append(pt); iters.append(it)
            self.wfit = wall
            lcb = lambda M: 0.5 + np.sign(M.mean(0) - 0.5) * np.maximum(np.abs(M.mean(0) - 0.5) - a.boot_z * M.std(0, ddof=1), 0.0)
            PV, PT = np.array(PV), np.array(PT)
            pval, ptst = lcb(PV), lcb(PT)
            self.boot_info = {'replicas': a.boot_gate, 'z': a.boot_z, 'sd_test_median': float(np.median(PT.std(0, ddof=1))),
                              'zero_lcb_share_test': float((ptst == 0.5).mean())}
        else:
            for sd in range(a.nseed):
                pv, pt, it = self.fit_one(sd, Xf, ytr, Xv, yv_fit, Xt)
                pval += pv
                ptst += pt
                iters.append(it)
            pval /= a.nseed
            ptst /= a.nseed
        if gain_mh is not None:                     # fold the indicator column's gain out of the per-feature report
            gain_mh += self.gain[:-1]
            self.gain = gain_mh
        vconf = np.abs(pval - 0.5)
        ftst = fwd[test_mask]
        ttst = ts[test_mask]
        mv = ftst != 0
        auc = float(cd.roc_auc_score((ftst[mv] > 0).astype(int), ptst[mv])) if mv.sum() > 20 else float('nan')
        uprate = float((ftst[mv] > 0).mean()) if mv.sum() else float('nan')
        row = {'test': label, 'n_fit': int(len(fit_pool)), 'n_val': int(len(val_idx)), 'n_test': int(test_mask.sum()),
               'best_iter': iters, 'auc': auc, 'up_rate': uprate, 'always': {}, 'bycov': {}}
        if a.specialist:
            row['specialist'] = self.spec_info              # AUC above is over every test row (non-setup rows at 0.5)
        if a.boot_gate:
            row['boot_gate'] = self.boot_info               # AUC above is over the lower-bound scores
        if self.stack_w:
            row['stack_weights_mean'] = np.mean(self.stack_w, axis=0).round(4).tolist()   # order lgb, logit, forest, nystroem
        if a.multi_horizon:
            row['multi_horizon'] = self.mh_info
        if a.drift_fe and a.model == 'lgb':
            row['lgb_drift_fe'] = self.lgbfe_info
        if a.model == 'nysagree':
            row['agree_gate'] = self.agree_info             # AUC above is over every test row (gated-out rows at 0.5)
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
                mm.fit(X[fit_pool], (np.abs(fwd[fit_pool]) >= med).astype(int))
                mval = mm.predict_proba(X[val_idx])[:, 1]
                mtst = mm.predict_proba(X[test_mask])[:, 1]
            gthr = float(np.quantile(mval, 1 - a.mag_keep))
            vgate = mval >= gthr
            tgate = mtst >= gthr
            big = (np.abs(ftst) >= med).astype(int)
            row['mag_gate'] = {'keep': a.mag_keep, 'median_abs_ret': med, 'gate_thr': gthr,
                               'test_pass_rate': float(tgate.mean()),
                               'mag_auc_test': float(cd.roc_auc_score(big, mtst)) if 0 < big.sum() < len(big) else None,
                               'tie_rate_pass': float((ftst[tgate] == 0).mean()) if tgate.any() else None,
                               'tie_rate_fail': float((ftst[~tgate] == 0).mean()) if (~tgate).any() else None}
            ptst = np.where(tgate, ptst, 0.5)        # gated-out rows get zero confidence: never selected
        if a.meta_keep is not None:
            # SWEEP_MATRIX A4 META-LABELER (scripts/m15_meta.py idea on the Deriv feed). Out-of-fold primary predictions on
            # the fit pool come from two chronological halves (the first ends one horizon before the split, so no label
            # of one half overlaps the other); a meta model learns P(primary side correct) on moved fit rows from
            # [features, primary confidence, primary side]; validation and test use the full-pool primary (more confident
            # than the half-pool OOF primary: a declared approximation); keep the top meta_keep share by meta score
            # (validation quantile); the direction threshold is the coverage quantile among validation rows in the gate.
            # (0.5: the half-pool primaries are fit by fit_one - the selected model class and recipe, seed 0, early stopping
            # on the validation rows - so a kernel or pooled primary gets a meta model learned on its own errors)
            tf_ = ts[fit_pool]
            mid = float(np.median(tf_))
            halves = (tf_ < mid - gap, tf_ >= mid)
            oof = np.full(len(fit_pool), np.nan)
            wall, gain0, nstack = self.wfit, self.gain.copy(), len(self.stack_w)
            for trm, tem in ((halves[0], halves[1]), (halves[1], halves[0])):
                self.wfit = None if wall is None else wall[trm]
                _, oof[tem], _ = self.fit_one(0, Xf[trm], ytr[trm], Xv, yv_fit, Xf[tem])
            self.wfit, self.gain = wall, gain0        # the reported gain and stack weights stay the primary's
            del self.stack_w[nstack:]
            fwf = fwd[fit_pool]
            ok = np.isfinite(oof) & (fwf != 0)
            Zf = np.column_stack([Xf, np.abs(oof - 0.5), (oof > 0.5).astype(np.float32)])
            mm = cd.lgb.LGBMClassifier(objective='binary', learning_rate=0.02, num_leaves=63, min_child_samples=400,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=20,
                                       n_estimators=400, n_jobs=a.n_jobs, verbosity=-1, random_state=0)
            mm.fit(Zf[ok], ((oof[ok] > 0.5) == (fwf[ok] > 0)).astype(int))
            mval = mm.predict_proba(np.column_stack([Xv, np.abs(pval - 0.5), (pval > 0.5).astype(np.float32)]))[:, 1]
            mtst = mm.predict_proba(np.column_stack([Xt, np.abs(ptst - 0.5), (ptst > 0.5).astype(np.float32)]))[:, 1]
            gthr = float(np.quantile(mval, 1 - a.meta_keep))
            vgate &= mval >= gthr
            tg = mtst >= gthr
            corr_t = ((ptst > 0.5) == (ftst > 0)) & (ftst != 0)
            row['meta_gate'] = {'keep': a.meta_keep, 'gate_thr': gthr, 'oof_rows': int(ok.sum()),
                                'oof_auc': float(cd.roc_auc_score((fwf[ok] > 0).astype(int), oof[ok])) if ok.sum() > 20 else None,
                                'test_pass_rate': float(tg.mean()),
                                'meta_auc_test': float(cd.roc_auc_score(corr_t.astype(int), mtst)) if 0 < corr_t.sum() < len(corr_t) else None}
            ptst = np.where(tg, ptst, 0.5)
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
        if a.dump_preds:                             # analysis only: keep the scored test predictions of every path
            self.dumps.append({'label': str(label), 'ts': ttst.copy(), 'p': ptst.copy(), 'fwd': ftst.copy(),
                               'thr': {f'{c}': float(np.quantile(vconf[vgate], 1 - c)) for c in self.covs}})
        if a.aci_gamma is not None:
            return aci_score_row(row, ptst, ftst, ttst, vconf[vgate], self.covs, gap, a.aci_gamma, a.aci_target)
        return cd.score_row(row, ptst, ftst, ttst, vconf[vgate], self.covs, gap)


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
                                                       'stack', 'lgbord4', 'nysagree'),
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
                    help='adaptive-conformal confidence gate step (e.g. 0.002): threshold += gamma * (loss - (1 - target)) '
                         'as each taken trade settles')
    ap.add_argument('--aci-target', type=float, default=0.56)
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
    ap.add_argument('--signed-only', action='store_true',
                    help='0.8 same-input controls: logit (e67drfe inputs and scaling, global intercept, no strata) or lgb '
                         '(the usual booster) on the signed coordinates only')
    ap.add_argument('--families', default=None,
                    help='SWEEP_MATRIX family blocks to keep (comma list of tf,sg,kf,rm,fd,cx,tk,pa,cm, or all / none); every '
                         'other block column is dropped from the --features selection')
    ap.add_argument('--include-blackout', action='store_true', help='keep Deriv purchase-blackout decisions (not Deriv-faithful)')
    ap.add_argument('--mag-keep', type=float, default=None,
                    help='magnitude gate: keep only the top share (e.g. 0.5) of rows by predicted big-move probability')
    ap.add_argument('--dataset', default=str(cd.LAB / 'cache' / 'dataset_dev_v0.1'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--n-jobs', type=int, default=16)
    ap.add_argument('--no-forward', action='store_true')
    ap.add_argument('--max-paths', type=int, default=15)
    a = ap.parse_args()
    a.train_session = a.train_session or a.session
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
    if a.multi_horizon and (a.horizon != 15 or a.reflect or a.meta_keep is not None or a.specialist or a.boot_gate
                            or a.model == 'lgbord4' or a.comp_gate is not None or a.weight != 'none'
                            or a.train_label != 'sign' or a.null_shift_weeks):
        raise SystemExit('--multi-horizon runs at --horizon 15 with no reflect / meta / specialist / boot / lgbord4 / comp '
                         'gate / weights / train label / null shift')
    covs = [float(c) for c in a.covs.split(',')]
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
    if a.null_shift_weeks:
        # SIGN-NULL REPLAY (literature: full-search null replay / reality check): every decision gets the settled label of
        # the decision exactly W weeks later (circular over the dataset span; same weekday and UTC time of day), which keeps
        # the label's drift and intraday structure and breaks any feature-label link. Missing matches become NaN.
        span = int(ts.max() - ts.min()) + 60
        tgt = ts.min() + ((ts - ts.min() + a.null_shift_weeks * 7 * 86400) % span)
        fwd = pd.Series(fwd, index=ts).reindex(tgt).to_numpy(np.float64)
    sess = session(ts, a.session)
    tsess = session(ts, a.train_session)
    if not a.include_blackout:                   # Deriv purchase blackout: untradeable decisions out of every role
        tr_ok = cd.tradeable(ts, a.horizon)
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
                                        'signed_only')}
    res = {
        'harness': X_VERSION, 'base_harness': cd.HARNESS_VERSION, 'tag': a.tag,
        'model': f'LightGBM @{a.horizon}m on the Deriv feed, eval session {a.session}, fit session {a.train_session}, knobs {knobs}',
        'key': f'{a.pair}.{a.horizon}m', 'data': 'Deriv 1-second mid ticks only (/mnt/data/deriv), no Dukascopy',
        'dataset': {'dir': a.dataset, 'manifest_sha256': cd.sha256(manifest)},
        'window': {'first_decision': day(ts[0]), 'last_decision': day(ts[-1])},
        'breakeven': cd.BE, 'session': a.session, 'tradeable_only': not a.include_blackout,
        'blackout_rule': None if a.include_blackout else cd.BLACKOUT_RULE,
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
    out.write_text(json.dumps(res, indent=1))
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
    if a.specialist and not {'ret_15m', 'volratio_15_240'} <= set(feats):
        raise SystemExit('--specialist needs the v0.1 columns ret_15m and volratio_15_240')
    if a.comp_gate is not None and 'rv_60m' not in feats:
        raise SystemExit('--comp-gate needs the v0.1 column rv_60m')
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
        out.write_text(json.dumps(res, indent=1))
    rng = np.random.default_rng(13)
    pathinfo = []
    for fi, testg in enumerate(combinations(range(cd.N_GROUPS), cd.K_TEST)):
        if fi >= a.max_paths:
            break
        tin = np.zeros(len(ts), bool)
        purged = np.zeros(len(ts), bool)
        purged30 = np.zeros(len(ts), bool)
        for g in testg:
            lo, hi = groups[g]
            tin |= (ts >= lo) & (ts < hi)
            purged |= (ts >= lo - gap) & (ts < hi + gap)
            purged30 |= (ts >= lo - 1800) & (ts < hi + 1800)    # --multi-horizon: 30-minute labels need a 30-minute purge
        fitter.ok30 = ~purged30
        row = fitter.split2(np.where((~purged) & moved & tsess)[0], sess, tin & sess & np.isfinite(fwd), rng, list(testg))
        pathinfo.append(row)
        print(f'  path {fi + 1:2d}/15 g{list(testg)} AUC={row["auc"]:.4f} n_fit={row["n_fit"]} iters={row["best_iter"]} | '
              f'cov{covs[0]}: {cd.fmt_side(row["bycov"][f"{covs[0]}"])} ({time.time() - t0:.0f}s)', flush=True)
    forward = []
    if not a.no_forward:
        frng = np.random.default_rng(17)
        for g in range(2, cd.N_GROUPS):
            lo, hi = groups[g]
            fitter.ok30 = ts < lo - 1800
            row = fitter.split2(np.where((ts < lo - gap) & moved & tsess)[0], sess, (ts >= lo) & (ts < hi) & sess & np.isfinite(fwd),
                                frng, f'forward g{g}')
            forward.append(row)
            print(f'  forward g{g} AUC={row["auc"]:.4f} | cov{covs[0]}: {cd.fmt_side(row["bycov"][f"{covs[0]}"])} '
                  f'({time.time() - t0:.0f}s)', flush=True)
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
    out.write_text(json.dumps(res, indent=1))
    if a.dump_preds:
        d = fitter.dumps
        np.savez_compressed(a.dump_preds, labels=np.array([x['label'] for x in d]),
                            thr=np.array([json.dumps(x['thr']) for x in d]),
                            **{f'{k}_{i}': x[k] for i, x in enumerate(d) for k in ('ts', 'p', 'fwd')})
    cd.print_summary(res, covs, f'{a.pair} {a.horizon}m x', t0, out)


if __name__ == '__main__':
    main()
