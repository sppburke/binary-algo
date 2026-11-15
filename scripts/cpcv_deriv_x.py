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

X_VERSION = 'cpcv_deriv_x/0.4'   # 0.4: --families (SWEEP_MATRIX family blocks), --train-label resid, --model; 0.3: shared
#                                  unrounded aggregation (cpcv_deriv 0.4); 0.2: blackout excluded
# Column prefixes of the SWEEP_MATRIX family blocks (tools/deriv_features_family.py, merged into dataset_dev_v0.3)
FAMILY_PREFIXES = ('tf_', 'sg_', 'kf_', 'rm_', 'fd_', 'cx_', 'tk_', 'pa_', 'cm_')   # cm_: CCM block (dataset_dev_v0.4)
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


class XFitter(cd.Fitter):
    def fit_one(self, sd, Xf, yf, Xv, yv, Xt):
        """One seed of the selected model class: (validation probabilities, test probabilities, iterations)."""
        a = self.a
        pv, pt, its = [], [], []
        if a.model in ('lgb', 'ens3'):
            m = make_model(a, sd)
            kw = {} if self.wfit is None else {'sample_weight': self.wfit}
            if a.no_early_stop:                     # validation rows only set the threshold
                m.fit(Xf, yf, **kw)
            else:
                m.fit(Xf, yf, eval_set=[(Xv, yv)], eval_metric='auc',
                      callbacks=[cd.lgb.early_stopping(80, verbose=False), cd.lgb.log_evaluation(0)], **kw)
            pv.append(m.predict_proba(Xv)[:, 1]); pt.append(m.predict_proba(Xt)[:, 1])
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
        if a.model == 'logit':                      # low-capacity linear model (L2, fixed C chosen before any run)
            from sklearn.linear_model import LogisticRegression
            F, V, T = standardize(Xf, Xv, Xt)
            m = LogisticRegression(C=a.logit_c, max_iter=3000)
            m.fit(F, yf)
            pv.append(m.predict_proba(V)[:, 1]); pt.append(m.predict_proba(T)[:, 1])
            its.append(int(np.max(m.n_iter_)))
        if a.model == 'nystroem':                   # SWEEP_MATRIX B6: RBF kernel (Nystroem features) + logistic SGD
            from sklearn.kernel_approximation import Nystroem
            from sklearn.linear_model import SGDClassifier
            F, V, T = standardize(Xf, Xv, Xt)
            ny = Nystroem(kernel='rbf', gamma=1.0 / F.shape[1], n_components=500, random_state=sd)
            Fk = ny.fit_transform(F)
            m = SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=30, tol=1e-4, random_state=sd)
            m.fit(Fk, yf)
            pv.append(m.predict_proba(ny.transform(V))[:, 1]); pt.append(m.predict_proba(ny.transform(T))[:, 1])
            its.append(int(m.n_iter_))
        return np.mean(pv, axis=0), np.mean(pt, axis=0), (its[0] if len(its) == 1 else its)

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
        pval = np.zeros(len(val_idx))
        ptst = np.zeros(int(test_mask.sum()))
        iters = []
        Xf, Xv, Xt = X[fit_pool], X[val_idx], X[test_mask]
        self.wfit = None
        if a.weight == 'absret':                    # |return|-weighted training (GMADL-style emphasis on large moves)
            w = np.abs(fwd[fit_pool])
            self.wfit = w / (w.mean() + 1e-18)
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
                    pv, pt, it = self.fit_one(sd, Xf[fm], ytr[fm], Xv[vm], yval[vm], Xt[tm])
                    pv_s += pv
                    pt_s += pt
                    iters.append(it)
                clip = np.maximum if k == 0 else np.minimum          # k 0 = UP specialist, 1 = DOWN specialist
                pval[vm] = clip(pv_s / a.nseed, 0.5)
                ptst[tm] = clip(pt_s / a.nseed, 0.5)
            self.spec_info = {'compression_thr': comp_thr, 'fit_up': int(setups(Xf)[0].sum()), 'fit_down': int(setups(Xf)[1].sum())}
        else:
            for sd in range(a.nseed):
                pv, pt, it = self.fit_one(sd, Xf, ytr, Xv, yval, Xt)
                pval += pv
                ptst += pt
                iters.append(it)
            pval /= a.nseed
            ptst /= a.nseed
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
        vgate = np.ones(len(val_idx), bool)
        if a.mag_keep is not None:
            # MAGNITUDE GATE (sign-invariant): a second model ranks "big move" (|ret| >= the fit pool's median |ret|);
            # keep only the top mag_keep share (threshold = validation quantile); the direction threshold is then the
            # coverage quantile among validation rows that pass the gate, so coverage is relative to gated rows.
            med = float(np.median(np.abs(fwd[fit_pool])))
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
            tf_ = ts[fit_pool]
            mid = float(np.median(tf_))
            halves = (tf_ < mid - gap, tf_ >= mid)
            oof = np.full(len(fit_pool), np.nan)
            for trm, tem in ((halves[0], halves[1]), (halves[1], halves[0])):
                m1 = make_model(a, 0)
                m1.fit(Xf[trm], ytr[trm])
                oof[tem] = m1.predict_proba(Xf[tem])[:, 1]
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
    ap.add_argument('--model', default='lgb', choices=('lgb', 'ens3', 'logit', 'nystroem'),
                    help='lgb (default, the mirror); ens3 = LightGBM + XGBoost + CatBoost mean (SWEEP_MATRIX A1; XGB/Cat use '
                         '--n-estimators/--lr); logit = L2 logistic regression on fit-pool-standardized features; '
                         'nystroem = RBF-kernel features + logistic SGD (SWEEP_MATRIX B6)')
    ap.add_argument('--cat-depth', type=int, default=8)
    ap.add_argument('--comp-gate', type=float, default=None,
                    help='SWEEP_MATRIX A2 compression gate: trade only rows with rv_60m <= its fit-pool quantile q (e.g. 0.33)')
    ap.add_argument('--meta-keep', type=float, default=None,
                    help='SWEEP_MATRIX A4 meta-labeler gate: keep the top share (e.g. 0.5) by P(primary side correct)')
    ap.add_argument('--weight', default='none', choices=('none', 'absret'),
                    help='absret: LightGBM sample weights |settled return| / mean (fit rows only)')
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
    ap.add_argument('--train-label', default='sign', choices=('sign', 'tb1', 'tb2', 'resid'),
                    help='fit target: Deriv settlement sign (default), triple-barrier first touch k=1/2, or the USD-basket '
                         'residualized sign (SWEEP_MATRIX F4); validation, thresholds and scoring always use the settlement sign')
    ap.add_argument('--label-dir', default=None, help='default: cache/labels_tb_dev_v0.1 (tb1/tb2), cache/labels_resid_dev_v0.1 (resid)')
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
    label_dir = a.label_dir or str(cd.LAB / 'cache' / ('labels_resid_dev_v0.1' if a.train_label == 'resid' else 'labels_tb_dev_v0.1'))
    X, fwd, ts, feats = cd.load(a.dataset, a.pair, a.horizon, a.features, 1, listed)
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
                                        'aci_gamma', 'aci_target', 'meta_keep', 'specialist_regime')}
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
    if a.specialist and not {'ret_15m', 'volratio_15_240'} <= set(feats):
        raise SystemExit('--specialist needs the v0.1 columns ret_15m and volratio_15_240')
    if a.comp_gate is not None and 'rv_60m' not in feats:
        raise SystemExit('--comp-gate needs the v0.1 column rv_60m')
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
        for g in testg:
            lo, hi = groups[g]
            tin |= (ts >= lo) & (ts < hi)
            purged |= (ts >= lo - gap) & (ts < hi + gap)
        row = fitter.split2(np.where((~purged) & moved & tsess)[0], sess, tin & sess & np.isfinite(fwd), rng, list(testg))
        pathinfo.append(row)
        print(f'  path {fi + 1:2d}/15 g{list(testg)} AUC={row["auc"]:.4f} n_fit={row["n_fit"]} iters={row["best_iter"]} | '
              f'cov{covs[0]}: {cd.fmt_side(row["bycov"][f"{covs[0]}"])} ({time.time() - t0:.0f}s)', flush=True)
    forward = []
    if not a.no_forward:
        frng = np.random.default_rng(17)
        for g in range(2, cd.N_GROUPS):
            lo, hi = groups[g]
            row = fitter.split2(np.where((ts < lo - gap) & moved & tsess)[0], sess, (ts >= lo) & (ts < hi) & sess & np.isfinite(fwd),
                                frng, f'forward g{g}')
            forward.append(row)
            print(f'  forward g{g} AUC={row["auc"]:.4f} | cov{covs[0]}: {cd.fmt_side(row["bycov"][f"{covs[0]}"])} '
                  f'({time.time() - t0:.0f}s)', flush=True)
    cd.aggregate(res, pathinfo, forward, covs, data_uprate, fitter.gain, feats)
    if a.max_paths < 15:                         # smoke runs never certify
        for k in res['bycov']:
            for s in cd.SIDES:
                res['bycov'][k]['verdict'][s]['CERTIFIED'] = False
        res['verdict'] = {s: f'SMOKE ({a.max_paths} paths)' for s in ('UP', 'DOWN')}
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
