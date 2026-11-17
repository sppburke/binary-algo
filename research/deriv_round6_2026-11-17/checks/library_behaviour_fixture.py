#!/usr/bin/env python3
"""Round 6, claims to verify (A-032, A-075): library behaviour on a synthetic fixture. No market data is read.

From the repository root (it fits small models on synthetic data; it ran as lab test x-r6-libcheck):

    ~/binary-algo-venv/bin/python -I research/deriv_round6_2026-11-17/checks/library_behaviour_fixture.py

1. CatBoost Ordered boosting with has_time=True: does the model use the row order of the input as its one permutation?
   a. resolved parameters (get_all_params);
   b. with every other source of randomness switched off (no bootstrap, rsm 1, random_strength 0), two seeds must
      give identical predictions when has_time=True and different ones when has_time=False (random permutations);
   c. reversing the row order must change an Ordered has_time model and must not change a Plain model beyond
      floating-point noise.
2. Truncation after early stopping, as the ens3 branch of cpcv_deriv_x.py calls the libraries:
   a. XGBoost sklearn API: predict_proba after early stopping equals the prediction limited to best_iteration + 1 trees;
   b. CatBoost with an eval_set: use_best_model resolved value and tree_count_ against get_best_iteration() + 1.
"""
import numpy as np
import catboost
import xgboost as xgb
from catboost import CatBoostClassifier

rng = np.random.default_rng(20261117)
n, k = 4000, 6
X = rng.normal(size=(n, k))
drift = np.linspace(-1.0, 1.0, n)                       # the relation changes along the row order ("time")
logit = 0.8 * X[:, 0] * drift + 0.5 * X[:, 1] + 0.3 * rng.normal(size=n)
y = (logit + rng.normal(size=n) > 0).astype(int)
Xf, yf, Xv, yv, Xt = X[:2800], y[:2800], X[2800:3400], y[2800:3400], X[3400:]
print('versions: catboost', catboost.__version__, '| xgboost', xgb.__version__)


def cb(boosting, has_time, seed):
    m = CatBoostClassifier(boosting_type=boosting, has_time=has_time, iterations=60, depth=4, learning_rate=0.03,
                           l2_leaf_reg=20, random_seed=seed, bootstrap_type='No', rsm=1.0, random_strength=0.0,
                           thread_count=2, verbose=False, allow_writing_files=False)
    return m


print('\n1a. resolved parameters')
for boosting, has_time in (('Ordered', True), ('Ordered', False), ('Plain', True)):
    m = cb(boosting, has_time, 0).fit(Xf, yf)
    p = m.get_all_params()
    print(f"  requested {boosting:7s} has_time={has_time!s:5s} -> boosting_type={p.get('boosting_type')} has_time={p.get('has_time')} "
          f"permutation_count={p.get('permutation_count')} fold_permutation_block={p.get('fold_permutation_block')} "
          f"task_type={p.get('task_type')} bootstrap_type={p.get('bootstrap_type')}")

print('\n1b. two seeds, all other randomness off (largest absolute difference of test probabilities)')
for boosting, has_time in (('Ordered', True), ('Ordered', False), ('Plain', False)):
    a = cb(boosting, has_time, 0).fit(Xf, yf).predict_proba(Xt)[:, 1]
    b = cb(boosting, has_time, 1).fit(Xf, yf).predict_proba(Xt)[:, 1]
    print(f'  {boosting:7s} has_time={has_time!s:5s}: seed 0 vs seed 1 max|dp| = {np.abs(a - b).max():.3e}')

print('\n1c. row order reversed (largest absolute difference of test probabilities)')
for boosting, has_time in (('Ordered', True), ('Plain', True), ('Plain', False)):
    a = cb(boosting, has_time, 0).fit(Xf, yf).predict_proba(Xt)[:, 1]
    b = cb(boosting, has_time, 0).fit(Xf[::-1].copy(), yf[::-1].copy()).predict_proba(Xt)[:, 1]
    print(f'  {boosting:7s} has_time={has_time!s:5s}: forward vs reversed rows max|dp| = {np.abs(a - b).max():.3e}')

print('\n2a. XGBoost sklearn API after early stopping (the ens3 call: eval_metric auc, early_stopping_rounds)')
g = xgb.XGBClassifier(n_estimators=400, learning_rate=0.3, max_depth=8, subsample=0.8, colsample_bytree=0.5, reg_lambda=10,
                      tree_method='hist', n_jobs=2, eval_metric='auc', early_stopping_rounds=20, random_state=0)
g.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
bi = int(g.best_iteration)
n_trees = g.get_booster().num_boosted_rounds()
p_default = g.predict_proba(Xt)[:, 1]
p_best = g.predict_proba(Xt, iteration_range=(0, bi + 1))[:, 1]
p_all = g.predict_proba(Xt, iteration_range=(0, n_trees))[:, 1]
print(f'  best_iteration {bi}, boosted rounds kept in the model {n_trees}')
print(f'  default predict vs first best_iteration+1 trees: max|dp| = {np.abs(p_default - p_best).max():.3e}')
print(f'  default predict vs all {n_trees} trees:             max|dp| = {np.abs(p_default - p_all).max():.3e}')

print('\n2b. CatBoost with an eval_set (the ens3 call: eval_metric AUC, early_stopping_rounds)')
c = CatBoostClassifier(iterations=400, learning_rate=0.3, depth=6, l2_leaf_reg=10, eval_metric='AUC', thread_count=2,
                       verbose=False, random_seed=0, allow_writing_files=False, early_stopping_rounds=20)
c.fit(Xf, yf, eval_set=(Xv, yv))
p = c.get_all_params()
print(f"  use_best_model={p.get('use_best_model')} best_iteration={c.get_best_iteration()} tree_count_={c.tree_count_} "
      f"boosting_type={p.get('boosting_type')} bootstrap_type={p.get('bootstrap_type')} task_type={p.get('task_type')} "
      f"nan_mode={p.get('nan_mode')} border_count={p.get('border_count')}")
