#!/usr/bin/env python3
"""Round 6: how large a true win rate the goal's path rule needs, by trades per path. A calculation: reads no data.

From the repository root (about seven minutes):

    ~/binary-algo-venv/bin/python -I research/deriv_round6_2026-11-17/checks/pass_probability.py

The goal rule for one side at its operating coverage (scripts/cpcv_deriv.py, verdict_goal): the run has 15 paths; a path
is counted when it holds at least 25 trades; at least 12 paths are counted and clear of 0.541; and the 10th percentile
(numpy.percentile, linear interpolation) of the COUNTED paths' win rates is at or above 0.541. How hard the percentile
condition is depends on how many paths are counted. With all 15 counted it is 0.6 x the second-lowest + 0.4 x the
third-lowest win rate, so 12 clear paths cannot pass and 13 pass only when the third-lowest exceeds 0.541 by at least
1.5 times the second-lowest's shortfall. With 12 counted paths all 12 must clear and the percentile then holds; with 13
counted, 12 clear paths pass; with 14 counted it is 0.7 x the second-lowest + 0.3 x the third-lowest (section 1).

Assumptions of sections 2 to 4, none of them measured on the stored runs: all 15 paths are counted and hold the same
number of trades n, and every trade wins independently (ties already charged as losses). The 15 paths are the 15 pairs
of six time groups. Two paths that share a group take their trades from the same rows of that group with different
refits; how much those trades overlap has not been measured. Two scenarios are computed:
  independent  each path has its own n trades (no overlap);
  shared       each of the six groups has n/2 trades and a path's trades are those of its two groups (complete overlap:
               every refit takes the same trades in a group).
They are two scenarios, not proven bounds: that a partial overlap gives a probability between the two is an assumption.
Sections 2 and 3 give every trade one win probability p. Section 4 lets the win probability differ between the six
groups at the same average p; there a path holds n/2 trades from each of its two groups in both scenarios. Monte Carlo
with a fixed seed; a printed probability has a standard error of at most 0.5 / sqrt(draws).
"""
import itertools
import math

import numpy as np

BE, DRAWS, SEED = 0.541, 400_000, 20261117
PAIRS = np.array(list(itertools.combinations(range(6), 2)))          # the 15 paths as pairs of the six groups


def win_rates(model, n, p, rng, draws):
    if model == 'independent':
        return rng.binomial(n, p, size=(draws, 15)) / n
    g = rng.binomial(n // 2, p, size=(draws, 6))                      # n is even in every call
    return (g[:, PAIRS[:, 0]] + g[:, PAIRS[:, 1]]) / n


def passes(w):
    clear = (w >= BE).sum(axis=1)
    p10 = np.percentile(w, 10, axis=1)
    return float(((clear >= 12) & (p10 >= BE)).mean()), float((clear >= 12).mean())


def rates(model, n, p, rng, draws=DRAWS):
    return passes(win_rates(model, n, p, rng, draws))


def needed(model, n, target, rng):
    lo, hi = BE, 0.80                                                  # pass probability rises with p
    for _ in range(11):
        mid = (lo + hi) / 2
        if rates(model, n, mid, rng)[0] >= target:
            hi = mid
        else:
            lo = mid
    return hi


def group_rates(model, n, pg, rng):
    """Pass probability when group g has win probability pg[:, g]. In both scenarios a path holds n/2 trades from each of
    its two groups, so the scenarios differ only in whether paths that share a group share its trades."""
    if model == 'independent':
        w = (rng.binomial(n // 2, pg[:, PAIRS[:, 0]]) + rng.binomial(n // 2, pg[:, PAIRS[:, 1]])) / n
    else:
        g = rng.binomial(n // 2, pg)
        w = (g[:, PAIRS[:, 0]] + g[:, PAIRS[:, 1]]) / n
    return passes(w)[0]


rng = np.random.default_rng(SEED)
print(f'numpy {np.__version__}; {DRAWS} draws per point; seed {SEED}; standard error of a probability <= {0.5 / math.sqrt(DRAWS):.4f}')

print('\n1. the percentile condition (exact check on constructed values)')
print('  all 15 paths counted:')
for k_clear, third in ((12, 0.60), (13, 0.545), (13, 0.60), (14, 0.545)):
    w = np.array([0.50] * (15 - k_clear) + [third] + [0.60] * (k_clear - 1))
    w.sort()
    print(f'  {k_clear} of 15 clear, lowest clear path at {third}: p10 = {np.percentile(w, 10):.4f} -> '
          f'{"passes" if np.percentile(w, 10) >= BE else "fails"} (the paths below 0.541 are at 0.50)')
print('  12 paths clear at 0.60 and the other counted paths at 0.54:')
for counted in (12, 13, 14, 15):
    w = np.array([0.54] * (counted - 12) + [0.60] * 12)
    ok = len(w) >= 12 and (w >= BE).sum() >= 12 and np.percentile(w, 10) >= BE
    print(f'  {counted} counted: p10 = {np.percentile(w, 10):.4f} -> {"passes" if ok else "fails"}')

print('\n2. true win rate p at which the goal rule passes with probability 0.5 and 0.8 (15 counted paths, equal trade counts)')
print('   trades/path   independent: 0.5    0.8      shared: 0.5    0.8')
for n in (50, 60, 100, 150, 200, 300, 500):
    r = [needed(m, n, t, rng) for m in ('independent', 'shared') for t in (0.5, 0.8)]
    print(f'   {n:9d}        {r[0]:.3f}  {r[1]:.3f}          {r[2]:.3f}  {r[3]:.3f}')

print('\n3. pass probability at a given true p (pass probability of the goal rule | probability of >= 12 of 15 clear)')
print('   trades/path  true p    independent                 shared')
for n in (60, 100, 200):
    for p in (0.500, 0.507, 0.520, 0.541, 0.560, 0.580, 0.600):
        a = rates('independent', n, p, rng, 2_000_000)
        b = rates('shared', n, p, rng, 2_000_000)
        print(f'   {n:9d}   {p:.3f}    {a[0]:.5f} | {a[1]:.5f}          {b[0]:.5f} | {b[1]:.5f}')

D4 = 300_000
print(f'\n4. a win probability that differs between the six time groups, at the same average p ({D4} draws per cell)')
print('   pattern A: three groups at p + d and three at p - d. pattern B: group p = p + N(0, d), recentred to average p.')
print('   scenario     trades/path  p      same p    d 0.02: A      B       d 0.04: A      B')
lower = higher = cells = 0
for model in ('independent', 'shared'):
    for n in (60, 100, 200):
        for p in (0.541, 0.56, 0.58, 0.60, 0.62, 0.64):
            base = group_rates(model, n, np.full((D4, 6), p), rng)
            row = []
            for d in (0.02, 0.04):
                a = group_rates(model, n, np.tile(np.array([p + d] * 3 + [p - d] * 3), (D4, 1)), rng)
                z = rng.normal(0, d, size=(D4, 6))
                z -= z.mean(axis=1, keepdims=True)
                b = group_rates(model, n, np.clip(p + z, 0.01, 0.99), rng)
                row += [a, b]
                for v in (a, b):
                    se = math.sqrt(2 * max(base * (1 - base), v * (1 - v), 1e-9) / D4)
                    cells += 1
                    lower += v < base - 3 * se
                    higher += v > base + 3 * se
            print(f'   {model:11s}  {n:9d}  {p:.3f}  {base:.4f}    {row[0]:.4f}  {row[1]:.4f}        {row[2]:.4f}  {row[3]:.4f}')
print(f'   of {cells} cells, the pass probability is lower than with one p by more than 3 standard errors in {lower}, '
      f'higher in {higher}, and within 3 standard errors in {cells - lower - higher}')
