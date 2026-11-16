#!/usr/bin/env python3
"""Discovery D2-01: signed shock-response curvature x forecast USD-factor variance (one feature per pair; causal).

    deriv_features_d2.py <out dir> <first day> <last day>

On a full wall-clock minute grid (minute bars as tools/deriv_dataset.py; missing minutes stay NaN, never filled):
  r_i  = a_i * one-minute log return, a_i = +1 for USD/XXX and -1 for XXX/USD (USD-strength orientation);
  z_i  = r_i / sd of the 60 preceding one-minute returns (>= 30 present);
  f_i  = median of the other six pairs' z (>= 3 present): the leave-target-out USD factor.
For each NY trading day D and pair i, using only the 20 preceding trading days (NY 07:00-18:00 minutes):
  v_clock(b)  = mean f_i^2 in 30-minute NY clock bin b;
  gamma_D     = ridge (lambda 10 on the non-intercept terms) coefficient of f^2 in
                z_i(k) = alpha_b + beta f_i(k) + gamma (f_i(k)^2 - v_clock(b)) + rho z_i(k-1);
  HAR         = OLS of V15(k) = sum_{m=k+1..k+15} f_i(m)^2 on [1, S15, S60/4, S240/16] (past sums of f_i^2 up to k).
Feature for every minute of day D: d2_shockvar = a_i * gamma_D * (HAR forecast - sum of the next 15 minutes' v_clock),
i.e. the expected signed response in the pair's UP direction. NaN until 20 trading days of history exist.
Columns: bar_start_ts, d2_shockvar (rows = every bar of the pair).
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deriv_dataset as dd  # noqa: E402

D2_VERSION = 'deriv_features_d2/0.1'
USD_BASE = ('USDJPY', 'USDCHF', 'USDCAD')
HIST_DAYS, RIDGE = 20, 10.0


def roll_sum(x, w):
    """Sum of the last w values up to and including each row (NaN-aware, scaled up from >= w/2 present values)."""
    s = pd.Series(x)
    return (s.rolling(w, min_periods=max(1, w // 2)).mean() * w).to_numpy()


def main():
    out = Path(sys.argv[1]); first = date.fromisoformat(sys.argv[2]); last = date.fromisoformat(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    closes, starts = {}, {}
    for p in dd.PAIRS:
        ts, px = dd.load_ticks(dd.day_files(p, first, last))
        b = dd.minute_bars(ts, px)
        starts[p] = b.index.values.astype(np.int64)
        closes[p] = pd.Series(np.log(b['close'].to_numpy(np.float64)), index=starts[p])
    g0 = min(s[0] for s in starts.values()); g1 = max(s[-1] for s in starts.values())
    grid = np.arange(g0, g1 + 60, 60, dtype=np.int64)
    pairs = list(dd.PAIRS)
    Z = np.full((len(grid), len(pairs)), np.nan)
    for j, p in enumerate(pairs):
        lc = closes[p].reindex(grid).to_numpy()
        r = np.diff(lc, prepend=np.nan) * (1.0 if p in USD_BASE else -1.0)
        sd = pd.Series(r).shift(1).rolling(60, min_periods=30).std().to_numpy()
        Z[:, j] = np.where(sd > 0, r / sd, np.nan)
    F = np.full_like(Z, np.nan)
    for j in range(len(pairs)):
        others = np.delete(Z, j, axis=1)
        ok = np.isfinite(others).sum(axis=1) >= 3
        F[ok, j] = np.nanmedian(others[ok], axis=1)
    ny = pd.to_datetime(grid, unit='s', utc=True).tz_convert('America/New_York')
    nyday = np.asarray(ny.date)
    mod = ny.hour.to_numpy() * 60 + ny.minute.to_numpy()
    b30 = mod // 30                                           # 48 clock bins
    win = (mod >= 7 * 60) & (mod < 18 * 60)                   # fitting window, NY 07:00-18:00
    days = [d for d in pd.unique(nyday) if np.isfinite(F[nyday == d]).any()]
    man = {'builder': D2_VERSION, 'first_day': first.isoformat(), 'last_day': last.isoformat(), 'hist_days': HIST_DAYS,
           'ridge': RIDGE, 'pairs': {}}
    for j, p in enumerate(pairs):
        a_i = 1.0 if p in USD_BASE else -1.0
        z, f = Z[:, j], F[:, j]
        zlag = np.roll(z, 1); zlag[0] = np.nan
        f2 = f ** 2
        S15, S60, S240 = roll_sum(f2, 15), roll_sum(f2, 60), roll_sum(f2, 240)
        V15 = np.roll(roll_sum(f2, 15), -15); V15[-15:] = np.nan   # sum over k+1 .. k+15
        feat = np.full(len(grid), np.nan)
        gammas = []
        for di, d in enumerate(days):
            if di < HIST_DAYS:
                continue
            hist = np.isin(nyday, days[di - HIST_DAYS:di])
            vclock = np.full(48, np.nan)
            for b in range(48):
                m = hist & (b30 == b) & np.isfinite(f2)
                if m.sum() >= 20:
                    vclock[b] = f2[m].mean()
            vclock = np.where(np.isfinite(vclock), vclock, np.nanmean(vclock))
            rows = hist & win & np.isfinite(z) & np.isfinite(f) & np.isfinite(zlag)
            if rows.sum() < 500:
                continue
            bins = b30[rows]
            ub = np.unique(bins)
            D = np.column_stack([(bins[:, None] == ub[None, :]).astype(float), f[rows], f2[rows] - vclock[bins], zlag[rows]])
            pen = np.r_[np.zeros(len(ub)), np.full(3, RIDGE)]
            coef = np.linalg.solve(D.T @ D + np.diag(pen), D.T @ z[rows])
            gam = coef[len(ub) + 1]
            hr = hist & win & np.isfinite(V15) & np.isfinite(S15) & np.isfinite(S60) & np.isfinite(S240)
            H = np.column_stack([np.ones(hr.sum()), S15[hr], S60[hr] / 4, S240[hr] / 16])
            hc = np.linalg.lstsq(H, V15[hr], rcond=None)[0]
            today = nyday == d
            Vhat = np.column_stack([np.ones(today.sum()), S15[today], S60[today] / 4, S240[today] / 16]) @ hc
            idx = ((mod[today][:, None] + np.arange(1, 16)[None, :]) // 30) % 48   # clock bins of minutes k+1 .. k+15
            vfut = vclock[idx].sum(axis=1)
            feat[today] = a_i * gam * (Vhat - vfut)
            gammas.append(gam)
        s = pd.Series(feat, index=grid).reindex(starts[p])
        df = pd.DataFrame({'bar_start_ts': starts[p], 'd2_shockvar': s.to_numpy(np.float32)})
        df.to_parquet(out / f'{p}.parquet', index=False)
        man['pairs'][p] = {'rows': int(len(df)), 'finite_share': round(float(np.isfinite(df['d2_shockvar']).mean()), 4),
                           'gamma_days': len(gammas), 'gamma_mean': round(float(np.mean(gammas)), 5) if gammas else None,
                           'gamma_sd': round(float(np.std(gammas)), 5) if gammas else None,
                           'gamma_positive_share': round(float(np.mean(np.array(gammas) > 0)), 3) if gammas else None,
                           'features': ['d2_shockvar']}
        print(p, man['pairs'][p], f'{time.time() - t0:.0f}s', flush=True)
    man['skipped_incomplete_days'] = [list(x) for x in dd.SKIPPED_INCOMPLETE]
    man['seconds'] = round(time.time() - t0, 1)
    (out / 'manifest.json').write_text(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
