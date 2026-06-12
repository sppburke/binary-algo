"""Sweep row B5a @300s: does RAW per-side bid/ask signed order flow (netps = bid-vol - ask-vol; sgnv = tick-rule
signed volume) carry EURUSD 5-min SIGN on MOVED bars, per-year, deriv-faithful? This is the one on-disk axis NOT
in the m5xp xpof feature set NOR the 76-candidate family probe. Retarget of _adj_perside_flow.py from 60s->300s.

Deriv-faithful: mid-to-mid, entry=next 1s tick, exit=last tick<=+300s, ties LOSE, NON-OVERLAP 300s, per-year,
MOVED bars only (|fwd_ret|>0.5pip). Single small LGBM, VAL-worst-half threshold selection, CI95 boot.

PRE-REGISTERED FALSIFIER (5m breakeven 0.541): KILL B5a unless some VAL-selected dir-conf coverage gives
moved-bar 5m-sign acc with CI95-lo >= 0.541 in BOTH 2025 AND 2026. Pre-register: expect ~0.51 (rawtick decays
by 5m + sign-invariance) -> KILLED, B5a confirmed null at 5m."""
import glob, json, numpy as np, pandas as pd, time
from min1_production import boot, nonoverlap_chrono
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

PIP = 1e-4
HOR = 300


def load_year_1s(year, months):
    rows = []
    for m in months:
        fs = sorted(glob.glob(f'/home/sean/git/raw/EURUSD/EURUSD_{year}-{m:02d}-*'))
        for f in fs:
            try: df = pd.read_parquet(f)
            except Exception: continue
            df['mid'] = (df['ask'].values + df['bid'].values) * 0.5
            df['ts'] = df['timestamp_utc'].values.astype('int64')
            df['netps'] = df['bid-vol'].values - df['ask-vol'].values
            df['totv'] = df['bid-vol'].values + df['ask-vol'].values
            dmid = np.sign(np.diff(df['mid'].values, prepend=df['mid'].values[0]))
            df['sgnv'] = dmid * df['totv'].values
            g = df.groupby('ts')
            agg = pd.DataFrame({'mid': g['mid'].last(), 'netps': g['netps'].sum(),
                                'sgnv': g['sgnv'].sum(), 'totv': g['totv'].sum(), 'nt': g.size()})
            rows.append(agg)
    if not rows: return None
    a = pd.concat(rows).groupby(level=0).agg({'mid': 'last', 'netps': 'sum', 'sgnv': 'sum', 'totv': 'sum', 'nt': 'sum'})
    return a.sort_index()


def feats(a):
    ts = a.index.values.astype('int64'); mid = a['mid'].values.astype(float)
    entry_t = ts + 1; ei = np.searchsorted(ts, entry_t, side='left')
    exit_t = entry_t + HOR; xi = np.searchsorted(ts, exit_t, side='right') - 1
    n = len(ts); eic = np.clip(ei, 0, n - 1); xic = np.clip(xi, 0, n - 1)
    valid = (ei < n) & (xi > ei) & ((ts[eic] - entry_t) <= 5) & ((exit_t - ts[xic]) <= 5)
    ret = np.where(valid, mid[xic] / np.where(mid[eic] == 0, np.nan, mid[eic]) - 1.0, np.nan)
    df = pd.DataFrame(index=a.index)
    for w in (30, 60, 120, 300):
        df[f'netps{w}'] = a['netps'].rolling(w, min_periods=1).sum().values
        df[f'sgnv{w}'] = a['sgnv'].rolling(w, min_periods=1).sum().values
        df[f'nt{w}'] = a['nt'].rolling(w, min_periods=1).sum().values
    df['flowimb'] = (a['netps'].rolling(120, min_periods=1).sum() / (a['totv'].rolling(120, min_periods=1).sum() + 1)).values
    return df, ret, valid, ts


def run_year(tag, a):
    df, ret, valid, ts = feats(a)
    yd = (ret > 0).astype(float)
    moved = valid & (np.abs(ret) > 0.5 * PIP)
    print(f"  [{tag}] secs={len(a):,} valid={int(np.nansum(valid)):,} moved={int(np.nansum(moved)):,} up_rate_moved={np.nanmean(yd[moved]):.3f}", flush=True)
    return df, yd, ret, valid, moved, ts


def main():
    t0 = time.time(); print("loading raw ticks...", flush=True)
    a24 = load_year_1s(2024, [1, 2, 3, 9, 10, 11]); print(f"2024 {time.time()-t0:.0f}s", flush=True)
    a25 = load_year_1s(2025, [2, 3, 9, 10]); print(f"2025 {time.time()-t0:.0f}s", flush=True)
    a26 = load_year_1s(2026, [2, 3, 4]); print(f"2026 {time.time()-t0:.0f}s", flush=True)
    D24, y24, r24, v24, m24, ts24 = run_year("2024tr", a24)
    D25, y25, r25, v25, m25, ts25 = run_year("2025", a25)
    D26, y26, r26, v26, m26, ts26 = run_year("2026", a26)
    n = len(D24); cut = int(n * 0.6)
    trm = m24.copy(); trm[cut:] = False
    vam = m24.copy(); vam[:cut] = False
    clf = lgb.LGBMClassifier(objective='binary', metric='auc', learning_rate=0.03, num_leaves=64,
        min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=8, n_estimators=600, n_jobs=20, verbosity=-1)
    clf.fit(D24.values[trm], y24[trm], eval_set=[(D24.values[vam], y24[vam])], eval_metric='auc',
            callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    pva = clf.predict_proba(D24.values[vam])[:, 1]
    valauc = float(roc_auc_score(y24[vam], pva)); confv = np.abs(pva - 0.5)
    print(f"\nVAL moved-bar dirAUC={valauc:.4f}", flush=True)
    W = {'2024H2': (clf.predict_proba(D24.values)[:, 1], y24, vam, ts24),
         '2025': (clf.predict_proba(D25.values)[:, 1], y25, m25, ts25),
         '2026': (clf.predict_proba(D26.values)[:, 1], y26, m26, ts26)}
    table = {}
    for cov in (1.0, 0.5, 0.25, 0.1, 0.05):
        dthr = float(np.quantile(confv, 1 - cov)); rows = {}
        for k in ('2024H2', '2025', '2026'):
            p_, y_, mm_, ts_ = W[k]
            sel = mm_ & (np.abs(p_ - 0.5) >= dthr)
            if sel.sum() == 0: rows[k] = {"n": 0, "acc": None, "ci": [None, None]}; continue
            s = nonoverlap_chrono(ts_, sel, HOR + 3)
            corr = ((p_[s] > 0.5).astype(int) == y_[s].astype(int)).astype(float)
            lo, hi = boot(corr)
            rows[k] = {"n": int(len(s)), "acc": round(float(corr.mean()), 4), "ci": [round(lo, 4), round(hi, 4)]}
        table[str(cov)] = rows
        print(f"  cov{cov:>5}: " + " | ".join(f"{k} n{rows[k]['n']} {rows[k]['acc']}{rows[k]['ci']}" for k in ('2024H2', '2025', '2026')), flush=True)
    # falsifier: any cov with CI95-lo>=0.541 in BOTH 2025 AND 2026
    survived = any(r['2025']['ci'][0] is not None and r['2025']['ci'][0] >= 0.541
                   and r['2026']['ci'][0] is not None and r['2026']['ci'][0] >= 0.541 for r in table.values())
    out = {"row": "B5a per-side raw signed flow @300s", "horizon_s": HOR, "breakeven": 0.541,
           "val_moved_dirAUC": valauc, "cov_table": table,
           "falsifier": {"SURVIVED": bool(survived),
                         "verdict": ("SURVIVED: a coverage clears 0.541 CI95-lo in BOTH 2025 and 2026" if survived else
                                     "KILLED: raw per-side signed flow (netps/sgnv) does NOT clear breakeven 0.541 CI-lo in 2025 AND 2026 "
                                     "at any coverage -> B5a null at 5m (rawtick decays by 300s + sign-invariance), confirming the family probe.")}}
    json.dump(out, open("m5_perside_flow_result.json", "w"), indent=1)
    print(f"\n[B5a] VAL dirAUC={valauc:.4f} -> {out['falsifier']['verdict']}", flush=True)
    print(f"[B5a] DONE {time.time()-t0:.0f}s -> m5_perside_flow_result.json", flush=True)


if __name__ == "__main__":
    main()
