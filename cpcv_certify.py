"""CPCV + DEFLATED-SHARPE / PBO CERTIFICATION  (Experiment #2)

PURPOSE — certify which HEADLINE numbers are REAL vs trial-deflation mirages.

We do NOT trust a single chronological train/val/test/oos partition: with ~70 configs searched and
corr(VAL,OOS)=-0.54 (ledger), one lucky path is expected. Instead we run COMBINATORIAL PURGED CV
(Lopez de Prado): N groups over the chronological 1-min bar index, choose k=2 as the test block,
PURGE+EMBARGO=1 label-horizon around each test group to kill the overlap leakage, enumerate C(N,k)
splits -> a DISTRIBUTION of OOS paths. A model whose 10th-percentile path still clears the bar is real;
one that only clears on the mean (or only on the original lucky split) is a multiple-testing mirage.

Targets:
  (1) 15m EURUSD DIRECTION  — retrain the m15-style LGBM (mk_lgb), MX_HOR=15 label (ret>0, ties excluded),
      report (a) raw AUC and (b) the GATED+selective accuracy (compression x NY x confidence, the actual book)
      across the C(N,2) purged paths; clear breakeven 0.541.
  (2) 30m MAGNITUDE        — |ret30|>=train-Q75 large-move classification (pe / rv30 / rv120 + LGBM),
      report large-move AUC across paths; clear 0.5+edge.

Then DEFLATION:
  - PBO (Probability of Backtest Overfitting, Bailey & Lopez de Prado, CSCV-style) on the config search.
  - Deflated metric: shrink the headline by the multiple-testing expectation given N_trials and the
    NEGATIVE corr(VAL,OOS)=-0.54 (selection actively anti-correlated => expected OOS BELOW the naive split).

MEMORY SAFETY: load ONE pair file at a time into a compact float32 feature matrix, build labels, then
del the frame; every LGBM fit SUBSAMPLES train to <=100k rows; only ~120 feature cols kept as float32.

  ~/binary-algo-venv/bin/python cpcv_certify.py
"""
import os, sys, json, time, math, itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from scipy.stats import norm

import harness as H

FEAT = H.FEAT_DIR
PAIR = "EURUSD"
YEARS = list(range(2012, 2027))          # 2012..2026 — full chronological span for the combinatorial design
RNG = np.random.default_rng(7)
SUBSAMPLE = 100_000                       # cap any single fit's TRAIN rows (memory + decorrelate overlap)
N_GROUPS = 8                              # CPCV groups over the chronological bar index
K_TEST = 2                               # test block size -> C(8,2)=28 purged-combinatorial paths
N_TRIALS = 70                            # configs the project searched (ledger: "~70+")
CORR_VAL_OOS = -0.54                     # ledger: corr(VALacc, OOSacc) across configs

FEATS = list(H.feature_cols(PAIR))       # 239 names; we keep a compact float32 subset for speed/memory
# direction book gate columns must be present:
GATE_BBW = "15m_bb_width"; GATE_NY = "sess_ny"


# ----------------------------------------------------------------------------- data
def _contig_fwd(idx_secs, c, hor):
    """forward ret over `hor` 1-min bars, only where the bar `hor` steps ahead is exactly hor*60s later."""
    n = len(c)
    contig = np.zeros(n, bool)
    if n > hor:
        contig[:n-hor] = (idx_secs[hor:] - idx_secs[:-hor]) == hor*60
    fwd = np.full(n, np.nan); fwd[:n-hor] = c[hor:]
    ret = fwd / c - 1.0
    return ret, contig


def load_direction(hor=15):
    """Pooled 2012-2026 frame for the 15m DIRECTION book. Returns float32 X (kept feats),
    y (ret>0, ties dropped), ts (sec), bbw, ny, and ret. ONE file at a time; del each frame."""
    Xs, ys, tss, bbws, nys, rets = [], [], [], [], [], []
    cols = FEATS + ["close"]
    for y in YEARS:
        p = f"{FEAT}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        df = pd.read_parquet(p, columns=cols)
        df = df[~df.index.duplicated(keep="last")]
        secs = df.index.values.astype("datetime64[s]").astype("int64")
        c = df["close"].values.astype(float)
        ret, contig = _contig_fwd(secs, c, hor)
        valid = contig & np.isfinite(ret) & (ret != 0.0)        # ties LOSE => excluded from the label set
        Xs.append(df.loc[valid, FEATS].to_numpy(np.float32))
        ys.append((ret[valid] > 0).astype(np.int8))
        tss.append(secs[valid])
        bbws.append(df[GATE_BBW].values.astype(np.float32)[valid])
        nys.append(df[GATE_NY].values.astype(np.float32)[valid])
        rets.append(ret[valid].astype(np.float32))
        del df
    X = np.concatenate(Xs); y = np.concatenate(ys); ts = np.concatenate(tss)
    bbw = np.concatenate(bbws); ny = np.concatenate(nys); ret = np.concatenate(rets)
    order = np.argsort(ts, kind="stable")                       # global chronological order
    return X[order], y[order], ts[order], bbw[order], ny[order], ret[order]


def load_magnitude(hor=30):
    """Pooled 2012-2026 frame for the 30m MAGNITUDE book. Predictors = perm-entropy(d4),rv30,rv120
    (m30_magnitude.py). Target = |ret30| large-move. Returns predictors P(n,3), aret, ts."""
    import math as _m
    def perm_entropy(r, d=4, tau=1, W=120):
        N=len(r); L=(d-1)*tau
        if N<=L+1: return np.full(N,np.nan)
        idx=np.arange(N-L)[:,None]+np.arange(0,d*tau,tau)[None,:]
        order=np.argsort(r[idx],axis=1,kind="stable"); code=(order*(d**np.arange(d))).sum(1).astype(np.int32)
        M=len(code); nb=d**d
        oh=np.zeros((M,nb),dtype=np.float32); oh[np.arange(M),code]=1.0
        cs=np.cumsum(oh,axis=0); cnt=cs.copy(); cnt[W:]=cs[W:]-cs[:-W]
        pp=cnt/np.maximum(cnt.sum(1,keepdims=True),1)
        with np.errstate(divide='ignore',invalid='ignore'):
            ent=-np.nansum(np.where(pp>0,pp*np.log(pp),0.0),axis=1)/_m.log(_m.factorial(d))
        out=np.full(N,np.nan); out[L:L+M]=ent; out[:L+W]=np.nan; return out
    Ps, arets, tss = [], [], []
    for y in YEARS:
        p=f"{FEAT}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        df=pd.read_parquet(p, columns=["close"]); df=df[~df.index.duplicated(keep="last")].sort_index()
        c=df["close"].values.astype(float); secs=df.index.values.astype("datetime64[s]").astype("int64")
        ret,contig=_contig_fwd(secs,c,hor)
        r=np.zeros(len(c)); r[1:]=np.diff(np.log(c))
        pe=perm_entropy(r,4,1,120)
        rs=pd.Series(r); rv30=rs.rolling(30).std().values; rv120=rs.rolling(120).std().values
        aret=np.abs(ret)
        valid=contig&np.isfinite(ret)&np.isfinite(pe)&np.isfinite(rv30)&np.isfinite(rv120)
        P=np.column_stack([-pe[valid], rv30[valid], rv120[valid]]).astype(np.float32)  # LOW pe -> large move (sign flip)
        Ps.append(P); arets.append(aret[valid].astype(np.float32)); tss.append(secs[valid])
        del df
    P=np.concatenate(Ps); aret=np.concatenate(arets); ts=np.concatenate(tss)
    order=np.argsort(ts,kind="stable")
    return P[order], aret[order], ts[order]


# ----------------------------------------------------------------------------- CPCV
def cpcv_groups(n, n_groups):
    """Contiguous chronological groups over the sorted bar index. Returns array group_id per row."""
    edges = np.linspace(0, n, n_groups+1).astype(int)
    g = np.zeros(n, dtype=np.int8)
    for k in range(n_groups):
        g[edges[k]:edges[k+1]] = k
    return g


def purge_embargo_mask(train_idx, test_ranges, ts, horizon_s, embargo_s):
    """Drop train rows whose [t, t+horizon] outcome window overlaps any test range, plus an embargo
    band after each test range. test_ranges: list of (t_lo, t_hi) in seconds (inclusive bounds)."""
    keep = np.ones(len(train_idx), bool)
    tt = ts[train_idx]
    for (lo, hi) in test_ranges:
        # a train obs at t labels the window [t, t+horizon]; it leaks if that window or its own past
        # touches the test block [lo,hi]. Purge t in [lo-horizon, hi]; embargo t in (hi, hi+embargo].
        bad = (tt >= lo - horizon_s) & (tt <= hi + embargo_s)
        keep &= ~bad
    return train_idx[keep]


def cpcv_paths(g, ts, horizon_min, n_groups=N_GROUPS, k=K_TEST):
    """Enumerate C(n_groups,k) combinations of test groups; yield (train_idx, test_idx) with purge+embargo."""
    horizon_s = horizon_min*60; embargo_s = horizon_min*60   # embargo = 1 label horizon
    all_idx = np.arange(len(g))
    for combo in itertools.combinations(range(n_groups), k):
        test_mask = np.isin(g, combo)
        test_idx = all_idx[test_mask]
        train_idx = all_idx[~test_mask]
        # contiguous test ranges (each chosen group is contiguous in chrono order)
        ranges = []
        for grp in combo:
            gi = all_idx[g == grp]
            ranges.append((ts[gi[0]], ts[gi[-1]]))
        train_idx = purge_embargo_mask(train_idx, ranges, ts, horizon_s, embargo_s)
        yield combo, train_idx, test_idx


def mk_lgb(n_estimators=1000):
    # matches m15_production.mk_lgb (objective/leaves/regularization), fewer trees for the 28x CPCV budget
    return lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.03, num_leaves=255,
        min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10,
        n_estimators=n_estimators, n_jobs=20, verbosity=-1)


def subsample_train(train_idx):
    if len(train_idx) > SUBSAMPLE:
        return np.sort(RNG.choice(train_idx, SUBSAMPLE, replace=False))
    return train_idx


# ----------------------------------------------------------------------------- (1) DIRECTION
def run_direction():
    t0=time.time()
    X, y, ts, bbw, ny, ret = load_direction(hor=15)
    n=len(y)
    print(f"[dir] pooled n={n:,} up-rate={y.mean():.4f} load {time.time()-t0:.0f}s", flush=True)
    g = cpcv_groups(n, N_GROUPS)
    # book gate threshold: bbw <= global-q33 (matches comp_q=33) AND NY session. Fit threshold on TRAIN each path.
    aucs=[]; sel_accs=[]; sel_ns=[]; paths=[]
    for ci,(combo, tr_idx, te_idx) in enumerate(cpcv_paths(g, ts, 15)):
        tr = subsample_train(tr_idx)
        m = mk_lgb()
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te_idx])[:,1]
        yte = y[te_idx]
        auc = roc_auc_score(yte, p)
        # gated selective book: compression(q33 of TRAIN bbw) x NY x confidence(top-cov of in-gate)
        bbw_thr = np.nanpercentile(bbw[tr], 33)
        gate = (bbw[te_idx] <= bbw_thr) & (ny[te_idx] > 0.5)
        conf = np.abs(p - 0.5)
        if gate.sum() >= 50:
            cthr = np.quantile(conf[gate], 1-0.10)          # top-10% confident in-gate (cov0.10)
            sel = gate & (conf >= cthr)
            if sel.sum() >= 25:
                acc = ((p[sel] > 0.5).astype(int) == yte[sel]).mean()
                sel_accs.append(acc); sel_ns.append(int(sel.sum()))
            else:
                sel_accs.append(np.nan); sel_ns.append(int(sel.sum()))
        else:
            sel_accs.append(np.nan); sel_ns.append(0)
        aucs.append(auc); paths.append(combo)
        del m, p
        if ci % 7 == 0:
            print(f"  [dir] path {ci+1}/28 combo{combo} AUC={auc:.4f} selacc={sel_accs[-1]} n={sel_ns[-1]} ({time.time()-t0:.0f}s)", flush=True)
    del X, y, ts, bbw, ny, ret
    aucs=np.array(aucs); sel=np.array(sel_accs,float); ns=np.array(sel_ns)
    valid_sel = sel[np.isfinite(sel) & (ns>=25)]
    res = dict(
        auc_mean=float(np.mean(aucs)), auc_p10=float(np.percentile(aucs,10)),
        auc_min=float(np.min(aucs)), auc_max=float(np.max(aucs)), n_paths=len(aucs),
        selacc_mean=float(np.mean(valid_sel)) if len(valid_sel) else float("nan"),
        selacc_p10=float(np.percentile(valid_sel,10)) if len(valid_sel) else float("nan"),
        selacc_min=float(np.min(valid_sel)) if len(valid_sel) else float("nan"),
        selacc_n_valid=int(len(valid_sel)), selacc_med_n=int(np.median(ns[ns>=25])) if (ns>=25).any() else 0,
        all_aucs=[round(float(a),4) for a in aucs],
        all_selaccs=[None if not np.isfinite(s) else round(float(s),4) for s in sel],
    )
    print(f"[dir] DONE {time.time()-t0:.0f}s  AUC mean={res['auc_mean']:.4f} p10={res['auc_p10']:.4f}"
          f"  SELacc mean={res['selacc_mean']:.4f} p10={res['selacc_p10']:.4f} (n_valid={res['selacc_n_valid']})", flush=True)
    return res, valid_sel, aucs


# ----------------------------------------------------------------------------- (2) MAGNITUDE
def run_magnitude():
    t0=time.time()
    P, aret, ts = load_magnitude(hor=30)
    n=len(aret)
    print(f"[mag] pooled n={n:,} load {time.time()-t0:.0f}s", flush=True)
    g = cpcv_groups(n, N_GROUPS)
    aucs=[]; lifts=[]
    for ci,(combo, tr_idx, te_idx) in enumerate(cpcv_paths(g, ts, 30)):
        tr = subsample_train(tr_idx)
        thr = np.nanquantile(aret[tr], 0.75)                 # large-move threshold from TRAIN only
        ytr = (aret[tr] >= thr).astype(int); yte = (aret[te_idx] >= thr).astype(int)
        if ytr.mean() in (0.0,1.0) or yte.mean() in (0.0,1.0):
            continue
        m = mk_lgb(n_estimators=600)
        m.fit(P[tr], ytr)
        pp = m.predict_proba(P[te_idx])[:,1]
        auc = roc_auc_score(yte, pp)
        # economic lift: mean |ret| in top-decile-predicted vs bottom-decile (magnitude separation)
        q=np.quantile(pp,[0.9,0.1]); hi=pp>=q[0]; lo=pp<=q[1]
        lift = float(aret[te_idx][hi].mean()/max(aret[te_idx][lo].mean(),1e-12))
        aucs.append(auc); lifts.append(lift)
        del m, pp
        if ci % 7 == 0:
            print(f"  [mag] path {ci+1}/28 combo{combo} AUC={auc:.4f} hi/lo|ret|lift={lift:.2f}x ({time.time()-t0:.0f}s)", flush=True)
    del P, aret, ts
    aucs=np.array(aucs); lifts=np.array(lifts)
    res=dict(auc_mean=float(np.mean(aucs)), auc_p10=float(np.percentile(aucs,10)),
             auc_min=float(np.min(aucs)), auc_max=float(np.max(aucs)), n_paths=len(aucs),
             lift_mean=float(np.mean(lifts)), lift_p10=float(np.percentile(lifts,10)),
             all_aucs=[round(float(a),4) for a in aucs])
    print(f"[mag] DONE {time.time()-t0:.0f}s  AUC mean={res['auc_mean']:.4f} p10={res['auc_p10']:.4f}"
          f" min={res['auc_min']:.4f}  hi/lo|ret| lift mean={res['lift_mean']:.2f}x", flush=True)
    return res, aucs


# ----------------------------------------------------------------------------- DEFLATION
def deflated_metric(headline, path_dist, bar, n_trials=N_TRIALS, corr=CORR_VAL_OOS):
    """Deflate a headline metric against the CPCV path distribution and the config-search multiplicity.

    1) Path-level: is the headline inside the body of the purged path distribution, or above its max?
       (Above max => the single chronological split was luckier than ANY honest purged path = mirage.)
    2) Multiple-testing expected-max under H0: with N independent trials the expected best in-sample is
       inflated by sigma*E[max of N standard normals] ~ sigma*sqrt(2 ln N). corr(VAL,OOS)<0 means selection
       on VAL pushes OOS the WRONG way, so the deflation is even harsher: subtract |corr|*that inflation
       from the OOS expectation (a conservative anti-selection penalty).
    3) Deflated p-value that the TRUE path metric exceeds `bar`, using the path dist mean/std and N_trials
       (Deflated-Sharpe-style: probability the best-of-N exceeds the bar by chance)."""
    pd_arr=np.asarray(path_dist,float); pd_arr=pd_arr[np.isfinite(pd_arr)]
    mu=float(pd_arr.mean()); sd=float(pd_arr.std(ddof=1)) if len(pd_arr)>1 else float("nan")
    pmax=float(pd_arr.max()); pmin=float(pd_arr.min()); p10=float(np.percentile(pd_arr,10))
    emax = sd*math.sqrt(2*math.log(max(n_trials,2))) if np.isfinite(sd) else float("nan")  # E[max-N] inflation
    anti = abs(corr)*emax if np.isfinite(emax) else float("nan")                            # neg-corr penalty
    deflated_expectation = mu - anti                                                        # honest OOS expectation
    # Deflated-Sharpe-style: prob that best-of-N purged paths exceeds the bar under N(mu,sd)
    if np.isfinite(sd) and sd>0:
        z=(bar-mu)/sd
        p_single_above=1-norm.cdf(z)
        p_bestN_above=1-(norm.cdf(z))**n_trials       # P(max of N > bar)
        # honest: prob a TYPICAL path clears the bar
        prob_typical_clears=1-norm.cdf((bar-mu)/sd)
    else:
        p_single_above=p_bestN_above=prob_typical_clears=float("nan")
    return dict(headline=headline, bar=bar, path_mean=round(mu,4), path_std=round(sd,4),
                path_min=round(pmin,4), path_p10=round(p10,4), path_max=round(pmax,4),
                headline_above_path_max=bool(headline>pmax),
                emax_inflation=round(emax,4) if np.isfinite(emax) else None,
                neg_corr_penalty=round(anti,4) if np.isfinite(anti) else None,
                deflated_expectation=round(deflated_expectation,4) if np.isfinite(deflated_expectation) else None,
                p10_clears_bar=bool(p10>bar),
                deflated_exp_clears_bar=bool(np.isfinite(deflated_expectation) and deflated_expectation>bar),
                prob_typical_path_clears_bar=round(float(prob_typical_clears),4) if np.isfinite(prob_typical_clears) else None)


# ----------------------------------------------------------------------------- main
def main():
    T0=time.time()
    out={}
    print("="*100); print("CPCV CERTIFICATION  (N=8 groups, k=2 test => C(8,2)=28 purged paths, embargo=1 horizon)"); print("="*100, flush=True)

    dir_res, dir_sel, dir_aucs = run_direction()
    mag_res, mag_aucs = run_magnitude()

    BE = 0.541        # deriv 15m breakeven at R~1.85
    # Direction: headline book = 0.647 combined (ledger). Bar = breakeven 0.541.
    dir_defl = deflated_metric(0.647, dir_sel if len(dir_sel) else dir_aucs, BE)
    # Direction AUC headline ~0.528 (val_auc in strategy.json). Bar = 0.50.
    dir_auc_defl = deflated_metric(0.528, dir_aucs, 0.50)
    # Magnitude: headline 0.79 large-move AUC (ledger). Bar = 0.50 + meaningful edge -> use 0.55 as "0.5+edge".
    mag_defl = deflated_metric(0.79, mag_aucs, 0.55)

    # 5m-stack PBO check: ledger row says 5m-stack 0.648@n45. We treat that as a headline whose honest
    # reference distribution is the DIRECTION selective book CPCV dist (same direction-edge machinery, 15m>=5m).
    # If 0.648 sits ABOVE the purged path max AND the deflated expectation < bar => PBO-positive (overfit).
    stack5m_defl = deflated_metric(0.648, dir_sel if len(dir_sel) else dir_aucs, BE)

    out["params"]=dict(n_groups=N_GROUPS, k_test=K_TEST, n_paths=28, embargo="1 label-horizon",
                       n_trials=N_TRIALS, corr_val_oos=CORR_VAL_OOS, subsample=SUBSAMPLE,
                       pair=PAIR, years=[YEARS[0],YEARS[-1]], breakeven_15m=BE)
    out["direction_15m"]=dir_res
    out["magnitude_30m"]=mag_res
    out["deflation"]=dict(direction_selective=dir_defl, direction_auc=dir_auc_defl,
                          magnitude_auc=mag_defl, stack5m_pbo=stack5m_defl)

    print("\n"+"="*100); print("DEFLATION SUMMARY"); print("="*100)
    for k,v in out["deflation"].items():
        print(f"\n[{k}] headline={v['headline']} bar={v['bar']}")
        print(f"   purged path dist: mean={v['path_mean']} p10={v['path_p10']} min={v['path_min']} max={v['path_max']}")
        print(f"   headline_above_path_max={v['headline_above_path_max']}  (TRUE => mirage: split luckier than any honest path)")
        print(f"   deflated_expectation={v['deflated_expectation']} (mean - |corr|*E[max-{N_TRIALS}])  clears_bar={v['deflated_exp_clears_bar']}")
        print(f"   p10_path_clears_bar={v['p10_clears_bar']}  prob_typical_path_clears_bar={v['prob_typical_path_clears_bar']}")

    json.dump(out, open("/home/sean/git/binary-algo/cpcv_certify_result.json","w"), indent=2)
    print(f"\n[ALL DONE {time.time()-T0:.0f}s] -> cpcv_certify_result.json", flush=True)
    return out


if __name__=="__main__":
    main()
