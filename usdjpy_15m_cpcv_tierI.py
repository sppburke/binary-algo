"""USDJPY 15-MIN — Tier-I improvement levers (I3 |ret|-weighted loss, I7 recency-weighting) on the
certified NY refit-CPCV folds. Runs the SAME 15 purged paths as usdjpy_15m_cpcv_session.py ny, but with
three TRAIN sample-weight schemes compared head-to-head per fold (identical splits/eval):

  control : uniform weights                         (== the certified base single-model CPCV)
  ret     : I3 — sample_weight = |fwd_ret|          (magnitude-emphasis / GMADL-style loss surrogate)
  recency : I7 — sample_weight = 0.5**(age_years/HL) (HL=3yr exp decay; recent bars weighted up)

EVAL is UNCHANGED for all three: fixed-15m deriv sign, ties LOSE, BE 0.541, NY decision rows, gap=900
nonoverlap, per-fold REFIT, |p-0.5|-rank coverage gate. Reports p10 / frac_clear_BE per scheme per cov.

I4 calibration is NOT run here: isotonic/Platt are strictly MONOTONIC in p, so they preserve the
|p-0.5| rank that the coverage gate selects on -> identical selected set -> identical win-rate. Closed
by rank-invariance (derivation, not experiment).

INCUMBENT: base single-model NY CPCV UP p10 .586 / DOWN .572 @cov3%. A lever SURVIVES-AS-IMPROVEMENT iff
its side p10 strictly exceeds control AND >= base incumbent with frac_clear_BE>=0.80.
Usage: ~/binary-algo-venv/bin/python usdjpy_15m_cpcv_tierI.py [stride=2] [cov=0.02,0.03]
"""
import os, sys, json, time, numpy as np, pandas as pd
from itertools import combinations
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from sessions import session_mask

TARGET="USDJPY"; HOR=15; STEP=60; GAP=HOR*STEP; BE=0.541
FEAT=H.FEAT_DIR; FEATS=H.feature_cols(TARGET)
YEARS=list(range(2012,2027))
N_GROUPS, K_TEST = 6, 2
SUB_FIT=150_000
PURGE=GAP; EMBARGO=GAP
NUM_LEAVES=255
SESSION="ny"
RECENCY_HL_YRS=3.0                      # recency half-life
SCHEMES=("control","ret","recency")

def _arg(i,d,cast):
    rest=[a for a in sys.argv[1:]]
    return cast(rest[i]) if len(rest)>i else d
STRIDE=_arg(0,2,int)
_covarg=_arg(1,"0.02,0.03",str); COVS=[float(x) for x in str(_covarg).split(",")]

def build_pair_ties(pair, stride):
    Xs=[]; fwds=[]; tss=[]
    for y in YEARS:
        p=f"{FEAT}/{pair}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p, columns=FEATS+["close"]); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float)
        ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        contig=np.zeros(n,bool); contig[:n-HOR]=(ts[HOR:]-ts[:-HOR])==GAP
        fr=np.full(n,np.nan); fr[:n-HOR]=c[HOR:]/c[:-HOR]-1.0
        X=d[FEATS].astype("float32"); keepf=X.isna().mean(axis=1).values<0.5
        valid=contig & np.isfinite(fr) & keepf
        idx=np.where(valid)[0]
        if stride>1: idx=idx[::stride]
        Xs.append(X.values[idx]); fwds.append(fr[idx]); tss.append(ts[idx])
    if not Xs: return None
    return np.concatenate(Xs), np.concatenate(fwds), np.concatenate(tss)

def mk_lgb(n=800, num_leaves=NUM_LEAVES, seed=0):
    return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=num_leaves,
        min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,
        n_estimators=n,n_jobs=16,verbosity=-1,random_state=seed,bagging_seed=seed,feature_fraction_seed=seed)

def nonoverlap_chrono(ts, mask, gap=GAP):
    take=[]; block=-1
    for i in np.where(mask)[0]:
        if ts[i]<block: continue
        take.append(i); block=int(ts[i])+gap
    return np.array(take,dtype=int)

def side_wr(pr, fwd, ts, thr, side):
    conf=np.abs(pr-0.5)
    want=(pr>0.5) if side=="UP" else (pr<0.5) if side=="DOWN" else np.ones(len(pr),bool)
    cand=want & (conf>=thr) & np.isfinite(fwd)
    tr=nonoverlap_chrono(ts, cand)
    if len(tr)==0: return 0, float("nan")
    pred=(pr[tr]>0.5).astype(int); ylab=(fwd[tr]>0).astype(int); moved=(fwd[tr]!=0)
    win=((pred==ylab) & moved).astype(float)
    return len(tr), float(win.mean())

def summ(a):
    a=np.asarray([x for x in a if np.isfinite(x)],float)
    if len(a)==0: return {"n_paths":0}
    return {"n_paths":int(len(a)),"mean":round(float(a.mean()),4),"p10":round(float(np.percentile(a,10)),4),
            "p50":round(float(np.percentile(a,50)),4),"min":round(float(a.min()),4),
            "frac_clear_BE":round(float((a>=BE).mean()),3)}

def main():
    t0=time.time()
    RESULT="usdjpy_15m_cpcv_tierI_result.json"
    print(f"[cpcv-tierI/{SESSION}] building USDJPY (stride {STRIDE}) covs={COVS} schemes={SCHEMES}...", flush=True)
    X,fwd,ts=build_pair_ties(TARGET, STRIDE)
    o=np.argsort(ts); X=X[o]; fwd=fwd[o]; ts=ts[o]
    sess=session_mask(ts, SESSION)
    moved=np.isfinite(fwd) & (fwd!=0.0)
    yr=pd.to_datetime(ts, unit="s").year.values.astype(float)
    ymax=yr.max()
    print(f"[cpcv-tierI/{SESSION}] USDJPY={len(ts):,} rows in-session={int(sess.sum()):,} built {time.time()-t0:.0f}s", flush=True)

    tgs=ts[sess]; bnds=[tgs[int(k*len(tgs)/N_GROUPS)] for k in range(N_GROUPS)]+[tgs[-1]+1]
    groups=[(int(bnds[g]), int(bnds[g+1])) for g in range(N_GROUPS)]

    res={"model":f"OWN-PAIR base-GBM @15m NY, TIES-STRICT per-fold-REFIT CPCV; Tier-I weight schemes {SCHEMES}",
         "key":"USDJPY.15m","breakeven":BE,"session":SESSION,"stride":STRIDE,"covs":COVS,"schemes":list(SCHEMES),
         "recency_hl_yrs":RECENCY_HL_YRS,
         "incumbent":"base single-model NY CPCV UP p10 .586 / DOWN .572 @cov3% (== control here)",
         "CERT_RULE":"lever IMPROVES iff side p10 > control AND >= incumbent AND frac_clear_BE>=0.80",
         "falsifier":{"registered_utc":"pre-paths","KILL_if":"no scheme's side p10 strictly exceeds control at matched cov"}}
    json.dump(res, open(RESULT,"w"), indent=2)

    rng=np.random.default_rng(13)
    paths={sc:{c:{"UP":[],"DOWN":[],"COMBINED":[]} for c in COVS} for sc in SCHEMES}
    aucs={sc:[] for sc in SCHEMES}; uprates=[]
    for fi,testg in enumerate(combinations(range(N_GROUPS), K_TEST)):
        tblocks=[groups[g] for g in testg]
        tin=np.zeros(len(ts),bool)
        for lo,hi in tblocks: tin |= (ts>=lo)&(ts<hi)
        test_mask=tin & sess & np.isfinite(fwd)
        purged=np.zeros(len(ts),bool)
        for lo,hi in tblocks: purged |= (ts>=lo-PURGE)&(ts<hi+EMBARGO)
        train_mask=(~purged) & moved & sess
        tr_idx=np.where(train_mask)[0]
        valsel=rng.random(len(tr_idx))<0.15
        val_idx=tr_idx[valsel]; fit_pool=tr_idx[~valsel]
        if len(fit_pool)>SUB_FIT: fit_pool=rng.choice(fit_pool, SUB_FIT, replace=False)
        ytr=(fwd[fit_pool]>0).astype(int); yval=(fwd[val_idx]>0).astype(int)
        ftst=fwd[test_mask]; ttst=ts[test_mask]
        oo=np.argsort(ttst); ftst=ftst[oo]; ttst=ttst[oo]
        mv=ftst!=0
        uprate=float((ftst[mv]>0).mean()) if mv.sum()>0 else float("nan"); uprates.append(uprate)
        # weight schemes
        absret=np.abs(fwd[fit_pool]);
        w_ret=absret/ (np.mean(absret)+1e-12)
        age=ymax - yr[fit_pool]; w_rec=0.5**(age/RECENCY_HL_YRS)
        wmap={"control":None,"ret":w_ret.astype(float),"recency":w_rec.astype(float)}
        for sc in SCHEMES:
            L=mk_lgb(seed=0)
            L.fit(X[fit_pool], ytr, sample_weight=wmap[sc], eval_set=[(X[val_idx], yval)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
            pval=L.predict_proba(X[val_idx])[:,1]; ptst=L.predict_proba(X[test_mask])[:,1][oo]
            vconf=np.abs(pval-0.5)
            auc=float(roc_auc_score((ftst[mv]>0).astype(int), ptst[mv])) if mv.sum()>20 else float("nan")
            aucs[sc].append(auc)
            for c in COVS:
                thr=float(np.quantile(vconf, 1-c))
                nC,wC=side_wr(ptst,ftst,ttst,thr,"COMBINED"); nU,wU=side_wr(ptst,ftst,ttst,thr,"UP"); nD,wD=side_wr(ptst,ftst,ttst,thr,"DOWN")
                if nC>=25: paths[sc][c]["COMBINED"].append(wC)
                if nU>=25: paths[sc][c]["UP"].append(wU)
                if nD>=25: paths[sc][c]["DOWN"].append(wD)
        print(f"  path {fi+1}/15 g{list(testg)} up={uprate:.4f} | AUC ctrl={aucs['control'][-1]:.4f} ret={aucs['ret'][-1]:.4f} rec={aucs['recency'][-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)

    INC={"UP":0.586,"DOWN":0.572,"COMBINED":0.586}
    res["uprate_tripwire_ok"]=bool(np.all([(0.45<=u<=0.55) for u in uprates if np.isfinite(u)]))
    res["auc_mean"]={sc:round(float(np.nanmean(aucs[sc])),4) for sc in SCHEMES}
    res["bycov"]={}
    ctrl_p10={c:{} for c in COVS}
    for c in COVS:
        res["bycov"][f"{c}"]={}
        for sc in SCHEMES:
            sm={s:summ(paths[sc][c][s]) for s in ("UP","DOWN","COMBINED")}
            vd={}
            for s in ("UP","DOWN","COMBINED"):
                p10=sm[s].get("p10",float("nan")); fc=sm[s].get("frac_clear_BE",0.0)
                if sc=="control": ctrl_p10[c][s]=p10
                imp=bool(np.isfinite(p10) and p10>ctrl_p10[c].get(s,99) and p10>=INC[s] and fc>=0.80) if sc!="control" else False
                vd[s]={"p10":p10,"frac_clear_BE":fc,"CERTIFIED":bool(np.isfinite(p10) and p10>=BE and fc>=0.80),"IMPROVES_control":imp}
            res["bycov"][f"{c}"][sc]={"summary":sm,"verdict":vd}
    json.dump(res, open(RESULT,"w"), indent=2)
    print(f"\n[cpcv-tierI/{SESSION}] === SUMMARY (AUC means {res['auc_mean']}) ===", flush=True)
    for c in COVS:
        print(f" cov{c}:", flush=True)
        for sc in SCHEMES:
            for s in ("UP","DOWN","COMBINED"):
                v=res["bycov"][f"{c}"][sc]["verdict"][s]
                print(f"   {sc:8s} {s:9s}: p10={v['p10']} frac={v['frac_clear_BE']} CERT={v['CERTIFIED']} IMPROVES={v['IMPROVES_control']}", flush=True)
    print(f"  trip={res['uprate_tripwire_ok']} done {time.time()-t0:.0f}s -> {RESULT}", flush=True)

if __name__=="__main__":
    main()
