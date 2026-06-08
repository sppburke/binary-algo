"""USDJPY 15-MIN — full per-fold-REFIT CPCV of the OWN-PAIR (single-pair) base GBM (THE certification gate).

SCOPE: USDJPY · 15m. The A1/A1b own-pair base showed a real UP-tilted edge (cov3% UP .55-.60, pt-est>BE every
year) whose AUC is capped ~.531 (data-independent). Pooling DILUTED it (signal is own-pair-specific). So the
certification gate is the OWN-PAIR refit-CPCV (NOT the pooled usdjpy_15m_cpcv.py). Decides whether the edge
survives per-fold gate re-tuning across purged-combinatorial time blocks — aggregating 15 purged paths so the
effective n at the operating gate is large.

Identical design to usdjpy_15m_cpcv.py EXCEPT PAIRS=["USDJPY"] (train on USDJPY moved bars only — no pooling).
6 contiguous time-GROUPS, C(6,2)=15 purged paths, purge+embargo=900s, per-fold REFIT, gate tuned on within-fold
USDJPY VAL, test=USDJPY moved bars, TIES-STRICT, nonoverlap gap=900, per-side UP/DOWN/COMBINED selective win-rate.

CERTIFY a side iff per-side p10 >= 0.541 AND >= ~80% paths clear 0.541. Breakeven 0.541.
Usage: ~/binary-algo-venv/bin/python usdjpy_15m_cpcv_base.py [stride=4] [cov=0.03] [nseed=1]
"""
import os, sys, json, time, numpy as np, pandas as pd
from itertools import combinations
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H

TARGET="USDJPY"; HOR=15; STEP=60; GAP=HOR*STEP; BE=0.541
PAIRS=["USDJPY"]                                   # OWN-PAIR ONLY (no pooling)
FEAT=H.FEAT_DIR; FEATS=H.feature_cols(TARGET)
YEARS=list(range(2012,2027))
N_GROUPS, K_TEST = 6, 2
SUB_FIT=150_000
PURGE=GAP; EMBARGO=GAP
INCUMBENT={"UP":0.541,"DOWN":0.541,"COMBINED":0.541}
STRIDE=int(sys.argv[1]) if len(sys.argv)>1 and sys.argv[1].isdigit() else 4
COV=float(sys.argv[2]) if len(sys.argv)>2 else 0.03
NSEED=int(sys.argv[3]) if len(sys.argv)>3 else 1
NUM_LEAVES=255

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
    RESULT="usdjpy_15m_cpcv_base_result.json" if NSEED==1 else f"usdjpy_15m_cpcv_base_k{NSEED}_result.json"
    print(f"[cpcv15m-base] building USDJPY (stride {STRIDE})...", flush=True)
    r=build_pair_ties(TARGET, STRIDE)
    X,fwd,ts=r
    order=np.argsort(ts); X=X[order]; fwd=fwd[order]; ts=ts[order]
    istgt=np.ones(len(ts),bool)
    moved=np.isfinite(fwd) & (fwd!=0.0)
    print(f"[cpcv15m-base] USDJPY={len(ts):,} rows  moved={int(moved.sum()):,}  feats={X.shape[1]}  built {time.time()-t0:.0f}s", flush=True)

    tg=ts; bnds=[tg[int(k*len(tg)/N_GROUPS)] for k in range(N_GROUPS)]+[tg[-1]+1]
    groups=[(int(bnds[g]), int(bnds[g+1])) for g in range(N_GROUPS)]
    print(f"[cpcv15m-base] groups: {[(pd.to_datetime(a,unit='s').date().isoformat(), pd.to_datetime(b,unit='s').date().isoformat()) for a,b in groups]}", flush=True)

    res={"model":"OWN-PAIR (single) base-GBM @15m, TIES-STRICT per-fold-REFIT CPCV","key":"USDJPY.15m","breakeven":BE,
         "stride":STRIDE,"cov":COV,"n_groups":N_GROUPS,"k_test":K_TEST,"sub_fit":SUB_FIT,"nseed":NSEED,"num_leaves":NUM_LEAVES,
         "incumbent_floor":INCUMBENT,
         "CERT_RULE":"side CERTIFIED iff p10>=0.541 AND frac_clear_BE>=0.80",
         "falsifier":{"registered_utc":"pre-paths","KILL_if":"each side: p10 < 0.541 OR frac_clear_BE < 0.80"}}
    json.dump(res, open(RESULT,"w"), indent=2)

    rng=np.random.default_rng(13)
    paths={"UP":[],"DOWN":[],"COMBINED":[]}; aucs=[]; uprates=[]; pathinfo=[]
    folds=list(combinations(range(N_GROUPS), K_TEST))
    for fi,testg in enumerate(folds):
        tblocks=[groups[g] for g in testg]
        tin=np.zeros(len(ts),bool)
        for lo,hi in tblocks: tin |= (ts>=lo)&(ts<hi)
        test_mask=tin & np.isfinite(fwd)
        purged=np.zeros(len(ts),bool)
        for lo,hi in tblocks: purged |= (ts>=lo-PURGE)&(ts<hi+EMBARGO)
        train_mask=(~purged) & moved
        tr_idx=np.where(train_mask)[0]
        valsel=rng.random(len(tr_idx))<0.15
        val_idx=tr_idx[valsel]; fit_pool=tr_idx[~valsel]
        if len(fit_pool)>SUB_FIT: fit_pool=rng.choice(fit_pool, SUB_FIT, replace=False)
        ytr=(fwd[fit_pool]>0).astype(int); yval=(fwd[val_idx]>0).astype(int)
        pval=np.zeros(len(val_idx)); ptst=np.zeros(int(test_mask.sum()))
        for sd in range(NSEED):
            L=mk_lgb(seed=sd)
            L.fit(X[fit_pool], ytr, eval_set=[(X[val_idx], yval)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
            pval+=L.predict_proba(X[val_idx])[:,1]; ptst+=L.predict_proba(X[test_mask])[:,1]
        pval/=NSEED; ptst/=NSEED
        vconf=np.abs(pval-0.5); thr=float(np.quantile(vconf, 1-COV))
        ftst=fwd[test_mask]; ttst=ts[test_mask]
        o=np.argsort(ttst); ptst=ptst[o]; ftst=ftst[o]; ttst=ttst[o]
        mv=ftst!=0
        auc=float(roc_auc_score((ftst[mv]>0).astype(int), ptst[mv])) if mv.sum()>20 else float("nan")
        uprate=float((ftst[mv]>0).mean()) if mv.sum()>0 else float("nan")
        nC,wC=side_wr(ptst,ftst,ttst,thr,"COMBINED")
        nU,wU=side_wr(ptst,ftst,ttst,thr,"UP")
        nD,wD=side_wr(ptst,ftst,ttst,thr,"DOWN")
        if nC>=25: paths["COMBINED"].append(wC)
        if nU>=25: paths["UP"].append(wU)
        if nD>=25: paths["DOWN"].append(wD)
        aucs.append(auc); uprates.append(uprate)
        pathinfo.append({"fold":list(testg),"auc":round(auc,4),"up_rate":round(uprate,4),
                         "UP":[nU,round(wU,4)],"DOWN":[nD,round(wD,4)],"COMBINED":[nC,round(wC,4)],"thr":round(thr,4)})
        print(f"  path {fi+1}/15 g{list(testg)} AUC={auc:.4f} up={uprate:.4f} | UP n{nU} {wU:.4f} | DOWN n{nD} {wD:.4f} | COMB n{nC} {wC:.4f} ({time.time()-t0:.0f}s)", flush=True)

    res["paths"]=pathinfo
    res["auc_summary"]={"mean":round(float(np.nanmean(aucs)),4),"min":round(float(np.nanmin(aucs)),4),"max":round(float(np.nanmax(aucs)),4)}
    res["uprate_tripwire_ok"]=bool(np.all([(0.46<=u<=0.54) for u in uprates if np.isfinite(u)]))
    res["summary"]={s:summ(paths[s]) for s in ("UP","DOWN","COMBINED")}
    verdict={}
    for s in ("UP","DOWN","COMBINED"):
        sm=res["summary"][s]; p10=sm.get("p10",float("nan")); fc=sm.get("frac_clear_BE",0.0)
        verdict[s]={"p10":p10,"frac_clear_BE":fc,
                    "CERTIFIED":bool(np.isfinite(p10) and p10>=BE and fc>=0.80),
                    "IMPROVES_incumbent":bool(np.isfinite(p10) and p10>INCUMBENT.get(s,BE))}
    res["verdict"]=verdict
    json.dump(res, open(RESULT,"w"), indent=2)
    print("\n[cpcv15m-base] === SUMMARY ===", flush=True)
    for s in ("UP","DOWN","COMBINED"):
        print(f"  {s}: {res['summary'][s]}  -> CERT={verdict[s]['CERTIFIED']}", flush=True)
    print(f"  AUC {res['auc_summary']}  uprate_tripwire_ok={res['uprate_tripwire_ok']}", flush=True)
    print(f"[cpcv15m-base] done {time.time()-t0:.0f}s -> {RESULT}", flush=True)

if __name__=="__main__":
    main()
