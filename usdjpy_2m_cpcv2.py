"""USDJPY 2-MIN — EXHAUSTIVE per-fold-REFIT CPCV cross-product (close the ~1pp cert gap on the pooled edge).

SCOPE: USDJPY · 2m. The pooled edge sits at CPCV p10 ~.531 (mean .546), ~1pp under breakeven 0.541. This runs the
Tier-I IMPROVE cross-product + COMBINATIONS the goal requires, each in ONE fit that yields MANY operating points:
  * model-class   : seedens (lgb K-seeds)  |  multialgo (lgb+xgb+cat, true error-decorrelation)
  * pool-composition: all7 (7 USD-majors)  |  usdbase (USDJPY+USDCHF+USDCAD only — same USD-base reversion dynamics)
  * coverage grid : reports p10/frac-clear at cov {0.01,0.02,0.03,0.05} from the SAME fitted models
  * regime combine: each cov reported BOTH ungated AND compression-gated (trade only when rv30 < VAL q33 —
                    reversion is strongest in compression; the pooled-model × regime-gate COMBINATION)
Per-fold refit, 6 groups / C(6,2)=15 purged paths, all-pair purge=embargo=120s, ties-strict, nonoverlap gap=120,
within-fold VAL threshold per cov. CERTIFY a side at an operating point iff p10>=0.541 AND frac_clear_BE>=0.80.
Memory-safe: pool built once, float32, subsample fit 150k.
Usage: ~/binary-algo-venv/bin/python usdjpy_2m_cpcv2.py [model=seedens|multialgo] [pool=all7|usdbase] [stride=16] [nseed=5]
"""
import os, sys, json, time, numpy as np, pandas as pd
from itertools import combinations
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import harness as H

TARGET="USDJPY"; HOR=2; STEP=60; GAP=HOR*STEP; BE=0.541
ALL7=["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCHF","USDCAD"]
USDBASE=["USDJPY","USDCHF","USDCAD"]
FEAT=H.FEAT_DIR; FEATS=H.feature_cols(TARGET)
YEARS=list(range(2012,2027))
N_GROUPS, K_TEST = 6, 2
SUB_FIT=150_000
PURGE=GAP; EMBARGO=GAP
COVS=(0.01,0.02,0.03,0.05)
MODEL=sys.argv[1] if len(sys.argv)>1 and not sys.argv[1].isdigit() else "seedens"
POOLSET=sys.argv[2] if len(sys.argv)>2 and sys.argv[2] in ("all7","usdbase") else "all7"
_ints=[a for a in sys.argv[1:] if a.isdigit()]
POOL_STRIDE=int(_ints[0]) if len(_ints)>0 else 16
NSEED=int(_ints[1]) if len(_ints)>1 else 5
PAIRS=ALL7 if POOLSET=="all7" else USDBASE

def build_pair_ties(pair, stride):
    """Base feats + raw 2m fwd return (ties kept) + rv30 (30-min realized vol) for ALL years."""
    Xs=[]; fwds=[]; tss=[]; rvs=[]
    for y in YEARS:
        p=f"{FEAT}/{pair}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p, columns=FEATS+["close"]); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float)
        ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        contig=np.zeros(n,bool); contig[:n-HOR]=(ts[HOR:]-ts[:-HOR])==GAP
        fr=np.full(n,np.nan); fr[:n-HOR]=c[HOR:]/c[:-HOR]-1.0
        rv30=pd.Series(c).pct_change().rolling(30).std().values
        X=d[FEATS].astype("float32"); keepf=X.isna().mean(axis=1).values<0.5
        valid=contig & np.isfinite(fr) & keepf
        idx=np.where(valid)[0]
        if stride>1: idx=idx[::stride]
        Xs.append(X.values[idx]); fwds.append(fr[idx]); tss.append(ts[idx]); rvs.append(rv30[idx])
    if not Xs: return None
    return np.concatenate(Xs), np.concatenate(fwds), np.concatenate(tss), np.concatenate(rvs)

def mk_lgb(seed=0):
    return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=255,
        min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,
        n_estimators=800,n_jobs=20,verbosity=-1,random_state=seed,bagging_seed=seed,feature_fraction_seed=seed)

def fit_predict(Xtr,ytr,Xva,Xte):
    """Return (pval, pte) averaged over the model spec."""
    pv=np.zeros(len(Xva)); pt=np.zeros(len(Xte)); k=0
    if MODEL=="seedens":
        for sd in range(NSEED):
            L=mk_lgb(sd); L.fit(Xtr,ytr); pv+=L.predict_proba(Xva)[:,1]; pt+=L.predict_proba(Xte)[:,1]; k+=1
    else:  # multialgo: lgb + xgb + cat (true cross-algorithm decorrelation)
        L=mk_lgb(0); L.fit(Xtr,ytr); pv+=L.predict_proba(Xva)[:,1]; pt+=L.predict_proba(Xte)[:,1]; k+=1
        Xg=xgb.XGBClassifier(n_estimators=450,learning_rate=0.03,max_depth=7,subsample=0.8,
            colsample_bytree=0.5,reg_lambda=20,n_jobs=20,eval_metric="auc",verbosity=0,tree_method="hist")
        Xg.fit(Xtr,ytr); pv+=Xg.predict_proba(Xva)[:,1]; pt+=Xg.predict_proba(Xte)[:,1]; k+=1
        Cb=CatBoostClassifier(iterations=450,learning_rate=0.04,depth=8,l2_leaf_reg=20,
            loss_function="Logloss",thread_count=20,verbose=0,allow_writing_files=False)
        Cb.fit(Xtr,ytr); pv+=Cb.predict_proba(Xva)[:,1]; pt+=Cb.predict_proba(Xte)[:,1]; k+=1
    return pv/k, pt/k

def nonoverlap_chrono(ts, mask, gap=GAP):
    take=[]; block=-1
    for i in np.where(mask)[0]:
        if ts[i]<block: continue
        take.append(i); block=int(ts[i])+gap
    return np.array(take,dtype=int)

def side_wr(pr, fwd, ts, thr, side, extra=None):
    conf=np.abs(pr-0.5)
    want=(pr>0.5) if side=="UP" else (pr<0.5) if side=="DOWN" else np.ones(len(pr),bool)
    cand=want & (conf>=thr) & np.isfinite(fwd)
    if extra is not None: cand=cand & extra
    tr=nonoverlap_chrono(ts, cand)
    if len(tr)==0: return 0, float("nan")
    pred=(pr[tr]>0.5).astype(int); ylab=(fwd[tr]>0).astype(int); moved=(fwd[tr]!=0)
    return len(tr), float(((pred==ylab)&moved).astype(float).mean())

def summ(a):
    a=np.asarray([x for x in a if np.isfinite(x)],float)
    if len(a)==0: return {"n_paths":0}
    return {"n_paths":int(len(a)),"mean":round(float(a.mean()),4),"p10":round(float(np.percentile(a,10)),4),
            "p50":round(float(np.percentile(a,50)),4),"min":round(float(a.min()),4),
            "frac_clear_BE":round(float((a>=BE).mean()),3)}

def main():
    t0=time.time()
    RESULT=f"usdjpy_2m_cpcv2_{MODEL}_{POOLSET}_result.json"
    print(f"[cpcv2] model={MODEL} pool={POOLSET}({len(PAIRS)}) stride={POOL_STRIDE} nseed={NSEED} building...", flush=True)
    Xs=[]; fwds=[]; tss=[]; rvs=[]; istgt=[]
    for p in PAIRS:
        r=build_pair_ties(p, POOL_STRIDE)
        if r is None: continue
        X,fwd,ts,rv=r; Xs.append(X); fwds.append(fwd); tss.append(ts); rvs.append(rv)
        istgt.append(np.full(len(ts), p==TARGET))
        print(f"   {p}: {len(ts):,} ({time.time()-t0:.0f}s)", flush=True)
    X=np.concatenate(Xs); fwd=np.concatenate(fwds); ts=np.concatenate(tss); rv=np.concatenate(rvs); istgt=np.concatenate(istgt)
    del Xs,fwds,tss,rvs
    moved=np.isfinite(fwd)&(fwd!=0.0)
    print(f"[cpcv2] pool={len(ts):,} USDJPY={int(istgt.sum()):,} built {time.time()-t0:.0f}s", flush=True)

    tg=np.sort(ts[istgt]); bnds=[tg[int(k*len(tg)/N_GROUPS)] for k in range(N_GROUPS)]+[tg[-1]+1]
    groups=[(int(bnds[g]),int(bnds[g+1])) for g in range(N_GROUPS)]
    rng=np.random.default_rng(13)
    # accumulate per (cov, gate, side) lists of path win-rates
    acc={f"{c}|{g}":{"UP":[],"DOWN":[],"COMBINED":[]} for c in COVS for g in ("nogate","comp")}
    pathlog=[]
    for fi,testg in enumerate(combinations(range(N_GROUPS),K_TEST)):
        tblocks=[groups[g] for g in testg]
        tin=np.zeros(len(ts),bool)
        for lo,hi in tblocks: tin|=(ts>=lo)&(ts<hi)
        test_mask=istgt & tin & np.isfinite(fwd)
        purged=np.zeros(len(ts),bool)
        for lo,hi in tblocks: purged|=(ts>=lo-PURGE)&(ts<hi+EMBARGO)
        train_mask=(~purged)&moved
        tgt_tr=np.where(train_mask&istgt)[0]
        val_idx=tgt_tr[rng.random(len(tgt_tr))<0.15]
        fit_pool=np.where(train_mask)[0]; fit_pool=fit_pool[~np.isin(fit_pool,val_idx)]
        if len(fit_pool)>SUB_FIT: fit_pool=rng.choice(fit_pool,SUB_FIT,replace=False)
        ytr=(fwd[fit_pool]>0).astype(int)
        pval,ptst=fit_predict(X[fit_pool],ytr,X[val_idx],X[test_mask])
        # test arrays sorted by ts
        ft=fwd[test_mask]; tt=ts[test_mask]; rvt=rv[test_mask]
        order=np.argsort(tt); ptst=ptst[order]; ft=ft[order]; tt=tt[order]; rvt=rvt[order]
        # compression gate threshold from VAL rv30 q33
        rvv=rv[val_idx]; rvq=float(np.nanpercentile(rvv,33)); compmask=rvt<rvq
        vconf=np.abs(pval-0.5)
        prow={"fold":list(testg)}
        for cov in COVS:
            thr=float(np.quantile(vconf,1-cov))
            for gname,extra in (("nogate",None),("comp",compmask)):
                for side in ("UP","DOWN","COMBINED"):
                    n,wr=side_wr(ptst,ft,tt,thr,side,extra)
                    if n>=25 and np.isfinite(wr): acc[f"{cov}|{gname}"][side].append(wr)
                    prow[f"{cov}_{gname}_{side}"]=[n,round(wr,4) if np.isfinite(wr) else None]
        pathlog.append(prow)
        print(f"  path {fi+1}/15 g{list(testg)} cov.03 nogate UP {prow['0.03_nogate_UP']} comp UP {prow['0.03_comp_UP']} ({time.time()-t0:.0f}s)", flush=True)

    res={"model":MODEL,"pool":POOLSET,"n_pairs":len(PAIRS),"stride":POOL_STRIDE,"nseed":NSEED,"breakeven":BE,
         "CERT_RULE":"p10>=0.541 AND frac_clear_BE>=0.80","grid":{}}
    print("\n[cpcv2] === GRID (p10 / frac_clear_BE) ===", flush=True)
    best={"UP":(0,None),"DOWN":(0,None),"COMBINED":(0,None)}
    for cov in COVS:
        for gname in ("nogate","comp"):
            key=f"{cov}|{gname}"; res["grid"][key]={s:summ(acc[key][s]) for s in ("UP","DOWN","COMBINED")}
            for s in ("UP","DOWN","COMBINED"):
                sm=res["grid"][key][s]; p10=sm.get("p10",0);
                if np.isfinite(p10) and p10>best[s][0]: best[s]=(p10,key)
            u=res["grid"][key]["UP"]; d=res["grid"][key]["DOWN"]; cm=res["grid"][key]["COMBINED"]
            print(f"  cov{cov} {gname:6s}: UP p10={u.get('p10')} fc={u.get('frac_clear_BE')} (n{u.get('n_paths')}) | "
                  f"DOWN p10={d.get('p10')} fc={d.get('frac_clear_BE')} | COMB p10={cm.get('p10')} fc={cm.get('frac_clear_BE')}", flush=True)
    res["best_operating_point"]={s:{"p10":best[s][0],"at":best[s][1]} for s in best}
    res["paths"]=pathlog
    res["verdict"]={s:{"best_p10":best[s][0],"at":best[s][1],
                       "CERTIFIED":bool(best[s][0]>=BE and res["grid"][best[s][1]][s].get("frac_clear_BE",0)>=0.80)} for s in best}
    json.dump(res,open(RESULT,"w"),indent=2)
    print("\n[cpcv2] BEST operating point per side:", flush=True)
    for s in ("UP","DOWN","COMBINED"):
        v=res["verdict"][s]; print(f"  {s}: best p10={v['best_p10']:.4f} @ {v['at']}  CERT={v['CERTIFIED']}", flush=True)
    print(f"[cpcv2] done {time.time()-t0:.0f}s -> {RESULT}", flush=True)

if __name__=="__main__":
    main()
