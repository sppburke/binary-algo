"""1-MINUTE v10 — MAGNITUDE model x compression-release, lift the large-sample (TEST) accuracy.

v9 proof: compression-release gives OOS 2026 0.872 but TEST 2024-25 only 0.668 (not robust). Hypothesis:
large 60s moves are MORE directional (small moves are spread/bounce noise that settles ~random). Train a
2nd model to predict P(|ret60| large), and in the release regime bet the direction only when BOTH the
direction model is confident AND the magnitude model predicts a large move. This should raise accuracy on
BOTH held-out periods. Direction probs reuse global v3 ensemble; magnitude model trained fresh.
Honest non-overlapping eval; TEST + OOS + 2026 halves + bootstrap CI.
"""
import time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
TICK="/home/sean/git/binary-algo/features_tick"; HS=60

def feats(b):
    mid=b["mid"]; imb=b["imb"].fillna(0); micro=b["micro"]; r1=mid.pct_change()
    X=pd.DataFrame(index=b.index)
    X["imb"]=imb
    for w in (3,5,10,20,40,80): X[f"imb_ema{w}"]=imb.ewm(span=w).mean()
    X["imb_acc"]=imb.ewm(span=5).mean()-imb.ewm(span=40).mean(); X["imb_chg"]=imb.diff(3)
    sgn=np.sign(imb)
    X["imb_sgn_ac30"]=(sgn*sgn.shift(1)).rolling(30).mean()
    X["imb_sameside30"]=(sgn==sgn.shift(1)).rolling(30).mean()
    rl=sgn.groupby((sgn!=sgn.shift()).cumsum()).cumcount()+1; X["imb_runlen"]=(rl*sgn).clip(-50,50)
    md=(micro-mid)/mid; X["micro_dev"]=md
    for w in (5,15,30,60): X[f"micro_dev_ema{w}"]=md.ewm(span=w).mean()
    X["micro_mom15"]=micro/micro.shift(15)-1; X["micro_mom60"]=micro/micro.shift(60)-1
    X["spread"]=b["spread"]; X["spread_ema30"]=b["spread"].ewm(span=30).mean()
    X["nt"]=b["nt"]; X["nt_ema30"]=b["nt"].ewm(span=30).mean()
    X["tsz"]=b["tsz"]; X["tsz_ema30"]=b["tsz"].ewm(span=30).mean()
    for w in (5,15,30,60,120,300,600,1800,3600): X[f"ret{w}"]=mid.pct_change(w)
    for w in (30,60,300,900,1800): X[f"rv{w}"]=r1.rolling(w).std()
    for w in (60,300,900,1800,3600): X[f"emadist{w}"]=mid/mid.ewm(span=w).mean()-1
    for w in (60,300,900):
        sd=r1.rolling(w).std()*np.sqrt(w); X[f"stretch{w}"]=(mid/mid.ewm(span=w).mean()-1)/(sd+1e-9)
    for w in (300,900,1800):
        hi=mid.rolling(w).max(); lo=mid.rolling(w).min(); X[f"rangepos{w}"]=(mid-lo)/(hi-lo+1e-12)
    for w in (300,900,1800): X[f"bbw{w}"]=(r1.rolling(w).std()*np.sqrt(w))
    X["rel_ratio"]=X["bbw300"]/(X["bbw1800"]+1e-12)
    hh=b.index.hour; X["hsin"]=np.sin(2*np.pi*hh/24); X["hcos"]=np.cos(2*np.pi*hh/24)
    X["dow"]=b.index.dayofweek.astype("float32")
    return X.replace([np.inf,-np.inf],np.nan).astype("float32")

def nonov(ts,conf,corr,thr,gap=HS):
    sel=np.where(conf>=thr)[0]
    if len(sel)==0: return np.nan,0,None
    order=sel[np.argsort(-conf[sel])]; used=np.empty(0,"int64"); take=[]
    for i in order:
        if used.size and np.any(np.abs(used-ts[i])<gap): continue
        take.append(i); used=np.append(used,ts[i])
    take=np.array(take); return corr[take].mean(), len(take), corr[take]

def main():
    t0=time.time()
    raw={sp:pd.read_parquet(f"{TICK}/{sp}_1s.parquet") for sp in ["train","val","test","oos"]}
    X={sp:feats(raw[sp]) for sp in raw}; MID={sp:raw[sp]["mid"] for sp in raw}
    TS={sp:raw[sp].index.values.astype("datetime64[s]").astype("int64") for sp in raw}
    cols=list(X["train"].columns)
    def move(sp): return (MID[sp].shift(-HS)/MID[sp]-1).values
    IDX={sp:np.where(np.isfinite(move(sp))&(move(sp)!=0))[0] for sp in raw}
    print(f"features {time.time()-t0:.0f}s",flush=True)
    # magnitude target: |ret60| in top tercile (TRAIN threshold)
    mvtr=np.abs(move("train")[IDX["train"]]); magthr=np.nanpercentile(mvtr,67)
    def magy(sp): return (np.abs(move(sp)[IDX[sp]])>=magthr).astype(int)
    Xtr=X["train"][cols].iloc[IDX["train"]][::5]; ymtr=magy("train")[::5]
    Xva=X["val"][cols].iloc[IDX["val"]]; ymva=magy("val")
    M=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.03,num_leaves=300,min_child_samples=150,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=8,n_estimators=2500,n_jobs=20,verbosity=-1)
    M.fit(Xtr,ymtr,eval_set=[(Xva,ymva)],eval_metric="auc",callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    pmv=M.predict_proba(Xva)[:,1]; pmt=M.predict_proba(X["test"][cols].iloc[IDX["test"]])[:,1]
    pmo=M.predict_proba(X["oos"][cols].iloc[IDX["oos"]])[:,1]
    print(f"magnitude model AUC val={roc_auc_score(ymva,pmv):.4f} (predicts |ret60|>=p67) {time.time()-t0:.0f}s",flush=True)
    # direction probs from global v3
    d=np.load("models/probs_min1_v3.npz"); pv,pt,po=d["pva"],d["pte"],d["poo"]; yv,yt,yo=d["yva"],d["yte"],d["yoo"]
    def Cc(sp,c): return X[sp][c].values[IDX[sp]]
    bbV,bbT,bbO=Cc("val","bbw1800"),Cc("test","bbw1800"),Cc("oos","bbw1800")
    rrV,rrT,rrO=Cc("val","rel_ratio"),Cc("test","rel_ratio"),Cc("oos","rel_ratio")
    qb=np.nanpercentile(bbV,33); rq=np.nanpercentile(rrV[bbV<=qb],90)   # comp33 & rel>p90 (v9 setup)
    ct=(pt>0.5).astype(int)==yt; co=(po>0.5).astype(int)==yo
    tst=TS["test"][IDX["test"]]; tso=TS["oos"][IDX["oos"]]; cft=np.abs(pt-0.5); cfo=np.abs(po-0.5); cfv=np.abs(pv-0.5)
    oo_h=len(IDX["oos"])//2; rng=np.random.default_rng(7)
    # combine: within release regime AND magnitude-predicted-large, select by direction confidence
    print(f"\n{'magcut':>7} {'cov':>6} {'TESTacc':>8} {'TESTn':>6} {'OOSacc':>7} {'OOSn':>5} {'OOS95CI':>15} {'H1':>5} {'H2':>5} {'75?':>4}")
    for magpc in (0,50,67,80):   # 0 = no magnitude filter (v9 baseline)
        mcut_v=np.nanpercentile(pmv,magpc) if magpc>0 else -1
        gv=(bbV<=qb)&(rrV>=rq)&(pmv>=mcut_v); gt=(bbT<=qb)&(rrT>=rq)&(pmt>=mcut_v); go=(bbO<=qb)&(rrO>=rq)&(pmo>=mcut_v)
        if gv.sum()<150:
            print(f"  p{magpc}: VAL n={int(gv.sum())} small"); continue
        cfvg=cfv[gv]
        for cov in (0.30,0.20,0.10):
            thr=np.quantile(cfvg,1-cov)
            at,nt,_=nonov(tst[gt],cft[gt],ct[gt],thr); ao,no,so=nonov(tso[go],cfo[go],co[go],thr)
            goh1=go.copy(); goh1[oo_h:]=False; goh2=go.copy(); goh2[:oo_h]=False
            a1,_,_=nonov(tso[goh1],cfo[goh1],co[goh1],thr); a2,_,_=nonov(tso[goh2],cfo[goh2],co[goh2],thr)
            ci=""
            if so is not None and no>=20:
                b=np.array([rng.choice(so,no,replace=True).mean() for _ in range(3000)]); ci=f"[{np.percentile(b,2.5):.2f},{np.percentile(b,97.5):.2f}]"
            ok="**" if (not np.isnan(ao) and ao>=0.75 and no>=50 and not np.isnan(at) and at>=0.73) else ""
            print(f"  mag>p{magpc:<3} {cov:>6.2%} {at:>8.3f} {nt:>6} {ao:>7.3f} {no:>5} {ci:>15} {a1:>5.2f} {a2:>5.2f} {ok:>4}",flush=True)
    print("\n** = OOS>=0.75(n>=50) AND TEST>=0.73.  MIN1-V10 DONE")

if __name__=="__main__":
    main()
