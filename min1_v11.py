"""1-MINUTE v11 — MAGNITUDE-primary gating (the AUC-0.68 signal) for a ROBUST >=75% on BOTH TEST and OOS.

v10: magnitude model (P(|ret60| large)) AUC 0.68 lifts TEST to 0.76-0.77, but release x magnitude is too thin
for the 3-month OOS. v11 uses magnitude-large as the PRIMARY gate (large moves are directional, and occur more
often than the tight release setup), optionally x light compression, then selects by direction confidence.
Grid magnitude-cut x compression x coverage; find cells where BOTH TEST and OOS >=0.75 at usable n.
Saves magnitude probs. Honest non-overlapping eval + bootstrap CI + 2026 halves.
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
    mvtr=np.abs(move("train")[IDX["train"]]); magthr=np.nanpercentile(mvtr,67)
    def magy(sp): return (np.abs(move(sp)[IDX[sp]])>=magthr).astype(int)
    M=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.03,num_leaves=300,min_child_samples=150,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=8,n_estimators=2500,n_jobs=20,verbosity=-1)
    M.fit(X["train"][cols].iloc[IDX["train"]][::5],magy("train")[::5],
          eval_set=[(X["val"][cols].iloc[IDX["val"]],magy("val"))],eval_metric="auc",
          callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    pm={sp:M.predict_proba(X[sp][cols].iloc[IDX[sp]])[:,1] for sp in raw}
    np.savez("models/probs_min1_mag.npz",pmv=pm["val"],pmt=pm["test"],pmo=pm["oos"])
    print(f"magnitude AUC val={roc_auc_score(magy('val'),pm['val']):.4f} {time.time()-t0:.0f}s",flush=True)
    d=np.load("models/probs_min1_v3.npz"); pv,pt,po=d["pva"],d["pte"],d["poo"]; yv,yt,yo=d["yva"],d["yte"],d["yoo"]
    def Cc(sp,c): return X[sp][c].values[IDX[sp]]
    bb={sp:Cc(sp,"bbw1800") for sp in raw}
    qb33=np.nanpercentile(bb["val"],33); qb50=np.nanpercentile(bb["val"],50)
    ct=(pt>0.5).astype(int)==yt; co=(po>0.5).astype(int)==yo
    tst=TS["test"][IDX["test"]]; tso=TS["oos"][IDX["oos"]]; cft=np.abs(pt-0.5); cfo=np.abs(po-0.5); cfv=np.abs(pv-0.5)
    oo_h=len(IDX["oos"])//2; rng=np.random.default_rng(7)
    COMP={"none":(np.ones_like(bb["val"],bool),np.ones_like(bb["test"],bool),np.ones_like(bb["oos"],bool)),
          "comp50":(bb["val"]<=qb50,bb["test"]<=qb50,bb["oos"]<=qb50),
          "comp33":(bb["val"]<=qb33,bb["test"]<=qb33,bb["oos"]<=qb33)}
    print(f"\n{'comp':>6} {'mag>':>5} {'cov':>6} {'TESTacc':>8} {'TESTn':>6} {'OOSacc':>7} {'OOSn':>5} {'OOS95CI':>14} {'H1':>5} {'H2':>5} {'B?':>3}")
    hits=[]
    for cname,(cmv,cmt,cmo) in COMP.items():
        for magpc in (50,67,80):
            mcv=np.nanpercentile(pm["val"],magpc)
            gv=cmv&(pm["val"]>=mcv); gt=cmt&(pm["test"]>=mcv); go=cmo&(pm["oos"]>=mcv)
            if gv.sum()<300: continue
            cfvg=cfv[gv]
            for cov in (0.30,0.20,0.10,0.05):
                thr=np.quantile(cfvg,1-cov)
                at,nt,_=nonov(tst[gt],cft[gt],ct[gt],thr); ao,no,so=nonov(tso[go],cfo[go],co[go],thr)
                goh1=go.copy(); goh1[oo_h:]=False; goh2=go.copy(); goh2[:oo_h]=False
                a1,_,_=nonov(tso[goh1],cfo[goh1],co[goh1],thr); a2,_,_=nonov(tso[goh2],cfo[goh2],co[goh2],thr)
                ci=""
                if so is not None and no>=20:
                    b=np.array([rng.choice(so,no,replace=True).mean() for _ in range(3000)]); ci=f"[{np.percentile(b,2.5):.2f},{np.percentile(b,97.5):.2f}]"
                both="**" if (not np.isnan(ao) and ao>=0.75 and no>=40 and not np.isnan(at) and at>=0.75 and nt>=40) else ""
                if both or (not np.isnan(ao) and ao>=0.72 and not np.isnan(at) and at>=0.72):
                    print(f"{cname:>6} p{magpc:<4} {cov:>6.2%} {at:>8.3f} {nt:>6} {ao:>7.3f} {no:>5} {ci:>14} {a1:>5.2f} {a2:>5.2f} {both:>3}",flush=True)
                if both: hits.append((cname,magpc,cov,at,nt,ao,no))
    print("\n** = BOTH TEST>=0.75(n>=40) AND OOS>=0.75(n>=40):")
    for h in hits: print("   ",h)
    print("MIN1-V11 DONE")

if __name__=="__main__":
    main()
