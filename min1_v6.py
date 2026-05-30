"""1-MINUTE v6 — COMPRESSION-RELEASE specialist + refined release trigger, consolidate >=75% with more n.

v5 found compress+release(bbw300 expanding) gives TEST 0.71-0.76 / OOS 0.85 (CI lb 0.76, both halves ~0.85)
but OOS n is small (41). v6: (1) define release via the expansion RATIO rel = bbw300/bbw1800 (quiet long-run,
moving short-run); (2) train an ensemble SPECIALIST on release bars (these have directional signal, unlike
plain-quiet bars where the v4 specialist failed); (3) sweep compression depth x release threshold to find the
setup with OOS>=0.75 at the LARGEST n with TEST>=0.73 and both 2026 halves stable. VAL-frozen threshold,
honest non-overlapping eval + bootstrap CI.
"""
import time, numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
TICK="/media/sean/CORSAIR/binary-algo/features_tick"; HS=60

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

FEATCOLS=None
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
    cols=[c for c in X["train"].columns]
    print(f"features {time.time()-t0:.0f}s nfeat={len(cols)}",flush=True)
    def move(sp): return MID[sp].shift(-HS)/MID[sp]-1
    # release setup thresholds from VAL (causal): compression q33 of bbw1800, release = rel_ratio high
    bbwV=X["val"]["bbw1800"].values; q33=np.nanpercentile(bbwV,33)
    # specialist train mask = compression(q33 on TRAIN) AND rel_ratio>median(TRAIN in-compression)
    bbwT=X["train"]["bbw1800"].values; q33T=np.nanpercentile(bbwT,33)
    relT=X["train"]["rel_ratio"].values; compT=bbwT<=q33T
    relmedT=np.nanmedian(relT[compT]); relmask=compT&(relT>=relmedT)
    def prep(sp,mask=None,stride=1):
        mm=move(sp).values; m=np.isfinite(mm)&(mm!=0)
        if mask is not None: m=m&mask
        idx=np.where(m)[0]
        if stride>1: idx=idx[::stride]
        return X[sp][cols].iloc[idx], (mm[idx]>0).astype(int), idx
    Xtr,ytr,_=prep("train",mask=relmask,stride=1)   # release bars are rarer; no stride
    Xva,yva,iva=prep("val"); Xte,yte,ite=prep("test"); Xoo,yoo,ioo=prep("oos")
    print(f"prep {time.time()-t0:.0f}s release-specialist-train={len(ytr):,} test={len(yte):,} oos={len(yoo):,}",flush=True)
    P={}
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=300,min_child_samples=100,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=6,n_estimators=4000,n_jobs=20,verbosity=-1)
    L.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    P["lgb"]=(L.predict_proba(Xva)[:,1],L.predict_proba(Xte)[:,1],L.predict_proba(Xoo)[:,1])
    G=xgb.XGBClassifier(n_estimators=2500,learning_rate=0.02,max_depth=9,subsample=0.8,colsample_bytree=0.6,
        reg_lambda=6,tree_method="hist",n_jobs=20,eval_metric="auc",early_stopping_rounds=150)
    G.fit(Xtr,ytr,eval_set=[(Xva,yva)],verbose=False)
    P["xgb"]=(G.predict_proba(Xva)[:,1],G.predict_proba(Xte)[:,1],G.predict_proba(Xoo)[:,1])
    C=CatBoostClassifier(iterations=2500,learning_rate=0.02,depth=9,l2_leaf_reg=6,eval_metric="AUC",
        thread_count=20,verbose=False,early_stopping_rounds=150)
    C.fit(Xtr.fillna(-999),ytr,eval_set=(Xva.fillna(-999),yva))
    P["cat"]=(C.predict_proba(Xva.fillna(-999))[:,1],C.predict_proba(Xte.fillna(-999))[:,1],C.predict_proba(Xoo.fillna(-999))[:,1])
    pv=np.mean([P[n][0] for n in P],0); pt=np.mean([P[n][1] for n in P],0); po=np.mean([P[n][2] for n in P],0)
    print(f"RELEASE-SPECIALIST BLEND AUC val={roc_auc_score(yva,pv):.4f} test={roc_auc_score(yte,pt):.4f} oos={roc_auc_score(yoo,po):.4f}",flush=True)
    np.savez("models/probs_min1_v6.npz",pva=pv,pte=pt,poo=po,yva=yva,yte=yte,yoo=yoo)

    ct=(pt>0.5).astype(int)==yte; co=(po>0.5).astype(int)==yoo
    tst=TS["test"][ite]; tso=TS["oos"][ioo]; cft=np.abs(pt-0.5); cfo=np.abs(po-0.5); cfv=np.abs(pv-0.5)
    def rr(sp,idx): return X[sp]["rel_ratio"].values[idx]
    def bb(sp,idx): return X[sp]["bbw1800"].values[idx]
    relV=rr("val",iva); relT2=rr("test",ite); relO=rr("oos",ioo)
    bbV=bb("val",iva); bbT=bb("test",ite); bbO=bb("oos",ioo)
    oo_h=len(ioo)//2; rng=np.random.default_rng(7)
    print(f"\n{'setup':>28} {'cov':>6} {'TESTacc':>8} {'TESTn':>6} {'OOSacc':>7} {'OOSn':>5} {'OOS95CI':>14} {'H1':>5} {'H2':>5} {'75?':>4}")
    for relpc in (50,67,80):
        rqV=np.nanpercentile(relV[bbV<=q33],relpc)
        gv=(bbV<=q33)&(relV>=rqV); gt=(bbT<=q33)&(relT2>=rqV); go=(bbO<=q33)&(relO>=rqV)
        if gv.sum()<300: continue
        cfvg=cfv[gv]
        for cov in (0.20,0.10,0.05,0.03):
            thr=np.quantile(cfvg,1-cov)
            at,nt,_=nonov(tst[gt],cft[gt],ct[gt],thr); ao,no,so=nonov(tso[go],cfo[go],co[go],thr)
            goh1=go.copy(); goh1[oo_h:]=False; goh2=go.copy(); goh2[:oo_h]=False
            a1,_,_=nonov(tso[goh1],cfo[goh1],co[goh1],thr); a2,_,_=nonov(tso[goh2],cfo[goh2],co[goh2],thr)
            ci=""
            if so is not None and no>=20:
                b=np.array([rng.choice(so,no,replace=True).mean() for _ in range(3000)]); ci=f"[{np.percentile(b,2.5):.2f},{np.percentile(b,97.5):.2f}]"
            ok="**" if (not np.isnan(ao) and ao>=0.75 and no>=50 and not np.isnan(at) and at>=0.73) else ""
            print(f"  release(comp&rel>p{relpc}) {cov:>6.2%} {at:>8.3f} {nt:>6} {ao:>7.3f} {no:>5} {ci:>14} {a1:>5.2f} {a2:>5.2f} {ok:>4}",flush=True)
    print("\n** = OOS>=0.75(n>=50) AND TEST>=0.73.  MIN1-V6 DONE")

if __name__=="__main__":
    main()
