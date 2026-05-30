"""1-MINUTE v13 — DIRECTION model trained ONLY on large-move bars + magnitude gate.

The global direction model is AUC 0.51 because tiny noise-moves (spread/bounce) dilute it. Hypothesis:
training the direction model ONLY on large 60s moves (|ret60|>=p67) yields a cleaner directional signal
(large moves carry real direction). Pair it with the magnitude gate (predict WHEN a large move happens,
AUC 0.68): bet the large-move direction model's call only on predicted-large moves. Goal: lift BOTH
held-out periods (TEST 2024-25 and OOS 2026) toward >=0.75. Honest non-overlapping eval + bootstrap CI.
Prints heartbeat progress lines (HB) for monitoring.
"""
import time, numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
TICK="/media/sean/CORSAIR/binary-algo/features_tick"; HS=60
T0=time.time()
def hb(m): print(f"HB[{time.time()-T0:.0f}s] {m}",flush=True)

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
    order=sel[np.argsort(-conf[sel])]
    if len(order)>60000: order=order[:60000]
    tmin=int(ts[order].min()); span=int(ts[order].max()-tmin)+gap+2
    blocked=np.zeros(span,bool); take=[]
    for i in order:
        t=int(ts[i]-tmin)
        if blocked[t]: continue
        take.append(i); blocked[max(0,t-gap+1):t+gap]=True
    take=np.array(take); return corr[take].mean(), len(take), corr[take]

def main():
    raw={sp:pd.read_parquet(f"{TICK}/{sp}_1s.parquet") for sp in ["train","val","test","oos"]}; hb("loaded parquet")
    X={sp:feats(raw[sp]) for sp in raw}; MID={sp:raw[sp]["mid"] for sp in raw}
    TS={sp:raw[sp].index.values.astype("datetime64[s]").astype("int64") for sp in raw}
    cols=list(X["train"].columns); hb(f"features built nfeat={len(cols)}")
    def move(sp): return (MID[sp].shift(-HS)/MID[sp]-1).values
    IDX={sp:np.where(np.isfinite(move(sp))&(move(sp)!=0))[0] for sp in raw}
    # large-move training mask (TRAIN p67 of |ret60|)
    mvtr=np.abs(move("train")[IDX["train"]]); magthr=np.nanpercentile(mvtr,67)
    big_tr=np.abs(move("train")[IDX["train"]])>=magthr
    ydir=lambda sp:(move(sp)[IDX[sp]]>0).astype(int)
    Xtr=X["train"][cols].iloc[IDX["train"]][big_tr]; ytr=ydir("train")[big_tr]
    Xva=X["val"][cols].iloc[IDX["val"]]; yva=ydir("val")
    hb(f"train DIRECTION-on-large: n={len(ytr):,} (|ret60|>=p67)")
    P={}
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=300,min_child_samples=120,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=8,n_estimators=4000,n_jobs=20,verbosity=-1)
    L.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",callbacks=[lgb.early_stopping(120),lgb.log_evaluation(0)]); hb("lgb done")
    Xte=X["test"][cols].iloc[IDX["test"]]; Xoo=X["oos"][cols].iloc[IDX["oos"]]
    P["lgb"]=(L.predict_proba(Xva)[:,1],L.predict_proba(Xte)[:,1],L.predict_proba(Xoo)[:,1])
    G=xgb.XGBClassifier(n_estimators=2500,learning_rate=0.02,max_depth=9,subsample=0.8,colsample_bytree=0.6,
        reg_lambda=8,tree_method="hist",n_jobs=20,eval_metric="auc",early_stopping_rounds=120)
    G.fit(Xtr,ytr,eval_set=[(Xva,yva)],verbose=False); hb("xgb done")
    P["xgb"]=(G.predict_proba(Xva)[:,1],G.predict_proba(Xte)[:,1],G.predict_proba(Xoo)[:,1])
    C=CatBoostClassifier(iterations=2500,learning_rate=0.02,depth=9,l2_leaf_reg=8,eval_metric="AUC",
        thread_count=20,verbose=False,early_stopping_rounds=120)
    C.fit(Xtr.fillna(-999),ytr,eval_set=(Xva.fillna(-999),yva)); hb("cat done")
    P["cat"]=(C.predict_proba(Xva.fillna(-999))[:,1],C.predict_proba(Xte.fillna(-999))[:,1],C.predict_proba(Xoo.fillna(-999))[:,1])
    pv=np.mean([P[n][0] for n in P],0); pt=np.mean([P[n][1] for n in P],0); po=np.mean([P[n][2] for n in P],0)
    yt=ydir("test"); yo=ydir("oos")
    hb(f"DIR-on-large AUC val={roc_auc_score(yva,pv):.4f} test={roc_auc_score(yt,pt):.4f} oos={roc_auc_score(yo,po):.4f}")
    # magnitude probs (reuse saved)
    m=np.load("models/probs_min1_mag.npz"); pmv,pmt,pmo=m["pmv"],m["pmt"],m["pmo"]
    def Cc(sp,c): return X[sp][c].values[IDX[sp]]
    bb={sp:Cc(sp,"bbw1800") for sp in raw}; qb33=np.nanpercentile(bb["val"],33)
    ct=(pt>0.5).astype(int)==yt; co=(po>0.5).astype(int)==yo
    tst=TS["test"][IDX["test"]]; tso=TS["oos"][IDX["oos"]]; cft=np.abs(pt-0.5); cfo=np.abs(po-0.5); cfv=np.abs(pv-0.5)
    oo_h=len(IDX["oos"])//2; rng=np.random.default_rng(7)
    print(f"\n{'gate':>22} {'cov':>6} {'TESTacc':>8} {'TESTn':>6} {'OOSacc':>7} {'OOSn':>5} {'OOS95CI':>14} {'H1':>5} {'H2':>5} {'B?':>3}",flush=True)
    for cname,cmask in (("magP67",(pmv>=np.nanpercentile(pmv,67),pmt>=np.nanpercentile(pmv,67),pmo>=np.nanpercentile(pmv,67))),
                        ("comp33&magP67",((bb["val"]<=qb33)&(pmv>=np.nanpercentile(pmv,67)),(bb["test"]<=qb33)&(pmt>=np.nanpercentile(pmv,67)),(bb["oos"]<=qb33)&(pmo>=np.nanpercentile(pmv,67)))),
                        ("magP80",(pmv>=np.nanpercentile(pmv,80),pmt>=np.nanpercentile(pmv,80),pmo>=np.nanpercentile(pmv,80)))):
        gv,gt,go=cmask
        if gv.sum()<300: continue
        cfvg=cfv[gv]
        for cov in (0.30,0.20,0.10,0.05):
            thr=np.quantile(cfvg,1-cov)
            at,nt,_=nonov(tst[gt],cft[gt],ct[gt],thr); ao,no,_=nonov(tso[go],cfo[go],co[go],thr)
            goh1=go.copy(); goh1[oo_h:]=False; goh2=go.copy(); goh2[:oo_h]=False
            a1,_,_=nonov(tso[goh1],cfo[goh1],co[goh1],thr); a2,_,_=nonov(tso[goh2],cfo[goh2],co[goh2],thr)
            ci=""
            if no>=20:
                _,_,so=nonov(tso[go],cfo[go],co[go],thr)
                if so is not None: b=np.array([rng.choice(so,no,replace=True).mean() for _ in range(3000)]); ci=f"[{np.percentile(b,2.5):.2f},{np.percentile(b,97.5):.2f}]"
            both="**" if (not np.isnan(ao) and ao>=0.75 and no>=40 and not np.isnan(at) and at>=0.75 and nt>=40) else ""
            print(f"{cname:>22} {cov:>6.2%} {at:>8.3f} {nt:>6} {ao:>7.3f} {no:>5} {ci:>14} {a1:>5.2f} {a2:>5.2f} {both:>3}",flush=True)
    hb("MIN1-V13 DONE")

if __name__=="__main__":
    main()
