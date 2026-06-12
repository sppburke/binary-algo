"""1-MINUTE v4 — COMPRESSION-SPECIALIST ensemble (the v3 winning regime), pushed to 75%.

v3 showed: at 60s, betting only in volatility-COMPRESSION (bbw1800<q33) gives TEST ~0.71 / OOS ~0.78-0.83
at 0.5-1% non-overlapping coverage — the clear lever. v4 trains an ensemble ONLY on compression-regime
bars (specialist) and maps the frontier across compression DEPTH (q33/q20/q10) x coverage, optionally
stacked with low realized vol. Goal: lift the LARGE-SAMPLE TEST accuracy past 0.75 with OOS confirmation.
Honest non-overlapping (60s-gap) eval, VAL-frozen threshold, OOS split H1/H2 + bootstrap CI.
"""
import time, numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
TICK="/home/sean/git/binary-algo/features_tick"
HS=60

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
    print(f"features {time.time()-t0:.0f}s",flush=True)
    def move(sp): return MID[sp].shift(-HS)/MID[sp]-1
    # compression thresholds from TRAIN
    bbwtr=X["train"]["bbw1800"].values
    QB={q:np.nanpercentile(bbwtr,q) for q in (10,20,33)}
    def prep(sp,compq=None,stride=1):
        mm=move(sp); m=mm.notna().values & (mm.values!=0)
        if compq is not None: m=m & (X[sp]["bbw1800"].values<=QB[compq])
        idx=np.where(m)[0]
        if stride>1: idx=idx[::stride]
        return X[sp].iloc[idx], (mm.values[idx]>0).astype(int), idx
    # SPECIALIST: train only on compression(q33) bars (more of them than q10/q20)
    Xtr,ytr,_=prep("train",compq=33,stride=2)
    Xva,yva,iva=prep("val")            # eval frames = all valid bars; gate applied at eval time
    Xte,yte,ite=prep("test"); Xoo,yoo,ioo=prep("oos")
    print(f"prep {time.time()-t0:.0f}s | specialist-train={len(ytr):,} (compress q33) test={len(yte):,} oos={len(yoo):,}",flush=True)
    P={}
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=300,min_child_samples=150,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=8,n_estimators=4000,n_jobs=20,verbosity=-1)
    L.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",callbacks=[lgb.early_stopping(120),lgb.log_evaluation(0)])
    P["lgb"]=(L.predict_proba(Xva)[:,1],L.predict_proba(Xte)[:,1],L.predict_proba(Xoo)[:,1])
    G=xgb.XGBClassifier(n_estimators=2500,learning_rate=0.02,max_depth=9,subsample=0.8,colsample_bytree=0.6,
        reg_lambda=8,tree_method="hist",n_jobs=20,eval_metric="auc",early_stopping_rounds=120)
    G.fit(Xtr,ytr,eval_set=[(Xva,yva)],verbose=False)
    P["xgb"]=(G.predict_proba(Xva)[:,1],G.predict_proba(Xte)[:,1],G.predict_proba(Xoo)[:,1])
    C=CatBoostClassifier(iterations=2500,learning_rate=0.02,depth=9,l2_leaf_reg=8,eval_metric="AUC",
        thread_count=20,verbose=False,early_stopping_rounds=120)
    C.fit(Xtr.fillna(-999),ytr,eval_set=(Xva.fillna(-999),yva))
    P["cat"]=(C.predict_proba(Xva.fillna(-999))[:,1],C.predict_proba(Xte.fillna(-999))[:,1],C.predict_proba(Xoo.fillna(-999))[:,1])
    pv=np.mean([P[n][0] for n in P],0); pt=np.mean([P[n][1] for n in P],0); po=np.mean([P[n][2] for n in P],0)
    print(f"SPECIALIST BLEND AUC val={roc_auc_score(yva,pv):.4f} test={roc_auc_score(yte,pt):.4f} oos={roc_auc_score(yoo,po):.4f}",flush=True)

    ct=(pt>0.5).astype(int)==yte; co=(po>0.5).astype(int)==yoo
    tst=TS["test"][ite]; tso=TS["oos"][ioo]; cft=np.abs(pt-0.5); cfo=np.abs(po-0.5); cfv=np.abs(pv-0.5)
    bbw_v=X["val"]["bbw1800"].values[iva]; bbw_t=X["test"]["bbw1800"].values[ite]; bbw_o=X["oos"]["bbw1800"].values[ioo]
    rv_v=X["val"]["rv900"].values[iva]; rv_t=X["test"]["rv900"].values[ite]; rv_o=X["oos"]["rv900"].values[ioo]
    rvq33=np.nanpercentile(rv_v,33)
    oo_mid=len(ioo)//2; rng=np.random.default_rng(1)
    def frontier(name,gv,gt,go):
        cfvg=cfv[gv]
        if gv.sum()<300: print(f"  {name}: VAL pocket n={int(gv.sum())} too small"); return
        for cov in (0.05,0.02,0.01,0.005):
            thr=np.quantile(cfvg,1-cov)
            at,nt,_=nonov(tst[gt],cft[gt],ct[gt],thr); ao,no,so=nonov(tso[go],cfo[go],co[go],thr)
            ci=""
            if so is not None and no>=20:
                b=np.array([rng.choice(so,no,replace=True).mean() for _ in range(3000)]); ci=f"[{np.percentile(b,2.5):.2f},{np.percentile(b,97.5):.2f}]"
            ok="**" if (not np.isnan(ao) and ao>=0.75 and no>=40 and not np.isnan(at) and at>=0.74) else ""
            print(f"  {name:>22} cov{cov:>6.2%}  TEST {at:.3f}(n{nt:>5})  OOS {ao:.3f}(n{no:>4}) {ci:>14} {ok}",flush=True)
    print("\n=== compression-specialist, gated frontier (non-overlapping) ===")
    for q in (33,20,10):
        frontier(f"compress(q{q})", bbw_v<=QB[q], bbw_t<=QB[q], bbw_o<=QB[q])
    frontier("compress(q20)&lowvol", (bbw_v<=QB[20])&(rv_v<=rvq33),(bbw_t<=QB[20])&(rv_t<=rvq33),(bbw_o<=QB[20])&(rv_o<=rvq33))
    frontier("compress(q10)&lowvol", (bbw_v<=QB[10])&(rv_v<=rvq33),(bbw_t<=QB[10])&(rv_t<=rvq33),(bbw_o<=QB[10])&(rv_o<=rvq33))
    np.savez("models/probs_min1_v4.npz",pva=pv,pte=pt,poo=po,yva=yva,yte=yte,yoo=yoo)
    print("\n** = OOS>=0.75 (n>=40) AND TEST>=0.74.  MIN1-V4 DONE")

if __name__=="__main__":
    main()
