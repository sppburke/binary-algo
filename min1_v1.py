"""1-MINUTE (60s) EURUSD binary direction — fuse tick microstructure + multi-timeframe regime.

Predict y = sign(mid(t+60s) - mid(t)) on the 1s tick grid. Features at decision time t:
  - MICROSTRUCTURE: order-book imbalance (imb) + EMAs/accel, microprice deviation + EMAs/momentum,
    spread + EMA, trade count (nt) + EMA, trade size (tsz) + EMA.
  - MULTI-TIMEFRAME REGIME (causal, from the 1s mid): returns at 5s..3600s (momentum/reversion),
    realized vol at 30s..1800s (vol regime), EMA-distance at 60s..3600s (trend position),
    stretch z-scores, range-position, compression (bb-width analog), time-of-day.

Splits (tick data): TRAIN 2021-2023 · VAL 2024-H1 · TEST 2024.09-2025.11 · OOS 2026.
HONEST EVAL: selective accuracy with a NON-OVERLAPPING constraint (no two bets within 60s) so the
reported OOS n is independent/tradeable, not 60x-inflated by 1s overlap. Threshold frozen on VAL.
"""
import sys, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
TICK="/home/sean/git/binary-algo/features_tick"
HS=60                       # 1-minute horizon in seconds
TRSTRIDE=5                  # subsample training to decorrelate overlapping labels

def feats(b):
    mid=b["mid"]; imb=b["imb"].fillna(0); micro=b["micro"]; r1=mid.pct_change()
    X=pd.DataFrame(index=b.index)
    # --- microstructure ---
    X["imb"]=imb
    for w in (3,5,10,20,40,80): X[f"imb_ema{w}"]=imb.ewm(span=w).mean()
    X["imb_acc"]=imb.ewm(span=5).mean()-imb.ewm(span=40).mean(); X["imb_chg"]=imb.diff(3)
    md=(micro-mid)/mid; X["micro_dev"]=md
    for w in (5,15,30,60): X[f"micro_dev_ema{w}"]=md.ewm(span=w).mean()
    X["micro_mom15"]=micro/micro.shift(15)-1; X["micro_mom60"]=micro/micro.shift(60)-1
    X["spread"]=b["spread"]; X["spread_ema30"]=b["spread"].ewm(span=30).mean()
    X["nt"]=b["nt"]; X["nt_ema30"]=b["nt"].ewm(span=30).mean()
    X["tsz"]=b["tsz"]; X["tsz_ema30"]=b["tsz"].ewm(span=30).mean()
    # --- multi-TF returns (momentum / reversion) ---
    for w in (5,15,30,60,120,300,600,1800,3600): X[f"ret{w}"]=mid.pct_change(w)
    # --- realized vol regime ---
    for w in (30,60,300,900,1800): X[f"rv{w}"]=r1.rolling(w).std()
    # --- trend position (EMA distance) ---
    for w in (60,300,900,1800,3600): X[f"emadist{w}"]=mid/mid.ewm(span=w).mean()-1
    # --- stretch z-scores (reversion setups) ---
    for w in (60,300,900):
        sd=r1.rolling(w).std()*np.sqrt(w)
        X[f"stretch{w}"]=(mid/mid.ewm(span=w).mean()-1)/(sd+1e-9)
    # --- range position ---
    for w in (300,900,1800):
        hi=mid.rolling(w).max(); lo=mid.rolling(w).min(); X[f"rangepos{w}"]=(mid-lo)/(hi-lo+1e-12)
    # --- compression (bb-width analog) ---
    for w in (300,900,1800): X[f"bbw{w}"]=(r1.rolling(w).std()*np.sqrt(w))
    # --- time ---
    hh=b.index.hour; X["hsin"]=np.sin(2*np.pi*hh/24); X["hcos"]=np.cos(2*np.pi*hh/24)
    X["dow"]=b.index.dayofweek.astype("float32")
    return X.replace([np.inf,-np.inf],np.nan).astype("float32")

def label(mid):
    ret=mid.shift(-HS)/mid-1; y=(ret>0).astype(float); y[ret==0]=np.nan; return y

def nonoverlap_acc(ts_sec, conf, correct, thr, gap=HS):
    """Greedy: among bars with conf>=thr, pick highest-conf first, skip any within `gap` sec. Honest n."""
    sel=np.where(conf>=thr)[0]
    if len(sel)==0: return np.nan,0
    order=sel[np.argsort(-conf[sel])]
    taken=[]; used=np.empty(0,dtype="int64")
    for i in order:
        t=ts_sec[i]
        if used.size and np.any(np.abs(used-t)<gap): continue
        taken.append(i); used=np.append(used,t)
    taken=np.array(taken)
    return correct[taken].mean(), len(taken)

def main():
    t0=time.time()
    raw={sp:pd.read_parquet(f"{TICK}/{sp}_1s.parquet") for sp in ["train","val","test","oos"]}
    X={sp:feats(raw[sp]) for sp in raw}; MID={sp:raw[sp]["mid"] for sp in raw}
    TS={sp:raw[sp].index.values.astype("datetime64[s]").astype("int64") for sp in raw}
    print(f"features {time.time()-t0:.0f}s | "+" ".join(f"{sp}={len(X[sp]):,}" for sp in X),flush=True)
    Y={sp:label(MID[sp]) for sp in raw}
    def prep(sp,stride=1):
        m=Y[sp].notna().values
        idx=np.where(m)[0]
        if stride>1: idx=idx[::stride]
        return X[sp].iloc[idx], Y[sp].iloc[idx].astype(int).values, idx
    Xtr,ytr,_=prep("train",TRSTRIDE); Xva,yva,iva=prep("val"); Xte,yte,ite=prep("test"); Xoo,yoo,ioo=prep("oos")
    print(f"prep {time.time()-t0:.0f}s | train={len(ytr):,} test={len(yte):,} oos={len(yoo):,} | nfeat={Xtr.shape[1]}",flush=True)
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=350,min_child_samples=200,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=8,n_estimators=4000,n_jobs=20,verbosity=-1)
    L.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",callbacks=[lgb.early_stopping(120),lgb.log_evaluation(0)])
    pv=L.predict_proba(Xva)[:,1]; pt=L.predict_proba(Xte)[:,1]; po=L.predict_proba(Xoo)[:,1]
    print(f"\nAUC val={roc_auc_score(yva,pv):.4f} test={roc_auc_score(yte,pt):.4f} oos={roc_auc_score(yoo,po):.4f}",flush=True)
    # honest non-overlapping selective frontier, threshold frozen on VAL
    ct=(pt>0.5).astype(int)==yte; co=(po>0.5).astype(int)==yoo
    tsv=TS["val"][iva]; tst=TS["test"][ite]; tso=TS["oos"][ioo]
    cfv=np.abs(pv-0.5); cft=np.abs(pt-0.5); cfo=np.abs(po-0.5)
    print(f"\n{'cov':>7} {'thr':>7} {'TESTacc':>8} {'TESTn':>7} {'OOSacc':>8} {'OOSn':>7} {'75both?':>8}  (non-overlapping)")
    for cov in (0.05,0.02,0.01,0.005,0.002,0.001):
        thr=np.quantile(cfv,1-cov)
        at,nt=nonoverlap_acc(tst,cft,ct,thr); ao,no=nonoverlap_acc(tso,cfo,co,thr)
        ok="YES" if (not np.isnan(at) and not np.isnan(ao) and at>=0.75 and ao>=0.75 and nt>=50 and no>=50) else ""
        print(f"{cov:>7.2%} {thr:>7.4f} {at:>8.4f} {nt:>7} {ao:>8.4f} {no:>7} {ok:>8}",flush=True)
    imp=sorted(zip(Xtr.columns,L.feature_importances_),key=lambda x:-x[1])[:15]
    print("\ntop features:",[f"{n}:{v}" for n,v in imp])
    np.savez("models/probs_min1_v1.npz",pva=pv,pte=pt,poo=po,yva=yva,yte=yte,yoo=yoo)
    print("MIN1-V1 DONE")

if __name__=="__main__":
    main()
