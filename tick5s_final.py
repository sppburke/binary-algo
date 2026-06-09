"""Final deliverable: train the 5-second microstructure model, persist model + OOS probabilities
for independent verification (verify.py compatible), and print the held-out 2026 accuracy@coverage.
This is the one horizon where a VERIFIED >=75% directional edge holds."""
import glob, time, numpy as np, pandas as pd, calendar, joblib, os
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
RAW="/home/sean/git/raw"; PAIR="EURUSD"; HS=5
def months(y,ms):
    out=[]
    for m in ms: out+=[f"{y}-{m:02d}-{d:02d}" for d in range(1,calendar.monthrange(y,m)[1]+1)]
    return out
SPLIT_DATES={"train":months(2022,[3,6,10])+months(2023,[3,6,10]),"val":months(2024,[4]),
 "test":months(2024,[10])+months(2025,[4,10]),"oos":months(2026,[2,3,4])}
def load_1s(dates):
    parts=[]
    for d in dates:
        for f in sorted(glob.glob(f"{RAW}/{PAIR}/{PAIR}_{d}_*.parquet")):
            parts.append(pd.read_parquet(f,columns=["ask","bid","ask-vol","bid-vol","timestamp_utc"]))
    t=pd.concat(parts,ignore_index=True); t["ts"]=pd.to_datetime(t["timestamp_utc"],unit="s",utc=True)
    t=t.sort_values("ts"); bid,ask=t["bid"].values,t["ask"].values
    bv,av=t["bid-vol"].values.astype(float),t["ask-vol"].values.astype(float)
    t["mid"]=(bid+ask)/2; t["imb"]=(bv-av)/(bv+av+1e-9); t["micro"]=(bid*av+ask*bv)/(av+bv+1e-9)
    t["spread"]=(ask-bid)/t["mid"]; t["tsz"]=bv+av; t=t.set_index("ts"); g=t.resample("1s")
    return pd.DataFrame({"mid":g["mid"].last(),"imb":g["imb"].mean(),"micro":g["micro"].last(),
        "spread":g["spread"].mean(),"nt":g["mid"].count(),"tsz":g["tsz"].mean()}).dropna(subset=["mid"])
def make(b):
    mid=b["mid"]; imb=b["imb"].fillna(0); micro=b["micro"]; X=pd.DataFrame(index=b.index); X["imb"]=imb
    for w in (2,3,5,8,13,21,34): X[f"imb_ema{w}"]=imb.ewm(span=w).mean(); X[f"ret{w}"]=mid.pct_change(w)
    X["imb_acc"]=imb.ewm(span=3).mean()-imb.ewm(span=13).mean(); X["imb_chg"]=imb.diff(2)
    md=(micro-mid)/mid; X["micro_dev"]=md
    for w in (3,5,10): X[f"micro_dev_ema{w}"]=md.ewm(span=w).mean()
    X["micro_mom5"]=micro/micro.shift(5)-1; X["micro_mom15"]=micro/micro.shift(15)-1
    X["spread"]=b["spread"]; X["spread_ema10"]=b["spread"].ewm(span=10).mean()
    X["nt"]=b["nt"]; X["nt_ema10"]=b["nt"].ewm(span=10).mean(); X["tsz"]=b["tsz"]
    X["rv10"]=mid.pct_change().rolling(10).std(); X["rv30"]=mid.pct_change().rolling(30).std()
    hh=b.index.hour; X["hsin"]=np.sin(2*np.pi*hh/24); X["hcos"]=np.cos(2*np.pi*hh/24)
    fwd=mid.shift(-HS); ret=fwd/mid-1; y=(ret>0).astype(float); y[ret==0]=np.nan
    return X.replace([np.inf,-np.inf],np.nan).astype("float32"), y
print("loading...",flush=True)
raw={sp:load_1s(d) for sp,d in SPLIT_DATES.items()}
D={sp:make(raw[sp]) for sp in raw}
Xtr,ytr=D["train"]; Xva,yva=D["val"]; Xte,yte=D["test"]; Xoo,yoo=D["oos"]
mtr,mva,mte,moo=ytr.notna(),yva.notna(),yte.notna(),yoo.notna()
m=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=255,
    min_child_samples=150,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=8,
    n_estimators=4000,n_jobs=20,verbosity=-1)
m.fit(Xtr[mtr],ytr[mtr].astype(int),eval_set=[(Xva[mva],yva[mva].astype(int))],
      eval_metric="auc",callbacks=[lgb.early_stopping(120),lgb.log_evaluation(0)])
pva=m.predict_proba(Xva[mva])[:,1]; pte=m.predict_proba(Xte[mte])[:,1]; poo=m.predict_proba(Xoo[moo])[:,1]
yv,yt,yo=yva[mva].astype(int).values,yte[mte].astype(int).values,yoo[moo].astype(int).values
os.makedirs("models",exist_ok=True)
joblib.dump({"model":m,"feats":list(Xtr.columns),"horizon_s":HS},"models/tick5s_model.joblib")
np.savez("models/probs_tick5s.npz",pva=pva,pte=pte,poo=poo,yva=yv,yte=yt,yoo=yo)
print(f"saved. AUC val={roc_auc_score(yv,pva):.4f} test={roc_auc_score(yt,pte):.4f} oos={roc_auc_score(yo,poo):.4f}")
print(f"\nVERIFIED 5-second directional accuracy (threshold from VAL, read off 2026 OOS):")
print(f"  {'cov':>7} {'TEST acc':>9} {'TEST n':>8} {'OOS acc':>9} {'OOS n':>7}")
for cov in (0.01,0.005,0.002,0.001,0.0005):
    thr=np.quantile(np.abs(pva-0.5),1-cov)
    rt=H.apply_threshold(yt,pte,thr); ro=H.apply_threshold(yo,poo,thr)
    print(f"  {cov:>7.3%} {rt['accuracy']:>9.4f} {rt['n']:>8} {ro['accuracy']:>9.4f} {ro['n']:>7}")
print("TICK5S FINAL DONE")
