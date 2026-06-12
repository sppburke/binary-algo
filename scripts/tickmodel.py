"""Tick-microstructure model: establish the TRUE achievable-accuracy frontier across horizons
using the rich raw-tick quote data (bid/ask + sizes). 1-second bars, microstructure features,
LightGBM, predict direction at H seconds. Proper TRAIN/VAL/TEST + 2026 OOS. Sampled months for
tractability. Reports selective accuracy@coverage at each horizon — the real current-best level + where it lives."""
import glob, sys, time, numpy as np, pandas as pd, calendar
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
RAW="/home/sean/git/raw"; PAIR="EURUSD"

def months(y, ms):
    out=[]
    for m in ms:
        out+=[f"{y}-{m:02d}-{d:02d}" for d in range(1,calendar.monthrange(y,m)[1]+1)]
    return out
# sampled splits (a few months each) to bound tick volume
SPLIT_DATES={
 "train": months(2022,[2,3,5,6,9,10])+months(2023,[2,3,5,6,9,10]),
 "val":   months(2024,[3,4]),
 "test":  months(2024,[9,10])+months(2025,[3,4,9,10]),
 "oos":   months(2026,[2,3,4]),
}

def load_1s(dates):
    parts=[]
    for d in dates:
        for f in sorted(glob.glob(f"{RAW}/{PAIR}/{PAIR}_{d}_*.parquet")):
            parts.append(pd.read_parquet(f, columns=["ask","bid","ask-vol","bid-vol","timestamp_utc"]))
    if not parts: return None
    t=pd.concat(parts,ignore_index=True)
    t["ts"]=pd.to_datetime(t["timestamp_utc"],unit="s",utc=True)
    t=t.sort_values("ts")
    bid,ask=t["bid"].values,t["ask"].values; bv,av=t["bid-vol"].values.astype(float),t["ask-vol"].values.astype(float)
    t["mid"]=(bid+ask)/2; t["imb"]=(bv-av)/(bv+av+1e-9)
    t["micro"]=(bid*av+ask*bv)/(av+bv+1e-9); t["spread"]=(ask-bid)/t["mid"]
    t=t.set_index("ts")
    g=t.resample("1s")
    b=pd.DataFrame({
        "mid":g["mid"].last(),"imb":g["imb"].mean(),"micro":g["micro"].last(),
        "spread":g["spread"].mean(),"nt":g["mid"].count(),
    }).dropna(subset=["mid"])
    return b

def feats_and_labels(b, horizons):
    mid=b["mid"]; imb=b["imb"].fillna(0); micro=b["micro"]
    X=pd.DataFrame(index=b.index)
    X["imb"]=imb
    for w in (3,5,10,30,60):
        X[f"imb_ema{w}"]=imb.ewm(span=w).mean()
        X[f"ret{w}"]=mid.pct_change(w)
    X["imb_chg"]=imb.diff(3)
    X["micro_dev"]=(micro-mid)/mid
    X["micro_dev_ema5"]=X["micro_dev"].ewm(span=5).mean()
    X["spread"]=b["spread"]; X["spread_ema10"]=b["spread"].ewm(span=10).mean()
    X["nt"]=b["nt"]; X["nt_ema10"]=b["nt"].ewm(span=10).mean()
    X["rv30"]=mid.pct_change().rolling(30).std()
    X["mom_micro"]=(micro/micro.shift(10)-1)
    Y={}
    for Hs in horizons:
        fwd=mid.shift(-Hs); ret=fwd/mid-1
        y=(ret>0).astype(float); y[ret==0]=np.nan
        Y[Hs]=y
    return X.replace([np.inf,-np.inf],np.nan).astype("float32"), Y

HORIZONS=[5,10,30,60,300]  # seconds; 300s = 5 minutes
print("loading tick splits...",flush=True)
data={}
for sp,dates in SPLIT_DATES.items():
    t0=time.time(); b=load_1s(dates)
    X,Y=feats_and_labels(b,HORIZONS)
    data[sp]=(X,Y)
    print(f"  {sp}: 1s_bars={len(X):,} ({time.time()-t0:.0f}s)",flush=True)

feat_cols=list(data["train"][0].columns)
print(f"features: {len(feat_cols)}",flush=True)
print(f"\n{'H(s)':>5} {'AUCval':>7} {'AUCtest':>7} {'AUCoos':>7} {'sel1%TEST':>9} {'sel1%OOS':>9} {'best75?':>30}")
for Hs in HORIZONS:
    Xtr,ytr=data["train"][0], data["train"][1][Hs]
    mtr=ytr.notna(); Xva,yva=data["val"][0],data["val"][1][Hs]; mva=yva.notna()
    Xte,yte=data["test"][0],data["test"][1][Hs]; mte=yte.notna()
    Xoo,yoo=data["oos"][0],data["oos"][1][Hs]; moo=yoo.notna()
    m=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.03,num_leaves=127,
        min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=10,
        n_estimators=2000,n_jobs=20,verbosity=-1)
    m.fit(Xtr[mtr],ytr[mtr].astype(int),eval_set=[(Xva[mva],yva[mva].astype(int))],
          eval_metric="auc",callbacks=[lgb.early_stopping(80),lgb.log_evaluation(0)])
    pv=m.predict_proba(Xva[mva])[:,1]; pt=m.predict_proba(Xte[mte])[:,1]; po=m.predict_proba(Xoo[moo])[:,1]
    yv=yva[mva].astype(int).values; yt=yte[mte].astype(int).values; yo=yoo[moo].astype(int).values
    av_,at_,ao_=roc_auc_score(yv,pv),roc_auc_score(yt,pt),roc_auc_score(yo,po)
    thr=np.quantile(np.abs(pv-0.5),0.99)
    rt=H.apply_threshold(yt,pt,thr); ro=H.apply_threshold(yo,po,thr)
    bv=H.threshold_for_target(yv,pv,0.75,min_n=200)
    b75="VAL75 thr->" + (f"TEST {H.apply_threshold(yt,pt,bv['conf_thr'])['accuracy']:.3f}/OOS {H.apply_threshold(yo,po,bv['conf_thr'])['accuracy']:.3f}" if bv else "not hit")
    print(f"{Hs:>5} {av_:>7.4f} {at_:>7.4f} {ao_:>7.4f} {rt['accuracy']:>9.4f} {ro['accuracy']:>9.4f} {b75:>30}",flush=True)
print("TICKMODEL DONE")
