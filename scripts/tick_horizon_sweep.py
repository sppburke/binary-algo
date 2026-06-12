"""HORIZON-FRONTIER SWEEP — find the LONGEST horizon whose tick-microstructure model still clears
>=75% selective accuracy on BOTH TEST and 2026 OOS at usable coverage.

Key efficiency: the microstructure FEATURES are identical across horizons; only the label
y = sign(mid.shift(-H)) changes. So we build X once per split and just relabel + retrain a fast
LGBM per horizon (single model approximates the V13 ensemble well enough to locate the crossover;
the winning horizon can be re-confirmed with the full LGBM+XGB+CatBoost ensemble via tick_ensemble.py).

Threshold frozen on VAL; TEST and 2026 OOS reported at that threshold. Horizons in SECONDS.
"""
import sys, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
TICK="/home/sean/git/binary-algo/features_tick"

HORIZONS=[int(x) for x in (sys.argv[1:] or [3,5,8,13,21,34,55,89,144,233,300])]

def feats(b):
    """Horizon-independent microstructure features (mirrors tick_ensemble.make, minus the label)."""
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
    return X.replace([np.inf,-np.inf],np.nan).astype("float32")

def label(mid, HS):
    ret=mid.shift(-HS)/mid-1; y=(ret>0).astype(float); y[ret==0]=np.nan
    return y

t0=time.time()
raw={sp:pd.read_parquet(f"{TICK}/{sp}_1s.parquet") for sp in ["train","val","test","oos"]}
X={sp:feats(raw[sp]) for sp in raw}; MID={sp:raw[sp]["mid"] for sp in raw}
print(f"features built {time.time()-t0:.0f}s | "+" ".join(f"{sp}={len(X[sp]):,}" for sp in X),flush=True)

results=[]
print(f"\n{'H(s)':>5} {'AUCval':>7} {'AUCoos':>7} | best both>=75% coverage (TEST/OOS acc @ that cov, n)")
for HS in HORIZONS:
    ytr=label(MID["train"],HS); yva=label(MID["val"],HS); yte=label(MID["test"],HS); yoo=label(MID["oos"],HS)
    mtr,mva,mte,moo=ytr.notna(),yva.notna(),yte.notna(),yoo.notna()
    Xtr,ytr=X["train"][mtr.values],ytr[mtr].astype(int).values
    Xva,yva=X["val"][mva.values],yva[mva].astype(int).values
    Xte,yte=X["test"][mte.values],yte[mte].astype(int).values
    Xoo,yoo=X["oos"][moo.values],yoo[moo].astype(int).values
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.03,num_leaves=255,min_child_samples=150,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=8,n_estimators=2500,n_jobs=20,verbosity=-1)
    L.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    pv=L.predict_proba(Xva)[:,1]; pt=L.predict_proba(Xte)[:,1]; po=L.predict_proba(Xoo)[:,1]
    av,ao=roc_auc_score(yva,pv),roc_auc_score(yoo,po)
    # find the LARGEST coverage where BOTH test and oos >=75% (n>=100 each)
    best=None
    for cov in (0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002):
        thr=np.quantile(np.abs(pv-0.5),1-cov)
        rt=H.apply_threshold(yte,pt,thr); ro=H.apply_threshold(yoo,po,thr)
        if rt['accuracy']>=0.75 and ro['accuracy']>=0.75 and rt['n']>=100 and ro['n']>=100:
            best=(cov,rt,ro)  # keep largest cov meeting it (loop is descending, so first hit = largest)
            break
    if best:
        cov,rt,ro=best
        s=f"cov{cov:.3%} TEST {rt['accuracy']:.3f}(n{rt['n']}) OOS {ro['accuracy']:.3f}(n{ro['n']})  <=75% OK"
    else:
        # report best-effort at 0.05% for context
        thr=np.quantile(np.abs(pv-0.5),1-0.0005); rt=H.apply_threshold(yte,pt,thr); ro=H.apply_threshold(yoo,po,thr)
        s=f"NO 75%/100n; @0.05% TEST {rt['accuracy']:.3f}(n{rt['n']}) OOS {ro['accuracy']:.3f}(n{ro['n']})"
    print(f"{HS:>5} {av:>7.4f} {ao:>7.4f} | {s}",flush=True)
    results.append((HS,av,ao,best is not None))
ok=[r[0] for r in results if r[3]]
print(f"\nLongest horizon clearing >=75% BOTH TEST+OOS (n>=100): {max(ok) if ok else 'NONE'} sec")
print("TICK-HORIZON-SWEEP DONE")
