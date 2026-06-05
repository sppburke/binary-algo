"""USDJPY 1-MIN MAGNITUDE |ret60|>=Q (sign-invariant edge) — Tier E1.

SCOPE: USDJPY · 1m magnitude. The program's one CPCV-certified edge is MOVE SIZE (sign-invariant),
not direction. Establish whether USDJPY 1m magnitude is forecastable (expected: yes, AUC ~0.7+, like
EURUSD 60s magAUC .787) and record in MAGNITUDE_FINDINGS.md. Also contrast magAUC vs dirAUC (~.52) on
identical data = the sign-invariance signature. NOT a direction key.

Label = (|1-min own-clock return| >= Q-percentile-of-train-|ret|). Features = 239 base. Metric = AUC +
top-decile |ret| capture (decile lift), per year. Usage: ~/binary-algo-venv/bin/python usdjpy_1m_magnitude.py
"""
import os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H

PAIR="USDJPY"; FEAT=H.FEAT_DIR; FEATS=H.feature_cols(PAIR)
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],
     "test24":["2024"],"test25":["2025"],"oos":["2026"]}
STRIDE=12; RESULT="usdjpy_1m_magnitude_result.json"

def build(years, stride=1):
    Xs=[]; ar=[]
    for y in years:
        p=f"{FEAT}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p,columns=FEATS+["close"]); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float); ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        contig=np.zeros(n,bool); contig[:n-1]=(ts[1:]-ts[:-1])==60
        fr=np.full(n,np.nan); fr[:n-1]=c[1:]/c[:-1]-1.0
        X=d[FEATS].astype("float32"); keepf=X.isna().mean(axis=1).values<0.5
        valid=contig&np.isfinite(fr)&keepf
        idx=np.where(valid)[0]
        if stride>1: idx=idx[::stride]
        Xs.append(X.iloc[idx]); ar.append(np.abs(fr[idx]))
    return pd.concat(Xs), np.concatenate(ar)

def mk(n=2500):
    return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=255,
        min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,
        n_estimators=n,n_jobs=20,verbosity=-1)

def main():
    t0=time.time(); res={"key":"USDJPY.1m.magnitude","metric":"AUC + top-decile |ret| lift","splits":SPL,
        "note":"sign-invariant — NO up/down key; record in MAGNITUDE_FINDINGS.md"}
    Xtr,atr=build(SPL["train"],STRIDE); Xva,ava=build(SPL["val"])
    res["by_Q"]={}
    for Q in (0.67,0.75,0.90):
        thr=float(np.quantile(atr,Q))
        ytr=(atr>=thr).astype(int); yva=(ava>=thr).astype(int)
        L=mk(); L.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",callbacks=[lgb.early_stopping(120),lgb.log_evaluation(0)])
        row={"Q":Q,"thr":thr,"val_auc":float(roc_auc_score(yva,L.predict_proba(Xva)[:,1])),"years":{}}
        for w in ("test24","test25","oos"):
            Xw,aw=build(SPL[w]); pr=L.predict_proba(Xw)[:,1]; yw=(aw>=thr).astype(int)
            auc=float(roc_auc_score(yw,pr))
            # top-decile lift: mean |ret| in top-10% predicted vs overall
            k=max(1,int(0.1*len(pr))); top=np.argsort(-pr)[:k]
            lift=float(aw[top].mean()/aw.mean())
            row["years"][w]={"auc":auc,"top10_absret_lift":lift,"base_rate":float(yw.mean())}
        res["by_Q"][f"{Q}"]=row
        print(f"[mag] Q={Q} VAL AUC={row['val_auc']:.4f} | "+" ".join(f"{w} AUC={row['years'][w]['auc']:.3f}(lift{row['years'][w]['top10_absret_lift']:.2f})" for w in row['years']),flush=True)
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[mag] saved {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
