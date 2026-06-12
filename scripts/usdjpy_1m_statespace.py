"""USDJPY 1-MIN — Tier-C state-space coverage (online-ARF keystone + Kalman forward-filter).

SCOPE: USDJPY · 1m. Coverage-rule 'run once' Tier-C, the honest-exhaustion confirmators:
 C5 ONLINE-ARF KEYSTONE (river): an ADAPTIVE concept-drift forest that retrains continuously (immune to
   stale-model regime drift). If it is ~.50 AUC every held-out year, the wall is GENUINE EFFICIENCY, not a
   stationary-GBM limitation — the keystone control. Chronological predict-BEFORE-learn (no look-ahead).
 C2 KALMAN forward-filter velocity sign (leakage trap #1: FILTER only, never RTS smoother) — classic
   state-space directional signal: sign of the filtered local drift predicts next-1m sign?
Usage: ~/binary-algo-venv/bin/python usdjpy_1m_statespace.py [stride]
"""
import sys, os, json, time, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import harness as H

PAIR="USDJPY"; FEAT=H.FEAT_DIR; FEATS=H.feature_cols(PAIR)
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],
     "test24":["2024"],"test25":["2025"],"oos":["2026"]}
STRIDE=int(sys.argv[1]) if len(sys.argv)>1 else 25     # online stream is sequential -> stride for tractability
RESULT="usdjpy_1m_statespace_result.json"
# compact informative feature subset for the online stream (full 239 too slow for river)
ARF_FEATS=["1m_ret_1","1m_ret_3","1m_ret_6","1m_ret_12","1m_rsi","1m_rangepos_12","1m_dist_ema10",
           "4h_slope_20","1m_rv_24","1m_macd_hist"]

def load(years, cols, stride=1):
    Xs=[];ys=[];cc=[];tss=[]
    for y in years:
        p=f"{FEAT}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p,columns=list(dict.fromkeys(cols+["close"]))); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float); ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        contig=np.zeros(n,bool); contig[:n-1]=(ts[1:]-ts[:-1])==60
        fr=np.full(n,np.nan); fr[:n-1]=c[1:]/c[:-1]-1.0
        X=d[cols].astype("float32"); keepf=X.notna().all(axis=1).values
        valid=contig&np.isfinite(fr)&(fr!=0)&keepf
        idx=np.where(valid)[0]
        if stride>1: idx=idx[::stride]
        Xs.append(X.iloc[idx].values); ys.append((fr[idx]>0).astype(int)); cc.append(c[idx]); tss.append(ts[idx])
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(cc), np.concatenate(tss)

def online_arf(res):
    from river import forest
    cols=[c for c in ARF_FEATS]
    # stream order: train -> val -> test24 -> test25 -> oos (chronological), predict-before-learn
    streams={k:load(SPL[k],cols,STRIDE) for k in ("train","val","test24","test25","oos")}
    model=forest.ARFClassifier(n_models=8, seed=7)
    preds={k:([],[]) for k in ("test24","test25","oos")}
    t0=time.time()
    for k in ("train","val","test24","test25","oos"):
        X,y,_,_=streams[k]
        for i in range(len(y)):
            xd={cols[j]:float(X[i,j]) for j in range(len(cols))}
            p=model.predict_proba_one(xd); pu=p.get(1,p.get(True,0.5)) if p else 0.5
            if k in preds: preds[k][0].append(pu); preds[k][1].append(int(y[i]))
            model.learn_one(xd,int(y[i]))
        print(f"  ARF streamed {k} ({len(y)} rows) {time.time()-t0:.0f}s",flush=True)
    res["online_arf"]={k:{"auc":float(roc_auc_score(preds[k][1],preds[k][0])),"n":len(preds[k][1])} for k in preds}
    print(f"[arf] per-year AUC: "+", ".join(f"{k} {res['online_arf'][k]['auc']:.4f}" for k in preds),flush=True)

def kalman(res):
    """1-D local-level+drift Kalman FORWARD filter on log-close; predict next-1m sign = sign of filtered drift."""
    cols=["1m_ret_1"]
    aucs={}
    for k in ("test24","test25","oos"):
        _,y,c,_=load(SPL[k],cols,1)
        lc=np.log(c); n=len(lc)
        # constant-velocity KF (level x, drift v); Q,R tuned coarse
        x=lc[0]; v=0.0; P=np.eye(2)*1e-4; Q=np.eye(2)*1e-9; R=1e-8
        F=np.array([[1,1],[0,1.0]]); Hm=np.array([[1.0,0.0]])
        drift=np.zeros(n)
        st=np.array([x,v])
        for t in range(1,n):
            st=F@st; P=F@P@F.T+Q
            yk=lc[t]-Hm@st; S=Hm@P@Hm.T+R; Kk=(P@Hm.T)/S
            st=st+(Kk.flatten()*yk); P=(np.eye(2)-Kk@Hm)@P
            drift[t]=st[1]                       # filtered drift AT t (uses data <=t only)
        pred=(drift>0).astype(int)               # predict next-1m up iff filtered drift>0
        aucs[k]=float(roc_auc_score(y, drift))   # AUC of drift as score
    res["kalman_drift"]={k:{"auc":aucs[k]} for k in aucs}
    print(f"[kalman] forward-filter drift sign AUC: "+", ".join(f"{k} {aucs[k]:.4f}" for k in aucs),flush=True)

def main():
    res={"key":"USDJPY.1m","tier":"C state-space coverage","stride_arf":STRIDE,
         "note":"online-ARF keystone (genuine-efficiency confirmator) + Kalman forward-filter (trap#1-safe)"}
    t0=time.time()
    kalman(res)
    online_arf(res)
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[statespace] saved {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
