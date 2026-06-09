"""AUDUSD 15m NY — cross-horizon STACK viability corr-check (confirm-or-kill the last on-disk structural lever).

SCOPE: AUDUSD · 15m · NY. The R2 backlog's cross-horizon stack (front-load a longer-horizon parent's confident
SIGN into the 15m child) is the strongest direction method elsewhere (EURUSD 5m) BUT was NULL for USDJPY 15m
(30m parent collinear with the 15m child, corr .957 — same own-pair GBM, no orthogonal sign). AUDUSD is the same
own-pair-specific case (A6 confirmed). FAST-KILL: train a 30m-NY parent + 15m-NY child on the SAME train, predict
VAL NY, measure corr(p30, p15). corr > 0.90 -> collinear -> cross-horizon stack adds no orthogonal sign -> SUBSUMED
(no full stack/CPCV needed). Also reports the 30m parent's own NY VAL AUC + cov2 selacc (must be a real edge to be
worth stacking).

Usage: ~/binary-algo-venv/bin/python audusd_15m_xhorizon.py
"""
import os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from sessions import session_mask

PAIR="AUDUSD"; STEP=60; BE=0.541; FEAT=H.FEAT_DIR; FEATS=H.feature_cols(PAIR)
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"]}
RESULT="audusd_15m_xhorizon_result.json"

def build_h(years, HOR, stride=1):
    GAP=HOR*STEP; Xs=[]; ys=[]; mv=[]; tss=[]
    for y in years:
        p=f"{FEAT}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p, columns=FEATS+["close"]); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float); ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        contig=np.zeros(n,bool); contig[:n-HOR]=(ts[HOR:]-ts[:-HOR])==GAP
        fr=np.full(n,np.nan); fr[:n-HOR]=c[HOR:]/c[:-HOR]-1.0
        X=d[FEATS].astype("float32"); keepf=X.isna().mean(axis=1).values<0.5
        valid=contig & np.isfinite(fr) & keepf; moved=valid&(fr!=0.0)
        idx=np.where(valid)[0]
        if stride>1: idx=idx[::stride]
        Xs.append(X.values[idx]); ys.append((fr[idx]>0).astype(int)); mv.append(moved[idx]); tss.append(ts[idx])
    return np.concatenate(Xs),np.concatenate(ys),np.concatenate(mv),np.concatenate(tss)

def mk(): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=127,
    min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,n_estimators=2000,n_jobs=20,verbosity=-1)

def fit_predict(HOR):
    Xtr,ytr,mtr,ttr=build_h(SPL["train"],HOR,6); s=session_mask(ttr,"ny"); itr=mtr&s
    Xva,yva,mva,tva=build_h(SPL["val"],HOR); sv=session_mask(tva,"ny"); iva=mva&sv
    L=mk(); L.fit(Xtr[itr],ytr[itr],eval_set=[(Xva[iva],yva[iva])],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pva=L.predict_proba(Xva)[:,1]
    auc=float(roc_auc_score(yva[iva],pva[iva]))
    return pva, yva, mva, tva, sv, auc

def main():
    t0=time.time()
    print("[xhorizon] fitting 15m NY child + 30m NY parent...",flush=True)
    p15,y15,m15,t15,s15,auc15=fit_predict(15)
    p30,y30,m30,t30,s30,auc30=fit_predict(30)
    print(f"[xhorizon] 15m NY VAL AUC={auc15:.4f} | 30m NY VAL AUC={auc30:.4f} ({time.time()-t0:.0f}s)",flush=True)
    # align p30 to the 15m NY rows by timestamp (parent prob known at each 15m decision bar)
    i15=np.where(s15 & m15)[0]; i30=np.where(s30 & m30)[0]
    df30=pd.Series(p30[i30], index=t30[i30]); df30=df30[~df30.index.duplicated(keep="last")]
    al=df30.reindex(t15[i15]); ok=np.isfinite(al.values)
    pc15=p15[i15][ok]; pc30=al.values[ok]
    corr=float(np.corrcoef(pc15,pc30)[0,1])
    # 30m parent own NY cov2 selacc on VAL (does it even have an edge to stack?)
    conf30=np.abs(p30[i30]-0.5); thr=float(np.quantile(conf30,1-0.02)); sel=conf30>=thr
    acc30=float(((p30[i30][sel]>0.5).astype(int)==y30[i30][sel]).mean()) if sel.sum()>20 else float("nan")
    res={"key":"AUDUSD.15m.NY","val_auc_15m":round(auc15,4),"val_auc_30m":round(auc30,4),
         "corr_p30_p15_valny":round(corr,4),"n_aligned":int(ok.sum()),
         "parent30_ny_cov2_valacc":round(acc30,4),
         "falsifier":"KILL (subsumed) if corr(p30,p15) > 0.90 -> collinear, no orthogonal sign (USDJPY 30m parent .957)",
         "verdict":{"collinear_subsumed":bool(corr>0.90),
            "note":("corr>0.90 -> 30m parent is the same own-pair GBM read at a coarser horizon; cross-horizon stack "
                    "adds no orthogonal direction sign -> SUBSUMED (USDJPY A5 reproduced)." if corr>0.90 else
                    "corr<=0.90 -> some decorrelation; a full agreement-gated stack CPCV is warranted.")}}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[xhorizon] corr(p30,p15)={corr:.4f} (n{ok.sum()}) parent30 cov2 valacc={acc30:.4f} -> collinear_subsumed={res['verdict']['collinear_subsumed']} ({time.time()-t0:.0f}s) -> {RESULT}",flush=True)

if __name__=="__main__":
    main()
