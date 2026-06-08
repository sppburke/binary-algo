"""USDJPY 30-MINUTE binary DIRECTION — single-pair base (A5 cross-horizon PARENT for the 15m stack).

SCOPE: USDJPY · 30m. Built to serve as the cross-horizon parent for the 15m A5 stack (longer-horizon drift
sign may add orthogonal info to the certified 15m NY edge). Same discipline as usdjpy_15m_base. label =
sign(close[t+30]-close[t]), ts[t+30]-ts[t]==1800 (30 clean 60s steps). Also reports NY-session split.

Usage: ~/binary-algo-venv/bin/python usdjpy_30m_base.py [stride] [leaves]
"""
import os, sys, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from sessions import session_mask

PAIR="USDJPY"; HOR=30; STEP=60; GAP=HOR*STEP; BE=0.541
FEAT=H.FEAT_DIR; FEATS=H.feature_cols(PAIR)
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],"test24":["2024"],"test25":["2025"],"oos":["2026"]}
def _argint(i,d): return int(sys.argv[i]) if len(sys.argv)>i and str(sys.argv[i]).isdigit() else d
TR_STRIDE=_argint(1,6); NUM_LEAVES=_argint(2,127)
RESULT="usdjpy_30m_base_result.json" if (TR_STRIDE==6 and NUM_LEAVES==127) else f"usdjpy_30m_base_s{TR_STRIDE}_l{NUM_LEAVES}_result.json"

def build(years, stride=1):
    Xs=[];ys=[];mv=[];tss=[]
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
        Xs.append(X.iloc[idx]); ys.append((fr[idx]>0).astype(int)); mv.append(moved[idx]); tss.append(ts[idx])
    return pd.concat(Xs), np.concatenate(ys), np.concatenate(mv), np.concatenate(tss)

def nonoverlap_chrono(ts, mask, gap=GAP):
    take=[];block=-1
    for i in np.where(mask)[0]:
        if ts[i]<block: continue
        take.append(i); block=int(ts[i])+gap
    return np.array(take,dtype=int)
def boot(corr, nb=5000, seed=7):
    corr=np.asarray(corr,float)
    if len(corr)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(corr)
    a=np.array([corr[rng.integers(0,n,n)].mean() for _ in range(nb)]); return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))
def side_eval(pr,y,moved,ts,thr):
    conf=np.abs(pr-0.5); tr=nonoverlap_chrono(ts,conf>=thr)
    if len(tr)==0: return None
    pred=(pr[tr]>0.5).astype(int); win=((pred==y[tr])&moved[tr]).astype(float); out={}
    lo,hi=boot(win); out["COMBINED"]={"n":int(len(tr)),"wr":float(win.mean()),"ci":[lo,hi]}
    for nm,msk in (("UP",pred==1),("DOWN",pred==0)):
        if msk.sum()>0: lo,hi=boot(win[msk]); out[nm]={"n":int(msk.sum()),"wr":float(win[msk].mean()),"ci":[lo,hi]}
        else: out[nm]={"n":0,"wr":float("nan"),"ci":[float("nan")]*2}
    return out
def mk_lgb(n=3000,num_leaves=127):
    return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=num_leaves,
        min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,n_estimators=n,n_jobs=16,verbosity=-1)

def main():
    t0=time.time()
    res={"key":"USDJPY.30m","model":"single-pair base GBM (A5 parent)","tr_stride":TR_STRIDE,"num_leaves":NUM_LEAVES,
         "falsifier":{"KILL_if":"VAL moved-AUC<=.515 OR (as a standalone edge) no held-out year CI-lo clears .541"}}
    Xtr,ytr,mtr,tstr=build(SPL["train"],TR_STRIDE); Xva,yva,mva,tsv=build(SPL["val"])
    print(f"[base30m] s{TR_STRIDE} train={int(mtr.sum()):,} val={int(mva.sum()):,} build={time.time()-t0:.0f}s",flush=True)
    L=mk_lgb(num_leaves=NUM_LEAVES); L.fit(Xtr[mtr],ytr[mtr],eval_set=[(Xva[mva],yva[mva])],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pva=L.predict_proba(Xva)[:,1]; val_auc=float(roc_auc_score(yva[mva],pva[mva])); res["val_auc"]=val_auc
    print(f"[base30m] VAL moved-AUC={val_auc:.4f}",flush=True)
    L.booster_.save_model("models/m30_USDJPY_direction_lgb.txt")
    res["years"]={}
    for w in ("test24","test25","oos"):
        Xw,yw,mw,tsw=build(SPL[w]); pr=L.predict_proba(Xw)[:,1]
        auc=float(roc_auc_score(yw[mw],pr[mw]))
        # all-session + NY split at cov3%
        confv=np.abs(pva-0.5); thr=float(np.quantile(confv,1-0.03))
        gall=side_eval(pr,yw,mw,tsw,thr)
        ny=session_mask(tsw,"ny"); wi=np.where(ny)[0]
        gny=side_eval(pr[wi],yw[wi],mw[wi],tsw[wi],thr)
        res["years"][w]={"moved_auc":auc,"all_cov3":{k:[gall[k]["n"],round(gall[k]["wr"],4)] for k in ("COMBINED","UP","DOWN")} if gall else None,
                         "ny_cov3":{k:[gny[k]["n"],round(gny[k]["wr"],4)] for k in ("COMBINED","UP","DOWN")} if gny else None}
        print(f"  {w}: AUC={auc:.4f} | all COMB {res['years'][w]['all_cov3']['COMBINED'] if gall else None} | NY COMB {res['years'][w]['ny_cov3']['COMBINED'] if gny else None}",flush=True)
    os.makedirs("models",exist_ok=True)
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[base30m] done {time.time()-t0:.0f}s -> {RESULT}",flush=True)

if __name__=="__main__":
    import os; os.makedirs("models",exist_ok=True); main()
