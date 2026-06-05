"""USDJPY — cross-horizon edge probe + 15m→1m STACK (Tier-A A5).

SCOPE: USDJPY. A5 cross-horizon stack front-loads a higher-horizon edge onto 1m. Prerequisite: does USDJPY
HAVE a higher-horizon direction edge to front-load? Probe base-GBM direction at H=5 and H=15 (USDJPY own
forward sign, contiguous H*60s, ties LOSE). If a parent horizon clears breakeven CI-lo, build the stack
(parent prob as a 1m feature/gate). If all near-efficient, A5 is moot (no parent edge) — documented null.
Usage: ~/binary-algo-venv/bin/python usdjpy_xhorizon.py [H1 H2 ...]   (default 5 15)
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from usdjpy_1m_base import nonoverlap_chrono, boot, mk_lgb

PAIR="USDJPY"; FEAT=H.FEAT_DIR; FEATS=H.feature_cols(PAIR); BE=0.541
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],
     "test24":["2024"],"test25":["2025"],"oos":["2026"]}
HORS=[int(x) for x in sys.argv[1:]] or [5,15]
STRIDE=12
RESULT="usdjpy_xhorizon_result.json"

def build(years, hor, stride=1):
    Xs=[];ys=[];mv=[];tss=[]
    gap=hor*60
    for y in years:
        p=f"{FEAT}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p,columns=FEATS+["close"]); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float); ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        contig=np.zeros(n,bool);
        if n>hor: contig[:n-hor]=(ts[hor:]-ts[:-hor])==gap
        fr=np.full(n,np.nan)
        if n>hor: fr[:n-hor]=c[hor:]/c[:-hor]-1.0
        X=d[FEATS].astype("float32"); keepf=X.isna().mean(axis=1).values<0.5
        valid=contig&np.isfinite(fr)&keepf
        idx=np.where(valid)[0]
        if stride>1: idx=idx[::stride]
        Xs.append(X.iloc[idx]); ys.append((fr[idx]>0).astype(int)); mv.append(fr[idx]!=0); tss.append(ts[idx])
    return pd.concat(Xs), np.concatenate(ys), np.concatenate(mv), np.concatenate(tss), gap

def evalyear(pr,y,mv,ts,gap,cov=0.05):
    conf=np.abs(pr-0.5); thr=float(np.quantile(conf,1-cov)); cand=conf>=thr
    sel=nonoverlap_chrono(ts,cand,gap)
    if len(sel)==0: return None
    pred=(pr[sel]>0.5).astype(int); win=((pred==y[sel])&mv[sel]).astype(float)
    lo,hi=boot(win); up=pred==1; dn=pred==0
    def s(m):
        if m.sum()==0: return None
        l,h=boot(win[m]); return [round(float(win[m].mean()),4),int(m.sum()),round(l,4)]
    return {"COMB":[round(float(win.mean()),4),int(len(sel)),round(lo,4)],"UP":s(up),"DOWN":s(dn)}

def main():
    t0=time.time(); res={"key":"USDJPY","tier":"A5 cross-horizon probe","stride":STRIDE,"by_H":{}}
    for hor in HORS:
        Xtr,ytr,mtr,_,_=build(SPL["train"],hor,STRIDE); Xva,yva,mva,tsv,gap=build(SPL["val"],hor)
        L=mk_lgb(num_leaves=255); L.fit(Xtr[mtr],ytr[mtr],eval_set=[(Xva[mva],yva[mva])],eval_metric="auc",
            callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
        val_auc=float(roc_auc_score(yva[mva],L.predict_proba(Xva)[mva][:,1]))
        row={"val_auc":val_auc,"years":{}}
        for w in ("test24","test25","oos"):
            Xw,yw,mw,tsw,_=build(SPL[w],hor); pr=L.predict_proba(Xw)[:,1]
            row["years"][w]={"auc":float(roc_auc_score(yw[mw],pr[mw])),"cov05":evalyear(pr,yw,mw,tsw,gap,0.05)}
        res["by_H"][f"{hor}"]=row
        print(f"[xhor] H={hor} VAL AUC={val_auc:.4f}",flush=True)
        for w in row["years"]:
            yy=row["years"][w]; print(f"   {w}: AUC={yy['auc']:.4f} cov5% {yy['cov05']}",flush=True)
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[xhor] saved {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
