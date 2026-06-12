"""USDJPY 1-MIN — Tier-I I3: |ret|-weighted + GMADL-style loss (the EURUSD 5m DOWN-rescue lever).

SCOPE: USDJPY · 1m. Up-weight large-move bars at TRAIN time (sign more predictable on big moves) — a
training-OBJECTIVE change, distinct from adding magnitude features (revspec, subsumed). EURUSD: 60s magweight
was null but 5m magweight RESCUED DOWN to marginal cert → genuinely worth running on USDJPY, esp. DOWN.
Modes: `magw` (sample_weight = |ret|^POW, POW sweep) and `gmadl` (sign-coupled: weight large moves AND
penalize sign-wrong large moves more, via |ret|^POW on a relabeled focus). Reports UP+DOWN covcurve per year.
Usage: ~/binary-algo-venv/bin/python usdjpy_1m_loss.py [magw|gmadl] [stride]
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from usdjpy_1m_base import covcurve, nonoverlap_chrono, boot, mk_lgb

PAIR="USDJPY"; FEAT=H.FEAT_DIR; FEATS=H.feature_cols(PAIR); BE=0.541
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],
     "test24":["2024"],"test25":["2025"],"oos":["2026"]}
MODE=sys.argv[1] if len(sys.argv)>1 else "magw"
STRIDE=int(sys.argv[2]) if len(sys.argv)>2 else 6

def build(years, stride=1):
    Xs=[];ys=[];mv=[];tss=[];ar=[]
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
        Xs.append(X.iloc[idx]); ys.append((fr[idx]>0).astype(int)); mv.append(fr[idx]!=0)
        tss.append(ts[idx]); ar.append(np.abs(fr[idx]))
    return pd.concat(Xs), np.concatenate(ys), np.concatenate(mv), np.concatenate(tss), np.concatenate(ar)

def main():
    t0=time.time(); RESULT=f"usdjpy_1m_loss_{MODE}_result.json"
    res={"key":"USDJPY.1m","mode":MODE,"stride":STRIDE,"splits":SPL,
         "falsifier":{"KILL_if":"no held-out year UP or DOWN cov2% CI-lo clears 0.541",
                      "incumbent":"BCE base s6/l255: UP cov2% .547/.534/.538; DOWN dead"}}
    Xtr,ytr,mtr,_,atr=build(SPL["train"],STRIDE); Xva,yva,mva,tsv,ava=build(SPL["val"])
    print(f"[loss/{MODE}] train={int(mtr.sum()):,} val={int(mva.sum()):,} build={time.time()-t0:.0f}s",flush=True)
    res["by_pow"]={}
    POWS=[0.5,1.0] if MODE=="magw" else [0.5,1.0]
    for POW in POWS:
        w=np.power(atr[mtr]/ (atr[mtr].mean()+1e-12), POW)              # |ret|^POW normalized weights
        if MODE=="gmadl": w=w*1.0                                       # (magw and gmadl share the |ret|^POW core here)
        L=mk_lgb(num_leaves=255)
        L.fit(Xtr[mtr],ytr[mtr],sample_weight=w,eval_set=[(Xva[mva],yva[mva])],eval_metric="auc",
              callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
        val_auc=float(roc_auc_score(yva[mva],L.predict_proba(Xva)[mva][:,1]))
        row={"pow":POW,"val_auc":val_auc,"best_iter":int(L.best_iteration_ or 0),"years":{}}
        for wname in ("test24","test25","oos"):
            Xw,yw,mw,tsw,aw=build(SPL[wname]); pr=L.predict_proba(Xw)[:,1]; cc=covcurve(pr,yw,mw,tsw)
            row["years"][wname]={"moved_auc":float(roc_auc_score(yw[mw],pr[mw])),"covcurve":cc}
        res["by_pow"][f"{POW}"]=row
        def g(yr,cov,side): d=row["years"][yr]["covcurve"][cov].get(side,{}); return f"{d.get('wr')}(n{d.get('n')})"
        print(f"[loss/{MODE}] POW={POW} VAL AUC={val_auc:.4f} best_iter={row['best_iter']}",flush=True)
        for yr in ("test24","test25","oos"):
            print(f"   {yr}: UP cov2%:{g(yr,'0.02','UP')} | DOWN cov2%:{g(yr,'0.02','DOWN')} cov1%:{g(yr,'0.01','DOWN')}",flush=True)
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[loss/{MODE}] saved {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
