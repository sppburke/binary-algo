"""USDJPY 2-MIN — |ret|-weighted / GMADL-family loss on the POOLED model (E1, ACTUALLY RUN not subsumed).

SCOPE: USDJPY · 2m. The goal's improve cross-product lists {BCE · |ret|/GMADL loss}. Sign-invariance predicts
|ret|-weighting reshapes toward MOVE SIZE (magnitude), but the goal demands the pooled×loss COMBINATION be RUN, not
argued. |ret|-weighted BCE upweights big-move bars (the GMADL idea: reward correct sign weighted by realized move).
If the big moves are MORE sign-predictable for the reversion edge, the worst-regime tail could lift. Pooled train
(7 majors, own 2m sign), sample_weight=|ret|^pow, eval USDJPY frozen-gate per-year.

Falsifier: KILL if no pow lifts a held-out UP cov2% wr over C1's .547/.546/.570 (CI-separated) AND VAL-AUC not > .524.
Usage: ~/binary-algo-venv/bin/python usdjpy_2m_loss.py [stride_per_pair=42]
"""
import os, sys, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from usdjpy_2m_base import side_eval, covcurve, nonoverlap_chrono, boot, mk_lgb

TARGET="USDJPY"; HOR=2; STEP=60; GAP=HOR*STEP; BE=0.541
PAIRS=["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCHF","USDCAD"]
FEAT=H.FEAT_DIR; FEATS=H.feature_cols(TARGET)
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],
     "test24":["2024"],"test25":["2025"],"oos":["2026"]}
STRIDE=int(sys.argv[1]) if len(sys.argv)>1 and sys.argv[1].isdigit() else 42
RESULT="usdjpy_2m_loss_result.json"

def build_pair(pair, years, stride, want_absret=False):
    Xs=[]; ys=[]; mv=[]; tss=[]; ars=[]
    for y in years:
        p=f"{FEAT}/{pair}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p,columns=FEATS+["close"]); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float); ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        contig=np.zeros(n,bool); contig[:n-HOR]=(ts[HOR:]-ts[:-HOR])==GAP
        fr=np.full(n,np.nan); fr[:n-HOR]=c[HOR:]/c[:-HOR]-1.0
        X=d[FEATS].astype("float32"); keepf=X.isna().mean(axis=1).values<0.5
        valid=contig&np.isfinite(fr)&keepf; idx=np.where(valid)[0]
        if stride>1: idx=idx[::stride]
        Xs.append(X.iloc[idx]); ys.append((fr[idx]>0).astype(int)); mv.append((fr[idx]!=0.0)); tss.append(ts[idx]); ars.append(np.abs(fr[idx]))
    if not Xs: return None
    return pd.concat(Xs), np.concatenate(ys), np.concatenate(mv), np.concatenate(tss), np.concatenate(ars)

def main():
    t0=time.time()
    res={"key":"USDJPY.2m","model":"POOLED + |ret|^pow-weighted BCE (GMADL-family)","stride":STRIDE,"splits":SPL,
         "falsifier":{"registered_utc":"pre-OOS","KILL_if":"no pow lifts held-out UP cov2% over C1 .547/.546/.570 CI-sep AND VAL-AUC<=.524"}}
    json.dump(res,open(RESULT,"w"),indent=2)
    Xtr=[];ytr=[];mtr=[];atr=[]
    for p in PAIRS:
        r=build_pair(p,SPL["train"],STRIDE)
        if r: Xtr.append(r[0]); ytr.append(r[1]); mtr.append(r[2]); atr.append(r[4])
    Xtr=pd.concat(Xtr); ytr=np.concatenate(ytr); mtr=np.concatenate(mtr); atr=np.concatenate(atr)
    Xva,yva,mva,tsv,_=build_pair(TARGET,SPL["val"],1)
    print(f"[loss] pooled-train={int(mtr.sum()):,} val={int(mva.sum()):,} build={time.time()-t0:.0f}s",flush=True)
    res["by_pow"]={}
    held={w:build_pair(TARGET,SPL[w],1) for w in ("test24","test25","oos")}
    for pow_ in (0.0, 0.5, 1.0):
        w_=(atr[mtr]**pow_) if pow_>0 else None
        if w_ is not None: w_=w_/np.mean(w_)
        L=mk_lgb(num_leaves=255)
        L.fit(Xtr[mtr],ytr[mtr],sample_weight=w_,eval_set=[(Xva[mva],yva[mva])],eval_metric="auc",
              callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
        pva=L.predict_proba(Xva)[:,1]; val_auc=float(roc_auc_score(yva[mva],pva[mva]))
        confv=np.abs(pva-0.5); THR=float(np.quantile(confv,1-0.02))
        row={"pow":pow_,"val_auc":val_auc,"years":{}}
        for w in ("test24","test25","oos"):
            Xw,yw,mw,tsw,_=held[w]; pr=L.predict_proba(Xw)[:,1]
            g=side_eval(pr,yw,mw,tsw,THR); cc=covcurve(pr,yw,mw,tsw)
            row["years"][w]={"moved_auc":float(roc_auc_score(yw[mw],pr[mw])),"gate_cov2":g,"covcurve":cc}
        res["by_pow"][str(pow_)]=row
        u=lambda w: row["years"][w]["gate_cov2"]["UP"] if row["years"][w]["gate_cov2"] else {}
        print(f"[loss] pow={pow_} VAL-AUC={val_auc:.4f} | UP cov2% "
              +" ".join(f"{w}:{u(w).get('wr')}(n{u(w).get('n')})" for w in ('test24','test25','oos')),flush=True)
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[loss] saved {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
