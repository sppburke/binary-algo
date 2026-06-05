"""USDJPY 2-MIN — RECENCY-WEIGHTED pooled training (ACTUALLY RUN; motivated by the CPCV recency finding).

SCOPE: USDJPY · 2m. The CPCV showed the RECENT regime (where we deploy) is the WEAKEST (recent-block paths .52-.54 vs
older .54+). Mechanism: the pooled model fits the full 2012-2021 distribution equally, but the deploy regime is 2024-26.
Upweight recent training years (sample_weight = exp(-(maxyr - yr)/tau)) so the fit matches the deploy regime → could
lift the recent/held-out win-rate that's the binding constraint. Genuinely untested (m30_recency exists for EURUSD 30m).
Frozen-gate per-year triage; escalate to CPCV if it beats C1 .547/.546/.570 on the held-out (esp. OOS 2026).
Falsifier: KILL if no tau lifts held-out UP cov2% over C1 (CI-separated) on 2025 OR 2026.
Usage: ~/binary-algo-venv/bin/python usdjpy_2m_recency.py [stride=42]
"""
import os, sys, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from usdjpy_2m_base import side_eval, covcurve, nonoverlap_chrono, boot, mk_lgb

TARGET="USDJPY"; HOR=2; STEP=60; GAP=HOR*STEP; BE=0.541
PAIRS=["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCHF","USDCAD"]
FEAT=H.FEAT_DIR; FEATS=H.feature_cols(TARGET)
TRAIN_YEARS=list(range(2012,2022)); MAXYR=2021
SPL={"val":["2022","2023"],"test24":["2024"],"test25":["2025"],"oos":["2026"]}
STRIDE=int(sys.argv[1]) if len(sys.argv)>1 and sys.argv[1].isdigit() else 42
RESULT="usdjpy_2m_recency_result.json"

def build_pair_yr(pair, years, stride):
    """returns X, y, moved, ts, year(int per row)."""
    Xs=[];ys=[];mv=[];tss=[];yr=[]
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
        Xs.append(X.iloc[idx]); ys.append((fr[idx]>0).astype(int)); mv.append((fr[idx]!=0.0)); tss.append(ts[idx]); yr.append(np.full(len(idx),int(y)))
    if not Xs: return None
    return pd.concat(Xs), np.concatenate(ys), np.concatenate(mv), np.concatenate(tss), np.concatenate(yr)

def main():
    t0=time.time()
    Xtr=[];ytr=[];mtr=[];yrtr=[]
    for p in PAIRS:
        r=build_pair_yr(p,[str(y) for y in TRAIN_YEARS],STRIDE)
        if r: Xtr.append(r[0]); ytr.append(r[1]); mtr.append(r[2]); yrtr.append(r[4])
    Xtr=pd.concat(Xtr); ytr=np.concatenate(ytr); mtr=np.concatenate(mtr); yrtr=np.concatenate(yrtr)
    Xva,yva,mva,tsv,_=build_pair_yr(TARGET,SPL["val"],1)
    held={w:build_pair_yr(TARGET,SPL[w],1) for w in ("test24","test25","oos")}
    print(f"[recency] pooled-train={int(mtr.sum()):,} build={time.time()-t0:.0f}s",flush=True)
    res={"key":"USDJPY.2m","model":"POOLED recency-weighted (exp decay by year)","stride":STRIDE,
         "falsifier":{"KILL_if":"no tau lifts held-out UP cov2% over C1 on 2025 or 2026 CI-sep"},"by_tau":{}}
    for tau in (1e9, 6.0, 3.0, 1.5):   # 1e9 = ~uniform (control); smaller tau = sharper recency
        w_=np.exp(-(MAXYR - yrtr[mtr])/tau); w_=w_/np.mean(w_)
        L=mk_lgb(num_leaves=255); L.fit(Xtr[mtr],ytr[mtr],sample_weight=w_,
            eval_set=[(Xva[mva],yva[mva])],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
        pva=L.predict_proba(Xva)[:,1]; val_auc=float(roc_auc_score(yva[mva],pva[mva]))
        THR=float(np.quantile(np.abs(pva-0.5),1-0.02)); row={"tau":tau,"val_auc":val_auc,"years":{}}
        for w in ("test24","test25","oos"):
            Xw,yw,mw,tsw,_=held[w]; pr=L.predict_proba(Xw)[:,1]; g=side_eval(pr,yw,mw,tsw,THR)
            row["years"][w]={"moved_auc":float(roc_auc_score(yw[mw],pr[mw])),"gate_cov2":g}
        res["by_tau"][str(tau)]=row
        uu=lambda w: (row["years"][w]["gate_cov2"]["UP"] if row["years"][w]["gate_cov2"] else {})
        print(f"[recency] tau={tau} VAL-AUC={val_auc:.4f} | UP cov2% "+" ".join(f"{w}:{uu(w).get('wr')}(n{uu(w).get('n')})" for w in ('test24','test25','oos')),flush=True)
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[recency] saved {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
