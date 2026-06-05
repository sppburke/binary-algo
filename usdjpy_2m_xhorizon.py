"""USDJPY 2-MIN — CROSS-HORIZON stack: longer-H pooled predictions as features for the 2m model (ACTUALLY RUN).

SCOPE: USDJPY · 2m. The horizon gradient says direction SIGN strengthens with H (USDJPY 15m UP 2024 .593, A5). The
2m model sees per-bar features but not the learned slow-drift sign a longer-horizon model extracts. COMBINE: train
POOLED H=5 and H=15 models, use their P(up) as 2 extra features for the POOLED H=2 model — front-loading the slower
directional drift into the 2m trade. Mechanism carries SIGN (longer-H sign), not just magnitude.

CAUSAL split (no leakage): H5/H15 trained on A=2012-2018; their predictions feature B=2019-2021 (+ val/test). H2
trained on B (with base+xh feats). Held-out 2024-26 features come from H5/H15 trained on FULL train (2012-2021),
predicting the clean future. Frozen-gate per-year. Falsifier: KILL if xh feats rank < top-30 by gain OR no held-out
UP cov2% beats C1 .547/.546/.570 CI-separated. (Prior LOW: A5 found the 15m parent itself sub-BE OOS.)
Usage: ~/binary-algo-venv/bin/python usdjpy_2m_xhorizon.py [stride_per_pair=30]
"""
import os, sys, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from usdjpy_2m_base import side_eval, covcurve, nonoverlap_chrono, boot, mk_lgb

TARGET="USDJPY"; STEP=60; BE=0.541
PAIRS=["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCHF","USDCAD"]
FEAT=H.FEAT_DIR; FEATS=H.feature_cols(TARGET)
STRIDE=int(sys.argv[1]) if len(sys.argv)>1 and sys.argv[1].isdigit() else 30
RESULT="usdjpy_2m_xhorizon_result.json"
A_YEARS=[str(y) for y in range(2012,2019)]          # H5/H15 train (early)
B_YEARS=[str(y) for y in range(2019,2022)]          # H2 train (late) — H5/H15 unseen
FULLTRAIN=[str(y) for y in range(2012,2022)]
VAL=["2022","2023"]; HELD={"test24":["2024"],"test25":["2025"],"oos":["2026"]}

def build_pair(pair, years, hor, stride, with_feats=True):
    """For horizon `hor` (minutes): base feats (optional) + label/moved/ts. gap=hor*60."""
    gap=hor*STEP; cols=(FEATS+["close"]) if with_feats else ["close"]
    Xs=[]; ys=[]; mv=[]; tss=[]
    for y in years:
        p=f"{FEAT}/{pair}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p,columns=cols); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float); ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        contig=np.zeros(n,bool); contig[:n-hor]=(ts[hor:]-ts[:-hor])==gap
        fr=np.full(n,np.nan); fr[:n-hor]=c[hor:]/c[:-hor]-1.0
        if with_feats:
            X=d[FEATS].astype("float32"); keepf=X.isna().mean(axis=1).values<0.5
        else:
            X=None; keepf=np.ones(n,bool)
        valid=contig&np.isfinite(fr)&keepf; idx=np.where(valid)[0]
        if stride>1: idx=idx[::stride]
        Xs.append(X.iloc[idx] if with_feats else None); ys.append((fr[idx]>0).astype(int)); mv.append((fr[idx]!=0.0)); tss.append(ts[idx])
    if with_feats:
        return pd.concat(Xs), np.concatenate(ys), np.concatenate(mv), np.concatenate(tss)
    return None, np.concatenate(ys), np.concatenate(mv), np.concatenate(tss)

def train_pooled(years, hor, stride):
    Xs=[];ys=[];mv=[]
    for p in PAIRS:
        r=build_pair(p,years,hor,stride); Xs.append(r[0]); ys.append(r[1]); mv.append(r[2])
    X=pd.concat(Xs); y=np.concatenate(ys); m=np.concatenate(mv)
    L=mk_lgb(num_leaves=255); L.fit(X[m],y[m]); return L

def xh_feats(L5,L15,Xdf):
    return pd.DataFrame({"xh_h5":L5.predict_proba(Xdf[FEATS])[:,1],"xh_h15":L15.predict_proba(Xdf[FEATS])[:,1]},index=Xdf.index)

def main():
    t0=time.time()
    res={"key":"USDJPY.2m","model":"POOLED H2 + xh(H5,H15 pooled P_up) features","stride":STRIDE,
         "falsifier":{"registered_utc":"pre-OOS","KILL_if":"xh feats rank<top-30 OR no held-out UP cov2% beats C1 CI-sep"}}
    json.dump(res,open(RESULT,"w"),indent=2)
    # H5/H15 on A (for B features) and on FULLTRAIN (for held-out features)
    print("[xh] training H5/H15 pooled on A=2012-18 ...",flush=True)
    L5a=train_pooled(A_YEARS,5,STRIDE); L15a=train_pooled(A_YEARS,15,STRIDE)
    print(f"[xh] training H5/H15 pooled on FULLTRAIN ... {time.time()-t0:.0f}s",flush=True)
    L5f=train_pooled(FULLTRAIN,5,STRIDE); L15f=train_pooled(FULLTRAIN,15,STRIDE)
    # H2 train on B with xh from A-models
    print(f"[xh] building H2 train (B=2019-21) ... {time.time()-t0:.0f}s",flush=True)
    XB=[];yB=[];mB=[]
    for p in PAIRS:
        r=build_pair(p,B_YEARS,2,STRIDE); XB.append(r[0].join(xh_feats(L5a,L15a,r[0]))); yB.append(r[1]); mB.append(r[2])
    XB=pd.concat(XB); yB=np.concatenate(yB); mB=np.concatenate(mB); ALL=FEATS+["xh_h5","xh_h15"]
    # VAL (USDJPY) with A-models xh (causal: A precedes val)
    Xv,yv,mv,tsv=build_pair(TARGET,VAL,2,1); Xv=Xv.join(xh_feats(L5a,L15a,Xv))
    L=mk_lgb(num_leaves=255)
    L.fit(XB[ALL][mB],yB[mB],eval_set=[(Xv[ALL][mv],yv[mv])],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pva=L.predict_proba(Xv[ALL])[:,1]; val_auc=float(roc_auc_score(yv[mv],pva[mv]))
    fi=dict(zip(ALL,L.feature_importances_)); order=sorted(ALL,key=lambda c:-fi[c])
    res["val_auc"]=val_auc; res["xh_ranks"]={"xh_h5":order.index("xh_h5"),"xh_h15":order.index("xh_h15")}
    THR=float(np.quantile(np.abs(pva-0.5),1-0.02))
    print(f"[xh] VAL-AUC={val_auc:.4f} xh ranks h5={order.index('xh_h5')} h15={order.index('xh_h15')} /{len(ALL)} {time.time()-t0:.0f}s",flush=True)
    res["years"]={}
    for w,yrs in HELD.items():
        Xw,yw,mw,tsw=build_pair(TARGET,yrs,2,1); Xw=Xw.join(xh_feats(L5f,L15f,Xw))   # held-out uses FULLTRAIN models (causal)
        pr=L.predict_proba(Xw[ALL])[:,1]; g=side_eval(pr,yw,mw,tsw,THR); cc=covcurve(pr,yw,mw,tsw)
        res["years"][w]={"moved_auc":float(roc_auc_score(yw[mw],pr[mw])),"gate_cov2":g,"covcurve":cc}
        u=g["UP"] if g else {}; print(f"=== {w} === AUC={res['years'][w]['moved_auc']:.4f} | UP cov2% n{u.get('n')} wr={u.get('wr')} CI{u.get('ci')}",flush=True)
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[xh] saved {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
