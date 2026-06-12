"""USDJPY 1-MIN — gotobi / Tokyo-fix CALENDAR as GBM conditioning feature (Tier-N UJ-N5).

SCOPE: USDJPY · 1m. Standalone gotobi/fix flow is already null (measured). This tests the weaker claim:
JST calendar feats (gotobi-day, day-of-month, minutes-to-Tokyo-fix, JST minute/hour) added to the base
239 let the GBM extract a CONDITIONAL signed effect (when any faint dip-buy fires). Prior .04 long-shot.
KILL if VAL AUC rise < .003 over base AND no held-out UP cov2% CI-lo clears .541.
Usage: ~/binary-algo-venv/bin/python usdjpy_1m_gotobi.py [stride]
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from usdjpy_1m_base import covcurve, side_eval, nonoverlap_chrono, boot, mk_lgb

PAIR="USDJPY"; FEAT=H.FEAT_DIR; BASE=H.feature_cols(PAIR); BE=0.541
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],
     "test24":["2024"],"test25":["2025"],"oos":["2026"]}
STRIDE=int(sys.argv[1]) if len(sys.argv)>1 else 6
RESULT=f"usdjpy_1m_gotobi_s{STRIDE}_result.json"
CAL=["jst_hour","jst_min","dom","gotobi","mins_to_fix","near_fix"]

def cal_feats(idx):
    jst=idx + pd.Timedelta(hours=9)          # approx JST (ignores DST nuance; fix window flagged broadly)
    dom=jst.day.values.astype(float)
    gob=np.isin(jst.day.values,[5,10,15,20,25,30]).astype(float)
    # Tokyo fix ~09:55 JST; minutes-to-fix within the day (signed, clipped)
    mins=(jst.hour.values*60+jst.minute.values).astype(float)
    fixm=9*60+55
    m2f=np.clip(fixm-mins,-120,120)
    near=((mins>=fixm-15)&(mins<=fixm+5)).astype(float)
    return {"jst_hour":jst.hour.values.astype(float),"jst_min":jst.minute.values.astype(float),
            "dom":dom,"gotobi":gob,"mins_to_fix":m2f,"near_fix":near}

def build(years, stride=1):
    Xs=[];ys=[];mv=[];tss=[]
    for y in years:
        p=f"{FEAT}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p,columns=BASE+["close"]); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float); ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        contig=np.zeros(n,bool); contig[:n-1]=(ts[1:]-ts[:-1])==60
        fr=np.full(n,np.nan); fr[:n-1]=c[1:]/c[:-1]-1.0
        cf=cal_feats(d.index); X=d[BASE].copy()
        for k in CAL: X[k]=cf[k].astype("float32")
        X=X.astype("float32"); keepf=d[BASE].isna().mean(axis=1).values<0.5
        valid=contig&np.isfinite(fr)&keepf
        idx=np.where(valid)[0]
        if stride>1: idx=idx[::stride]
        Xs.append(X.iloc[idx]); ys.append((fr[idx]>0).astype(int)); mv.append(fr[idx]!=0); tss.append(ts[idx])
    return pd.concat(Xs), np.concatenate(ys), np.concatenate(mv), np.concatenate(tss)

def main():
    t0=time.time(); cols=BASE+CAL
    res={"key":"USDJPY.1m","model":"base239 + JST gotobi/fix calendar conditioning (UJ-N5)","splits":SPL,
         "falsifier":{"KILL_if":"VAL AUC rise <.003 over base .5234 AND no UP cov2% CI-lo>.541","prior":0.04}}
    Xtr,ytr,mtr,_=build(SPL["train"],STRIDE); Xva,yva,mva,tsv=build(SPL["val"])
    print(f"[gotobi] train={int(mtr.sum()):,} feats={len(cols)} build={time.time()-t0:.0f}s",flush=True)
    L=mk_lgb(num_leaves=255); L.fit(Xtr[cols][mtr],ytr[mtr],eval_set=[(Xva[cols][mva],yva[mva])],eval_metric="auc",
        callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pva=L.predict_proba(Xva[cols])[:,1]; val_auc=float(roc_auc_score(yva[mva],pva[mva]))
    fi=dict(zip(cols,L.feature_importances_)); order=sorted(fi,key=lambda c:-fi[c])
    ranks={k:order.index(k)+1 for k in CAL}
    print(f"[gotobi] VAL AUC={val_auc:.4f} (base .5234; rise {val_auc-0.5234:+.4f}) cal ranks: {ranks}",flush=True)
    res["val_auc"]=val_auc; res["cal_ranks"]=ranks; res["years"]={}
    for w in ("test24","test25","oos"):
        Xw,yw,mw,tsw=build(SPL[w]); pr=L.predict_proba(Xw[cols])[:,1]; cc=covcurve(pr,yw,mw,tsw)
        res["years"][w]={"moved_auc":float(roc_auc_score(yw[mw],pr[mw])),"covcurve":cc}
        def g(cov,side): d=cc[cov].get(side,{}); return f"{d.get('wr')}(n{d.get('n')})"
        print(f"=== {w} === AUC={res['years'][w]['moved_auc']:.4f} | UP cov2%:{g('0.02','UP')} | DOWN cov2%:{g('0.02','DOWN')}",flush=True)
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[gotobi] saved {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
