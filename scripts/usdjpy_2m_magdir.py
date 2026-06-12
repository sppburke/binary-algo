"""USDJPY 2-MIN — MAGNITUDE→DIRECTION bridge (ACTUALLY RUN the sign-invariance test, not argue it).

SCOPE: USDJPY · 2m. The magnitude model is STRONG (magAUC ~.78). Sign-invariance theorem predicts big-move bars are
NOT more sign-predictable — but RUN it: on the pooled DIRECTION model's confident bars, does restricting to bars where
the MAGNITUDE model predicts a big 2m move (top-magnitude quantile) RAISE the direction win-rate? Mechanism (if any):
the reversion edge may be stronger on big moves (big dip → stronger bounce) — the OPPOSITE conditioning to the (killed)
compression/small-move filter. Two probes per year: (a) direction win-rate on high-mag vs low-mag confident bars;
(b) mag-gated UP/DOWN win-rate vs ungated. Falsifier: KILL if high-mag confident-bar UP win-rate is NOT CI-separated
above low-mag (theorem holds) AND mag-gated UP cov2% does not beat C1 .547/.546/.570.
Usage: ~/binary-algo-venv/bin/python usdjpy_2m_magdir.py [stride=42]
"""
import os, sys, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from usdjpy_2m_base import nonoverlap_chrono, boot, mk_lgb
from usdjpy_2m_xpair import build_pair, build_pool, PAIRS

TARGET="USDJPY"; HOR=2; STEP=60; GAP=HOR*STEP; BE=0.541
FEAT=H.FEAT_DIR; FEATS=H.feature_cols(TARGET)
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],
     "test24":["2024"],"test25":["2025"],"oos":["2026"]}
STRIDE=int(sys.argv[1]) if len(sys.argv)>1 and sys.argv[1].isdigit() else 42
RESULT="usdjpy_2m_magdir_result.json"

def build_pair_mag(pair, years, stride):
    """Like build_pair but also returns |ret| (abs forward return) for the magnitude label."""
    Xs=[];ys=[];mv=[];tss=[];ars=[]
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
    return pd.concat(Xs), np.concatenate(ys), np.concatenate(mv), np.concatenate(tss), np.concatenate(ars)

def main():
    t0=time.time()
    # direction: pooled; magnitude: pooled (|ret|>=train-Q75)
    Xtr,ytr,mtr=build_pool(SPL["train"],STRIDE)
    Xtr_m=[];atr_m=[]
    for p in PAIRS:
        r=build_pair_mag(p,SPL["train"],STRIDE); Xtr_m.append(r[0]); atr_m.append(r[4])
    Xtr_m=pd.concat(Xtr_m); atr_m=np.concatenate(atr_m)
    magthr=float(np.quantile(atr_m,0.75)); ymag=(atr_m>=magthr).astype(int)
    print(f"[magdir] train dir={int(mtr.sum()):,} build={time.time()-t0:.0f}s",flush=True)
    Ld=mk_lgb(num_leaves=255); Ld.fit(Xtr[mtr],ytr[mtr])
    Lm=mk_lgb(num_leaves=255); Lm.fit(Xtr_m,ymag)
    print(f"[magdir] dir+mag models fit {time.time()-t0:.0f}s",flush=True)
    res={"key":"USDJPY.2m","model":"mag-gated direction (pooled dir x pooled mag-Q75)","stride":STRIDE,"magthr":magthr,
         "falsifier":{"KILL_if":"high-mag confident UP wr NOT CI-sep above low-mag AND mag-gated UP cov2% not > C1"},"years":{}}
    for w in ("test24","test25","oos"):
        Xw,yw,mw,tsw,aw=build_pair_mag(TARGET,SPL[w],1)
        pdir=Ld.predict_proba(Xw)[:,1]; pmag=Lm.predict_proba(Xw)[:,1]
        conf=np.abs(pdir-0.5); thr=float(np.quantile(conf,1-0.02))   # cov2% direction-confident
        # split confident UP bars by predicted magnitude (median of pmag among confident)
        cand=(conf>=thr)&np.isfinite(aw)&(pdir>0.5)
        idx=nonoverlap_chrono(tsw,cand)
        if len(idx)>10:
            pm=pmag[idx]; med=np.median(pm); hi=pm>=med; lo=~hi
            def wr(sel):
                if sel.sum()<10: return None
                pr=(pdir[idx][sel]>0.5).astype(int); yl=(yw[idx][sel]); mo=mw[idx][sel]
                wins=((pr==yl)&mo).astype(float); lo_,hi_=boot(wins); return [int(sel.sum()),round(float(wins.mean()),4),[round(lo_,3),round(hi_,3)]]
            res["years"][w]={"UP_highmag":wr(hi),"UP_lowmag":wr(lo)}
        else:
            res["years"][w]={"UP_highmag":None,"UP_lowmag":None}
        hm=res["years"][w]["UP_highmag"]; lm=res["years"][w]["UP_lowmag"]
        print(f"=== {w} === UP high-mag {hm} | UP low-mag {lm}",flush=True)
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[magdir] saved {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
