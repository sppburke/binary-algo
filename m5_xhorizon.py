"""CROSS-HORIZON test: does the REAL 15-min edge (m15_production, 0.647) front-load into the 5-MINUTE sub-move?
The 5-min move is the first third of the 15-min move. If the 15m model's confident direction also predicts the 5-min
outcome, that's a NEW 5-min signal that inherits a genuine edge instead of fighting the 5-min noise floor.

Score the 15m ENSEMBLE's predicted sign at its confident NY×compression entries against BOTH labels:
  y15 = sign(close(t+15m)-close(t))  (native — sanity, should ~0.65)
  y5  = sign(close(t+5m)-close(t))   (the question)
Non-overlap per the TRADE horizon (5-min trades -> gap 300s). Held-out TEST24/TEST25/OOS26, CI95.
"""
import os, json, numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
import harness as H
MODELS="/home/sean/git/binary-algo/models"; PAIR="EURUSD"
base=list(H.feature_cols("EURUSD"))
SPL={"test24":["2024"],"test25":["2025"],"oos":["2026"]}

def art(n): return f"{MODELS}/m15_{PAIR}_{n}"
def load15():
    p=json.load(open(art("strategy.json")))
    L=lgb.Booster(model_file=art("direction_lgb.txt"))
    G=xgb.XGBClassifier(); G.load_model(art("direction_xgb.json"))
    C=CatBoostClassifier(); C.load_model(art("direction_cat.cbm"))
    return p,L,G,C
def proba(L,G,C,X): return (L.predict(X.values)+G.predict_proba(X)[:,1]+C.predict_proba(X.fillna(-999))[:,1])/3.0
def boot(c,nb=4000,seed=7):
    c=np.asarray(c,float)
    if len(c)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(c); a=np.array([c[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))
def nonoverlap_chrono(ts,mask,gap):
    take=[]; block=-1
    for i in np.where(mask)[0]:
        if ts[i]<block: continue
        take.append(i); block=int(ts[i])+gap
    return np.array(take,dtype=int)

def labels(df,hor):
    c=df["close"].values; n=len(c); secs=df.index.values.astype("datetime64[s]").astype("int64")
    fwd=np.full(n,np.nan)
    if n>hor:
        contig=(secs[hor:]-secs[:-hor])==hor*60
        fwd[:n-hor]=np.where(contig,c[hor:]/c[:-hor]-1.0,np.nan)
    return fwd,secs

def run():
    p,L,G,C=load15(); thr=p["conf_thr"]; bbw=p["bb_width_thr"]
    print(f"[xhorizon] 15m ensemble @ its frozen gate (15m_bb_width<={bbw:.4f} & NY, conf>={thr:.4f}) scored vs 5-min outcome")
    for HORtest,gap,lab in ((15,900,"y15 native"),(5,300,"y5  5-min")):
        print(f"\n--- target {lab} (non-overlap {gap}s) ---")
        allc=[]
        for w in ("test24","test25","oos"):
            df=pd.read_parquet(f"{H.FEAT_DIR}/{PAIR}_{SPL[w][0]}.parquet",columns=base+["close"])
            df=df[~df.index.duplicated(keep="last")]
            fwd,secs=labels(df,HORtest); valid=np.isfinite(fwd)&(fwd!=0)
            X=df.loc[valid,base].astype("float32"); pr=proba(L,G,C,X)
            y=(fwd[valid]>0).astype(int); ts=secs[valid]
            g=(df["15m_bb_width"].values.astype(float)<=bbw)&(df["sess_ny"].values.astype(float)>0.5)
            g=g[valid]; conf=np.abs(pr-0.5); m=g&(conf>=thr)
            sel=nonoverlap_chrono(ts,m,gap)
            corr=((pr[sel]>0.5).astype(int)==y[sel]).astype(float) if len(sel) else np.array([])
            acc=corr.mean() if len(sel) else float("nan"); lo,hi=boot(corr)
            print(f"=== {w} === n={len(sel)} acc={acc:.3f} CI95=[{lo:.3f},{hi:.3f}]")
            allc.append(corr)
        A=np.concatenate(allc); lo,hi=boot(A)
        print(f"=== COMBINED {lab} === n={len(A)} acc={A.mean():.3f} CI95=[{lo:.3f},{hi:.3f}]  (breakeven~0.541)")

if __name__=="__main__": run()
