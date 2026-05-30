"""V24 — push the VALIDATED EURUSD compress×NY pocket (~0.60) toward 0.75 with orthogonal in-pocket
conditions, gating every candidate on the 4-window stability test (TEST24, TEST25, OOS-H1, OOS-H2).

Base pocket = EURUSD, compression (15m_bb_width <= TRAIN q33), NY session. Within it the global LGBM
gives ~0.60 selective. We test whether adding ONE orthogonal condition lifts accuracy *in all four
windows at once* (real) vs only some (overfit). Confidence threshold frozen on VAL within each pocket.
Target = up/down binary endpoint sign. Coverage kept moderate so OOS n stays usable per window.
"""
import numpy as np, pandas as pd, os
import lightgbm as lgb
import harness as H

base=list(H.feature_cols("EURUSD")); HOR=15; STRIDE=3
def load(years, stride=1):
    parts=[]
    for y in years:
        p=f"{H.FEAT_DIR}/EURUSD_{y}.parquet"
        if not os.path.exists(p): continue
        df=pd.read_parquet(p,columns=base+H.META_COLS); df=df[~df.index.duplicated(keep="last")]
        idx=df.index; c=df["close"].values; n=len(c)
        secs=idx.values.astype("datetime64[s]").astype("int64")
        contig=np.zeros(n,bool)
        if n>HOR: contig[:n-HOR]=(secs[HOR:]-secs[:-HOR])==HOR*60
        fwd=np.full(n,np.nan); fwd[:n-HOR]=c[HOR:]; ret=fwd/c-1.0
        yv=(ret>0).astype(float); valid=contig&np.isfinite(ret)&(ret!=0)
        d=df.loc[valid,base].copy(); d["_y"]=yv[valid]; d["_h"]=idx[valid].hour
        parts.append(d.iloc[::stride] if stride>1 else d)
    return pd.concat(parts)

TR=load(H.SPLITS["train"],STRIDE); VA=load(H.SPLITS["val"])
Q={q:np.nanpercentile(TR["15m_bb_width"].values.astype(float),q) for q in (10,20,33)}
def f(D,c): return D[c].values.astype(float)
def base_mask(D): return (f(D,"15m_bb_width")<=Q[33]) & (f(D,"sess_ny")>0.5)

# orthogonal enhancer masks (each ANDed onto base)
ENH={
 "base(compress×NY)":   lambda D: np.ones(len(D),bool),
 "+tightcompress(q20)": lambda D: f(D,"15m_bb_width")<=Q[20],
 "+tightcompress(q10)": lambda D: f(D,"15m_bb_width")<=Q[10],
 "+NYmorning(13-16)":   lambda D: (D["_h"].values>=13)&(D["_h"].values<16),
 "+MTFalign_hi":        lambda D: np.abs(f(D,"mtf_trend_align"))>=np.nanmedian(np.abs(f(VA,"mtf_trend_align"))),
 "+stretch_OB/OS":      lambda D: (f(D,"15m_bb_pctb")>0.8)|(f(D,"15m_bb_pctb")<0.2),
 "+lowvol_z":           lambda D: f(D,"vol_z")<=np.nanmedian(f(VA,"vol_z")),
 "+rsi_extreme":        lambda D: (f(D,"15m_rsi")>=65)|(f(D,"15m_rsi")<=35),
}

L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=255,
    min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=10,
    n_estimators=3000,n_jobs=20,verbosity=-1)
ytr=TR["_y"].astype(int).values; yva=VA["_y"].astype(int).values
L.fit(TR[base].astype("float32"),ytr,eval_set=[(VA[base].astype("float32"),yva)],
      eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
def prob(D): return L.predict_proba(D[base].astype("float32"))[:,1]
pva=prob(VA)

WINDOWS={"TEST24":load(["2024"]),"TEST25":load(["2025"]),"OOS26":load(["2026"])}
WP={k:prob(v) for k,v in WINDOWS.items()}
# split OOS26 into halves
oo=WINDOWS["OOS26"]; h=len(oo)//2
WINDOWS["OOS_H1"]=oo.iloc[:h]; WINDOWS["OOS_H2"]=oo.iloc[h:]
WP["OOS_H1"]=WP["OOS26"][:h]; WP["OOS_H2"]=WP["OOS26"][h:]
ORDER=["TEST24","TEST25","OOS_H1","OOS_H2"]

def acc_sel(y,p,thr):
    m=np.abs(p-0.5)>=thr
    if m.sum()==0: return np.nan,0
    return ((p[m]>0.5).astype(int)==y[m]).mean(), int(m.sum())

import sys
COV=float(sys.argv[1]) if len(sys.argv)>1 else 0.5  # within-pocket confidence coverage
print(f"comp q33={Q[33]:.6g} q20={Q[20]:.6g} q10={Q[10]:.6g} | within-pocket conf cov={COV:.0%}\n")
print(f"{'enhancer':>22} | "+" ".join(f"{w:>13}" for w in ORDER)+"  all>=.70?")
for name,em in ENH.items():
    # VAL pocket + threshold
    mv=base_mask(VA)&em(VA)
    if mv.sum()<200:
        print(f"{name:>22} | (VAL pocket too small n={int(mv.sum())})"); continue
    confv=np.abs(pva[mv]-0.5); thr=np.quantile(confv,1-COV)
    cells=[]; oks=[]
    for w in ORDER:
        D=WINDOWS[w]; p=WP[w]; m=base_mask(D)&em(D)
        a,n=acc_sel(D["_y"].astype(int).values[m],p[m],thr)
        cells.append(f"{a:.3f}(n{n})"); oks.append((not np.isnan(a)) and a>=0.70 and n>=30)
    flag="**" if all(oks) else ""
    print(f"{name:>22} | "+" ".join(f"{c:>13}" for c in cells)+f"   {flag}",flush=True)
print("\n** = >=0.70 in ALL four independent windows.  V24-STACK DONE")
