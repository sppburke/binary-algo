"""V23 — Reconcile EURUSD-only ~0.63 vs pooled ~0.54: is EURUSD compress×NY STABLE across windows?

Honest protocol: EURUSD only. Train LGBM on TRAIN. Compression = TRAIN q33 of 15m_bb_width.
Threshold frozen on VAL (compression×NY subset) at fixed coverages. Then report selective accuracy +
bootstrap CI on independent windows that CANNOT share luck:
  - TEST 2024 alone, TEST 2025 alone   (two separate years)
  - OOS 2026 first-half, OOS 2026 second-half
A real edge is stable across all four; an up-fluctuation is not. Target = up/down binary endpoint sign.
"""
import numpy as np, pandas as pd, os
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
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
        if stride>1: d=d.iloc[::stride]
        parts.append(d)
    D=pd.concat(parts); return D

def cmask(D,thr): return D["15m_bb_width"].values.astype(float)<=thr
def nymask(D):    return D["sess_ny"].values.astype(float)>0.5
def sel(y,p,thr):
    m=np.abs(p-0.5)>=thr
    if m.sum()==0: return np.nan,0,None
    s=((p[m]>0.5).astype(int)==y[m]); return s.mean(),int(m.sum()),s

def report(tag,D,p,thr,rng):
    m=cmask(D,thr_bw)&nymask(D)
    a,n,s=sel(D["_y"].astype(int).values[m],p[m],thr)
    ci=""
    if s is not None and n>=20:
        b=np.array([rng.choice(s,n,replace=True).mean() for _ in range(5000)])
        ci=f"[{np.percentile(b,2.5):.3f},{np.percentile(b,97.5):.3f}]"
    print(f"   {tag:>22}: acc={a:.3f} n={n:>5} CI95={ci}",flush=True)

TR=load(H.SPLITS["train"],STRIDE); VA=load(H.SPLITS["val"])
thr_bw=np.nanpercentile(TR["15m_bb_width"].values.astype(float),33)
L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=255,
    min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=10,
    n_estimators=3000,n_jobs=20,verbosity=-1)
ytr=TR["_y"].astype(int).values; yva=VA["_y"].astype(int).values
L.fit(TR[base].astype("float32"),ytr,eval_set=[(VA[base].astype("float32"),yva)],
      eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
pv=L.predict_proba(VA[base].astype("float32"))[:,1]
print(f"comp thr={thr_bw:.6g}",flush=True)

# freeze threshold on VAL compression×NY at each coverage
mv=cmask(VA,thr_bw)&nymask(VA); confv=np.abs(pv[mv]-0.5)
rng=np.random.default_rng(11)
for cov in (0.05,0.02):
    thr=np.quantile(confv,1-cov)
    print(f"\n=== EURUSD compress×NY, VAL-frozen thr @cov~{int(cov*100)}% (thr={thr:.4f}) ===")
    for tag,yrs in [("TEST 2024",["2024"]),("TEST 2025",["2025"]),("OOS 2026 all",["2026"])]:
        D=load(yrs); p=L.predict_proba(D[base].astype("float32"))[:,1]
        report(tag,D,p,thr,rng)
        if tag=="OOS 2026 all":
            half=len(D)//2
            for sub,lab in [(D.iloc[:half],"OOS 2026 H1"),(D.iloc[half:],"OOS 2026 H2")]:
                ps=L.predict_proba(sub[base].astype("float32"))[:,1]
                report(lab,sub,ps,thr,rng)
print("\nStable across TEST24/TEST25/OOS-H1/OOS-H2 => real EURUSD edge. Scattered => fluctuation. V23 DONE")
