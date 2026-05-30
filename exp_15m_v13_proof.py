"""V27 — PROOF pipeline: beat 67% on 15m EURUSD binary on fully held-out data, no peeking.

Everything is chosen on TRAIN(2012-21)+VAL(2022-23) ONLY:
  - model: LGBM+XGB+CatBoost ensemble, trained on 2012-21.
  - regime gate: volatility-compression depth (bb_width<=TRAIN q{10,20,33}) x NY session.
  - config (compression depth, coverage) and the confidence threshold: picked to MAXIMIZE VAL accuracy
    subject to VAL n>=150 (so the choice is stable). 2024-2026 is NEVER consulted in selection.
Then evaluate the frozen pipeline on 2024, 2025, 2026 separately AND combined (max held-out n),
with 5000x bootstrap 95% CIs. A clean >67% requires the held-out accuracy (esp. combined, tight CI)
to clear 0.67. Target = up/down 15m binary endpoint sign.
"""
import numpy as np, pandas as pd, os, time
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
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
        parts.append(d.iloc[::stride] if stride>1 else d)
    return pd.concat(parts)

t0=time.time()
TR=load(H.SPLITS["train"],STRIDE); VA=load(H.SPLITS["val"])
Q={q:np.nanpercentile(TR["15m_bb_width"].values.astype(float),q) for q in (10,20,33)}
def gate(D,q): return (D["15m_bb_width"].values.astype(float)<=Q[q])&(D["sess_ny"].values.astype(float)>0.5)
ytr=TR["_y"].astype(int).values; yva=VA["_y"].astype(int).values
print(f"load {time.time()-t0:.0f}s; train={len(TR):,}",flush=True)

# ---- ensemble trained on 2012-21 ----
L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=255,min_child_samples=200,
    subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=10,n_estimators=3000,n_jobs=20,verbosity=-1)
L.fit(TR[base].astype("float32"),ytr,eval_set=[(VA[base].astype("float32"),yva)],
      eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
G=xgb.XGBClassifier(n_estimators=2000,learning_rate=0.02,max_depth=8,subsample=0.8,colsample_bytree=0.5,
    reg_lambda=10,tree_method="hist",n_jobs=20,eval_metric="auc",early_stopping_rounds=150)
G.fit(TR[base].astype("float32"),ytr,eval_set=[(VA[base].astype("float32"),yva)],verbose=False)
C=CatBoostClassifier(iterations=2000,learning_rate=0.02,depth=8,l2_leaf_reg=10,eval_metric="AUC",
    thread_count=20,verbose=False,early_stopping_rounds=150)
C.fit(TR[base].fillna(-999),ytr,eval_set=(VA[base].fillna(-999),yva))
def prob(D):
    X=D[base].astype("float32")
    return (L.predict_proba(X)[:,1]+G.predict_proba(X)[:,1]+C.predict_proba(D[base].fillna(-999))[:,1])/3.0
pva=prob(VA)
print(f"ensemble trained {time.time()-t0:.0f}s; VAL AUC={roc_auc_score(yva,pva):.4f}",flush=True)

# ---- SELECT config (depth, coverage) + threshold on VAL only ----
best=None
for q in (10,20,33):
    gv=gate(VA,q)
    if gv.sum()<300: continue
    confv=np.abs(pva[gv]-0.5)
    for cov in (0.10,0.05,0.02):
        thr=np.quantile(confv,1-cov)
        m=gv&(np.abs(pva-0.5)>=thr)
        if m.sum()<150: continue
        acc=((pva[m]>0.5).astype(int)==yva[m]).mean()
        if best is None or acc>best[0]:
            best=(acc,q,cov,thr,int(m.sum()))
accV,Q_SEL,COV_SEL,THR,nV=best
print(f"\nSELECTED on VAL: compress(q{Q_SEL})×NY @cov{COV_SEL:.0%}  VALacc={accV:.3f} (nVAL={nV}, thr={THR:.4f})",flush=True)

# ---- FROZEN evaluation on held-out 2024 / 2025 / 2026 + combined ----
rng=np.random.default_rng(2024)
def evalyear(years,label):
    D=load(years); p=prob(D); y=D["_y"].astype(int).values
    m=gate(D,Q_SEL)&(np.abs(p-0.5)>=THR)
    if m.sum()==0: print(f"   {label:>14}: n=0"); return None
    s=((p[m]>0.5).astype(int)==y[m]); acc=s.mean(); n=int(m.sum())
    b=np.array([rng.choice(s,n,replace=True).mean() for _ in range(5000)])
    lo,hi=np.percentile(b,[2.5,97.5])
    flag=" >67% ✓" if lo>0.67 else (" >67%(pt)" if acc>0.67 else "")
    print(f"   {label:>14}: acc={acc:.3f}  n={n:>4}  CI95=[{lo:.3f},{hi:.3f}]{flag}",flush=True)
    return s
print("\n==== FROZEN pipeline on HELD-OUT data (never used in selection) ====")
s24=evalyear(["2024"],"2024"); s25=evalyear(["2025"],"2025"); s26=evalyear(["2026"],"2026")
# combined
alls=[x for x in (s24,s25,s26) if x is not None]
if alls:
    S=np.concatenate(alls); n=len(S); acc=S.mean()
    b=np.array([rng.choice(S,n,replace=True).mean() for _ in range(5000)]); lo,hi=np.percentile(b,[2.5,97.5])
    print(f"   {'2024-26 COMBINED':>14}: acc={acc:.3f}  n={n:>4}  CI95=[{lo:.3f},{hi:.3f}]"
          f"{'  >67% PROVEN ✓' if lo>0.67 else ('  >67% point, CI spans' if acc>0.67 else '  <=67%')}")
print("\nV27-PROOF DONE")
