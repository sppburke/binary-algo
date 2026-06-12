"""V26 — how HIGH can 15m EURUSD binary accuracy go (OOS-verified, honestly)?

Map accuracy vs coverage to the limit on the validated pockets. For each coverage report:
  VAL acc | TEST(2024+25) acc | OOS-2026 acc + bootstrap 95% CI | OOS-H1 | OOS-H2 | nOOS
A number only counts as the 'achievable ceiling' if it holds in BOTH OOS halves at n that gives a
CI excluding chance. Threshold frozen on VAL within each pocket. Pockets stack the validated levers
(compression depth, NY session, low realized vol). Target = up/down 15m binary endpoint sign.
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
        parts.append(d.iloc[::stride] if stride>1 else d)
    return pd.concat(parts)

TR=load(H.SPLITS["train"],STRIDE); VA=load(H.SPLITS["val"])
Q={q:np.nanpercentile(TR["15m_bb_width"].values.astype(float),q) for q in (10,20,33)}
vzmed=np.nanmedian(VA["vol_z"].values.astype(float))
def col(D,c): return D[c].values.astype(float)
POCKETS={
 "compress×NY":            lambda D:(col(D,"15m_bb_width")<=Q[33])&(col(D,"sess_ny")>0.5),
 "compress×NY×lowvol":     lambda D:(col(D,"15m_bb_width")<=Q[33])&(col(D,"sess_ny")>0.5)&(col(D,"vol_z")<=vzmed),
 "tightcompress(q20)×NY":  lambda D:(col(D,"15m_bb_width")<=Q[20])&(col(D,"sess_ny")>0.5),
 "tightcompress(q10)×NY":  lambda D:(col(D,"15m_bb_width")<=Q[10])&(col(D,"sess_ny")>0.5),
}
L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=255,min_child_samples=200,
    subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=10,n_estimators=3000,n_jobs=20,verbosity=-1)
ytr=TR["_y"].astype(int).values; yva=VA["_y"].astype(int).values
L.fit(TR[base].astype("float32"),ytr,eval_set=[(VA[base].astype("float32"),yva)],
      eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
def prob(D): return L.predict_proba(D[base].astype("float32"))[:,1]
pva=prob(VA)
TE=load(["2024","2025"]); OO=load(["2026"]); pte=prob(TE); poo=prob(OO)
yte=TE["_y"].astype(int).values; yoo=OO["_y"].astype(int).values
h=len(OO)//2; oh1=slice(0,h); oh2=slice(h,None)
rng=np.random.default_rng(3)
def a(y,p,m,thr):
    mm=m&(np.abs(p-0.5)>=thr)
    if mm.sum()==0: return np.nan,0,None
    s=((p[mm]>0.5).astype(int)==y[mm]); return s.mean(),int(mm.sum()),s

print(f"AUC val={roc_auc_score(yva,pva):.4f} test={roc_auc_score(yte,pte):.4f} oos={roc_auc_score(yoo,poo):.4f}\n")
for name,fn in POCKETS.items():
    mv,mt,mo=fn(VA),fn(TE),fn(OO); confv=np.abs(pva[mv]-0.5)
    print(f"### {name}  (VAL pocket n={int(mv.sum())}, OOS pocket n={int(mo.sum())})")
    print(f"   {'cov':>5} {'VALacc':>7} {'TESTacc':>8} {'OOSacc':>7} {'OOS95CI':>15} {'H1':>6} {'H2':>6} {'nOOS':>6}")
    for cov in (0.10,0.05,0.02,0.01,0.005):
        if mv.sum()<100: break
        thr=np.quantile(confv,1-cov)
        av,_,_=a(yva,pva,mv,thr); at,_,_=a(yte,pte,mt,thr)
        ao,no,so=a(yoo,poo,mo,thr)
        mo1=np.zeros(len(OO),bool); mo1[oh1]=True; mo2=np.zeros(len(OO),bool); mo2[oh2]=True
        a1,_,_=a(yoo,poo,mo&mo1,thr); a2,_,_=a(yoo,poo,mo&mo2,thr)
        ci=""
        if so is not None and no>=20:
            b=np.array([rng.choice(so,no,replace=True).mean() for _ in range(5000)])
            ci=f"[{np.percentile(b,2.5):.2f},{np.percentile(b,97.5):.2f}]"
        print(f"   {cov:>5.1%} {av:>7.3f} {at:>8.3f} {ao:>7.3f} {ci:>15} {a1:>6.3f} {a2:>6.3f} {no:>6}",flush=True)
    print()
print("V26-MAX DONE")
