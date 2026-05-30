"""15-MINUTE v6 — REGIME-SPECIALIST on the up/down binary (sign of 15m return).

V18 found vol-compression (bottom-tertile 15m bb_width) is the most generalizing gate (OOS 0.598@5%).
Hypothesis: a model trained ONLY on compression-regime bars learns the in-regime signal better than the
global model, and stacking compression × session pushes the selective OOS frontier higher.

- Compression threshold = TRAIN q33 of 15m_bb_width (fixed, causal; applied to VAL/TEST/OOS unchanged).
- Train SPECIALIST LGBM on compression-only TRAIN rows; also keep a GLOBAL LGBM for comparison.
- Evaluate within: compression, compression×london, compression×ny, compression×londonfix(15-16utc).
- Threshold frozen on the matching VAL subset; map selective accuracy down to 1% coverage on TEST + OOS.
"""
import time, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import exp_15m_v5_gates as V5   # reuse load_split / base_cols (read-only import)

STRIDE=3
COVS=[0.5,0.2,0.1,0.05,0.02,0.01]

def mk_lgb():
    return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=255,
        min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=10,
        n_estimators=3000,n_jobs=20,verbosity=-1)

def comp_mask(g, thr):  return g["15m_bb_width"].values.astype(float)<=thr
def sess_mask(g, s):    return g[s].values.astype(float)>0.5
def hourwin(g, a, b):   return (g["utc_hour"].values>=a)&(g["utc_hour"].values<b)

def sel(y,p,thr):
    c=np.abs(p-0.5); m=c>=thr
    if m.sum()==0: return np.nan,0
    return ((p[m]>0.5).astype(int)==y[m]).mean(), int(m.sum())

def frontier(tag, yv,pv, yt,pt, yo,po):
    if len(yv)<80 or len(yt)<80 or len(yo)<80:
        print(f"  {tag:>30} | too few (v{len(yv)} t{len(yt)} o{len(yo)})"); return
    cv=np.abs(pv-0.5); segs=[]
    for cov in COVS:
        thr=np.quantile(cv,1-cov)
        at,nt=sel(yt,pt,thr); ao,no=sel(yo,po,thr)
        f="**" if (not np.isnan(at) and not np.isnan(ao) and at>=0.75 and ao>=0.75 and nt>=40 and no>=40) else ""
        segs.append(f"c{int(cov*100)}:{at:.3f}/{ao:.3f}(o{no}){f}")
    print(f"  {tag:>30} | base up te={yt.mean():.3f} oo={yo.mean():.3f} | "+" ".join(segs),flush=True)

def main():
    t0=time.time()
    Xtr,ytr,gtr=V5.load_split("train",STRIDE)
    Xva,yva,gva=V5.load_split("val")
    Xte,yte,gte=V5.load_split("test")
    Xoo,yoo,goo=V5.load_split("oos")
    thr_bw=np.nanpercentile(gtr["15m_bb_width"].values.astype(float),33)
    print(f"v6 load={time.time()-t0:.0f}s  compression thr(15m_bb_width q33 TRAIN)={thr_bw:.6g}",flush=True)

    ctr=comp_mask(gtr,thr_bw)
    print(f"  compression rows: tr={ctr.mean():.2%} va={comp_mask(gva,thr_bw).mean():.2%} "
          f"te={comp_mask(gte,thr_bw).mean():.2%} oo={comp_mask(goo,thr_bw).mean():.2%}",flush=True)

    # ---- global model (baseline within-regime comparison) ----
    G=mk_lgb(); G.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",
                      callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    gpv=G.predict_proba(Xva)[:,1]; gpt=G.predict_proba(Xte)[:,1]; gpo=G.predict_proba(Xoo)[:,1]
    print(f"GLOBAL  AUC oos={roc_auc_score(yoo,gpo):.4f}",flush=True)

    # ---- specialist: trained on compression-only TRAIN ----
    S=mk_lgb(); S.fit(Xtr[ctr],ytr[ctr],eval_set=[(Xva[comp_mask(gva,thr_bw)],yva[comp_mask(gva,thr_bw)])],
                      eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    spv=S.predict_proba(Xva)[:,1]; spt=S.predict_proba(Xte)[:,1]; spo=S.predict_proba(Xoo)[:,1]

    cv=comp_mask(gva,thr_bw); ct=comp_mask(gte,thr_bw); co=comp_mask(goo,thr_bw)
    aucS=roc_auc_score(yoo[co],spo[co]); aucG=roc_auc_score(yoo[co],gpo[co])
    print(f"SPECIALIST AUC oos@compression={aucS:.4f}  (global@compression={aucG:.4f})\n",flush=True)

    print("================ FRONTIER  (acc TEST/OOS @cov, VAL-frozen) ================")
    print("[GLOBAL model, ungated]")
    frontier("ALL/global", yva,gpv, yte,gpt, yoo,gpo)
    print("[GLOBAL model inside compression]")
    frontier("compress/global", yva[cv],gpv[cv], yte[ct],gpt[ct], yoo[co],gpo[co])
    print("[SPECIALIST model inside compression and stacks]")
    frontier("compress/spec", yva[cv],spv[cv], yte[ct],spt[ct], yoo[co],spo[co])
    # stacks
    lv,lt,lo=sess_mask(gva,"sess_london"),sess_mask(gte,"sess_london"),sess_mask(goo,"sess_london")
    nv,nt,no=sess_mask(gva,"sess_ny"),sess_mask(gte,"sess_ny"),sess_mask(goo,"sess_ny")
    fv,ft,fo=hourwin(gva,15,16),hourwin(gte,15,16),hourwin(goo,15,16)
    frontier("compress&london/spec", yva[cv&lv],spv[cv&lv], yte[ct&lt],spt[ct&lt], yoo[co&lo],spo[co&lo])
    frontier("compress&ny/spec",     yva[cv&nv],spv[cv&nv], yte[ct&nt],spt[ct&nt], yoo[co&no],spo[co&no])
    frontier("compress&londonfix/spec", yva[cv&fv],spv[cv&fv], yte[ct&ft],spt[ct&ft], yoo[co&fo],spo[co&fo])
    print("\n** = both TEST & OOS >=0.75 at >=40 bets.  V6-SPECIALIST DONE")

if __name__=="__main__":
    main()
