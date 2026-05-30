"""15-MINUTE v7 — META-LABELING on the up/down binary (López de Prado AFML).

Primary model gives a directional side = sign(p1-0.5). A SECOND model (meta) predicts whether that
side will be CORRECT, using the 239 features + p1 + regime context. We then select bets by META
confidence instead of raw |p1-0.5|. Meta-labeling can lift accuracy@coverage because it learns *where*
the primary is trustworthy (regime/vol/session-conditioned), which a flat confidence threshold cannot.

Leakage control: primary trained on TRAIN only. Meta trained on VAL (predict primary correctness on VAL,
where the primary never saw labels). TEST 2024-25 and 2026 OOS stay fully held out — meta threshold frozen
on a VAL holdout slice. Target stays the binary endpoint sign; we only change the SELECTION rule.
"""
import time, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import exp_15m_v5_gates as V5

STRIDE=3
COVS=[0.5,0.2,0.1,0.05,0.02,0.01]

def mk(n=3000,leaves=255,mcs=200):
    return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=leaves,
        min_child_samples=mcs,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=10,
        n_estimators=n,n_jobs=20,verbosity=-1)

def sel(y,p,thr):
    m=p>=thr
    if m.sum()==0: return np.nan,0
    return (y[m]).mean(), int(m.sum())   # y here = primary-correct indicator -> selective ACCURACY

def main():
    t0=time.time()
    Xtr,ytr,gtr=V5.load_split("train",STRIDE)
    Xva,yva,gva=V5.load_split("val")
    Xte,yte,gte=V5.load_split("test")
    Xoo,yoo,goo=V5.load_split("oos")
    print(f"v7 load={time.time()-t0:.0f}s",flush=True)

    # ---- primary (direction) ----
    P=mk(); P.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",
                  callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    p1v=P.predict_proba(Xva)[:,1]; p1t=P.predict_proba(Xte)[:,1]; p1o=P.predict_proba(Xoo)[:,1]
    print(f"PRIMARY AUC oos={roc_auc_score(yoo,p1o):.4f}",flush=True)

    # primary side + correctness label
    def side(p): return (p>0.5).astype(int)
    cor_v=(side(p1v)==yva).astype(int); cor_t=(side(p1t)==yte).astype(int); cor_o=(side(p1o)==yoo).astype(int)

    # ---- meta features = base feats + primary prob + |edge| + a few regime cols ----
    def metaX(X,p,g):
        M=X.copy()
        M["_p1"]=p; M["_edge"]=np.abs(p-0.5)
        for c in ("15m_bb_width","15m_rv_20","sess_london","sess_ny","hour_sin","hour_cos","vol_z"):
            if c in g: M["_"+c]=g[c].values
        M["_uhr"]=g["utc_hour"].values
        return M.astype("float32")
    MXv=metaX(Xva,p1v,gva); MXt=metaX(Xte,p1t,gte); MXo=metaX(Xoo,p1o,goo)

    # split VAL: first 70% meta-train, last 30% meta-threshold-calibration
    n=len(MXv); cut=int(n*0.70)
    Mtr_X,Mtr_y=MXv.iloc[:cut],cor_v[:cut]
    Mca_X,Mca_y=MXv.iloc[cut:],cor_v[cut:]
    Mt=mk(n=2000); Mt.fit(Mtr_X,Mtr_y,eval_set=[(Mca_X,Mca_y)],eval_metric="auc",
                          callbacks=[lgb.early_stopping(120),lgb.log_evaluation(0)])
    mca=Mt.predict_proba(Mca_X)[:,1]; mt=Mt.predict_proba(MXt)[:,1]; mo=Mt.predict_proba(MXo)[:,1]
    print(f"META AUC(correctness) cal={roc_auc_score(Mca_y,mca):.4f} "
          f"test={roc_auc_score(cor_t,mt):.4f} oos={roc_auc_score(cor_o,mo):.4f}",flush=True)

    print("\n========== SELECTION-RULE FRONTIER (acc TEST/OOS @cov) ==========")
    # (A) baseline: select by primary |edge|, threshold on full VAL
    print("[A] select by primary |p1-0.5|  (threshold frozen on VAL)")
    cvv=np.abs(p1v-0.5)
    for cov in COVS:
        thr=np.quantile(cvv,1-cov)
        at,nt=sel(cor_t,np.abs(p1t-0.5),thr); ao,no=sel(cor_o,np.abs(p1o-0.5),thr)
        print(f"   cov{int(cov*100):>2} thr={thr:.4f}  TEST {at:.3f}(n{nt})  OOS {ao:.3f}(n{no})")
    # (B) select by META prob, threshold frozen on VAL calibration slice
    print("[B] select by META correctness prob  (threshold frozen on VAL-cal slice)")
    for cov in COVS:
        thr=np.quantile(mca,1-cov)
        at,nt=sel(cor_t,mt,thr); ao,no=sel(cor_o,mo,thr)
        f="**" if (not np.isnan(at) and not np.isnan(ao) and at>=0.75 and ao>=0.75 and nt>=40 and no>=40) else ""
        print(f"   cov{int(cov*100):>2} thr={thr:.4f}  TEST {at:.3f}(n{nt})  OOS {ao:.3f}(n{no}) {f}")
    print("\n** = both TEST & OOS >=0.75 at >=40 bets.  V7-META DONE")

if __name__=="__main__":
    main()
