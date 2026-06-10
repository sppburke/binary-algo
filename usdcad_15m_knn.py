"""USDCAD 15m — kNN regime-matcher (instance-based, orthogonal to GBM) direction screen (R1-3) — FAST SCREEN.

SCOPE: USDCAD · 15m (R1-3). Mechanism: signal = sign(mean subsequent 15m return of the K nearest NY bars in
Z-scored top-feature state space). Instance-based -> methodologically ORTHOGONAL to the GBM tree-partition; if
it decorrelates AND carries comparable signal, a kNN-GBM blend could lift. PRIOR: LOW (instance-based rarely
beats GBM on tabular; the AUDUSD cross-horizon-stack lesson: decorrelated-but-equally-weak = REDUNDANT).

Screen: (1) fit a quick GBM on NY train -> top-20 features; (2) Z-score top-20 on train, subsample 80k train as
reference; (3) kNN (K=100) val probabilities; (4) compare kNN VAL AUC vs GBM VAL AUC, corr(kNN,GBM), and a
50/50 blend AUC. KILL if kNN AUC < GBM AUC AND blend AUC <= GBM AUC + 0.003 (no standalone edge AND no blend
lift -> redundant/subsumed). If blend lifts -> escalate to NY refit-CPCV blend.
Usage: ~/binary-algo-venv/bin/python usdcad_15m_knn.py
"""
import json, time, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
import harness as H
from sessions import session_mask
from usdcad_15m_base import build, BE, SPL

PAIR="USDCAD"; SESSION="ny"; K=100; NREF=80_000; NTOP=20
FEATS=H.feature_cols(PAIR)

def mk(): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=127,
    min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,
    n_estimators=3000,n_jobs=14,verbosity=-1)

def main():
    t0=time.time(); RESULT="usdcad_15m_knn_result.json"
    res={"key":"USDCAD.15m","model":f"kNN regime-matcher (K={K}, top{NTOP} Z-space) NY screen","session":SESSION,"breakeven":BE,
         "falsifier":{"registered":"pre-OOS","KILL_if":"kNN VAL AUC < GBM VAL AUC AND blend AUC <= GBM AUC + 0.003 (no edge + no blend lift = redundant)"}}
    json.dump(res,open(RESULT,"w"),indent=2)
    Xtr_df,ytr,mtr,tstr=build(SPL["train"],6); Xva_df,yva,mva,tsv=build(SPL["val"])
    str_s=session_mask(tstr,SESSION); va_s=session_mask(tsv,SESSION)
    itr=np.where(mtr&str_s)[0]; iva=np.where(mva&va_s)[0]
    Xtr=Xtr_df.values.astype("float32"); Xva=Xva_df.values.astype("float32")
    print(f"[knn] NY train(moved)={len(itr):,} val(moved)={len(iva):,} build={time.time()-t0:.0f}s",flush=True)
    # GBM arm + top-20
    L=mk(); L.fit(Xtr[itr],ytr[itr],eval_set=[(Xva[iva],yva[iva])],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pgbm=L.predict_proba(Xva)[:,1][iva]; gbm_auc=float(roc_auc_score(yva[iva],pgbm))
    top_idx=np.argsort(L.feature_importances_)[::-1][:NTOP]
    print(f"[knn] GBM VAL NY moved-AUC={gbm_auc:.4f}; top{NTOP} feats={[FEATS[i] for i in top_idx][:8]}...",flush=True)
    # kNN in Z-scored top-feature space
    Xt=Xtr[itr][:,top_idx].astype("float64"); Xv=Xva[iva][:,top_idx].astype("float64")
    mu=np.nanmean(Xt,axis=0); sd=np.nanstd(Xt,axis=0)+1e-9
    Zt=np.nan_to_num((Xt-mu)/sd); Zv=np.nan_to_num((Xv-mu)/sd)
    rng=np.random.default_rng(7)
    ref=rng.choice(len(Zt), min(NREF,len(Zt)), replace=False)
    nn=NearestNeighbors(n_neighbors=K, algorithm="auto", n_jobs=14).fit(Zt[ref])
    yt_ref=ytr[itr][ref].astype(float)
    _,nbr=nn.kneighbors(Zv)
    pknn=yt_ref[nbr].mean(axis=1)           # P(up) = fraction of K neighbors that went up
    knn_auc=float(roc_auc_score(yva[iva],pknn))
    corr=float(np.corrcoef(pgbm,pknn)[0,1])
    blend=0.5*pgbm+0.5*pknn; blend_auc=float(roc_auc_score(yva[iva],blend))
    print(f"[knn] kNN VAL AUC={knn_auc:.4f} | corr(kNN,GBM)={corr:.3f} | blend AUC={blend_auc:.4f} (GBM {gbm_auc:.4f}, Δ{blend_auc-gbm_auc:+.4f}) {time.time()-t0:.0f}s",flush=True)
    kill=bool(knn_auc<gbm_auc and blend_auc<=gbm_auc+0.003)
    res.update({"gbm_val_auc":round(gbm_auc,4),"knn_val_auc":round(knn_auc,4),"corr_knn_gbm":round(corr,3),
                "blend_val_auc":round(blend_auc,4),"blend_lift":round(blend_auc-gbm_auc,4),"K":K,"nref":int(len(ref)),
                "verdict":{"KILLED":kill,"note":"if KILLED, kNN redundant (no standalone edge + no blend lift; AUDUSD decorrelated-but-equally-weak lesson). If blend lifts -> escalate NY refit-CPCV blend."}})
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[knn] VERDICT {'KILLED' if kill else 'SURVIVES->escalate blend refit-CPCV'} (kNN {knn_auc:.4f} vs GBM {gbm_auc:.4f}; blend Δ{blend_auc-gbm_auc:+.4f}) -> {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__": main()
