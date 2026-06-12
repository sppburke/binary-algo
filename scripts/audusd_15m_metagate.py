"""AUDUSD 15m NY — meta-label 'avoid-losers' gate FAST PRE-CHECK (confirm-or-kill).

SCOPE: AUDUSD · 15m · NY. A 2nd LGBM predicts P(primary directional call CORRECT) from axes ORTHOGONAL to the
239 own-pair feats (AUDNZD-RV sign, commodity/safe-haven RISK sign, cross-pair agreement, session/hour), to gate
coverage more sharply than raw |p-0.5|. FAST falsifier (computable before any backtest): if meta-correctness VAL
AUC <= 0.53 the gate is random (EURUSD 15m meta was .502 NULL) -> KILL without a full nested-refit CPCV.

Strong prior of NULL: orthochan already showed these exact axes (audnzd/risk) carry no conditional DIRECTION
signal over base (ADDS=False). This checks whether they carry incremental WHEN-CORRECT info anyway.

Train primary on 2012-21 NY (moved); predict VAL 2022-23 NY -> correctness label; fit meta on the orthogonal
axes with an internal VAL/test split; report meta-AUC.
Usage: ~/binary-algo-venv/bin/python audusd_15m_metagate.py
"""
import json, time, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sessions import session_mask
from audusd_15m_base import build, mk_lgb, SPL, FEATS
from audusd_15m_xpair import build_xp_aud, SPL as XSPL

RESULT="audusd_15m_metagate_result.json"
SESSION="ny"
META_AXES_PREFIX=("audnzd","risk","audrisk","usdbask","catchup","agree","disp","sess_","comp")

def main():
    t0=time.time()
    # primary on train NY moved
    Xtr,ytr,mtr,ttr=build(SPL["train"],6); s=session_mask(ttr,SESSION); itr=mtr&s
    P=mk_lgb();
    # quick early-stop on a slice of train val
    Xva0,yva0,mva0,tva0=build(SPL["val"]); sv=session_mask(tva0,SESSION); iva=mva0&sv
    P.fit(Xtr[itr],ytr[itr],eval_set=[(Xva0[iva],yva0[iva])],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    # primary probs on VAL NY moved -> correctness label
    pva=P.predict_proba(Xva0[iva])[:,1]; yv=yva0[iva]
    pred=(pva>0.5).astype(int); correct=(pred==yv).astype(int)
    conf=np.abs(pva-0.5)
    print(f"[metagate] primary VAL NY moved n={iva.sum()} acc={correct.mean():.4f} base-correct-rate (the meta must beat AUC .5)",flush=True)
    # orthogonal meta-axes at VAL NY (align by timestamp via build_xp_aud)
    F=build_xp_aud(XSPL["val"],1); fm=session_mask(F["_ts"].values.astype("int64"),SESSION); F=F.loc[fm]
    axes=[c for c in F.columns if c.startswith(META_AXES_PREFIX)]
    # align meta-axes to the primary VAL NY moved rows by timestamp
    import pandas as pd
    fts=F["_ts"].values.astype("int64"); fmap=pd.DataFrame(F[axes].values, index=fts, columns=axes)
    vts=tva0[iva]
    A=fmap.reindex(vts).values.astype("float32")
    ok=np.isfinite(A).any(axis=1)
    A=A[ok]; corr=correct[ok]; cf=conf[ok]
    Ameta=np.column_stack([A, cf])     # axes + primary confidence
    n=len(Ameta); h=n//2
    M=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.03,num_leaves=63,min_child_samples=200,
        subsample=0.8,colsample_bytree=0.6,reg_lambda=10,n_estimators=400,n_jobs=20,verbosity=-1)
    M.fit(Ameta[:h],corr[:h],eval_set=[(Ameta[h:],corr[h:])],eval_metric="auc",callbacks=[lgb.early_stopping(50),lgb.log_evaluation(0)])
    meta_auc=float(roc_auc_score(corr[h:], M.predict_proba(Ameta[h:])[:,1]))
    # also: meta-auc using ONLY orthogonal axes (drop primary conf) to isolate the axes' value
    M2=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.03,num_leaves=63,min_child_samples=200,
        subsample=0.8,colsample_bytree=0.6,reg_lambda=10,n_estimators=400,n_jobs=20,verbosity=-1)
    M2.fit(A[:h],corr[:h],eval_set=[(A[h:],corr[h:])],eval_metric="auc",callbacks=[lgb.early_stopping(50),lgb.log_evaluation(0)])
    meta_auc_axesonly=float(roc_auc_score(corr[h:], M2.predict_proba(A[h:])[:,1]))
    res={"key":"AUDUSD.15m.NY","n_meta_axes":len(axes),"meta_axes":axes[:20],
         "primary_val_ny_acc":round(float(correct.mean()),4),
         "meta_correctness_auc_axes+conf":round(meta_auc,4),"meta_correctness_auc_axesonly":round(meta_auc_axesonly,4),
         "falsifier":"KILL if meta-correctness AUC <= 0.53 (random gate, as EURUSD 15m meta .502)",
         "verdict":{"meta_gate_viable":bool(meta_auc_axesonly>0.53),
            "note":"axes-only AUC is the test of whether orthogonal axes carry WHEN-CORRECT info (orthochan showed they carry no DIRECTION info)."}}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[metagate] meta-correctness AUC: axes+conf={meta_auc:.4f} axesONLY={meta_auc_axesonly:.4f} -> viable={res['verdict']['meta_gate_viable']} ({time.time()-t0:.0f}s) -> {RESULT}",flush=True)

if __name__=="__main__":
    main()
