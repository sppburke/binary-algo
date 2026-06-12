"""USDCAD 15m NY — meta-label 'avoid-losers' gate FAST PRE-CHECK (confirm-or-kill).

SCOPE: USDCAD · 15m · NY. Fork of audusd_15m_metagate.py. A 2nd LGBM predicts P(primary directional call
CORRECT) from axes ORTHOGONAL to the 239 own-pair feats (cross-pair USD-residual/catchup, commodity-bloc/risk,
agreement, session), to gate coverage more sharply than raw |p-0.5|. FAST falsifier: if meta-correctness
axes-only VAL AUC <= 0.53 the gate is random -> KILL (EURUSD 15m meta .502, AUDUSD .5294 both NULL).

Strong prior of NULL: USDCAD xpair (A6) already showed these cross-pair/risk axes carry no incremental
DIRECTION over base (IMPROVES_base=False). This checks the WHEN-CORRECT angle anyway (run-don't-argue).
Usage: ~/binary-algo-venv/bin/python usdcad_15m_metagate.py
"""
import json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sessions import session_mask
from usdcad_15m_base import build, mk_lgb, SPL, FEATS
from usdcad_15m_xpair import build_xp, SPL as XSPL

RESULT="usdcad_15m_metagate_result.json"
SESSION="ny"
META_AXES_PREFIX=("usdbask","catchup","tgtresid","risk","cadcommod","agree","disp","sess_","comp")

def main():
    t0=time.time()
    Xtr,ytr,mtr,ttr=build(SPL["train"],6); s=session_mask(ttr,SESSION); itr=mtr&s
    P=mk_lgb()
    Xva0,yva0,mva0,tva0=build(SPL["val"]); sv=session_mask(tva0,SESSION); iva=mva0&sv
    P.fit(Xtr[itr],ytr[itr],eval_set=[(Xva0[iva],yva0[iva])],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pva=P.predict_proba(Xva0[iva])[:,1]; yv=yva0[iva]
    pred=(pva>0.5).astype(int); correct=(pred==yv).astype(int); conf=np.abs(pva-0.5)
    print(f"[metagate] primary VAL NY moved n={int(iva.sum())} acc={correct.mean():.4f}",flush=True)
    F=build_xp(XSPL["val"],1,"xp"); fm=session_mask(F["_ts"].values.astype("int64"),SESSION); F=F.loc[fm]
    axes=[c for c in F.columns if c.startswith(META_AXES_PREFIX)]
    fts=F["_ts"].values.astype("int64"); fmap=pd.DataFrame(F[axes].values, index=fts, columns=axes)
    fmap=fmap[~fmap.index.duplicated(keep="last")]
    A=fmap.reindex(tva0[iva]).values.astype("float32")
    ok=np.isfinite(A).any(axis=1); A=A[ok]; corr=correct[ok]; cf=conf[ok]
    Ameta=np.column_stack([A, cf]); n=len(Ameta); h=n//2
    def mkmeta(): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.03,num_leaves=63,
        min_child_samples=200,subsample=0.8,colsample_bytree=0.6,reg_lambda=10,n_estimators=400,n_jobs=14,verbosity=-1)
    M=mkmeta(); M.fit(Ameta[:h],corr[:h],eval_set=[(Ameta[h:],corr[h:])],eval_metric="auc",callbacks=[lgb.early_stopping(50),lgb.log_evaluation(0)])
    meta_auc=float(roc_auc_score(corr[h:], M.predict_proba(Ameta[h:])[:,1]))
    M2=mkmeta(); M2.fit(A[:h],corr[:h],eval_set=[(A[h:],corr[h:])],eval_metric="auc",callbacks=[lgb.early_stopping(50),lgb.log_evaluation(0)])
    meta_auc_axesonly=float(roc_auc_score(corr[h:], M2.predict_proba(A[h:])[:,1]))
    res={"key":"USDCAD.15m.NY","n_meta_axes":len(axes),"meta_axes":axes[:20],
         "primary_val_ny_acc":round(float(correct.mean()),4),
         "meta_correctness_auc_axes+conf":round(meta_auc,4),"meta_correctness_auc_axesonly":round(meta_auc_axesonly,4),
         "falsifier":"KILL if meta-correctness axes-only AUC <= 0.53 (random gate, as EURUSD .502 / AUDUSD .5294)",
         "verdict":{"meta_gate_viable":bool(meta_auc_axesonly>0.53),
            "note":"axes-only AUC tests whether orthogonal axes carry WHEN-CORRECT info (xpair A6 showed they carry no DIRECTION info)."}}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[metagate] meta-correctness AUC: axes+conf={meta_auc:.4f} axesONLY={meta_auc_axesonly:.4f} -> viable={res['verdict']['meta_gate_viable']} ({time.time()-t0:.0f}s) -> {RESULT}",flush=True)

if __name__=="__main__":
    main()
