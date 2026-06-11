"""USDCHF 15m — signed ORDER-FLOW (OFI/Kyle) direction screen (discovery R1-5) — CONFIRM-ONCE FAST SCREEN.

SCOPE: USDCHF · 15m (R1-5). USDCHF has on-disk minute-aligned order-flow features (features_of/USDCHF_*: 18
cols — OF_of_norm/sum at LB 1/3/5/10/15/30, uptick_5/15, kyle_5/15, of_accel, of_persist). Mechanism: signed
order-flow imbalance as a DIRECTION feature (positive OFI -> buy pressure -> up). PRIOR: LOW. Tier-1 cross-pair
SUBSUMPTION says signed-OF direction is run-and-KILLED at 15m on USDJPY (tickmicro forward-holdout 2026 −.035)
and EURUSD (cross-impact .5015) — sign-invariance: order-flow imbalance gates SIZE; directional sign lives at
SECONDS, the 900s label is too far out. But the data is ON-DISK and the attitude says other pairs' nulls are
not a wall -> RUN it ONCE as a cheap NY VAL-AUC screen (base vs base+OF). KILL if no lift; escalate to NY
refit-CPCV only if it lifts.

KILL if base+OF NY VAL moved-AUC <= base NY VAL AUC + 0.003 OR no OF_ feature in top-20 SHAP.
Usage: ~/binary-algo-venv/bin/python usdchf_15m_ofi.py
"""
import os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from sessions import session_mask
from usdchf_15m_base import build, BE, SPL

PAIR="USDCHF"; SESSION="ny"; OFDIR="features_of"
FEATS=H.feature_cols(PAIR)

def load_of(years):
    """Load OF features for years, indexed by epoch-second int64 (UTC), deduped."""
    parts=[]
    for y in years:
        p=f"{OFDIR}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p)
        d.index=d.index.values.astype("datetime64[s]").astype("int64")
        d=d[~d.index.duplicated(keep="last")]          # dedup AT second-resolution (post-conversion)
        parts.append(d.astype("float32"))
    if not parts: return None
    out=pd.concat(parts); return out[~out.index.duplicated(keep="last")]

def align_of(of_all, ts):
    """Reindex OF to the base rows' epoch seconds (ts). Missing -> NaN (LGBM-native)."""
    return of_all.reindex(ts)

def mk(): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=127,
    min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,
    n_estimators=3000,n_jobs=14,verbosity=-1)

def main():
    t0=time.time(); RESULT="usdchf_15m_ofi_result.json"
    res={"key":"USDCHF.15m","model":"signed order-flow (OFI/Kyle) NY direction screen","session":SESSION,"breakeven":BE,
         "incumbent":"NY base VAL AUC ~.532 (cpcv mean); NY-certified base UP p10 .6044/DOWN .5791",
         "falsifier":{"registered":"pre-OOS","KILL_if":"base+OF NY VAL AUC <= base NY VAL AUC + 0.003 OR no OF_ feat in top-20"}}
    json.dump(res,open(RESULT,"w"),indent=2)
    Xtr_df,ytr,mtr,tstr=build(SPL["train"],6); Xva_df,yva,mva,tsv=build(SPL["val"])
    str_s=session_mask(tstr,SESSION); va_s=session_mask(tsv,SESSION)
    itr=mtr&str_s; iva=mva&va_s
    of_tr=load_of(SPL["train"]); of_va=load_of(SPL["val"])
    OFcols=list(of_tr.columns)
    Otr=align_of(of_tr,tstr); Ova=align_of(of_va,tsv)
    cov_tr=float(np.isfinite(Otr.values).any(axis=1)[itr].mean()); cov_va=float(np.isfinite(Ova.values).any(axis=1)[iva].mean())
    print(f"[ofi] NY train(moved)={int(itr.sum()):,} val(moved)={int(iva.sum()):,} OFcols={len(OFcols)} OF-coverage tr={cov_tr:.2f} va={cov_va:.2f} build={time.time()-t0:.0f}s",flush=True)
    Xtr=Xtr_df.values.astype("float32"); Xva=Xva_df.values.astype("float32")
    # base arm
    Lb=mk(); Lb.fit(Xtr[itr],ytr[itr],eval_set=[(Xva[iva],yva[iva])],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    base_auc=float(roc_auc_score(yva[iva],Lb.predict_proba(Xva)[:,1][iva]))
    print(f"[ofi/base] VAL NY moved-AUC={base_auc:.4f}",flush=True)
    # base+OF arm
    Xtr2=np.hstack([Xtr,Otr.values.astype("float32")]); Xva2=np.hstack([Xva,Ova.values.astype("float32")])
    cols2=list(FEATS)+OFcols
    L2=mk(); L2.fit(Xtr2[itr],ytr[itr],eval_set=[(Xva2[iva],yva[iva])],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    aug_auc=float(roc_auc_score(yva[iva],L2.predict_proba(Xva2)[:,1][iva]))
    imp=sorted(zip(cols2,L2.feature_importances_),key=lambda z:-z[1])[:20]
    top20=[c for c,_ in imp]; of_in_top20=[c for c in top20 if c.startswith("OF_")]
    lift=aug_auc-base_auc
    print(f"[ofi/aug] VAL NY moved-AUC={aug_auc:.4f} (base {base_auc:.4f}, Δ{lift:+.4f})",flush=True)
    print(f"  OF_ in top20: {of_in_top20}",flush=True)
    print(f"  top20: {', '.join(top20)}",flush=True)
    kill=bool(lift<=0.003 or len(of_in_top20)==0)
    res.update({"base_val_auc":round(base_auc,4),"aug_val_auc":round(aug_auc,4),"lift":round(lift,4),
                "of_coverage":{"train":round(cov_tr,3),"val":round(cov_va,3)},
                "of_in_top20":of_in_top20,"top20":top20,
                "verdict":{"KILLED":kill,"note":"if KILLED, signed-OF direction subsumed at 15m (USDJPY/EURUSD precedent + sign-invariance: OF gates SIZE, sign lives at seconds). If SURVIVES -> escalate NY refit-CPCV."}})
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[ofi] VERDICT {'KILLED' if kill else 'SURVIVES->escalate refit-CPCV'} (lift {lift:+.4f}, OF_top20={len(of_in_top20)}) -> {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__": main()
