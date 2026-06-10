"""USDCAD 15m — TWO-SPEED momentum SIGN-agreement as a DIRECTION feature (discovery R1-1) — FAST SCREEN.

SCOPE: USDCAD · 15m (R1-1). Mechanism: when SLOW (1h/4h/30m) and FAST (1m/5m/15m) momentum AGREE on sign,
the agreed sign IS the predicted direction; disagreement = correction/rebound (no edge). The agreed-sign vote
carries SIGN (survives sign-invariance). CAVEAT (prior-lowering): the base 239 already has `mtf_trend_align`
+ all multi-tf ret/ema-slope/macd, so the GBM may already capture this. So this is a CHEAP FAST-KILL SCREEN
first (NY single-fit, base vs base+twospeed VAL moved-AUC + binding-year tail); escalate to a full refit-CPCV
ONLY if it lifts.

Two-speed features (from base multi-tf ret_6 at 1m/5m/15m/30m/1h/4h, NY rows; all causal):
  net_vote = sum_tf sign(ret_tf)            (signed consensus across speeds -> directional)
  agree_frac = mean_tf 1{sign(ret_tf)==sign(net_vote)}
  slow_fast_prod_{s,f} = sign(slow_ret)*sign(fast_ret) for (4h,15m),(1h,5m),(30m,1m)  (+1 agree / -1 disagree)
  slow_vote_{s,f} = sign(slow_ret) * 1{agree}    (slow's directional vote when fast confirms)
  wsum = sum_tf z(ret_tf)                    (magnitude-weighted consensus, z on train)

KILL if base+twospeed NY VAL moved-AUC <= base NY VAL AUC + 0.003 (no incremental sign info; subsumed by base
mtf features) OR no twospeed feature ranks in the top-20 SHAP. Pre-registered in result JSON BEFORE held-out.
Usage: ~/binary-algo-venv/bin/python usdcad_15m_twospeed.py
"""
import os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from sessions import session_mask
from usdcad_15m_base import build, side_eval, BE, SPL, nonoverlap_chrono

PAIR="USDCAD"; HOR=15; GAP=900; SESSION="ny"
FEATS=H.feature_cols(PAIR)
TFS=["1m","5m","15m","30m","1h","4h"]
RETC=[f"{t}_ret_6" for t in TFS]                 # one momentum per speed
SLOWFAST=[("4h","15m"),("1h","5m"),("30m","1m"),("4h","30m"),("1h","15m")]

def twospeed_feats(Xdf, zmu=None, zsd=None):
    """Build two-speed signed-agreement features from base ret_6 columns. Returns (DataFrame, zmu, zsd)."""
    R=Xdf[RETC].values.astype(float)
    sign=np.sign(R)
    net_vote=np.nansum(sign,axis=1)
    consensus_sign=np.sign(net_vote)
    agree_frac=np.nanmean((sign==consensus_sign[:,None]).astype(float),axis=1)
    if zmu is None: zmu=np.nanmean(R,axis=0); zsd=np.nanstd(R,axis=0)+1e-12
    Z=(R-zmu)/zsd
    wsum=np.nansum(Z,axis=1)
    out={"ts_net_vote":net_vote,"ts_agree_frac":agree_frac,"ts_wsum":wsum,"ts_consensus":consensus_sign}
    for s,f in SLOWFAST:
        ss=np.sign(Xdf[f"{s}_ret_6"].values.astype(float)); fs=np.sign(Xdf[f"{f}_ret_6"].values.astype(float))
        out[f"ts_prod_{s}_{f}"]=ss*fs
        out[f"ts_slowvote_{s}_{f}"]=ss*(ss==fs).astype(float)
    return pd.DataFrame(out,index=Xdf.index), zmu, zsd

def mk(): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=127,
    min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,
    n_estimators=3000,n_jobs=20,verbosity=-1)

def fit_eval(Xtr,ytr,itr,Xva,yva,iva,tsv,mva,tag):
    L=mk(); L.fit(Xtr[itr],ytr[itr],eval_set=[(Xva[iva],yva[iva])],eval_metric="auc",
        callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pva=L.predict_proba(Xva)[:,1]; auc=float(roc_auc_score(yva[iva],pva[iva]))
    print(f"[twospeed/{tag}] VAL NY moved-AUC={auc:.4f} best_iter={L.best_iteration_}",flush=True)
    return L,auc

def main():
    t0=time.time(); RESULT="usdcad_15m_twospeed_result.json"
    res={"key":"USDCAD.15m","model":"two-speed sign-agreement screen (NY)","session":SESSION,"breakeven":BE,
         "incumbent":"NY base VAL AUC .5319 (cpcv mean); NY-certified base UP p10 .6044/DOWN .5791",
         "falsifier":{"registered":"pre-OOS","KILL_if":"base+twospeed NY VAL AUC <= base NY VAL AUC + 0.003 OR no ts_ feature in top-20 SHAP"}}
    json.dump(res,open(RESULT,"w"),indent=2)
    # build NY-restricted train/val on moved bars
    Xtr_df,ytr,mtr,tstr=build(SPL["train"],6); Xva_df,yva,mva,tsv=build(SPL["val"])
    str_s=session_mask(tstr,SESSION); va_s=session_mask(tsv,SESSION)
    itr=mtr&str_s; iva=mva&va_s
    print(f"[twospeed] NY train(moved)={int(itr.sum()):,} val(moved)={int(iva.sum()):,} build={time.time()-t0:.0f}s",flush=True)
    # base-only arm
    Xtr=Xtr_df.values.astype("float32"); Xva=Xva_df.values.astype("float32")
    _,base_auc=fit_eval(Xtr,ytr,itr,Xva,yva,iva,tsv,mva,"base")
    # base+twospeed arm
    Ttr,zmu,zsd=twospeed_feats(Xtr_df); Tva,_,_=twospeed_feats(Xva_df,zmu,zsd)
    Xtr2=np.hstack([Xtr, Ttr.values.astype("float32")]); Xva2=np.hstack([Xva, Tva.values.astype("float32")])
    cols2=list(FEATS)+list(Ttr.columns)
    L2=mk(); L2.fit(Xtr2[itr],ytr[itr],eval_set=[(Xva2[iva],yva[iva])],eval_metric="auc",
        callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    p2=L2.predict_proba(Xva2)[:,1]; aug_auc=float(roc_auc_score(yva[iva],p2[iva]))
    imp=sorted(zip(cols2,L2.feature_importances_),key=lambda z:-z[1])[:20]
    top20=[c for c,_ in imp]; ts_in_top20=[c for c in top20 if c.startswith("ts_")]
    print(f"[twospeed/aug] VAL NY moved-AUC={aug_auc:.4f} (base {base_auc:.4f}, Δ{aug_auc-base_auc:+.4f})",flush=True)
    print(f"  ts_ feats in top20: {ts_in_top20}",flush=True)
    print(f"  top20: {', '.join(top20)}",flush=True)
    lift=aug_auc-base_auc
    kill=bool(lift<=0.003 or len(ts_in_top20)==0)
    res.update({"base_val_auc":round(base_auc,4),"aug_val_auc":round(aug_auc,4),"lift":round(lift,4),
                "ts_in_top20":ts_in_top20,"top20":top20,
                "verdict":{"KILLED":kill,"note":"if KILLED, two-speed subsumed by base mtf_trend_align+multi-tf feats (run-don't-argue). If SURVIVES -> escalate to NY refit-CPCV."}})
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[twospeed] VERDICT {'KILLED' if kill else 'SURVIVES->escalate refit-CPCV'} (lift {lift:+.4f}, ts_top20={len(ts_in_top20)}) -> {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__": main()
