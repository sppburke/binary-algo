"""NZDUSD 15-min direction — Antipodean cousin pooling SCREEN (A6).

SCOPE: NZDUSD · 15m (A6 — Antipodean bloc test). Does adding AUDUSD cross-pair features lift the
all-session val AUC above the A1 base floor (0.5219)? If yes: escalate to NY-session CPCV with xpair feats.
If no: KILL A6, own-pair specific (AUDUSD-precedent pattern confirmed).

Feature set: 239 NZDUSD base + 4 Antipodean cross-pair channels:
  aln(AUDUSD)=+1 (same-direction: NZDUSD-up = NZD-strength = USD-weakness, AUDUSD-up = AUD-strength = same)
  antipodean_basket_1: AUDUSD 1-bar return in NZDUSD-up frame
  antipodean_catchup_k: NZDUSD owes AUDUSD move = AUDUSD_ret_k - NZD_ret_k (signed catch-up)
  antipodean_risk_k: (AUDUSD_ret_k + NZDUSD_ret_k)/2 (common Pacific risk factor)
  antipodean_resid_k: NZDUSD_ret_k - AUDUSD_ret_k (NZD-specific residual orthogonal to AUD common)

If xpbase (239 + xpair) does NOT lift val_auc > BASE_AUCREC=0.5219 AND val_auc > 0.5210 (trivial margin):
  -> A6 KILLED. Own-pair specific pattern confirmed for NZDUSD (same as AUDUSD precedent).

Usage: ~/binary-algo-venv/bin/python nzdusd_15m_xpair_aud.py [mode=xpbase|xp] [stride=4]
"""
import os, sys, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H

FEAT="/home/sean/git/binary-algo/features"
TARGET="NZDUSD"; COUSIN="AUDUSD"; HOR=15; GAP=900
ALL_YEARS=[str(y) for y in range(2012,2027)]
SPL={"train":[str(y) for y in range(2012,2022)], "val":[str(y) for y in range(2022,2024)],
     "test24":["2024"], "test25":["2025"], "oos":["2026"]}
BASE_AUCREC=0.5219   # A1 all-session base val_auc — must beat this to escalate
LB=[1,2,4,8,16,32,64,128]  # lookback bars for cross-pair features
STRIDE=int(sys.argv[2]) if len(sys.argv)>2 else 4
MODE=sys.argv[1] if len(sys.argv)>1 else "xpbase"   # xpbase=239base+xp, xp=xp only

def nonoverlap_chrono(ts, mask, gap=GAP):
    take=[]; block=-1
    for i in np.where(mask)[0]:
        if ts[i]<block: continue
        take.append(i); block=int(ts[i])+gap
    return np.array(take,dtype=int)

def build_xp(years, stride=1, keep_ties=False):
    """Build Antipodean xpair feature matrix for NZDUSD-as-target."""
    frames=[]
    for y in years:
        ft=f"{FEAT}/{TARGET}_{y}.parquet"; fc=f"{FEAT}/{COUSIN}_{y}.parquet"
        if not (os.path.exists(ft) and os.path.exists(fc)): continue
        dt=pd.read_parquet(ft); dc=pd.read_parquet(fc)
        dt.index=pd.to_datetime(dt.index,utc=True); dc.index=pd.to_datetime(dc.index,utc=True)
        df=dt.join(dc.add_suffix("_aud"), how="inner")
        n=len(df)
        lnzd=np.log(df["close"].values); laud=np.log(df["close_aud"].values)
        rets_nzd={k:np.concatenate([[np.nan]*k, lnzd[k:]-lnzd[:-k]]) for k in LB}
        rets_aud={k:np.concatenate([[np.nan]*k, laud[k:]-laud[:-k]]) for k in LB}
        feats={}
        for k in LB:
            feats[f"aud_basket_{k}"]=rets_aud[k]            # AUDUSD return (same-direction for NZDUSD)
            feats[f"aud_catchup_{k}"]=rets_aud[k]-rets_nzd[k]  # NZDUSD owes AUDUSD catch-up
            feats[f"aud_risk_{k}"]=(rets_aud[k]+rets_nzd[k])/2  # common Pacific factor
            feats[f"aud_resid_{k}"]=rets_nzd[k]-rets_aud[k]   # NZD residual vs AUD
        fwd=np.concatenate([lnzd[HOR:]-lnzd[:-HOR], [np.nan]*HOR])
        ts=df.index.astype("int64")//10**9
        base_cols=H.feature_cols(TARGET)
        xp_feat=pd.DataFrame(feats, index=df.index)
        if MODE=="xpbase":
            base_sub=dt[[c for c in base_cols if c in dt.columns]]
            F=pd.concat([xp_feat, base_sub], axis=1, join='inner')
        else:
            F=xp_feat
        F["_y"]=(fwd>0).astype("int8"); F["_ts"]=ts
        F["_fwd"]=fwd; F["_moved"]=(fwd!=0)
        if stride>1: F=F.iloc[::stride]
        frames.append(F)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def mk(seed=0):
    return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=127,
        min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,
        n_estimators=3000,n_jobs=20,verbosity=-1,random_state=seed,bagging_seed=seed,feature_fraction_seed=seed)

def main():
    t0=time.time()
    RESULT=f"nzdusd_15m_xpair_aud_{MODE}_result.json"
    print(f"[xpair-aud] {TARGET} Antipodean pooling screen mode={MODE} stride={STRIDE}...",flush=True)
    TR=build_xp(SPL["train"], STRIDE); VA=build_xp(SPL["val"], STRIDE)
    feat_cols=[c for c in TR.columns if not c.startswith("_")]
    Xtr=TR[feat_cols].values; ytr=TR["_y"].values; mtr=TR["_moved"].values
    Xva=VA[feat_cols].values; yva=VA["_y"].values; mva=VA["_moved"].values
    Xva_ts=VA["_ts"].values; Xva_fwd=VA["_fwd"].values
    print(f"[xpair-aud] features={len(feat_cols)} train={int(mtr.sum()):,} val={int(mva.sum()):,} built={time.time()-t0:.0f}s",flush=True)
    L=mk(); L.fit(Xtr[mtr], ytr[mtr], eval_set=[(Xva[mva], yva[mva])], eval_metric="auc",
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
    pva=L.predict_proba(Xva)[:,1]; val_auc=float(roc_auc_score(yva[mva], pva[mva]))
    verdict="LIFT->ESCALATE" if val_auc>BASE_AUCREC else "FLAT/KILL"
    print(f"[xpair-aud] val_auc={val_auc:.4f} base={BASE_AUCREC} -> {verdict} ({time.time()-t0:.0f}s)",flush=True)

    # Per-year froward check at cov=5%
    results={}
    for w,yrs in SPL.items():
        if w=="train": continue
        TW=build_xp(yrs,1)
        if len(TW)==0: continue
        Xw=TW[feat_cols].values; yw=TW["_y"].values; mw=TW["_moved"].values
        tw=TW["_ts"].values; fw=TW["_fwd"].values
        pw=L.predict_proba(Xw)[:,1]
        conf=np.abs(pw-0.5); thr=float(np.quantile(np.abs(pva[mva]-0.5), 0.95))
        cand=(conf>=thr) & mw & np.isfinite(fw)
        take=nonoverlap_chrono(tw, cand)
        if len(take)==0: results[w]={"n":0,"wr":float("nan")}; continue
        pred=(pw[take]>0.5).astype(int); ylab=(fw[take]>0).astype(int); moved=(fw[take]!=0)
        wr=float(((pred==ylab)&moved).mean()); n=int(len(take))
        results[w]={"n":n,"wr":round(wr,4)}
        print(f"  {w}: n={n} wr={wr:.4f} @cov5%",flush=True)

    out={"target":TARGET,"cousin":COUSIN,"mode":MODE,"val_auc":round(val_auc,4),
         "base_val_auc":BASE_AUCREC,"verdict":verdict,"forward":results,
         "n_features":len(feat_cols),"stride_train":STRIDE,"elapsed":round(time.time()-t0,1)}
    json.dump(out, open(RESULT,"w"), indent=2)
    print(f"[xpair-aud] done -> {RESULT}",flush=True)

if __name__=="__main__":
    main()
