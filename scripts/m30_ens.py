"""30m EURUSD — Iteration 3: ENSEMBLE (lgb+xgb+cat) + agreement confidence + two-factor reversion overlay.

Rationale: a single shallow LGB barely fits (AUC ~0.52). An ensemble gives a more stable probability tail; ensemble
AGREEMENT (all 3 models same side, all far from 0.5) is a stricter confidence notion than |p-0.5| of one model. We
also test a TWO-FACTOR overlay: keep a bet only when the ML direction AND a structural reversion signal agree
(intersection of two weak edges). All honest: select on VAL, verify across TEST24/TEST25/OOS26, indep non-overlap.

  python m30_ens.py train <tag> [cross]    # cross => add USD-basket peer feats
  python m30_ens.py eval  <tag>
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import harness as H, crosspair as CP
from m30_lab import HOR, GAP_S, base, REGIME, WINDOWS, nonoverlap_chrono, boot, indep_acc, thr_for_cov, build_gates, MODELS

def load_x(years, stride=1, use_cross=False):
    cross_cols = CP.cross_feature_names("EURUSD") if use_cross else []
    feats = base + cross_cols
    parts=[]
    for y in years:
        if use_cross:
            df=CP.load_year_merged("EURUSD", y, base_cols=base+["close"])
            if df is None: continue
        else:
            p=f"{H.FEAT_DIR}/EURUSD_{y}.parquet"
            if not os.path.exists(p): continue
            df=pd.read_parquet(p, columns=base+["close"])
        df=df[~df.index.duplicated(keep="last")]
        c=df["close"].values; n=len(c)
        secs=df.index.values.astype("datetime64[s]").astype("int64")
        contig=np.zeros(n,bool)
        if n>HOR: contig[:n-HOR]=(secs[HOR:]-secs[:-HOR])==HOR*60
        fwd=np.full(n,np.nan); fwd[:n-HOR]=c[HOR:]; ret=fwd/c-1.0
        yv=(ret>0).astype(float); valid=contig&np.isfinite(ret)&(ret!=0)
        keep=[x for x in feats if x in df.columns]
        d=df.loc[valid, keep].copy(); d["_y"]=yv[valid]; d["_ts"]=secs[valid]
        parts.append(d.iloc[::stride] if stride>1 else d)
    D=pd.concat(parts); return D, [x for x in feats if x in D.columns]

def main_train(tag, use_cross=False):
    t0=time.time(); os.makedirs(MODELS,exist_ok=True); STRIDE=3
    print(f"[ens:{tag}] load TRAIN stride{STRIDE} cross={use_cross}...",flush=True)
    TR,feats=load_x([str(y) for y in range(2012,2022)],STRIDE,use_cross)
    VA,_=load_x(WINDOWS["val"],1,use_cross)
    ytr=TR["_y"].astype(int).values; yva=VA["_y"].astype(int).values
    Xtr=TR[feats].astype("float32"); Xva=VA[feats].astype("float32")
    print(f"[ens:{tag}] train={len(TR):,} val={len(VA):,} feats={len(feats)} load={time.time()-t0:.0f}s",flush=True)
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=255,min_child_samples=300,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=10,n_estimators=3000,n_jobs=20,verbosity=-1)
    L.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",callbacks=[lgb.early_stopping(200),lgb.log_evaluation(0)])
    print(f"[ens:{tag}] lgb iter={L.best_iteration_} {time.time()-t0:.0f}s",flush=True)
    G=xgb.XGBClassifier(n_estimators=2000,learning_rate=0.02,max_depth=8,subsample=0.8,colsample_bytree=0.5,
        reg_lambda=10,tree_method="hist",n_jobs=20,eval_metric="auc",early_stopping_rounds=200)
    G.fit(Xtr,ytr,eval_set=[(Xva,yva)],verbose=False)
    print(f"[ens:{tag}] xgb {time.time()-t0:.0f}s",flush=True)
    C=CatBoostClassifier(iterations=2000,learning_rate=0.02,depth=8,l2_leaf_reg=10,eval_metric="AUC",
        thread_count=20,verbose=False,early_stopping_rounds=200)
    C.fit(Xtr.fillna(-999),ytr,eval_set=(Xva.fillna(-999),yva))
    print(f"[ens:{tag}] cat {time.time()-t0:.0f}s",flush=True)
    del TR,VA,Xtr,Xva
    # TRAIN-side percentile thresholds for compression/expansion gates (one strided pass)
    Dq,_=load_x([str(y) for y in range(2016,2022)],8,False); qthr={}
    for c in REGIME:
        if c.endswith("bb_width") and c in Dq.columns:
            v=Dq[c].values.astype(float); qthr[c]=float(np.nanpercentile(v,33)); qthr[c+"_hi"]=float(np.nanpercentile(v,67))
    del Dq
    cache={"feats":feats,"qthr":json.dumps(qthr)}; aucs={}
    for w,yrs in WINDOWS.items():
        D,_=load_x(yrs,1,use_cross); X=D[feats].astype("float32")
        pl=L.predict_proba(X)[:,1]; pg=G.predict_proba(X)[:,1]; pc=C.predict_proba(X.fillna(-999))[:,1]
        pe=(pl+pg+pc)/3.0; y=D["_y"].astype(int).values; ts=D["_ts"].values.astype("int64")
        aucs[w]=float(roc_auc_score(y,pe))
        cache.update({f"{w}_p":pe.astype("float32"),f"{w}_pl":pl.astype("float32"),f"{w}_pg":pg.astype("float32"),
                      f"{w}_pc":pc.astype("float32"),f"{w}_y":y.astype("int8"),f"{w}_ts":ts})
        for c in REGIME:
            if c in D.columns: cache[f"{w}_r_{c}"]=D[c].values.astype("float32")
        print(f"[ens:{tag}] {w}: n={len(y):,} ensAUC={aucs[w]:.4f} (lgb {roc_auc_score(y,pl):.4f} xgb {roc_auc_score(y,pg):.4f} cat {roc_auc_score(y,pc):.4f})",flush=True)
        del D,X
    np.savez_compressed(f"{MODELS}/m30_{tag}.npz", **cache)
    print(f"[ens:{tag}] cached. ensAUC "+" ".join(f"{w}={aucs[w]:.4f}" for w in WINDOWS)+f" DONE {time.time()-t0:.0f}s",flush=True)

def _load_window(z,w):
    R={"_y":z[f"{w}_y"],"_p":z[f"{w}_p"],"_ts":z[f"{w}_ts"],
       "_pl":z[f"{w}_pl"],"_pg":z[f"{w}_pg"],"_pc":z[f"{w}_pc"]}
    for k in z.files:
        if k.startswith(f"{w}_r_"): R[k[len(f"{w}_r_"):]]=z[k]
    return R

def agree_conf(R):
    """confidence = ensemble |p-0.5| but ONLY where all 3 models agree on side; else 0 (never selected)."""
    s=np.sign(R["_pl"]-0.5)+np.sign(R["_pg"]-0.5)+np.sign(R["_pc"]-0.5)
    allagree=np.abs(s)==3
    conf=np.abs(R["_p"]-0.5); conf=np.where(allagree, conf, -1.0)
    return conf

def main_eval(tag, covs=(0.05,0.02,0.01)):
    z=np.load(f"{MODELS}/m30_{tag}.npz",allow_pickle=True); qthr=json.loads(str(z["qthr"]))
    W={w:_load_window(z,w) for w in WINDOWS}
    print(f"\n===== EVAL m30_{tag} ENSEMBLE =====  (indep {GAP_S}s chrono; CI95; breakeven~0.541)")
    print("ensAUC: "+"  ".join(f"{w}={roc_auc_score(W[w]['_y'],W[w]['_p']):.4f}" for w in WINDOWS))
    Gs={w:build_gates(W[w],qthr) for w in WINDOWS}
    # confidence variants: ensemble |p-.5|, and agreement-gated
    def run(conf_of, gname, label):
        gval=Gs["val"][gname]
        if gval.sum()<200: return
        print(f"\n--- {label} | gate={gname} (VAL n={int(gval.sum()):,}) ---")
        for cov in covs:
            cval=conf_of(W["val"]); valmask=gval&(cval>=0)
            cc=cval[valmask]
            if len(cc)<50: continue
            thr=float(np.quantile(cc,1-cov))
            row=[]
            for w in WINDOWS:
                Rw=W[w]; cw=conf_of(Rw); g=Gs[w].get(gname,np.ones(len(Rw["_y"]),bool))
                m=g&(cw>=thr)
                if m.sum()==0: row.append(f"{w}:n0"); continue
                sel=nonoverlap_chrono(Rw["_ts"],m)
                corr=((Rw["_p"][sel]>0.5).astype(int)==Rw["_y"][sel]).astype(float)
                acc=corr.mean(); lo,hi=boot(corr)
                row.append(f"{w}:n{len(sel)} {acc:.3f}[{lo:.2f},{hi:.2f}]")
            print(f"  cov{cov:.0%} thr={thr:.4f}  "+"  ".join(row))
    ens_conf=lambda R: np.abs(R["_p"]-0.5)
    for gname in ("ny","comp_1h","comp1h_ny","comp30_ny","comp15_ny","trend_align"):
        if gname in Gs["val"]: run(ens_conf, gname, "ENS-conf")
    for gname in ("comp1h_ny","comp30_ny","comp_1h","ny"):
        if gname in Gs["val"]: run(agree_conf, gname, "AGREE")

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "eval"; tag=sys.argv[2] if len(sys.argv)>2 else "ens"
    if mode=="train": main_train(tag, len(sys.argv)>3 and sys.argv[3]=="cross")
    else: main_eval(tag)
