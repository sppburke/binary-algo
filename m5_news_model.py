"""Model-based EVENT-TIME CONDITIONING: add macro-surprise features to the 5m cross-pair+OF model and test whether
(a) it lifts overall 5-min direction AUC, and (b) gating to NEWS WINDOWS (recent high-impact surprise) gives a higher-accuracy
selective subset. Surprise features per 1-min bar via merge_asof to the most recent release. Held-out TEST24/25 + OOS26, CI95.
"""
import sys, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H, m5_xpair as MX
CAL="/home/sean/git/binary-algo/macro_calendar.parquet"
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],"test24":["2024"],"test25":["2025"],"oos":["2026"]}
WIN=30  # minutes a release stays "active"

def news_features(index):
    ev=pd.read_parquet(CAL); ev["ts"]=pd.to_datetime(ev["ts"],utc=True,format="ISO8601")
    ev["ishigh"]=(ev["volatility"]=="HIGH").astype(float)
    g=(ev.groupby("ts").agg(sig=("sig_z","sum"),absz=("sig_z",lambda s:s.abs().max()),ishigh=("ishigh","max"))
         .reset_index().sort_values("ts"))
    g["ts"]=g["ts"].dt.as_unit("ns"); g["rel_ts"]=g["ts"]
    df=pd.DataFrame({"ts":pd.to_datetime(pd.Index(index),utc=True).as_unit("ns")})
    order=np.argsort(df["ts"].values)                      # ensure sorted for merge_asof, map back after
    dfs=df.iloc[order].reset_index(drop=True)
    m=pd.merge_asof(dfs,g[["ts","rel_ts","sig","absz","ishigh"]],on="ts",direction="backward")
    mins=(m["ts"]-m["rel_ts"]).dt.total_seconds().values/60.0
    active=np.isfinite(mins)&(mins<=WIN)
    out_s=pd.DataFrame({
        "nz_sig":np.where(active,np.nan_to_num(m["sig"].values),0.0),
        "nz_absz":np.where(active,np.nan_to_num(m["absz"].values),0.0),
        "nz_high":np.where(active,np.nan_to_num(m["ishigh"].values),0.0),
        "nz_mins":np.where(active,np.minimum(np.nan_to_num(mins,nan=WIN),WIN),WIN),
        "nz_inwin":active.astype(float)})
    out=pd.DataFrame(index=range(len(df)),columns=out_s.columns,dtype=float)
    out.iloc[order]=out_s.values                            # un-sort back to original index order
    out.index=index
    return out

def build(window, mode="xpof", stride=1):
    F=MX.build_xp(SPL[window],stride); F=MX.augment(F,SPL[window],mode)
    nf=news_features(F.index)
    for c in nf.columns: F[c]=nf[c].values
    return F

def boot(c,nb=4000,seed=7):
    c=np.asarray(c,float)
    if len(c)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(c); a=np.array([c[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))

def main():
    t0=__import__("time").time()
    TR=build("train",stride=4); VA=build("val")
    xpc=MX.xp_cols(TR); base=list(H.feature_cols("EURUSD"))
    NEWS=["nz_sig","nz_absz","nz_high","nz_mins","nz_inwin"]
    cols=list(dict.fromkeys(xpc+[c for c in base if c in TR.columns]+MX.OF_COLS+NEWS))
    cols=[c for c in cols if c in TR.columns and c in VA.columns]
    print(f"[news_model] train={len(TR):,} feats={len(cols)} (news={NEWS}) {__import__('time').time()-t0:.0f}s",flush=True)
    ytr=TR["_y"].astype(int).values; yva=VA["_y"].astype(int).values
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=127,min_child_samples=400,
        subsample=0.8,colsample_bytree=0.5,reg_lambda=20,n_estimators=2500,n_jobs=20,verbosity=-1)
    L.fit(TR[cols].astype("float32"),ytr,eval_set=[(VA[cols].astype("float32"),yva)],eval_metric="auc",
          callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    imp=sorted(zip(cols,L.feature_importances_),key=lambda z:-z[1])
    newsrank={c:i for i,(c,_) in enumerate(imp)}
    print(f"[news_model] VAL AUC={roc_auc_score(yva,L.predict_proba(VA[cols].astype('float32'))[:,1]):.4f}; news-feat ranks: "+
          ", ".join(f"{c}#{newsrank[c]}" for c in NEWS),flush=True)
    for w in ("test24","test25","oos"):
        D=build(w); pr=L.predict_proba(D[cols].astype("float32"))[:,1]; y=D["_y"].astype(int).values
        ts=D["_ts"].values.astype("int64"); ny=D["sess_ny"].values>0.5
        inwin=D["nz_inwin"].values>0.5; bignews=inwin&(D["nz_absz"].values>=1.0)
        auc=roc_auc_score(y,pr); aucn=roc_auc_score(y[inwin],pr[inwin]) if inwin.sum()>50 and len(set(y[inwin]))>1 else float("nan")
        # accuracy on news-window bars at high model confidence vs overall
        def selacc(mask,thr):
            m=mask&(np.abs(pr-0.5)>=thr); sel=MX.nonoverlap_chrono(ts,m,300)
            if len(sel)<10: return (0,float("nan"))
            return (len(sel),((pr[sel]>0.5).astype(int)==y[sel]).mean())
        n_all,a_all=selacc(ny,np.quantile(np.abs(pr-0.5)[ny],0.95))
        n_nw,a_nw=selacc(ny&bignews,0.0)
        print(f"=== {w} === AUC={auc:.4f} | news-win AUC={aucn:.4f} (n_inwin={int(inwin.sum())}) | "
              f"NY top5%%conf acc={a_all:.3f}(n{n_all}) | NY&bignews acc={a_nw:.3f}(n{n_nw})",flush=True)

if __name__=="__main__": main()
