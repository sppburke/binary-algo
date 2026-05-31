"""5-MIN EURUSD direction via CROSS-PAIR / USD-common-factor / lead-lag features (GENUINELY UNTRIED block).

Motivation (probe m5_xpair_probe.py): the cross-pair reversion/catch-up signal is WEAK (~0.508 full-cov) but SIGN-STABLE
across TEST2024 and OOS2026 — unlike the prior OHLCV momentum which flipped on test25. We build the full orthogonal block
and test (a) XP-only and (b) XP+base(239) selective, honestly across TEST24/TEST25/OOS26 with CI95 + non-overlap.

Sign conventions: USD-quote pairs {EURUSD,GBPUSD,AUDUSD,NZDUSD} up => USD down; USD-base {USDJPY,USDCHF,USDCAD} up => USD up.
'eu_equiv' = each pair's move expressed in EURUSD-up (=USD-weakness) direction: +r for USD-quote, -r for USD-base.

Usage: python m5_xpair.py [mode=xp|xpbase] [stride_train=4]
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H

FEAT="/media/sean/CORSAIR/binary-algo/features"
PAIRS=["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCHF","USDCAD"]
USD_BASE={"USDJPY","USDCHF","USDCAD"}
NONEU=[p for p in PAIRS if p!="EURUSD"]
LB=[1,3,5,10,15,30]
HOR=int(os.environ.get("MX_HOR","5")); GAP_S=HOR*60   # MX_HOR=15 tests the cross-pair block at the 15-min horizon
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],"test24":["2024"],"test25":["2025"],"oos":["2026"]}

def eu_equiv_sign(p): return -1.0 if p in USD_BASE else +1.0

def build_xp(years, stride=1):
    """Return DataFrame indexed by timestamp with cross-pair features + label _y + _ts (+ gate cols)."""
    out=[]
    for y in years:
        cl={}
        ok=True
        for p in PAIRS:
            fp=f"{FEAT}/{p}_{y}.parquet"
            if not os.path.exists(fp): ok=False; break
            d=pd.read_parquet(fp,columns=["close"]); d=d[~d.index.duplicated(keep="last")]
            cl[p]=d["close"]
        if not ok: continue
        df=pd.DataFrame(cl).dropna()
        if len(df)<100: continue
        idx=df.index; secs=idx.values.astype("datetime64[s]").astype("int64"); n=len(df)
        lr={p:np.log(df[p].values) for p in PAIRS}
        feats={}
        # per-lookback returns in EURUSD-equivalent terms
        euq={}  # euq[p][k]
        rets={}
        for p in PAIRS:
            rets[p]={}
            for k in LB:
                r=np.concatenate([[np.nan]*k, lr[p][k:]-lr[p][:-k]])
                rets[p][k]=r
        for k in LB:
            eu_r=rets["EURUSD"][k]
            basket=np.nanmean(np.vstack([eu_equiv_sign(p)*rets[p][k] for p in NONEU]),axis=0)  # USD-weakness basket = EURUSD-equiv
            disp=np.nanstd(np.vstack([eu_equiv_sign(p)*rets[p][k] for p in NONEU]),axis=0)
            agree=np.nanmean(np.vstack([(np.sign(eu_equiv_sign(p)*rets[p][k])==np.sign(basket)).astype(float) for p in NONEU]),axis=0)
            feats[f"eu_r{k}"]=eu_r
            feats[f"usdbask{k}"]=basket                  # basket move in EURUSD-up direction
            feats[f"catchup{k}"]=basket-eu_r             # EURUSD owes the basket move (predicts +)
            feats[f"eurresid{k}"]=eu_r-basket            # EUR idiosyncratic (beyond USD)
            feats[f"disp{k}"]=disp                       # cross-pair dispersion (low=coherent USD move)
            feats[f"agree{k}"]=agree                     # fraction of pairs agreeing with basket
            for p in NONEU:
                feats[f"ll_{p}{k}"]=eu_equiv_sign(p)*rets[p][k]-eu_r  # pair lead-lag residual vs EURUSD
        # session + compression gate cols (from EURUSD close only)
        hours=idx.hour.values+idx.minute.values/60.0
        feats["sess_ny"]=((hours>=13.0)&(hours<22.0)).astype(float)
        feats["sess_ln"]=((hours>=7.0)&(hours<16.0)).astype(float)
        feats["hour"]=hours
        # compression proxy: rolling 60-min std of 1-min EURUSD returns
        r1=rets["EURUSD"][1]
        comp=pd.Series(r1).rolling(60,min_periods=20).std().values
        feats["comp60"]=comp
        # label: next-5min EURUSD sign, wall-clock contiguous (exactly 300s), ties excluded
        fwd=np.full(n,np.nan)
        if n>HOR:
            contig=(secs[HOR:]-secs[:-HOR])==HOR*60
            fr=lr["EURUSD"][HOR:]-lr["EURUSD"][:-HOR]
            fwd[:n-HOR]=np.where(contig,fr,np.nan)
        yv=(fwd>0).astype(float)
        F=pd.DataFrame(feats,index=idx); F["_y"]=yv; F["_ts"]=secs; F["_fwd"]=fwd
        valid=np.isfinite(fwd)&(fwd!=0)
        F=F.loc[valid]
        if stride>1: F=F.iloc[::stride]
        out.append(F)
    return pd.concat(out)

def nonoverlap_chrono(ts, mask, gap=GAP_S):
    take=[]; block=-1
    for i in np.where(mask)[0]:
        if ts[i]<block: continue
        take.append(i); block=int(ts[i])+gap
    return np.array(take,dtype=int)

def boot(corr, nb=5000, seed=7):
    corr=np.asarray(corr,float)
    if len(corr)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(corr)
    a=np.array([corr[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))

XPCOLS=None
def xp_cols(df):
    return [c for c in df.columns if c not in ("_y","_ts","_fwd","hour")]

OF_COLS=['OF_of_norm_1','OF_of_sum_1','OF_of_norm_3','OF_of_sum_3','OF_of_norm_5','OF_of_sum_5','OF_of_norm_10',
    'OF_of_sum_10','OF_of_norm_15','OF_of_sum_15','OF_of_norm_30','OF_of_sum_30','OF_of_uptick_5','OF_of_uptick_15',
    'OF_kyle_5','OF_kyle_15','OF_of_accel','OF_of_persist']
OFDIR="/media/sean/CORSAIR/binary-algo/features_of"

def _read_years(dirpath, pair, years, cols):
    parts=[]
    for y in years:
        p=f"{dirpath}/{pair}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p,columns=cols); d=d[~d.index.duplicated(keep="last")]; parts.append(d)
    return pd.concat(parts) if parts else None

def augment(F, years, mode):
    """Join base(239) and/or order-flow(18) onto the XP frame F (index=timestamp), dropping overlaps."""
    if mode in ("xpbase","xpof"):
        B=_read_years(FEAT,"EURUSD",years,list(H.feature_cols("EURUSD")))
        if B is not None: F=F.join(B[[c for c in B.columns if c not in F.columns]],how="left")
    if mode=="xpof":
        O=_read_years(OFDIR,"EURUSD",years,OF_COLS)
        if O is not None: F=F.join(O[[c for c in O.columns if c not in F.columns]],how="left")
    return F

def feat_cols(mode, df, xpc):
    base=list(H.feature_cols("EURUSD"))
    cols=list(xpc)
    if mode in ("xpbase","xpof"): cols+=[c for c in base if c in df.columns]
    if mode=="xpof": cols+=[c for c in OF_COLS if c in df.columns]
    return list(dict.fromkeys(cols))

def covcurve(pr,y,ts,gate,thrs=None):
    """Descriptive per-window selective curve (NY-gated, non-overlap) at fixed CONF thresholds (cross-window comparable)."""
    out={}
    conf=np.abs(pr-0.5)
    for thr in (thrs or [0.04,0.06,0.08,0.10,0.13,0.16]):
        m=gate&(conf>=thr); sel=nonoverlap_chrono(ts,m)
        if len(sel)<20: out[thr]=(len(sel),float("nan")); continue
        acc=((pr[sel]>0.5).astype(int)==y[sel]).mean(); out[thr]=(len(sel),acc)
    return out

def main(mode="xp", stride=4):
    t0=time.time()
    print(f"[m5_xpair] mode={mode} stride_train={stride}",flush=True)
    TR=build_xp(SPL["train"],stride); VA=build_xp(SPL["val"])
    xpc=xp_cols(TR)
    TR=augment(TR,SPL["train"],mode); VA=augment(VA,SPL["val"],mode)
    cols=feat_cols(mode,TR,xpc)
    print(f"[m5_xpair] train={len(TR):,} val={len(VA):,} feats={len(cols)} build={time.time()-t0:.0f}s",flush=True)
    ytr=TR["_y"].astype(int).values; yva=VA["_y"].astype(int).values
    Xtr=TR[cols].astype("float32"); Xva=VA[cols].astype("float32")
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=127,min_child_samples=400,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,n_estimators=3000,n_jobs=20,verbosity=-1)
    L.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pva=L.predict_proba(Xva)[:,1]
    aucv=roc_auc_score(yva,pva)
    imp=sorted(zip(cols,L.feature_importances_),key=lambda z:-z[1])[:20]
    print(f"[m5_xpair] best_iter={L.best_iteration_} VAL AUC={aucv:.4f}",flush=True)
    print("  top20:",", ".join(f"{c}" for c,_ in imp),flush=True)
    # freeze operating point on VAL: gate=NY, pick coverage maximizing VAL indep acc (constrained n>=200)
    tsv=VA["_ts"].values.astype("int64"); gny=VA["sess_ny"].values>0.5; confv=np.abs(pva-0.5)
    best=None
    for cov in (0.10,0.05,0.03,0.02):
        if gny.sum()<1000: break
        thr=float(np.quantile(confv[gny],1-cov)); m=gny&(confv>=thr); sel=nonoverlap_chrono(tsv,m)
        if len(sel)<150: continue
        acc=((pva[sel]>0.5).astype(int)==yva[sel]).mean()
        if best is None or acc>best[0]: best=(acc,cov,thr,len(sel))
    accV,COV,THR,nV=best
    print(f"[m5_xpair] FROZEN gate=NY cov{COV:.0%} thr={THR:.4f} VAL_indep_acc={accV:.3f} (n{nV})",flush=True)
    # report held-out windows at frozen thr + dump diagnostics for offline conditional analysis
    allc=[]; diag={}
    # VAL diagnostics (for HONEST offline gate selection — select on VAL, verify held-out)
    _agc=[c for c in VA.columns if c.startswith("agree") or c.startswith("disp") or c=="comp60"]
    diag["val"]=dict(pr=pva.astype("float32"),y=yva.astype("int8"),ts=tsv,ny=gny,
        **{c:VA[c].values.astype("float32") for c in _agc},
        **{c:VA[c].values.astype("float32") for c in OF_COLS if c in VA.columns})
    for w in ("test24","test25","oos"):
        D=build_xp(SPL[w]); D=augment(D,SPL[w],mode)
        X=D[cols].astype("float32"); pr=L.predict_proba(X)[:,1]; y=D["_y"].astype(int).values
        ts=D["_ts"].values.astype("int64"); g=D["sess_ny"].values>0.5
        m=g&(np.abs(pr-0.5)>=THR); sel=nonoverlap_chrono(ts,m)
        corr=((pr[sel]>0.5).astype(int)==y[sel]).astype(float) if len(sel) else np.array([])
        acc=corr.mean() if len(sel) else float("nan"); lo,hi=boot(corr)
        aucw=roc_auc_score(y,pr)
        cc=covcurve(pr,y,ts,g)
        print(f"=== {w} === AUC={aucw:.4f} indep_trades={len(sel)} acc={acc:.3f} CI95=[{lo:.3f},{hi:.3f}]",flush=True)
        print("    NY covcurve(conf->n,acc): "+"  ".join(f"{t:.2f}:n{cc[t][0]},{cc[t][1]:.3f}" for t in cc),flush=True)
        allc.append(corr)
        # save full per-window arrays for offline analysis (agreement/OF/regime conditioning)
        agcols=[c for c in D.columns if c.startswith("agree") or c.startswith("disp") or c=="comp60"]
        ofd={c:D[c].values.astype("float32") for c in OF_COLS if c in D.columns}
        diag[w]=dict(pr=pr.astype("float32"),y=y.astype("int8"),ts=ts,ny=g,
                     **{c:D[c].values.astype("float32") for c in agcols},**ofd)
    A=np.concatenate(allc); acc=A.mean(); lo,hi=boot(A)
    print(f"=== HELD-OUT COMBINED === n={len(A)} acc={acc:.3f} CI95=[{lo:.3f},{hi:.3f}]  (breakeven~0.541)  {time.time()-t0:.0f}s",flush=True)
    np.savez_compressed(f"models/m5xp_{mode}_diag.npz",
        **{f"{w}__{k}":v for w,d in diag.items() for k,v in d.items()},
        frozen=np.array([COV,THR],dtype="float64"))
    print(f"[m5_xpair] saved diagnostics -> models/m5xp_{mode}_diag.npz",flush=True)

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "xp"
    stride=int(sys.argv[2]) if len(sys.argv)>2 else 4
    main(mode,stride)
