"""5m EURUSD — integrate Sofien Kaabar's CUSTOM indicators (orthogonal to the 239 base feats) + expansive training.

Adds Kaabar indicators (formulas verified from his corpus code) at multiple TFs {1,5,15,30,60}min, computed causally
from the 1m close: Disparity, MAD z-pos, RVI-on-volatility (RSI of vol direction), CMO, Trend Intensity Index,
KAMA Efficiency Ratio, vol-ratio (VAMA alpha), Choppiness-proxy, Fib-timing count. Top-orthogonal per the mine:
RVI-vol, KAMA-ER, TII, Fib-timing. Train LGB on base+Sofien, honest selective vs the base 0.518 OOS / 0.56 selective.

  python m5_sofien.py train <tag>   # cache probs -> models/m5_<tag>.npz ; eval prints honest selective per window
  python m5_sofien.py eval <tag>
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from m5_lab import HOR, GAP_S, base, REGIME, WINDOWS, nonoverlap_chrono, boot, indep_acc, thr_for_cov, build_gates, MODELS

def rma(s,n): return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def _disp(c,n): return (c/c.rolling(n).mean()-1)*100
def _madz(c,n,k=2.0):
    ma=c.rolling(n).mean(); mad=(c-ma)/ma; return mad/(mad.rolling(n).std()*k+1e-12)
def _rvivol(c,n):
    vol=c.rolling(n).std(); dv=vol.diff(); ru=rma(dv.clip(lower=0),n); rd=rma((-dv).clip(lower=0),n)
    return 100-100/(1+ru/(rd+1e-12))
def _cmo(c,n):
    d=c.diff(); su=d.clip(lower=0).rolling(n).sum(); sd=(-d).clip(lower=0).rolling(n).sum()
    return (su-sd)/(su+sd+1e-12)*100
def _tii(c,n):
    ma=c.rolling(n).mean(); up=(c>ma).rolling(n).sum(); dn=(c<ma).rolling(n).sum(); return up/(up+dn+1e-12)*100
def _kamaer(c,n): return ((c-c.shift(n)).abs()/(c.diff().abs().rolling(n).sum()+1e-12)).fillna(0)
def _volr(c,a,b): return c.rolling(a).std()/(c.rolling(b).std()+1e-12)
def _chop(c,n):
    r=c.diff().abs(); rng=c.rolling(n).max()-c.rolling(n).min()
    return 100*np.log10((r.rolling(n).sum())/(rng+1e-12)+1e-12)/np.log10(n)
def _fibtiming(c, count=8, s1=5, s2=3, s3=2):
    cv=c.values; n=len(cv); buy=np.zeros(n); sell=np.zeros(n); cb=-1; cs=1
    for i in range(s1,n):
        if cv[i]<cv[i-s1] and cv[i]<cv[i-s2] and cv[i]<cv[i-s3]:
            buy[i]=cb; cb-=1
            if cb==-count-1: cb=0
        elif cv[i]>=cv[i-s1]: cb=-1
        if cv[i]>cv[i-s1] and cv[i]>cv[i-s2] and cv[i]>cv[i-s3]:
            sell[i]=cs; cs+=1
            if cs==count+1: cs=0
        elif cv[i]<=cv[i-s1]: cs=1
    return pd.Series(buy+sell, index=c.index)

SOF=[("disp",_disp,14),("madz",_madz,20),("rvivol",_rvivol,10),("cmo",_cmo,14),
     ("tii",_tii,30),("kamaer",_kamaer,10),("volr",lambda c,n:_volr(c,5,n),20),("chop",_chop,14)]
def sofien_block(close1m):
    idx=close1m.index; out={}
    for tf in (1,5,15,30,60):
        c = close1m if tf==1 else close1m.resample(f"{tf}min").last().dropna()
        for nm,fn,n in SOF:
            v=fn(c,n)
            out[f"SF{tf}_{nm}"]= v.reindex(idx) if tf==1 else v.shift(1).reindex(idx,method="ffill")
        ft=_fibtiming(c)
        out[f"SF{tf}_fib"]= ft.reindex(idx) if tf==1 else ft.shift(1).reindex(idx,method="ffill")
    return pd.DataFrame(out,index=idx).replace([np.inf,-np.inf],np.nan).astype("float32")

SOF_COLS=None
def load(years, stride=1):
    global SOF_COLS
    parts=[]
    for y in years:
        p=f"{H.FEAT_DIR}/EURUSD_{y}.parquet"
        if not os.path.exists(p): continue
        df=pd.read_parquet(p, columns=base+["close"]); df=df[~df.index.duplicated(keep="last")]
        sof=sofien_block(df["close"])
        c=df["close"].values; n=len(c); secs=df.index.values.astype("datetime64[s]").astype("int64")
        contig=np.zeros(n,bool)
        if n>HOR: contig[:n-HOR]=(secs[HOR:]-secs[:-HOR])==HOR*60
        fwd=np.full(n,np.nan); fwd[:n-HOR]=c[HOR:]; ret=fwd/c-1.0
        yv=(ret>0).astype(float); valid=contig&np.isfinite(ret)&(ret!=0)
        d=pd.concat([df[base], sof], axis=1).loc[valid].copy()
        d["_y"]=yv[valid]; d["_ts"]=secs[valid]
        parts.append(d.iloc[::stride] if stride>1 else d)
    D=pd.concat(parts); SOF_COLS=[c for c in D.columns if c.startswith("SF")]
    return D

def main_train(tag):
    t0=time.time(); os.makedirs(MODELS,exist_ok=True); STRIDE=3
    print(f"[sofien:{tag}] loading TRAIN+sofien stride{STRIDE}...",flush=True)
    TR=load([str(y) for y in range(2012,2022)],STRIDE); VA=load(WINDOWS["val"])
    feats=base+SOF_COLS
    ytr=TR["_y"].astype(int).values; yva=VA["_y"].astype(int).values
    print(f"[sofien:{tag}] train={len(TR):,} val={len(VA):,} base={len(base)} sofien={len(SOF_COLS)} feats={len(feats)} load={time.time()-t0:.0f}s",flush=True)
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=255,min_child_samples=300,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=10,n_estimators=3000,n_jobs=20,verbosity=-1)
    L.fit(TR[feats].astype("float32"),ytr,eval_set=[(VA[feats].astype("float32"),yva)],eval_metric="auc",
          callbacks=[lgb.early_stopping(200),lgb.log_evaluation(0)])
    print(f"[sofien:{tag}] lgb iter={L.best_iteration_} {time.time()-t0:.0f}s",flush=True)
    # feature importance: how many top-30 are Sofien?
    imp=pd.Series(L.feature_importances_, index=feats).sort_values(ascending=False)
    top30=imp.head(30); nsof=sum(1 for c in top30.index if c.startswith("SF"))
    print(f"[sofien:{tag}] top-30 importances: {nsof} are Sofien. Top Sofien: "+", ".join([c for c in imp.index if c.startswith('SF')][:6]),flush=True)
    qthr={tf:None for tf in ()}
    qthr={}
    for c in REGIME:
        if c.endswith("bb_width") and c in TR.columns:
            v=TR[c].values.astype(float); qthr[c]=float(np.nanpercentile(v,33)); qthr[c+"_hi"]=float(np.nanpercentile(v,67))
    cache={"feats":json.dumps(feats),"qthr":json.dumps(qthr)}; aucs={}
    for w,yrs in WINDOWS.items():
        D=load(yrs); p=L.predict_proba(D[feats].astype("float32"))[:,1]
        y=D["_y"].astype(int).values; ts=D["_ts"].values.astype("int64"); aucs[w]=float(roc_auc_score(y,p))
        cache[f"{w}_p"]=p.astype("float32"); cache[f"{w}_y"]=y.astype("int8"); cache[f"{w}_ts"]=ts
        for c in REGIME:
            if c in D.columns: cache[f"{w}_r_{c}"]=D[c].values.astype("float32")
        print(f"[sofien:{tag}] {w}: n={len(y):,} AUC={aucs[w]:.4f}",flush=True); del D
    np.savez_compressed(f"{MODELS}/m5_{tag}.npz", **cache)
    print(f"[sofien:{tag}] cached. AUC "+" ".join(f"{w}={aucs[w]:.4f}" for w in WINDOWS)+f" (base was oos=0.5175) DONE {time.time()-t0:.0f}s",flush=True)
    main_eval(tag)

def _load_window(z,w):
    R={"_y":z[f"{w}_y"],"_p":z[f"{w}_p"],"_ts":z[f"{w}_ts"]}
    for k in z.files:
        if k.startswith(f"{w}_r_"): R[k[len(f"{w}_r_"):]]=z[k]
    return R
def main_eval(tag, covs=(0.05,0.02,0.01)):
    z=np.load(f"{MODELS}/m5_{tag}.npz",allow_pickle=True); qthr=json.loads(str(z["qthr"]))
    W={w:_load_window(z,w) for w in WINDOWS}
    print(f"\n===== EVAL m5_{tag} (base+Sofien) =====  indep {GAP_S}s chrono; CI95; breakeven~0.541")
    print("AUC: "+"  ".join(f"{w}={roc_auc_score(W[w]['_y'],W[w]['_p']):.4f}" for w in WINDOWS))
    Gs={w:build_gates(W[w],qthr) for w in WINDOWS}
    for gname in ("none","ny","comp_1h","comp1h_ny","comp30_ny","comp15_ny"):
        if gname not in Gs["val"] or Gs["val"][gname].sum()<200: continue
        print(f"\n--- gate={gname} ---")
        for cov in covs:
            thr=thr_for_cov(W["val"]["_p"],W["val"]["_y"],Gs["val"][gname],cov)
            if thr is None: continue
            row=[]
            for w in WINDOWS:
                g=Gs[w].get(gname,np.ones(len(W[w]["_y"]),bool)); n,acc,lo,hi,ev=indep_acc(W[w]["_p"],W[w]["_y"],W[w]["_ts"],g,thr)
                row.append(f"{w}:n{n} {acc:.3f}[{lo:.2f},{hi:.2f}]" if n else f"{w}:n0")
            print(f"  cov{cov:.0%}  "+"  ".join(row))

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "eval"; tag=sys.argv[2] if len(sys.argv)>2 else "sofien"
    (main_train if mode=="train" else main_eval)(tag)
