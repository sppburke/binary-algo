"""NON-TIME BARS — volume & dollar bars from 10s OHLCV (Lopez de Prado AFML ch2). The "different story" test.

Sample the market by INFORMATION ARRIVAL (cumulative volume / dollar) instead of the clock. Build features on the
event-bar series (incl. BAR DURATION = info speed) and predict the tradeable 30-MINUTE time-forward direction. If
event-bar sampling reveals predictability the time bars hide, AUC/selective should beat the ~0.52 time-bar wall.

  python vbars.py build      # build volume & dollar bar caches (per year) from 10s
  python vbars.py model V     # train LGB on volume bars -> 30m label, honest selective
  python vbars.py model D     # dollar bars
"""
import os, sys, glob, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
RAW="/media/sean/CORSAIR/tick_data/processed/EURUSD"
CACHE="/media/sean/CORSAIR/binary-algo/vbar_cache"; os.makedirs(CACHE,exist_ok=True)
HOR_S=1800
V_THR=1.0e9   # ~5-min-equivalent volume per bar (median 10s vol ~3.5e7 * ~30)
D_THR=1.5e9   # ~5-min-equivalent dollar per bar
WINDOWS={"train":[str(y) for y in range(2016,2022)],"val":["2022","2023"],"test24":["2024"],"test25":["2025"],"oos":["2026"]}

def _load10s(year):
    fs=sorted(glob.glob(f"{RAW}/EURUSD_10s_{year}-*.parquet"))
    if not fs: return None
    dfs=[]
    for f in fs:
        d=pd.read_parquet(f)
        if "datetime_utc" in d.columns: d=d.set_index("datetime_utc")
        d.index=pd.to_datetime(d.index,utc=True,errors="coerce"); dfs.append(d[["open","high","low","close","volume"]])
    df=pd.concat(dfs); df=df[df.index.notna()]; df=df[~df.index.duplicated(keep="last")].sort_index()
    for c in ("open","high","low","close","volume"): df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.dropna(subset=["close","volume"])

def _bars(df, key, thr):
    """Vectorized event bars: assign each 10s row to a bar by floor(cumsum(key)/thr)."""
    ts=df.index.values.astype("datetime64[s]").astype("int64")
    o=df["open"].values; h=df["high"].values; l=df["low"].values; c=df["close"].values; v=df["volume"].values
    accum = v if key=="V" else c*v
    bid=np.floor(np.cumsum(accum)/thr).astype("int64")
    # group by bid (contiguous) -> bar OHLC, vol, vwap, t_close, n10s
    chg=np.r_[True, bid[1:]!=bid[:-1]]; starts=np.where(chg)[0]; ends=np.r_[starts[1:],len(bid)]
    rows=[]
    for s,e in zip(starts,ends):
        sl=slice(s,e)
        rows.append((ts[e-1], o[s], h[sl].max(), l[sl].min(), c[e-1], v[sl].sum(),
                     (c[sl]*v[sl]).sum()/(v[sl].sum()+1e-12), e-s, ts[e-1]-ts[s]+10))
    b=pd.DataFrame(rows, columns=["t","open","high","low","close","vol","vwap","n10s","dur_s"])
    # 30m-forward close from the 10s series (deriv-faithful: price 1800s after bar close), contiguity-enforced
    tgt=b["t"].values+HOR_S
    j=np.searchsorted(ts, tgt, side="left"); j=np.clip(j,0,len(ts)-1)
    ok=(np.abs(ts[j]-tgt)<=30)  # forward price within 30s of +1800s (no gap)
    fwd=c[j]; ret=fwd/b["close"].values-1.0
    b["y"]=(ret>0).astype(int); b["valid"]=ok&np.isfinite(ret)&(ret!=0); b["ret"]=ret
    return b

def build():
    t0=time.time()
    for year in [str(y) for y in range(2016,2027)]:
        for key,thr,tag in (("V",V_THR,"V"),("D",D_THR,"D")):
            cp=f"{CACHE}/{tag}_{year}.parquet"
            if os.path.exists(cp): continue
            df=_load10s(year)
            if df is None: continue
            b=_bars(df,key,thr); b.to_parquet(cp)
            print(f"[build] {tag} {year}: bars={len(b):,} valid={int(b['valid'].sum()):,} med_dur={b['dur_s'].median():.0f}s ({time.time()-t0:.0f}s)",flush=True)

def feats(b):
    c=b["close"]; X=pd.DataFrame(index=b.index)
    r=c.pct_change()
    for k in (1,2,3,5,8,13,21): X[f"ret{k}"]=c.pct_change(k)
    for k in (5,10,20,50): X[f"dist_ema{k}"]=c/c.ewm(span=k).mean()-1
    d=c.diff()
    for n in (7,14,21):
        X[f"rsi{n}"]=100-100/(1+d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()/((-d).clip(lower=0).ewm(alpha=1/n,adjust=False).mean()+1e-12))
    for n in (10,20,50): X[f"rv{n}"]=r.rolling(n).std()
    for n in (14,30):
        hh=b["high"].rolling(n).max(); ll=b["low"].rolling(n).min(); X[f"rangepos{n}"]=(c-ll)/(hh-ll+1e-12)
    X["vwap_dist"]=c/b["vwap"]-1
    X["dur_s"]=b["dur_s"]; X["dur_z"]=(b["dur_s"]-b["dur_s"].rolling(50).mean())/(b["dur_s"].rolling(50).std()+1e-9)  # info speed
    X["vol_z"]=(b["vol"]-b["vol"].rolling(50).mean())/(b["vol"].rolling(50).std()+1e-9)
    X["n10s"]=b["n10s"]; X["ibs"]=(c-b["low"])/(b["high"]-b["low"]+1e-12)
    X["bb_pctb"]=(c-c.rolling(20).mean())/(2*c.rolling(20).std()+1e-12)
    X["ac10"]=r.rolling(10).corr(r.shift(1))
    hh=c.index.to_series().dt.hour if isinstance(c.index,pd.DatetimeIndex) else None
    return X.replace([np.inf,-np.inf],np.nan)

def load_bars(tag, years):
    parts=[pd.read_parquet(f"{CACHE}/{tag}_{y}.parquet") for y in years if os.path.exists(f"{CACHE}/{tag}_{y}.parquet")]
    b=pd.concat(parts).reset_index(drop=True)
    b.index=pd.to_datetime(b["t"],unit="s",utc=True)
    return b

def nonoverlap_chrono(ts,mask,gap=HOR_S):
    take=[]; bu=-1
    for i in np.where(mask)[0]:
        if ts[i]<bu: continue
        take.append(i); bu=int(ts[i])+gap
    return np.array(take,dtype=int)
def boot(corr,nb=3000,seed=7):
    corr=np.asarray(corr,float)
    if len(corr)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(corr); a=np.array([corr[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))

def model(tag):
    t0=time.time()
    B={w:load_bars(tag,yrs) for w,yrs in WINDOWS.items()}
    for w in B: B[w]["_X"]=None
    Xall={w:feats(B[w]) for w in WINDOWS}
    feat=[c for c in Xall["train"].columns]
    TR=Xall["train"]; ytr=B["train"]["y"].values; vtr=B["train"]["valid"].values
    VA=Xall["val"]; yva=B["val"]["y"].values; vva=B["val"]["valid"].values
    itr=np.where(vtr)[0]; iva=np.where(vva)[0]
    print(f"[model:{tag}] train_bars={len(TR):,} valid={len(itr):,} val_valid={len(iva):,} feats={len(feat)} med_dur(train)={B['train']['dur_s'].median():.0f}s ({time.time()-t0:.0f}s)",flush=True)
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=255,min_child_samples=100,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=8,n_estimators=2000,n_jobs=20,verbosity=-1)
    L.fit(TR.iloc[itr],ytr[itr],eval_set=[(VA.iloc[iva],yva[iva])],eval_metric="auc",callbacks=[lgb.early_stopping(120),lgb.log_evaluation(0)])
    print(f"[model:{tag}] lgb iter={L.best_iteration_} {time.time()-t0:.0f}s",flush=True)
    print(f"[model:{tag}] AUC "+" ".join(f"{w}={roc_auc_score(B[w]['y'].values[B[w]['valid'].values], L.predict_proba(Xall[w].iloc[np.where(B[w]['valid'].values)[0]])[:,1]):.4f}" for w in WINDOWS),flush=True)
    # honest selective (non-overlap chrono 1800s on bar t_close)
    print(f"\n{'gate':10s} "+" ".join(f"{w:>20s}" for w in WINDOWS))
    for cov in (0.10,0.05,0.02,0.01):
        pv=L.predict_proba(VA)[:,1]; conf=np.abs(pv-0.5); thr=float(np.quantile(conf[iva],1-cov))
        cells=[]
        for w in WINDOWS:
            p=L.predict_proba(Xall[w])[:,1]; v=B[w]["valid"].values; ts=B[w]["t"].values.astype("int64")
            m=v&(np.abs(p-0.5)>=thr); sel=nonoverlap_chrono(ts,m)
            if len(sel)==0: cells.append("n0".rjust(20)); continue
            corr=((p[sel]>0.5).astype(int)==B[w]["y"].values[sel]).astype(float); acc=corr.mean(); lo,hi=boot(corr)
            cells.append(f"n{len(sel):>5} {acc:.3f}[{lo:.2f},{hi:.2f}]".rjust(20))
        print(f"cov{cov:.0%}    "+" ".join(cells),flush=True)
    print(f"\n[model:{tag}] DONE {time.time()-t0:.0f}s  (time-bar 30m baseline AUC ~0.52, selective ~0.59)",flush=True)

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "build"
    if mode=="build": build()
    else: model(sys.argv[2] if len(sys.argv)>2 else "V")
