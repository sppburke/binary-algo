"""5m EURUSD — test the NEW sofien price-action SETUPS (untested at any horizon) for a 5-minute binary.

Builds 5-minute OHLC from the 10s source (cached). Each setup fires at a 5m bar close = a 5-min decision; label =
sign(close(next 5m bar) - close(this bar)) with contiguity (next bar exactly +300s, no session gap). Trades are
naturally non-overlapping (one per 5m boundary). Honest: report accuracy + CI95 per window (train-ish / VAL / TEST24 /
TEST25 / OOS26); a real edge holds across ALL held-out windows.

Setups (from sofien corpus deep-mine):
  strat_322  : bar[-2]=outside(3), bar[-1]=directional(2 down), break high[-1] -> long (mirror short). Hit 67.4% n181.
  strat_312  : bar[-2]=outside(3), bar[-1]=inside(1), break inside high -> long (mirror). Hit 81.6% n629 (PF<1).
  round_rev  : close AT a 50-pip round level having come from above -> long (support bounce); from below -> short.
  round_mom  : round level & rsi14<50 & bearish close -> long (mirror). (momentum-filtered variant)
  nr7        : narrowest range in 7 bars, then break its high->long / low->short (trend-filtered by sma).

  python m5_patterns.py
"""
import os, glob, numpy as np, pandas as pd, time

RAW="/home/sean/git/processed/EURUSD"
CACHE="/home/sean/git/binary-algo/ohlc_cache"; os.makedirs(CACHE,exist_ok=True)
WINDOWS={"train":[str(y) for y in range(2016,2022)],"val":["2022","2023"],"test24":["2024"],"test25":["2025"],"oos":["2026"]}

def build_5m(year):
    cp=f"{CACHE}/EURUSD_5m_{year}.parquet"
    if os.path.exists(cp): return pd.read_parquet(cp)
    files=sorted(glob.glob(f"{RAW}/EURUSD_10s_{year}-*.parquet"))
    if not files: return None
    dfs=[]
    for f in files:
        d=pd.read_parquet(f)
        if "datetime_utc" in d.columns: d=d.set_index("datetime_utc")
        d.index=pd.to_datetime(d.index, utc=True, errors="coerce"); dfs.append(d[["open","high","low","close"]])
    df=pd.concat(dfs); df=df[df.index.notna()]; df=df[~df.index.duplicated(keep="last")].sort_index()
    o=df["open"].resample("5min").first(); h=df["high"].resample("5min").max()
    l=df["low"].resample("5min").min(); c=df["close"].resample("5min").last()
    m=pd.DataFrame({"open":o,"high":h,"low":l,"close":c}).dropna(subset=["close"])
    m.to_parquet(cp); return m

def rsi(c,n=14):
    d=c.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return (100-100/(1+up/dn.replace(0,np.nan))).fillna(50.0)

def signals(m):
    """Return dict name -> pred array (+1 long / -1 short / 0 none), aligned to m bars (decision at bar close)."""
    o,h,l,c=m["open"].values,m["high"].values,m["low"].values,m["close"].values; n=len(c)
    # bar typing vs prior bar
    bt=np.zeros(n,int)
    for i in range(1,n):
        if h[i]>h[i-1] and l[i]<l[i-1]: bt[i]=3
        elif h[i]<=h[i-1] and l[i]>=l[i-1]: bt[i]=1
        else: bt[i]=2
    down=c<o; up=c>o
    g={}
    # Strat 3-2-2: t-2 outside, t-1 directional-down, current bar breaks high[t-1] -> long
    s322=np.zeros(n,int)
    for i in range(2,n):
        if bt[i-2]==3 and bt[i-1]==2 and down[i-1] and h[i]>h[i-1]: s322[i]=1
        if bt[i-2]==3 and bt[i-1]==2 and up[i-1]   and l[i]<l[i-1]: s322[i]=-1
    g["strat_322"]=s322
    # Strat 3-1-2: t-2 outside, t-1 inside, current breaks inside high -> long
    s312=np.zeros(n,int)
    for i in range(2,n):
        if bt[i-2]==3 and bt[i-1]==1 and h[i]>h[i-1]: s312[i]=1
        if bt[i-2]==3 and bt[i-1]==1 and l[i]<l[i-1]: s312[i]=-1
    g["strat_312"]=s312
    # round-number reaction (50-pip grid). closed at round level.
    grid=0.0050; nearest=np.round(c/grid)*grid; at_round=np.abs(c-nearest)<0.00005
    prev3_above=np.zeros(n,bool); prev3_below=np.zeros(n,bool)
    for i in range(3,n):
        lvl=nearest[i]; prev3_above[i]=np.all(c[i-3:i]>lvl); prev3_below[i]=np.all(c[i-3:i]<lvl)
    rrev=np.zeros(n,int); rrev[at_round&prev3_above]=1; rrev[at_round&prev3_below]=-1   # support bounce / resistance
    g["round_rev"]=rrev
    rsiv=rsi(m["close"]).values
    rmom=np.zeros(n,int); rmom[at_round&(rsiv<50)&down]=1; rmom[at_round&(rsiv>50)&up]=-1
    g["round_mom"]=rmom
    # NR7 breakout, trend-filtered by sma50
    rng=h-l; sma=pd.Series(c).rolling(50).mean().values
    nr7=np.zeros(n,bool)
    for i in range(6,n): nr7[i]=rng[i]==rng[i-6:i+1].min()
    nb=np.zeros(n,int)
    for i in range(7,n):
        if nr7[i-1] and h[i]>h[i-1] and c[i]>sma[i]: nb[i]=1
        if nr7[i-1] and l[i]<l[i-1] and c[i]<sma[i]: nb[i]=-1
    g["nr7_brk"]=nb
    return g

def boot(corr,nb=4000,seed=7):
    corr=np.asarray(corr,float)
    if len(corr)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(corr)
    a=np.array([corr[rng.integers(0,n,n)].mean() for _ in range(nb)]); return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))

def main():
    t0=time.time(); data={}
    for w,yrs in WINDOWS.items():
        ms=[build_5m(y) for y in yrs]; ms=[x for x in ms if x is not None]
        m=pd.concat(ms); m=m[~m.index.duplicated(keep="last")].sort_index()
        c=m["close"].values; secs=m.index.values.astype("datetime64[s]").astype("int64"); n=len(c)
        fwd=np.full(n,np.nan); fwd[:n-1]=c[1:]; contig=np.zeros(n,bool); contig[:n-1]=(secs[1:]-secs[:-1])==300
        ret=fwd/c-1.0; y=(ret>0).astype(int); valid=contig&np.isfinite(ret)&(ret!=0)
        data[w]={"m":m,"sig":signals(m),"y":y,"valid":valid}
        print(f"[patterns] {w}: 5m_bars={n:,} valid={int(valid.sum()):,} ({time.time()-t0:.0f}s)",flush=True)
    names=list(data["train"]["sig"].keys())
    print(f"\n{'setup':12s} "+" ".join(f"{w:>20s}" for w in WINDOWS))
    for nm in names:
        cells=[]
        for w in WINDOWS:
            d=data[w]; pr=d["sig"][nm]; m=(pr!=0)&d["valid"]
            if m.sum()==0: cells.append("n0".rjust(20)); continue
            pu=(pr[m]>0); corr=(pu==(d["y"][m]==1)).astype(float)
            acc=corr.mean(); lo,hi=boot(corr); cells.append(f"n{int(m.sum()):>5} {acc:.3f}[{lo:.2f},{hi:.2f}]".rjust(20))
        print(f"{nm:12s} "+" ".join(cells),flush=True)
    print(f"\n[patterns] DONE {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
