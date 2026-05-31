"""Test the top Sofien HIGH-PRECISION confluence rules (quality_tier=high) as DISCRETE 5-min directional signals.
Prior session tested individual indicators (0 OOS AUC) and structural setups (null) — but NOT these specific confluence
combos (trend-filtered Connors RSI2, BB+RSI reversion, pullback-in-trend). Honest OOS test across TEST24/TEST25/OOS26,
non-overlap 300s, CI95. Direction predicted by the rule vs sign(close(t+5m)-close(t)).
"""
import numpy as np, pandas as pd
FEAT="/media/sean/CORSAIR/binary-algo/features"; HOR=5
def rsi(c,n):
    d=np.diff(c,prepend=c[0]); up=np.where(d>0,d,0.0); dn=np.where(d<0,-d,0.0)
    ru=pd.Series(up).ewm(alpha=1/n,adjust=False).mean().values; rd=pd.Series(dn).ewm(alpha=1/n,adjust=False).mean().values
    return 100-100/(1+ru/(rd+1e-12))
def sma(c,n): return pd.Series(c).rolling(n,min_periods=n//2).mean().values
def boot(corr,nb=3000,seed=7):
    corr=np.asarray(corr,float)
    if len(corr)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(corr); a=np.array([corr[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))
def nonoverlap(ts,idx,gap=300):
    take=[]; block=-1
    for i in idx:
        if ts[i]<block: continue
        take.append(i); block=int(ts[i])+gap
    return np.array(take,dtype=int)

def load(y):
    df=pd.read_parquet(f"{FEAT}/EURUSD_{y}.parquet",columns=["close","sess_ny"]); df=df[~df.index.duplicated(keep="last")]
    c=df["close"].values; n=len(c); secs=df.index.values.astype("datetime64[s]").astype("int64")
    fwd=np.full(n,np.nan)
    if n>HOR:
        contig=(secs[HOR:]-secs[:-HOR])==HOR*60; fwd[:n-HOR]=np.where(contig,c[HOR:]/c[:-HOR]-1.0,np.nan)
    return c,secs,fwd,(df["sess_ny"].values.astype(float)>0.5)

def rules(c):
    r2=rsi(c,2); r14=rsi(c,14); s200=sma(c,200); s20=sma(c,20)
    std20=pd.Series(c).rolling(20,min_periods=10).std().values; mb=sma(c,20)
    lower=mb-2*std20; upper=mb+2*std20
    above=c>s200; below=c<s200
    pcb=(c<s20)&(np.concatenate([[False],c[:-1]>=s20[:-1]]))  # just crossed below sma20
    pca=(c>s20)&(np.concatenate([[False],c[:-1]<=s20[:-1]]))
    # each rule -> dir array: +1 long, -1 short, 0 none
    R={}
    d=np.zeros(len(c)); d[(r2<10)&above]=1; d[(r2>90)&below]=-1; R["trend_rsi2(200sma)"]=d.copy()
    d=np.zeros(len(c)); d[(r2<5)]=1; d[(r2>95)]=-1; R["rsi2_pure"]=d.copy()
    d=np.zeros(len(c)); d[(c<lower)&(r14<30)]=1; d[(c>upper)&(r14>70)]=-1; R["bb_rsi_revert"]=d.copy()
    d=np.zeros(len(c)); d[above&pcb]=1; d[below&pca]=-1; R["pullback_in_trend"]=d.copy()
    d=np.zeros(len(c)); d[(r2<10)&above]=1; d[(r2>90)&below]=-1
    d[(r2<5)&below]=0; R["trend_rsi2_strict"]=d.copy()  # only trend-aligned
    return R

def run():
    data={y:load(y) for y in ("2024","2025","2026")}
    names=list(rules(data["2024"][0]).keys())
    print(f"{'rule':<22}{'gate':<6} "+ "".join(f"{w:>16}" for w in ("test24","test25","oos"))+f"{'COMBINED':>20}")
    for gate in ("all","ny"):
        for nm in names:
            allc=[]; cells=[]
            for y,w in (("2024","test24"),("2025","test25"),("2026","oos")):
                c,secs,fwd,ny=data[y]; R=rules(c)[nm]
                sig=(R!=0)&np.isfinite(fwd)&(fwd!=0)
                if gate=="ny": sig=sig&ny
                idx=np.where(sig)[0]; sel=nonoverlap(secs,idx)
                if len(sel)==0: cells.append("n0"); allc.append(np.array([])); continue
                pred=R[sel]; ytrue=np.sign(fwd[sel]); corr=(pred==ytrue).astype(float)
                allc.append(corr); cells.append(f"{corr.mean():.3f}(n{len(sel)})")
            A=np.concatenate(allc);
            if len(A)<20: continue
            lo,hi=boot(A); cells.append(f"{A.mean():.3f}[{lo:.3f},{hi:.3f}]")
            print(f"{nm:<22}{gate:<6} "+"".join(f"{x:>16}" for x in cells))
if __name__=="__main__": run()
