"""30m EURUSD — ORTHOGONAL-MATH PREDICTABILITY GATES ("avoid losers").

Hypothesis: 30m direction is ~0.52 predictable ON AVERAGE, but the average mixes predictable + coin-flip regimes.
Complexity/persistence measures from nonlinear dynamics tag the predictable subset. Gate the bet there; abstain elsewhere.
Measures (all causal, trailing-window, on 1m log-returns): PERMUTATION ENTROPY (Bandt-Pompe), lag-1 return
AUTOCORRELATION (ac20), VARIANCE-RATIO Hurst proxy. Direction = momentum (sign of recent move) or reversion (opposite).
We bin decision points by each measure and ask: does conditional 30m-direction accuracy RISE in predictable bins, and
does it hold across TEST24/TEST25/OOS26? Honest: per-bin non-overlap (1800s) accuracy + CI for any promising bin.

  python m30_complexity.py
"""
import os, math, time, numpy as np, pandas as pd
import harness as H
HOR=30; GAP_S=HOR*60
WINDOWS={"train":[str(y) for y in range(2016,2022)],"val":["2022","2023"],"test24":["2024"],"test25":["2025"],"oos":["2026"]}

def perm_entropy(r, d=3, tau=1, W=60):
    N=len(r); L=(d-1)*tau
    if N<=L+1: return np.full(N,np.nan)
    idx=np.arange(N-L)[:,None]+np.arange(0,d*tau,tau)[None,:]
    emb=r[idx]; order=np.argsort(emb,axis=1,kind="stable")
    code=(order*(d**np.arange(d))).sum(1).astype(np.int32)
    M=len(code); nb=d**d
    oh=np.zeros((M,nb),dtype=np.float32); oh[np.arange(M),code]=1.0
    cs=np.cumsum(oh,axis=0); cnt=cs.copy(); cnt[W:]=cs[W:]-cs[:-W]
    tot=np.maximum(cnt.sum(1,keepdims=True),1); p=cnt/tot
    with np.errstate(divide='ignore',invalid='ignore'):
        ent=-np.nansum(np.where(p>0,p*np.log(p),0.0),axis=1)/math.log(math.factorial(d))
    out=np.full(N,np.nan); out[L:L+M]=ent; out[:L+W]=np.nan
    return out

def load_window(years):
    base=pd.concat([pd.read_parquet(f"{H.FEAT_DIR}/EURUSD_{y}.parquet",columns=["close"]) for y in years])
    base=base[~base.index.duplicated(keep="last")].sort_index()
    c=base["close"].values.astype(float); n=len(c); secs=base.index.values.astype("datetime64[s]").astype("int64")
    contig=np.zeros(n,bool); contig[:n-HOR]=(secs[HOR:]-secs[:-HOR])==HOR*60
    fwd=np.full(n,np.nan); fwd[:n-HOR]=c[HOR:]; ret=fwd/c-1
    y=(ret>0).astype(int); valid=contig&np.isfinite(ret)&(ret!=0)
    r=np.zeros(n); r[1:]=np.diff(np.log(c))
    rs=pd.Series(r)
    pe30=perm_entropy(r,3,1,30); pe60=perm_entropy(r,3,1,60); pe120=perm_entropy(r,3,1,120)
    ac20=rs.rolling(20).corr(rs.shift(1)).values
    # variance-ratio Hurst proxy over 60: var(sum_k r)/k vs var(r) ; H = 0.5*log(VR)/log(k)+0.5
    k=10; agg=rs.rolling(k).sum(); vr=(agg.rolling(120).var())/(k*rs.rolling(120).var()+1e-18)
    hurst=(0.5*np.log(vr.clip(1e-6,1e6))/math.log(k)+0.5).values
    mom30=np.sign(c-np.concatenate([np.full(30,np.nan),c[:-30]]))   # sign of recent 30m move
    return dict(c=c,ts=secs,y=y,valid=valid,pe30=pe30,pe60=pe60,pe120=pe120,ac20=ac20,hurst=hurst,mom=mom30)

def nonoverlap_chrono(ts,mask,gap=GAP_S):
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

def bin_accuracy(data, measure, direction, nbins=5, label=""):
    """Overlapping conditional accuracy of `direction` signal across quantile bins of `measure`. Shows the SHAPE."""
    edges=np.nanquantile(data["train"][measure][data["train"]["valid"]], np.linspace(0,1,nbins+1))
    edges[0]=-np.inf; edges[-1]=np.inf
    print(f"\n[{label}] accuracy of {direction}-signal by {measure} quintile (overlapping; base 0.50):")
    print(f"  {'bin':>18} "+" ".join(f"{w:>14}" for w in WINDOWS))
    for b in range(nbins):
        cells=[]
        for w in WINDOWS:
            d=data[w]; mv=d[measure]; m=d["valid"]&(mv>=edges[b])&(mv<edges[b+1])&np.isfinite(d["mom"])&(d["mom"]!=0)
            if m.sum()<50: cells.append("n<50".rjust(14)); continue
            pred = d["mom"][m] if direction=="momentum" else -d["mom"][m]
            acc=((pred>0)==(d["y"][m]==1)).mean()
            cells.append(f"{acc:.3f}(n{int(m.sum())//1000}k)".rjust(14))
        print(f"  [{edges[b]:+.2e},{edges[b+1]:+.2e}) "+" ".join(cells))

def gate_eval(data, name, gate_fn, dir_fn):
    """Honest INDEPENDENT non-overlap accuracy + CI of a specific gate+direction across windows."""
    cells=[]; held=[]
    for w in WINDOWS:
        d=data[w]; fire=gate_fn(d)&d["valid"]; dirn=dir_fn(d)
        m=fire&np.isfinite(dirn)&(dirn!=0)
        if m.sum()==0: cells.append("n0".rjust(20)); continue
        sel=nonoverlap_chrono(d["ts"],m)
        if len(sel)==0: cells.append("n0".rjust(20)); continue
        corr=((dirn[sel]>0)==(d["y"][sel]==1)).astype(float); acc=corr.mean(); lo,hi=boot(corr)
        cells.append(f"n{len(sel):>5} {acc:.3f}[{lo:.2f},{hi:.2f}]".rjust(20))
        if w in ("test24","test25","oos"): held.append(acc)
    flag=" <== STABLE>=.58" if (len(held)==3 and all(a>=0.58 for a in held)) else ""
    print(f"{name:34s} "+" ".join(cells)+flag)

def main():
    t0=time.time(); data={}
    for w,yrs in WINDOWS.items():
        data[w]=load_window(yrs); print(f"[cx] {w}: n={len(data[w]['y']):,} valid={int(data[w]['valid'].sum()):,} ({time.time()-t0:.0f}s)",flush=True)
    # SHAPE: does momentum/reversion accuracy depend on PE / ac20 / hurst?
    for meas in ("pe60","pe120","ac20","hurst"):
        for dirn in ("momentum","reversion"):
            bin_accuracy(data, meas, dirn, 5, label=f"{meas} x {dirn}")
    # honest gates the shape suggests + the Sofien ac20>=0.70 contrarian
    print(f"\n[cx] honest INDEPENDENT gates (non-overlap {GAP_S}s, CI95):")
    print(f"{'gate':34s} "+" ".join(f"{w:>20s}" for w in WINDOWS))
    gate_eval(data,"ac20>=0.70 reversion", lambda d:d["ac20"]>=0.70, lambda d:-d["mom"])
    gate_eval(data,"ac20<=-0.30 momentum", lambda d:d["ac20"]<=-0.30, lambda d:d["mom"])
    gate_eval(data,"loPE60 momentum", lambda d:d["pe60"]<=np.nanquantile(data["train"]["pe60"][data["train"]["valid"]],0.2), lambda d:d["mom"])
    gate_eval(data,"loPE60 reversion", lambda d:d["pe60"]<=np.nanquantile(data["train"]["pe60"][data["train"]["valid"]],0.2), lambda d:-d["mom"])
    gate_eval(data,"hiPE60 reversion", lambda d:d["pe60"]>=np.nanquantile(data["train"]["pe60"][data["train"]["valid"]],0.8), lambda d:-d["mom"])
    gate_eval(data,"hurst>0.55 momentum", lambda d:d["hurst"]>0.55, lambda d:d["mom"])
    gate_eval(data,"hurst<0.45 reversion", lambda d:d["hurst"]<0.45, lambda d:-d["mom"])
    gate_eval(data,"loPE60 & hurst>.55 mom", lambda d:(d["pe60"]<=np.nanquantile(data["train"]["pe60"][data["train"]["valid"]],0.3))&(d["hurst"]>0.55), lambda d:d["mom"])
    gate_eval(data,"loPE60 & hurst<.45 rev", lambda d:(d["pe60"]<=np.nanquantile(data["train"]["pe60"][data["train"]["valid"]],0.3))&(d["hurst"]<0.45), lambda d:-d["mom"])
    print(f"\n[cx] DONE {time.time()-t0:.0f}s")

if __name__=="__main__":
    main()
