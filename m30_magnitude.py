"""30m EURUSD — VALIDATE THE THEOREM on our data: orthogonal complexity predicts MAGNITUDE, not direction.

arXiv:2512.15720: order-flow/permutation entropy is sign-invariant -> gates |move| (volatility), not sign. Here we
confirm on EURUSD: do permutation entropy (d=4) and recent realized vol predict the 30m FORWARD ABSOLUTE return,
out-of-sample? If yes, the data IS predictable (magnitude) even though SIGN is ~EMH (~0.515) — the honest "different
story". (For an up/down binary this raises EV via tie/cost avoidance, not directional accuracy.)

  python m30_magnitude.py
"""
import os, math, time, numpy as np, pandas as pd
import harness as H
from sklearn.metrics import roc_auc_score
HOR=30
WINDOWS={"train":[str(y) for y in range(2016,2022)],"val":["2022","2023"],"test24":["2024"],"test25":["2025"],"oos":["2026"]}

def perm_entropy(r, d=4, tau=1, W=120):
    N=len(r); L=(d-1)*tau
    if N<=L+1: return np.full(N,np.nan)
    idx=np.arange(N-L)[:,None]+np.arange(0,d*tau,tau)[None,:]
    order=np.argsort(r[idx],axis=1,kind="stable"); code=(order*(d**np.arange(d))).sum(1).astype(np.int32)
    M=len(code); nb=d**d
    oh=np.zeros((M,nb),dtype=np.float32); oh[np.arange(M),code]=1.0
    cs=np.cumsum(oh,axis=0); cnt=cs.copy(); cnt[W:]=cs[W:]-cs[:-W]
    p=cnt/np.maximum(cnt.sum(1,keepdims=True),1)
    with np.errstate(divide='ignore',invalid='ignore'):
        ent=-np.nansum(np.where(p>0,p*np.log(p),0.0),axis=1)/math.log(math.factorial(d))
    out=np.full(N,np.nan); out[L:L+M]=ent; out[:L+W]=np.nan; return out

def load(years):
    base=pd.concat([pd.read_parquet(f"{H.FEAT_DIR}/EURUSD_{y}.parquet",columns=["close"]) for y in years])
    base=base[~base.index.duplicated(keep="last")].sort_index()
    c=base["close"].values.astype(float); n=len(c); secs=base.index.values.astype("datetime64[s]").astype("int64")
    contig=np.zeros(n,bool); contig[:n-HOR]=(secs[HOR:]-secs[:-HOR])==HOR*60
    fwd=np.full(n,np.nan); fwd[:n-HOR]=c[HOR:]; ret=fwd/c-1
    valid=contig&np.isfinite(ret)
    r=np.zeros(n); r[1:]=np.diff(np.log(c)); rs=pd.Series(r)
    pe=perm_entropy(r,4,1,120)
    rv30=rs.rolling(30).std().values; rv120=rs.rolling(120).std().values
    aret=np.abs(ret)
    return dict(aret=aret, valid=valid, pe=pe, rv30=rv30, rv120=rv120)

def main():
    t0=time.time(); D={w:load(yrs) for w,yrs in WINDOWS.items()}
    print(f"[mag] loaded ({time.time()-t0:.0f}s). Target = 30m forward |return|. Does complexity/vol predict MAGNITUDE OOS?\n")
    # 1) correlations of predictors with future |return|
    print("Spearman-ish corr(predictor, future|ret|):")
    for pred in ("pe","rv30","rv120"):
        cs=[]
        for w in WINDOWS:
            d=D[w]; m=d["valid"]&np.isfinite(d[pred])&np.isfinite(d["aret"])
            x=pd.Series(d[pred][m]).rank(); yv=pd.Series(d["aret"][m]).rank()
            cs.append(f"{w}={np.corrcoef(x,yv)[0,1]:+.3f}")
        print(f"  {pred:6s}: "+"  ".join(cs))
    # 2) large-move classification: predict whether 30m |ret| is in TOP quartile (threshold from TRAIN), AUC per window
    thr=np.nanquantile(D["train"]["aret"][D["train"]["valid"]],0.75)
    print(f"\nLarge-move (|ret|>=train-Q75={thr:.2e}) classification AUC by predictor:")
    for pred in ("pe","rv30","rv120"):
        cells=[]
        for w in WINDOWS:
            d=D[w]; m=d["valid"]&np.isfinite(d[pred])&np.isfinite(d["aret"])
            y=(d["aret"][m]>=thr).astype(int); x=d[pred][m]
            if pred=="pe": x=-x   # LOW pe -> large move
            cells.append(f"{w}={roc_auc_score(y,x):.3f}")
        print(f"  {pred:6s}: "+"  ".join(cells))
    # 3) low-PE gate concentrates large moves? mean |ret| in low-PE vs high-PE quintile, OOS
    print(f"\nMean 30m |ret| by PE quintile (low PE = predictable-large-move regime?):")
    edges=np.nanquantile(D["train"]["pe"][D["train"]["valid"]],np.linspace(0,1,6)); edges[0]=-np.inf; edges[-1]=np.inf
    print(f"  {'PE bin':>14} "+" ".join(f"{w:>12}" for w in WINDOWS))
    for b in range(5):
        cells=[]
        for w in WINDOWS:
            d=D[w]; m=d["valid"]&(d["pe"]>=edges[b])&(d["pe"]<edges[b+1])&np.isfinite(d["aret"])
            cells.append(f"{d['aret'][m].mean()*1e4:.1f}bp".rjust(12) if m.sum()>100 else "n<100".rjust(12))
        print(f"  {'Q'+str(b+1):>14} "+" ".join(cells))
    print(f"\n[mag] DONE {time.time()-t0:.0f}s  (if low-PE/high-vol predicts larger |moves| OOS -> magnitude IS forecastable; sign is not)")

if __name__=="__main__":
    main()
