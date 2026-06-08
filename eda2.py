"""EDA2: stacked extreme-stretch reversion. Does fading multi-TF stretch concentrate edge?
Train+Val years only (2012-2023). Reports fade accuracy & coverage vs stretch threshold."""
import glob, os
import numpy as np, pandas as pd

FEAT_DIR="/home/sean/git/binary-algo/features"
YEARS=[str(y) for y in range(2012,2024)]
bb=[f"{tf}_bb_pctb" for tf in ["1m","5m","15m","30m","1h"]]
rsi=[f"{tf}_rsi" for tf in ["1m","5m","15m","30m","1h"]]
de=[f"{tf}_dist_ema20" for tf in ["1m","5m","15m","1h"]]
cols=["y","valid","fwd_ret"]+bb+rsi+de
parts=[]
for y in YEARS:
    p=f"{FEAT_DIR}/EURUSD_{y}.parquet"
    if os.path.exists(p): parts.append(pd.read_parquet(p,columns=cols))
df=pd.concat(parts); df=df[df.valid==True].copy(); df["up"]=df.y.astype(int)
N=len(df)
# composite signed stretch: + = stretched up (fade -> predict down)
df["stretch"]=df[bb].mean(axis=1)  # bb %b centered at 0, +/-1 ~ band edge
df["rsi_str"]=df[rsi].mean(axis=1) # rsi centered (already (rsi-50)/50)
df["comp"]=(df["stretch"]+df["rsi_str"])/2
print(f"N={N:,}")

def fade_eval(score_col, label):
    print(f"\n== Fade on |{score_col}| (predict opposite of stretch sign) ==")
    print(f"  {'thr':>6} {'coverage':>9} {'n':>9} {'fade_acc':>9}")
    s=df[score_col]
    for thr in [0,0.3,0.5,0.7,0.9,1.1,1.3,1.5]:
        sel=s.abs()>=thr
        m=df[sel]
        if len(m)<500: continue
        # fade: if stretched up (s>0) predict down(0); else predict up(1)
        pred=(m[score_col]<0).astype(int)
        acc=(pred==m.up).mean()
        print(f"  {thr:6.2f} {sel.mean():9.4%} {len(m):9,} {acc:9.4f}")

fade_eval("stretch","bb")
fade_eval("comp","bb+rsi")

print("\n== Require ALL TFs stretched same direction (unanimous) + fade ==")
print(f"  {'bb_thr':>6} {'coverage':>9} {'n':>9} {'fade_acc':>9}")
for thr in [0.5,0.8,1.0,1.2]:
    up_all=(df[bb]>thr).all(axis=1)
    dn_all=(df[bb]<-thr).all(axis=1)
    sel=up_all|dn_all
    m=df[sel]
    if len(m)<300: continue
    pred=np.where(up_all[sel],0,1)  # fade
    acc=(pred==m.up.values).mean()
    print(f"  {thr:6.2f} {sel.mean():9.4%} {len(m):9,} {acc:9.4f}")

print("\n== Extreme tail: top-stretch quantiles fade_acc ==")
for q in [0.9,0.95,0.99,0.995,0.999]:
    thr=df["comp"].abs().quantile(q)
    sel=df["comp"].abs()>=thr
    m=df[sel]
    pred=(m["comp"]<0).astype(int)
    acc=(pred==m.up).mean()
    print(f"  q{q}: thr={thr:.3f} n={len(m):,} fade_acc={acc:.4f}")
