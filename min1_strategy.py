"""1-MINUTE EURUSD binary — FINAL strategy (compression-release selective book) + honest economics.

THE strategy (all components pre-committed; nothing tuned on the evaluation periods):
  1. Direction model: global LGBM+XGB+CatBoost ensemble on 53 microstructure+multi-TF-regime features
     (models/probs_min1_v3.npz). 60s direction is near-efficient (~0.51 AUC) — the model is used only to
     RANK confidence within a regime, not to predict every bar.
  2. Regime gate (the lever): bet ONLY in the volatility COMPRESSION-RELEASE regime — bbw1800 in its bottom
     tercile (quiet over the last 30 min) AND rel_ratio = bbw300/bbw1800 high (short-term vol expanding = a
     directional breakout is underway). Selection percentiles fixed on VAL.
  3. Selective: among release bars, bet the most-confident |p-0.5| fraction (threshold frozen on VAL),
     with a non-overlapping 60s constraint so every bet is independent/tradeable.
Reports TEST 2024-25, OOS 2026 (+ halves) with bootstrap CIs, and the binary-payout economics.
"""
import numpy as np, pandas as pd, time
TICK="/media/sean/CORSAIR/binary-algo/features_tick"; HS=60
def feats(b):
    mid=b["mid"]; r1=mid.pct_change(); X=pd.DataFrame(index=b.index)
    X["bbw1800"]=r1.rolling(1800).std()*np.sqrt(1800); X["bbw300"]=r1.rolling(300).std()*np.sqrt(300)
    X["rel_ratio"]=X["bbw300"]/(X["bbw1800"]+1e-12); return X
def nonov(ts,conf,corr,thr,gap=HS):
    sel=np.where(conf>=thr)[0]
    if len(sel)==0: return np.nan,0,None
    order=sel[np.argsort(-conf[sel])]
    if len(order)>60000: order=order[:60000]
    tmin=int(ts[order].min()); span=int(ts[order].max()-tmin)+gap+2; blk=np.zeros(span,bool); tk=[]
    for i in order:
        t=int(ts[i]-tmin)
        if blk[t]: continue
        tk.append(i); blk[max(0,t-gap+1):t+gap]=True
    tk=np.array(tk); return corr[tk].mean(), len(tk), corr[tk]

t0=time.time()
raw={sp:pd.read_parquet(f"{TICK}/{sp}_1s.parquet") for sp in ["val","test","oos"]}
F={sp:feats(raw[sp]) for sp in raw}; MID={sp:raw[sp]["mid"] for sp in raw}
TS={sp:raw[sp].index.values.astype("datetime64[s]").astype("int64") for sp in raw}
def move(sp): return (MID[sp].shift(-HS)/MID[sp]-1).values
IDX={sp:np.where(np.isfinite(move(sp))&(move(sp)!=0))[0] for sp in raw}
d=np.load("models/probs_min1_v3.npz"); pv,pt,po=d["pva"],d["pte"],d["poo"]; yv,yt,yo=d["yva"],d["yte"],d["yoo"]
def Cc(sp,c): return F[sp][c].values[IDX[sp]]
bbV,bbT,bbO=Cc("val","bbw1800"),Cc("test","bbw1800"),Cc("oos","bbw1800")
rrV,rrT,rrO=Cc("val","rel_ratio"),Cc("test","rel_ratio"),Cc("oos","rel_ratio")
ct=(pt>0.5).astype(int)==yt; co=(po>0.5).astype(int)==yo
tst=TS["test"][IDX["test"]]; tso=TS["oos"][IDX["oos"]]; cft=np.abs(pt-0.5); cfo=np.abs(po-0.5); cfv=np.abs(pv-0.5)
# pre-committed setup (fixed on VAL): compression q33, release p90, direction coverage 10%
qb=np.nanpercentile(bbV,33); rq=np.nanpercentile(rrV[bbV<=qb],90); COV=0.10
gv=(bbV<=qb)&(rrV>=rq); gt=(bbT<=qb)&(rrT>=rq); go=(bbO<=qb)&(rrO>=rq)
thr=np.quantile(cfv[gv],1-COV)
print(f"loaded; {time.time()-t0:.0f}s. Strategy: compression(bbw1800<=q33) & release(rel_ratio>=p90) & top-{COV:.0%} confidence")
rng=np.random.default_rng(7)
def rep(tag,ts,cf,corr,mask):
    a,n,s=nonov(ts[mask],cf[mask],corr[mask],thr)
    ci=""
    if s is not None and n>=20:
        b=np.array([rng.choice(s,n,replace=True).mean() for _ in range(5000)]); ci=f"[{np.percentile(b,2.5):.3f},{np.percentile(b,97.5):.3f}]"
    print(f"  {tag:>16}: acc={a:.3f}  n={n:>4}  CI95={ci}"); return a,n
print("\n==== compression-release selective 1-min binary, frozen pipeline ====")
at,ntt=rep("TEST 2024-25",tst,cft,ct,gt)
oo_h=len(IDX["oos"])//2; goh1=go.copy(); goh1[oo_h:]=False; goh2=go.copy(); goh2[:oo_h]=False
ao,noo=rep("OOS 2026",tso,cfo,co,go)
rep("OOS 2026 H1",tso,cfo,co,goh1); rep("OOS 2026 H2",tso,cfo,co,goh2)
# economics: binary payout breakeven
for payout in (0.75,0.80,0.90):
    be=1.0/(1.0+payout)
    print(f"  payout {payout:.2f}: breakeven={be:.3f}  -> TEST EV/bet={(at*payout-(1-at)):+.3f}  OOS EV/bet={(ao*payout-(1-ao)):+.3f}")
print("\nMIN1-STRATEGY DONE")
