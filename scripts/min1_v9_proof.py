"""1-MINUTE v9 — PRE-COMMITTED proof of the compression-release strategy.

Mechanism fixed a priori (from research + v3-v7): bet the global ensemble's direction ONLY in the
volatility compression-RELEASE regime (compression bbw1800 low AND rel_ratio bbw300/bbw1800 high).
Selection (release percentile + coverage + confidence threshold) done on VAL ONLY. TEST (2024-25) and
OOS 2026 are both held out from selection. Honest non-overlapping (60s-gap) eval; report TEST + OOS +
2026 halves + bootstrap CI. Reuses global v3 ensemble probs (no peeking-driven retrain).
"""
import numpy as np, pandas as pd, time
TICK="/home/sean/git/binary-algo/features_tick"; HS=60

def feats(b):
    mid=b["mid"]; r1=mid.pct_change(); X=pd.DataFrame(index=b.index)
    X["bbw1800"]=r1.rolling(1800).std()*np.sqrt(1800); X["bbw300"]=r1.rolling(300).std()*np.sqrt(300)
    X["rel_ratio"]=X["bbw300"]/(X["bbw1800"]+1e-12); return X

def nonov(ts,conf,corr,thr,gap=HS):
    sel=np.where(conf>=thr)[0]
    if len(sel)==0: return np.nan,0,None
    order=sel[np.argsort(-conf[sel])]; used=np.empty(0,"int64"); take=[]
    for i in order:
        if used.size and np.any(np.abs(used-ts[i])<gap): continue
        take.append(i); used=np.append(used,ts[i])
    take=np.array(take); return corr[take].mean(), len(take), corr[take]

t0=time.time()
raw={sp:pd.read_parquet(f"{TICK}/{sp}_1s.parquet") for sp in ["val","test","oos"]}
F={sp:feats(raw[sp]) for sp in raw}; MID={sp:raw[sp]["mid"] for sp in raw}
TS={sp:raw[sp].index.values.astype("datetime64[s]").astype("int64") for sp in raw}
def move(sp): return (MID[sp].shift(-HS)/MID[sp]-1).values
IDX={sp:np.where(np.isfinite(move(sp))&(move(sp)!=0))[0] for sp in raw}
d=np.load("models/probs_min1_v3.npz"); pv,pt,po=d["pva"],d["pte"],d["poo"]; yv,yt,yo=d["yva"],d["yte"],d["yoo"]
def Cv(sp,c): return F[sp][c].values[IDX[sp]]
bbV,bbT,bbO=Cv("val","bbw1800"),Cv("test","bbw1800"),Cv("oos","bbw1800")
rrV,rrT,rrO=Cv("val","rel_ratio"),Cv("test","rel_ratio"),Cv("oos","rel_ratio")
cv_=(pv>0.5).astype(int)==yv; ct=(pt>0.5).astype(int)==yt; co=(po>0.5).astype(int)==yo
tsv=TS["val"][IDX["val"]]; tst=TS["test"][IDX["test"]]; tso=TS["oos"][IDX["oos"]]
cfv=np.abs(pv-0.5); cft=np.abs(pt-0.5); cfo=np.abs(po-0.5)
qb=np.nanpercentile(bbV,33)
print(f"loaded; feats {time.time()-t0:.0f}s | compression bbw1800<=q33={qb:.2e}",flush=True)

# ---- SELECT (release pct, coverage) on VAL only: maximize VAL non-overlapping accuracy with VAL n>=120 ----
best=None
for relpc in (70,80,85,90,95):
    rq=np.nanpercentile(rrV[bbV<=qb],relpc)
    gv=(bbV<=qb)&(rrV>=rq)
    if gv.sum()<400: continue
    cfvg=cfv[gv]
    for cov in (0.30,0.20,0.10):
        thr=np.quantile(cfvg,1-cov)
        av,nv,_=nonov(tsv[gv],cfv[gv],cv_[gv],thr)
        if nv<120: continue
        if best is None or av>best[0]:
            best=(av,relpc,cov,rq,thr,nv)
av,RELPC,COV,RQ,THR,nV=best
print(f"\nSELECTED on VAL: compression-release (comp q33 & rel>p{RELPC}) @cov{COV:.0%}  VALacc={av:.3f} (nVAL={nV})",flush=True)

def report(tag,ts,cf,corr,mask,thr):
    a,n,s=nonov(ts[mask],cf[mask],corr[mask],thr)
    ci=""
    if s is not None and n>=20:
        rng=np.random.default_rng(7); b=np.array([rng.choice(s,n,replace=True).mean() for _ in range(5000)])
        ci=f"[{np.percentile(b,2.5):.3f},{np.percentile(b,97.5):.3f}]"
    print(f"   {tag:>16}: acc={a:.3f}  n={n:>4}  CI95={ci}",flush=True)
    return s
gt=(bbT<=qb)&(rrT>=RQ); go=(bbO<=qb)&(rrO>=RQ)
print("\n==== FROZEN compression-release pipeline on HELD-OUT data ====")
report("TEST 2024-25",tst,cft,ct,gt,THR)
oo_h=len(IDX["oos"])//2
goh1=go.copy(); goh1[oo_h:]=False; goh2=go.copy(); goh2[:oo_h]=False
so=report("OOS 2026",tso,cfo,co,go,THR)
report("OOS 2026 H1",tso,cfo,co,goh1,THR)
report("OOS 2026 H2",tso,cfo,co,goh2,THR)
print("\nMIN1-V9-PROOF DONE")
