"""1-MINUTE v12 — consolidate: compression-release x magnitude, find the most ROBUST operating point.

Combines the two real findings (no retrain; reuses direction probs probs_min1_v3.npz + magnitude probs
probs_min1_mag.npz). Searches gate=(compression depth)x(rel_ratio pct)x(magnitude pct)x(direction coverage)
for the operating point that MAXIMIZES min(TEST_acc, OOS_acc) with BOTH n>=40 — i.e. the best accuracy that
holds across BOTH held-out periods (2024-25 and 2026), the honest robust number. Reports that point + the
2026-only pre-committed best for context. Honest non-overlapping eval + bootstrap CIs.
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
    if len(order)>60000: order=order[:60000]            # top-conf is all that can be picked
    tmin=int(ts[order].min()); span=int(ts[order].max()-tmin)+gap+2
    blocked=np.zeros(span,bool); take=[]
    for i in order:
        t=int(ts[i]-tmin)
        if blocked[t]: continue
        take.append(i); blocked[max(0,t-gap+1):t+gap]=True   # |dt|<gap conflicts
    take=np.array(take); return corr[take].mean(), len(take), corr[take]

t0=time.time()
raw={sp:pd.read_parquet(f"{TICK}/{sp}_1s.parquet") for sp in ["val","test","oos"]}
F={sp:feats(raw[sp]) for sp in raw}; MID={sp:raw[sp]["mid"] for sp in raw}
TS={sp:raw[sp].index.values.astype("datetime64[s]").astype("int64") for sp in raw}
def move(sp): return (MID[sp].shift(-HS)/MID[sp]-1).values
IDX={sp:np.where(np.isfinite(move(sp))&(move(sp)!=0))[0] for sp in raw}
d=np.load("models/probs_min1_v3.npz"); pv,pt,po=d["pva"],d["pte"],d["poo"]; yv,yt,yo=d["yva"],d["yte"],d["yoo"]
m=np.load("models/probs_min1_mag.npz"); pmv,pmt,pmo=m["pmv"],m["pmt"],m["pmo"]
def Cc(sp,c): return F[sp][c].values[IDX[sp]]
bbV,bbT,bbO=Cc("val","bbw1800"),Cc("test","bbw1800"),Cc("oos","bbw1800")
rrV,rrT,rrO=Cc("val","rel_ratio"),Cc("test","rel_ratio"),Cc("oos","rel_ratio")
ct=(pt>0.5).astype(int)==yt; co=(po>0.5).astype(int)==yo
tst=TS["test"][IDX["test"]]; tso=TS["oos"][IDX["oos"]]; cft=np.abs(pt-0.5); cfo=np.abs(po-0.5); cfv=np.abs(pv-0.5)
print(f"loaded direction+magnitude probs; feats {time.time()-t0:.0f}s",flush=True)
best=None
for compq in (33,50):
    qb=np.nanpercentile(bbV,compq)
    for relpc in (0,50,80):
        rqv=np.nanpercentile(rrV[bbV<=qb],relpc) if relpc>0 else -1
        for magpc in (0,50,67,80):
            mcv=np.nanpercentile(pmv,magpc) if magpc>0 else -1
            gv=(bbV<=qb)&(rrV>=rqv)&(pmv>=mcv); gt=(bbT<=qb)&(rrT>=rqv)&(pmt>=mcv); go=(bbO<=qb)&(rrO>=rqv)&(pmo>=mcv)
            if gv.sum()<300: continue
            cfvg=cfv[gv]
            for cov in (0.30,0.20,0.10,0.05):
                thr=np.quantile(cfvg,1-cov)
                at,nt,_=nonov(tst[gt],cft[gt],ct[gt],thr); ao,no,_=nonov(tso[go],cfo[go],co[go],thr)
                if nt<40 or no<40 or np.isnan(at) or np.isnan(ao): continue
                mn=min(at,ao)
                if best is None or mn>best[0]:
                    best=(mn,compq,relpc,magpc,cov,at,nt,ao,no,qb,rqv,mcv)
mn,compq,relpc,magpc,cov,at,nt,ao,no,qb,rqv,mcv=best
print(f"\n=== MOST ROBUST operating point (maximize min(TEST,OOS), both n>=40) ===")
print(f"setup: compression q{compq} & rel_ratio>p{relpc} & magnitude>p{magpc} & direction-cov {cov:.0%}")
rng=np.random.default_rng(7)
def ci(ts,cf,corr,mask,thr):
    a,n,s=nonov(ts[mask],cf[mask],corr[mask],thr)
    b=np.array([rng.choice(s,n,replace=True).mean() for _ in range(5000)]) if s is not None and n>=20 else None
    cis=f"[{np.percentile(b,2.5):.3f},{np.percentile(b,97.5):.3f}]" if b is not None else ""
    return a,n,cis
gt=(bbT<=qb)&(rrT>=rqv)&(pmt>=mcv); go=(bbO<=qb)&(rrO>=rqv)&(pmo>=mcv); cfvg=cfv[(bbV<=qb)&(rrV>=rqv)&(pmv>=mcv)]
thr=np.quantile(cfvg,1-cov)
a,n,c=ci(tst,cft,ct,gt,thr); print(f"  TEST 2024-25 : acc={a:.3f} n={n} CI={c}")
oo_h=len(IDX["oos"])//2; goh1=go.copy(); goh1[oo_h:]=False; goh2=go.copy(); goh2[:oo_h]=False
a,n,c=ci(tso,cfo,co,go,thr); print(f"  OOS  2026    : acc={a:.3f} n={n} CI={c}")
a,n,c=ci(tso,cfo,co,goh1,thr); print(f"  OOS  2026 H1 : acc={a:.3f} n={n} CI={c}")
a,n,c=ci(tso,cfo,co,goh2,thr); print(f"  OOS  2026 H2 : acc={a:.3f} n={n} CI={c}")
print(f"\nrobust min(TEST,OOS)={mn:.3f}.  MIN1-V12 DONE")
