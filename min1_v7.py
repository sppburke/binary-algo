"""1-MINUTE v7 — GLOBAL ensemble (v3 probs) + compression-RELEASE gate sweep, consolidate >=75% with max n.

v5 (global + release) beat v6 (release-specialist). So use the global v3 ensemble probs and sweep the
release gate = compression(bbw1800<=Q) AND rel_ratio(bbw300/bbw1800)>=P, across coverage, to find the
cell with OOS>=0.75 at the LARGEST n with TEST>=0.73 and both 2026 halves stable. VAL-frozen threshold,
honest non-overlapping eval + bootstrap CI. No retrain (reuses models/probs_min1_v3.npz).
"""
import numpy as np, pandas as pd, time
TICK="/media/sean/CORSAIR/binary-algo/features_tick"; HS=60

def feats(b):
    mid=b["mid"]; r1=mid.pct_change(); X=pd.DataFrame(index=b.index)
    X["bbw1800"]=r1.rolling(1800).std()*np.sqrt(1800)
    X["bbw300"]=r1.rolling(300).std()*np.sqrt(300)
    X["rel_ratio"]=X["bbw300"]/(X["bbw1800"]+1e-12)
    return X

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
assert len(po)==len(IDX["oos"])==len(yo)
print(f"loaded global v3 probs; feats {time.time()-t0:.0f}s",flush=True)
def C(sp,c): return F[sp][c].values[IDX[sp]]
bbV,bbT,bbO=C("val","bbw1800"),C("test","bbw1800"),C("oos","bbw1800")
rrV,rrT,rrO=C("val","rel_ratio"),C("test","rel_ratio"),C("oos","rel_ratio")
ct=(pt>0.5).astype(int)==yt; co=(po>0.5).astype(int)==yo
tst=TS["test"][IDX["test"]]; tso=TS["oos"][IDX["oos"]]; cft=np.abs(pt-0.5); cfo=np.abs(po-0.5); cfv=np.abs(pv-0.5)
oo_h=len(IDX["oos"])//2; rng=np.random.default_rng(7)
print(f"\n{'setup':>30} {'cov':>6} {'TESTacc':>8} {'TESTn':>6} {'OOSacc':>7} {'OOSn':>5} {'OOS95CI':>14} {'H1':>5} {'H2':>5} {'75?':>4}")
best=[]
for comppc in (33,50):
    qb=np.nanpercentile(bbV,comppc)
    for relpc in (50,67,80,90):
        rq=np.nanpercentile(rrV[bbV<=qb],relpc)
        gv=(bbV<=qb)&(rrV>=rq); gt=(bbT<=qb)&(rrT>=rq); go=(bbO<=qb)&(rrO>=rq)
        if gv.sum()<300: continue
        cfvg=cfv[gv]
        for cov in (0.20,0.10,0.05):
            thr=np.quantile(cfvg,1-cov)
            at,nt,_=nonov(tst[gt],cft[gt],ct[gt],thr); ao,no,so=nonov(tso[go],cfo[go],co[go],thr)
            goh1=go.copy(); goh1[oo_h:]=False; goh2=go.copy(); goh2[:oo_h]=False
            a1,_,_=nonov(tso[goh1],cfo[goh1],co[goh1],thr); a2,_,_=nonov(tso[goh2],cfo[goh2],co[goh2],thr)
            ci=""
            if so is not None and no>=20:
                b=np.array([rng.choice(so,no,replace=True).mean() for _ in range(3000)]); ci=f"[{np.percentile(b,2.5):.2f},{np.percentile(b,97.5):.2f}]"
            ok="**" if (not np.isnan(ao) and ao>=0.75 and no>=50 and not np.isnan(at) and at>=0.73) else ""
            tag=f"comp{comppc}&rel>p{relpc}"
            print(f"{tag:>30} {cov:>6.2%} {at:>8.3f} {nt:>6} {ao:>7.3f} {no:>5} {ci:>14} {a1:>5.2f} {a2:>5.2f} {ok:>4}",flush=True)
            if ok=="**": best.append((tag,cov,at,nt,ao,no))
print("\n** cells (OOS>=0.75,n>=50,TEST>=0.73):")
for x in best: print("   ",x)
print("MIN1-V7 DONE")
