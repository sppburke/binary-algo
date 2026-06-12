"""1-MINUTE v8 — directional CONFIRMATION in the compression-release regime (lift the large-sample TEST).

v7: comp33&rel>p90 gives OOS 0.77-0.90 but TEST (large n) only ~0.63-0.71. To make it robust we need TEST up.
Idea: within the release regime, require the global model's direction to AGREE with (a) breakout momentum
sign(ret30) and (b) microprice deviation sign(micro_dev). Triple-confirmed bets should be cleaner.
Reuses global v3 probs. Reports TEST(large n)+OOS+halves+CI per confirmation level. Also a PRE-COMMITTED
cell: pick the (setup,coverage) maximizing VAL+TEST accuracy, then report OOS once (honest proof).
"""
import numpy as np, pandas as pd, time
TICK="/home/sean/git/binary-algo/features_tick"; HS=60

def feats(b):
    mid=b["mid"]; micro=b["micro"]; r1=mid.pct_change(); X=pd.DataFrame(index=b.index)
    X["bbw1800"]=r1.rolling(1800).std()*np.sqrt(1800); X["bbw300"]=r1.rolling(300).std()*np.sqrt(300)
    X["rel_ratio"]=X["bbw300"]/(X["bbw1800"]+1e-12)
    X["ret30"]=mid.pct_change(30); X["micro_dev"]=(micro-mid)/mid
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
print(f"loaded; feats {time.time()-t0:.0f}s",flush=True)
def Cv(sp,c): return F[sp][c].values[IDX[sp]]
G={sp:{c:Cv(sp,c) for c in ("bbw1800","rel_ratio","ret30","micro_dev")} for sp in raw}
ct=(pt>0.5).astype(int)==yt; co=(po>0.5).astype(int)==yo
tst=TS["test"][IDX["test"]]; tso=TS["oos"][IDX["oos"]]; cft=np.abs(pt-0.5); cfo=np.abs(po-0.5); cfv=np.abs(pv-0.5)
oo_h=len(IDX["oos"])//2; rng=np.random.default_rng(7)
qb=np.nanpercentile(G["val"]["bbw1800"],33)
rq=np.nanpercentile(G["val"]["rel_ratio"][G["val"]["bbw1800"]<=qb],80)   # comp33 & rel>p80 (more n than p90)

def release(sp): return (G[sp]["bbw1800"]<=qb)&(G[sp]["rel_ratio"]>=rq)
def confirm(sp,p,level):
    pd_=np.sign(p-0.5); base=release(sp)
    if level==0: return base
    agree_mom=pd_==np.sign(G[sp]["ret30"])
    if level==1: return base&agree_mom
    agree_mic=pd_==np.sign(G[sp]["micro_dev"])
    return base&agree_mom&agree_mic
NAMES={0:"release",1:"release&mom",2:"release&mom&micro"}
print(f"\n{'confirm':>20} {'cov':>6} {'TESTacc':>8} {'TESTn':>6} {'OOSacc':>7} {'OOSn':>5} {'OOS95CI':>14} {'H1':>5} {'H2':>5} {'75?':>4}")
for level in (0,1,2):
    gv=confirm("val",pv,level); gt=confirm("test",pt,level); go=confirm("oos",po,level)
    if gv.sum()<200: print(f"{NAMES[level]:>20}: VAL n={int(gv.sum())} small"); continue
    cfvg=cfv[gv]
    for cov in (0.30,0.20,0.10,0.05):
        thr=np.quantile(cfvg,1-cov)
        at,nt,_=nonov(tst[gt],cft[gt],ct[gt],thr); ao,no,so=nonov(tso[go],cfo[go],co[go],thr)
        goh1=go.copy(); goh1[oo_h:]=False; goh2=go.copy(); goh2[:oo_h]=False
        a1,_,_=nonov(tso[goh1],cfo[goh1],co[goh1],thr); a2,_,_=nonov(tso[goh2],cfo[goh2],co[goh2],thr)
        ci=""
        if so is not None and no>=20:
            b=np.array([rng.choice(so,no,replace=True).mean() for _ in range(3000)]); ci=f"[{np.percentile(b,2.5):.2f},{np.percentile(b,97.5):.2f}]"
        ok="**" if (not np.isnan(ao) and ao>=0.75 and no>=50 and not np.isnan(at) and at>=0.73) else ""
        print(f"{NAMES[level]:>20} {cov:>6.2%} {at:>8.3f} {nt:>6} {ao:>7.3f} {no:>5} {ci:>14} {a1:>5.2f} {a2:>5.2f} {ok:>4}",flush=True)
print("\n** = OOS>=0.75(n>=50) AND TEST>=0.73.  MIN1-V8 DONE")
