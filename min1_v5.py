"""1-MINUTE v5 — refine the compression regime: COMPRESSION-RELEASE / breakout-trigger setups.

Reuses the v3 GLOBAL ensemble probs (models/probs_min1_v3.npz) — no retrain. v4 showed plain
in-compression caps TEST ~0.71. Hypothesis (research compression→expansion): the predictable 60s
move is the DIRECTIONAL RELEASE of a quiet period, not just being quiet. Test setups that add a
breakout/alignment trigger on top of compression, with clean WITHIN-SETUP coverage, VAL-frozen
threshold, TEST (large n) + OOS + OOS H1/H2 + bootstrap CI. Honest non-overlapping (60s-gap) eval.
"""
import numpy as np, pandas as pd, time
TICK="/media/sean/CORSAIR/binary-algo/features_tick"; HS=60

def feats(b):
    mid=b["mid"]; imb=b["imb"].fillna(0); micro=b["micro"]; r1=mid.pct_change()
    X=pd.DataFrame(index=b.index)
    X["imb_ema20"]=imb.ewm(span=20).mean()
    X["micro_dev"]=(micro-mid)/mid
    for w in (15,30,60): X[f"ret{w}"]=mid.pct_change(w)
    X["rv900"]=r1.rolling(900).std()
    X["bbw1800"]=r1.rolling(1800).std()*np.sqrt(1800)
    X["bbw300"]=r1.rolling(300).std()*np.sqrt(300)
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
def idxs(sp):
    mm=move(sp); m=np.isfinite(mm)&(mm!=0); return np.where(m)[0]
IDX={sp:idxs(sp) for sp in raw}
d=np.load("models/probs_min1_v3.npz"); pv,pt,po=d["pva"],d["pte"],d["poo"]; yv,yt,yo=d["yva"],d["yte"],d["yoo"]
assert len(po)==len(IDX["oos"])==len(yo), (len(po),len(IDX["oos"]),len(yo))
print(f"loaded v3 probs, aligned. feats {time.time()-t0:.0f}s",flush=True)

def col(sp,c): return F[sp][c].values[IDX[sp]]
# train-percentiles approximated from VAL (causal: VAL precedes test/oos)
bbwV=col("val","bbw1800"); q33=np.nanpercentile(bbwV,33); q20=np.nanpercentile(bbwV,20)
r30V=np.abs(col("val","ret30"))
def setup_masks(sp):
    bbw=col(sp,"bbw1800"); bbw3=col(sp,"bbw300"); r30=col(sp,"ret30"); r60=col(sp,"ret60")
    md=col(sp,"micro_dev"); imbe=col(sp,"imb_ema20")
    comp=bbw<=q33
    r30med=np.nanmedian(np.abs(r30V))
    return {
     "compress(q33)":        comp,
     "compress+breakout":    comp & (np.abs(r30)>=r30med),
     "compress+microalign":  comp & (np.sign(md)==np.sign(r30)) & (np.abs(r30)>0),
     "compress+imbalign":    comp & (np.sign(imbe)==np.sign(r30)) & (np.abs(r30)>0),
     "compress+release(bbw3up)": comp & (bbw3>np.nanmedian(col("val","bbw300"))),
     "compress(q20)+breakout": (bbw<=q20) & (np.abs(r30)>=r30med),
    }
Mv=setup_masks("val"); Mt=setup_masks("test"); Mo=setup_masks("oos")
ct=(pt>0.5).astype(int)==yt; co=(po>0.5).astype(int)==yo
tst=TS["test"][IDX["test"]]; tso=TS["oos"][IDX["oos"]]; cft=np.abs(pt-0.5); cfo=np.abs(po-0.5); cfv=np.abs(pv-0.5)
oo_h=len(IDX["oos"])//2; rng=np.random.default_rng(7)
print(f"\n{'setup':>26} {'cov':>6} {'TESTacc':>8} {'TESTn':>6} {'OOSacc':>7} {'OOSn':>5} {'OOS95CI':>14} {'H1':>5} {'H2':>5} {'75?':>4}")
for name in Mv:
    gv,gt,go=Mv[name],Mt[name],Mo[name]
    if gv.sum()<400:
        print(f"{name:>26} VAL n={int(gv.sum())} too small"); continue
    cfvg=cfv[gv]
    for cov in (0.05,0.02,0.01,0.005):
        thr=np.quantile(cfvg,1-cov)
        at,nt,_=nonov(tst[gt],cft[gt],ct[gt],thr); ao,no,so=nonov(tso[go],cfo[go],co[go],thr)
        goh1=go.copy(); goh1[oo_h:]=False; goh2=go.copy(); goh2[:oo_h]=False
        a1,_,_=nonov(tso[goh1],cfo[goh1],co[goh1],thr); a2,_,_=nonov(tso[goh2],cfo[goh2],co[goh2],thr)
        ci=""
        if so is not None and no>=20:
            b=np.array([rng.choice(so,no,replace=True).mean() for _ in range(3000)]); ci=f"[{np.percentile(b,2.5):.2f},{np.percentile(b,97.5):.2f}]"
        ok="**" if (not np.isnan(ao) and ao>=0.75 and no>=40 and not np.isnan(at) and at>=0.73) else ""
        print(f"{name:>26} {cov:>6.2%} {at:>8.3f} {nt:>6} {ao:>7.3f} {no:>5} {ci:>14} {a1:>5.2f} {a2:>5.2f} {ok:>4}",flush=True)
print("\n** = OOS>=0.75(n>=40) AND TEST>=0.73.  MIN1-V5 DONE")
