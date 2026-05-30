"""2-min phase B: load cached probs, sweep regimes x coverage selecting on VAL, judge TEST/OOS once.

Each candidate = (regime mask def, confidence coverage). We freeze on VAL (max VAL non-overlap accuracy with
n>=NMIN), then report TEST + OOS non-overlapping accuracy + per-month breakdown. Non-overlap gap=120s.
"""
import sys, numpy as np
MODELS="/media/sean/CORSAIR/binary-algo/models"
GAP=120; NMIN_VAL=120; NMIN_REPORT=20
Z=np.load(f"{MODELS}/probs_min2_v1.npz",allow_pickle=True)

def get(sp):
    return dict(p=Z[f"{sp}_p"],mag=Z[f"{sp}_mag"],y=Z[f"{sp}_y"],ts=Z[f"{sp}_ts"],
        bbw1800=Z[f"{sp}_bbw1800"],bbw3600=Z[f"{sp}_bbw3600"],rel=Z[f"{sp}_rel"],rel2=Z[f"{sp}_rel2"],
        sess=Z[f"{sp}_sess"],month=Z[f"{sp}_month"])
VA,TE,OO=get("va"),get("te"),get("oo")

def nonoverlap(ts,conf,thr,gap=GAP):
    sel=np.where(conf>=thr)[0]
    if len(sel)==0: return np.array([],dtype=int)
    order=sel[np.argsort(-conf[sel])]
    if len(order)>300000: order=order[:300000]
    tmin=int(ts[order].min()); span=int(ts[order].max()-tmin)+gap+2
    blk=np.zeros(span,bool); take=[]
    for i in order:
        t=int(ts[i]-tmin)
        if blk[t]: continue
        take.append(i); blk[max(0,t-gap+1):t+gap]=True
    return np.sort(np.array(take))

# ---- regime definitions: each returns a boolean mask + a label. Thresholds are fit on VA only. ----
def regime_builders():
    R={}
    # thresholds computed on VA
    for q in (25,33,50):
        b18=np.nanpercentile(VA["bbw1800"],q)
        for rp in (80,90):
            rq=np.nanpercentile(VA["rel"][VA["bbw1800"]<=b18],rp)
            R[f"comp{q}_rel{rp}"]=(lambda d,b=b18,r=rq:(d["bbw1800"]<=b)&(d["rel"]>=r))
    for q in (25,33,50):
        b36=np.nanpercentile(VA["bbw3600"],q)
        for rp in (80,90):
            rq=np.nanpercentile(VA["rel2"][VA["bbw3600"]<=b36],rp)
            R[f"comp36_{q}_rel{rp}"]=(lambda d,b=b36,r=rq:(d["bbw3600"]<=b)&(d["rel2"]>=r))
    for mp in (50,67,80):
        mq=np.nanpercentile(VA["mag"],mp)
        R[f"mag{mp}"]=(lambda d,m=mq:d["mag"]>=m)
    for mp in (67,80):
        mq=np.nanpercentile(VA["mag"],mp)
        R[f"mag{mp}_ny"]=(lambda d,m=mq:(d["mag"]>=m)&(d["sess"]>0.5))
    # compression x magnitude
    b18_33=np.nanpercentile(VA["bbw1800"],33)
    for mp in (50,67):
        mq=np.nanpercentile(VA["mag"],mp)
        R[f"comp33_mag{mp}"]=(lambda d,b=b18_33,m=mq:(d["bbw1800"]<=b)&(d["mag"]>=m))
    R["ny"]=(lambda d:d["sess"]>0.5)
    R["all"]=(lambda d:np.ones(len(d["p"]),bool))
    return R

def evalcfg(regfn,cov):
    gv=regfn(VA); confv=np.abs(VA["p"]-0.5)[gv]
    if gv.sum()<200: return None
    thr=float(np.quantile(confv,1-cov))
    res={}
    for nm,D in (("va",VA),("te",TE),("oo",OO)):
        g=regfn(D); conf=np.abs(D["p"]-0.5)
        cand=g&(conf>=thr)
        tr=nonoverlap(D["ts"],np.where(cand,conf,-1.0),0.0); tr=tr[cand[tr]]
        corr=((D["p"][tr]>0.5).astype(int)==D["y"][tr])
        res[nm]=(len(tr),corr.mean() if len(tr) else float("nan"),corr,tr,D)
    return thr,res

def main():
    R=regime_builders()
    rows=[]
    for rname,rfn in R.items():
        for cov in (0.20,0.10,0.05,0.02,0.01):
            out=evalcfg(rfn,cov)
            if out is None: continue
            thr,res=out
            nva,accva=res["va"][0],res["va"][1]
            if nva<NMIN_VAL: continue
            nte,accte=res["te"][0],res["te"][1]
            noo,accoo=res["oo"][0],res["oo"][1]
            rows.append((accva,rname,cov,thr,nva,accva,nte,accte,noo,accoo))
    rows.sort(reverse=True)
    print(f"{'regime':18s} {'cov':>5s} {'nVA':>5s} {'accVA':>6s} {'nTE':>5s} {'accTE':>6s} {'nOO':>4s} {'accOO':>6s}")
    for r in rows[:30]:
        _,rname,cov,thr,nva,accva,nte,accte,noo,accoo=r
        print(f"{rname:18s} {cov:5.2f} {nva:5d} {accva:6.3f} {nte:5d} {accte:6.3f} {noo:4d} {accoo:6.3f}")
    # detail on top VAL pick
    print("\n=== TOP VAL PICK detail ===")
    _,rname,cov,thr,nva,accva,nte,accte,noo,accoo=rows[0]
    print(f"regime={rname} cov={cov} thr={thr:.5f}")
    out=evalcfg(R[rname],cov); _,res=out
    for nm in ("te","oo"):
        n,acc,corr,tr,D=res[nm]
        print(f"-- {nm}: n={n} acc={acc:.3f} EV@0.80={acc*0.8-(1-acc):+.3f}")
        mo=D["month"][tr]
        for m in sorted(set(mo.tolist())):
            k=mo==m; print(f"     {m}: n={int(k.sum()):>4} acc={corr[k].mean():.3f}")

if __name__=="__main__": main()
