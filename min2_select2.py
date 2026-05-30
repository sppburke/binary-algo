"""2-min phase B v2: cross compression-release with magnitude gate + trend alignment, bootstrap CIs.

Adds richer regimes (comp x mag, comp x rel x mag) and reports bootstrap 95% CI on OOS accuracy.
Selection rule pre-committed: argmax VAL non-overlap accuracy subject to nVA>=NFLOOR. Judge TEST+OOS once.
"""
import sys, numpy as np
MODELS="/media/sean/CORSAIR/binary-algo/models"
GAP=120
NFLOOR=int(sys.argv[1]) if len(sys.argv)>1 else 250
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

def boot_ci(corr,n_boot=2000):
    if len(corr)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(12345); n=len(corr)
    accs=[corr[rng.integers(0,n,n)].mean() for _ in range(n_boot)]
    return (float(np.percentile(accs,2.5)),float(np.percentile(accs,97.5)))

def regime_builders():
    R={}
    for q in (33,50,67):
        b18=np.nanpercentile(VA["bbw1800"],q)
        for rp in (70,80,90):
            sub=VA["rel"][VA["bbw1800"]<=b18]
            rq=np.nanpercentile(sub,rp)
            R[f"c{q}_r{rp}"]=(lambda d,b=b18,r=rq:(d["bbw1800"]<=b)&(d["rel"]>=r))
            for mp in (50,67):
                mq=np.nanpercentile(VA["mag"][(VA["bbw1800"]<=b18)&(VA["rel"]>=rq)],mp) if ((VA["bbw1800"]<=b18)&(VA["rel"]>=rq)).sum()>50 else np.nanpercentile(VA["mag"],mp)
                R[f"c{q}_r{rp}_m{mp}"]=(lambda d,b=b18,r=rq,m=mq:(d["bbw1800"]<=b)&(d["rel"]>=r)&(d["mag"]>=m))
    # rel-only (no compression floor) x mag
    for rp in (80,90):
        rq=np.nanpercentile(VA["rel"],rp)
        R[f"relonly{rp}"]=(lambda d,r=rq:d["rel"]>=r)
        for mp in (50,67):
            mq=np.nanpercentile(VA["mag"],mp)
            R[f"relonly{rp}_m{mp}"]=(lambda d,r=rq,m=mq:(d["rel"]>=r)&(d["mag"]>=m))
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
    R=regime_builders(); rows=[]
    for rname,rfn in R.items():
        for cov in (0.10,0.05,0.03,0.02,0.01):
            out=evalcfg(rfn,cov)
            if out is None: continue
            thr,res=out
            nva,accva=res["va"][0],res["va"][1]
            if nva<NFLOOR: continue
            rows.append((accva,rname,cov,thr,res))
    rows.sort(key=lambda r:-r[0])
    print(f"NFLOOR(nVA)>={NFLOOR}")
    print(f"{'regime':16s} {'cov':>5s} {'nVA':>4s} {'aVA':>5s} {'nTE':>5s} {'aTE':>5s} {'nOO':>4s} {'aOO':>5s} {'OO95CI':>14s}")
    for accva,rname,cov,thr,res in rows[:25]:
        nva=res["va"][0]; nte,ate=res["te"][0],res["te"][1]; noo,aoo=res["oo"][0],res["oo"][1]
        lo,hi=boot_ci(res["oo"][2])
        print(f"{rname:16s} {cov:5.2f} {nva:4d} {accva:5.3f} {nte:5d} {ate:5.3f} {noo:4d} {aoo:5.3f} [{lo:.3f},{hi:.3f}]")
    print("\n=== TOP VAL PICK (pre-committed) detail ===")
    accva,rname,cov,thr,res=rows[0]
    print(f"regime={rname} cov={cov} thr={thr:.5f} VALacc={accva:.3f}")
    for nm in ("te","oo"):
        n,acc,corr,tr,D=res[nm]; lo,hi=boot_ci(corr)
        print(f"-- {nm}: n={n} acc={acc:.3f} CI95=[{lo:.3f},{hi:.3f}] EV@0.80={acc*0.8-(1-acc):+.3f}")
        mo=D["month"][tr]
        for m in sorted(set(mo.tolist())):
            k=mo==m; print(f"     {m}: n={int(k.sum()):>4} acc={corr[k].mean():.3f}")

if __name__=="__main__": main()
