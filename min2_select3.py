"""2-min v3 selector: specialist probs + trend alignment. Pre-committed argmax VAL acc s.t. nVA>=NFLOOR."""
import sys, numpy as np
MODELS="/home/sean/git/binary-algo/models"; GAP=120
NFLOOR=int(sys.argv[1]) if len(sys.argv)>1 else 300
Z=np.load(f"{MODELS}/probs_min2_v3.npz",allow_pickle=True)
BQ=float(Z["bq"]); RQ=float(Z["rq"])
def get(sp):
    return {k:Z[f"{sp}_{k}"] for k in ["p","y","ts","bbw1800","rel","bbw3600","rel2","emad900","emad3600","ret300","sess","month"]}
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
def boot_ci(corr,nb=2000):
    if len(corr)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(7); n=len(corr)
    a=[corr[rng.integers(0,n,n)].mean() for _ in range(nb)]
    return (float(np.percentile(a,2.5)),float(np.percentile(a,97.5)))
def base_regime(d):  # specialist betting regime, optionally re-tightened on rel percentile
    return (d["bbw1800"]<=BQ)&(d["rel"]>=RQ)
def trend_filter(d,kind):
    dirsign=np.sign(d["p"]-0.5)
    if kind=="none": return np.ones(len(d["p"]),bool)
    if kind=="cont900": return dirsign==np.sign(d["emad900"])
    if kind=="rev900":  return dirsign==-np.sign(d["emad900"])
    if kind=="cont3600":return dirsign==np.sign(d["emad3600"])
    if kind=="rev3600": return dirsign==-np.sign(d["emad3600"])
    if kind=="cont300": return dirsign==np.sign(d["ret300"])
    if kind=="rev300":  return dirsign==-np.sign(d["ret300"])
    return np.ones(len(d["p"]),bool)
def regfn(reltight,trend,ny):
    rqt=np.nanpercentile(VA["rel"][base_regime(VA)],reltight) if reltight>0 else -1e9
    def f(d):
        m=base_regime(d)&(d["rel"]>=rqt)&trend_filter(d,trend)
        if ny: m=m&(d["sess"]>0.5)
        return m
    return f
def evalcfg(rf,cov):
    gv=rf(VA); confv=np.abs(VA["p"]-0.5)[gv]
    if gv.sum()<150: return None
    thr=float(np.quantile(confv,1-cov)); res={}
    for nm,D in (("va",VA),("te",TE),("oo",OO)):
        g=rf(D); conf=np.abs(D["p"]-0.5); cand=g&(conf>=thr)
        tr=nonoverlap(D["ts"],np.where(cand,conf,-1.0),0.0); tr=tr[cand[tr]]
        corr=((D["p"][tr]>0.5).astype(int)==D["y"][tr])
        res[nm]=(len(tr),corr.mean() if len(tr) else float("nan"),corr,tr,D)
    return thr,res
def main():
    print(f"specialist regime bbw1800<={BQ:.2e} rel>={RQ:.3f}  NFLOOR>={NFLOOR}")
    rows=[]
    for reltight in (0,70,85):
        for trend in ("none","cont900","rev900","cont3600","rev3600","cont300","rev300"):
            for ny in (False,True):
                for cov in (0.20,0.10,0.05,0.03):
                    out=evalcfg(regfn(reltight,trend,ny),cov)
                    if out is None: continue
                    thr,res=out; nva,accva=res["va"][0],res["va"][1]
                    if nva<NFLOOR: continue
                    rows.append((accva,reltight,trend,ny,cov,res))
    rows.sort(key=lambda r:-r[0])
    print(f"{'rel':>3s} {'trend':8s} {'ny':2s} {'cov':>4s} {'nVA':>4s} {'aVA':>5s} {'nTE':>5s} {'aTE':>5s} {'nOO':>4s} {'aOO':>5s} {'OO95CI':>14s}")
    for accva,rt,tr,ny,cov,res in rows[:25]:
        nva=res["va"][0]; nte,ate=res["te"][0],res["te"][1]; noo,aoo=res["oo"][0],res["oo"][1]
        lo,hi=boot_ci(res["oo"][2])
        print(f"{rt:3d} {tr:8s} {int(ny):2d} {cov:4.2f} {nva:4d} {accva:5.3f} {nte:5d} {ate:5.3f} {noo:4d} {aoo:5.3f} [{lo:.3f},{hi:.3f}]")
    print("\n=== TOP VAL PICK detail ===")
    accva,rt,tr,ny,cov,res=rows[0]
    print(f"reltight={rt} trend={tr} ny={ny} cov={cov} VALacc={accva:.3f}")
    for nm in ("te","oo"):
        n,acc,corr,trd,D=res[nm]; lo,hi=boot_ci(corr)
        print(f"-- {nm}: n={n} acc={acc:.3f} CI95=[{lo:.3f},{hi:.3f}] EV@0.80={acc*0.8-(1-acc):+.3f}")
        mo=D["month"][trd]
        for m in sorted(set(mo.tolist())):
            k=mo==m; print(f"     {m}: n={int(k.sum()):>4} acc={corr[k].mean():.3f}")
if __name__=="__main__": main()
