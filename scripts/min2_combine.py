"""2-min: ensemble v1 (all-bars) + v3 (regime-specialist) direction probs, re-run regime selection.

Averaging two near-independent direction models (different training distributions) should denoise the
in-regime direction signal. Uses v3's regime cols + trend cols. Pre-committed argmax VAL acc s.t. nVA>=NFLOOR.
"""
import sys, numpy as np
MODELS="/home/sean/git/binary-algo/models"; GAP=120
NFLOOR=int(sys.argv[1]) if len(sys.argv)>1 else 300
W=float(sys.argv[2]) if len(sys.argv)>2 else 0.5     # weight on v3 specialist
Z1=np.load(f"{MODELS}/probs_min2_v1.npz",allow_pickle=True)
Z3=np.load(f"{MODELS}/probs_min2_v3.npz",allow_pickle=True)
BQ=float(Z3["bq"]); RQ=float(Z3["rq"])
def get(sp):
    d={k:Z3[f"{sp}_{k}"] for k in ["y","ts","bbw1800","rel","emad900","emad3600","ret300","sess","month"]}
    d["p"]=(1-W)*Z1[f"{sp}_p"]+W*Z3[f"{sp}_p"]     # ensemble direction prob
    return d
VA,TE,OO=get("va"),get("te"),get("oo")
from sklearn.metrics import roc_auc_score
for nm,D in (("va",VA),("te",TE),("oo",OO)):
    reg=(D["bbw1800"]<=BQ)&(D["rel"]>=RQ)
    print(f"[{nm}] ensemble AUC_inregime={roc_auc_score(D['y'][reg],D['p'][reg]):.4f} regn={int(reg.sum())}")
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
    rng=np.random.default_rng(7); n=len(corr); a=[corr[rng.integers(0,n,n)].mean() for _ in range(nb)]
    return (float(np.percentile(a,2.5)),float(np.percentile(a,97.5)))
def base(d): return (d["bbw1800"]<=BQ)&(d["rel"]>=RQ)
def tfilter(d,k):
    s=np.sign(d["p"]-0.5)
    return {"none":np.ones(len(d["p"]),bool),"rev300":s==-np.sign(d["ret300"]),"rev900":s==-np.sign(d["emad900"]),
            "rev3600":s==-np.sign(d["emad3600"]),"cont300":s==np.sign(d["ret300"])}[k]
def regfn(reltight,trend,ny):
    rqt=np.nanpercentile(VA["rel"][base(VA)],reltight) if reltight>0 else -1e9
    def f(d):
        m=base(d)&(d["rel"]>=rqt)&tfilter(d,trend)
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
    print(f"W(v3)={W} NFLOOR>={NFLOOR}"); rows=[]
    for reltight in (0,70,85):
        for trend in ("none","rev300","rev900","rev3600","cont300"):
            for ny in (False,True):
                for cov in (0.20,0.10,0.05,0.03):
                    out=evalcfg(regfn(reltight,trend,ny),cov)
                    if out is None: continue
                    thr,res=out; nva,accva=res["va"][0],res["va"][1]
                    if nva<NFLOOR: continue
                    rows.append((accva,reltight,trend,ny,cov,res))
    rows.sort(key=lambda r:-r[0])
    print(f"{'rel':>3s} {'trend':8s} {'ny':2s} {'cov':>4s} {'nVA':>4s} {'aVA':>5s} {'nTE':>5s} {'aTE':>5s} {'nOO':>4s} {'aOO':>5s} {'OO95CI':>14s}")
    for accva,rt,tr,ny,cov,res in rows[:20]:
        nva=res["va"][0]; nte,ate=res["te"][0],res["te"][1]; noo,aoo=res["oo"][0],res["oo"][1]; lo,hi=boot_ci(res["oo"][2])
        print(f"{rt:3d} {tr:8s} {int(ny):2d} {cov:4.2f} {nva:4d} {accva:5.3f} {nte:5d} {ate:5.3f} {noo:4d} {aoo:5.3f} [{lo:.3f},{hi:.3f}]")
if __name__=="__main__": main()
