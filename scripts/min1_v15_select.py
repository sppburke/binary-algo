"""1-min v15 selector: test 2-min lessons on the 60s book. Pre-committed argmax VAL acc s.t. nVA>=NFLOOR.

Levers (each toggled in the sweep):
  - direction blend  p = (1-w)*pall + w*pspec      (specialist transfer)
  - regime gate      compression-release bbw1800<=bq & rel>=rq  (+ rel re-tighten)
  - magnitude gate   mag>=mq      (helps at 60s)
  - trend filter     rev/cont vs ret60/ret120/ret300
  - 2m confirmation  sign(p-0.5)==sign(p2m-0.5)   AND optionally p2m in its regime/confidence
  - coverage         confidence quantile
Baseline to beat: TEST 0.682 (n759) / OOS 0.780 (n50, April-heavy).
"""
import sys, numpy as np
MODELS="/home/sean/git/binary-algo/models"; GAP=60
NFLOOR=int(sys.argv[1]) if len(sys.argv)>1 else 120
Z=np.load(f"{MODELS}/probs_min1_v15.npz",allow_pickle=True)
BQ=float(Z["bq"]); RQ=float(Z["rq"])
def get(sp):
    return {k:Z[f"{sp}_{k}"] for k in ["pall","pspec","p2m","mag","y","ts","bbw1800","rel","ret60","ret120","ret300","sess","month"]}
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
def boot(corr,nb=2000):
    if len(corr)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(7); n=len(corr); a=[corr[rng.integers(0,n,n)].mean() for _ in range(nb)]
    return (float(np.percentile(a,2.5)),float(np.percentile(a,97.5)))
def blendp(d,w,w2=0.0): return (1-w-w2)*d["pall"]+w*d["pspec"]+w2*d["p2m"]
def tfilter(d,p,kind):
    s=np.sign(p-0.5)
    R={"none":np.ones(len(p),bool),
       "rev60":s==-np.sign(d["ret60"]),"rev120":s==-np.sign(d["ret120"]),"rev300":s==-np.sign(d["ret300"]),
       "cont60":s==np.sign(d["ret60"]),"cont300":s==np.sign(d["ret300"])}
    return R[kind]
def regfn(d,p,w,reltight,mag_q,trend,confirm2m):
    m=(d["bbw1800"]<=BQ)&(d["rel"]>=RQ)
    if reltight>0:
        rqt=np.nanpercentile(VA["rel"][(VA["bbw1800"]<=BQ)&(VA["rel"]>=RQ)],reltight); m=m&(d["rel"]>=rqt)
    if mag_q>0:
        mq=np.nanpercentile(VA["mag"],mag_q); m=m&(d["mag"]>=mq)
    m=m&tfilter(d,p,trend)
    if confirm2m: m=m&(np.sign(p-0.5)==np.sign(d["p2m"]-0.5))
    return m
def eval_split(D,w,w2,reltight,mag_q,trend,confirm2m,thr):
    p=blendp(D,w,w2); g=regfn(D,p,w,reltight,mag_q,trend,confirm2m); conf=np.abs(p-0.5); cand=g&(conf>=thr)
    tr=nonoverlap(D["ts"],np.where(cand,conf,-1.0),0.0); tr=tr[cand[tr]]
    corr=((p[tr]>0.5).astype(int)==D["y"][tr])
    return (len(tr),corr.mean() if len(tr) else float("nan"),corr,tr,D)
def eval_va(w,w2,reltight,mag_q,trend,confirm2m,cov):
    pV=blendp(VA,w,w2); gv=regfn(VA,pV,w,reltight,mag_q,trend,confirm2m)
    if gv.sum()<120: return None
    thr=float(np.quantile(np.abs(pV-0.5)[gv],1-cov))
    return thr, eval_split(VA,w,w2,reltight,mag_q,trend,confirm2m,thr)
def main():
    print(f"NFLOOR>={NFLOOR}  (baseline TEST 0.682/n759, OOS 0.780/n50)"); rows=[]
    grid=[]
    for w in (0.0,0.3,0.5):              # specialist weight
        for w2 in (0.0,0.3,0.5,0.7):     # 2-min cross-horizon weight
            if w+w2>1.0: continue
            for reltight in (0,80,90):
                for mag_q in (0,50,67):
                    for trend in ("none","rev60","rev120","rev300"):
                        for confirm2m in (False,True):
                            for cov in (0.10,0.05,0.02):
                                grid.append((w,w2,reltight,mag_q,trend,confirm2m,cov))
    # phase 1: rank on VAL only (cheap)
    va=[]
    for cfg in grid:
        out=eval_va(*cfg)
        if out is None: continue
        thr,r=out; nva,accva=r[0],r[1]
        if nva<NFLOOR: continue
        va.append((accva,thr,*cfg))
    va.sort(key=lambda r:-r[0])
    print(f"phase1: {len(va)} configs pass nVA>={NFLOOR}; evaluating TEST/OOS for top 40")
    # phase 2: TEST/OOS only for top-40 VAL configs
    rows=[]
    for accva,thr,w,w2,rlt,mag,trend,c2,cov in va[:40]:
        te=eval_split(TE,w,w2,rlt,mag,trend,c2,thr); oo=eval_split(OO,w,w2,rlt,mag,trend,c2,thr)
        rows.append((accva,w,w2,rlt,mag,trend,c2,cov,te,oo))
    print(f"{'w':>3s} {'w2':>3s} {'rlt':>3s} {'mag':>3s} {'trend':6s} {'2m':2s} {'cov':>4s} {'aVA':>5s} {'nTE':>4s} {'aTE':>5s} {'nOO':>3s} {'aOO':>5s} {'OO95CI':>13s}")
    for accva,w,w2,rlt,mag,trend,c2,cov,te,oo in rows:
        nte,ate=te[0],te[1]; noo,aoo=oo[0],oo[1]; lo,hi=boot(oo[2])
        print(f"{w:3.1f} {w2:3.1f} {rlt:3d} {mag:3d} {trend:6s} {int(c2):2d} {cov:4.2f} {accva:5.3f} {nte:4d} {ate:5.3f} {noo:3d} {aoo:5.3f} [{lo:.2f},{hi:.2f}]")
    print("\n=== TOP VAL PICK detail ===")
    accva,w,w2,rlt,mag,trend,c2,cov,te,oo=rows[0]
    print(f"w={w} w2={w2} reltight={rlt} mag_q={mag} trend={trend} confirm2m={c2} cov={cov} VALacc={accva:.3f}")
    for nm,res in (("te",te),("oo",oo)):
        n,acc,corr,tr,D=res; lo,hi=boot(corr)
        print(f"-- {nm}: n={n} acc={acc:.3f} CI95=[{lo:.3f},{hi:.3f}] EV@0.80={acc*0.8-(1-acc):+.3f}")
        mo=D["month"][tr]
        for m in sorted(set(mo.tolist())):
            k=mo==m; print(f"     {m}: n={int(k.sum()):>4} acc={corr[k].mean():.3f}")
if __name__=="__main__": main()
