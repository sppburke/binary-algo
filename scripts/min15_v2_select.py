"""15-min v2 selector: test specialist blend + reversion + compression-release on the 15m book.

Baseline V27: comp(15m_bb_width<=q)xNY all-bars → 0.642 combined (2024 .691/2025 .590/2026 .592).
Selection: pre-committed argmax VAL acc s.t. nVA>=NFLOOR; report TEST + OOS + combined (V27 metric) + per-year.
"""
import sys, numpy as np
MODELS="/home/sean/git/binary-algo/models"
NFLOOR=int(sys.argv[1]) if len(sys.argv)>1 else 150
Z=np.load(f"{MODELS}/probs_min15_v2.npz",allow_pickle=True)
BQ=float(Z["bq"])
def get(sp): return {k:Z[f"{sp}_{k}"] for k in ["pall","pspec","y","bbw15","bbw5","bbw30","ny","ret15_1","ret5_3","ret15_3","month"]}
VA,TE,OO=get("va"),get("te"),get("oo")
def boot(corr,nb=2000):
    if len(corr)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(7); n=len(corr); a=[corr[rng.integers(0,n,n)].mean() for _ in range(nb)]
    return (float(np.percentile(a,2.5)),float(np.percentile(a,97.5)))
def blendp(d,w): return (1-w)*d["pall"]+w*d["pspec"]
def base(d,bbw_q): return (d["bbw15"]<=bbw_q)&(d["ny"]>0.5)
def tfilter(d,p,kind):
    s=np.sign(p-0.5)
    return {"none":np.ones(len(p),bool),"rev15_1":s==-np.sign(d["ret15_1"]),"rev5_3":s==-np.sign(d["ret5_3"]),
            "rev15_3":s==-np.sign(d["ret15_3"]),"cont15_1":s==np.sign(d["ret15_1"])}[kind]
def release(d,kind):
    if kind=="none": return np.ones(len(d["pall"]),bool)
    rr=d["bbw5"]/(d["bbw30"]+1e-12)
    return rr>=RELQ[kind]
RELQ={}
def regfn(d,p,bbw_q,trend,rel):
    m=base(d,bbw_q)&tfilter(d,p,trend)
    if rel!="none": m=m&release(d,rel)
    return m
def evalcfg(w,bbw_q,trend,rel,cov):
    pV=blendp(VA,w); gv=regfn(VA,pV,bbw_q,trend,rel)
    if gv.sum()<NFLOOR: return None
    thr=float(np.quantile(np.abs(pV-0.5)[gv],1-cov)); res={}
    for nm,D in (("va",VA),("te",TE),("oo",OO)):
        p=blendp(D,w); g=regfn(D,p,bbw_q,trend,rel); m=g&(np.abs(p-0.5)>=thr)
        corr=((p[m]>0.5).astype(int)==D["y"][m])
        res[nm]=(int(m.sum()),corr.mean() if m.sum() else float("nan"),corr,m,D)
    return thr,res
def main():
    global RELQ
    # release thresholds on VA
    rrva=VA["bbw5"]/(VA["bbw30"]+1e-12)
    RELQ={"rel70":np.nanpercentile(rrva,70),"rel50":np.nanpercentile(rrva,50)}
    bbwq={"q33":np.nanpercentile(VA["bbw15"],33),"q50":np.nanpercentile(VA["bbw15"],50),"q67":BQ}
    print(f"NFLOOR>={NFLOOR}  baseline V27 combined 0.642 (2024 .691/2025 .590/2026 .592)")
    rows=[]
    for w in (0.0,0.3,0.5):
        for bk,bq in bbwq.items():
            for trend in ("none","rev15_1","rev5_3","rev15_3","cont15_1"):
                for rel in ("none","rel50","rel70"):
                    for cov in (0.10,0.05,0.02):
                        out=evalcfg(w,bq,trend,rel,cov)
                        if out is None: continue
                        thr,res=out; nva,accva=res["va"][0],res["va"][1]
                        if nva<NFLOOR: continue
                        # combined across te+oo (V27-style held-out)
                        cc=np.concatenate([res["te"][2],res["oo"][2]]); comb=cc.mean() if len(cc) else float("nan")
                        rows.append((accva,w,bk,trend,rel,cov,res,comb,len(cc)))
    rows.sort(key=lambda r:-r[0])
    print(f"{'w':>3s} {'bbwq':4s} {'trend':8s} {'rel':6s} {'cov':>4s} {'nVA':>4s} {'aVA':>5s} {'nTE':>5s} {'aTE':>5s} {'nOO':>4s} {'aOO':>5s} {'comb':>5s} {'ncomb':>5s}")
    for accva,w,bk,trend,rel,cov,res,comb,ncomb in rows[:25]:
        nva=res["va"][0]; nte,ate=res["te"][0],res["te"][1]; noo,aoo=res["oo"][0],res["oo"][1]
        print(f"{w:3.1f} {bk:4s} {trend:8s} {rel:6s} {cov:4.2f} {nva:4d} {accva:5.3f} {nte:5d} {ate:5.3f} {noo:4d} {aoo:5.3f} {comb:5.3f} {ncomb:5d}")
    print("\n=== TOP VAL PICK detail ===")
    accva,w,bk,trend,rel,cov,res,comb,ncomb=rows[0]
    print(f"w={w} bbwq={bk} trend={trend} rel={rel} cov={cov} VALacc={accva:.3f} | COMBINED {comb:.3f} (n{ncomb}) vs V27 0.642")
    for nm in ("te","oo"):
        n,acc,corr,m,D=res[nm]; lo,hi=boot(corr)
        print(f"-- {nm}: n={n} acc={acc:.3f} CI95=[{lo:.3f},{hi:.3f}]")
        mo=D["month"][m]; yrs=sorted(set(s[:4] for s in mo.tolist()))
        for yr in yrs:
            k=np.array([s[:4]==yr for s in mo.tolist()]); print(f"     {yr}: n={int(k.sum()):>4} acc={corr[k].mean():.3f}")
if __name__=="__main__": main()
