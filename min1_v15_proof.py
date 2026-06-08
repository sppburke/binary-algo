"""1-min IMPROVEMENT proof: the 2-min reversion lesson applied to the 60s book. Two pre-committed configs.

Baseline (min1_production): TEST 0.682 (n759) / OOS 0.780 (n50, 44 in April).
Lesson transferred: REVERSION filter (bet AGAINST the last 5-min move, rev300) — the 1m book had no trend filter.

A) argmax-VAL pick:   all-bars dir, rel re-tightened to p90, rev300, cov0.05
B) robust large-n:    0.5*all-bars + 0.5*specialist, rel p80, rev300, cov0.05  (3.7x the OOS trades)
"""
import sys, numpy as np
MODELS="/home/sean/git/binary-algo/models"; GAP=60
Z=np.load(f"{MODELS}/probs_min1_v15.npz",allow_pickle=True)
BQ=float(Z["bq"]); RQ=float(Z["rq"])
def get(sp): return {k:Z[f"{sp}_{k}"] for k in ["pall","pspec","p2m","mag","y","ts","bbw1800","rel","ret300","month"]}
VA,TE,OO=get("va"),get("te"),get("oo")
def nonoverlap(ts,conf,thr,gap=GAP):
    sel=np.where(conf>=thr)[0]
    if len(sel)==0: return np.array([],dtype=int)
    order=sel[np.argsort(-conf[sel])]
    tmin=int(ts[order].min()); span=int(ts[order].max()-tmin)+gap+2
    blk=np.zeros(span,bool); take=[]
    for i in order:
        t=int(ts[i]-tmin)
        if blk[t]: continue
        take.append(i); blk[max(0,t-gap+1):t+gap]=True
    return np.sort(np.array(take))
def boot(corr,nb=5000):
    if len(corr)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(7); n=len(corr); a=[corr[rng.integers(0,n,n)].mean() for _ in range(nb)]
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))
def run(name,w,w2,reltight,cov):
    rqt=np.nanpercentile(VA["rel"][(VA["bbw1800"]<=BQ)&(VA["rel"]>=RQ)],reltight)
    def p(d): return (1-w-w2)*d["pall"]+w*d["pspec"]+w2*d["p2m"]
    def gate(d):
        pp=p(d)
        return (d["bbw1800"]<=BQ)&(d["rel"]>=rqt)&(np.sign(pp-0.5)==-np.sign(d["ret300"]))
    pV=p(VA); gv=gate(VA); thr=float(np.quantile(np.abs(pV-0.5)[gv],1-cov))
    print(f"\n##### {name}: w={w} w2={w2} rel_p{reltight} rev300 cov={cov} thr={thr:.5f} #####")
    for nm,D in (("VAL",VA),("TEST 2024-25",TE),("OOS 2026",OO)):
        pp=p(D); g=gate(D); conf=np.abs(pp-0.5); cand=g&(conf>=thr)
        tr=nonoverlap(D["ts"],np.where(cand,conf,-1.0),0.0); tr=tr[cand[tr]]
        corr=((pp[tr]>0.5).astype(int)==D["y"][tr]); acc=corr.mean() if len(tr) else float("nan"); lo,hi=boot(corr)
        print(f"  {nm:14s} n={len(tr):4d} acc={acc:.3f} CI95=[{lo:.3f},{hi:.3f}] EV@0.80={acc*0.8-(1-acc):+.3f}")
        if nm.startswith("OOS") or nm.startswith("TEST"):
            mo=D["month"][tr]
            for m in sorted(set(mo.tolist())):
                k=mo==m; print(f"        {m}: n={int(k.sum()):>4} acc={corr[k].mean():.3f}")
print("BASELINE: TEST 0.682 (n759) / OOS 0.780 (n50, 44 in April)")
run("A) argmax-VAL (all-bars + reversion)", 0.0,0.0,90,0.05)
run("B) robust large-n (specialist blend + reversion)", 0.5,0.0,80,0.05)
