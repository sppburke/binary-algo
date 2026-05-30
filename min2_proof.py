"""2-min FINAL pre-committed proof. Locks the pipeline selected by VAL-only criteria, judges TEST+OOS once.

Pre-committed config (no peeking at OOS to choose any of these):
  W=0.5  (argmax VAL in-regime AUC over {0.2..0.6})
  regime = compression-release: bbw1800<=train_q67 AND rel_ratio>=train_p70
  trend  = reversion vs last 5-min move: bet against sign(ret300)   (rev300)
  coverage = 3% (confidence threshold = 97th pct of |p-0.5| over VAL regime bars)
  selection family = {reltight 0/70/85, trend none/rev300/rev900/rev3600, cov .20/.10/.05/.03}, no session
                     sub-filter; argmax VAL accuracy s.t. nVA>=350  ->  rev300, reltight 0, cov 0.03.
"""
import numpy as np
MODELS="/media/sean/CORSAIR/binary-algo/models"; GAP=120; W=0.5
Z1=np.load(f"{MODELS}/probs_min2_v1.npz",allow_pickle=True)
Z3=np.load(f"{MODELS}/probs_min2_v3.npz",allow_pickle=True)
Z4=np.load(f"{MODELS}/probs_min2_v4.npz",allow_pickle=True)
BQ=float(Z3["bq"]); RQ=float(Z3["rq"])
def get(sp):
    d={k:Z3[f"{sp}_{k}"] for k in ["y","ts","bbw1800","rel","ret300","month"]}
    d["p"]=(1-W)*Z1[f"{sp}_p"]+W*Z4[f"{sp}_p"]; return d
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
    rng=np.random.default_rng(7); n=len(corr); a=[corr[rng.integers(0,n,n)].mean() for _ in range(nb)]
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))
def regime(d): # compression-release AND reversion-vs-ret300
    return (d["bbw1800"]<=BQ)&(d["rel"]>=RQ)&(np.sign(d["p"]-0.5)==-np.sign(d["ret300"]))
COV=0.03
gv=regime(VA); THR=float(np.quantile(np.abs(VA["p"]-0.5)[gv],1-COV))
print(f"pre-committed: W={W} bbw1800<={BQ:.3e} rel>={RQ:.3f} rev(ret300) cov={COV} conf_thr={THR:.5f}")
for nm,D in (("VAL",VA),("TEST 2024-25",TE),("OOS 2026",OO)):
    g=regime(D); conf=np.abs(D["p"]-0.5); cand=g&(conf>=THR)
    tr=nonoverlap(D["ts"],np.where(cand,conf,-1.0),0.0); tr=tr[cand[tr]]
    corr=((D["p"][tr]>0.5).astype(int)==D["y"][tr]); acc=corr.mean()
    lo,hi=boot(corr)
    print(f"\n=== {nm} === trades={len(tr)} acc={acc:.4f} CI95=[{lo:.3f},{hi:.3f}] EV@0.80={acc*0.8-(1-acc):+.3f}")
    mo=D["month"][tr]
    for m in sorted(set(mo.tolist())):
        k=mo==m; print(f"     {m}: n={int(k.sum()):>4} acc={corr[k].mean():.3f}")
