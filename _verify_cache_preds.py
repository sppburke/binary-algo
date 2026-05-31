"""Stage 1: compute and cache cand/conf/correct/ts arrays so the random-priority sweep is cheap + memory-light."""
import sys; sys.argv=["x"]
import numpy as np, gc, min1_production as M
p,L,G,C,S=M._load()
out={}
for sp in ("test","oos"):
    b=M.load_split(sp); X,y,mag,valid,ts,idx=M.prep(b)
    pr=M._blend(p,L,G,C,S,X)
    bbw=X["bbw1800"].values; rel=X["rel_ratio"].values; r300=X["ret300"].values
    gate=valid&(bbw<=p["bbw1800_q67"])&(rel>=p["rel_tighten"])&(np.sign(pr-0.5)==-np.sign(r300))
    conf=np.abs(pr-0.5); cand=gate&(conf>=p["conf_thr"])
    correct=((pr>0.5).astype(int)==y).astype(np.int8)
    out[f"{sp}_ts"]=ts.astype(np.int64)
    out[f"{sp}_conf"]=conf.astype(np.float64)
    out[f"{sp}_cand"]=cand
    out[f"{sp}_correct"]=correct
    del b,X,pr,bbw,rel,r300,gate; gc.collect()
np.savez("/tmp/_gb_cache.npz", **out)
print("cached", {k:v.shape for k,v in out.items() if k.endswith("cand")})
print("GAP=",M.GAP)
