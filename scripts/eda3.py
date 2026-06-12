"""EDA3 — confluence-fade high-precision probe.
Stack orthogonal reversion conditions and measure fade accuracy + coverage.
Rule: price stretched (multi-TF bb%b extreme) AND order-flow opposing the move
(OFI sign opposite, i.e. flow already reversing) AND optional low-vol regime.
We DEVELOP the rule on TRAIN+VAL (2012-2023), then VALIDATE on TEST(2024-25)+OOS(2026)."""
import os, numpy as np, pandas as pd
import orderflow as OF
FEAT_DIR="/home/sean/git/binary-algo/features"; OF_DIR=OF.OF_DIR

bb=[f"{tf}_bb_pctb" for tf in ["1m","5m","15m","30m","1h"]]
need=["y","valid","fwd_ret","5m_rv_24","15m_rv_24","5m_ret_1","15m_ret_1"]+bb
ofneed=["OF_of_norm_5","OF_of_norm_15","OF_of_uptick_15"]

def load(years):
    parts=[]
    for y in years:
        p=f"{FEAT_DIR}/EURUSD_{y}.parquet"; o=f"{OF_DIR}/EURUSD_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p,columns=need)
        if os.path.exists(o):
            of=pd.read_parquet(o,columns=ofneed).reindex(d.index,method="ffill",limit=3)
            d=pd.concat([d,of],axis=1)
        parts.append(d)
    d=pd.concat(parts); d=d[d.valid==True].copy(); d["up"]=d.y.astype(int)
    d["stretch"]=d[bb].mean(axis=1)
    d["unan_up"]=(d[bb]>0.8).all(axis=1); d["unan_dn"]=(d[bb]<-0.8).all(axis=1)
    return d

dev=load([str(y) for y in range(2012,2024)])
te =load(["2024","2025"]); oo=load(["2026"])
print(f"dev N={len(dev):,} test N={len(te):,} oos N={len(oo):,}")

def fade_rule(d, bb_thr=0.8, use_of=True, of_thr=0.0, lowvol_q=None, vol_ref=None):
    up_str=(d[bb]>bb_thr).all(axis=1)   # stretched up -> fade short (predict down=0)
    dn_str=(d[bb]<-bb_thr).all(axis=1)
    cond_up=up_str.copy(); cond_dn=dn_str.copy()
    if use_of:  # require flow already turning against the stretch
        cond_up &= (d["OF_of_norm_5"]<-of_thr)
        cond_dn &= (d["OF_of_norm_5"]> of_thr)
    if lowvol_q is not None:
        thr=vol_ref["5m_rv_24"].quantile(lowvol_q)
        lv=d["5m_rv_24"]<=thr
        cond_up&=lv; cond_dn&=lv
    sel=cond_up|cond_dn
    if sel.sum()==0: return None
    pred=np.where(cond_up[sel],0,1)
    acc=(pred==d.up[sel].values).mean()
    return {"n":int(sel.sum()),"cov":float(sel.mean()),"acc":float(acc)}

print("\n== Develop on dev: confluence fade variants ==")
configs=[]
for bb_thr in [0.6,0.8,1.0]:
    for use_of in [False,True]:
        for of_thr in ([0.0,0.1] if use_of else [0.0]):
            for lvq in [None,0.5,0.3]:
                r=fade_rule(dev,bb_thr,use_of,of_thr,lvq,dev)
                if r and r["n"]>=300:
                    configs.append((bb_thr,use_of,of_thr,lvq,r))
configs.sort(key=lambda x:-x[4]["acc"])
print(f"  {'bb':>4} {'of':>5} {'ofth':>4} {'lvq':>4} {'dev_n':>7} {'dev_cov':>8} {'dev_acc':>8}")
for bb_thr,use_of,of_thr,lvq,r in configs[:15]:
    print(f"  {bb_thr:4.1f} {str(use_of):>5} {of_thr:4.1f} {str(lvq):>4} {r['n']:7,} {r['cov']:8.4%} {r['acc']:8.4f}")

print("\n== Validate top-5 dev configs on TEST and OOS ==")
print(f"  {'config':>26} {'TEST n/cov/acc':>26} {'OOS n/cov/acc':>26}")
for bb_thr,use_of,of_thr,lvq,r in configs[:5]:
    rt=fade_rule(te,bb_thr,use_of,of_thr,lvq,dev); ro=fade_rule(oo,bb_thr,use_of,of_thr,lvq,dev)
    cfg=f"bb{bb_thr} of{use_of} t{of_thr} lv{lvq}"
    ts=f"{rt['n']}/{rt['cov']:.3%}/{rt['acc']:.3f}" if rt else "n/a"
    os_=f"{ro['n']}/{ro['cov']:.3%}/{ro['acc']:.3f}" if ro else "n/a"
    print(f"  {cfg:>26} {ts:>26} {os_:>26}")
