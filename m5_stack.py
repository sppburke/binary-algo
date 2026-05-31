"""CROSS-HORIZON STACK: combine the 15m ensemble (0.597 on the 5-min sub-move) with the 5m cross-pair meta-labeler.
Two semi-independent edges; require AGREEMENT to concentrate accuracy and (hopefully) lift the binding 2025 window.
All scored vs the 5-MIN label, held-out TEST24/TEST25/OOS26, non-overlap 300s, CI95.
"""
import json, numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
import harness as H, m5_xpair as MX
from m5_xpair_production import meta_feats, _Xmeta, art as art5
MODELS="/media/sean/CORSAIR/binary-algo/models"; PAIR="EURUSD"
base=list(H.feature_cols("EURUSD"))
SPL={"test24":["2024"],"test25":["2025"],"oos":["2026"]}

def a15(n): return f"{MODELS}/m15_{PAIR}_{n}"
def load15():
    p=json.load(open(a15("strategy.json")))
    L=lgb.Booster(model_file=a15("direction_lgb.txt")); G=xgb.XGBClassifier(); G.load_model(a15("direction_xgb.json"))
    C=CatBoostClassifier(); C.load_model(a15("direction_cat.cbm")); return p,L,G,C
def p15f(L,G,C,X): return (L.predict(X.values)+G.predict_proba(X)[:,1]+C.predict_proba(X.fillna(-999))[:,1])/3.0
def load5():
    p=json.load(open(art5("strategy.json"))); P=lgb.Booster(model_file=art5("primary_lgb.txt"))
    M=lgb.Booster(model_file=art5("meta_lgb.txt")); return p,P,M
def boot(c,nb=4000,seed=7):
    c=np.asarray(c,float)
    if len(c)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(c); a=np.array([c[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))

def report(name,per):
    A=np.concatenate([per[w][2] for w in per]); lo,hi=boot(A)
    cells=" ".join(f"{w}:{per[w][1]:.3f}(n{per[w][0]})" for w in per)
    flo=min(per[w][1] for w in per)
    print(f"  {name:<34} {cells}  FLOOR={flo:.3f} COMB n{len(A)} {A.mean():.3f}[{lo:.3f},{hi:.3f}]")

def run():
    p15,L,G,C=load15(); s5,P,M=load5()
    bbw=p15["bb_width_thr"]; thr15=p15["conf_thr"]; pf=s5["primary_feats"]; mf=s5["meta_feats"]; tm=s5["meta_thr"]
    print(f"[stack] 15m gate(bbw<={bbw:.4f}&NY,conf>={thr15:.3f}) x 5m meta(thr>={tm:.3f}); target=5-min; non-overlap 300s")
    # precompute per-window arrays
    Wd={}
    for w in ("test24","test25","oos"):
        D=MX.build_xp(SPL[w]); D=MX.augment(D,SPL[w],"xpof")
        p15v=p15f(L,G,C,D[base].astype("float32"))
        p5v=P.predict(D[pf].astype("float32")); m5v=M.predict(_Xmeta(D,p5v,mf))
        y=D["_y"].astype(int).values; ts=D["_ts"].values.astype("int64"); ny=D["sess_ny"].values>0.5
        g15=(D["15m_bb_width"].values.astype(float)<=bbw)&ny
        Wd[w]=dict(p15=p15v,p5=p5v,m5=m5v,y=y,ts=ts,ny=ny,g15=g15)
    def evalrule(maskfn,dirfn,gap=300):
        per={}
        for w in ("test24","test25","oos"):
            d=Wd[w]; m=maskfn(d); sel=MX.nonoverlap_chrono(d["ts"],m,gap)
            if len(sel)==0: per[w]=(0,float("nan"),np.array([])); continue
            pred=dirfn(d)[sel]; corr=(pred==d["y"][sel]).astype(float); per[w]=(len(sel),corr.mean(),corr)
        return per
    dir15=lambda d:(d["p15"]>0.5).astype(int); dir5=lambda d:(d["p5"]>0.5).astype(int)
    agree=lambda d:(np.sign(d["p15"]-0.5)==np.sign(d["p5"]-0.5))
    c15=lambda d:np.abs(d["p15"]-0.5)>=thr15
    print("\n-- baselines --")
    report("R1: 15m-confident -> dir15", evalrule(lambda d: d["g15"]&c15(d), dir15))
    report("R4: 5m-meta -> dir5",        evalrule(lambda d: d["ny"]&(d["m5"]>=tm), dir5))
    print("-- cross-horizon AGREEMENT stacks --")
    report("R2: 15m-conf & agree -> dir15", evalrule(lambda d: d["g15"]&c15(d)&agree(d), dir15))
    report("R3: 15m-conf & 5m-meta & agree", evalrule(lambda d: d["g15"]&c15(d)&(d["m5"]>=tm)&agree(d), dir15))
    report("R5: 5m-meta & 15m-confirms dir", evalrule(lambda d: d["ny"]&(d["m5"]>=tm)&agree(d), dir5))
    report("R6: union-conf & agree (NY)",   evalrule(lambda d: d["ny"]&c15(d)&(d["m5"]>=tm)&agree(d), dir15))
    # threshold sweep on the agreement stack R3 (raise BOTH thresholds together via meta quantile proxy)
    print("-- R3 sweep: raise 5m-meta threshold (15m gate fixed) --")
    for mt in (tm, 0.60, 0.63, 0.66, 0.70):
        report(f"R3 meta>={mt:.2f}", evalrule(lambda d,mt=mt: d["g15"]&c15(d)&(d["m5"]>=mt)&agree(d), dir15))

if __name__=="__main__": run()
