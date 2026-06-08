"""Can the 15m PARENT be lifted by a meta-labeler? (the lever that would raise the whole cross-horizon stack)
m15 ensemble (0.647 native) gated by a learned meta predicting P(dir15 correct on the 15-MIN outcome) from ORTHOGONAL
cross-pair/order-flow axes + 15m confidence. If this beats 0.647 with verifiable coverage, re-stack at 5-min.
Run with MX_HOR=15. Held-out TEST24/TEST25/OOS26, non-overlap 900s, CI95.
"""
import os
assert os.environ.get("MX_HOR")=="15", "run with MX_HOR=15"
import json, numpy as np
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
import harness as H, m5_xpair as MX
MODELS="/home/sean/git/binary-algo/models"; PAIR="EURUSD"
base=list(H.feature_cols("EURUSD"))
SPL={"val":["2022","2023"],"test24":["2024"],"test25":["2025"],"oos":["2026"]}
def a15(n): return f"{MODELS}/m15_{PAIR}_{n}"
def load15():
    p=json.load(open(a15("strategy.json"))); L=lgb.Booster(model_file=a15("direction_lgb.txt"))
    G=xgb.XGBClassifier(); G.load_model(a15("direction_xgb.json")); C=CatBoostClassifier(); C.load_model(a15("direction_cat.cbm"))
    return p,L,G,C
def p15f(L,G,C,X): return (L.predict(X.values)+G.predict_proba(X)[:,1]+C.predict_proba(X.fillna(-999))[:,1])/3.0
def boot(c,nb=4000,seed=7):
    c=np.asarray(c,float)
    if len(c)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(c); a=np.array([c[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))
def yr(ts): return (np.asarray(ts,dtype="datetime64[s]").astype("datetime64[Y]").astype(int)+1970)

def feats(w,L,G,C):
    D=MX.build_xp(SPL[w]); D=MX.augment(D,SPL[w],"xpof")
    p15=p15f(L,G,C,D[base].astype("float32"))
    xpm=[c for c in D.columns if c.startswith("agree") or c.startswith("disp") or c=="comp60" or c.startswith("OF_")]
    conf=np.abs(p15-0.5).astype("float32")
    Xm=np.column_stack([conf,D["15m_bb_width"].values.astype("float32"),D["sess_ny"].values.astype("float32")]+
                       [D[c].values.astype("float32") for c in xpm])
    return dict(p15=p15,dir=(p15>0.5).astype(int),y=D["_y"].astype(int).values,ts=D["_ts"].values.astype("int64"),
                ny=D["sess_ny"].values>0.5,Xm=Xm)

def main():
    p,L,G,C=load15(); W={w:feats(w,L,G,C) for w in SPL}
    va=W["val"]; vny=va["ny"]; ycorr=(va["dir"]==va["y"]).astype(int)
    print(f"[m15_meta] target=15-min outcome; VAL dir15-correct={ycorr[vny].mean():.3f}; baseline m15 native combined=0.647",flush=True)
    M=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=15,min_child_samples=800,
        subsample=0.8,colsample_bytree=0.6,reg_lambda=20,n_estimators=400,n_jobs=20,verbosity=-1)
    M.fit(va["Xm"][vny],ycorr[vny]); S={w:M.predict_proba(W[w]["Xm"])[:,1] for w in SPL}
    vts=va["ts"]; vyr=yr(vts)
    def winsel(w,thr):
        d=W[w]; m=d["ny"]&(S[w]>=thr); sel=MX.nonoverlap_chrono(d["ts"],m,900)
        if len(sel)==0: return (0,float("nan"),np.array([]))
        corr=(d["dir"][sel]==d["y"][sel]).astype(float); return (len(sel),float(corr.mean()),corr)
    def hmin(thr):
        a=[]
        for yy in sorted(set(vyr.tolist())):
            m=vny&(vyr==yy)&(S["val"]>=thr); sel=MX.nonoverlap_chrono(vts,m,900)
            if len(sel)<30: return float("nan")
            a.append((va["dir"][sel]==va["y"][sel]).mean())
        return min(a)
    grid=[float(np.quantile(S["val"][vny],q)) for q in (0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.96,0.98)]
    print(f"{'thr':>7} {'hmin':>6}  {'t24':>13} {'t25':>13} {'oos':>13}  {'FLOOR':>6} {'COMB':>20}",flush=True)
    for thr in grid:
        nv,av,_=winsel("val",thr)
        if nv<200: continue
        h=hmin(thr); res={w:winsel(w,thr) for w in ("test24","test25","oos")}
        if any(res[w][0]<12 for w in res): continue
        A=np.concatenate([res[w][2] for w in res]); lo,hi=boot(A); fl=min(res[w][1] for w in res)
        print(f"{thr:7.3f} {h:6.3f}  {res['test24'][1]:.3f}(n{res['test24'][0]:>3}) {res['test25'][1]:.3f}(n{res['test25'][0]:>3}) "
              f"{res['oos'][1]:.3f}(n{res['oos'][0]:>3})  {fl:6.3f} {A.mean():.3f}[{lo:.3f},{hi:.3f}]n{len(A)}",flush=True)
if __name__=="__main__": main()
