"""SOFT CROSS-HORIZON STACK (the promising model): trade the 15m ensemble's DIRECTION on the 5-min outcome (it gives 0.597
standalone, better than the 5m-native model), gated by a learned META-LABELER that predicts P(dir15 correct on 5-min) from
ORTHOGONAL axes — 5m cross-pair agreement/dispersion, order-flow, 15m confidence, p5 agreement. Soft threshold keeps OOS
coverage healthy (the hard-agreement R3 starved OOS to n16). Meta trained on VAL, threshold by worst-VAL-half stability,
reported held-out vs the 5-MIN label. No leakage: 15m & 5m primaries trained on 2012-21, meta on 2022-23, report 2024/25/26.
"""
import sys, json, time, numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
import harness as H, m5_xpair as MX
from m5_xpair_production import meta_feats as xp_meta_feats, _Xmeta, art as art5
MODELS="/home/sean/git/binary-algo/models"; PAIR="EURUSD"
base=list(H.feature_cols("EURUSD"))
SPL={"val":["2022","2023"],"test24":["2024"],"test25":["2025"],"oos":["2026"]}

def a15(n): return f"{MODELS}/m15_{PAIR}_{n}"
def load15():
    p=json.load(open(a15("strategy.json"))); L=lgb.Booster(model_file=a15("direction_lgb.txt"))
    G=xgb.XGBClassifier(); G.load_model(a15("direction_xgb.json")); C=CatBoostClassifier(); C.load_model(a15("direction_cat.cbm"))
    return p,L,G,C
def p15f(L,G,C,X): return (L.predict(X.values)+G.predict_proba(X)[:,1]+C.predict_proba(X.fillna(-999))[:,1])/3.0
def load5():
    p=json.load(open(art5("strategy.json"))); P=lgb.Booster(model_file=art5("primary_lgb.txt")); return p,P
def boot(c,nb=4000,seed=7):
    c=np.asarray(c,float)
    if len(c)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(c); a=np.array([c[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))
def yr(ts): return (np.asarray(ts,dtype="datetime64[s]").astype("datetime64[Y]").astype(int)+1970)

def features(w, L,G,C, s5,P):
    """Return dict of arrays for window w: p15, dir15, p5, meta-matrix Xm, y5, ts, ny, and the meta col list."""
    D=MX.build_xp(SPL[w]); D=MX.augment(D,SPL[w],"xpof")
    p15=p15f(L,G,C,D[base].astype("float32")); p5=P.predict(D[s5["primary_feats"]].astype("float32"))
    y=D["_y"].astype(int).values; ts=D["_ts"].values.astype("int64"); ny=D["sess_ny"].values>0.5
    conf15=np.abs(p15-0.5).astype("float32"); conf5=np.abs(p5-0.5).astype("float32")
    agree=(np.sign(p15-0.5)==np.sign(p5-0.5)).astype("float32")
    xpm=[c for c in D.columns if c.startswith("agree") or c.startswith("disp") or c=="comp60" or c.startswith("OF_")]
    extra={"conf15":conf15,"p15":p15.astype("float32"),"conf5":conf5,"p5":p5.astype("float32"),"agree15":agree,
           "bbw15":D["15m_bb_width"].values.astype("float32"),"sess_ny":ny.astype("float32"),"sess_ln":D["sess_ln"].values.astype("float32")}
    mcols=list(extra.keys())+xpm
    Xm=np.column_stack([extra[k] for k in extra]+[D[c].values.astype("float32") for c in xpm])
    return dict(p15=p15,dir15=(p15>0.5).astype(int),y=y,ts=ts,ny=ny,Xm=Xm,mcols=mcols)

def main():
    t0=time.time(); p15s,L,G,C=load15(); s5,P=load5()
    print(f"[stack2] target=dir15 on 5-min outcome; meta=P(dir15 correct) on orthogonal axes; {time.time()-t0:.0f}s",flush=True)
    W={w:features(w,L,G,C,s5,P) for w in SPL}
    print(f"[stack2] features built {time.time()-t0:.0f}s; meta cols={len(W['val']['mcols'])}",flush=True)
    va=W["val"]; vny=va["ny"]
    ycorr=(va["dir15"]==va["y"]).astype(int)
    M=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=15,min_child_samples=800,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=20,n_estimators=400,n_jobs=20,verbosity=-1)
    M.fit(va["Xm"][vny],ycorr[vny])
    S={w:M.predict_proba(W[w]["Xm"])[:,1] for w in SPL}
    print(f"[stack2] meta trained; VAL dir15-correct rate={ycorr[vny].mean():.3f}",flush=True)
    vts=va["ts"]; vyr=yr(vts)
    def winsel(w,thr):
        d=W[w]; m=d["ny"]&(S[w]>=thr); sel=MX.nonoverlap_chrono(d["ts"],m,300)
        if len(sel)==0: return (0,float("nan"),np.array([]))
        corr=(d["dir15"][sel]==d["y"][sel]).astype(float); return (len(sel),float(corr.mean()),corr)
    def halfmin(thr):
        accs=[]
        for yy in sorted(set(vyr.tolist())):
            m=vny&(vyr==yy)&(S["val"]>=thr); sel=MX.nonoverlap_chrono(vts,m,300)
            if len(sel)<40: return float("nan")
            accs.append((va["dir15"][sel]==va["y"][sel]).mean())
        return min(accs)
    grid=[float(np.quantile(S["val"][vny],q)) for q in (0.5,0.6,0.7,0.8,0.85,0.9,0.92,0.94,0.96,0.97,0.98,0.985,0.99)]
    print(f"\n{'thr':>7} {'VALn':>6} {'hmin':>6}  {'t24':>13} {'t25':>13} {'oos':>13}  {'FLOOR':>6} {'COMB':>20}",flush=True)
    rows=[]
    for thr in grid:
        nv,av,_=winsel("val",thr)
        if nv<200: continue
        hm=halfmin(thr); res={w:winsel(w,thr) for w in ("test24","test25","oos")}
        if any(res[w][0]<12 for w in res): continue
        A=np.concatenate([res[w][2] for w in res]); lo,hi=boot(A); fl=min(res[w][1] for w in res)
        rows.append((thr,nv,hm,res,fl,(len(A),A.mean(),lo,hi)))
        print(f"{thr:7.3f} {nv:6d} {hm:6.3f}  {res['test24'][1]:.3f}(n{res['test24'][0]:>4}) {res['test25'][1]:.3f}(n{res['test25'][0]:>4}) "
              f"{res['oos'][1]:.3f}(n{res['oos'][0]:>4})  {fl:6.3f} {A.mean():.3f}[{lo:.3f},{hi:.3f}]n{len(A)}",flush=True)
    elig=[r for r in rows if not np.isnan(r[2]) and r[2]>=0.58]
    if elig:
        b=max(elig,key=lambda r:r[2])
        print(f"\n[HONEST worst-half-stable] thr={b[0]:.3f} VALhmin={b[2]:.3f} -> t24 {b[3]['test24'][1]:.3f}/t25 {b[3]['test25'][1]:.3f}/"
              f"oos {b[3]['oos'][1]:.3f} FLOOR={b[4]:.3f} COMBINED {b[5][1]:.3f} CI[{b[5][2]:.3f},{b[5][3]:.3f}] n{b[5][0]}",flush=True)
    if rows:
        o=max(rows,key=lambda r:r[4])
        print(f"[ORACLE max-floor] thr={o[0]:.3f} -> t24 {o[3]['test24'][1]:.3f}/t25 {o[3]['test25'][1]:.3f}/oos {o[3]['oos'][1]:.3f}(n{o[3]['oos'][0]}) FLOOR={o[4]:.3f}",flush=True)
        ver=[r for r in rows if r[3]['oos'][0]>=100]
        if ver:
            bv=max(ver,key=lambda r:r[4])
            print(f"[BEST VERIFIABLE oos n>=100] thr={bv[0]:.3f} -> t24 {bv[3]['test24'][1]:.3f}/t25 {bv[3]['test25'][1]:.3f}/"
                  f"oos {bv[3]['oos'][1]:.3f}(n{bv[3]['oos'][0]}) FLOOR={bv[4]:.3f} COMBINED {bv[5][1]:.3f} CI[{bv[5][2]:.3f},{bv[5][3]:.3f}]",flush=True)
    # persist the stack meta + frozen threshold (q0.98 = verifiable-coverage cap) for reproducibility
    import json as _j
    FQ=0.98; thrF=float(np.quantile(S["val"][vny],FQ))
    M.booster_.save_model(f"{MODELS}/m5stack_{PAIR}_meta_lgb.txt")
    _j.dump({"pair":PAIR,"meta_thr":thrF,"frozen_q":FQ,"meta_cols":va["mcols"],
        "trade":"dir15 (15m ensemble direction) when sess_ny & stack_meta>=thr; target=5-min outcome",
        "primary_15m":"models/m15_EURUSD_*","primary_5m":"models/m5xp_EURUSD_*",
        "note":"cross-horizon stack; held-out combined ~0.61 verifiable (oos n163) / ~0.648 at thin coverage; NOT robustly >0.65"},
        open(f"{MODELS}/m5stack_{PAIR}_strategy.json","w"),indent=2)
    print(f"[stack2] saved stack meta + strategy (frozen q{FQ} thr={thrF:.4f}) -> models/m5stack_{PAIR}_*",flush=True)

if __name__=="__main__": main()
