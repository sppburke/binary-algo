"""CULMINATION: WALK-FORWARD CROSS-HORIZON STACK — combine every lever that helped.
Per fold (expanding window, 1yr gap): retrain a 15m parent (lgb) + a 5m primary (lgb) on cross-pair+base+OF; a stack meta
predicts P(dir15 correct on the 5-MIN outcome) from orthogonal axes (cross-pair agree/disp + OF + conf15 + p5); abstain unless
meta>=thr & NY. Aggregate held-out 2024/2025/2026 (each predicted by a model trained only on prior years), non-overlap 300s,
CI95. Reports the coverage curve + best verifiable (oos n>=120) and honest worst-fold-stable picks.
"""
import numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H, m5_xpair as MX
base=list(H.feature_cols("EURUSD"))
FOLDS=[("2024",[str(y) for y in range(2012,2023)],"2023"),
       ("2025",[str(y) for y in range(2012,2024)],"2024"),
       ("2026",[str(y) for y in range(2012,2025)],"2025")]

def build5(years,stride=1):
    MX.HOR=5; MX.GAP_S=300
    F=MX.build_xp(years,stride); return MX.augment(F,years,"xpof")
def build15(years,stride=1):
    MX.HOR=15; MX.GAP_S=900
    F=MX.build_xp(years,stride); return MX.augment(F,years,"xpof")
def lgbfit(X,y,Xv,yv):
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=127,min_child_samples=400,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,n_estimators=2500,n_jobs=20,verbosity=-1)
    L.fit(X,y,eval_set=[(Xv,yv)],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    return L
def boot(c,nb=4000,seed=7):
    c=np.asarray(c,float)
    if len(c)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(c); a=np.array([c[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))

def metacols(F): return [c for c in F.columns if c.startswith("agree") or c.startswith("disp") or c=="comp60" or c.startswith("OF_")]
def Xmeta(F,p15,p5):
    mc=metacols(F)
    return np.column_stack([np.abs(p15-0.5),p15,p5,np.abs(p5-0.5),(np.sign(p15-0.5)==np.sign(p5-0.5)).astype(float),
        F["15m_bb_width"].values.astype("float32"),F["sess_ny"].values.astype("float32")]+[F[c].values.astype("float32") for c in mc])

def run():
    print("[wf_stack] walk-forward cross-horizon stack; target=5-min; per-fold 1yr gap; non-overlap 300s",flush=True)
    fold_data={}
    for ty,tr,va in FOLDS:
        # primary5 + parent15 trained on expanding window (stride5), val=va year
        TR5=build5(tr,5); VA5=build5([va]); TE5=build5([ty])
        xpc=MX.xp_cols(TR5); cols=MX.feat_cols("xpof",TR5,xpc)
        cols=[c for c in cols if c in TR5.columns and c in VA5.columns and c in TE5.columns]
        TR15=build15(tr,5); VA15=build15([va])
        cols15=[c for c in cols if c in TR15.columns]
        P5=lgbfit(TR5[cols].astype("float32"),TR5["_y"].astype(int).values,VA5[cols].astype("float32"),VA5["_y"].astype(int).values)
        P15=lgbfit(TR15[cols15].astype("float32"),TR15["_y"].astype(int).values,VA15[cols15].astype("float32"),VA15["_y"].astype(int).values)
        def apply(F):
            p5=P5.predict_proba(F[cols].astype("float32"))[:,1]; p15=P15.predict_proba(F[cols15].astype("float32"))[:,1]
            return p15,p5
        pv15,pv5=apply(VA5); pt15,pt5=apply(TE5)
        # stack meta trained on val year
        Xv=Xmeta(VA5,pv15,pv5); yv5=VA5["_y"].astype(int).values; dirv=(pv15>0.5).astype(int)
        M=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=15,min_child_samples=600,
            subsample=0.8,colsample_bytree=0.6,reg_lambda=20,n_estimators=400,n_jobs=20,verbosity=-1)
        vny=VA5["sess_ny"].values>0.5; M.fit(Xv[vny],(dirv==yv5).astype(int)[vny])
        sval=M.predict_proba(Xv)[:,1]; Xt=Xmeta(TE5,pt15,pt5); stest=M.predict_proba(Xt)[:,1]
        fold_data[ty]=dict(sval=sval,vny=vny,vy=yv5,vdir=dirv,vts=VA5["_ts"].values.astype("int64"),
            stest=stest,tny=TE5["sess_ny"].values>0.5,ty5=TE5["_y"].astype(int).values,
            tdir=(pt15>0.5).astype(int),tts=TE5["_ts"].values.astype("int64"))
        a=roc_auc_score(TE5["_y"].astype(int).values,pt15)
        print(f"[wf_stack] fold {ty} trained (P15 AUC={a:.4f}); val n={len(yv5)}",flush=True)
    # global threshold via per-fold val quantiles -> use the median val quantile mapping; sweep q
    def sel_test(ty,q):
        d=fold_data[ty]; thr=float(np.quantile(d["sval"][d["vny"]],q))
        m=d["tny"]&(d["stest"]>=thr); sel=MX.nonoverlap_chrono(d["tts"],m,300)
        if len(sel)==0: return (0,float("nan"),np.array([]))
        corr=(d["tdir"][sel]==d["ty5"][sel]).astype(float); return (len(sel),float(corr.mean()),corr)
    print(f"\n{'q':>6}  {'t24':>13} {'t25':>13} {'oos':>13}  {'FLOOR':>6} {'COMB':>20}",flush=True)
    rows=[]
    for q in (0.5,0.7,0.8,0.85,0.9,0.93,0.95,0.97,0.98,0.99):
        res={ty:sel_test(ty,q) for ty,_,_ in FOLDS}
        if any(res[ty][0]<12 for ty,_,_ in FOLDS): continue
        A=np.concatenate([res[ty][2] for ty,_,_ in FOLDS]); lo,hi=boot(A); fl=min(res[ty][1] for ty,_,_ in FOLDS)
        rows.append((q,res,fl,(len(A),A.mean(),lo,hi)))
        print(f"{q:6.2f}  {res['2024'][1]:.3f}(n{res['2024'][0]:>4}) {res['2025'][1]:.3f}(n{res['2025'][0]:>4}) "
              f"{res['2026'][1]:.3f}(n{res['2026'][0]:>4})  {fl:6.3f} {A.mean():.3f}[{lo:.3f},{hi:.3f}]n{len(A)}",flush=True)
    ver=[r for r in rows if r[1]['2026'][0]>=120]
    if ver:
        bv=max(ver,key=lambda r:r[2])
        print(f"[BEST VERIFIABLE oos n>=120] q={bv[0]:.2f} -> t24 {bv[1]['2024'][1]:.3f}/t25 {bv[1]['2025'][1]:.3f}/"
              f"oos {bv[1]['2026'][1]:.3f}(n{bv[1]['2026'][0]}) FLOOR={bv[2]:.3f} COMBINED {bv[3][1]:.3f} CI[{bv[3][2]:.3f},{bv[3][3]:.3f}]",flush=True)
if __name__=="__main__": run()
