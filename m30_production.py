"""30-MINUTE EURUSD binary — PRODUCTION pipeline (train / backtest / live inference).

HONEST result (see m30_research_log.md): 30m EURUSD up/down is ~0.52 AUC unconditional (noise floor, confirmed 6 ways
+ literature). The best HONEST, OOS-verified, tradeable edge is a SELECTIVE ensemble inside the volatility-compression ×
NY-session regime: ~0.62–0.64 independent accuracy across TEST24 / TEST25 / OOS26 — i.e. PROFITABLE vs deriv's payout-
deduction breakeven (~0.541 at a 15% deduction), but NOT 75%. A robust >75% at 30m does not exist in this data; that is
a seconds-scale phenomenon (research_log V25; m30_research_log). 30m IS deriv-tradeable (above the 15m forex floor).

Settlement = deriv Rise/Fall mid-to-mid close-to-close, NO spread, ties LOSE; next-tick entry negligible at 30m.
Reported accuracy = INDEPENDENT non-overlapping (1800s) chronological/live-faithful trades, bootstrap CI95.

Discipline: gate (compression TF) + coverage + conf threshold chosen on VAL 2022-23, frozen, reported on TEST 2024,
TEST 2025, OOS 2026 at that same threshold (corr(VAL,OOS) was -0.54 historically — a result counts only if it holds
across all held-out windows with CI95 excluding breakeven).

Usage:
  python m30_production.py train [PAIR]     # train 2012-2021 ensemble, freeze (comp_tf,cov,thr) on VAL, save models/m30_*, backtest
  python m30_production.py backtest [PAIR]
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import harness as H

MODELS="/media/sean/CORSAIR/binary-algo/models"
HOR=30; GAP_S=HOR*60; PAIR="EURUSD"
base=list(H.feature_cols("EURUSD"))
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],"test24":["2024"],"test25":["2025"],"oos":["2026"]}

def set_pair(p):
    global PAIR; PAIR=p; return p
def art(name): return f"{MODELS}/m30_{PAIR}_{name}"

def load(years, stride=1):
    parts=[]
    for y in years:
        p=f"{H.FEAT_DIR}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        df=pd.read_parquet(p, columns=base+["close"]); df=df[~df.index.duplicated(keep="last")]
        c=df["close"].values; n=len(c); secs=df.index.values.astype("datetime64[s]").astype("int64")
        contig=np.zeros(n,bool)
        if n>HOR: contig[:n-HOR]=(secs[HOR:]-secs[:-HOR])==HOR*60
        fwd=np.full(n,np.nan); fwd[:n-HOR]=c[HOR:]; ret=fwd/c-1.0
        yv=(ret>0).astype(float); valid=contig&np.isfinite(ret)&(ret!=0)
        d=df.loc[valid,base].copy(); d["_y"]=yv[valid]; d["_ts"]=secs[valid]
        parts.append(d.iloc[::stride] if stride>1 else d)
    return pd.concat(parts)

def gate_mask(D, comp_tf, bbw_thr):
    return (D[f"{comp_tf}_bb_width"].values.astype(float)<=bbw_thr)&(D["sess_ny"].values.astype(float)>0.5)

def nonoverlap_chrono(ts, mask, gap=GAP_S):
    take=[]; block_until=-1
    for i in np.where(mask)[0]:
        if ts[i] < block_until: continue
        take.append(i); block_until=int(ts[i])+gap
    return np.array(take,dtype=int)
def boot(corr, nb=5000, seed=7):
    corr=np.asarray(corr,float)
    if len(corr)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(corr)
    a=np.array([corr[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))

def _proba(L,G,C,X):
    return (L.predict(X.values)+G.predict_proba(X)[:,1]+C.predict_proba(X.fillna(-999))[:,1])/3.0

def train():
    t0=time.time(); os.makedirs(MODELS,exist_ok=True)
    TR=load(SPL["train"],3); VA=load(SPL["val"])
    ytr=TR["_y"].astype(int).values; yva=VA["_y"].astype(int).values
    Xtr=TR[base].astype("float32"); Xva=VA[base].astype("float32")
    print(f"[train] train={len(TR):,} val={len(VA):,} load={time.time()-t0:.0f}s",flush=True)
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=255,min_child_samples=300,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=10,n_estimators=3000,n_jobs=20,verbosity=-1)
    L.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",callbacks=[lgb.early_stopping(200),lgb.log_evaluation(0)])
    L.booster_.save_model(art("direction_lgb.txt")); print(f"[train] lgb {time.time()-t0:.0f}s",flush=True)
    G=xgb.XGBClassifier(n_estimators=2000,learning_rate=0.02,max_depth=8,subsample=0.8,colsample_bytree=0.5,
        reg_lambda=10,tree_method="hist",n_jobs=20,eval_metric="auc",early_stopping_rounds=200)
    G.fit(Xtr,ytr,eval_set=[(Xva,yva)],verbose=False); G.save_model(art("direction_xgb.json"))
    print(f"[train] xgb {time.time()-t0:.0f}s",flush=True)
    C=CatBoostClassifier(iterations=2000,learning_rate=0.02,depth=8,l2_leaf_reg=10,eval_metric="AUC",
        thread_count=20,verbose=False,early_stopping_rounds=200)
    C.fit(Xtr.fillna(-999),ytr,eval_set=(Xva.fillna(-999),yva)); C.save_model(art("direction_cat.cbm"))
    print(f"[train] cat {time.time()-t0:.0f}s",flush=True)
    Lb=lgb.Booster(model_file=art("direction_lgb.txt"))
    pva=_proba(Lb,G,C,Xva)
    # select comp_tf in {15m,30m,1h} (bottom-tertile TRAIN bb_width) x NY, coverage on VAL maximizing VAL indep acc
    # Pre-committed FAMILY = volatility-compression x NY-session, MODERATE coverage (n>=150) — the validated 15m mechanism
    # (V23). corr(VAL,OOS)=-0.54 historically, so we deliberately avoid the ultra-low-coverage tail where VAL-max anti-
    # transfers; we select (comp_tf, cov) on VAL only over a small constrained grid, then freeze & report held-out.
    qthr={tf:float(np.nanpercentile(TR[f"{tf}_bb_width"].values.astype(float),33)) for tf in ("15m","30m","1h")}
    tsv=VA["_ts"].values.astype("int64"); best=None
    for tf in ("15m","30m","1h"):
        gv=gate_mask(VA,tf,qthr[tf]); confv=np.abs(pva-0.5)
        if gv.sum()<2000: continue
        for cov in (0.05,0.03):
            thr=float(np.quantile(confv[gv],1-cov)); m=gv&(confv>=thr)
            sel=nonoverlap_chrono(tsv,m)
            if len(sel)<150: continue
            acc=((pva[sel]>0.5).astype(int)==yva[sel]).mean()
            if best is None or acc>best[0]: best=(acc,tf,cov,thr,len(sel))
    accV,TF,COV,THR,nV=best
    params={"feature_names":base,"comp_tf":TF,"bb_width_thr":qthr[TF],"coverage":COV,"conf_thr":THR,"horizon_bars":HOR,
            "pair":PAIR,"gate":f"{TF}_bb_width<=tertile AND sess_ny>0.5","val_indep_acc":float(accV),"val_n":nV,
            "splits":{"train":"2012-2021","val":"2022-2023","test":"2024 & 2025","oos":"2026"},
            "val_auc":float(roc_auc_score(yva,pva)),"settlement":"deriv Rise/Fall mid-to-mid, ties lose, breakeven~0.541"}
    json.dump(params,open(art("strategy.json"),"w"),indent=2)
    print(f"[train] SELECTED comp({TF})xNY cov{COV:.0%} thr={THR:.4f} VAL_indep_acc={accV:.3f} (n{nV}) valAUC={params['val_auc']:.4f}  {time.time()-t0:.0f}s",flush=True)
    backtest()

def _load():
    p=json.load(open(art("strategy.json")))
    L=lgb.Booster(model_file=art("direction_lgb.txt"))
    G=xgb.XGBClassifier(); G.load_model(art("direction_xgb.json"))
    C=CatBoostClassifier(); C.load_model(art("direction_cat.cbm"))
    return p,L,G,C

def backtest():
    p,L,G,C=_load()
    print(f"[backtest] deriv Rise/Fall {HOR}m | gate=comp({p['comp_tf']})xNY cov{p['coverage']:.0%} thr={p['conf_thr']:.4f} | INDEPENDENT non-overlap {GAP_S}s chrono, CI95",flush=True)
    allc=[]
    for w in ("test24","test25","oos"):
        D=load(SPL[w])
        if len(D)==0: continue
        X=D[base].astype("float32"); pr=_proba(L,G,C,X); y=D["_y"].astype(int).values
        ts=D["_ts"].values.astype("int64"); g=gate_mask(D,p["comp_tf"],p["bb_width_thr"])
        m=g&(np.abs(pr-0.5)>=p["conf_thr"]); sel=nonoverlap_chrono(ts,m)
        corr=((pr[sel]>0.5).astype(int)==y[sel]).astype(float) if len(sel) else np.array([])
        acc=corr.mean() if len(sel) else float("nan"); lo,hi=boot(corr)
        ev=acc*0.85-(1-acc) if len(sel) else float("nan")
        print(f"=== {w} === indep_trades={len(sel)} acc={acc:.3f} CI95=[{lo:.3f},{hi:.3f}] EV/bet@payout0.85={ev:+.3f}",flush=True)
        if w in ("test24","test25","oos"): allc.append(corr)
    if allc:
        A=np.concatenate(allc); acc=A.mean(); lo,hi=boot(A)
        print(f"=== HELD-OUT COMBINED (test24+test25+oos) === n={len(A)} acc={acc:.3f} CI95=[{lo:.3f},{hi:.3f}]",flush=True)
        print(f"    deriv payout-deduction EV: "+"  ".join(f"R{po:.2f}->be{1/(1+po):.3f}:EV{acc*po-(1-acc):+.3f}" for po in (0.80,0.85,0.90)),flush=True)

class Min30Strategy:
    def __init__(self, pair="EURUSD", models_dir=MODELS):
        global MODELS; MODELS=models_dir; set_pair(pair); self.p,self.L,self.G,self.C=_load()
    def signal(self, feature_row):
        X=feature_row.to_frame().T if isinstance(feature_row,pd.Series) else feature_row
        X=X[self.p["feature_names"]].astype("float32"); pr=float(_proba(self.L,self.G,self.C,X)[0]); conf=abs(pr-0.5)
        tf=self.p["comp_tf"]
        in_regime=(float(X[f"{tf}_bb_width"].iloc[0])<=self.p["bb_width_thr"]) and (float(X["sess_ny"].iloc[0])>0.5)
        return {"trade":bool(in_regime and conf>=self.p["conf_thr"]),"direction":int(np.sign(pr-0.5)) or 1,
                "confidence":conf,"p_up":pr,"in_regime":in_regime}

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "backtest"
    if len(sys.argv)>2: set_pair(sys.argv[2])
    {"train":train,"backtest":backtest}.get(mode,backtest)()
