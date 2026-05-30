"""15-MINUTE EURUSD binary — PRODUCTION pipeline (train / backtest / live inference).

Best 15m strategy (research_log V27): EURUSD up/down 15m binary, ensemble on the 239 multi-timeframe
features (pipeline.py), bet ONLY in the volatility-compression x NY-session regime, selective by confidence.
Honest result: pre-committed pipeline = 0.642 on held-out 2024-2026 (CI [0.624,0.659]); profitable vs an
0.80 binary payout (breakeven 0.556) but NOT >=75% — the 15m frontier is ~0.64.

Usage:
  python m15_production.py train      # train on 2012-2021, freeze (depth,coverage,threshold) on VAL, save models/m15_*, report
  python m15_production.py backtest   # load artifacts, replay 2024 / 2025 / 2026 (trades/acc/EV per year)

Live: from m15_production import Min15Strategy ; s=Min15Strategy() ; s.signal(feature_row)
  feature_row = a pandas Series/1-row DataFrame of the 239 pipeline.py features for the current 15m decision bar
  (compute with pipeline.py). Returns {"trade":bool,"direction":+/-1,"confidence":float,"in_regime":bool}.

Artifacts (models/, labeled m15_*):
  m15_direction_lgb.txt · m15_direction_xgb.json · m15_direction_cat.cbm   (direction ensemble)
  m15_strategy.json   (feature_names, comp_q (selected), bb_width_thr, coverage, conf_thr, horizon=15 bars)
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import harness as H

MODELS = "/media/sean/CORSAIR/binary-algo/models"
HOR = 15; STRIDE = 3
base = list(H.feature_cols("EURUSD"))

def load(years, stride=1):
    parts=[]
    for y in years:
        p=f"{H.FEAT_DIR}/EURUSD_{y}.parquet"
        if not os.path.exists(p): continue
        df=pd.read_parquet(p, columns=base+H.META_COLS); df=df[~df.index.duplicated(keep="last")]
        idx=df.index; c=df["close"].values; n=len(c)
        secs=idx.values.astype("datetime64[s]").astype("int64")
        contig=np.zeros(n,bool)
        if n>HOR: contig[:n-HOR]=(secs[HOR:]-secs[:-HOR])==HOR*60
        fwd=np.full(n,np.nan); fwd[:n-HOR]=c[HOR:]; ret=fwd/c-1.0
        yv=(ret>0).astype(float); valid=contig&np.isfinite(ret)&(ret!=0)
        d=df.loc[valid,base].copy(); d["_y"]=yv[valid]
        parts.append(d.iloc[::stride] if stride>1 else d)
    return pd.concat(parts)

def gate_mask(D, bbw_thr):
    return (D["15m_bb_width"].values.astype(float)<=bbw_thr)&(D["sess_ny"].values.astype(float)>0.5)

def mk_lgb(): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=255,
    min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=10,n_estimators=3000,n_jobs=20,verbosity=-1)

def train():
    t0=time.time(); os.makedirs(MODELS,exist_ok=True)
    TR=load(H.SPLITS["train"],STRIDE); VA=load(H.SPLITS["val"])
    ytr=TR["_y"].astype(int).values; yva=VA["_y"].astype(int).values
    print(f"[train] load {time.time()-t0:.0f}s; train={len(TR):,} val={len(VA):,}",flush=True)
    L=mk_lgb(); L.fit(TR[base].astype("float32"),ytr,eval_set=[(VA[base].astype("float32"),yva)],
        eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    L.booster_.save_model(f"{MODELS}/m15_direction_lgb.txt"); print(f"[train] lgb done {time.time()-t0:.0f}s",flush=True)
    G=xgb.XGBClassifier(n_estimators=2000,learning_rate=0.02,max_depth=8,subsample=0.8,colsample_bytree=0.5,
        reg_lambda=10,tree_method="hist",n_jobs=20,eval_metric="auc",early_stopping_rounds=150)
    G.fit(TR[base].astype("float32"),ytr,eval_set=[(VA[base].astype("float32"),yva)],verbose=False)
    G.save_model(f"{MODELS}/m15_direction_xgb.json"); print(f"[train] xgb done {time.time()-t0:.0f}s",flush=True)
    C=CatBoostClassifier(iterations=2000,learning_rate=0.02,depth=8,l2_leaf_reg=10,eval_metric="AUC",
        thread_count=20,verbose=False,early_stopping_rounds=150)
    C.fit(TR[base].fillna(-999),ytr,eval_set=(VA[base].fillna(-999),yva))
    C.save_model(f"{MODELS}/m15_direction_cat.cbm"); print(f"[train] cat done {time.time()-t0:.0f}s",flush=True)
    def prob(D):
        X=D[base].astype("float32"); return (L.predict_proba(X)[:,1]+G.predict_proba(X)[:,1]+C.predict_proba(D[base].fillna(-999))[:,1])/3.0
    pva=prob(VA)
    Q={q:float(np.nanpercentile(TR["15m_bb_width"].values.astype(float),q)) for q in (10,20,33)}
    # select (depth q, coverage) on VAL maximizing VAL accuracy (n>=150) — V27 protocol
    best=None
    for q in (10,20,33):
        gv=gate_mask(VA,Q[q]); confv=np.abs(pva[gv]-0.5)
        if gv.sum()<300: continue
        for cov in (0.10,0.05,0.02):
            thr=float(np.quantile(confv,1-cov)); m=gv&(np.abs(pva-0.5)>=thr)
            if m.sum()<150: continue
            acc=((pva[m]>0.5).astype(int)==yva[m]).mean()
            if best is None or acc>best[0]: best=(acc,q,cov,thr,int(m.sum()))
    accV,Q_SEL,COV_SEL,THR,nV=best
    params={"feature_names":base,"comp_q":Q_SEL,"bb_width_thr":Q[Q_SEL],"coverage":COV_SEL,"conf_thr":THR,
            "horizon_bars":HOR,"gate":"15m_bb_width<=q AND sess_ny>0.5","val_acc":float(accV),"val_n":nV,
            "splits":{"train":"2012-2021","val":"2022-2023","test":"2024-2025","oos":"2026"},"val_auc":float(roc_auc_score(yva,pva))}
    json.dump(params,open(f"{MODELS}/m15_strategy.json","w"),indent=2)
    print(f"[train] saved models/m15_* | SELECTED comp(q{Q_SEL})xNY @cov{COV_SEL:.0%} bb<={Q[Q_SEL]:.2e} conf_thr={THR:.4f} VALacc={accV:.3f} valAUC={params['val_auc']:.4f}",flush=True)
    print(f"[train] DONE {time.time()-t0:.0f}s\n"); backtest()

def _load():
    p=json.load(open(f"{MODELS}/m15_strategy.json"))
    L=lgb.Booster(model_file=f"{MODELS}/m15_direction_lgb.txt")
    G=xgb.XGBClassifier(); G.load_model(f"{MODELS}/m15_direction_xgb.json")
    C=CatBoostClassifier(); C.load_model(f"{MODELS}/m15_direction_cat.cbm")
    return p,L,G,C

def _dirproba(p,L,G,C,X):
    Xo=X[p["feature_names"]]
    return (L.predict(Xo.values)+G.predict_proba(Xo)[:,1]+C.predict_proba(Xo.fillna(-999))[:,1])/3.0

def backtest():
    p,L,G,C=_load()
    allc=[]; alln=0
    for yrs,label in (([("2024")],"2024"),([("2025")],"2025"),([("2026")],"2026")):
        D=load(yrs)
        if len(D)==0: continue
        pr=_dirproba(p,L,G,C,D); y=D["_y"].astype(int).values
        g=gate_mask(D,p["bb_width_thr"]); m=g&(np.abs(pr-0.5)>=p["conf_thr"])
        corr=((pr[m]>0.5).astype(int)==y[m]); acc=corr.mean() if m.sum() else float("nan")
        ev=acc*0.80-(1-acc) if m.sum() else float("nan")
        print(f"=== {label} === trades(selective,overlapping)={int(m.sum())} accuracy={acc:.3f} EV/bet@0.80={ev:+.3f}",flush=True)
        allc.append(corr); alln+=int(m.sum())
    C_=np.concatenate(allc); acc=C_.mean()
    print(f"=== 2024-2026 COMBINED === trades={alln} accuracy={acc:.3f} EV/bet@0.80={acc*0.8-(1-acc):+.3f}  (V27 held-out: ~0.642)",flush=True)

class Min15Strategy:
    def __init__(self, models_dir=MODELS):
        global MODELS; MODELS=models_dir
        self.p,self.L,self.G,self.C=_load()
    def signal(self, feature_row):
        """feature_row: 1-row DataFrame (or Series) of the 239 pipeline.py features for the current 15m bar."""
        X=feature_row.to_frame().T if isinstance(feature_row,pd.Series) else feature_row
        X=X.astype("float32")
        pr=float(_dirproba(self.p,self.L,self.G,self.C,X)[0]); conf=abs(pr-0.5)
        in_regime=(float(X["15m_bb_width"].iloc[0])<=self.p["bb_width_thr"]) and (float(X["sess_ny"].iloc[0])>0.5)
        return {"trade":bool(in_regime and conf>=self.p["conf_thr"]),"direction":int(np.sign(pr-0.5)) or 1,
                "confidence":conf,"p_up":pr,"in_regime":in_regime}

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "backtest"
    {"train":train,"backtest":backtest}.get(mode, backtest)()
