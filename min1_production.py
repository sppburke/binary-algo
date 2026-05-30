"""1-MINUTE EURUSD binary — PRODUCTION pipeline (train / backtest / live inference).

Strategy = compression-release selective book on the 60s up/down binary (see README "The 1-minute strategy").
Honest expectation: a selective ~0.66 book (large-sample 2024/2025), profitable vs typical binary payouts,
that abstains most of the time and fires in bursts. The OOS-2026 0.872 is a thin, April-concentrated sample.

Usage:
  python min1_production.py train      # train on 2021-23, freeze params on VAL, serialize models/min1_*, report
  python min1_production.py backtest   # load artifacts, replay TEST 2024-25 + OOS 2026 (trades/acc/EV by period)

Live: from min1_production import Min1Strategy
      s = Min1Strategy()                       # loads models/min1_*
      out = s.signal(buffer_df)                # buffer = >=3700 recent 1s bars [mid,imb,micro,spread,nt,tsz]
      # out = {"trade": bool, "direction": +1/-1, "confidence": float, "in_regime": bool}

Artifacts written to models/:
  min1_direction_lgb.txt · min1_direction_xgb.json · min1_direction_cat.cbm   (direction ensemble)
  min1_magnitude.joblib                                                       (P(|ret60| large), AUC ~0.68)
  min1_strategy.json   (feature_names, bbw1800_q33, rel_ratio_p90, conf_thr, horizon_s, gap_s, train info)
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
import joblib
from sklearn.metrics import roc_auc_score

TICK = "/media/sean/CORSAIR/binary-algo/features_tick"
MODELS = "/media/sean/CORSAIR/binary-algo/models"
HS, GAP = 60, 60                 # 60s horizon, 60s non-overlap
COMP_PC, REL_PC, COV = 33, 90, 0.10
TRSTRIDE = 5
SPLIT_YEARS = {"train": "2021-2023", "val": "2024-H1", "test": "2024.09-2025.11", "oos": "2026"}

# ----------------------------- features (53, causal) -----------------------------
def feats(b):
    mid=b["mid"]; imb=b["imb"].fillna(0); micro=b["micro"]; r1=mid.pct_change()
    X=pd.DataFrame(index=b.index)
    X["imb"]=imb
    for w in (3,5,10,20,40,80): X[f"imb_ema{w}"]=imb.ewm(span=w).mean()
    X["imb_acc"]=imb.ewm(span=5).mean()-imb.ewm(span=40).mean(); X["imb_chg"]=imb.diff(3)
    sgn=np.sign(imb)
    X["imb_sgn_ac30"]=(sgn*sgn.shift(1)).rolling(30).mean()
    X["imb_sameside30"]=(sgn==sgn.shift(1)).rolling(30).mean()
    rl=sgn.groupby((sgn!=sgn.shift()).cumsum()).cumcount()+1; X["imb_runlen"]=(rl*sgn).clip(-50,50)
    md=(micro-mid)/mid; X["micro_dev"]=md
    for w in (5,15,30,60): X[f"micro_dev_ema{w}"]=md.ewm(span=w).mean()
    X["micro_mom15"]=micro/micro.shift(15)-1; X["micro_mom60"]=micro/micro.shift(60)-1
    X["spread"]=b["spread"]; X["spread_ema30"]=b["spread"].ewm(span=30).mean()
    X["nt"]=b["nt"]; X["nt_ema30"]=b["nt"].ewm(span=30).mean()
    X["tsz"]=b["tsz"]; X["tsz_ema30"]=b["tsz"].ewm(span=30).mean()
    for w in (5,15,30,60,120,300,600,1800,3600): X[f"ret{w}"]=mid.pct_change(w)
    for w in (30,60,300,900,1800): X[f"rv{w}"]=r1.rolling(w).std()
    for w in (60,300,900,1800,3600): X[f"emadist{w}"]=mid/mid.ewm(span=w).mean()-1
    for w in (60,300,900):
        sd=r1.rolling(w).std()*np.sqrt(w); X[f"stretch{w}"]=(mid/mid.ewm(span=w).mean()-1)/(sd+1e-9)
    for w in (300,900,1800):
        hi=mid.rolling(w).max(); lo=mid.rolling(w).min(); X[f"rangepos{w}"]=(mid-lo)/(hi-lo+1e-12)
    for w in (300,900,1800): X[f"bbw{w}"]=(r1.rolling(w).std()*np.sqrt(w))
    X["rel_ratio"]=X["bbw300"]/(X["bbw1800"]+1e-12)
    hh=b.index.hour; X["hsin"]=np.sin(2*np.pi*hh/24); X["hcos"]=np.cos(2*np.pi*hh/24)
    X["dow"]=b.index.dayofweek.astype("float32")
    return X.replace([np.inf,-np.inf],np.nan).astype("float32")

def load_split(sp):
    return pd.read_parquet(f"{TICK}/{sp}_1s.parquet")

def prep(b):
    """features + 60s mid label + valid mask + ts + regime cols."""
    X=feats(b); mid=b["mid"]
    ret=(mid.shift(-HS)/mid-1).values
    valid=np.isfinite(ret)&(ret!=0)
    ts=b.index.values.astype("datetime64[s]").astype("int64")
    return X, (ret>0).astype(int), np.abs(ret), valid, ts, b.index

def nonoverlap(ts, conf, thr, gap=GAP):
    """greedy highest-confidence non-overlapping selection; returns sorted indices of trades."""
    sel=np.where(conf>=thr)[0]
    if len(sel)==0: return np.array([],dtype=int)
    order=sel[np.argsort(-conf[sel])]
    if len(order)>200000: order=order[:200000]
    tmin=int(ts[order].min()); span=int(ts[order].max()-tmin)+gap+2
    blk=np.zeros(span,bool); take=[]
    for i in order:
        t=int(ts[i]-tmin)
        if blk[t]: continue
        take.append(i); blk[max(0,t-gap+1):t+gap]=True
    return np.sort(np.array(take))

# ----------------------------- train -----------------------------
def mk_lgb(n=4000): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=350,
    min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=8,n_estimators=n,n_jobs=20,verbosity=-1)

def train():
    t0=time.time(); os.makedirs(MODELS,exist_ok=True)
    btr,bva=load_split("train"),load_split("val")
    Xtr_,ytr_,mtr_,vtr,_,_=prep(btr); Xva_,yva_,mva_,vva,_,_=prep(bva)
    feat_names=list(Xtr_.columns)
    itr=np.where(vtr)[0][::TRSTRIDE]; iva=np.where(vva)[0]
    Xtr=Xtr_.iloc[itr]; ytr=ytr_[itr]; Xva=Xva_.iloc[iva]; yva=yva_[iva]
    print(f"[train] features+prep {time.time()-t0:.0f}s; dir-train n={len(ytr):,} val n={len(yva):,}",flush=True)
    # direction ensemble
    L=mk_lgb(); L.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",callbacks=[lgb.early_stopping(120),lgb.log_evaluation(0)])
    L.booster_.save_model(f"{MODELS}/min1_direction_lgb.txt"); print(f"[train] lgb done {time.time()-t0:.0f}s",flush=True)
    G=xgb.XGBClassifier(n_estimators=2500,learning_rate=0.02,max_depth=9,subsample=0.8,colsample_bytree=0.6,
        reg_lambda=8,tree_method="hist",n_jobs=20,eval_metric="auc",early_stopping_rounds=120)
    G.fit(Xtr,ytr,eval_set=[(Xva,yva)],verbose=False); G.save_model(f"{MODELS}/min1_direction_xgb.json")
    print(f"[train] xgb done {time.time()-t0:.0f}s",flush=True)
    C=CatBoostClassifier(iterations=2500,learning_rate=0.02,depth=9,l2_leaf_reg=8,eval_metric="AUC",
        thread_count=20,verbose=False,early_stopping_rounds=120)
    C.fit(Xtr.fillna(-999),ytr,eval_set=(Xva.fillna(-999),yva)); C.save_model(f"{MODELS}/min1_direction_cat.cbm")
    print(f"[train] cat done {time.time()-t0:.0f}s",flush=True)
    # magnitude model (|ret60| top-tercile)
    magthr=np.nanpercentile(mtr_[itr],67); ymag=(mtr_[itr]>=magthr).astype(int)
    M=mk_lgb(2500); M.fit(Xtr,ymag,eval_set=[(Xva,(mva_[iva]>=magthr).astype(int))],eval_metric="auc",
        callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    joblib.dump(M,f"{MODELS}/min1_magnitude.joblib"); print(f"[train] magnitude done {time.time()-t0:.0f}s",flush=True)
    # direction val probs -> freeze regime + confidence thresholds
    pv=(L.predict_proba(Xva)[:,1]+G.predict_proba(Xva)[:,1]+C.predict_proba(Xva.fillna(-999))[:,1])/3.0
    bbw=Xva["bbw1800"].values; rel=Xva["rel_ratio"].values
    qb=float(np.nanpercentile(bbw,COMP_PC)); rq=float(np.nanpercentile(rel[bbw<=qb],REL_PC))
    gate=(bbw<=qb)&(rel>=rq); thr=float(np.quantile(np.abs(pv[gate]-0.5),1-COV))
    params={"feature_names":feat_names,"bbw1800_q33":qb,"rel_ratio_p90":rq,"conf_thr":thr,
            "horizon_s":HS,"gap_s":GAP,"comp_pc":COMP_PC,"rel_pc":REL_PC,"cov":COV,
            "mag_top_tercile_thr":float(magthr),"splits":SPLIT_YEARS,"val_auc":float(roc_auc_score(yva,pv))}
    json.dump(params,open(f"{MODELS}/min1_strategy.json","w"),indent=2)
    print(f"[train] saved models/min1_* | gate bbw1800<={qb:.2e} rel>={rq:.3f} conf_thr={thr:.4f} valAUC={params['val_auc']:.4f}",flush=True)
    print(f"[train] DONE {time.time()-t0:.0f}s\n"); backtest()

# ----------------------------- load + backtest -----------------------------
def _load():
    p=json.load(open(f"{MODELS}/min1_strategy.json"))
    L=lgb.Booster(model_file=f"{MODELS}/min1_direction_lgb.txt")
    G=xgb.XGBClassifier(); G.load_model(f"{MODELS}/min1_direction_xgb.json")
    C=CatBoostClassifier(); C.load_model(f"{MODELS}/min1_direction_cat.cbm")
    M=joblib.load(f"{MODELS}/min1_magnitude.joblib")
    return p,L,G,C,M

def _dirproba(p,L,G,C,X):
    Xo=X[p["feature_names"]]
    return (L.predict(Xo.values)+G.predict_proba(Xo)[:,1]+C.predict_proba(Xo.fillna(-999))[:,1])/3.0

def backtest():
    p,L,G,C,M=_load()
    for sp,label in (("test","TEST 2024-25"),("oos","OOS 2026")):
        b=load_split(sp); X,y,mag,valid,ts,idx=prep(b)
        pr=_dirproba(p,L,G,C,X)
        bbw=X["bbw1800"].values; rel=X["rel_ratio"].values
        gate=valid&(bbw<=p["bbw1800_q33"])&(rel>=p["rel_ratio_p90"])
        conf=np.abs(pr-0.5); cand=gate&(conf>=p["conf_thr"])
        tr=nonoverlap(ts, np.where(cand,conf,-1.0), 0.0)   # only cand bars (conf>=thr) eligible
        tr=tr[cand[tr]]
        correct=((pr[tr]>0.5).astype(int)==y[tr])
        acc=correct.mean() if len(tr) else float("nan")
        print(f"\n=== {label} ({SPLIT_YEARS[sp]}) === trades={len(tr)} accuracy={acc:.3f}")
        mo=idx[tr].to_period("M").astype(str)
        for m in sorted(set(mo)):
            k=mo==m; print(f"    {m}: trades={int(k.sum()):>4}  acc={correct[k.values].mean():.3f}")
        for payout in (0.80,):
            be=1/(1+payout); ev=acc*payout-(1-acc) if len(tr) else float("nan")
            print(f"    payout {payout:.2f}: breakeven={be:.3f}  EV/bet={ev:+.3f}")

# ----------------------------- live inference -----------------------------
class Min1Strategy:
    def __init__(self, models_dir=MODELS):
        global MODELS; MODELS=models_dir
        self.p,self.L,self.G,self.C,self.M=_load()
    def signal(self, buffer_df):
        """buffer_df: >=3700 consecutive 1s bars with cols [mid,imb,micro,spread,nt,tsz], tz-aware index.
        Returns dict for the LAST bar. trade=True only if in compression-release regime AND confident."""
        X=feats(buffer_df).iloc[[-1]]
        pr=float(_dirproba(self.p,self.L,self.G,self.C,X)[0])
        bbw=float(X["bbw1800"].iloc[0]); rel=float(X["rel_ratio"].iloc[0]); conf=abs(pr-0.5)
        in_regime=(bbw<=self.p["bbw1800_q33"]) and (rel>=self.p["rel_ratio_p90"])
        trade=bool(in_regime and conf>=self.p["conf_thr"])
        return {"trade":trade,"direction":int(np.sign(pr-0.5)) or 1,"confidence":conf,
                "p_up":pr,"in_regime":in_regime,"mag_p_large":float(self.M.predict_proba(X[self.p["feature_names"]])[:,1][0])}

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "backtest"
    {"train":train,"backtest":backtest}.get(mode, backtest)()
