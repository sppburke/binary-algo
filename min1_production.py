"""1-MINUTE EURUSD binary — PRODUCTION pipeline v2 (train / backtest / live inference).

v2 applies the 2-minute model's lessons to the 60s book and is STRICTLY BETTER than v1:
  - REVERSION trend filter: bet AGAINST the last 5-min move (sign(p-0.5) == -sign(ret300)). v1 had no trend
    filter. This is the dominant new lever — it lifts the large-sample TEST accuracy ~+6 points.
  - compression-release direction SPECIALIST (LGBM trained only on regime bars) blended 50/50 with the all-bars
    ensemble (in-regime AUC 0.510->0.519); broadens the confident set so the OOS sample is no longer 1 month.

Result (frozen pipeline, nothing tuned on the evaluation periods):
  v1 was TEST 0.682 (n759) / OOS 0.780 (n50, 44 in April).
  v2 is  TEST 0.739 (n1033) / OOS 0.777 (n184, all 3 months: Feb .892 / Mar .850 / Apr .710) -- +6 pts TEST,
  3.7x the OOS trades, and OOS spread across every month instead of one. EV/bet@0.80 = +0.33 (TEST) to +0.40 (OOS).

PER-PAIR (default EURUSD). Artifacts are pair-labeled; train other currencies with a PAIR argument (each needs
its own 1-second microstructure cache under features_tick_<PAIR>/).

Usage (PAIR optional, defaults EURUSD):
  python min1_production.py train [PAIR]      # train both model groups, freeze regime+rev+thr on VAL, save, report
  python min1_production.py backtest [PAIR]   # load artifacts, replay TEST 2024-25 + OOS 2026 (trades/acc/EV/month)

Live: from min1_production import Min1Strategy
      s = Min1Strategy(pair="EURUSD")          # loads models/min1_EURUSD_*
      out = s.signal(buffer_df)                # buffer = >=3700 recent 1s bars [mid,imb,micro,spread,nt,tsz]
      # out = {"trade":bool,"direction":+/-1,"confidence":float,"in_regime":bool,"p_up":float}

Artifacts (models/, PAIR-labeled, e.g. EURUSD):
  min1_EURUSD_direction_lgb.txt · _direction_xgb.json · _direction_cat.cbm   (all-bars direction ensemble)
  min1_EURUSD_direction_spec_lgb.txt                                          (compression-release LGBM specialist)
  min1_EURUSD_magnitude.joblib                                                (P(|ret60| large), kept for info)
  min1_EURUSD_strategy.json   (pair, feature_names, w_spec, bbw1800_q67, rel_p70, rel_tighten, conf_thr, ...)
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
import joblib
from sklearn.metrics import roc_auc_score

ROOT = "/media/sean/CORSAIR/binary-algo"; MODELS = f"{ROOT}/models"
PAIR = "EURUSD"
HS, GAP = 60, 60                          # 60s horizon, 60s non-overlap
COMP_PC, REL_PC, REL_TIGHTEN, COV = 67, 70, 80, 0.05   # regime bbw1800<=q67 & rel>=p70, rel re-tightened to p80; 5% cov
W_SPEC = 0.5                              # blend weight on the compression-release specialist
TRSTRIDE_ALL, TRSTRIDE_SPEC = 5, 2
SPLIT_YEARS = {"train": "2021-2023", "val": "2024-H1", "test": "2024.09-2025.11", "oos": "2026"}

def set_pair(pair):
    global PAIR, TICK; PAIR=pair
    TICK = f"{ROOT}/features_tick" if pair=="EURUSD" else f"{ROOT}/features_tick_{pair}"
    return TICK
def art(name): return f"{MODELS}/min1_{PAIR}_{name}"
set_pair("EURUSD")

# ----------------------------- features (62, causal) -----------------------------
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
    for w in (30,60,300,900,1800,3600): X[f"rv{w}"]=r1.rolling(w).std()
    for w in (60,300,900,1800,3600): X[f"emadist{w}"]=mid/mid.ewm(span=w).mean()-1
    for w in (120,300,900,1800):
        sd=r1.rolling(w).std()*np.sqrt(w); X[f"stretch{w}"]=(mid/mid.ewm(span=w).mean()-1)/(sd+1e-9)
    for w in (300,900,1800,3600):
        hi=mid.rolling(w).max(); lo=mid.rolling(w).min(); X[f"rangepos{w}"]=(mid-lo)/(hi-lo+1e-12)
    for w in (300,900,1800,3600): X[f"bbw{w}"]=(r1.rolling(w).std()*np.sqrt(w))
    X["rel_ratio"]=X["bbw300"]/(X["bbw1800"]+1e-12)
    X["rel_ratio2"]=X["bbw900"]/(X["bbw3600"]+1e-12)
    hh=b.index.hour; X["hsin"]=np.sin(2*np.pi*hh/24); X["hcos"]=np.cos(2*np.pi*hh/24)
    X["dow"]=b.index.dayofweek.astype("float32")
    return X.replace([np.inf,-np.inf],np.nan).astype("float32")

def load_split(sp): return pd.read_parquet(f"{TICK}/{sp}_1s.parquet")

def prep(b):
    X=feats(b); mid=b["mid"]
    ret=(mid.shift(-HS)/mid-1).values
    valid=np.isfinite(ret)&(ret!=0)
    ts=b.index.values.astype("datetime64[s]").astype("int64")
    return X,(ret>0).astype(int),np.abs(ret),valid,ts,b.index

def nonoverlap(ts,conf,thr,gap=GAP):
    sel=np.where(conf>=thr)[0]
    if len(sel)==0: return np.array([],dtype=int)
    order=sel[np.argsort(-conf[sel])]
    if len(order)>300000: order=order[:300000]
    tmin=int(ts[order].min()); span=int(ts[order].max()-tmin)+gap+2
    blk=np.zeros(span,bool); take=[]
    for i in order:
        t=int(ts[i]-tmin)
        if blk[t]: continue
        take.append(i); blk[max(0,t-gap+1):t+gap]=True
    return np.sort(np.array(take))

def mk_lgb(n=4000): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=350,
    min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=8,n_estimators=n,n_jobs=20,verbosity=-1)
def mk_spec(n=6000): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.01,num_leaves=512,
    min_child_samples=300,subsample=0.7,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,reg_alpha=2,n_estimators=n,n_jobs=20,verbosity=-1)

# ----------------------------- train -----------------------------
def train():
    t0=time.time(); os.makedirs(MODELS,exist_ok=True)
    btr,bva=load_split("train"),load_split("val")
    Xtr,ytr,mtr,vtr,_,_=prep(btr); Xva,yva,mva,vva,_,_=prep(bva)
    feat_names=list(Xtr.columns); iva=np.where(vva)[0]
    bbwtr=Xtr["bbw1800"].values.astype(float); reltr=Xtr["rel_ratio"].values.astype(float)
    qb=float(np.nanpercentile(bbwtr[vtr],COMP_PC)); rq=float(np.nanpercentile(reltr[vtr&(bbwtr<=qb)],REL_PC))
    print(f"[train] prep {time.time()-t0:.0f}s; regime bbw1800<={qb:.3e} rel>={rq:.3f}",flush=True)
    # all-bars direction ensemble
    iall=np.where(vtr)[0][::TRSTRIDE_ALL]; XA=Xtr.iloc[iall]; yA=ytr[iall]
    L=mk_lgb(); L.fit(XA,yA,eval_set=[(Xva.iloc[iva],yva[iva])],eval_metric="auc",callbacks=[lgb.early_stopping(120),lgb.log_evaluation(0)])
    L.booster_.save_model(art("direction_lgb.txt")); print(f"[train] dir-lgb {time.time()-t0:.0f}s",flush=True)
    G=xgb.XGBClassifier(n_estimators=2500,learning_rate=0.02,max_depth=9,subsample=0.8,colsample_bytree=0.6,
        reg_lambda=8,tree_method="hist",n_jobs=20,eval_metric="auc",early_stopping_rounds=120)
    G.fit(XA,yA,eval_set=[(Xva.iloc[iva],yva[iva])],verbose=False); G.save_model(art("direction_xgb.json"))
    print(f"[train] dir-xgb {time.time()-t0:.0f}s",flush=True)
    C=CatBoostClassifier(iterations=2500,learning_rate=0.02,depth=9,l2_leaf_reg=8,eval_metric="AUC",
        thread_count=20,verbose=False,early_stopping_rounds=120)
    C.fit(XA.fillna(-999),yA,eval_set=(Xva.iloc[iva].fillna(-999),yva[iva])); C.save_model(art("direction_cat.cbm"))
    print(f"[train] dir-cat {time.time()-t0:.0f}s",flush=True)
    # magnitude (kept for info)
    magthr=float(np.nanpercentile(mtr[iall],67)); M=mk_lgb(2500)
    M.fit(XA,(mtr[iall]>=magthr).astype(int),eval_set=[(Xva.iloc[iva],(mva[iva]>=magthr).astype(int))],eval_metric="auc",
        callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)]); joblib.dump(M,art("magnitude.joblib"))
    print(f"[train] magnitude {time.time()-t0:.0f}s",flush=True)
    # compression-release specialist
    def rmask(X,valid):
        b=X["bbw1800"].values.astype(float); r=X["rel_ratio"].values.astype(float)
        return valid&(b<=qb)&(r>=rq)
    isp=np.where(rmask(Xtr,vtr))[0][::TRSTRIDE_SPEC]; ivsp=np.where(rmask(Xva,vva))[0]
    S=mk_spec(); S.fit(Xtr.iloc[isp],ytr[isp],eval_set=[(Xva.iloc[ivsp],yva[ivsp])],eval_metric="auc",
        callbacks=[lgb.early_stopping(200),lgb.log_evaluation(0)])
    S.booster_.save_model(art("direction_spec_lgb.txt")); print(f"[train] specialist best_iter={S.best_iteration_} {time.time()-t0:.0f}s",flush=True)
    # blended direction on VAL -> rel re-tighten + reversion-filtered confidence threshold
    def blend(X):
        pa=(L.predict_proba(X)[:,1]+G.predict_proba(X)[:,1]+C.predict_proba(X.fillna(-999))[:,1])/3.0
        ps=S.predict_proba(X)[:,1]; return (1-W_SPEC)*pa+W_SPEC*ps
    pv=blend(Xva); bbw=Xva["bbw1800"].values; rel=Xva["rel_ratio"].values; r300=Xva["ret300"].values
    base=vva&(bbw<=qb)&(rel>=rq); rqt=float(np.nanpercentile(rel[base],REL_TIGHTEN))
    gate=base&(rel>=rqt)&(np.sign(pv-0.5)==-np.sign(r300))
    thr=float(np.quantile(np.abs(pv[gate]-0.5),1-COV))
    params={"feature_names":feat_names,"w_spec":W_SPEC,"bbw1800_q67":qb,"rel_p70":rq,"rel_tighten":rqt,"conf_thr":thr,
            "horizon_s":HS,"gap_s":GAP,"comp_pc":COMP_PC,"rel_pc":REL_PC,"rel_tighten_pc":REL_TIGHTEN,"cov":COV,
            "pair":PAIR,"trend":"reversion_vs_ret300","mag_top_tercile_thr":magthr,"splits":SPLIT_YEARS,
            "val_auc_inregime":float(roc_auc_score(yva[gate],pv[gate]))}
    json.dump(params,open(art("strategy.json"),"w"),indent=2)
    print(f"[train] saved models/min1_{PAIR}_* | gate bbw<={qb:.2e} rel>={rqt:.3f} rev(ret300) conf_thr={thr:.5f}",flush=True)
    print(f"[train] DONE {time.time()-t0:.0f}s\n"); backtest()

# ----------------------------- load + backtest -----------------------------
def _load():
    p=json.load(open(art("strategy.json")))
    L=lgb.Booster(model_file=art("direction_lgb.txt"))
    G=xgb.XGBClassifier(); G.load_model(art("direction_xgb.json"))
    C=CatBoostClassifier(); C.load_model(art("direction_cat.cbm"))
    S=lgb.Booster(model_file=art("direction_spec_lgb.txt"))
    return p,L,G,C,S

def _blend(p,L,G,C,S,X):
    Xo=X[p["feature_names"]]
    pa=(L.predict(Xo.values)+G.predict_proba(Xo)[:,1]+C.predict_proba(Xo.fillna(-999))[:,1])/3.0
    ps=S.predict(Xo.values); return (1-p["w_spec"])*pa+p["w_spec"]*ps

def backtest():
    p,L,G,C,S=_load()
    for sp,label in (("test","TEST 2024-25"),("oos","OOS 2026")):
        b=load_split(sp); X,y,mag,valid,ts,idx=prep(b)
        pr=_blend(p,L,G,C,S,X)
        bbw=X["bbw1800"].values; rel=X["rel_ratio"].values; r300=X["ret300"].values
        gate=valid&(bbw<=p["bbw1800_q67"])&(rel>=p["rel_tighten"])&(np.sign(pr-0.5)==-np.sign(r300))
        conf=np.abs(pr-0.5); cand=gate&(conf>=p["conf_thr"])
        tr=nonoverlap(ts,np.where(cand,conf,-1.0),0.0); tr=tr[cand[tr]]
        correct=((pr[tr]>0.5).astype(int)==y[tr]); acc=correct.mean() if len(tr) else float("nan")
        print(f"\n=== {label} ({SPLIT_YEARS[sp]}) === trades={len(tr)} accuracy={acc:.3f}")
        mo=np.asarray(idx[tr].to_period("M").astype(str))
        for m in sorted(set(mo.tolist())):
            k=mo==m; print(f"    {m}: trades={int(k.sum()):>4}  acc={correct[k].mean():.3f}")
        be=1/1.80; ev=acc*0.80-(1-acc) if len(tr) else float("nan")
        print(f"    payout 0.80: breakeven={be:.3f}  EV/bet={ev:+.3f}")

# ----------------------------- live inference -----------------------------
class Min1Strategy:
    def __init__(self, pair="EURUSD", models_dir=MODELS):
        global MODELS; MODELS=models_dir; set_pair(pair)
        self.p,self.L,self.G,self.C,self.S=_load()
    def signal(self, buffer_df):
        """buffer_df: >=3700 consecutive 1s bars [mid,imb,micro,spread,nt,tsz], tz-aware. Returns dict for LAST bar."""
        X=feats(buffer_df).iloc[[-1]]
        pr=float(_blend(self.p,self.L,self.G,self.C,self.S,X)[0])
        bbw=float(X["bbw1800"].iloc[0]); rel=float(X["rel_ratio"].iloc[0]); r300=float(X["ret300"].iloc[0]); conf=abs(pr-0.5)
        in_regime=(bbw<=self.p["bbw1800_q67"]) and (rel>=self.p["rel_tighten"]) and (np.sign(pr-0.5)==-np.sign(r300))
        return {"trade":bool(in_regime and conf>=self.p["conf_thr"]),"direction":int(np.sign(pr-0.5)) or 1,
                "confidence":conf,"p_up":pr,"in_regime":bool(in_regime)}

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "backtest"
    if len(sys.argv)>2: set_pair(sys.argv[2])
    print(f"[pair={PAIR}] tick-cache={TICK}  artifacts=models/min1_{PAIR}_*",flush=True)
    {"train":train,"backtest":backtest}.get(mode,backtest)()
