"""1-MINUTE EURUSD binary — PRODUCTION pipeline v3 (train / backtest / live inference).

Target = a deriv.com Rise/Fall 60-SECOND binary (verified vs deriv T&C 2.2.1.3 / 2.2.3.1 / 2.3.1 + tick API
schemas): entry spot = the NEXT mid-tick after the order; exit spot = the last mid-tick at/before +60s; "Rise"
wins iff exit > entry STRICTLY (ties LOSE). Settlement is MID-to-MID with NO spread — deriv's edge is the payout
deduction (~15% -> payout R~1.85, breakeven win-rate ~0.541), modeled by the backtest's payout-sensitivity table.

v3 METHODOLOGY FIX (2026-05, post bias-audit) — read research_log "BIAS AUDIT" entry:
  The 1-second bars drop empty seconds (tick1s_cache.py), so the old label `mid.shift(-60)` shifted 60
  *bars*, not 60 *seconds*. On gap-dropped data that spanned a MEDIAN of ~111 wall-clock seconds (p90 233s),
  i.e. a variable, ~2x-longer-than-advertised horizon that does not correspond to any tradeable fixed-expiry
  binary, and broke trade independence (the non-overlap gap was 60s while outcome windows ran ~111s, so the
  "independent" trades overlapped and the bootstrap CIs were too narrow / n inflated ~30-40%).
  FIX: the label is now a strict WALL-CLOCK lookup (price at t+60s, matched within TOL_S, window must not
  cross a data gap), and non-overlap blocks HS+TOL_S seconds so selected trades are genuinely independent.
  The edge is robust to the fix (it survives strict independence — see backtest CIs); what changed is that
  the numbers now describe a REAL 60s binary, with honest independence-correct confidence intervals.

v2 levers (unchanged): REVERSION trend filter (bet AGAINST the last 5-min move, sign(p-0.5)==-sign(ret300));
  compression-release direction SPECIALIST (LGBM trained only on regime bars) blended 50/50 with the all-bars
  ensemble. The backtest reports bootstrap CI95, per-month accuracy, payout sensitivity, and a +1s entry-latency
  robustness row. Numbers are re-verified by `python min1_production.py train` (see research_log for current figures).

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

ROOT = "/home/sean/git/binary-algo"; MODELS = f"{ROOT}/models"
PAIR = "EURUSD"
HS = 60                                   # 60-SECOND fixed wall-clock expiry (a real binary option expiry)
TOL_S = 10                                # settlement/entry matched within 10s of the exact instant, else the
                                          # [entry, expiry] window crosses a data gap and the row is dropped
GAP = HS + TOL_S                          # non-overlap block (s): guarantees selected trades' expiry windows are disjoint
ENTRY_LAG_S = 1                           # deriv enters on the NEXT tick AFTER the order (T&C 2.2.3.1) -> 1s entry lag
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

def wc_ret(ts, mid, horizon_s=HS, tol_s=TOL_S, lag_s=ENTRY_LAG_S):
    """deriv.com Rise/Fall settlement, modeled faithfully (verified vs deriv T&C + API tick schemas):
      * spot = bid/ask MID, <=1 tick/sec; our 1s bars use the last tick/sec -> matches deriv's feed. NO spread
        crosses the win/loss decision (T&C 2.2.1.3); the broker edge is the payout deduction (2.3.1), modeled
        downstream via the payout-sensitivity EV table.
      * ENTRY spot = the NEXT tick after the order is processed (T&C 2.2.3.1): first tick at/after decision+lag_s.
      * EXIT  spot = the LATEST tick at or before the end time:                last tick with ts <= entry_ts+horizon_s.
      * 'Rise' wins iff exit > entry STRICTLY; a tie (ret==0) LOSES (we keep ties as valid losing trades).
    VALID iff the entry and exit ticks exist within tol_s of their target instants and there are >=2 ticks in
    the window (deriv refunds <2-tick / gap-straddling windows). Returns (ret, valid)."""
    ts=np.asarray(ts); mid=np.asarray(mid,dtype=float); n=len(ts)
    entry_t=ts+lag_s;            ei=np.searchsorted(ts, entry_t, side="left")        # first tick at/after order+lag
    exit_t=entry_t+horizon_s;    xi=np.searchsorted(ts, exit_t, side="right")-1      # last tick at/before expiry
    eic=np.clip(ei,0,n-1); xic=np.clip(xi,0,n-1)
    valid=(ei<n)&(xi>ei)                                                             # entry exists; >=2 distinct ticks
    valid&=(ts[eic]-entry_t)<=tol_s                                                  # entry tick within tol after order
    valid&=(exit_t-ts[xic])<=tol_s                                                   # exit tick within tol before expiry
    ret=mid[xic]/mid[eic]-1.0
    valid&=np.isfinite(ret)                                                          # ret==0 stays valid: a losing tie
    return ret, valid

def prep(b, lag_s=ENTRY_LAG_S):
    X=feats(b); mid=b["mid"].values.astype(float)
    ts=b.index.values.astype("datetime64[s]").astype("int64")
    ret,valid=wc_ret(ts,mid,HS,TOL_S,lag_s)
    return X,(ret>0).astype(int),np.abs(ret),valid,ts,b.index

def boot(corr, nb=5000, seed=7):
    """Bootstrap CI95 of the win-rate over INDEPENDENT (non-overlapping) trades."""
    corr=np.asarray(corr,dtype=float)
    if len(corr)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(corr)
    a=np.array([corr[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))

def nonoverlap(ts,conf,thr,gap=GAP):
    """GREEDY-by-confidence de-overlap. Kept only as an OPTIMISTIC reference: it peeks ahead within each
    overlap cluster to keep the most-confident bar, which a live trader cannot do. Use nonoverlap_chrono
    for the headline tradeable number."""
    sel=np.where(conf>=thr)[0]
    if len(sel)==0: return np.array([],dtype=int)
    order=sel[np.argsort(-conf[sel])]
    if len(order)>2_000_000: order=order[:2_000_000]
    tmin=int(ts[order].min()); span=int(ts[order].max()-tmin)+gap+2
    blk=np.zeros(span,bool); take=[]
    for i in order:
        t=int(ts[i]-tmin)
        if blk[t]: continue
        take.append(i); blk[max(0,t-gap+1):t+gap]=True
    return np.sort(np.array(take))

def nonoverlap_chrono(ts, mask, gap=GAP):
    """LIVE-FAITHFUL first-come de-overlap (NO look-ahead): scan bars in time order; take each signaled bar,
    then block the next `gap` seconds. This is the honest tradeable policy — at the first signal you cannot
    know a more-confident bar is coming, so (unlike greedy-by-confidence) we never peek ahead to pick the best
    bar in an overlap cluster. This is the PRIMARY accuracy reported by the backtest."""
    take=[]; block_until=-1
    for i in np.where(mask)[0]:
        if ts[i] < block_until: continue
        take.append(i); block_until=int(ts[i])+gap
    return np.array(take,dtype=int)

def mk_lgb(n=4000): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=350,
    min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=8,n_estimators=n,n_jobs=20,verbosity=-1)
def mk_spec(n=6000): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.01,num_leaves=512,
    min_child_samples=300,subsample=0.7,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,reg_alpha=2,n_estimators=n,n_jobs=20,verbosity=-1)

# ----------------------------- train -----------------------------
def train():
    t0=time.time(); os.makedirs(MODELS,exist_ok=True)
    btr,bva=load_split("train"),load_split("val")
    Xtr,ytr,mtr,vtr,_,_=prep(btr); Xva,yva,mva,vva,_,_=prep(bva)
    feat_names=list(Xtr.columns); iva=np.where(vva&(mva>0))[0]   # drop exact ties from training/early-stop
    bbwtr=Xtr["bbw1800"].values.astype(float); reltr=Xtr["rel_ratio"].values.astype(float)
    qb=float(np.nanpercentile(bbwtr[vtr],COMP_PC)); rq=float(np.nanpercentile(reltr[vtr&(bbwtr<=qb)],REL_PC))
    print(f"[train] prep {time.time()-t0:.0f}s; regime bbw1800<={qb:.3e} rel>={rq:.3f}",flush=True)
    # all-bars direction ensemble
    iall=np.where(vtr&(mtr>0))[0][::TRSTRIDE_ALL]; XA=Xtr.iloc[iall]; yA=ytr[iall]
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
    isp=np.where(rmask(Xtr,vtr)&(mtr>0))[0][::TRSTRIDE_SPEC]; ivsp=np.where(rmask(Xva,vva)&(mva>0))[0]
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
    print(f"[backtest] deriv Rise/Fall {HS}s | entry=next tick (lag {ENTRY_LAG_S}s), exit=last tick<=expiry, mid-to-mid, ties LOSE | gap={GAP}s | de-overlap=chronological (no look-ahead)")
    for sp,label in (("test","TEST 2024-25"),("oos","OOS 2026")):
        b=load_split(sp); X,y,mag,valid,ts,idx=prep(b)   # prep uses the deriv next-tick entry lag ENTRY_LAG_S
        mid=b["mid"].values.astype(float)
        pr=_blend(p,L,G,C,S,X); pred=(pr>0.5).astype(int)
        bbw=X["bbw1800"].values; rel=X["rel_ratio"].values; r300=X["ret300"].values
        gate=valid&(bbw<=p["bbw1800_q67"])&(rel>=p["rel_tighten"])&(np.sign(pr-0.5)==-np.sign(r300))
        conf=np.abs(pr-0.5); cand=gate&(conf>=p["conf_thr"])
        # PRIMARY: chronological first-come de-overlap (live policy); ties (mag==0) are LOSING trades (deriv strict)
        tr=nonoverlap_chrono(ts,cand)
        correct=((pred[tr]==y[tr])&(mag[tr]>0)).astype(float); acc=correct.mean() if len(tr) else float("nan")
        lo,hi=boot(correct); ngate=int(gate.sum()); cov=len(tr)/ngate if ngate else float("nan")
        print(f"\n=== {label} ({SPLIT_YEARS[sp]}) === independent_trades={len(tr)} accuracy={acc:.3f} CI95=[{lo:.3f},{hi:.3f}]"
              f"  (in-regime bars={ngate}; realized coverage={cov:.2%})")
        mo=np.asarray(idx[tr].to_period("M").astype(str))
        for m in sorted(set(mo.tolist())):
            k=mo==m; print(f"    {m}: trades={int(k.sum()):>4}  acc={correct[k].mean():.3f}")
        # deriv edge = payout deduction (T&C 2.3.1), NOT a spread; win returns stake*(1+payout). breakeven p*=1/(1+payout)
        print(f"    deriv payout-deduction EV (0.85 ≈ the ~15% deduction; R floats with vol/duration):")
        for po in (0.80,0.85,0.90):
            print(f"        payout {po:.2f}: breakeven={1/(1+po):.3f}  EV/bet={acc*po-(1-acc):+.3f}")
        # GREEDY-by-confidence reference (OPTIMISTIC: peeks ahead within overlap clusters — not live-achievable)
        trg=nonoverlap(ts,np.where(cand,conf,-1.0),0.0); trg=trg[cand[trg]]
        accg=((pred[trg]==y[trg])&(mag[trg]>0)).mean() if len(trg) else float("nan")
        print(f"    [ref] greedy-by-confidence de-overlap (optimistic upper bound): n={len(trg)} acc={accg:.3f}")
        # IDEAL reference: same-tick entry (NOT achievable on deriv, which enters the NEXT tick) — shows the entry-lag cost
        r0,v0=wc_ret(ts,mid,HS,TOL_S,lag_s=0); y0=(r0>0).astype(int); m0=np.abs(r0)
        t0=tr[v0[tr]]; acc0=((pred[t0]==y0[t0])&(m0[t0]>0)).mean() if len(t0) else float("nan")
        print(f"    [ref] same-tick entry (ideal, not achievable on deriv): n={len(t0)} acc={acc0:.3f}")

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
