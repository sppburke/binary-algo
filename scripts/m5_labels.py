"""LABEL-ENGINEERING BATCH (path-dependent / redefined TRAIN labels) on the 5m cross-pair primary — UP+DOWN.

GOAL: does REDEFINING the TRAINING LABEL (not reweighting it) unlock a 5m direction edge? The IDENTICAL certified
cross-pair feature matrix + primary hyperparams are reused; ONLY the TRAIN label set changes. Eval ALWAYS uses the
TRUE unsmoothed 300s mid-to-mid sign (`_y`); label redefinition is TRAIN-only.

Four alternative TRAIN-label variants on the same anchors:
  (1) triple_barrier_k{0.5,1.0}: symmetric +/- k*sigma_t barriers on the forward mid PATH within 300s;
      label = sign(first touch); if no touch, label = sign(300s return). (1m-bar PROXY path — see PATH NOTE.)
  (2) trend_scan: label = sign(OLS slope of forward mid over 60..300s); sample_weight = |t-stat| of the slope.
  (3) jump_filtered: simple jump test on the 300s window (max |1m return| > J*sigma_t => jump-dominated);
      DROP jump-dominated bars from TRAIN, label the remainder by 300s sign. (1m-bar PROXY path.)
  (4) deadband3: 3-class vol-adaptive deadband — DROP |ret| < deadband_t (deadband_t prop sigma_t) bars from TRAIN,
      train 3-class {down,flat,up}, direction = sign(p_up - p_down).

For each variant: train UP and DOWN heads with the SAME LGBM hyperparams as m5_lossbatch.py's focal/asym models
(n_estimators=1200, lr=0.03, num_leaves=127, min_child=300, subsample=0.8, colsample=0.5, reg_lambda=20, n_jobs=20).
The "UP/DOWN head" distinction is the EVAL side mask (per_year sv=1/0), identical to m5_lossbatch — one model per
variant scored on both sides (deadband uses its own 3-class score; deadband_down = same model, eval side mask flips).

PATH NOTE (1s alignment resolution): native 1s mids live in features_tick/{train,val,test,oos}_1s.parquet but train_1s
only covers 2021-02-28+ while the 5m TRAIN anchors span years 2012-2021 (XP.SPL["train"]). 1s is therefore INFEASIBLE
for ~9/10 TRAIN years. We FALL BACK to a 1m-bar PROXY path from features/EURUSD_<year>.parquet `close` (1-min spaced,
datetime_utc index; XP frame index-epoch == _ts EXACTLY, 100% anchor + +300s coverage verified). Triple-barrier and
jump labels use 5 forward 1m bars (t+60..t+300). sigma_t = 5m_atr_pct (fractional vol; fallback 1m_rv_12*sqrt(5)).

EVAL (EXACTLY like m5_lossbatch.per_year): COV=0.05, NY gate, side mask, conf=|s-0.5|, cthr=quantile(conf[g],1-COV),
nonoverlap_chrono(ts,mask,300), per-year 2024/2025/2026, boot() CI95, win=(y==side). DERIV-FAITHFUL: breakeven 0.541,
ties LOSE (true `_y` already drops fwd==0). Reports moved-bar up-rate per side/year, tripwire [0.47,0.53].
SELECTION on WORST-VAL-half (k swept per variant on VAL min-of-two-time-halves), NEVER VAL-acc-max.

PRE-REGISTERED FALSIFIER (written to m5_labels_result.json at start): KILL the label-engineering family unless some
variant yields UP binding-2025 (min per-year, TRUE-300s-sign, ties-LOSE, worst-VAL-half-selected) win-rate CI95-lo
>= 0.541 AND beats incumbent UP .577 by > 1 SE; OR DOWN binding-2025 CI95-lo >= 0.541 AND beats magweight DOWN p10
.5441 by > 1 SE. If the best variant's 2025 up-rate not in [0.47,0.53] or it only wins on a smoothed label -> subsumed.

  ~/binary-algo-venv/bin/python m5_labels.py
"""
import os, sys; sys.argv=["x"]
import json, time, numpy as np, pandas as pd
import lightgbm as lgb
import m5_xpair as MX, m5_xpair_production as XP

ROOT="/home/sean/git/binary-algo"; FEAT=MX.FEAT
BE=0.541; STRIDE=6; COV=0.05
INCUMBENT_UP=0.577        # incumbent UP binding-2025 headline (refit floor .553); beat by >1SE
MAGWEIGHT_DOWN=0.5441     # magweight POW=0.5 DOWN cov0.05 p10; beat by >1SE
# triple-barrier multipliers, jump multiplier, deadband multiplier, trend min-window (in 1m bars)
KS=(0.5,1.0); JMULT=3.0; DB_MULT=0.5
NBARS=5                   # 5 forward 1-min bars span the 300s horizon (t+60..t+300)
RESULT=f"{ROOT}/m5_labels_result.json"
LGB_KW=dict(n_estimators=1200,learning_rate=0.03,num_leaves=127,min_child_samples=300,subsample=0.8,
            subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,n_jobs=20,verbosity=-1)

def boot(c,nb=2500,seed=7):
    c=np.asarray(c,float)
    if len(c)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(c); a=np.array([c[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))

def se_win(win,n):
    """1 SE of a binomial win-rate (used for the >1SE beat test)."""
    if not n or n<=0 or win is None: return float("nan")
    return float(np.sqrt(max(win*(1-win),1e-9)/n))

# ---------------------------------------------------------------------------
#  per_year — IDENTICAL discipline to m5_lossbatch.per_year (eval ALWAYS on true `_y`)
#  scorefn returns a [0,1] direction-confidence score; sv=1 (UP) / sv=0 (DOWN) side mask.
#  Also reports moved-bar up-rate per year for the [0.47,0.53] tripwire.
# ---------------------------------------------------------------------------
def per_year(EV,scorefn,sv):
    res={}
    for w,D in EV.items():
        s=scorefn(D); y=D["_y"].astype(int).values; ts=D["_ts"].values.astype("int64"); ny=D["sess_ny"].values>0.5
        conf=np.abs(s-0.5); g=ny&((s>0.5) if sv==1 else (s<0.5))
        if g.sum()<20: continue
        cthr=np.quantile(conf[g],1-COV); m=g&(conf>=cthr); sel=MX.nonoverlap_chrono(ts,m,300)
        if len(sel)>=20:
            cc=(y[sel]==sv).astype(float); lo,hi=boot(cc)
            Y=2024 if w=="test24" else 2025 if w=="test25" else 2026
            uprate=float(y[sel].mean())   # up-rate of the MOVED (selected) bars -> tripwire
            res[Y]=dict(win=round(float(cc.mean()),4),n=len(sel),ci=[round(lo,4),round(hi,4)],
                        uprate=round(uprate,4),uprate_ok=bool(0.47<=uprate<=0.53))
    b=[res[Y]["win"] for Y in res]; lo=[res[Y]["ci"][0] for Y in res]; ns=[res[Y]["n"] for Y in res]
    res["binding"]=dict(win=min(b) if b else None,ci_lo=min(lo) if lo else None,min_n=min(ns) if ns else 0,
                        uprate_2025=(res.get(2025,{}) or {}).get("uprate"),
                        uprate_2025_ok=(res.get(2025,{}) or {}).get("uprate_ok"))
    return res

# ---------------------------------------------------------------------------
#  WORST-VAL-HALF k selection (NEVER VAL-acc-max). For a family of candidate scorefns keyed by k,
#  pick the k whose VAL min-over-two-time-halves selective acc (NY, side, cov0.05, non-overlap) is largest.
#  VAL split into two chronological halves by median _ts.
# ---------------------------------------------------------------------------
def val_worsthalf_acc(VA,scorefn,sv):
    s=scorefn(VA); y=VA["_y"].astype(int).values; ts=VA["_ts"].values.astype("int64"); ny=VA["sess_ny"].values>0.5
    conf=np.abs(s-0.5); g=ny&((s>0.5) if sv==1 else (s<0.5))
    if g.sum()<40: return float("nan"),0
    cthr=np.quantile(conf[g],1-COV); m=g&(conf>=cthr); sel=MX.nonoverlap_chrono(ts,m,300)
    if len(sel)<40: return float("nan"),len(sel)
    cut=np.median(ts[sel]); h1=sel[ts[sel]<=cut]; h2=sel[ts[sel]>cut]
    accs=[]
    for h in (h1,h2):
        if len(h)<15: return float("nan"),len(sel)
        accs.append(float((y[h]==sv).mean()))
    return float(min(accs)),len(sel)

def select_k(VA,cand,sv):
    """cand: dict k->scorefn. Return (best_k, best_scorefn, worsthalf_acc, n)."""
    best=None
    for k,fn in cand.items():
        a,n=val_worsthalf_acc(VA,fn,sv)
        if np.isnan(a) or n<40: continue
        if best is None or a>best[2]: best=(k,fn,a,n)
    return best  # may be None

# ---------------------------------------------------------------------------
#  1m-bar PROXY forward path. Returns, per anchor _ts: a (Nanchors x NBARS) matrix of forward 1m
#  log-returns r_i = log(close[t+i*60]) - log(close[t]), i=1..NBARS (NaN where bar missing),
#  plus per-bar sigma_t (fractional vol from 5m_atr_pct, fallback 1m_rv_12*sqrt(5)).
# ---------------------------------------------------------------------------
def build_path_proxy(F, years):
    ts=F["_ts"].values.astype("int64")
    # assemble a single epoch->close map across the requested years
    cmap_parts=[]
    for y in years:
        fp=f"{FEAT}/EURUSD_{y}.parquet"
        if not os.path.exists(fp): continue
        c=pd.read_parquet(fp,columns=["close"]); c=c[~c.index.duplicated(keep="last")]
        ce=c.index.values.astype("datetime64[s]").astype("int64")
        cmap_parts.append(pd.Series(np.log(c["close"].values.astype("float64")),index=ce))
    if not cmap_parts:
        R=np.full((len(F),NBARS),np.nan); return R
    logc=pd.concat(cmap_parts); logc=logc[~logc.index.duplicated(keep="last")]
    base=logc.reindex(ts).values                       # log mid at the anchor
    R=np.empty((len(F),NBARS),dtype="float64")
    for i in range(1,NBARS+1):
        R[:,i-1]=logc.reindex(ts+60*i).values - base   # cumulative log-return to +i minutes
    return R

def sigma_t(F):
    """Per-bar fractional vol for barrier/jump/deadband scaling. 5m_atr_pct primary, 1m_rv_12*sqrt(5) fallback."""
    if "5m_atr_pct" in F.columns:
        s=F["5m_atr_pct"].values.astype("float64")
    elif "1m_rv_12" in F.columns:
        s=F["1m_rv_12"].values.astype("float64")*np.sqrt(5.0)
    else:
        # last-resort: empirical std of _fwd (constant); keeps the script runnable
        s=np.full(len(F),float(np.nanstd(F["_fwd"].values))+1e-9)
    s=np.where(np.isfinite(s)&(s>0),s,np.nan)
    med=np.nanmedian(s); s=np.where(np.isfinite(s),s,med)   # impute missing with median
    return np.maximum(s,1e-6)

# ---------------------------------------------------------------------------
#  TRAIN-label builders. Each returns (label_array_or_None_per_row, sample_weight_or_None, keep_mask).
#  keep_mask drops bars the variant excludes from TRAIN. Eval is unaffected (true `_y`).
# ---------------------------------------------------------------------------
def lab_triple_barrier(F, R, sig, k):
    """First-touch of +/- k*sigma on the 1m path within 300s; else sign(300s return)."""
    n=len(F); fwd=F["_fwd"].values.astype("float64")
    lab=np.where(fwd>0,1,0).astype("int8")             # default = sign(300s return)
    up=(k*sig); dn=-(k*sig)
    touched=np.zeros(n,dtype=bool)
    for i in range(NBARS):
        r=R[:,i]
        hit_up=(~touched)&np.isfinite(r)&(r>=up)
        hit_dn=(~touched)&np.isfinite(r)&(r<=dn)
        lab[hit_up]=1; lab[hit_dn]=0
        touched|=(hit_up|hit_dn)
    keep=np.ones(n,dtype=bool)                          # triple-barrier keeps ALL bars
    return lab,None,keep

def lab_trend_scan(F, R, sig):
    """sign(OLS slope of forward mid over 60..300s); weight=|t-stat|. Anchor at i=0 (return 0)."""
    n=len(F)
    # design: x = [0,1,..,NBARS] minutes (include the anchor 0 point); y = [0, R[:,0..NBARS-1]]
    xs=np.arange(0,NBARS+1,dtype="float64")
    xbar=xs.mean(); sxx=np.sum((xs-xbar)**2)
    Y=np.column_stack([np.zeros(n),R])                 # (n, NBARS+1), anchor return 0 prepended
    # per-row OLS slope & t-stat with NaN-robust handling (require >=3 finite path points)
    slope=np.full(n,np.nan); tstat=np.full(n,np.nan)
    finite=np.isfinite(Y)
    nf=finite.sum(axis=1)
    Yf=np.where(finite,Y,0.0)
    # use full xs for rows with all points finite (the common case); rows with gaps -> fallback to sign(_fwd)
    full=nf==(NBARS+1)
    if full.any():
        Yc=Y[full]                                     # (m, NBARS+1)
        ybar=Yc.mean(axis=1,keepdims=True)
        sxy=np.sum((xs-xbar)[None,:]*(Yc-ybar),axis=1)
        b=sxy/sxx
        yhat=ybar+b[:,None]*(xs-xbar)[None,:]
        sse=np.sum((Yc-yhat)**2,axis=1)
        dof=(NBARS+1)-2
        se=np.sqrt(np.maximum(sse/max(dof,1),1e-18)/sxx)
        slope[full]=b; tstat[full]=b/np.maximum(se,1e-12)
    fwd=F["_fwd"].values.astype("float64")
    lab=np.where(np.isfinite(slope),(slope>0).astype("int8"),(fwd>0).astype("int8"))
    w=np.where(np.isfinite(tstat),np.abs(tstat),1.0).astype("float64")
    w=np.clip(w,0.0,50.0)                              # cap runaway t-stats
    keep=np.ones(n,dtype=bool)
    return lab,w,keep

def lab_jump_filtered(F, R, sig):
    """Drop jump-dominated bars (max per-minute |return| > J*sigma) from TRAIN; label remainder by sign(300s)."""
    n=len(F); fwd=F["_fwd"].values.astype("float64")
    lab=np.where(fwd>0,1,0).astype("int8")
    # per-minute increments delta_j = R[:,j]-R[:,j-1] (R[:, -1]:=0 at anchor)
    Rp=np.column_stack([np.zeros(n),R])                # (n, NBARS+1)
    dlt=np.diff(Rp,axis=1)                             # (n, NBARS) per-minute log returns
    maxabs=np.nanmax(np.abs(dlt),axis=1)
    jump=np.isfinite(maxabs)&(maxabs>(JMULT*sig))
    keep=~jump                                         # DROP jumps from TRAIN
    return lab,None,keep

def lab_deadband3(F, sig):
    """3-class {0=down,1=flat,2=up}: DROP |ret|<deadband (deadband=DB_MULT*sigma) from TRAIN as 'flat',
    train remaining as 2-class up/down? NO — spec: train 3-class, drop the dead-zone from TRAIN, dir=sign(p_up-p_down).
    We keep ALL bars and assign 3 classes; 'flat' = |ret|<deadband (these are the dead-zone bars).
    Direction score at eval = p_up - p_down mapped to [0,1]."""
    fwd=F["_fwd"].values.astype("float64")
    db=DB_MULT*sig
    cls=np.where(np.abs(fwd)<db,1,np.where(fwd>0,2,0)).astype("int8")   # 0 down / 1 flat / 2 up
    keep=np.ones(len(F),dtype=bool)
    return cls,None,keep

# ---------------------------------------------------------------------------
def sigmoid(z): return 1.0/(1.0+np.exp(-z))

def fit_binary(X,y,w=None):
    m=lgb.LGBMClassifier(objective="binary",**LGB_KW)
    m.fit(X,y,sample_weight=w)
    return m

def fit_multiclass(X,y,w=None):
    m=lgb.LGBMClassifier(objective="multiclass",num_class=3,**LGB_KW)
    m.fit(X,y,sample_weight=w)
    return m

def main():
    t0=time.time()
    # ---- PRE-REGISTER falsifier stub BEFORE any training ----
    stub={"test":"label-engineering batch (triple-barrier / trend-scan / jump-filter / 3-class deadband) "
                 "on the 5m cross-pair primary — TRAIN-label redefinition only, eval on TRUE 300s sign",
          "breakeven":BE,"cov":COV,"stride":STRIDE,
          "path_resolution":("1m-bar PROXY from features/EURUSD_<year>.parquet close (native 1s INFEASIBLE for "
                             "TRAIN years 2012-2020; train_1s.parquet starts 2021-02-28). XP index-epoch==_ts verified; "
                             "5 forward 1m bars span 300s; sigma_t=5m_atr_pct (fallback 1m_rv_12*sqrt5)."),
          "incumbent_UP_2025":INCUMBENT_UP,"magweight_DOWN_p10":MAGWEIGHT_DOWN,
          "PREREGISTERED_FALSIFIER":("KILL the label-engineering family unless some variant yields UP binding-2025 "
            "(min per-year, TRUE-300s-sign, ties-LOSE, worst-VAL-half-selected) win-rate CI95-lo >= 0.541 AND beats "
            "incumbent UP .577 by >1 SE; OR DOWN binding-2025 CI95-lo >= 0.541 AND beats magweight DOWN p10 .5441 by "
            ">1 SE. If best variant's 2025 up-rate not in [0.47,0.53] or only wins on a smoothed label -> subsumed."),
          "status":"RUNNING","variants":{}}
    json.dump(stub,open(RESULT,"w"),indent=2,default=str)
    print(f"[labels] pre-registered falsifier -> {RESULT}",flush=True)

    # ---- identical certified feature matrix + primary feats ----
    p=json.load(open(XP.art("strategy.json"))); cols=p["primary_feats"]
    TR=MX.augment(MX.build_xp(XP.SPL["train"],STRIDE),XP.SPL["train"],XP.MODE)
    VA=MX.augment(MX.build_xp(XP.SPL["val"]),XP.SPL["val"],XP.MODE)
    EV={w:MX.augment(MX.build_xp(XP.SPL[w]),XP.SPL[w],XP.MODE) for w in ("test24","test25","oos")}
    Xtr=TR[cols].astype("float32")
    print(f"[labels] train={len(TR):,} val={len(VA):,} feats={len(cols)} build={time.time()-t0:.0f}s",flush=True)

    # ---- forward 1m PROXY path + sigma for TRAIN (eval windows never need a redefined label) ----
    Rtr=build_path_proxy(TR,XP.SPL["train"]); sigtr=sigma_t(TR)
    cov_path=float(np.isfinite(Rtr[:,-1]).mean())
    print(f"[labels] TRAIN 1m-proxy path coverage(+300s)={cov_path:.3f} sigma_med={np.nanmedian(sigtr):.2e} {time.time()-t0:.0f}s",flush=True)

    out=stub; out["train_path_coverage_300s"]=round(cov_path,4); out["variants"]={}

    # =====================================================================
    #  Build (label,weight,keep) per variant, fit UP+DOWN heads, eval per_year.
    #  k-selection (where a variant has a knob, i.e. triple-barrier) is on WORST-VAL-HALF.
    # =====================================================================

    # ---- (1) TRIPLE BARRIER: k swept, selected per side on worst-VAL-half ----
    # Pre-fit one binary model per k once (label is side-agnostic; UP/DOWN differ only in eval side mask).
    tb_models={}
    for k in KS:
        lab,w,keep=lab_triple_barrier(TR,Rtr,sigtr,k)
        m=fit_binary(Xtr[keep],lab[keep],None)
        tb_models[k]=m
        print(f"  [tb k{k}] fit on {int(keep.sum()):,} bars (lab up-frac {lab[keep].mean():.3f}) {time.time()-t0:.0f}s",flush=True)
    def tb_score(k): return lambda D,m=tb_models[k]: m.predict_proba(D[cols].astype("float32"))[:,1]
    cand={k:tb_score(k) for k in KS}
    for side,sv in (("UP",1),("DOWN",0)):
        sel=select_k(VA,cand,sv)
        if sel is None:
            out["variants"].setdefault("triple_barrier",{})[side]={"binding":{"win":None,"ci_lo":None,"min_n":0},"note":"no valid k on VAL"}
            print(f"  [tb {side}] no valid k on VAL",flush=True); continue
        k,fn,wha,nval=sel
        r=per_year(EV,fn,sv); r["selected_k"]=k; r["val_worsthalf"]=round(wha,4); r["val_n"]=nval
        out["variants"].setdefault("triple_barrier",{})[side]=r
        print(f"  [tb {side}] selected k{k} (VAL worst-half {wha:.3f} n{nval}) binding={r['binding']}",flush=True)
    json.dump(out,open(RESULT,"w"),indent=2,default=str)

    # ---- (2) TREND SCAN (slope sign, weight=|t-stat|) ----
    lab,w,keep=lab_trend_scan(TR,Rtr,sigtr)
    m=fit_binary(Xtr[keep],lab[keep],w[keep] if w is not None else None)
    ts_fn=lambda D,m=m: m.predict_proba(D[cols].astype("float32"))[:,1]
    print(f"  [trend] fit on {int(keep.sum()):,} bars (lab up-frac {lab[keep].mean():.3f}, w_mean {w[keep].mean():.2f}) {time.time()-t0:.0f}s",flush=True)
    out["variants"]["trend_scan"]={"UP":per_year(EV,ts_fn,1),"DOWN":per_year(EV,ts_fn,0)}
    print(f"  [trend] UP={out['variants']['trend_scan']['UP']['binding']} DOWN={out['variants']['trend_scan']['DOWN']['binding']}",flush=True)
    json.dump(out,open(RESULT,"w"),indent=2,default=str)

    # ---- (3) JUMP-FILTERED diffusive-only (drop jumps from TRAIN) ----
    lab,w,keep=lab_jump_filtered(TR,Rtr,sigtr)
    dropped=int((~keep).sum())
    m=fit_binary(Xtr[keep],lab[keep],None)
    jf_fn=lambda D,m=m: m.predict_proba(D[cols].astype("float32"))[:,1]
    print(f"  [jump] dropped {dropped:,}/{len(TR):,} ({dropped/len(TR):.1%}) jump bars; fit on {int(keep.sum()):,} {time.time()-t0:.0f}s",flush=True)
    out["variants"]["jump_filtered"]={"UP":per_year(EV,jf_fn,1),"DOWN":per_year(EV,jf_fn,0),"dropped_frac":round(dropped/len(TR),4)}
    print(f"  [jump] UP={out['variants']['jump_filtered']['UP']['binding']} DOWN={out['variants']['jump_filtered']['DOWN']['binding']}",flush=True)
    json.dump(out,open(RESULT,"w"),indent=2,default=str)

    # ---- (4) 3-CLASS VOL-ADAPTIVE DEADBAND (dir = p_up - p_down) ----
    cls,w,keep=lab_deadband3(TR,sigtr)
    flat_frac=float((cls==1).mean())
    m=fit_multiclass(Xtr[keep],cls[keep],None)
    # 3-class score -> direction confidence in [0,1]: 0.5 + 0.5*(p_up - p_down)
    def db_fn(D,m=m):
        pp=m.predict_proba(D[cols].astype("float32"))   # cols: [0=down,1=flat,2=up]
        return 0.5+0.5*(pp[:,2]-pp[:,0])
    print(f"  [deadband] flat-class frac {flat_frac:.3f}; 3-class fit on {int(keep.sum()):,} {time.time()-t0:.0f}s",flush=True)
    out["variants"]["deadband3"]={"UP":per_year(EV,db_fn,1),"DOWN":per_year(EV,db_fn,0),"flat_frac":round(flat_frac,4)}
    print(f"  [deadband] UP={out['variants']['deadband3']['UP']['binding']} DOWN={out['variants']['deadband3']['DOWN']['binding']}",flush=True)
    json.dump(out,open(RESULT,"w"),indent=2,default=str)

    # =====================================================================
    #  BINDING blocks + VERDICT (pre-registered falsifier)
    # =====================================================================
    def binding_of(side):
        """Best variant on this side by binding CI95-lo; carry the beat tests."""
        best=None
        for name,v in out["variants"].items():
            b=(v.get(side,{}) or {}).get("binding",{})
            lo=b.get("ci_lo")
            if lo is None: continue
            if best is None or lo>best["ci_lo"]:
                best={"variant":name,"win":b.get("win"),"ci_lo":lo,"min_n":b.get("min_n"),
                      "uprate_2025":b.get("uprate_2025"),"uprate_2025_ok":b.get("uprate_2025_ok")}
        return best
    up_b=binding_of("UP"); dn_b=binding_of("DOWN")

    def beats_up(b):
        if not b or b["ci_lo"] is None or (b["min_n"] or 0)<100: return False
        if b["ci_lo"]<BE: return False
        se=se_win(b["win"],b["min_n"])
        return bool((b["win"] or 0) > INCUMBENT_UP+se) and bool(b.get("uprate_2025_ok"))
    def beats_down(b):
        if not b or b["ci_lo"] is None or (b["min_n"] or 0)<100: return False
        if b["ci_lo"]<BE: return False
        se=se_win(b["win"],b["min_n"])
        return bool((b["win"] or 0) > MAGWEIGHT_DOWN+se) and bool(b.get("uprate_2025_ok"))

    UP_beats=beats_up(up_b); DOWN_beats=beats_down(dn_b)
    out["binding"]={"UP":up_b,"DOWN":dn_b}
    statement=(f"label-engineering: UP {'BEATS incumbent .577' if UP_beats else 'fails'} "
               f"(best {up_b['variant'] if up_b else '-'} 2025 ci_lo {up_b['ci_lo'] if up_b else None}); "
               f"DOWN {'BEATS magweight .5441' if DOWN_beats else 'fails'} "
               f"(best {dn_b['variant'] if dn_b else '-'} 2025 ci_lo {dn_b['ci_lo'] if dn_b else None}). ")
    if not (UP_beats or DOWN_beats):
        statement+=("Redefining the TRAIN label (triple-barrier / trend-scan / jump-filter / 3-class deadband) does "
                    "NOT lift UP over .577+1SE nor DOWN over .5441+1SE on the TRUE 300s sign — the label definition is "
                    "not the binding constraint at 5m; path-dependent/redefined-label family joins the killed set "
                    "(focal/quantile/asym/magweight/seedens/rankloss). Eval used the real deriv label (ties-LOSE); "
                    "selection was worst-VAL-half.")
    out["VERDICT"]={"UP_beats":UP_beats,"DOWN_beats":DOWN_beats,"statement":statement}
    out["status"]="DONE"
    json.dump(out,open(RESULT,"w"),indent=2,default=str)
    print(f"VERDICT: {statement}\n-> {RESULT} ({time.time()-t0:.0f}s)",flush=True)

if __name__=="__main__": main()
