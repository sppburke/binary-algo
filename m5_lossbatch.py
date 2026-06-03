"""LOSS-LEVER BATCH (corpus UNTESTED direction-predictor losses, RUN) on the cross-pair primary — DOWN+UP.
Genuinely-distinct objectives vs the already-run BCE / magweight(|ret|) / lambdarank(N12):
  FOCAL  custom focal-loss fobj (γ=2): down-weight easy bars, emphasize hard/borderline → DOWN-rescue attempt.
  QUANT  quantile-regression primary on the forward RETURN (lgb objective=quantile α=0.5): sign = sign(median ret).
  ASYM   asymmetric class weight (scale_pos_weight sweep): rebalance toward the DOWN class.
Eval DOWN+UP selective per-year (cov0.05) vs incumbent. PRE-REGISTERED FALSIFIER: KILL unless a loss lifts DOWN
binding-2025 CI95-lo ≥0.541 (n≥100) OR UP binding over the .553 floor. Honest prior null (same cross-pair features
whose DOWN 2025 sign is coin-flip across magweight/seedens/pooling/rankloss). RUN to confirm, not infer.

  ~/binary-algo-venv/bin/python m5_lossbatch.py
"""
import os, sys; sys.argv=["x"]
import json, time, numpy as np
import lightgbm as lgb
import m5_xpair as MX, m5_xpair_production as XP
ROOT="/media/sean/CORSAIR/binary-algo"; BE=0.541; STRIDE=6; COV=0.05
def boot(c,nb=2500,seed=7):
    c=np.asarray(c,float)
    if len(c)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(c); a=np.array([c[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))
def focal_obj(gamma=2.0):
    def f(y,raw):
        p=1.0/(1.0+np.exp(-raw)); p=np.clip(p,1e-6,1-1e-6)
        # focal binary grad/hess wrt raw logit (standard form)
        pt=np.where(y==1,p,1-p); g_ce=p-y                      # BCE grad
        mod=(1-pt)**gamma
        # d/draw of focal ~ mod*(g_ce) - gamma*(1-pt)^(gamma-1)*pt*log(pt)*sign... use stable approx:
        grad=mod*g_ce + gamma*(1-pt)**(gamma-1)*pt*np.log(np.clip(pt,1e-6,1))*np.where(y==1,1,-1)*(p*(1-p))/np.maximum(pt,1e-6)
        hess=mod*p*(1-p)+1e-6
        return grad,hess
    return f
def per_year(EV,cols,scorefn,sv):
    res={}
    for w,D in EV.items():
        s=scorefn(D); y=D["_y"].astype(int).values; ts=D["_ts"].values.astype("int64"); ny=D["sess_ny"].values>0.5
        conf=np.abs(s-0.5); g=ny&((s>0.5) if sv==1 else (s<0.5))
        if g.sum()<20: continue
        cthr=np.quantile(conf[g],1-COV); m=g&(conf>=cthr); sel=MX.nonoverlap_chrono(ts,m,300)
        if len(sel)>=20:
            cc=(y[sel]==sv).astype(float); lo,hi=boot(cc); Y=2024 if w=="test24" else 2025 if w=="test25" else 2026
            res[Y]=dict(win=round(float(cc.mean()),4),n=len(sel),ci=[round(lo,4),round(hi,4)])
    b=[res[Y]["win"] for Y in res]; lo=[res[Y]["ci"][0] for Y in res]; ns=[res[Y]["n"] for Y in res]
    res["binding"]=dict(win=min(b) if b else None,ci_lo=min(lo) if lo else None,min_n=min(ns) if ns else 0); return res

def main():
    t0=time.time(); p=json.load(open(XP.art("strategy.json"))); cols=p["primary_feats"]
    TR=MX.augment(MX.build_xp(XP.SPL["train"],STRIDE),XP.SPL["train"],XP.MODE)
    EV={w:MX.augment(MX.build_xp(XP.SPL[w]),XP.SPL[w],XP.MODE) for w in ("test24","test25","oos")}
    Xtr=TR[cols].astype("float32"); ytr=TR["_y"].astype(int).values; fwtr=TR["_fwd"].astype("float32").values
    print(f"[lossbatch] train={len(Xtr):,} {time.time()-t0:.0f}s",flush=True)
    out={"test":"loss-lever batch (focal/quantile/asym) on cross-pair primary","breakeven":BE,"cov":COV,"variants":{}}
    def sigmoid(z): return 1/(1+np.exp(-z))
    # FOCAL
    fm=lgb.LGBMClassifier(n_estimators=1200,learning_rate=0.03,num_leaves=127,min_child_samples=300,subsample=0.8,
        subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,n_jobs=20,verbosity=-1,objective=focal_obj(2.0))
    fm.fit(Xtr,ytr)
    fsc=lambda D: sigmoid(fm.predict(D[cols].astype("float32"),raw_score=True))
    out["variants"]["focal_g2"]={"UP":per_year(EV,cols,fsc,1),"DOWN":per_year(EV,cols,fsc,0)}
    print(f"  focal done {time.time()-t0:.0f}s UP={out['variants']['focal_g2']['UP']['binding']} DOWN={out['variants']['focal_g2']['DOWN']['binding']}",flush=True)
    # QUANTILE regression on forward return -> sign(median) mapped to [0,1] via rank within split for gating
    qm=lgb.LGBMRegressor(objective="quantile",alpha=0.5,n_estimators=1200,learning_rate=0.03,num_leaves=127,
        min_child_samples=300,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,n_jobs=20,verbosity=-1)
    qm.fit(Xtr,fwtr)
    def qsc(D):
        q=qm.predict(D[cols].astype("float32"))                # predicted median fwd return
        return 0.5+0.5*np.tanh(q/ (np.std(q)+1e-9))            # map to (0,1), sign-preserving, conf=|.-.5|
    out["variants"]["quantile_median"]={"UP":per_year(EV,cols,qsc,1),"DOWN":per_year(EV,cols,qsc,0)}
    print(f"  quantile done {time.time()-t0:.0f}s UP={out['variants']['quantile_median']['UP']['binding']} DOWN={out['variants']['quantile_median']['DOWN']['binding']}",flush=True)
    # ASYM class weight (favor DOWN class y=0): sample_weight up on y==0
    for spw in (1.5,2.0):
        w=np.where(ytr==0,spw,1.0)
        am=lgb.LGBMClassifier(n_estimators=1200,learning_rate=0.03,num_leaves=127,min_child_samples=300,subsample=0.8,
            subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,n_jobs=20,verbosity=-1,objective="binary")
        am.fit(Xtr,ytr,sample_weight=w); asc=lambda D,m=am: m.predict_proba(D[cols].astype("float32"))[:,1]
        out["variants"][f"asym_down_w{spw}"]={"DOWN":per_year(EV,cols,asc,0)}
        print(f"  asym w{spw} done {time.time()-t0:.0f}s DOWN={out['variants'][f'asym_down_w{spw}']['DOWN']['binding']}",flush=True)
    # verdict
    def beats(v,side,thr):
        b=v.get(side,{}).get("binding",{}); return bool(b.get("ci_lo") is not None and b["ci_lo"]>BE and (b.get("min_n") or 0)>=100 and (b.get("win") or 0)>=thr)
    down_win=any(beats(v,"DOWN",BE) for v in out["variants"].values())
    up_win=any(beats(v,"UP",0.553) for v in out["variants"].values() if "UP" in v)
    out["VERDICT"]=dict(DOWN_certified=down_win,UP_beats_floor=up_win,
        statement=(f"loss-batch: DOWN {'CERTIFIES' if down_win else 'still fails'}, UP {'beats floor' if up_win else 'no'}. "
            + ("" if (down_win or up_win) else "Focal/quantile/asym-weight losses do NOT lift DOWN over breakeven nor UP over .553 — the objective is not the binding constraint (DOWN 2025 sign is coin-flip in the features). Loss-reoptimization family exhausted (joins magweight/seedens/rankloss).")))
    json.dump(out,open(f"{ROOT}/m5_lossbatch_result.json","w"),indent=2,default=str)
    print(f"VERDICT: {out['VERDICT']['statement']}\n-> m5_lossbatch_result.json ({time.time()-t0:.0f}s)")

if __name__=="__main__": main()
