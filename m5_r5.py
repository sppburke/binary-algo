"""R5 — RECENCY-DECAY sample-weighting τ-sweep on the m5xp primary (does the 2026 collapse = slow drift?).

Refit the cross-pair PRIMARY with exponential chronological sample weights w_t=exp(−(T−t)/τ), τ∈{2,4,8,∞}yr
(τ=∞ = current uniform book). Refit the orthogonal meta on VAL each time (fair), gate at the deployed VAL-q
threshold, evaluate UP per-year 2024/25/26. SIGN from the direction model; recency is a stationarity prior
(theorem-safe). HYPOTHESIS: if the 2026 collapse is slow drift, upweighting recent TRAIN lifts the binding year.

PRE-REGISTERED FALSIFIER: KILL R5 unless a finite τ raises the binding (worst) held-out UP year above τ=∞
(the current book) at comparable trade count, with binding-year still clearing breakeven. Incumbent: m5xp UP
binding-2025 .577 (refit floor .553).

  ~/binary-algo-venv/bin/python m5_r5.py
"""
import os, sys; sys.argv=["x"]
import json, time, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import m5_xpair as MX, m5_xpair_production as XP
ROOT="/home/sean/git/binary-algo"; BE=0.541
def boot(c,nb=2000,seed=7):
    c=np.asarray(c,float)
    if len(c)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(c); a=np.array([c[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))
def yr(ts): return (np.asarray(ts,dtype="datetime64[s]").astype("datetime64[Y]").astype(int)+1970)

def main():
    t0=time.time()
    p0=json.load(open(XP.art("strategy.json"))); cols=p0["primary_feats"]; Q=p0.get("val_q",0.95)
    TR=MX.build_xp(XP.SPL["train"],4); TR=MX.augment(TR,XP.SPL["train"],XP.MODE)
    VA=MX.build_xp(XP.SPL["val"]); VA=MX.augment(VA,XP.SPL["val"],XP.MODE)
    EV={w:MX.augment(MX.build_xp(XP.SPL[w]),XP.SPL[w],XP.MODE) for w in ("test24","test25","oos")}
    ytr=TR["_y"].astype(int).values; yva=VA["_y"].astype(int).values
    tstr=TR["_ts"].values.astype("int64"); Tmax=tstr.max(); yrs=(Tmax-tstr)/ (365.25*86400)   # years before end
    print(f"[r5] train={len(TR):,} val={len(VA):,} feats={len(cols)} {time.time()-t0:.0f}s",flush=True)
    mcols=XP.meta_feats(VA); vny=VA["sess_ny"].values>0.5; tsv=VA["_ts"].values.astype("int64")
    Xva=VA[cols].astype("float32"); EVx={w:EV[w][cols].astype("float32") for w in EV}
    res={}
    for tau in (2.0,4.0,8.0,np.inf):
        w=np.ones(len(ytr)) if np.isinf(tau) else np.exp(-yrs/tau); w=w/w.mean()
        P=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=127,min_child_samples=400,
            subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,n_estimators=1500,n_jobs=20,verbosity=-1)
        P.fit(TR[cols].astype("float32"),ytr,sample_weight=w,
              eval_set=[(Xva,yva)],eval_metric="auc",callbacks=[lgb.early_stopping(120),lgb.log_evaluation(0)])
        pva=P.predict_proba(Xva)[:,1]
        M=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=15,min_child_samples=1000,
            subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=20,n_estimators=400,n_jobs=20,verbosity=-1)
        Xm=XP._Xmeta(VA,pva,mcols); ycorr=((pva>0.5).astype(int)==yva).astype(int); M.fit(Xm[vny],ycorr[vny])
        sval=M.predict_proba(Xm)[:,1]; THR=float(np.quantile(sval[vny],Q))
        py={}
        for ww in ("test24","test25","oos"):
            D=EV[ww]; pr=P.predict_proba(EVx[ww])[:,1]; sm=M.predict_proba(XP._Xmeta(D,pr,mcols))[:,1]
            y=D["_y"].astype(int).values; ts=D["_ts"].values.astype("int64"); ny=D["sess_ny"].values>0.5
            g=ny&(sm>=THR)&(pr>0.5); sel=MX.nonoverlap_chrono(ts,g,300)
            if len(sel)>=20:
                cc=((pr[sel]>0.5).astype(int)==y[sel]).astype(float); lo,hi=boot(cc)
                YR=(2024 if ww=="test24" else 2025 if ww=="test25" else 2026)
                py[YR]=dict(win=round(float(cc.mean()),4),n=len(sel),ci=[round(lo,4),round(hi,4)])
        b=[py[Y]["win"] for Y in py]; lo=[py[Y]["ci"][0] for Y in py]
        tk="inf" if np.isinf(tau) else f"{tau:.0f}"
        res[tk]=dict(per_year=py, val_auc=round(float(roc_auc_score(yva,pva)),4), thr=round(THR,4),
                     binding=dict(win=round(min(b),4) if b else None, ci_lo=round(min(lo),4) if lo else None))
        print(f"  τ={tk:>3}yr: VALauc={res[tk]['val_auc']} per-year={ {Y:py[Y]['win'] for Y in py} } binding={res[tk]['binding']}",flush=True)
    inf_bind=res["inf"]["binding"]["win"]
    best_fin=max([t for t in res if t!="inf"], key=lambda t:(res[t]["binding"]["win"] or 0))
    bf=res[best_fin]["binding"]
    survive=bool(bf["win"] is not None and inf_bind is not None and bf["win"]>inf_bind+1e-4 and bf["ci_lo"]>BE)
    out=dict(test="R5 recency-decay sample-weighting τ-sweep (m5xp primary)", breakeven=BE, val_q=Q, by_tau=res,
        VERDICT=dict(tau_inf_binding=inf_bind, best_finite_tau=best_fin, best_finite_binding=bf, SURVIVES=survive,
          statement=("R5 SURVIVES: a finite τ raises the binding year above the uniform book, CI clears breakeven."
            if survive else
            f"R5 KILLED: best finite τ={best_fin}yr binding {bf['win']} (CI-lo {bf['ci_lo']}) does NOT beat uniform τ=∞ "
            f"binding {inf_bind}. The 2026 collapse is NOT slow drift fixable by recency-weighting the 2012-21 train "
            f"(it is regime non-stationarity, consistent with corr(VAL,OOS)=−0.54).")))
    json.dump(out,open(f"{ROOT}/m5_r5_result.json","w"),indent=2,default=str)
    print(f"\nVERDICT: {out['VERDICT']['statement']}\n-> m5_r5_result.json ({time.time()-t0:.0f}s)")

if __name__=="__main__": main()
