"""RED-TEAM angle (c)/(d): at 60s, does conditioning DIRECTION on predicted-MAGNITUDE create a stable >0.65 subset?

The verdict rests on the sign-invariance theorem: magnitude (move SIZE) is forecastable (AUC ~0.68) but is
ORTHOGONAL to sign, so gating direction by predicted-magnitude should NOT lift directional accuracy. This was
RUN at 10m (m10_magdir.py, null) but never at the 60s tick scale where magnitude AUC is highest. If the theorem
is wrong here, the HIGH-magnitude bucket would show directional acc > the unconditional ~0.51 AND be 2025-stable.

Also tests angle (c) directly: the directional CONDITIONAL accuracy ceiling on 2025 across confidence buckets.
A meta cannot exceed this ceiling, so if it is ~0.55 on 2025, no meta can reach 0.65.

Deriv-faithful 60s, ties lose, 1s entry lag, non-overlap, per-window 2024/2025/2026, CI95.
"""
import sys, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from min1_production import feats, wc_ret, nonoverlap_chrono, boot, load_split, set_pair
set_pair("EURUSD"); HS=60; LAG=1; TOL=3

def prep(sp):
    b=load_split(sp); X=feats(b)
    mid=b["mid"].values.astype(float); ts=b.index.values.astype("datetime64[s]").astype("int64")
    yr=b.index.year.values
    ret,valid=wc_ret(ts,mid,HS,TOL,LAG)
    ydir=(ret>0).astype(int); amag=np.abs(ret)
    return X,ydir,amag,valid,ts,yr

def mk(): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.03,num_leaves=255,
    min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=8,n_estimators=1500,n_jobs=20,verbosity=-1)

def main():
    t0=time.time()
    Xtr,ydtr,amtr,vtr,_,_=prep("train"); itr=np.where(vtr)[0][::20]
    Xva,ydva,amva,vva,tsva,_=prep("val")
    q75=float(np.quantile(amtr[vtr],0.75))
    print(f"[magdir60] fit={len(itr):,} val={vva.sum():,} q75|ret60|={q75:.2e} {time.time()-t0:.0f}s",flush=True)
    # direction model
    D=mk(); D.fit(Xtr.iloc[itr],ydtr[itr],eval_set=[(Xva.iloc[np.where(vva)[0]],ydva[np.where(vva)[0]])],
        eval_metric="auc",callbacks=[lgb.early_stopping(120),lgb.log_evaluation(0)])
    # magnitude model
    ymtr=(amtr>=q75).astype(int)
    M=mk(); M.fit(Xtr.iloc[itr],ymtr[itr],eval_set=[(Xva.iloc[np.where(vva)[0]],(amva[np.where(vva)[0]]>=q75).astype(int))],
        eval_metric="auc",callbacks=[lgb.early_stopping(120),lgb.log_evaluation(0)])
    del Xtr
    pdva=D.predict_proba(Xva)[:,1]; pmva=M.predict_proba(Xva)[:,1]; del Xva
    print(f"[magdir60] VAL dirAUC={roc_auc_score(ydva[vva],pdva[vva]):.4f} magAUC={roc_auc_score((amva[vva]>=q75).astype(int),pmva[vva]):.4f}",flush=True)

    Xte,ydte,amte,vte,tste,yrte=prep("test"); pdte=D.predict_proba(Xte)[:,1]; pmte=M.predict_proba(Xte)[:,1]; del Xte
    Xoo,ydoo,amoo,voo,tsoo,_=prep("oos");   pdoo=D.predict_proba(Xoo)[:,1]; pmoo=M.predict_proba(Xoo)[:,1]; del Xoo

    # assemble windows
    W={"2024":(pdte[yrte==2024],pmte[yrte==2024],ydte[yrte==2024],vte[yrte==2024],tste[yrte==2024]),
       "2025":(pdte[yrte==2025],pmte[yrte==2025],ydte[yrte==2025],vte[yrte==2025],tste[yrte==2025]),
       "2026":(pdoo,pmoo,ydoo,voo,tsoo)}

    # (1) magnitude-gated direction: within HIGH predicted-magnitude bars, is direction conditional-acc higher & stable?
    pmthr=float(np.quantile(pmva[vva],0.75))  # top quartile predicted magnitude (set on VAL)
    print(f"\n(1) MAGNITUDE-GATED DIRECTION (top-25% predicted magnitude, dir-conf selective). pmthr={pmthr:.3f}",flush=True)
    for dcov in (1.0,0.5,0.25,0.1):
        # within high-mag, take top dcov by direction confidence; threshold set on VAL high-mag bars
        hv=vva&(pmva>=pmthr); confv=np.abs(pdva-0.5)
        dthr=float(np.quantile(confv[hv],1-dcov)) if hv.sum() else 0.0
        rows=[]
        for k in ("2024","2025","2026"):
            pd_,pm_,yd_,v_,ts_=W[k]
            m=v_&(pm_>=pmthr)&(np.abs(pd_-0.5)>=dthr)
            if m.sum()==0: rows.append((0,float('nan'),float('nan'),float('nan'))); continue
            s=nonoverlap_chrono(ts_,m,HS+TOL)
            corr=((pd_[s]>0.5).astype(int)==yd_[s]).astype(float)
            rows.append((len(s),corr.mean(),*boot(corr)))
        win=all(r[0]>=25 and r[2]>0.65 for r in rows)
        print(f"   dcov{dcov:>4}: "+" | ".join(f"{k} n{r[0]} {r[1]:.3f}[{r[2]:.2f},{r[3]:.2f}]" for k,r in zip(('24','25','26'),rows))+("  <<<WIN" if win else ""),flush=True)

    # (2) directional conditional-accuracy CEILING per window (pure dir confidence, no mag gate) — the meta wall
    print(f"\n(2) DIRECTIONAL CONDITIONAL-ACC CEILING (the wall a meta cannot exceed), by dir-confidence coverage:",flush=True)
    confv=np.abs(pdva-0.5)[vva]
    for cov in (0.1,0.05,0.02,0.01,0.005):
        dthr=float(np.quantile(confv,1-cov)); rows=[]
        for k in ("2024","2025","2026"):
            pd_,pm_,yd_,v_,ts_=W[k]
            m=v_&(np.abs(pd_-0.5)>=dthr)
            if m.sum()==0: rows.append((0,float('nan'),float('nan'),float('nan'))); continue
            s=nonoverlap_chrono(ts_,m,HS+TOL)
            corr=((pd_[s]>0.5).astype(int)==yd_[s]).astype(float)
            rows.append((len(s),corr.mean(),*boot(corr)))
        win=all(r[0]>=25 and r[2]>0.65 for r in rows)
        print(f"   cov{cov:>6.1%}: "+" | ".join(f"{k} n{r[0]} {r[1]:.3f}[{r[2]:.2f},{r[3]:.2f}]" for k,r in zip(('24','25','26'),rows))+("  <<<WIN" if win else ""),flush=True)
    print(f"\n[magdir60] DONE {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__": main()
