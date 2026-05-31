"""RED-TEAM angle (a)/(b) strongest form: TRIGGERED SUBPOPULATION + 60s path-dependent up/down label.

Different from everything tried: don't predict every bar. Define a seconds-scale TRIGGER (extreme tick imbalance
run / micro-momentum burst) and TRAIN a 60s-direction model ONLY on triggered bars, then test 60s settlement on
triggered bars in held-out years. The label is still binary up/down at 60s (deriv-faithful), but the traded
POPULATION is the post-trigger regime — a naturally-selective, plausibly-non-efficient subset. If informed flow
exists, the post-trigger 60s sign is where it would show. n>=25 per year is easy (triggers are frequent).

Two trigger families (both sign-stable definitions, not magnitude): (A) strong same-side imbalance run +
micro-momentum continuation; (B) compression-then-release (low rv then imbalance burst). For each, the model
learns post-trigger sign. Per-window 2024/2025/2026, CI95. WIN = each window n>=25 & CI95-lo>0.65.
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
    ret,valid=wc_ret(ts,mid,HS,TOL,LAG); ydir=(ret>0).astype(int)
    return X,ydir,valid,ts,yr

def triggers(X):
    """Return dict of boolean trigger masks. Defined on tick microstructure, sign-symmetric (a trigger fires for
    both up- and down-bursts; the model predicts which)."""
    imb_run=X["imb_runlen"].values; imb_ema5=X["imb_ema5"].values
    mm15=X["micro_mom15"].values; rv60=X["rv60"].values; rv300=X["rv300"].values
    # A: strong directional imbalance run with micro-momentum (informed-flow burst)
    A=(np.abs(imb_ema5)>np.nanquantile(np.abs(imb_ema5),0.85))&(imb_run>=5)&(np.sign(imb_ema5)==np.sign(mm15))
    # B: compression then release — low recent rv (compressed) then imbalance burst
    comp=rv300<np.nanquantile(rv300,0.30)
    B=comp&(np.abs(imb_ema5)>np.nanquantile(np.abs(imb_ema5),0.80))
    return {"A_flowburst":A,"B_comprelease":B}

def mk(): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.03,num_leaves=128,
    min_child_samples=100,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=8,n_estimators=1200,n_jobs=20,verbosity=-1)

def main():
    t0=time.time()
    Xtr,ydtr,vtr,_,_=prep("train"); Ttr=triggers(Xtr)
    Xva,ydva,vva,tsva,_=prep("val"); Tva=triggers(Xva)
    Xte,ydte,vte,tste,yrte=prep("test"); Tte=triggers(Xte)
    Xoo,ydoo,voo,tsoo,_=prep("oos");   Too=triggers(Xoo)
    print(f"[trigpop] data ready {time.time()-t0:.0f}s",flush=True)
    for name in ("A_flowburst","B_comprelease"):
        mtr=vtr&Ttr[name]; mva=vva&Tva[name]
        itr=np.where(mtr)[0]
        if len(itr)>400000: itr=itr[::len(itr)//400000+1]
        D=mk(); D.fit(Xtr.iloc[itr],ydtr[itr],eval_set=[(Xva.iloc[np.where(mva)[0]],ydva[np.where(mva)[0]])],
            eval_metric="auc",callbacks=[lgb.early_stopping(120),lgb.log_evaluation(0)])
        pva=D.predict_proba(Xva)[:,1]
        aucv=roc_auc_score(ydva[mva],pva[mva]) if mva.sum()>50 else float('nan')
        confv=np.abs(pva-0.5)[mva]
        print(f"\n=== TRIGGER {name} === train_fired={mtr.sum():,} val_fired={mva.sum():,} VAL_post-trigger_dirAUC={aucv:.4f}",flush=True)
        pte=D.predict_proba(Xte)[:,1]; poo=D.predict_proba(Xoo)[:,1]
        W={"2024":(pte[yrte==2024],ydte[yrte==2024],vte[yrte==2024],Tte[name][yrte==2024],tste[yrte==2024]),
           "2025":(pte[yrte==2025],ydte[yrte==2025],vte[yrte==2025],Tte[name][yrte==2025],tste[yrte==2025]),
           "2026":(poo,ydoo,voo,Too[name],tsoo)}
        for cov in (1.0,0.5,0.25,0.1):
            dthr=float(np.quantile(confv,1-cov)) if len(confv) else 0.0; rows=[]
            for k in ("2024","2025","2026"):
                p_,y_,v_,t_,ts_=W[k]
                m=v_&t_&(np.abs(p_-0.5)>=dthr)
                if m.sum()==0: rows.append((0,float('nan'),float('nan'),float('nan'))); continue
                s=nonoverlap_chrono(ts_,m,HS+TOL)
                corr=((p_[s]>0.5).astype(int)==y_[s]).astype(float)
                rows.append((len(s),corr.mean(),*boot(corr)))
            win=all(r[0]>=25 and r[2]>0.65 for r in rows)
            print(f"   cov{cov:>4}: "+" | ".join(f"{k} n{r[0]} {r[1]:.3f}[{r[2]:.2f},{r[3]:.2f}]" for k,r in zip(('24','25','26'),rows))+("  <<<WIN" if win else ""),flush=True)
    print(f"\n[trigpop] DONE {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__": main()
