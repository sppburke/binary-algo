"""RED-TEAM angle (a): TRIGGERED ENTRY carrying the seconds-scale edge to a 60s binary settlement.

Hypothesis to break the exhaustion verdict: the tick ensemble has a real ~0.65 edge at 5s. If that edge is a
DIRECTIONAL signal (informed flow) rather than a microstructure-reversion artifact, then firing only on its
confident signals and SETTLING AT 60s wall-clock should carry >0.65 to the 60s binary on a naturally-selective
(but n>=25) traded subset. If instead the 5s edge is sub-minute mean-reversion / bid-ask bounce, it will decay or
FLIP by 60s. tickhz.py never tested this: it trains a FRESH model per horizon (so 60s model = 60s-noise). This
trains where the edge lives and holds to 60s.

Deriv-faithful: label = sign(mid[entry+60] - mid[entry]), ties LOSE, entry = next tick (1s lag), non-overlapping
60s trades, chronological first-come de-overlap. Threshold chosen on VAL (2024H1). Reported on EACH window
2024(test<2025) / 2025(test) / 2026(oos) with bootstrap CI95. WIN requires every window n>=25 AND CI95-lo>0.65.
"""
import sys, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from min1_production import feats, wc_ret, nonoverlap_chrono, boot, load_split, set_pair
set_pair("EURUSD")
LAG=1
HS_TRAIN=int(sys.argv[1]) if len(sys.argv)>1 else 5     # horizon the model is TRAINED to predict
HS_SETTLE=60                                            # horizon the binary actually SETTLES at (wall-clock 60s)
def TOL(hs): return max(2, hs//30+1)

def prep(sp):
    b=load_split(sp); X=feats(b)
    mid=b["mid"].values.astype(float); ts=b.index.values.astype("datetime64[s]").astype("int64")
    yr=b.index.year.values; hour=b.index.hour.values
    # training label at HS_TRAIN; settlement label at HS_SETTLE (both deriv-faithful)
    rtr,vtr=wc_ret(ts,mid,HS_TRAIN,TOL(HS_TRAIN),LAG); ytr=(rtr>0).astype(int)
    rst,vst=wc_ret(ts,mid,HS_SETTLE,TOL(HS_SETTLE),LAG); yst=(rst>0).astype(int)
    return X,ytr,vtr,yst,vst,ts,yr,hour

def mk(): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.03,num_leaves=300,
    min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=8,n_estimators=2500,n_jobs=20,verbosity=-1)

def settle_eval(p,yst,vst,ts,thr):
    """fire when |p-0.5|>=thr; SETTLE at 60s; non-overlap 60s; return n, acc, CI."""
    gap=HS_SETTLE+TOL(HS_SETTLE)
    m=vst&(np.abs(p-0.5)>=thr)
    if m.sum()==0: return (0,float("nan"),float("nan"),float("nan"))
    s=nonoverlap_chrono(ts,m,gap)
    if len(s)==0: return (0,float("nan"),float("nan"),float("nan"))
    corr=((p[s]>0.5).astype(int)==yst[s]).astype(float)
    return (len(s),corr.mean(),*boot(corr))

def main():
    t0=time.time()
    Xtr,ytr,vtr,_,_,tstr,_,_=prep("train"); itr=np.where(vtr)[0][::20]
    Xva,yva,vva,ystv,vstv,tsva,_,_=prep("val")
    print(f"[trig60] HS_TRAIN={HS_TRAIN} HS_SETTLE={HS_SETTLE}  fit={len(itr):,} val={vva.sum():,}  {time.time()-t0:.0f}s",flush=True)
    L=mk(); L.fit(Xtr.iloc[itr],ytr[itr],eval_set=[(Xva.iloc[np.where(vva)[0]],yva[np.where(vva)[0]])],
        eval_metric="auc",callbacks=[lgb.early_stopping(120),lgb.log_evaluation(0)])
    del Xtr
    # VAL: also report whether the TRAIN-horizon signal even predicts the 60s settlement at all (AUC)
    pva=L.predict_proba(Xva)[:,1]
    both=vva&vstv
    auc_settle=roc_auc_score(ystv[both],pva[both])
    print(f"[trig60] VAL: {HS_TRAIN}s-trained signal AUC vs 60s-settle outcome = {auc_settle:.4f} (0.50=no carry; >0.55 = real carry)",flush=True)
    # choose conf thresholds on VAL by coverage
    del Xva
    # build TEST(2024,2025) and OOS(2026) once
    Xte,_,_,yste,vste,tste,yrte,_=prep("test"); pte=L.predict_proba(Xte)[:,1]; del Xte
    Xoo,_,_,ysto,vsto,tsoo,yroo,_=prep("oos"); poo=L.predict_proba(Xoo)[:,1]; del Xoo
    confv=np.abs(pva-0.5)[both]
    print(f"\n{'cov':>6} {'thr':>7} | {'2024 (n/acc/CI)':>26} | {'2025':>26} | {'2026':>26}",flush=True)
    for cov in (0.05,0.02,0.01,0.005,0.002,0.001):
        thr=float(np.quantile(confv,1-cov))
        # 2024 = test rows with year 2024 ; 2025 = test rows year 2025 ; 2026 = oos
        m24=yrte==2024; m25=yrte==2025
        r24=settle_eval(pte[m24],yste[m24],vste[m24],tste[m24],thr)
        r25=settle_eval(pte[m25],yste[m25],vste[m25],tste[m25],thr)
        r26=settle_eval(poo,ysto,vsto,tsoo,thr)
        def fmt(r): return f"n{r[0]:<5} {r[1]:.3f}[{r[2]:.2f},{r[3]:.2f}]" if r[0] else "n0 ----"
        win = all(r[0]>=25 and r[2]>0.65 for r in (r24,r25,r26))
        print(f"{cov:>6.1%} {thr:>7.4f} | {fmt(r24):>26} | {fmt(r25):>26} | {fmt(r26):>26}  {'<<<WIN' if win else ''}",flush=True)
    print(f"\n[trig60] DONE {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__": main()
