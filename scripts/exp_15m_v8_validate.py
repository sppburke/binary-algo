"""15-MINUTE v8 — HONEST validation of the gated selective frontier (anti-multiple-testing).

V18-V19 chose the compress×NY @low-cov pocket partly by eyeballing OOS — a soft leak. Here we:
  1. Train the global LGBM on TRAIN only.
  2. Enumerate a grid of (gate × coverage) candidate pockets.
  3. SELECT the single best pocket by VAL selective accuracy (with a VAL-n floor) — OOS never consulted.
  4. Confirm on TEST, then judge on 2026 OOS exactly once. Bootstrap 95% CI on the OOS accuracy.
  5. Deflation context: report how high the best-of-grid VAL accuracy would look under the null
     (binomial spread at the selected n) so the reader can discount the multiple comparisons.
Target stays the up/down binary endpoint sign. Selection rule = |p-0.5| within the chosen gate.
"""
import time, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import exp_15m_v5_gates as V5

STRIDE=3
COVS=[0.5,0.2,0.1,0.05,0.02,0.01]
VAL_NFLOOR=300     # require >=300 VAL bets so the chosen threshold is stable

def gates_for(g):
    h=g["utc_hour"].values
    london=g["sess_london"].values.astype(float)>0.5 if "sess_london" in g else np.zeros(len(g),bool)
    ny=g["sess_ny"].values.astype(float)>0.5 if "sess_ny" in g else np.zeros(len(g),bool)
    return {"london":london,"ny":ny,"londonfix":(h>=15)&(h<16),"usdata":(h>=13)&(h<14)}

def comp_mask(g,thr): return g["15m_bb_width"].values.astype(float)<=thr

def acc_at(y,p,thr):
    m=np.abs(p-0.5)>=thr
    if m.sum()==0: return np.nan,0,None
    sel=((p[m]>0.5).astype(int)==y[m])
    return sel.mean(), int(m.sum()), sel

def main():
    t0=time.time()
    Xtr,ytr,gtr=V5.load_split("train",STRIDE)
    Xva,yva,gva=V5.load_split("val")
    Xte,yte,gte=V5.load_split("test")
    Xoo,yoo,goo=V5.load_split("oos")
    thr_bw=np.nanpercentile(gtr["15m_bb_width"].values.astype(float),33)
    print(f"v8 load={time.time()-t0:.0f}s  comp thr={thr_bw:.6g}",flush=True)
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=255,
        min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=10,
        n_estimators=3000,n_jobs=20,verbosity=-1)
    L.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pv=L.predict_proba(Xva)[:,1]; pt=L.predict_proba(Xte)[:,1]; po=L.predict_proba(Xoo)[:,1]
    print(f"AUC oos={roc_auc_score(yoo,po):.4f}",flush=True)

    cv=comp_mask(gva,thr_bw); ct=comp_mask(gte,thr_bw); co=comp_mask(goo,thr_bw)
    Sv=gates_for(gva); St=gates_for(gte); So=gates_for(goo)
    # candidate gate masks (val/test/oos triples)
    cand={"ALL":(np.ones(len(gva),bool),np.ones(len(gte),bool),np.ones(len(goo),bool)),
          "compress":(cv,ct,co)}
    for s in Sv:
        cand[s]=(Sv[s],St[s],So[s])
        cand["compress&"+s]=(cv&Sv[s],ct&St[s],co&So[s])

    # ---- 1) select best pocket by VAL accuracy (OOS not consulted) ----
    rows=[]
    for name,(mv,mt,mo) in cand.items():
        yv_g,pv_g=yva[mv],pv[mv]
        if len(yv_g)<200: continue
        confv=np.abs(pv_g-0.5)
        for cov in COVS:
            thr=np.quantile(confv,1-cov)
            av,nv,_=acc_at(yv_g,pv_g,thr)
            if nv<VAL_NFLOOR: continue
            rows.append((name,cov,thr,av,nv))
    rows.sort(key=lambda r:-r[3])
    print("\n--- top VAL pockets (selection set; OOS NOT consulted) ---")
    for r in rows[:8]:
        print(f"   {r[0]:>18} cov{int(r[1]*100):>2} VALacc={r[3]:.4f} (nVAL={r[4]})")
    best=rows[0]; name,cov,thr=best[0],best[1],best[2]
    print(f"\nSELECTED pocket = '{name}' @cov{int(cov*100)}  (VALacc={best[3]:.4f}, nVAL={best[4]})")

    # ---- DIAGNOSTIC: does VAL ranking carry ANY OOS validity? show TEST/OOS for top-8 VAL pockets ----
    print("\n--- transfer diagnostic: top-8 VAL pockets, frozen-threshold TEST/OOS (assess VAL->OOS, not re-select) ---")
    print(f"   {'pocket':>20} {'cov':>4} {'VALacc':>7} {'TESTacc':>8} {'OOSacc':>8} {'nOOS':>6}")
    for r in rows[:8]:
        nm,cv2,th=r[0],r[1],r[2]; mv2,mt2,mo2=cand[nm]
        a_t,n_t,_=acc_at(yte[mt2],pt[mt2],th); a_o,n_o,_=acc_at(yoo[mo2],po[mo2],th)
        print(f"   {nm:>20} {int(cv2*100):>4} {r[3]:>7.3f} {a_t:>8.3f} {a_o:>8.3f} {n_o:>6}")
    import numpy as _np
    vals=_np.array([r[3] for r in rows[:8]]); oos=_np.array([acc_at(yoo[cand[r[0]][2]],po[cand[r[0]][2]],r[2])[0] for r in rows[:8]])
    msk=_np.isfinite(oos)
    if msk.sum()>=3:
        cc=_np.corrcoef(vals[msk],oos[msk])[0,1]
        print(f"   corr(VALacc, OOSacc) across top-8 = {cc:+.2f}   (≈0 or negative => VAL ranking has no OOS validity)")

    # ---- 2) confirm TEST, 3) judge OOS once, 4) bootstrap CI ----
    mv,mt,mo=cand[name]
    at,nt,_=acc_at(yte[mt],pt[mt],thr)
    ao,no,selo=acc_at(yoo[mo],po[mo],thr)
    print(f"  TEST acc={at:.4f} (n={nt})")
    print(f"  OOS  acc={ao:.4f} (n={no})   <-- single held-out judgment")
    if selo is not None and no>=20:
        rng=np.random.default_rng(7)
        boot=np.array([rng.choice(selo,size=no,replace=True).mean() for _ in range(5000)])
        lo,hi=np.percentile(boot,[2.5,97.5])
        print(f"  OOS 95% bootstrap CI = [{lo:.3f}, {hi:.3f}]")
        # binomial p-value vs 0.5
        from math import comb
        k=int(round(ao*no))
        # normal approx p-value (one-sided)
        z=(ao-0.5)/np.sqrt(0.25/no); print(f"  vs 0.50: z={z:.2f}  (>=75%? {'YES' if lo>=0.75 else 'no'})")

    # ---- 5) deflation context: spread of best-of-grid under the null ----
    ntrials=len(rows)
    nsel=best[4]
    # expected max of `ntrials` binomial(n, .5) accuracies — quick MC
    rng=np.random.default_rng(1)
    nulls=rng.binomial(nsel,0.5,size=(2000,ntrials))/nsel
    exp_max=nulls.max(axis=1).mean()
    print(f"\nDEFLATION: {ntrials} pockets tried; under null the best VALacc at n={nsel} "
          f"averages ~{exp_max:.3f} by chance. (Selected VALacc {best[3]:.3f}.)")
    print("V8-VALIDATE DONE")

if __name__=="__main__":
    main()
