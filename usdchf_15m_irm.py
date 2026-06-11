"""USDCHF 15m — R2-1: ENVIRONMENT-INVARIANT (IRM-style) feature-stability filter on the xpair-NY book.

SCOPE: USDCHF · 15m (discovery R2 survivor). The certified deliverable USDCHF.m15ny_xpair.v1 (EUR-bloc
xpair-NY, BOTH sides >65% refit-CPCV) has ONE weakness: it is REFIT-DEPENDENT — the frozen 2012-21 vintage
decays forward (full-337 frozen cov1 COMB .7924(2024)→.4775(2026), DOWN .43). HYPOTHESIS (R2-1): some of the
337 features have a sign→Y relationship that FLIPS across eras (era-local sign), and those drive the forward
decay. Selecting only features whose DIRECTED sign is INVARIANT across regime-partitioned environments, then
refitting on that subset, should DECAY LESS forward without losing the carried sign.

METHOD (IRM-style, on-disk, no NN):
  1. Build the xpair-NY matrix (337 feats; usdchf_15m_xpair_frozen.build_mat), restrict to NY moved bars.
  2. Partition TRAIN (2012-21 NY) into N_ENV chronological environments (eras).
  3. Per feature, per env: single-feature DIRECTIONAL AUC = roc_auc_score(y_env, feat_env). The feature's
     directed sign in that env = sign(AUC - 0.5). A feature is INVARIANT iff that sign AGREES across ALL envs
     (no era-local flip) AND it is non-trivially predictive (median |AUC-0.5| >= EPS). Sign-flippers dropped.
  4. Train the FROZEN book (2012-21 NY) on the invariant subset, gate on VAL worst-half, evaluate forward
     2024/25/26 NY at the same cov gates (matched to usdchf_15m_xpair_frozen / _ownpair_frozen).
  5. Compare frozen-forward to the FULL-337 frozen incumbent (usdchf_15m_xpair_frozen_result.json).

FALSIFIER (pre-registered, R2 fast-KILL): SURVIVES iff (a) the filter prunes >= ~10% of features AND
(b) the invariant subset's frozen 2026 cov1 COMB is MATERIALLY above the full-337 .4775 on BOTH sides
(>= +0.02). Else KILLED (recent-era weakness is genuine signal / info-bound decay — no feature selection helps).

Usage: ~/binary-algo-venv/bin/python usdchf_15m_irm.py [n_env=6] [eps=0.004]
"""
import os, sys, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sessions import session_mask
# parse OUR args BEFORE importing xpair_frozen (it parses sys.argv at module-load: NSEED=int(argv[2]))
N_ENV=int(sys.argv[1]) if len(sys.argv)>1 else 6
EPS=float(sys.argv[2]) if len(sys.argv)>2 else 0.004
_argv=sys.argv; sys.argv=[_argv[0]]                       # neutralize argv for the import
import usdchf_15m_xpair_frozen as XF   # reuse build_mat, side_eval, nonoverlap_chrono
sys.argv=_argv

HOR=15; GAP=900; BE=0.541; NUM_LEAVES=255; STRIDE=3
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],"test24":["2024"],"test25":["2025"],"oos":["2026"]}
COVS=(0.05,0.03,0.02,0.01)
RESULT="usdchf_15m_irm_result.json"

def main():
    t0=time.time()
    full=json.load(open("usdchf_15m_xpair_frozen_result.json"))   # full-337 frozen incumbent
    f26=full["years"]["oos"]["bycov"]
    inc_2026_cov1={"COMBINED":f26["0.01"]["COMBINED"]["wr"],"UP":f26["0.01"]["UP"].get("wr"),"DOWN":f26["0.01"]["DOWN"].get("wr")}
    res={"key":"USDCHF.15m","model":"R2-1 IRM env-invariant feature-stability filter on xpair-NY frozen book",
         "n_env":N_ENV,"eps":EPS,"full337_frozen_2026_cov1":inc_2026_cov1,
         "falsifier":{"registered":"pre-eval",
            "SURVIVES_if":"prune_frac>=0.10 AND invariant-subset frozen 2026 cov1 COMB >= full-337 +0.02 on BOTH sides",
            "KILL_if":"prune<10% OR no material 2026 decay reduction (recent weakness = genuine signal/info-bound)"}}
    json.dump(res,open(RESULT,"w"),indent=2)

    allyears=[y for k in ("train","val","test24","test25","oos") for y in SPL[k]]
    print(f"[irm] building xpair matrix (stride {STRIDE})...",flush=True)
    X,fwd,ts,cols=XF.build_mat(allyears,STRIDE)
    ny=session_mask(ts,"ny"); yr=pd.to_datetime(ts,unit="s").year.values
    moved=np.isfinite(fwd)&(fwd!=0)
    itr=np.isin(yr,[int(y) for y in SPL["train"]])&ny&moved
    iva=np.isin(yr,[int(y) for y in SPL["val"]])&ny&np.isfinite(fwd)
    print(f"[irm] rows={len(ts):,} train={int(itr.sum()):,} val={int(iva.sum()):,} feats={len(cols)} build={time.time()-t0:.0f}s",flush=True)

    # --- per-env single-feature directional AUC sign-consistency ---
    tr_idx=np.where(itr)[0]; tr_ts=ts[tr_idx]; o=np.argsort(tr_ts); tr_idx=tr_idx[o]
    env_bounds=[int(k*len(tr_idx)/N_ENV) for k in range(N_ENV)]+[len(tr_idx)]
    envs=[tr_idx[env_bounds[e]:env_bounds[e+1]] for e in range(N_ENV)]
    ytr_full=(fwd>0).astype(int)
    sign_mat=np.zeros((len(cols),N_ENV)); absauc=np.zeros((len(cols),N_ENV))
    Xtr_all=X[itr]; ytr_all=ytr_full[itr]   # for speed, compute on the strided train
    # recompute env masks within the itr-subset ordering
    tr_order=np.argsort(ts[itr]);
    Xo=Xtr_all[tr_order]; yo=ytr_all[tr_order]
    eb=[int(k*len(yo)/N_ENV) for k in range(N_ENV)]+[len(yo)]
    for e in range(N_ENV):
        sl=slice(eb[e],eb[e+1]); ye=yo[sl]; Xe=Xo[sl]
        if len(ye)<200 or ye.mean() in (0,1): continue
        for j in range(len(cols)):
            col=Xe[:,j]; m=np.isfinite(col)
            if m.sum()<100 or len(np.unique(col[m]))<3:
                absauc[j,e]=0; continue
            try:
                a=roc_auc_score(ye[m], col[m])
            except Exception:
                a=0.5
            sign_mat[j,e]=np.sign(a-0.5); absauc[j,e]=abs(a-0.5)
    # invariant = sign agrees across ALL envs (nonzero, same sign) AND median |AUC-.5| >= EPS
    sign_ok=np.array([ (len(set(s[s!=0]))==1 and (s!=0).sum()>=N_ENV-1) for s in sign_mat ])
    pred_ok=np.median(absauc,axis=1)>=EPS
    invariant=sign_ok & pred_ok
    inv_idx=np.where(invariant)[0]
    prune_frac=1.0-len(inv_idx)/len(cols)
    res["n_feats_full"]=len(cols); res["n_feats_invariant"]=int(len(inv_idx)); res["prune_frac"]=round(float(prune_frac),3)
    print(f"[irm] invariant feats={len(inv_idx)}/{len(cols)} (prune {prune_frac:.1%}); dropped sign-flippers={int((~sign_ok).sum())}, weak={int((sign_ok&~pred_ok).sum())} ({time.time()-t0:.0f}s)",flush=True)
    if prune_frac<0.10:
        res["verdict"]={"SURVIVES":False,"reason":f"prune {prune_frac:.1%} < 10% — filter near-trivial; the 337 feats are mostly sign-stable, no era-local sign-flip driving decay"}
        json.dump(res,open(RESULT,"w"),indent=2)
        print(f"[irm] VERDICT KILLED (prune<10%) -> {RESULT}",flush=True); return

    # --- train frozen book on invariant subset, evaluate forward ---
    Xi=X[:,inv_idx]
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=NUM_LEAVES,
        min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,
        n_estimators=800,n_jobs=16,verbosity=-1,random_state=0)
    L.fit(Xi[itr],(fwd[itr]>0).astype(int),eval_set=[(Xi[iva],(fwd[iva]>0).astype(int))],eval_metric="auc",
          callbacks=[lgb.early_stopping(80),lgb.log_evaluation(0)])
    pva=L.predict_proba(Xi[iva])[:,1]; mvv=(fwd[iva]!=0)
    vauc=float(roc_auc_score((fwd[iva][mvv]>0).astype(int),pva[mvv]))
    tsv=ts[iva]; fwv=fwd[iva]; oo=np.argsort(tsv); tsv=tsv[oo]; fwv=fwv[oo]; pvas=pva[oo]
    half=len(pvas)//2; confv=np.abs(pvas-0.5); gates={}
    for cov in COVS:
        thr=float(np.quantile(confv,1-cov))
        hv=[(lambda r: r["COMBINED"]["wr"] if r else np.nan)(XF.side_eval(pvas[s:e],fwv[s:e],tsv[s:e],thr)) for s,e in ((0,half),(half,len(pvas)))]
        gates[f"{cov:.2f}"]={"thr":thr,"val_worst_half":round(float(np.nanmin(hv)),4)}
    res["val_auc"]=round(vauc,4); res["gates"]=gates
    print(f"[irm] invariant-subset VAL AUC={vauc:.4f} (full-337 .5572)",flush=True)
    res["years"]={}
    for w in ("test24","test25","oos"):
        m=np.isin(yr,[int(y) for y in SPL[w]])&ny&np.isfinite(fwd)
        pw=L.predict_proba(Xi[m])[:,1]; fww=fwd[m]; tsw=ts[m]
        oo=np.argsort(tsw); pw=pw[oo]; fww=fww[oo]; tsw=tsw[oo]
        mv=fww!=0; auc=float(roc_auc_score((fww[mv]>0).astype(int),pw[mv]))
        res["years"][w]={"moved_auc":round(auc,4),"bycov":{}}
        print(f"=== {w} === inv frozen moved-AUC={auc:.4f}",flush=True)
        for cov in COVS:
            r=XF.side_eval(pw,fww,tsw,gates[f"{cov:.2f}"]["thr"])
            res["years"][w]["bycov"][f"{cov:.2f}"]=r
            if r: print(f"   cov{cov}: COMB {r['COMBINED']['wr']} (full {full['years'][w]['bycov'][f'{cov:.2f}']['COMBINED']['wr']}) | UP {r['UP'].get('wr')} | DOWN {r['DOWN'].get('wr')}",flush=True)
    # verdict: 2026 cov1 COMB materially above full-337 on BOTH sides
    inv26=res["years"]["oos"]["bycov"]["0.01"]
    up_better = (inv26["UP"].get("wr") or 0) >= (inc_2026_cov1["UP"] or 0)+0.02
    dn_better = (inv26["DOWN"].get("wr") or 0) >= (inc_2026_cov1["DOWN"] or 0)+0.02
    survives = bool(prune_frac>=0.10 and up_better and dn_better)
    res["verdict"]={"SURVIVES":survives,"prune_frac":round(float(prune_frac),3),
                    "inv_2026_cov1":{"UP":inv26["UP"].get("wr"),"DOWN":inv26["DOWN"].get("wr"),"COMB":inv26["COMBINED"]["wr"]},
                    "full_2026_cov1":inc_2026_cov1,"up_better_by_2pt":up_better,"down_better_by_2pt":dn_better,
                    "note":"SURVIVES only if pruning era-local-sign features MATERIALLY reduces the 2026 decay on BOTH sides; else recent weakness is genuine signal/info-bound."}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[irm] VERDICT SURVIVES={survives} (prune {prune_frac:.1%}; inv 2026 cov1 UP {inv26['UP'].get('wr')}/DOWN {inv26['DOWN'].get('wr')} vs full {inc_2026_cov1['UP']}/{inc_2026_cov1['DOWN']}) -> {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
