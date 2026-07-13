"""USDCHF 15-MIN — adversarial FROZEN-PAST forward holdout of the XPAIR-NY book (trap#9 check).

SCOPE: USDCHF · 15m. The xpair-NY refit-CPCV is the new incumbent (BOTH sides certified every cov;
UP @cov1 p10 .6515 ≥ .65). This harness answers the MANDATORY adversarial question: is that a
deployable frozen edge, refit-dependent decay (sibling-universal USD-factor non-stationarity), or
era-local memorization (trap#9 — would look ~.50 flat in EVERY forward year)?

Train ONCE on 2012-21 NY xpbase matrix (stride 3, optional K-seed avg), FREEZE, evaluate per-year
2024/2025/2026 NY at cov gates frozen on VAL (2022-23 NY) by worst-half. Ties-strict, nonoverlap 900s,
bootstrap CI95. Compare per-year frozen WR vs the refit-CPCV p10 (UP .5962/.6515, DOWN .5908/.6229 @cov5/1):
  - frozen ≈ refit in 2024-25 then decays      -> REFIT-DEPENDENT (deploy w/ retrain; expected)
  - frozen ~.50 ALL years                       -> era-local memorization (DOWNGRADE the cert)
  - frozen ≥ refit everywhere                   -> frozen-deployable (upgrade)

Usage: ~/binary-algo-venv/bin/python usdchf_15m_xpair_frozen.py [stride=3] [nseed=1]
"""
import os, sys, json, time, numpy as np, pandas as pd
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sessions import session_mask
import harness as H
import usdchf_15m_xpair as XP

HOR=15; GAP=HOR*60; BE=0.541; NUM_LEAVES=255
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],"test24":["2024"],"test25":["2025"],"oos":["2026"]}
COVS=(0.05,0.03,0.02,0.01)
REPO_ROOT=Path(__file__).resolve().parents[1]
RESULTS_JSON=REPO_ROOT/"results"/"json"

def build_mat(years, stride):
    """xpbase matrix on the joint clock, ties kept; memory-sane per-year base join."""
    F=XP.build_xp(years, stride, "xpbase", keep_ties=True)
    xpc=XP.xp_cols(F)
    base_cols=[c for c in H.feature_cols("USDCHF") if c not in F.columns]
    idx=F.index; yr=idx.year.values
    Xb=np.full((len(F), len(base_cols)), np.nan, dtype="float32")
    for y in years:
        m=yr==int(y)
        if not m.any(): continue
        p=f"{XP.FEAT}/USDCHF_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p, columns=base_cols); d=d[~d.index.duplicated(keep="last")]
        Xb[np.where(m)[0]]=d.reindex(idx[m]).values.astype("float32")
        del d
    X=np.concatenate([F[xpc].values.astype("float32"), Xb], axis=1); del Xb
    fwd=F["_fwd"].values; ts=F["_ts"].values.astype("int64"); del F
    if len(ts)>1 and np.any(ts[1:]<=ts[:-1]):
        raise ValueError("USDCHF xpair rows must be strictly increasing and unique")
    return X, fwd, ts, list(xpc)+base_cols

def nonoverlap_chrono(ts, mask, gap=GAP):
    take=[]; block=-1
    for i in np.where(mask)[0]:
        if ts[i]<block: continue
        take.append(i); block=int(ts[i])+gap
    return np.array(take,dtype=int)

def boot(w, nb=5000, seed=7):
    w=np.asarray(w,float)
    if len(w)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(w)
    a=np.array([w[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))

def side_eval(pr, fwd, ts, thr):
    conf=np.abs(pr-0.5); tr=nonoverlap_chrono(ts, (conf>=thr)&np.isfinite(fwd))
    if len(tr)==0: return None
    pred=(pr[tr]>0.5).astype(int); ylab=(fwd[tr]>0).astype(int); moved=(fwd[tr]!=0)
    win=((pred==ylab)&moved).astype(float); out={}
    lo,hi=boot(win); out["COMBINED"]={"n":int(len(tr)),"wr":round(float(win.mean()),4),"ci":[round(lo,3),round(hi,3)]}
    for nm,msk in (("UP",pred==1),("DOWN",pred==0)):
        if msk.sum()>0:
            lo,hi=boot(win[msk]); out[nm]={"n":int(msk.sum()),"wr":round(float(win[msk].mean()),4),"ci":[round(lo,3),round(hi,3)]}
        else: out[nm]={"n":0}
    return out

def main(stride=3, nseed=1):
    t0=time.time()
    with (RESULTS_JSON/"usdchf_15m_cpcv_xpair_ny_multicov_result.json").open() as fh:
        xinc=json.load(fh)
    RESULT=f"usdchf_15m_xpair_frozen_result.json" if nseed==1 else f"usdchf_15m_xpair_frozen_seedens{nseed}_result.json"
    res={"key":"USDCHF.15m","model":f"FROZEN-2012-21 xpair-NY book (xpbase, stride {stride}, nseed {nseed}), per-year forward NY",
         "breakeven":BE,"purpose":"trap#9 adversarial: refit-dependent decay vs era-local memorization vs frozen-deployable",
         "refit_cpcv_cert":{"UP_p10":{c:xinc["bycov"][c]["summary"]["UP"]["p10"] for c in xinc["bycov"]},
                            "DOWN_p10":{c:xinc["bycov"][c]["summary"]["DOWN"]["p10"] for c in xinc["bycov"]}},
         "falsifier":{"registered":"pre-eval",
            "MEMORIZATION_if":"all forward years ~coin-flip (no year COMBINED wr >= 0.55 at any cov)",
            "REFIT_DEPENDENT_if":"2024 strong then monotone decay to sub-BE by 2026 (sibling pattern)"}}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[xpfrozen USDCHF] building (stride {stride}, nseed {nseed})...",flush=True)
    # build once over all years then slice (joint clock identical to cert harness)
    allyears=[y for k in ("train","val","test24","test25","oos") for y in SPL[k]]
    X,fwd,ts,cols=build_mat(allyears,stride)
    ny=session_mask(ts,"ny")
    yr=pd.to_datetime(ts,unit="s").year.values
    moved=np.isfinite(fwd)&(fwd!=0)
    def yrmask(years): return np.isin(yr,[int(y) for y in years])
    itr=yrmask(SPL["train"])&ny&moved
    iva=yrmask(SPL["val"])&ny&np.isfinite(fwd)
    print(f"[xpfrozen USDCHF] rows={len(ts):,} train={int(itr.sum()):,} val={int(iva.sum()):,} feats={len(cols)} build={time.time()-t0:.0f}s",flush=True)
    pva=np.zeros(int(iva.sum()))
    for sd in range(nseed):
        L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=NUM_LEAVES,
            min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,
            n_estimators=800,n_jobs=16,verbosity=-1,random_state=sd,bagging_seed=sd,feature_fraction_seed=sd)
        L.fit(X[itr],(fwd[itr]>0).astype(int),eval_set=[(X[iva],(fwd[iva]>0).astype(int))],eval_metric="auc",
              callbacks=[lgb.early_stopping(80),lgb.log_evaluation(0)])
        pva+=L.predict_proba(X[iva])[:,1]
        if sd==0: models=[L]
        else: models.append(L)
    pva/=nseed
    mvv=(fwd[iva]!=0)
    vauc=float(roc_auc_score((fwd[iva][mvv]>0).astype(int), pva[mvv]))
    # gate per cov on VAL worst-half (chronological halves of the NY val rows)
    tsv=ts[iva]; fwv=fwd[iva]
    oo=np.argsort(tsv); tsv=tsv[oo]; fwv=fwv[oo]; pvas=pva[oo]
    half=len(pvas)//2; confv=np.abs(pvas-0.5)
    gates={}
    for cov in COVS:
        thr=float(np.quantile(confv,1-cov))
        worst=np.nanmin([ (lambda r: r["COMBINED"]["wr"] if r else np.nan)(side_eval(pvas[s:e],fwv[s:e],tsv[s:e],thr))
                          for s,e in ((0,half),(half,len(pvas))) ])
        gates[f"{cov:.2f}"]={"thr":thr,"val_worst_half":round(float(worst),4)}
    res["val_auc"]=round(vauc,4); res["gates"]=gates
    print(f"[xpfrozen USDCHF] VAL moved-AUC={vauc:.4f} gates={ {k:round(v['thr'],4) for k,v in gates.items()} }",flush=True)
    res["years"]={}
    for w in ("test24","test25","oos"):
        m=yrmask(SPL[w])&ny&np.isfinite(fwd)
        pw=np.zeros(int(m.sum()))
        for L in models: pw+=L.predict_proba(X[m])[:,1]
        pw/=nseed
        fww=fwd[m]; tsw=ts[m]
        oo=np.argsort(tsw); pw=pw[oo]; fww=fww[oo]; tsw=tsw[oo]
        mv=fww!=0
        auc=float(roc_auc_score((fww[mv]>0).astype(int),pw[mv]))
        res["years"][w]={"moved_auc":round(auc,4),"up_rate":round(float((fww[mv]>0).mean()),4),"bycov":{}}
        print(f"=== {w} === frozen moved-AUC={auc:.4f}",flush=True)
        for cov in COVS:
            r=side_eval(pw,fww,tsw,gates[f"{cov:.2f}"]["thr"])
            res["years"][w]["bycov"][f"{cov:.2f}"]=r
            if r: print(f"   cov{cov}: COMB n{r['COMBINED']['n']} {r['COMBINED']['wr']} CI{r['COMBINED']['ci']} | UP {r['UP'].get('wr')} | DOWN {r['DOWN'].get('wr')}",flush=True)
    # verdict
    any_strong=any(res["years"][w]["bycov"][c] and res["years"][w]["bycov"][c]["COMBINED"]["wr"]>=0.55
                   for w in res["years"] for c in res["years"][w]["bycov"])
    res["verdict"]={"memorization": not any_strong,
                    "note":"if not memorization: compare decay profile to refit p10 -> refit-dependent (expected) or frozen-deployable"}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[xpfrozen USDCHF] memorization={not any_strong} -> {RESULT}  {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    cli_stride=int(sys.argv[1]) if len(sys.argv)>1 else 3
    cli_nseed=int(sys.argv[2]) if len(sys.argv)>2 else 1
    main(cli_stride,cli_nseed)
