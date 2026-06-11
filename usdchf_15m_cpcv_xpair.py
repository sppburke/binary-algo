"""USDCHF 15-MIN — NY-session per-fold-refit CPCV on the CROSS-PAIR (xpbase) feature matrix.

SCOPE: USDCHF · 15m (A6 escalation — the v2 EUR-bloc prediction's decisive test). USDCHF certified
on ALL-SESSION (EUR-bloc breadth signature, unlike the own-pair havens) AND NY concentrates it far
higher (NY AUC .5401, cov2 p10 UP .6221/DOWN .6161). The question: does the EUR-bloc cross-pair POOLED
feature matrix LIFT the NY refit-CPCV p10 over the OWN-PAIR NY incumbent — i.e. is USDCHF the EUR-bloc
case (pooling ADDS, like EURUSD/GBPUSD) or the own-pair case (pooling dilutes, like USDJPY/AUDUSD/USDCAD)?
Run only AFTER the all-session xpair SCREEN (usdchf_15m_xpair.py xpbase) lifts base; this is the
escalation CERT harness.

Harness identical to usdchf_15m_cpcv_session.py (6 groups, k=2, purge=embargo=900s, 15 paths,
SUB_FIT 150k, NY DST-correct decision bars, ties-STRICT charged as losses, nonoverlap gap=900) —
only the feature matrix differs: XP.build_xp(keep_ties=True) + base-239 augment ≈ 100 xp + 239 base.
Stride 3 default (~341 float32 cols; keeps the matrix ~2.5GB, OOM discipline).

IMPROVES iff p10 > incumbent p10 at matched cov on BOTH sides; CERT rule unchanged.
Usage: ~/binary-algo-venv/bin/python usdchf_15m_cpcv_xpair.py [stride=3] [covs=0.05,0.03,0.02,0.01] [nseed=1]
"""
import os, sys, json, time, numpy as np, pandas as pd
from itertools import combinations
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sessions import session_mask
import usdchf_15m_xpair as XP

HOR=15; GAP=HOR*60; BE=0.541
N_GROUPS,K_TEST=6,2; SUB_FIT=150_000; PURGE=GAP; EMBARGO=GAP; NUM_LEAVES=255
STRIDE=int(sys.argv[1]) if len(sys.argv)>1 else 3
COVS=[float(x) for x in (sys.argv[2] if len(sys.argv)>2 else "0.05,0.03,0.02,0.01").split(",")]
NSEED=int(sys.argv[3]) if len(sys.argv)>3 else 1
ALL_YEARS=[str(y) for y in range(2012,2027)]
INC=json.load(open("usdchf_15m_cpcv_session_ny_multicov_result.json"))
INC_P10={c:{s:INC["bycov"][c]["summary"][s]["p10"] for s in ("UP","DOWN","COMBINED")} for c in INC["bycov"]}

def mk_lgb(n=800, seed=0):
    return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=NUM_LEAVES,
        min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,
        n_estimators=n,n_jobs=16,verbosity=-1,random_state=seed,bagging_seed=seed,feature_fraction_seed=seed)

def nonoverlap_chrono(ts, mask, gap=GAP):
    take=[]; block=-1
    for i in np.where(mask)[0]:
        if ts[i]<block: continue
        take.append(i); block=int(ts[i])+gap
    return np.array(take,dtype=int)

def side_wr(pr, fwd, ts, thr, side):
    conf=np.abs(pr-0.5)
    want=(pr>0.5) if side=="UP" else (pr<0.5) if side=="DOWN" else np.ones(len(pr),bool)
    cand=want & (conf>=thr) & np.isfinite(fwd)
    tr=nonoverlap_chrono(ts, cand)
    if len(tr)==0: return 0, float("nan")
    pred=(pr[tr]>0.5).astype(int); ylab=(fwd[tr]>0).astype(int); moved=(fwd[tr]!=0)
    win=((pred==ylab) & moved).astype(float)
    return len(tr), float(win.mean())

def summ(a):
    a=np.asarray([x for x in a if np.isfinite(x)],float)
    if len(a)==0: return {"n_paths":0}
    return {"n_paths":int(len(a)),"mean":round(float(a.mean()),4),"p10":round(float(np.percentile(a,10)),4),
            "p50":round(float(np.percentile(a,50)),4),"min":round(float(a.min()),4),
            "frac_clear_BE":round(float((a>=BE).mean()),3)}

def main():
    t0=time.time()
    RESULT=(f"usdchf_15m_cpcv_xpair_ny_multicov_result.json" if NSEED==1
            else f"usdchf_15m_cpcv_xpair_ny_seedens{NSEED}_result.json")
    print(f"[cpcv-xpair/ny] building xpbase matrix (stride {STRIDE}) covs={COVS} nseed={NSEED}...", flush=True)
    F=XP.build_xp(ALL_YEARS, STRIDE, "xpbase", keep_ties=True)
    xpc=XP.xp_cols(F)
    # memory-sane augment: per-year float32 aligned join of the 239 base feats onto F's strided
    # joint-clock index (XP.augment concats 15yr of UNSTRIDED float64 parquets ~10GB -> OOM-killed).
    import harness as H
    base_cols=[c for c in H.feature_cols("USDCHF") if c not in F.columns]
    idx=F.index; yr_arr=idx.year.values
    Xb=np.full((len(F), len(base_cols)), np.nan, dtype="float32")
    for y in ALL_YEARS:
        msk=yr_arr==int(y)
        if not msk.any(): continue
        p=f"{XP.FEAT}/USDCHF_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p, columns=base_cols); d=d[~d.index.duplicated(keep="last")]
        Xb[np.where(msk)[0]]=d.reindex(idx[msk]).values.astype("float32")
        del d
    cols=list(xpc)+base_cols
    X=np.concatenate([F[xpc].values.astype("float32"), Xb], axis=1)
    del Xb
    fwd=F["_fwd"].values; ts=F["_ts"].values.astype("int64")
    del F
    o=np.argsort(ts); X=X[o]; fwd=fwd[o]; ts=ts[o]
    sess=session_mask(ts, "ny")
    moved=np.isfinite(fwd)&(fwd!=0.0)
    print(f"[cpcv-xpair/ny] rows={len(ts):,} in-NY={int(sess.sum()):,} feats={len(cols)} built {time.time()-t0:.0f}s", flush=True)

    tgs=ts[sess]; bnds=[tgs[int(k*len(tgs)/N_GROUPS)] for k in range(N_GROUPS)]+[tgs[-1]+1]
    groups=[(int(bnds[g]), int(bnds[g+1])) for g in range(N_GROUPS)]

    res={"model":f"XPAIR(xpbase)-GBM @15m, SESSION=ny, TIES-STRICT per-fold-REFIT CPCV (multi-cov)","key":"USDCHF.15m",
         "breakeven":BE,"stride":STRIDE,"covs":COVS,"nseed":NSEED,"n_feats":len(cols),
         "incumbent":"own-pair NY refit-CPCV (usdchf_15m_cpcv_session_ny_multicov_result.json)",
         "incumbent_p10":INC_P10,
         "CERT_RULE":"side CERTIFIED iff p10>=0.541 AND frac_clear_BE>=0.80",
         "falsifier":{"registered":"pre-paths",
            "IMPROVES_if":"p10 > incumbent own-pair NY p10 at matched cov on BOTH sides (cov0.02 UP .6221 / DOWN .6161; cov1 UP .6413 / DOWN .6361); else NON-ADDITIVE (own-pair book stands = own-pair-specific haven case)",
            "KILL_if":"p10<0.541 OR frac_clear_BE<0.80 per side"}}
    json.dump(res, open(RESULT,"w"), indent=2)

    rng=np.random.default_rng(13)
    paths={c:{"UP":[],"DOWN":[],"COMBINED":[]} for c in COVS}
    npaths={c:{"UP":[],"DOWN":[],"COMBINED":[]} for c in COVS}
    aucs=[]; uprates=[]; pathinfo=[]
    for fi,testg in enumerate(combinations(range(N_GROUPS), K_TEST)):
        tblocks=[groups[g] for g in testg]
        tin=np.zeros(len(ts),bool)
        for lo,hi in tblocks: tin |= (ts>=lo)&(ts<hi)
        test_mask=tin & sess & np.isfinite(fwd)
        purged=np.zeros(len(ts),bool)
        for lo,hi in tblocks: purged |= (ts>=lo-PURGE)&(ts<hi+EMBARGO)
        train_mask=(~purged) & moved & sess
        tr_idx=np.where(train_mask)[0]
        valsel=rng.random(len(tr_idx))<0.15
        val_idx=tr_idx[valsel]; fit_pool=tr_idx[~valsel]
        if len(fit_pool)>SUB_FIT: fit_pool=rng.choice(fit_pool, SUB_FIT, replace=False)
        ytr=(fwd[fit_pool]>0).astype(int); yval=(fwd[val_idx]>0).astype(int)
        pval=np.zeros(len(val_idx)); ptst=np.zeros(int(test_mask.sum()))
        for sd in range(NSEED):
            L=mk_lgb(seed=sd)
            L.fit(X[fit_pool], ytr, eval_set=[(X[val_idx], yval)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
            pval+=L.predict_proba(X[val_idx])[:,1]; ptst+=L.predict_proba(X[test_mask])[:,1]
        pval/=NSEED; ptst/=NSEED
        vconf=np.abs(pval-0.5)
        ftst=fwd[test_mask]; ttst=ts[test_mask]
        oo=np.argsort(ttst); ptst=ptst[oo]; ftst=ftst[oo]; ttst=ttst[oo]
        mv=ftst!=0
        auc=float(roc_auc_score((ftst[mv]>0).astype(int), ptst[mv])) if mv.sum()>20 else float("nan")
        uprate=float((ftst[mv]>0).mean()) if mv.sum()>0 else float("nan")
        aucs.append(auc); uprates.append(uprate)
        rowinfo={"fold":list(testg),"auc":round(auc,4),"up_rate":round(uprate,4),"bycov":{}}
        for c in COVS:
            thr=float(np.quantile(vconf, 1-c))
            nC,wC=side_wr(ptst,ftst,ttst,thr,"COMBINED"); nU,wU=side_wr(ptst,ftst,ttst,thr,"UP"); nD,wD=side_wr(ptst,ftst,ttst,thr,"DOWN")
            if nC>=25: paths[c]["COMBINED"].append(wC); npaths[c]["COMBINED"].append(nC)
            if nU>=25: paths[c]["UP"].append(wU); npaths[c]["UP"].append(nU)
            if nD>=25: paths[c]["DOWN"].append(wD); npaths[c]["DOWN"].append(nD)
            rowinfo["bycov"][f"{c}"]={"UP":[nU,round(wU,4)],"DOWN":[nD,round(wD,4)],"COMBINED":[nC,round(wC,4)],"thr":round(thr,4)}
        pathinfo.append(rowinfo)
        bc=rowinfo["bycov"][f"{COVS[0]}"]
        print(f"  path {fi+1}/15 g{list(testg)} AUC={auc:.4f} up={uprate:.4f} | @cov{COVS[0]}: UP n{bc['UP'][0]} {bc['UP'][1]} | DOWN n{bc['DOWN'][0]} {bc['DOWN'][1]} | COMB n{bc['COMBINED'][0]} {bc['COMBINED'][1]} ({time.time()-t0:.0f}s)", flush=True)

    res["paths"]=pathinfo
    res["auc_summary"]={"mean":round(float(np.nanmean(aucs)),4),"min":round(float(np.nanmin(aucs)),4),"max":round(float(np.nanmax(aucs)),4)}
    res["uprate_tripwire_ok"]=bool(np.all([(0.45<=u<=0.55) for u in uprates if np.isfinite(u)]))
    res["bycov"]={}
    for c in COVS:
        sm={s:summ(paths[c][s]) for s in ("UP","DOWN","COMBINED")}
        med_n={s:(int(np.median(npaths[c][s])) if npaths[c][s] else 0) for s in ("UP","DOWN","COMBINED")}
        vd={}
        for s in ("UP","DOWN","COMBINED"):
            p10=sm[s].get("p10",float("nan")); fc=sm[s].get("frac_clear_BE",0.0)
            inc=INC_P10.get(f"{c}",{}).get(s)
            vd[s]={"p10":p10,"frac_clear_BE":fc,"med_n_per_path":med_n[s],
                   "CERTIFIED":bool(np.isfinite(p10) and p10>=BE and fc>=0.80),
                   "incumbent_p10":inc,"BEATS_incumbent":bool(inc is not None and np.isfinite(p10) and p10>inc)}
        res["bycov"][f"{c}"]={"summary":sm,"verdict":vd}
    json.dump(res, open(RESULT,"w"), indent=2)
    print(f"\n[cpcv-xpair/ny] === MULTI-COV SUMMARY (vs own-pair incumbent) ===", flush=True)
    for c in COVS:
        print(f" cov{c}:", flush=True)
        for s in ("UP","DOWN","COMBINED"):
            v=res["bycov"][f"{c}"]["verdict"][s]; sm=res["bycov"][f"{c}"]["summary"][s]
            print(f"   {s}: p10={v['p10']} (inc {v['incumbent_p10']}, beats={v['BEATS_incumbent']}) mean={sm.get('mean')} frac={v['frac_clear_BE']} med_n={v['med_n_per_path']} CERT={v['CERTIFIED']}", flush=True)
    print(f"  AUC {res['auc_summary']} trip={res['uprate_tripwire_ok']}  done {time.time()-t0:.0f}s -> {RESULT}", flush=True)

if __name__=="__main__":
    main()
