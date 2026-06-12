"""USDJPY 15-MIN — refit-CPCV of the NY edge with a TRIPLE-BARRIER FIRST-TOUCH train label.

SCOPE: USDJPY · 15m · NY session. The frozen-past screen (usdjpy_15m_tbfirsttouch.py) showed the
path-aware first-touch train label (k=2.0) LIFTS VAL fixed-15m AUC .5313 -> .5439 and its NY cov3%
win-rate CI-lo clears BE in test24+test25 (survived the falsifier). It is the ONLY lever that beat
base VAL-AUC, so per the IMPROVE mandate it must face the SAME refit-CPCV that certified the base book
before we believe or dismiss it.

This is the certified NY refit-CPCV (usdjpy_15m_cpcv_session.py ny) with EXACTLY ONE change: the per-fold
TRAIN target is the triple-barrier first-touch label (upper/lower = +/-k*sigma_t, vertical=15m timeout ->
endpoint sign; sigma_t = trailing causal realized 1-bar vol). EVERYTHING ELSE is identical and deriv-faithful:
  * VAL early-stopping label, test win-rate, and the moved-AUC are the UNCHANGED fixed-15m sign label,
    ties (move==0) LOSE, BE 0.541, gap=900 nonoverlap_chrono, NY decision rows.
  * k=2.0 was frozen from the frozen-past screen, where it was picked on VAL (2022-23) fixed-15m AUC
    across a 5-value grid {0.5,0.75,1.0,1.5,2.0}. CAVEAT (corrected 2026-06-08 after adversarial verify):
    this is NOT "train-agreement, no leak" — the VAL years 2022-23 appear inside some CPCV folds, so the
    single scalar k is mildly informed by data that later serves as test in those folds. It is a fixed
    scalar (not re-tuned per fold), so the contamination is minor (1 of 5 grid points on an aggregate
    metric) and, critically, it would only INFLATE TB's apparent edge — yet TB still does NOT beat the
    seed-ens incumbent (see below), so the caveat only strengthens the negative verdict. A fully clean
    design would re-freeze k inside each fold's train.
  * where the TB label is undefined (early rows / sigma<=0 / non-contig) the fit row falls back to the
    fixed-15m sign so the train pool is identical to the base CPCV (only the label content differs).

INCUMBENT to beat: base single-model NY CPCV UP p10 .586 / DOWN p10 .572 @cov3%; seed-ens K=3
UP .6005 / DOWN .5738 @cov2%. SURVIVE-AS-IMPROVEMENT iff a side's p10 strictly exceeds the base
single-model incumbent at matched cov AND frac_clear_BE>=0.80.

Usage: ~/binary-algo-venv/bin/python usdjpy_15m_cpcv_tbfirsttouch.py [stride=2] [cov=0.02,0.03] [nseed=1]
"""
import os, sys, json, time, numpy as np, pandas as pd
from itertools import combinations
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from sessions import session_mask

TARGET="USDJPY"; HOR=15; STEP=60; GAP=HOR*STEP; BE=0.541
FEAT=H.FEAT_DIR; FEATS=H.feature_cols(TARGET)
YEARS=list(range(2012,2027))
N_GROUPS, K_TEST = 6, 2
SUB_FIT=150_000
PURGE=GAP; EMBARGO=GAP
NUM_LEAVES=255
VOL_WIN=900; VOL_MINP=60          # trailing realized-vol window for sigma_t (causal) — matches tbfirsttouch
TB_K=2.0                          # TRAIN-frozen barrier multiplier (chosen_k from the frozen-past screen)
SESSION="ny"

def _arg(i,d,cast):
    rest=[a for a in sys.argv[1:]]
    return cast(rest[i]) if len(rest)>i else d
STRIDE=_arg(0,2,int)
_covarg=_arg(1,"0.02,0.03",str); COVS=[float(x) for x in str(_covarg).split(",")]
NSEED=_arg(2,1,int)

def _first_touch_label(c, fr, contig, k):
    """VECTORIZED triple-barrier first-touch label for a contiguous per-year close series `c`.
    +1 if upper (+k*sigma_t) touched strictly before lower within the forward HOR window; 0 if lower
    first; timeout/tie -> endpoint sign(fr). nan where non-contig / sigma undefined / fr non-finite.
    sigma_t = trailing rolling std of 1-bar returns (causal)."""
    n=len(c)
    r1=np.full(n,np.nan); r1[1:]=c[1:]/c[:-1]-1.0
    sig=pd.Series(r1).rolling(VOL_WIN, min_periods=VOL_MINP).std().values   # past-only sigma at bar t
    cpad=np.concatenate([c, np.full(HOR,np.nan)])
    win=np.lib.stride_tricks.sliding_window_view(cpad, HOR)
    fwd=win[1:n+1]                                                          # c[t+1 .. t+HOR]
    c0=c[:,None]
    cummax=np.maximum.accumulate(fwd,axis=1); cummin=np.minimum.accumulate(fwd,axis=1)
    end_sign=(fr>0).astype("float64")
    up=c0*(1.0+k*sig[:,None]); dn=c0*(1.0-k*sig[:,None])
    up_hit=cummax>=up; dn_hit=cummin<=dn
    tu=np.where(up_hit.any(axis=1), up_hit.argmax(axis=1), HOR)
    td=np.where(dn_hit.any(axis=1), dn_hit.argmax(axis=1), HOR)
    ylab=end_sign.copy(); ylab[tu<td]=1.0; ylab[td<tu]=0.0
    bad=(~contig)|(~np.isfinite(sig))|(sig<=0)|(~np.isfinite(fr))
    ylab=ylab.astype("float64"); ylab[bad]=np.nan
    return ylab

def build_pair_ties(pair, stride):
    """Returns X, fwd(fixed-15m fwd ret = EVAL label src), ts, tb (first-touch TRAIN label, k=TB_K)."""
    Xs=[]; fwds=[]; tss=[]; tbs=[]
    for y in YEARS:
        p=f"{FEAT}/{pair}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p, columns=FEATS+["close"]); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float)
        ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        contig=np.zeros(n,bool); contig[:n-HOR]=(ts[HOR:]-ts[:-HOR])==GAP
        fr=np.full(n,np.nan); fr[:n-HOR]=c[HOR:]/c[:-HOR]-1.0
        tb=_first_touch_label(c, fr, contig, TB_K)                         # per-year, aligned to full series
        X=d[FEATS].astype("float32"); keepf=X.isna().mean(axis=1).values<0.5
        valid=contig & np.isfinite(fr) & keepf
        idx=np.where(valid)[0]
        if stride>1: idx=idx[::stride]
        Xs.append(X.values[idx]); fwds.append(fr[idx]); tss.append(ts[idx]); tbs.append(tb[idx])
    if not Xs: return None
    return np.concatenate(Xs), np.concatenate(fwds), np.concatenate(tss), np.concatenate(tbs)

def mk_lgb(n=800, num_leaves=NUM_LEAVES, seed=0):
    return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=num_leaves,
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
    RESULT=(f"usdjpy_15m_cpcv_tbfirsttouch_{SESSION}_multicov_result.json" if NSEED==1
            else f"usdjpy_15m_cpcv_tbfirsttouch_{SESSION}_seedens{NSEED}_result.json")
    print(f"[cpcv-tbft/{SESSION}] building USDJPY (stride {STRIDE}) covs={COVS} TB_K={TB_K} nseed={NSEED}...", flush=True)
    X,fwd,ts,tb=build_pair_ties(TARGET, STRIDE)
    o=np.argsort(ts); X=X[o]; fwd=fwd[o]; ts=ts[o]; tb=tb[o]
    sess=session_mask(ts, SESSION)
    moved=np.isfinite(fwd) & (fwd!=0.0)
    tb_def=np.isfinite(tb)
    # fraction of moved in-session rows where TB label actually differs from the fixed-15m sign
    msf=moved & sess & tb_def
    flip=float((tb[msf]!=(fwd[msf]>0).astype(int)).mean()) if msf.sum() else float("nan")
    print(f"[cpcv-tbft/{SESSION}] USDJPY={len(ts):,} rows  in-session={int(sess.sum()):,}  "
          f"TB-defined={int(tb_def.sum()):,}  TB!=sign frac(moved,NY)={flip:.4f}  built {time.time()-t0:.0f}s", flush=True)

    tgs=ts[sess]; bnds=[tgs[int(k*len(tgs)/N_GROUPS)] for k in range(N_GROUPS)]+[tgs[-1]+1]
    groups=[(int(bnds[g]), int(bnds[g+1])) for g in range(N_GROUPS)]

    res={"model":f"OWN-PAIR base-GBM @15m, SESSION={SESSION}, TRAIN-LABEL=triple-barrier first-touch k={TB_K}, "
                 f"EVAL=fixed-15m deriv sign (ties LOSE), TIES-STRICT per-fold-REFIT CPCV (multi-cov)",
         "key":"USDJPY.15m","breakeven":BE,"session":SESSION,"train_label":f"tb_first_touch_k{TB_K}",
         "tb_flip_frac_moved_ny":round(flip,4),"stride":STRIDE,"covs":COVS,"n_groups":N_GROUPS,"k_test":K_TEST,"nseed":NSEED,
         "incumbent":"base single-model NY CPCV UP p10 .586 / DOWN .572 @cov3%; seedensK3 UP .6005 / DOWN .5738 @cov2%",
         "CERT_RULE":"side CERTIFIED iff p10>=0.541 AND frac_clear_BE>=0.80; IMPROVEMENT iff p10>base-incumbent at matched cov",
         "falsifier":{"registered_utc":"pre-paths",
                      "KILL_as_improvement_if":"no side's p10 strictly exceeds the base single-model incumbent (UP .586/DOWN .572 @cov3%) at matched cov"}}
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
        # TRAIN label = triple-barrier first-touch where defined, else fixed-15m sign (pool unchanged vs base)
        ytr=np.where(tb_def[fit_pool], tb[fit_pool], (fwd[fit_pool]>0).astype(int)).astype(int)
        yval=(fwd[val_idx]>0).astype(int)                  # EVAL/early-stop label stays deriv-faithful
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
    # MATCHED incumbent (corrected after adversarial verify): compare K=3 vs the K=3 seed-ens base, and
    # single-seed vs the single-model base. NOTE the p10-vs-fixed-incumbent flag below is WEAK — the proper
    # test is PAIRED per-path (tb fold i vs base fold i), since CPCV p10 is a single noisy order statistic.
    # The paired test (run externally) shows TB does NOT robustly beat the matched seed-ens base.
    INC=({"UP":0.6005,"DOWN":0.5738,"COMBINED":0.5962} if NSEED>=3      # matched seed-ens K3 @cov2%
         else {"UP":0.586,"DOWN":0.572,"COMBINED":0.580})              # matched single-model @cov3%
    res["incumbent_matched_to_nseed"]=INC
    res["bycov"]={}
    for c in COVS:
        sm={s:summ(paths[c][s]) for s in ("UP","DOWN","COMBINED")}
        med_n={s:(int(np.median(npaths[c][s])) if npaths[c][s] else 0) for s in ("UP","DOWN","COMBINED")}
        vd={}
        for s in ("UP","DOWN","COMBINED"):
            p10=sm[s].get("p10",float("nan")); fc=sm[s].get("frac_clear_BE",0.0)
            vd[s]={"p10":p10,"frac_clear_BE":fc,"med_n_per_path":med_n[s],
                   "CERTIFIED":bool(np.isfinite(p10) and p10>=BE and fc>=0.80),
                   "p10_gt_matched_incumbent_WEAK":bool(np.isfinite(p10) and p10>INC[s] and fc>=0.80)}
        res["bycov"][f"{c}"]={"summary":sm,"verdict":vd}
    json.dump(res, open(RESULT,"w"), indent=2)
    print(f"\n[cpcv-tbft/{SESSION}] === MULTI-COV SUMMARY (TB_K={TB_K}, flip={flip:.3f}) ===", flush=True)
    for c in COVS:
        print(f" cov{c}:", flush=True)
        for s in ("UP","DOWN","COMBINED"):
            v=res["bycov"][f"{c}"]["verdict"][s]; sm=res["bycov"][f"{c}"]["summary"][s]
            print(f"   {s}: p10={v['p10']} mean={sm.get('mean')} frac={v['frac_clear_BE']} med_n={v['med_n_per_path']} "
                  f"-> CERT={v['CERTIFIED']} p10>matched_inc(WEAK)={v['p10_gt_matched_incumbent_WEAK']}", flush=True)
    print(f"  AUC {res['auc_summary']} trip={res['uprate_tripwire_ok']}  done {time.time()-t0:.0f}s -> {RESULT}", flush=True)

if __name__=="__main__":
    main()
