"""USDJPY 15-MIN — own-pair refit-CPCV restricted to a SESSION (certify the NY-concentrated edge).

SCOPE: USDJPY · 15m. A9 single-fit showed the NY session lifts conditional AUC (.531->.544) and the
COMBINED/UP win-rate (NY UP ~.58 across ALL years incl 2026; COMBINED CI-lo clears 2024+2025). This
certifies the SESSION-concentrated own-pair edge: train + test DECISION bars restricted to the session
(features stay causal/continuous), per-fold REFIT, ties-strict, nonoverlap gap=900.

CERTIFY a side iff per-side p10 >= 0.541 AND >= ~80% paths clear 0.541.
Usage: ~/binary-algo-venv/bin/python usdjpy_15m_cpcv_session.py <ny|ldn|asia> [stride=2] [cov=0.03] [nseed=1]
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
SESSION=sys.argv[1] if len(sys.argv)>1 and sys.argv[1] in ("ny","ldn","asia","all") else "ny"
def _arg(i,d,cast):
    rest=[a for a in sys.argv[2:]]
    return cast(rest[i]) if len(rest)>i else d
STRIDE=_arg(0,2,int)
_covarg=_arg(1,"0.03",str); COVS=[float(x) for x in str(_covarg).split(",")]
NSEED=_arg(2,1,int)

def build_pair_ties(pair, stride):
    Xs=[]; fwds=[]; tss=[]
    for y in YEARS:
        p=f"{FEAT}/{pair}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p, columns=FEATS+["close"]); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float)
        ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        contig=np.zeros(n,bool); contig[:n-HOR]=(ts[HOR:]-ts[:-HOR])==GAP
        fr=np.full(n,np.nan); fr[:n-HOR]=c[HOR:]/c[:-HOR]-1.0
        X=d[FEATS].astype("float32"); keepf=X.isna().mean(axis=1).values<0.5
        valid=contig & np.isfinite(fr) & keepf
        idx=np.where(valid)[0]
        if stride>1: idx=idx[::stride]
        Xs.append(X.values[idx]); fwds.append(fr[idx]); tss.append(ts[idx])
    if not Xs: return None
    return np.concatenate(Xs), np.concatenate(fwds), np.concatenate(tss)

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
    RESULT=(f"usdjpy_15m_cpcv_session_{SESSION}_multicov_result.json" if NSEED==1
            else f"usdjpy_15m_cpcv_session_{SESSION}_seedens{NSEED}_result.json")
    print(f"[cpcv-sess/{SESSION}] building USDJPY (stride {STRIDE}) covs={COVS}...", flush=True)
    X,fwd,ts=build_pair_ties(TARGET, STRIDE)
    o=np.argsort(ts); X=X[o]; fwd=fwd[o]; ts=ts[o]
    sess=session_mask(ts, SESSION)
    moved=np.isfinite(fwd) & (fwd!=0.0)
    print(f"[cpcv-sess/{SESSION}] USDJPY={len(ts):,} rows  in-session={int(sess.sum()):,}  built {time.time()-t0:.0f}s", flush=True)

    # groups over IN-SESSION timeline (so each test block has comparable in-session counts)
    tgs=ts[sess]; bnds=[tgs[int(k*len(tgs)/N_GROUPS)] for k in range(N_GROUPS)]+[tgs[-1]+1]
    groups=[(int(bnds[g]), int(bnds[g+1])) for g in range(N_GROUPS)]
    print(f"[cpcv-sess/{SESSION}] groups: {[(pd.to_datetime(a,unit='s').date().isoformat(), pd.to_datetime(b,unit='s').date().isoformat()) for a,b in groups]}", flush=True)

    res={"model":f"OWN-PAIR base-GBM @15m, SESSION={SESSION}, TIES-STRICT per-fold-REFIT CPCV (multi-cov)","key":"USDJPY.15m","breakeven":BE,
         "session":SESSION,"stride":STRIDE,"covs":COVS,"n_groups":N_GROUPS,"k_test":K_TEST,"nseed":NSEED,
         "CERT_RULE":"side CERTIFIED iff p10>=0.541 AND frac_clear_BE>=0.80",
         "falsifier":{"registered_utc":"pre-paths","KILL_if":"each side: p10<0.541 OR frac_clear_BE<0.80"}}
    json.dump(res, open(RESULT,"w"), indent=2)

    rng=np.random.default_rng(13)
    # paths[cov][side] = list of per-path win-rates; also track per-path n at each cov
    paths={c:{"UP":[],"DOWN":[],"COMBINED":[]} for c in COVS}
    npaths={c:{"UP":[],"DOWN":[],"COMBINED":[]} for c in COVS}
    aucs=[]; uprates=[]; pathinfo=[]
    for fi,testg in enumerate(combinations(range(N_GROUPS), K_TEST)):
        tblocks=[groups[g] for g in testg]
        tin=np.zeros(len(ts),bool)
        for lo,hi in tblocks: tin |= (ts>=lo)&(ts<hi)
        test_mask=tin & sess & np.isfinite(fwd)            # test = in-session decision bars in test blocks
        purged=np.zeros(len(ts),bool)
        for lo,hi in tblocks: purged |= (ts>=lo-PURGE)&(ts<hi+EMBARGO)
        train_mask=(~purged) & moved & sess                # train on in-session moved bars outside purge
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
            vd[s]={"p10":p10,"frac_clear_BE":fc,"med_n_per_path":med_n[s],"CERTIFIED":bool(np.isfinite(p10) and p10>=BE and fc>=0.80)}
        res["bycov"][f"{c}"]={"summary":sm,"verdict":vd}
    json.dump(res, open(RESULT,"w"), indent=2)
    print(f"\n[cpcv-sess/{SESSION}] === MULTI-COV SUMMARY ===", flush=True)
    for c in COVS:
        print(f" cov{c}:", flush=True)
        for s in ("UP","DOWN","COMBINED"):
            v=res["bycov"][f"{c}"]["verdict"][s]; sm=res["bycov"][f"{c}"]["summary"][s]
            print(f"   {s}: p10={v['p10']} mean={sm.get('mean')} frac={v['frac_clear_BE']} med_n={v['med_n_per_path']} -> CERT={v['CERTIFIED']}", flush=True)
    print(f"  AUC {res['auc_summary']} trip={res['uprate_tripwire_ok']}  done {time.time()-t0:.0f}s -> {RESULT}", flush=True)

if __name__=="__main__":
    main()
