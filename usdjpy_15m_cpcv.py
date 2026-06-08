"""USDJPY 15-MIN — full per-fold-REFIT CPCV of the CROSS-PAIR POOLED primary (the certification keystone).

SCOPE: USDJPY · 15m. Decides whether the cross-pair pooled edge CERTIFIES — i.e. survives per-fold gate
re-tuning across purged-combinatorial time blocks, not just one VAL-frozen pocket. This is the right test
for a THIN-but-real edge: it aggregates 15 purged paths so the effective n at the operating gate is large.
15m is the deriv-FX-deployable floor; EURUSD 15m cross-pair certified BOTH sides here (UP p10 .567 / DOWN
p10 .574). The horizon-gradient (none@60s→UP@5m→BOTH@10m&15m) predicts USDJPY 15m pooling should cross BE.

Design (mirrors usdjpy_2m_cpcv.py): 6 contiguous time-GROUPS over the USDJPY timeline, C(6,2)=15 purged test
paths, purge+embargo = 1 horizon (900s) each side of every test block (applied to ALL pairs), per-fold REFIT
of the POOLED base GBM (train = all 7 majors' base feats outside the purged test windows, subsampled to 150k),
gate (conf-coverage) tuned on a within-fold USDJPY VAL split, test = USDJPY moved bars, TIES-STRICT (zero-move
bet charged a LOSS), nonoverlap gap=900, per-side UP/DOWN/COMBINED selective win-rate. Reports per-path AUC +
test up-rate so a high-acc/low-AUC mirage or an up-rate-drift mirage shows. Memory-safe: pool built once at
equal stride, float32, subsample fit.

CERTIFY a side iff per-side p10 >= 0.541 AND >= ~80% paths clear 0.541. Breakeven 0.541.
Usage: ~/binary-algo-venv/bin/python usdjpy_15m_cpcv.py [pool_stride=12] [cov=0.03] [nseed=1]
"""
import os, sys, json, time, numpy as np, pandas as pd
from itertools import combinations
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H

TARGET="USDJPY"; HOR=15; STEP=60; GAP=HOR*STEP; BE=0.541
PAIRS=["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCHF","USDCAD"]
FEAT=H.FEAT_DIR; FEATS=H.feature_cols(TARGET)
YEARS=list(range(2012,2027))
N_GROUPS, K_TEST = 6, 2
SUB_FIT=150_000
PURGE=GAP; EMBARGO=GAP
INCUMBENT={"UP":0.541,"DOWN":0.541,"COMBINED":0.541}   # new horizon: no USDJPY incumbent; cert is absolute vs BE
POOL_STRIDE=int(sys.argv[1]) if len(sys.argv)>1 and sys.argv[1].isdigit() else 12
COV=float(sys.argv[2]) if len(sys.argv)>2 else 0.03
NSEED=int(sys.argv[3]) if len(sys.argv)>3 else 1   # >1 => seed-ensemble per fold (Tier-I variance reduction)

def build_pair_ties(pair, stride):
    """Base feats + RAW 15-min forward return (ties kept) for ALL years. Returns X(float32 ndarray),
    fwd(raw return; nan if invalid), ts(int64). label/moved derived downstream."""
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

def mk_lgb(n=800, num_leaves=255, seed=0):
    return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=num_leaves,
        min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,
        n_estimators=n,n_jobs=20,verbosity=-1,random_state=seed,bagging_seed=seed,feature_fraction_seed=seed)

def nonoverlap_chrono(ts, mask, gap=GAP):
    take=[]; block=-1
    for i in np.where(mask)[0]:
        if ts[i]<block: continue
        take.append(i); block=int(ts[i])+gap
    return np.array(take,dtype=int)

def boot(corr, nb=2000, seed=7):
    corr=np.asarray(corr,float)
    if len(corr)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(corr)
    a=np.array([corr[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))

def side_wr(pr, fwd, ts, thr, side):
    """Selective at conf>=thr, predicted `side`, nonoverlap; ties (fwd==0) LOSE. Returns (n, wr)."""
    conf=np.abs(pr-0.5)
    want=(pr>0.5) if side=="UP" else (pr<0.5) if side=="DOWN" else np.ones(len(pr),bool)
    cand=want & (conf>=thr) & np.isfinite(fwd)
    tr=nonoverlap_chrono(ts, cand)
    if len(tr)==0: return 0, float("nan")
    pred=(pr[tr]>0.5).astype(int); ylab=(fwd[tr]>0).astype(int); moved=(fwd[tr]!=0)
    win=((pred==ylab) & moved).astype(float)   # tie => loss
    return len(tr), float(win.mean())

def summ(a):
    a=np.asarray([x for x in a if np.isfinite(x)],float)
    if len(a)==0: return {"n_paths":0}
    return {"n_paths":int(len(a)),"mean":round(float(a.mean()),4),"p10":round(float(np.percentile(a,10)),4),
            "p50":round(float(np.percentile(a,50)),4),"min":round(float(a.min()),4),
            "frac_clear_BE":round(float((a>=BE).mean()),3)}

def main():
    t0=time.time()
    RESULT="usdjpy_15m_cpcv_result.json" if NSEED==1 else f"usdjpy_15m_cpcv_k{NSEED}_result.json"
    print(f"[cpcv15m] building pool (stride {POOL_STRIDE}, 7 majors)...", flush=True)
    Xs=[]; fwds=[]; tss=[]; istgt=[]
    for p in PAIRS:
        r=build_pair_ties(p, POOL_STRIDE)
        if r is None: continue
        X,fwd,ts=r; Xs.append(X); fwds.append(fwd); tss.append(ts)
        istgt.append(np.full(len(ts), p==TARGET))
        print(f"   {p}: {len(ts):,} rows ({time.time()-t0:.0f}s)", flush=True)
    X=np.concatenate(Xs); fwd=np.concatenate(fwds); ts=np.concatenate(tss); istgt=np.concatenate(istgt)
    del Xs, fwds, tss
    moved=np.isfinite(fwd) & (fwd!=0.0)
    print(f"[cpcv15m] pool={len(ts):,} rows  USDJPY={int(istgt.sum()):,}  feats={X.shape[1]}  built {time.time()-t0:.0f}s", flush=True)

    # 6 contiguous time-groups over the USDJPY timeline (equal USDJPY-bar counts)
    tg=np.sort(ts[istgt]); bnds=[tg[int(k*len(tg)/N_GROUPS)] for k in range(N_GROUPS)]+[tg[-1]+1]
    groups=[(int(bnds[g]), int(bnds[g+1])) for g in range(N_GROUPS)]   # [lo,hi) ts ranges
    print(f"[cpcv15m] groups (ts ranges): {[(pd.to_datetime(a,unit='s').date().isoformat(), pd.to_datetime(b,unit='s').date().isoformat()) for a,b in groups]}", flush=True)

    res={"model":"cross-pair POOLED base-GBM @15m, TIES-STRICT per-fold-REFIT CPCV","key":"USDJPY.15m","breakeven":BE,
         "pool_stride":POOL_STRIDE,"cov":COV,"n_groups":N_GROUPS,"k_test":K_TEST,"sub_fit":SUB_FIT,"nseed":NSEED,
         "incumbent_floor":INCUMBENT,
         "CERT_RULE":"side CERTIFIED iff p10>=0.541 AND frac_clear_BE>=0.80; IMPROVES incumbent iff p10>incumbent_floor",
         "falsifier":{"registered_utc":"pre-paths","KILL_if":"each side: p10 < 0.541 OR frac_clear_BE < 0.80"}}
    json.dump(res, open(RESULT,"w"), indent=2)

    rng=np.random.default_rng(13)
    paths={"UP":[],"DOWN":[],"COMBINED":[]}; aucs=[]; uprates=[]; pathinfo=[]
    folds=list(combinations(range(N_GROUPS), K_TEST))
    for fi,testg in enumerate(folds):
        tblocks=[groups[g] for g in testg]
        # test = USDJPY moved+tie bars inside the test blocks
        tin=np.zeros(len(ts),bool)
        for lo,hi in tblocks: tin |= (ts>=lo)&(ts<hi)
        test_mask=istgt & tin & np.isfinite(fwd)
        # purge: drop ALL rows within [lo-PURGE, hi+EMBARGO] of any test block from train
        purged=np.zeros(len(ts),bool)
        for lo,hi in tblocks: purged |= (ts>=lo-PURGE)&(ts<hi+EMBARGO)
        train_mask= (~purged) & moved                      # train on moved bars only, any pair, outside purge
        # within-fold VAL = random 15% of USDJPY train-moved bars (threshold calibration)
        tgt_tr=np.where(train_mask & istgt)[0]
        valsel=rng.random(len(tgt_tr))<0.15
        val_idx=tgt_tr[valsel]; fit_pool=np.where(train_mask)[0]
        fit_pool=fit_pool[~np.isin(fit_pool, val_idx)]     # exclude val from fit
        if len(fit_pool)>SUB_FIT: fit_pool=rng.choice(fit_pool, SUB_FIT, replace=False)
        ytr=(fwd[fit_pool]>0).astype(int)
        yval=(fwd[val_idx]>0).astype(int)
        # fit NSEED models (seed-ensemble if NSEED>1), average probabilities on VAL + TEST
        pval=np.zeros(len(val_idx)); ptst=np.zeros(int(test_mask.sum()))
        for sd in range(NSEED):
            L=mk_lgb(seed=sd)
            L.fit(X[fit_pool], ytr, eval_set=[(X[val_idx], yval)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
            pval+=L.predict_proba(X[val_idx])[:,1]; ptst+=L.predict_proba(X[test_mask])[:,1]
        pval/=NSEED; ptst/=NSEED
        # threshold per side from VAL at coverage COV (conf quantile among VAL moved)
        vconf=np.abs(pval-0.5)
        thr=float(np.quantile(vconf, 1-COV))
        # test
        ftst=fwd[test_mask]; ttst=ts[test_mask]
        order=np.argsort(ttst); ptst=ptst[order]; ftst=ftst[order]; ttst=ttst[order]
        mv=ftst!=0
        auc=float(roc_auc_score((ftst[mv]>0).astype(int), ptst[mv])) if mv.sum()>20 else float("nan")
        uprate=float((ftst[mv]>0).mean()) if mv.sum()>0 else float("nan")
        nC,wC=side_wr(ptst,ftst,ttst,thr,"COMBINED")
        nU,wU=side_wr(ptst,ftst,ttst,thr,"UP")
        nD,wD=side_wr(ptst,ftst,ttst,thr,"DOWN")
        if nC>=25: paths["COMBINED"].append(wC)
        if nU>=25: paths["UP"].append(wU)
        if nD>=25: paths["DOWN"].append(wD)
        aucs.append(auc); uprates.append(uprate)
        pathinfo.append({"fold":list(testg),"auc":round(auc,4),"up_rate":round(uprate,4),
                         "UP":[nU,round(wU,4)],"DOWN":[nD,round(wD,4)],"COMBINED":[nC,round(wC,4)],"thr":round(thr,4)})
        print(f"  path {fi+1}/15 test-g{list(testg)} AUC={auc:.4f} up-rate={uprate:.4f} | "
              f"UP n{nU} {wU:.4f} | DOWN n{nD} {wD:.4f} | COMB n{nC} {wC:.4f} ({time.time()-t0:.0f}s)", flush=True)

    res["paths"]=pathinfo
    res["auc_summary"]={"mean":round(float(np.nanmean(aucs)),4),"min":round(float(np.nanmin(aucs)),4),"max":round(float(np.nanmax(aucs)),4)}
    res["uprate_tripwire_ok"]=bool(np.all([(0.46<=u<=0.54) for u in uprates if np.isfinite(u)]))
    res["summary"]={s:summ(paths[s]) for s in ("UP","DOWN","COMBINED")}
    verdict={}
    for s in ("UP","DOWN","COMBINED"):
        sm=res["summary"][s]; p10=sm.get("p10",float("nan")); fc=sm.get("frac_clear_BE",0.0)
        verdict[s]={"p10":p10,"frac_clear_BE":fc,
                    "CERTIFIED":bool(np.isfinite(p10) and p10>=BE and fc>=0.80),
                    "IMPROVES_incumbent":bool(np.isfinite(p10) and p10>INCUMBENT.get(s,BE))}
    res["verdict"]=verdict
    json.dump(res, open(RESULT,"w"), indent=2)
    print("\n[cpcv15m] === SUMMARY ===", flush=True)
    for s in ("UP","DOWN","COMBINED"):
        print(f"  {s}: {res['summary'][s]}  -> CERT={verdict[s]['CERTIFIED']} IMPROVES={verdict[s]['IMPROVES_incumbent']}", flush=True)
    print(f"  AUC {res['auc_summary']}  uprate_tripwire_ok={res['uprate_tripwire_ok']}", flush=True)
    print(f"[cpcv15m] done {time.time()-t0:.0f}s -> {RESULT}", flush=True)

if __name__=="__main__":
    main()
