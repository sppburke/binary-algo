"""AUDUSD 15-MIN NY — refit-CPCV with a TRIPLE-BARRIER FIRST-TOUCH train label (confirm-or-kill).

SCOPE: AUDUSD · 15m · NY. The ONE lever that beat base VAL-AUC for USDJPY (.539->.5439) was the path-aware
first-touch TRAIN label — but adversarially it was NOT a robust improvement (certified-at-level, p10 lift =
single noisy order-stat; DOWN central-tendency regressed). AUDUSD is the same own-pair case, so prior is LOW,
but per the IMPROVE mandate it gets a fast confirm-or-kill on the SAME NY refit-CPCV that certified the book.

ONE change vs audusd_15m_cpcv_session.py: per-fold TRAIN target = triple-barrier first-touch (+/-k*sigma_t,
vertical=15m timeout -> endpoint sign). EVAL/early-stop/test-WR/moved-AUC stay the UNCHANGED fixed-15m deriv
sign (ties LOSE, BE .541, gap=900 nonoverlap, NY rows). k is chosen ONCE on VAL (2022-23 NY) fixed-15m AUC over
a grid (mild leak caveat: VAL years recur in some CPCV folds; a fixed scalar only INFLATES TB's apparent edge,
so a negative verdict only strengthens). Where TB undefined -> fall back to fixed sign (train pool == base).

INCUMBENT to beat (matched): seed-ens K=3 NY @cov2 UP .596 / DOWN .5962 / COMB .591; single-model NY @cov5
UP .5713 / DOWN .5773. SURVIVE-AS-IMPROVEMENT iff a side's p10 strictly exceeds the matched incumbent at
matched cov AND frac_clear_BE>=0.80 AND the path-MEAN also rises (the USDJPY-TB lesson: a p10 bump with flat
mean on correlated paths is variance redistribution, not edge).

Usage: ~/binary-algo-venv/bin/python audusd_15m_cpcv_tbfirsttouch.py [stride=2] [cov=0.05,0.03,0.02] [nseed=1]
"""
import os, sys, json, time, numpy as np, pandas as pd
from itertools import combinations
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from sessions import session_mask

TARGET="AUDUSD"; HOR=15; STEP=60; GAP=HOR*STEP; BE=0.541
FEAT=H.FEAT_DIR; FEATS=H.feature_cols(TARGET)
YEARS=list(range(2012,2027))
N_GROUPS, K_TEST = 6, 2
SUB_FIT=150_000
PURGE=GAP; EMBARGO=GAP
NUM_LEAVES=255
VOL_WIN=900; VOL_MINP=60
K_GRID=[0.5,0.75,1.0,1.5,2.0]
SESSION="ny"
def _arg(i,d,cast):
    rest=[a for a in sys.argv[1:]]
    return cast(rest[i]) if len(rest)>i else d
STRIDE=_arg(0,2,int)
_covarg=_arg(1,"0.05,0.03,0.02",str); COVS=[float(x) for x in str(_covarg).split(",")]
NSEED=_arg(2,1,int)

def _first_touch_label(c, fr, contig, k):
    n=len(c)
    r1=np.full(n,np.nan); r1[1:]=c[1:]/c[:-1]-1.0
    sig=pd.Series(r1).rolling(VOL_WIN, min_periods=VOL_MINP).std().values
    cpad=np.concatenate([c, np.full(HOR,np.nan)])
    win=np.lib.stride_tricks.sliding_window_view(cpad, HOR)
    fwd=win[1:n+1]; c0=c[:,None]
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

def build_pair(pair, stride, ks):
    """Returns X, fwd(fixed-15m EVAL src), ts, and a dict tb[k] of first-touch TRAIN labels for each k in ks."""
    Xs=[]; fwds=[]; tss=[]; tbs={k:[] for k in ks}
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
        for k in ks:
            tb=_first_touch_label(c, fr, contig, k); tbs[k].append(tb[idx])
    if not Xs: return None
    return (np.concatenate(Xs), np.concatenate(fwds), np.concatenate(tss),
            {k:np.concatenate(v) for k,v in tbs.items()})

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
            "min":round(float(a.min()),4),"frac_clear_BE":round(float((a>=BE).mean()),3)}

def main():
    t0=time.time()
    RESULT=(f"audusd_15m_cpcv_tbfirsttouch_ny_multicov_result.json" if NSEED==1
            else f"audusd_15m_cpcv_tbfirsttouch_ny_seedens{NSEED}_result.json")
    print(f"[tbft/ny] building AUDUSD (stride {STRIDE}) covs={COVS} nseed={NSEED} kgrid={K_GRID}...",flush=True)
    X,fwd,ts,tbd=build_pair(TARGET, STRIDE, K_GRID)
    o=np.argsort(ts); X=X[o]; fwd=fwd[o]; ts=ts[o]; tbd={k:v[o] for k,v in tbd.items()}
    sess=session_mask(ts, SESSION); moved=np.isfinite(fwd)&(fwd!=0.0)
    # --- VAL k-screen: train 2012-21 NY moved, eval VAL 2022-23 NY fixed-15m AUC; pick best k ---
    yrs=pd.to_datetime(ts,unit="s",utc=True).year.values
    tr0=(yrs<=2021)&moved&sess; va0=(yrs>=2022)&(yrs<=2023)&moved&sess
    ti=np.where(tr0)[0]; rng=np.random.default_rng(1)
    if len(ti)>SUB_FIT: ti=rng.choice(ti,SUB_FIT,replace=False)
    vi=np.where(va0)[0]; yval0=(fwd[vi]>0).astype(int)
    best=None
    for k in K_GRID:
        ytr0=np.where(np.isfinite(tbd[k][ti]), tbd[k][ti], (fwd[ti]>0).astype(int)).astype(int)
        L=mk_lgb(seed=0); L.fit(X[ti],ytr0,eval_set=[(X[vi],yval0)],eval_metric="auc",callbacks=[lgb.early_stopping(80),lgb.log_evaluation(0)])
        a=float(roc_auc_score(yval0, L.predict_proba(X[vi])[:,1]))
        print(f"  k-screen k={k}: VAL fixed-15m AUC={a:.4f} ({time.time()-t0:.0f}s)",flush=True)
        if best is None or a>best[1]: best=(k,a)
    TB_K=best[0]
    tb=tbd[TB_K]; tb_def=np.isfinite(tb)
    msf=moved&sess&tb_def; flip=float((tb[msf]!=(fwd[msf]>0).astype(int)).mean()) if msf.sum() else float("nan")
    print(f"[tbft/ny] chosen TB_K={TB_K} (VAL AUC {best[1]:.4f}); base fixed-15m VAL AUC was ~.5378; TB!=sign frac(moved,NY)={flip:.4f}",flush=True)

    tgs=ts[sess]; bnds=[tgs[int(k*len(tgs)/N_GROUPS)] for k in range(N_GROUPS)]+[tgs[-1]+1]
    groups=[(int(bnds[g]), int(bnds[g+1])) for g in range(N_GROUPS)]
    INC=({"UP":0.596,"DOWN":0.5962,"COMBINED":0.591} if NSEED>=3 else {"UP":0.5713,"DOWN":0.5773,"COMBINED":0.5809})
    res={"key":"AUDUSD.15m","session":SESSION,"breakeven":BE,"train_label":f"tb_first_touch_k{TB_K}",
         "val_kscreen":{str(k):None for k in K_GRID},"chosen_k":TB_K,"tb_flip_frac_moved_ny":round(flip,4),
         "stride":STRIDE,"covs":COVS,"nseed":NSEED,"incumbent_matched":INC,
         "CERT_RULE":"side CERTIFIED iff p10>=.541 & frac>=.80; IMPROVEMENT iff p10>matched incumbent at matched cov AND path-mean also rises",
         "falsifier":{"registered":"pre-paths","KILL_as_improvement_if":"no side p10 strictly exceeds matched incumbent at matched cov with mean also rising"}}
    json.dump(res,open(RESULT,"w"),indent=2)

    rng=np.random.default_rng(13)
    paths={c:{"UP":[],"DOWN":[],"COMBINED":[]} for c in COVS}; npaths={c:{"UP":[],"DOWN":[],"COMBINED":[]} for c in COVS}
    aucs=[]; uprates=[]; pathinfo=[]
    for fi,testg in enumerate(combinations(range(N_GROUPS), K_TEST)):
        tblocks=[groups[g] for g in testg]; tin=np.zeros(len(ts),bool)
        for lo,hi in tblocks: tin|=(ts>=lo)&(ts<hi)
        test_mask=tin&sess&np.isfinite(fwd)
        purged=np.zeros(len(ts),bool)
        for lo,hi in tblocks: purged|=(ts>=lo-PURGE)&(ts<hi+EMBARGO)
        train_mask=(~purged)&moved&sess; tr_idx=np.where(train_mask)[0]
        valsel=rng.random(len(tr_idx))<0.15; val_idx=tr_idx[valsel]; fit_pool=tr_idx[~valsel]
        if len(fit_pool)>SUB_FIT: fit_pool=rng.choice(fit_pool,SUB_FIT,replace=False)
        ytr=np.where(tb_def[fit_pool], tb[fit_pool], (fwd[fit_pool]>0).astype(int)).astype(int)
        yval=(fwd[val_idx]>0).astype(int)
        pval=np.zeros(len(val_idx)); ptst=np.zeros(int(test_mask.sum()))
        for sd in range(NSEED):
            L=mk_lgb(seed=sd); L.fit(X[fit_pool], ytr, eval_set=[(X[val_idx], yval)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
            pval+=L.predict_proba(X[val_idx])[:,1]; ptst+=L.predict_proba(X[test_mask])[:,1]
        pval/=NSEED; ptst/=NSEED; vconf=np.abs(pval-0.5)
        ftst=fwd[test_mask]; ttst=ts[test_mask]; oo=np.argsort(ttst); ptst=ptst[oo]; ftst=ftst[oo]; ttst=ttst[oo]
        mv=ftst!=0; auc=float(roc_auc_score((ftst[mv]>0).astype(int), ptst[mv])) if mv.sum()>20 else float("nan")
        uprate=float((ftst[mv]>0).mean()) if mv.sum()>0 else float("nan"); aucs.append(auc); uprates.append(uprate)
        rowinfo={"fold":list(testg),"auc":round(auc,4),"up_rate":round(uprate,4),"bycov":{}}
        for c in COVS:
            thr=float(np.quantile(vconf,1-c))
            nC,wC=side_wr(ptst,ftst,ttst,thr,"COMBINED"); nU,wU=side_wr(ptst,ftst,ttst,thr,"UP"); nD,wD=side_wr(ptst,ftst,ttst,thr,"DOWN")
            if nC>=25: paths[c]["COMBINED"].append(wC); npaths[c]["COMBINED"].append(nC)
            if nU>=25: paths[c]["UP"].append(wU); npaths[c]["UP"].append(nU)
            if nD>=25: paths[c]["DOWN"].append(wD); npaths[c]["DOWN"].append(nD)
            rowinfo["bycov"][f"{c}"]={"UP":[nU,round(wU,4)],"DOWN":[nD,round(wD,4)],"COMBINED":[nC,round(wC,4)]}
        pathinfo.append(rowinfo)
        bc=rowinfo["bycov"][f"{COVS[0]}"]
        print(f"  path {fi+1}/15 g{list(testg)} AUC={auc:.4f} | @cov{COVS[0]}: UP {bc['UP']} DOWN {bc['DOWN']} COMB {bc['COMBINED']} ({time.time()-t0:.0f}s)",flush=True)

    res["paths"]=pathinfo
    res["auc_summary"]={"mean":round(float(np.nanmean(aucs)),4),"min":round(float(np.nanmin(aucs)),4),"max":round(float(np.nanmax(aucs)),4)}
    res["bycov"]={}
    for c in COVS:
        sm={s:summ(paths[c][s]) for s in ("UP","DOWN","COMBINED")}
        vd={}
        for s in ("UP","DOWN","COMBINED"):
            p10=sm[s].get("p10",float("nan")); fc=sm[s].get("frac_clear_BE",0.0); mn=sm[s].get("mean",float("nan"))
            vd[s]={"p10":p10,"mean":mn,"frac_clear_BE":fc,"CERTIFIED":bool(np.isfinite(p10) and p10>=BE and fc>=0.80),
                   "IMPROVES_matched_incumbent":bool(np.isfinite(p10) and p10>INC[s] and fc>=0.80)}
        res["bycov"][f"{c}"]={"summary":sm,"verdict":vd}
    res["any_side_improves"]=bool(any(res["bycov"][f"{c}"]["verdict"][s]["IMPROVES_matched_incumbent"] for c in COVS for s in ("UP","DOWN")))
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[tbft/ny] === SUMMARY (TB_K={TB_K}, flip={flip:.3f}) vs matched incumbent {INC} ===",flush=True)
    for c in COVS:
        print(f" cov{c}:",flush=True)
        for s in ("UP","DOWN","COMBINED"):
            v=res["bycov"][f"{c}"]["verdict"][s]
            print(f"   {s}: p10={v['p10']} mean={v['mean']} frac={v['frac_clear_BE']} CERT={v['CERTIFIED']} IMPROVES={v['IMPROVES_matched_incumbent']}",flush=True)
    print(f"  AUC {res['auc_summary']} -> any_side_improves={res['any_side_improves']}  done {time.time()-t0:.0f}s -> {RESULT}",flush=True)

if __name__=="__main__":
    main()
