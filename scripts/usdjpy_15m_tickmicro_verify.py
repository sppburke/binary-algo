"""DECISIVE leakage control for the tick-microstructure lift (bars+tk cov2% COMB p10 .58->.61, 14/15 folds).
A fresh-tick 15m DIRECTION edge would contradict sign-invariance -> suspect a boundary/contemporaneity leak
(the asof tick 'at t' may sit just inside the [t,t+900s] label window). Test with a LAG control: recompute
the tick features asof <= t - LAG_S (strictly causal, cannot touch the label window). Plus a label-SHUFFLE
control. Per-fold PAIRED capture (matched folds) of AUC + cov2/3% COMB/UP/DOWN win-rate.

Sets on the SAME NY refit-CPCV folds:
  bars            : 239 bar feats (control)
  bars+tk         : bars + tick feats asof<=t            (the candidate; possibly boundary-leaky)
  bars+tk_lag60   : bars + tick feats asof<=t-60s        (strictly causal — the leakage discriminator)
  tkonly          : tick feats asof<=t
  tkonly_lag60    : tick feats asof<=t-60s
  bars_SHUF       : bars, TRAIN labels permuted          (harness leakage floor -> must be ~.50)

If bars+tk_lag60 ~ bars+tk  -> edge is GENUINE predictive microstructure (escalate: seed-ens + freeze).
If bars+tk_lag60 ~ bars     -> the lift was a fresh-second boundary leak (discard).
Usage: ~/binary-algo-venv/bin/python usdjpy_15m_tickmicro_verify.py [stride=2]
"""
import os, sys, json, time, numpy as np, pandas as pd
from itertools import combinations
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from sessions import session_mask
from usdjpy_15m_tickmicro import (TARGET, FEAT, FEATS, YEARS, HOR, GAP, BE, TKDIR, WINS,
                                  N_GROUPS, K_TEST, SUB_FIT, PURGE, EMBARGO, NUM_LEAVES,
                                  mk_lgb, nonoverlap_chrono, side_wr, summ)
SESSION="ny"
STRIDE=int(sys.argv[1]) if len(sys.argv)>1 and sys.argv[1].isdigit() else 2
COVS=[0.02,0.03]
LAG_S=60
TKCOLS=["micro_dev","spread"]+[f"{p}_{W}" for W in WINS for p in ("imb","mom","rv","nt","flow","sprd")]

def build_aligned():
    """bar feats + fwd + ts + tick feats asof<=t (Xtk0) and asof<=t-LAG_S (XtkL)."""
    Xb=[]; frs=[]; tss=[]; T0=[]; TL=[]
    for y in YEARS:
        p=f"{FEAT}/{TARGET}_{y}.parquet"; tp=f"{TKDIR}/tickmicro_{y}.parquet"
        if not os.path.exists(p) or not os.path.exists(tp): continue
        d=pd.read_parquet(p, columns=FEATS+["close"]); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float); ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        contig=np.zeros(n,bool); contig[:n-HOR]=(ts[HOR:]-ts[:-HOR])==GAP
        fr=np.full(n,np.nan); fr[:n-HOR]=c[HOR:]/c[:-HOR]-1.0
        X=d[FEATS].astype("float32"); keepf=X.isna().mean(axis=1).values<0.5
        valid=contig & np.isfinite(fr) & keepf; idx=np.where(valid)[0]
        if STRIDE>1: idx=idx[::STRIDE]
        tk=pd.read_parquet(tp).drop_duplicates("epoch").sort_values("epoch")
        tke=tk["epoch"].values; tkv=tk[TKCOLS].values.astype("float32")
        be=ts[idx]
        # asof<=t
        pos0=np.searchsorted(tke, be, side="right")-1
        r0=np.full((len(be),len(TKCOLS)),np.nan,dtype="float32"); ok0=pos0>=0; r0[ok0]=tkv[pos0[ok0]]
        # asof<=t-LAG_S (strictly causal)
        posL=np.searchsorted(tke, be-LAG_S, side="right")-1
        rL=np.full((len(be),len(TKCOLS)),np.nan,dtype="float32"); okL=posL>=0; rL[okL]=tkv[posL[okL]]
        Xb.append(X.values[idx]); frs.append(fr[idx]); tss.append(ts[idx]); T0.append(r0); TL.append(rL)
    return (np.concatenate(Xb),np.concatenate(frs),np.concatenate(tss),np.concatenate(T0),np.concatenate(TL))

def main():
    t0=time.time()
    Xbar,fwd,ts,Xtk0,XtkL=build_aligned()
    o=np.argsort(ts); Xbar=Xbar[o]; fwd=fwd[o]; ts=ts[o]; Xtk0=Xtk0[o]; XtkL=XtkL[o]
    sess=session_mask(ts,SESSION); moved=np.isfinite(fwd)&(fwd!=0.0)
    print(f"[verify] rows={len(ts):,} NY={int(sess.sum()):,} tk0-complete(NY)={float(np.isfinite(Xtk0[sess]).all(1).mean()):.3f} "
          f"tkL-complete(NY)={float(np.isfinite(XtkL[sess]).all(1).mean()):.3f}  {time.time()-t0:.0f}s",flush=True)
    SETS={"bars":Xbar,"bars+tk":np.concatenate([Xbar,Xtk0],1),"bars+tk_lag60":np.concatenate([Xbar,XtkL],1),
          "tkonly":Xtk0,"tkonly_lag60":XtkL}
    tgs=ts[sess]; bnds=[tgs[int(k*len(tgs)/N_GROUPS)] for k in range(N_GROUPS)]+[tgs[-1]+1]
    groups=[(int(bnds[g]),int(bnds[g+1])) for g in range(N_GROUPS)]
    rng=np.random.default_rng(13)
    # per-fold matched arrays
    pf={s:{"auc":[], "0.02_COMB":[], "0.03_COMB":[], "0.02_UP":[], "0.02_DOWN":[]} for s in list(SETS)+["bars_SHUF"]}
    for fi,testg in enumerate(combinations(range(N_GROUPS),K_TEST)):
        tblocks=[groups[g] for g in testg]; tin=np.zeros(len(ts),bool)
        for lo,hi in tblocks: tin|=(ts>=lo)&(ts<hi)
        test_mask=tin&sess&np.isfinite(fwd)
        purged=np.zeros(len(ts),bool)
        for lo,hi in tblocks: purged|=(ts>=lo-PURGE)&(ts<hi+EMBARGO)
        train_mask=(~purged)&moved&sess; tr_idx=np.where(train_mask)[0]
        valsel=rng.random(len(tr_idx))<0.15; val_idx=tr_idx[valsel]; fit_pool=tr_idx[~valsel]
        if len(fit_pool)>SUB_FIT: fit_pool=rng.choice(fit_pool,SUB_FIT,replace=False)
        ytr=(fwd[fit_pool]>0).astype(int); yval=(fwd[val_idx]>0).astype(int)
        ftst=fwd[test_mask]; ttst=ts[test_mask]; oo=np.argsort(ttst); ftst=ftst[oo]; ttst=ttst[oo]
        mv=ftst!=0
        def run_set(name, Xs, ytr_use):
            L=mk_lgb(seed=0); L.fit(Xs[fit_pool],ytr_use,eval_set=[(Xs[val_idx],yval)],eval_metric="auc",
                callbacks=[lgb.early_stopping(80),lgb.log_evaluation(0)])
            pval=L.predict_proba(Xs[val_idx])[:,1]; ptst=L.predict_proba(Xs[test_mask])[:,1][oo]
            vconf=np.abs(pval-0.5)
            auc=float(roc_auc_score((ftst[mv]>0).astype(int),ptst[mv])) if mv.sum()>20 else np.nan
            pf[name]["auc"].append(auc)
            for c in COVS:
                thr=float(np.quantile(vconf,1-c))
                for side in (["COMB","UP","DOWN"] if c==0.02 else ["COMB"]):
                    sd={"COMB":"COMBINED","UP":"UP","DOWN":"DOWN"}[side]
                    nS,wS=side_wr(ptst,ftst,ttst,thr,sd)
                    pf[name][f"{c}_{side}"].append(wS if nS>=25 else np.nan)
        for s,Xs in SETS.items(): run_set(s, Xs, ytr)
        run_set("bars_SHUF", Xbar, rng.permutation(ytr))                 # shuffle control
        print(f"  path {fi+1}/15 g{list(testg)} | bars {pf['bars']['auc'][-1]:.4f} tk {pf['bars+tk']['auc'][-1]:.4f} "
              f"tkLag {pf['bars+tk_lag60']['auc'][-1]:.4f} tkonly {pf['tkonly']['auc'][-1]:.4f} "
              f"tkonlyLag {pf['tkonly_lag60']['auc'][-1]:.4f} SHUF {pf['bars_SHUF']['auc'][-1]:.4f} ({time.time()-t0:.0f}s)",flush=True)
    # paired summaries vs bars
    def paired(a,b):
        a=np.array(pf[a]["auc"]); b=np.array(pf[b]["auc"]); d=a-b
        se=np.nanstd(d,ddof=1)/np.sqrt(np.isfinite(d).sum())
        return {"mean_dAUC":round(float(np.nanmean(d)),4),"wins":int((d>0).sum()),"t_corr":round(float(np.nanmean(d)/se),2) if se>0 else None}
    res={"key":"USDJPY.15m.ny","test":"tick-feature leakage control (lag60 + shuffle)","stride":STRIDE,"lag_s":LAG_S,
         "auc_mean":{s:round(float(np.nanmean(pf[s]["auc"])),4) for s in pf},
         "paired_vs_bars":{s:paired(s,"bars") for s in ("bars+tk","bars+tk_lag60","tkonly","tkonly_lag60","bars_SHUF")},
         "cov_p10":{s:{k:round(float(np.nanpercentile([x for x in pf[s][k] if np.isfinite(x)],10)),4)
                       for k in ("0.02_COMB","0.03_COMB","0.02_UP","0.02_DOWN") if any(np.isfinite(pf[s][k]))}
                    for s in pf}}
    # verdict
    af=res["auc_mean"]; lift_fresh=af["bars+tk"]-af["bars"]; lift_lag=af["bars+tk_lag60"]-af["bars"]
    leak = bool(lift_fresh>0.0015 and lift_lag<lift_fresh*0.5)     # fresh lifts but lag collapses -> boundary leak
    genuine = bool(lift_lag>=0.0015 and res["paired_vs_bars"]["bars+tk_lag60"]["wins"]>=11)
    res["verdict"]={"fresh_lift":round(lift_fresh,4),"lag60_lift":round(lift_lag,4),
        "shuffle_auc":af["bars_SHUF"],"LEAK_boundary":leak,"GENUINE_causal":genuine,
        "note":("GENUINE: lag60 (strictly causal) preserves the lift -> real predictive microstructure; escalate to seed-ens+freeze."
                if genuine else "BOUNDARY-LEAK: lift vanishes with 60s-lagged features -> the fresh-second tick was peeking into the label window; discard."
                if leak else "NULL/NOISE: no material lift even fresh; bar bound holds.")}
    json.dump(res,open("usdjpy_15m_tickmicro_verify_result.json","w"),indent=2)
    print(f"\n[verify] AUC means: {res['auc_mean']}",flush=True)
    print(f"  paired vs bars: {res['paired_vs_bars']}",flush=True)
    print(f"  cov2% COMB p10: bars {res['cov_p10']['bars'].get('0.02_COMB')} | tk {res['cov_p10']['bars+tk'].get('0.02_COMB')} | tkLag60 {res['cov_p10']['bars+tk_lag60'].get('0.02_COMB')}",flush=True)
    print(f"  SHUFFLE control AUC={res['auc_mean']['bars_SHUF']} (must be ~.50)",flush=True)
    print(f"\n[verify] VERDICT: fresh_lift={lift_fresh:+.4f} lag60_lift={lift_lag:+.4f} -> "
          f"{'GENUINE' if genuine else 'BOUNDARY-LEAK' if leak else 'NULL'}  -> usdjpy_15m_tickmicro_verify_result.json  {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
