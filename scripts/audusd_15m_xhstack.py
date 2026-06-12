"""AUDUSD 15m NY — CROSS-HORIZON STACK refit-CPCV (30m parent x 15m child; corr .81, partially decorrelated).

SCOPE: AUDUSD · 15m · NY. xhorizon showed the 30m-NY parent is only .81-correlated with the 15m child (NOT
collinear like USDJPY .957) and has a real edge (parent cov2 .595). This tests whether combining the two
semi-independent directional reads lifts the 15m NY win-rate over the child-only book. Per CPCV fold: REFIT a
15m-NY child (label sign(close[t+15]-close[t])) AND a 30m-NY parent (label sign(close[t+30]-close[t])) on the
fold's purged NY-moved train; on the test fold predict p15 + p30 on the SAME 15m decision rows; combine. Eval =
the 15m deriv label (ties LOSE, BE .541), NY rows, nonoverlap gap=900.

ARMS (all at MATCHED coverage via VAL quantile so win-rates are comparable):
  base   = child-only: bet sign(p15-.5), gate |p15-.5|>=thr      (matched single-seed control)
  blend  = bet sign(pb-.5), pb=(p15+p30)/2, gate |pb-.5|>=thr     (mix both horizons)
  agree  = child bet, require sign(p30-.5)==sign(p15-.5), child thr set to hit the same overall coverage
           (the 30m parent as a confirmation filter — the EURUSD cross-horizon mechanism)

INCUMBENT: seed-ens K=3 NY @cov2 UP .596 / DOWN .5962 / COMB .591. Also compare blend/agree vs THIS script's
single-seed base arm (matched harness). IMPROVEMENT iff a combined arm's p10 > base arm p10 at matched cov on
BOTH sides AND the path-mean rises AND frac_clear>=.80. If it clears -> escalate to seed-ens + freeze.

Usage: ~/binary-algo-venv/bin/python audusd_15m_xhstack.py [stride=2] [cov=0.05,0.03,0.02] [nseed=1]
"""
import os, sys, json, time, numpy as np, pandas as pd
from itertools import combinations
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from sessions import session_mask

TARGET="AUDUSD"; STEP=60; BE=0.541; FEAT=H.FEAT_DIR; FEATS=H.feature_cols(TARGET)
YEARS=list(range(2012,2027)); N_GROUPS,K_TEST=6,2; SUB_FIT=150_000; NUM_LEAVES=255
HOR_C=15; HOR_P=30; GAP_C=HOR_C*STEP; GAP_P=HOR_P*STEP
PURGE=GAP_P; EMBARGO=GAP_P    # purge by the LONGER (parent) horizon to be safe
def _arg(i,d,cast):
    rest=[a for a in sys.argv[1:]]; return cast(rest[i]) if len(rest)>i else d
STRIDE=_arg(0,2,int); COVS=[float(x) for x in str(_arg(1,"0.05,0.03,0.02",str)).split(",")]; NSEED=_arg(2,1,int)
SESSION="ny"

def build():
    Xs=[]; f15=[]; f30=[]; tss=[]
    for y in YEARS:
        p=f"{FEAT}/{TARGET}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p, columns=FEATS+["close"]); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float); ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        X=d[FEATS].astype("float32"); keepf=X.isna().mean(axis=1).values<0.5
        cg15=np.zeros(n,bool); cg15[:n-HOR_C]=(ts[HOR_C:]-ts[:-HOR_C])==GAP_C
        cg30=np.zeros(n,bool); cg30[:n-HOR_P]=(ts[HOR_P:]-ts[:-HOR_P])==GAP_P
        fr15=np.full(n,np.nan); fr15[:n-HOR_C]=c[HOR_C:]/c[:-HOR_C]-1.0
        fr30=np.full(n,np.nan); fr30[:n-HOR_P]=c[HOR_P:]/c[:-HOR_P]-1.0
        fr15[~(cg15&keepf)]=np.nan; fr30[~(cg30&keepf)]=np.nan
        idx=np.where(keepf)[0]
        if STRIDE>1: idx=idx[::STRIDE]
        Xs.append(X.values[idx]); f15.append(fr15[idx]); f30.append(fr30[idx]); tss.append(ts[idx])
    return np.concatenate(Xs),np.concatenate(f15),np.concatenate(f30),np.concatenate(tss)

def mk(seed=0): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=NUM_LEAVES,
    min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,n_estimators=800,
    n_jobs=16,verbosity=-1,random_state=seed,bagging_seed=seed,feature_fraction_seed=seed)

def nonoverlap_chrono(ts, mask, gap=GAP_C):
    take=[]; block=-1
    for i in np.where(mask)[0]:
        if ts[i]<block: continue
        take.append(i); block=int(ts[i])+gap
    return np.array(take,dtype=int)

def wr_side(sel_mask, betdir, fwd, ts, side):
    want=np.ones(len(betdir),bool) if side=="COMBINED" else (betdir==1) if side=="UP" else (betdir==0)
    cand=sel_mask & want & np.isfinite(fwd)
    tr=nonoverlap_chrono(ts, cand)
    if len(tr)==0: return 0,float("nan")
    win=((betdir[tr]==(fwd[tr]>0).astype(int)) & (fwd[tr]!=0)).astype(float)
    return len(tr), float(win.mean())

def summ(a):
    a=np.asarray([x for x in a if np.isfinite(x)],float)
    if len(a)==0: return {"n_paths":0}
    return {"n_paths":int(len(a)),"mean":round(float(a.mean()),4),"p10":round(float(np.percentile(a,10)),4),
            "min":round(float(a.min()),4),"frac_clear_BE":round(float((a>=BE).mean()),3)}

def main():
    t0=time.time()
    RESULT=f"audusd_15m_xhstack_ny_result.json"
    print(f"[xhstack] building (stride {STRIDE}) covs={COVS} nseed={NSEED}...",flush=True)
    X,f15,f30,ts=build(); o=np.argsort(ts); X=X[o]; f15=f15[o]; f30=f30[o]; ts=ts[o]
    sess=session_mask(ts,SESSION)
    mv15=np.isfinite(f15)&(f15!=0.0); mv30=np.isfinite(f30)&(f30!=0.0)
    print(f"[xhstack] rows={len(ts):,} NY={int(sess.sum()):,} 15m-moved-NY={int((mv15&sess).sum()):,} 30m-moved-NY={int((mv30&sess).sum()):,} built {time.time()-t0:.0f}s",flush=True)
    tgs=ts[sess]; bnds=[tgs[int(k*len(tgs)/N_GROUPS)] for k in range(N_GROUPS)]+[tgs[-1]+1]
    groups=[(int(bnds[g]),int(bnds[g+1])) for g in range(N_GROUPS)]
    INC={"UP":0.596,"DOWN":0.5962,"COMBINED":0.591}
    ARMS=["base","blend","agree"]
    paths={c:{a:{s:[] for s in ("UP","DOWN","COMBINED")} for a in ARMS} for c in COVS}
    rng=np.random.default_rng(13); corrs=[]
    for fi,testg in enumerate(combinations(range(N_GROUPS),K_TEST)):
        tin=np.zeros(len(ts),bool)
        for g in testg: lo,hi=groups[g]; tin|=(ts>=lo)&(ts<hi)
        purged=np.zeros(len(ts),bool)
        for g in testg: lo,hi=groups[g]; purged|=(ts>=lo-PURGE)&(ts<hi+EMBARGO)
        test_mask=tin & sess & np.isfinite(f15)
        tr15=np.where((~purged)&mv15&sess)[0]; tr30=np.where((~purged)&mv30&sess)[0]
        vsel=rng.random(len(tr15))<0.15; val_idx=tr15[vsel]; fit15=tr15[~vsel]
        if len(fit15)>SUB_FIT: fit15=rng.choice(fit15,SUB_FIT,replace=False)
        if len(tr30)>SUB_FIT: tr30=rng.choice(tr30,SUB_FIT,replace=False)
        p15v=np.zeros(len(val_idx)); ti=np.where(test_mask)[0]; p15t=np.zeros(len(ti)); p30t=np.zeros(len(ti)); p30v=np.zeros(len(val_idx))
        for sd in range(NSEED):
            Lc=mk(sd).fit(X[fit15],(f15[fit15]>0).astype(int),eval_set=[(X[val_idx],(f15[val_idx]>0).astype(int))],eval_metric="auc",callbacks=[lgb.early_stopping(80),lgb.log_evaluation(0)])
            Lp=mk(sd+100).fit(X[tr30],(f30[tr30]>0).astype(int))
            p15v+=Lc.predict_proba(X[val_idx])[:,1]; p15t+=Lc.predict_proba(X[ti])[:,1]
            p30t+=Lp.predict_proba(X[ti])[:,1]; p30v+=Lp.predict_proba(X[val_idx])[:,1]
        p15v/=NSEED; p15t/=NSEED; p30t/=NSEED; p30v/=NSEED
        fwd=f15[ti]; tst=ts[ti]
        oo=np.argsort(tst); p15t=p15t[oo]; p30t=p30t[oo]; fwd=fwd[oo]; tst=tst[oo]
        mv=fwd!=0
        if mv.sum()>20: corrs.append(float(np.corrcoef(p15t[mv],p30t[mv])[0,1]))
        bet15=(p15t>0.5).astype(int); betb=((p15t+p30t)/2>0.5).astype(int); agree=(p30t>0.5)==(p15t>0.5)
        cf15=np.abs(p15t-0.5); cfb=np.abs((p15t+p30t)/2-0.5)
        cf15v=np.abs(p15v-0.5); cfbv=np.abs((p15v+p30v)/2-0.5); agreev=(p30v>0.5)==(p15v>0.5)
        for c in COVS:
            thr15=float(np.quantile(cf15v,1-c)); thrb=float(np.quantile(cfbv,1-c))
            # agree: choose child thr on VAL agreed bars to hit overall coverage c
            af=max(agreev.mean(),1e-6); q=min(max(1-c/af,0.0),0.999); thra=float(np.quantile(cf15v[agreev],q)) if agreev.sum()>50 else thr15
            sel_base=cf15>=thr15; sel_blend=cfb>=thrb; sel_agree=agree&(cf15>=thra)
            for arm,(sel,bet) in {"base":(sel_base,bet15),"blend":(sel_blend,betb),"agree":(sel_agree,bet15)}.items():
                for s in ("UP","DOWN","COMBINED"):
                    nS,wS=wr_side(sel,bet,fwd,tst,s)
                    if nS>=25: paths[c][arm][s].append(wS)
        print(f"  path {fi+1}/15 corr(p15,p30)={corrs[-1] if corrs else float('nan'):.3f} ({time.time()-t0:.0f}s)",flush=True)
    res={"key":"AUDUSD.15m.NY.xhstack","breakeven":BE,"stride":STRIDE,"covs":COVS,"nseed":NSEED,
         "incumbent_seedensK3_cov2":INC,"mean_fold_corr_p15_p30":round(float(np.mean(corrs)),4) if corrs else None,
         "arms":ARMS,"bycov":{}}
    improves=False
    for c in COVS:
        res["bycov"][f"{c}"]={}
        for a in ARMS:
            res["bycov"][f"{c}"][a]={s:summ(paths[c][a][s]) for s in ("UP","DOWN","COMBINED")}
        b=res["bycov"][f"{c}"]["base"]
        for a in ("blend","agree"):
            ar=res["bycov"][f"{c}"][a]
            beats_base=all(ar[s].get("p10",-9)>b[s].get("p10",9) for s in ("UP","DOWN")) and all(ar[s].get("frac_clear_BE",0)>=0.8 for s in ("UP","DOWN"))
            beats_inc=all(ar[s].get("p10",-9)>INC[s] for s in ("UP","DOWN")) and all(ar[s].get("frac_clear_BE",0)>=0.8 for s in ("UP","DOWN"))
            res["bycov"][f"{c}"][a+"_beats_base_arm"]=bool(beats_base); res["bycov"][f"{c}"][a+"_beats_seedens_inc"]=bool(beats_inc)
            if beats_inc: improves=True
    res["IMPROVES_over_seedens"]=improves
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[xhstack] mean fold corr(p15,p30)={res['mean_fold_corr_p15_p30']}",flush=True)
    for c in COVS:
        print(f" cov{c}:",flush=True)
        for a in ARMS:
            r=res["bycov"][f"{c}"][a]
            print(f"   {a}: UP p10={r['UP'].get('p10')} mean={r['UP'].get('mean')} | DOWN p10={r['DOWN'].get('p10')} mean={r['DOWN'].get('mean')} | COMB p10={r['COMBINED'].get('p10')}",flush=True)
        print(f"   -> blend beats_inc={res['bycov'][f'{c}']['blend_beats_seedens_inc']} agree beats_inc={res['bycov'][f'{c}']['agree_beats_seedens_inc']}",flush=True)
    print(f"[xhstack] IMPROVES_over_seedens={improves}  done {time.time()-t0:.0f}s -> {RESULT}",flush=True)

if __name__=="__main__":
    main()
