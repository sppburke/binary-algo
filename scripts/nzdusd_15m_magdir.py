"""NZDUSD 15m NY — E2/I3: direction-conditioned-on-magnitude (magdir). Does bucketing the
certified seed-ens direction trades by predicted move-size lift per-side hit-rate?

Sign-invariance theorem + 3-major prior kills (m15_magdir_result.json EURUSD,
usdchf_15m_magdir_result.json, usdcad_15m_magdir_result.json) predict NULL.
NZDUSD-specific Tier-1 run closes the "SUBSUMED pending magdir confirmation here"
deferral in sweeps/NZDUSD_15m.md for both E2 (magnitude gating) and I3 (|ret|-weight).

Falsifier: KILL unless HIGH-mag bucket lifts binding-year UP or DOWN CI95-lo above the
frozen-forward incumbent floor by > 1 SE (~0.02).
Incumbent floors: test24 UP .5059 / DOWN .6443; test25 UP .5140 / DOWN .5816; oos UP .4184 / DOWN .4646.
"""
import os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sessions import session_mask
from nzdusd_15m_base import build, boot, BE, SPL, FEATS, GAP
import harness as H

HOR = 15
FEAT_DIR = H.FEAT_DIR
PAIR = "NZDUSD"
MODELS = "/home/sean/git/binary-algo/models"
RESULT = "nzdusd_15m_magdir_result.json"
CONF_THR = 0.0835073173947228  # frozen cov2% gate from m15ny_NZDUSD_seedens_strategy.json

# Incumbent frozen-forward CI-lo (from books/NZDUSD.m15ny_seedens.v1.manifest.json)
INCUMBENT_CI_LO = {
    "test24": {"UP": 0.5059, "DOWN": 0.6443},
    "test25": {"UP": 0.5140, "DOWN": 0.5816},
    "oos":    {"UP": 0.4184, "DOWN": 0.4646},
}

def mag_build(years, stride=1):
    """Same logic as nzdusd_15m_base.build() but returns |ret_15m| as magnitude target."""
    Xs=[]; mags=[]; mvs=[]; tss=[]
    for y in years:
        p=f"{FEAT_DIR}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p, columns=list(FEATS)+["close"])
        d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float)
        ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        contig=np.zeros(n,bool); contig[:n-HOR]=(ts[HOR:]-ts[:-HOR])==GAP
        fr=np.full(n,np.nan); fr[:n-HOR]=c[HOR:]/c[:-HOR]-1.0
        X=d[list(FEATS)].astype("float32")
        keepf=X.isna().mean(axis=1).values<0.5
        valid=contig & np.isfinite(fr) & keepf
        moved=valid & (fr!=0.0)
        idx=np.where(valid)[0]
        if stride>1: idx=idx[::stride]
        Xs.append(X.iloc[idx]); mags.append(np.abs(fr[idx]))
        mvs.append(moved[idx]); tss.append(ts[idx])
    return pd.concat(Xs), np.concatenate(mags), np.concatenate(mvs), np.concatenate(tss)

def nonoverlap(ts, mask):
    take=[]; block=-1
    for i in np.where(mask)[0]:
        if ts[i]<block: continue
        take.append(i); block=int(ts[i])+GAP
    return np.array(take, dtype=int)

def main():
    t0=time.time()

    # 1. Magnitude model
    print("[magdir] Building magnitude model on TRAIN ...", flush=True)
    Xtr,mtr,mvtr,_=mag_build(SPL["train"],6)
    Xva,mva,mvva,_=mag_build(SPL["val"])
    med=float(np.median(mtr[mvtr]))   # train-median on MOVED bars
    ytr_m=(mtr>=med).astype(int); yva_m=(mva>=med).astype(int)
    Mg=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=63,
        min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,
        n_estimators=2000,n_jobs=20,verbosity=-1)
    Mg.fit(Xtr,ytr_m,eval_set=[(Xva,yva_m)],eval_metric="auc",
           callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    mag_val_auc=float(roc_auc_score(yva_m[mvva],Mg.predict_proba(Xva[mvva])[:,1]))
    print(f"[magdir] Mag VAL AUC={mag_val_auc:.4f} iter={Mg.best_iteration_} {time.time()-t0:.0f}s", flush=True)

    # 2. Load frozen seed-ens direction models
    seeds=[lgb.Booster(model_file=f"{MODELS}/m15ny_NZDUSD_s{s}_lgb.txt") for s in range(3)]
    print(f"[magdir] Loaded {len(seeds)} frozen seed models {time.time()-t0:.0f}s", flush=True)

    res={"key":"NZDUSD.15m.ny.magdir","mag_train_median":float(med),"mag_val_auc":round(mag_val_auc,4),
         "conf_thr":CONF_THR,
         "lever":"E2 direction-conditioned-on-magnitude (own-pair NY seed-ens K=3 x HIGH/LOW magnitude buckets)",
         "falsifier":{"KILL_if":"HIGH-mag bucket does NOT lift binding-year UP or DOWN CI95-lo above incumbent floor by >0.02",
                      "incumbent_ci_lo":INCUMBENT_CI_LO},
         "years":{}}

    # 3. Held-out evaluation
    for w in ("test24","test25","oos"):
        Xw,yw,mvw,tsw=build(SPL[w])
        nyw=session_mask(tsw,"ny")
        pr_dir=np.mean([s.predict(Xw.values) for s in seeds],axis=0)
        pr_mag=Mg.predict_proba(Xw)[:,1]
        conf=np.abs(pr_dir-0.5)
        gated=nonoverlap(tsw, nyw & (conf>=CONF_THR))
        if len(gated)==0:
            res["years"][w]={"n_gated":0}; continue
        g_pdir=pr_dir[gated]; g_pmag=pr_mag[gated]
        g_y=yw[gated]; g_mv=mvw[gated]
        g_pred=(g_pdir>0.5).astype(int)
        g_win=((g_pred==g_y)&g_mv).astype(float)

        yr={"n_gated":int(len(gated)),"wr_all":round(float(g_win.mean()),4)}
        for bucket,bmask in [("HIGH",g_pmag>=0.5),("LOW",g_pmag<0.5)]:
            if bmask.sum()==0: yr[bucket]=None; continue
            bw=g_win[bmask]; bp=(g_pdir[bmask]>0.5).astype(int)
            bu=bp==1; bd=bp==0
            bd_out={"n":int(bmask.sum()),"wr":round(float(bw.mean()),4)}
            for side,sm in [("UP",bu),("DOWN",bd)]:
                if sm.sum()>0:
                    lo,hi=boot(bw[sm]); bd_out[side]={"n":int(sm.sum()),"wr":round(float(bw[sm].mean()),4),"ci":[round(lo,4),round(hi,4)]}
                else:
                    bd_out[side]={"n":0}
            yr[bucket]=bd_out
        res["years"][w]=yr

        print(f"  {w}: n={len(gated)} wr_all={g_win.mean():.4f} | HIGH n={int((g_pmag>=0.5).sum())} wr={float(g_win[g_pmag>=0.5].mean() if (g_pmag>=0.5).any() else float('nan')):.4f} | LOW n={int((g_pmag<0.5).sum())} wr={float(g_win[g_pmag<0.5].mean() if (g_pmag<0.5).any() else float('nan')):.4f}", flush=True)
        for bucket in ("HIGH","LOW"):
            b=yr.get(bucket)
            if b:
                for side in ("UP","DOWN"):
                    s=b.get(side)
                    if s and s.get("ci"): print(f"        {bucket}/{side}: n={s['n']} wr={s['wr']:.4f} CI[{s['ci'][0]:.3f},{s['ci'][1]:.3f}]", flush=True)

    # 4. Falsifier
    SE=0.02
    lifts=[]
    for w,yr in res["years"].items():
        b=yr.get("HIGH")
        if not b: continue
        for side in ("UP","DOWN"):
            s=b.get(side,{})
            ci=s.get("ci")
            if ci and not np.isnan(ci[0]):
                floor=INCUMBENT_CI_LO.get(w,{}).get(side,float("nan"))
                if not np.isnan(floor) and ci[0]>floor+SE:
                    lifts.append({"year":w,"side":side,"ci_lo":ci[0],"incumbent_floor":floor,"lift":round(ci[0]-floor,4)})

    beats_base_auc=mag_val_auc>0.55   # modest bar for any magnitude predictability
    res["verdict"]={
        "mag_val_auc":round(mag_val_auc,4),
        "lifts_above_threshold":lifts,
        "beats_base_auc":beats_base_auc,
        "KILLED":len(lifts)==0,
        "note":"E2 magdir KILLED → I3 |ret|-weight SUBSUMED (sign-invariance confirmed for NZDUSD)" if len(lifts)==0
               else "LIFT FOUND — check for overfit / small-n"}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[magdir] VERDICT: {'KILLED' if res['verdict']['KILLED'] else 'SURVIVED'} | lifts={lifts} -> {RESULT}  total={time.time()-t0:.0f}s", flush=True)

if __name__=="__main__":
    main()
