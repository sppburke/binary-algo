"""AUDUSD 15m NY edge — FROZEN-PAST FORWARD HOLDOUT (the mandatory trap#9 adversarial verification).

SCOPE: AUDUSD · 15m. The NY-session refit-CPCV CERTIFIED both sides (UP p10 .571/.583, DOWN .577/.590 @cov5/2,
15/15 paths). Refit-CPCV refits per fold, so it does NOT test forward transfer (skill trap #9: CPCV folds are
flanked by train folds → era-local memorization can certify with no forward transfer). The all-session FROZEN
base showed a 2026 UP COLLAPSE (.493) while DOWN held (.589). This script is the deployment-faithful check:
train ONCE on 2012-2021 NY-only, FREEZE, evaluate per-year 2024/2025/2026 NY-only at the cov gates with
bootstrap CI. Compares frozen-forward win-rate vs the refit-CPCV cert:
  frozen-forward ~= refit-CPCV  -> cert is HONEST (genuinely deployable, periodic-retrain floor = refit p10).
  frozen-forward << refit-CPCV  -> refit cert OVERSTATES forward transfer (regime-dependent; trust per-era refit).
Reports BOTH sides per year so we can see if NY rescues the all-session 2026 UP collapse.

Usage: ~/binary-algo-venv/bin/python audusd_15m_ny_frozen.py [stride] [leaves] [nseed]
"""
import sys, json, time, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sessions import session_mask
from audusd_15m_base import build, side_eval, covcurve, boot, mk_lgb, BE, SPL, FEATS

SESSION="ny"
def _argint(i,d): return int(sys.argv[i]) if len(sys.argv)>i and str(sys.argv[i]).isdigit() else d
TR_STRIDE=_argint(1,6); NUM_LEAVES=_argint(2,127); NSEED=_argint(3,1)
RESULT=f"audusd_15m_ny_frozen{'_seedens'+str(NSEED) if NSEED>1 else ''}_result.json"

def ny_restrict(X,y,mv,ts):
    m=session_mask(ts, SESSION)
    return X[m], y[m], mv[m], ts[m]

def main():
    t0=time.time()
    res={"key":"AUDUSD.15m","session":SESSION,"model":f"FROZEN 2012-21 NY-only base GBM (stride{TR_STRIDE} leaves{NUM_LEAVES} nseed{NSEED})",
         "purpose":"trap#9 frozen-past forward holdout — does the NY refit-CPCV cert transfer FORWARD (esp UP, which collapsed all-session in 2026)?",
         "breakeven":BE,"tr_stride":TR_STRIDE,"num_leaves":NUM_LEAVES,"nseed":NSEED,
         "refit_cpcv_cert":{"UP_p10":{"cov05":.5713,"cov03":.5719,"cov02":.5833},"DOWN_p10":{"cov05":.5773,"cov03":.5972,"cov02":.5897}},
         "falsifier":{"registered":"pre-OOS","note":"if a side's frozen-forward per-year win-rate at the gate is >2pp below its refit-CPCV p10 in the binding year, the refit cert OVERSTATES forward transfer for that side (regime-dependent)."}}
    json.dump(res,open(RESULT,"w"),indent=2)

    Xtr,ytr,mtr,ttr=build(SPL["train"], TR_STRIDE); Xtr,ytr,mtr,ttr=ny_restrict(Xtr,ytr,mtr,ttr)
    Xva,yva,mva,tva=build(SPL["val"]); Xva,yva,mva,tva=ny_restrict(Xva,yva,mva,tva)
    itr=mtr; iva=mva
    print(f"[ny-frozen] train_NY={int(itr.sum()):,} val_NY={int(iva.sum()):,} feats={len(FEATS)} nseed={NSEED} build={time.time()-t0:.0f}s",flush=True)
    boosters=[]
    pva=np.zeros(len(Xva))
    for sd in range(NSEED):
        L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=NUM_LEAVES,
            min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,
            n_estimators=3000,n_jobs=20,verbosity=-1,random_state=sd,bagging_seed=sd,feature_fraction_seed=sd)
        L.fit(Xtr[itr], ytr[itr], eval_set=[(Xva[iva], yva[iva])], eval_metric="auc",
              callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        boosters.append(L); pva+=L.predict_proba(Xva)[:,1]
    pva/=NSEED
    val_auc=float(roc_auc_score(yva[iva], pva[iva]))
    print(f"[ny-frozen] VAL_NY moved-AUC={val_auc:.4f} {time.time()-t0:.0f}s",flush=True)
    res["val_auc"]=val_auc

    # gate: VAL worst-half stability at each cov (report all covs, no single pick — we compare per-cov to refit cert)
    confv=np.abs(pva-0.5)
    res["years"]={}
    for w in ("test24","test25","oos"):
        Xw,yw,mw,tw=build(SPL[w]); Xw,yw,mw,tw=ny_restrict(Xw,yw,mw,tw)
        pr=np.mean([b.predict_proba(Xw)[:,1] for b in boosters],axis=0)
        auc=float(roc_auc_score(yw[mw], pr[mw])); uprate=float(yw[mw].mean())
        cc={}
        for cov in (0.05,0.03,0.02,0.01):
            thr=float(np.quantile(confv,1-cov))
            r=side_eval(pr,yw,mw,tw,thr)
            if r: cc[f"{cov}"]={k:{"n":v["n"],"wr":round(v["wr"],4),"ci_lo":round(v["ci"][0],4)} for k,v in r.items()}
        res["years"][w]={"moved_auc":round(auc,4),"up_rate":round(uprate,4),"tripwire_ok":bool(0.47<=uprate<=0.53),"bycov":cc}
        c5=cc.get("0.05",{}); c2=cc.get("0.02",{})
        print(f"=== {w} === AUC={auc:.4f} up={uprate:.4f} trip={res['years'][w]['tripwire_ok']}",flush=True)
        for cv,cd in (("cov5",c5),("cov2",c2)):
            print(f"    {cv}: COMB {cd.get('COMBINED')} | UP {cd.get('UP')} | DOWN {cd.get('DOWN')}",flush=True)

    # verdict: per side, does frozen-forward hold near the refit cert in the binding (worst) year?
    def side_min_over_years(side, cov):
        xs=[res["years"][w]["bycov"].get(f"{cov}",{}).get(side,{}).get("wr",float("nan")) for w in ("test24","test25","oos")]
        xs=[x for x in xs if np.isfinite(x)]; return min(xs) if xs else float("nan")
    res["frozen_forward_min_year"]={
        "UP_cov05":round(side_min_over_years("UP",0.05),4),"UP_cov02":round(side_min_over_years("UP",0.02),4),
        "DOWN_cov05":round(side_min_over_years("DOWN",0.05),4),"DOWN_cov02":round(side_min_over_years("DOWN",0.02),4)}
    fwd=res["frozen_forward_min_year"]
    res["verdict"]={
        "UP_forward_holds": bool(fwd["UP_cov05"]>=BE),
        "DOWN_forward_holds": bool(fwd["DOWN_cov05"]>=BE),
        "UP_overstated_by_refit": bool(fwd["UP_cov05"] < 0.5713-0.02),
        "DOWN_overstated_by_refit": bool(fwd["DOWN_cov05"] < 0.5773-0.02),
        "note":"frozen-2012-21 forward; compare per-year-min to refit-CPCV p10. UP collapsed all-session in 2026 — does NY rescue it forward?"}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[ny-frozen] frozen-forward per-year MIN: UP cov5 {fwd['UP_cov05']} cov2 {fwd['UP_cov02']} | DOWN cov5 {fwd['DOWN_cov05']} cov2 {fwd['DOWN_cov02']}",flush=True)
    print(f"[ny-frozen] verdict {res['verdict']}  -> {RESULT}  {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
