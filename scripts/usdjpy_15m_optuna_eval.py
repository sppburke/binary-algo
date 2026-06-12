"""Fast recovery: I6 Optuna held-out eval using the RECOVERED best params (study crashed post-tuning on a
predict_proba[:,1] bug; the 40-trial search completed: best worst-half VAL AUC .5437). Skips the re-tune;
fits default base vs the tuned config and evaluates frozen-past per-year (deriv-faithful, NY) — the same
null-detector as usdjpy_15m_optuna.py. Writes usdjpy_15m_optuna_result.json."""
import json, time, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sessions import session_mask
from usdjpy_15m_base import build, side_eval, boot, BE, SPL

SESSION="ny"; STRIDE=6; FIT_CAP=130_000
BEST=dict(num_leaves=299, learning_rate=0.006320332012933287, min_child_samples=994,
          subsample=0.7299578442746574, colsample_bytree=0.5362864481723499, reg_lambda=39.22199824544969,
          n_estimators=4000)
DEFAULT=dict(num_leaves=127, learning_rate=0.02, min_child_samples=400, subsample=0.8,
             colsample_bytree=0.5, reg_lambda=20.0, n_estimators=3000)

def ny(years, stride):
    X,y,m,ts=build(years,stride); return X,y,m,ts,session_mask(ts,SESSION)
def worst_half(pr,y,mv,ts):
    o=np.argsort(ts); pr,y,mv,ts=pr[o],y[o],mv[o],ts[o]; h=len(ts)//2; a=[]
    for sl in (slice(0,h),slice(h,None)):
        if mv[sl].sum()>50: a.append(roc_auc_score(y[sl][mv[sl]],pr[sl][mv[sl]]))
    return float(min(a)) if a else 0.5

def main():
    t0=time.time()
    Xtr,ytr,mtr,tstr,nytr=ny(SPL["train"],STRIDE); itr=mtr&nytr
    Xva,yva,mva,tsva,nyva=ny(SPL["val"],1); iva=mva&nyva
    rng=np.random.default_rng(13); fp=np.where(itr)[0]
    if len(fp)>FIT_CAP: fp=rng.choice(fp,FIT_CAP,replace=False)
    nyva_idx=np.where(nyva)[0]
    print(f"[opt-eval] fit={len(fp):,} val_moved={int(iva.sum()):,}  {time.time()-t0:.0f}s",flush=True)
    def heldout(params,tag):
        L=lgb.LGBMClassifier(objective="binary",metric="auc",n_jobs=16,verbosity=-1,random_state=0,
            bagging_seed=0,feature_fraction_seed=0,subsample_freq=1,**params)
        L.fit(Xtr.values[fp],ytr[fp],eval_set=[(Xva.values[iva],yva[iva])],eval_metric="auc",
              callbacks=[lgb.early_stopping(120),lgb.log_evaluation(0)])
        pva=L.predict_proba(Xva.values)[:,1]
        thr=float(np.quantile(np.abs(pva[nyva_idx]-0.5),1-0.03))
        out={"val_worst_half_auc":round(worst_half(pva[nyva_idx],yva[nyva_idx],mva[nyva_idx],tsva[nyva_idx]),4),
             "val_full_auc":round(float(roc_auc_score(yva[iva],pva[iva])),4),"years":{}}
        for w in ("test24","test25","oos"):
            Xw,yw,mw,tsw=build(SPL[w],1); wi=np.where(session_mask(tsw,SESSION))[0]
            pr=L.predict_proba(Xw.values)[:,1][wi]; mvw=mw[wi]
            auc=float(roc_auc_score(yw[wi][mvw],pr[mvw])) if mvw.sum()>20 else float("nan")
            g=side_eval(pr,yw[wi],mw[wi],tsw[wi],thr); cb=g["COMBINED"] if g else None
            out["years"][w]={"moved_auc":round(auc,4),"up_rate":round(float(yw[wi][mvw].mean()),4),
                "cov3_COMB":({"n":cb["n"],"wr":round(cb["wr"],4),"ci":[round(c,4) for c in cb["ci"]]} if cb else None)}
            print(f"  [{tag}] {w}: moved-AUC={auc:.4f} | cov3 COMB "+(f"n{cb['n']} wr={cb['wr']:.4f} CI[{cb['ci'][0]:.3f},{cb['ci'][1]:.3f}]" if cb else "—"),flush=True)
        return out
    print("[opt-eval] DEFAULT:",flush=True); d=heldout(DEFAULT,"default")
    print("[opt-eval] TUNED:",flush=True);   t=heldout(BEST,"tuned")
    yrs=("test24","test25","oos")
    beats=[w for w in yrs if (t["years"][w]["moved_auc"] or 0)>(d["years"][w]["moved_auc"] or 0)]
    clears=[w for w in yrs if t["years"][w]["cov3_COMB"] and t["years"][w]["cov3_COMB"]["ci"][0]>=BE]
    dmin=min(d["years"][w]["moved_auc"] for w in yrs); tmin=min(t["years"][w]["moved_auc"] for w in yrs)
    surv=bool(len(beats)>=2 and len(clears)>=2 and tmin>dmin)
    res={"key":"USDJPY.15m.ny","lever":"I6 Optuna TPE (worst-VAL-half obj), recovered eval","n_trials":40,
         "best_params":BEST,"best_val_worst_half_auc":0.5437,"default_config":DEFAULT,
         "default_heldout":d,"tuned_heldout":t,
         "verdict":{"SURVIVES":surv,"tuned_auc_beats_default_years":beats,"tuned_cov3_clears_years":clears,
                    "default_binding_auc":round(dmin,4),"tuned_binding_auc":round(tmin,4),
                    "note":("SURVIVED -> escalate to CPCV paired" if surv else
                            "KILLED: Optuna-tuned hyperparams do NOT beat default base on held-out binding year — "
                            "hyperparameters are not the constraint; ~.539 AUC bound holds. Matches USDJPY-2m anti-transfer.")}}
    json.dump(res,open("usdjpy_15m_optuna_result.json","w"),indent=2)
    print(f"\n[opt-eval] VERDICT: {'SURVIVED' if surv else 'KILLED'} (tuned binding AUC {tmin:.4f} vs default {dmin:.4f}; "
          f"auc-beats={beats}; cov3-clears={clears}) -> usdjpy_15m_optuna_result.json  {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__": main()
