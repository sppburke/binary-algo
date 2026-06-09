"""USDJPY 15m — I6 Optuna hyperparameter search (NY session), the last Tier-I lever, RUN not subsumed.

SCOPE: USDJPY · 15m · NY. USDJPY-2m Optuna ANTI-TRANSFERRED (VAL up, OOS worse; corr(VAL,OOS)=-.54).
This re-tests I6 at 15m NY rather than citing the 2m null. Optuna (TPE) maximizes the WORST-VAL-HALF
moved-AUC (per the skill's worst-half stability selection — NEVER VAL-acc-max), then the tuned config is
evaluated frozen-past per-year (deriv-faithful: NY decision rows, ties LOSE, nonoverlap gap=900, cov3%
gate from VAL) AGAINST the DEFAULT base (A1: num_leaves 127, lr .02). This is a fair RELATIVE null-detector:
if tuned does not beat default frozen-past it cannot beat the seed-ens deliverable (refit-CPCV) either.

Falsifier (pre-registered): SURVIVE only if tuned held-out moved-AUC > default in >=2 held-out years AND
tuned cov3% COMBINED CI-lo clears BE in >=2 years AND beats default's binding (worst) year. Else KILLED
(reproduces 2m anti-transfer / confirms the ~.539 AUC bound is not a hyperparameter deficiency). If it
SURVIVES, escalate to the full NY refit-CPCV with a PAIRED per-path comparison vs the matched incumbent.

Usage: ~/binary-algo-venv/bin/python usdjpy_15m_optuna.py [n_trials=40] [stride=6]
"""
import os, sys, json, time, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import optuna
import harness as H
from sessions import session_mask
from usdjpy_15m_base import build, side_eval, covcurve, boot, BE, SPL, FEATS

PAIR="USDJPY"; SESSION="ny"; HOR=15; GAP=900
N_TRIALS=int(sys.argv[1]) if len(sys.argv)>1 and sys.argv[1].isdigit() else 40
STRIDE=int(sys.argv[2]) if len(sys.argv)>2 and sys.argv[2].isdigit() else 6
FIT_CAP=130_000
RESULT="usdjpy_15m_optuna_result.json"
optuna.logging.set_verbosity(optuna.logging.WARNING)

def ny_rows(years, stride):
    X,y,m,ts=build(years,stride); ny=session_mask(ts,SESSION)
    return X, y, m, ts, ny

def worst_half_auc(pr, y, moved, ts):
    """min moved-AUC over the two chronological halves of the (already NY) rows."""
    o=np.argsort(ts); pr,y,moved,ts=pr[o],y[o],moved[o],ts[o]
    half=len(ts)//2; aucs=[]
    for sl in (slice(0,half), slice(half,None)):
        mv=moved[sl]
        if mv.sum()>50: aucs.append(roc_auc_score(y[sl][mv], pr[sl][mv]))
    return float(min(aucs)) if aucs else 0.5

def fit_eval(params, Xtr,ytr,itr, Xva,yva,mva,tsva,iva, seed=0):
    L=lgb.LGBMClassifier(objective="binary",metric="auc",n_jobs=16,verbosity=-1,
        random_state=seed,bagging_seed=seed,feature_fraction_seed=seed,
        subsample_freq=1, n_estimators=params.pop("n_estimators",2000), **params)
    L.fit(Xtr[itr], ytr[itr], eval_set=[(Xva[iva], yva[iva])], eval_metric="auc",
          callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)])
    pva=L.predict_proba(Xva)[:,1]
    return L, pva

def main():
    t0=time.time()
    print(f"[optuna] building NY train/val (stride {STRIDE})...", flush=True)
    Xtr,ytr,mtr,tstr,nytr = ny_rows(SPL["train"], STRIDE); itr=mtr&nytr
    Xva,yva,mva,tsva,nyva = ny_rows(SPL["val"], 1); iva=mva&nyva
    # cap fit pool
    rng=np.random.default_rng(13); fitidx=np.where(itr)[0]
    if len(fitidx)>FIT_CAP: fitidx=rng.choice(fitidx,FIT_CAP,replace=False)
    itr_mask=np.zeros(len(ytr),bool); itr_mask[fitidx]=True
    print(f"[optuna] NY train moved={int(itr.sum()):,} (fit {int(itr_mask.sum()):,}) val moved={int(iva.sum()):,}  {time.time()-t0:.0f}s", flush=True)

    nyva_idx=np.where(nyva)[0]
    def objective(trial):
        params=dict(
            num_leaves=trial.suggest_int("num_leaves",31,511),
            learning_rate=trial.suggest_float("learning_rate",0.005,0.05,log=True),
            min_child_samples=trial.suggest_int("min_child_samples",100,1000),
            subsample=trial.suggest_float("subsample",0.6,0.95),
            colsample_bytree=trial.suggest_float("colsample_bytree",0.3,0.8),
            reg_lambda=trial.suggest_float("reg_lambda",1.0,50.0,log=True),
            n_estimators=2000)
        L,pva=fit_eval(dict(params), Xtr,ytr,itr_mask, Xva,yva,mva,tsva,iva)
        wha=worst_half_auc(pva[nyva_idx], yva[nyva_idx], mva[nyva_idx], tsva[nyva_idx])
        return wha
    study=optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=13))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best=study.best_params; best_wha=study.best_value
    print(f"[optuna] best worst-half VAL AUC={best_wha:.4f} params={best}  {time.time()-t0:.0f}s", flush=True)

    # default base config (A1)
    default=dict(num_leaves=127, learning_rate=0.02, min_child_samples=400, subsample=0.8,
                 colsample_bytree=0.5, reg_lambda=20.0, n_estimators=3000)

    def heldout(params, tag):
        L,pva=fit_eval(dict(params), Xtr,ytr,itr_mask, Xva,yva,mva,tsva,iva)
        vconf=np.abs(pva[nyva_idx]-0.5); thr=float(np.quantile(vconf,1-0.03))   # cov3% gate from VAL NY
        out={"val_worst_half_auc":worst_half_auc(pva[nyva_idx],yva[nyva_idx],mva[nyva_idx],tsva[nyva_idx]),
             "val_full_auc":float(roc_auc_score(yva[iva],pva[iva])),"years":{}}
        for w in ("test24","test25","oos"):
            Xw,yw,mw,tsw=build(SPL[w],1); wi=np.where(session_mask(tsw,SESSION))[0]
            pr=L.predict_proba(Xw.values)[:,1][wi]
            mvw=mw[wi]; auc=float(roc_auc_score(yw[wi][mvw], pr[mvw])) if mvw.sum()>20 else float("nan")
            g=side_eval(pr, yw[wi], mw[wi], tsw[wi], thr)
            comb=g["COMBINED"] if g else None
            out["years"][w]={"moved_auc":round(auc,4),"up_rate":round(float(yw[wi][mvw].mean()),4),
                             "cov3_COMB":({"n":comb["n"],"wr":round(comb["wr"],4),"ci":[round(c,4) for c in comb["ci"]]} if comb else None)}
            print(f"  [{tag}] {w}: moved-AUC={auc:.4f} | cov3% COMB " + (f"n{comb['n']} wr={comb['wr']:.4f} CI[{comb['ci'][0]:.3f},{comb['ci'][1]:.3f}]" if comb else "—"), flush=True)
        return out

    print("[optuna] === DEFAULT base held-out ===", flush=True)
    d_eval=heldout(default,"default")
    print("[optuna] === TUNED held-out ===", flush=True)
    t_eval=heldout(best,"tuned")

    yrs=("test24","test25","oos")
    tuned_auc_beats=[w for w in yrs if (t_eval["years"][w]["moved_auc"] or 0) > (d_eval["years"][w]["moved_auc"] or 0)]
    tuned_cov_clears=[w for w in yrs if t_eval["years"][w]["cov3_COMB"] and t_eval["years"][w]["cov3_COMB"]["ci"][0]>=BE]
    default_min_auc=min(d_eval["years"][w]["moved_auc"] for w in yrs)
    tuned_min_auc=min(t_eval["years"][w]["moved_auc"] for w in yrs)
    survives=bool(len(tuned_auc_beats)>=2 and len(tuned_cov_clears)>=2 and tuned_min_auc>default_min_auc)
    res={"key":"USDJPY.15m.ny","lever":"I6 Optuna TPE hyperparameter search (worst-VAL-half objective)",
         "n_trials":N_TRIALS,"stride":STRIDE,"best_params":best,"best_val_worst_half_auc":round(best_wha,4),
         "default_config":default,"default_heldout":d_eval,"tuned_heldout":t_eval,
         "falsifier":{"registered_utc":"pre-heldout","SURVIVE_if":"tuned moved-AUC>default in >=2 yrs AND cov3% COMB CI-lo>=BE in >=2 yrs AND tuned binding-year AUC>default binding-year AUC"},
         "verdict":{"SURVIVES":survives,"tuned_auc_beats_default_years":tuned_auc_beats,"tuned_cov3_clears_years":tuned_cov_clears,
                    "default_binding_auc":round(default_min_auc,4),"tuned_binding_auc":round(tuned_min_auc,4),
                    "note":("SURVIVED -> escalate to NY refit-CPCV paired vs incumbent" if survives else
                            "KILLED: Optuna-tuned hyperparams do NOT beat the default base on held-out (binding year) — hyperparameters are not the constraint; the ~.539 AUC bound holds. Consistent with USDJPY-2m anti-transfer.")}}
    json.dump(res, open(RESULT,"w"), indent=2)
    print(f"\n[optuna] VERDICT: {'SURVIVED' if survives else 'KILLED'} (tuned binding AUC {tuned_min_auc:.4f} vs default {default_min_auc:.4f}; "
          f"auc-beats={tuned_auc_beats}; cov3-clears={tuned_cov_clears}) -> {RESULT}  {time.time()-t0:.0f}s", flush=True)

if __name__=="__main__":
    main()
