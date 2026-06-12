"""USDJPY 1-MIN — Tier-I I6: Optuna TPE hyperparameter search, selected on the DEPLOY objective.

SCOPE: USDJPY · 1m. I only sampled 2 GBM configs (127/255 leaves). This sweeps LGBM hyperparams with Optuna
TPE, selecting on WORST-VAL-HALF combined win-rate at cov2% (the actual deploy objective, never VAL-acc-max).
Tests whether tuning can cross breakeven where the keystone (online-ARF .50) says the wall is signal-level.
Usage: ~/binary-algo-venv/bin/python usdjpy_1m_optuna.py [n_trials] [stride]
"""
import sys, os, json, time, numpy as np
import lightgbm as lgb, optuna
from sklearn.metrics import roc_auc_score
from usdjpy_1m_base import build, side_eval, covcurve, nonoverlap_chrono, boot
import harness as H

BE=0.541; SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],
     "test24":["2024"],"test25":["2025"],"oos":["2026"]}
NTRIALS=int(sys.argv[1]) if len(sys.argv)>1 else 10
STRIDE=int(sys.argv[2]) if len(sys.argv)>2 else 12
RESULT="usdjpy_1m_optuna_result.json"
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("loading...",flush=True)
Xtr,ytr,mtr,_=build(SPL["train"],STRIDE); Xva,yva,mva,tsv=build(SPL["val"])
half=len(yva)//2

def worst_half_wr(pva, cov=0.02):
    confv=np.abs(pva-0.5); thr=float(np.quantile(confv,1-cov)); accs=[]
    for s,e in ((0,half),(half,len(pva))):
        r=side_eval(pva[s:e],yva[s:e],mva[s:e],tsv[s:e],thr); accs.append(r["COMBINED"]["wr"] if r else np.nan)
    return float(np.nanmin(accs))

def objective(trial):
    p=dict(objective="binary",metric="auc",n_jobs=20,verbosity=-1,
        learning_rate=trial.suggest_float("lr",0.01,0.05,log=True),
        num_leaves=trial.suggest_int("leaves",63,511),
        min_child_samples=trial.suggest_int("mcs",100,800),
        colsample_bytree=trial.suggest_float("cs",0.4,0.9),
        subsample=trial.suggest_float("ss",0.6,0.95), subsample_freq=1,
        reg_lambda=trial.suggest_float("rl",1.0,40.0,log=True),
        n_estimators=2500)
    m=lgb.LGBMClassifier(**p)
    m.fit(Xtr[mtr],ytr[mtr],eval_set=[(Xva[mva],yva[mva])],eval_metric="auc",
          callbacks=[lgb.early_stopping(120),lgb.log_evaluation(0)])
    return worst_half_wr(m.predict_proba(Xva)[:,1])

def main():
    t0=time.time()
    st=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=7))
    st.optimize(objective,n_trials=NTRIALS)
    best=st.best_params; print(f"[optuna] best worst-VAL-half WR={st.best_value:.4f} params={best} {time.time()-t0:.0f}s",flush=True)
    # eval best config on held-out
    p=dict(objective="binary",metric="auc",n_jobs=20,verbosity=-1,n_estimators=3000,subsample_freq=1,
        learning_rate=best["lr"],num_leaves=best["leaves"],min_child_samples=best["mcs"],
        colsample_bytree=best["cs"],subsample=best["ss"],reg_lambda=best["rl"])
    m=lgb.LGBMClassifier(**p); m.fit(Xtr[mtr],ytr[mtr],eval_set=[(Xva[mva],yva[mva])],eval_metric="auc",
        callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    res={"key":"USDJPY.1m","lever":"I6 Optuna(worst-VAL-half)","n_trials":NTRIALS,"best_params":best,
         "best_val_worsthalf_wr":float(st.best_value),"val_auc":float(roc_auc_score(yva[mva],m.predict_proba(Xva)[mva][:,1])),"years":{}}
    for w in ("test24","test25","oos"):
        Xw,yw,mw,tsw=build(SPL[w]); pr=m.predict_proba(Xw)[:,1]; cc=covcurve(pr,yw,mw,tsw)
        res["years"][w]={"moved_auc":float(roc_auc_score(yw[mw],pr[mw])),"covcurve":cc}
        def g(cov,side): d=cc[cov].get(side,{}); return f"{d.get('wr')}(n{d.get('n')})"
        print(f"=== {w} === AUC={res['years'][w]['moved_auc']:.4f} | UP cov2%:{g('0.02','UP')} cov1%:{g('0.01','UP')} | DOWN cov2%:{g('0.02','DOWN')}",flush=True)
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[optuna] saved {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
