"""USDJPY 2-MIN — Optuna hyperparameter search on the POOLED model (worst-VAL-half objective, ACTUALLY RUN).

SCOPE: USDJPY · 2m. The pooled model used fixed hyperparams (255 leaves etc.). 1m Optuna was SINGLE-PAIR (.5347<BE);
the POOLED model was never tuned. Tune num_leaves/min_child/colsample/reg_lambda/lr to MAXIMIZE the worst-VAL-half
COMBINED win-rate at cov2% (the deploy objective, NOT VAL-AUC-max). If a config lifts the worst-VAL-half meaningfully,
escalate to CPCV. Falsifier: KILL if best worst-VAL-half COMBINED wr <= 0.541 (can't even pass VAL) OR no held-out
UP cov2% beats C1 .547/.546/.570 CI-separated.
Usage: ~/binary-algo-venv/bin/python usdjpy_2m_optuna.py [n_trials=14] [stride=42]
"""
import os, sys, json, time, numpy as np, pandas as pd
import lightgbm as lgb, optuna
from sklearn.metrics import roc_auc_score
import harness as H
from usdjpy_2m_base import side_eval, covcurve, nonoverlap_chrono, boot
from usdjpy_2m_xpair import build_pair, build_pool, PAIRS

optuna.logging.set_verbosity(optuna.logging.WARNING)
TARGET="USDJPY"; BE=0.541
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],
     "test24":["2024"],"test25":["2025"],"oos":["2026"]}
NTRIAL=int(sys.argv[1]) if len(sys.argv)>1 and sys.argv[1].isdigit() else 14
STRIDE=int(sys.argv[2]) if len(sys.argv)>2 and sys.argv[2].isdigit() else 42
RESULT="usdjpy_2m_optuna_result.json"

def main():
    t0=time.time()
    Xtr,ytr,mtr=build_pool(SPL["train"],STRIDE)
    Xva,yva,mva,tsv=build_pair(TARGET,SPL["val"],1)
    half=len(Xva)//2
    print(f"[optuna] pooled-train={int(mtr.sum()):,} val={int(mva.sum()):,} build={time.time()-t0:.0f}s",flush=True)
    Xtr_f=Xtr[mtr]; ytr_f=ytr[mtr]

    def worst_half(pva):
        confv=np.abs(pva-0.5); thr=float(np.quantile(confv,1-0.02)); accs=[]
        for s,e in ((0,half),(half,len(pva))):
            r=side_eval(pva[s:e],yva[s:e],mva[s:e],tsv[s:e],thr); accs.append(r["COMBINED"]["wr"] if r else float("nan"))
        return float(np.nanmin(accs))

    def objective(tr):
        p=dict(objective="binary",metric="auc",
               learning_rate=tr.suggest_float("lr",0.01,0.05,log=True),
               num_leaves=tr.suggest_int("num_leaves",63,511),
               min_child_samples=tr.suggest_int("min_child",100,800),
               colsample_bytree=tr.suggest_float("colsample",0.3,0.8),
               subsample=tr.suggest_float("subsample",0.6,0.95),subsample_freq=1,
               reg_lambda=tr.suggest_float("reg_lambda",1,50,log=True),
               n_estimators=tr.suggest_int("n_est",400,1200),n_jobs=20,verbosity=-1)
        L=lgb.LGBMClassifier(**p); L.fit(Xtr_f,ytr_f)
        return worst_half(L.predict_proba(Xva)[:,1])

    study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=7))
    study.optimize(objective,n_trials=NTRIAL,show_progress_bar=False)
    best=study.best_params; bestval=study.best_value
    print(f"[optuna] best worst-VAL-half COMBINED wr={bestval:.4f} (BE .541) params={best} {time.time()-t0:.0f}s",flush=True)
    res={"key":"USDJPY.2m","model":"POOLED Optuna (worst-VAL-half cov2% objective)","n_trials":NTRIAL,
         "best_worst_val_half":bestval,"best_params":best,
         "falsifier":{"KILL_if":"best worst-VAL-half <=0.541 OR no held-out UP cov2% beats C1 CI-sep"},"years":{}}
    # refit best, held-out
    L=lgb.LGBMClassifier(objective="binary",metric="auc",n_jobs=20,verbosity=-1,subsample_freq=1,**best)
    L.fit(Xtr_f,ytr_f)
    confv=np.abs(L.predict_proba(Xva)[:,1]-0.5); THR=float(np.quantile(confv,1-0.02))
    for w in ("test24","test25","oos"):
        Xw,yw,mw,tsw=build_pair(TARGET,SPL[w],1); pr=L.predict_proba(Xw)[:,1]
        g=side_eval(pr,yw,mw,tsw,THR); cc=covcurve(pr,yw,mw,tsw)
        res["years"][w]={"moved_auc":float(roc_auc_score(yw[mw],pr[mw])),"gate_cov2":g,"covcurve":cc}
        u=g["UP"] if g else {}; print(f"=== {w} === AUC={res['years'][w]['moved_auc']:.4f} | UP cov2% n{u.get('n')} wr={u.get('wr')} CI{u.get('ci')}",flush=True)
    res["verdict"]={"KILLED": bool(bestval<=BE)}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[optuna] saved {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
