"""Freeze the IMPROVED USDJPY 15m NY seed-ensemble direction book (Tier-I I2 winning combination).

SCOPE: USDJPY · 15m · NY · combined. Trains K seed-diverse own-pair base GBMs on TRAIN 2012-21 NY-moved
bars, averages their probabilities, freezes the cov2% gate on VAL 2022-23 NY worst-half. This is the
best certified config: NY seed-ens(K=3) refit-CPCV UP p10 .6005 / DOWN p10 .5738 @cov2% (15/15 paths).
Book id: USDJPY.m15ny_seedens.v1 (sibling of the single-model USDJPY.m15ny.v1).

Usage: ~/binary-algo-venv/bin/python usdjpy_15m_freeze_ny_seedens.py
"""
import os, json, time, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from sessions import session_mask
from usdjpy_15m_base import build, side_eval, BE, SPL
import manifest as MAN

PAIR="USDJPY"; SESSION="ny"; K=3; COV=0.02; NUM_LEAVES=127; GAP=900
os.makedirs("models", exist_ok=True); BOOK_ID="USDJPY.m15ny_seedens.v1"

def mk_lgb(seed):
    return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=NUM_LEAVES,
        min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,
        n_estimators=3000,n_jobs=16,verbosity=-1,random_state=seed,bagging_seed=seed,feature_fraction_seed=seed)

def main():
    t0=time.time()
    Xtr,ytr,mtr,tstr=build(SPL["train"],6); Xva,yva,mva,tsv=build(SPL["val"])
    nytr=session_mask(tstr,SESSION); nyva=session_mask(tsv,SESSION); itr=mtr&nytr; iva=mva&nyva
    models=[]; pva=np.zeros(len(yva))
    for s in range(K):
        L=mk_lgb(s); L.fit(Xtr[itr],ytr[itr],eval_set=[(Xva[iva],yva[iva])],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
        mp=f"models/m15ny_seedens_USDJPY_s{s}_lgb.txt"; L.booster_.save_model(mp); models.append(mp)
        pva+=L.predict_proba(Xva)[:,1]; print(f"[freeze-se] seed {s} best_iter={L.best_iteration_} ({time.time()-t0:.0f}s)",flush=True)
    pva/=K
    val_auc=float(roc_auc_score(yva[iva],pva[iva]))
    THR=float(np.quantile(np.abs(pva[np.where(nyva)[0]]-0.5),1-COV))
    print(f"[freeze-se] K={K} VAL(NY) AUC={val_auc:.4f} cov{COV:.0%} thr={THR:.4f}",flush=True)
    boosters=[lgb.Booster(model_file=m) for m in models]
    fwd={}
    for w in ("test24","test25","oos"):
        Xw,yw,mw,tsw=build(SPL[w]); wi=np.where(session_mask(tsw,SESSION))[0]
        pr=np.mean([b.predict(Xw.values)[wi] for b in boosters],axis=0)
        g=side_eval(pr,yw[wi],mw[wi],tsw[wi],THR)
        fwd[w]={k:{"n":g[k]["n"],"wr":round(g[k]["wr"],4),"ci":[round(c,4) for c in g[k]["ci"]]} for k in ("COMBINED","UP","DOWN")} if g else None
        print(f"  {w}: "+" | ".join(f"{k} n{fwd[w][k]['n']} {fwd[w][k]['wr']}" for k in ("COMBINED","UP","DOWN")),flush=True)
    strat={"book":BOOK_ID,"pair":PAIR,"timeframe":"15m","horizon_min":15,"session":SESSION,"seed_ensemble_K":K,
           "label":"sign(close[t+15]-close[t]), ties LOSE","settlement":"bar-close, gap=900s nonoverlap",
           "gate":{"cov":COV,"conf_thr":THR,"rule":"avg of K seed models; trade |p-0.5|>=thr AND ts in NY session"},
           "breakeven":BE,"val_ny_auc":val_auc,
           "cert":"NY seed-ens(K=3) per-fold-refit CPCV: UP p10 .6005 / DOWN p10 .5738 / COMB p10 .5962 @cov2% (15/15) — usdjpy_15m_cpcv_session_ny_seedens3_result.json",
           "deploy_note":"Best certified config. Refit-dependent (retrain periodically; size on refit floor). UP robust frozen-forward; DOWN refit-dependent.",
           "frozen_forward":fwd,"models":models}
    strat_path="models/m15ny_seedens_USDJPY_strategy.json"; json.dump(strat,open(strat_path,"w"),indent=2)
    man=MAN.build(book_id=BOOK_ID,timeframe="15m",side="combined",role="direction",
                  script="usdjpy_15m_cpcv_session.py ny ... 3 (cert) + usdjpy_15m_freeze_ny_seedens.py (freeze)",
                  summary="USDJPY 15m NY-session own-pair seed-ensemble(K=3) GBM; BOTH sides certified NY refit-CPCV (UP p10 .6005 / DOWN .5738 @cov2%, 15/15). Best USDJPY 15m direction config. NY-concentrated + USDJPY-specific.",
                  metrics={"up_refit_cpcv_p10":0.6005,"down_refit_cpcv_p10":0.5738,"comb_refit_cpcv_p10":0.5962,
                           "up_mean":0.617,"down_mean":0.616,"breakeven":BE,"cov":COV,"session":"ny","seed_K":K,
                           "val_ny_auc":round(val_auc,4),"frozen_forward":fwd},
                  artifacts=models, hyperparams={"num_leaves":NUM_LEAVES,"lr":0.02,"reg_lambda":20,"n_est":3000,"K":K,"tr_stride":6},
                  strategy_json=strat_path, feature_fingerprint=MAN.dir_fingerprint(H.FEAT_DIR,("USDJPY_*.parquet",)),
                  created_utc="2026-06-08",
                  notes="seed-ensemble avg of K=3; cov2% gate frozen on VAL NY worst-half; NY=America/New_York 08-17 DST-correct. DOWN refit-dependent.")
    man["currency"]="USDJPY"
    path=MAN.freeze(man,artifacts_src=models+[strat_path]); json.dump(man,open(f"books/{BOOK_ID}.manifest.json","w"),indent=2)
    print(f"[freeze-se] FROZEN {BOOK_ID} content_id={man['content_id']} -> {path} ({time.time()-t0:.0f}s)",flush=True)

if __name__=="__main__":
    main()
