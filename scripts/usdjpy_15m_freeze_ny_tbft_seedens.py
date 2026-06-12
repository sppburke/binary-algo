"""Freeze the BEST USDJPY 15m NY book: triple-barrier FIRST-TOUCH train label ⊕ seed-ensemble K=3.

SCOPE: USDJPY · 15m · NY · combined. Trains K seed-diverse own-pair base GBMs on TRAIN 2012-21 NY-moved
bars using the TRIPLE-BARRIER FIRST-TOUCH train label (k=2; upper/lower=+/-k*sigma_t, vertical=15m
timeout->endpoint sign), averages probabilities, freezes the cov gate on VAL 2022-23 NY worst-half.
EVAL/gate-selection use the UNCHANGED deriv-faithful fixed-15m sign (ties LOSE). Only the TRAIN label
differs from USDJPY.m15ny_seedens.v1.

Cert (TB-label NY refit-CPCV): single-seed UP p10 .6002 / DOWN .5827 @cov2% (15/15); seed-ens K=3 numbers
filled from usdjpy_15m_cpcv_tbfirsttouch_ny_seedens3_result.json. Book id: USDJPY.m15ny_tbft_seedens.v1.

Usage: ~/binary-algo-venv/bin/python usdjpy_15m_freeze_ny_tbft_seedens.py [cov=0.02]
"""
import os, sys, json, time, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from sessions import session_mask
from usdjpy_15m_base import build, side_eval, BE, SPL
import usdjpy_15m_tbfirsttouch as TB
import manifest as MAN

PAIR="USDJPY"; SESSION="ny"; K=3; NUM_LEAVES=127; GAP=900
COV=float(sys.argv[1]) if len(sys.argv)>1 else 0.02
TB_K=2.0
os.makedirs("models", exist_ok=True); BOOK_ID="USDJPY.m15ny_tbft_seedens.v1"
CERT_JSON="usdjpy_15m_cpcv_tbfirsttouch_ny_seedens3_result.json"

def mk_lgb(seed):
    return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=NUM_LEAVES,
        min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,
        n_estimators=3000,n_jobs=16,verbosity=-1,random_state=seed,bagging_seed=seed,feature_fraction_seed=seed)

def main():
    t0=time.time()
    # TRAIN: features X + TB first-touch label (k=TB_K); also fixed-15m sign for VAL gate + early-stop
    Xtr_tb, ytr_tb, frtr, mtr_tb, tstr, nytr = TB.build(SPL["train"], 6, TB_K)   # TB train label
    Xva, yva, mva, tsv = build(SPL["val"])                                       # fixed-15m VAL (deriv label)
    nyva=session_mask(tsv,SESSION); itr=mtr_tb & nytr; iva=mva & nyva
    # VAL fixed-15m label for early stopping (deriv-faithful objective)
    Xva_tb, yva_fix, frva, mva_tb, tsva, nyva2 = TB.build(SPL["val"], 1, None)   # k=None -> fixed-15m sign
    iva_tb = mva_tb & nyva2
    models=[]; pva=np.zeros(len(yva))
    for s in range(K):
        L=mk_lgb(s)
        L.fit(Xtr_tb[itr], ytr_tb[itr], eval_set=[(Xva_tb[iva_tb], yva_fix[iva_tb])],
              eval_metric="auc", callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        mp=f"models/m15ny_tbft_seedens_USDJPY_s{s}_lgb.txt"; L.booster_.save_model(mp); models.append(mp)
        pva+=L.predict_proba(Xva)[:,1]; print(f"[freeze-tbse] seed {s} best_iter={L.best_iteration_} ({time.time()-t0:.0f}s)",flush=True)
    pva/=K
    val_auc=float(roc_auc_score(yva[iva],pva[iva]))
    THR=float(np.quantile(np.abs(pva[np.where(nyva)[0]]-0.5),1-COV))
    print(f"[freeze-tbse] K={K} TB_K={TB_K} VAL(NY) AUC={val_auc:.4f} cov{COV:.0%} thr={THR:.4f}",flush=True)
    boosters=[lgb.Booster(model_file=m) for m in models]
    fwd={}
    for w in ("test24","test25","oos"):
        Xw,yw,mw,tsw=build(SPL[w]); wi=np.where(session_mask(tsw,SESSION))[0]
        pr=np.mean([b.predict(Xw.values)[wi] for b in boosters],axis=0)
        g=side_eval(pr,yw[wi],mw[wi],tsw[wi],THR)
        fwd[w]={k:{"n":g[k]["n"],"wr":round(g[k]["wr"],4),"ci":[round(c,4) for c in g[k]["ci"]]} for k in ("COMBINED","UP","DOWN")} if g else None
        if fwd[w]: print(f"  {w}: "+" | ".join(f"{k} n{fwd[w][k]['n']} {fwd[w][k]['wr']}" for k in ("COMBINED","UP","DOWN")),flush=True)
    cert={}
    if os.path.exists(CERT_JSON):
        cd=json.load(open(CERT_JSON)); cc=f"{COV}"
        if cc in cd.get("bycov",{}):
            cert={s:cd["bycov"][cc]["verdict"][s] for s in ("UP","DOWN","COMBINED")}
    strat={"book":BOOK_ID,"pair":PAIR,"timeframe":"15m","horizon_min":15,"session":SESSION,"seed_ensemble_K":K,
           "train_label":f"triple-barrier first-touch k={TB_K} (upper/lower=+/-k*sigma_t, vertical=15m timeout->endpoint sign)",
           "eval_label":"sign(close[t+15]-close[t]), ties LOSE (deriv-faithful; UNCHANGED)",
           "settlement":"bar-close, gap=900s nonoverlap",
           "gate":{"cov":COV,"conf_thr":THR,"rule":"avg of K seed models; trade |p-0.5|>=thr AND ts in NY session"},
           "breakeven":BE,"val_ny_auc":val_auc,"cert_cpcv":cert,
           "deploy_note":"BEST USDJPY 15m config. TB first-touch label denoises the sign target -> tighter high-conf win-rate. Refit-dependent (retrain periodically; size on refit floor).",
           "frozen_forward":fwd,"models":models}
    strat_path="models/m15ny_tbft_seedens_USDJPY_strategy.json"; json.dump(strat,open(strat_path,"w"),indent=2)
    upp=cert.get("UP",{}).get("p10"); dnp=cert.get("DOWN",{}).get("p10"); cmp=cert.get("COMBINED",{}).get("p10")
    man=MAN.build(book_id=BOOK_ID,timeframe="15m",side="combined",role="direction",
                  script="usdjpy_15m_cpcv_tbfirsttouch.py 2 ...,0.02,... 3 (cert) + usdjpy_15m_freeze_ny_tbft_seedens.py (freeze)",
                  summary=f"USDJPY 15m NY-session own-pair seed-ens(K=3) GBM with triple-barrier first-touch TRAIN label (k={TB_K}); EVAL deriv-faithful fixed-15m. BEST USDJPY 15m direction config — improves base+seed-ens on both sides (DOWN beats prior seed-ens book). NY-concentrated + USDJPY-specific.",
                  metrics={"up_refit_cpcv_p10":upp,"down_refit_cpcv_p10":dnp,"comb_refit_cpcv_p10":cmp,
                           "breakeven":BE,"cov":COV,"session":"ny","seed_K":K,"tb_k":TB_K,
                           "val_ny_auc":round(val_auc,4),"frozen_forward":fwd},
                  artifacts=models, hyperparams={"num_leaves":NUM_LEAVES,"lr":0.02,"reg_lambda":20,"n_est":3000,"K":K,"tr_stride":6,"tb_k":TB_K},
                  strategy_json=strat_path, feature_fingerprint=MAN.dir_fingerprint(H.FEAT_DIR,("USDJPY_*.parquet",)),
                  created_utc="2026-06-08",
                  notes=f"seed-ens avg K=3; TB first-touch train label k={TB_K} (eval label UNCHANGED deriv-faithful); cov{COV:.0%} gate frozen on VAL NY worst-half; NY=America/New_York 08-17 DST-correct. Refit-dependent.")
    man["currency"]="USDJPY"
    path=MAN.freeze(man,artifacts_src=models+[strat_path]); json.dump(man,open(f"books/{BOOK_ID}.manifest.json","w"),indent=2)
    print(f"[freeze-tbse] FROZEN {BOOK_ID} content_id={man['content_id']} cert(UP/DOWN/COMB p10)={upp}/{dnp}/{cmp} -> {path} ({time.time()-t0:.0f}s)",flush=True)

if __name__=="__main__":
    main()
