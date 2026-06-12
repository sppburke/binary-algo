"""Freeze the CERTIFIED AUDUSD 15m NY-session own-pair direction book (the deployable deliverable).

SCOPE: AUDUSD · 15m · combined (both sides via confidence selection). Trains the own-pair base GBM (optionally
a K-seed ensemble) on TRAIN 2012-21 restricted to NY-session decision rows (features causal/continuous),
freezes the cov3% gate on VAL 2022-23 NY worst-half, records the frozen-forward per-year side-split, saves the
booster(s) + strategy.json, then manifest.build + freeze + writes the book manifest.

CERTIFICATION OF RECORD = per-fold-refit NY-CPCV (audusd_15m_cpcv_session_ny_*_result.json): both sides
p10 ≥ BE, 15/15 paths, every cov. REFIT-DEPENDENT — the frozen-2021 vintage decays forward (.60→.57→.53 by
2026); deploy NY-only with periodic retraining and size on the refit per-era floor, NOT the frozen book.

Usage: ~/binary-algo-venv/bin/python audusd_15m_freeze_ny.py [nseed=1]
  nseed=1 -> book AUDUSD.m15ny.v1 ; nseed>1 -> book AUDUSD.m15ny_seedens.v1
"""
import os, sys, json, time, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from sessions import session_mask
from audusd_15m_base import build, side_eval, BE, SPL
import manifest as MAN

PAIR="AUDUSD"; HOR=15; GAP=900; SESSION="ny"; NUM_LEAVES=127
NSEED=int(sys.argv[1]) if len(sys.argv)>1 and sys.argv[1].isdigit() else 1
MODELS="models"; os.makedirs(MODELS, exist_ok=True)
BOOK_ID="AUDUSD.m15ny.v1" if NSEED==1 else "AUDUSD.m15ny_seedens.v1"
CERT_JSON=("audusd_15m_cpcv_session_ny_multicov_result.json" if NSEED==1
           else f"audusd_15m_cpcv_session_ny_seedens{NSEED}_result.json")
COV=0.03

def mk(seed):
    return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=NUM_LEAVES,
        min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,
        n_estimators=3000,n_jobs=20,verbosity=-1,random_state=seed,bagging_seed=seed,feature_fraction_seed=seed)

def main():
    t0=time.time()
    cert=json.load(open(CERT_JSON)) if os.path.exists(CERT_JSON) else {}
    cv=cert.get("bycov",{}).get(str(COV),{}).get("verdict",{}) if cert else {}
    cvs=cert.get("bycov",{}).get(str(COV),{}).get("summary",{}) if cert else {}
    Xtr,ytr,mtr,tstr=build(SPL["train"], 6); Xva,yva,mva,tsv=build(SPL["val"])
    str_s=session_mask(tstr,SESSION); va_s=session_mask(tsv,SESSION)
    itr=mtr & str_s; iva=mva & va_s
    print(f"[freeze-ny AUDUSD] nseed={NSEED} train(NY-moved)={int(itr.sum()):,} val(NY-moved)={int(iva.sum()):,} build={time.time()-t0:.0f}s",flush=True)
    boosters=[]; pva=np.zeros(len(Xva))
    for sd in range(NSEED):
        L=mk(sd); L.fit(Xtr[itr], ytr[itr], eval_set=[(Xva[iva], yva[iva])], eval_metric="auc",
              callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        boosters.append(L); pva+=L.predict_proba(Xva)[:,1]
    pva/=NSEED; val_auc=float(roc_auc_score(yva[iva], pva[iva]))
    vi=np.where(va_s)[0]; confv=np.abs(pva[vi]-0.5); THR=float(np.quantile(confv,1-COV))
    print(f"[freeze-ny AUDUSD] VAL(NY) AUC={val_auc:.4f} cov{COV:.0%} thr={THR:.4f}",flush=True)

    fwd={}
    for w in ("test24","test25","oos"):
        Xw,yw,mw,tsw=build(SPL[w]); ws=session_mask(tsw,SESSION); wi=np.where(ws)[0]
        pr=np.mean([b.predict_proba(Xw)[:,1] for b in boosters],axis=0)[wi]
        g=side_eval(pr, yw[wi], mw[wi], tsw[wi], THR)
        fwd[w]={k:{"n":g[k]["n"],"wr":round(g[k]["wr"],4),"ci":[round(c,4) for c in g[k]["ci"]]} for k in ("COMBINED","UP","DOWN")} if g else None
        if fwd[w]: print(f"  {w}: "+" | ".join(f"{k} n{fwd[w][k]['n']} {fwd[w][k]['wr']}" for k in ("COMBINED","UP","DOWN")),flush=True)

    model_paths=[]
    for sd,L in enumerate(boosters):
        mp=f"{MODELS}/m15ny_AUDUSD_s{sd}_lgb.txt" if NSEED>1 else f"{MODELS}/m15ny_AUDUSD_direction_lgb.txt"
        L.booster_.save_model(mp); model_paths.append(mp)
    up_p10=cv.get("UP",{}).get("p10"); dn_p10=cv.get("DOWN",{}).get("p10"); cb_p10=cv.get("COMBINED",{}).get("p10")
    strat={"book":BOOK_ID,"pair":PAIR,"timeframe":"15m","horizon_min":15,"session":SESSION,"nseed":NSEED,
           "label":"sign(close[t+15]-close[t]), ties LOSE","settlement":"bar-close, gap=900s nonoverlap_chrono",
           "gate":{"cov":COV,"conf_thr":THR,"rule":"trade when |p-0.5|>=conf_thr AND ts in NY session (America/New_York 08-17, DST-correct)"},
           "ensemble":"mean of seed probabilities" if NSEED>1 else "single model",
           "breakeven":BE,"val_ny_auc":val_auc,
           "cert":f"NY per-fold-refit CPCV @cov{COV}: UP p10 {up_p10} / DOWN p10 {dn_p10} / COMB p10 {cb_p10} (15/15 paths) — {CERT_JSON}",
           "deploy_note":"REFIT-DEPENDENT: frozen-2021 vintage decays forward (.60->.57->.53 cov5, sub-BE by 2026). Deploy NY-only with PERIODIC RETRAINING; size on the refit per-era floor (~.57-.60), NOT this frozen artifact. Both sides own-pair-specific (pooling/AUDNZD/risk add nothing); Asia/LDN sub-BE.",
           "frozen_forward":fwd}
    strat_path=f"{MODELS}/m15ny_AUDUSD{'_seedens' if NSEED>1 else ''}_strategy.json"; json.dump(strat, open(strat_path,"w"), indent=2)

    man=MAN.build(book_id=BOOK_ID, timeframe="15m", side="combined", role="direction",
                  script="audusd_15m_base.py (model) + audusd_15m_cpcv_session.py ny (cert) + audusd_15m_ny_frozen.py (fwd verify) + audusd_15m_freeze_ny.py (freeze)",
                  summary=f"AUDUSD 15m NY-session own-pair base GBM{' seed-ens K='+str(NSEED) if NSEED>1 else ''}; BOTH sides certified via NY per-fold-refit CPCV (UP p10 {up_p10} / DOWN p10 {dn_p10} @cov{COV}, 15/15). Direction edge NY-concentrated + AUDUSD-own-pair-specific (cross-pair pooling/AUDNZD/risk add nothing; Asia/LDN sub-BE). REFIT-DEPENDENT (frozen-2021 fwd decays).",
                  metrics={"up_refit_cpcv_p10":up_p10,"down_refit_cpcv_p10":dn_p10,"comb_refit_cpcv_p10":cb_p10,
                           "up_refit_mean":cvs.get("UP",{}).get("mean"),"down_refit_mean":cvs.get("DOWN",{}).get("mean"),
                           "breakeven":BE,"cov":COV,"val_ny_auc":round(val_auc,4),"frozen_forward":fwd,
                           "refit_dependent":True},
                  artifacts=model_paths, hyperparams={"num_leaves":NUM_LEAVES,"lr":0.02,"reg_lambda":20,"n_est":3000,"early_stop":150,"tr_stride":6,"nseed":NSEED},
                  strategy_json=strat_path,
                  feature_fingerprint=MAN.dir_fingerprint(H.FEAT_DIR, ("AUDUSD_*.parquet",)),
                  created_utc="2026-06-09",
                  notes="NY session = America/New_York 08-17 (DST-correct, sessions.py). cov3% gate frozen on VAL NY worst-half. Two-sided via confidence selection. BOTH sides REFIT-DEPENDENT (frozen-2021 forward decays to sub-BE by 2026); deploy with periodic retrain, size on refit floor.")
    man["currency"]="AUDUSD"   # manifest.build hardcodes EURUSD; override for this key
    path=MAN.freeze(man, artifacts_src=model_paths+[strat_path])
    json.dump(man, open(f"books/{BOOK_ID}.manifest.json","w"), indent=2)
    print(f"[freeze-ny AUDUSD] FROZEN {BOOK_ID} -> {path} content_id={man['content_id']} ({time.time()-t0:.0f}s)",flush=True)

if __name__=="__main__":
    main()
