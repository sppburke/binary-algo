"""Freeze a CERTIFIED USDCHF 15m session-restricted own-pair direction book (the deployable deliverable).

SCOPE: USDCHF · 15m · combined (both sides via confidence selection). Trains the own-pair base GBM (optionally
a K-seed ensemble) on TRAIN 2012-21 restricted to SESSION decision rows (features causal/continuous), freezes
the cov gate on VAL 2022-23 session worst-half, records the frozen-forward per-year side-split, saves the
booster(s) + strategy.json, then manifest.build + freeze + writes the book manifest.

CERTIFICATION OF RECORD = per-fold-refit session-CPCV (usdchf_15m_cpcv_session_<sess>_*_result.json): both
sides p10 ≥ BE, ≥80% of 15 paths, at the operating cov. REFIT-DEPENDENT (the program-wide 15m signature) —
the frozen-2021 vintage decays forward; deploy session-only with periodic retraining and size on the refit
per-era floor, NOT the frozen book.

Usage: ~/binary-algo-venv/bin/python usdchf_15m_freeze.py [nseed=1] [session=ny] [cov=0.03]
  nseed=1 -> book USDCHF.m15<sess>.v1 ; nseed>1 -> book USDCHF.m15<sess>_seedens.v1
"""
import os, sys, json, time, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from sessions import session_mask
from usdchf_15m_base import build, side_eval, BE, SPL
import manifest as MAN

PAIR="USDCHF"; HOR=15; GAP=900; NUM_LEAVES=127
NSEED=int(sys.argv[1]) if len(sys.argv)>1 and sys.argv[1].isdigit() else 1
SESSION=sys.argv[2] if len(sys.argv)>2 and sys.argv[2] in ("ny","ldn","asia","all") else "ny"
COV=float(sys.argv[3]) if len(sys.argv)>3 else 0.03
MODELS="models"; os.makedirs(MODELS, exist_ok=True)
_sfx=SESSION if SESSION!="ny" else "ny"
BOOK_ID=f"USDCHF.m15{_sfx}.v1" if NSEED==1 else f"USDCHF.m15{_sfx}_seedens.v1"
CERT_JSON=(f"usdchf_15m_cpcv_session_{SESSION}_multicov_result.json" if NSEED==1
           else f"usdchf_15m_cpcv_session_{SESSION}_seedens{NSEED}_result.json")

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
    print(f"[freeze {SESSION} USDCHF] nseed={NSEED} cov{COV} train(moved)={int(itr.sum()):,} val(moved)={int(iva.sum()):,} build={time.time()-t0:.0f}s",flush=True)
    boosters=[]; pva=np.zeros(len(Xva))
    for sd in range(NSEED):
        L=mk(sd); L.fit(Xtr[itr], ytr[itr], eval_set=[(Xva[iva], yva[iva])], eval_metric="auc",
              callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        boosters.append(L); pva+=L.predict_proba(Xva)[:,1]
    pva/=NSEED; val_auc=float(roc_auc_score(yva[iva], pva[iva]))
    vi=np.where(va_s)[0]; confv=np.abs(pva[vi]-0.5); THR=float(np.quantile(confv,1-COV))
    print(f"[freeze {SESSION} USDCHF] VAL AUC={val_auc:.4f} cov{COV:.0%} thr={THR:.4f}",flush=True)

    fwd={}
    for w in ("test24","test25","oos"):
        Xw,yw,mw,tsw=build(SPL[w]); ws=session_mask(tsw,SESSION); wi=np.where(ws)[0]
        pr=np.mean([b.predict_proba(Xw)[:,1] for b in boosters],axis=0)[wi]
        g=side_eval(pr, yw[wi], mw[wi], tsw[wi], THR)
        fwd[w]={k:{"n":g[k]["n"],"wr":round(g[k]["wr"],4),"ci":[round(c,4) for c in g[k]["ci"]]} for k in ("COMBINED","UP","DOWN")} if g else None
        if fwd[w]: print(f"  {w}: "+" | ".join(f"{k} n{fwd[w][k]['n']} {fwd[w][k]['wr']}" for k in ("COMBINED","UP","DOWN")),flush=True)

    model_paths=[]
    for sd,L in enumerate(boosters):
        mp=f"{MODELS}/m15{_sfx}_USDCHF_s{sd}_lgb.txt" if NSEED>1 else f"{MODELS}/m15{_sfx}_USDCHF_direction_lgb.txt"
        L.booster_.save_model(mp); model_paths.append(mp)
    up_p10=cv.get("UP",{}).get("p10"); dn_p10=cv.get("DOWN",{}).get("p10"); cb_p10=cv.get("COMBINED",{}).get("p10")
    strat={"book":BOOK_ID,"pair":PAIR,"timeframe":"15m","horizon_min":15,"session":SESSION,"nseed":NSEED,
           "label":"sign(close[t+15]-close[t]), ties LOSE","settlement":"bar-close, gap=900s nonoverlap_chrono",
           "gate":{"cov":COV,"conf_thr":THR,"rule":f"trade when |p-0.5|>=conf_thr AND ts in {SESSION} session"},
           "ensemble":"mean of seed probabilities" if NSEED>1 else "single model",
           "breakeven":BE,"val_auc":val_auc,
           "cert":f"{SESSION} per-fold-refit CPCV @cov{COV}: UP p10 {up_p10} / DOWN p10 {dn_p10} / COMB p10 {cb_p10} — {CERT_JSON}",
           "deploy_note":"REFIT-DEPENDENT (program-wide 15m signature): frozen-2021 vintage decays forward. Deploy session-only with PERIODIC RETRAINING; size on the refit per-era floor, NOT this frozen artifact.",
           "frozen_forward":fwd}
    strat_path=f"{MODELS}/m15{_sfx}_USDCHF{'_seedens' if NSEED>1 else ''}_strategy.json"; json.dump(strat, open(strat_path,"w"), indent=2)

    man=MAN.build(book_id=BOOK_ID, timeframe="15m", side="combined", role="direction",
                  script=f"usdchf_15m_base.py (model) + usdchf_15m_cpcv_session.py {SESSION} (cert) + usdchf_15m_freeze.py (freeze)",
                  summary=f"USDCHF 15m {SESSION}-session own-pair base GBM{' seed-ens K='+str(NSEED) if NSEED>1 else ''}; BOTH sides certified via {SESSION} per-fold-refit CPCV (UP p10 {up_p10} / DOWN p10 {dn_p10} @cov{COV}). USDCHF = European safe-haven (USD-base/CHF-quote); HYBRID edge = broad (all-session certifies, EUR-bloc breadth) + NY-concentrated carrier (NY AUC .540 >> all-session). EUR-bloc cross-pair pooling does NOT add (own-pair-specific, contemporaneous coupling). REFIT-DEPENDENT.",
                  metrics={"up_refit_cpcv_p10":up_p10,"down_refit_cpcv_p10":dn_p10,"comb_refit_cpcv_p10":cb_p10,
                           "up_refit_mean":cvs.get("UP",{}).get("mean"),"down_refit_mean":cvs.get("DOWN",{}).get("mean"),
                           "breakeven":BE,"cov":COV,"session":SESSION,"val_auc":round(val_auc,4),"frozen_forward":fwd,
                           "refit_dependent":True},
                  artifacts=model_paths, hyperparams={"num_leaves":NUM_LEAVES,"lr":0.02,"reg_lambda":20,"n_est":3000,"early_stop":150,"tr_stride":6,"nseed":NSEED},
                  strategy_json=strat_path,
                  feature_fingerprint=MAN.dir_fingerprint(H.FEAT_DIR, ("USDCHF_*.parquet",)),
                  created_utc="2026-06-10",
                  notes=f"{SESSION} session (sessions.py, DST-correct). cov{COV} gate frozen on VAL {SESSION} worst-half. Two-sided via confidence selection. REFIT-DEPENDENT; deploy with periodic retrain, size on refit floor.")
    man["currency"]="USDCHF"   # manifest.build hardcodes EURUSD; override for this key
    path=MAN.freeze(man, artifacts_src=model_paths+[strat_path])
    json.dump(man, open(f"books/{BOOK_ID}.manifest.json","w"), indent=2)
    print(f"[freeze {SESSION} USDCHF] FROZEN {BOOK_ID} -> {path} content_id={man['content_id']} ({time.time()-t0:.0f}s)",flush=True)

if __name__=="__main__":
    main()
