"""Freeze the CERTIFIED USDJPY 15m NY-session own-pair direction book (the deployable deliverable).

SCOPE: USDJPY · 15m · combined (both sides via confidence selection). Trains the own-pair base GBM on
TRAIN 2012-21, restricted to NY-session decision rows (features causal/continuous), freezes the cov3% gate
on VAL 2022-23 worst-half, records the frozen-forward per-year side-split (2024/25/26), saves the booster +
strategy.json, then manifest.build + freeze + writes the book manifest. Book id: USDJPY.m15ny.v1.

The CERTIFICATION OF RECORD is the per-fold-refit NY-CPCV (UP p10 .586 / DOWN p10 .572, 15/15 paths).
This frozen single-model is the deployable artifact; UP is robust frozen-forward (~.58 all years), DOWN is
refit-dependent (frozen 2026 thin/weak; size on the refit floor, retrain periodically).

Usage: ~/binary-algo-venv/bin/python usdjpy_15m_freeze_ny.py
"""
import os, json, time, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from sessions import session_mask
from usdjpy_15m_base import build, side_eval, covcurve, mk_lgb, BE, SPL
import manifest as MAN

PAIR="USDJPY"; HOR=15; GAP=900; SESSION="ny"; NUM_LEAVES=127
MODELS=os.path.join(H.FEAT_DIR.rsplit("/",1)[0], "models") if False else "models"
os.makedirs(MODELS, exist_ok=True)
BOOK_ID="USDJPY.m15ny.v1"

def main():
    t0=time.time()
    Xtr,ytr,mtr,tstr=build(SPL["train"], 6)
    Xva,yva,mva,tsv=build(SPL["val"])
    str_s=session_mask(tstr,SESSION); va_s=session_mask(tsv,SESSION)
    itr=mtr & str_s; iva=mva & va_s
    print(f"[freeze-ny] train(NY-moved)={int(itr.sum()):,} val(NY-moved)={int(iva.sum()):,} build={time.time()-t0:.0f}s",flush=True)
    L=mk_lgb(num_leaves=NUM_LEAVES)
    L.fit(Xtr[itr], ytr[itr], eval_set=[(Xva[iva], yva[iva])], eval_metric="auc",
          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    pva=L.predict_proba(Xva)[:,1]; val_auc=float(roc_auc_score(yva[iva], pva[iva]))
    # cov3% gate frozen on VAL NY worst-half (match cert operating point)
    vi=np.where(va_s)[0]; pvi=pva[vi]; confv=np.abs(pvi-0.5)
    COV=0.03; THR=float(np.quantile(confv,1-COV))
    print(f"[freeze-ny] best_iter={L.best_iteration_} VAL(NY) AUC={val_auc:.4f} cov{COV:.0%} thr={THR:.4f}",flush=True)

    # frozen-forward per-year side-split (NY)
    fwd={}
    for w in ("test24","test25","oos"):
        Xw,yw,mw,tsw=build(SPL[w]); ws=session_mask(tsw,SESSION); wi=np.where(ws)[0]
        pr=L.predict_proba(Xw)[:,1][wi]
        g=side_eval(pr, yw[wi], mw[wi], tsw[wi], THR)
        fwd[w]={k:{"n":g[k]["n"],"wr":round(g[k]["wr"],4),"ci":[round(c,4) for c in g[k]["ci"]]} for k in ("COMBINED","UP","DOWN")} if g else None
        print(f"  {w}: "+" | ".join(f"{k} n{fwd[w][k]['n']} {fwd[w][k]['wr']}" for k in ("COMBINED","UP","DOWN")),flush=True)

    model_path=f"{MODELS}/m15ny_USDJPY_direction_lgb.txt"; L.booster_.save_model(model_path)
    strat={"book":BOOK_ID,"pair":PAIR,"timeframe":"15m","horizon_min":15,"session":SESSION,
           "label":"sign(close[t+15]-close[t]), ties LOSE","settlement":"bar-close, gap=900s nonoverlap_chrono",
           "gate":{"cov":COV,"conf_thr":THR,"rule":"trade when |p-0.5|>=conf_thr AND ts in NY session"},
           "breakeven":BE,"val_auc":val_auc,"best_iter":int(L.best_iteration_ or 0),
           "cert":"NY per-fold-refit CPCV: UP p10 .586 / DOWN p10 .572 / COMB p10 .580 (15/15 paths) — usdjpy_15m_cpcv_session_ny_cov0.03_result.json",
           "deploy_note":"UP robust frozen-forward (~.58 all yrs); DOWN refit-dependent (size on refit floor, retrain periodically). Refit-CPCV is the certified floor.",
           "frozen_forward":fwd}
    strat_path=f"{MODELS}/m15ny_USDJPY_strategy.json"; json.dump(strat, open(strat_path,"w"), indent=2)

    man=MAN.build(book_id=BOOK_ID, timeframe="15m", side="combined", role="direction",
                  script="usdjpy_15m_session.py (model) + usdjpy_15m_cpcv_session.py ny (cert) + usdjpy_15m_freeze_ny.py (freeze)",
                  summary="USDJPY 15m NY-session own-pair base GBM; BOTH sides certified via NY per-fold-refit CPCV (UP p10 .586 / DOWN p10 .572, 15/15). Direction edge is NY-concentrated + USDJPY-specific (pooling dilutes).",
                  metrics={"up_refit_cpcv_p10":0.586,"down_refit_cpcv_p10":0.572,"comb_refit_cpcv_p10":0.580,
                           "up_refit_mean":0.602,"down_refit_mean":0.597,"breakeven":BE,"cov":COV,
                           "val_ny_auc":round(val_auc,4),"frozen_forward":fwd},
                  artifacts=[model_path], hyperparams={"num_leaves":NUM_LEAVES,"lr":0.02,"reg_lambda":20,"n_est":3000,"early_stop":150,"tr_stride":6},
                  strategy_json=strat_path,
                  feature_fingerprint=MAN.dir_fingerprint(H.FEAT_DIR, ("USDJPY_*.parquet",)),
                  created_utc="2026-06-08",
                  notes="NY session = America/New_York 08-17 (DST-correct, sessions.py). cov3% gate frozen on VAL NY worst-half. "
                        "Two-sided via confidence selection. DOWN refit-dependent.")
    man["currency"]="USDJPY"   # manifest.build hardcodes EURUSD; override for this key
    path=MAN.freeze(man, artifacts_src=[model_path, strat_path])
    json.dump(man, open(f"books/{BOOK_ID}.manifest.json","w"), indent=2)
    print(f"[freeze-ny] FROZEN {BOOK_ID} -> {path}  ({time.time()-t0:.0f}s)",flush=True)
    print(f"[freeze-ny] content_id={man['content_id']}",flush=True)

if __name__=="__main__":
    main()
