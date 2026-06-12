"""Freeze the CERTIFIED GBPUSD 15m XPAIR-NY K=8 seed-ensemble direction book.

SCOPE: GBPUSD · 15m · combined (both sides via confidence selection). Trains the xpair-NY 8-seed GBM
ensemble on TRAIN 2012-21 restricted to NY-session decision rows (340-feat xpbase matrix), freezes
per-cov gates on VAL 2022-23 NY worst-half, saves boosters + strategy.json, then manifest.build + freeze.

CERTIFICATION OF RECORD = per-fold-refit xpair-NY CPCV (gbpusd_15m_cpcv_xpair_ny_seedens8_result.json):
BOTH sides p10 ≥ BE 15/15 paths every cov; K=8 cov1 UP p10 .6552 / DOWN .6395; cov2 UP .6251 / DOWN .6255.
K=8 LIFTS over K=3 (unexpected — xpair 340-feat richer diversity allows continued seed-ens variance reduction):
DOWN @cov2 +.0101 / UP @cov2 +.0042 / cov1 UP +.0082 / cov1 DOWN +.0072. K=3 own-pair siblings were
SATURATED at K=3; GBPUSD xpair is NOT — larger feature space explains the continued lift.
REFIT-DEPENDENT (gbpusd_15m_xpair_frozen_seedens3_result.json: 2024 .7437 → 2026 .4679) —
deploy NY-only with periodic retraining, size on refit p10 minus tick haircut (mean Δ −.0078).

Usage: ~/binary-algo-venv/bin/python gbpusd_15m_freeze_xpny_k8.py
  -> book GBPUSD.m15ny_xpair_seedens8.v1
"""
import os, sys, json, time, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sessions import session_mask
from gbpusd_15m_xpair_frozen import build_mat, nonoverlap_chrono, side_eval
import manifest as MAN

HOR=15; GAP=900; BE=0.541; NUM_LEAVES=255; STRIDE=3; NSEED=8
MODELS="models"; os.makedirs(MODELS, exist_ok=True)
BOOK_ID="GBPUSD.m15ny_xpair_seedens8.v1"
CERT_JSON="gbpusd_15m_cpcv_xpair_ny_seedens8_result.json"
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],"test24":["2024"],"test25":["2025"],"oos":["2026"]}
COVS=(0.05,0.03,0.02,0.01)

def main():
    t0=time.time()
    allyears=[y for k in ("train","val","test24","test25","oos") for y in SPL[k]]
    X,fwd,ts,cols=build_mat(allyears,STRIDE)
    ny=session_mask(ts,"ny")
    import pandas as pd
    yr=pd.to_datetime(ts,unit="s").year.values
    moved=np.isfinite(fwd)&(fwd!=0)
    itr=np.isin(yr,[int(y) for y in SPL["train"]])&ny&moved
    iva=np.isin(yr,[int(y) for y in SPL["val"]])&ny&np.isfinite(fwd)
    print(f"[freeze xpny-k8] rows={len(ts):,} train={int(itr.sum()):,} val={int(iva.sum()):,} feats={len(cols)} build={time.time()-t0:.0f}s",flush=True)
    model_paths=[]; pva=np.zeros(int(iva.sum())); models=[]
    for sd in range(NSEED):
        L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=NUM_LEAVES,
            min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,
            n_estimators=800,n_jobs=16,verbosity=-1,random_state=sd,bagging_seed=sd,feature_fraction_seed=sd)
        L.fit(X[itr],(fwd[itr]>0).astype(int),eval_set=[(X[iva],(fwd[iva]>0).astype(int))],eval_metric="auc",
              callbacks=[lgb.early_stopping(80),lgb.log_evaluation(0)])
        pva+=L.predict_proba(X[iva])[:,1]; models.append(L)
        mp=f"{MODELS}/m15xpny_GBPUSD_k8_s{sd}.txt"; L.booster_.save_model(mp); model_paths.append(mp)
        print(f"[freeze xpny-k8] seed {sd} best_iter={L.best_iteration_} saved {mp} ({time.time()-t0:.0f}s)",flush=True)
    pva/=NSEED
    mvv=(fwd[iva]!=0)
    vauc=float(roc_auc_score((fwd[iva][mvv]>0).astype(int), pva[mvv]))
    # per-cov gates on VAL NY worst-half
    tsv=ts[iva]; fwv=fwd[iva]
    oo=np.argsort(tsv); tsv=tsv[oo]; fwv=fwv[oo]; pvas=pva[oo]
    half=len(pvas)//2; confv=np.abs(pvas-0.5); gates={}
    for cov in COVS:
        thr=float(np.quantile(confv,1-cov))
        halves=[]
        for s,e in ((0,half),(half,len(pvas))):
            r=side_eval(pvas[s:e],fwv[s:e],tsv[s:e],thr)
            halves.append(r["COMBINED"]["wr"] if r else float("nan"))
        gates[f"{cov:.2f}"]={"conf_thr":thr,"val_worst_half_wr":round(float(np.nanmin(halves)),4)}
    print(f"[freeze xpny-k8] VAL moved-AUC={vauc:.4f} gates={ {k:round(v['conf_thr'],4) for k,v in gates.items()} }",flush=True)

    cert=json.load(open(CERT_JSON))
    frozen=json.load(open("gbpusd_15m_xpair_frozen_seedens3_result.json"))
    strat={"book_id":BOOK_ID,"pair":"GBPUSD","horizon_min":HOR,"session":"ny (America/New_York 08:00-17:00, DST-correct, sessions.session_mask)",
           "feature_recipe":"gbpusd_15m_xpair.build_xp_gbp(years, stride, keep_ties=True) xp-cols + per-year float32 join of harness.feature_cols('GBPUSD') (gbpusd_15m_xpair_frozen.build_mat)",
           "n_features":len(cols),"feature_cols":cols,"nseed":NSEED,"stride_train":STRIDE,
           "predict":"p̄ = mean over 8 seed boosters' P(up); bet UP if p̄>0.5 else DOWN when |p̄-0.5| ≥ conf_thr[cov]",
           "gates":gates,"breakeven":BE,
           "deploy":{"mode":"REFIT-DEPENDENT — retrain periodically (frozen-2021 vintage decays to sub-BE by 2026); "
                            "size on refit-CPCV per-era floor minus tick haircut",
                     "refit_floor_p10_cov1":{"UP":cert["bycov"]["0.01"]["summary"]["UP"]["p10"],
                                              "DOWN":cert["bycov"]["0.01"]["summary"]["DOWN"]["p10"]},
                     "refit_floor_p10_cov2":{"UP":cert["bycov"]["0.02"]["summary"]["UP"]["p10"],
                                              "DOWN":cert["bycov"]["0.02"]["summary"]["DOWN"]["p10"]},
                     "tick_haircut":-0.0078,"kelly":"1/8 on (floor - haircut - BE)"}}
    strat_path=f"{MODELS}/m15xpny_GBPUSD_k8_strategy.json"; json.dump(strat,open(strat_path,"w"),indent=2)

    man=MAN.build(book_id=BOOK_ID, timeframe="15m", side="combined", role="direction",
        script="gbpusd_15m_freeze_xpny_k8.py",
        summary=("GBPUSD 15m XPAIR-NY K=8 seed-ens direction book: pooled USD cross-section + EURGBP-triangular/"
                 "eurobloc/risk channels + 239 own-pair feats, NY decision rows, 8-seed mean, per-cov conf gates. "
                 "BOTH sides certified via per-fold-refit CPCV K=8 (15/15 paths every cov); REFIT-DEPENDENT. "
                 "K=8 LIFTS over K=3: DOWN @cov2 +.0101 / UP @cov2 +.0042 / cov1 UP +.0082 / DOWN +.0072 p10."),
        metrics={"breakeven":BE,"val_moved_auc":round(vauc,4),
                 "refit_cpcv_p10_cov1_UP":cert["bycov"]["0.01"]["summary"]["UP"]["p10"],
                 "refit_cpcv_p10_cov1_DOWN":cert["bycov"]["0.01"]["summary"]["DOWN"]["p10"],
                 "refit_cpcv_p10_cov2_UP":cert["bycov"]["0.02"]["summary"]["UP"]["p10"],
                 "refit_cpcv_p10_cov2_DOWN":cert["bycov"]["0.02"]["summary"]["DOWN"]["p10"],
                 "refit_cpcv_mean_cov1_UP":cert["bycov"]["0.01"]["summary"]["UP"]["mean"],
                 "refit_cpcv_mean_cov1_DOWN":cert["bycov"]["0.01"]["summary"]["DOWN"]["mean"],
                 "k3_p10_cov2_UP":0.6209,"k3_p10_cov2_DOWN":0.6154,
                 "k3_p10_cov1_UP":0.6470,"k3_p10_cov1_DOWN":0.6323,
                 "delta_vs_k3_cov2_UP":round(cert["bycov"]["0.02"]["summary"]["UP"]["p10"]-0.6209,4),
                 "delta_vs_k3_cov2_DOWN":round(cert["bycov"]["0.02"]["summary"]["DOWN"]["p10"]-0.6154,4),
                 "frozen_fwd_2024_cov1_COMB":frozen["years"]["test24"]["bycov"]["0.01"]["COMBINED"]["wr"],
                 "frozen_fwd_2026_cov1_COMB":frozen["years"]["oos"]["bycov"]["0.01"]["COMBINED"]["wr"],
                 "ticksettle_mean_delta":-0.0078},
        artifacts=model_paths,
        hyperparams={"num_leaves":NUM_LEAVES,"lr":0.02,"reg_lambda":20,"n_est":800,"early_stop":80,
                     "tr_stride":STRIDE,"nseed":NSEED,"n_features":len(cols)},
        strategy_json=strat_path,
        notes=("Cert of record: gbpusd_15m_cpcv_xpair_ny_seedens8_result.json. "
               "K=8 UNEXPECTED LIFT over K=3 (unlike 3/3 own-pair siblings which saturated at K=3 — "
               "xpair 340-feat richer diversity allows continued seed-ensemble variance reduction). "
               "Adversarial: gbpusd_15m_xpair_frozen_seedens3_result.json (refit-dependent, NOT memorization); "
               "gbpusd_15m_ticksettle_result.json (bar proxy PRESERVED, mean -0.0078). "
               "K=8 unseats K=3 at cov1 and cov2 on BOTH sides. New deliverable config."))
    man["currency"]="GBPUSD"   # manifest.build hardcodes EURUSD; override for this key
    path=MAN.freeze(man, artifacts_src=model_paths+[strat_path])
    print(f"[freeze xpny-k8] FROZEN {BOOK_ID} content_id={man['content_id']} -> {path}  ({time.time()-t0:.0f}s)",flush=True)

if __name__=="__main__":
    main()
