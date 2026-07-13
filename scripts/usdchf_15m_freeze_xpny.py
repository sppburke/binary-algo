"""Freeze the CERTIFIED USDCHF 15m XPAIR-NY seed-ensemble direction book (the deployable deliverable).

SCOPE: USDCHF · 15m · combined (both sides via confidence selection). Trains the xpair-NY K-seed GBM
ensemble on TRAIN 2012-21 restricted to NY-session decision rows (337-feat xpbase matrix: cross-pair
USD-residual/lead-lag + EUR-bloc/risk channels + 239 own-pair base; features
causal/continuous), freezes per-cov gates on VAL 2022-23 NY worst-half, saves boosters + strategy.json,
then manifest.build + freeze (+ currency override).

CERTIFICATION OF RECORD = per-fold-refit xpair-NY CPCV (usdchf_15m_cpcv_xpair_ny_seedens3_result.json +
single-seed usdchf_15m_cpcv_xpair_ny_multicov_result.json): BOTH sides p10 ≥ BE 15/15 paths every cov;
K=3 cov1 UP p10 .6935 / DOWN .6682 (means .7252/.7230); single-seed cov1 UP .6997 / DOWN .662. Seed-ens
LIFTS single-seed on path-MEAN 15/15 cov×side cells + p10 13/15 (only thin cov0.01/0.005 UP p10 dip,
mean still up) → genuine variance-reduction floor lift, not redistribution. REFIT-DEPENDENT (single-seed
frozen-fwd proxy usdchf_15m_xpair_frozen_result.json: 2024 cov1 COMB .7924 → 2026 .4775) — deploy NY-only
with periodic retraining, size on the refit per-era floor minus the bar-close haircut (-.0035, no tick data).

Usage: ~/binary-algo-venv/bin/python usdchf_15m_freeze_xpny.py [nseed=3]
  -> book USDCHF.m15ny_xpair_seedens.v1 (nseed>1) / USDCHF.m15ny_xpair.v1 (nseed=1)
"""
import os, sys, json, time, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sessions import session_mask
from usdchf_15m_xpair_frozen import build_mat, nonoverlap_chrono, side_eval
import manifest as MAN

HOR=15; GAP=900; BE=0.541; NUM_LEAVES=255; STRIDE=3
NSEED=int(sys.argv[1]) if len(sys.argv)>1 and sys.argv[1].isdigit() else 3
MODELS="models"; os.makedirs(MODELS, exist_ok=True)
BOOK_ID="USDCHF.m15ny_xpair_seedens.v1" if NSEED>1 else "USDCHF.m15ny_xpair.v1"
CERT_JSON=("usdchf_15m_cpcv_xpair_ny_seedens3_result.json" if NSEED>1
           else "usdchf_15m_cpcv_xpair_ny_multicov_result.json")
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
    print(f"[freeze xpny] rows={len(ts):,} train={int(itr.sum()):,} val={int(iva.sum()):,} feats={len(cols)} build={time.time()-t0:.0f}s",flush=True)
    model_paths=[]; pva=np.zeros(int(iva.sum())); models=[]
    for sd in range(NSEED):
        L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=NUM_LEAVES,
            min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,
            n_estimators=800,n_jobs=16,verbosity=-1,random_state=sd,bagging_seed=sd,feature_fraction_seed=sd)
        L.fit(X[itr],(fwd[itr]>0).astype(int),eval_set=[(X[iva],(fwd[iva]>0).astype(int))],eval_metric="auc",
              callbacks=[lgb.early_stopping(80),lgb.log_evaluation(0)])
        pva+=L.predict_proba(X[iva])[:,1]; models.append(L)
        mp=f"{MODELS}/m15xpny_USDCHF_s{sd}.txt"; L.booster_.save_model(mp); model_paths.append(mp)
        print(f"[freeze xpny] seed {sd} best_iter={L.best_iteration_} saved {mp} ({time.time()-t0:.0f}s)",flush=True)
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
    print(f"[freeze xpny] VAL moved-AUC={vauc:.4f} gates={ {k:round(v['conf_thr'],4) for k,v in gates.items()} }",flush=True)

    cert=json.load(open(CERT_JSON))
    frozen=json.load(open("usdchf_15m_xpair_frozen_result.json"))   # single-seed adversarial frozen-forward (no seedens3 frozen run)
    strat={"book_id":BOOK_ID,"pair":"USDCHF","horizon_min":HOR,"session":"ny (America/New_York 08:00-17:00, DST-correct, sessions.session_mask)",
           "feature_recipe":"usdchf_15m_xpair.build_xp(years, stride, 'xpbase', keep_ties=True) xp-cols (EUR-bloc sign-flipped, USDCHF~-EURUSD) + per-year float32 join of harness.feature_cols('USDCHF') (usdchf_15m_xpair_frozen.build_mat)",
           "n_features":len(cols),"feature_cols":cols,"nseed":NSEED,"stride_train":STRIDE,
           "predict":"p̄ = mean over K seed boosters' P(up); bet UP if p̄>0.5 else DOWN when |p̄-0.5| ≥ conf_thr[cov]",
           "gates":gates,"breakeven":BE,
           "deploy":{"mode":"REFIT-DEPENDENT — retrain periodically (frozen-2021 vintage decays to sub-BE by 2026); "
                            "size on refit-CPCV per-era floor minus bar-close haircut",
                     "refit_floor_p10_cov1":{"UP":cert["bycov"]["0.01"]["summary"]["UP"]["p10"],
                                              "DOWN":cert["bycov"]["0.01"]["summary"]["DOWN"]["p10"]},
                     "refit_floor_p10_cov2":{"UP":cert["bycov"]["0.02"]["summary"]["UP"]["p10"],
                                              "DOWN":cert["bycov"]["0.02"]["summary"]["DOWN"]["p10"]},
                     "barclose_haircut":-0.0035,"haircut_note":"no USDCHF tick data; bar-close proxy haircut from USDJPY/AUDUSD tick validation (-.0035)","kelly":"1/8 on (floor - haircut - BE)"}}
    strat_path=f"{MODELS}/m15xpny_USDCHF_strategy.json"; json.dump(strat,open(strat_path,"w"),indent=2)

    man=MAN.build(book_id=BOOK_ID, timeframe="15m", side="combined", role="direction",
        script="usdchf_15m_freeze_xpny.py",
        summary=("USDCHF 15m XPAIR-NY seed-ens direction book: pooled USD cross-section + EURGBP-triangular/"
                 "eurobloc/risk channels + 239 own-pair feats, NY decision rows, K-seed mean, per-cov conf gates. "
                 "BOTH sides certified via per-fold-refit CPCV (15/15 paths every cov); REFIT-DEPENDENT."),
        metrics={"breakeven":BE,"val_moved_auc":round(vauc,4),
                 "refit_cpcv_p10_cov1_UP":cert["bycov"]["0.01"]["summary"]["UP"]["p10"],
                 "refit_cpcv_p10_cov1_DOWN":cert["bycov"]["0.01"]["summary"]["DOWN"]["p10"],
                 "refit_cpcv_p10_cov2_UP":cert["bycov"]["0.02"]["summary"]["UP"]["p10"],
                 "refit_cpcv_p10_cov2_DOWN":cert["bycov"]["0.02"]["summary"]["DOWN"]["p10"],
                 "refit_cpcv_mean_cov1_UP":cert["bycov"]["0.01"]["summary"]["UP"]["mean"],
                 "refit_cpcv_mean_cov1_DOWN":cert["bycov"]["0.01"]["summary"]["DOWN"]["mean"],
                 "singleseed_cov1_UP_p10":0.6997,"singleseed_cov1_DOWN_p10":0.662,
                 "frozen_fwd_2024_cov1_COMB":frozen["years"]["test24"]["bycov"]["0.01"]["COMBINED"]["wr"],
                 "frozen_fwd_2026_cov1_COMB":frozen["years"]["oos"]["bycov"]["0.01"]["COMBINED"]["wr"],
                 "barclose_haircut":-0.0035},
        artifacts=model_paths,
        hyperparams={"num_leaves":NUM_LEAVES,"lr":0.02,"reg_lambda":20,"n_est":800,"early_stop":80,
                     "tr_stride":STRIDE,"nseed":NSEED,"n_features":len(cols)},
        strategy_json=strat_path,
        notes=("Cert of record: "+CERT_JSON+" (+ single-seed multicov usdchf_15m_cpcv_xpair_ny_multicov_result.json: "
               "cov1 UP p10 .6997 / DOWN .662 — FIRST major to certify >65% BOTH sides). Adversarial: "
               "usdchf_15m_xpair_frozen_result.json (refit-dependent, NOT memorization; xpair-frozen BEATS "
               "own-pair-frozen all 6 cells = pooling genuine) + usdchf_15m_ownpair_frozen_result.json. No USDCHF "
               "tick data; bar-close proxy haircut -.0035 (USDJPY/AUDUSD-validated). EUR-bloc pooling case "
               "(USDCHF~-EURUSD, SNB-managed); NY carrier + all-session breadth. p10 binds on recent era."))
    man["currency"]="USDCHF"   # manifest.build hardcodes EURUSD; override for this key
    path=MAN.freeze(man, artifacts_src=model_paths+[strat_path])
    print(f"[freeze xpny] FROZEN {BOOK_ID} content_id={man['content_id']} -> {path}  ({time.time()-t0:.0f}s)",flush=True)

if __name__=="__main__":
    main()
