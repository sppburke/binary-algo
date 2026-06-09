"""FROZEN-PAST FORWARD holdout of the tick-microstructure lift — the trap #9 discriminator.
CPCV said bars+tk is GENUINE+leak-clean (lift survives 60s-lag, shuffle=.50). But CPCV folds are flanked by
train folds, so slow/regime tick feats (spread, intensity) can MEMORIZE era-local structure with no forward
transfer (skill trap #9). This trains on TRAIN(2012-2021) NY only, freezes the cov2% gate on VAL(2022-2023)
NY worst-half, and evaluates FORWARD on held-out 2024/2025/2026 — deriv-faithful (bar label, ties LOSE,
nonoverlap gap=900). Tick feats are asof <= t-60s (strictly causal, as the verified-clean variant).

DECIDE: if bars+tk beats bars on the BINDING (worst) forward year COMBINED win-rate -> real deployable edge
(escalate: seed-ens + freeze). If the CPCV lift decays forward (bars+tk <= bars on binding year) -> trap #9
era-memorization -> discard the CPCV lift, keep the seed-ens book as the deliverable.
Usage: ~/binary-algo-venv/bin/python usdjpy_15m_tickmicro_frozen.py [K=1]
"""
import os, sys, glob, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sessions import session_mask
from usdjpy_15m_base import build, side_eval, boot, BE, SPL
from usdjpy_15m_tickmicro import TKDIR, WINS, FEATS

SESSION="ny"; LAG_S=60; COV=0.02
K=int(sys.argv[1]) if len(sys.argv)>1 and sys.argv[1].isdigit() else 1
TKCOLS=["micro_dev","spread"]+[f"{p}_{W}" for W in WINS for p in ("imb","mom","rv","nt","flow","sprd")]

def load_tk_all():
    parts=[]
    for f in sorted(glob.glob(f"{TKDIR}/tickmicro_*.parquet")):
        parts.append(pd.read_parquet(f))
    tk=pd.concat(parts,ignore_index=True).drop_duplicates("epoch").sort_values("epoch")
    return tk["epoch"].values, tk[TKCOLS].values.astype("float32")

def tk_at(ts_arr, tke, tkv):
    pos=np.searchsorted(tke, ts_arr-LAG_S, side="right")-1
    r=np.full((len(ts_arr),len(TKCOLS)),np.nan,dtype="float32")
    ok=(pos>=0) & ((ts_arr-LAG_S-tke[np.clip(pos,0,len(tke)-1)])<=120)
    r[ok]=tkv[pos[ok]]; return r

def mk_lgb(seed=0):
    return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=127,
        min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,
        n_estimators=3000,n_jobs=16,verbosity=-1,random_state=seed,bagging_seed=seed,feature_fraction_seed=seed)

def fit_set(Xtr,ytr,itr,Xva,yva,iva):
    models=[]; pv=np.zeros(len(yva))
    for s in range(K):
        L=mk_lgb(s); L.fit(Xtr[itr],ytr[itr],eval_set=[(Xva[iva],yva[iva])],eval_metric="auc",
            callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
        models.append(L); pv+=L.predict_proba(Xva)[:,1]
    return models, pv/K

def main():
    t0=time.time()
    tke,tkv=load_tk_all(); print(f"[frozen] tick cache {len(tke):,} rows  {time.time()-t0:.0f}s",flush=True)
    # TRAIN
    Xtr,ytr,mtr,tstr=build(SPL["train"],6); nytr=session_mask(tstr,SESSION); itr=mtr&nytr
    Xtr_tk=np.concatenate([Xtr.values, tk_at(tstr.astype(float),tke,tkv)],axis=1)
    # VAL
    Xva,yva,mva,tsv=build(SPL["val"],1); nyva=session_mask(tsv,SESSION); iva=mva&nyva
    Xva_tk=np.concatenate([Xva.values, tk_at(tsv.astype(float),tke,tkv)],axis=1)
    nyva_idx=np.where(nyva)[0]
    print(f"[frozen] train NY moved={int(itr.sum()):,} val NY moved={int(iva.sum()):,} (K={K})  {time.time()-t0:.0f}s",flush=True)
    res={"key":"USDJPY.15m.ny","test":"frozen-past forward holdout (trap#9) of tick lift; tick feats asof<=t-60s",
         "K":K,"cov":COV,"feat_sets":["bars","bars+tk"],"years":{}}
    out={}
    for tag,Xtr_s,Xva_s in [("bars",Xtr.values,Xva.values),("bars+tk",Xtr_tk,Xva_tk)]:
        models,pva=fit_set(Xtr_s,ytr,itr,Xva_s,yva,iva)
        thr=float(np.quantile(np.abs(pva[nyva_idx]-0.5),1-COV))
        valauc=float(roc_auc_score(yva[iva],pva[iva]))
        yr={}
        for w in ("test24","test25","oos"):
            Xw,yw,mw,tsw=build(SPL[w],1); wi=np.where(session_mask(tsw,SESSION))[0]
            if tag=="bars": Xs=Xw.values[wi]
            else: Xs=np.concatenate([Xw.values, tk_at(tsw.astype(float),tke,tkv)],axis=1)[wi]
            pr=np.mean([m.predict_proba(Xs)[:,1] for m in models],axis=0)
            mvw=mw[wi]; auc=float(roc_auc_score(yw[wi][mvw],pr[mvw])) if mvw.sum()>20 else float("nan")
            g=side_eval(pr,yw[wi],mw[wi],tsw[wi],thr)
            yr[w]={"moved_auc":round(auc,4),**({s:{"n":g[s]["n"],"wr":round(g[s]["wr"],4),"ci":[round(c,4) for c in g[s]["ci"]]} for s in ("COMBINED","UP","DOWN")} if g else {})}
        out[tag]={"val_ny_auc":round(valauc,4),"thr":round(thr,4),"years":yr}
        res["years"][tag]=out[tag]
        print(f"[frozen] {tag}: VAL-AUC {valauc:.4f} | "+" | ".join(f"{w} COMB {yr[w].get('COMBINED',{}).get('wr')}({yr[w].get('COMBINED',{}).get('n')}) AUC {yr[w]['moved_auc']}" for w in ("test24","test25","oos")),flush=True)
    # feature importance of the bars+tk model (last fit), tick-feat share
    fi=models[-1].feature_importances_; nbar=Xtr.values.shape[1]
    tk_imp=float(fi[nbar:].sum()); tot=float(fi.sum())
    res["tk_importance_share"]=round(tk_imp/tot,4) if tot>0 else None
    res["top_tk_feats"]=sorted([(TKCOLS[i],int(fi[nbar+i])) for i in range(len(TKCOLS))],key=lambda x:-x[1])[:6]
    # verdict: does bars+tk beat bars on the BINDING (worst) forward-year COMBINED wr?
    def binding(tag):
        wrs=[res["years"][tag]["years"][w].get("COMBINED",{}).get("wr") for w in ("test24","test25","oos")]
        wrs=[x for x in wrs if x is not None]; return min(wrs) if wrs else None
    bb=binding("bars"); bt=binding("bars+tk")
    deltas={w: round((res["years"]["bars+tk"]["years"][w].get("COMBINED",{}).get("wr",0) or 0)-(res["years"]["bars"]["years"][w].get("COMBINED",{}).get("wr",0) or 0),4) for w in ("test24","test25","oos")}
    forward_genuine=bool(bt is not None and bb is not None and bt>bb and all(d>=-0.005 for d in deltas.values()))
    res["verdict"]={"bars_binding_COMB":bb,"barstk_binding_COMB":bt,"per_year_COMB_delta":deltas,
        "FORWARD_GENUINE":forward_genuine,
        "note":("FORWARD-GENUINE: bars+tk beats bars on the binding forward year AND no year regresses -> real deployable tick edge; escalate to seed-ens + freeze."
                if forward_genuine else
                "TRAP#9 / NOT-FORWARD: the CPCV tick lift does NOT transfer to the frozen-past forward holdout (binding year not improved or a year regresses) -> era-memorization, not a deployable edge. Discard; seed-ens book stays the deliverable.")}
    json.dump(res,open("usdjpy_15m_tickmicro_frozen_result.json","w"),indent=2)
    print(f"\n[frozen] tk importance share={res['tk_importance_share']} top_tk={res['top_tk_feats']}",flush=True)
    print(f"[frozen] per-year COMB delta (bars+tk - bars): {deltas}",flush=True)
    print(f"[frozen] VERDICT: {'FORWARD-GENUINE' if forward_genuine else 'TRAP#9/NOT-FORWARD'} "
          f"(bars binding {bb} -> bars+tk binding {bt}) -> usdjpy_15m_tickmicro_frozen_result.json  {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
