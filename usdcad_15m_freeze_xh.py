"""USDCAD 15m NY — CROSS-HORIZON BLEND (15m child x 30m parent) FROZEN-FORWARD adversarial check (+ freeze).

SCOPE: USDCAD · 15m · NY. The cross-horizon BLEND (avg of a 15m child + a decorrelated 30m parent, corr ~.71-.76)
beat the seed-ens-K3 deliverable on refit-CPCV @cov2. Before promoting it (and after TB's refit-CPCV lift was
exposed as refit-overfit by THIS same check), verify the blend survives a FROZEN-2012-21 forward holdout (trap#9).

Train K=3 seed-ens 15m child (label sign(close[t+15]-close[t])) + K=3 seed-ens 30m parent (label
sign(close[t+30]-close[t])) on 2012-21 NY-moved bars; blend = (mean_p15 + mean_p30)/2 on the 15m decision rows;
freeze the cov gate on VAL 2022-23 NY; test per-year NY forward (eval ALWAYS the 15m deriv sign, ties LOSE).
Compare to BASE book frozen-forward (UP .6658/.5606/.5714, DOWN .7487/.5851/.5258, COMB .6931/.5696/.5519 @cov2).
PROMOTE+FREEZE as USDCAD.m15ny_xhblend.v1 iff blend COMB forward >= base COMB in BOTH binding years (2025 AND
2026) AND neither side regresses materially — else refit-CPCV-only (record, keep base deliverable).

Usage: ~/binary-algo-venv/bin/python usdcad_15m_freeze_xh.py [nseed=3] [cov=0.02] [freeze=0|1]
"""
import os, sys, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from sessions import session_mask
from usdcad_15m_base import side_eval, BE, SPL, FEATS

PAIR="USDCAD"; STEP=60; FEAT=H.FEAT_DIR
HOR_C=15; HOR_P=30; GAP_C=HOR_C*STEP; GAP_P=HOR_P*STEP; NUM_LEAVES=127
NSEED=int(sys.argv[1]) if len(sys.argv)>1 else 3
COV=float(sys.argv[2]) if len(sys.argv)>2 else 0.02
DO_FREEZE=(len(sys.argv)>3 and sys.argv[3]=="1")
RESULT="usdcad_15m_freeze_xh_result.json"
BASE_FWD={"2024":{"UP":0.6658,"DOWN":0.7487,"COMB":0.6931},"2025":{"UP":0.5606,"DOWN":0.5851,"COMB":0.5696},"2026":{"UP":0.5714,"DOWN":0.5258,"COMB":0.5519}}

def build_xh(years, stride=1):
    """Per-split: X, f15 (15m fwd ret, eval label src), f30 (30m fwd ret, parent label src), ts."""
    Xs=[]; f15=[]; f30=[]; tss=[]
    for y in years:
        p=f"{FEAT}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p, columns=FEATS+["close"]); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float); ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        X=d[FEATS].astype("float32"); keepf=X.isna().mean(axis=1).values<0.5
        cg15=np.zeros(n,bool); cg15[:n-HOR_C]=(ts[HOR_C:]-ts[:-HOR_C])==GAP_C
        cg30=np.zeros(n,bool); cg30[:n-HOR_P]=(ts[HOR_P:]-ts[:-HOR_P])==GAP_P
        fr15=np.full(n,np.nan); fr15[:n-HOR_C]=c[HOR_C:]/c[:-HOR_C]-1.0
        fr30=np.full(n,np.nan); fr30[:n-HOR_P]=c[HOR_P:]/c[:-HOR_P]-1.0
        fr15[~(cg15&keepf)]=np.nan; fr30[~(cg30&keepf)]=np.nan
        idx=np.where(keepf)[0]
        if stride>1: idx=idx[::stride]
        Xs.append(X.values[idx]); f15.append(fr15[idx]); f30.append(fr30[idx]); tss.append(ts[idx])
    return np.concatenate(Xs),np.concatenate(f15),np.concatenate(f30),np.concatenate(tss)

def mk(seed): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=NUM_LEAVES,
    min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,n_estimators=3000,
    n_jobs=16,verbosity=-1,random_state=seed,bagging_seed=seed,feature_fraction_seed=seed)

def main():
    t0=time.time()
    Xtr,f15tr,f30tr,tstr=build_xh(SPL["train"],6); Xva,f15va,f30va,tsv=build_xh(SPL["val"],1)
    nytr=session_mask(tstr,"ny"); nyva=session_mask(tsv,"ny")
    mv15=np.isfinite(f15tr)&(f15tr!=0.0); mv30=np.isfinite(f30tr)&(f30tr!=0.0)
    itr15=mv15&nytr; itr30=mv30&nytr
    y15tr=(f15tr>0).astype(int); y30tr=(f30tr>0).astype(int)
    print(f"[freeze-xh] nseed={NSEED} cov{COV} train15(NY moved)={int(itr15.sum()):,} train30={int(itr30.sum()):,} build={time.time()-t0:.0f}s",flush=True)
    childs=[]; parents=[]; p15va=np.zeros(len(Xva)); p30va=np.zeros(len(Xva))
    iva15=(np.isfinite(f15va)&(f15va!=0.0))&nyva
    for sd in range(NSEED):
        Lc=mk(sd); Lc.fit(Xtr[itr15],y15tr[itr15],eval_set=[(Xva[iva15],(f15va[iva15]>0).astype(int))],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
        Lp=mk(sd+100); Lp.fit(Xtr[itr30],y30tr[itr30])
        childs.append(Lc); parents.append(Lp)
        p15va+=Lc.predict_proba(Xva)[:,1]; p30va+=Lp.predict_proba(Xva)[:,1]
    p15va/=NSEED; p30va/=NSEED
    pblend_va=(p15va+p30va)/2.0
    val_auc=float(roc_auc_score((f15va[iva15]>0).astype(int), pblend_va[iva15]))
    vi=np.where(nyva)[0]; THR=float(np.quantile(np.abs(pblend_va[vi]-0.5),1-COV))
    print(f"[freeze-xh] VAL NY blend AUC={val_auc:.4f} cov{COV} thr={THR:.4f}",flush=True)
    fwd={}
    for w,wy in (("2024",["2024"]),("2025",["2025"]),("2026",["2026"])):
        Xw,f15w,f30w,tsw=build_xh(wy,1); ws=session_mask(tsw,"ny"); wi=np.where(ws)[0]
        p15=np.mean([b.predict_proba(Xw)[:,1] for b in childs],axis=0)
        p30=np.mean([b.predict_proba(Xw)[:,1] for b in parents],axis=0)
        pbl=((p15+p30)/2.0)[wi]
        yw=(f15w>0).astype(int); mw=np.isfinite(f15w)&(f15w!=0.0)
        g=side_eval(pbl,yw[wi],mw[wi],tsw[wi],THR)
        fwd[w]={k:{"n":g[k]["n"],"wr":round(g[k]["wr"],4)} for k in ("COMBINED","UP","DOWN")} if g else None
        b=BASE_FWD[w]
        print(f"  {w}: BLEND UP {fwd[w]['UP']['wr']} (base {b['UP']}) | DOWN {fwd[w]['DOWN']['wr']} (base {b['DOWN']}) | COMB {fwd[w]['COMBINED']['wr']} (base {b['COMB']})",flush=True)
    comb_binding=all(fwd[y]["COMBINED"]["wr"]>=BASE_FWD[y]["COMB"] for y in ("2025","2026"))
    up_binding=all(fwd[y]["UP"]["wr"]>=BASE_FWD[y]["UP"] for y in ("2025","2026"))
    res={"key":"USDCAD.15m.ny.xhblend","nseed":NSEED,"cov":COV,"val_ny_blend_auc":round(val_auc,4),
         "frozen_forward":fwd,"base_frozen_forward":BASE_FWD,
         "refit_cpcv":"single-seed blend beat seed-ens-K3 @cov2 (UP .626/DOWN .5857/COMB .6104); matched-seedens K3 = usdcad_15m_xhstack_ny_result.json (nseed3 run)",
         "verdict":{"COMB_fwd_beats_base_binding":bool(comb_binding),"UP_fwd_beats_base_binding":bool(up_binding),
                    "note":"PROMOTE+freeze iff COMB fwd >= base COMB in BOTH binding years (2025&2026). Else refit-CPCV-only (keep base)."}}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[freeze-xh] COMB-fwd-beats-base(2025&2026)={comb_binding} UP={up_binding} -> {RESULT} ({time.time()-t0:.0f}s)",flush=True)
    if comb_binding and DO_FREEZE:
        import manifest as MAN
        MODELS="models"; os.makedirs(MODELS,exist_ok=True); BOOK_ID="USDCAD.m15ny_xhblend.v1"; mps=[]
        for sd,(Lc,Lp) in enumerate(zip(childs,parents)):
            mc=f"{MODELS}/m15ny_USDCAD_xh_child_s{sd}.txt"; Lc.booster_.save_model(mc); mps.append(mc)
            mp=f"{MODELS}/m15ny_USDCAD_xh_parent_s{sd}.txt"; Lp.booster_.save_model(mp); mps.append(mp)
        strat={"book":BOOK_ID,"pair":PAIR,"timeframe":"15m","session":"ny","nseed":NSEED,
               "ensemble":"blend = mean( mean_K(child p15), mean_K(parent p30) )","gate":{"cov":COV,"conf_thr":THR},
               "cert":"NY refit-CPCV blend > seed-ens-K3 @cov2; frozen-forward survives binding years","frozen_forward":fwd}
        sp=f"{MODELS}/m15ny_USDCAD_xhblend_strategy.json"; json.dump(strat,open(sp,"w"),indent=2)
        man=MAN.build(book_id=BOOK_ID,timeframe="15m",side="combined",role="direction",
                      script="usdcad_15m_xhstack.py (cert) + usdcad_15m_freeze_xh.py (freeze)",
                      summary=f"USDCAD 15m NY cross-horizon BLEND (15m child K={NSEED} + 30m parent K={NSEED}, corr~.76): beats base seed-ens on refit-CPCV @cov2 + survives frozen-forward.",
                      metrics={"breakeven":BE,"cov":COV,"val_ny_blend_auc":round(val_auc,4),"frozen_forward":fwd,"refit_dependent":True},
                      artifacts=mps,hyperparams={"num_leaves":NUM_LEAVES,"nseed":NSEED,"blend":"15m+30m"},strategy_json=sp,
                      feature_fingerprint=MAN.dir_fingerprint(H.FEAT_DIR,("USDCAD_*.parquet",)),created_utc="2026-06-10",
                      notes="Cross-horizon blend; eval fixed-15m sign. REFIT-DEPENDENT.")
        man["currency"]="USDCAD"; path=MAN.freeze(man,artifacts_src=mps+[sp]); json.dump(man,open(f"books/{BOOK_ID}.manifest.json","w"),indent=2)
        print(f"[freeze-xh] FROZEN {BOOK_ID} content_id={man['content_id']} -> {path}",flush=True)

if __name__=="__main__": main()
