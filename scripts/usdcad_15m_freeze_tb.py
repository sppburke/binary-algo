"""USDCAD 15m NY — TB-label seed-ens K=3 FROZEN-FORWARD adversarial check (+ freeze if it survives).

SCOPE: USDCAD · 15m · NY. The TB-label (triple-barrier first-touch, k=2.0) + seed-ens K=3 gave a REAL UP-side
refit-CPCV lift over the base seed-ens deliverable (UP p10 .6135@cov2 vs .5968, consistent all covs, mean
confirmed; DOWN-neutral; AUC flat = operating-point gain). Before promoting it to the (USDCAD,15m,UP) leader,
ADVERSARIALLY VERIFY (trap#9): does the UP lift survive a FROZEN-2012-21 forward holdout, or is it refit-only?

Train K=3 seed-ens on 2012-21 NY-moved bars with the TB first-touch TRAIN label (eval ALWAYS the deriv-faithful
fixed-15m sign, ties LOSE); freeze the cov2 gate on VAL 2022-23 NY worst-cov; test per-year NY forward
(2024/25/26) UP/DOWN/COMB. Compare to the BASE book frozen-forward (UP .6658/.5606/.5714, DOWN .7487/.5851/
.5258 @cov2). PROMOTE+FREEZE as USDCAD.m15ny_tbseedens.v1 iff TB UP forward >= base UP forward in the binding
years (2025 AND 2026) — else TB is a refit-CPCV-only lift (record, do NOT replace the deliverable).

Usage: ~/binary-algo-venv/bin/python usdcad_15m_freeze_tb.py [k=2.0] [nseed=3] [cov=0.02] [freeze=0|1]
"""
import os, sys, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from sessions import session_mask
from usdcad_15m_base import side_eval, BE, SPL, FEATS

PAIR="USDCAD"; HOR=15; STEP=60; GAP=HOR*STEP; FEAT=H.FEAT_DIR
VOL_WIN=900; VOL_MINP=60; NUM_LEAVES=127
TB_K=float(sys.argv[1]) if len(sys.argv)>1 else 2.0
NSEED=int(sys.argv[2]) if len(sys.argv)>2 else 3
COV=float(sys.argv[3]) if len(sys.argv)>3 else 0.02
DO_FREEZE=(len(sys.argv)>4 and sys.argv[4]=="1")
RESULT="usdcad_15m_freeze_tb_result.json"
BASE_FWD={"2024":{"UP":0.6658,"DOWN":0.7487,"COMB":0.6931},"2025":{"UP":0.5606,"DOWN":0.5851,"COMB":0.5696},"2026":{"UP":0.5714,"DOWN":0.5258,"COMB":0.5519}}

def _first_touch_label(c, fr, contig, k):
    n=len(c); r1=np.full(n,np.nan); r1[1:]=c[1:]/c[:-1]-1.0
    sig=pd.Series(r1).rolling(VOL_WIN, min_periods=VOL_MINP).std().values
    cpad=np.concatenate([c, np.full(HOR,np.nan)])
    win=np.lib.stride_tricks.sliding_window_view(cpad, HOR); fwd=win[1:n+1]; c0=c[:,None]
    cummax=np.maximum.accumulate(fwd,axis=1); cummin=np.minimum.accumulate(fwd,axis=1)
    end_sign=(fr>0).astype("float64"); up=c0*(1.0+k*sig[:,None]); dn=c0*(1.0-k*sig[:,None])
    up_hit=cummax>=up; dn_hit=cummin<=dn
    tu=np.where(up_hit.any(axis=1), up_hit.argmax(axis=1), HOR); td=np.where(dn_hit.any(axis=1), dn_hit.argmax(axis=1), HOR)
    ylab=end_sign.copy(); ylab[tu<td]=1.0; ylab[td<tu]=0.0
    bad=(~contig)|(~np.isfinite(sig))|(sig<=0)|(~np.isfinite(fr)); ylab=ylab.astype("float64"); ylab[bad]=np.nan
    return ylab

def build_tb(years, stride=1):
    Xs=[]; ys=[]; mv=[]; tss=[]; tbs=[]
    for y in years:
        p=f"{FEAT}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p, columns=FEATS+["close"]); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float); ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        contig=np.zeros(n,bool); contig[:n-HOR]=(ts[HOR:]-ts[:-HOR])==GAP
        fr=np.full(n,np.nan); fr[:n-HOR]=c[HOR:]/c[:-HOR]-1.0
        X=d[FEATS].astype("float32"); keepf=X.isna().mean(axis=1).values<0.5
        valid=contig & np.isfinite(fr) & keepf; moved=valid & (fr!=0.0)
        tb=_first_touch_label(c, fr, contig, TB_K)
        idx=np.where(valid)[0]
        if stride>1: idx=idx[::stride]
        Xs.append(X.values[idx]); ys.append((fr[idx]>0).astype(int)); mv.append(moved[idx]); tss.append(ts[idx]); tbs.append(tb[idx])
    return np.concatenate(Xs),np.concatenate(ys),np.concatenate(mv),np.concatenate(tss),np.concatenate(tbs)

def mk(seed): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=NUM_LEAVES,
    min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,n_estimators=3000,
    n_jobs=16,verbosity=-1,random_state=seed,bagging_seed=seed,feature_fraction_seed=seed)

def main():
    t0=time.time()
    Xtr,ytr,mtr,tstr,tbtr=build_tb(SPL["train"],6); Xva,yva,mva,tsv,_=build_tb(SPL["val"],1)
    nytr=session_mask(tstr,"ny"); nyva=session_mask(tsv,"ny")
    # TRAIN target = TB first-touch where defined, else fixed sign; restricted to NY moved bars
    itr=mtr&nytr; ytb=np.where(np.isfinite(tbtr), tbtr, ytr).astype(int)
    iva=mva&nyva
    print(f"[freeze-tb] k={TB_K} nseed={NSEED} cov{COV} train(NY moved)={int(itr.sum()):,} val(NY moved)={int(iva.sum()):,} build={time.time()-t0:.0f}s",flush=True)
    boosters=[]; pva=np.zeros(len(Xva))
    for sd in range(NSEED):
        L=mk(sd); L.fit(Xtr[itr],ytb[itr],eval_set=[(Xva[iva],yva[iva])],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
        boosters.append(L); pva+=L.predict_proba(Xva)[:,1]
    pva/=NSEED; val_auc=float(roc_auc_score(yva[iva],pva[iva]))
    vi=np.where(nyva)[0]; THR=float(np.quantile(np.abs(pva[vi]-0.5),1-COV))
    print(f"[freeze-tb] VAL NY AUC={val_auc:.4f} cov{COV} thr={THR:.4f}",flush=True)
    fwd={}
    for w,wy in (("2024",["2024"]),("2025",["2025"]),("2026",["2026"])):
        Xw,yw,mw,tsw,_=build_tb(wy,1); ws=session_mask(tsw,"ny"); wi=np.where(ws)[0]
        pr=np.mean([b.predict_proba(Xw)[:,1] for b in boosters],axis=0)[wi]
        g=side_eval(pr,yw[wi],mw[wi],tsw[wi],THR)
        fwd[w]={k:{"n":g[k]["n"],"wr":round(g[k]["wr"],4)} for k in ("COMBINED","UP","DOWN")} if g else None
        b=BASE_FWD[w]
        print(f"  {w}: TB UP {fwd[w]['UP']['wr']} (base {b['UP']}) | DOWN {fwd[w]['DOWN']['wr']} (base {b['DOWN']}) | COMB {fwd[w]['COMBINED']['wr']} (base {b['COMB']})",flush=True)
    up_binding=all(fwd[y]["UP"]["wr"]>=BASE_FWD[y]["UP"] for y in ("2025","2026"))
    res={"key":"USDCAD.15m.ny.tbseedens","tb_k":TB_K,"nseed":NSEED,"cov":COV,"val_ny_auc":round(val_auc,4),
         "frozen_forward":fwd,"base_frozen_forward":BASE_FWD,
         "refit_cpcv":"TB+K3 UP p10 .6135@cov2 vs base .5968 (+.017, all covs, mean-confirmed) — usdcad_15m_cpcv_tbfirsttouch_ny_seedens3_result.json",
         "verdict":{"UP_fwd_beats_base_binding_2025_2026":bool(up_binding),
                    "note":"PROMOTE TB to UP leader + freeze iff UP fwd >= base UP fwd in BOTH binding years. Else refit-CPCV-only lift (record, keep base deliverable)."}}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[freeze-tb] UP-fwd-beats-base(2025&2026)={up_binding} -> {RESULT} ({time.time()-t0:.0f}s)",flush=True)
    if up_binding and DO_FREEZE:
        import manifest as MAN
        MODELS="models"; os.makedirs(MODELS,exist_ok=True); BOOK_ID="USDCAD.m15ny_tbseedens.v1"; mps=[]
        for sd,L in enumerate(boosters):
            mp=f"{MODELS}/m15ny_USDCAD_tb_s{sd}.txt"; L.booster_.save_model(mp); mps.append(mp)
        strat={"book":BOOK_ID,"pair":PAIR,"timeframe":"15m","session":"ny","nseed":NSEED,"tb_k":TB_K,
               "label_train":"triple-barrier first-touch k=2.0; EVAL fixed-15m sign ties-LOSE","gate":{"cov":COV,"conf_thr":THR},
               "cert":"NY refit-CPCV TB+K3: UP p10 .6135@cov2 (+.017 vs base); frozen-forward survives binding years","frozen_forward":fwd}
        sp=f"{MODELS}/m15ny_USDCAD_tbseedens_strategy.json"; json.dump(strat,open(sp,"w"),indent=2)
        man=MAN.build(book_id=BOOK_ID,timeframe="15m",side="combined",role="direction",
                      script="usdcad_15m_tbfirsttouch.py (cert) + usdcad_15m_freeze_tb.py (freeze)",
                      summary=f"USDCAD 15m NY TB-label(k={TB_K}) seed-ens K={NSEED}: UP-side improvement over base seed-ens (refit-CPCV UP p10 .6135@cov2 +.017, frozen-fwd survives). DOWN-neutral. AUC-flat operating-point gain.",
                      metrics={"up_refit_cpcv_p10_cov2":0.6135,"down_refit_cpcv_p10_cov2":0.5756,"breakeven":BE,"cov":COV,"val_ny_auc":round(val_auc,4),"frozen_forward":fwd,"refit_dependent":True},
                      artifacts=mps,hyperparams={"num_leaves":NUM_LEAVES,"nseed":NSEED,"tb_k":TB_K},strategy_json=sp,
                      feature_fingerprint=MAN.dir_fingerprint(H.FEAT_DIR,("USDCAD_*.parquet",)),created_utc="2026-06-10",
                      notes="TB first-touch TRAIN label; eval fixed-15m. UP-side upgrade; DOWN use base book. REFIT-DEPENDENT.")
        man["currency"]="USDCAD"; path=MAN.freeze(man,artifacts_src=mps+[sp]); json.dump(man,open(f"books/{BOOK_ID}.manifest.json","w"),indent=2)
        print(f"[freeze-tb] FROZEN {BOOK_ID} content_id={man['content_id']} -> {path}",flush=True)

if __name__=="__main__": main()
