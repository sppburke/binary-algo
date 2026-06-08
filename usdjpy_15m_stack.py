"""USDJPY 15m NY — A5 cross-horizon STACK (30m parent confidence front-loads the 15m child gate).

SCOPE: USDJPY · 15m · NY. Mechanism (non-subsumed): the 30m parent's directional confidence may carry
lower-frequency drift sign orthogonal to the 15m own-pair signal. Two tests vs the certified 15m-alone cert:
 (1) HARD-AGREEMENT gate: trade the 15m signal only when the 30m parent AGREES on direction (causal — both
     use features known at bar t). Classic parent→child front-load.
 (2) FEATURE-ADD: add the 30m P(up) as a meta-feature to the 15m model, retrain, compare VAL-AUC.
EVAL deriv-faithful fixed-15m, ties LOSE, nonoverlap gap=900, per-year CI95. Incumbent: 15m NY cert
(UP cov3% .586 / seed-ens cov2% .60). Falsifier: stack must beat the 15m-alone binding-year win-rate.

Usage: ~/binary-algo-venv/bin/python usdjpy_15m_stack.py   (requires models/m30_USDJPY_direction_lgb.txt)
"""
import os, json, time, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sessions import session_mask
from usdjpy_15m_base import build, side_eval, boot, mk_lgb, BE, SPL, FEATS

GAP=900; RESULT="usdjpy_15m_stack_result.json"
M30="models/m30_USDJPY_direction_lgb.txt"

def main():
    t0=time.time()
    if not os.path.exists(M30):
        json.dump({"error":"30m parent model missing — run usdjpy_30m_base.py first","verdict":"SKIPPED"},open(RESULT,"w"),indent=2)
        print("[stack] 30m parent missing -> SKIPPED",flush=True); return
    par=lgb.Booster(model_file=M30)
    Xtr,ytr,mtr,tstr=build(SPL["train"],6); Xva,yva,mva,tsv=build(SPL["val"])
    nytr=session_mask(tstr,"ny"); nyva=session_mask(tsv,"ny")
    itr=mtr&nytr; iva=mva&nyva
    # 15m NY model (incumbent)
    L=mk_lgb(num_leaves=127); L.fit(Xtr[itr],ytr[itr],eval_set=[(Xva[iva],yva[iva])],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pva15=L.predict_proba(Xva)[:,1]; thr15=float(np.quantile(np.abs(pva15[np.where(nyva)[0]]-0.5),1-0.03))
    # parent thr on VAL NY
    pva30=par.predict(Xva.values); thr30=float(np.quantile(np.abs(pva30[np.where(nyva)[0]]-0.5),1-0.10))  # looser parent gate
    res={"key":"USDJPY.15m.ny","model":"A5 cross-horizon: 30m parent front-loads 15m child","thr15":round(thr15,4),"thr30":round(thr30,4),
         "incumbent":"15m NY cert UP cov3% .586 / seed-ens cov2% .60",
         "falsifier":{"KILL_if":"agreement-gated binding-year wr <= 15m-alone binding-year wr (no orthogonal lift)"}}
    print(f"[stack] 15m thr={thr15:.4f} 30m thr={thr30:.4f} ({time.time()-t0:.0f}s)",flush=True)

    # ---- (2) FEATURE-ADD: add p30 to 15m feats, retrain, VAL-AUC ----
    def addp30(X): X=X.copy(); X["p30_parent"]=par.predict(X.values[:, :len(FEATS)]).astype("float32"); return X
    Xtr2=Xtr.copy(); Xtr2["p30_parent"]=pva30_tr=par.predict(Xtr.values).astype("float32")
    Xva2=Xva.copy(); Xva2["p30_parent"]=pva30.astype("float32")
    L2=mk_lgb(num_leaves=127); L2.fit(Xtr2[itr],ytr[itr],eval_set=[(Xva2[iva],yva[iva])],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    val_auc_feat=float(roc_auc_score(yva[iva],L2.predict_proba(Xva2[iva])[:,1]))
    res["feature_add_val_auc"]=round(val_auc_feat,4); res["feature_add_beats_base"]=bool(val_auc_feat>0.539)
    print(f"[stack] FEATURE-ADD VAL-AUC(NY)={val_auc_feat:.4f} (base .539) beats={val_auc_feat>0.539}",flush=True)

    res["years"]={}
    for w in ("test24","test25","oos"):
        Xw,yw,mw,tsw=build(SPL[w]); ny=session_mask(tsw,"ny"); wi=np.where(ny)[0]
        p15=L.predict_proba(Xw)[:,1][wi]; p30=par.predict(Xw.values)[wi]; ywi=yw[wi]; mwi=mw[wi]; twi=tsw[wi]
        # 15m-alone cov3%
        g_alone=side_eval(p15,ywi,mwi,twi,thr15)
        # (1) hard-agreement: 15m confident AND 30m agrees direction
        agree=(np.sign(p15-0.5)==np.sign(p30-0.5))
        conf15=np.abs(p15-0.5)>=thr15
        cand=agree & conf15 & np.isfinite(ywi.astype(float))
        order=np.argsort(twi); take=[];block=-1
        for i in order[cand[order]]:
            if twi[i]<block: continue
            take.append(i); block=int(twi[i])+GAP
        take=np.array(take,dtype=int)
        if len(take)>=5:
            pred=(p15[take]>0.5).astype(int); win=((pred==ywi[take])&mwi[take]).astype(float)
            lo,hi=boot(win); up=pred==1; dn=pred==0
            agg={"COMBINED":[int(len(take)),round(float(win.mean()),4),[round(lo,4),round(hi,4)]],
                 "UP":[int(up.sum()),round(float(win[up].mean()),4) if up.sum() else None],
                 "DOWN":[int(dn.sum()),round(float(win[dn].mean()),4) if dn.sum() else None]}
        else: agg={"COMBINED":[len(take),None,None]}
        res["years"][w]={"alone_cov3":{k:[g_alone[k]["n"],round(g_alone[k]["wr"],4)] for k in ("COMBINED","UP","DOWN")} if g_alone else None,
                         "agreement_gate":agg}
        print(f"  {w}: 15m-alone COMB {res['years'][w]['alone_cov3']['COMBINED'] if g_alone else None} | agree-gate COMB {agg['COMBINED']}",flush=True)

    def binding(getter):
        vals=[getter(w) for w in res["years"] if getter(w) is not None]
        return round(min(vals),4) if vals else None
    al=binding(lambda w: res["years"][w]["alone_cov3"]["COMBINED"][1] if res["years"][w]["alone_cov3"] else None)
    ag=binding(lambda w: res["years"][w]["agreement_gate"]["COMBINED"][1] if res["years"][w]["agreement_gate"]["COMBINED"][1] else None)
    res["verdict"]={"alone_binding":al,"agree_binding":ag,"stack_improves":bool(ag and al and ag>al),
                    "feature_add_beats_base":res["feature_add_beats_base"]}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[stack] binding COMB: alone {al} vs agree-gate {ag} -> stack_improves={res['verdict']['stack_improves']}; feature-add beats base={res['feature_add_beats_base']}  {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
