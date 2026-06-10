"""USDCAD 15m NY — A8 separately-trained UP/DOWN SPECIALISTS + A2 compression-regime GATE (coverage-rule RUN).

SCOPE: USDCAD · 15m · NY. Fork of usdjpy_15m_spec.py. RUN (not argue) the Tier rows previously subsumed by EURUSD/USDJPY-1m precedent:
 A8: separately-trained per-side specialists (one-vs-rest target incl. ties as negatives) vs the symmetric
     certified model's same-side win-rate. Tests the "subset-training destroys ranking" claim AT THIS KEY.
 A2: compression-regime gate — trade NY bars only when a compression feature (low bb_width/rv tercile) holds,
     on top of the confidence gate. Tests if compression conditioning lifts the gated win-rate.
EVAL deriv-faithful fixed-15m, ties LOSE, nonoverlap gap=900, per-year CI95.

Usage: ~/binary-algo-venv/bin/python usdjpy_15m_spec.py
"""
import os, json, time, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sessions import session_mask
from usdcad_15m_base import build, side_eval, boot, mk_lgb, BE, SPL, FEATS

GAP=900; RESULT="usdcad_15m_spec_result.json"

def gate_thr(pva_ny, cov=0.03): return float(np.quantile(np.abs(pva_ny-0.5),1-cov))

def main():
    t0=time.time()
    Xtr,ytr,mtr,tstr=build(SPL["train"],6); Xva,yva,mva,tsv=build(SPL["val"])
    nytr=session_mask(tstr,"ny"); nyva=session_mask(tsv,"ny")
    res={"key":"USDCAD.15m.ny","checks":["A8 up/down separately-trained specialists","A2 compression-regime gate"],
         "incumbent":"symmetric NY seed-ens base (UP refit-p10 .5968@cov2 / DOWN .5814@cov5)",
         "falsifier":{"A8":"specialist same-side win-rate <= symmetric same-side => specialist KILLED (subset-train hurts)",
                      "A2":"compression-gated win-rate <= ungated => no lift"}}
    # ---- symmetric base (incumbent) ----
    sym=mk_lgb(num_leaves=127); itr=mtr&nytr; iva=mva&nyva
    sym.fit(Xtr[itr],ytr[itr],eval_set=[(Xva[iva],yva[iva])],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pva_sym=sym.predict_proba(Xva)[:,1]; thr=gate_thr(pva_sym[np.where(nyva)[0]])
    print(f"[spec] symmetric VAL-AUC(NY)={roc_auc_score(yva[iva],pva_sym[iva]):.4f} thr={thr:.4f} ({time.time()-t0:.0f}s)",flush=True)

    # ---- A8 specialists: one-vs-rest on ALL NY bars (ties as negatives) ----
    # need ties in train: rebuild with moved info; ytr is up/down on moved only -> reconstruct side targets on NY rows
    # up-spec target=1{up&moved}; down-spec target=1{down&moved}; trained on ALL NY rows (incl flats as 0)
    def side_target(y,m,side): return ((y==(1 if side=="UP" else 0))&m).astype(int)
    specs={}
    for side in ("UP","DOWN"):
        ytr_s=side_target(ytr,mtr,side); yva_s=side_target(yva,mva,side)
        Ls=mk_lgb(num_leaves=127)
        Ls.fit(Xtr[nytr],ytr_s[nytr],eval_set=[(Xva[nyva],yva_s[nyva])],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
        specs[side]=Ls
        print(f"[spec] {side}-specialist trained ({time.time()-t0:.0f}s)",flush=True)

    # ---- A2 compression feature: pick a bb_width / rv feature, low tercile = compressed ----
    comp_feat=next((c for c in FEATS if "bb_width" in c.lower() or "bbw" in c.lower()), None) \
              or next((c for c in FEATS if c.lower().endswith("rv") or "_rv" in c.lower()), FEATS[0])
    res["compression_feature"]=comp_feat
    comp_q=float(np.nanquantile(Xva[comp_feat].values[np.where(nyva)[0]],0.5))
    print(f"[spec] A2 compression feature={comp_feat} median(VAL-NY)={comp_q:.5g}",flush=True)

    res["years"]={}
    for w in ("test24","test25","oos"):
        Xw,yw,mw,tsw=build(SPL[w]); ny=session_mask(tsw,"ny"); wi=np.where(ny)[0]
        psym=sym.predict_proba(Xw)[:,1][wi]; ywi=yw[wi]; mwi=mw[wi]; twi=tsw[wi]
        gsym=side_eval(psym,ywi,mwi,twi,thr)
        # A8: each specialist's own-side selective win-rate (gate on specialist prob top-cov)
        a8={}
        for side in ("UP","DOWN"):
            ps=specs[side].predict_proba(Xw)[:,1][wi]
            thr_s=float(np.quantile(ps,1-0.03))           # top 3% most-confident for that side
            cand=ps>=thr_s
            # nonoverlap
            order=np.argsort(twi); idx=order[cand[order]]
            take=[];block=-1
            for i in idx:
                if twi[i]<block: continue
                take.append(i); block=int(twi[i])+GAP
            take=np.array(take,dtype=int)
            if len(take)>=5:
                # specialist predicts 'side' on selected bars; win iff actual move == side
                actual_side=np.where(ywi[take]==1,"UP","DOWN"); win=((actual_side==side)&mwi[take]).astype(float)
                lo,hi=boot(win); a8[side]={"n":int(len(take)),"wr":round(float(win.mean()),4),"ci":[round(lo,4),round(hi,4)]}
            else: a8[side]={"n":int(len(take)),"wr":None}
        # A2: symmetric model BUT only on compressed NY bars (comp_feat <= median)
        comp_mask=Xw[comp_feat].values[wi]<=comp_q
        ca=psym[comp_mask]; ya=ywi[comp_mask]; ma=mwi[comp_mask]; ta=twi[comp_mask]
        g_comp=side_eval(ca,ya,ma,ta,thr) if comp_mask.sum()>20 else None
        res["years"][w]={
            "symmetric_cov3":{k:[gsym[k]["n"],round(gsym[k]["wr"],4)] for k in ("COMBINED","UP","DOWN")} if gsym else None,
            "A8_specialist":a8,
            "A2_compressed_cov3":{k:[g_comp[k]["n"],round(g_comp[k]["wr"],4)] for k in ("COMBINED","UP","DOWN")} if g_comp else None}
        print(f"  {w}: SYM UP {res['years'][w]['symmetric_cov3']['UP'] if gsym else None} DOWN {res['years'][w]['symmetric_cov3']['DOWN'] if gsym else None} | "
              f"A8 UP {a8['UP']} DOWN {a8['DOWN']} | A2comp COMB {res['years'][w]['A2_compressed_cov3']['COMBINED'] if g_comp else None}",flush=True)

    # verdicts (binding worst-year)
    def binding(getter):
        vals=[getter(w) for w in res["years"] if getter(w) is not None]
        return round(min(vals),4) if vals else None
    sym_up=binding(lambda w: res["years"][w]["symmetric_cov3"]["UP"][1] if res["years"][w]["symmetric_cov3"] else None)
    a8_up =binding(lambda w: res["years"][w]["A8_specialist"]["UP"]["wr"] if res["years"][w]["A8_specialist"]["UP"]["wr"] else None)
    sym_dn=binding(lambda w: res["years"][w]["symmetric_cov3"]["DOWN"][1] if res["years"][w]["symmetric_cov3"] else None)
    a8_dn =binding(lambda w: res["years"][w]["A8_specialist"]["DOWN"]["wr"] if res["years"][w]["A8_specialist"]["DOWN"]["wr"] else None)
    res["verdict"]={"A8_UP_spec_beats_sym":bool(a8_up and sym_up and a8_up>sym_up),
                    "A8_DOWN_spec_beats_sym":bool(a8_dn and sym_dn and a8_dn>sym_dn),
                    "binding":{"sym_up":sym_up,"a8_up":a8_up,"sym_dn":sym_dn,"a8_dn":a8_dn}}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[spec] A8 binding: UP sym {sym_up} vs spec {a8_up}; DOWN sym {sym_dn} vs spec {a8_dn} -> spec_beats UP={res['verdict']['A8_UP_spec_beats_sym']} DOWN={res['verdict']['A8_DOWN_spec_beats_sym']}  {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
