"""USDJPY 15-MIN — SESSION-CONCENTRATED own-pair direction (A9: the highest-prior combo lever).

SCOPE: USDJPY · 15m. The EURUSD direction edge is decisively NY-session-concentrated (cross-pair certs
BOTH sides in NY only, null in LDN/Asia). For USDJPY the carrier session is unknown a-priori: NY = USD-home
+ carry flow; Asia/Tokyo = JPY-home + gotobi/Tokyo-fix flow. This script restricts the DECISION bars
(train + VAL gate-select + per-year eval) to a session (DST-correct exchange-tz via sessions.py) on the
OWN-PAIR base GBM (the A1 carrier; pooling diluted it). Features stay causal/continuous; only label/decision
rows are filtered (campaign rule). Tests whether session concentration lifts the binding-year UP/DOWN CI-lo
over breakeven 0.541.

Usage: ~/binary-algo-venv/bin/python usdjpy_15m_session.py <ny|ldn|asia|all> [stride] [leaves]
"""
import os, sys, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from sessions import session_mask
from usdjpy_15m_base import build, side_eval, covcurve, nonoverlap_chrono, boot, mk_lgb, BE, SPL, FEATS

PAIR="USDJPY"; HOR=15; GAP=900
SESSION = sys.argv[1] if len(sys.argv)>1 and not sys.argv[1].isdigit() else "ny"
def _argint(i, d):
    rest=[a for a in sys.argv[1:] if a.isdigit()]
    return int(rest[i]) if len(rest)>i else d
TR_STRIDE=_argint(0,6); NUM_LEAVES=_argint(1,127)
RESULT=f"usdjpy_15m_session_{SESSION}_s{TR_STRIDE}_l{NUM_LEAVES}_result.json"

def main():
    t0=time.time()
    res={"key":"USDJPY.15m","model":f"single-pair base GBM, SESSION={SESSION}-restricted decision bars (239 feats, 15m label)",
         "session":SESSION,"settlement":"bar-close approx, ties LOSE, BE 0.541, gap=900s","splits":SPL,
         "tr_stride":TR_STRIDE,"num_leaves":NUM_LEAVES,
         "falsifier":{"registered_utc":"pre-OOS",
            "KILL_if":"VAL moved-AUC <= 0.515 OR no held-out year (COMBINED or UP) CI95-lower clears 0.541 at the gate",
            "rationale":"A1 base UP cov3% .601/.566/.552 (CI-lo clears 2024 only). Session concentration (EURUSD edge is "
                        "NY-only) may lift the binding-year CI-lo over BE. Test ny/ldn/asia; the carrier session for "
                        "USDJPY is unknown (NY=USD+carry, Asia=JPY-home/Tokyo-fix)."}}
    json.dump(res,open(RESULT,"w"),indent=2)

    Xtr,ytr,mtr,tstr=build(SPL["train"], TR_STRIDE)
    Xva,yva,mva,tsv=build(SPL["val"])
    # session filter on DECISION rows (features already causal/continuous in X)
    str_s=session_mask(tstr,SESSION); va_s=session_mask(tsv,SESSION)
    itr=mtr & str_s; iva=mva & va_s
    print(f"[sess/{SESSION}] s{TR_STRIDE} l{NUM_LEAVES} train(sess-moved)={int(itr.sum()):,} val(sess-moved)={int(iva.sum()):,} build={time.time()-t0:.0f}s",flush=True)
    L=mk_lgb(num_leaves=NUM_LEAVES)
    L.fit(Xtr[itr], ytr[itr], eval_set=[(Xva[iva], yva[iva])], eval_metric="auc",
          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    pva=L.predict_proba(Xva)[:,1]
    val_auc=float(roc_auc_score(yva[iva], pva[iva]))
    print(f"[sess/{SESSION}] best_iter={L.best_iteration_} VAL(sess) moved-AUC={val_auc:.4f} {time.time()-t0:.0f}s",flush=True)
    res["val_auc"]=val_auc; res["best_iter"]=int(L.best_iteration_ or 0)

    # VAL worst-half gate on SESSION val decision rows
    vi=np.where(va_s)[0]; pvi=pva[vi]; yvi=yva[vi]; mvi=mva[vi]; tvi=tsv[vi]
    half=len(pvi)//2; confv=np.abs(pvi-0.5); best=None
    for cov in (0.20,0.10,0.05,0.03,0.02):
        thr=float(np.quantile(confv,1-cov)); accs=[]
        for s,e in ((0,half),(half,len(pvi))):
            r=side_eval(pvi[s:e],yvi[s:e],mvi[s:e],tvi[s:e],thr); accs.append(r["COMBINED"]["wr"] if r else float("nan"))
        worst=np.nanmin(accs)
        if best is None or worst>best[0]: best=(worst,cov,thr,accs)
    worst_half,COV,THR,halfaccs=best
    res["gate"]={"cov":COV,"conf_thr":THR,"val_worst_half_wr":float(worst_half),"val_half_wrs":[float(a) for a in halfaccs]}
    print(f"[sess/{SESSION}] FROZEN gate cov{COV:.0%} thr={THR:.4f} VAL worst-half WR={worst_half:.4f}",flush=True)

    res["years"]={}
    for w in ("test24","test25","oos"):
        Xw,yw,mw,tsw=build(SPL[w])
        ws=session_mask(tsw,SESSION)
        wi=np.where(ws)[0]
        pr=L.predict_proba(Xw)[:,1][wi]; ywi=yw[wi]; mwi=mw[wi]; twi=tsw[wi]
        auc=float(roc_auc_score(ywi[mwi], pr[mwi])) if mwi.sum()>20 else float("nan")
        up_rate=float(ywi[mwi].mean()) if mwi.sum()>0 else float("nan")
        gate=side_eval(pr,ywi,mwi,twi,THR); cc=covcurve(pr,ywi,mwi,twi)
        res["years"][w]={"moved_auc":auc,"moved_up_rate":up_rate,"gate":gate,"covcurve":cc,
                         "tripwire_ok":bool(0.46<=up_rate<=0.54)}
        g=gate["COMBINED"] if gate else {"n":0,"wr":float('nan'),"ci":[float('nan')]*2}
        u=gate["UP"] if gate else {"n":0,"wr":float('nan'),"ci":[float('nan')]*2}
        d=gate["DOWN"] if gate else {"n":0,"wr":float('nan'),"ci":[float('nan')]*2}
        print(f"=== {w} === AUC={auc:.4f} up-rate={up_rate:.4f} cov{COV:.0%}: COMB n{g['n']} {g['wr']:.4f} CI[{g['ci'][0]:.3f},{g['ci'][1]:.3f}] | "
              f"UP n{u['n']} {u['wr']:.4f} CI[{u['ci'][0]:.3f},{u['ci'][1]:.3f}] | DOWN n{d['n']} {d['wr']:.4f} CI[{d['ci'][0]:.3f},{d['ci'][1]:.3f}]",flush=True)

    up_clears=[w for w in ("test24","test25","oos") if res["years"][w]["gate"] and res["years"][w]["gate"]["UP"]["ci"][0]>=BE]
    dn_clears=[w for w in ("test24","test25","oos") if res["years"][w]["gate"] and res["years"][w]["gate"]["DOWN"]["ci"][0]>=BE]
    comb_clears=[w for w in ("test24","test25","oos") if res["years"][w]["gate"] and res["years"][w]["gate"]["COMBINED"]["ci"][0]>=BE]
    res["verdict"]={"val_auc_le_0515":bool(val_auc<=0.515),"UP_years_CIlo_clears_BE":up_clears,
                    "DOWN_years_CIlo_clears_BE":dn_clears,"COMBINED_years_CIlo_clears_BE":comb_clears,
                    "KILLED":bool(val_auc<=0.515 or (len(up_clears)==0 and len(dn_clears)==0 and len(comb_clears)==0))}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[sess/{SESSION}] VERDICT: {'KILLED' if res['verdict']['KILLED'] else 'SURVIVED'} "
          f"(val_auc={val_auc:.4f}; UP-clears={up_clears}; DOWN-clears={dn_clears}; COMB-clears={comb_clears}) -> {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
