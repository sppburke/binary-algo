"""NZDUSD 15m NY — Tier-I I1 (adaptive-conformal/ACI gate) + I4 (temperature calibration) on the certified book.

SCOPE: NZDUSD · 15m · NY. The certified deliverable uses a FIXED cov2% confidence gate. ACI (Gibbs-Candes
adaptive conformal) adapts the confidence threshold ONLINE to target a coverage α*, trading MORE in-regime and
LESS off-regime (causal, no look-ahead). I1 was the EURUSD-5m WINNER (regime-robust, +trades) — plain ACI
KILLED for AUDUSD 15m own-pair family; this is the NZDUSD-specific Tier-1 run.

Compares, per held-out NY year, the ACI gate vs the fixed cov2% gate: realized win-rate + coverage + CI95.
Falsifier: ACI must beat the fixed gate's binding-year win-rate at >= equal coverage (else no improvement).

Usage: ~/binary-algo-venv/bin/python nzdusd_15m_aci.py [target_cov=0.02] [eta=0.02]
"""
import os, sys, json, time, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sessions import session_mask
from nzdusd_15m_base import build, side_eval, boot, mk_lgb, BE, SPL

TARGET_COV=float(sys.argv[1]) if len(sys.argv)>1 else 0.02
ETA=float(sys.argv[2]) if len(sys.argv)>2 else 0.02
GAP=900; RESULT=f"nzdusd_15m_aci_result.json"

def temp_fit(p, y):
    """1-D temperature scaling: minimize logloss over T on logit(p). Returns T."""
    p=np.clip(p,1e-6,1-1e-6); z=np.log(p/(1-p))
    best=(1e9,1.0)
    for T in np.linspace(0.5,3.0,51):
        q=1/(1+np.exp(-z/T)); q=np.clip(q,1e-6,1-1e-6)
        ll=-np.mean(y*np.log(q)+(1-y)*np.log(1-q))
        if ll<best[0]: best=(ll,T)
    return best[1]

def aci_stream(pr, fwd, ts, target_cov, eta, thr0):
    """Online ACI on the chronological NY moved-bar stream. Adapt conf threshold to target trade-rate target_cov.
    Causal: threshold updated AFTER each bar from realized take/skip. nonoverlap gap enforced. ties LOSE.
    Returns (n_trades, win_rate, coverage)."""
    conf=np.abs(pr-0.5)
    order=np.argsort(ts); pr=pr[order]; fwd=fwd[order]; ts=ts[order]; conf=conf[order]
    thr=thr0; block=-1; taken=[]; wins=[]
    for i in range(len(ts)):
        # ACI threshold update toward target trade-rate: if we'd trade (conf>=thr) push thr up, else down
        would=conf[i]>=thr
        thr=thr+eta*((1 if would else 0)-target_cov)*0.5   # nudge so long-run trade-rate ~ target_cov
        thr=max(0.0,min(0.5,thr))
        if not would: continue
        if ts[i]<block: continue                            # nonoverlap
        block=int(ts[i])+GAP
        pred=1 if pr[i]>0.5 else 0; yl=1 if fwd[i]>0 else 0; moved=fwd[i]!=0
        taken.append(i); wins.append(1.0 if (pred==yl and moved) else 0.0)
    if not taken: return 0,float("nan"),0.0
    return len(taken), float(np.mean(wins)), len(taken)/max(1,len(ts))

def main():
    t0=time.time()
    Xtr,ytr,mtr,tstr=build(SPL["train"],6); Xva,yva,mva,tsv=build(SPL["val"])
    nytr=session_mask(tstr,"ny"); nyva=session_mask(tsv,"ny")
    itr=mtr&nytr; iva=mva&nyva
    L=mk_lgb(num_leaves=127)
    L.fit(Xtr[itr],ytr[itr],eval_set=[(Xva[iva],yva[iva])],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pva=L.predict_proba(Xva)[:,1]
    # I4 calibration: fit temperature on VAL NY moved
    T=temp_fit(pva[iva],yva[iva])
    def cal(p):
        p=np.clip(p,1e-6,1-1e-6); z=np.log(p/(1-p)); return 1/(1+np.exp(-z/T))
    # fixed gate thr0 from VAL NY at target_cov (on calibrated conf)
    cva=np.abs(cal(pva[np.where(nyva)[0]])-0.5); thr0=float(np.quantile(cva,1-TARGET_COV))
    res={"key":"NZDUSD.15m.ny","model":"certified NY base + I1 ACI gate + I4 temp-calibration","target_cov":TARGET_COV,"eta":ETA,
         "temperature":round(T,3),"val_auc":round(float(roc_auc_score(yva[iva],pva[iva])),4),"fixed_thr0":round(thr0,4),
         "falsifier":{"KILL_if":"ACI does NOT beat the fixed cov-gate binding-year win-rate at >= equal coverage"}}
    print(f"[aci] T={T:.3f} thr0={thr0:.4f} target_cov={TARGET_COV}",flush=True)
    res["years"]={}
    for w in ("test24","test25","oos"):
        Xw,yw,mw,tsw=build(SPL[w]); ny=session_mask(tsw,"ny"); wi=np.where(ny)[0]
        prc=cal(L.predict_proba(Xw)[:,1][wi]); fw=yw[wi].astype(float); mwi=mw[wi]; twi=tsw[wi]
        frw=np.where(mwi, np.where(yw[wi]==1,1.0,-1.0), 0.0)   # sign proxy for ties-lose (moved up=+,down=-,flat=0)
        # FIXED gate (calibrated): reuse side_eval on calibrated probs
        gfix=side_eval(prc, yw[wi], mwi, twi, thr0)
        # ACI gate
        nA,wA,covA=aci_stream(prc, frw, twi, TARGET_COV, ETA, thr0)
        loA,hiA=boot(np.concatenate([np.ones(int(round(wA*nA))),np.zeros(nA-int(round(wA*nA)))])) if nA>=5 else (float("nan"),float("nan"))
        gf=gfix["COMBINED"] if gfix else {"n":0,"wr":float("nan"),"ci":[float("nan")]*2}
        res["years"][w]={"fixed":{"n":gf["n"],"wr":round(gf["wr"],4),"ci":[round(c,4) for c in gf["ci"]]},
                         "aci":{"n":nA,"wr":round(wA,4),"cov":round(covA,4),"ci":[round(loA,4),round(hiA,4)]}}
        print(f"  {w}: FIXED n{gf['n']} wr={gf['wr']:.4f} CI[{gf['ci'][0]:.3f},{gf['ci'][1]:.3f}] | ACI n{nA} wr={wA:.4f} cov={covA:.4f} CI[{loA:.3f},{hiA:.3f}]",flush=True)
    # verdict: does ACI beat fixed on the binding (worst) year at >= coverage?
    fb=min(res["years"][w]["fixed"]["wr"] for w in res["years"])
    ab=min(res["years"][w]["aci"]["wr"] for w in res["years"])
    res["verdict"]={"fixed_binding_wr":round(fb,4),"aci_binding_wr":round(ab,4),"ACI_improves":bool(ab>fb)}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[aci] binding-year: FIXED {fb:.4f} vs ACI {ab:.4f} -> ACI_improves={res['verdict']['ACI_improves']}  {time.time()-t0:.0f}s -> {RESULT}",flush=True)

if __name__=="__main__":
    main()
