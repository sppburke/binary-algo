"""USDJPY 1-MIN — Tier-I I1: Adaptive-Conformal (ACI) gate (the EURUSD-5m WINNER lever).

SCOPE: USDJPY · 1m. ACI (Gibbs-Candes) adapts the confidence threshold online to hold realized win-rate near
a target w* (trades more when recent calls win, less when they miss) — causal, no look-ahead. EURUSD 5m: ACI
beat the fixed gate. Test on the best base (s6/l255): can an ADAPTIVE gate sustain win-rate >= breakeven at
nonzero coverage where the fixed gate cannot? Both sides, per held-out year. (If unconditional WR is sub-BE,
ACI shrinks coverage toward 0 — expected null, but RUN don't argue.)
Usage: ~/binary-algo-venv/bin/python usdjpy_1m_aci.py [stride]
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from usdjpy_1m_base import build, nonoverlap_chrono, boot, mk_lgb

BE=0.541; SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],
     "test24":["2024"],"test25":["2025"],"oos":["2026"]}
STRIDE=int(sys.argv[1]) if len(sys.argv)>1 else 6
RESULT="usdjpy_1m_aci_result.json"

def aci_gate(pr, y, moved, ts, side, w_target=0.56, gamma=0.02, thr0=0.5):
    """Causal ACI: walk bars in time; maintain conf-threshold thr_t; take predicted-`side` bar iff conf>=thr_t;
    after a TAKEN trade, thr += gamma*( (1 if loss else 0) - (1-w_target) )  -> raise thr after losses."""
    order=np.argsort(ts); thr=thr0; taken=[]; wins=[]
    want_up = side=="UP"
    block_until=-1
    for i in order:
        if (pr[i]>0.5)!=want_up:    # only this side's predictions
            continue
        conf=abs(pr[i]-0.5)
        if conf>=thr and ts[i]>=block_until:
            win = 1.0 if (((pr[i]>0.5)==(y[i]==1)) and moved[i]) else 0.0
            taken.append(i); wins.append(win); block_until=int(ts[i])+60
            thr = max(0.0, thr + gamma*((1.0-win) - (1.0-w_target)))   # err - target_miss
    if len(taken)<5: return {"n":len(taken),"wr":float("nan"),"ci":[float("nan")]*2}
    wins=np.array(wins); lo,hi=boot(wins)
    return {"n":int(len(taken)),"wr":float(wins.mean()),"ci":[lo,hi],"final_thr":float(thr)}

def main():
    t0=time.time()
    res={"key":"USDJPY.1m","model":"ACI adaptive gate on s6/l255 base (Tier-I I1)","stride":STRIDE,
         "falsifier":{"KILL_if":"no held-out year UP or DOWN ACI win-rate CI-lo clears 0.541 at n>=75",
                      "note":"ACI is the EURUSD-5m winner lever; expected null here if unconditional WR sub-BE"}}
    Xtr,ytr,mtr,_=build(SPL["train"],STRIDE); Xva,yva,mva,tsv=build(SPL["val"])
    L=mk_lgb(num_leaves=255); L.fit(Xtr[mtr],ytr[mtr],eval_set=[(Xva[mva],yva[mva])],eval_metric="auc",
        callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    print(f"[aci] base VAL AUC={roc_auc_score(yva[mva],L.predict_proba(Xva)[mva][:,1]):.4f} {time.time()-t0:.0f}s",flush=True)
    res["years"]={}
    for w in ("test24","test25","oos"):
        Xw,yw,mw,tsw=build(SPL[w]); pr=L.predict_proba(Xw)[:,1]
        up=aci_gate(pr,yw,mw,tsw,"UP"); dn=aci_gate(pr,yw,mw,tsw,"DOWN")
        res["years"][w]={"UP":up,"DOWN":dn}
        print(f"=== {w} === UP n{up['n']} wr={up['wr']:.4f} CI[{up['ci'][0]:.3f},{up['ci'][1]:.3f}] | DOWN n{dn['n']} wr={dn['wr']:.4f} CI[{dn['ci'][0]:.3f},{dn['ci'][1]:.3f}]",flush=True)
    upc=[w for w in res["years"] if res["years"][w]["UP"]["n"]>=75 and res["years"][w]["UP"]["ci"][0]>=BE]
    dnc=[w for w in res["years"] if res["years"][w]["DOWN"]["n"]>=75 and res["years"][w]["DOWN"]["ci"][0]>=BE]
    res["verdict"]={"UP_clears":upc,"DOWN_clears":dnc,"KILLED":len(upc)==0 and len(dnc)==0}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[aci] {'KILLED' if res['verdict']['KILLED'] else 'SURVIVES'} (UP {upc}, DOWN {dnc}) {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
