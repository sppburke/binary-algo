"""USDJPY 1-MIN — REGIME-GATED side FILTER (attack v1/v2: lift the UP side over breakeven).

SCOPE: USDJPY · 1m. Motivated by Tier-1 RAW conditional structure (this session): USDJPY 1m has a
signed MEAN-REVERSION (dip-buy) tilt that is OOS-stable and AMPLIFIED by low-vol compression and the
Tokyo session — raw moved-bar UP-rate after-dip .515/.510/.517 (vs after-rally .495/.495/.497);
comp×Tokyo×dip .513/.520/.524 (2024/25/26, OOS-best). The GBM's confident tail selects WITHIN this
structure. So: train a SYMMETRIC GBM (subset-trained specialists destroy ranking — EURUSD finding),
then FILTER at inference to {dip × compression × session} × predicted-side, take the model-confident
bars. Gate params (compression pct, session) FROZEN on VAL worst-half; reported per-year held-out.

Settlement bar-close approx, ties LOSE, breakeven 0.541, gap=60s, per-year CI95, moved-only.
Falsifier pre-registered. Usage: ~/binary-algo-venv/bin/python usdjpy_1m_regime.py [stride]
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from usdjpy_1m_base import nonoverlap_chrono, boot, mk_lgb

PAIR="USDJPY"; HOR=1; GAP=60; BE=0.541
FEAT=H.FEAT_DIR; FEATS=H.feature_cols(PAIR)
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],
     "test24":["2024"],"test25":["2025"],"oos":["2026"]}
STRIDE=int(sys.argv[1]) if len(sys.argv)>1 else 8
RESULT=f"usdjpy_1m_regime_s{STRIDE}_result.json"

def build_g(years, stride=1):
    Xs=[];ys=[];mv=[];tss=[];p5=[];rv=[];hh=[]
    for y in years:
        p=f"{FEAT}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p,columns=FEATS+["close"]); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float); ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        contig=np.zeros(n,bool); contig[:n-1]=(ts[1:]-ts[:-1])==60
        fr=np.full(n,np.nan); fr[:n-1]=c[1:]/c[:-1]-1.0
        prior5=np.full(n,np.nan); prior5[5:]=c[5:]/c[:-5]-1.0
        rv30=pd.Series(c).pct_change().rolling(30).std().values
        hours=d.index.hour.values+d.index.minute.values/60.0
        X=d[FEATS].astype("float32"); keepf=X.isna().mean(axis=1).values<0.5
        valid=contig&np.isfinite(fr)&np.isfinite(prior5)&np.isfinite(rv30)&keepf
        idx=np.where(valid)[0]
        if stride>1: idx=idx[::stride]
        Xs.append(X.iloc[idx]); ys.append((fr[idx]>0).astype(int)); mv.append((fr[idx]!=0))
        tss.append(ts[idx]); p5.append(prior5[idx]); rv.append(rv30[idx]); hh.append(hours[idx])
    return (pd.concat(Xs),np.concatenate(ys),np.concatenate(mv),np.concatenate(tss),
            np.concatenate(p5),np.concatenate(rv),np.concatenate(hh))

def filt_eval(pr,y,moved,ts,gatemask,side,thr):
    """Within gatemask, take predicted-`side` bars with conf>=thr, nonoverlap; ties LOSE."""
    conf=np.abs(pr-0.5)
    want=(pr>0.5) if side=="UP" else (pr<0.5)
    cand=gatemask&want&(conf>=thr)
    tr=nonoverlap_chrono(ts,cand)
    if len(tr)==0: return {"n":0,"wr":float("nan"),"ci":[float("nan")]*2}
    pred=(pr[tr]>0.5).astype(int)
    win=((pred==y[tr])&moved[tr]).astype(float)
    lo,hi=boot(win); return {"n":int(len(tr)),"wr":float(win.mean()),"ci":[lo,hi]}

def main():
    t0=time.time()
    res={"key":"USDJPY.1m","model":f"regime-gated side FILTER on symmetric GBM (stride{STRIDE})",
         "mechanism":"dip-buy (prior5<0) x compression (rv30 low) x session; symmetric GBM + side+regime filter",
         "settlement":"bar-close approx, ties LOSE, BE 0.541, gap60","splits":SPL,
         "falsifier":{"registered_utc":"pre-OOS",
            "KILL_if":"no held-out year UP win-rate CI95-lower clears 0.541 at the VAL-frozen gate (n>=150)",
            "incumbent":"single-pair baseline UP cov1% .538/.540/.545 (sub-BE)"}}
    json.dump(res,open(RESULT,"w"),indent=2)

    Xtr,ytr,mtr,_,_,_,_=build_g(SPL["train"],STRIDE)
    Xva,yva,mva,tsv,p5v,rvv,hhv=build_g(SPL["val"])
    print(f"[regime] train={int(mtr.sum()):,} val={int(mva.sum()):,} build={time.time()-t0:.0f}s",flush=True)
    L=mk_lgb(num_leaves=255)
    L.fit(Xtr[mtr],ytr[mtr],eval_set=[(Xva[mva],yva[mva])],eval_metric="auc",
          callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pva=L.predict_proba(Xva)[:,1]; val_auc=float(roc_auc_score(yva[mva],pva[mva]))
    print(f"[regime] best_iter={L.best_iteration_} VAL AUC={val_auc:.4f} {time.time()-t0:.0f}s",flush=True)
    res["val_auc"]=val_auc

    # compression threshold from VAL (q33 of rv30), session masks
    cq=float(np.nanpercentile(rvv,33))
    def masks(p5,rv,hh):
        dip=p5<0; rally=p5>0; comp=rv<cq; tk=(hh>=0)&(hh<9); ny=(hh>=13)&(hh<22)
        return {"dip":dip,"dip_comp":dip&comp,"dip_comp_tk":dip&comp&tk,
                "rally":rally,"rally_comp":rally&comp,"rally_comp_ny":rally&comp&ny}
    mv_=masks(p5v,rvv,hhv); res["comp_q33"]=cq

    # FREEZE on VAL worst-half: for UP gates pick (gate,conf-cov) maximizing worst-half UP wr (n>=150)
    half=len(pva)//2; confv=np.abs(pva-0.5)
    def pick(side, gates):
        best=None
        for g in gates:
            for cov in (0.5,0.3,0.2,0.1):   # coverage WITHIN the gate
                gm=mv_[g]
                cg=confv[gm]
                if len(cg)<300: continue
                thr=float(np.quantile(cg,1-cov))
                accs=[]
                for s,e in ((0,half),(half,len(pva))):
                    sl=np.zeros(len(pva),bool); sl[s:e]=True
                    r=filt_eval(pva,yva,mva,tsv,gm&sl,side,thr)
                    accs.append(r["wr"] if r["n"]>=80 else np.nan)
                worst=np.nanmin(accs) if np.isfinite(accs).any() else np.nan
                if np.isfinite(worst) and (best is None or worst>best[0]): best=(worst,g,cov,thr)
        return best
    upg=pick("UP",["dip","dip_comp","dip_comp_tk"])
    dng=pick("DOWN",["rally","rally_comp","rally_comp_ny"])
    res["UP_gate"]={"val_worst_half_wr":upg[0],"gate":upg[1],"cov_in_gate":upg[2],"thr":upg[3]} if upg else None
    res["DOWN_gate"]={"val_worst_half_wr":dng[0],"gate":dng[1],"cov_in_gate":dng[2],"thr":dng[3]} if dng else None
    print(f"[regime] UP gate={upg}  DOWN gate={dng}",flush=True)

    res["years"]={}
    for w in ("test24","test25","oos"):
        Xw,yw,mw,tsw,p5w,rvw,hhw=build_g(SPL[w]); pr=L.predict_proba(Xw)[:,1]
        mwk=masks(p5w,rvw,hhw)
        yr={"moved_auc":float(roc_auc_score(yw[mw],pr[mw])),"moved_up_rate":float(yw[mw].mean())}
        if upg: yr["UP"]=filt_eval(pr,yw,mw,tsw,mwk[upg[1]],"UP",upg[3])
        if dng: yr["DOWN"]=filt_eval(pr,yw,mw,tsw,mwk[dng[1]],"DOWN",dng[3])
        res["years"][w]=yr
        u=yr.get("UP",{}); dn=yr.get("DOWN",{})
        print(f"=== {w} === AUC={yr['moved_auc']:.4f} up-rate={yr['moved_up_rate']:.4f} | "
              f"UP n{u.get('n','-')} wr={u.get('wr',float('nan')):.4f} CI[{u.get('ci',[0,0])[0]:.3f},{u.get('ci',[0,0])[1]:.3f}] | "
              f"DOWN n{dn.get('n','-')} wr={dn.get('wr',float('nan')):.4f} CI[{dn.get('ci',[0,0])[0]:.3f},{dn.get('ci',[0,0])[1]:.3f}]",flush=True)

    up_clears=[w for w in res["years"] if res["years"][w].get("UP",{}).get("n",0)>=150 and res["years"][w]["UP"]["ci"][0]>=BE]
    dn_clears=[w for w in res["years"] if res["years"][w].get("DOWN",{}).get("n",0)>=150 and res["years"][w]["DOWN"]["ci"][0]>=BE]
    res["verdict"]={"UP_years_CIlo_clears_BE":up_clears,"DOWN_years_CIlo_clears_BE":dn_clears,
                    "UP_KILLED":len(up_clears)==0,"DOWN_KILLED":len(dn_clears)==0}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[regime] UP {'KILLED' if res['verdict']['UP_KILLED'] else 'SURVIVES'} (clears {up_clears}); "
          f"DOWN {'KILLED' if res['verdict']['DOWN_KILLED'] else 'SURVIVES'} (clears {dn_clears}) -> {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
