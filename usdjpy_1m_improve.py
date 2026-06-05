"""USDJPY 1-MIN — Tier-I IMPROVE: seed-ensemble of the best base + pure UP/DOWN specialist controls.

SCOPE: USDJPY · 1m. The certified-book improve mandate (SWEEP_MATRIX Tier-I I2) + the (a)-(e) step-c
specialist control, on the best base (s6/l255). Tests whether variance reduction (K-seed prob-avg)
stabilizes/lifts the UP tail over breakeven, and confirms (EURUSD finding) that subset-trained
specialists are WORSE than the symmetric filter. Reports UP/DOWN covcurve per year.

Modes: `seedens` (K=5 seed ensemble), `spec` (pure UP-only / DOWN-only specialists).
Usage: ~/binary-algo-venv/bin/python usdjpy_1m_improve.py [seedens|spec] [stride]
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from usdjpy_1m_base import build, covcurve, side_eval, nonoverlap_chrono, boot

PAIR="USDJPY"; BE=0.541
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],
     "test24":["2024"],"test25":["2025"],"oos":["2026"]}
MODE=sys.argv[1] if len(sys.argv)>1 else "seedens"
STRIDE=int(sys.argv[2]) if len(sys.argv)>2 else 6

def lgbm(seed, leaves=255, n=3000):
    return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=leaves,
        min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,
        n_estimators=n,n_jobs=20,verbosity=-1,random_state=seed)

def prior5(years):
    """recompute prior-5min sign per row aligned to build() output, for specialist subset masks."""
    # build() already returns rows in the same order; reconstruct prior5 from close per year
    import harness as H
    out=[]
    for y in years:
        p=f"{H.FEAT_DIR}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p,columns=["close"]); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float); ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        contig=np.zeros(n,bool); contig[:n-1]=(ts[1:]-ts[:-1])==60
        fr=np.full(n,np.nan); fr[:n-1]=c[1:]/c[:-1]-1.0
        p5=np.full(n,np.nan); p5[5:]=c[5:]/c[:-5]-1.0
        X=pd.read_parquet(p,columns=H.feature_cols(PAIR)); keepf=X.isna().mean(axis=1).values<0.5
        valid=contig&np.isfinite(fr)&keepf
        out.append(p5[valid])
    return np.concatenate(out)

def main():
    t0=time.time(); RESULT=f"usdjpy_1m_improve_{MODE}_result.json"
    res={"key":"USDJPY.1m","mode":MODE,"stride":STRIDE,"splits":SPL,
         "falsifier":{"KILL_if":"no held-out year UP (or DOWN) cov1-2% CI-lo clears 0.541",
                      "incumbent":"s6/l255 UP cov2% .547/.534/.538 (worst .534, sub-BE)"}}
    Xtr,ytr,mtr,_=build(SPL["train"],STRIDE); Xva,yva,mva,tsv=build(SPL["val"])
    print(f"[imp/{MODE}] train={int(mtr.sum()):,} val={int(mva.sum()):,} build={time.time()-t0:.0f}s",flush=True)

    if MODE=="seedens":
        seeds=[11,23,37,51,67]; P=[]
        for s in seeds:
            L=lgbm(s); L.fit(Xtr[mtr],ytr[mtr],eval_set=[(Xva[mva],yva[mva])],eval_metric="auc",
                callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
            P.append(L); print(f"  seed{s} done {time.time()-t0:.0f}s",flush=True)
        def pred(X): return np.mean([m.predict_proba(X)[:,1] for m in P],axis=0)
        val_auc=float(roc_auc_score(yva[mva],pred(Xva)[mva])); res["val_auc"]=val_auc
        print(f"[imp/seedens] K={len(seeds)} VAL AUC={val_auc:.4f}",flush=True)
        res["years"]={}
        for w in ("test24","test25","oos"):
            Xw,yw,mw,tsw=build(SPL[w]); pr=pred(Xw); cc=covcurve(pr,yw,mw,tsw)
            res["years"][w]={"moved_auc":float(roc_auc_score(yw[mw],pr[mw])),"covcurve":cc}
            def g(cov,side): d=cc[cov].get(side,{}); return f"{d.get('wr')}(n{d.get('n')})"
            print(f"=== {w} === AUC={res['years'][w]['moved_auc']:.4f} | UP cov2%:{g('0.02','UP')} cov1%:{g('0.01','UP')} | DOWN cov2%:{g('0.02','DOWN')}",flush=True)

    elif MODE=="spec":
        p5tr=prior5(SPL["train"])
        res["years"]={}
        for sidename,subset,want in (("UP",p5tr<0,"UP"),("DOWN",p5tr>0,"DOWN")):
            sub=mtr&subset
            L=lgbm(7); L.fit(Xtr[sub],ytr[sub],eval_set=[(Xva[mva],yva[mva])],eval_metric="auc",
                callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
            print(f"[imp/spec] {sidename}-specialist trained on n={int(sub.sum()):,} (after-{'dip' if want=='UP' else 'rally'} bars) {time.time()-t0:.0f}s",flush=True)
            for w in ("test24","test25","oos"):
                Xw,yw,mw,tsw=build(SPL[w]); pr=L.predict_proba(Xw)[:,1]; cc=covcurve(pr,yw,mw,tsw)
                d=cc["0.02"].get(want,{})
                res.setdefault("years",{}).setdefault(w,{})[f"{sidename}_spec_cov2"]={"wr":d.get("wr"),"n":d.get("n")}
                print(f"  {sidename}-spec {w}: cov2% {want} wr={d.get('wr')} n{d.get('n')}",flush=True)
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[imp/{MODE}] saved {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
