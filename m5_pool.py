"""POOLED CROSS-PAIR direction model (§3 'the one untested STRUCTURAL lever') — UP + DOWN, focus the 2025 DOWN wall.

Distinct from m5xp (which uses other pairs' residuals as FEATURES for EURUSD): here we TRAIN one weight-shared GBM
on the POOLED bars of all 7 USD majors (each pair's own base-239 features → its own 5m direction label), so the
model learns a shared direction function from 7× the data + 7 regimes. HYPOTHESIS (sign-aware; both sides): other
majors carry tradeable 2025 DOWN structure EURUSD-only training misses → pooling may lift the DOWN 2025 binding
regime that killed every EURUSD-only config. Also run a |return|-weighted (POW=0.5) pooled variant for DOWN.

PRE-REGISTERED FALSIFIER: KILL unless pooled (plain OR magweight) lifts EURUSD DOWN binding-2025 moved-acc CI95-lo
≥0.541 at usable cover (n≥100), beating the EURUSD-only wall (.52-.53). UP reported vs floor .553. Settlement:
wc-contiguous 300s, moved-only, ties LOSE, nonoverlap, per-year CI95, up-rate tripwire.

  ~/binary-algo-venv/bin/python m5_pool.py
"""
import os, sys; sys.argv=["x"]
import json, time, numpy as np, pandas as pd
import lightgbm as lgb
import harness as H
ROOT="/home/sean/git/binary-algo"; FEAT=H.FEAT_DIR; BE=0.541; HOR=5
PAIRS=["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCHF","USDCAD"]
FEATS=list(H.feature_cols("EURUSD")); TRAIN=range(2012,2022)
def boot(c,nb=2500,seed=7):
    c=np.asarray(c,float)
    if len(c)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(c); a=np.array([c[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))
def yr(ts): return (np.asarray(ts,dtype="datetime64[s]").astype("datetime64[Y]").astype(int)+1970)

def load(pair, years, want_ny=False):
    Xs,ys,tss,fws,nys=[],[],[],[],[]
    for y in years:
        p=f"{FEAT}/{pair}_{y}.parquet"
        if not os.path.exists(p): continue
        df=pd.read_parquet(p,columns=FEATS+["close"])
        df=df[~df.index.duplicated(keep="last")].sort_index()
        secs=df.index.values.astype("datetime64[s]").astype("int64"); c=df["close"].values.astype(float); n=len(c)
        fwd=np.full(n,np.nan)
        if n>HOR:
            contig=(secs[HOR:]-secs[:-HOR])==HOR*60; fwd[:n-HOR]=np.where(contig, np.log(c[HOR:])-np.log(c[:-HOR]), np.nan)
        valid=np.isfinite(fwd)&(fwd!=0)
        hour=((secs%86400)//3600)            # UTC hour; NY session ~13:00-21:00 UTC
        ny=(hour>=13)&(hour<21)
        Xs.append(df.loc[valid,FEATS].to_numpy(np.float32)); ys.append((fwd[valid]>0).astype(np.int8))
        tss.append(secs[valid]); fws.append(fwd[valid].astype(np.float32)); nys.append(ny[valid])
        del df
    if not Xs: return None
    return (np.concatenate(Xs),np.concatenate(ys),np.concatenate(tss),np.concatenate(fws),np.concatenate(nys))

def mk(): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.03,num_leaves=255,
    min_child_samples=300,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,n_estimators=1200,n_jobs=20,verbosity=-1)

def main():
    t0=time.time(); rng=np.random.default_rng(7)
    # pooled TRAIN across 7 majors
    Xs,ys,ws=[],[],[]
    for p in PAIRS:
        d=load(p,TRAIN)
        if d is None: continue
        X,y,ts,fw,ny=d
        if len(X)>120000:
            idx=np.sort(rng.choice(len(X),120000,replace=False)); X,y,fw=X[idx],y[idx],fw[idx]
        a=np.abs(fw); med=np.median(a[a>0]) or 1e-9; w=np.clip((a/med)**0.5,0.1,10.0)
        Xs.append(X); ys.append(y); ws.append(w)
        print(f"  pooled {p}: {len(X):,} ({time.time()-t0:.0f}s)",flush=True)
    Xtr=np.concatenate(Xs); ytr=np.concatenate(ys); wtr=np.concatenate(ws)
    print(f"[pool] pooled train n={len(Xtr):,} across {len(Xs)} pairs {time.time()-t0:.0f}s",flush=True)
    models={}
    for name,sw in (("plain",None),("magw",wtr)):
        m=mk(); m.fit(Xtr,ytr,sample_weight=sw); models[name]=m; print(f"  fit {name} {time.time()-t0:.0f}s",flush=True)
    # eval on EURUSD held-out, NY-gated, per-year UP+DOWN sel
    EV={str(Y):load("EURUSD",[Y]) for Y in (2024,2025,2026)}
    def per_year(model, side, sv, cov):
        res={}
        for Y in (2024,2025,2026):
            d=EV[str(Y)]
            if d is None: continue
            X,y,ts,fw,ny=d; pr=model.predict_proba(X)[:,1]; conf=np.abs(pr-0.5)
            gate=ny&((pr>0.5) if sv==1 else (pr<0.5))
            if gate.sum()<20: continue
            cthr=np.quantile(conf[gate],1-cov); m=gate&(conf>=cthr); sel=_nonoverlap(ts,m)
            if len(sel)>=20:
                cc=(y[sel]==sv).astype(float); lo,hi=boot(cc); res[Y]=dict(win=round(float(cc.mean()),4),n=len(sel),ci=[round(lo,4),round(hi,4)],uprate=round(float((y[sel]==1).mean()),3))
        return res
    def _nonoverlap(ts,mask,gap=300):
        idx=np.where(mask)[0]; out=[]; last=-1e18
        for i in idx:
            if ts[i]>=last+gap: out.append(i); last=ts[i]
        return np.array(out,int)
    out={"test":"POOLED cross-pair (7 majors) direction — UP+DOWN, plain & magweight","breakeven":BE,"covs":[0.05,0.10]}
    best={}
    for name in models:
        out[name]={}
        for side,sv in (("UP",1),("DOWN",0)):
            for cov in (0.05,0.10):
                py=per_year(models[name],side,sv,cov); out[name][f"{side}_cov{cov}"]=py
                b=[py[Y]["win"] for Y in py]; lo=[py[Y]["ci"][0] for Y in py]; ns=[py[Y]["n"] for Y in py]
                ok=bool(lo and min(lo)>BE and ns and min(ns)>=100 and len(py)==3)
                best[f"{name}_{side}_cov{cov}"]=dict(bind=min(b) if b else None, bind_lo=min(lo) if lo else None, min_n=min(ns) if ns else 0, all3=ok)
    down_ok=any(best[k]["all3"] for k in best if "_DOWN_" in k)
    up_ok=any(best[k]["all3"] for k in best if "_UP_" in k)
    out["cert"]=best
    out["VERDICT"]=dict(DOWN_pooled_certified=down_ok, UP_pooled_certified=up_ok,
        statement=(f"POOLED cross-pair: DOWN {'CERTIFIED (escalate to CPCV)' if down_ok else 'still fails the 2025 wall'}, "
            f"UP {'clears' if up_ok else 'below floor'}. "
            + ("Pooling cracks the DOWN 2025 regime." if down_ok else
               "Pooling 7 majors does NOT crack the DOWN 2025 regime wall — the structural lever is applied-and-failed; DOWN exhausted incl. pooling.")))
    json.dump(out,open(f"{ROOT}/m5_pool_result.json","w"),indent=2,default=str)
    for k,v in best.items(): print(f"  {k}: bind {v['bind']} CI-lo {v['bind_lo']} n{v['min_n']} all3={v['all3']}")
    print(f"VERDICT: {out['VERDICT']['statement']}\n-> m5_pool_result.json ({time.time()-t0:.0f}s)")

if __name__=="__main__": main()
