"""DOWN IMPROVE (§3, untried on DOWN): SEED-ENSEMBLE the POW=0.5 magweight primary, forward per-year DOWN+UP-sel.

The magweight-DOWN survivor (m5_magweight_cpcv: DOWN cov0.05 refit p10 .5441/89% MEETS §4) FAILS §1(e) only on the
FORWARD 2025 CI-lo .533<.541. Seed-ensembling (K independent seeds, prob-avg) REDUCES tail variance → narrows the
forward CI and may lift the point. This is the mandated §3 improve lever, never applied to the DOWN side.

STAGE 1 (this script, cheap): train K=4 magweight seeds on full TRAIN, forward per-year DOWN+UP selective at
cov{0.03,0.05,0.10}, CI95, nonoverlap, up-rate tripwire. PRE-REGISTERED: if seed-ens lifts forward 2025 DOWN CI-lo
≥0.541 at any cov (n≥100) → DOWN candidate-certified, ESCALATE to nested-refit CPCV. Else DOWN improve-lever
applied-and-failed → genuinely exhausted. (UP reported too: does seed-ens magweight beat UP floor .553?)

  ~/binary-algo-venv/bin/python m5_magweight_seedens.py
"""
import os, sys; sys.argv=["x"]
import json, time, numpy as np
import lightgbm as lgb
import m5_xpair as MX, m5_xpair_production as XP
ROOT="/home/sean/git/binary-algo"; BE=0.541; K=4; POW=0.5; COVS=[0.03,0.05,0.10]
def magw(fwd):
    a=np.abs(fwd).astype(float); med=np.median(a[a>0]) or 1e-9
    return np.clip((a/med)**POW,0.1,10.0)
def mk(seed): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=127,
    min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,n_estimators=1500,
    n_jobs=20,verbosity=-1,random_state=seed,bagging_seed=seed,feature_fraction_seed=seed)
def boot(c,nb=3000,seed=7):
    c=np.asarray(c,float)
    if len(c)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(c); a=np.array([c[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))
def yr(ts): return (np.asarray(ts,dtype="datetime64[s]").astype("datetime64[Y]").astype(int)+1970)

def main():
    t0=time.time(); p=json.load(open(XP.art("strategy.json"))); cols=p["primary_feats"]
    TR=MX.augment(MX.build_xp(XP.SPL["train"],4),XP.SPL["train"],XP.MODE)
    EV={w:MX.augment(MX.build_xp(XP.SPL[w]),XP.SPL[w],XP.MODE) for w in ("test24","test25","oos")}
    ytr=TR["_y"].astype(int).values; w=magw(TR["_fwd"].astype("float32").values); Xtr=TR[cols].astype("float32")
    rng=np.random.default_rng(0)
    if len(Xtr)>120000:
        idx=np.sort(rng.choice(len(Xtr),120000,replace=False)); Xtr=Xtr.iloc[idx]; ytr=ytr[idx]; w=w[idx]
    print(f"[mw-seedens] train n={len(Xtr):,} K={K} POW={POW} {time.time()-t0:.0f}s",flush=True)
    models=[]
    for s in range(K):
        m=mk(7+s*101); m.fit(Xtr,ytr,sample_weight=w); models.append(m)
        print(f"  seed {s} fit {time.time()-t0:.0f}s",flush=True)
    def prob(D): return np.mean([m.predict_proba(D[cols].astype("float32"))[:,1] for m in models],axis=0)
    EP={w_:(prob(EV[w_]), EV[w_]["_y"].astype(int).values, EV[w_]["_ts"].values.astype("int64"),
            EV[w_]["sess_ny"].values>0.5) for w_ in EV}
    def per_year(side, sv, cov):
        res={}
        for w_,(pr,y,ts,ny) in EP.items():
            conf=np.abs(pr-0.5); gate=ny&((pr>0.5) if sv==1 else (pr<0.5))
            if gate.sum()<20: continue
            cthr=np.quantile(conf[gate],1-cov); m=gate&(conf>=cthr)
            sel=MX.nonoverlap_chrono(ts,m,300)
            if len(sel)>=20:
                cc=(y[sel]==sv).astype(float); lo,hi=boot(cc); Y=2024 if w_=="test24" else 2025 if w_=="test25" else 2026
                res[Y]=dict(win=round(float(cc.mean()),4),n=len(sel),ci=[round(lo,4),round(hi,4)],uprate=round(float((y[sel]==1).mean()),3))
        return res
    out={"test":"DOWN improve: K=4 seed-ensemble magweight POW=0.5, forward per-year","breakeven":BE,"K":K,"covs":COVS}
    cert={}
    for side,sv in (("UP",1),("DOWN",0)):
        out[side]={}
        for cov in COVS:
            py=per_year(side,sv,cov); out[side][f"cov{cov}"]=py
            b=[py[Y]["win"] for Y in py]; lo=[py[Y]["ci"][0] for Y in py]; ns=[py[Y]["n"] for Y in py]
            ok=bool(lo and min(lo)>BE and ns and min(ns)>=100 and len(py)==3)
            cert[f"{side}_cov{cov}"]=dict(binding=min(b) if b else None, binding_ci_lo=min(lo) if lo else None, min_n=min(ns) if ns else 0, all3yr_clears=ok)
    down_ok=any(cert[f"DOWN_cov{c}"]["all3yr_clears"] for c in COVS)
    up_ok=any(cert[f"UP_cov{c}"]["all3yr_clears"] for c in COVS)
    out["cert"]=cert
    out["VERDICT"]=dict(DOWN_forward_certified=down_ok, UP_forward_certified=up_ok,
        statement=(f"DOWN {'FORWARD-CERTIFIED (escalate to CPCV)' if down_ok else 'still fails'}: "
            + "; ".join(f"{c}: bind {cert['DOWN_cov'+str(c)]['binding']} CI-lo {cert['DOWN_cov'+str(c)]['binding_ci_lo']} n{cert['DOWN_cov'+str(c)]['min_n']}" for c in COVS)
            + (". Seed-ensemble lifts the DOWN forward binding CI over breakeven." if down_ok else
               ". Seed-ensembling does NOT lift the forward DOWN binding CI over breakeven — the §3 improve lever is applied-and-failed on DOWN; DOWN genuinely exhausted on-disk.")))
    json.dump(out,open(f"{ROOT}/m5_magweight_seedens_result.json","w"),indent=2,default=str)
    for side in ("UP","DOWN"):
        for cov in COVS:
            c=cert[f"{side}_cov{cov}"]; print(f"  {side} cov{cov}: bind {c['binding']} CI-lo {c['binding_ci_lo']} n{c['min_n']} all3clear={c['all3yr_clears']}  per-year={ {Y:out[side]['cov'+str(cov)][Y]['win'] for Y in out[side]['cov'+str(cov)]} }")
    print(f"VERDICT: {out['VERDICT']['statement']}\n-> m5_magweight_seedens_result.json ({time.time()-t0:.0f}s)")

if __name__=="__main__": main()
