"""DOWN §3 lever RUN LITERALLY (per user): hyperparameter TUNING (Optuna-substitute random search, n_trials capped
+ logged, select on worst-VAL-half) of the magweight-DOWN cross-pair primary. Optuna pkg absent in the uv venv, so
this is a faithful random search over the same hyperparameter space with worst-VAL-half selection (the lever's intent).

Tune (num_leaves, min_child, reg_lambda, lr, colsample, POW) to MAXIMIZE worst-VAL-half DOWN selective accuracy;
take the best config, retrain, evaluate forward DOWN per-year + binding-2025 CI.

PRE-REGISTERED FALSIFIER: KILL unless the tuned config lifts DOWN binding-2025 moved-acc CI95-lo ≥0.541 at n≥100.
Expected null (the 2025 wall is information/regime, not hyperparams; walk-forward was +.015 'often worse'). RUN to confirm.

  ~/binary-algo-venv/bin/python m5_down_optuna.py
"""
import os, sys; sys.argv=["x"]
import json, time, numpy as np
import lightgbm as lgb
import m5_xpair as MX, m5_xpair_production as XP
ROOT="/media/sean/CORSAIR/binary-algo"; BE=0.541; N_TRIALS=20; STRIDE=8; COV=0.05
def boot(c,nb=2500,seed=7):
    c=np.asarray(c,float)
    if len(c)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(c); a=np.array([c[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))
def yr(ts): return (np.asarray(ts,dtype="datetime64[s]").astype("datetime64[Y]").astype(int)+1970)
def magw(fwd,pw):
    a=np.abs(fwd).astype(float); med=np.median(a[a>0]) or 1e-9; return np.clip((a/med)**pw,0.1,10.0)

def main():
    t0=time.time(); rng=np.random.default_rng(12345)
    p=json.load(open(XP.art("strategy.json"))); cols=p["primary_feats"]
    TR=MX.augment(MX.build_xp(XP.SPL["train"],STRIDE),XP.SPL["train"],XP.MODE)
    VA=MX.augment(MX.build_xp(XP.SPL["val"]),XP.SPL["val"],XP.MODE)
    EV={w:MX.augment(MX.build_xp(XP.SPL[w]),XP.SPL[w],XP.MODE) for w in ("test24","test25","oos")}
    Xtr=TR[cols].astype("float32").to_numpy(); ytr=TR["_y"].astype(int).values; fwtr=TR["_fwd"].astype("float32").values
    Xva=VA[cols].astype("float32").to_numpy(); yva=VA["_y"].astype(int).values; tsv=VA["_ts"].values.astype("int64"); nyv=VA["sess_ny"].values>0.5
    half=np.median(tsv); h1=tsv<half; h2=~h1
    print(f"[down-tune] train={len(Xtr):,} (stride {STRIDE}) trials={N_TRIALS} {time.time()-t0:.0f}s",flush=True)
    def down_wh(pr):  # worst-VAL-half DOWN selective acc at COV
        conf=np.abs(pr-0.5); accs=[]
        for h in (h1,h2):
            g=h&nyv&(pr<0.5)
            if g.sum()<40: return -1
            cthr=np.quantile(conf[g],1-COV); sel=g&(conf>=cthr)
            if sel.sum()<15: return -1
            accs.append((yva[sel]==0).mean())
        return min(accs)
    trials=[]; best=None
    for i in range(N_TRIALS):
        hp=dict(num_leaves=int(rng.choice([63,127,255,350])),min_child_samples=int(rng.choice([200,300,500,800])),
                reg_lambda=float(rng.choice([8,10,20,40])),learning_rate=float(rng.choice([0.02,0.03,0.05])),
                colsample_bytree=float(rng.choice([0.4,0.5,0.6])),n_estimators=int(rng.choice([500,800])))
        pw=float(rng.choice([0.25,0.5,0.75]))
        m=lgb.LGBMClassifier(objective="binary",subsample=0.8,subsample_freq=1,n_jobs=20,verbosity=-1,**hp)
        m.fit(Xtr,ytr,sample_weight=magw(fwtr,pw))
        pr=m.predict_proba(Xva)[:,1]; wh=down_wh(pr)
        trials.append({"trial":i,"POW":pw,**hp,"down_worsthalf":round(wh,4)})
        if best is None or wh>best[0]: best=(wh,hp,pw,m)
        print(f"  trial {i}: POW={pw} nl={hp['num_leaves']} wh-DOWN={wh:.4f} {time.time()-t0:.0f}s",flush=True)
    whb,hpb,pwb,mb=best
    # forward per-year DOWN at COV with the best config
    def fwd_py(model):
        res={}
        for w,D in EV.items():
            pr=model.predict_proba(D[cols].astype("float32"))[:,1]; y=D["_y"].astype(int).values
            ts=D["_ts"].values.astype("int64"); ny=D["sess_ny"].values>0.5; conf=np.abs(pr-0.5)
            g=ny&(pr<0.5);
            if g.sum()<20: continue
            cthr=np.quantile(conf[g],1-COV); m=g&(conf>=cthr); sel=MX.nonoverlap_chrono(ts,m,300)
            if len(sel)>=20:
                cc=(y[sel]==0).astype(float); lo,hi=boot(cc); Y=2024 if w=="test24" else 2025 if w=="test25" else 2026
                res[Y]=dict(win=round(float(cc.mean()),4),n=len(sel),ci=[round(lo,4),round(hi,4)])
        return res
    py=fwd_py(mb); b=[py[Y]["win"] for Y in py]; lo=[py[Y]["ci"][0] for Y in py]; ns=[py[Y]["n"] for Y in py]
    bind=min(b) if b else None; bind_lo=min(lo) if lo else None
    survive=bool(bind_lo is not None and bind_lo>BE and ns and min(ns)>=100 and len(py)==3)
    out={"test":"DOWN hyperparameter tuning (random search, worst-VAL-half) of magweight cross-pair primary",
         "breakeven":BE,"n_trials":N_TRIALS,"cov":COV,"trials":trials,
         "best":{"down_worsthalf":round(whb,4),"POW":pwb,**hpb},"forward_per_year":py,
         "binding":dict(win=bind,ci_lo=bind_lo,min_n=min(ns) if ns else 0),
         "VERDICT":dict(SURVIVES=survive,
            statement=(f"DOWN tuning {'SURVIVES' if survive else 'KILLED'}: best worst-VAL-half DOWN {whb:.4f} (POW={pwb}), "
                f"forward binding {bind} (CI-lo {bind_lo}) per-year {py}. "
                + ("" if survive else "Hyperparameter tuning does NOT lift DOWN binding-2025 CI over breakeven — the 2025 wall is information/regime, not model-capacity. §3 tuning lever RUN+failed on DOWN.")))}
    json.dump(out,open(f"{ROOT}/m5_down_optuna_result.json","w"),indent=2,default=str)
    print(f"[down-tune] best wh-DOWN={whb:.4f} POW={pwb} {hpb}")
    print(f"[down-tune] forward DOWN per-year: {py}  binding {bind} CI-lo {bind_lo}")
    print(f"VERDICT: {out['VERDICT']['statement']}\n-> m5_down_optuna_result.json ({time.time()-t0:.0f}s)")

if __name__=="__main__": main()
