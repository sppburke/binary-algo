"""§3 hyperparameter tuning with REAL OPTUNA (TPE) — DOWN (primary) + UP — of the magweight cross-pair primary.
n_trials capped + logged, selection on WORST-VAL-HALF (per §3/§4 discipline). Supersedes the random-search substitute.

Objective per side = worst-VAL-half selective accuracy at cov0.05 (DOWN: NY&pred<0.5; UP: NY&pred>0.5). Search space:
num_leaves, min_child, reg_lambda, lr, colsample, n_estimators, POW(|ret|-weight). Best config → forward per-year + CI.

PRE-REGISTERED FALSIFIER: DOWN — KILL unless tuned DOWN binding-2025 CI95-lo ≥0.541 (n≥100). UP — KILL unless tuned
UP binding beats the certified floor .553 (CI-lo ≥.541). Expected null both (2025 wall is information/regime, and the
primary is already near-tuned). RUN to confirm.

  ~/binary-algo-venv/bin/python m5_down_optuna2.py
"""
import os, sys; sys.argv=["x"]
import json, time, numpy as np, optuna
import lightgbm as lgb
import m5_xpair as MX, m5_xpair_production as XP
optuna.logging.set_verbosity(optuna.logging.WARNING)
ROOT="/home/sean/git/binary-algo"; BE=0.541; STRIDE=8; COV=0.05; N_DOWN=40; N_UP=30
def boot(c,nb=2500,seed=7):
    c=np.asarray(c,float)
    if len(c)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(c); a=np.array([c[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))
def magw(fwd,pw):
    a=np.abs(fwd).astype(float); med=np.median(a[a>0]) or 1e-9; return np.clip((a/med)**pw,0.1,10.0)

def main():
    t0=time.time(); p=json.load(open(XP.art("strategy.json"))); cols=p["primary_feats"]
    TR=MX.augment(MX.build_xp(XP.SPL["train"],STRIDE),XP.SPL["train"],XP.MODE)
    VA=MX.augment(MX.build_xp(XP.SPL["val"]),XP.SPL["val"],XP.MODE)
    EV={w:MX.augment(MX.build_xp(XP.SPL[w]),XP.SPL[w],XP.MODE) for w in ("test24","test25","oos")}
    Xtr=TR[cols].astype("float32").to_numpy(); ytr=TR["_y"].astype(int).values; fwtr=TR["_fwd"].astype("float32").values
    Xva=VA[cols].astype("float32").to_numpy(); yva=VA["_y"].astype(int).values
    tsv=VA["_ts"].values.astype("int64"); nyv=VA["sess_ny"].values>0.5; half=np.median(tsv); h1=tsv<half; h2=~h1
    print(f"[optuna] train={len(Xtr):,} stride={STRIDE} {time.time()-t0:.0f}s",flush=True)
    def wh(pr,sv):
        conf=np.abs(pr-0.5); accs=[]
        for h in (h1,h2):
            g=h&nyv&((pr>0.5) if sv==1 else (pr<0.5))
            if g.sum()<40: return -1
            cthr=np.quantile(conf[g],1-COV); sel=g&(conf>=cthr)
            if sel.sum()<15: return -1
            accs.append((yva[sel]==sv).mean())
        return float(min(accs))
    def make_obj(sv):
        def obj(tr):
            hp=dict(num_leaves=tr.suggest_categorical("num_leaves",[63,127,255,350]),
                    min_child_samples=tr.suggest_categorical("min_child_samples",[200,300,500,800]),
                    reg_lambda=tr.suggest_float("reg_lambda",5.0,50.0,log=True),
                    learning_rate=tr.suggest_float("learning_rate",0.015,0.06,log=True),
                    colsample_bytree=tr.suggest_float("colsample_bytree",0.4,0.7),
                    n_estimators=tr.suggest_categorical("n_estimators",[500,800]))
            pw=tr.suggest_categorical("POW",[0.25,0.5,0.75])
            m=lgb.LGBMClassifier(objective="binary",subsample=0.8,subsample_freq=1,n_jobs=20,verbosity=-1,**hp)
            m.fit(Xtr,ytr,sample_weight=magw(fwtr,pw))
            return wh(m.predict_proba(Xva)[:,1],sv)
        return obj
    def fwd_py(hp,pw,sv):
        m=lgb.LGBMClassifier(objective="binary",subsample=0.8,subsample_freq=1,n_jobs=20,verbosity=-1,**hp)
        m.fit(Xtr,ytr,sample_weight=magw(fwtr,pw)); res={}
        for w,D in EV.items():
            pr=m.predict_proba(D[cols].astype("float32"))[:,1]; y=D["_y"].astype(int).values
            ts=D["_ts"].values.astype("int64"); ny=D["sess_ny"].values>0.5; conf=np.abs(pr-0.5)
            g=ny&((pr>0.5) if sv==1 else (pr<0.5))
            if g.sum()<20: continue
            cthr=np.quantile(conf[g],1-COV); mk=g&(conf>=cthr); sel=MX.nonoverlap_chrono(ts,mk,300)
            if len(sel)>=20:
                cc=(y[sel]==sv).astype(float); lo,hi=boot(cc); Y=2024 if w=="test24" else 2025 if w=="test25" else 2026
                res[Y]=dict(win=round(float(cc.mean()),4),n=len(sel),ci=[round(lo,4),round(hi,4)])
        return res
    out={"test":"OPTUNA TPE tuning (worst-VAL-half) magweight cross-pair primary — DOWN + UP","breakeven":BE,
         "cov":COV,"stride":STRIDE,"optuna":optuna.__version__}
    res={}
    for side,sv,nt in (("DOWN",0,N_DOWN),("UP",1,N_UP)):
        st=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=7))
        st.optimize(make_obj(sv),n_trials=nt,show_progress_bar=False)
        bp=st.best_params; pw=bp.pop("POW"); py=fwd_py(bp,pw,sv)
        b=[py[Y]["win"] for Y in py]; lo=[py[Y]["ci"][0] for Y in py]; ns=[py[Y]["n"] for Y in py]
        bind=min(b) if b else None; bind_lo=min(lo) if lo else None
        thr=BE if side=="DOWN" else 0.553
        ok=bool(bind_lo is not None and bind_lo>BE and ns and min(ns)>=100 and len(py)==3 and (bind or 0)>=thr)
        res[side]=dict(best_wh=round(st.best_value,4),best_params={**bp,"POW":pw},n_trials=nt,forward_per_year=py,
                       binding=dict(win=bind,ci_lo=bind_lo,min_n=min(ns) if ns else 0),beats_bar=ok)
        print(f"[optuna] {side}: best wh={st.best_value:.4f} POW={pw} -> forward {py} binding {bind} CI-lo {bind_lo} beats={ok} ({time.time()-t0:.0f}s)",flush=True)
    out["results"]=res
    out["VERDICT"]=dict(DOWN_certified=res["DOWN"]["beats_bar"], UP_beats_floor=res["UP"]["beats_bar"],
        statement=(f"OPTUNA tuning: DOWN {'CERTIFIES' if res['DOWN']['beats_bar'] else 'still fails'} "
            f"(binding {res['DOWN']['binding']}); UP {'BEATS floor .553' if res['UP']['beats_bar'] else 'does not beat floor .553'} "
            f"(binding {res['UP']['binding']}). "
            + ("" if (res['DOWN']['beats_bar'] or res['UP']['beats_bar']) else
               "Real-optuna TPE (70 trials) does NOT lift DOWN over breakeven nor UP over the .553 floor — hyperparameter tuning is not the binding constraint (the 2025 wall is information/regime). §3 tuning lever RUN+failed.")))
    json.dump(out,open(f"{ROOT}/m5_down_optuna2_result.json","w"),indent=2,default=str)
    print(f"VERDICT: {out['VERDICT']['statement']}\n-> m5_down_optuna2_result.json ({time.time()-t0:.0f}s)")

if __name__=="__main__": main()
