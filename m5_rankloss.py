"""N12 — PAIRWISE SIGN-RANKING LOSS (LambdaRank) for the cross-pair direction primary vs BCE (the incumbent).

MECHANISM (sign-aware): the deriv gate bets on the TOP-confidence UP (and bottom = DOWN) bars, so what matters is
the ORDERING of P(up) at the tail. BCE optimizes full-sample log-loss; a pairwise ranking loss (LambdaRank,
top-weighted) optimizes the ordering directly — it MAY sharpen the high-confidence tail the gate uses. Retrain the
SAME cross-pair feature set with objective=lambdarank; gate NY + top-cov by RANK score; compare UP & DOWN
selective accuracy per-year to the incumbent BCE primary at the same gate+cover.

PRE-REGISTERED FALSIFIER: KILL unless the ranker beats the BCE primary on the binding (worst) held-out year for at
least one side at matched coverage, with binding CI95-lo>0.541. Incumbent: m5xp UP refit floor .553; DOWN .5441.
(Honest prior: BCE already early-stops on AUC = an ordering metric, and seed-ensemble (ordering/variance) + magweight
(|ret|-loss) already failed to lift the floor — so this is largely subsumed; RUN to confirm, not infer.)

  ~/binary-algo-venv/bin/python m5_rankloss.py
"""
import os, sys; sys.argv=["x"]
import json, time, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import m5_xpair as MX, m5_xpair_production as XP
ROOT="/media/sean/CORSAIR/binary-algo"; BE=0.541; COV=0.10
def boot(c,nb=2000,seed=7):
    c=np.asarray(c,float)
    if len(c)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(c); a=np.array([c[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))
def yr(ts): return (np.asarray(ts,dtype="datetime64[s]").astype("datetime64[Y]").astype(int)+1970)

def main():
    t0=time.time()
    p0=json.load(open(XP.art("strategy.json"))); cols=p0["primary_feats"]
    P_bce=lgb.Booster(model_file=XP.art("primary_lgb.txt"))     # incumbent BCE primary (baseline)
    TR=MX.augment(MX.build_xp(XP.SPL["train"],4),XP.SPL["train"],XP.MODE)
    VA=MX.augment(MX.build_xp(XP.SPL["val"]),XP.SPL["val"],XP.MODE)
    EV={w:MX.augment(MX.build_xp(XP.SPL[w]),XP.SPL[w],XP.MODE) for w in ("test24","test25","oos")}
    ytr=TR["_y"].astype(int).values
    print(f"[rankloss] train={len(TR):,} feats={len(cols)} {time.time()-t0:.0f}s",flush=True)
    R=lgb.LGBMRanker(objective="lambdarank",n_estimators=1500,learning_rate=0.02,num_leaves=127,
        min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,n_jobs=20,verbosity=-1)
    yva=VA["_y"].astype(int).values
    R.fit(TR[cols].astype("float32"),ytr,group=[len(TR)],
          eval_set=[(VA[cols].astype("float32"),yva)],eval_group=[len(VA)],eval_at=[max(1,int(0.1*len(VA)))],
          callbacks=[lgb.early_stopping(120),lgb.log_evaluation(0)])
    print(f"[rankloss] ranker trained {time.time()-t0:.0f}s",flush=True)
    # gate: NY; UP=top-cov by score, DOWN=bottom-cov by score. Threshold frozen on VAL.
    def scores(model,D,kind):
        X=D[cols].astype("float32")
        return (model.predict(X) if kind=="rank" else model.predict(X))  # both return a score; BCE booster -> prob
    # VAL thresholds (frozen)
    sV_r=R.predict(VA[cols].astype("float32")); sV_b=P_bce.predict(VA[cols].astype("float32"))
    nyV=VA["sess_ny"].values>0.5
    thr_r_up=np.quantile(sV_r[nyV],1-COV); thr_r_dn=np.quantile(sV_r[nyV],COV)
    thr_b_up=np.quantile(sV_b[nyV],1-COV); thr_b_dn=np.quantile(sV_b[nyV],COV)
    def per_year(model,kind,thr_up,thr_dn):
        res={"UP":{}, "DOWN":{}}
        for w,D in EV.items():
            s=model.predict(D[cols].astype("float32")); y=D["_y"].astype(int).values
            ts=D["_ts"].values.astype("int64"); ny=D["sess_ny"].values>0.5; Y=2024 if w=="test24" else 2025 if w=="test25" else 2026
            for side,sel_mask,correct in (("UP", ny&(s>=thr_up), (y==1)),("DOWN", ny&(s<=thr_dn), (y==0))):
                sel=MX.nonoverlap_chrono(ts,sel_mask,300)
                if len(sel)>=20:
                    cc=correct[sel].astype(float); lo,hi=boot(cc); res[side][Y]=dict(win=round(float(cc.mean()),4),n=len(sel),ci=[round(lo,4),round(hi,4)])
        return res
    rk=per_year(R,"rank",thr_r_up,thr_r_dn); bc=per_year(P_bce,"bce",thr_b_up,thr_b_dn)
    def binding(res,side):
        ys=[res[side][Y]["win"] for Y in res[side]]; lo=[res[side][Y]["ci"][0] for Y in res[side]]
        return (min(ys) if ys else None, min(lo) if lo else None)
    out={"test":"N12 LambdaRank pairwise-ranking-loss primary vs BCE","breakeven":BE,"cov":COV,
         "ranker":rk,"bce_baseline":bc}
    verdict={}
    for side in ("UP","DOWN"):
        rb,rlo=binding(rk,side); bb,blo=binding(bc,side)
        beats=bool(rb is not None and bb is not None and rb>bb+1e-4 and rlo is not None and rlo>BE)
        verdict[side]=dict(ranker_binding=rb,ranker_ci_lo=rlo,bce_binding=bb,beats_bce_and_clears=beats)
    survive=any(verdict[s]["beats_bce_and_clears"] for s in verdict)
    out["VERDICT"]=dict(per_side=verdict, SURVIVES=survive,
        statement=(f"N12 {'SURVIVES' if survive else 'KILLED'}: "
            f"UP ranker {verdict['UP']['ranker_binding']}(CI-lo {verdict['UP']['ranker_ci_lo']}) vs BCE {verdict['UP']['bce_binding']}; "
            f"DOWN ranker {verdict['DOWN']['ranker_binding']}(CI-lo {verdict['DOWN']['ranker_ci_lo']}) vs BCE {verdict['DOWN']['bce_binding']}. "
            + ("" if survive else "Pairwise ranking loss does NOT beat BCE at the gated tail on either side — confirms BCE+AUC-early-stop already optimizes the ordering (joins seed-ens null + magweight). Empirically subsumes the loss-reoptimization lever.")))
    json.dump(out,open(f"{ROOT}/m5_rankloss_result.json","w"),indent=2,default=str)
    print(f"[rankloss] UP ranker {rk['UP']} \n           BCE {bc['UP']}")
    print(f"[rankloss] DOWN ranker {rk['DOWN']} \n           BCE {bc['DOWN']}")
    print(f"VERDICT: {out['VERDICT']['statement']}\n-> m5_rankloss_result.json ({time.time()-t0:.0f}s)")

if __name__=="__main__": main()
