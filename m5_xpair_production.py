"""5-MIN EURUSD binary direction — PRODUCTION pipeline for the SESSION-2 honest best book.

Pipeline = cross-pair/USD-common-factor + base(239) + order-flow PRIMARY (lgb) -> ORTHOGONAL META-LABELER (lgb) that
predicts P(primary correct) from agreement/dispersion/order-flow/confidence (NOT the 239). Abstain unless meta>=thr & NY.
Threshold frozen by WORST-VAL-HALF stability (2022 & 2023 both), NEVER VAL-acc-max (that is the corr(VAL,OOS)=-0.54 trap).

HONEST RESULT (m5_research_log.md iter 8-11), FROZEN pre-committed (worst-VAL-half-stable thr, q0.95): combined held-out
**0.583** CI95[0.570,0.595] (test24 0.607 / test25 0.555 / oos 0.606), EV +0.078/bet @R0.85 — profitable vs the 0.541
breakeven and the best honest 5-min book to date (up from m5_production 0.566). The meta-labeler REACHES ~0.61 at a more
selective (higher-variance, lower-n) operating point, but the stability rule correctly prefers the conservative q0.95 point.
It does NOT reach >=0.65: the 2025 regime caps test25 at ~0.55-0.60 even under a hindsight ORACLE gate (max-floor ~0.60).
The genuine >=0.65 directional edge is seconds-scale (mtick3 3s 0.657/0.667), not 5m.

Settlement = mid-to-mid, ties lose, non-overlap 300s chrono, bootstrap CI95.
Usage: python m5_xpair_production.py train [PAIR] | backtest [PAIR]
"""
import sys, os, json, time, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import m5_xpair as MX
import harness as H

MODELS="/media/sean/CORSAIR/binary-algo/models"
PAIR="EURUSD"; MODE="xpof"
SPL=MX.SPL
def art(n): return f"{MODELS}/m5xp_{PAIR}_{n}"

def meta_feats(F):
    return [c for c in F.columns if c.startswith("agree") or c.startswith("disp") or c=="comp60" or c.startswith("OF_")]

def _Xmeta(F, pr, mcols):
    import numpy as np
    conf=np.abs(pr-0.5).astype("float32")
    return np.column_stack([conf]+[F[c].values.astype("float32") for c in mcols])

def yr(ts): return (np.asarray(ts,dtype="datetime64[s]").astype("datetime64[Y]").astype(int)+1970)

def train(stride=4):
    t0=time.time(); os.makedirs(MODELS,exist_ok=True)
    # ---- PRIMARY ----
    TR=MX.build_xp(SPL["train"],stride); VA=MX.build_xp(SPL["val"])
    xpc=MX.xp_cols(TR)
    TR=MX.augment(TR,SPL["train"],MODE); VA=MX.augment(VA,SPL["val"],MODE)
    cols=MX.feat_cols(MODE,TR,xpc)
    print(f"[prod] primary train={len(TR):,} val={len(VA):,} feats={len(cols)} {time.time()-t0:.0f}s",flush=True)
    ytr=TR["_y"].astype(int).values; yva=VA["_y"].astype(int).values
    P=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=127,min_child_samples=400,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,n_estimators=3000,n_jobs=20,verbosity=-1)
    P.fit(TR[cols].astype("float32"),ytr,eval_set=[(VA[cols].astype("float32"),yva)],eval_metric="auc",
          callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    P.booster_.save_model(art("primary_lgb.txt"))
    pva=P.predict_proba(VA[cols].astype("float32"))[:,1]
    print(f"[prod] primary VAL AUC={roc_auc_score(yva,pva):.4f} {time.time()-t0:.0f}s",flush=True)
    # ---- META on VAL (orthogonal axes), NY rows ----
    mcols=meta_feats(VA); vny=VA["sess_ny"].values>0.5
    Xm=_Xmeta(VA,pva,mcols); ycorr=((pva>0.5).astype(int)==yva).astype(int)
    M=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=15,min_child_samples=1000,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=20,n_estimators=400,n_jobs=20,verbosity=-1)
    M.fit(Xm[vny],ycorr[vny])
    M.booster_.save_model(art("meta_lgb.txt"))
    sval=M.predict_proba(Xm)[:,1]
    # ---- freeze threshold by worst-VAL-half stability ----
    tsv=VA["_ts"].values.astype("int64"); vyr=yr(tsv)
    def half_acc(thr):
        accs=[]
        for yy in sorted(set(vyr.tolist())):
            m=(VA["sess_ny"].values>0.5)&(vyr==yy)&(sval>=thr); sel=MX.nonoverlap_chrono(tsv,m)
            if len(sel)<40: continue
            accs.append(((pva[sel]>0.5).astype(int)==yva[sel]).mean())
        return min(accs) if len(accs)==len(set(vyr.tolist())) else float("nan")
    def val_n(thr):
        m=vny&(sval>=thr); return len(MX.nonoverlap_chrono(tsv,m))
    best=None
    for q in (0.80,0.85,0.88,0.90,0.92,0.94,0.95):
        thr=float(np.quantile(sval[vny],q)); hm=half_acc(thr); nn=val_n(thr)
        if nn<200 or np.isnan(hm): continue
        if best is None or hm>best[1]: best=(thr,hm,nn,q)
    THR,HM,NV,Q=best
    json.dump({"pair":PAIR,"mode":MODE,"primary_feats":cols,"meta_feats":mcols,"meta_thr":THR,"val_q":Q,
        "val_halfmin":HM,"val_n":NV,"gate":"sess_ny & meta>=thr","settlement":"mid-to-mid ties lose breakeven~0.541",
        "splits":{"train":"2012-2021","val":"2022-2023","test":"2024 & 2025","oos":"2026"}},open(art("strategy.json"),"w"),indent=2)
    print(f"[prod] FROZEN meta_thr={THR:.4f} (VAL q{Q:.2f}) worst-half={HM:.3f} val_n={NV} {time.time()-t0:.0f}s",flush=True)
    backtest()

def _load():
    p=json.load(open(art("strategy.json")))
    P=lgb.Booster(model_file=art("primary_lgb.txt")); M=lgb.Booster(model_file=art("meta_lgb.txt"))
    return p,P,M

def backtest():
    p,P,M=_load(); cols=p["primary_feats"]; mcols=p["meta_feats"]; THR=p["meta_thr"]
    print(f"[backtest] cross-pair+OF primary -> orthogonal meta-labeler | gate=NY & meta>={THR:.4f} | non-overlap 300s CI95",flush=True)
    allc=[]
    for w in ("test24","test25","oos"):
        D=MX.build_xp(SPL[w]); D=MX.augment(D,SPL[w],MODE)
        pr=P.predict(D[cols].astype("float32")); y=D["_y"].astype(int).values
        sm=M.predict(_Xmeta(D,pr,mcols)); ts=D["_ts"].values.astype("int64"); ny=D["sess_ny"].values>0.5
        m=ny&(sm>=THR); sel=MX.nonoverlap_chrono(ts,m)
        corr=((pr[sel]>0.5).astype(int)==y[sel]).astype(float) if len(sel) else np.array([])
        acc=corr.mean() if len(sel) else float("nan"); lo,hi=MX.boot(corr)
        print(f"=== {w} === indep_trades={len(sel)} acc={acc:.3f} CI95=[{lo:.3f},{hi:.3f}] AUC={roc_auc_score(y,pr):.4f}",flush=True)
        allc.append(corr)
    A=np.concatenate(allc); acc=A.mean(); lo,hi=MX.boot(A)
    print(f"=== HELD-OUT COMBINED === n={len(A)} acc={acc:.3f} CI95=[{lo:.3f},{hi:.3f}] (breakeven~0.541)",flush=True)
    print("    deriv EV: "+"  ".join(f"R{po:.2f}->EV{acc*po-(1-acc):+.3f}" for po in (0.80,0.85,0.90)),flush=True)

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "backtest"
    if len(sys.argv)>2: PAIR=sys.argv[2]
    {"train":train,"backtest":backtest}.get(mode,backtest)()
