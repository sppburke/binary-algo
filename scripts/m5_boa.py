"""E1 — ONLINE performance-weighted (Hedge/BOA) ENSEMBLE over the 4 frozen direction books.

Books (all predict DIRECTION; applied to the 5-min outcome like the cross-horizon stack): m5xp (cross-pair
primary), m15, m10, m5 (each a 3-model lgb+xgb+cat ensemble). SIGN comes only from the books; online weighting
is regime-adaptation (sign-aware → theorem-safe). HYPOTHESIS: re-weighting books by RECENT performance adapts to
the 2026 regime shift and beats the static equal-weight blend (which tied m5xp) + the incumbent m5xp-alone.
HONEST RISK: corr(VAL,OOS)=−0.54 ⇒ recent performance can ANTI-predict; performance-chasing may HURT.

NO LOOK-AHEAD: online weights update over the NON-OVERLAPPING NY-gated trade sequence (gap 300s) — trade k's
outcome resolves before trade k+1's entry, so the weight at k uses only resolved past outcomes.

PRE-REGISTERED FALSIFIER: KILL E1 unless an online variant BEATS BOTH (a) the static equal-weight blend AND
(b) incumbent m5xp-alone on the binding (worst) held-out year at matched coverage, with binding-year CI95-lo>0.541.
Incumbent: m5xp UP refit floor 0.553 / forward binding-2025 0.577.

  ~/binary-algo-venv/bin/python m5_boa.py
"""
import os, sys; sys.argv=["x"]
import json, time, numpy as np
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
import harness as H, m5_xpair as MX
from m5_xpair_production import _Xmeta, art as art5
MODELS="/home/sean/git/binary-algo/models"; ROOT="/home/sean/git/binary-algo"; PAIR="EURUSD"; BE=0.541
base=list(H.feature_cols("EURUSD"))
SPL={"val":["2022","2023"],"test24":["2024"],"test25":["2025"],"oos":["2026"]}

def a(tag,n): return f"{MODELS}/{tag}_{PAIR}_{n}"
def load_ens(tag):
    L=lgb.Booster(model_file=a(tag,"direction_lgb.txt"))
    G=xgb.XGBClassifier(); G.load_model(a(tag,"direction_xgb.json"))
    C=CatBoostClassifier(); C.load_model(a(tag,"direction_cat.cbm"))
    return ("ens",L,G,C)
def load_m5xp():
    p=json.load(open(art5("strategy.json"))); P=lgb.Booster(model_file=art5("primary_lgb.txt")); return ("xp",p,P)
def predict(book,D):
    if book[0]=="ens":
        _,L,G,C=book; X=D[base].astype("float32")
        return (L.predict(X.values)+G.predict_proba(X)[:,1]+C.predict_proba(X.fillna(-999))[:,1])/3.0
    else:
        _,p,P=book; return P.predict(D[p["primary_feats"]].astype("float32"))
def boot(c,nb=2000,seed=7):
    c=np.asarray(c,float)
    if len(c)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(c); a_=np.array([c[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a_,2.5)),float(np.percentile(a_,97.5))
def yr(ts): return (np.asarray(ts,dtype="datetime64[s]").astype("datetime64[Y]").astype(int)+1970)

def main():
    t0=time.time()
    books={"m5xp":load_m5xp(),"m15":load_ens("m15"),"m10":load_ens("m10"),"m5":load_ens("m5")}
    order=["m5xp","m15","m10","m5"]; K=len(order)
    print(f"[boa] loaded {K} books {time.time()-t0:.0f}s",flush=True)
    # aligned per-bar streams, NY-gated, non-overlapping, chronological across val->test24->test25->oos
    seqs=[]
    for w in ("val","test24","test25","oos"):
        D=MX.build_xp(SPL[w]); D=MX.augment(D,SPL[w],"xpof")
        P={k:predict(books[k],D) for k in order}
        y=D["_y"].astype(int).values; ts=D["_ts"].values.astype("int64"); ny=D["sess_ny"].values>0.5
        sel=MX.nonoverlap_chrono(ts,ny,300)            # independent NY trades, resolved-before-next
        seqs.append(dict(w=w,ts=ts[sel],y=y[sel],P={k:P[k][sel] for k in order}))
        print(f"  [{w}] indep NY trades={len(sel)}",flush=True); del D
    ts=np.concatenate([s["ts"] for s in seqs]); y=np.concatenate([s["y"] for s in seqs])
    Pm=np.column_stack([np.concatenate([s["P"][k] for s in seqs]) for k in order])  # (T,K) book up-probs
    o=np.argsort(ts,kind="stable"); ts=ts[o]; y=y[o]; Pm=Pm[o]; T=len(ts); years=yr(ts)
    warm=years<=2024                                   # warm online weights on val(2022-23)+test24; report 24/25/26

    # ---- aggregators ----
    def static_equal(): return Pm.mean(1)
    def m5xp_alone(): return Pm[:,0]
    def ewa(eta):
        w=np.ones(K)/K; agg=np.empty(T)
        for t in range(T):
            w=w/w.sum(); agg[t]=float(w@Pm[t])
            loss=(Pm[t]-y[t])**2                        # square loss; outcome resolved before next trade
            w=w*np.exp(-eta*loss)
        return agg
    aggs={"static_equal":static_equal(),"m5xp_alone":m5xp_alone()}
    for eta in (0.25,0.5,1.0,2.0,4.0): aggs[f"ewa_eta{eta}"]=ewa(eta)

    COV=0.10
    def per_year(agg):
        # UP side: pred up = agg>0.5; among NY trades pick top-COV by confidence; per-year win
        conf=np.abs(agg-0.5); up=agg>0.5
        res={}
        # threshold from warmup confidence (held-out years use the same fixed thr)
        thr=np.quantile(conf[warm & up], 1-COV) if (warm&up).sum()>50 else np.quantile(conf[up],1-COV)
        selsel=up&(conf>=thr)
        corr=((agg>0.5).astype(int)==y)
        for Y in (2024,2025,2026):
            m=selsel&(years==Y)
            if m.sum()>=20:
                cc=corr[m].astype(float); lo,hi=boot(cc); res[Y]=dict(win=round(float(cc.mean()),4),n=int(m.sum()),ci=[round(lo,4),round(hi,4)])
        hh=selsel&(years>=2025)   # held-out (25+26)
        res["heldout_2526"]=dict(win=round(float(corr[hh].mean()),4),n=int(hh.sum())) if hh.sum() else {}
        bind=[res[Y]["win"] for Y in (2024,2025,2026) if Y in res]; blo=[res[Y]["ci"][0] for Y in (2024,2025,2026) if Y in res]
        res["binding"]=dict(win=round(min(bind),4) if bind else None, ci_lo=round(min(blo),4) if blo else None)
        return res
    results={k:per_year(v) for k,v in aggs.items()}
    for k in aggs: print(f"  {k:16s} binding={results[k]['binding']}  2025={results[k].get(2025)}",flush=True)

    base_bind=results["m5xp_alone"]["binding"]["win"]; stat_bind=results["static_equal"]["binding"]["win"]
    online=[k for k in aggs if k.startswith("ewa")]
    best_online=max(online,key=lambda k:(results[k]["binding"]["win"] or 0))
    bo=results[best_online]["binding"]
    beats=(bo["win"] is not None and base_bind is not None and bo["win"]>base_bind+1e-4
           and stat_bind is not None and bo["win"]>stat_bind+1e-4)
    clears=(bo["ci_lo"] is not None and bo["ci_lo"]>BE)
    survive=bool(beats and clears)
    out=dict(test="E1 online BOA/Hedge ensemble over 4 frozen books", breakeven=BE, books=order, cov=COV,
        per_year={k:results[k] for k in aggs},
        m5xp_alone_binding=base_bind, static_equal_binding=stat_bind,
        best_online=best_online, best_online_binding=bo,
        VERDICT=dict(beats_static_and_incumbent=beats, binding_ci_clears_breakeven=clears, SURVIVES=survive,
            statement=("E1 SURVIVES: online ensemble beats static blend + m5xp-alone on the binding year, CI clears breakeven."
                if survive else
                f"E1 KILLED: best online {best_online} binding {bo['win']} (CI-lo {bo['ci_lo']}) does NOT beat both "
                f"m5xp-alone {base_bind} and static-equal {stat_bind} on the binding year / clear breakeven. "
                f"Online performance-weighting does not fix the 2026 collapse (consistent with corr(VAL,OOS)=−0.54: "
                f"recent-performance chasing anti-transfers).")))
    json.dump(out,open(f"{ROOT}/m5_boa_result.json","w"),indent=2,default=str)
    print(f"\n[E1] m5xp-alone binding={base_bind}  static-equal binding={stat_bind}  best-online {best_online} binding={bo['win']} (CI-lo {bo['ci_lo']})")
    print(f"VERDICT: {out['VERDICT']['statement']}")
    print(f"-> m5_boa_result.json  ({time.time()-t0:.0f}s)")

if __name__=="__main__": main()
