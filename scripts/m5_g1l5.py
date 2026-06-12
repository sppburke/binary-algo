"""G1 + L5 — cheap gate re-scoring / re-selection on the m5xp UP stream (no retrain).

G1 — VOL-NORMALIZED CONFORMITY GATE: score the book's confidence by conf/(vol+eps) instead of raw conf, so the
  gate is regime-stationary (the cheap static analogue of a vol-normalized ACI conformity score; if the STATIC
  vol-normalized score doesn't lift the floor, the adaptive ACI version won't either, and we skip the heavy
  nested-refit re-run). Sign from the direction model; vol only reshapes the gate (theorem-safe).
L5 — MADL AS THE SELECTION METRIC: pick the gate threshold on worst-VAL-half by maximizing the |ret|-weighted
  directional PnL (Mean Absolute Directional Loss objective) instead of accuracy, then evaluate win-rate per year.
  (For deriv the per-trade payoff is FIXED R, so |ret|-weighting may pick a worse-winrate gate — that's the test.)

Baseline = accuracy-selected raw-confidence cover (the incumbent operating rule).
PRE-REGISTERED FALSIFIER: KILL G1/L5 unless it BEATS the raw-conf accuracy-selected baseline on the binding (worst)
  held-out year at matched coverage AND binding-year CI95-lo>0.541. Incumbent: m5xp UP refit floor 0.553 / fwd 0.577.

  ~/binary-algo-venv/bin/python m5_g1l5.py
"""
import os, sys; sys.argv=["x"]
import json, time, numpy as np
import m5_xpair as MX, m5_xpair_production as XP
ROOT="/home/sean/git/binary-algo"; BE=0.541; COV=0.10
def boot(c,nb=2000,seed=7):
    c=np.asarray(c,float)
    if len(c)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(c); a=np.array([c[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))
def yr(ts): return (np.asarray(ts,dtype="datetime64[s]").astype("datetime64[Y]").astype(int)+1970)

def main():
    t0=time.time(); p,P,M=XP._load(); cols=p["primary_feats"]; mcols=p["meta_feats"]; THR=p["meta_thr"]
    S={}
    for w in ("val","test24","test25","oos"):
        D=MX.build_xp(XP.SPL[w]); D=MX.augment(D,XP.SPL[w],XP.MODE)
        pr=P.predict(D[cols].astype("float32")); sm=M.predict(XP._Xmeta(D,pr,mcols))
        y=D["_y"].astype(int).values; ts=D["_ts"].values.astype("int64"); ny=D["sess_ny"].values>0.5
        fwd=np.abs(D["_fwd"].values.astype(float)); vol=D["5m_rv_24"].values.astype(float)
        g=ny&(sm>=THR)&(pr>0.5)                       # UP bars in book gate
        sel=MX.nonoverlap_chrono(ts,g,300)
        S[w]=dict(ts=ts[sel],conf=np.abs(pr[sel]-0.5),corr=((pr[sel]>0.5).astype(int)==y[sel]).astype(float),
                  aret=fwd[sel],vol=vol[sel])
        print(f"  [{w}] UP gated indep={len(sel)}",flush=True); del D
    print(f"[g1l5] streams built {time.time()-t0:.0f}s",flush=True)

    EV={k:np.concatenate([S[w][k] for w in ("test24","test25","oos")]) for k in ("ts","conf","corr","aret","vol")}
    V=S["val"]; eps=1e-9
    # ex-ante vol RANK (trailing 5m_rv_24 is known at trade time); NaN/inf -> median
    def volrank(vol):
        v=np.where(np.isfinite(vol),vol,np.nan)
        med=np.nanmedian(v); v=np.where(np.isfinite(v),v,med)
        r=np.argsort(np.argsort(v))/max(len(v)-1,1); return r          # in [0,1]
    Vvr=volrank(V["vol"]); Evr=volrank(EV["vol"])
    # EX-ANTE scores ONLY (no forward |ret| in any test-time score):
    scores={
      "baseline_rawconf": (V["conf"],               EV["conf"]),                       # incumbent rule
      "G1_volnorm":       (V["conf"]/(Vvr+0.1),     EV["conf"]/(Evr+0.1)),             # conf down-weighted in high-vol
    }
    def per_year(scoreE, thr):
        sel=scoreE>=thr; ys=yr(EV["ts"][sel]); res={}
        for Y in (2024,2025,2026):
            m=ys==Y
            if m.sum()>=20:
                cc=EV["corr"][sel][m]; lo,hi=boot(cc); res[Y]=dict(win=round(float(cc.mean()),4),n=int(m.sum()),ci=[round(lo,4),round(hi,4)])
        res["pooled"]=dict(win=round(float(EV["corr"][sel].mean()),4),n=int(sel.sum()))
        b=[res[Y]["win"] for Y in (2024,2025,2026) if Y in res]; lo=[res[Y]["ci"][0] for Y in (2024,2025,2026) if Y in res]
        res["binding"]=dict(win=round(min(b),4) if b else None, ci_lo=round(min(lo),4) if lo else None); return res
    halfmed=np.median(V["ts"]); h1=V["ts"]<halfmed; h2=~h1
    def worsthalf(scoreV, thr, metric):   # metric: 'acc' or 'madl' (|ret|-weighted PnL) — uses VAL outcomes (legit for tuning)
        vals=[]
        for h in (h1,h2):
            s=scoreV[h]; c=V["corr"][h]; ar=V["aret"][h]; sel=s>=thr
            if h.sum()<40 or sel.sum()<15: return -1e9
            vals.append(c[sel].mean() if metric=="acc" else ((2*c[sel]-1)*ar[sel]).mean())
        return min(vals)

    out={"test":"G1 vol-normalized conformity + L5 MADL-selection on m5xp UP (ex-ante, leakage-fixed)","breakeven":BE}
    # --- baseline + G1: gate by ex-ante score, cov picked by VAL worst-half ACCURACY ---
    for name,(sV,sE) in scores.items():
        covg=[0.05,0.10,0.15,0.20,0.30]
        best=max(covg,key=lambda cov: worsthalf(sV,np.quantile(sV,1-cov),"acc"))
        thr=float(np.quantile(sV,1-best))      # FROZEN from VAL, applied to test
        out[name]=dict(cov_selected=best, worsthalf_acc=round(worsthalf(sV,thr,"acc"),4), per_year=per_year(sE,thr))
    # --- L5: SAME ex-ante conf gate, but cov picked by VAL worst-half MADL (|ret|-weighted) instead of accuracy ---
    sV,sE=V["conf"],EV["conf"]; covg=[0.05,0.10,0.15,0.20,0.30]
    best_madl=max(covg,key=lambda cov: worsthalf(sV,np.quantile(sV,1-cov),"madl"))
    thrL=float(np.quantile(sV,1-best_madl))
    out["L5_madl_select"]=dict(cov_selected=best_madl, worsthalf_madl=round(worsthalf(sV,thrL,"madl"),8), per_year=per_year(sE,thrL))
    base_py=out["baseline_rawconf"]["per_year"]; g1_py=out["G1_volnorm"]["per_year"]; l5_py=out["L5_madl_select"]["per_year"]

    bb=base_py["binding"]["win"]; bclo=base_py["binding"]["ci_lo"]
    def verdict(name,py):
        w=py["binding"]["win"]; lo=py["binding"]["ci_lo"]
        beats=(w is not None and bb is not None and w>bb+1e-4); clears=(lo is not None and lo>BE)
        return dict(binding=w,ci_lo=lo,beats_baseline=beats,clears_breakeven=clears,SURVIVES=bool(beats and clears))
    out["VERDICT"]=dict(baseline_binding=dict(win=bb,ci_lo=bclo),
        G1=verdict("G1",g1_py), L5=verdict("L5",l5_py),
        statement=(f"baseline(rawconf,acc-select) binding={bb} (CI-lo {bclo}). "
            f"G1(vol-norm) binding={g1_py['binding']['win']} (CI-lo {g1_py['binding']['ci_lo']}). "
            f"L5(MADL-select) binding={l5_py['binding']['win']} (CI-lo {l5_py['binding']['ci_lo']}). "
            "KILLED unless a variant beats baseline binding + clears breakeven CI."))
    json.dump(out,open(f"{ROOT}/m5_g1l5_result.json","w"),indent=2,default=str)
    print("\n[G1/L5] cov%.2f"%COV)
    print(f"  baseline rawconf: binding={bb} CI-lo={bclo}  per-year={ {Y:base_py[Y]['win'] for Y in (2024,2025,2026) if Y in base_py} }")
    print(f"  G1 vol-norm:      binding={g1_py['binding']['win']} CI-lo={g1_py['binding']['ci_lo']}  per-year={ {Y:g1_py[Y]['win'] for Y in (2024,2025,2026) if Y in g1_py} }")
    print(f"  L5 MADL-select:   binding={l5_py['binding']['win']} CI-lo={l5_py['binding']['ci_lo']}  per-year={ {Y:l5_py[Y]['win'] for Y in (2024,2025,2026) if Y in l5_py} }")
    print(f"VERDICT G1 SURVIVES={out['VERDICT']['G1']['SURVIVES']}  L5 SURVIVES={out['VERDICT']['L5']['SURVIVES']}")
    print(f"-> m5_g1l5_result.json ({time.time()-t0:.0f}s)")

if __name__=="__main__": main()
