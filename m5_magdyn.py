"""R1 — MAGNITUDE-AS-DYNAMIC-THRESHOLD GATE (magnitude×direction bridge, sign-invariance-safe).

MECHANISM (which side's sign: BOTH — it only modulates the gate, never the sign).
  Sign of every bet comes ONLY from the frozen m5xp DIRECTION model (pr>0.5 => UP). A separately-trained 5m
  MAGNITUDE model (target |ret300|>=train-Q75, the program's one strong edge) forecasts move SIZE ex-ante.
  CLAIM: among the book's confident bars, deriv win-rate is HIGHER when the forecast move is large (a small
  forecast move => near-flat outcome => more ties/coin-flips => lower win-rate, since ties LOSE). So a DYNAMIC
  confidence threshold τ(mag) — stricter when mag-forecast is low — should lift the win-rate floor vs the
  incumbent's FIXED confidence cover at matched coverage. Magnitude is used ONLY to gate (legit per the
  sign-invariance theorem), never to choose direction. (Distinct from the null E2 "direction|magnitude-quartile".)

STAGE A (mechanism-first, cheap): on val+test gated UP bars, win-rate by mag-forecast quartile. If win-rate does
  NOT rise with magnitude, the mechanism is absent => KILL before building the gate.
STAGE B (if mechanism present): dynamic gate score = conf + λ·mag_pred (λ picked on worst-VAL-half), matched
  coverage, vs the λ=0 fixed-conf baseline; per-year 2024/25/26 CI95.

PRE-REGISTERED FALSIFIER: KILL R1 unless (i) Stage-A win-rate lift (top mag quartile − bottom) ≥ +0.02 with n≥100/bin,
  AND (ii) the dynamic gate BEATS the λ=0 fixed-conf baseline on worst-VAL-half AND on binding-2025 forward at
  matched coverage, AND (iii) binding-2025 moved-acc CI95-lower > 0.541. Incumbent to beat: m5xp UP refit floor
  0.553 / forward binding-2025 0.577.

  ~/binary-algo-venv/bin/python m5_magdyn.py
"""
import os, sys; sys.argv=["x"]
import json, math, time
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import m5_xpair as MX, m5_xpair_production as XP
import harness as H

ROOT="/media/sean/CORSAIR/binary-algo"; BE=0.541
FEAT=H.FEAT_DIR; HOR=5
import re
MAGF=[c for c in H.feature_cols("EURUSD") if re.search(r"rv|bb_width|atr|rangepos|_std|semivar|range", c, re.I)]

def boot(c, n=2000, seed=7):
    c=np.asarray(c,float)
    if len(c)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); bs=[c[rng.integers(0,len(c),len(c))].mean() for _ in range(n)]
    return float(np.percentile(bs,2.5)), float(np.percentile(bs,97.5))

# ---- 1) train a light 5m MAGNITUDE model on EURUSD features (no cross-pair needed) ----
def train_mag():
    Xs=[];ys=[]
    thr=None; arets_tr=[]
    for y in range(2012,2022):
        p=f"{FEAT}/EURUSD_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p, columns=MAGF+["close"]); d=d[~d.index.duplicated(keep="last")].sort_index()
        secs=d.index.values.astype("datetime64[s]").astype("int64"); c=d["close"].values.astype(float)
        n=len(c); fwd=np.full(n,np.nan)
        if n>HOR:
            contig=(secs[HOR:]-secs[:-HOR])==HOR*60
            fr=np.log(c[HOR:])-np.log(c[:-HOR]); fwd[:n-HOR]=np.where(contig,fr,np.nan)
        aret=np.abs(fwd); valid=np.isfinite(aret)&(aret>0)
        Xs.append(d.loc[valid,MAGF].to_numpy(np.float32)); arets_tr.append(aret[valid]); del d
    X=np.concatenate(Xs); aret=np.concatenate(arets_tr)
    thr=np.quantile(aret,0.75); ym=(aret>=thr).astype(int)
    # cap
    if len(X)>150000:
        rng=np.random.default_rng(7); idx=np.sort(rng.choice(len(X),150000,replace=False)); X=X[idx]; ym=ym[idx]
    m=lgb.LGBMClassifier(objective="binary",n_estimators=400,learning_rate=0.03,num_leaves=63,
        min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=10,n_jobs=20,verbosity=-1)
    m.fit(X,ym)
    return m, float(thr)

# ---- 2) per-split: direction pr + meta sm + mag_pred + y + fwd, within book gate ----
def eval_split(w, P, M, cols, mcols, THR, magm):
    D=MX.build_xp(XP.SPL[w]); D=MX.augment(D,XP.SPL[w],XP.MODE)
    pr=P.predict(D[cols].astype("float32")); y=D["_y"].astype(int).values
    sm=M.predict(XP._Xmeta(D,pr,mcols)); ts=D["_ts"].values.astype("int64"); ny=D["sess_ny"].values>0.5
    fwd=D["_fwd"].values.astype(float)
    magp=magm.predict_proba(D[MAGF].astype("float32"))[:,1]
    gate=ny&(sm>=THR)
    out=dict(ts=ts[gate],pr=pr[gate],y=y[gate],fwd=fwd[gate],mag=magp[gate],conf=np.abs(pr[gate]-0.5))
    del D
    return out

def yr(ts): return (np.asarray(ts,dtype="datetime64[s]").astype("datetime64[Y]").astype(int)+1970)

def main():
    t0=time.time()
    magm,magthr=train_mag(); print(f"[mag] trained 5m magnitude model (|ret300|>=Q75={magthr:.2e}), {len(MAGF)} feats, {time.time()-t0:.0f}s",flush=True)
    p,P,M=XP._load(); cols=p["primary_feats"]; mcols=p["meta_feats"]; THR=p["meta_thr"]
    S={w:eval_split(w,P,M,cols,mcols,THR,magm) for w in ("val","test24","test25","oos")}
    for w in S: print(f"  [{w}] gated bars={len(S[w]['ts'])}",flush=True)

    # ----- STAGE A: mechanism on val+test24+test25 gated UP bars -----
    up=lambda d:(d["pr"]>0.5)
    cat=lambda *ws:{k:np.concatenate([S[w][k][up(S[w])] for w in ws]) for k in ("ts","pr","y","fwd","mag","conf")}
    A=cat("val","test24","test25")
    correctA=(A["pr"]>0.5).astype(int)==A["y"]
    q=np.quantile(A["mag"],[0.25,0.5,0.75])
    bins=np.digitize(A["mag"],q)
    stageA=[]
    for b in range(4):
        m=bins==b
        wr=correctA[m].mean() if m.sum() else float("nan")
        stageA.append(dict(quartile=b,n=int(m.sum()),win=round(float(wr),4),
                           mean_aret=round(float(np.abs(A["fwd"][m]).mean()),6)))
    lift=stageA[3]["win"]-stageA[0]["win"]
    mech_ok=(lift>=0.02) and all(s["n"]>=100 for s in stageA)
    print("\n[STAGE A] win-rate by mag-forecast quartile (val+test UP bars):")
    for s in stageA: print(f"   q{s['quartile']}: n={s['n']:5d} win={s['win']:.4f} mean|ret|={s['mean_aret']:.2e}")
    print(f"   top−bottom lift={lift:+.4f}  mechanism_present={mech_ok}")

    # ----- STAGE B: dynamic gate vs fixed-conf baseline (always run for full picture) -----
    COV=0.10   # matched coverage among UP gated bars (book ships ~0.05-0.10)
    def winrate_at(dat, score, cov):
        # top-cov by score among UP bars; return per-year acc + n + pooled
        u=up(dat); sc=score[u]; corr=((dat["pr"][u]>0.5).astype(int)==dat["y"][u]); ts=dat["ts"][u]
        if len(sc)<20: return {}
        thr=np.quantile(sc,1-cov); sel=sc>=thr
        ys=yr(ts[sel]); res={}
        for Y in (2024,2025,2026):
            mm=ys==Y
            if mm.sum()>=20:
                cc=corr[sel][mm]; lo,hi=boot(cc.astype(float)); res[Y]=(round(float(cc.mean()),4),int(mm.sum()),round(lo,4),round(hi,4))
        res["pooled"]=(round(float(corr[sel].mean()),4),int(sel.sum())); return res
    # pick λ on worst-VAL-half: maximize min(val-half1,val-half2) win-rate at cov among UP
    V=S["val"]; uV=up(V); tsV=V["ts"][uV]; confV=V["conf"][uV]; magV=V["mag"][uV]; corrV=((V["pr"][uV]>0.5).astype(int)==V["y"][uV])
    half=np.median(tsV); h1=tsV<half; h2=~h1
    def worsthalf(lam):
        sc=confV+lam*(magV-magV.mean())
        accs=[]
        for h in (h1,h2):
            s=sc[h]; c=corrV[h]
            if h.sum()<40: return -1
            thr=np.quantile(s,1-COV); sel=s>=thr
            if sel.sum()<15: return -1
            accs.append(c[sel].mean())
        return min(accs)
    lams=[0.0,0.05,0.1,0.2,0.4,0.8,1.5,3.0]
    wh=[(lam,worsthalf(lam)) for lam in lams]
    best_lam=max(wh,key=lambda x:x[1])[0]
    base_wh=dict(wh)[0.0]; best_whv=dict(wh)[best_lam]
    print(f"\n[STAGE B] worst-VAL-half win@cov{COV}: λ=0 baseline={base_wh:.4f}; best λ={best_lam} -> {best_whv:.4f}")
    # held-out per-year: baseline (λ=0) vs dynamic (best_lam), pooled across test+oos eval splits
    def full(dat): return dat
    EV=cat("test24","test25","oos")   # held-out UP bars
    base_score=EV["conf"]; dyn_score=EV["conf"]+best_lam*(EV["mag"]-EV["mag"].mean())
    base=winrate_at({k:EV[k] for k in EV}, np.where(up(EV),0,0)+0, COV) if False else None
    # winrate_at expects full dict incl pr/y; build per-year using EV directly:
    def per_year(score):
        sc=score; corr=((EV["pr"]>0.5).astype(int)==EV["y"]); ts=EV["ts"]
        thr=np.quantile(sc,1-COV); sel=sc>=thr; ys=yr(ts[sel]); res={}
        for Y in (2024,2025,2026):
            mm=ys==Y
            if mm.sum()>=20:
                cc=corr[sel][mm].astype(float); lo,hi=boot(cc); res[Y]=dict(win=round(float(cc.mean()),4),n=int(mm.sum()),ci=[round(lo,4),round(hi,4)])
        res["pooled"]=dict(win=round(float(corr[sel].mean()),4),n=int(sel.sum())); return res
    base_py=per_year(base_score); dyn_py=per_year(dyn_score)
    print(f"   BASELINE(λ=0)  per-year: {base_py}")
    print(f"   DYNAMIC(λ={best_lam}) per-year: {dyn_py}")

    # ----- verdict -----
    def bind(py):
        ys=[py[Y]["win"] for Y in (2024,2025,2026) if Y in py]; lo=[py[Y]["ci"][0] for Y in (2024,2025,2026) if Y in py]
        return (min(ys) if ys else float("nan")), (min(lo) if lo else float("nan"))
    b_bind,b_lo=bind(base_py); d_bind,d_lo=bind(dyn_py)
    beats_baseline = (best_whv>base_wh+1e-4) and (d_bind>b_bind+1e-4)
    binding_clears = d_lo>BE
    survive = bool(mech_ok and beats_baseline and binding_clears)
    out=dict(test="R1 magnitude-as-dynamic-threshold gate (5m)", breakeven=BE, magnitude_Q75=magthr,
        mag_feats=len(MAGF), cov=COV, best_lambda=best_lam,
        stageA_winrate_by_mag_quartile=stageA, stageA_lift=round(lift,4), mechanism_present=mech_ok,
        worst_val_half=dict(baseline_lam0=round(base_wh,4), best_lam=best_lam, best=round(best_whv,4)),
        baseline_per_year=base_py, dynamic_per_year=dyn_py,
        baseline_binding=dict(win=round(b_bind,4),ci_lo=round(b_lo,4)),
        dynamic_binding=dict(win=round(d_bind,4),ci_lo=round(d_lo,4)),
        incumbent_to_beat=dict(refit_floor=0.553, forward_binding_2025=0.577),
        VERDICT=dict(mechanism_present=mech_ok, beats_fixed_conf_baseline=beats_baseline,
            binding_year_ci_clears_breakeven=binding_clears, SURVIVES=survive,
            statement=("R1 SURVIVES: magnitude-dynamic gate beats fixed-conf baseline on worst-VAL-half + binding year, CI clears breakeven."
                if survive else
                f"R1 KILLED: mechanism_present={mech_ok} (lift {lift:+.4f}), beats_baseline={beats_baseline} "
                f"(dyn binding {d_bind:.4f} vs base {b_bind:.4f}; worst-half dyn {best_whv:.4f} vs base {base_wh:.4f}), "
                f"binding_clears={binding_clears} (CI-lo {d_lo:.4f}). Magnitude gate adds nothing beyond the book's existing compression+confidence gate.")))
    json.dump(out, open(f"{ROOT}/m5_magdyn_result.json","w"), indent=2, default=str)
    print(f"\nVERDICT: {out['VERDICT']['statement']}")
    print(f"-> m5_magdyn_result.json  ({time.time()-t0:.0f}s)")

if __name__=="__main__":
    main()
