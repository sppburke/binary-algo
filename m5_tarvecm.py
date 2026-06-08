"""N10 — TAR-VECM error-correction SPEED-OF-ADJUSTMENT SIGN as a 5m EURUSD direction signal.

MECHANISM (sign-aware): EURUSD co-moves with the USD majors; estimate the cointegrating vector β (Engle-Granger
first step) on TRAIN: z_t = logEUR_t − β·log(majors)_t (deviation from equilibrium). The VECM error-correction
term predicts the SIGN of the next move: if α<0 (reversion), pred_up = (z_t < −thr) — below equilibrium ⇒ revert
UP; pred_dn = (z_t > +thr). TAR (threshold) band: only act when |z|>thr (band of inaction). Distinct from C3 RMT
eigen-residual (KILLED by USD-factor sign-flip) and N2 triangular residual: this uses the SIGN of the EC
adjustment with a threshold, not the residual LEVEL fed to a GBM. Polarity (revert vs momentum) chosen on
worst-VAL-half (pre-registered: reversion is primary).

KNOWN RISK (pre-registered): the cross-pair sign relationship inverted in 2025 (it killed C3) — if the EC-sign
anti-transfers 2024→2025, this is regime-dependent, not a stable edge.

PRE-REGISTERED FALSIFIER: KILL unless the EC-sign signal's binding (worst of 2024/25/26) moved-acc CI95-lower
clears breakeven 0.541 at usable coverage (n≥100/yr), AND VAL worst-half acc ≥ 0.541. Single-year hit or
2024-only = KILL (the C3 failure mode). Incumbent: m5xp UP refit floor 0.553.

  ~/binary-algo-venv/bin/python m5_tarvecm.py
"""
import os, sys; sys.argv=["x"]
import json, time, numpy as np, pandas as pd
import harness as H
ROOT="/home/sean/git/binary-algo"; FEAT=H.FEAT_DIR; BE=0.541; HOR=5
PAIRS=["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCHF","USDCAD"]; OTH=PAIRS[1:]
TRAIN=range(2012,2022); VAL=[2022,2023]
def boot(c,nb=2000,seed=7):
    c=np.asarray(c,float)
    if len(c)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(c); a=np.array([c[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))

def load_year(y):
    """EURUSD-own-clock log-closes for all majors (NO ffill; drop bars missing any leg)."""
    cols={}
    for p in PAIRS:
        fp=f"{FEAT}/{p}_{y}.parquet"
        if not os.path.exists(fp): return None
        d=pd.read_parquet(fp,columns=["close"]); d=d[~d.index.duplicated(keep="last")]
        cols[p]=np.log(d["close"])
    df=pd.concat(cols,axis=1); df.columns=PAIRS
    df=df.reindex(cols["EURUSD"].index).dropna()        # EURUSD clock, require all legs present
    return df

def main():
    t0=time.time()
    # ---- fit cointegrating β on TRAIN (logEUR ~ const + 6 majors) ----
    TRdf=[load_year(y) for y in TRAIN]; TRdf=[d for d in TRdf if d is not None]; TRdf=pd.concat(TRdf)
    Xtr=np.column_stack([np.ones(len(TRdf))]+[TRdf[p].values for p in OTH]); ytr=TRdf["EURUSD"].values
    beta,_,_,_=np.linalg.lstsq(Xtr,ytr,rcond=None)
    ztr=ytr-Xtr@beta; thr_grid=np.quantile(np.abs(ztr),[0.5,0.6,0.7,0.8])
    print(f"[tarvecm] β fit on TRAIN n={len(TRdf):,}; |z| std={ztr.std():.2e}; thr grid={thr_grid} {time.time()-t0:.0f}s",flush=True)
    del TRdf

    def eval_years(years, thr, polarity):
        # polarity +1=reversion (pred_up=z<−thr), −1=momentum
        accs={};
        for y in years:
            df=load_year(y)
            if df is None: continue
            secs=df.index.values.astype("datetime64[s]").astype("int64"); n=len(df)
            X=np.column_stack([np.ones(n)]+[df[p].values for p in OTH]); z=df["EURUSD"].values-X@beta
            # forward 300s sign, wall-clock contiguous
            fwd=np.full(n,np.nan)
            if n>HOR:
                contig=(secs[HOR:]-secs[:-HOR])==HOR*60
                fr=df["EURUSD"].values[HOR:]-df["EURUSD"].values[:-HOR]; fwd[:n-HOR]=np.where(contig,fr,np.nan)
            moved=np.isfinite(fwd)&(fwd!=0)
            sig=np.zeros(n)                                    # +1 predict up, -1 predict down
            sig[z<-thr]=1*polarity; sig[z>thr]=-1*polarity
            m=moved&(sig!=0)
            # nonoverlap chrono (gap 300s)
            idx=np.where(m)[0]; sel=[]; last=-1e18
            for i in idx:
                if secs[i]>=last+300: sel.append(i); last=secs[i]
            sel=np.array(sel,int)
            if len(sel)>=20:
                correct=((sig[sel]>0).astype(int)==(fwd[sel]>0).astype(int)).astype(float)
                lo,hi=boot(correct); uprate=float((fwd[sel]>0).mean())
                accs[y]=dict(acc=round(float(correct.mean()),4),n=len(sel),ci=[round(lo,4),round(hi,4)],uprate=round(uprate,3))
            del df
        return accs

    # ---- select polarity + thr on worst-VAL-half ----
    best=None
    for pol in (1,-1):
        for thr in thr_grid:
            va=eval_years(VAL,thr,pol)
            if len(va)<2: continue
            wh=min(va[y]["acc"] for y in va); n=min(va[y]["n"] for y in va)
            if n<50: continue
            if best is None or wh>best[0]: best=(wh,pol,float(thr),va)
    if best is None:
        print("[tarvecm] no usable VAL operating point"); return
    whacc,pol,thr,va=best
    poln="reversion" if pol==1 else "momentum"
    print(f"[tarvecm] VAL-selected: {poln} thr={thr:.2e} worst-half-acc={whacc:.4f} VAL={ {y:va[y]['acc'] for y in va} }",flush=True)
    # ---- held-out per-year ----
    ho=eval_years([2024,2025,2026],thr,pol)
    b=[ho[y]["acc"] for y in ho]; lo=[ho[y]["ci"][0] for y in ho]; ur=[ho[y]["uprate"] for y in ho]
    binding=min(b) if b else None; binding_lo=min(lo) if lo else None
    tripwire_ok=all(0.47<=u<=0.53 for u in ur) if ur else False
    survive=bool(binding_lo is not None and binding_lo>BE and whacc>BE and all(ho[y]["n"]>=100 for y in ho))
    out=dict(test="N10 TAR-VECM error-correction sign — 5m EURUSD direction", breakeven=BE, polarity=poln, thr=thr,
        val_worsthalf_acc=round(whacc,4), per_year=ho, binding=dict(acc=binding,ci_lo=binding_lo),
        uprate_tripwire_ok=tripwire_ok,
        VERDICT=dict(SURVIVES=survive,
            statement=(f"N10 SURVIVES: EC-sign {poln} binding {binding} (CI-lo {binding_lo}) clears breakeven all years."
                if survive else
                f"N10 KILLED: EC-sign {poln} binding {binding} (CI-lo {binding_lo}); per-year {{ {','.join(f'{y}:{ho[y]['acc']}' for y in ho)} }}; "
                f"tripwire_ok={tripwire_ok}. Cross-pair cointegration sign does NOT carry stable 5m direction "
                f"(same USD-factor regime-dependence that killed C3 RMT-residual).")))
    json.dump(out,open(f"{ROOT}/m5_tarvecm_result.json","w"),indent=2,default=str)
    print(f"[tarvecm] held-out per-year: { {y:(ho[y]['acc'],ho[y]['n'],ho[y]['uprate']) for y in ho} }")
    print(f"VERDICT: {out['VERDICT']['statement']}\n-> m5_tarvecm_result.json ({time.time()-t0:.0f}s)")

if __name__=="__main__": main()
