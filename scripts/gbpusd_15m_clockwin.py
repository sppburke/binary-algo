"""GBPUSD 15-MIN — GBP-clock window levers (discovery R1): UK-data sub-window CPCV + raw-rule probes.

SCOPE: GBPUSD · 15m. Three discovery-R1 levers sharing the London-clock mechanism (sources in
SWEEP_MATRIX.md N47/N48 + backlog):
  MODE subwin — **LDN-morning 07:00-09:30 Europe/London sub-window refit-CPCV** (UK data releases land
    07:00 LDN; full-LDN null may be non-event dilution — this isolates the event-adjacent bars).
    Same per-fold-refit CPCV harness as gbpusd_15m_cpcv_session.py with a custom DST-correct mask.
  MODE probe — raw-rule conditional probes (no ML; near-free):
    N47 LDN-open momentum: at decision bars 08:30-09:00 LDN, does sign(ret 08:00->bar) predict the
        next-15m sign? (persistence-of-informed-order at the GBP vol spike; Martins-Lopes 2024.)
    N48 pre-WMR-fix: at decision bars 15:45-16:00 LDN, does sign(ret 15:30->bar) predict the next-15m
        sign (into the 16:00 fix)? (Osler fix-flow concentration.)

Falsifiers (pre-registered, written before evaluation):
  subwin: KILL if p10 < 0.541 on BOTH sides at every cov OR med_n/path < 200 (underpowered).
  N47/N48 probe: KILL if no held-out year (2024/2025/2026) conditional WR CI95-lo >= 0.541 on either
    sign-side (raw mechanism absent at tradeable strength).

Usage: ~/binary-algo-venv/bin/python gbpusd_15m_clockwin.py <subwin|probe>
"""
import os, sys, json, time, numpy as np, pandas as pd

MODE = sys.argv[1] if len(sys.argv) > 1 else "probe"
PAIR = "GBPUSD"; HOR = 15; GAP = HOR*60; BE = 0.541
FEAT = "/home/sean/git/binary-algo/features"
YEARS = list(range(2012, 2027))

def ldn_hours(ts):
    """fractional Europe/London local hour for epoch-seconds array (DST-correct)."""
    t = pd.to_datetime(ts, unit="s", utc=True).tz_convert("Europe/London")
    return t.hour.values + t.minute.values/60.0

def nonoverlap_chrono(ts, mask, gap=GAP):
    take=[]; block=-1
    for i in np.where(mask)[0]:
        if ts[i]<block: continue
        take.append(i); block=int(ts[i])+gap
    return np.array(take,dtype=int)

def boot(corr, nb=5000, seed=7):
    corr=np.asarray(corr,float)
    if len(corr)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(corr)
    a=np.array([corr[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))

# ---------------- probe mode (raw close series, no ML) ----------------
def load_close(years):
    parts=[]
    for y in years:
        p=f"{FEAT}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p, columns=["close"]); d=d[~d.index.duplicated(keep="last")]
        parts.append(d)
    d=pd.concat(parts)
    c=d["close"].values.astype(float)
    ts=d.index.values.astype("datetime64[s]").astype("int64")
    return c, ts

def probe():
    t0=time.time()
    RESULT="gbpusd_15m_clockwin_probe_result.json"
    res={"key":"GBPUSD.15m","mode":"probe","breakeven":BE,
         "falsifier":{"registered":"pre-eval",
           "KILL_if":"no held-out year (2024/2025/2026) conditional WR CI95-lo >= 0.541 on either sign-side, per lever"}}
    json.dump(res,open(RESULT,"w"),indent=2)
    c,ts=load_close([str(y) for y in YEARS])
    n=len(c)
    contig=np.zeros(n,bool); contig[:n-HOR]=(ts[HOR:]-ts[:-HOR])==GAP
    fwd=np.full(n,np.nan); fwd[:n-HOR]=c[HOR:]/c[:-HOR]-1.0
    lh=ldn_hours(ts)
    yr=pd.to_datetime(ts,unit="s").year.values
    levers={
      # name: (decision-window lo/hi in LDN hours, anchor hour for the conditioning return)
      "N47_ldnopen": (8.5, 9.0, 8.0),
      "N48_prefix": (15.75, 16.0, 15.5),
    }
    out={}
    for name,(wlo,whi,anch) in levers.items():
        # anchor price: last bar at/just before the anchor hour on the same LDN day
        day=pd.to_datetime(ts,unit="s",utc=True).tz_convert("Europe/London").normalize().asi8
        df=pd.DataFrame({"i":np.arange(n),"day":day,"lh":lh})
        anchor_idx=np.full(n,-1,dtype=int)
        # per-day last index with lh <= anch
        for d_,grp in df.groupby("day", sort=False):
            m=grp["lh"].values<=anch
            if not m.any(): continue
            ai=int(grp["i"].values[m][-1])
            sel=grp["i"].values
            anchor_idx[sel]=ai
        cond_ret=np.where(anchor_idx>=0, c/np.where(anchor_idx>=0,c[anchor_idx],np.nan)-1.0, np.nan)
        dec=(lh>=wlo)&(lh<whi)&contig&np.isfinite(fwd)&(fwd!=0)&np.isfinite(cond_ret)&(cond_ret!=0)
        out[name]={"window_ldn":[wlo,whi],"anchor_ldn":anch,"years":{}}
        for Y in (2024,2025,2026):
            m=dec&(yr==Y)
            tr=nonoverlap_chrono(ts, m)
            if len(tr)<5:
                out[name]["years"][str(Y)]={"n":int(len(tr))}; continue
            pred_up=cond_ret[tr]>0           # momentum-persistence rule
            win=(np.sign(fwd[tr])==np.where(pred_up,1,-1))
            lo,hi=boot(win.astype(float))
            # per sign-side
            sides={}
            for nm,msk in (("UP",pred_up),("DOWN",~pred_up)):
                if msk.sum()>=5:
                    l2,h2=boot(win[msk].astype(float))
                    sides[nm]={"n":int(msk.sum()),"wr":round(float(win[msk].mean()),4),"ci":[round(l2,3),round(h2,3)]}
                else: sides[nm]={"n":int(msk.sum())}
            out[name]["years"][str(Y)]={"n":int(len(tr)),"wr":round(float(win.mean()),4),
                                         "ci":[round(lo,3),round(hi,3)],"sides":sides}
            print(f"[{name}] {Y}: n={len(tr)} WR={win.mean():.4f} CI[{lo:.3f},{hi:.3f}] "
                  f"UP {sides.get('UP')} DOWN {sides.get('DOWN')}", flush=True)
        # anti-persistence (reversion) read is the mirror: 1-wr; record rule orientation only
    res["levers"]=out
    # verdicts
    verd={}
    for name in levers:
        clears=[]
        for Y in (2024,2025,2026):
            yres=out[name]["years"].get(str(Y),{})
            for s in ("UP","DOWN"):
                sd=yres.get("sides",{}).get(s,{})
                if sd.get("ci") and sd["ci"][0]>=BE: clears.append(f"{Y}-{s}")
        verd[name]={"clears":clears,"KILLED":len(clears)==0}
    res["verdict"]=verd
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[probe] verdicts: {verd}  ({time.time()-t0:.0f}s) -> {RESULT}", flush=True)

# ---------------- subwin mode (refit-CPCV on the 07:00-09:30 LDN mask) ----------------
def subwin():
    import lightgbm as lgb
    from itertools import combinations
    from sklearn.metrics import roc_auc_score
    import harness as H
    FEATS=H.feature_cols(PAIR)
    N_GROUPS,K_TEST=6,2; SUB_FIT=150_000; PURGE=GAP; EMBARGO=GAP
    COVS=[0.05,0.03,0.02]
    t0=time.time()
    RESULT="gbpusd_15m_clockwin_subwin_result.json"
    res={"key":"GBPUSD.15m","mode":"subwin 07:00-09:30 Europe/London refit-CPCV","breakeven":BE,
         "covs":COVS,"CERT_RULE":"side CERTIFIED iff p10>=0.541 AND frac_clear_BE>=0.80",
         "falsifier":{"registered":"pre-paths",
           "KILL_if":"p10<0.541 BOTH sides at every cov OR med_n/path<200 (underpowered)"}}
    json.dump(res,open(RESULT,"w"),indent=2)
    # build (mirrors gbpusd_15m_cpcv_session.build_pair_ties, stride 1 — window is already thin)
    Xs=[];fwds=[];tss=[]
    for y in YEARS:
        p=f"{FEAT}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p,columns=FEATS+["close"]); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float)
        ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        contig=np.zeros(n,bool); contig[:n-HOR]=(ts[HOR:]-ts[:-HOR])==GAP
        fr=np.full(n,np.nan); fr[:n-HOR]=c[HOR:]/c[:-HOR]-1.0
        X=d[FEATS].astype("float32"); keepf=X.isna().mean(axis=1).values<0.5
        valid=contig&np.isfinite(fr)&keepf
        idx=np.where(valid)[0]
        Xs.append(X.values[idx]); fwds.append(fr[idx]); tss.append(ts[idx])
    X=np.concatenate(Xs); fwd=np.concatenate(fwds); ts=np.concatenate(tss)
    o=np.argsort(ts); X=X[o]; fwd=fwd[o]; ts=ts[o]
    lh=ldn_hours(ts); sess=(lh>=7.0)&(lh<9.5)
    moved=np.isfinite(fwd)&(fwd!=0.0)
    print(f"[subwin] rows={len(ts):,} in-window={int(sess.sum()):,} build={time.time()-t0:.0f}s",flush=True)
    tgs=ts[sess]; bnds=[tgs[int(k*len(tgs)/N_GROUPS)] for k in range(N_GROUPS)]+[tgs[-1]+1]
    groups=[(int(bnds[g]),int(bnds[g+1])) for g in range(N_GROUPS)]
    def mk(seed=0):
        return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=255,
            min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,
            n_estimators=800,n_jobs=16,verbosity=-1,random_state=seed,bagging_seed=seed,feature_fraction_seed=seed)
    def side_wr(pr,fw,t,thr,side):
        conf=np.abs(pr-0.5)
        want=(pr>0.5) if side=="UP" else (pr<0.5) if side=="DOWN" else np.ones(len(pr),bool)
        cand=want&(conf>=thr)&np.isfinite(fw)
        tr=nonoverlap_chrono(t,cand)
        if len(tr)==0: return 0,float("nan")
        pred=(pr[tr]>0.5).astype(int); ylab=(fw[tr]>0).astype(int); mv=(fw[tr]!=0)
        return len(tr), float(((pred==ylab)&mv).mean())
    def summ(a):
        a=np.asarray([x for x in a if np.isfinite(x)],float)
        if len(a)==0: return {"n_paths":0}
        return {"n_paths":int(len(a)),"mean":round(float(a.mean()),4),"p10":round(float(np.percentile(a,10)),4),
                "frac_clear_BE":round(float((a>=BE).mean()),3)}
    rng=np.random.default_rng(13)
    paths={c:{"UP":[],"DOWN":[],"COMBINED":[]} for c in COVS}
    npaths={c:{"UP":[],"DOWN":[],"COMBINED":[]} for c in COVS}
    aucs=[]
    for fi,testg in enumerate(combinations(range(N_GROUPS),K_TEST)):
        tblocks=[groups[g] for g in testg]
        tin=np.zeros(len(ts),bool)
        for lo,hi in tblocks: tin|=(ts>=lo)&(ts<hi)
        test_mask=tin&sess&np.isfinite(fwd)
        purged=np.zeros(len(ts),bool)
        for lo,hi in tblocks: purged|=(ts>=lo-PURGE)&(ts<hi+EMBARGO)
        train_mask=(~purged)&moved&sess
        tr_idx=np.where(train_mask)[0]
        valsel=rng.random(len(tr_idx))<0.15
        val_idx=tr_idx[valsel]; fit_pool=tr_idx[~valsel]
        if len(fit_pool)>SUB_FIT: fit_pool=rng.choice(fit_pool,SUB_FIT,replace=False)
        L=mk()
        L.fit(X[fit_pool],(fwd[fit_pool]>0).astype(int),
              eval_set=[(X[val_idx],(fwd[val_idx]>0).astype(int))],eval_metric="auc",
              callbacks=[lgb.early_stopping(80),lgb.log_evaluation(0)])
        pval=L.predict_proba(X[val_idx])[:,1]; ptst=L.predict_proba(X[test_mask])[:,1]
        vconf=np.abs(pval-0.5)
        ftst=fwd[test_mask]; ttst=ts[test_mask]
        oo=np.argsort(ttst); ptst=ptst[oo]; ftst=ftst[oo]; ttst=ttst[oo]
        mv=ftst!=0
        auc=float(roc_auc_score((ftst[mv]>0).astype(int),ptst[mv])) if mv.sum()>20 else float("nan")
        aucs.append(auc)
        for cv in COVS:
            thr=float(np.quantile(vconf,1-cv))
            for s in ("UP","DOWN","COMBINED"):
                nn,w=side_wr(ptst,ftst,ttst,thr,s)
                if nn>=25: paths[cv][s].append(w); npaths[cv][s].append(nn)
        print(f"  path {fi+1}/15 AUC={auc:.4f} ({time.time()-t0:.0f}s)",flush=True)
    res["auc_summary"]={"mean":round(float(np.nanmean(aucs)),4),"min":round(float(np.nanmin(aucs)),4)}
    res["bycov"]={}
    for cv in COVS:
        sm={s:summ(paths[cv][s]) for s in ("UP","DOWN","COMBINED")}
        med={s:(int(np.median(npaths[cv][s])) if npaths[cv][s] else 0) for s in ("UP","DOWN","COMBINED")}
        res["bycov"][f"{cv}"]={"summary":sm,"med_n":med}
        print(f" cov{cv}: "+" | ".join(f"{s} p10={sm[s].get('p10')} frac={sm[s].get('frac_clear_BE')} med_n={med[s]}" for s in ("UP","DOWN","COMBINED")),flush=True)
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[subwin] done {time.time()-t0:.0f}s -> {RESULT}",flush=True)

if __name__=="__main__":
    probe() if MODE=="probe" else subwin()
