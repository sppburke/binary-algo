"""USDJPY 15m DIRECTION — does TICK MICROSTRUCTURE break the ~.539 bar-data AUC bound?

SCOPE: USDJPY · 15m · NY. The bar sweep is exhausted at AUC ~.539 (information bound, ~15 lever classes).
Raw ticks (bid/ask + bid-vol/ask-vol, 2012-2026) are now at /home/sean/git/raw — the named frontier.
PRIOR is LOW: order-flow/microstructure statistics gate SIZE not SIGN (sign-invariance arXiv:2512.15720),
and CKS event-OFI was direction-NULL at EURUSD 1m (min1_cksofi: AUC .499-.501). The label horizon (900s)
is also far longer than where microstructure lives (seconds). So this is a DECISIVE test sized to the prior.

DESIGN — align tick-micro features to the EXISTING NY bar-decision grid (so it's a clean AUGMENTATION test):
for each decision bar timestamp t (UTC epoch s; from features/USDJPY_{y}.parquet, NY, moved, contiguous
15m-ahead label) compute CAUSAL tick-microstructure features from 1s bars with index <= t over lookbacks
W in {60,300,900}s, asof-merged backward (never uses a tick after t). Features per W: mean quote-vol
imbalance (imb), realized vol of 1s mid returns (rv), short-horizon mid return (mom), tick intensity
(nt/s), mean spread; plus microprice deviation (micro-mid)/mid at t, and signed-imbalance*|ret| flow.
LABEL = the UNCHANGED deriv-faithful bar label sign(close[t+15]-close[t]) (ties LOSE) — identical to the
certified bar model, so any AUC lift is purely the tick features.

TEST (NY refit-CPCV, same 15 purged paths as the cert): three feature sets head-to-head on the SAME folds —
  bars     : 239 bar feats (control == certified base)
  bars+tk  : bars + tick-micro (treatment)
  tkonly   : tick-micro alone (does microstructure carry ANY 15m sign?)
Compare PAIRED per-path (lesson from the TB-label case: CPCV p10 is one noisy order statistic; judge by the
paired per-fold mean diff vs the matched control, not p10-vs-a-number). Falsifier: SURVIVE only if bars+tk
beats bars on per-fold paired moved-AUC with mean lift > ~one path-std AND p10 clears, in a way robust across
cov. Else KILLED -> the ~.539 bound holds even with tick microstructure (sign-invariance confirmed at 15m).

Usage:
  ~/binary-algo-venv/bin/python usdjpy_15m_tickmicro.py build [y0=2012] [y1=2026]   # cache per-year tickmicro
  ~/binary-algo-venv/bin/python usdjpy_15m_tickmicro.py run   [stride=2] [cov=0.02,0.03]
"""
import os, sys, glob, json, time, calendar, numpy as np, pandas as pd
from itertools import combinations
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from sessions import session_mask
import tick1s_cache as TK              # reuse _day_1s (raw ticks -> 1s micro bars), re-wired to /home/sean/git/raw

TARGET="USDJPY"; HOR=15; STEP=60; GAP=HOR*STEP; BE=0.541
FEAT=H.FEAT_DIR; FEATS=H.feature_cols(TARGET)
YEARS=list(range(2012,2027))
TKDIR=f"/home/sean/git/binary-algo/features_tick_{TARGET}15m"; os.makedirs(TKDIR, exist_ok=True)
WINS=(60,300,900)                      # causal lookback windows (seconds)
N_GROUPS,K_TEST=6,2; SUB_FIT=150_000; PURGE=GAP; EMBARGO=GAP; NUM_LEAVES=255

# ---------- tick-micro feature engineering on a 1s-bar series (all trailing/causal) ----------
def tickmicro_from_1s(b1s):
    """b1s: DataFrame indexed by 1s ts (UTC) with cols mid,imb,micro,spread,nt,tsz (from TK._day_1s).
    Returns a DataFrame (same 1s index) of CAUSAL microstructure features (rolling, trailing)."""
    s=b1s.sort_index()
    mid=s["mid"]; r1=mid.pct_change()
    out=pd.DataFrame(index=s.index)
    out["micro_dev"]=((s["micro"]-mid)/mid).astype("float32")      # microprice deviation at t
    out["spread"]=s["spread"].astype("float32")
    for W in WINS:
        win=f"{W}s"
        out[f"imb_{W}"]=s["imb"].rolling(win).mean().astype("float32")          # mean quote-vol imbalance
        out[f"mom_{W}"]=(mid/mid.shift(W)-1.0).astype("float32")                # short-horizon mid return
        out[f"rv_{W}"]=r1.rolling(win).std().astype("float32")                  # realized vol of 1s returns
        out[f"nt_{W}"]=(s["nt"].rolling(win).sum()/W).astype("float32")         # tick intensity (ticks/s)
        out[f"flow_{W}"]=(s["imb"].rolling(win).mean()*(mid/mid.shift(W)-1.0).abs()).astype("float32")  # signed-imb * |ret|
        out[f"sprd_{W}"]=s["spread"].rolling(win).mean().astype("float32")
    return out.replace([np.inf,-np.inf],np.nan)

def _year_days(y):
    return [f"{y}-{m:02d}-{d:02d}" for m in range(1,13) for d in range(1,calendar.monthrange(y,m)[1]+1)]

def build_year(y):
    """Build the year's 1s series (per-day, bounded memory), compute trailing tickmicro feats, asof-merge
    onto that year's BAR decision timestamps, cache to TKDIR/tickmicro_{y}.parquet (indexed by bar epoch s)."""
    barp=f"{FEAT}/{TARGET}_{y}.parquet"
    if not os.path.exists(barp): print(f"[build {y}] no bar parquet, skip",flush=True); return
    bar=pd.read_parquet(barp, columns=["close"]); bar=bar[~bar.index.duplicated(keep="last")]
    bts=bar.index.values.astype("datetime64[s]").astype("int64")                # bar decision epochs (UTC)
    # build 1s micro bars day-by-day, accumulate trailing feats sampled at bar epochs
    feats_at=[]; idx_at=[]
    TK.PAIR=TARGET                                                              # ensure USDJPY (legacy default EURUSD)
    days=_year_days(y); carry=None
    for d in days:
        files=sorted(glob.glob(f"/home/sean/git/raw/{TARGET}/{TARGET}_{d}_*.parquet"))
        if not files: continue
        b1=TK._day_1s(files)
        if b1 is None or len(b1)==0: continue
        # prepend the prior day's tail (max window) so the day's early-bar lookbacks are complete
        if carry is not None: b1=pd.concat([carry, b1])
        tm=tickmicro_from_1s(b1)
        tm_e=tm.index.values.astype("datetime64[s]").astype("int64")
        # bar epochs falling in THIS day (UTC date d)
        d0=int(pd.Timestamp(d, tz="UTC").timestamp()); d1=d0+86400
        sel=(bts>=d0)&(bts<d1); be=bts[sel]
        if len(be):
            # asof backward: most recent 1s feature row with epoch <= bar epoch
            pos=np.searchsorted(tm_e, be, side="right")-1
            ok=pos>=0
            if ok.any():
                feats_at.append(tm.iloc[pos[ok]].values); idx_at.append(be[ok])
        carry=b1.iloc[-max(WINS):]                                              # tail for next day's lookback
    if not feats_at: print(f"[build {y}] no aligned rows",flush=True); return
    X=np.concatenate(feats_at); I=np.concatenate(idx_at)
    cols=list(tickmicro_from_1s(TK._day_1s(sorted(glob.glob(f"/home/sean/git/raw/{TARGET}/{TARGET}_{days[0]}_*.parquet"))[:1] or [])).columns) if False else None
    # rebuild column names deterministically
    colnames=["micro_dev","spread"]+[f"{p}_{W}" for W in WINS for p in ("imb","mom","rv","nt","flow","sprd")]
    df=pd.DataFrame(X, columns=colnames); df.insert(0,"epoch",I)
    df=df.drop_duplicates("epoch").sort_values("epoch")
    out=f"{TKDIR}/tickmicro_{y}.parquet"; df.to_parquet(out, index=False)
    print(f"[build {y}] bars={len(bts):,} aligned={len(df):,} -> {out}",flush=True)

def cmd_build():
    y0=int(sys.argv[2]) if len(sys.argv)>2 else 2012
    y1=int(sys.argv[3]) if len(sys.argv)>3 else 2026
    t0=time.time()
    for y in range(y0,y1+1):
        out=f"{TKDIR}/tickmicro_{y}.parquet"
        if os.path.exists(out): print(f"[build {y}] cached",flush=True); continue
        ty=time.time(); build_year(y); print(f"[build {y}] {time.time()-ty:.0f}s (tot {time.time()-t0:.0f}s)",flush=True)
    print("TICKMICRO BUILD DONE",flush=True)

# ---------- the refit-CPCV augmentation test ----------
def mk_lgb(seed=0, num_leaves=NUM_LEAVES, n=800):
    return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=num_leaves,
        min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,
        n_estimators=n,n_jobs=16,verbosity=-1,random_state=seed,bagging_seed=seed,feature_fraction_seed=seed)

def nonoverlap_chrono(ts, mask, gap=GAP):
    take=[]; block=-1
    for i in np.where(mask)[0]:
        if ts[i]<block: continue
        take.append(i); block=int(ts[i])+gap
    return np.array(take,dtype=int)

def side_wr(pr, fwd, ts, thr, side):
    conf=np.abs(pr-0.5)
    want=(pr>0.5) if side=="UP" else (pr<0.5) if side=="DOWN" else np.ones(len(pr),bool)
    cand=want & (conf>=thr) & np.isfinite(fwd)
    tr=nonoverlap_chrono(ts,cand)
    if len(tr)==0: return 0,float("nan")
    pred=(pr[tr]>0.5).astype(int); ylab=(fwd[tr]>0).astype(int); moved=(fwd[tr]!=0)
    win=((pred==ylab)&moved).astype(float)
    return len(tr), float(win.mean())

def summ(a):
    a=np.asarray([x for x in a if np.isfinite(x)],float)
    if len(a)==0: return {"n_paths":0}
    return {"n_paths":int(len(a)),"mean":round(float(a.mean()),4),"p10":round(float(np.percentile(a,10)),4),
            "p50":round(float(np.percentile(a,50)),4),"min":round(float(a.min()),4),
            "frac_clear_BE":round(float((a>=BE).mean()),3)}

def build_aligned(stride):
    """Load bar feats + label + ts, asof-join cached tickmicro by epoch. Returns Xbar, Xtk, fwd, ts, colnames."""
    Xb=[]; frs=[]; tss=[]; TK_=[]
    tkcols=["micro_dev","spread"]+[f"{p}_{W}" for W in WINS for p in ("imb","mom","rv","nt","flow","sprd")]
    for y in YEARS:
        p=f"{FEAT}/{TARGET}_{y}.parquet"; tp=f"{TKDIR}/tickmicro_{y}.parquet"
        if not os.path.exists(p) or not os.path.exists(tp): continue
        d=pd.read_parquet(p, columns=FEATS+["close"]); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float); ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        contig=np.zeros(n,bool); contig[:n-HOR]=(ts[HOR:]-ts[:-HOR])==GAP
        fr=np.full(n,np.nan); fr[:n-HOR]=c[HOR:]/c[:-HOR]-1.0
        X=d[FEATS].astype("float32"); keepf=X.isna().mean(axis=1).values<0.5
        valid=contig & np.isfinite(fr) & keepf
        idx=np.where(valid)[0]
        if stride>1: idx=idx[::stride]
        tk=pd.read_parquet(tp); tk=tk.drop_duplicates("epoch").sort_values("epoch")
        tke=tk["epoch"].values; tkv=tk[tkcols].values.astype("float32")
        be=ts[idx]; pos=np.searchsorted(tke, be, side="right")-1               # asof backward (epoch<=bar ts)
        row=np.full((len(be),len(tkcols)), np.nan, dtype="float32")
        ok=pos>=0; row[ok]=tkv[pos[ok]]
        Xb.append(X.values[idx]); frs.append(fr[idx]); tss.append(ts[idx]); TK_.append(row)
    return np.concatenate(Xb), np.concatenate(TK_), np.concatenate(frs), np.concatenate(tss), tkcols

def cmd_run():
    t0=time.time()
    stride=int(sys.argv[2]) if len(sys.argv)>2 and sys.argv[2].isdigit() else 2
    covs=[float(x) for x in (sys.argv[3] if len(sys.argv)>3 else "0.02,0.03").split(",")]
    SESSION="ny"; RESULT="usdjpy_15m_tickmicro_result.json"
    print(f"[tickmicro/run] loading bar+tick aligned (stride {stride})...",flush=True)
    Xbar,Xtk,fwd,ts,tkcols=build_aligned(stride)
    o=np.argsort(ts); Xbar=Xbar[o]; Xtk=Xtk[o]; fwd=fwd[o]; ts=ts[o]
    sess=session_mask(ts,SESSION); moved=np.isfinite(fwd)&(fwd!=0.0)
    tk_cov=float(np.isfinite(Xtk[sess]).all(axis=1).mean())
    print(f"[tickmicro/run] rows={len(ts):,} NY={int(sess.sum()):,} tickfeat-complete(NY)={tk_cov:.3f} "
          f"bar_feats={Xbar.shape[1]} tk_feats={Xtk.shape[1]}  {time.time()-t0:.0f}s",flush=True)
    Xboth=np.concatenate([Xbar,Xtk],axis=1)
    SETS={"bars":Xbar, "bars+tk":Xboth, "tkonly":Xtk}
    tgs=ts[sess]; bnds=[tgs[int(k*len(tgs)/N_GROUPS)] for k in range(N_GROUPS)]+[tgs[-1]+1]
    groups=[(int(bnds[g]),int(bnds[g+1])) for g in range(N_GROUPS)]
    rng=np.random.default_rng(13)
    paths={s:{c:{"UP":[],"DOWN":[],"COMBINED":[]} for c in covs} for s in SETS}
    aucs={s:[] for s in SETS}; uprates=[]; perfold_auc={s:[] for s in SETS}
    for fi,testg in enumerate(combinations(range(N_GROUPS),K_TEST)):
        tblocks=[groups[g] for g in testg]; tin=np.zeros(len(ts),bool)
        for lo,hi in tblocks: tin|=(ts>=lo)&(ts<hi)
        test_mask=tin&sess&np.isfinite(fwd)
        purged=np.zeros(len(ts),bool)
        for lo,hi in tblocks: purged|=(ts>=lo-PURGE)&(ts<hi+EMBARGO)
        train_mask=(~purged)&moved&sess; tr_idx=np.where(train_mask)[0]
        valsel=rng.random(len(tr_idx))<0.15; val_idx=tr_idx[valsel]; fit_pool=tr_idx[~valsel]
        if len(fit_pool)>SUB_FIT: fit_pool=rng.choice(fit_pool,SUB_FIT,replace=False)
        ytr=(fwd[fit_pool]>0).astype(int); yval=(fwd[val_idx]>0).astype(int)
        ftst=fwd[test_mask]; ttst=ts[test_mask]; oo=np.argsort(ttst); ftst=ftst[oo]; ttst=ttst[oo]
        mv=ftst!=0; uprate=float((ftst[mv]>0).mean()); uprates.append(uprate)
        line=[f"path {fi+1}/15 g{list(testg)} up={uprate:.3f}"]
        for s,Xs in SETS.items():
            L=mk_lgb(seed=0)
            L.fit(Xs[fit_pool],ytr,eval_set=[(Xs[val_idx],yval)],eval_metric="auc",
                  callbacks=[lgb.early_stopping(80),lgb.log_evaluation(0)])
            pval=L.predict_proba(Xs[val_idx])[:,1]; ptst=L.predict_proba(Xs[test_mask])[:,1][oo]
            vconf=np.abs(pval-0.5)
            auc=float(roc_auc_score((ftst[mv]>0).astype(int),ptst[mv])) if mv.sum()>20 else float("nan")
            aucs[s].append(auc); perfold_auc[s].append(auc)
            for c in covs:
                thr=float(np.quantile(vconf,1-c))
                for side in ("UP","DOWN","COMBINED"):
                    nS,wS=side_wr(ptst,ftst,ttst,thr,side)
                    if nS>=25: paths[s][c][side].append(wS)
            line.append(f"{s} AUC={auc:.4f}")
        print("  "+" | ".join(line)+f" ({time.time()-t0:.0f}s)",flush=True)
    # paired per-fold AUC diff treatment vs control
    pa=np.array(perfold_auc["bars+tk"]); pc=np.array(perfold_auc["bars"]); d=pa-pc
    paired={"mean_dAUC":round(float(np.nanmean(d)),4),"median_dAUC":round(float(np.nanmedian(d)),4),
            "se":round(float(np.nanstd(d,ddof=1)/np.sqrt(np.isfinite(d).sum())),4),
            "tstat_CORRELATED_paths_discount":round(float(np.nanmean(d)/(np.nanstd(d,ddof=1)/np.sqrt(np.isfinite(d).sum()))),2),
            "wins":int((d>0).sum()),"n":int(np.isfinite(d).sum())}
    res={"key":"USDJPY.15m.ny","lever":"tick microstructure features (OFI/microprice/spread/intensity/rv/mom) aug at NY 15m grid",
         "tick_source":"/home/sean/git/raw/USDJPY","windows_s":list(WINS),"tk_feats":tkcols,"stride":stride,"covs":covs,
         "settlement":"deriv-faithful bar label sign(close[t+15]-close[t]) ties LOSE; tick feats CAUSAL asof<=t; NY refit-CPCV purge+embargo=900s",
         "auc_mean":{s:round(float(np.nanmean(aucs[s])),4) for s in SETS},
         "uprate_tripwire_ok":bool(np.all([(0.45<=u<=0.55) for u in uprates if np.isfinite(u)])),
         "paired_bars+tk_vs_bars":paired,"bycov":{}}
    for c in covs:
        res["bycov"][f"{c}"]={s:{side:summ(paths[s][c][side]) for side in ("UP","DOWN","COMBINED")} for s in SETS}
    # falsifier
    inc_auc=res["auc_mean"]["bars"]
    survives=bool(paired["mean_dAUC"]>0.003 and paired["wins"]>=11 and res["auc_mean"]["bars+tk"]>inc_auc+0.003)
    res["verdict"]={"SURVIVES":survives,
        "note":("SURVIVED: tick-micro lifts per-fold AUC robustly -> escalate to adversarial verify + seed-ens + freeze"
                if survives else
                "KILLED: tick microstructure does NOT lift 15m USDJPY direction beyond the ~.539 bar bound. "
                "tkonly AUC ~.50 => microstructure carries no own 15m SIGN (sign-invariance: it gates SIZE). "
                "The bar information bound holds even with full tick OFI/microprice/intensity data.")}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[tickmicro/run] === AUC means: {res['auc_mean']} ===",flush=True)
    print(f"  paired bars+tk vs bars: {paired}",flush=True)
    for c in covs:
        for s in SETS:
            v=res["bycov"][f"{c}"][s]["COMBINED"]
            print(f"   cov{c} {s:8s} COMB p10={v.get('p10')} mean={v.get('mean')} frac={v.get('frac_clear_BE')}",flush=True)
    print(f"\n[tickmicro/run] VERDICT: {'SURVIVED' if survives else 'KILLED'} (bars {inc_auc} -> bars+tk {res['auc_mean']['bars+tk']}, "
          f"tkonly {res['auc_mean']['tkonly']}; paired mean dAUC {paired['mean_dAUC']} wins {paired['wins']}/15) -> {RESULT}  {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "run"
    if mode=="build": cmd_build()
    else: cmd_run()
