"""USDJPY 2-MIN — ESN reservoir temporal features → POOLED GBM (N-R1a, ACTUALLY RUN not subsumed).

SCOPE: USDJPY · 2m. The 239 base feats are windowed AGGREGATES (rv/rangepos/macd) that discard the intra-window
PATH ORDER. A fixed random echo-state reservoir (Jaeger ESN) over the W recent bars preserves order; its final state
is a nonlinear temporal embedding. The GRU (trained recurrence + LINEAR readout) found AUC .514 < GBM .524 — but a
GBM readout on the reservoir states can catch NONLINEAR order-interactions the linear readout misses. So: append the
reservoir final-state to the 239 base feats, POOL-train (7 majors, own 2m sign), eval USDJPY frozen-gate per-year.
ECHO-OF-MAGNITUDE control: also report whether reservoir feats rank in the GBM's top-40 by gain (if not, no path info).

Frozen-gate triage (cheaper than CPCV); escalate to CPCV only if it lifts the C1 pooled tail. Falsifier:
KILL if VAL moved-AUC <= C1's ~.524 OR no held-out year UP cov2% wr beats C1's .547/.546/.570 by a CI-separated margin
OR reservoir feats all rank below top-60 of 303 by gain.
Usage: ~/binary-algo-venv/bin/python usdjpy_2m_esn.py [stride_per_pair=42] [units=64]
"""
import os, sys, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from usdjpy_2m_base import side_eval, covcurve, nonoverlap_chrono, boot, mk_lgb

TARGET="USDJPY"; HOR=2; STEP=60; GAP=HOR*STEP; BE=0.541; W=30
PAIRS=["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCHF","USDCAD"]
FEAT=H.FEAT_DIR; FEATS=H.feature_cols(TARGET)
CH=["1m_ret_1","1m_ret_3","1m_ret_6","1m_ret_12","1m_ret_24","1m_rsi","1m_rangepos_12",
    "1m_dist_ema10","1m_dist_ema20","1m_rv_12","1m_rv_24","1m_macd_hist","5m_ret_3","5m_ret_12","4h_slope_20"]
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],
     "test24":["2024"],"test25":["2025"],"oos":["2026"]}
STRIDE=int(sys.argv[1]) if len(sys.argv)>1 and sys.argv[1].isdigit() else 42
UNITS=int(sys.argv[2]) if len(sys.argv)>2 and sys.argv[2].isdigit() else 64
RESULT=f"usdjpy_2m_esn_s{STRIDE}_result.json"

# fixed random reservoir (seeded, NOT trained)
_rng=np.random.default_rng(7)
W_in=(_rng.standard_normal((UNITS,len(CH)))*0.5).astype("float32")
M=_rng.standard_normal((UNITS,UNITS)).astype("float32")
mask=(_rng.random((UNITS,UNITS))<0.1); M*=mask                       # 10% sparse
sr=max(abs(np.linalg.eigvals(M))); W_res=(M*(0.9/sr)).astype("float32")  # spectral radius 0.9
LEAK=0.3
RES_COLS=[f"res{j}" for j in range(UNITS)]

def reservoir(win):
    """win: [N,W,C] -> final reservoir state [N,UNITS] (vectorized over batch, sequential over W)."""
    N=win.shape[0]; h=np.zeros((N,UNITS),dtype="float32")
    for s in range(W):
        x=win[:,s,:]                                                  # [N,C]
        h=(1-LEAK)*h + LEAK*np.tanh(x@W_in.T + h@W_res.T)
    return h

def build_pair(pair, years, stride):
    """base239 + reservoir(W bars) + 2m label. Returns X(df with base+res), y, moved, ts."""
    Xb=[]; Rs=[]; ys=[]; mv=[]; tss=[]
    for y in years:
        p=f"{FEAT}/{pair}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p,columns=list(dict.fromkeys(FEATS+CH+["close"]))); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float); ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        fr=np.full(n,np.nan); fr[:n-HOR]=c[HOR:]/c[:-HOR]-1.0
        Fbase=d[FEATS].astype("float32"); chmat=d[CH].astype("float32").fillna(0.0).values
        keepf=Fbase.isna().mean(axis=1).values<0.5
        # eligible bars: window contiguous + label horizon clean + base ok
        elig=[]
        for t in range(W-1,n-HOR):
            if (ts[t]-ts[t-W+1])!=(W-1)*STEP: continue
            if (ts[t+HOR]-ts[t])!=GAP: continue
            if not np.isfinite(fr[t]) or not keepf[t]: continue
            elig.append(t)
        elig=np.array(elig,dtype=int)
        if stride>1: elig=elig[::stride]
        if len(elig)==0: continue
        win=np.stack([chmat[t-W+1:t+1] for t in elig]).astype("float32")  # [N,W,C]
        Rs.append(reservoir(win)); Xb.append(Fbase.values[elig]); ys.append((fr[elig]>0).astype(int))
        mv.append(fr[elig]!=0.0); tss.append(ts[elig])
    if not Xb: return None
    Xall=np.hstack([np.concatenate(Xb), np.concatenate(Rs)])
    return pd.DataFrame(Xall,columns=FEATS+RES_COLS), np.concatenate(ys), np.concatenate(mv), np.concatenate(tss)

def main():
    t0=time.time(); ALLCOLS=FEATS+RES_COLS
    res={"key":"USDJPY.2m","model":f"POOLED base239+ESN{UNITS} reservoir (W={W}), eval USDJPY @2m","stride":STRIDE,
         "settlement":"bar-close approx, ties LOSE, BE 0.541, gap=120","splits":SPL,
         "falsifier":{"registered_utc":"pre-OOS",
            "KILL_if":"VAL moved-AUC <= 0.524 (C1) OR no held-out yr UP cov2% beats C1 .547/.546/.570 CI-separated OR all res feats rank < top-60/303 by gain"}}
    json.dump(res,open(RESULT,"w"),indent=2)
    Xtr=[];ytr=[];mtr=[]
    for p in PAIRS:
        r=build_pair(p,SPL["train"],STRIDE)
        if r: Xtr.append(r[0]); ytr.append(r[1]); mtr.append(r[2]); print(f"   train {p} {len(r[1]):,} ({time.time()-t0:.0f}s)",flush=True)
    Xtr=pd.concat(Xtr,ignore_index=True); ytr=np.concatenate(ytr); mtr=np.concatenate(mtr)
    rv=build_pair(TARGET,SPL["val"],1); Xva,yva,mva,tsv=rv
    print(f"[esn] pooled-train={int(mtr.sum()):,} USDJPY-val={int(mva.sum()):,} feats={len(ALLCOLS)} build={time.time()-t0:.0f}s",flush=True)
    L=mk_lgb(num_leaves=255)
    L.fit(Xtr[mtr],ytr[mtr],eval_set=[(Xva[mva],yva[mva])],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pva=L.predict_proba(Xva)[:,1]; val_auc=float(roc_auc_score(yva[mva],pva[mva]))
    fi=dict(zip(ALLCOLS,L.feature_importances_)); order=sorted(ALLCOLS,key=lambda c:-fi[c])
    res_ranks=[order.index(c) for c in RES_COLS]; best_res_rank=min(res_ranks)
    res["val_auc"]=val_auc; res["best_res_feat_rank"]=int(best_res_rank); res["n_res_in_top60"]=int(sum(r<60 for r in res_ranks))
    print(f"[esn] VAL moved-AUC={val_auc:.4f} (C1 .524) | best reservoir-feat rank {best_res_rank}/303, {res['n_res_in_top60']} res-feats in top60 {time.time()-t0:.0f}s",flush=True)
    # frozen gate cov2% (worst-VAL-half)
    half=len(pva)//2; confv=np.abs(pva-0.5); best=None
    for cov in (0.05,0.03,0.02):
        thr=float(np.quantile(confv,1-cov)); accs=[]
        for s,e in ((0,half),(half,len(pva))):
            r=side_eval(pva[s:e],yva[s:e],mva[s:e],tsv[s:e],thr); accs.append(r["COMBINED"]["wr"] if r else float("nan"))
        worst=np.nanmin(accs)
        if best is None or worst>best[0]: best=(worst,cov,thr)
    _,COV,THR=best; res["gate"]={"cov":COV,"thr":THR}
    res["years"]={}
    for w in ("test24","test25","oos"):
        D=build_pair(TARGET,SPL[w],1); Xw,yw,mw,tsw=D; pr=L.predict_proba(Xw)[:,1]
        auc=float(roc_auc_score(yw[mw],pr[mw])); cc=covcurve(pr,yw,mw,tsw); g=side_eval(pr,yw,mw,tsw,THR)
        res["years"][w]={"moved_auc":auc,"covcurve":cc,"gate":g}
        u=g["UP"] if g else {}; print(f"=== {w} === AUC={auc:.4f} | UP cov{COV:.0%} n{u.get('n')} wr={u.get('wr')} CI{u.get('ci')}",flush=True)
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[esn] saved {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
