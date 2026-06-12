"""USDJPY 1-MIN direction via CROSS-PAIR / USD-common-factor features — USDJPY-as-TARGET.

SCOPE: USDJPY · 1m. Mirror of m5_xpair.py (which is EURUSD-target) but with USDJPY as the TARGET
and HOR=1. Mechanism (attack v1 cause #1 = weak UP channel): USDJPY IS a USD pair (USDJPY up ⇒ USD up),
so the broad USD-strength basket built from the OTHER 6 majors is a DIRECT common-factor signal for
USDJPY's own next move, plus catch-up/residual/lead-lag of USDJPY vs that basket. EURUSD cross-pair was
none@60s, but USDJPY's exposure is direct + we have on-disk OF — distinct channel, run once.

Sign convention (everything in USD-UP terms): USD-base {USDJPY,USDCHF,USDCAD} up ⇒ USD up ⇒ sign +1;
USD-quote {EURUSD,GBPUSD,AUDUSD,NZDUSD} up ⇒ USD down ⇒ sign −1. USDJPY's own move IS USD-up (+1).

Label = USDJPY's OWN 1-min forward sign (the deliverable binary), wall-clock contiguous 60s, ties
LOSE for win-rate / excluded from train+AUC. Deriv-faithful within bar resolution (gap=60s, ties LOSE,
breakeven 0.541), COMBINED+UP+DOWN per-year CI95, selection on VAL worst-half. Falsifier pre-registered.

Usage: ~/binary-algo-venv/bin/python usdjpy_xpair.py [mode=xp|xpbase|xpof] [stride_train=8]
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from usdjpy_1m_base import nonoverlap_chrono, boot, side_eval, covcurve, mk_lgb

FEAT = H.FEAT_DIR; OFDIR = "/home/sean/git/binary-algo/features_of"
TARGET = "USDJPY"; HOR = 1; GAP = 60; BE = 0.541
PAIRS = ["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCHF","USDCAD"]
USD_BASE = {"USDJPY","USDCHF","USDCAD"}
OTHERS = [p for p in PAIRS if p != TARGET]
LB = [1,3,5,10,15,30]
SPL = {"train":[str(y) for y in range(2012,2022)], "val":["2022","2023"],
       "test24":["2024"], "test25":["2025"], "oos":["2026"]}
OF_COLS = ['OF_of_norm_1','OF_of_sum_1','OF_of_norm_3','OF_of_sum_3','OF_of_norm_5','OF_of_sum_5',
    'OF_of_norm_10','OF_of_sum_10','OF_of_norm_15','OF_of_sum_15','OF_of_norm_30','OF_of_sum_30',
    'OF_of_uptick_5','OF_of_uptick_15','OF_kyle_5','OF_kyle_15','OF_of_accel','OF_of_persist']

def usd_up_sign(p): return +1.0 if p in USD_BASE else -1.0

def build_xp(years, stride=1):
    """Cross-pair USD-factor features, USDJPY target. Returns F(df) with xp feats + _y,_ts,_fwd,_moved
    + gate cols (sess_ny/sess_ln/hour/comp60). Ties kept (_moved=False) so win-rate can charge them."""
    out=[]
    for y in years:
        cl={}; ok=True
        for p in PAIRS:
            fp=f"{FEAT}/{p}_{y}.parquet"
            if not os.path.exists(fp): ok=False; break
            d=pd.read_parquet(fp,columns=["close"]); d=d[~d.index.duplicated(keep="last")]; cl[p]=d["close"]
        if not ok: continue
        df=pd.DataFrame(cl).dropna()                       # inner-join: no ffill (avoids fake-flat trap #2)
        if len(df)<100: continue
        idx=df.index; secs=idx.values.astype("datetime64[s]").astype("int64"); n=len(df)
        lr={p:np.log(df[p].values) for p in PAIRS}
        rets={p:{k:np.concatenate([[np.nan]*k, lr[p][k:]-lr[p][:-k]]) for k in LB} for p in PAIRS}
        feats={}
        for k in LB:
            own=rets[TARGET][k]                                                  # USDJPY own move (=USD-up)
            mat=np.vstack([usd_up_sign(p)*rets[p][k] for p in OTHERS])           # 6 others in USD-up terms
            basket=np.nanmean(mat,axis=0); disp=np.nanstd(mat,axis=0)
            agree=np.nanmean((np.sign(mat)==np.sign(basket)).astype(float),axis=0)
            feats[f"own_r{k}"]=own
            feats[f"usdbask{k}"]=basket           # broad USD strength (predicts USDJPY up if it follows)
            feats[f"catchup{k}"]=basket-own       # USDJPY owes the USD move -> predicts +
            feats[f"resid{k}"]=own-basket         # USDJPY-idiosyncratic (JPY risk flow)
            feats[f"disp{k}"]=disp
            feats[f"agree{k}"]=agree
            for p in OTHERS:
                feats[f"ll_{p}{k}"]=usd_up_sign(p)*rets[p][k]-own                # lead-lag residual vs USDJPY
        hours=idx.hour.values+idx.minute.values/60.0
        feats["sess_ny"]=((hours>=13.0)&(hours<22.0)).astype(float)
        feats["sess_ln"]=((hours>=7.0)&(hours<16.0)).astype(float)
        feats["sess_tk"]=((hours>=0.0)&(hours<9.0)).astype(float)               # Tokyo (JPY-relevant)
        feats["hour"]=hours
        feats["comp60"]=pd.Series(rets[TARGET][1]).rolling(60,min_periods=20).std().values
        # label: USDJPY own next-1min sign, contiguous, ties tracked
        fwd=np.full(n,np.nan)
        if n>HOR:
            contig=(secs[HOR:]-secs[:-HOR])==HOR*60
            fr=lr[TARGET][HOR:]-lr[TARGET][:-HOR]
            fwd[:n-HOR]=np.where(contig,fr,np.nan)
        F=pd.DataFrame(feats,index=idx); F["_y"]=(fwd>0).astype(float); F["_ts"]=secs; F["_fwd"]=fwd
        F["_moved"]=(np.isfinite(fwd)&(fwd!=0)).astype(float)
        F=F.loc[np.isfinite(fwd)]                                               # keep ties (moved=0) for eval
        if stride>1: F=F.iloc[::stride]
        out.append(F)
    return pd.concat(out)

def xp_cols(df): return [c for c in df.columns if c not in ("_y","_ts","_fwd","_moved","hour")]

def _read_years(dirpath, pair, years, cols):
    parts=[]
    for y in years:
        p=f"{dirpath}/{pair}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p,columns=cols); d=d[~d.index.duplicated(keep="last")]; parts.append(d)
    return pd.concat(parts) if parts else None

def augment(F, years, mode):
    if mode in ("xpbase","xpof"):
        B=_read_years(FEAT,TARGET,years,list(H.feature_cols(TARGET)))
        if B is not None: F=F.join(B[[c for c in B.columns if c not in F.columns]],how="left")
    if mode=="xpof":
        O=_read_years(OFDIR,TARGET,years,OF_COLS)
        if O is not None: F=F.join(O[[c for c in O.columns if c not in F.columns]],how="left")
    return F

def feat_cols(mode, df, xpc):
    cols=list(xpc)
    if mode in ("xpbase","xpof"): cols+=[c for c in H.feature_cols(TARGET) if c in df.columns]
    if mode=="xpof": cols+=[c for c in OF_COLS if c in df.columns]
    return list(dict.fromkeys(cols))

def main(mode="xpof", stride=8):
    t0=time.time(); RESULT=f"usdjpy_1m_xpair_{mode}_result.json"
    res={"key":"USDJPY.1m","model":f"cross-pair USDJPY-target ({mode}) @MX_HOR=1","stride_train":stride,
         "settlement":"bar-close approx, ties LOSE, breakeven 0.541, gap=60s","splits":SPL,
         "falsifier":{"registered_utc":"pre-OOS",
            "KILL_if":"VAL moved-AUC <= 0.515 OR no held-out year (COMBINED or UP) CI95-lower clears 0.541 at any tradeable cov",
            "incumbent":"single-pair baseline UP cov1% .538/.540/.545 (sub-BE); beat = lift a UP CI-lo over 0.541"}}
    json.dump(res,open(RESULT,"w"),indent=2)

    TR=build_xp(SPL["train"],stride); VA=build_xp(SPL["val"])
    xpc=xp_cols(TR)
    TR=augment(TR,SPL["train"],mode); VA=augment(VA,SPL["val"],mode)
    cols=feat_cols(mode,TR,xpc)
    mtr=TR["_moved"].values>0.5; mva=VA["_moved"].values>0.5
    ytr=TR["_y"].astype(int).values; yva=VA["_y"].astype(int).values
    Xtr=TR[cols].astype("float32"); Xva=VA[cols].astype("float32")
    tsv=VA["_ts"].values.astype("int64")
    print(f"[xp/{mode}] train={int(mtr.sum()):,} val={int(mva.sum()):,} feats={len(cols)} build={time.time()-t0:.0f}s",flush=True)
    L=mk_lgb()
    L.fit(Xtr[mtr],ytr[mtr],eval_set=[(Xva[mva],yva[mva])],eval_metric="auc",
          callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pva=L.predict_proba(Xva)[:,1]; val_auc=float(roc_auc_score(yva[mva],pva[mva]))
    imp=sorted(zip(cols,L.feature_importances_),key=lambda z:-z[1])[:15]
    print(f"[xp/{mode}] best_iter={L.best_iteration_} VAL moved-AUC={val_auc:.4f} {time.time()-t0:.0f}s",flush=True)
    print("  top15:",", ".join(c for c,_ in imp),flush=True)
    res["val_auc"]=val_auc; res["best_iter"]=int(L.best_iteration_ or 0); res["top_feats"]=[c for c,_ in imp]

    # VAL worst-half gate (COMBINED)
    half=len(pva)//2; confv=np.abs(pva-0.5); best=None
    for cov in (0.20,0.10,0.05,0.03,0.02):
        thr=float(np.quantile(confv,1-cov)); accs=[]
        for s,e in ((0,half),(half,len(pva))):
            r=side_eval(pva[s:e],yva[s:e],mva[s:e],tsv[s:e],thr); accs.append(r["COMBINED"]["wr"] if r else float("nan"))
        worst=np.nanmin(accs)
        if best is None or worst>best[0]: best=(worst,cov,thr,accs)
    worst_half,COV,THR,halfaccs=best
    res["gate"]={"cov":COV,"conf_thr":THR,"val_worst_half_wr":float(worst_half)}
    print(f"[xp/{mode}] FROZEN gate cov{COV:.0%} thr={THR:.4f} VAL worst-half WR={worst_half:.4f}",flush=True)

    res["years"]={}
    for w in ("test24","test25","oos"):
        D=build_xp(SPL[w]); D=augment(D,SPL[w],mode)
        X=D[cols].astype("float32"); pr=L.predict_proba(X)[:,1]
        yw=D["_y"].astype(int).values; mw=D["_moved"].values>0.5; tsw=D["_ts"].values.astype("int64")
        auc=float(roc_auc_score(yw[mw],pr[mw])); up_rate=float(yw[mw].mean())
        gate=side_eval(pr,yw,mw,tsw,THR); cc=covcurve(pr,yw,mw,tsw)
        res["years"][w]={"moved_auc":auc,"moved_up_rate":up_rate,"gate":gate,"covcurve":cc,
                         "tripwire_ok":bool(0.47<=up_rate<=0.53)}
        g=gate["COMBINED"] if gate else {"n":0,"wr":float('nan'),"ci":[float('nan')]*2}
        u=gate["UP"] if gate else {"n":0,"wr":float('nan'),"ci":[float('nan')]*2}
        print(f"=== {w} === AUC={auc:.4f} up-rate={up_rate:.4f} cov{COV:.0%}: COMB n{g['n']} wr={g['wr']:.4f} CI[{g['ci'][0]:.3f},{g['ci'][1]:.3f}] | UP n{u['n']} wr={u['wr']:.4f} CI[{u['ci'][0]:.3f},{u['ci'][1]:.3f}]",flush=True)

    clears=[w for w in ("test24","test25","oos") if res["years"][w]["gate"] and
            (res["years"][w]["gate"]["COMBINED"]["ci"][0]>=BE or res["years"][w]["gate"]["UP"]["ci"][0]>=BE)]
    res["verdict"]={"val_auc_le_0515":bool(val_auc<=0.515),"years_CIlo_clears_BE_at_gate":clears,
                    "KILLED":bool(val_auc<=0.515 or len(clears)==0)}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[xp/{mode}] VERDICT: {'KILLED' if res['verdict']['KILLED'] else 'SURVIVED'} (val_auc={val_auc:.4f}; clears={clears}) -> {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "xpof"
    stride=int(sys.argv[2]) if len(sys.argv)>2 else 8
    main(mode,stride)
