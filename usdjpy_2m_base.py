"""USDJPY 2-MINUTE binary DIRECTION — single-pair BASELINE (foundation for the 2m sweep).

SCOPE: USDJPY · 2m. The foundational baseline for the (USDJPY,2m,UP) and (USDJPY,2m,DOWN) keys.
Sibling of usdjpy_1m_base.py (H=1) — this is the H=2 fork (the repo forks per horizon:
min1/min2/m5/m10/m15/m30). The usdjpy_2m_* method suite imports build/side_eval/covcurve/
nonoverlap_chrono/boot/mk_lgb from HERE.

Bar-based path (USDJPY has NO 1s tick cache; only EURUSD does). Uses the 239 multi-TF base
features in features/USDJPY_<year>.parquet. The parquet's precomputed y/fwd_ret are the 5-MIN
label, so the 2-min label is RE-DERIVED here from the faithful `close` series:
  label_t = sign(close[t+2]-close[t]),  requiring ts[t+2]-ts[t]==120  (two clean 60s steps, no gap),
ties (move==0) EXCLUDED from train/AUC but counted as LOSSES for the tradeable win-rate.

Deriv-faithful WITHIN bar resolution: entry≈close[t], exit≈close[t+2] (bar-close approximation —
no tick precision available for USDJPY; mild OPTIMISTIC proxy of the tick-tradeable edge). Breakeven
win-rate 0.541 (deriv payout R~1.85). de-overlap = nonoverlap_chrono gap=120s (live first-come, no
look-ahead). Per-year held-out (2024/2025/2026=strict OOS) with bootstrap CI95. Selection on VAL by
WORST-half stability (never VAL-acc-max). COMBINED + UP-split + DOWN-split reported separately.

Pre-registered falsifier written to usdjpy_2m_base_result.json BEFORE the held-out read.

Usage: ~/binary-algo-venv/bin/python usdjpy_2m_base.py [stride] [leaves]
"""
import os, sys, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H

PAIR = "USDJPY"; HOR = 2; STEP = 60; GAP = HOR*STEP; BE = 0.541   # H=2 -> 120s horizon, gap 120s
FEAT = H.FEAT_DIR
SPL = {"train":[str(y) for y in range(2012,2022)], "val":["2022","2023"],
       "test24":["2024"], "test25":["2025"], "oos":["2026"]}
# defaults reproduce the recorded baseline; override via argv: python usdjpy_2m_base.py <stride> <leaves>
def _argint(i, default): return int(sys.argv[i]) if len(sys.argv)>i and str(sys.argv[i]).isdigit() else default
TR_STRIDE = _argint(1, 24)     # train ~3.5M moved bars -> stride 24 ~= 150k
NUM_LEAVES = _argint(2, 127)
RESULT = "usdjpy_2m_base_result.json" if (TR_STRIDE==24 and NUM_LEAVES==127) else f"usdjpy_2m_base_s{TR_STRIDE}_l{NUM_LEAVES}_result.json"

FEATS = H.feature_cols(PAIR)   # 239 base features

def build(years, stride=1):
    """Per-year: 239 feats + 2-min contiguous label. Returns X(df), y(1=up/0=down on moved),
    moved(bool), ts(secs int64). Ties kept in rows (moved=False) so win-rate can charge them.
    label_t = sign(close[t+HOR]-close[t]); requires ts[t+HOR]-ts[t]==GAP (no intervening gap)."""
    Xs=[]; ys=[]; mv=[]; tss=[]
    for y in years:
        p=f"{FEAT}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p, columns=FEATS+["close"])
        d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float)
        ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        contig=np.zeros(n,bool); contig[:n-HOR]=(ts[HOR:]-ts[:-HOR])==GAP   # exactly HOR clean 60s steps
        fr=np.full(n,np.nan); fr[:n-HOR]=c[HOR:]/c[:-HOR]-1.0
        X=d[FEATS].astype("float32")
        keepf=X.isna().mean(axis=1).values<0.5           # drop warmup / mostly-NaN rows
        valid=contig & np.isfinite(fr) & keepf
        moved=valid & (fr!=0.0)
        idx=np.where(valid)[0]
        if stride>1: idx=idx[::stride]
        Xs.append(X.iloc[idx]); ys.append((fr[idx]>0).astype(int))
        mv.append(moved[idx]); tss.append(ts[idx])
    X=pd.concat(Xs); return X, np.concatenate(ys), np.concatenate(mv), np.concatenate(tss)

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

def mk_lgb(n=3000, num_leaves=127):
    return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=num_leaves,
        min_child_samples=400,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,
        n_estimators=n,n_jobs=20,verbosity=-1)

def side_eval(pr, y, moved, ts, thr):
    """At conf>=thr, nonoverlap_chrono trades; ties LOSE. Return COMBINED/UP/DOWN win-rates+CI+n."""
    conf=np.abs(pr-0.5); cand=conf>=thr
    tr=nonoverlap_chrono(ts, cand)
    if len(tr)==0: return None
    pred=(pr[tr]>0.5).astype(int)
    win=((pred==y[tr]) & moved[tr]).astype(float)      # tie (moved=False) => loss
    out={}
    lo,hi=boot(win); out["COMBINED"]={"n":int(len(tr)),"wr":float(win.mean()),"ci":[lo,hi]}
    up=pred==1
    if up.sum()>0:
        lo,hi=boot(win[up]); out["UP"]={"n":int(up.sum()),"wr":float(win[up].mean()),"ci":[lo,hi]}
    else: out["UP"]={"n":0,"wr":float("nan"),"ci":[float("nan")]*2}
    dn=pred==0
    if dn.sum()>0:
        lo,hi=boot(win[dn]); out["DOWN"]={"n":int(dn.sum()),"wr":float(win[dn].mean()),"ci":[lo,hi]}
    else: out["DOWN"]={"n":0,"wr":float("nan"),"ci":[float("nan")]*2}
    return out

def covcurve(pr, y, moved, ts, covs=(0.30,0.20,0.10,0.05,0.03,0.02,0.01)):
    conf=np.abs(pr-0.5); rows={}
    for cov in covs:
        thr=float(np.quantile(conf,1-cov))
        r=side_eval(pr,y,moved,ts,thr)
        rows[f"{cov:.2f}"]={"thr":thr, **({k:{"n":v["n"],"wr":round(v["wr"],4)} for k,v in r.items()} if r else {})}
    return rows

def main():
    t0=time.time()
    # ---- pre-register falsifier BEFORE held-out read ----
    res={"key":"USDJPY.2m", "model":"single-pair LGBM baseline (239 base feats, 2-min own-clock label)",
         "settlement":"bar-close approx, ties LOSE, breakeven 0.541, gap=120s nonoverlap_chrono",
         "splits":SPL, "tr_stride":TR_STRIDE, "num_leaves":NUM_LEAVES,
         "falsifier":{"registered_utc":"pre-OOS",
            "KILL_if":"VAL moved-AUC <= 0.515  OR  no held-out year COMBINED win-rate CI95-lower clears 0.541",
            "rationale":"USDJPY 1m was near-efficient (best UP worst-yr .534 < .541 BE). 2m is the transition "
                        "horizon: EURUSD 1m~.50 -> 2m UP .555/DOWN .540, so the dip-buy reversion + USD-common "
                        "factor begin to carry by 120s. Measure directly whether USDJPY 2m clears BE."}}
    json.dump(res, open(RESULT,"w"), indent=2)

    Xtr,ytr,mtr,_=build(SPL["train"], TR_STRIDE)
    Xva,yva,mva,tsv=build(SPL["val"])
    itr=mtr; iva=mva                                  # train/early-stop on MOVED bars (ties excluded)
    print(f"[base2m] stride={TR_STRIDE} leaves={NUM_LEAVES} train={int(itr.sum()):,} val={int(iva.sum()):,} feats={len(FEATS)} build={time.time()-t0:.0f}s", flush=True)
    L=mk_lgb(num_leaves=NUM_LEAVES)
    L.fit(Xtr[itr], ytr[itr], eval_set=[(Xva[iva], yva[iva])], eval_metric="auc",
          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    pva=L.predict_proba(Xva)[:,1]
    val_auc=float(roc_auc_score(yva[iva], pva[iva]))
    print(f"[base2m] best_iter={L.best_iteration_} VAL moved-AUC={val_auc:.4f} {time.time()-t0:.0f}s", flush=True)
    res["val_auc"]=val_auc; res["best_iter"]=int(L.best_iteration_ or 0)

    # ---- VAL gate selection: WORST-half stability (never VAL-acc-max) ----
    half=len(pva)//2
    confv=np.abs(pva-0.5); best=None
    for cov in (0.20,0.10,0.05,0.03,0.02):
        thr=float(np.quantile(confv,1-cov))
        accs=[]
        for s,e in ((0,half),(half,len(pva))):
            r=side_eval(pva[s:e], yva[s:e], mva[s:e], tsv[s:e], thr)
            accs.append(r["COMBINED"]["wr"] if r else float("nan"))
        worst=np.nanmin(accs)
        if best is None or worst>best[0]: best=(worst, cov, thr, accs)
    worst_half_wr, COV, THR, halfaccs = best
    res["gate"]={"cov":COV, "conf_thr":THR, "val_worst_half_wr":float(worst_half_wr), "val_half_wrs":[float(a) for a in halfaccs]}
    print(f"[base2m] FROZEN gate cov{COV:.0%} thr={THR:.4f} VAL worst-half WR={worst_half_wr:.4f} (halves {halfaccs})", flush=True)

    # ---- held-out per year: COMBINED + UP + DOWN, at frozen gate + full covcurve ----
    res["years"]={}
    for w in ("test24","test25","oos"):
        Xw,yw,mw,tsw=build(SPL[w])
        pr=L.predict_proba(Xw)[:,1]
        auc=float(roc_auc_score(yw[mw], pr[mw]))
        up_rate=float(yw[mw].mean())
        gate=side_eval(pr, yw, mw, tsw, THR)
        cc=covcurve(pr, yw, mw, tsw)
        res["years"][w]={"moved_auc":auc, "moved_up_rate":up_rate, "gate":gate, "covcurve":cc,
                         "tripwire_ok": bool(0.47<=up_rate<=0.53)}
        g=gate["COMBINED"] if gate else {"n":0,"wr":float('nan'),"ci":[float('nan')]*2}
        print(f"=== {w} === moved-AUC={auc:.4f} up-rate={up_rate:.4f} | gate cov{COV:.0%}: "
              f"COMB n{g['n']} wr={g['wr']:.4f} CI[{g['ci'][0]:.3f},{g['ci'][1]:.3f}]", flush=True)
        if gate:
            for side in ("UP","DOWN"):
                s=gate[side]; print(f"        {side}: n{s['n']} wr={s['wr']:.4f} CI[{s['ci'][0]:.3f},{s['ci'][1]:.3f}]", flush=True)

    # ---- apply falsifier ----
    kill_auc = val_auc<=0.515
    clears = [w for w in ("test24","test25","oos")
              if res["years"][w]["gate"] and res["years"][w]["gate"]["COMBINED"]["ci"][0]>=BE]
    kill_wr = len(clears)==0
    res["verdict"]={"val_auc_le_0515":bool(kill_auc), "years_COMBINED_CIlo_clears_BE":clears,
                    "KILLED":bool(kill_auc or kill_wr),
                    "note":"baseline as a standalone tradeable edge" }
    json.dump(res, open(RESULT,"w"), indent=2)
    print(f"\n[base2m] VERDICT: {'KILLED' if res['verdict']['KILLED'] else 'SURVIVED'} "
          f"(val_auc={val_auc:.4f}; COMBINED-clears={clears}) -> {RESULT}  total={time.time()-t0:.0f}s", flush=True)

if __name__=="__main__":
    main()
