"""USDCAD 15-MINUTE binary DIRECTION — single-pair BASELINE (foundation for the 15m sweep).

SCOPE: USDCAD · 15m. The foundational baseline for the (USDCAD,15m,UP) and (USDCAD,15m,DOWN) keys.
Fork of audusd_15m_base.py (itself the usdjpy_15m H=15 template) with PAIR=USDCAD. 15m is the
DERIV-FX-DEPLOYABLE minimum expiry (deriv forex Rise/Fall floor = 15m) AND historically the program's
strongest direction horizon (EURUSD 15m xpair certified BOTH sides p10 .567/.574; USDJPY 15m NY own-pair
seed-ens p10 .60/.57; AUDUSD 15m NY own-pair seed-ens p10 .596/.596; GBPUSD 15m xpair-NY seed-ens K=8
p10 .655/.640). The usdcad_15m_* method suite imports build/side_eval/covcurve/nonoverlap_chrono/boot/
mk_lgb from HERE.

MECHANISTIC PRIOR (v0 — USDCAD is a doubly-determined pair). USDCAD is a **USD-BASE / CAD-QUOTE** pair
(USDCAD UP ⇒ USD strengthens vs CAD; DOWN ⇒ CAD strengthens). It is simultaneously:
  (1) a **USD major** with USD on the NUMERATOR (like USDJPY) — loads on the USD common factor, so cross-pair
      USD-residual POOLING is a candidate lever, BUT both my closest analogs were OWN-PAIR-SPECIFIC: USDJPY
      (USD-base) and AUDUSD (commodity) both DILUTED under pooling. Strong prior: USDCAD is own-pair-specific
      too → pooling NULL. RUN it anyway (don't argue). NOTE the USD sign FLIPS vs AUDUSD/NZDUSD (USD in the
      denominator there) — any cross-pair residual must be sign-aligned carefully.
  (2) a **COMMODITY / petrocurrency** — CAD is oil-linked (WTI down ⇒ CAD weak ⇒ USDCAD UP). Oil/WTI is the
      dominant external signed driver, OFF-DISK (acquisition frontier, like AUDUSD's iron-ore/rate-diff).
  (3) the **MOST North-American pair** (both legs trade US/Canada hours; BoC + Fed + US/CA data + oil all hit
      in the LDN/NY window) → the NY-session-concentration prior is STRONGER here than for any prior pair.
      Asia is a very low prior for CAD (it barely trades in Tokyo).

SIDE-ASYMMETRY PREDICTION (falsifiable): for AUDUSD (AUD base) the DOWN side (= risk-off / USD-strength,
the sharp directional move) was the more forecastable. USDCAD has USD on the OPPOSITE side of the quote, so
the analogous sharp risk-off / oil-down / USD-strength move is USDCAD **UP**. Prediction: USDCAD UP is the
more forecastable/robust side (inverse of AUDUSD). Tested in the side-split below.

Bar-based path. Uses the 239 multi-TF base features in features/USDCAD_<year>.parquet. The 15-min label is
RE-DERIVED here from the faithful `close` series:
  label_t = sign(close[t+15]-close[t]),  requiring ts[t+15]-ts[t]==900  (15 clean 60s steps, no gap),
ties (move==0) EXCLUDED from train/AUC but counted as LOSSES for the tradeable win-rate.

Deriv-faithful WITHIN bar resolution: entry≈close[t], exit≈close[t+15] (bar-close approximation; at 15m the
next-tick entry latency is negligible — faithful proxy, validated on USDJPY/AUDUSD via true tick settlement).
Breakeven win-rate 0.541 (deriv payout R~1.85). de-overlap = nonoverlap_chrono gap=900s (live first-come, no
look-ahead). Per-year held-out (2024/2025/2026=strict OOS) with bootstrap CI95. Selection on VAL by
WORST-half stability (never VAL-acc-max). COMBINED + UP-split + DOWN-split reported separately.

Pre-registered falsifier written to usdcad_15m_base_result.json BEFORE the held-out read.

Usage: ~/binary-algo-venv/bin/python usdcad_15m_base.py [stride] [leaves]
"""
import os, sys, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H

PAIR = "USDCAD"; HOR = 15; STEP = 60; GAP = HOR*STEP; BE = 0.541   # H=15 -> 900s horizon, gap 900s
FEAT = H.FEAT_DIR
SPL = {"train":[str(y) for y in range(2012,2022)], "val":["2022","2023"],
       "test24":["2024"], "test25":["2025"], "oos":["2026"]}
# defaults reproduce the recorded baseline; override via argv: python usdcad_15m_base.py <stride> <leaves>
def _argint(i, default): return int(sys.argv[i]) if len(sys.argv)>i and str(sys.argv[i]).isdigit() else default
TR_STRIDE = _argint(1, 6)       # 15m drops more rows at gaps; stride 6 keeps a generous train
NUM_LEAVES = _argint(2, 127)
RESULT = "usdcad_15m_base_result.json" if (TR_STRIDE==6 and NUM_LEAVES==127) else f"usdcad_15m_base_s{TR_STRIDE}_l{NUM_LEAVES}_result.json"

FEATS = H.feature_cols(PAIR)   # 239 base features

def build(years, stride=1):
    """Per-year: 239 feats + 15-min contiguous label. Returns X(df), y(1=up/0=down on moved),
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
    res={"key":"USDCAD.15m", "model":"single-pair LGBM baseline (239 base feats, 15-min own-clock label)",
         "settlement":"bar-close approx, ties LOSE, breakeven 0.541, gap=900s nonoverlap_chrono",
         "splits":SPL, "tr_stride":TR_STRIDE, "num_leaves":NUM_LEAVES,
         "falsifier":{"registered_utc":"pre-OOS",
            "KILL_if":"VAL moved-AUC <= 0.515  OR  no held-out year COMBINED win-rate CI95-lower clears 0.541",
            "rationale":"USDCAD is a USD-base/CAD-quote major (USD common factor -> cross-pair pooling candidate, "
                        "but USDJPY+AUDUSD analogs were own-pair-specific = strong NULL prior) AND a petrocurrency "
                        "(oil/WTI = dominant signed driver, OFF-DISK) AND the most North-American pair (NY-session "
                        "concentration = strongest prior of any pair). Side-asymmetry prediction: USDCAD UP "
                        "(=USD-strength/risk-off/oil-down) is the more forecastable side (inverse of AUDUSD, whose "
                        "USD sits in the denominator). Measure the SINGLE-PAIR all-session base floor + moved-AUC "
                        "ceiling first; session + pooling are the levers tested next. 15m = deriv-FX floor + "
                        "program's best direction horizon."}}
    json.dump(res, open(RESULT,"w"), indent=2)

    Xtr,ytr,mtr,_=build(SPL["train"], TR_STRIDE)
    Xva,yva,mva,tsv=build(SPL["val"])
    itr=mtr; iva=mva                                  # train/early-stop on MOVED bars (ties excluded)
    print(f"[base15m USDCAD] stride={TR_STRIDE} leaves={NUM_LEAVES} train={int(itr.sum()):,} val={int(iva.sum()):,} feats={len(FEATS)} build={time.time()-t0:.0f}s", flush=True)
    L=mk_lgb(num_leaves=NUM_LEAVES)
    L.fit(Xtr[itr], ytr[itr], eval_set=[(Xva[iva], yva[iva])], eval_metric="auc",
          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    pva=L.predict_proba(Xva)[:,1]
    val_auc=float(roc_auc_score(yva[iva], pva[iva]))
    print(f"[base15m USDCAD] best_iter={L.best_iteration_} VAL moved-AUC={val_auc:.4f} {time.time()-t0:.0f}s", flush=True)
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
    print(f"[base15m USDCAD] FROZEN gate cov{COV:.0%} thr={THR:.4f} VAL worst-half WR={worst_half_wr:.4f} (halves {halfaccs})", flush=True)

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
    print(f"\n[base15m USDCAD] VERDICT: {'KILLED' if res['verdict']['KILLED'] else 'SURVIVED'} "
          f"(val_auc={val_auc:.4f}; COMBINED-clears={clears}) -> {RESULT}  total={time.time()-t0:.0f}s", flush=True)

if __name__=="__main__":
    main()
