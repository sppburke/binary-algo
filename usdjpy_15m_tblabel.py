"""USDJPY 15-MIN — TRAIN-LABEL sharpening (TN1: triple-barrier / averaged-horizon labels).

SCOPE: USDJPY · 15m. The own-pair base AUC is signal-capped ~.531 under the raw endpoint label
sign(close[t+15]-close[t]). Hypothesis (López de Prado AFML; Prata et al. 2024; arXiv:2504.02249): the
ENDPOINT label is noisy (a single forward mid); a path-aware TRAIN label denoises the sign target so the
model learns a cleaner sign map → higher AUC → higher selective win-rate.

CRITICAL DISCIPLINE: only the TRAINING label changes. The VAL gate selection and ALL held-out EVALUATION
use the UNCHANGED deriv-faithful fixed-15m label (sign(close[t+15]-close[t]), ties LOSE) — because deriv
settles at exactly the 15m endpoint. A train-label that lifts held-out fixed-15m win-rate is a real edge;
one that only lifts its own surrogate metric is not.

Modes:
  avg  : train_label = sign(mean(close[t+1..t+15]) - close[t])     (averaged-horizon, denoised endpoint)
  tb   : triple-barrier first-touch with barriers ±LAMBDA*rolling_std(ret,window); timeout->endpoint sign

Usage: ~/binary-algo-venv/bin/python usdjpy_15m_tblabel.py <avg|tb> [stride] [leaves] [lambda]
"""
import os, sys, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from usdjpy_15m_base import side_eval, covcurve, nonoverlap_chrono, boot, mk_lgb, BE, SPL, FEATS

PAIR="USDJPY"; HOR=15; STEP=60; GAP=HOR*STEP; FEAT=H.FEAT_DIR
MODE = sys.argv[1] if len(sys.argv)>1 and sys.argv[1] in ("avg","tb") else "avg"
def _argint(i,d):
    rest=[a for a in sys.argv[2:] if a.replace('.','',1).isdigit()]
    return rest[i] if len(rest)>i else d
TR_STRIDE=int(_argint(0,6)); NUM_LEAVES=int(_argint(1,127)); LAMBDA=float(_argint(2,1.0))
RESULT=f"usdjpy_15m_tblabel_{MODE}_s{TR_STRIDE}_l{NUM_LEAVES}_result.json"

def build(years, stride=1, label_mode="eval"):
    """Returns X(df), y_train (mode label), fr (fixed-15m fwd ret = EVAL label src), moved, ts.
    label_mode 'eval' => y is fixed-15m sign (for VAL/eval). else MODE-specific train label."""
    Xs=[]; ytr=[]; frs=[]; tss=[]
    for y in years:
        p=f"{FEAT}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p, columns=FEATS+["close"]); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float)
        ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        contig=np.zeros(n,bool); contig[:n-HOR]=(ts[HOR:]-ts[:-HOR])==GAP
        fr=np.full(n,np.nan); fr[:n-HOR]=c[HOR:]/c[:-HOR]-1.0          # fixed-15m fwd ret (EVAL)
        # train label per mode
        if label_mode=="avg":
            avg=np.full(n,np.nan)
            for k in range(1,HOR+1):
                pass
            # mean of close[t+1..t+HOR] via cumulative sum (vectorized)
            cs=np.cumsum(c)
            mean_fwd=np.full(n,np.nan)
            mean_fwd[:n-HOR]=(cs[HOR:]-cs[:-HOR])/HOR                  # mean(c[t+1..t+HOR])
            ylab=(mean_fwd> c).astype(float); ylab[~np.isfinite(mean_fwd)]=np.nan
        elif label_mode=="tb":
            # rolling std of 1-bar returns (causal), barrier = LAMBDA*std
            r1=np.full(n,np.nan); r1[1:]=c[1:]/c[:-1]-1.0
            sig=pd.Series(r1).rolling(900,min_periods=60).std().values    # ~15h vol window
            ylab=np.full(n,np.nan)
            for t in range(n-HOR):
                if not np.isfinite(sig[t]) or sig[t]<=0: continue
                up=c[t]*(1+LAMBDA*sig[t]); dn=c[t]*(1-LAMBDA*sig[t])
                seg=c[t+1:t+HOR+1]
                hit_up=np.where(seg>=up)[0]; hit_dn=np.where(seg<=dn)[0]
                tu=hit_up[0] if len(hit_up) else 10**9; td=hit_dn[0] if len(hit_dn) else 10**9
                if tu<td: ylab[t]=1.0
                elif td<tu: ylab[t]=0.0
                else: ylab[t]=float(c[t+HOR]>c[t])                     # timeout -> endpoint sign
        else:  # eval = fixed-15m sign
            ylab=(fr>0).astype(float); ylab[~np.isfinite(fr)]=np.nan
        X=d[FEATS].astype("float32"); keepf=X.isna().mean(axis=1).values<0.5
        valid=contig & np.isfinite(fr) & keepf & np.isfinite(ylab)
        idx=np.where(valid)[0]
        if stride>1: idx=idx[::stride]
        Xs.append(X.iloc[idx]); ytr.append(ylab[idx].astype(int)); frs.append(fr[idx]); tss.append(ts[idx])
    X=pd.concat(Xs); fr=np.concatenate(frs)
    return X, np.concatenate(ytr), fr, (fr!=0.0), np.concatenate(tss)

def main():
    t0=time.time()
    res={"key":"USDJPY.15m","model":f"single-pair base GBM, TRAIN-LABEL={MODE} (lambda={LAMBDA}), EVAL=fixed-15m deriv label",
         "label_mode":MODE,"lambda":LAMBDA,"tr_stride":TR_STRIDE,"num_leaves":NUM_LEAVES,
         "settlement":"EVAL=bar-close fixed-15m, ties LOSE, BE 0.541, gap=900s; TRAIN label denoised","splits":SPL,
         "falsifier":{"registered_utc":"pre-OOS",
            "KILL_if":"VAL fixed-15m moved-AUC <= base .531 (no AUC lift from relabeling) OR no held-out year (COMB/UP) CI-lo clears 0.541",
            "rationale":"raw-endpoint base AUC capped ~.531. Path-aware train label may denoise the sign target. EVAL stays "
                        "deriv-faithful fixed-15m; only a held-out fixed-15m lift counts."}}
    json.dump(res,open(RESULT,"w"),indent=2)

    # TRAIN on mode label; build VAL/eval with fixed-15m label for selection+eval
    Xtr,ytr,frtr,mtr,_=build(SPL["train"], TR_STRIDE, MODE)
    Xva,yva_tr,frva,mva,tsv=build(SPL["val"], 1, MODE)        # yva_tr=mode label (unused for eval)
    yva_eval=(frva>0).astype(int)                            # fixed-15m eval label
    itr=mtr; iva=mva
    print(f"[tbl/{MODE}] s{TR_STRIDE} l{NUM_LEAVES} lam{LAMBDA} train={int(itr.sum()):,} val={int(iva.sum()):,} build={time.time()-t0:.0f}s",flush=True)
    L=mk_lgb(num_leaves=NUM_LEAVES)
    L.fit(Xtr[itr], ytr[itr], eval_set=[(Xva[iva], yva_eval[iva])], eval_metric="auc",
          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    pva=L.predict_proba(Xva)[:,1]
    val_auc=float(roc_auc_score(yva_eval[iva], pva[iva]))     # AUC vs FIXED-15m label
    print(f"[tbl/{MODE}] best_iter={L.best_iteration_} VAL fixed15m moved-AUC={val_auc:.4f} (base .531) {time.time()-t0:.0f}s",flush=True)
    res["val_auc"]=val_auc; res["best_iter"]=int(L.best_iteration_ or 0)

    half=len(pva)//2; confv=np.abs(pva-0.5); best=None
    for cov in (0.20,0.10,0.05,0.03,0.02):
        thr=float(np.quantile(confv,1-cov)); accs=[]
        for s,e in ((0,half),(half,len(pva))):
            r=side_eval(pva[s:e],yva_eval[s:e],mva[s:e],tsv[s:e],thr); accs.append(r["COMBINED"]["wr"] if r else float("nan"))
        worst=np.nanmin(accs)
        if best is None or worst>best[0]: best=(worst,cov,thr,accs)
    worst_half,COV,THR,halfaccs=best
    res["gate"]={"cov":COV,"conf_thr":THR,"val_worst_half_wr":float(worst_half)}
    print(f"[tbl/{MODE}] FROZEN gate cov{COV:.0%} thr={THR:.4f} VAL worst-half WR={worst_half:.4f}",flush=True)

    res["years"]={}
    for w in ("test24","test25","oos"):
        Xw,_,frw,mw,tsw=build(SPL[w], 1, MODE)
        yw=(frw>0).astype(int)
        pr=L.predict_proba(Xw)[:,1]
        auc=float(roc_auc_score(yw[mw],pr[mw])); up_rate=float(yw[mw].mean())
        gate=side_eval(pr,yw,mw,tsw,THR); cc=covcurve(pr,yw,mw,tsw)
        res["years"][w]={"moved_auc":auc,"moved_up_rate":up_rate,"gate":gate,"covcurve":cc,
                         "tripwire_ok":bool(0.46<=up_rate<=0.54)}
        g=gate["COMBINED"] if gate else {"n":0,"wr":float('nan'),"ci":[float('nan')]*2}
        u=gate["UP"] if gate else {"n":0,"wr":float('nan'),"ci":[float('nan')]*2}
        d=gate["DOWN"] if gate else {"n":0,"wr":float('nan'),"ci":[float('nan')]*2}
        print(f"=== {w} === AUC={auc:.4f} up={up_rate:.4f} cov{COV:.0%}: COMB n{g['n']} {g['wr']:.4f} CI[{g['ci'][0]:.3f},{g['ci'][1]:.3f}] | "
              f"UP n{u['n']} {u['wr']:.4f} CI[{u['ci'][0]:.3f},{u['ci'][1]:.3f}] | DOWN n{d['n']} {d['wr']:.4f} CI[{d['ci'][0]:.3f},{d['ci'][1]:.3f}]",flush=True)

    up_clears=[w for w in ("test24","test25","oos") if res["years"][w]["gate"] and res["years"][w]["gate"]["UP"]["ci"][0]>=BE]
    dn_clears=[w for w in ("test24","test25","oos") if res["years"][w]["gate"] and res["years"][w]["gate"]["DOWN"]["ci"][0]>=BE]
    comb_clears=[w for w in ("test24","test25","oos") if res["years"][w]["gate"] and res["years"][w]["gate"]["COMBINED"]["ci"][0]>=BE]
    res["verdict"]={"val_auc":val_auc,"beats_base_auc":bool(val_auc>0.5313),
                    "UP_years_CIlo_clears_BE":up_clears,"DOWN_years_CIlo_clears_BE":dn_clears,"COMBINED_years_CIlo_clears_BE":comb_clears,
                    "KILLED":bool(val_auc<=0.515 or (len(up_clears)==0 and len(dn_clears)==0 and len(comb_clears)==0))}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[tbl/{MODE}] VERDICT: {'KILLED' if res['verdict']['KILLED'] else 'SURVIVED'} beats_base_AUC={res['verdict']['beats_base_auc']} "
          f"(val_auc={val_auc:.4f}; UP={up_clears}; DOWN={dn_clears}; COMB={comb_clears}) -> {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
