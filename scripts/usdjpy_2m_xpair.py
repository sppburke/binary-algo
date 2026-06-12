"""USDJPY 2-MIN — CROSS-PAIR POOLED training (C1: the 30m 94%-of-gain lever, retargeted to 120s).

SCOPE: USDJPY · 2m. The single highest-prior on-disk improve lever. The EURUSD 30m result showed the
cross-pair edge came ~94% from POOLING the TRAINING ROWS across the 7 majors (base features) — NOT from
cross-pair OF/residual factors (~6%). Mechanism: the 239 base features are pair-agnostic (normalized rv /
bb_width / rangepos / macd / session), so a model trained on {all 7 majors' base features → each pair's OWN
2m-forward sign} learns the GENERIC base-feature→direction map with 7x data + cross-pair regime diversity,
then transfers to USDJPY. Decorrelates pair-idiosyncratic noise; attacks the data-starvation cause behind the
binding-year (2025) collapse seen in the single-pair base.

Pool = train rows from EURUSD,GBPUSD,AUDUSD,NZDUSD,USDJPY,USDCHF,USDCAD (each labeled by its OWN 2m sign,
own-clock contiguous, ties excluded from train/AUC). Early-stop + gate-select + held-out eval are ALL on the
USDJPY TARGET only (deploy on USDJPY). Same discipline as usdjpy_2m_base (worst-VAL-half gate, nonoverlap
gap=120, per-year CI95, moved up-rate tripwire). Falsifier pre-registered.

Usage: ~/binary-algo-venv/bin/python usdjpy_2m_xpair.py pool [stride_per_pair=42] [leaves=255]
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from usdjpy_2m_base import side_eval, covcurve, nonoverlap_chrono, boot, mk_lgb

TARGET="USDJPY"; HOR=2; STEP=60; GAP=HOR*STEP; BE=0.541
PAIRS=["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCHF","USDCAD"]
FEAT=H.FEAT_DIR; FEATS=H.feature_cols(TARGET)   # canonical 239 base feats (same set across majors)
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],
     "test24":["2024"],"test25":["2025"],"oos":["2026"]}

def build_pair(pair, years, stride=1):
    """Base feats + 2-min own-clock label for ONE pair. label_t=sign(close[t+HOR]-close[t]),
    requires ts[t+HOR]-ts[t]==GAP. Returns X(df), y, moved(bool), ts(int64)."""
    Xs=[]; ys=[]; mv=[]; tss=[]
    for y in years:
        p=f"{FEAT}/{pair}_{y}.parquet"
        if not os.path.exists(p): continue
        cols=[c for c in FEATS if True]+["close"]
        d=pd.read_parquet(p, columns=cols)
        d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float)
        ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        contig=np.zeros(n,bool); contig[:n-HOR]=(ts[HOR:]-ts[:-HOR])==GAP
        fr=np.full(n,np.nan); fr[:n-HOR]=c[HOR:]/c[:-HOR]-1.0
        X=d[FEATS].astype("float32")
        keepf=X.isna().mean(axis=1).values<0.5
        valid=contig & np.isfinite(fr) & keepf
        moved=valid & (fr!=0.0)
        idx=np.where(valid)[0]
        if stride>1: idx=idx[::stride]
        Xs.append(X.iloc[idx]); ys.append((fr[idx]>0).astype(int))
        mv.append(moved[idx]); tss.append(ts[idx])
    if not Xs: return None
    return pd.concat(Xs), np.concatenate(ys), np.concatenate(mv), np.concatenate(tss)

def build_pool(years, stride):
    Xs=[]; ys=[]; mv=[]
    for p in PAIRS:
        r=build_pair(p, years, stride)
        if r is None: continue
        X,y,m,_=r; Xs.append(X); ys.append(y); mv.append(m)
    return pd.concat(Xs), np.concatenate(ys), np.concatenate(mv)

def main(mode="pool", stride=42, leaves=255):
    t0=time.time(); RESULT=f"usdjpy_2m_xpair_{mode}_s{stride}_l{leaves}_result.json"
    res={"key":"USDJPY.2m","model":f"cross-pair POOLED base-GBM (7 majors, {mode}), eval USDJPY @2m",
         "stride_per_pair":stride,"num_leaves":leaves,"pairs":PAIRS,
         "settlement":"bar-close approx, ties LOSE, breakeven 0.541, gap=120s nonoverlap_chrono","splits":SPL,
         "falsifier":{"registered_utc":"pre-OOS",
            "KILL_if":"VAL(USDJPY) moved-AUC <= 0.515 OR no held-out USDJPY year (COMBINED or UP) CI95-lower clears 0.541 at the gate",
            "incumbent":"single-pair 2m base UP cov2% .546/.525/.527 (worst 2025 .524 sub-BE); beat = lift a worst-year CI-lo >= 0.541"}}
    json.dump(res,open(RESULT,"w"),indent=2)

    # pooled TRAIN; TARGET-only VAL for early stopping + gate selection
    Xtr,ytr,mtr=build_pool(SPL["train"], stride)
    Xva,yva,mva,tsv=build_pair(TARGET, SPL["val"], 1)
    print(f"[xpair/{mode}] pooled-train={int(mtr.sum()):,} (7 majors) USDJPY-val={int(mva.sum()):,} feats={len(FEATS)} build={time.time()-t0:.0f}s",flush=True)
    L=mk_lgb(num_leaves=leaves)
    L.fit(Xtr[mtr], ytr[mtr], eval_set=[(Xva[mva], yva[mva])], eval_metric="auc",
          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    pva=L.predict_proba(Xva)[:,1]; val_auc=float(roc_auc_score(yva[mva], pva[mva]))
    print(f"[xpair/{mode}] best_iter={L.best_iteration_} USDJPY-VAL moved-AUC={val_auc:.4f} {time.time()-t0:.0f}s",flush=True)
    res["val_auc"]=val_auc; res["best_iter"]=int(L.best_iteration_ or 0)

    # VAL worst-half gate (USDJPY target)
    half=len(pva)//2; confv=np.abs(pva-0.5); best=None
    for cov in (0.20,0.10,0.05,0.03,0.02):
        thr=float(np.quantile(confv,1-cov)); accs=[]
        for s,e in ((0,half),(half,len(pva))):
            r=side_eval(pva[s:e],yva[s:e],mva[s:e],tsv[s:e],thr); accs.append(r["COMBINED"]["wr"] if r else float("nan"))
        worst=np.nanmin(accs)
        if best is None or worst>best[0]: best=(worst,cov,thr,accs)
    worst_half,COV,THR,halfaccs=best
    res["gate"]={"cov":COV,"conf_thr":THR,"val_worst_half_wr":float(worst_half),"val_half_wrs":[float(a) for a in halfaccs]}
    print(f"[xpair/{mode}] FROZEN gate cov{COV:.0%} thr={THR:.4f} VAL worst-half WR={worst_half:.4f}",flush=True)

    res["years"]={}
    for w in ("test24","test25","oos"):
        Xw,yw,mw,tsw=build_pair(TARGET, SPL[w], 1)
        pr=L.predict_proba(Xw)[:,1]
        auc=float(roc_auc_score(yw[mw],pr[mw])); up_rate=float(yw[mw].mean())
        gate=side_eval(pr,yw,mw,tsw,THR); cc=covcurve(pr,yw,mw,tsw)
        res["years"][w]={"moved_auc":auc,"moved_up_rate":up_rate,"gate":gate,"covcurve":cc,
                         "tripwire_ok":bool(0.47<=up_rate<=0.53)}
        g=gate["COMBINED"] if gate else {"n":0,"wr":float('nan'),"ci":[float('nan')]*2}
        u=gate["UP"] if gate else {"n":0,"wr":float('nan'),"ci":[float('nan')]*2}
        d=gate["DOWN"] if gate else {"n":0,"wr":float('nan'),"ci":[float('nan')]*2}
        print(f"=== {w} === AUC={auc:.4f} up-rate={up_rate:.4f} cov{COV:.0%}: COMB n{g['n']} wr={g['wr']:.4f} CI[{g['ci'][0]:.3f},{g['ci'][1]:.3f}] | "
              f"UP n{u['n']} wr={u['wr']:.4f} CI[{u['ci'][0]:.3f},{u['ci'][1]:.3f}] | DOWN n{d['n']} wr={d['wr']:.4f} CI[{d['ci'][0]:.3f},{d['ci'][1]:.3f}]",flush=True)

    up_clears=[w for w in ("test24","test25","oos") if res["years"][w]["gate"] and res["years"][w]["gate"]["UP"]["ci"][0]>=BE]
    comb_clears=[w for w in ("test24","test25","oos") if res["years"][w]["gate"] and res["years"][w]["gate"]["COMBINED"]["ci"][0]>=BE]
    res["verdict"]={"val_auc_le_0515":bool(val_auc<=0.515),"UP_years_CIlo_clears_BE":up_clears,
                    "COMBINED_years_CIlo_clears_BE":comb_clears,
                    "KILLED":bool(val_auc<=0.515 or (len(up_clears)==0 and len(comb_clears)==0))}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[xpair/{mode}] VERDICT: {'KILLED' if res['verdict']['KILLED'] else 'SURVIVED'} "
          f"(val_auc={val_auc:.4f}; UP-clears={up_clears}; COMB-clears={comb_clears}) -> {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 and not sys.argv[1].isdigit() else "pool"
    rest=[a for a in sys.argv[1:] if a.isdigit()]
    stride=int(rest[0]) if len(rest)>0 else 42
    leaves=int(rest[1]) if len(rest)>1 else 255
    main(mode, stride, leaves)
