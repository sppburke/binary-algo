"""USDJPY 15-MIN — TAR-VECM cointegration-velocity SIGN (TN4: the one intrinsically-directional new lever).

SCOPE: USDJPY · 15m. Engle-Granger cointegrating residual z_t of log(USDJPY) on log(6 other USD majors),
β estimated on TRAIN (frozen). The error-correction VELOCITY is signed: if USDJPY is RICH vs the basket
(z_t>0) it should revert DOWN → directional feature f_t = -z_t. Band-TAR regime gate |z_t|>c (reversion
only outside the no-arb band). Inputs = 7-major aligned closes (ALL on disk). Construction is econometrics-
canon (Engle-Granger + Balke-Fagan band-TAR), not corpus-sourced.

Two tests, EVAL = deriv-faithful fixed-15m sign (ties LOSE):
  (1) STANDALONE fast-KILL: does sign(-z_t) (optionally gated |z|>c) predict USDJPY 15m sign? moved-AUC + win-rate.
  (2) INTEGRATION: add [z, |z|, regime, ec_vel] to the 239 base feats, retrain NY GBM, VAL-AUC vs base .539.

Fast-KILL: no stable cointegration (ADF on z fails) OR standalone sign(-z) held-out AUC <= 0.51 OR no AUC lift.
Usage: ~/binary-algo-venv/bin/python usdjpy_15m_tarvecm.py
"""
import os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from sessions import session_mask
from usdjpy_15m_base import side_eval, covcurve, mk_lgb, BE, SPL, FEATS
try:
    from statsmodels.tsa.stattools import adfuller
    HAVE_SM=True
except Exception:
    HAVE_SM=False

PAIR="USDJPY"; HOR=15; GAP=900; FEAT=H.FEAT_DIR
BASKET=["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDCHF","USDCAD"]   # 6 others; USDJPY is target
RESULT="usdjpy_15m_tarvecm_result.json"

def load_close(pair):
    """epoch-second-indexed log-close series across all years (1-min bars)."""
    ss=[]
    for y in range(2012,2027):
        p=f"{FEAT}/{pair}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p, columns=["close"]); d=d[~d.index.duplicated(keep="last")]
        s=pd.Series(np.log(d["close"].values.astype(float)),
                    index=d.index.values.astype("datetime64[s]").astype("int64"))
        ss.append(s)
    return pd.concat(ss)

def build_usdjpy_bars(years):
    """USDJPY 15m candidate bars: ts, fwd_ret (fixed-15m), moved, plus the 239 base feats df indexed by ts."""
    Xs=[]; frs=[]; tss=[]
    for y in years:
        p=f"{FEAT}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p, columns=FEATS+["close"]); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float)
        ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        contig=np.zeros(n,bool); contig[:n-HOR]=(ts[HOR:]-ts[:-HOR])==GAP
        fr=np.full(n,np.nan); fr[:n-HOR]=c[HOR:]/c[:-HOR]-1.0
        X=d[FEATS].astype("float32"); keepf=X.isna().mean(axis=1).values<0.5
        valid=contig & np.isfinite(fr) & keepf
        idx=np.where(valid)[0]
        Xs.append(X.iloc[idx]); frs.append(fr[idx]); tss.append(ts[idx])
    return pd.concat(Xs), np.concatenate(frs), np.concatenate(tss)

def main():
    t0=time.time()
    res={"key":"USDJPY.15m","model":"TAR-VECM cointegration-velocity sign (Engle-Granger z_t of USDJPY vs 6-major basket)",
         "basket":BASKET,"eval":"deriv-faithful fixed-15m sign, ties LOSE, BE 0.541",
         "falsifier":{"registered_utc":"pre-OOS",
            "KILL_if":"ADF on train z not stationary OR standalone sign(-z) held-out moved-AUC <= 0.51 OR base+z VAL-AUC <= .539 (no lift)"}}
    json.dump(res, open(RESULT,"w"), indent=2)

    logc={p:load_close(p) for p in [PAIR]+BASKET}
    # train: estimate cointegrating beta on USDJPY decision bars (2012-21)
    Xtr,frtr,tstr=build_usdjpy_bars(SPL["train"])
    def basket_matrix(ts):
        cols=[]
        for p in BASKET:
            s=logc[p].reindex(ts)
            cols.append(s.values)
        B=np.column_stack(cols);
        return B
    ytr_lc=logc[PAIR].reindex(tstr).values
    Btr=basket_matrix(tstr)
    ok=np.isfinite(ytr_lc) & np.all(np.isfinite(Btr),axis=1)
    A=np.column_stack([np.ones(ok.sum()), Btr[ok]])
    coef,_,_,_=np.linalg.lstsq(A, ytr_lc[ok], rcond=None)   # [const, b1..b6]
    def resid(ts):
        lc=logc[PAIR].reindex(ts).values; B=basket_matrix(ts)
        z=lc - (coef[0] + B@coef[1:])
        return z
    ztr=resid(tstr)
    c_band=float(np.nanquantile(np.abs(ztr),0.5))   # band threshold = median |z| on train
    res["beta"]={"const":float(coef[0]),**{BASKET[i]:float(coef[i+1]) for i in range(6)}}
    res["band_c"]=c_band
    if HAVE_SM:
        zz=ztr[np.isfinite(ztr)]
        adf=adfuller(zz[::50] if len(zz)>20000 else zz, maxlag=20, autolag="AIC")
        res["adf_train"]={"stat":float(adf[0]),"pvalue":float(adf[1]),"stationary_5pct":bool(adf[1]<0.05)}
        print(f"[tarvecm] ADF z train: stat={adf[0]:.3f} p={adf[1]:.4f} stationary={adf[1]<0.05}  band_c={c_band:.5f}",flush=True)
    # ECM gamma: Δz_t = gamma * z_{t-1} (+ regime); fit on train (sign of gamma should be negative = reversion)
    dz=np.diff(ztr); zlag=ztr[:-1]; m=np.isfinite(dz)&np.isfinite(zlag)
    gamma=float(np.polyfit(zlag[m], dz[m], 1)[0])
    res["ecm_gamma"]=gamma
    print(f"[tarvecm] beta const={coef[0]:.3f} ECM gamma={gamma:.5f} (neg=reversion)  ({time.time()-t0:.0f}s)",flush=True)

    # ---- (1) standalone sign(-z) test per held-out year (NY-session, matching the cert regime) ----
    res["standalone"]={}
    for w in ("test24","test25","oos"):
        Xw,frw,tsw=build_usdjpy_bars(SPL[w])
        zw=resid(tsw); yw=(frw>0).astype(int); mw=(frw!=0.0)
        ny=session_mask(tsw,"ny")
        # signal: predict UP when z<0 (cheap vs basket -> rises). score = -z (higher => more UP)
        sc=-zw
        sel=ny & mw & np.isfinite(sc)
        if sel.sum()<50: res["standalone"][w]={"n":int(sel.sum()),"auc":None}; continue
        auc=float(roc_auc_score(yw[sel], sc[sel]))
        # gated win-rate: trade only |z|>c, side=sign(-z)
        gate=(np.abs(zw)>c_band)&sel
        pred=(sc[gate]>0).astype(int); win=(pred==yw[gate]).astype(float)
        res["standalone"][w]={"n":int(sel.sum()),"auc":round(auc,4),
                              "gated_n":int(gate.sum()),"gated_wr":round(float(win.mean()),4) if gate.sum()>0 else None}
        print(f"  [standalone {w}] NY sign(-z) moved-AUC={auc:.4f} | band-gated n{int(gate.sum())} wr={res['standalone'][w]['gated_wr']}",flush=True)

    standalone_auc=[v["auc"] for v in res["standalone"].values() if v.get("auc")]
    kill_standalone = (len(standalone_auc)==0) or (max(standalone_auc)<=0.51)

    # ---- (2) integration: add z-features to 239 base, retrain NY GBM, VAL-AUC vs base .539 ----
    def add_z(X, ts):
        z=resid(ts); zc=np.clip(z,-5*c_band,5*c_band)
        X=X.copy()
        X["cvecm_z"]=zc.astype("float32"); X["cvecm_absz"]=np.abs(zc).astype("float32")
        X["cvecm_regime"]=(np.abs(z)>c_band).astype("float32"); X["cvecm_ecvel"]=(gamma*z).astype("float32")
        return X
    Xva,frva,tsv=build_usdjpy_bars(SPL["val"]); yva=(frva>0).astype(int); mva=(frva!=0.0)
    nytr=session_mask(tstr,"ny"); nyva=session_mask(tsv,"ny")
    Xtr2=add_z(Xtr,tstr); Xva2=add_z(Xva,tsv)
    itr=(frtr!=0.0)&nytr; iva=mva&nyva
    L=mk_lgb(num_leaves=127)
    L.fit(Xtr2[itr], (frtr[itr]>0).astype(int), eval_set=[(Xva2[iva], yva[iva])], eval_metric="auc",
          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    val_auc=float(roc_auc_score(yva[iva], L.predict_proba(Xva2[iva])[:,1]))
    res["integration_val_ny_auc"]=round(val_auc,4)
    res["integration_beats_base"]=bool(val_auc>0.539)
    # feature importance of the z-features
    imp=dict(zip(L.booster_.feature_name(), L.booster_.feature_importance(importance_type="gain")))
    zimp={k:int(imp.get(k,0)) for k in ["cvecm_z","cvecm_absz","cvecm_regime","cvecm_ecvel"]}
    tot=sum(imp.values()) or 1; res["z_feat_gain_pct"]={k:round(100*v/tot,3) for k,v in zimp.items()}
    print(f"[tarvecm] INTEGRATION NY VAL-AUC={val_auc:.4f} (base .539) beats={val_auc>0.539}  z-feat gain%={res['z_feat_gain_pct']}",flush=True)

    res["verdict"]={"KILLED":bool((kill_standalone) and (val_auc<=0.539)),
                    "standalone_max_auc":round(max(standalone_auc),4) if standalone_auc else None,
                    "note":"directional iff standalone sign(-z) AUC>.51 OR integration lifts VAL-AUC>.539"}
    json.dump(res, open(RESULT,"w"), indent=2)
    print(f"\n[tarvecm] VERDICT: {'KILLED' if res['verdict']['KILLED'] else 'SURVIVED-some-signal'} "
          f"(standalone max AUC {res['verdict']['standalone_max_auc']}; integration VAL .{int(val_auc*10000)}) -> {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
