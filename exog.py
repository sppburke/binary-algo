"""Exogenous features from the OTHER 6 USD pairs, tailored for 15-min prediction:
 (A) richer per-peer LONGER-horizon features (15m/30m/1h trend, momentum, stretch, vol),
 (B) constructed cross-pair factors: USD-strength basket, EURUSD idiosyncratic residual vs the
     basket (+ z-scores at 15/30/60m) = the stat-arb mean-reversion signal, and peer dispersion.
All causal (info <= t); aligned to EURUSD's 1m grid."""
import os, numpy as np, pandas as pd
import harness as H
FEAT=H.FEAT_DIR
PEERS=["GBPUSD","AUDUSD","NZDUSD","USDCAD","USDCHF","USDJPY"]
ALL=["EURUSD"]+PEERS
USD_BASE={"USDCAD":-1.0,"USDCHF":-1.0,"USDJPY":-1.0}  # invert USDxxx so + == USD weak (EUR-aligned)
PEER_COLS=["5m_ret_3","15m_ret_1","15m_ret_3","15m_dist_ema20","15m_above_ema50",
           "30m_ret_1","30m_ret_3","1h_ret_1","15m_rv_24","15m_autocorr_10","15m_bb_pctb","1h_dist_ema20"]

def _close(pair,year):
    p=f"{FEAT}/{pair}_{year}.parquet"
    if not os.path.exists(p): return None
    s=pd.read_parquet(p,columns=["close"])["close"]
    return s[~s.index.duplicated(keep="last")]

def load_exog_year(year, base_index):
    """Return exogenous feature DataFrame aligned to base_index (EURUSD 1m index for `year`)."""
    out=[]
    # (A) richer peer features
    for pr in PEERS:
        p=f"{FEAT}/{pr}_{year}.parquet"
        if not os.path.exists(p): continue
        pf=pd.read_parquet(p,columns=PEER_COLS)
        pf=pf[~pf.index.duplicated(keep="last")].reindex(base_index,method="ffill",limit=3)
        pf.columns=[f"X{pr[:6]}_{c}" for c in pf.columns]
        out.append(pf)
    # (B) constructed factors
    closes={}
    for pair in ALL:
        s=_close(pair,year)
        if s is not None: closes[pair]=s.reindex(base_index,method="ffill",limit=3)
    C=pd.DataFrame(closes)
    lret=np.log(C).diff()
    oriented=lret.copy()
    for pair,sgn in USD_BASE.items():
        if pair in oriented: oriented[pair]*=sgn
    basket=oriented.mean(axis=1)              # USD-strength factor (EUR-aligned)
    eur=lret["EURUSD"]
    resid=eur-basket                           # idiosyncratic EURUSD move vs basket
    F=pd.DataFrame(index=base_index)
    for w in (5,15,30,60):
        F[f"FAC_basket_{w}"]=basket.rolling(w).sum()
        cum=resid.rolling(w).sum(); F[f"FAC_resid_{w}"]=cum
        F[f"FAC_resid_z_{w}"]=(cum-cum.rolling(240).mean())/(cum.rolling(240).std()+1e-9)
    F["FAC_disp_15"]=oriented.std(axis=1).rolling(15).mean()
    F["FAC_disp_30"]=oriented.std(axis=1).rolling(30).mean()
    F["FAC_eur_basket_gap_15"]=basket.rolling(15).sum()-eur.rolling(15).sum()
    F["FAC_eur_basket_gap_30"]=basket.rolling(30).sum()-eur.rolling(30).sum()
    out.append(F)
    res=pd.concat(out,axis=1)
    return res.replace([np.inf,-np.inf],np.nan).astype("float32")

def exog_feature_names():
    names=[f"X{pr[:6]}_{c}" for pr in PEERS for c in PEER_COLS]
    for w in (5,15,30,60): names+=[f"FAC_basket_{w}",f"FAC_resid_{w}",f"FAC_resid_z_{w}"]
    names+=["FAC_disp_15","FAC_disp_30","FAC_eur_basket_gap_15","FAC_eur_basket_gap_30"]
    return names

if __name__=="__main__":
    import time
    base=pd.read_parquet(f"{FEAT}/EURUSD_2015.parquet",columns=["close"])
    base=base[~base.index.duplicated(keep="last")]
    t=time.time(); E=load_exog_year("2015",base.index)
    print("exog shape",E.shape,"time",round(time.time()-t,1),"s","ncols",E.shape[1])
    print("nan frac",round(float(E.iloc[3000:].isna().mean().mean()),3))
    print("names match:",len(exog_feature_names())==E.shape[1], E.shape[1], len(exog_feature_names()))
