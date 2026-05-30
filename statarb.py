"""Statistical-arbitrage residual features (NEW methodology).
All 7 pairs share USD => a strong common factor. Decompose each pair's return into:
  common (USD-factor) component  +  idiosyncratic residual.
The common component is ~random walk (unpredictable); the IDIOSYNCRATIC residual is
stationary & mean-reverting. EURUSD's cumulative idiosyncratic drift (z-scored) should
REVERT -> a stronger, cleaner directional signal than raw price reversion.

We build, causally, at the 1m grid:
  - PCA/basket factor from the 7 aligned returns (loadings fit on TRAIN only).
  - EURUSD idiosyncratic residual, cumulative over windows, z-scored (the fade signal).
  - basket-implied EURUSD drift (relative-value).
"""
import os, numpy as np, pandas as pd
import harness as H

FEAT_DIR=H.FEAT_DIR
PAIRS=["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDCAD","USDCHF","USDJPY"]
# sign so that "+ return" == "USD weakens" for every pair (USDXXX inverted)
USD_BASE={"USDCAD":-1.0,"USDCHF":-1.0,"USDJPY":-1.0}  # base=USD -> invert to get USD-strength-aligned

def load_close_matrix(years):
    cols=["close","valid"]
    series={}
    for p in PAIRS:
        parts=[]
        for y in years:
            fp=f"{FEAT_DIR}/{p}_{y}.parquet"
            if os.path.exists(fp): parts.append(pd.read_parquet(fp,columns=["close"]))
        if parts:
            s=pd.concat(parts)["close"]
            s=s[~s.index.duplicated(keep="last")]
            series[p]=s
    df=pd.DataFrame(series).sort_index()
    df=df[~df.index.duplicated(keep="last")]
    return df

def build_residual_features(close_df, betas=None, fit=False):
    """Returns residual feature frame for EURUSD aligned to close_df.index, plus betas."""
    lret=np.log(close_df).diff()
    # orient all to "anti-USD" so they're positively correlated (EUR up == USD down)
    oriented=lret.copy()
    for p,s in USD_BASE.items():
        if p in oriented: oriented[p]=oriented[p]*s
    # common USD factor = cross-sectional mean of oriented returns (equal-weight basket)
    factor=oriented.mean(axis=1)
    out={}
    eur=lret["EURUSD"]
    # static beta of EURUSD on factor (fit on train window)
    if fit or betas is None:
        mask=eur.notna()&factor.notna()
        b=np.polyfit(factor[mask].values, eur[mask].values, 1)
        betas={"slope":float(b[0]),"intercept":float(b[1])}
    pred=betas["slope"]*factor+betas["intercept"]
    resid=eur-pred   # EURUSD idiosyncratic 1m return
    # cumulative idiosyncratic drift over windows, z-scored (stretch -> fade)
    for w in (5,15,30,60):
        cum=resid.rolling(w).sum()
        z=(cum-cum.rolling(240).mean())/(cum.rolling(240).std()+1e-9)
        out[f"resid_cum_{w}"]=cum
        out[f"resid_z_{w}"]=z
    out["resid_1"]=resid
    out["factor_cum_5"]=factor.rolling(5).sum()
    out["factor_cum_15"]=factor.rolling(15).sum()
    # basket-implied: if basket says USD weakening (factor>0) but EURUSD lagged, expect EUR up
    out["rv_gap_5"]=factor.rolling(5).sum()-eur.rolling(5).sum()
    out["rv_gap_15"]=factor.rolling(15).sum()-eur.rolling(15).sum()
    F=pd.DataFrame(out,index=close_df.index).replace([np.inf,-np.inf],np.nan).astype("float32")
    return F, betas

STATARB_COLS=["resid_cum_5","resid_z_5","resid_cum_15","resid_z_15","resid_cum_30","resid_z_30",
              "resid_cum_60","resid_z_60","resid_1","factor_cum_5","factor_cum_15","rv_gap_5","rv_gap_15"]

if __name__=="__main__":
    tr=load_close_matrix([str(y) for y in range(2012,2022)])
    F,betas=build_residual_features(tr, fit=True)
    print("train residual feats",F.shape,"betas",betas)
    print("resid_z_15 std",float(F["resid_z_15"].std()),"resid_1 std",float(F["resid_1"].std()))
    # quick fade test on TRAIN: does cumulative idiosyncratic stretch revert?
    eurp=tr["EURUSD"]; fwd=np.log(eurp.shift(-5))-np.log(eurp); up=(fwd>0).astype(int)
    for w in (5,15,30,60):
        z=F[f"resid_z_{w}"]; m=z.notna()&fwd.notna()&(fwd!=0)
        # fade: z>0 (stretched up idiosyncratically) -> predict down
        for q in (0.0,0.9,0.95,0.99):
            thr=z[m].abs().quantile(q) if q>0 else 0
            sel=m&(z.abs()>=thr)
            pred=(z[sel]<0).astype(int); acc=(pred.values==up[sel].values).mean()
            print(f"  resid_z_{w} fade q{q}: n={int(sel.sum()):,} acc={acc:.4f}")
