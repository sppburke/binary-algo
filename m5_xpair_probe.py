"""Cheap feasibility probe: does cross-pair / USD-common-factor lead-lag predict EURUSD NEXT-5min sign?
Single-column (close) load, 2 held-out years (2024 TEST, 2026 OOS). No training. Tier-1 sanity check before a full build.

Sign conventions (Tier-1 FX): EURUSD/GBPUSD/AUDUSD/NZDUSD = USD is QUOTE (pair up => USD down).
USDJPY/USDCHF/USDCAD = USD is BASE (pair up => USD up). USD-basket return excludes EURUSD (no trivial leakage).
EURUSD instantaneous move implied by USD = -USD_basket_ret.
"""
import sys, numpy as np, pandas as pd
FEAT="/home/sean/git/binary-algo/features"
PAIRS=["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCHF","USDCAD"]
USD_BASE={"USDJPY","USDCHF","USDCAD"}            # +ret contributes to USD strength
HOR=5

def load_year(y):
    cl={}
    for p in PAIRS:
        d=pd.read_parquet(f"{FEAT}/{p}_{y}.parquet",columns=["close"])
        d=d[~d.index.duplicated(keep="last")]
        cl[p]=d["close"]
    df=pd.DataFrame(cl).dropna()                 # inner-join aligned on common 1-min timestamps
    return df

def sign_agree(pred,target,mask):
    pred=pred[mask]; target=target[mask]
    ok=np.isfinite(pred)&np.isfinite(target)&(target!=0)&(pred!=0)
    pred=pred[ok]; target=target[ok]
    if len(pred)<500: return (np.nan,len(pred))
    return (float((np.sign(pred)==np.sign(target)).mean()), len(pred))

def spearman(pred,target,mask):
    pred=pred[mask]; target=target[mask]
    ok=np.isfinite(pred)&np.isfinite(target)
    pred=pred[ok]; target=target[ok]
    if len(pred)<500: return np.nan
    from scipy.stats import spearmanr
    return float(spearmanr(pred,target).correlation)

def probe(y):
    df=load_year(y)
    secs=df.index.values.astype("datetime64[s]").astype("int64")
    n=len(df)
    # 1-min log returns
    lr={p:np.log(df[p].values) for p in PAIRS}
    r1={p:np.concatenate([[np.nan],np.diff(lr[p])]) for p in PAIRS}
    # rolling 5-min return = lr[t]-lr[t-5]
    r5={p:np.concatenate([[np.nan]*HOR, lr[p][HOR:]-lr[p][:-HOR]]) for p in PAIRS}
    # USD basket return (exclude EURUSD), sign-adjusted, 1-min and 5-min
    def basket(rdict):
        cols=[]
        for p in PAIRS:
            if p=="EURUSD": continue
            s=+1.0 if p in USD_BASE else -1.0
            cols.append(s*rdict[p])
        return np.nanmean(np.vstack(cols),axis=0)
    usd1=basket(r1); usd5=basket(r5)
    rEU1=r1["EURUSD"]; rEU5=r5["EURUSD"]; rGB5=r5["GBPUSD"]
    # forward EURUSD 5-min return (next-bar to +5), contiguity-checked (wall clock exactly 300s)
    fwd=np.full(n,np.nan)
    if n>HOR:
        contig=(secs[HOR:]-secs[:-HOR])==HOR*60
        fr=lr["EURUSD"][HOR:]-lr["EURUSD"][:-HOR]
        fwd[:n-HOR]=np.where(contig,fr,np.nan)
    target=fwd
    # candidate predictors of next 5-min EURUSD direction
    preds={
      "own_momo_r5":      rEU5,                 # continuation of own move
      "own_revert_r5":   -rEU5,                 # mean-reversion of own move
      "usd_co_r1":       -usd1,                 # last-1min USD move -> EURUSD (co-move/continuation)
      "usd_co_r5":       -usd5,
      "usd_catchup_lag": (-usd5 - rEU5),        # EURUSD 'owes' the USD move not yet reflected
      "gbp_lead":        (rGB5 - rEU5),         # GBP moved, EUR to catch up
      "gbp_co_r5":        rGB5,
    }
    allmask=np.ones(n,bool)
    print(f"\n=== {y}  (aligned rows={n:,}) ===")
    print(f"{'predictor':<18}{'sign_agree':>11}{'n':>9}{'spearman':>11}")
    for k,v in preds.items():
        sa,nn=sign_agree(v,target,allmask); sp=spearman(v,target,allmask)
        print(f"{k:<18}{sa:>11.4f}{nn:>9}{sp:>11.4f}")
    # selective concentration: does sign-agreement rise when |predictor| is large? (rank by |pred|)
    def sel_curve(pred,label):
        ok=np.isfinite(pred)&np.isfinite(target)&(target!=0)&(pred!=0)
        p=pred[ok]; t=target[ok]; order=np.argsort(-np.abs(p))
        agree=(np.sign(p[order])==np.sign(t[order])).astype(float)
        row=f"  sel[{label:<16}]"
        for cov in (1.0,0.2,0.1,0.05,0.02):
            k=max(50,int(cov*len(agree))); row+=f"  c{int(cov*100)}%={agree[:k].mean():.3f}(n{k})"
        print(row)
    # combined standardized score over the 3 consistent reversion/catch-up signals (per-year z = probe-only)
    zs=[]
    for k in ("usd_catchup_lag","gbp_lead","own_revert_r5"):
        v=preds[k]; m=np.nanmean(v); s=np.nanstd(v); zs.append((v-m)/(s+1e-12))
    combo=np.nansum(np.vstack(zs),axis=0)
    for k in ("usd_catchup_lag","gbp_lead","own_revert_r5"): sel_curve(preds[k],k)
    sel_curve(combo,"COMBO_z3")

if __name__=="__main__":
    for y in (sys.argv[1:] or ["2024","2026"]):
        probe(y)
