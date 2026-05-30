"""Cross-pair lead-lag feature merger.
At EURUSD decision time t, append the (causal) recent behaviour of the 6 other USD pairs,
aligned on the same 1m timestamp grid. Lets the model exploit lead-lag + relative-value.
All peer features are computed the same causal way (info <= t), so same-timestamp join
introduces no lookahead."""
import os
import numpy as np, pandas as pd

FEAT_DIR="/media/sean/CORSAIR/binary-algo/features"
ALL_PAIRS=["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDCAD","USDCHF","USDJPY"]
PEER_COLS=["1m_ret_1","1m_ret_3","5m_ret_1","5m_ret_3","15m_ret_1",
           "1m_bb_pctb","5m_bb_pctb","1m_dist_ema20","5m_rv_24","1m_autocorr_10"]

def load_year_merged(target, year, base_cols=None):
    """Return EURUSD(target) features for `year` with peer columns appended."""
    tp=f"{FEAT_DIR}/{target}_{year}.parquet"
    if not os.path.exists(tp): return None
    base=pd.read_parquet(tp) if base_cols is None else pd.read_parquet(tp, columns=base_cols)
    peers=[p for p in ALL_PAIRS if p!=target]
    for pr in peers:
        pp=f"{FEAT_DIR}/{pr}_{year}.parquet"
        if not os.path.exists(pp): continue
        pf=pd.read_parquet(pp, columns=PEER_COLS)
        pf=pf.reindex(base.index, method="ffill", limit=3)
        pf.columns=[f"X{pr[:6]}_{c}" for c in pf.columns]
        base=pd.concat([base, pf], axis=1)
    return base

def cross_feature_names(target="EURUSD"):
    peers=[p for p in ALL_PAIRS if p!=target]
    return [f"X{pr[:6]}_{c}" for pr in peers for c in PEER_COLS]

if __name__=="__main__":
    import time; t=time.time()
    m=load_year_merged("EURUSD","2015")
    print("merged shape",m.shape,"time",round(time.time()-t,1),"s")
    xc=cross_feature_names()
    print("n cross feats",len(xc))
    print("nan frac cross (valid rows):",
          round(float(m[m.valid==True][xc].isna().mean().mean()),4))
