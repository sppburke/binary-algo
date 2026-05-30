"""Daily/weekly context features for longer-horizon (15m) prediction, aligned causally to the
1m grid. Built from the 1m close in the base feature parquets. All higher-than-intraday info uses
only COMPLETED prior days (shift by 1 trading day); intraday position uses running high/low to t."""
import numpy as np, pandas as pd

def _ema(s,n): return s.ewm(span=n,adjust=False).mean()

def build_daily_features(close_1m):
    """close_1m: pd.Series of 1-minute close, DatetimeIndex (one year). Returns DataFrame on same index."""
    c=close_1m
    # daily bars (trading days)
    d_open=c.resample("1D").first(); d_high=c.resample("1D").max()
    d_low=c.resample("1D").min(); d_close=c.resample("1D").last()
    daily=pd.DataFrame({"o":d_open,"h":d_high,"l":d_low,"c":d_close}).dropna(subset=["c"])
    dc=daily["c"]
    D={}
    for k in (1,2,3,5,10):
        D[f"dret_{k}"]=dc.pct_change(k)
    for n in (5,10,20):
        D[f"dist_dema{n}"]=dc/_ema(dc,n)-1.0
        D[f"above_dema{n}"]=(dc>_ema(dc,n)).astype("float32")
    for n in (10,20):
        hi=daily["h"].rolling(n).max(); lo=daily["l"].rolling(n).min()
        D[f"d_rangepos_{n}"]=(dc-lo)/(hi-lo+1e-12)
        D[f"d_dist_hi_{n}"]=dc/hi-1.0; D[f"d_dist_lo_{n}"]=dc/lo-1.0
    D["d_rv_10"]=dc.pct_change().rolling(10).std()
    D["d_rv_20"]=dc.pct_change().rolling(20).std()
    D["d_rv_pct"]=D["d_rv_20"].rolling(60).rank(pct=True)
    D["overnight_gap"]=(daily["o"]/dc.shift(1)-1.0)
    dfd=pd.DataFrame(D,index=daily.index).shift(1)  # causal: only completed prior days
    dfd=dfd.reindex(c.index, method="ffill")
    # intraday causal: running high/low of TODAY up to t, position, minutes since session start
    day=c.index.normalize()
    grp=c.groupby(day)
    run_hi=grp.cummax(); run_lo=grp.cummin()
    out=pd.DataFrame(index=c.index)
    out["intraday_pos"]=(c-run_lo)/(run_hi-run_lo+1e-12)
    out["intraday_ret"]=c/grp.transform("first")-1.0
    out["min_of_day"]=(c.index.hour*60+c.index.minute).astype("float32")
    out["dow"]=c.index.dayofweek.astype("float32")
    out=pd.concat([out,dfd],axis=1)
    out.columns=[f"DLY_{x}" for x in out.columns]
    return out.replace([np.inf,-np.inf],np.nan).astype("float32")

def daily_feature_names():
    idx=pd.date_range("2020-01-06",periods=3000,freq="1min",tz="UTC")
    import numpy as np
    s=pd.Series(np.cumsum(np.random.RandomState(0).randn(3000))*0+1.1,index=idx)
    return list(build_daily_features(s).columns)
