"""Signed order-flow proxy features from 10s bars (the most-cited short-horizon signal).
We lack a true limit-order book, so we proxy order-flow imbalance via tick-rule signed
volume: per 10s bar, sign(close-open)*volume; aggregate to 1m and roll over windows.
Cached separately in features_of/ and merged on demand."""
import os, numpy as np, pandas as pd
import pipeline as P

OF_DIR="/home/sean/git/binary-algo/features_of"
os.makedirs(OF_DIR, exist_ok=True)

def of_1m(df10):
    o,c,v=df10["open"],df10["close"],df10["volume"]
    r=(c-o)
    sgn=np.sign(r.values)
    # tick-rule fallback: if bar flat (r==0), use sign of close vs prev close
    flat=sgn==0
    sgn_prev=np.sign(c.diff().fillna(0).values)
    sgn=np.where(flat, sgn_prev, sgn)
    svol=pd.Series(sgn*v.values, index=df10.index)
    absret=r.abs()
    g=pd.DataFrame({"svol":svol,"vol":v,"absret":absret,"up":(sgn>0).astype(float)})
    m=pd.DataFrame({
        "ofi": g["svol"].resample("1min").sum(),
        "vol": g["vol"].resample("1min").sum(),
        "uptick": g["up"].resample("1min").mean(),
        "absret": g["absret"].resample("1min").sum(),
        "nbar": g["vol"].resample("1min").count(),
    }).dropna(subset=["nbar"])
    m["ofi_norm"]=m["ofi"]/(m["vol"]+1e-9)
    return m

def build_of_features(m):
    out={}
    on=m["ofi_norm"].fillna(0); ofi=m["ofi"].fillna(0); vol=m["vol"]
    for k in (1,3,5,10,15,30):
        out[f"of_norm_{k}"]=on.rolling(k).mean()
        out[f"of_sum_{k}"]=ofi.rolling(k).sum()/(vol.rolling(k).sum()+1e-9)
    out["of_uptick_5"]=m["uptick"].rolling(5).mean()-0.5
    out["of_uptick_15"]=m["uptick"].rolling(15).mean()-0.5
    # Kyle-lambda proxy: price impact per unit signed flow (abs ret / vol)
    out["kyle_5"]=m["absret"].rolling(5).sum()/(vol.rolling(5).sum()+1e-9)
    out["kyle_15"]=m["absret"].rolling(15).sum()/(vol.rolling(15).sum()+1e-9)
    # OFI acceleration / divergence
    out["of_accel"]=on.rolling(3).mean()-on.rolling(15).mean()
    # signed-flow vs price divergence handled downstream; persistence:
    out["of_persist"]=np.sign(on.rolling(5).mean())*np.sign(on.rolling(15).mean())
    df=pd.DataFrame(out, index=m.index)
    df.columns=[f"OF_{c}" for c in df.columns]
    return df

OF_COLS=None
def of_feature_names():
    global OF_COLS
    if OF_COLS is None:
        idx=pd.date_range("2020-01-01",periods=50,freq="1min",tz="UTC")
        dummy=pd.DataFrame({"ofi_norm":0.0,"ofi":0.0,"vol":1.0,"uptick":0.5,"absret":0.0},index=idx)
        OF_COLS=list(build_of_features(dummy).columns)
    return OF_COLS

def process_pair(pair, years=None):
    if years is None: years=[str(y) for y in range(2012,2027)]
    for year in years:
        out=f"{OF_DIR}/{pair}_{year}.parquet"
        if os.path.exists(out): print(f"  {pair} {year}: skip"); continue
        df10=P.load_pair_year_10s(pair,year)
        if df10 is None: print(f"  {pair} {year}: no data"); continue
        m=of_1m(df10); F=build_of_features(m)
        F=F.replace([np.inf,-np.inf],np.nan).astype("float32")
        F.to_parquet(out)
        print(f"  {pair} {year}: rows={len(F)} cols={F.shape[1]}", flush=True)

if __name__=="__main__":
    import sys
    for p in (sys.argv[1:] or ["EURUSD"]):
        print(f"=== OF {p} ==="); process_pair(p)
