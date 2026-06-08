"""Cache 1-second microstructure bars per split to parquet (one-time cost), so seconds-horizon
model iterations are fast. Broad date coverage to improve generalization (close the TEST gap)."""
import glob, time, os, numpy as np, pandas as pd, calendar
RAW="/media/sean/CORSAIR/tick_data/raw"; PAIR="EURUSD"
OUT="/home/sean/git/binary-algo/features_tick"; os.makedirs(OUT, exist_ok=True)
def mo(y,ms):
    out=[]
    for m in ms: out+=[f"{y}-{m:02d}-{d:02d}" for d in range(1,calendar.monthrange(y,m)[1]+1)]
    return out
SPLIT_DATES={
 "train": mo(2021,[3,6,9,12])+mo(2022,[2,4,6,8,10,12])+mo(2023,[2,4,6,8,10,12]),
 "val":   mo(2024,[4,5]),
 "test":  mo(2024,[9,10,11])+mo(2025,[2,3,4,9,10,11]),
 "oos":   mo(2026,[2,3,4]),
}
def _day_1s(files):
    """Resample ONE day's hourly tick files to 1s bars (bounded memory)."""
    parts=[pd.read_parquet(f,columns=["ask","bid","ask-vol","bid-vol","timestamp_utc"]) for f in files]
    t=pd.concat(parts,ignore_index=True); t["ts"]=pd.to_datetime(t["timestamp_utc"],unit="s",utc=True)
    t=t.sort_values("ts"); bid,ask=t["bid"].values,t["ask"].values
    bv,av=t["bid-vol"].values.astype(float),t["ask-vol"].values.astype(float)
    t["mid"]=(bid+ask)/2; t["imb"]=(bv-av)/(bv+av+1e-9); t["micro"]=(bid*av+ask*bv)/(av+bv+1e-9)
    t["spread"]=(ask-bid)/t["mid"]; t["tsz"]=bv+av; t=t.set_index("ts"); g=t.resample("1s")
    return pd.DataFrame({"mid":g["mid"].last(),"imb":g["imb"].mean(),"micro":g["micro"].last(),
        "spread":g["spread"].mean(),"nt":g["mid"].count(),"tsz":g["tsz"].mean()}).dropna(subset=["mid"]).astype("float32")

def load_1s(dates):
    """Memory-safe: resample per-day, accumulate only the (small) 1s bars."""
    days=[]
    for d in dates:
        files=sorted(glob.glob(f"{RAW}/{PAIR}/{PAIR}_{d}_*.parquet"))
        if files: days.append(_day_1s(files))
    if not days: return None
    return pd.concat(days)
if __name__=="__main__":
    for sp,dates in SPLIT_DATES.items():
        p=f"{OUT}/{sp}_1s.parquet"
        if os.path.exists(p): print(f"{sp}: exists",flush=True); continue
        t0=time.time(); b=load_1s(dates)
        b.to_parquet(p); print(f"{sp}: {len(b):,} bars saved ({time.time()-t0:.0f}s)",flush=True)
    print("CACHE DONE")
