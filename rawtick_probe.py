"""Probe: do raw-tick QUOTE IMBALANCE and MICROPRICE predict 5-min direction?
Raw ticks have bid/ask + bid-vol/ask-vol (quote sizes) = order-book imbalance, the documented
short-horizon predictor I previously lacked. Test on a sample before scaling."""
import glob, os, numpy as np, pandas as pd
RAW="/home/sean/git/raw"

def load_ticks(pair, dates):
    parts=[]
    for d in dates:
        fs=sorted(glob.glob(f"{RAW}/{pair}/{pair}_{d}_*.parquet"))
        for f in fs:
            parts.append(pd.read_parquet(f))
    if not parts: return None
    t=pd.concat(parts, ignore_index=True)
    t["ts"]=pd.to_datetime(t["timestamp_utc"], unit="s", utc=True)
    t=t.sort_values("ts").reset_index(drop=True)
    return t

def tick_features_1m(t):
    bid,ask=t["bid"].values,t["ask"].values
    bv,av=t["bid-vol"].values.astype(float),t["ask-vol"].values.astype(float)
    mid=(bid+ask)/2
    micro=(bid*av+ask*bv)/(av+bv+1e-9)          # microprice (size-weighted)
    imb=(bv-av)/(bv+av+1e-9)                      # quote imbalance in [-1,1]
    micro_dev=(micro-mid)/mid                     # microprice deviation from mid
    spread=(ask-bid)/mid
    df=pd.DataFrame({"mid":mid,"imb":imb,"micro_dev":micro_dev,"spread":spread},index=t["ts"])
    g=df.resample("1min")
    out=pd.DataFrame({
        "mid": g["mid"].last(),
        "imb_mean": g["imb"].mean(),
        "imb_last": g["imb"].last(),
        "micro_dev_mean": g["micro_dev"].mean(),
        "spread_mean": g["spread"].mean(),
        "nticks": g["mid"].count(),
    }).dropna(subset=["mid"])
    return out

# sample: 3 spread-out months to avoid regime bias
import calendar
def month_dates(y,m):
    nd=calendar.monthrange(y,m)[1]
    return [f"{y}-{m:02d}-{d:02d}" for d in range(1,nd+1)]

for (y,m,lab) in [(2015,3,"2015-03"),(2020,6,"2020-06"),(2025,9,"2025-09")]:
    t=load_ticks("EURUSD", month_dates(y,m))
    if t is None: print(lab,"no data"); continue
    f=tick_features_1m(t)
    mid=f["mid"]; fwd=mid.shift(-5); ret=fwd/mid-1; up=(ret>0).astype(int)
    m_=ret.notna()&(ret!=0)
    print(f"\n=== {lab}: ticks={len(t):,} 1m_bars={len(f):,} ===")
    # predictive accuracy of each signal (predict up if signal>0)
    for col,desc in [("imb_mean","imbalance->follow"),("imb_last","imb_last->follow"),
                     ("micro_dev_mean","microdev->follow")]:
        s=f[col]
        # follow signal: predict up if imbalance positive (more bid size -> price up)
        pred=(s>0).astype(int); acc=(pred[m_]==up[m_]).mean()
        # also test fade
        accf=((s<0).astype(int)[m_]==up[m_]).mean()
        # extreme decile
        thr=s[m_].abs().quantile(0.9); ext=m_&(s.abs()>=thr)
        acce=(pred[ext]==up[ext]).mean()
        print(f"  {desc:22s}: follow_acc={acc:.4f} fade_acc={accf:.4f} extreme10%_follow={acce:.4f} (n_ext={int(ext.sum())})")
