"""Does quote imbalance predict direction at SHORT horizons but decay by 5 min?
If imbalance is real at seconds and ~0.50 at 5min, that definitively explains the 5m current best level
even WITH order-book data. Tests imbalance-follow accuracy at horizons 1..30 bars (sec & min)."""
import glob, numpy as np, pandas as pd, calendar
RAW="/media/sean/CORSAIR/tick_data/raw"
def load_ticks(pair,dates):
    parts=[]
    for d in dates:
        for f in sorted(glob.glob(f"{RAW}/{pair}/{pair}_{d}_*.parquet")): parts.append(pd.read_parquet(f))
    t=pd.concat(parts,ignore_index=True); t["ts"]=pd.to_datetime(t["timestamp_utc"],unit="s",utc=True)
    return t.sort_values("ts").reset_index(drop=True)
def mdates(y,m): return [f"{y}-{m:02d}-{d:02d}" for d in range(1,calendar.monthrange(y,m)[1]+1)]

t=load_ticks("EURUSD",mdates(2020,6))
bid,ask=t["bid"].values,t["ask"].values; bv,av=t["bid-vol"].values.astype(float),t["ask-vol"].values.astype(float)
mid=(bid+ask)/2; imb=(bv-av)/(bv+av+1e-9); micro=(bid*av+ask*bv)/(av+bv+1e-9)
df=pd.DataFrame({"mid":mid,"imb":imb,"micro":micro},index=t["ts"])

print("== TICK-LEVEL imbalance predicting price change after N ticks ==")
mp=df["mid"].values; im=df["imb"].values; mc=df["micro"].values
for ntick in (1,3,10,30,100):
    fwd=np.roll(mp,-ntick); ch=fwd-mp; ch[-ntick:]=np.nan
    m=np.isfinite(ch)&(ch!=0)
    # follow imbalance: imb>0 -> predict up
    acc=((im[m]>0)==(ch[m]>0)).mean()
    # microprice deviation predicts: micro>mid -> up
    accmd=(((mc[m]-mp[m])>0)==(ch[m]>0)).mean()
    print(f"  +{ntick:4d} ticks: imb_follow_acc={acc:.4f}  microdev_acc={accmd:.4f}  n={int(m.sum()):,}")

print("\n== 1-MINUTE bars: imbalance predicting direction at horizon H minutes ==")
g=df.resample("1min")
b=pd.DataFrame({"mid":g["mid"].last(),"imb":g["imb"].mean()}).dropna()
mm=b["mid"].values; ii=b["imb"].values
for H in (1,2,3,5,10,15,30):
    fwd=np.roll(mm,-H); ch=fwd-mm; ch[-H:]=np.nan; m=np.isfinite(ch)&(ch!=0)
    acc=((ii[m]>0)==(ch[m]>0)).mean()
    thr=np.nanquantile(np.abs(ii[m]),0.9); ext=m&(np.abs(ii)>=thr)
    acce=((ii[ext]>0)==(ch[ext]>0)).mean()
    print(f"  H={H:2d}m: imb_follow_acc={acc:.4f}  extreme10%={acce:.4f}  n={int(m.sum()):,}")
