"""Compliant calendar PROXY (no scraping): scheduled economic releases (NFP/CPI/FOMC/ECB) occur
at fixed (day-of-week, time-of-day) slots and create systematic volatility. We learn, from TRAIN
ONLY, the expected forward volatility per (dow, minute-of-day) cell -> an 'expected event intensity'
feature, plus an indicator for top release windows. This captures the calendar's main mechanism
(event timing) without any external data, and is fully causal (lookup fit on train, applied to all)."""
import numpy as np, pandas as pd

def fit_event_intensity(close_1m_by_year_train, horizon=15):
    """Build a (dow, minute_of_day) -> mean |fwd-return| table from training closes."""
    cells={}; counts={}
    for c in close_1m_by_year_train:
        c=c[~c.index.duplicated(keep="last")]
        fwd=c.shift(-horizon); ret=(fwd/c-1.0).abs()
        dow=c.index.dayofweek.values; mod=(c.index.hour*60+c.index.minute).values
        key=dow*1440+mod
        df=pd.DataFrame({"k":key,"r":ret.values}).dropna()
        g=df.groupby("k")["r"]
        for k,v in g.sum().items(): cells[k]=cells.get(k,0.0)+v
        for k,v in g.count().items(): counts[k]=counts.get(k,0)+v
    table={k:cells[k]/counts[k] for k in cells if counts[k]>50}
    arr=np.array(list(table.values()))
    thr=np.quantile(arr,0.95) if len(arr) else 0.0
    return {"table":table, "thr":thr, "median":float(np.median(arr)) if len(arr) else 0.0}

def apply_event_intensity(index, model):
    dow=index.dayofweek.values; mod=(index.hour*60+index.minute).values
    key=dow*1440+mod
    tab=model["table"]; med=model["median"]
    ev=np.array([tab.get(int(k),med) for k in key],dtype="float32")
    out=pd.DataFrame(index=index)
    out["EVT_intensity"]=ev
    out["EVT_top_window"]=(ev>=model["thr"]).astype("float32")
    # neighbours (release approaching / just passed) via local max over +-15 min on the day-grid
    s=pd.Series(ev,index=index)
    out["EVT_intensity_max15"]=s.rolling(31,center=True,min_periods=1).max().values
    out["EVT_rising"]=(s.diff(5)>0).astype("float32").values
    return out.astype("float32")

EVENT_COLS=["EVT_intensity","EVT_top_window","EVT_intensity_max15","EVT_rising"]
