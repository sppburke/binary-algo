"""Feasibility test: does CME ES (S&P futures) carry 30-min-FORWARD directional info for EURUSD? (E1 lever)

LEAN ES minute data is UTC-timestamped (market-hours db: Future-cme-ES dataTimeZone=UTC), front contract = the
per-day CSV with the most rows. We align ES to the EURUSD 1-min UTC grid and measure:
  (a) contemporaneous corr(ES_ret, EUR_ret)  -> alignment + co-movement sanity check (should be strongly +).
  (b) LEAD corr(ES_ret up to t, EUR forward-30m ret) -> the tradeable signal. Literature says ~0 at 30m.
  (c) directional AUC of an ES-only LGB for sign(EUR(t+30)-EUR(t)), overall and NY-session (13:30-20:00 UTC).
Run per year so any signal must be stable. No look-ahead: all ES features use info <= t; label is forward.
"""
import os, sys, glob, zipfile, io, time, numpy as np, pandas as pd

LEAN="/home/sean/git/reverse-engineered-trading/lean/data/future/cme/minute"
FEAT="/media/sean/CORSAIR/binary-algo/features"

def load_future_minute(sym, year):
    """Front-contract minute close, UTC-indexed, for one symbol+year. Front = contract CSV with most rows that day."""
    files=sorted(glob.glob(f"{LEAN}/{sym}/{year}*_trade.zip"))
    rows=[]
    for fp in files:
        day=os.path.basename(fp)[:8]
        try:
            z=zipfile.ZipFile(fp)
        except Exception: continue
        best=None; bestn=-1
        for nm in z.namelist():
            raw=z.read(nm)
            n=raw.count(b"\n")
            if n>bestn: bestn=n; best=raw
        if best is None: continue
        df=pd.read_csv(io.BytesIO(best), header=None, names=["ms","o","h","l","c","v"])
        base=pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:8]}", tz="UTC")
        df.index=base+pd.to_timedelta(df["ms"], unit="ms")
        rows.append(df[["c","v"]])
    if not rows: return None
    s=pd.concat(rows); s=s[~s.index.duplicated(keep="last")].sort_index()
    return s

def eur_1m(year):
    p=f"{FEAT}/EURUSD_{year}.parquet"
    df=pd.read_parquet(p, columns=["close"]); df=df[~df.index.duplicated(keep="last")]
    return df["close"]

def main(years):
    for year in years:
        t0=time.time()
        es=load_future_minute("es", year)
        if es is None: print(f"{year}: no ES data"); continue
        eur=eur_1m(year)
        # align ES close onto EUR 1-min grid (causal ffill, limit 3 min)
        esc=es["c"].reindex(eur.index, method="ffill", limit=3)
        d=pd.DataFrame({"eur":eur, "es":esc}).dropna()
        er=d["eur"]; sr=d["es"]
        # returns
        for k in (1,5,15,30):
            d[f"eur_r{k}"]=er.pct_change(k); d[f"es_r{k}"]=sr.pct_change(k)
        d["eur_fwd30"]=er.shift(-30)/er-1.0
        # contiguity: only rows where t..t+30 are contiguous minutes (drop session gaps)
        secs=d.index.values.astype("datetime64[s]").astype("int64"); n=len(secs)
        contig=np.zeros(n,bool); contig[:n-30]=(secs[30:]-secs[:-30])==1800
        d["valid"]=contig & np.isfinite(d["eur_fwd30"]) & (d["eur_fwd30"]!=0)
        hour=d.index.hour; ny=(hour>=13)&(hour<20)   # ES core / US session UTC
        v=d["valid"].values
        def corr(a,b,mask):
            x=d[a].values; y=d[b].values; m=mask&np.isfinite(x)&np.isfinite(y)
            return np.corrcoef(x[m],y[m])[0,1] if m.sum()>100 else float("nan")
        print(f"\n=== {year} ===  rows={len(d):,} valid30={int(v.sum()):,} load={time.time()-t0:.0f}s")
        print(f"  [align] contemporaneous corr(es_r15, eur_r15): ALL={corr('es_r15','eur_r15',v):+.3f}  NY={corr('es_r15','eur_r15',v&ny):+.3f}  (expect strongly + if aligned)")
        print(f"  [lead]  corr(es_r15(t), eur_fwd30):           ALL={corr('es_r15','eur_fwd30',v):+.3f}  NY={corr('es_r15','eur_fwd30',v&ny):+.3f}")
        print(f"  [lead]  corr(es_r5 (t), eur_fwd30):           ALL={corr('es_r5','eur_fwd30',v):+.3f}  NY={corr('es_r5','eur_fwd30',v&ny):+.3f}")
        print(f"  [lead]  corr(es_r30(t), eur_fwd30):           ALL={corr('es_r30','eur_fwd30',v):+.3f}  NY={corr('es_r30','eur_fwd30',v&ny):+.3f}")
        # E1's EXACT mechanism: EURUSD under-reacts to an ES move -> catches up over next 30m. gap = ES outperf vs EUR.
        d["gap15"]=d["es_r15"]-d["eur_r15"]; d["gap30"]=d["es_r30"]-d["eur_r30"]
        print(f"  [E1gap] corr(es15-eur15 gap, eur_fwd30):     ALL={corr('gap15','eur_fwd30',v):+.3f}  NY={corr('gap15','eur_fwd30',v&ny):+.3f}  (catch-up hypothesis)")
        print(f"  [E1gap] corr(es30-eur30 gap, eur_fwd30):     ALL={corr('gap30','eur_fwd30',v):+.3f}  NY={corr('gap30','eur_fwd30',v&ny):+.3f}")
        # directional value: sign(es_r15) -> sign(eur_fwd30) (continuation); and sign(gap15) (catch-up)
        for nm_,col in (("sign(es_r15)","es_r15"),("sign(gap15)","gap15")):
            for win,mask in (("ALL",v),("NY",v&ny)):
                x=d[col].values; fwd=d["eur_fwd30"].values
                m=mask&np.isfinite(x)&np.isfinite(fwd)&(x!=0)
                acc=(np.sign(x[m])==np.sign(fwd[m])).mean()
                print(f"  [dir]   {nm_}==sign(eur_fwd30) {win}: acc={acc:.3f} (n={int(m.sum()):,})  base0.50")

if __name__=="__main__":
    yrs=sys.argv[1:] if len(sys.argv)>1 else ["2025","2026"]
    main(yrs)
