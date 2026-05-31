"""Fetch the FXStreet economic calendar (free, no key) for 2012-2026, filter to USD/EUR impactful events with BOTH
actual and consensus, attach the EURUSD directional signal (event_signs), save to macro_calendar.parquet.
Verified endpoint: GET calendar-api.fxstreet.com/.../eventDates/{from}/{to}  header Referer: https://www.fxstreet.com/
"""
import sys, json, time, urllib.request, numpy as np, pandas as pd
import event_signs as ES
BASE="https://calendar-api.fxstreet.com/en/api/v1/eventDates"
HDR={"Referer":"https://www.fxstreet.com/","User-Agent":"Mozilla/5.0"}
OUT="/media/sean/CORSAIR/binary-algo/macro_calendar.parquet"

def fetch_year(y):
    url=f"{BASE}/{y}-01-01T00:00:00Z/{y}-12-31T23:59:59Z"
    req=urllib.request.Request(url,headers=HDR)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req,timeout=90) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            print(f"  {y} attempt {attempt+1} err {e}",flush=True); time.sleep(5)
    return []

def main(years):
    rows=[]
    for y in years:
        d=fetch_year(y); n0=len(d)
        kept=0
        for r in d:
            if r.get("currencyCode") not in ("USD","EUR"): continue
            if r.get("volatility") not in ("HIGH","MEDIUM"): continue
            a=r.get("actual"); c=r.get("consensus")
            if a is None or c is None: continue
            ccy,bull=ES.classify(r.get("name"), r.get("currencyCode"))
            if ccy is None: continue
            surp=float(a)-float(c)
            rows.append(dict(dateUtc=r.get("dateUtc"),ccy=r.get("currencyCode"),name=r.get("name"),
                actual=float(a),consensus=float(c),previous=r.get("previous"),surprise=surp,
                ratioDeviation=r.get("ratioDeviation"),volatility=r.get("volatility"),
                isPreliminary=bool(r.get("isPreliminary")),sig_ccy=ccy,bull=bull,
                eurusd_signal=ES.eurusd_signal(ccy,bull,surp)))
            kept+=1
        print(f"  {y}: {n0} rows -> {kept} USD/EUR impactful with actual+consensus+sign",flush=True)
        time.sleep(1.0)
    df=pd.DataFrame(rows)
    df["ts"]=pd.to_datetime(df["dateUtc"],utc=True,format="ISO8601")
    df=df.sort_values("ts").reset_index(drop=True)
    # z-score the surprise PER event-name using ONLY past data is ideal; here global per-name std (rescale later causally)
    df["surp_z"]=df.groupby("name")["surprise"].transform(lambda s:(s-s.mean())/(s.std()+1e-9))
    df["sig_z"]=df["surp_z"]*df["bull"]*np.where(df["sig_ccy"].values=="EUR",1.0,-1.0)
    df.to_parquet(OUT)
    print(f"\nSAVED {len(df)} events -> {OUT}")
    print("by ccy:",df["ccy"].value_counts().to_dict())
    print("by volatility:",df["volatility"].value_counts().to_dict())
    print("year span:",df["ts"].dt.year.min(),"-",df["ts"].dt.year.max(),"| per-year counts:")
    print(df["ts"].dt.year.value_counts().sort_index().to_dict())
    print("\ntop event names:",df["name"].value_counts().head(15).to_dict())

if __name__=="__main__":
    years=sys.argv[1:] or [str(y) for y in range(2012,2027)]
    main(years)
