"""Validate the certified AUDUSD 15m NY seed-ens book under TRUE TICK SETTLEMENT (deriv-faithful wc_ret).

SCOPE: AUDUSD · 15m. The cert + freeze used the BAR-CLOSE label sign(close[t+15]-close[t]); this re-settles the
frozen book's confident NY trades on held-out years with the deriv-faithful tick rule (mirror of
usdjpy_15m_ticksettle.py, which caught 2 settler bugs and PRESERVED the USDJPY edge): ENTRY = first tick at/after
bar-close+1s lag, EXIT = last tick at/before entry+900s, mid-to-mid, ties (ret==0) LOSE. Reports tick-WR vs
bar-close-WR per year/side. NOT a new edge — an integrity gate on the DELIVERABLE before any deploy claim.

Book: AUDUSD.m15ny_seedens.v1 (K=3 seed-ens, cov3% gate thr=0.06625).
Usage: ~/binary-algo-venv/bin/python audusd_15m_ticksettle.py
"""
import os, glob, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sessions import session_mask
from audusd_15m_base import build, boot, BE, SPL

PAIR="AUDUSD"; HOR=15; GAP=900; HOR_S=900; LAG=1; TOL=30.0; SESSION="ny"   # TOL=30s appropriate for a 900s horizon
BARCLOSE=60   # bar INDEXED at minute START; close[t]=mid at minute END (~index+60s). Anchor entry at index+BARCLOSE (else ~59s window misalignment, trap#8). Verified faithful on USDJPY.
THR=0.06624753336867155
MODELS=[f"models/m15ny_AUDUSD_s{s}_lgb.txt" for s in range(3)]
YEAR_OF={"test24":2024,"test25":2025,"oos":2026}
RESULT="audusd_15m_ticksettle_result.json"

def nonoverlap_chrono(ts, mask, gap=GAP):
    take=[]; block=-1
    for i in np.where(mask)[0]:
        if ts[i]<block: continue
        take.append(i); block=int(ts[i])+gap
    return np.array(take,dtype=int)

def day_ticks(date):
    # tick FILES are broker-local (UTC+~8): content of file F = UTC [F-1 ~16:00, F ~15:59].
    # UTC date d's afternoon lives in file d+1 -> load BOTH {d, d+1} to cover UTC date d.
    dnext=(pd.Timestamp(date, tz="UTC")+pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fs=sorted(glob.glob(f"/home/sean/git/raw/{PAIR}/{PAIR}_{date}_*.parquet")
              +glob.glob(f"/home/sean/git/raw/{PAIR}/{PAIR}_{dnext}_*.parquet"))
    if not fs: return None,None
    parts=[pd.read_parquet(f,columns=["timestamp_utc","bid","ask"]) for f in fs]
    t=pd.concat(parts,ignore_index=True).drop_duplicates("timestamp_utc").sort_values("timestamp_utc")
    ts=t["timestamp_utc"].values.astype(float); mid=((t["bid"]+t["ask"])/2.0).values.astype(float)
    return ts,mid

def settle(dts, tk_ts, tk_mid):
    n=len(tk_ts); entry_t=dts+BARCLOSE+LAG
    ei=np.searchsorted(tk_ts, entry_t, side="left"); exit_t=entry_t+HOR_S
    xi=np.searchsorted(tk_ts, exit_t, side="right")-1
    eic=np.clip(ei,0,n-1); xic=np.clip(xi,0,n-1)
    entry_ok=(ei<n)&((tk_ts[eic]-entry_t)<=TOL)
    exit_ok=(xi>ei)&((exit_t-tk_ts[xic])<=TOL)
    valid=entry_ok&exit_ok
    ret=np.where(valid, tk_mid[xic]/tk_mid[eic]-1.0, np.nan)
    return ret, valid, ~entry_ok, (entry_ok & ~exit_ok)

def wr_ci(win):
    win=np.asarray(win,float)
    if len(win)==0: return {"n":0,"wr":float("nan"),"ci":[float("nan")]*2}
    lo,hi=boot(win); return {"n":int(len(win)),"wr":round(float(win.mean()),4),"ci":[round(lo,4),round(hi,4)]}

def main():
    t0=time.time()
    boosters=[lgb.Booster(model_file=m) for m in MODELS]
    res={"book":"AUDUSD.m15ny_seedens.v1","thr":THR,"breakeven":BE,
         "settlement":"TRUE TICK wc_ret: entry=first tick>=barclose+1s, exit=last tick<=entry+900s, mid-to-mid, ties LOSE",
         "purpose":"validate the certified book's frozen-forward win-rate under deriv-faithful tick settlement vs bar-close",
         "years":{}}
    for tag,yr in YEAR_OF.items():
        Xw,yw,mw,tsw=build([str(yr)],1); ny=session_mask(tsw,SESSION)
        pr=np.mean([b.predict(Xw.values) for b in boosters],axis=0)
        conf=np.abs(pr-0.5)
        tr=nonoverlap_chrono(tsw, (conf>=THR)&ny)
        if len(tr)==0: continue
        dts=tsw[tr].astype(float); pdir=(pr[tr]>0.5).astype(int)
        barwin=((pdir==yw[tr]) & mw[tr]).astype(float)
        dates=pd.to_datetime(dts, unit="s", utc=True).strftime("%Y-%m-%d")
        tret=np.full(len(dts), np.nan); tval=np.zeros(len(dts), bool)
        ebad=np.zeros(len(dts),bool); xbad=np.zeros(len(dts),bool); nofile=np.zeros(len(dts),bool)
        for d in pd.unique(dates):
            m=dates==d
            tk_ts,tk_mid=day_ticks(d)
            if tk_ts is None: nofile[m]=True; continue
            r,v,eb,xb=settle(dts[m], tk_ts, tk_mid); tret[m]=r; tval[m]=v; ebad[m]=eb; xbad[m]=xb
        twin=( (np.sign(tret)>0).astype(int)==pdir ) & tval & (tret!=0.0)
        vmask=tval & np.isfinite(tret)
        up=pdir==1; dn=pdir==0
        yr_res={
          "n_trades":int(len(tr)),"n_tick_valid":int(vmask.sum()),"tick_valid_frac":round(float(vmask.mean()),4),
          "bar_close_wr":{s:wr_ci(barwin[vmask & mm]) for s,mm in [("COMBINED",np.ones(len(tr),bool)),("UP",up),("DOWN",dn)]},
          "tick_wr":{s:wr_ci(twin[vmask & mm].astype(float)) for s,mm in [("COMBINED",np.ones(len(tr),bool)),("UP",up),("DOWN",dn)]},
          "tie_rate_tick":round(float(((tret==0.0)&tval).mean()),4),
          "invalid_breakdown":{"entry_bad":round(float(ebad.mean()),4),"exit_bad":round(float(xbad.mean()),4),"no_file":round(float(nofile.mean()),4)},
        }
        res["years"][tag]=yr_res
        b=yr_res["bar_close_wr"]["COMBINED"]; k=yr_res["tick_wr"]["COMBINED"]
        print(f"[ticksettle] {tag} ({yr}): n={yr_res['n_trades']} valid={yr_res['tick_valid_frac']:.3f} "
              f"(fail: entry {ebad.mean():.2f} exit {xbad.mean():.2f} nofile {nofile.mean():.2f}) | "
              f"SAME-SUBSET bar-WR {b['wr']} (n{b['n']}) -> tick-WR {k['wr']} (n{k['n']}) CI{k['ci']} | tie {yr_res['tie_rate_tick']} ({time.time()-t0:.0f}s)",flush=True)
    deltas=[]
    for tag in res["years"]:
        b=res["years"][tag]["bar_close_wr"]["COMBINED"]["wr"]; k=res["years"][tag]["tick_wr"]["COMBINED"]["wr"]
        if np.isfinite(b) and np.isfinite(k): deltas.append(k-b)
    res["mean_tick_minus_bar_COMB"]=round(float(np.mean(deltas)),4) if deltas else None
    res["verdict"]={"tick_settlement_preserves_edge":bool(deltas and abs(np.mean(deltas))<0.02),
        "note":("Tick settlement reproduces the bar-close win-rate within ~2pp -> the certified 15m edge is deriv-faithful, NOT a bar-shift artifact."
                if (deltas and abs(np.mean(deltas))<0.02) else
                "Tick vs bar-close win-rate differs >2pp on average -> bar-close proxy biased at 15m; see per-year.")}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[ticksettle] mean(tick-bar) COMB WR delta = {res['mean_tick_minus_bar_COMB']} -> "
          f"{'PRESERVED' if res['verdict']['tick_settlement_preserves_edge'] else 'DIVERGES'}  -> {RESULT}  {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
