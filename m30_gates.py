"""30m EURUSD — LARGER-TIMEFRAME REVERSAL & INDICATOR-FLAG GATE harness (user direction).

Tests pure rule-based gates: "only bet (a 30m reversal/momentum) when indicator X at TF Y is in flag-state Z".
Larger TFs {30m,1h,4h,1D} OHLC resampled from the 5m cache, indicators computed causally and aligned to the 1m
decision grid (last COMPLETED higher-TF bar). Label = deriv-faithful 30m (contiguity-enforced). Each rule => fire mask
+ direction; honest INDEPENDENT (non-overlap chrono 1800s) accuracy + CI95 across train/val/test24/test25/oos. A rule
counts ONLY if it holds across ALL held-out windows (corr(VAL,OOS) was -0.54). Base rate ~0.50.

  python m30_gates.py
"""
import os, time, numpy as np, pandas as pd
import harness as H

HOR=30; GAP_S=HOR*60
CACHE="/home/sean/git/binary-algo/ohlc_cache"
WINDOWS={"train":[str(y) for y in range(2016,2022)],"val":["2022","2023"],"test24":["2024"],"test25":["2025"],"oos":["2026"]}
TFS={"30m":"30min","1h":"1h","4h":"4h","1D":"1D"}

def rma(s,n): return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def _rsi(c,n=14):
    d=c.diff(); return 100-100/(1+rma(d.clip(lower=0),n)/(rma((-d).clip(lower=0),n)+1e-12))
def _stoch(h,l,c,n=14):
    hh=h.rolling(n).max(); ll=l.rolling(n).min(); return (c-ll)/(hh-ll+1e-12)*100
def _willr(h,l,c,n=14):
    hh=h.rolling(n).max(); ll=l.rolling(n).min(); return (hh-c)/(hh-ll+1e-12)*-100
def _cci(h,l,c,n=20):
    tp=(h+l+c)/3; ma=tp.rolling(n).mean(); md=(tp-ma).abs().rolling(n).mean(); return (tp-ma)/(0.015*md+1e-12)
def _demarker(h,l,n=14):
    a=h.diff().clip(lower=0).rolling(n).mean(); b=(-l.diff()).clip(lower=0).rolling(n).mean(); return a/(a+b+1e-12)
def _fisher(h,l,c,n=10):
    hh=h.rolling(n).max(); ll=l.rolling(n).min(); st=((c-ll)/(hh-ll+1e-12)).clip(1e-6,1-1e-6)
    x=(2*st-1).clip(-0.999,0.999); return 0.5*np.log((1+x)/(1-x))
def _cmo(c,n=14):
    d=c.diff(); su=d.clip(lower=0).rolling(n).sum(); sd=(-d).clip(lower=0).rolling(n).sum(); return (su-sd)/(su+sd+1e-12)*100
def _bbpctb(c,n=20):
    ma=c.rolling(n).mean(); sd=c.rolling(n).std(); return (c-ma)/(2*sd+1e-12)
def _bbwidth(c,n=20):
    ma=c.rolling(n).mean(); sd=c.rolling(n).std(); return 4*sd/(ma+1e-12)
def _z(c,n=20): return (c-c.rolling(n).mean())/(c.rolling(n).std()+1e-12)
def _atrp(h,l,c,n=14):
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1); return rma(tr,n)/c
def _rangepos(h,l,c,n=24):
    hh=h.rolling(n).max(); ll=l.rolling(n).min(); return (c-ll)/(hh-ll+1e-12)
def _chop(h,l,c,n=14):
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    rng=h.rolling(n).max()-l.rolling(n).min(); return 100*np.log10(tr.rolling(n).sum()/(rng+1e-12)+1e-12)/np.log10(n)
def _rvivol(c,n=10):
    v=c.rolling(n).std(); dv=v.diff(); return 100-100/(1+rma(dv.clip(lower=0),n)/(rma((-dv).clip(lower=0),n)+1e-12))
def _adx(h,l,c,n=14):
    up=h.diff(); dn=-l.diff()
    plus=pd.Series(np.where((up>dn)&(up>0),up,0.0),index=h.index); minus=pd.Series(np.where((dn>up)&(dn>0),dn,0.0),index=h.index)
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1); atr=rma(tr,n)
    pdi=100*rma(plus,n)/(atr+1e-12); mdi=100*rma(minus,n)/(atr+1e-12)
    dx=100*(pdi-mdi).abs()/(pdi+mdi+1e-12); return rma(dx,n)

def _ohlc(years):
    parts=[pd.read_parquet(f"{CACHE}/EURUSD_5m_{y}.parquet") for y in years if os.path.exists(f"{CACHE}/EURUSD_5m_{y}.parquet")]
    m=pd.concat(parts); m=m[~m.index.duplicated(keep="last")].sort_index(); return m

def tf_indicators(o5, tf_rule):
    o=o5["open"].resample(tf_rule).first(); h=o5["high"].resample(tf_rule).max()
    l=o5["low"].resample(tf_rule).min(); c=o5["close"].resample(tf_rule).last()
    b=pd.DataFrame({"open":o,"high":h,"low":l,"close":c}).dropna(subset=["close"])
    o,h,l,c=b["open"],b["high"],b["low"],b["close"]
    out={}
    out["rsi"]=_rsi(c); out["stoch"]=_stoch(h,l,c); out["willr"]=_willr(h,l,c); out["cci"]=_cci(h,l,c)
    out["demarker"]=_demarker(h,l); out["fisher"]=_fisher(h,l,c); out["cmo"]=_cmo(c)
    out["bbpctb"]=_bbpctb(c); out["bbwidth"]=_bbwidth(c); out["z"]=_z(c); out["atrp"]=_atrp(h,l,c)
    out["rangepos"]=_rangepos(h,l,c); out["chop"]=_chop(h,l,c); out["rvivol"]=_rvivol(c); out["adx"]=_adx(h,l,c)
    out["rsi_rising"]=(out["rsi"].diff()>0).astype(float); out["fisher_turn_up"]=(out["fisher"].diff()>0).astype(float)
    out["ret1"]=c.pct_change(); out["above_ema50"]=(c>c.ewm(span=50).mean()).astype(float)
    out["above_sma200"]=(c>c.rolling(200).mean()).astype(float)
    out["ibs"]=(c-l)/(h-l+1e-12)                                   # Internal Bar Strength
    out["rsi2"]=_rsi(c,2)                                          # Connors RSI(2)
    out["td_buy9"]=((c<c.shift(4)).rolling(9).sum()>=9).astype(float)   # DeMark TD buy setup-9
    out["td_sell9"]=((c>c.shift(4)).rolling(9).sum()>=9).astype(float)
    ma20=c.rolling(20).mean(); sd20=c.rolling(20).std()
    lb=ma20-2*sd20; out["bb_reentry_lo"]=((c.shift(1)<lb.shift(1))&(c>=lb)).astype(float)  # lower-band reclaim
    out["bb_reentry_hi"]=((c.shift(1)>(ma20.shift(1)+2*sd20.shift(1)))&(c<=ma20+2*sd20)).astype(float)
    return pd.DataFrame(out, index=b.index)

def load_window(years):
    base=pd.concat([pd.read_parquet(f"{H.FEAT_DIR}/EURUSD_{y}.parquet",columns=["close"]) for y in years])
    base=base[~base.index.duplicated(keep="last")].sort_index()
    idx=base.index; c=base["close"].values; n=len(c); secs=idx.values.astype("datetime64[s]").astype("int64")
    contig=np.zeros(n,bool); contig[:n-HOR]=(secs[HOR:]-secs[:-HOR])==HOR*60
    fwd=np.full(n,np.nan); fwd[:n-HOR]=c[HOR:]; ret=fwd/c-1
    y=(ret>0).astype(int); valid=contig&np.isfinite(ret)&(ret!=0)
    o5=_ohlc(years); F={}
    for tf,rule in TFS.items():
        ind=tf_indicators(o5,rule).shift(1).reindex(idx,method="ffill")  # last COMPLETED higher-TF bar
        for col in ind.columns: F[f"{tf}_{col}"]=ind[col].values
    F=pd.DataFrame(F,index=idx)
    hour=idx.hour.values
    return {"F":F,"y":y,"ts":secs,"valid":valid,"hour":hour}

def boot(corr,nb=4000,seed=7):
    corr=np.asarray(corr,float)
    if len(corr)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(corr); a=np.array([corr[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))
def nonoverlap_chrono(ts,mask,gap=GAP_S):
    take=[]; bu=-1
    for i in np.where(mask)[0]:
        if ts[i]<bu: continue
        take.append(i); bu=int(ts[i])+gap
    return np.array(take,dtype=int)

def evaluate(data, rules):
    print(f"{'rule':30s} "+" ".join(f"{w:>20s}" for w in WINDOWS),flush=True)
    keep=[]
    for nm,fn in rules:
        cells=[]; oos_ok=True; held=[]
        for w in WINDOWS:
            d=data[w]; fire,dirn=fn(d["F"],d["hour"])
            m=fire&d["valid"]&np.isfinite(dirn)&(dirn!=0)
            if m.sum()==0: cells.append("n0".rjust(20)); continue
            sel=nonoverlap_chrono(d["ts"],m)
            if len(sel)==0: cells.append("n0".rjust(20)); continue
            pu=(dirn[sel]>0); corr=(pu==(d["y"][sel]==1)).astype(float); acc=corr.mean(); lo,hi=boot(corr)
            cells.append(f"n{len(sel):>5} {acc:.3f}[{lo:.2f},{hi:.2f}]".rjust(20))
            if w in ("test24","test25","oos"): held.append((acc,len(sel),lo))
        flag=""
        if len(held)==3 and all(a>=0.60 for a,_,_ in held): flag=" <== STABLE>=.60"
        if len(held)==3 and all(a>=0.65 for a,_,_ in held): flag=" <=== STABLE>=.65 !!"
        print(f"{nm:30s} "+" ".join(cells)+flag,flush=True)
        if flag: keep.append(nm)
    return keep

# ---- larger-TF reversal / indicator-flag gate rules (user direction) ----
def col(F,c):
    return F[c].values.astype(float) if c in F.columns else np.full(len(F),np.nan)
def rev(hi,lo):  # reversal: short on overbought(hi), long on oversold(lo)
    d=np.zeros(len(hi)); d[hi]=-1; d[lo]=+1; return d
def RULES():
    R=[]
    def add(name,fn): R.append((name,fn))
    # single larger-TF oscillator-extreme reversals
    for tf in ("1h","4h","1D"):
        add(f"rev_{tf}_rsi", lambda F,h,tf=tf: (np.ones(len(F),bool)&((col(F,f"{tf}_rsi")>70)|(col(F,f"{tf}_rsi")<30)), rev(col(F,f"{tf}_rsi")>70,col(F,f"{tf}_rsi")<30)))
        add(f"rev_{tf}_demarker", lambda F,h,tf=tf: (((col(F,f"{tf}_demarker")>0.7)|(col(F,f"{tf}_demarker")<0.3)), rev(col(F,f"{tf}_demarker")>0.7,col(F,f"{tf}_demarker")<0.3)))
        add(f"rev_{tf}_willr", lambda F,h,tf=tf: (((col(F,f"{tf}_willr")>-10)|(col(F,f"{tf}_willr")<-90)), rev(col(F,f"{tf}_willr")>-10,col(F,f"{tf}_willr")<-90)))
        add(f"rev_{tf}_cci", lambda F,h,tf=tf: (((col(F,f"{tf}_cci")>150)|(col(F,f"{tf}_cci")<-150)), rev(col(F,f"{tf}_cci")>150,col(F,f"{tf}_cci")<-150)))
        add(f"rev_{tf}_bb", lambda F,h,tf=tf: (((col(F,f"{tf}_bbpctb")>1)|(col(F,f"{tf}_bbpctb")<-1)), rev(col(F,f"{tf}_bbpctb")>1,col(F,f"{tf}_bbpctb")<-1)))
        add(f"rev_{tf}_z", lambda F,h,tf=tf: (((col(F,f"{tf}_z")>2)|(col(F,f"{tf}_z")<-2)), rev(col(F,f"{tf}_z")>2,col(F,f"{tf}_z")<-2)))
        add(f"rev_{tf}_fisher", lambda F,h,tf=tf: (((col(F,f"{tf}_fisher")>1.5)|(col(F,f"{tf}_fisher")<-1.5)), rev(col(F,f"{tf}_fisher")>1.5,col(F,f"{tf}_fisher")<-1.5)))
    # multi-TF confluence reversals (daily + 4h same extreme)
    add("rev_4h_1D_rsi_conf", lambda F,h: (((col(F,"1D_rsi")>65)&(col(F,"4h_rsi")>70))|((col(F,"1D_rsi")<35)&(col(F,"4h_rsi")<30)),
        rev((col(F,"1D_rsi")>65)&(col(F,"4h_rsi")>70),(col(F,"1D_rsi")<35)&(col(F,"4h_rsi")<30))))
    add("rev_1D_demarker_4h_rsi", lambda F,h: (((col(F,"1D_demarker")>0.7)&(col(F,"4h_rsi")>65))|((col(F,"1D_demarker")<0.3)&(col(F,"4h_rsi")<35)),
        rev((col(F,"1D_demarker")>0.7)&(col(F,"4h_rsi")>65),(col(F,"1D_demarker")<0.3)&(col(F,"4h_rsi")<35))))
    # ranging-regime gate (daily Choppiness high = ranging) + 1h oscillator reversal
    add("rev_1h_rsi_in_chop", lambda F,h: (((col(F,"1D_chop")>55)&((col(F,"1h_rsi")>70)|(col(F,"1h_rsi")<30))),
        rev((col(F,"1D_chop")>55)&(col(F,"1h_rsi")>70),(col(F,"1D_chop")>55)&(col(F,"1h_rsi")<30))))
    add("rev_4h_rsi_in_chop", lambda F,h: (((col(F,"4h_chop")>55)&((col(F,"4h_rsi")>68)|(col(F,"4h_rsi")<32))),
        rev((col(F,"4h_chop")>55)&(col(F,"4h_rsi")>68),(col(F,"4h_chop")>55)&(col(F,"4h_rsi")<32))))
    # trend-regime gate (low chop=trending) + momentum continuation on 4h
    add("mom_4h_trend_lowchop", lambda F,h: ((col(F,"4h_chop")<38)&((col(F,"4h_rsi")>55)|(col(F,"4h_rsi")<45)),
        np.where(col(F,"4h_rsi")>55,1.0,np.where(col(F,"4h_rsi")<45,-1.0,0.0))))
    # daily band touch + 4h fisher turn (reversal trigger)
    add("rev_1D_bb_4h_fisher", lambda F,h: (((col(F,"1D_bbpctb")>0.9)&(col(F,"4h_fisher")>1))|((col(F,"1D_bbpctb")<-0.9)&(col(F,"4h_fisher")<-1)),
        rev((col(F,"1D_bbpctb")>0.9)&(col(F,"4h_fisher")>1),(col(F,"1D_bbpctb")<-0.9)&(col(F,"4h_fisher")<-1))))
    # ---- Sofien mined rules (distinct mechanisms): IBS, Connors RSI2, TD setup, BB re-entry ----
    for tf in ("30m","1h","4h"):
        add(f"ibs_rev_{tf}", lambda F,h,tf=tf: (((col(F,f"{tf}_ibs")<0.1)|(col(F,f"{tf}_ibs")>0.9)), rev(col(F,f"{tf}_ibs")>0.9,col(F,f"{tf}_ibs")<0.1)))
        add(f"connors_rsi2_{tf}", lambda F,h,tf=tf: (((col(F,f"{tf}_rsi2")<5)&(col(F,f"{tf}_above_sma200")>0.5))|((col(F,f"{tf}_rsi2")>95)&(col(F,f"{tf}_above_sma200")<0.5)),
            rev((col(F,f"{tf}_rsi2")>95)&(col(F,f"{tf}_above_sma200")<0.5),(col(F,f"{tf}_rsi2")<5)&(col(F,f"{tf}_above_sma200")>0.5))))
        add(f"td_setup9_{tf}", lambda F,h,tf=tf: (((col(F,f"{tf}_td_buy9")>0.5)|(col(F,f"{tf}_td_sell9")>0.5)), rev(col(F,f"{tf}_td_sell9")>0.5,col(F,f"{tf}_td_buy9")>0.5)))
        add(f"bb_reentry_{tf}", lambda F,h,tf=tf: (((col(F,f"{tf}_bb_reentry_lo")>0.5)|(col(F,f"{tf}_bb_reentry_hi")>0.5)), rev(col(F,f"{tf}_bb_reentry_hi")>0.5,col(F,f"{tf}_bb_reentry_lo")>0.5)))
    # IBS reversion gated to ranging regime (daily chop high)
    add("ibs_rev_30m_in_chop", lambda F,h: (((col(F,"1D_chop")>55)&((col(F,"30m_ibs")<0.1)|(col(F,"30m_ibs")>0.9))), rev((col(F,"1D_chop")>55)&(col(F,"30m_ibs")>0.9),(col(F,"1D_chop")>55)&(col(F,"30m_ibs")<0.1))))
    # Connors RSI2 confluence across 30m+1h
    add("connors_rsi2_30m_1h", lambda F,h: (((col(F,"30m_rsi2")<10)&(col(F,"1h_rsi2")<15)&(col(F,"1h_above_sma200")>0.5))|((col(F,"30m_rsi2")>90)&(col(F,"1h_rsi2")>85)&(col(F,"1h_above_sma200")<0.5)),
        rev((col(F,"30m_rsi2")>90)&(col(F,"1h_rsi2")>85)&(col(F,"1h_above_sma200")<0.5),(col(F,"30m_rsi2")<10)&(col(F,"1h_rsi2")<15)&(col(F,"1h_above_sma200")>0.5))))
    return R

def main():
    t0=time.time(); data={}
    for w,yrs in WINDOWS.items():
        data[w]=load_window(yrs); print(f"[gates] loaded {w}: n={len(data[w]['y']):,} valid={int(data[w]['valid'].sum()):,} ({time.time()-t0:.0f}s)",flush=True)
    print(f"\n[gates] indep non-overlap {GAP_S}s chrono; CI95; base~0.50; breakeven~0.541. STABLE = all 3 held-out windows >= threshold.\n")
    keep=evaluate(data, RULES())
    print(f"\n[gates] STABLE rules (>=0.60 across all held-out): {keep if keep else 'NONE'}  DONE {time.time()-t0:.0f}s")

if __name__=="__main__":
    main()
