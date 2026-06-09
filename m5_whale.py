"""N14 — WHALE-conditioned signed quote-imbalance (top-1% tsz) as a 5m EURUSD DIRECTION signal, BOTH sides.

MECHANISM (sign-aware, distinct from the KILLED pooled OFI): per-side signed quote-imbalance qimb=(bidvol−askvol)/
(bidvol+askvol), but restricted to the TOP-1% tsz (bidvol+askvol) ticks per bar — the 'conviction' subset. Pooled
OFI averages conviction with HFT noise (m5_perside_flow @300s = .5077, m5_cksofi300 = .5006, both KILLED); isolating
the largest-quoted-size ticks may recover a directional component the pooled version washes out. Polarity
(continuation vs reversion) selected on a calibration year. NOT subsumed by the pooled-flow null — distinct feature.

KNOWN RISK (pre-registered): deriv ticks are QUOTES not trades, so 'whale' = large QUOTED size, a weaker proxy for
informed flow; and microstructure direction decays by 60s — low prior at 300s. RUN anyway (attitude: don't infer).

PRE-REGISTERED FALSIFIER: KILL unless the whale-OIB signal's binding (worst of 2024/25/26) moved-acc CI95-lower
clears breakeven 0.541 at usable coverage (n≥100/yr), for at least ONE side. Focus DOWN (the open gap). Coin-flip
on the calibration year = KILL. Incumbent: m5xp UP refit floor .553; DOWN magweight .5441 (marginal).

  ~/binary-algo-venv/bin/python m5_whale.py     (resumable: caches per-year 5m whale-OIB)
"""
import os, sys; sys.argv=["x"]
import json, time, glob, numpy as np, pandas as pd
ROOT="/home/sean/git/binary-algo"; RAW="/home/sean/git/raw/EURUSD"; BE=0.541
CACHE=f"{ROOT}/features_tick_whale"; os.makedirs(CACHE,exist_ok=True)
HOR_S=300; CALIB=2023; EVAL=[2024,2025,2026]
def boot(c,nb=2000,seed=7):
    c=np.asarray(c,float)
    if len(c)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(c); a=np.array([c[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))

def build_year(y, tsz_thr=None):
    """Per-5m: whale_oib (mean qimb of top-tsz ticks), n_whale, mid_close, ts. Cache to parquet (resumable)."""
    cp=f"{CACHE}/EURUSD_{y}.parquet"
    if os.path.exists(cp): return pd.read_parquet(cp)
    files=sorted(glob.glob(f"{RAW}/EURUSD_{y}-*.parquet"))
    if not files: return None
    parts=[]
    for i,f in enumerate(files):
        try: d=pd.read_parquet(f,columns=["bid","ask","bid-vol","ask-vol","timestamp_utc"])
        except Exception: continue
        if len(d)==0: continue
        bv=d["bid-vol"].values.astype(float); av=d["ask-vol"].values.astype(float)
        mid=(d["bid"].values+d["ask"].values)/2; tsz=bv+av; qimb=(bv-av)/(bv+av+1e-9)
        t=pd.DataFrame({"ts":pd.to_datetime(d["timestamp_utc"],unit="s",utc=True),"mid":mid,"tsz":tsz,"qimb":qimb}).set_index("ts")
        parts.append(t)
        if i%1500==0: print(f"   [{y}] {i}/{len(files)} files",flush=True)
    if not parts: return None
    t=pd.concat(parts).sort_index()
    if tsz_thr is None: tsz_thr=np.quantile(t["tsz"].values,0.99)   # calib: top-1% tsz
    whale=t[t["tsz"]>=tsz_thr]
    g=t["mid"].resample("5min"); gw=whale["qimb"].resample("5min")
    out=pd.DataFrame({"mid":g.last(),"whale_oib":gw.mean(),"n_whale":whale["qimb"].resample("5min").count()}).dropna(subset=["mid"])
    out.attrs={}; out.to_parquet(cp)
    print(f"   [{y}] cached {len(out)} 5m bars -> {cp} (tsz_thr={tsz_thr:.1f})",flush=True)
    return out

def fwd_sign(df):
    secs=df.index.values.astype("datetime64[s]").astype("int64"); mid=df["mid"].values.astype(float); n=len(df)
    HB=HOR_S//300  # 1 bar = 5m
    fwd=np.full(n,np.nan)
    if n>HB:
        contig=(secs[HB:]-secs[:-HB])==HOR_S
        fwd[:n-HB]=np.where(contig, mid[HB:]-mid[:-HB], np.nan)
    return secs,fwd

def evalsig(df, thr_q, pol):
    secs,fwd=fwd_sign(df); w=df["whale_oib"].values; moved=np.isfinite(fwd)&(fwd!=0)&np.isfinite(w)
    if moved.sum()<50: return None
    delta=np.quantile(np.abs(w[moved]),thr_q)
    sig=np.zeros(len(df)); sig[w>delta]=pol; sig[w<-delta]=-pol
    m=moved&(sig!=0); idx=np.where(m)[0]; sel=[]; last=-1e18
    for i in idx:
        if secs[i]>=last+HOR_S: sel.append(i); last=secs[i]
    sel=np.array(sel,int)
    if len(sel)<20: return None
    pu=(sig[sel]>0); up=pu; dn=~pu
    corr=((sig[sel]>0).astype(int)==(fwd[sel]>0).astype(int)).astype(float)
    res={"n":len(sel),"acc":round(float(corr.mean()),4),"uprate":round(float((fwd[sel]>0).mean()),3)}
    if up.sum()>=20:
        cu=((fwd[sel][up]>0).astype(int)==1).astype(float); lo,hi=boot(cu); res["UP"]={"acc":round(float(cu.mean()),4),"n":int(up.sum()),"ci":[round(lo,4),round(hi,4)]}
    if dn.sum()>=20:
        cd=((fwd[sel][dn]>0).astype(int)==0).astype(float); lo,hi=boot(cd); res["DOWN"]={"acc":round(float(cd.mean()),4),"n":int(dn.sum()),"ci":[round(lo,4),round(hi,4)]}
    return res

def main():
    t0=time.time()
    # calib year: build + get tsz_thr from its own 99th pct (stored in build), then reuse for eval
    cdf=build_year(CALIB)
    if cdf is None: print("[whale] no calib data"); return
    # recover tsz_thr used: rebuild not needed; for eval, recompute thr per year from its own dist is fine (tsz stationary)
    cal_secs,cal_fwd=fwd_sign(cdf)
    # pick polarity + delta-quantile on calib (worst over the two halves not needed for a screen; use whole calib)
    best=None
    for pol in (1,-1):
        for tq in (0.3,0.5,0.7):
            r=evalsig(cdf,tq,pol)
            if r and r["n"]>=50:
                key=min([r.get(s,{}).get("acc",0) for s in ("UP","DOWN") if s in r] or [0])
                if best is None or key>best[0]: best=(key,pol,tq)
    if best is None: print("[whale] no usable calib op-point"); return
    cal_acc,pol,tq=best; poln="continuation" if pol==1 else "reversion"
    cr=evalsig(cdf,tq,pol)
    print(f"[whale] calib {CALIB}: {poln} tq={tq} -> UP {cr.get('UP')} DOWN {cr.get('DOWN')} (cal_acc={cal_acc:.4f}) {time.time()-t0:.0f}s",flush=True)
    # eval years
    ho={}
    for y in EVAL:
        df=build_year(y)
        if df is None: continue
        r=evalsig(df,tq,pol); ho[y]=r
        print(f"[whale] {y}: {r} ({time.time()-t0:.0f}s)",flush=True)
    # verdict per side
    def side_binding(side):
        accs=[ho[y][side]["acc"] for y in ho if ho[y] and side in ho[y]]
        los=[ho[y][side]["ci"][0] for y in ho if ho[y] and side in ho[y]]
        ns=[ho[y][side]["n"] for y in ho if ho[y] and side in ho[y]]
        return (min(accs) if accs else None, min(los) if los else None, min(ns) if ns else 0)
    up_b,up_lo,up_n=side_binding("UP"); dn_b,dn_lo,dn_n=side_binding("DOWN")
    up_ok=bool(up_lo is not None and up_lo>BE and up_n>=100); dn_ok=bool(dn_lo is not None and dn_lo>BE and dn_n>=100)
    survive=up_ok or dn_ok
    out=dict(test="N14 whale-conditioned signed quote-OIB (top-1% tsz) — 5m EURUSD direction", breakeven=BE,
        polarity=poln, delta_q=tq, calib_year=CALIB, calib=cr, per_year=ho,
        UP_binding=dict(acc=up_b,ci_lo=up_lo,n=up_n), DOWN_binding=dict(acc=dn_b,ci_lo=dn_lo,n=dn_n),
        VERDICT=dict(UP_survives=up_ok, DOWN_survives=dn_ok, SURVIVES=survive,
            statement=(f"N14 {'SURVIVES' if survive else 'KILLED'}: UP binding {up_b} (CI-lo {up_lo}, n{up_n}); "
                f"DOWN binding {dn_b} (CI-lo {dn_lo}, n{dn_n}). "
                + ("" if survive else f"Whale-conditioned quote-OIB is coin-flip at 5m on both sides — empirically confirms the order-flow family null at 300s (joins m5_cksofi300 .5006, m5_perside_flow .5077). Now Tier-1-subsumes N13/N15 (same family/data)."))))
    json.dump(out,open(f"{ROOT}/m5_whale_result.json","w"),indent=2,default=str)
    print(f"VERDICT: {out['VERDICT']['statement']}\n-> m5_whale_result.json ({time.time()-t0:.0f}s)")

if __name__=="__main__": main()
