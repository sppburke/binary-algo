"""EVENT-TIME CONDITIONING: does the macro SURPRISE direction predict EURUSD's 5-min move right after a release?
Hypothesis (post-announcement drift / underreaction, Andersen-Bollerslev-Diebold-Vega): in the minutes after a surprise,
EURUSD drifts in the surprise direction. Rule (no training): predict dir = sign(eurusd_signal); enter at release+delay,
hold H min. Measure accuracy per year (2024/2025/2026 held-out), CI95, vs surprise magnitude, entry delay, hold horizon.
"""
import sys, numpy as np, pandas as pd
FEAT="/home/sean/git/binary-algo/features"
CAL="/home/sean/git/binary-algo/macro_calendar.parquet"

def load_closes(years):
    parts=[]
    for y in years:
        try:
            d=pd.read_parquet(f"{FEAT}/EURUSD_{y}.parquet",columns=["close"])
            d=d[~d.index.duplicated(keep="last")]; parts.append(d)
        except FileNotFoundError: pass
    s=pd.concat(parts).sort_index(); s=s[~s.index.duplicated(keep="last")]
    return s

def boot(c,nb=4000,seed=7):
    c=np.asarray(c,float)
    if len(c)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(c); a=np.array([c[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))

def eval_rule(ev, closes, delay, H, sigcol="eurusd_signal", min_abs_z=0.0, vol=None):
    idx=closes.index; vals=closes.values; secs=idx.values.astype("datetime64[s]").astype("int64")
    n=len(idx)
    e=ev.copy()
    if vol: e=e[e["volatility"].isin(vol)]
    e=e[np.abs(e["sig_z"])>=min_abs_z]
    e=e[e["eurusd_signal"]!=0]
    rec={}  # year -> list of correct(0/1)
    for _,r in e.iterrows():
        t0=int(pd.Timestamp(r["ts"]).timestamp())+delay*60
        j=np.searchsorted(secs,t0,side="left")
        if j>=n: continue
        # entry bar must be within 2 min of intended entry (avoid weekend/gap holes)
        if secs[j]-t0>120: continue
        k=np.searchsorted(secs,secs[j]+H*60,side="left")
        if k>=n or (secs[k]-secs[j])!=H*60: continue   # require exact H-min contiguous hold
        pred=1 if r[sigcol]>0 else -1
        out=np.sign(vals[k]-vals[j])
        if out==0: continue
        yy=pd.Timestamp(r["ts"]).year
        rec.setdefault(yy,[]).append(1.0 if pred==out else 0.0)
    return rec

def summarize(rec, label):
    allc=[]; cells=[]
    for yy in (2024,2025,2026):
        c=np.array(rec.get(yy,[]))
        cells.append(f"{yy}:{c.mean():.3f}(n{len(c)})" if len(c)>=10 else f"{yy}:n{len(c)}")
        allc.append(c)
    A=np.concatenate(allc) if allc else np.array([])
    tr=np.concatenate([np.array(rec.get(yy,[])) for yy in range(2012,2024)]) if rec else np.array([])
    if len(A)>=10:
        lo,hi=boot(A)
        print(f"  {label:<40} train(12-23):{tr.mean():.3f}(n{len(tr)})  "+"  ".join(cells)+f"  HELDOUT n{len(A)} {A.mean():.3f}[{lo:.3f},{hi:.3f}]")
    else:
        print(f"  {label:<40} insufficient held-out n")

def eval_jump(ev, closes, J, H, mode="cont", min_abs_z=0.0, vol=None, agree_surprise=False):
    """Use the release-minute JUMP (return over [release, release+J]) to predict the next H min.
    mode='cont' predicts continuation (sign jump), 'rev' predicts reversal. agree_surprise: require jump sign == surprise sign."""
    idx=closes.index; vals=closes.values; secs=idx.values.astype("datetime64[s]").astype("int64"); n=len(idx)
    e=ev.copy()
    if vol: e=e[e["volatility"].isin(vol)]
    e=e[np.abs(e["sig_z"])>=min_abs_z]
    rec={}
    for _,r in e.iterrows():
        t0=int(pd.Timestamp(r["ts"]).timestamp())
        j0=np.searchsorted(secs,t0,side="left")
        if j0>=n or secs[j0]-t0>120: continue
        jj=np.searchsorted(secs,secs[j0]+J*60,side="left")
        if jj>=n or (secs[jj]-secs[j0])!=J*60: continue
        k=np.searchsorted(secs,secs[jj]+H*60,side="left")
        if k>=n or (secs[k]-secs[jj])!=H*60: continue
        jump=vals[jj]-vals[j0]
        if jump==0: continue
        if agree_surprise and np.sign(jump)!=np.sign(r["eurusd_signal"]): continue
        pred=np.sign(jump) if mode=="cont" else -np.sign(jump)
        out=np.sign(vals[k]-vals[jj])
        if out==0: continue
        rec.setdefault(pd.Timestamp(r["ts"]).year,[]).append(1.0 if pred==out else 0.0)
    return rec

def main():
    ev=pd.read_parquet(CAL); ev["ts"]=pd.to_datetime(ev["ts"],utc=True)
    print(f"[m5_news] {len(ev)} events {ev['ts'].dt.year.min()}-{ev['ts'].dt.year.max()}; HIGH={int((ev.volatility=='HIGH').sum())} MED={int((ev.volatility=='MEDIUM').sum())}")
    closes=load_closes([str(y) for y in range(2012,2027)])
    print(f"[m5_news] closes {len(closes):,} bars\n")
    print("=== HIGH-impact only, by entry delay & hold horizon (rule: trade surprise direction) ===")
    for delay in (0,1,2):
        for H in (5,):
            for mz in (0.0,):
                summarize(eval_rule(ev,closes,delay,H,min_abs_z=mz,vol=["HIGH"]), f"HIGH delay={delay}m hold={H}m")
    print("\n=== HIGH-impact, large surprises only (|sig_z|>=1.0), entry delay sweep ===")
    for delay in (0,1,2,3):
        summarize(eval_rule(ev,closes,delay,5,min_abs_z=1.0,vol=["HIGH"]), f"HIGH |z|>=1.0 delay={delay}m hold=5m")
    print("\n=== hold-horizon sweep (HIGH, delay=1, |z|>=0.5) ===")
    for H in (3,5,10,15):
        summarize(eval_rule(ev,closes,1,H,min_abs_z=0.5,vol=["HIGH"]), f"HIGH |z|>=0.5 delay=1m hold={H}m")
    print("\n=== HIGH+MEDIUM, delay=1, hold=5, |z| sweep ===")
    for mz in (0.0,0.5,1.0,1.5):
        summarize(eval_rule(ev,closes,1,5,min_abs_z=mz,vol=["HIGH","MEDIUM"]), f"HIGH+MED |z|>={mz} delay=1m hold=5m")
    print("\n=== JUMP-continuation: release-minute jump (J min) predicts next H min (HIGH) ===")
    for J in (1,2):
        for H in (3,5):
            summarize(eval_jump(ev,closes,J,H,mode="cont",vol=["HIGH"]), f"CONT jump={J}m hold={H}m HIGH")
            summarize(eval_jump(ev,closes,J,H,mode="rev",vol=["HIGH"]),  f"REV  jump={J}m hold={H}m HIGH")
    print("\n=== JUMP-continuation, jump AGREES with surprise sign (HIGH+MED, |z|>=0.5) ===")
    for J in (1,2):
        summarize(eval_jump(ev,closes,J,5,mode="cont",min_abs_z=0.5,vol=["HIGH","MEDIUM"],agree_surprise=True), f"CONT+agree jump={J}m hold=5m")
        summarize(eval_jump(ev,closes,J,5,mode="rev", min_abs_z=0.5,vol=["HIGH","MEDIUM"],agree_surprise=True), f"REV +agree jump={J}m hold=5m")

if __name__=="__main__": main()
