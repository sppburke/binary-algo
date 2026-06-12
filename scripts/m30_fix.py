"""30m EURUSD — Iteration 4: FX FIXING-WINDOW reversal (literature-backed, mechanistic).

Krohn, Mueller & Whelan (J.Finance 2024) + Evans (2018, "Forex Trading and the WMR Fix"): a W-shaped USD drift INTO
the benchmark fixes (WMR London 16:00 London, ECB ~13:15 CET, Tokyo ~09:55 JST) and a REVERSAL after — persistent for
21 years across all G10. This is the one orthogonal, clock-deterministic directional signal the literature backs at the
~0.5-3% coverage / 30-min scale. Test: at the fix minute, bet the REVERSAL of the pre-fix drift over the next 30 min.

Discipline: (a) MAP time-of-day reversal edge on TRAIN only (exploration); (b) verify literature-pre-committed fix
windows across VAL/TEST24/TEST25/OOS; a real edge holds in sign across ALL. Independent (non-overlap 1800s), CI95.

  python m30_fix.py
"""
import os, numpy as np, pandas as pd, time
from m30_lab import HOR, nonoverlap_chrono, boot

WINDOWS={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],
         "test24":["2024"],"test25":["2025"],"oos":["2026"]}
FEAT="/home/sean/git/binary-algo/features"

def load_close(years):
    parts=[]
    for y in years:
        p=f"{FEAT}/EURUSD_{y}.parquet"
        if not os.path.exists(p): continue
        df=pd.read_parquet(p, columns=["close"]); df=df[~df.index.duplicated(keep="last")]
        parts.append(df)
    df=pd.concat(parts)
    idx=df.index; c=df["close"].values.astype(float); n=len(c)
    secs=idx.values.astype("datetime64[s]").astype("int64")
    contig=np.zeros(n,bool)
    if n>HOR: contig[:n-HOR]=(secs[HOR:]-secs[:-HOR])==HOR*60
    fwd=np.full(n,np.nan); fwd[:n-HOR]=c[HOR:]; ret=fwd/c-1.0
    y=(ret>0).astype(float); valid=contig&np.isfinite(ret)&(ret!=0)
    hour=idx.hour.values; minute=idx.minute.values
    return dict(c=c, secs=secs, y=y, valid=valid, hour=hour, minute=minute, idx=idx)

def drift_sign(c, secs, t_idx, lookback_s):
    """sign of close[t]-close[t-lookback] using nearest bar at t-lookback (causal)."""
    target=secs[t_idx]-lookback_s
    j=np.searchsorted(secs, target, side="right")-1
    j=np.clip(j,0,len(c)-1)
    ok=(secs[t_idx]-secs[j])<=lookback_s+120  # within 2 min of intended lookback
    d=np.sign(c[t_idx]-c[j]); d[~ok]=0
    return d

def eval_rule(W, decmask, pred, label="rule"):
    """decmask: bool over all rows (decision fires). pred: +1/-1/0 array. Returns per-window line."""
    cells=[]
    for w in WINDOWS:
        D=W[w]; m=decmask[w]&(pred[w]!=0)&D["valid"]
        if m.sum()==0: cells.append(f"{w}:n0"); continue
        sel=nonoverlap_chrono(D["secs"], m)
        if len(sel)==0: cells.append(f"{w}:n0"); continue
        pu=(pred[w][sel]>0); corr=(pu==(D["y"][sel]==1)).astype(float)
        acc=corr.mean(); lo,hi=boot(corr)
        cells.append(f"{w}:n{len(sel)} {acc:.3f}[{lo:.2f},{hi:.2f}]")
    return f"{label:30s} "+"  ".join(cells)

def main():
    t0=time.time()
    W={w:load_close(yrs) for w,yrs in WINDOWS.items()}
    print(f"[fix] loaded ({time.time()-t0:.0f}s). indep 1800s; base~0.50; breakeven~0.541\n",flush=True)
    # --- (A) time-of-day MAP on TRAIN: reversal of prior-30min drift, decided at each (hour, minute in {0,30}) ---
    Dtr=W["train"]; print("[A] TRAIN time-of-day map — reversal of prior-30m drift, by decision hour (minute=0):")
    rows=[]
    for hh in range(24):
        dm=(Dtr["hour"]==hh)&(Dtr["minute"]==0)&Dtr["valid"]
        idxs=np.where(dm)[0]
        if len(idxs)<200: continue
        dr=drift_sign(Dtr["c"], Dtr["secs"], idxs, 1800)
        pr=-dr  # reversal
        good=pr!=0; sel=nonoverlap_chrono(Dtr["secs"], dm&_scatter(pr!=0,idxs,len(Dtr["c"])))
        # simpler: evaluate directly on idxs
        pu=(pr[good]>0); yy=(Dtr["y"][idxs][good]==1); acc=(pu==yy).mean(); n=good.sum()
        rows.append((hh,n,acc))
    for hh,n,acc in rows:
        star=" <==" if (acc>0.55 or acc<0.45) else ""
        print(f"   hour {hh:02d}:00 UTC  n={n:5d}  reversal_acc={acc:.3f}{star}")
    # --- (B) literature fix windows, reversal of prior-{30,60}m drift, decided at fix minute ---
    print("\n[B] Fix-window reversal (pre-committed windows), across all splits:")
    fixes={"WMR_16utc":(16,0),"WMR_15utc":(15,0),"ECB_1215":(12,15),"ECB_1115":(11,15),
           "Tokyo_0055":(0,55),"NYopen_1330":(13,30),"NYopen_1400":(14,0)}
    for fname,(fh,fm) in fixes.items():
        for lb in (1800,3600):
            decmask={}; pred={}
            for w in WINDOWS:
                D=W[w]; dm=(D["hour"]==fh)&(D["minute"]==fm)
                idxs=np.where(dm)[0]
                pr=np.zeros(len(D["c"]))
                if len(idxs)>0:
                    dr=drift_sign(D["c"], D["secs"], idxs, lb); pr[idxs]=-dr
                decmask[w]=dm; pred[w]=pr
            print(eval_rule(W, decmask, pred, f"{fname} rev(drift{lb//60}m)"))
    # --- (C) the same windows, CONTINUATION (sign check) for WMR ---
    print("\n[C] WMR continuation (opposite sign — to see the drift direction):")
    for fname,(fh,fm) in {"WMR_16utc":(16,0),"WMR_15utc":(15,0)}.items():
        decmask={}; pred={}
        for w in WINDOWS:
            D=W[w]; dm=(D["hour"]==fh)&(D["minute"]==fm); idxs=np.where(dm)[0]
            pr=np.zeros(len(D["c"]))
            if len(idxs)>0: pr[idxs]=drift_sign(D["c"], D["secs"], idxs, 1800)
            decmask[w]=dm; pred[w]=pr
        print(eval_rule(W, decmask, pred, f"{fname} cont(drift30m)"))
    print(f"\n[fix] DONE {time.time()-t0:.0f}s")

def _scatter(vals, idxs, n):
    out=np.zeros(n,bool); out[idxs]=vals; return out

if __name__=="__main__":
    main()
