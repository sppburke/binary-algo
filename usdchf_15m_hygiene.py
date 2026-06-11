"""USDCHF 15m data-hygiene gate (run BEFORE any compute).
Replicates usdcad_15m_base.build() label derivation to verify:
  (1) moved up-rate per year ∈ [0.47,0.53] (fake-flat mirage tripwire),
  (2) flat fraction small (no ffill mirage),
  (3) per-session row fractions (ny/ldn/asia) to set session priors —
      CHF is European so LDN prior should be materially higher than for USDCAD/USDJPY.
"""
import os, numpy as np, pandas as pd
import harness as H
from sessions import session_mask

PAIR="USDCHF"; HOR=15; STEP=60; GAP=HOR*STEP
FEATS=H.feature_cols(PAIR)
print(f"[hygiene] PAIR={PAIR} n_feats={len(FEATS)}")

def yr(y):
    p=f"{H.FEAT_DIR}/{PAIR}_{y}.parquet"
    if not os.path.exists(p): return None
    d=pd.read_parquet(p, columns=FEATS+["close"]); d=d[~d.index.duplicated(keep="last")]
    c=d["close"].values.astype(float)
    ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
    contig=np.zeros(n,bool); contig[:n-HOR]=(ts[HOR:]-ts[:-HOR])==GAP
    fr=np.full(n,np.nan); fr[:n-HOR]=c[HOR:]/c[:-HOR]-1.0
    X=d[FEATS].astype("float32"); keepf=X.isna().mean(axis=1).values<0.5
    valid=contig & np.isfinite(fr) & keepf
    moved=valid & (fr!=0.0)
    return ts, fr, valid, moved

allrows=0; allsess={"ny":0,"ldn":0,"asia":0}
for y in [2015,2018,2021,2024,2025,2026]:
    r=yr(y)
    if r is None:
        print(f"  {y}: MISSING"); continue
    ts,fr,valid,moved=r
    nv=int(valid.sum()); nm=int(moved.sum())
    up=float((fr[moved]>0).mean()) if nm else float("nan")
    flat=float((valid & (fr==0.0)).sum()/max(1,nv))
    trip="OK" if 0.47<=up<=0.53 else "*** TRIPWIRE FAIL ***"
    # session fractions among MOVED decision bars
    sm={s:float(session_mask(ts[moved], s).mean()) for s in ("ny","ldn","asia")}
    print(f"  {y}: valid={nv:,} moved={nm:,} up-rate={up:.4f} flat_frac={flat:.4f} {trip} | "
          f"session(moved): ny={sm['ny']:.2f} ldn={sm['ldn']:.2f} asia={sm['asia']:.2f}")
