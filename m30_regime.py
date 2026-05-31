"""30m EURUSD — STRUCTURAL conditional-direction analysis (mechanism-first, not ML-confidence).

Tests whether any a-priori REGIME has a stable directional bias in the 30-min forward return — i.e. a conditional
base rate that deviates from 0.50 and HOLDS across TRAIN/VAL/TEST24/TEST25/OOS26. Rules are pre-committed from theory
(reversion of oscillator extremes in compressed/quiet ranges; continuation under multi-TF trend confluence). Each rule
emits a direction per row using ONLY causal contemporaneous features; we bet where it fires, take INDEPENDENT
(non-overlap chronological, 1800s) trades, and report accuracy + CI95 per window. A real edge is consistent in sign
and magnitude across ALL held-out windows (corr(VAL,OOS) was -0.54 historically — consistency is the only guard).

  python m30_regime.py
"""
import numpy as np, pandas as pd, time
from m30_lab import load, nonoverlap_chrono, boot, GAP_S, base

NEED=["30m_rsi","1h_rsi","15m_rsi","4h_rsi","30m_bb_pctb","1h_bb_pctb","15m_bb_pctb",
      "30m_bb_width","1h_bb_width","15m_bb_width","30m_rangepos_24","1h_rangepos_24","30m_rangepos_48",
      "sess_ny","sess_london","sess_overlap","30m_above_ema50","1h_above_ema50","4h_above_ema50",
      "30m_macd_pos","1h_macd_pos","30m_autocorr_10","1h_autocorr_10","30m_dist_ema50","1h_dist_ema50",
      "30m_slope_20","1h_slope_20","mtf_trend_align","30m_atr_pct","1h_atr_pct","30m_ret_6","1h_ret_6"]
NEED=[c for c in NEED if c in base]
WINDOWS={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],
         "test24":["2024"],"test25":["2025"],"oos":["2026"]}

def col(R,c,n): return R[c].astype(float) if c in R.columns else np.full(n,np.nan)

def rules(R):
    """Return dict name-> (pred array in {+1,-1,0}). +1 = bet UP, -1 = bet DOWN, 0 = no bet."""
    n=len(R); g={}
    rsi30=col(R,"30m_rsi",n); rsi1h=col(R,"1h_rsi",n)          # normalized (rsi-50)/50
    pctb30=col(R,"30m_bb_pctb",n); pctb1h=col(R,"1h_bb_pctb",n)
    rp30=col(R,"30m_rangepos_24",n); rp1h=col(R,"1h_rangepos_24",n)
    bw30=col(R,"30m_bb_width",n); bw1h=col(R,"1h_bb_width",n)
    ny=col(R,"sess_ny",n)>0.5; lon=col(R,"sess_london",n)>0.5
    ae30=col(R,"30m_above_ema50",n)>0.5; ae1h=col(R,"1h_above_ema50",n)>0.5; ae4h=col(R,"4h_above_ema50",n)>0.5
    autoc=col(R,"30m_autocorr_10",n)
    bw30_med=np.nanmedian(bw30); bw30_lo=np.nanpercentile(bw30,33)
    def rev(sig_hi, sig_lo):  # reversion: predict DOWN when overbought(hi), UP when oversold(lo)
        p=np.zeros(n); p[sig_hi]=-1; p[sig_lo]=+1; return p
    def cont(up, dn):         # continuation
        p=np.zeros(n); p[up]=+1; p[dn]=-1; return p
    # --- pure reversion of oscillator extremes ---
    g["rev_rsi30"]      = rev(rsi30>0.4,  rsi30<-0.4)
    g["rev_rsi30_x"]    = rev(rsi30>0.5,  rsi30<-0.5)
    g["rev_pctb30"]     = rev(pctb30>1.0, pctb30<-1.0)
    g["rev_rangepos30"] = rev(rp30>0.9,   rp30<0.1)
    # --- reversion gated to COMPRESSED (quiet) regime — sofien: reversion lives in quiet ranges ---
    comp = bw30<=bw30_lo
    g["rev_rsi30_comp"]   = rev((rsi30>0.4)&comp, (rsi30<-0.4)&comp)
    g["rev_pctb30_comp"]  = rev((pctb30>1.0)&comp,(pctb30<-1.0)&comp)
    # --- reversion gated to NY session ---
    g["rev_rsi30_ny"]     = rev((rsi30>0.4)&ny,   (rsi30<-0.4)&ny)
    g["rev_pctb30_ny"]    = rev((pctb30>1.0)&ny,  (pctb30<-1.0)&ny)
    g["rev_rsi30_comp_ny"]= rev((rsi30>0.4)&comp&ny,(rsi30<-0.4)&comp&ny)
    # --- multi-TF extreme confluence (30m & 1h both stretched) ---
    g["rev_rsi_mtf"]   = rev((rsi30>0.4)&(rsi1h>0.3), (rsi30<-0.4)&(rsi1h<-0.3))
    g["rev_pctb_mtf"]  = rev((pctb30>1.0)&(pctb1h>0.6),(pctb30<-1.0)&(pctb1h<-0.6))
    # --- continuation under trend confluence ---
    up3=ae30&ae1h&ae4h; dn3=(~ae30)&(~ae1h)&(~ae4h)
    g["cont_trend3"]   = cont(up3, dn3)
    g["cont_trend3_ny"]= cont(up3&ny, dn3&ny)
    # --- momentum-regime conditioned: continuation when autocorr>0, reversion when autocorr<0 ---
    momo=autoc>0.05; mr=autoc<-0.05
    g["cont_momo_rsi"] = cont((rsi30>0.2)&momo, (rsi30<-0.2)&momo)
    g["rev_mr_rsi"]    = rev((rsi30>0.4)&mr,   (rsi30<-0.4)&mr)
    return g

def main():
    t0=time.time()
    print(f"[regime] indep non-overlap {GAP_S}s, chronological; CI95 boot; base rate ~0.50; breakeven~0.541",flush=True)
    data={}
    for w,yrs in WINDOWS.items():
        D=load(yrs, 1, cols=NEED)
        data[w]={"R":D[[c for c in NEED if c in D.columns]], "y":D["_y"].astype(int).values, "ts":D["_ts"].values.astype("int64")}
        print(f"[regime] loaded {w}: n={len(D):,} ({time.time()-t0:.0f}s)",flush=True)
    # discover candidate rule names from train
    names=list(rules(data["train"]["R"]).keys())
    print(f"\n{'rule':22s} "+" ".join(f"{w:>22s}" for w in WINDOWS))
    for nm in names:
        cells=[]
        for w in WINDOWS:
            d=data[w]; pr=rules(d["R"])[nm]
            mask=pr!=0
            if mask.sum()==0: cells.append("n0".rjust(22)); continue
            sel=nonoverlap_chrono(d["ts"], mask)
            pred_up=(pr[sel]>0)            # True=bet up
            corr=(pred_up==(d["y"][sel]==1)).astype(float)
            acc=corr.mean(); lo,hi=boot(corr)
            cells.append(f"n{len(sel):>5} {acc:.3f}[{lo:.2f},{hi:.2f}]".rjust(22))
        print(f"{nm:22s} "+" ".join(cells),flush=True)
    print(f"\n[regime] DONE {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
