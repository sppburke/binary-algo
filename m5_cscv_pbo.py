"""V2 — CSCV / PBO + Deflated-Sharpe + N̂ + MinBTL on the m5xp (5m) gated-PnL matrix.

Companion to V1: replaces the E[max] proxy in cpcv_certify.py with the EXACT overfit probability (Bailey &
Lopez de Prado, "The Probability of Backtest Overfitting"). One heavy job (reload the frozen m5xp book once),
then a pure combinatorially-symmetric cross-validation on the per-bar gated returns.

CONFIG FAMILY (what the sweep selected AMONG): on the book's deployed gate (NY & meta>=THR, non-overlap 300s),
configs = {UP (pred>0.5), DOWN (pred<0.5)} x {fixed confidence covers ~30/15/10/5/2%}. Fixed conf thresholds
(computed once on the full sample) = FIXED strategies → no per-split threshold look-ahead. Binary deriv PnL:
win=+R(0.85), loss/tie=-1. For a binary +-/- strategy Sharpe is MONOTONE in win-rate, so CSCV's rank-by-Sharpe
== rank-by-win-rate == the actual selection metric.

OUTPUTS: PBO (prob the IS-best config is OOS-below-median), OOS-degradation slope, Deflated-Sharpe (DSR) of the
deployed UP config at N=M and at N̂=ρ̄+(1−ρ̄)M, MinBTL vs actual sample length.
PRE-REGISTERED FALSIFIER (IDEAS_LOG V2): DOWNGRADE m5xp UP if CSCV PBO>0.05 OR DSR-deflated edge <0.541 (DSR<0.95);
if the 5m sample is below MinBTL at realized N̂, the VAL-selected edge is noise.

  ~/binary-algo-venv/bin/python m5_cscv_pbo.py
"""
import sys; sys.argv=["x"]
import os, json, math, itertools, time
import numpy as np
from scipy.stats import norm
import m5_xpair as MX
import m5_xpair_production as XP

ROOT="/home/sean/git/binary-algo"
BE=0.541; R=0.85                  # deriv: win +0.85, loss/tie -1; breakeven win-rate 0.541
GAMMA=0.5772156649
S_BLOCKS=16                       # CSCV blocks (C(16,8)=12870 symmetric splits)
COVERS=[0.30,0.15,0.10,0.05,0.02] # confidence-cover levels per side -> fixed conf thresholds
RHO=0.4                           # avg cross-trial correlation (V1-consistent)
MIN_TR=20                         # min trades for a config's perf to be valid in a submatrix

def emax_norm(N):
    N=max(float(N),2.0)
    return (1-GAMMA)*norm.ppf(1-1.0/N)+GAMMA*norm.ppf(1-1.0/(N*math.e))

# ---------------- stage 1: per-bar gated streams (one book load) ----------------
def gated_stream():
    p,P,M=XP._load(); cols=p["primary_feats"]; mcols=p["meta_feats"]; THR=p["meta_thr"]
    TS=[];PR=[];Y=[]
    for w in ("test24","test25","oos"):
        D=MX.build_xp(XP.SPL[w]); D=MX.augment(D,XP.SPL[w],XP.MODE)
        pr=P.predict(D[cols].astype("float32")); y=D["_y"].astype(int).values
        sm=M.predict(XP._Xmeta(D,pr,mcols)); ts=D["_ts"].values.astype("int64"); ny=D["sess_ny"].values>0.5
        gate=ny&(sm>=THR)
        sel=MX.nonoverlap_chrono(ts,gate)          # independent (non-overlapping 300s) gated trades
        TS.append(ts[sel]); PR.append(pr[sel]); Y.append(y[sel])
        print(f"  [stream] {w}: gated indep trades={len(sel)}",flush=True)
        del D,pr,sm
    ts=np.concatenate(TS); pr=np.concatenate(PR); y=np.concatenate(Y)
    o=np.argsort(ts,kind="stable")
    return ts[o],pr[o],y[o].astype(int),THR

# ---------------- main ----------------
def main():
    t0=time.time()
    ts,pr,y,THR=gated_stream()
    conf=np.abs(pr-0.5); pred=(pr>0.5).astype(int); correct=(pred==y).astype(int)
    T=len(ts); print(f"[stream] total independent gated bars T={T} (meta>={THR:.4f}&NY) load {time.time()-t0:.0f}s",flush=True)

    # ---- build config family: side x fixed conf-cover threshold ----
    configs=[]   # (name, row_mask)
    for side,smask in (("UP",pred==1),("DOWN",pred==0)):
        cs=conf[smask]
        for cov in COVERS:
            thr=np.quantile(cs,1-cov) if len(cs) else np.inf      # fixed threshold (global) for this cover
            mask=smask&(conf>=thr)
            if mask.sum()>=MIN_TR:
                configs.append((f"{side}_cov{cov}", mask))
    names=[c[0] for c in configs]; Mn=len(configs)
    print(f"[family] {Mn} configs: {names}",flush=True)

    # per-bar return for each config (NaN where not traded)
    ret=np.where(correct==1, R, -1.0)             # +0.85 win / -1 loss-or-tie (ties already lose: moved-bars, y is sign)
    # ---- CSCV: S blocks, C(S,S/2) symmetric splits ----
    edges=np.linspace(0,T,S_BLOCKS+1).astype(int)
    blk=np.zeros(T,int)
    for b in range(S_BLOCKS): blk[edges[b]:edges[b+1]]=b
    # precompute per-block, per-config wins + counts
    W=np.zeros((Mn,S_BLOCKS)); N=np.zeros((Mn,S_BLOCKS))
    for ci,(nm,mask) in enumerate(configs):
        for b in range(S_BLOCKS):
            sub=mask&(blk==b)
            N[ci,b]=sub.sum(); W[ci,b]=correct[sub].sum()
    def winrate(wins,cnts):
        return np.where(cnts>=1, wins/np.maximum(cnts,1), np.nan)
    half=S_BLOCKS//2
    lambdas=[]; oos_isbest=[]; is_sr=[]; oos_sr=[]
    combos=list(itertools.combinations(range(S_BLOCKS),half))
    for combo in combos:
        isb=np.zeros(S_BLOCKS,bool); isb[list(combo)]=True
        is_w=W[:,isb].sum(1); is_n=N[:,isb].sum(1)
        oo_w=W[:,~isb].sum(1); oo_n=N[:,~isb].sum(1)
        valid=(is_n>=MIN_TR)&(oo_n>=MIN_TR)
        if valid.sum()<2: continue
        is_p=np.where(valid, is_w/np.maximum(is_n,1), -np.inf)
        oo_p=np.where(valid, oo_w/np.maximum(oo_n,1), np.nan)
        nstar=int(np.argmax(is_p))                         # IS-best config
        # OOS relative rank of n* among valid configs
        vv=np.where(valid)[0]; oos_vals=oo_p[vv]
        rank=(oos_vals< oo_p[nstar]).sum()                 # how many strictly worse OOS
        omega=(rank+1)/(len(vv)+1)                          # relative rank in (0,1)
        omega=min(max(omega,1e-6),1-1e-6)
        lambdas.append(math.log(omega/(1-omega)))
        oos_isbest.append(oo_p[nstar])
        is_sr.append(is_p[nstar]); oos_sr.append(oo_p[nstar])
    lambdas=np.array(lambdas)
    PBO=float((lambdas<0).mean())                           # prob IS-best is OOS-below-median
    oos_isbest=np.array(oos_isbest)
    # OOS-degradation: fraction of splits where the IS-best OOS win-rate < breakeven
    frac_isbest_below_BE=float((oos_isbest<BE).mean())
    # linear OOS-vs-IS degradation slope across splits
    iss=np.array(is_sr); ooss=np.array(oos_sr)
    slope=float(np.polyfit(iss,ooss,1)[0]) if len(iss)>2 else float("nan")

    # ---- Deflated-Sharpe of the DEPLOYED UP config (tightest cover that the book ships ~cov0.05) ----
    dep_name="UP_cov0.05"
    di=names.index(dep_name) if dep_name in names else int(np.argmax([ (W[i].sum()/max(N[i].sum(),1)) for i in range(Mn)]))
    dmask=configs[di][1]; dret=ret[dmask]; dwin=correct[dmask].mean(); dn=int(dmask.sum())
    sr=dret.mean()/dret.std(ddof=1) if dret.std(ddof=1)>0 else float("nan")     # per-trade Sharpe
    # trial-SR dispersion across the family (full sample)
    fam_sr=[]
    for i in range(Mn):
        rr=ret[configs[i][1]]
        if rr.std(ddof=1)>0: fam_sr.append(rr.mean()/rr.std(ddof=1))
    fam_sr=np.array(fam_sr); var_sr=float(fam_sr.var(ddof=1))
    # skew/kurt of deployed returns
    z=(dret-dret.mean())/dret.std(ddof=1); g3=float((z**3).mean()); g4=float((z**4).mean())
    def DSR(N_trials):
        sr0=math.sqrt(var_sr)*emax_norm(N_trials)
        denom=math.sqrt(max(1-g3*sr+(g4-1)/4*sr*sr,1e-9))
        return float(norm.cdf((sr-sr0)*math.sqrt(dn-1)/denom)), sr0
    dsr_M,sr0_M=DSR(Mn)
    Nhat=RHO+(1-RHO)*Mn
    dsr_Nhat,sr0_Nhat=DSR(Nhat)
    dsr_prog,_=DSR(70)                                  # program-wide ~70 trials (V1-consistent)
    # MinBTL (Bailey-LdP): min sample (trades) for SR to be significant given N trials
    def minBTL(N_trials):
        e=emax_norm(N_trials)
        return float((e/sr)**2 + 1) if sr>0 else float("inf")
    minbtl_M=minBTL(Mn); minbtl_prog=minBTL(70)

    out=dict(test="V2 CSCV/PBO + DSR + N̂ + MinBTL on m5xp 5m gated-PnL matrix",
        breakeven=BE, payout_R=R, T_independent_gated_bars=T, meta_thr=round(float(THR),4),
        family=names, M_configs=Mn, S_blocks=S_BLOCKS, n_splits=len(lambdas),
        CSCV=dict(PBO=round(PBO,4), frac_ISbest_OOS_below_breakeven=round(frac_isbest_below_BE,4),
                  oos_degradation_slope=round(slope,3),
                  oos_isbest_winrate_mean=round(float(oos_isbest.mean()),4),
                  oos_isbest_winrate_p10=round(float(np.percentile(oos_isbest,10)),4)),
        deployed_config=dict(name=dep_name if dep_name in names else names[di], n=dn, win_rate=round(float(dwin),4),
                  per_trade_sharpe=round(sr,4), skew=round(g3,3), kurt=round(g4,3)),
        DSR=dict(var_trial_sr=round(var_sr,5),
                 at_M=dict(N=Mn, sr0=round(sr0_M,4), DSR=round(dsr_M,4)),
                 at_Nhat=dict(N=round(Nhat,1), sr0=round(sr0_Nhat,4), DSR=round(dsr_Nhat,4)),
                 at_program70=dict(N=70, DSR=round(dsr_prog,4))),
        MinBTL=dict(at_M=round(minbtl_M,0), at_program70=round(minbtl_prog,0), actual_T=T,
                    sample_exceeds_minBTL_atM=bool(T>minbtl_M), sample_exceeds_minBTL_prog=bool(T>minbtl_prog)))
    # verdict (pre-registered falsifier)
    pbo_fail=PBO>0.05; dsr_fail=dsr_Nhat<0.95 or dwin<BE; btl_fail=T<minbtl_prog
    downgrade=bool(pbo_fail or dsr_fail or btl_fail)
    out["VERDICT"]=dict(
        PBO_le_0p05=not pbo_fail, DSR_at_Nhat_ge_0p95=not (dsr_Nhat<0.95),
        deployed_winrate_ge_breakeven=bool(dwin>=BE), sample_ge_minBTL=not btl_fail,
        downgrades_per_falsifier=downgrade,
        statement=(f"PBO={PBO:.3f} ({'PASS' if not pbo_fail else 'FAIL >0.05'}); "
                   f"deployed {dep_name} win={dwin:.4f} n={dn}, DSR@N̂={dsr_Nhat:.3f} ({'PASS' if dsr_Nhat>=0.95 else 'FAIL <0.95'}); "
                   f"MinBTL@70={minbtl_prog:.0f} vs T={T} ({'PASS' if not btl_fail else 'FAIL'}). "
                   f"{'m5xp UP DOWNGRADES under V2.' if downgrade else 'm5xp UP SURVIVES V2 (low overfit probability).'}"))
    json.dump(out, open(f"{ROOT}/m5_cscv_pbo_result.json","w"), indent=2, default=str)

    print("="*92); print("V2 CSCV/PBO + DSR + MinBTL — m5xp 5m"); print("="*92)
    print(f"T={T} indep gated bars | {Mn} configs | {len(lambdas)} CSCV splits (S={S_BLOCKS})")
    print(f"PBO={PBO:.4f}  frac(IS-best OOS<breakeven)={frac_isbest_below_BE:.4f}  OOS-degradation slope={slope:.3f}")
    print(f"  IS-best OOS win-rate: mean={oos_isbest.mean():.4f} p10={np.percentile(oos_isbest,10):.4f}")
    print(f"deployed {out['deployed_config']['name']}: win={dwin:.4f} n={dn} per-trade-SR={sr:.4f} skew={g3:.2f} kurt={g4:.2f}")
    print(f"DSR: @M={Mn} {dsr_M:.4f} (sr0 {sr0_M:.3f}) | @N̂={Nhat:.0f} {dsr_Nhat:.4f} | @prog70 {dsr_prog:.4f}")
    print(f"MinBTL: @M={minbtl_M:.0f} @prog70={minbtl_prog:.0f}  vs actual T={T}  (exceeds@prog70={T>minbtl_prog})")
    print(f"VERDICT: {out['VERDICT']['statement']}")
    print(f"-> m5_cscv_pbo_result.json  ({time.time()-t0:.0f}s)")

if __name__=="__main__":
    main()
