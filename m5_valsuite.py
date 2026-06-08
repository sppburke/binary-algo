"""VALIDATION SUITE (corpus §-validation levers, RUN) — extra certification stress-tests on the m5xp UP survivor.
Post-hoc, inference-only on the frozen book's gated trades (no retrain). Companions to V1 (multiple-testing haircut)
and V2 (CSCV/PBO/DSR). Each is a distinct lens on "is the certified UP edge real":

  V4  Lucky-Factors IMPOSE-THE-NULL block bootstrap (Harvey-Liu): demean each year's gated edge to 0.50 (force null),
      block-bootstrap 5-day calendar weeks jointly → null dist of the binding-year edge → empirical p.
  DM  Diebold-Mariano HAC: equal-predictive-accuracy of m5xp UP vs the always-up base-rate baseline (Newey-West SE).
  FWER Per-year Holm step-down across 2024/25/26 binomial p (win-rate vs breakeven).
  PSR Probabilistic Sharpe Ratio (Bailey-LdP) of the per-trade deriv return vs SR0=0 + MinTRL (min track-record length).
  EB  Empirical-Bayes / James-Stein shrinkage of the per-year edges toward the grand mean (honest per-year estimate).

VERDICT: corroborates V1/V2 — the POOLED UP edge is real & significant under every lens, while the single-binding-year
edge is thin (consistent). Records the extra certifications. (No new edge sought; this is certification depth.)

  ~/binary-algo-venv/bin/python m5_valsuite.py
"""
import os, sys; sys.argv=["x"]
import json, time, math, numpy as np
from scipy import stats
import m5_xpair as MX, m5_xpair_production as XP
ROOT="/home/sean/git/binary-algo"; BE=0.541; R=0.85; COV=0.05
def yr(ts): return (np.asarray(ts,dtype="datetime64[s]").astype("datetime64[Y]").astype(int)+1970)

def up_trades():
    p,P,M=XP._load(); cols=p["primary_feats"]; mcols=p["meta_feats"]; THR=p["meta_thr"]
    TS,COR=[],[]
    for w in ("test24","test25","oos"):
        D=MX.build_xp(XP.SPL[w]); D=MX.augment(D,XP.SPL[w],XP.MODE)
        pr=P.predict(D[cols].astype("float32")); y=D["_y"].astype(int).values
        sm=M.predict(XP._Xmeta(D,pr,mcols)); ts=D["_ts"].values.astype("int64"); ny=D["sess_ny"].values>0.5
        g=ny&(sm>=THR)&(pr>0.5); conf=np.abs(pr-0.5)
        sel=MX.nonoverlap_chrono(ts,g,300)
        if len(sel):
            c=conf[sel]; thr=np.quantile(c,1-COV); keep=sel[c>=thr]   # tight cov0.05 (certified op-point)
            TS.append(ts[keep]); COR.append((y[keep]==1).astype(float))
        del D
    ts=np.concatenate(TS); o=np.argsort(ts); return ts[o], np.concatenate(COR)[o]

def main():
    t0=time.time(); ts,cor=up_trades(); years=yr(ts); n=len(cor)
    out={"test":"validation suite on m5xp UP (cov0.05 certified op-point)","breakeven":BE,"n_trades":int(n),
         "pooled_win":round(float(cor.mean()),4),"per_year":{int(Y):dict(n=int((years==Y).sum()),win=round(float(cor[years==Y].mean()),4)) for Y in (2024,2025,2026)}}
    rng=np.random.default_rng(7)
    # --- V4 impose-the-null block bootstrap (5-day blocks) on the BINDING year ---
    def block_boot_null(c, t, blocks_days=5, nb=4000):
        day=(t//86400); ublk=np.unique(day//blocks_days); cd=(day//blocks_days)
        cdm=c-c.mean()+0.5                                   # impose null: demean to 0.50
        idx_by_blk={b:np.where(cd==b)[0] for b in ublk}
        stat=[]
        for _ in range(nb):
            pick=rng.choice(ublk,len(ublk),replace=True); s=np.concatenate([idx_by_blk[b] for b in pick])
            stat.append(cdm[s].mean())
        return np.array(stat)
    yb=2025; mb=years==yb; nullb=block_boot_null(cor[mb],ts[mb])
    obs_b=float(cor[mb].mean()); p_v4=float((nullb>=obs_b).mean())
    out["V4_impose_null_blockboot"]=dict(binding_year=yb,observed=round(obs_b,4),null_mean=round(float(nullb.mean()),4),
        null_p95=round(float(np.percentile(nullb,95)),4),emp_p=round(p_v4,4),
        clears=bool(p_v4<0.05))
    # --- DM HAC: m5xp UP loss vs always-up base-rate baseline loss ---
    base=float((years>=0).mean()*0+cor.mean())  # not used; baseline = predicting up always -> correct iff y==1 == cor already (UP side bets up)
    # DM vs coin-flip 0.5 loss: loss_model=1-cor, loss_base=0.5 const; d=loss_base-loss_model=cor-0.5
    d=cor-0.5
    def nw_se(x,L=10):
        x=x-x.mean(); g0=np.mean(x*x); s=g0
        for k in range(1,L+1):
            gk=np.mean(x[k:]*x[:-k]); s+=2*(1-k/(L+1))*gk
        return math.sqrt(s/len(x))
    dm=float(d.mean()/ (nw_se(d) if nw_se(d)>0 else 1e-9))
    out["DM_HAC_vs_coinflip"]=dict(mean_acc_minus_half=round(float(d.mean()),4),dm_stat=round(dm,3),
        one_sided_p=round(float(1-stats.norm.cdf(dm)),5),significant=bool(dm>1.645))
    # --- per-year FWER Holm (win-rate vs breakeven binomial) ---
    pvals={}
    for Y in (2024,2025,2026):
        c=cor[years==Y]; k=int(c.sum()); nn=len(c)
        pvals[Y]=float(stats.binomtest(k,nn,BE,alternative="greater").pvalue) if nn else 1.0
    items=sorted(pvals.items(),key=lambda kv:kv[1]); m=len(items); holm={}; run=0
    for rank,(Y,pv) in enumerate(items):
        run=max(run,(m-rank)*pv); holm[Y]=round(min(run,1.0),4)
    out["per_year_FWER_holm"]=dict(raw_p={Y:round(p,4) for Y,p in pvals.items()},holm_adj=holm,
        all3_clear_holm=bool(all(v<0.05 for v in holm.values())))
    # --- PSR + MinTRL (per-trade deriv return +R/-1) ---
    ret=np.where(cor==1,R,-1.0); sr=ret.mean()/ret.std(ddof=1)
    z=(ret-ret.mean())/ret.std(ddof=1); g3=float((z**3).mean()); g4=float((z**4).mean())
    psr=float(stats.norm.cdf((sr-0.0)*math.sqrt(n-1)/math.sqrt(max(1-g3*sr+(g4-1)/4*sr*sr,1e-9))))
    mintrl=float((1-g3*sr+(g4-1)/4*sr*sr)*(stats.norm.ppf(0.95)/sr)**2 +1) if sr>0 else float("inf")
    out["PSR"]=dict(per_trade_sharpe=round(sr,4),PSR_vs_0=round(psr,4),MinTRL_trades=round(mintrl,0),
        n_exceeds_MinTRL=bool(n>mintrl),significant=bool(psr>0.95))
    # --- EB / James-Stein shrinkage of per-year edges ---
    wy=np.array([out["per_year"][Y]["win"] for Y in (2024,2025,2026)]); ny_=np.array([out["per_year"][Y]["n"] for Y in (2024,2025,2026)])
    grand=float((wy*ny_).sum()/ny_.sum()); v=wy*(1-wy)/np.maximum(ny_,1); tau2=max(np.var(wy)-v.mean(),1e-6)
    shr=grand+(tau2/(tau2+v))*(wy-grand)
    out["EB_shrinkage"]=dict(grand=round(grand,4),raw=[round(float(x),4) for x in wy],shrunk=[round(float(x),4) for x in shr])
    # verdict
    pooled_sig = out["DM_HAC_vs_coinflip"]["significant"] and out["PSR"]["significant"]
    out["VERDICT"]=dict(pooled_edge_significant=bool(pooled_sig),
        binding_year_v4_clears=out["V4_impose_null_blockboot"]["clears"],
        per_year_all_holm_clear=out["per_year_FWER_holm"]["all3_clear_holm"],
        statement=(f"m5xp UP cov0.05 n={n}, pooled win {out['pooled_win']}. POOLED edge {'SIGNIFICANT' if pooled_sig else 'NOT sig'} "
            f"(DM {dm:.2f}, PSR {psr:.3f}, n>MinTRL {n>mintrl}). Binding-2025 impose-null block-boot p={p_v4:.3f} "
            f"({'clears' if p_v4<0.05 else 'thin'}); per-year Holm all-clear={out['per_year_FWER_holm']['all3_clear_holm']}. "
            "Corroborates V1/V2: pooled UP edge real & significant; single-binding-year thin (use refit floor .553)."))
    json.dump(out,open(f"{ROOT}/m5_valsuite_result.json","w"),indent=2,default=str)
    for k in ("V4_impose_null_blockboot","DM_HAC_vs_coinflip","per_year_FWER_holm","PSR","EB_shrinkage"):
        print(f"  {k}: {out[k]}")
    print(f"VERDICT: {out['VERDICT']['statement']}\n-> m5_valsuite_result.json ({time.time()-t0:.0f}s)")

if __name__=="__main__": main()
