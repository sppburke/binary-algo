"""USDCHF 15-MIN direction — ★NEW synthetic-EURCHF LEVEL error-correction band (SNB-pinned), discovery R1 cand #3.

SCOPE: USDCHF · 15m (discovery R1). The ONE genuinely-novel CHF-unique on-disk lever from discovery: the SNB pins
the EURCHF *LEVEL* into a slow band (2011-15 hard floor, then soft management + neg rates), so the synthetic cross
  logEURCHF = logEURUSD + logUSDCHF        (EURCHF = EUR/USD * USD/CHF; both legs on disk)
mean-reverts toward a slow anchor. A CAUSAL rolling z-score of that LEVEL carries a RESTORING-FORCE SIGN
(target-zone / Krugman S-curve conditional-MEAN drift) — which, because the SNB manages via the CHF leg, maps to a
signed prediction on USDCHF. This is a DIRECTION lever (conditional-mean drift on a policy-stationary LEVEL), NOT a
vol/entropy statistic → survives sign-invariance. It is ORTHOGONAL to cross-pair POOLING: pooling uses RETURN
residuals / lead-lag (corr~0 at 15m here); ECM uses the LEVEL, an axis the 239-feat base + xpair return-residuals do
NOT span (verified: no level/band/ECM feature in usdchf_15m_xpair.py).

FEATURES (causal, multi-window anchors W minutes): zW=(logEURCHF - rollmean_W)/rollstd_W, |zW|, slope of anchor,
and the leg-share (rolling variance attribution of EURCHF moves to the USDCHF leg vs the EURUSD leg) so the GBM can
learn WHEN the reversion routes through CHF. Augmented onto the base 239 USDCHF feats. all-session SCREEN: does
base+ECM LIFT base VAL AUC .5301 and the 2026 tail? If yes → escalate to carrier-session refit-CPCV.

ADVERSARIAL (trap: a non-stationary LEVEL feature can memorize era-local band structure — strategy-eval trap#9 cousin):
report the sign of corr(z, fwd) per era (2024 vs 2026); KILL if the z↔fwd sign FLIPS across eras (regime-local, no
forward transfer) even if VAL lifts.

Pre-registered falsifier in usdchf_15m_ecm_result.json BEFORE OOS.
Usage: ~/binary-algo-venv/bin/python usdchf_15m_ecm.py [mode=ecmbase|ecmonly] [stride_train=4]
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H

FEAT="/home/sean/git/binary-algo/features"
TARGET="USDCHF"; LEGS=["EURUSD","USDCHF"]    # logEURCHF = logEURUSD + logUSDCHF
HOR=int(os.environ.get("MX_HOR","15")); GAP_S=HOR*60; BE=0.541
ANCHORS=[120,240,480,960,1920]               # slow-band anchor windows (minutes), causal
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],"test24":["2024"],"test25":["2025"],"oos":["2026"]}
BASE_VAL_AUC=0.5301; BASE_2026_COV2_COMB=0.5103   # A1 incumbent to beat (usdchf_15m_base_result.json)

def build_ecm(years, stride=1, mode="ecmbase"):
    """Synthetic-EURCHF LEVEL z-band ECM feats + USDCHF 15m label. Ties excluded from train/AUC."""
    out=[]
    for y in years:
        cl={}; ok=True
        for p in LEGS:
            fp=f"{FEAT}/{p}_{y}.parquet"
            if not os.path.exists(fp): ok=False; break
            d=pd.read_parquet(fp,columns=["close"]); d=d[~d.index.duplicated(keep="last")]
            cl[p]=d["close"]
        if not ok: continue
        df=pd.DataFrame(cl).dropna(); df=df[~df.index.duplicated(keep="last")]
        if len(df)<max(ANCHORS)+100: continue
        idx=df.index; secs=idx.values.astype("datetime64[s]").astype("int64"); n=len(df)
        leur=np.log(df["EURUSD"].values); lchf=np.log(df["USDCHF"].values)
        lcross=leur+lchf                                  # logEURCHF (synthetic, both legs on disk)
        feats={}
        scr=pd.Series(lcross,index=idx)
        # per-leg 1-step log-returns for the leg-share (variance attribution)
        rE=pd.Series(np.concatenate([[np.nan],np.diff(leur)]),index=idx)
        rC=pd.Series(np.concatenate([[np.nan],np.diff(lchf)]),index=idx)
        for W in ANCHORS:
            m=scr.rolling(W,min_periods=W//2).mean()
            s=scr.rolling(W,min_periods=W//2).std()
            z=(scr-m)/s.replace(0,np.nan)                 # causal LEVEL z vs slow anchor
            feats[f"z{W}"]=z.values
            feats[f"absz{W}"]=z.abs().values
            feats[f"anchslope{W}"]=(m-m.shift(W//4)).values   # band drift (anchor slope)
            # leg-share: fraction of recent EURCHF level-variance carried by the CHF leg
            vC=rC.rolling(W,min_periods=W//2).var(); vE=rE.rolling(W,min_periods=W//2).var()
            feats[f"chfshare{W}"]=(vC/(vC+vE).replace(0,np.nan)).values
        # label (raw USDCHF direction, wall-clock contiguous)
        fwd=np.full(n,np.nan)
        if n>HOR:
            contig=(secs[HOR:]-secs[:-HOR])==HOR*60
            fr=lchf[HOR:]-lchf[:-HOR]
            fwd[:n-HOR]=np.where(contig,fr,np.nan)
        F=pd.DataFrame(feats,index=idx).astype("float32")
        F["_y"]=(fwd>0).astype("float32"); F["_ts"]=secs.astype("int64"); F["_fwd"]=fwd.astype("float64")
        valid=np.isfinite(fwd)&(fwd!=0)
        F=F.loc[valid]
        if stride>1: F=F.iloc[::stride]
        out.append(F)
    return pd.concat(out)

def ecm_cols(df): return [c for c in df.columns if c not in ("_y","_ts","_fwd")]

def augment(F, years, mode):
    if mode!="ecmbase": return F
    cols=[c for c in H.feature_cols(TARGET) if c not in F.columns]
    idx=F.index; parts=[]
    for y in years:
        p=f"{FEAT}/{TARGET}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p,columns=cols); d=d[~d.index.duplicated(keep="last")]
        keep=idx.intersection(d.index)
        if len(keep): parts.append(d.reindex(keep).astype("float32"))
        del d
    if parts:
        B=pd.concat(parts); F=F.join(B,how="left"); del B
    return F

def feat_cols(mode, df, ecmc):
    cols=list(ecmc)
    if mode=="ecmbase": cols+=[c for c in H.feature_cols(TARGET) if c in df.columns]
    return list(dict.fromkeys(cols))

def nonoverlap_chrono(ts, mask, gap=GAP_S):
    take=[]; block=-1
    for i in np.where(mask)[0]:
        if ts[i]<block: continue
        take.append(i); block=int(ts[i])+gap
    return np.array(take,dtype=int)

def boot(corr, nb=5000, seed=7):
    corr=np.asarray(corr,float)
    if len(corr)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(corr)
    a=np.array([corr[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))

def side_eval(pr,y,fwd,ts,thr):
    conf=np.abs(pr-0.5); tr=nonoverlap_chrono(ts, conf>=thr)
    if len(tr)==0: return None
    pred=(pr[tr]>0.5).astype(int); moved=(fwd[tr]!=0)
    win=((pred==y[tr]) & moved).astype(float); out={}
    lo,hi=boot(win); out["COMBINED"]={"n":int(len(tr)),"wr":round(float(win.mean()),4),"ci":[round(lo,3),round(hi,3)]}
    for nm,msk in (("UP",pred==1),("DOWN",pred==0)):
        if msk.sum()>0:
            lo,hi=boot(win[msk]); out[nm]={"n":int(msk.sum()),"wr":round(float(win[msk].mean()),4),"ci":[round(lo,3),round(hi,3)]}
        else: out[nm]={"n":0,"wr":float("nan"),"ci":[float("nan")]*2}
    return out

def covcurve(pr,y,fwd,ts,covs=(0.10,0.05,0.03,0.02,0.01)):
    conf=np.abs(pr-0.5); rows={}
    for cov in covs:
        thr=float(np.quantile(conf,1-cov)); r=side_eval(pr,y,fwd,ts,thr)
        rows[f"{cov:.2f}"]={"thr":round(thr,4), **({k:{"n":v["n"],"wr":v["wr"]} for k,v in r.items()} if r else {})}
    return rows

def main(mode="ecmbase", stride=4):
    t0=time.time()
    RESULT=f"usdchf_15m_ecm_{mode}_result.json"
    res={"key":"USDCHF.15m","model":f"synthetic-EURCHF LEVEL error-correction z-band (SNB-pinned), mode={mode}",
         "incumbent":"A1 base VAL AUC .5301, 2026 cov2% COMB .5103","breakeven":BE,"anchors":ANCHORS,
         "hypothesis":"SNB pins EURCHF LEVEL -> causal rolling-z of logEURCHF=logEURUSD+logUSDCHF carries restoring-force SIGN -> maps to USDCHF leg. Orthogonal to return-residual pooling.",
         "falsifier":{"registered":"pre-OOS",
            "IMPROVES_if":"VAL moved-AUC > 0.5301 AND 2026 cov2% COMBINED wr > 0.5103",
            "ADVERSARIAL_KILL_if":"sign(corr(z960,fwd)) FLIPS between 2024 and 2026 (era-local band memorization, no forward transfer)"}}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[ecm USDCHF] mode={mode} stride={stride} HOR={HOR} anchors={ANCHORS}",flush=True)
    TR=build_ecm(SPL["train"],stride,mode); VA=build_ecm(SPL["val"],1,mode)
    ecmc=ecm_cols(TR); TR=augment(TR,SPL["train"],mode); VA=augment(VA,SPL["val"],mode)
    cols=feat_cols(mode,TR,ecmc)
    ytr=TR["_y"].astype(int).values; yva=VA["_y"].astype(int).values
    Xtr=TR[cols].astype("float32"); Xva=VA[cols].astype("float32")
    print(f"[ecm USDCHF] train={len(TR):,} val={len(VA):,} feats={len(cols)} (ecm={len(ecmc)}) build={time.time()-t0:.0f}s",flush=True)
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=127,min_child_samples=400,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,n_estimators=3000,n_jobs=12,verbosity=-1)
    L.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pva=L.predict_proba(Xva)[:,1]; aucv=float(roc_auc_score(yva,pva))
    imp=sorted(zip(cols,L.feature_importances_),key=lambda z:-z[1])[:20]
    ecm_in_top=[c for c in (cc for cc,_ in imp) if c in ecmc]
    res["val_auc"]=aucv; res["best_iter"]=int(L.best_iteration_ or 0); res["top20"]=[c for c,_ in imp]; res["ecm_in_top20"]=ecm_in_top
    print(f"[ecm USDCHF] best_iter={L.best_iteration_} VAL AUC={aucv:.4f} (base .5301); ecm feats in top20: {ecm_in_top}",flush=True)
    # gate on VAL worst-half
    tsv=VA["_ts"].values.astype("int64"); fwv=VA["_fwd"].values; half=len(pva)//2; confv=np.abs(pva-0.5); best=None
    for cov in (0.10,0.05,0.03,0.02):
        thr=float(np.quantile(confv,1-cov)); accs=[]
        for s,e in ((0,half),(half,len(pva))):
            r=side_eval(pva[s:e],yva[s:e],fwv[s:e],tsv[s:e],thr); accs.append(r["COMBINED"]["wr"] if r else float("nan"))
        worst=np.nanmin(accs)
        if best is None or worst>best[0]: best=(worst,cov,thr)
    worst,COV,THR=best; res["gate"]={"cov":COV,"thr":THR,"val_worst_half":round(float(worst),4)}
    print(f"[ecm USDCHF] gate cov{COV:.0%} thr={THR:.4f} VAL worst-half={worst:.4f}",flush=True)
    res["years"]={}; z_sign={}
    for w in ("test24","test25","oos"):
        D=build_ecm(SPL[w],1,mode); D=augment(D,SPL[w],mode)
        X=D[cols].astype("float32"); pr=L.predict_proba(X)[:,1]; y=D["_y"].astype(int).values
        fwd=D["_fwd"].values; ts=D["_ts"].values.astype("int64")
        mv=fwd!=0; auc=float(roc_auc_score(y[mv],pr[mv])); uprate=float(y[mv].mean())
        # adversarial: sign of corr(z960, fwd) per era
        if "z960" in D.columns:
            zz=D["z960"].values; m2=mv & np.isfinite(zz)
            z_sign[w]=float(np.corrcoef(zz[m2], fwd[m2])[0,1]) if m2.sum()>50 else float("nan")
        gate=side_eval(pr,y,fwd,ts,THR); cc=covcurve(pr,y,fwd,ts)
        res["years"][w]={"moved_auc":round(auc,4),"up_rate":round(uprate,4),"tripwire_ok":bool(0.47<=uprate<=0.53),
                         "gate":gate,"covcurve":cc}
        g=gate["COMBINED"] if gate else {}
        print(f"=== {w} === AUC={auc:.4f} up={uprate:.4f} corr(z960,fwd)={z_sign.get(w,float('nan')):.4f} | gate cov{COV:.0%} COMB n{g.get('n')} wr={g.get('wr')} CI{g.get('ci')}",flush=True)
        c2=cc.get("0.02",{}); print(f"    cov2%: COMB {c2.get('COMBINED')} UP {c2.get('UP')} DOWN {c2.get('DOWN')}",flush=True)
    a26=res["years"]["oos"]["covcurve"].get("0.02",{}).get("COMBINED",{}).get("wr",float("nan"))
    res["z960_corr_by_era"]=z_sign
    sign_flip=bool(np.isfinite(z_sign.get("test24",np.nan)) and np.isfinite(z_sign.get("oos",np.nan)) and (np.sign(z_sign["test24"])!=np.sign(z_sign["oos"])))
    improves=bool(aucv>BASE_VAL_AUC and np.isfinite(a26) and a26>BASE_2026_COV2_COMB)
    res["verdict"]={"val_auc_gt_base":bool(aucv>BASE_VAL_AUC),"oos_cov2_comb":a26,"oos_cov2_gt_base":bool(np.isfinite(a26) and a26>BASE_2026_COV2_COMB),
                    "z960_sign_flip_2024_vs_2026":sign_flip,"IMPROVES_base":bool(improves and not sign_flip),
                    "note":"IMPROVES requires VAL>base AND 2026 cov2>base AND NO z-sign flip across eras (else era-local band memorization, no transfer)."}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[ecm USDCHF] VERDICT IMPROVES_base={res['verdict']['IMPROVES_base']} (VAL {aucv:.4f} vs .5301; 2026 cov2 COMB {a26} vs .5103; z-sign-flip={sign_flip}) -> {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "ecmbase"
    stride=int(sys.argv[2]) if len(sys.argv)>2 else 4
    main(mode,stride)
