"""AUDUSD 15-MIN direction via CROSS-PAIR / USD-common-factor + AUDUSD-SPECIFIC own-pair channels.

SCOPE: AUDUSD · 15m (A6 keystone). Retarget of m5_xpair.py to AUDUSD-as-TARGET at MX_HOR=15. The EURUSD 15m
certifying keystone is cross-pair USD-residual+lead-lag pooling; AUDUSD is a USD-major so the same lever is the
strong prior. ADDS AUDUSD-specific ON-DISK channels the EURUSD builder lacks:
  (1) AUDNZD relative value — AUD vs its closest cousin NZD (NZDUSD on disk); the AUD-NZD spread is
      mean-reverting and DECORRELATED from the USD common factor → a genuine OWN-pair direction channel.
  (2) commodity-dollar (AUD/NZD/CAD) vs safe-haven (JPY/CHF) RISK-ON/OFF factor — a second factor orthogonal
      to the USD level, extractable from the 7-pair panel WITHOUT external VIX/equity data; AUDUSD loads
      positively on risk-on.

Sign convention (same as EURUSD: AUDUSD is USD-QUOTE): equiv_sign(p)=+1 for USD-quote {EURUSD,GBPUSD,AUDUSD,
NZDUSD}, -1 for USD-base {USDJPY,USDCHF,USDCAD}. 'aud_equiv' of p = equiv_sign(p)*r_p = move in AUDUSD-up
(=USD-weakness) direction. basket = mean over the other 6 = USD-weakness factor; catchup = basket - aud_r
(AUDUSD owes the basket move → predicts +).

Label: next-15m AUDUSD sign, wall-clock contiguous (exactly 900s), ties excluded from train/AUC (charged as
losses in win-rate). Per-year held-out (2024/25/26) COMBINED+UP+DOWN covcurve. all-session (A9 session test is
separate). This is the SCREEN: does pooling+aud-specific LIFT the base AUC .5232 and rescue the binding-2026
tail? If yes → refit-CPCV cert via the xpair feature matrix.

Pre-registered falsifier in audusd_15m_xpair_<mode>_result.json BEFORE held-out read.
Usage: ~/binary-algo-venv/bin/python audusd_15m_xpair.py [mode=xp|xpbase] [stride_train=4]
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H

FEAT="/home/sean/git/binary-algo/features"
PAIRS=["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCHF","USDCAD"]
USD_BASE={"USDJPY","USDCHF","USDCAD"}
TARGET="AUDUSD"
NONAUD=[p for p in PAIRS if p!=TARGET]
COMMOD_EXAUD=["NZDUSD","USDCAD"]   # commodity dollars ex-AUD (NZD USD-quote, CAD USD-base)
SAFEHAVEN=["USDJPY","USDCHF"]
LB=[1,3,5,10,15,30]
HOR=int(os.environ.get("MX_HOR","15")); GAP_S=HOR*60
BE=0.541
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],"test24":["2024"],"test25":["2025"],"oos":["2026"]}
BASE_VAL_AUC=0.5232; BASE_2026_COV2_COMB=0.5588   # A1 incumbent to beat (audusd_15m_base_result.json)

def equiv_sign(p): return -1.0 if p in USD_BASE else +1.0

def build_xp_aud(years, stride=1):
    """Cross-pair (AUDUSD-target) + AUDUSD-specific features + label _y, _ts, _fwd. Ties excluded."""
    out=[]
    for y in years:
        cl={}; ok=True
        for p in PAIRS:
            fp=f"{FEAT}/{p}_{y}.parquet"
            if not os.path.exists(fp): ok=False; break
            d=pd.read_parquet(fp,columns=["close"]); d=d[~d.index.duplicated(keep="last")]
            cl[p]=d["close"]
        if not ok: continue
        df=pd.DataFrame(cl).dropna()
        if len(df)<100: continue
        idx=df.index; secs=idx.values.astype("datetime64[s]").astype("int64"); n=len(df)
        lr={p:np.log(df[p].values) for p in PAIRS}
        rets={p:{k:np.concatenate([[np.nan]*k, lr[p][k:]-lr[p][:-k]]) for k in LB} for p in PAIRS}
        feats={}
        for k in LB:
            aud_r=rets[TARGET][k]
            basket=np.nanmean(np.vstack([equiv_sign(p)*rets[p][k] for p in NONAUD]),axis=0)
            disp=np.nanstd(np.vstack([equiv_sign(p)*rets[p][k] for p in NONAUD]),axis=0)
            agree=np.nanmean(np.vstack([(np.sign(equiv_sign(p)*rets[p][k])==np.sign(basket)).astype(float) for p in NONAUD]),axis=0)
            feats[f"aud_r{k}"]=aud_r
            feats[f"usdbask{k}"]=basket
            feats[f"catchup{k}"]=basket-aud_r          # AUDUSD owes the basket move (predicts +)
            feats[f"audresid{k}"]=aud_r-basket          # AUD idiosyncratic (beyond USD factor)
            feats[f"disp{k}"]=disp
            feats[f"agree{k}"]=agree
            for p in NONAUD:
                feats[f"ll_{p}{k}"]=equiv_sign(p)*rets[p][k]-aud_r   # pair lead-lag residual vs AUDUSD
            # --- AUDUSD-specific block ---
            nzd_r=rets["NZDUSD"][k]                      # NZDUSD USD-quote -> raw return = NZD USD-weakness
            feats[f"audnzd_r{k}"]=aud_r-nzd_r            # AUDNZD cross return (own-pair, USD-factor-free)
            commod=np.nanmean(np.vstack([equiv_sign(p)*rets[p][k] for p in COMMOD_EXAUD]),axis=0)
            safeh=np.nanmean(np.vstack([equiv_sign(p)*rets[p][k] for p in SAFEHAVEN]),axis=0)
            feats[f"risk{k}"]=commod-safeh               # risk-on factor (commodity$ vs safe-haven); AUDUSD loads +
            feats[f"audrisk_resid{k}"]=aud_r-(commod-safeh)  # AUD beyond the risk factor
        # AUDNZD mean-reversion: deviation of log(AUDNZD) from its rolling anchor
        audnzd=lr[TARGET]-lr["NZDUSD"]
        for win in (60,240):
            anchor=pd.Series(audnzd).rolling(win,min_periods=win//2).mean().values
            feats[f"audnzd_dev{win}"]=audnzd-anchor      # >0 AUD rich vs NZD -> reversion predicts AUD down
        # session + compression gate cols
        hours=idx.hour.values+idx.minute.values/60.0
        feats["sess_ny"]=((hours>=13.0)&(hours<22.0)).astype(float)
        feats["sess_ln"]=((hours>=7.0)&(hours<16.0)).astype(float)
        feats["sess_as"]=((hours>=0.0)&(hours<9.0)).astype(float)
        feats["hour"]=hours
        r1=rets[TARGET][1]
        feats["comp60"]=pd.Series(r1).rolling(60,min_periods=20).std().values
        # label
        fwd=np.full(n,np.nan)
        if n>HOR:
            contig=(secs[HOR:]-secs[:-HOR])==HOR*60
            fr=lr[TARGET][HOR:]-lr[TARGET][:-HOR]
            fwd[:n-HOR]=np.where(contig,fr,np.nan)
        F=pd.DataFrame(feats,index=idx); F["_y"]=(fwd>0).astype(float); F["_ts"]=secs; F["_fwd"]=fwd
        valid=np.isfinite(fwd)&(fwd!=0)
        F=F.loc[valid]
        if stride>1: F=F.iloc[::stride]
        out.append(F)
    return pd.concat(out)

def xp_cols(df): return [c for c in df.columns if c not in ("_y","_ts","_fwd","hour")]

def augment(F, years, mode):
    if mode=="xpbase":
        cols=list(H.feature_cols(TARGET))
        parts=[]
        for y in years:
            p=f"{FEAT}/{TARGET}_{y}.parquet"
            if not os.path.exists(p): continue
            d=pd.read_parquet(p,columns=cols); d=d[~d.index.duplicated(keep="last")]; parts.append(d)
        B=pd.concat(parts) if parts else None
        if B is not None: F=F.join(B[[c for c in B.columns if c not in F.columns]],how="left")
    return F

def feat_cols(mode, df, xpc):
    cols=list(xpc)
    if mode=="xpbase": cols+=[c for c in H.feature_cols(TARGET) if c in df.columns]
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

def main(mode="xpbase", stride=4):
    t0=time.time()
    RESULT=f"audusd_15m_xpair_{mode}_result.json"
    res={"key":"AUDUSD.15m","model":f"cross-pair USD-residual + AUDUSD-specific (audnzd RV, risk factor), mode={mode}",
         "incumbent":"A1 base VAL AUC .5232, 2026 cov2% COMB .5588","breakeven":BE,
         "falsifier":{"registered":"pre-OOS",
            "IMPROVES_if":"VAL moved-AUC > 0.5232 AND 2026 cov2% COMBINED wr > 0.5588 (else base stands; pooling subsumed)"}}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[xpair AUDUSD] mode={mode} stride={stride} HOR={HOR}",flush=True)
    TR=build_xp_aud(SPL["train"],stride); VA=build_xp_aud(SPL["val"])
    xpc=xp_cols(TR); TR=augment(TR,SPL["train"],mode); VA=augment(VA,SPL["val"],mode)
    cols=feat_cols(mode,TR,xpc)
    ytr=TR["_y"].astype(int).values; yva=VA["_y"].astype(int).values
    Xtr=TR[cols].astype("float32"); Xva=VA[cols].astype("float32")
    print(f"[xpair AUDUSD] train={len(TR):,} val={len(VA):,} feats={len(cols)} build={time.time()-t0:.0f}s",flush=True)
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=127,min_child_samples=400,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,n_estimators=3000,n_jobs=20,verbosity=-1)
    L.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pva=L.predict_proba(Xva)[:,1]; aucv=float(roc_auc_score(yva,pva))
    imp=sorted(zip(cols,L.feature_importances_),key=lambda z:-z[1])[:20]
    res["val_auc"]=aucv; res["best_iter"]=int(L.best_iteration_ or 0); res["top20"]=[c for c,_ in imp]
    print(f"[xpair AUDUSD] best_iter={L.best_iteration_} VAL AUC={aucv:.4f} (base .5232)",flush=True)
    print("  top20:",", ".join(c for c,_ in imp),flush=True)
    # VAL worst-half gate (cov sweep), never VAL-acc-max
    tsv=VA["_ts"].values.astype("int64"); fwv=VA["_fwd"].values; half=len(pva)//2; confv=np.abs(pva-0.5); best=None
    for cov in (0.10,0.05,0.03,0.02):
        thr=float(np.quantile(confv,1-cov)); accs=[]
        for s,e in ((0,half),(half,len(pva))):
            r=side_eval(pva[s:e],yva[s:e],fwv[s:e],tsv[s:e],thr); accs.append(r["COMBINED"]["wr"] if r else float("nan"))
        worst=np.nanmin(accs)
        if best is None or worst>best[0]: best=(worst,cov,thr)
    worst,COV,THR=best; res["gate"]={"cov":COV,"thr":THR,"val_worst_half":round(float(worst),4)}
    print(f"[xpair AUDUSD] gate cov{COV:.0%} thr={THR:.4f} VAL worst-half={worst:.4f}",flush=True)
    res["years"]={}
    for w in ("test24","test25","oos"):
        D=build_xp_aud(SPL[w]); D=augment(D,SPL[w],mode)
        X=D[cols].astype("float32"); pr=L.predict_proba(X)[:,1]; y=D["_y"].astype(int).values
        fwd=D["_fwd"].values; ts=D["_ts"].values.astype("int64")
        mv=fwd!=0; auc=float(roc_auc_score(y[mv],pr[mv])); uprate=float(y[mv].mean())
        gate=side_eval(pr,y,fwd,ts,THR); cc=covcurve(pr,y,fwd,ts)
        res["years"][w]={"moved_auc":round(auc,4),"up_rate":round(uprate,4),"tripwire_ok":bool(0.47<=uprate<=0.53),
                         "gate":gate,"covcurve":cc}
        g=gate["COMBINED"] if gate else {}
        print(f"=== {w} === AUC={auc:.4f} up={uprate:.4f} | gate cov{COV:.0%} COMB n{g.get('n')} wr={g.get('wr')} CI{g.get('ci')}",flush=True)
        c2=cc.get("0.02",{}); print(f"    cov2%: COMB {c2.get('COMBINED')} UP {c2.get('UP')} DOWN {c2.get('DOWN')}",flush=True)
    a26=res["years"]["oos"]["covcurve"].get("0.02",{}).get("COMBINED",{}).get("wr",float("nan"))
    improves=bool(aucv>BASE_VAL_AUC and np.isfinite(a26) and a26>BASE_2026_COV2_COMB)
    res["verdict"]={"val_auc_gt_base":bool(aucv>BASE_VAL_AUC),"oos_cov2_comb":a26,"oos_cov2_gt_base":bool(np.isfinite(a26) and a26>BASE_2026_COV2_COMB),
                    "IMPROVES_base":improves,"note":"if not IMPROVES, base stands; pooling subsumed at 15m (record + move to session/cert)"}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[xpair AUDUSD] VERDICT IMPROVES_base={improves} (VAL {aucv:.4f} vs .5232; 2026 cov2 COMB {a26} vs .5588) -> {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "xpbase"
    stride=int(sys.argv[2]) if len(sys.argv)>2 else 4
    main(mode,stride)
