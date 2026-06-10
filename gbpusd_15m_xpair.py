"""GBPUSD 15-MIN direction via CROSS-PAIR / USD-common-factor + GBPUSD-SPECIFIC own-pair channels.

SCOPE: GBPUSD · 15m (A6 keystone — the EUR-bloc pooling discriminator). Fork of audusd_15m_xpair.py with
TARGET=GBPUSD. Cross-pair USD-residual+lead-lag pooling was THE certifying keystone at EURUSD 15m (the one
sibling where pooling WON) but DILUTED own-pair signal at USDJPY+AUDUSD. GBPUSD is the tightest EURUSD
correlate among the majors → this screen decides which precedent GBPUSD follows. ADDS GBPUSD-specific
ON-DISK channels the EURUSD builder lacks:
  (1) EURGBP relative value (the SWEEP_MATRIX **N2 triangular USD-canceling residual**, uniquely on-disk
      here) — GBP vs its closest cousin EUR; log(EURGBP)=log(EURUSD)-log(GBPUSD) shares the USD leg, so the
      USD factor is ALGEBRAICALLY removed → immune to the USD-factor sign-inversion that killed RMT/xpair
      elsewhere; mean-reverting RV + velocity channels.
  (2) European-bloc factor (EUR+CHF USD-weakness mean) and GBP-beyond-bloc residual — GBP loads on the
      European bloc; the residual isolates UK-idiosyncratic (BoE/gilt) flow.
  (3) commodity-dollar vs safe-haven RISK-ON/OFF factor (GBP loads mildly risk-on; nearly free from the
      same panel).

Sign convention (GBPUSD is USD-QUOTE): equiv_sign(p)=+1 for USD-quote {EURUSD,GBPUSD,AUDUSD,NZDUSD}, -1 for
USD-base {USDJPY,USDCHF,USDCAD}. 'gbp_equiv' of p = equiv_sign(p)*r_p = move in GBPUSD-up (=USD-weakness)
direction. basket = mean over the other 6 = USD-weakness factor; catchup = basket - gbp_r (GBPUSD owes the
basket move → predicts +).

Label: next-15m GBPUSD sign, wall-clock contiguous (exactly 900s), ties excluded from train/AUC (charged as
losses in win-rate). Per-year held-out (2024/25/26) COMBINED+UP+DOWN covcurve. all-session (A9 session test
is separate). This is the SCREEN: does pooling+gbp-specific LIFT the base VAL AUC and rescue the dead-2026
frozen tail? If yes → refit-CPCV cert via the xpair feature matrix.

Incumbent constants are LOADED from gbpusd_15m_base_result.json (no transcription).
Pre-registered falsifier in gbpusd_15m_xpair_<mode>_result.json BEFORE held-out read.
Usage: ~/binary-algo-venv/bin/python gbpusd_15m_xpair.py [mode=xp|xpbase] [stride_train=4]
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H

FEAT="/home/sean/git/binary-algo/features"
PAIRS=["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCHF","USDCAD"]
USD_BASE={"USDJPY","USDCHF","USDCAD"}
TARGET="GBPUSD"
NONGBP=[p for p in PAIRS if p!=TARGET]
COMMOD=["AUDUSD","NZDUSD","USDCAD"]   # commodity dollars (AUD/NZD USD-quote, CAD USD-base)
SAFEHAVEN=["USDJPY","USDCHF"]
LB=[1,3,5,10,15,30]
HOR=int(os.environ.get("MX_HOR","15")); GAP_S=HOR*60
BE=0.541
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],"test24":["2024"],"test25":["2025"],"oos":["2026"]}
_base=json.load(open("gbpusd_15m_base_result.json"))
BASE_VAL_AUC=float(_base["val_auc"])                                            # .5285
BASE_2026_COV2_COMB=float(_base["years"]["oos"]["covcurve"]["0.02"]["COMBINED"]["wr"])  # ~.501

def equiv_sign(p): return -1.0 if p in USD_BASE else +1.0

def build_xp_gbp(years, stride=1, keep_ties=False):
    """Cross-pair (GBPUSD-target) + GBPUSD-specific features + label _y, _ts, _fwd. Ties excluded by
    default (screen behavior); keep_ties=True retains fwd==0 rows so a cert harness can charge them
    as losses (deriv-faithful ties-LOSE)."""
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
            gbp_r=rets[TARGET][k]
            basket=np.nanmean(np.vstack([equiv_sign(p)*rets[p][k] for p in NONGBP]),axis=0)
            disp=np.nanstd(np.vstack([equiv_sign(p)*rets[p][k] for p in NONGBP]),axis=0)
            agree=np.nanmean(np.vstack([(np.sign(equiv_sign(p)*rets[p][k])==np.sign(basket)).astype(float) for p in NONGBP]),axis=0)
            feats[f"gbp_r{k}"]=gbp_r
            feats[f"usdbask{k}"]=basket
            feats[f"catchup{k}"]=basket-gbp_r          # GBPUSD owes the basket move (predicts +)
            feats[f"gbpresid{k}"]=gbp_r-basket          # GBP idiosyncratic (beyond USD factor)
            feats[f"disp{k}"]=disp
            feats[f"agree{k}"]=agree
            for p in NONGBP:
                feats[f"ll_{p}{k}"]=equiv_sign(p)*rets[p][k]-gbp_r   # pair lead-lag residual vs GBPUSD
            # --- GBPUSD-specific block ---
            eur_r=rets["EURUSD"][k]                      # EURUSD USD-quote -> raw return = EUR USD-weakness
            feats[f"eurgbp_r{k}"]=gbp_r-eur_r            # GBP-vs-EUR cross return (N2 triangular, USD-factor-free)
            eurobloc=np.nanmean(np.vstack([rets["EURUSD"][k], equiv_sign("USDCHF")*rets["USDCHF"][k]]),axis=0)
            feats[f"eurobloc{k}"]=eurobloc               # European-bloc USD-weakness factor (EUR+CHF)
            feats[f"gbpbloc_resid{k}"]=gbp_r-eurobloc    # GBP beyond the European bloc (UK-idiosyncratic)
            commod=np.nanmean(np.vstack([equiv_sign(p)*rets[p][k] for p in COMMOD]),axis=0)
            safeh=np.nanmean(np.vstack([equiv_sign(p)*rets[p][k] for p in SAFEHAVEN]),axis=0)
            feats[f"risk{k}"]=commod-safeh               # risk-on factor; GBP loads mildly +
        # EURGBP mean-reversion: deviation of log(EURGBP) from its rolling anchor (N2 RV channel)
        eurgbp=lr["EURUSD"]-lr[TARGET]                   # log(EURGBP)
        for win in (60,240):
            anchor=pd.Series(eurgbp).rolling(win,min_periods=win//2).mean().values
            feats[f"eurgbp_dev{win}"]=eurgbp-anchor      # >0 EUR rich vs GBP -> reversion predicts GBPUSD up vs EURUSD
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
        valid=np.isfinite(fwd) if keep_ties else (np.isfinite(fwd)&(fwd!=0))
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
    RESULT=f"gbpusd_15m_xpair_{mode}_result.json"
    res={"key":"GBPUSD.15m","model":f"cross-pair USD-residual + GBPUSD-specific (EURGBP N2 RV, euro-bloc, risk factor), mode={mode}",
         "incumbent":f"A1 base VAL AUC {BASE_VAL_AUC:.4f}, 2026 cov2% COMB {BASE_2026_COV2_COMB:.4f}","breakeven":BE,
         "falsifier":{"registered":"pre-OOS",
            "IMPROVES_if":f"VAL moved-AUC > {BASE_VAL_AUC:.4f} AND 2026 cov2% COMBINED wr > {BASE_2026_COV2_COMB:.4f} (else base stands; pooling subsumed)"}}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[xpair GBPUSD] mode={mode} stride={stride} HOR={HOR} (base VAL {BASE_VAL_AUC:.4f}, base 2026cov2 {BASE_2026_COV2_COMB:.4f})",flush=True)
    TR=build_xp_gbp(SPL["train"],stride); VA=build_xp_gbp(SPL["val"])
    xpc=xp_cols(TR); TR=augment(TR,SPL["train"],mode); VA=augment(VA,SPL["val"],mode)
    cols=feat_cols(mode,TR,xpc)
    ytr=TR["_y"].astype(int).values; yva=VA["_y"].astype(int).values
    Xtr=TR[cols].astype("float32"); Xva=VA[cols].astype("float32")
    print(f"[xpair GBPUSD] train={len(TR):,} val={len(VA):,} feats={len(cols)} build={time.time()-t0:.0f}s",flush=True)
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=127,min_child_samples=400,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,n_estimators=3000,n_jobs=20,verbosity=-1)
    L.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pva=L.predict_proba(Xva)[:,1]; aucv=float(roc_auc_score(yva,pva))
    imp=sorted(zip(cols,L.feature_importances_),key=lambda z:-z[1])[:20]
    res["val_auc"]=aucv; res["best_iter"]=int(L.best_iteration_ or 0); res["top20"]=[c for c,_ in imp]
    print(f"[xpair GBPUSD] best_iter={L.best_iteration_} VAL AUC={aucv:.4f} (base {BASE_VAL_AUC:.4f})",flush=True)
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
    print(f"[xpair GBPUSD] gate cov{COV:.0%} thr={THR:.4f} VAL worst-half={worst:.4f}",flush=True)
    res["years"]={}
    for w in ("test24","test25","oos"):
        D=build_xp_gbp(SPL[w]); D=augment(D,SPL[w],mode)
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
    print(f"\n[xpair GBPUSD] VERDICT IMPROVES_base={improves} (VAL {aucv:.4f} vs {BASE_VAL_AUC:.4f}; 2026 cov2 COMB {a26} vs {BASE_2026_COV2_COMB:.4f}) -> {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "xpbase"
    stride=int(sys.argv[2]) if len(sys.argv)>2 else 4
    main(mode,stride)
