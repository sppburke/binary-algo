"""USDCAD 15-MIN direction via CROSS-PAIR / USD-common-factor + USDCAD-specific channels (A6 keystone screen).

SCOPE: USDCAD · 15m (A6 / discovery R1-2). Fork of audusd_15m_xpair.py with TARGET=USDCAD. The EUR-bloc 15m
certifying keystone is cross-pair USD-residual+lead-lag pooling. USDCAD is a USD-major so the lever is a
candidate, BUT both closest analogs (USDJPY USD-base, AUDUSD commodity) DILUTED under pooling → low prior;
RUN-don't-argue.

SIGN ALIGNMENT (load-bearing — USDCAD is USD-BASE, USD in the NUMERATOR). equiv_sign(p)=+1 for USD-quote
{EURUSD,GBPUSD,AUDUSD,NZDUSD}, -1 for USD-base {USDJPY,USDCHF,USDCAD}. To keep the TARGET in its RAW frame
("USDCAD-up = USD-strength"), align every pair to the target-up frame with
   aln(p) = equiv_sign(TARGET) * equiv_sign(p)
so aln(TARGET)=+1 (target raw) and aln(p)*r_p = pair move in the USDCAD-up (=USD-strength) direction. The
AUDUSD builder implicitly had equiv_sign(TARGET)=+1; here equiv_sign(USDCAD)=-1 INVERTS the basket vs the
AUDUSD/NZDUSD case. basket = mean over the other 6 of aln(p)*r_p = USD-STRENGTH factor (USDCAD-up direction);
catchup = basket - usdcad_r (USDCAD owes the basket move → predicts +). USDCAD-specific block: commodity-bloc
(AUD/NZD) residual (oil-proxy) + risk factor (commod$ vs safe-haven) — USDCAD loads NEGATIVELY on risk-on, but
the GBM handles sign; the residual carries the CAD-idiosyncratic (oil-driven) component.

modes: xp (xpair feats only) | xpbase (xpair + 239 base) | dblortho (xpbase + double-orthogonalized
CAD-idiosyncratic residual: r_USDCAD purged of fac AND of commodity-bloc mean → on-disk OIL PROXY, R1-4).

Label: next-15m USDCAD sign (raw), wall-clock contiguous (900s), ties excluded from train/AUC (charged as
losses in win-rate). all-session SCREEN: does pooling LIFT base VAL AUC .5234 and the binding-year tail? If
yes → escalate to NY refit-CPCV. Pre-registered falsifier in usdcad_15m_xpair_<mode>_result.json BEFORE OOS.
Usage: ~/binary-algo-venv/bin/python usdcad_15m_xpair.py [mode=xpbase|xp|dblortho] [stride_train=4]
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H

FEAT="/home/sean/git/binary-algo/features"
PAIRS=["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCHF","USDCAD"]
USD_BASE={"USDJPY","USDCHF","USDCAD"}
TARGET="USDCAD"
NONTGT=[p for p in PAIRS if p!=TARGET]
COMMOD_EXTGT=["AUDUSD","NZDUSD"]   # commodity dollars ex-CAD (both USD-quote)
SAFEHAVEN=["USDJPY","USDCHF"]
LB=[1,3,5,10,15,30]
HOR=int(os.environ.get("MX_HOR","15")); GAP_S=HOR*60
BE=0.541
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],"test24":["2024"],"test25":["2025"],"oos":["2026"]}
BASE_VAL_AUC=0.5234; BASE_2026_COV2_COMB=0.537   # A1 incumbent to beat (usdcad_15m_base_result.json)

def equiv_sign(p): return -1.0 if p in USD_BASE else +1.0
def aln(p): return equiv_sign(TARGET)*equiv_sign(p)   # target-up frame: aln(TARGET)=+1

def build_xp(years, stride=1, mode="xpbase"):
    """Cross-pair (USDCAD-target, sign-aligned) + USDCAD-specific features + label _y,_ts,_fwd. Ties excluded."""
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
            tgt_r=rets[TARGET][k]                                  # raw USDCAD return (USD-strength up)
            basket=np.nanmean(np.vstack([aln(p)*rets[p][k] for p in NONTGT]),axis=0)   # USD-strength factor (USDCAD-up frame)
            disp=np.nanstd(np.vstack([aln(p)*rets[p][k] for p in NONTGT]),axis=0)
            agree=np.nanmean(np.vstack([(np.sign(aln(p)*rets[p][k])==np.sign(basket)).astype(float) for p in NONTGT]),axis=0)
            feats[f"tgt_r{k}"]=tgt_r
            feats[f"usdbask{k}"]=basket
            feats[f"catchup{k}"]=basket-tgt_r          # USDCAD owes the basket move (predicts +)
            feats[f"tgtresid{k}"]=tgt_r-basket          # USDCAD idiosyncratic (beyond USD factor)
            feats[f"disp{k}"]=disp
            feats[f"agree{k}"]=agree
            for p in NONTGT:
                feats[f"ll_{p}{k}"]=aln(p)*rets[p][k]-tgt_r   # pair lead-lag residual vs USDCAD (target frame)
            # --- USDCAD-specific block (commodity / oil-proxy) ---
            commod=np.nanmean(np.vstack([aln(p)*rets[p][k] for p in COMMOD_EXTGT]),axis=0)  # commod-dollars in USDCAD-up frame
            safeh=np.nanmean(np.vstack([aln(p)*rets[p][k] for p in SAFEHAVEN]),axis=0)
            feats[f"risk{k}"]=commod-safeh               # risk factor in USDCAD-up frame (USDCAD loads -, GBM handles)
            feats[f"cadcommod_resid{k}"]=tgt_r-commod    # USDCAD beyond the commodity bloc (CAD-idiosyncratic ~ oil-proxy)
            if mode=="dblortho":
                # double-orthogonalized: purge USD factor AND commodity-bloc-common -> CAD-idiosyncratic (R1-4)
                feats[f"dblortho{k}"]=tgt_r-basket-(commod-basket)   # = tgt_r - commod, purged of fac via basket; isolate vs both
        # session + compression gate cols
        hours=idx.hour.values+idx.minute.values/60.0
        feats["sess_ny"]=((hours>=13.0)&(hours<22.0)).astype(float)
        feats["sess_ln"]=((hours>=7.0)&(hours<16.0)).astype(float)
        feats["hour"]=hours
        r1=rets[TARGET][1]
        feats["comp60"]=pd.Series(r1).rolling(60,min_periods=20).std().values
        # label (raw USDCAD direction)
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
    if mode in ("xpbase","dblortho"):
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
    if mode in ("xpbase","dblortho"): cols+=[c for c in H.feature_cols(TARGET) if c in df.columns]
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
    RESULT=f"usdcad_15m_xpair_{mode}_result.json"
    res={"key":"USDCAD.15m","model":f"cross-pair USD-residual (sign-aligned, USD-numerator) + USDCAD-specific (commod-bloc/oil-proxy, risk), mode={mode}",
         "incumbent":"A1 base VAL AUC .5234, 2026 cov2% COMB .537; NY-certified base UP p10 .6044/DOWN .5791","breakeven":BE,
         "falsifier":{"registered":"pre-OOS",
            "IMPROVES_if":"VAL moved-AUC > 0.5234 AND 2026 cov2% COMBINED wr > 0.537 (else base stands; pooling subsumed = USDJPY/AUDUSD case). If screen lifts -> escalate to NY refit-CPCV."}}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[xpair USDCAD] mode={mode} stride={stride} HOR={HOR}",flush=True)
    TR=build_xp(SPL["train"],stride,mode); VA=build_xp(SPL["val"],1,mode)
    xpc=xp_cols(TR); TR=augment(TR,SPL["train"],mode); VA=augment(VA,SPL["val"],mode)
    cols=feat_cols(mode,TR,xpc)
    ytr=TR["_y"].astype(int).values; yva=VA["_y"].astype(int).values
    Xtr=TR[cols].astype("float32"); Xva=VA[cols].astype("float32")
    print(f"[xpair USDCAD] train={len(TR):,} val={len(VA):,} feats={len(cols)} build={time.time()-t0:.0f}s",flush=True)
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=127,min_child_samples=400,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,n_estimators=3000,n_jobs=20,verbosity=-1)
    L.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pva=L.predict_proba(Xva)[:,1]; aucv=float(roc_auc_score(yva,pva))
    imp=sorted(zip(cols,L.feature_importances_),key=lambda z:-z[1])[:20]
    res["val_auc"]=aucv; res["best_iter"]=int(L.best_iteration_ or 0); res["top20"]=[c for c,_ in imp]
    print(f"[xpair USDCAD] best_iter={L.best_iteration_} VAL AUC={aucv:.4f} (base .5234)",flush=True)
    print("  top20:",", ".join(c for c,_ in imp),flush=True)
    tsv=VA["_ts"].values.astype("int64"); fwv=VA["_fwd"].values; half=len(pva)//2; confv=np.abs(pva-0.5); best=None
    for cov in (0.10,0.05,0.03,0.02):
        thr=float(np.quantile(confv,1-cov)); accs=[]
        for s,e in ((0,half),(half,len(pva))):
            r=side_eval(pva[s:e],yva[s:e],fwv[s:e],tsv[s:e],thr); accs.append(r["COMBINED"]["wr"] if r else float("nan"))
        worst=np.nanmin(accs)
        if best is None or worst>best[0]: best=(worst,cov,thr)
    worst,COV,THR=best; res["gate"]={"cov":COV,"thr":THR,"val_worst_half":round(float(worst),4)}
    print(f"[xpair USDCAD] gate cov{COV:.0%} thr={THR:.4f} VAL worst-half={worst:.4f}",flush=True)
    res["years"]={}
    for w in ("test24","test25","oos"):
        D=build_xp(SPL[w],1,mode); D=augment(D,SPL[w],mode)
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
                    "IMPROVES_base":improves,"note":"if not IMPROVES, base stands; pooling subsumed at 15m (USDJPY/AUDUSD case). If IMPROVES -> escalate to NY refit-CPCV."}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[xpair USDCAD] VERDICT IMPROVES_base={improves} (VAL {aucv:.4f} vs .5234; 2026 cov2 COMB {a26} vs .537) -> {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "xpbase"
    stride=int(sys.argv[2]) if len(sys.argv)>2 else 4
    main(mode,stride)
