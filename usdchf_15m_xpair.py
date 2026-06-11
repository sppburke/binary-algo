"""USDCHF 15-MIN direction via CROSS-PAIR / USD-common-factor + USDCHF-specific channels (A6 keystone screen).

SCOPE: USDCHF · 15m (A6 — THE DIFFERENTIATING TEST). Fork of usdcad_15m_xpair.py with TARGET=USDCHF,
pair-specific block repurposed from commodity/oil → EUR-BLOC. THE central question: USDCHF≈−EURUSD (CHF is
SNB-managed, tracks EUR via the 2011-15 EURCHF floor + intervention), so the cross-pair EUR-bloc POOLING lever
— the EURUSD/GBPUSD 15m CERTIFYING keystone — may WIN here, UNLIKE the own-pair-specific safe-haven analogs
(USDJPY/AUDUSD/USDCAD all DILUTED under pooling). Genuine fork, MED prior (not low).

SIGN ALIGNMENT (load-bearing — USDCHF is USD-BASE, USD in the NUMERATOR). equiv_sign(p)=+1 for USD-quote
{EURUSD,GBPUSD,AUDUSD,NZDUSD}, -1 for USD-base {USDJPY,USDCHF,USDCAD}. To keep the TARGET in its RAW frame
("USDCHF-up = USD-strength"), align every pair to the target-up frame with
   aln(p) = equiv_sign(TARGET) * equiv_sign(p)
so aln(TARGET)=+1 (target raw) and aln(p)*r_p = pair move in the USDCHF-up (=USD-strength) direction. Because
equiv_sign(USDCHF)=-1, aln(EURUSD)=-1: a EURUSD RISE (EUR up = USD weak) maps to the USDCHF-DOWN direction —
exactly the inverse co-movement USDCHF≈−EURUSD. basket = mean over the other 6 of aln(p)*r_p = USD-STRENGTH
factor (USDCHF-up direction); catchup = basket - usdchf_r (USDCHF owes the basket move → predicts +). The
EUR-BLOC block sharpens the pooling signal: eurbloc = mean of {EURUSD,GBPUSD} in USDCHF-up frame; eurcatch =
eurbloc − tgt_r (EUR-bloc lead-lag → USDCHF catch-up); chf_eurresid = tgt_r − eurbloc (CHF beyond EUR-bloc ~
SNB/idiosyncratic); risk = eurbloc − JPY-haven.

modes: xp (xpair feats only) | xpbase (xpair + 239 base — THE keystone) | dblortho (xpbase + double-
orthogonalized CHF-idiosyncratic residual: r_USDCHF purged of USD-fac AND EUR-bloc mean → SNB-proxy).

Label: next-15m USDCHF sign (raw), wall-clock contiguous (900s), ties excluded from train/AUC (charged as
losses in win-rate). all-session SCREEN: does EUR-bloc pooling LIFT base VAL AUC .5301 and the binding-year
tail? If yes → escalate to carrier-session refit-CPCV. Pre-registered falsifier in result JSON BEFORE OOS.
Usage: ~/binary-algo-venv/bin/python usdchf_15m_xpair.py [mode=xpbase|xp|dblortho] [stride_train=4]
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H

FEAT="/home/sean/git/binary-algo/features"
PAIRS=["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCHF","USDCAD"]
USD_BASE={"USDJPY","USDCHF","USDCAD"}
TARGET="USDCHF"
NONTGT=[p for p in PAIRS if p!=TARGET]
EUR_BLOC=["EURUSD","GBPUSD"]        # European currencies CHF co-moves with (the EUR-bloc pooling hypothesis)
SAFEHAVEN=["USDJPY"]               # other safe-haven ex-target (USDCHF is TARGET, excluded)
LB=[1,3,5,10,15,30]
HOR=int(os.environ.get("MX_HOR","15")); GAP_S=HOR*60
BE=0.541
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],"test24":["2024"],"test25":["2025"],"oos":["2026"]}
BASE_VAL_AUC=0.5301; BASE_2026_COV2_COMB=0.5103   # A1 incumbent to beat (usdchf_15m_base_result.json)

def equiv_sign(p): return -1.0 if p in USD_BASE else +1.0
def aln(p): return equiv_sign(TARGET)*equiv_sign(p)   # target-up frame: aln(TARGET)=+1

def build_xp(years, stride=1, mode="xpbase"):
    """Cross-pair (USDCHF-target, sign-aligned) + USDCHF-specific features + label _y,_ts,_fwd. Ties excluded."""
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
        df=df[~df.index.duplicated(keep="last")]   # dedup (prevents join-explosion / OOM)
        if len(df)<100: continue
        idx=df.index; secs=idx.values.astype("datetime64[s]").astype("int64"); n=len(df)
        lr={p:np.log(df[p].values) for p in PAIRS}
        rets={p:{k:np.concatenate([[np.nan]*k, lr[p][k:]-lr[p][:-k]]) for k in LB} for p in PAIRS}
        feats={}
        for k in LB:
            tgt_r=rets[TARGET][k]                                  # raw USDCHF return (USD-strength up)
            basket=np.nanmean(np.vstack([aln(p)*rets[p][k] for p in NONTGT]),axis=0)   # USD-strength factor (USDCHF-up frame)
            disp=np.nanstd(np.vstack([aln(p)*rets[p][k] for p in NONTGT]),axis=0)
            agree=np.nanmean(np.vstack([(np.sign(aln(p)*rets[p][k])==np.sign(basket)).astype(float) for p in NONTGT]),axis=0)
            feats[f"tgt_r{k}"]=tgt_r
            feats[f"usdbask{k}"]=basket
            feats[f"catchup{k}"]=basket-tgt_r          # USDCHF owes the basket move (predicts +)
            feats[f"tgtresid{k}"]=tgt_r-basket          # USDCHF idiosyncratic (beyond USD factor)
            feats[f"disp{k}"]=disp
            feats[f"agree{k}"]=agree
            for p in NONTGT:
                feats[f"ll_{p}{k}"]=aln(p)*rets[p][k]-tgt_r   # pair lead-lag residual vs USDCHF (target frame)
            # --- USDCHF-specific block (EUR-BLOC pooling — the differentiating hypothesis) ---
            eurbloc=np.nanmean(np.vstack([aln(p)*rets[p][k] for p in EUR_BLOC]),axis=0)  # EUR-bloc (EURUSD,GBPUSD) in USDCHF-up frame (sign-FLIPPED: EUR up=USD weak=USDCHF down)
            safeh=np.nanmean(np.vstack([aln(p)*rets[p][k] for p in SAFEHAVEN]),axis=0)  # USDJPY safe-haven in USDCHF-up frame
            feats[f"eurbloc{k}"]=eurbloc                 # EUR-bloc strength (USDCHF-up frame) — the pooling signal if CHF≈EUR
            feats[f"eurcatch{k}"]=eurbloc-tgt_r          # USDCHF owes the EUR-bloc move (EUR-bloc lead-lag → predicts catch-up)
            feats[f"risk{k}"]=eurbloc-safeh              # risk factor: EUR-bloc vs JPY-haven in USDCHF-up frame (USDCHF loads -, GBM handles)
            feats[f"chf_eurresid{k}"]=tgt_r-eurbloc      # USDCHF beyond the EUR-bloc (CHF-idiosyncratic ~ SNB/intervention component)
            if mode=="dblortho":
                # double-orthogonalized: purge USD factor AND EUR-bloc-common -> CHF-idiosyncratic (SNB proxy)
                feats[f"dblortho{k}"]=tgt_r-basket-(eurbloc-basket)   # = tgt_r - eurbloc, purged of fac via basket; isolate vs both
        # session + compression gate cols
        hours=idx.hour.values+idx.minute.values/60.0
        feats["sess_ny"]=((hours>=13.0)&(hours<22.0)).astype(float)
        feats["sess_ln"]=((hours>=7.0)&(hours<16.0)).astype(float)
        feats["hour"]=hours
        r1=rets[TARGET][1]
        feats["comp60"]=pd.Series(r1).rolling(60,min_periods=20).std().values
        # label (raw USDCHF direction)
        fwd=np.full(n,np.nan)
        if n>HOR:
            contig=(secs[HOR:]-secs[:-HOR])==HOR*60
            fr=lr[TARGET][HOR:]-lr[TARGET][:-HOR]
            fwd[:n-HOR]=np.where(contig,fr,np.nan)
        F=pd.DataFrame(feats,index=idx).astype("float32"); F["_y"]=(fwd>0).astype("float32"); F["_ts"]=secs.astype("int64"); F["_fwd"]=fwd.astype("float64")
        valid=np.isfinite(fwd)&(fwd!=0)
        F=F.loc[valid]
        if stride>1: F=F.iloc[::stride]
        out.append(F)
    return pd.concat(out)

def xp_cols(df): return [c for c in df.columns if c not in ("_y","_ts","_fwd","hour")]

def augment(F, years, mode):
    """Join base 239 feats — MEMORY-FRUGAL: reindex each year's base to ONLY F's (strided) rows
    before concat, so we never hold the full-res base matrix (the OOM cause)."""
    if mode in ("xpbase","dblortho"):
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
    RESULT=f"usdchf_15m_xpair_{mode}_result.json"
    res={"key":"USDCHF.15m","model":f"cross-pair USD-residual (sign-aligned, USD-numerator, sign-FLIPPED EUR-bloc) + EUR-bloc/SNB-resid, mode={mode}",
         "incumbent":"A1 base VAL AUC .5301, 2026 cov2% COMB .5103","breakeven":BE,
         "hypothesis":"USDCHF≈-EURUSD (SNB-managed EUR-bloc) so cross-pair EUR-bloc POOLING may WIN here (like EURUSD/GBPUSD) UNLIKE the own-pair-specific havens (USDJPY/AUDUSD/USDCAD). This is the DIFFERENTIATING test.",
         "falsifier":{"registered":"pre-OOS",
            "IMPROVES_if":"VAL moved-AUC > 0.5301 AND 2026 cov2% COMBINED wr > 0.5103 (else base stands; pooling subsumed = own-pair-specific haven case). If screen lifts -> escalate to carrier-session refit-CPCV."}}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[xpair USDCHF] mode={mode} stride={stride} HOR={HOR}",flush=True)
    TR=build_xp(SPL["train"],stride,mode); VA=build_xp(SPL["val"],1,mode)
    xpc=xp_cols(TR); TR=augment(TR,SPL["train"],mode); VA=augment(VA,SPL["val"],mode)
    cols=feat_cols(mode,TR,xpc)
    ytr=TR["_y"].astype(int).values; yva=VA["_y"].astype(int).values
    Xtr=TR[cols].astype("float32"); Xva=VA[cols].astype("float32")
    print(f"[xpair USDCHF] train={len(TR):,} val={len(VA):,} feats={len(cols)} build={time.time()-t0:.0f}s",flush=True)
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=127,min_child_samples=400,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,n_estimators=3000,n_jobs=12,verbosity=-1)
    L.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pva=L.predict_proba(Xva)[:,1]; aucv=float(roc_auc_score(yva,pva))
    imp=sorted(zip(cols,L.feature_importances_),key=lambda z:-z[1])[:20]
    res["val_auc"]=aucv; res["best_iter"]=int(L.best_iteration_ or 0); res["top20"]=[c for c,_ in imp]
    print(f"[xpair USDCHF] best_iter={L.best_iteration_} VAL AUC={aucv:.4f} (base .5234)",flush=True)
    print("  top20:",", ".join(c for c,_ in imp),flush=True)
    tsv=VA["_ts"].values.astype("int64"); fwv=VA["_fwd"].values; half=len(pva)//2; confv=np.abs(pva-0.5); best=None
    for cov in (0.10,0.05,0.03,0.02):
        thr=float(np.quantile(confv,1-cov)); accs=[]
        for s,e in ((0,half),(half,len(pva))):
            r=side_eval(pva[s:e],yva[s:e],fwv[s:e],tsv[s:e],thr); accs.append(r["COMBINED"]["wr"] if r else float("nan"))
        worst=np.nanmin(accs)
        if best is None or worst>best[0]: best=(worst,cov,thr)
    worst,COV,THR=best; res["gate"]={"cov":COV,"thr":THR,"val_worst_half":round(float(worst),4)}
    print(f"[xpair USDCHF] gate cov{COV:.0%} thr={THR:.4f} VAL worst-half={worst:.4f}",flush=True)
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
    print(f"\n[xpair USDCHF] VERDICT IMPROVES_base={improves} (VAL {aucv:.4f} vs .5234; 2026 cov2 COMB {a26} vs .537) -> {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "xpbase"
    stride=int(sys.argv[2]) if len(sys.argv)>2 else 4
    main(mode,stride)
