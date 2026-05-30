"""15-MINUTE v5 — CONDITIONAL-POCKET selective analysis on the up/down BINARY (sign of 15m return).

Target stays the binary endpoint sign. We do NOT change the product. We ask one question:
is there a *regime / time / confluence GATE* under which the selective book generalizes to >=75%
OOS at usable coverage? Two things are measured per gate, frozen-on-VAL, reported on TEST + 2026 OOS:

  (1) MODEL-FREE directional drift inside the gate:  P(up | gate) on TEST/OOS  (exogenous-flow test,
      e.g. London-fix window, month-end, session) — does the gate itself tilt direction off 0.50?
  (2) GATED selective accuracy: train one LGBM on the 239-feature base stack, then within the gated
      rows pick the VAL confidence threshold for each target coverage and read TEST/OOS accuracy.

Gates tested: volatility compression (bottom bb_width tertile) / expansion (top), sessions, hour
buckets incl. 15-16 UTC London-fix and 13-14 UTC US-data, day-of-week, month-end, and the EDA
confluence-extreme (all-TF bb_pctb>1 fade / <0 fade). Baseline = ungated.
"""
import time, numpy as np, pandas as pd, os
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H

HOR=15; STRIDE=3
base_cols=list(H.feature_cols("EURUSD"))
# gate-source columns we must keep aligned (subset of base_cols + index-derived)
GATECOLS=[c for c in base_cols if c in (
    "15m_bb_width","5m_bb_width","1h_bb_width","15m_atr_pct","15m_rv_20",
    "sess_london","sess_ny","sess_overlap","hour_sin","hour_cos","dow","vol_z",
    "15m_bb_pctb","5m_bb_pctb","1h_bb_pctb","30m_bb_pctb","1m_bb_pctb","mtf_trend_align")]

def load_split(split, stride=1):
    parts=[]
    for y in H.SPLITS[split]:
        p=f"{H.FEAT_DIR}/EURUSD_{y}.parquet"
        if not os.path.exists(p): continue
        df=pd.read_parquet(p, columns=base_cols+H.META_COLS); df=df[~df.index.duplicated(keep="last")]
        idx=df.index; c=df["close"].values; n=len(c)
        secs=idx.values.astype("datetime64[s]").astype("int64")
        contig=np.zeros(n,dtype=bool)
        if n>HOR: contig[:n-HOR]=(secs[HOR:]-secs[:-HOR])==HOR*60
        fwd=np.full(n,np.nan); fwd[:n-HOR]=c[HOR:]; ret=fwd/c-1.0
        y_=(ret>0).astype(float); valid=contig&np.isfinite(ret)&(ret!=0)
        d=df.loc[valid, base_cols].copy(); d["_y"]=y_[valid]
        if stride>1: d=d.iloc[::stride]
        parts.append(d)
    df=pd.concat(parts)
    X=df[base_cols].astype("float32"); y=df["_y"].astype(int).values
    # gate frame (index preserved)
    g=pd.DataFrame(index=df.index)
    for col in GATECOLS:
        if col in df.columns: g[col]=df[col].values
    ix=df.index
    g["utc_hour"]=ix.hour
    g["utc_min"]=ix.minute
    g["dow_i"]=ix.dayofweek
    # month-end: last 2 business days of month
    g["dom"]=ix.day
    g["is_monthend"]=(ix + pd.Timedelta(days=3)).month != ix.month
    return X, y, g

def build_gates(g):
    """Return dict name->boolean mask (np array) over the rows of this split."""
    G={}
    G["ALL"]=np.ones(len(g),bool)
    bw=g.get("15m_bb_width")
    if bw is not None:
        bwv=bw.values.astype(float)
        q33,q67=np.nanpercentile(bwv,[33,67])
        G["vol_compress(bbw<q33)"]=bwv<=q33
        G["vol_expand(bbw>q67)"]=bwv>=q67
    for s in ("sess_london","sess_ny","sess_overlap"):
        if s in g: G[s]=g[s].values.astype(float)>0.5
    h=g["utc_hour"].values
    G["londonfix_15-16utc"]=(h>=15)&(h<16)
    G["usdata_13-14utc"]=(h>=13)&(h<14)
    G["asia_0-6utc"]=(h>=0)&(h<6)
    G["monthend"]=g["is_monthend"].values
    G["monday"]=g["dow_i"].values==0
    G["friday"]=g["dow_i"].values==4
    # EDA confluence-extreme fade: all available bb_pctb>1 (overbought) or <0 (oversold)
    pcols=[c for c in ("1m_bb_pctb","5m_bb_pctb","15m_bb_pctb","30m_bb_pctb","1h_bb_pctb") if c in g]
    if len(pcols)>=3:
        P=g[pcols].values.astype(float)
        allov=np.all(P>1.0,axis=1); allos=np.all(P<0.0,axis=1)
        G["confl_allOB(fade=down)"]=allov
        G["confl_allOS(fade=up)"]=allos
    return G

def sel_acc(y,p,thr):
    conf=np.abs(p-0.5); sel=conf>=thr
    if sel.sum()==0: return np.nan,0
    return ((p[sel]>0.5).astype(int)==y[sel]).mean(), int(sel.sum())

def main():
    t0=time.time()
    Xtr,ytr,_=load_split("train",STRIDE)
    Xva,yva,gva=load_split("val")
    Xte,yte,gte=load_split("test")
    Xoo,yoo,goo=load_split("oos")
    print(f"v5-gates shapes tr={Xtr.shape} te={Xte.shape} oo={Xoo.shape} load={time.time()-t0:.0f}s",flush=True)
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=255,
        min_child_samples=300,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=10,
        n_estimators=3000,n_jobs=20,verbosity=-1)
    L.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pv=L.predict_proba(Xva)[:,1]; pt=L.predict_proba(Xte)[:,1]; po=L.predict_proba(Xoo)[:,1]
    print(f"LGBM AUC val={roc_auc_score(yva,pv):.4f} test={roc_auc_score(yte,pt):.4f} oos={roc_auc_score(yoo,po):.4f}",flush=True)

    Gv=build_gates(gva); Gt=build_gates(gte); Go=build_gates(goo)
    COVS=[0.5,0.2,0.1,0.05]
    print("\n================ CONDITIONAL POCKETS (binary endpoint sign) ================")
    print("gate | n(te/oo) | P(up) base te/oo | gated selective acc @cov (VAL-frozen thr): TEST / OOS(n)")
    for name in Gv:
        if name not in Gt or name not in Go: continue
        mv,mt,mo=Gv[name],Gt[name],Go[name]
        nv,nt,no=int(mv.sum()),int(mt.sum()),int(mo.sum())
        if min(nt,no)<150:
            print(f"  {name:>26} | te{nt} oo{no} | (too few)"); continue
        # model-free directional drift inside gate
        bup_t=yte[mt].mean(); bup_o=yoo[mo].mean()
        line=f"  {name:>26} | te{nt} oo{no} | up te={bup_t:.3f} oo={bup_o:.3f} |"
        # gated selective: threshold from gated VAL subset
        yv_g,pv_g=yva[mv],pv[mv]; yt_g,pt_g=yte[mt],pt[mt]; yo_g,po_g=yoo[mo],po[mo]
        confv=np.abs(pv_g-0.5)
        segs=[]
        for cov in COVS:
            if len(confv)<50: break
            thr=np.quantile(confv,1-cov)
            at,ntb=sel_acc(yt_g,pt_g,thr); ao,nob=sel_acc(yo_g,po_g,thr)
            flag="**" if (not np.isnan(at) and not np.isnan(ao) and at>=0.75 and ao>=0.75 and ntb>=50 and nob>=50) else ""
            segs.append(f"c{int(cov*100)}:{at:.3f}/{ao:.3f}(n{nob}){flag}")
        print(line+" "+" ".join(segs),flush=True)
    print("\n** = both TEST and OOS >=0.75 at >=50 bets in-gate. Baseline gate is 'ALL'.")
    print("V5-GATES DONE")

if __name__=="__main__":
    main()
