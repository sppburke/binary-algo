"""USDJPY 1-MIN — signed REVERSION / PATH-STATE features (Tier-N UJ-N1 + UJ-N2).

SCOPE: USDJPY · 1m. Discovery round-1 levers:
 UJ-N1 (Osler spike-into-extreme signed reversion, DOWN-revive): sharp signed thrust into a range
   extreme reverts (agent probe: sharp-up@range-top next-bar up-rate .473 = DOWN bias).
 UJ-N2 (signed path-state dip-buy, UP-lift): signed cumret / run-length / dist-from-running-extreme.
Adds ~16 SIGNED features (flip under return-flip → NOT sign-invariant) to the 239 base; trains the
best config (stride6, 255 leaves). Tests whether explicit signed interactions add DIRECTION the base
GBM didn't already extract. Reports UP + DOWN coverage curves + the new feats' importance rank.
Falsifier: KILL if (UP cov1-2% CI-lo doesn't clear .541 in any year) AND (DOWN ditto) AND (new feats
don't dominate importance). Usage: ~/binary-algo-venv/bin/python usdjpy_1m_revspec.py [stride]
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from usdjpy_1m_base import nonoverlap_chrono, boot, side_eval, covcurve, mk_lgb

PAIR="USDJPY"; FEAT=H.FEAT_DIR; BASE=H.feature_cols(PAIR); BE=0.541
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],
     "test24":["2024"],"test25":["2025"],"oos":["2026"]}
STRIDE=int(sys.argv[1]) if len(sys.argv)>1 else 6
RESULT=f"usdjpy_1m_revspec_s{STRIDE}_result.json"
NEW=["sret3","sret6","sret12","sret30","rangepos48","dist_hi48","dist_lo48","dfrommax30","dfrommin30",
     "up_run","down_run","atr_pct","spk12_pos","spk6_lo","spk12_dmax"]

def add_signed(c):
    n=len(c); lc=np.log(c); s=pd.Series(c)
    f={}
    for k in (3,6,12,30):
        r=np.full(n,np.nan); r[k:]=lc[k:]-lc[:-k]; f[f"sret{k}"]=r
    hi48=s.rolling(48,min_periods=20).max().values; lo48=s.rolling(48,min_periods=20).min().values
    f["rangepos48"]=(c-lo48)/(hi48-lo48+1e-12)
    f["dist_hi48"]=(hi48-c)/c; f["dist_lo48"]=(c-lo48)/c
    mx30=s.rolling(30,min_periods=15).max().values; mn30=s.rolling(30,min_periods=15).min().values
    f["dfrommax30"]=c/mx30-1.0; f["dfrommin30"]=c/mn30-1.0          # signed (<=0, >=0)
    r1=np.concatenate([[0.0],np.diff(lc)]); sg=np.sign(r1)
    rl=pd.Series(sg).groupby((pd.Series(sg)!=pd.Series(sg).shift()).cumsum()).cumcount().values+1
    f["up_run"]=np.where(sg>0,rl,0).astype(float); f["down_run"]=np.where(sg<0,rl,0).astype(float)
    f["atr_pct"]=pd.Series(np.abs(r1)).rolling(30,min_periods=15).mean().values
    f["spk12_pos"]=f["sret12"]*f["rangepos48"]                      # spike x position (UJ-N1 interaction)
    f["spk6_lo"]=f["sret6"]*f["dist_lo48"]
    f["spk12_dmax"]=f["sret12"]*f["dfrommax30"]
    return f

def build(years, stride=1):
    Xs=[];ys=[];mv=[];tss=[]
    for y in years:
        p=f"{FEAT}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p,columns=BASE+["close"]); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float); ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        contig=np.zeros(n,bool); contig[:n-1]=(ts[1:]-ts[:-1])==60
        fr=np.full(n,np.nan); fr[:n-1]=c[1:]/c[:-1]-1.0
        nf=add_signed(c); X=d[BASE].copy()
        for k in NEW: X[k]=nf[k].astype("float32")
        X=X.astype("float32"); keepf=X[BASE].isna().mean(axis=1).values<0.5
        valid=contig&np.isfinite(fr)&keepf
        idx=np.where(valid)[0]
        if stride>1: idx=idx[::stride]
        Xs.append(X.iloc[idx]); ys.append((fr[idx]>0).astype(int)); mv.append(fr[idx]!=0); tss.append(ts[idx])
    return pd.concat(Xs), np.concatenate(ys), np.concatenate(mv), np.concatenate(tss)

def main():
    t0=time.time()
    res={"key":"USDJPY.1m","model":f"base239 + 16 signed reversion/path-state feats (UJ-N1+N2), stride{STRIDE} leaves255",
         "settlement":"bar-close, ties LOSE, BE .541, gap60","splits":SPL,"new_feats":NEW,
         "falsifier":{"registered_utc":"pre-OOS",
            "KILL_if":"no held-out year UP OR DOWN win-rate CI95-lo clears 0.541 at cov<=2% (n>=75) AND new feats don't lift tail vs base",
            "incumbent":"s6/l255 base: UP cov2% .547/.534/.538; DOWN dead"}}
    json.dump(res,open(RESULT,"w"),indent=2)
    cols=BASE+NEW
    Xtr,ytr,mtr,_=build(SPL["train"],STRIDE); Xva,yva,mva,tsv=build(SPL["val"])
    print(f"[rev] train={int(mtr.sum()):,} val={int(mva.sum()):,} feats={len(cols)} build={time.time()-t0:.0f}s",flush=True)
    L=mk_lgb(num_leaves=255); L.fit(Xtr[mtr],ytr[mtr],eval_set=[(Xva[mva],yva[mva])],eval_metric="auc",
        callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pva=L.predict_proba(Xva)[:,1]; val_auc=float(roc_auc_score(yva[mva],pva[mva]))
    fi=dict(zip(cols,L.feature_importances_)); order=sorted(fi,key=lambda c:-fi[c])
    new_ranks={k:order.index(k)+1 for k in NEW}
    print(f"[rev] best_iter={L.best_iteration_} VAL AUC={val_auc:.4f} {time.time()-t0:.0f}s",flush=True)
    print(f"[rev] top15: {order[:15]}",flush=True)
    print(f"[rev] new-feat ranks (of {len(cols)}): "+", ".join(f"{k}#{new_ranks[k]}" for k in sorted(NEW,key=lambda k:new_ranks[k])[:8]),flush=True)
    res["val_auc"]=val_auc; res["new_feat_ranks"]=new_ranks; res["top15"]=order[:15]
    res["years"]={}
    for w in ("test24","test25","oos"):
        Xw,yw,mw,tsw=build(SPL[w]); pr=L.predict_proba(Xw[cols])[:,1]
        cc=covcurve(pr,yw,mw,tsw); auc=float(roc_auc_score(yw[mw],pr[mw]))
        res["years"][w]={"moved_auc":auc,"moved_up_rate":float(yw[mw].mean()),"covcurve":cc}
        def g(cov,side): d=cc[cov].get(side,{}); return f"{d.get('wr')}(n{d.get('n')})"
        print(f"=== {w} === AUC={auc:.4f} | UP cov2%:{g('0.02','UP')} cov1%:{g('0.01','UP')} | DOWN cov2%:{g('0.02','DOWN')} cov1%:{g('0.01','DOWN')}",flush=True)
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[rev] saved {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
