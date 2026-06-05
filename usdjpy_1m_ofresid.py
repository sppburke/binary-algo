"""USDJPY 1-MIN — cross-pair ORDER-FLOW residual (Tier-N UJ-N3).

SCOPE: USDJPY · 1m. The killed A6a cross-pair model used price RETURNS only. This tests the untested
delta: signed ORDER-FLOW residual = USDJPY own signed OF − USD-up-basket OF (6 other majors' OF,
sign-flipped to USD-up terms). Isolates JPY-IDIOSYNCRATIC flow (carry / risk-on-off / BoJ) vs broad-USD
flow. Signed algebraic residual of signed flow → NOT the sign-invariant |OF| channel. Added to base 239
+ USDJPY own 18 OF. Fast-KILL (low prior .06: own-OF weak at 1m, EURUSD cross-impact OFI was .5015).
Usage: ~/binary-algo-venv/bin/python usdjpy_1m_ofresid.py [stride]
"""
import sys, os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from usdjpy_1m_base import nonoverlap_chrono, boot, side_eval, covcurve, mk_lgb

PAIR="USDJPY"; FEAT=H.FEAT_DIR; OFDIR="/media/sean/CORSAIR/binary-algo/features_of"; BASE=H.feature_cols(PAIR); BE=0.541
PAIRS=["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCHF","USDCAD"]; USD_BASE={"USDJPY","USDCHF","USDCAD"}
OTHERS=[p for p in PAIRS if p!=PAIR]
OWN_OF=['OF_of_norm_1','OF_of_sum_1','OF_of_norm_3','OF_of_sum_3','OF_of_norm_5','OF_of_sum_5','OF_of_norm_10',
    'OF_of_sum_10','OF_of_norm_15','OF_of_sum_15','OF_of_norm_30','OF_of_sum_30','OF_of_uptick_5','OF_of_uptick_15',
    'OF_kyle_5','OF_kyle_15','OF_of_accel','OF_of_persist']
SUMK=['OF_of_sum_5','OF_of_sum_15','OF_of_sum_30']        # signed-flow windows for basket+residual
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],
     "test24":["2024"],"test25":["2025"],"oos":["2026"]}
STRIDE=int(sys.argv[1]) if len(sys.argv)>1 else 8
RESULT=f"usdjpy_1m_ofresid_s{STRIDE}_result.json"
NEW=[f"baskOF_{k}" for k in ("5","15","30")]+[f"residOF_{k}" for k in ("5","15","30")]
def usd_up_sign(p): return +1.0 if p in USD_BASE else -1.0

def build(years, stride=1):
    Xs=[];ys=[];mv=[];tss=[]
    for y in years:
        # intersect index across 7 closes (for label + alignment)
        cl={}; ok=True
        for p in PAIRS:
            fp=f"{FEAT}/{p}_{y}.parquet"
            if not os.path.exists(fp): ok=False;break
            d=pd.read_parquet(fp,columns=["close"]); d=d[~d.index.duplicated(keep="last")]; cl[p]=d["close"]
        if not ok: continue
        cdf=pd.DataFrame(cl).dropna(); idx=cdf.index
        secs=idx.values.astype("datetime64[s]").astype("int64"); n=len(idx)
        contig=np.zeros(n,bool); contig[:n-1]=(secs[1:]-secs[:-1])==60
        cj=cdf[PAIR].values.astype(float); fr=np.full(n,np.nan); fr[:n-1]=cj[1:]/cj[:-1]-1.0
        # OF for all 7 on this index
        ofb={}
        for p in PAIRS:
            op=f"{OFDIR}/{p}_{y}.parquet"
            if not os.path.exists(op): ok=False;break
            o=pd.read_parquet(op,columns=SUMK); o=o[~o.index.duplicated(keep="last")]
            ofb[p]=o.reindex(idx)
        if not ok: continue
        feats={}
        for k in SUMK:
            mat=np.vstack([usd_up_sign(p)*ofb[p][k].values for p in OTHERS])
            bask=np.nanmean(mat,axis=0); own=usd_up_sign(PAIR)*ofb[PAIR][k].values
            kk=k.split("_")[-1]
            feats[f"baskOF_{kk}"]=bask; feats[f"residOF_{kk}"]=own-bask
        # base 239 + own 18 OF on idx
        B=pd.read_parquet(f"{FEAT}/{PAIR}_{y}.parquet",columns=BASE); B=B[~B.index.duplicated(keep="last")].reindex(idx)
        O=pd.read_parquet(f"{OFDIR}/{PAIR}_{y}.parquet",columns=OWN_OF); O=O[~O.index.duplicated(keep="last")].reindex(idx)
        X=pd.concat([B, O, pd.DataFrame(feats,index=idx)],axis=1).astype("float32")
        keepf=B.isna().mean(axis=1).values<0.5
        valid=contig&np.isfinite(fr)&keepf
        ii=np.where(valid)[0]
        if stride>1: ii=ii[::stride]
        Xs.append(X.iloc[ii]); ys.append((fr[ii]>0).astype(int)); mv.append(fr[ii]!=0); tss.append(secs[ii])
    return pd.concat(Xs), np.concatenate(ys), np.concatenate(mv), np.concatenate(tss)

def main():
    t0=time.time()
    res={"key":"USDJPY.1m","model":f"base239 + own18OF + cross-pair OF-basket+RESIDUAL (UJ-N3), stride{STRIDE}",
         "settlement":"bar-close, ties LOSE, BE .541","splits":SPL,"new_feats":NEW,
         "falsifier":{"registered_utc":"pre-OOS","KILL_if":"VAL AUC<=.515 OR no year UP/DOWN cov2% CI-lo>.541 AND residOF feats rank low",
            "prior":0.06,"incumbent":"s6/l255 base UP cov2% .547/.534/.538; DOWN dead"}}
    json.dump(res,open(RESULT,"w"),indent=2)
    cols=BASE+OWN_OF+NEW
    Xtr,ytr,mtr,_=build(SPL["train"],STRIDE); Xva,yva,mva,tsv=build(SPL["val"])
    print(f"[ofres] train={int(mtr.sum()):,} val={int(mva.sum()):,} feats={len(cols)} build={time.time()-t0:.0f}s",flush=True)
    L=mk_lgb(num_leaves=255); L.fit(Xtr[cols][mtr],ytr[mtr],eval_set=[(Xva[cols][mva],yva[mva])],eval_metric="auc",
        callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pva=L.predict_proba(Xva[cols])[:,1]; val_auc=float(roc_auc_score(yva[mva],pva[mva]))
    fi=dict(zip(cols,L.feature_importances_)); order=sorted(fi,key=lambda c:-fi[c])
    ranks={k:order.index(k)+1 for k in NEW}
    print(f"[ofres] best_iter={L.best_iteration_} VAL AUC={val_auc:.4f} | newfeat ranks(of {len(cols)}): {ranks} {time.time()-t0:.0f}s",flush=True)
    res["val_auc"]=val_auc; res["new_feat_ranks"]=ranks
    res["years"]={}
    for w in ("test24","test25","oos"):
        Xw,yw,mw,tsw=build(SPL[w]); pr=L.predict_proba(Xw[cols])[:,1]
        cc=covcurve(pr,yw,mw,tsw); auc=float(roc_auc_score(yw[mw],pr[mw]))
        res["years"][w]={"moved_auc":auc,"covcurve":cc}
        def g(cov,side): d=cc[cov].get(side,{}); return f"{d.get('wr')}(n{d.get('n')})"
        print(f"=== {w} === AUC={auc:.4f} | UP cov2%:{g('0.02','UP')} | DOWN cov2%:{g('0.02','DOWN')}",flush=True)
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[ofres] saved {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
