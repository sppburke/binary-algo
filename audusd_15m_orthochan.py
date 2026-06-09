"""AUDUSD 15m NY — isolate the AUD-specific ORTHOGONAL own-pair channels (famonly confirm-or-kill).

SCOPE: AUDUSD · 15m · NY. A6 (audusd_15m_xpair.py) put cross-pair + AUDNZD-RV + risk-bloc features into ONE big
model where they ranked below the top-20 (own-pair base dominated) and did not lift VAL AUC. That is suggestive
but NOT a clean isolation. This runs a FAMONLY test of the two most mechanistically-distinct AUD-specific
channels (the base 239 contain NO NZD/cross-pair info, so these are genuinely orthogonal candidates):
  - AUDNZD residual-difference s = aud_r - nzd_r (USD factor cancels; RBA-RBNZ mean reversion) + audnzd_dev
  - commodity-vs-safe-haven RISK factor = commod(AUD,NZD,CAD) - haven(CHF,JPY) + audrisk_resid
plus a SIGNED realized-semivariance asymmetry block RS+/RS- (backlog #11; EURUSD D7 passed its sign-null but was
sub-BE). NY-restricted, single-fit, VAL worst-half gate. Reports each block ALONE (famonly), base-only, and
base+block, per-year cov2% per side. A channel ADDS only if base+block VAL-AUC > base + 0.005 AND 2026 cov2%
COMB > base. Falsifier pre-registered in result JSON.

Usage: ~/binary-algo-venv/bin/python audusd_15m_orthochan.py
"""
import os, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
from sessions import session_mask
from audusd_15m_xpair import build_xp_aud, augment, SPL, BE
from audusd_15m_base import side_eval, boot

SESSION="ny"; RESULT="audusd_15m_orthochan_result.json"
FEAT="/home/sean/git/binary-algo/features"

def add_semivar(F, years):
    """SIGNED realized-semivariance asymmetry from AUDUSD 1m returns: rsasym_k = (RS+ - RS-)/(RS+ + RS-)."""
    parts=[]
    for y in years:
        p=f"{FEAT}/AUDUSD_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p,columns=["close"]); d=d[~d.index.duplicated(keep="last")]
        parts.append(d)
    if not parts: return F
    c=pd.concat(parts); lr=np.log(c["close"].values); r1=np.concatenate([[np.nan],np.diff(lr)])
    s=pd.Series(r1,index=c.index)
    feats={}
    for k in (15,30,60):
        rp=(s.clip(lower=0)**2).rolling(k,min_periods=k//2).sum()
        rm=(s.clip(upper=0)**2).rolling(k,min_periods=k//2).sum()
        feats[f"rsasym{k}"]=((rp-rm)/(rp+rm+1e-12)).values
    SV=pd.DataFrame(feats,index=c.index)
    return F.join(SV[[col for col in SV.columns if col not in F.columns]],how="left")

def cols_of(df, kind):
    xp=[c for c in df.columns if c not in ("_y","_ts","_fwd","hour")]
    base=[c for c in H.feature_cols("AUDUSD") if c in df.columns]
    audnzd=[c for c in xp if c.startswith("audnzd")]
    risk=[c for c in xp if c.startswith("risk") or c.startswith("audrisk")]
    semi=[c for c in df.columns if c.startswith("rsasym")]
    return {"base":base,"audnzd":audnzd,"risk":risk,"semi":semi}[kind]

def fit_eval(TR,VA,YR,cols,tag):
    ytr=TR["_y"].astype(int).values; yva=VA["_y"].astype(int).values
    Xtr=TR[cols].astype("float32").values
    if len(Xtr)>200_000:                       # subsample fit (memory + matches cpcv SUB_FIT)
        rng=np.random.default_rng(7); sel=rng.choice(len(Xtr),200_000,replace=False)
        Xtr=Xtr[sel]; ytr=ytr[sel]
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=127,min_child_samples=400,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,n_estimators=2000,n_jobs=20,verbosity=-1)
    L.fit(Xtr,ytr,eval_set=[(VA[cols].astype("float32").values,yva)],eval_metric="auc",
          callbacks=[lgb.early_stopping(120),lgb.log_evaluation(0)])
    pva=L.predict_proba(VA[cols].astype("float32"))[:,1]; aucv=float(roc_auc_score(yva,pva))
    confv=np.abs(pva-0.5); THR=float(np.quantile(confv,1-0.02))
    out={"val_auc":round(aucv,4),"n_feats":len(cols),"years":{}}
    for w in YR:
        D=YR[w]; pr=L.predict_proba(D[cols].astype("float32"))[:,1]
        y=D["_y"].astype(int).values; fwd=D["_fwd"].values; ts=D["_ts"].values.astype("int64")
        mv=fwd!=0; moved=mv
        g=side_eval(pr,y,moved,ts,THR)
        out["years"][w]={"auc":round(float(roc_auc_score(y[mv],pr[mv])),4),
            "cov2":{k:{"n":g[k]["n"],"wr":round(g[k]["wr"],4)} for k in ("COMBINED","UP","DOWN")} if g else None}
    c=out["years"];
    print(f"  [{tag}] feats={len(cols)} VAL_AUC={aucv:.4f} | 2026 cov2 COMB {c['oos']['cov2']['COMBINED'] if c['oos']['cov2'] else None}",flush=True)
    return out

def main():
    t0=time.time()
    print("[orthochan] building NY-restricted panel...",flush=True)
    def prep(years,stride):
        F=build_xp_aud(years,stride); F=augment(F,years,"xpbase"); F=add_semivar(F,years)
        m=session_mask(F["_ts"].values.astype("int64"),SESSION)
        return F.loc[m]
    TR=prep(SPL["train"],6); VA=prep(SPL["val"],2); YR={w:prep(SPL[w],1) for w in ("test24","test25","oos")}
    print(f"[orthochan] train_NY={len(TR):,} val_NY={len(VA):,} build={time.time()-t0:.0f}s",flush=True)
    base=cols_of(TR,"base"); audnzd=cols_of(TR,"audnzd"); risk=cols_of(TR,"risk"); semi=cols_of(TR,"semi")
    print(f"[orthochan] |base|={len(base)} |audnzd|={len(audnzd)} |risk|={len(risk)} |semi|={len(semi)}",flush=True)
    res={"key":"AUDUSD.15m.NY","breakeven":BE,
         "falsifier":{"registered":"pre-OOS","ADDS_if":"base+block VAL-AUC > base_VAL_AUC+0.005 AND 2026 cov2 COMB > base 2026 cov2"},
         "arms":{}}
    json.dump(res,open(RESULT,"w"),indent=2)
    res["arms"]["base"]=fit_eval(TR,VA,YR,base,"base")
    for nm,blk in (("audnzd",audnzd),("risk",risk),("semi",semi)):
        if blk: res["arms"][f"{nm}_only"]=fit_eval(TR,VA,YR,blk,f"{nm}_only")
        res["arms"][f"base+{nm}"]=fit_eval(TR,VA,YR,base+blk,f"base+{nm}")
    b=res["arms"]["base"]; b_auc=b["val_auc"]; b26=b["years"]["oos"]["cov2"]["COMBINED"]["wr"] if b["years"]["oos"]["cov2"] else float("nan")
    verdict={}
    for nm in ("audnzd","risk","semi"):
        a=res["arms"].get(f"base+{nm}");
        if not a: continue
        a26=a["years"]["oos"]["cov2"]["COMBINED"]["wr"] if a["years"]["oos"]["cov2"] else float("nan")
        verdict[nm]={"base+block_val_auc":a["val_auc"],"delta_auc":round(a["val_auc"]-b_auc,4),
            "block_2026_cov2":a26,"ADDS":bool(a["val_auc"]>b_auc+0.005 and np.isfinite(a26) and a26>b26)}
    res["verdict"]=verdict; res["base_val_auc"]=b_auc; res["base_2026_cov2"]=b26
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[orthochan] base VAL_AUC={b_auc} 2026cov2={b26}",flush=True)
    for nm,v in verdict.items(): print(f"  {nm}: base+block AUC {v['base+block_val_auc']} (Δ{v['delta_auc']}) 2026cov2 {v['block_2026_cov2']} -> ADDS={v['ADDS']}",flush=True)
    print(f"[orthochan] done {time.time()-t0:.0f}s -> {RESULT}",flush=True)

if __name__=="__main__":
    main()
