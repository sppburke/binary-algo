"""G1 VERIFICATION — full-refit CPCV: does the VOL-NORMALIZED gate beat the baseline CONFIDENCE gate per path?

Extends m5_cpcv_refit.py: refit the cross-pair primary on each of C(8,2)=28 purged paths; on the held-out test
fold evaluate UP-selective accuracy TWO ways at matched covers — (BASE) top-cov by confidence (the incumbent
rule) and (G1) top-cov by conf/volrank (vol-normalized conformity, ex-ante 5m_rv_24). The cert standard: G1 is a
real improvement ONLY if its refit p10 / frac-clear EXCEEDS baseline's at the SAME cover (frozen-forward
over-states; refit is the truth). Covers {0.05,0.10,0.15}. Settlement book-native, moved-only, ties at build.

  M5_STRIDE=6 ~/binary-algo-venv/bin/python m5_g1_cpcv.py
"""
import os, json, time, itertools, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import m5_xpair as MX, m5_xpair_production as XP
N_GROUPS,K_TEST=8,2; SUBSAMPLE=100_000; STRIDE=int(os.environ.get("M5_STRIDE","6")); BE=0.541
COVS=[0.05,0.10,0.15]; ALL_YEARS=[str(y) for y in range(2012,2027)]

def build():
    import gc; Xs,ys,tss,nys,vols=[],[],[],[],[]
    cols=json.load(open(XP.art("strategy.json")))["primary_feats"]
    for yr in ALL_YEARS:
        D=MX.build_xp([yr],stride=STRIDE)
        if len(D)==0: continue
        D=MX.augment(D,[yr],XP.MODE)
        Xs.append(D[cols].astype("float32").to_numpy()); ys.append(D["_y"].astype(np.int8).values)
        tss.append(D["_ts"].values.astype("int64")); nys.append((D["sess_ny"].values>0.5))
        vols.append(D["5m_rv_24"].values.astype("float32")); del D; gc.collect()
    return cols,(np.concatenate(Xs),np.concatenate(ys),np.concatenate(tss),np.concatenate(nys),np.concatenate(vols))

def mk(): return lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.03,num_leaves=127,
    min_child_samples=300,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,n_estimators=700,n_jobs=20,verbosity=-1)
def vrank(v):
    v=np.where(np.isfinite(v),v,np.nanmedian(v[np.isfinite(v)]) if np.isfinite(v).any() else 0.0)
    return np.argsort(np.argsort(v))/max(len(v)-1,1)

def main():
    t0=time.time(); cols,(X,y,ts,ny,vol)=build()
    o=np.argsort(ts,kind="stable"); X,y,ts,ny,vol=X[o],y[o],ts[o],ny[o],vol[o]; n=len(y)
    print(f"[g1-cpcv] n={n:,} stride={STRIDE} up-rate={y.mean():.4f} build {time.time()-t0:.0f}s",flush=True)
    edges=np.linspace(0,n,N_GROUPS+1).astype(int); g=np.zeros(n,np.int8)
    for k in range(N_GROUPS): g[edges[k]:edges[k+1]]=k
    rng=np.random.default_rng(7); HOR=300; idx=np.arange(n)
    base={c:[] for c in COVS}; g1={c:[] for c in COVS}; bn={c:[] for c in COVS}; gn={c:[] for c in COVS}
    for ci,combo in enumerate(itertools.combinations(range(N_GROUPS),K_TEST)):
        te=idx[np.isin(g,combo)]; tr=idx[~np.isin(g,combo)]; keep=np.ones(len(tr),bool); tt=ts[tr]
        for grp in combo:
            gi=idx[g==grp]; lo,hi=ts[gi[0]],ts[gi[-1]]; keep&=~((tt>=lo-HOR)&(tt<=hi+HOR))
        tr=tr[keep]
        if len(tr)>SUBSAMPLE: tr=np.sort(rng.choice(tr,SUBSAMPLE,replace=False))
        m=mk(); m.fit(X[tr],y[tr]); pr=m.predict_proba(X[te])[:,1]; yte=y[te]
        gate=ny[te]&(pr>0.5); conf=np.abs(pr-0.5)
        if gate.sum()>=40:
            vr=vrank(vol[te][gate]); confg=conf[gate]; y_g=yte[gate]
            sc_g1=confg/(vr+0.1)
            for c in COVS:
                # baseline: top-c by confidence
                s=confg>=np.quantile(confg,1-c)
                base[c].append(float((y_g[s]==1).mean()) if s.sum()>=20 else np.nan); bn[c].append(int(s.sum()))
                # G1: top-c by vol-normalized score
                s2=sc_g1>=np.quantile(sc_g1,1-c)
                g1[c].append(float((y_g[s2]==1).mean()) if s2.sum()>=20 else np.nan); gn[c].append(int(s2.sum()))
        else:
            for c in COVS: base[c].append(np.nan); g1[c].append(np.nan); bn[c].append(0); gn[c].append(0)
        del m,pr
        if ci%7==0: print(f"  path {ci+1}/28 ({time.time()-t0:.0f}s)",flush=True)
    def summ(d,nd):
        a=np.array(d,float); v=a[np.isfinite(a)]
        return dict(p10=round(float(np.percentile(v,10)),4) if len(v) else None,
                    mean=round(float(v.mean()),4) if len(v) else None,
                    frac_clear=round(float((v>=BE).mean()),3) if len(v) else None,
                    n_valid=len(v), med_n=int(np.median([x for x in nd if x>=20])) if any(x>=20 for x in nd) else 0)
    out={"test":"G1 vol-normalized gate vs baseline confidence gate — full-refit CPCV","breakeven":BE,"stride":STRIDE,"covs":COVS,
         "baseline":{str(c):summ(base[c],bn[c]) for c in COVS},"G1_volnorm":{str(c):summ(g1[c],gn[c]) for c in COVS}}
    # verdict: G1 beats baseline if its p10 exceeds baseline p10 at the same cov AND clears breakeven
    beats={}
    for c in COVS:
        b=out["baseline"][str(c)]["p10"]; gg=out["G1_volnorm"][str(c)]["p10"]
        beats[str(c)]=bool(gg is not None and b is not None and gg>b+1e-4 and gg>=BE)
    g1_wins=any(beats.values())
    out["VERDICT"]=dict(g1_beats_baseline_any_cov=g1_wins, per_cov=beats,
        statement=("G1 CERTIFIED improvement: vol-normalized gate refit p10 exceeds baseline at >=1 cover and clears breakeven."
            if g1_wins else
            "G1 NOT an improvement under refit-CPCV: vol-normalized gate refit p10 does NOT exceed the baseline confidence gate at any cover. The frozen-forward 'flattening' was a selection artifact (corr(VAL,OOS)=−0.54)."))
    json.dump(out,open("m5_g1_cpcv_result.json","w"),indent=2,default=str)
    print("\n[G1-CPCV] cov | baseline p10/frac | G1 p10/frac")
    for c in COVS:
        b=out["baseline"][str(c)]; gg=out["G1_volnorm"][str(c)]
        print(f"  {c:.2f} | base p10={b['p10']} frac={b['frac_clear']} medn={b['med_n']} | G1 p10={gg['p10']} frac={gg['frac_clear']} medn={gg['med_n']} | G1>base={beats[str(c)]}")
    print(f"VERDICT: {out['VERDICT']['statement']}\n-> m5_g1_cpcv_result.json ({time.time()-t0:.0f}s)")

if __name__=="__main__": main()
