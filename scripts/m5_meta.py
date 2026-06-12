"""Learned META-LABELER ('avoid losers') on ORTHOGONAL axes — the user's explicit ask, done properly.
Primary = the xpof cross-pair+base+OF lgb (probs cached in models/m5xp_xpof_diag.npz for val/test24/test25/oos).
Meta-model M = regularized lgb trained on VAL to predict P(primary correct) from ORTHOGONAL meta-features ONLY
(cross-pair agreement/dispersion at all lookbacks, order-flow, base confidence, compression) — NOT the raw 239.
Abstention threshold selected by WORST-VAL-half stability (NOT VAL-acc-max — that is the corr(VAL,OOS)=-0.54 trap).
Trade where M>=thr & NY; direction=sign(primary p-0.5); non-overlap 300s; report EACH held-out window + floor. No leakage:
primary trained on 2012-21, meta trained on 2022-23, reported on 2024/2025/2026.
"""
import sys, numpy as np
import lightgbm as lgb
MODE="xpof"
D=np.load(f"models/m5xp_{MODE}_diag.npz")
WINS=["val","test24","test25","oos"]
def win(w):
    d={}
    for key in D.files:
        if key=="frozen": continue
        ww,_,k=key.partition("__")
        if ww==w: d[k]=D[key]
    return d
W={w:win(w) for w in WINS}
META=[k for k in W["val"] if k.startswith("agree") or k.startswith("disp") or k=="comp60" or k.startswith("OF_")]
print(f"[m5_meta] meta-features ({len(META)}): {META}")

def nonoverlap(ts,mask,gap=300):
    take=[]; block=-1
    for i in np.where(mask)[0]:
        if ts[i]<block: continue
        take.append(i); block=int(ts[i])+gap
    return np.array(take,dtype=int)
def boot(corr,nb=4000,seed=7):
    corr=np.asarray(corr,float)
    if len(corr)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(corr)
    a=np.array([corr[rng.integers(0,n,n)].mean() for _ in range(nb)]); return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))
def yr(ts): return (np.asarray(ts,dtype="datetime64[s]").astype("datetime64[Y]").astype(int)+1970)
def Xmeta(w):
    d=W[w]; conf=np.abs(d["pr"]-0.5)
    cols=[conf]+[d[k] for k in META]
    return np.vstack(cols).T.astype("float32")

# meta-train on VAL, NY-gated rows only (we only ever trade NY)
va=W["val"]; vny=va["ny"].astype(bool)
Xtr=Xmeta("val")[vny]
ycorr=((va["pr"]>0.5).astype(int)==va["y"]).astype(int)[vny]
M=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=15,min_child_samples=1000,
    subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=20,n_estimators=400,n_jobs=20,verbosity=-1)
M.fit(Xtr,ycorr)
print(f"[m5_meta] meta trained on VAL NY n={len(ycorr)} base_correct_rate={ycorr.mean():.3f}")

# meta-score per window
S={w:M.predict_proba(Xmeta(w))[:,1] for w in WINS}
vts=va["ts"].astype("int64"); vyr=yr(vts)
def winsel(w,thr):
    d=W[w]; ny=d["ny"].astype(bool); ts=d["ts"].astype("int64")
    m=ny&(S[w]>=thr); sel=nonoverlap(ts,m)
    if len(sel)==0: return (0,float("nan"),np.array([]))
    corr=((d["pr"][sel]>0.5).astype(int)==d["y"][sel]).astype(float); return (len(sel),float(corr.mean()),corr)
def valhalf(thr):
    out={}
    for yy in sorted(set(vyr.tolist())):
        idx=np.where(vny&(vyr==yy)&(S["val"]>=thr))[0]
        ts=vts; sel=nonoverlap(ts, np.isin(np.arange(len(ts)),idx))
        if len(sel)<40: out[yy]=(len(sel),float("nan")); continue
        corr=((va["pr"][sel]>0.5).astype(int)==va["y"][sel]).astype(float); out[yy]=(len(sel),float(corr.mean()))
    return out

# threshold grid from VAL meta-score quantiles
grid=[float(np.quantile(S["val"][vny],q)) for q in (0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.97,0.98,0.985,0.99,0.993,0.996)]
print(f"\n{'thr':>7} {'VALn':>6} {'VALacc':>7} {'halfmin':>8}  {'t24':>14} {'t25':>14} {'oos':>14}  {'FLOOR':>6} {'COMB':>20}")
rows=[]
for thr in grid:
    nv,av,_=winsel("val",thr)
    if nv<150: continue
    half=valhalf(thr); hv=[v[1] for v in half.values() if v[0]>=40]; hmin=min(hv) if hv else float("nan")
    res={w:winsel(w,thr) for w in ("test24","test25","oos")}
    corrs=np.concatenate([res[w][2] for w in res]); cl,ch=boot(corrs)
    floor=min(res[w][1] for w in res)
    rows.append((thr,nv,av,hmin,res,floor,(len(corrs),corrs.mean(),cl,ch)))
    print(f"{thr:7.3f} {nv:6d} {av:7.3f} {hmin:8.3f}  "
          f"{res['test24'][1]:.3f}(n{res['test24'][0]:>4}) {res['test25'][1]:.3f}(n{res['test25'][0]:>4}) {res['oos'][1]:.3f}(n{res['oos'][0]:>4})  "
          f"{floor:6.3f} {corrs.mean():.3f}[{cl:.3f},{ch:.3f}]n{len(corrs)}")
# honest pick = worst-half-stable (max hmin with hmin>=0.56), report its held-out floor
elig=[r for r in rows if not np.isnan(r[3]) and r[3]>=0.56]
if elig:
    b=max(elig,key=lambda r:r[3])
    print(f"\n[HONEST worst-half-stable pick] thr={b[0]:.3f} VALhalfmin={b[3]:.3f} -> "
          f"t24 {b[4]['test24'][1]:.3f} / t25 {b[4]['test25'][1]:.3f} / oos {b[4]['oos'][1]:.3f}  FLOOR={b[5]:.3f}  "
          f"COMBINED {b[6][1]:.3f} CI[{b[6][2]:.3f},{b[6][3]:.3f}] n{b[6][0]}")
else:
    print("\n[HONEST pick] no threshold with both VAL halves >=0.56")
if rows:
    orc=max(rows,key=lambda r:r[5])
    print(f"[ORACLE max-floor, NOT honest] thr={orc[0]:.3f} -> t24 {orc[4]['test24'][1]:.3f}/t25 {orc[4]['test25'][1]:.3f}/oos {orc[4]['oos'][1]:.3f} FLOOR={orc[5]:.3f}")
