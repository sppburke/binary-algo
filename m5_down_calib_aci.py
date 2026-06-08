"""DOWN §3 levers RUN LITERALLY (per user): CALIBRATION (temperature + isotonic/Venn-Abers) + ADAPTIVE-CONFORMAL
gate (ACI), on the m5xp DOWN side. Inference-only on the frozen book (no retrain).

Mirrors m5_conformal.py (ACI threshold feedback to a target win-rate) but DOWN candidates (NY & pred<0.5,
win=(y==0)); calibrates the DOWN P(down) on VAL first (temperature scaling + isotonic Venn-Abers proxy), then runs
ACI on the calibrated score. Tests whether calibration+ACI can hold DOWN win-rate over breakeven in the 2025 wall.

PRE-REGISTERED FALSIFIER: KILL unless some (calibration × ACI w*) yields DOWN binding-2025 win ≥0.541 with boot
CI95-lo ≥0.541 at n≥100. Expected null (calibration is monotonic → rank-invariant for cover gates; ACI can only
starve coverage where the 2025 edge is absent) — RUN to confirm, not infer.

  ~/binary-algo-venv/bin/python m5_down_calib_aci.py
"""
import os, sys; sys.argv=["x"]
import json, time, numpy as np
from sklearn.isotonic import IsotonicRegression
import m5_xpair as MX, m5_xpair_production as XP
ROOT="/home/sean/git/binary-algo"; BE=0.541
YEARS=(("val","VAL"),("test24","2024"),("test25","2025"),("oos","2026"))
def boot(c,nb=3000,seed=7):
    c=np.asarray(c,float)
    if len(c)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(c); a=np.array([c[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))

def collect_down():
    p,P,M=XP._load(); cols=p["primary_feats"]; mcols=p["meta_feats"]
    TS,PD,WIN,YR=[],[],[],[]
    for w,label in YEARS:
        D=MX.build_xp(XP.SPL[w]); D=MX.augment(D,XP.SPL[w],XP.MODE)
        pr=P.predict(D[cols].astype("float32")); y=D["_y"].astype(int).values; ts=D["_ts"].values.astype("int64")
        ny=D["sess_ny"].values>0.5; cand=ny&(pr<0.5)                       # DOWN candidates
        sel=MX.nonoverlap_chrono(ts,cand)
        TS.append(ts[sel]); PD.append((1.0-pr[sel]))                        # P(down)=1-pr (since pr=P(up))
        WIN.append((y[sel]==0).astype(float)); YR.append([label]*len(sel)); del D
    ts=np.concatenate(TS); o=np.argsort(ts)
    return ts[o],np.concatenate(PD)[o],np.concatenate(WIN)[o],np.concatenate(YR)[o]

def temp_scale(pdwn,win,mask_val):
    # fit temperature T on VAL by minimizing logloss of DOWN-win vs sigmoid(logit(pdwn)/T)
    eps=1e-6; z=np.log(np.clip(pdwn,eps,1-eps)/(1-np.clip(pdwn,eps,1-eps)))
    best=(1.0,1e9)
    for T in np.linspace(0.5,3.0,26):
        q=1/(1+np.exp(-z/T)); ll=-np.mean(win[mask_val]*np.log(np.clip(q[mask_val],eps,1))+(1-win[mask_val])*np.log(np.clip(1-q[mask_val],eps,1)))
        if ll<best[1]: best=(T,ll)
    T=best[0]; return 1/(1+np.exp(-z/T)), T

def aci(score,win,wstar,gamma=0.02):
    astar=1-wstar; th=float(np.quantile(score,0.5)); lo,hi=float(np.quantile(score,0.05)),float(np.quantile(score,0.995))
    tr=np.zeros(len(score),bool)
    for t in range(len(score)):
        if score[t]>=th:
            tr[t]=True; th=min(hi,max(lo,th+gamma*((1-win[t])-astar)))
    return tr

def per_year(mask,win,yr):
    out={}
    for _,l in YEARS:
        if l=="VAL": continue
        m=mask&(yr==l);
        if m.sum()>=20:
            cc=win[m]; loi,hii=boot(cc); out[l]=dict(win=round(float(cc.mean()),4),n=int(m.sum()),ci=[round(loi,4),round(hii,4)])
    b=[out[l]["win"] for l in out]; lo=[out[l]["ci"][0] for l in out]; ns=[out[l]["n"] for l in out]
    out["binding"]=dict(win=min(b) if b else None, ci_lo=min(lo) if lo else None, min_n=min(ns) if ns else 0)
    return out

def main():
    t0=time.time(); ts,pdwn,win,yr=collect_down()
    mval=yr=="VAL"
    print(f"[down-calib-aci] DOWN candidates={len(pdwn)} VAL={mval.sum()} overall down-rate={win.mean():.4f} {time.time()-t0:.0f}s",flush=True)
    # calibrations
    p_temp,T=temp_scale(pdwn,win,mval)
    iso=IsotonicRegression(out_of_bounds="clip"); iso.fit(pdwn[mval],win[mval]); p_iso=iso.predict(pdwn)
    cals={"raw":pdwn,"temperature":p_temp,"isotonic_vennabers":p_iso}
    out={"test":"DOWN calibration (temp/isotonic) + ACI gate","breakeven":BE,"T":round(float(T),3),"n_cand":int(len(pdwn))}
    survive=False; detail={}
    for cname,score in cals.items():
        # fixed calibrated-prob gate at a few thresholds + ACI at w*
        for wstar in (0.54,0.55,0.56):
            tr=aci(score,win,wstar); py=per_year(tr,win,yr)
            key=f"{cname}|ACI_w{wstar}"; detail[key]=py
            bd=py["binding"]
            if bd["win"] is not None and bd["ci_lo"] is not None and bd["ci_lo"]>BE and bd["min_n"]>=100:
                survive=True
    out["results"]=detail
    # report the best DOWN binding across all
    best=max(detail.items(), key=lambda kv:(kv[1]["binding"]["ci_lo"] or -9))
    out["best"]=dict(config=best[0], binding=best[1]["binding"])
    out["VERDICT"]=dict(SURVIVES=survive,
        statement=(f"DOWN calib+ACI {'SURVIVES' if survive else 'KILLED'}: best {best[0]} binding {best[1]['binding']}. "
            + ("" if survive else "Calibration+ACI does NOT hold DOWN over breakeven in 2025 — calibration is rank-monotone and ACI only starves coverage where the 2025 edge is absent. §3 gate/calibration levers applied-and-failed on DOWN.")))
    json.dump(out,open(f"{ROOT}/m5_down_calib_aci_result.json","w"),indent=2,default=str)
    for k,v in detail.items():
        b=v["binding"]; print(f"  {k}: binding win {b['win']} CI-lo {b['ci_lo']} n{b['min_n']}  2025={v.get('2025')}")
    print(f"VERDICT: {out['VERDICT']['statement']}\n-> m5_down_calib_aci_result.json ({time.time()-t0:.0f}s)")

if __name__=="__main__": main()
