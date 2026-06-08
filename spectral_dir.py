"""SCOPE: EURUSD (key-specific). LEVER FAMILY — CORPUS_LEVER_INVENTORY.md (UNTESTED, prior 0.03):
LITERAL causal DWT / SSA multiresolution BAND-SPLIT -> per-band feature -> recombine -> SIGN, as a
DIRECTION model at MX_HOR minutes. The wavelet/spectral analog of the already-KILLED HAVOK/Hankel-DMD
signed-decomposition; sign-invariance theorem demotes "sign from a spectral component".

CAUSALITY (leakage trap): every decomposition is computed on the TRAILING window [t-L+1, t] ONLY (no
future bar). DWT = pywt.wavedec(window, db4) -> per-level last-coeff + energy + slope. SSA = Hankel SVD
of the window -> top-r components, last-value + last-increment per component. Two readouts:
  (a) LEARNED recombine: all band features -> certified GBM (fwd_holdout.mk_lgb) -> P(up).
  (b) LITERAL recombine: sign(sum of leading SSA components' last increment).

Discipline + faithful gate identical to nbeats_nhits_dir.py (imported). Per-year held-out 2024/2025/2026.
PRE-REGISTERED FALSIFIER: KILL if VAL dirAUC<=0.515 OR no held-out year selacc CI95-lo>=0.541.
Usage: MX_HOR=15 ~/binary-algo-venv/bin/python spectral_dir.py -> spectral_dir_15m_result.json
"""
import os, sys, json, time
import numpy as np
import pywt
from sklearn.metrics import roc_auc_score
import nbeats_nhits_dir as NB
from fwd_holdout import mk_lgb, _sel_acc

H=NB.H; L=NB.L; COV=NB.COV; BE=NB.BE; SEED=NB.SEED; ROOT=NB.ROOT
DWT_LEVEL=int(os.environ.get("SP_DWTLVL","4")); SSA_M=int(os.environ.get("SP_SSAM", str(max(8,L//3))))
SSA_R=int(os.environ.get("SP_SSAR","6"))
TR_STRIDE=int(os.environ.get("SP_TRSTRIDE", str(max(8,H*3))))     # bound train n (per-window SVD is costly)
YR_STRIDE=int(os.environ.get("SP_YRSTRIDE","6"))                  # bound eval n per year
np.random.seed(SEED)

def _dwt_feats(W):
    """W [n,L] return windows -> DWT per-level features (causal: window-only). Returns [n,F1]."""
    feats=[]
    for x in W:
        cs=pywt.wavedec(x, 'db4', level=DWT_LEVEL, mode='periodization')
        row=[]
        for c in cs:
            row += [float(c[-1]), float(np.std(c)), float(c[-1]-c.mean())]   # last coeff(@t), energy, last-vs-mean slope
        feats.append(row)
    return np.asarray(feats,np.float32)

def _ssa_feats(W):
    """W [n,L] -> SSA top-r component features + a literal recombine signal. Returns (feats[n,F2], lit[n])."""
    M=SSA_M; feats=[]; lit=[]
    for x in W:
        K=len(x)-M+1
        Hk=np.lib.stride_tricks.sliding_window_view(x, M)[:K]        # [K,M] Hankel rows
        Hk=Hk.T                                                       # [M,K]
        try:
            U,s,Vt=np.linalg.svd(Hk, full_matrices=False)
        except np.linalg.LinAlgError:
            feats.append([0.0]*(SSA_R*2)); lit.append(0.0); continue
        r=min(SSA_R,len(s)); row=[]; lit_sum=0.0
        for k in range(r):
            comp=s[k]*np.outer(U[:,k],Vt[k])                          # rank-1 [M,K]
            # diagonal-average -> reconstructed series (length L); take last value + last increment (causal @ t)
            rec=np.array([comp[::-1,:].diagonal(j).mean() for j in range(-M+1,K)])
            row += [float(rec[-1]), float(rec[-1]-rec[-2]) if len(rec)>1 else 0.0]
            lit_sum += (rec[-1]-rec[-2]) if len(rec)>1 else 0.0
        if r<SSA_R: row += [0.0]*((SSA_R-r)*2)
        feats.append(row); lit.append(lit_sum)
    return np.asarray(feats,np.float32), np.asarray(lit,np.float32)

def build_feats(split, stride):
    X,_,y,ts=NB.build_windows(NB.SPLITS[split], NB.SINGLE_COLS, stride)   # channel 0 = r_EURUSD
    if len(X)==0: return None
    W=X[:,:,0]
    dwt=_dwt_feats(W); ssa,lit=_ssa_feats(W)
    F=np.concatenate([dwt,ssa],axis=1)
    return F, y.astype(int), lit, ts

def _boot_ci(correct, B=2000):
    if len(correct)==0: return (float('nan'),float('nan'))
    rng=np.random.default_rng(SEED); n=len(correct)
    a=np.array([correct[rng.integers(0,n,n)].mean() for _ in range(B)])
    return float(np.quantile(a,.025)), float(np.quantile(a,.975))

def metrics(p,y,cov=COV):
    y=y.astype(int); auc=float(roc_auc_score(y,p)) if len(np.unique(y))>1 else float('nan')
    conf=np.abs(p-0.5); thr=np.quantile(conf,1-cov); sel=conf>=thr
    if sel.sum()<25: return dict(acc=float('nan'),n=int(sel.sum()),ci_lo=float('nan'),ci_hi=float('nan'),uprate=float(y.mean()),auc=auc)
    correct=((p[sel]>0.5).astype(int)==y[sel]).astype(float); lo,hi=_boot_ci(correct)
    return dict(acc=float(correct.mean()),n=int(sel.sum()),ci_lo=lo,ci_hi=hi,uprate=float(y.mean()),auc=auc)

if __name__=="__main__":
    t0=time.time(); print(f"[spectral_dir] H={H} L={L} DWTlvl={DWT_LEVEL} SSA(M={SSA_M},r={SSA_R}) trstride={TR_STRIDE}",flush=True)
    tr=build_feats("train",TR_STRIDE); va=build_feats("val",max(4,TR_STRIDE//2))
    yrs={k:build_feats(f"y{k}",YR_STRIDE) for k in ["2024","2025","2026"]}
    res={"key":f"EURUSD.{H}m.direction","lever":"causal DWT/SSA band-split->recombine->sign",
         "falsifier":"KILL if VAL dirAUC<=0.515 OR no held-out year selacc CI95-lo>=0.541",
         "n_train":int(len(tr[0])) if tr else 0,"arms":[]}
    if tr and va:
        Ftr,ytr,_,_=tr; Fva,yva,_,_=va
        # (a) LEARNED recombine via certified GBM
        gbm=mk_lgb(600).fit(Ftr,ytr)
        pva=gbm.predict_proba(Fva)[:,1]; arm={"readout":"GBM_recombine","val":metrics(pva,yva)}
        arm["val_dirAUC"]=arm["val"]["auc"]; arm["years"]={}
        for k in ["2024","2025","2026"]:
            if yrs[k] is None: arm["years"][k]=None; continue
            Fy,yy,_,_=yrs[k]; arm["years"][k]=metrics(gbm.predict_proba(Fy)[:,1],yy)
        # (b) LITERAL recombine: sign(sum leading-SSA increments) -> pseudo-prob by |lit|
        litva=va[2]; pj=0.5+0.5*np.tanh(litva/(np.abs(litva).std()+1e-12)); litarm={"readout":"literal_SSA_recombine","val":metrics(pj,yva)}
        litarm["val_dirAUC"]=litarm["val"]["auc"]; litarm["years"]={}
        for k in ["2024","2025","2026"]:
            if yrs[k] is None: litarm["years"][k]=None; continue
            lit=yrs[k][2]; pj=0.5+0.5*np.tanh(lit/(np.abs(lit).std()+1e-12)); litarm["years"][k]=metrics(pj,yrs[k][1])
        for a in (arm,litarm):
            yy=[a["years"][k] for k in ["2024","2025","2026"] if a["years"][k]]
            val_ok=(a["val_dirAUC"] is not None) and a["val_dirAUC"]>0.515
            any_clear=any((r["ci_lo"]==r["ci_lo"]) and r["ci_lo"]>=BE for r in yy)
            a["KILL"]=not(val_ok and any_clear); a["verdict"]="KILLED" if a["KILL"] else "SURVIVED-escalate-to-CPCV"
            res["arms"].append(a)
            print(f"  {a['readout']:22s} valAUC={a['val_dirAUC']:.4f} "
                  f"2024={a['years']['2024']['acc'] if a['years']['2024'] else None} "
                  f"2025={a['years']['2025']['acc'] if a['years']['2025'] else None} "
                  f"2026={a['years']['2026']['acc'] if a['years']['2026'] else None} -> {a['verdict']}",flush=True)
    res["any_survived"]=any(not a["KILL"] for a in res["arms"]); res["secs"]=round(time.time()-t0,1)
    op=f"{ROOT}/spectral_dir_{H}m_result.json"; json.dump(res,open(op,"w"),indent=1)
    print(f"[done] {op} any_survived={res['any_survived']} ({res['secs']}s)",flush=True)
