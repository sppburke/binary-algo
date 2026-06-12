"""USDJPY 2-MIN — Tier-D sequence/DL coverage (GRU on bar-feature path). H=2 fork.

SCOPE: USDJPY · 2m. Coverage-rule 'run once' Tier-D. A GRU over W recent 1-min bars → next-2m sign. Tests whether a
trained SEQUENCE model captures intra-path direction structure the per-bar pooled GBM misses (the on-disk new-signal
question). EURUSD D1-D6 + USDJPY-1m GRU were null (info-bound); confirm at USDJPY 2m. Selection on VAL; per-year
held-out AUC + cov UP/DOWN win-rate (nonoverlap gap=120, ties LOSE). Modest CPU GRU.
Usage: ~/binary-algo-venv/bin/python usdjpy_2m_dl.py
"""
import os, json, time, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import torch, torch.nn as nn
import harness as H
from usdjpy_2m_base import nonoverlap_chrono, boot   # gap defaults to 120 here

PAIR="USDJPY"; FEAT=H.FEAT_DIR; W=30; HOR=2; STEP=60; GAP=HOR*STEP; BE=0.541
SPL={"train":[str(y) for y in range(2012,2022)],"val":["2022","2023"],
     "test24":["2024"],"test25":["2025"],"oos":["2026"]}
COLS=["1m_ret_1","1m_ret_3","1m_ret_6","1m_ret_12","1m_ret_24","1m_rsi","1m_rangepos_12",
      "1m_dist_ema10","1m_dist_ema20","1m_rv_12","1m_rv_24","1m_macd_hist","5m_ret_3","5m_ret_12","4h_slope_20"]
RESULT="usdjpy_2m_dl_result.json"
torch.manual_seed(7); np.random.seed(7)

def windows(years, stride, cap):
    """(Xw [N,W,F], y [N], ts [N], moved [N]) of consecutive-bar windows ending at t, label=sign(close[t+HOR]-close[t])."""
    Xs=[];ys=[];tss=[];mv=[]
    for y in years:
        p=f"{FEAT}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p,columns=COLS+["close"]); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float); ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        F=d[COLS].astype("float32").fillna(0.0).values
        fr=np.full(n,np.nan); fr[:n-HOR]=c[HOR:]/c[:-HOR]-1.0           # 2-min forward return
        for t in range(W-1, n-HOR):
            if (ts[t]-ts[t-W+1])!=(W-1)*60: continue                    # window contiguous
            if (ts[t+HOR]-ts[t])!=GAP: continue                         # label horizon clean
            if not np.isfinite(fr[t]) or fr[t]==0: continue
            Xs.append(F[t-W+1:t+1]); ys.append(1 if fr[t]>0 else 0); tss.append(ts[t]); mv.append(True)
    if not Xs: return None
    idx=np.arange(len(Xs))
    if stride>1: idx=idx[::stride]
    if len(idx)>cap: idx=idx[np.linspace(0,len(idx)-1,cap).astype(int)]
    return (np.asarray(Xs,dtype="float32")[idx], np.asarray(ys)[idx],
            np.asarray(tss)[idx], np.asarray(mv)[idx])

class GRU(nn.Module):
    def __init__(self,f,h=32):
        super().__init__(); self.g=nn.GRU(f,h,batch_first=True); self.fc=nn.Linear(h,1)
    def forward(self,x): o,_=self.g(x); return self.fc(o[:,-1,:]).squeeze(-1)

def main():
    t0=time.time()
    tr=windows(SPL["train"],stride=40,cap=120000); va=windows(SPL["val"],stride=4,cap=60000)
    mu=tr[0].reshape(-1,tr[0].shape[2]).mean(0); sd=tr[0].reshape(-1,tr[0].shape[2]).std(0)+1e-6
    def norm(X): return (X-mu)/sd
    Xtr=torch.tensor(norm(tr[0])); ytr=torch.tensor(tr[1],dtype=torch.float32)
    Xva=torch.tensor(norm(va[0])); yva=va[1]
    print(f"[dl2m] train windows={len(tr[1]):,} val={len(va[1]):,} F={len(COLS)} W={W} build={time.time()-t0:.0f}s",flush=True)
    dev="cpu"; m=GRU(len(COLS)).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
    lossf=nn.BCEWithLogitsLoss(); bs=2048; best=(0.0,None)
    for ep in range(8):
        m.train(); perm=torch.randperm(len(ytr))
        for i in range(0,len(ytr),bs):
            j=perm[i:i+bs]; opt.zero_grad()
            out=m(Xtr[j].to(dev)); l=lossf(out,ytr[j].to(dev)); l.backward(); opt.step()
        m.eval()
        with torch.no_grad(): pv=torch.sigmoid(m(Xva.to(dev))).cpu().numpy()
        auc=roc_auc_score(yva,pv)
        if auc>best[0]: best=(auc,{k:v.clone() for k,v in m.state_dict().items()})
        print(f"  ep{ep} VAL AUC={auc:.4f} {time.time()-t0:.0f}s",flush=True)
    m.load_state_dict(best[1]); res={"key":"USDJPY.2m","tier":"D GRU","val_auc":float(best[0]),"W":W,"feats":COLS,"years":{}}
    for w in ("test24","test25","oos"):
        D=windows(SPL[w],stride=2,cap=120000)
        with torch.no_grad(): pr=torch.sigmoid(m(torch.tensor(norm(D[0])).to(dev))).cpu().numpy()
        yw=D[1]; ts=D[2]; auc=float(roc_auc_score(yw,pr)); conf=np.abs(pr-0.5); out={}
        for cov in (0.05,0.02):
            thr=float(np.quantile(conf,1-cov)); cand=conf>=thr; sel=nonoverlap_chrono(ts,cand)
            if len(sel)==0: continue
            pred=(pr[sel]>0.5).astype(int); win=(pred==yw[sel]).astype(float)
            up=pred==1; dn=pred==0
            out[f"cov{cov}"]={"UP":[round(float(win[up].mean()),4),int(up.sum())] if up.sum() else None,
                              "DOWN":[round(float(win[dn].mean()),4),int(dn.sum())] if dn.sum() else None}
        res["years"][w]={"auc":auc,"sel":out}
        print(f"=== {w} === AUC={auc:.4f} sel={out}",flush=True)
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"[dl2m] saved {RESULT} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
