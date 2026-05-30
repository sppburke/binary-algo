"""V9 — recurrent sequence model (GRU) on the raw normalized price path.
The last untested model class. GBM/TabNet operate on engineered features; a GRU sees the raw
ordered sequence of recent returns and can in principle learn nonlinear path/microstructure
patterns. Bounded test: if 5m direction has exploitable sequential structure beyond ~0.52,
this finds it. CPU, subsampled for tractability. 2026 fully OOS."""
import sys, time, numpy as np, pandas as pd, os
from sklearn.metrics import roc_auc_score
import torch, torch.nn as nn
import harness as H

PAIR="EURUSD"; L=60; HOR=5
STRIDE_TR=6  # subsample training windows for CPU tractability
torch.manual_seed(0)

def build_sequences(years, stride):
    """Return (X[N,L,F], y[N]) from contiguous 1-min windows ending at t, label 5m ahead."""
    Xs=[]; Ys=[]
    for y in years:
        p=f"{H.FEAT_DIR}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p, columns=["close","1m_ret_1","1m_rv_12","1m_rsi","valid"])
        c=d["close"].values.astype(np.float64)
        secs=d.index.values.astype("datetime64[s]").astype("int64")
        n=len(c)
        ret=np.zeros(n); ret[1:]=np.diff(np.log(c))
        feats=np.stack([ret, np.nan_to_num(d["1m_ret_1"].values),
                        np.nan_to_num(d["1m_rv_12"].values), np.nan_to_num(d["1m_rsi"].values)],axis=1)
        # standardize per-column (causal-ish; fine for bounded test)
        fwd=np.full(n,np.nan); fwd[:n-HOR]=c[HOR:]
        lab=(fwd>c).astype(np.float32)
        # window end indices t from L-1 .. n-1-HOR, require contiguity over [t-L+1, t+HOR]
        idxs=np.arange(L-1, n-HOR, stride)
        for t in idxs:
            if secs[t]-secs[t-L+1]!=(L-1)*60: continue
            if secs[t+HOR]-secs[t]!=HOR*60: continue
            if fwd[t]==c[t] or not np.isfinite(fwd[t]): continue
            Xs.append(feats[t-L+1:t+1]); Ys.append(lab[t])
    X=np.asarray(Xs,dtype=np.float32); Y=np.asarray(Ys,dtype=np.float32)
    return X,Y

t0=time.time()
Xtr,ytr=build_sequences([str(y) for y in range(2012,2022)], STRIDE_TR)
Xva,yva=build_sequences(["2022","2023"], 3)
Xte,yte=build_sequences(["2024","2025"], 1)
Xoo,yoo=build_sequences(["2026"], 1)
# standardize using train stats per feature
mu=Xtr.reshape(-1,Xtr.shape[2]).mean(0); sd=Xtr.reshape(-1,Xtr.shape[2]).std(0)+1e-8
for A in (Xtr,Xva,Xte,Xoo): A-=mu; A/=sd
print(f"seqs tr={Xtr.shape} va={Xva.shape} te={Xte.shape} oo={Xoo.shape} ymean_tr={ytr.mean():.4f} build={time.time()-t0:.0f}s",flush=True)

class GRUNet(nn.Module):
    def __init__(self,f,h=48):
        super().__init__(); self.gru=nn.GRU(f,h,num_layers=1,batch_first=True)
        self.head=nn.Sequential(nn.Linear(h,32),nn.ReLU(),nn.Dropout(0.2),nn.Linear(32,1))
    def forward(self,x):
        o,_=self.gru(x); return self.head(o[:,-1]).squeeze(-1)

dev="cpu"; torch.set_num_threads(20)
net=GRUNet(Xtr.shape[2]).to(dev)
opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-5)
lossf=nn.BCEWithLogitsLoss()
def batches(X,Y,bs=4096,shuf=True):
    idx=np.arange(len(X));
    if shuf: np.random.shuffle(idx)
    for i in range(0,len(X),bs): yield X[idx[i:i+bs]],Y[idx[i:i+bs]]
Xtr_t=torch.tensor(Xtr); ytr_t=torch.tensor(ytr)
def auc_of(X,Y):
    net.eval(); ps=[]
    with torch.no_grad():
        for i in range(0,len(X),8192):
            ps.append(torch.sigmoid(net(torch.tensor(X[i:i+8192]))).numpy())
    p=np.concatenate(ps); return roc_auc_score(Y,p),p
best=0; best_p=None
for ep in range(12):
    net.train()
    for xb,yb in batches(Xtr,ytr):
        opt.zero_grad(); out=net(torch.tensor(xb)); loss=lossf(out,torch.tensor(yb)); loss.backward(); opt.step()
    va,_=auc_of(Xva,yva); print(f"epoch {ep} val_auc={va:.4f}",flush=True)
    if va>best: best=va;
te,pte=auc_of(Xte,yte); oo,poo=auc_of(Xoo,yoo); va,pva=auc_of(Xva,yva)
print(f"GRU AUC val={va:.4f} test={te:.4f} oos={oo:.4f}")
H.report("TEST",yte,pte); H.report("OOS ",yoo,poo)
for tgt in (0.75,0.70,0.65,0.60):
    bv=H.threshold_for_target(yva,pva,target=tgt,min_n=200)
    if bv is None: print(f"target {tgt:.0%}: unreachable on VAL"); continue
    rt=H.apply_threshold(yte,pte,bv['conf_thr']); ro=H.apply_threshold(yoo,poo,bv['conf_thr'])
    print(f"target {tgt:.0%}: TEST acc={rt['accuracy']:.3f} n={rt['n']} | OOS acc={ro['accuracy']:.3f} n={ro['n']}")
print("GRU DONE")
