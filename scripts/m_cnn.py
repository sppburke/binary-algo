"""Seconds-scale 1D-CNN on the RAW tick path — creative model type vs the GBM ceiling (AUC 0.527).

GBMs see only hand-crafted summaries. This reads the actual last-W-seconds path of microstructure channels
[1s mid-return, order-flow imbalance, microprice deviation, imbalance-EMA, signed-return] and learns the
breakout/flow dynamics directly. Label = deriv-faithful wc_ret at HS=5s (next-tick entry, true expiry, ties lose).
Honest selective eval: chronological non-overlap (no look-ahead), CI95. If it lifts AUC past ~0.53 and the selective
tail past the GBM's ~0.64, it's the creative win toward >0.65.

  python m_cnn.py [HS] [W]
"""
import sys, time, numpy as np, pandas as pd
import torch, torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import roc_auc_score
from min1_production import wc_ret, nonoverlap_chrono, boot, load_split, set_pair
set_pair("EURUSD")
TICK="/home/sean/git/binary-algo/features_tick"
HS=int(sys.argv[1]) if len(sys.argv)>1 else 5
W=int(sys.argv[2]) if len(sys.argv)>2 else 60
ARCH=sys.argv[3] if len(sys.argv)>3 else "cnn"   # cnn | gru
TOL=max(2,HS//30+1); LAG=1
torch.set_num_threads(20); T0=time.time()
def hb(m): print(f"HB[{time.time()-T0:.0f}s] {m}",flush=True)

def channels(b):
    mid=b["mid"].values.astype("float64"); imb=b["imb"].fillna(0).values.astype("float64")
    micro=b["micro"].values.astype("float64")
    ret1=np.diff(mid,prepend=mid[0])/mid
    md=(micro-mid)/mid
    imb_ema=pd.Series(imb).ewm(span=20).mean().values
    sret=np.sign(ret1)
    return np.stack([ret1,imb,md,imb_ema,sret],axis=1)   # (N,5)

def build(split):
    b=load_split(split)
    mid=b["mid"].values.astype("float64"); ts=b.index.values.astype("datetime64[s]").astype("int64")
    hour=b.index.hour.values
    C=channels(b)
    ret,valid=wc_ret(ts,mid,HS,TOL,LAG); y=(ret>0).astype("float32")
    return mid,ts,hour,C,y,valid

class Net(nn.Module):
    def __init__(s,ch):
        super().__init__()
        s.net=nn.Sequential(
            nn.Conv1d(ch,32,5,padding=2),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(32,64,5,padding=2),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(64,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        s.fc=nn.Linear(64,1)
    def forward(s,x): return s.fc(s.net(x).squeeze(-1)).squeeze(-1)

class GRUNet(nn.Module):
    def __init__(s,ch,hid=48):
        super().__init__(); s.gru=nn.GRU(ch,hid,num_layers=1,batch_first=True); s.fc=nn.Linear(hid,1)
    def forward(s,x):                          # x:(B,ch,W) -> (B,W,ch)
        o,_=s.gru(x.transpose(1,2)); return s.fc(o[:,-1,:]).squeeze(-1)

def main():
    data={sp:build(sp) for sp in ["train","val","test","oos"]}; hb(f"channels+labels HS={HS} W={W}")
    mu=data["train"][3].mean(0); sd=data["train"][3].std(0)+1e-9
    def XY(sp,stride):
        mid,ts,hour,C,y,valid=data[sp]; n=len(mid)
        ok=np.zeros(n,bool); ok[W-1:]=True; ok&=valid
        idx=np.where(ok)[0][::stride]
        Cn=((C-mu)/sd).astype("float32")
        sw=sliding_window_view(Cn,(W,Cn.shape[1]))[:,0,:,:]   # (N-W+1, W, ch)
        Xw=sw[idx-(W-1)].transpose(0,2,1).astype("float32")   # (len, ch, W)
        return Xw, y[idx], ts[idx], hour[idx]
    Xtr,ytr,_,_=XY("train",20); hb(f"train windows {Xtr.shape}")
    Xva,yva,tsv,hv=XY("val",3); Xte,yte,tst,ht=XY("test",3); Xoo,yoo,tso,ho=XY("oos",2)
    hb(f"eval windows te={Xte.shape} oo={Xoo.shape}")
    net=GRUNet(Xtr.shape[1]) if ARCH=="gru" else Net(Xtr.shape[1])
    hb(f"arch={ARCH}")
    opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-5); lossf=nn.BCEWithLogitsLoss()
    Xtr_t=torch.from_numpy(Xtr); ytr_t=torch.from_numpy(ytr); bs=4096; nb=len(Xtr)//bs
    for ep in range(6):
        net.train(); perm=torch.randperm(len(Xtr)); tot=0.0
        for bi in range(nb):
            j=perm[bi*bs:(bi+1)*bs]; opt.zero_grad(); out=net(Xtr_t[j]); loss=lossf(out,ytr_t[j]); loss.backward(); opt.step(); tot+=loss.item()
        net.eval()
        with torch.no_grad(): pv=torch.sigmoid(net(torch.from_numpy(Xva))).numpy()
        hb(f"epoch {ep} loss={tot/nb:.4f} valAUC={roc_auc_score(yva,pv):.4f}")
    net.eval()
    def predict(X, bs=16384):
        out=[]
        with torch.no_grad():
            for k in range(0,len(X),bs): out.append(torch.sigmoid(net(torch.from_numpy(X[k:k+bs]))).numpy())
        return np.concatenate(out)
    pv=predict(Xva); pt=predict(Xte); po=predict(Xoo)
    hb(f"CNN AUC val={roc_auc_score(yva,pv):.4f} test={roc_auc_score(yte,pt):.4f} oos={roc_auc_score(yoo,po):.4f}")
    gap=HS+TOL
    def gate(hour,nm): ny=(hour>=12)&(hour<21); return {"none":np.ones(len(hour),bool),"ny":ny}[nm]
    def indep(p,y,ts,g,thr):
        m=g&(np.abs(p-0.5)>=thr)
        if m.sum()==0: return (0,np.nan,np.nan,np.nan)
        s=nonoverlap_chrono(ts,m,gap)
        if len(s)==0: return (0,np.nan,np.nan,np.nan)
        corr=((p[s]>0.5).astype(int)==y[s].astype(int)).astype(float); return (len(s),corr.mean(),*boot(corr))
    print(f"\n{'gate':>5} {'cov':>6} {'TEST':>22} {'OOS':>22}")
    for gnm in ("none","ny"):
        gv=gate(hv,gnm)
        for cov in (0.02,0.01,0.005):
            thr=float(np.quantile(np.abs(pv-0.5)[gv],1-cov))
            nt,at,lot,hit=indep(pt,yte,tst,gate(ht,gnm),thr); no,ao,loo,hoo=indep(po,yoo,tso,gate(ho,gnm),thr)
            print(f"{gnm:>5} {cov:>6.1%}  TEST:n{nt} {at:.3f}[{lot:.2f},{hit:.2f}]   OOS:n{no} {ao:.3f}[{loo:.2f},{hoo:.2f}]",flush=True)
    hb("CNN DONE")

if __name__=="__main__":
    main()
