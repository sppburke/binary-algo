"""1-MINUTE v14 — temporal 1D-CNN on the raw 1s microstructure path (direction).

GBMs see only hand-crafted summaries and cap at ~0.51 direction AUC. A small 1D-CNN reads the actual
last-W-seconds path of [1s mid-returns, order-book imbalance, microprice deviation] to try to extract
the breakout/flow dynamics directly. Subsampled training (CPU). If it lifts direction AUC past ~0.53 it's
a new signal; evaluate its selective frontier in the compression-release regime. Heartbeat (HB) progress.
"""
import time, numpy as np, pandas as pd
import torch, torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import roc_auc_score
TICK="/media/sean/CORSAIR/binary-algo/features_tick"; HS=60; W=60
torch.set_num_threads(20); T0=time.time()
def hb(m): print(f"HB[{time.time()-T0:.0f}s] {m}",flush=True)

def chans(b):
    mid=b["mid"].values.astype("float64"); imb=b["imb"].fillna(0).values.astype("float64")
    micro=b["micro"].values.astype("float64")
    ret1=np.diff(mid,prepend=mid[0])/mid                       # 1s return
    md=(micro-mid)/mid
    return np.stack([ret1,imb,md],axis=1)                      # (N,3)

def build(split):
    b=pd.read_parquet(f"{TICK}/{split}_1s.parquet")
    mid=b["mid"].values.astype("float64"); ts=b.index.values.astype("datetime64[s]").astype("int64")
    C=chans(b); n=len(mid)
    fwd=np.full(n,np.nan); fwd[:n-HS]=mid[HS:]; ret=fwd/mid-1
    # bbw regime for gating (compression-release)
    r1=pd.Series(mid).pct_change(); bbw1800=(r1.rolling(1800).std()*np.sqrt(1800)).values
    bbw300=(r1.rolling(300).std()*np.sqrt(300)).values; rel=bbw300/(bbw1800+1e-12)
    return mid,ts,C,ret,bbw1800,rel

class Net(nn.Module):
    def __init__(s,ch=3):
        super().__init__()
        s.net=nn.Sequential(
            nn.Conv1d(ch,32,5,padding=2),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(32,64,5,padding=2),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(64,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        s.fc=nn.Linear(64,1)
    def forward(s,x): return s.fc(s.net(x).squeeze(-1)).squeeze(-1)

def windows(C,idx):
    # C:(N,3) -> windows ending at i: C[i-W+1:i+1]; return (len,3,W)
    sw=sliding_window_view(C,(W,3))[:,0,:,:]    # (N-W+1, W, 3) view; row j = C[j:j+W]
    # window ending at i corresponds to start i-W+1 -> sw index i-W+1
    sel=idx-(W-1)
    return sw[sel].transpose(0,2,1)             # (len,3,W)

def main():
    data={sp:build(sp) for sp in ["train","val","test","oos"]}; hb("built channels+labels")
    # normalization stats from train channels
    mu=data["train"][2].mean(0); sd=data["train"][2].std(0)+1e-9
    def valid_idx(sp,stride):
        mid,ts,C,ret,bbw,rel=data[sp]; n=len(mid)
        ok=np.zeros(n,bool); ok[W-1:n-HS]=True
        ok&=np.isfinite(ret)&(ret!=0)&np.isfinite(bbw)
        idx=np.where(ok)[0]
        return idx[::stride]
    def XY(sp,stride):
        mid,ts,C,ret,bbw,rel=data[sp]; idx=valid_idx(sp,stride)
        Cn=(C-mu)/sd
        Xw=windows(Cn,idx).astype("float32")
        y=(ret[idx]>0).astype("float32")
        return Xw,y,idx
    Xtr,ytr,_=XY("train",24); hb(f"train windows {Xtr.shape}")
    Xva,yva,iva=XY("val",6);  Xte,yte,ite=XY("test",6); Xoo,yoo,ioo=XY("oos",3)
    hb(f"eval windows te={Xte.shape} oo={Xoo.shape}")
    dev="cpu"; net=Net().to(dev); opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-5)
    lossf=nn.BCEWithLogitsLoss()
    Xtr_t=torch.from_numpy(Xtr); ytr_t=torch.from_numpy(ytr)
    bs=4096; nb=len(Xtr)//bs
    for ep in range(6):
        net.train(); perm=torch.randperm(len(Xtr))
        tot=0.0
        for bi in range(nb):
            j=perm[bi*bs:(bi+1)*bs]; xb=Xtr_t[j].to(dev); yb=ytr_t[j].to(dev)
            opt.zero_grad(); out=net(xb); loss=lossf(out,yb); loss.backward(); opt.step(); tot+=loss.item()
        # quick val AUC
        net.eval()
        with torch.no_grad():
            pv=torch.sigmoid(net(torch.from_numpy(Xva).to(dev))).cpu().numpy()
        hb(f"epoch {ep} loss={tot/nb:.4f} valAUC={roc_auc_score(yva,pv):.4f}")
    net.eval()
    with torch.no_grad():
        pv=torch.sigmoid(net(torch.from_numpy(Xva))).numpy()
        pt=torch.sigmoid(net(torch.from_numpy(Xte))).numpy()
        po=torch.sigmoid(net(torch.from_numpy(Xoo))).numpy()
    hb(f"CNN AUC val={roc_auc_score(yva,pv):.4f} test={roc_auc_score(yte,pt):.4f} oos={roc_auc_score(yoo,po):.4f}")
    # selective frontier in compression-release regime (non-overlapping)
    def regime(sp,idx):
        mid,ts,C,ret,bbw,rel=data[sp]
        qb=np.nanpercentile(data["val"][3]*0+data["val"][4],33) if False else None
        return bbw[idx],rel[idx],ts[idx]
    bbV,relV,tsv=regime("val",iva); bbT,relT,tst=regime("test",ite); bbO,relO,tso=regime("oos",ioo)
    qb=np.nanpercentile(bbV,33); rq=np.nanpercentile(relV[bbV<=qb],80)
    def nonov(ts,conf,corr,thr,gap=HS):
        sel=np.where(conf>=thr)[0]
        if len(sel)==0: return np.nan,0
        order=sel[np.argsort(-conf[sel])]
        if len(order)>60000: order=order[:60000]
        tmin=int(ts[order].min()); span=int(ts[order].max()-tmin)+gap+2; blk=np.zeros(span,bool); tk=[]
        for i in order:
            t=int(ts[i]-tmin)
            if blk[t]: continue
            tk.append(i); blk[max(0,t-gap+1):t+gap]=True
        tk=np.array(tk); return corr[tk].mean(), len(tk)
    ct=(pt>0.5).astype(int)==yte.astype(int); co=(po>0.5).astype(int)==yoo.astype(int)
    cft=np.abs(pt-0.5); cfo=np.abs(po-0.5); cfv=np.abs(pv-0.5)
    gt=(bbT<=qb)&(relT>=rq); go=(bbO<=qb)&(relO>=rq); gv=(bbV<=qb)&(relV>=rq)
    print(f"\n{'gate':>16} {'cov':>6} {'TESTacc':>8} {'TESTn':>6} {'OOSacc':>7} {'OOSn':>5}",flush=True)
    for nm,(mt,mo,mvk) in (("ALL",(np.ones(len(yte),bool),np.ones(len(yoo),bool),np.ones(len(yva),bool))),
                           ("comp-release",(gt,go,gv))):
        cfvg=cfv[mvk]
        for cov in (0.10,0.05,0.02):
            thr=np.quantile(cfvg,1-cov)
            at,nt=nonov(tst[mt],cft[mt],ct[mt],thr); ao,no=nonov(tso[mo],cfo[mo],co[mo],thr)
            print(f"{nm:>16} {cov:>6.2%} {at:>8.3f} {nt:>6} {ao:>7.3f} {no:>5}",flush=True)
    hb("MIN1-V14 DONE")

if __name__=="__main__":
    main()
