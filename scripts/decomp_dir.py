"""SCOPE: EURUSD (key-specific). LEVER FAMILY — CORPUS_LEVER_INVENTORY.md:333,359 (UNTESTED, prior 0.03):
DLinear / Autoformer-style series-decomposition + AutoCorrelation, and a TFT-style QUANTILE head, as
DIRECTION models at MX_HOR minutes (single-pair AND cross-pair 7-pair panel input). GPU + AMP.
FEDformer's mechanism (frequency-domain mixing) is covered by the `freq` decomposition variant.

Readouts: (a) forecast cumulative return -> sign (logistic VAL-fit -> P(up)); (b) TFT quantile fan ->
P(up) from where 0 sits in the predicted quantile distribution, abstain when the fan straddles 0.
Also the "cross-sectional vol/quantile GATE" framing (predict |move| quantile, gate direction) — which
the sign-invariance theorem predicts collapses to a magnitude gate (already KILLED); reported for completeness.

Discipline + faithful gate + pre-registered falsifier identical to nbeats_nhits_dir.py (imported).
KILL the arm if VAL dirAUC<=0.515 OR no held-out year selacc CI95-lo>=0.541.
Usage: MX_HOR=10 ~/binary-algo-venv/bin/python decomp_dir.py  -> decomp_dir_10m_result.json
"""
import os, sys, json, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
import nbeats_nhits_dir as NB   # reuse build_windows, sel_metrics, SPLITS, *_COLS, standardize, gate constants

H=NB.H; L=NB.L; COV=NB.COV; BE=NB.BE; DEV=NB.DEV; SEED=NB.SEED; ROOT=NB.ROOT
EPOCHS=int(os.environ.get("NB_EPOCHS","12")); STRIDE_TR=NB.STRIDE_TR; MAXTR=NB.MAXTR
QUANTS=[0.1,0.25,0.5,0.75,0.9]
torch.manual_seed(SEED); np.random.seed(SEED)

# ----------------------------- decomposition --------------------------------
def moving_avg_causal(x, k):
    """Trailing (causal) moving average along time. x [B,L,C]."""
    pad = x[:, :1, :].repeat(1, k-1, 1)
    xp = torch.cat([pad, x], dim=1)
    w = torch.ones(1,1,k,device=x.device)/k
    B,Lt,C = x.shape
    t = xp.transpose(1,2).reshape(B*C,1,-1)
    tr = torch.nn.functional.conv1d(t, w).reshape(B,C,Lt).transpose(1,2)
    return tr                                  # trend [B,L,C]

class SeriesDecomp(nn.Module):
    def __init__(self,k=25): super().__init__(); self.k=k
    def forward(self,x): tr=moving_avg_causal(x,self.k); return tr, x-tr   # trend, seasonal

class DLinear(nn.Module):
    """Decomposition + per-component linear map L->H (arXiv:2205.13504; beats Autoformer on FX)."""
    def __init__(self,L,C,H,freq=False):
        super().__init__(); self.dec=SeriesDecomp(25); self.C=C; self.L=L; self.H=H; self.freq=freq
        self.lt=nn.Linear(L*C,H); self.ls=nn.Linear(L*C,H)
    def forward(self,x):
        tr,se=self.dec(x); B=x.shape[0]
        return self.lt(tr.reshape(B,-1))+self.ls(se.reshape(B,-1))

class AutoCorr(nn.Module):
    """FFT AutoCorrelation + time-delay aggregation (Autoformer arXiv:2106.13008), fully vectorized:
    weights = softmax(autocorr over lags); agg = circular delay-aggregation of x by those weights (done as
    a circular convolution via FFT, no per-sample python loop)."""
    def __init__(self,C): super().__init__(); self.proj=nn.Linear(C,C)
    def forward(self,x):                       # x [B,L,C]
        B,Lt,C=x.shape; q=self.proj(x)
        fq=torch.fft.rfft(q,dim=1); corr=torch.fft.irfft(fq*torch.conj(fq),n=Lt,dim=1).mean(-1)  # [B,L] autocorr
        w=torch.softmax(corr,dim=1)                                                                # lag weights [B,L]
        Fx=torch.fft.rfft(x,dim=1); Fw=torch.fft.rfft(w,dim=1).unsqueeze(-1)                       # [B,L//2+1,1]
        return torch.fft.irfft(Fx*Fw, n=Lt, dim=1)                                                 # delay-aggregated

class AutoformerLite(nn.Module):
    def __init__(self,L,C,H,freq=False):
        super().__init__(); self.dec=SeriesDecomp(25); self.ac=AutoCorr(C); self.freq=freq
        self.enc=nn.Sequential(nn.Linear(L*C,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU())
        self.head=nn.Linear(128,H)
    def forward(self,x):
        tr,se=self.dec(x); s=self.ac(se); B=x.shape[0]
        z=self.enc((tr+s).reshape(B,-1)); return self.head(z)

class FreqLinear(nn.Module):
    """FEDformer-style frequency-enhanced linear forecaster (arXiv:2201.12740): linear mixing over the
    low-frequency Fourier modes of the lookback window -> H-step forecast. A genuinely distinct mechanism
    from DLinear's time-domain decomposition."""
    def __init__(self,L,C,H,modes=16):
        super().__init__(); self.modes=min(modes, L//2+1); self.H=H
        self.wr=nn.Parameter(torch.randn(self.modes*C, H)*0.02)
        self.wi=nn.Parameter(torch.randn(self.modes*C, H)*0.02)
    def forward(self,x):                        # x [B,L,C]
        f=torch.fft.rfft(x,dim=1)[:, :self.modes, :]           # low Fourier modes [B,modes,C]
        B=x.shape[0]
        return f.real.reshape(B,-1)@self.wr + f.imag.reshape(B,-1)@self.wi   # [B,H]

class QuantileNet(nn.Module):
    """TFT-style multi-quantile head: predicts Q quantiles of the H-step cumulative return."""
    def __init__(self,L,C,nq):
        super().__init__(); self.dec=SeriesDecomp(25)
        self.body=nn.Sequential(nn.Linear(L*C,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU())
        self.head=nn.Linear(128,nq)
    def forward(self,x):
        tr,se=self.dec(x); B=x.shape[0]; z=self.body((tr+se).reshape(B,-1)); return self.head(z)

def make(arch,L,C,H):
    if arch=="dlinear":   return DLinear(L,C,H)
    if arch=="autoformer":return AutoformerLite(L,C,H)
    if arch=="freq":      return FreqLinear(L,C,H)          # FEDformer-style frequency-domain forecaster
    raise ValueError(arch)

# ----------------------------- train ---------------------------------------
def _to(x): return torch.tensor(x)
def train_mse(arch,Xtr,Rtr,Xva,Rva,C):
    torch.manual_seed(SEED); net=make(arch,L,C,H).to(DEV)
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-5)
    sc=torch.amp.GradScaler(DEV) if DEV=="cuda" else None
    rsc=1.0/(Rtr.std()+1e-8); xtr=_to(Xtr); rtr=_to(Rtr*rsc)
    xva=_to(Xva); rva=_to(Rva*rsc)   # keep on CPU; batch in eval
    n=len(xtr); bs=4096; best=1e9; bstate=None; bad=0
    for ep in range(EPOCHS):
        net.train(); perm=torch.randperm(n)
        for j in range(0,n,bs):
            idx=perm[j:j+bs]; xb=xtr[idx].to(DEV); rb=rtr[idx].to(DEV); opt.zero_grad()
            if sc is not None:
                with torch.amp.autocast(DEV): loss=((net(xb)-rb)**2).mean()
                sc.scale(loss).backward(); sc.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(),5.0); sc.step(opt); sc.update()
            else: loss=((net(xb)-rb)**2).mean(); loss.backward()
            if sc is None: torch.nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step()
        net.eval()
        with torch.no_grad():
            se=0.0; cnt=0
            for jj in range(0,len(xva),8192):
                vb=xva[jj:jj+8192].to(DEV); rb=rva[jj:jj+8192].to(DEV)
                if DEV=="cuda":
                    with torch.amp.autocast(DEV): vp=net(vb)
                else: vp=net(vb)
                se+=((vp.float()-rb)**2).sum().item(); cnt+=rb.numel()
            vl=se/max(cnt,1)
        if vl<best-1e-6: best=vl; bstate={k:v.detach().clone() for k,v in net.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=3: break
    if bstate: net.load_state_dict(bstate)
    net.eval(); return net,rsc

def train_quant(Xtr,Rtr,Xva,Rva,C):
    """Pinball-loss quantile net on cumulative return target."""
    torch.manual_seed(SEED); nq=len(QUANTS); net=QuantileNet(L,C,nq).to(DEV)
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-5)
    qs=torch.tensor(QUANTS,device=DEV).view(1,-1)
    ctr=Rtr.sum(1,keepdims=True); cva=Rva.sum(1,keepdims=True)         # cumulative return target
    rsc=1.0/(ctr.std()+1e-8)
    xtr=_to(Xtr); ytr=_to(ctr*rsc); xva=_to(Xva); yva=_to(cva*rsc)   # val on CPU; batch in eval
    n=len(xtr); bs=4096; best=1e9; bstate=None; bad=0
    for ep in range(EPOCHS):
        net.train(); perm=torch.randperm(n)
        for j in range(0,n,bs):
            idx=perm[j:j+bs]; xb=xtr[idx].to(DEV); yb=ytr[idx].to(DEV); opt.zero_grad()
            pred=net(xb); err=yb-pred; loss=torch.maximum(qs*err,(qs-1)*err).mean()
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            num=0.0; cnt=0
            for jj in range(0,len(xva),8192):
                vb=xva[jj:jj+8192].to(DEV); yb=yva[jj:jj+8192].to(DEV)
                ve=yb-net(vb); num+=torch.maximum(qs*ve,(qs-1)*ve).sum().item(); cnt+=ve.numel()
            vl=num/max(cnt,1)
        if vl<best-1e-6: best=vl; bstate={k:v.detach().clone() for k,v in net.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=3: break
    if bstate: net.load_state_dict(bstate)
    net.eval(); return net,rsc

def cumret(net,X,rsc):
    out=[]
    with torch.no_grad():
        for j in range(0,len(X),8192):
            xb=_to(X[j:j+8192]).to(DEV)
            if DEV=="cuda":
                with torch.amp.autocast(DEV): p=net(xb)
            else: p=net(xb)
            out.append(p.float().sum(1).cpu().numpy())
    return np.nan_to_num(np.concatenate(out)/(rsc if rsc else 1.0), nan=0.0, posinf=0.0, neginf=0.0)

def quant_pred(net,X):
    out=[]
    with torch.no_grad():
        for j in range(0,len(X),8192):
            out.append(net(_to(X[j:j+8192]).to(DEV)).float().cpu().numpy())
    return np.nan_to_num(np.concatenate(out), nan=0.0, posinf=0.0, neginf=0.0)   # [N,nq] predicted quantiles

def p_up_from_quantiles(q):
    """P(up)=fraction of predicted quantile mass above 0, interpolated; abstain-conf via fan position of 0."""
    qa=np.array(QUANTS)
    # for each row, find where 0 sits among predicted quantiles -> implied CDF(0) -> P(up)=1-CDF(0)
    P=np.empty(len(q))
    for i in range(len(q)):
        v=q[i]; v=np.maximum.accumulate(v)     # enforce monotone
        if 0<=v[0]: cdf0=qa[0]*(0)/max(v[0],1e-12) if v[0]>0 else 0.0; cdf0=0.0
        elif 0>=v[-1]: cdf0=1.0
        else:
            j=np.searchsorted(v,0.0); j=min(max(j,1),len(v)-1)
            frac=(0.0-v[j-1])/max(v[j]-v[j-1],1e-12); cdf0=qa[j-1]+frac*(qa[j]-qa[j-1])
        P[i]=1.0-cdf0
    return np.clip(P,1e-4,1-1e-4)

# ----------------------------- run -----------------------------------------
def run_forecast_arm(arch,input_mode):
    cols=NB.XPAIR_COLS if input_mode=="xpair" else NB.SINGLE_COLS; C=len(cols); t0=time.time()
    Xtr,Rtr,ytr,_=NB.build_windows(NB.SPLITS["train"],cols,STRIDE_TR)
    if len(Xtr)>MAXTR:
        s=np.random.default_rng(SEED).choice(len(Xtr),MAXTR,replace=False); s.sort(); Xtr,Rtr,ytr=Xtr[s],Rtr[s],ytr[s]
    Xva,Rva,yva,_=NB.build_windows(NB.SPLITS["val"],cols,max(1,STRIDE_TR//2))
    yrs={y:NB.build_windows(NB.SPLITS[y],cols,1) for y in ["y2024","y2025","y2026"]}
    (Xtr_s,Xva_s,*ys),_=NB.standardize(Xtr,Xva,*[yrs[y][0] for y in ["y2024","y2025","y2026"]])
    net,rsc=train_mse(arch,Xtr_s,Rtr,Xva_s,Rva,C)
    cva=cumret(net,Xva_s,rsc); lr=LogisticRegression(max_iter=1000).fit(cva.reshape(-1,1),yva.astype(int))
    out={"arch":arch,"input":input_mode,"readout":"forecast_sign","H":H,"years":{},"n_train":int(len(Xtr))}
    pva=lr.predict_proba(cva.reshape(-1,1))[:,1]; out["val"]=NB.sel_metrics(pva,yva); out["val_dirAUC"]=out["val"]["auc"]
    for y,k in zip(["y2024","y2025","y2026"],["2024","2025","2026"]):
        Xs=ys[["y2024","y2025","y2026"].index(y)]; yy=yrs[y][2]
        if len(Xs)==0: out["years"][k]=None; continue
        cc=cumret(net,Xs,rsc); out["years"][k]=NB.sel_metrics(lr.predict_proba(cc.reshape(-1,1))[:,1],yy)
    _verdict(out); out["secs"]=round(time.time()-t0,1); return out

def run_quant_arm(input_mode):
    cols=NB.XPAIR_COLS if input_mode=="xpair" else NB.SINGLE_COLS; C=len(cols); t0=time.time()
    Xtr,Rtr,ytr,_=NB.build_windows(NB.SPLITS["train"],cols,STRIDE_TR)
    if len(Xtr)>MAXTR:
        s=np.random.default_rng(SEED).choice(len(Xtr),MAXTR,replace=False); s.sort(); Xtr,Rtr,ytr=Xtr[s],Rtr[s],ytr[s]
    Xva,Rva,yva,_=NB.build_windows(NB.SPLITS["val"],cols,max(1,STRIDE_TR//2))
    yrs={y:NB.build_windows(NB.SPLITS[y],cols,1) for y in ["y2024","y2025","y2026"]}
    (Xtr_s,Xva_s,*ys),_=NB.standardize(Xtr,Xva,*[yrs[y][0] for y in ["y2024","y2025","y2026"]])
    net,rsc=train_quant(Xtr_s,Rtr,Xva_s,Rva,C)
    out={"arch":"tft_quantile","input":input_mode,"readout":"quantile_fan","H":H,"years":{},"n_train":int(len(Xtr))}
    qva=quant_pred(net,Xva_s); pva=p_up_from_quantiles(qva); out["val"]=NB.sel_metrics(pva,yva); out["val_dirAUC"]=out["val"]["auc"]
    for y,k in zip(["y2024","y2025","y2026"],["2024","2025","2026"]):
        Xs=ys[["y2024","y2025","y2026"].index(y)]; yy=yrs[y][2]
        if len(Xs)==0: out["years"][k]=None; continue
        out["years"][k]=NB.sel_metrics(p_up_from_quantiles(quant_pred(net,Xs)),yy)
    _verdict(out); out["secs"]=round(time.time()-t0,1); return out

def _verdict(out):
    yrs=[out["years"][k] for k in ["2024","2025","2026"] if out["years"][k]]
    val_ok=(out["val_dirAUC"] is not None) and (out["val_dirAUC"]>0.515)
    any_clear=any((r["ci_lo"]==r["ci_lo"]) and r["ci_lo"]>=BE for r in yrs)
    out["KILL"]=not(val_ok and any_clear); out["verdict"]="KILLED" if out["KILL"] else "SURVIVED-escalate-to-CPCV"

if __name__=="__main__":
    print(f"[decomp_dir] H={H} L={L} dev={DEV} epochs={EPOCHS}",flush=True)
    res={"key":f"EURUSD.{H}m.direction","lever":"CORPUS_LEVER_INVENTORY.md:333,359 DLinear/Autoformer/FEDformer/TFT-quantile",
         "falsifier":"KILL if VAL dirAUC<=0.515 OR no held-out year selacc CI95-lo>=0.541","arms":[]}
    def _show(r):
        print(f"  {r['arch']:12s} {r['input']:6s} valAUC={r['val_dirAUC']:.4f} "
              f"2024={r['years']['2024']['acc'] if r['years']['2024'] else None} "
              f"2025={r['years']['2025']['acc'] if r['years']['2025'] else None} "
              f"2026={r['years']['2026']['acc'] if r['years']['2026'] else None} -> {r['verdict']} ({r['secs']}s)",flush=True)
    jobs=[("forecast",a,i) for a in ["dlinear","autoformer","freq"] for i in ["single","xpair"]] + \
         [("quant",None,i) for i in ["single","xpair"]]
    for kind,arch,inp in jobs:
        try:
            r=run_forecast_arm(arch,inp) if kind=="forecast" else run_quant_arm(inp)
            res["arms"].append(r); _show(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            res["arms"].append({"arch":arch or "tft_quantile","input":inp,"H":H,"error":f"{type(e).__name__}: {e}","KILL":True,"verdict":"ERROR"})
        finally:
            if DEV=="cuda": torch.cuda.empty_cache()
    res["any_survived"]=any(not a["KILL"] for a in res["arms"])
    op=f"{ROOT}/decomp_dir_{H}m_result.json"; json.dump(res,open(op,"w"),indent=1)
    print(f"[done] {op} any_survived={res['any_survived']}",flush=True)
