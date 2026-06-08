"""SCOPE: EURUSD (key-specific). LEVER FAMILY — CORPUS_LEVER_INVENTORY.md:356-357 (UNTESTED, prior 0.03):
from-scratch N-BEATS / N-HiTS torch path-forecaster -> SIGN, as a DIRECTION model at MX_HOR minutes,
SINGLE-PAIR (EURUSD returns only) AND CROSS-PAIR (7-pair USD panel) input. GPU + AMP.

Mechanism: forecast the H-step forward return path with a doubly-residual basis-expansion stack
(N-BEATS: generic FC basis; N-HiTS: multi-rate MaxPool input pooling + hierarchical interpolation),
then direction = sign(sum of predicted forward returns). A causal logistic (fit on VAL only) maps the
predicted cumulative return -> P(up) for the deriv-faithful selective-accuracy gate.

DISCIPLINE (strategy-eval §2): harness splits TRAIN 2012-21 | VAL 22-23 | held-out per-year 2024,2025,2026
(2026 strict OOS). Label = sign(close(t+H)-close(t)) on the panel's own 1-min wall clock with strict
contiguity (lookback AND [t,t+H] gap-free); exact ties DROPPED; moved-bars only -> up-rate must be in
[0.47,0.53]. Metric = deriv-faithful selective accuracy among top-COV |p-0.5| (fwd_holdout._sel_acc) +
plain AUC, bootstrap CI95. Breakeven 0.541.

PRE-REGISTERED FALSIFIER (write before reading OOS): KILL the (arch,input) arm if
  VAL dirAUC <= 0.515  OR  no held-out year's selective-acc CI95-lower clears 0.541.
Sign-invariance prior (THEORY): a price/return PATH forecaster carries SIZE not SIGN -> expect ~0.50.

Usage:  MX_HOR=5 ~/binary-algo-venv/bin/python nbeats_nhits_dir.py     (writes nbeats_nhits_dir_5m_result.json)
"""
import os, sys, json, time, glob
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

H        = int(os.environ.get("MX_HOR", "5"))     # forecast/label horizon in minutes
L        = int(os.environ.get("NB_L", "64"))      # lookback window (1-min bars)
COV      = float(os.environ.get("COV", "0.10"))
BE       = 0.541
EPOCHS   = int(os.environ.get("NB_EPOCHS", "12"))
STRIDE_TR= int(os.environ.get("NB_STRIDE", str(max(1, H))))  # decorrelate overlapping train windows
MAXTR    = int(os.environ.get("NB_MAXTR", "150000"))
SEED     = int(os.environ.get("NB_SEED", "0"))
DEV      = "cuda" if torch.cuda.is_available() else "cpu"
ROOT     = "/home/sean/git/binary-algo"
torch.manual_seed(SEED); np.random.seed(SEED)

SPLITS = {"train": list(range(2012, 2022)), "val": [2022, 2023], "y2024": [2024], "y2025": [2025], "y2026": [2026]}
XPAIR_COLS = ["r_EURUSD","r_GBPUSD","r_AUDUSD","r_NZDUSD","r_USDJPY","r_USDCHF","r_USDCAD",
              "fac","e_EURUSD","e_GBPUSD","e_AUDUSD","e_NZDUSD","e_USDJPY","e_USDCHF","e_USDCAD"]
SINGLE_COLS = ["r_EURUSD"]

# ----------------------------- data ----------------------------------------
def _load_year(y):
    p = f"{ROOT}/features/panel_{y}.parquet"
    if not os.path.exists(p): return None
    d = pd.read_parquet(p)
    return d

def build_windows(years, cols, stride):
    """Causal lookback windows ending at decision bar i; target = H forward 1-min returns; label = sign(c[i+H]-c[i]).
    Strict wall-clock contiguity over [i-L+1, i+H] (all 60s steps). Ties dropped. Returns X[N,L,C], R[N,H], yup[N], ts[N]."""
    Xs, Rs, Ys, Ts = [], [], [], []
    for y in years:
        d = _load_year(y)
        if d is None: continue
        t = d["t"].values.astype(np.int64)
        c = d["c_eur"].values.astype(np.float64)
        chan = d[cols].values.astype(np.float32)
        chan = np.nan_to_num(chan, nan=0.0, posinf=0.0, neginf=0.0)
        n = len(c)
        # future cumulative path: r_fwd[k] for k=1..H  (log returns)
        logc = np.log(c)
        idxs = np.arange(L-1, n-H, stride)
        for i in idxs:
            if t[i] - t[i-L+1] != (L-1)*60:      continue   # contiguous lookback
            if t[i+H] - t[i] != H*60:            continue   # contiguous forward window
            if c[i+H] == c[i]:                   continue   # tie -> drop (deriv: ties lose, excluded from sign metric)
            Xs.append(chan[i-L+1:i+1])
            Rs.append(np.diff(logc[i:i+H+1]).astype(np.float32))   # H forward 1-min log-returns
            Ys.append(1.0 if c[i+H] > c[i] else 0.0)
            Ts.append(t[i])
    if not Xs:
        return (np.zeros((0,L,len(cols)),np.float32), np.zeros((0,H),np.float32),
                np.zeros((0,),np.float32), np.zeros((0,),np.int64))
    return (np.asarray(Xs,np.float32), np.asarray(Rs,np.float32),
            np.asarray(Ys,np.float32), np.asarray(Ts,np.int64))

# ----------------------------- models --------------------------------------
class NBeatsBlock(nn.Module):
    def __init__(self, in_dim, H, width=256):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim,width), nn.ReLU(), nn.Linear(width,width), nn.ReLU(),
                                nn.Linear(width,width), nn.ReLU())
        self.back = nn.Linear(width, in_dim)   # generic basis backcast
        self.fore = nn.Linear(width, H)        # generic basis forecast
    def forward(self, x):
        h = self.fc(x); return self.back(h), self.fore(h)

class NBeats(nn.Module):
    """Generic doubly-residual stack (arXiv:1905.10437). Flattened L*C input."""
    def __init__(self, L, C, H, nblocks=3, width=256):
        super().__init__()
        self.in_dim = L*C; self.H = H
        self.blocks = nn.ModuleList([NBeatsBlock(self.in_dim, H, width) for _ in range(nblocks)])
    def forward(self, x):                       # x [B,L,C]
        r = x.reshape(x.shape[0], -1); fsum = 0.0
        for b in self.blocks:
            bc, fc = b(r); r = r - bc; fsum = fsum + fc
        return fsum                              # [B,H] forecast returns

class NHiTSBlock(nn.Module):
    def __init__(self, L, C, H, pool_k, n_freq, width=256):
        super().__init__()
        self.L, self.C, self.H, self.pool_k = L, C, H, pool_k
        self.n_freq = max(1, n_freq)
        self.pool = nn.MaxPool1d(pool_k, ceil_mode=True)
        Lp = -(-L // pool_k)                      # ceil
        self.in_dim = Lp * C
        self.fc = nn.Sequential(nn.Linear(self.in_dim,width), nn.ReLU(), nn.Linear(width,width), nn.ReLU())
        self.back = nn.Linear(width, L*C)
        self.fore = nn.Linear(width, self.n_freq)  # coarse forecast -> interpolate to H
    def forward(self, x):                          # x [B,L,C]
        B = x.shape[0]
        xp = self.pool(x.transpose(1,2)).transpose(1,2).reshape(B,-1)   # multi-rate pooled
        h = self.fc(xp)
        bc = self.back(h).reshape(B, self.L, self.C)
        coarse = self.fore(h)                                          # [B,n_freq]
        fore = torch.nn.functional.interpolate(coarse.unsqueeze(1), size=self.H, mode="linear",
                                               align_corners=False).squeeze(1)  # hierarchical interpolation -> H
        return bc, fore

class NHiTS(nn.Module):
    """Multi-rate MaxPool input pyramid + hierarchical interpolation (arXiv:2201.12886)."""
    def __init__(self, L, C, H, pools=(8,4,1), freqs=None, width=256):
        super().__init__()
        if freqs is None: freqs = [max(1,H//8), max(1,H//2), H]
        self.blocks = nn.ModuleList([NHiTSBlock(L,C,H,pk,fr,width) for pk,fr in zip(pools,freqs)])
    def forward(self, x):
        r = x; fsum = 0.0
        for b in self.blocks:
            bc, fc = b(r); r = r - bc; fsum = fsum + fc
        return fsum

def make_model(arch, L, C, H):
    return NBeats(L,C,H) if arch=="nbeats" else NHiTS(L,C,H)

# ----------------------------- train / eval --------------------------------
def standardize(Xtr, *others):
    mu = Xtr.reshape(-1, Xtr.shape[2]).mean(0); sd = Xtr.reshape(-1, Xtr.shape[2]).std(0) + 1e-8
    out = [(Xtr-mu)/sd] + [((A-mu)/sd if len(A) else A) for A in others]
    return out, (mu, sd)

def train_forecaster(arch, Xtr, Rtr, Xva, Rva, C):
    torch.manual_seed(SEED)
    net = make_model(arch, L, C, H).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(DEV) if DEV=="cuda" else None
    # target scaling (returns are ~1e-4) -> scale up for stable MSE
    rsc = 1.0/ (Rtr.std() + 1e-8)
    xtr = torch.tensor(Xtr); rtr = torch.tensor(Rtr*rsc)
    xva = torch.tensor(Xva); rva = torch.tensor(Rva*rsc)   # keep on CPU; batch to GPU in eval (avoids full-batch OOM)
    n = len(xtr); bs = 4096; best=1e9; best_state=None; patience=3; bad=0
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(n)
        for j in range(0, n, bs):
            idx = perm[j:j+bs]; xb = xtr[idx].to(DEV); rb = rtr[idx].to(DEV)
            opt.zero_grad()
            if scaler is not None:
                with torch.amp.autocast(DEV):
                    pred = net(xb); loss = ((pred-rb)**2).mean()
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                scaler.step(opt); scaler.update()
            else:
                pred = net(xb); loss = ((pred-rb)**2).mean(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        net.eval()
        with torch.no_grad():
            se=0.0; cnt=0
            for jj in range(0,len(xva),8192):
                vb=xva[jj:jj+8192].to(DEV); rb=rva[jj:jj+8192].to(DEV)
                if DEV=="cuda":
                    with torch.amp.autocast(DEV): vp=net(vb)
                else: vp=net(vb)
                se+=((vp.float()-rb)**2).sum().item(); cnt+=rb.numel()
            vloss=se/max(cnt,1)
        if vloss < best-1e-6: best=vloss; best_state={k:v.detach().clone() for k,v in net.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=patience: break
    if best_state is not None: net.load_state_dict(best_state)
    net.eval()
    return net, rsc

def predict_cumret(net, X, rsc):
    out=[]
    with torch.no_grad():
        for j in range(0, len(X), 8192):
            xb = torch.tensor(X[j:j+8192]).to(DEV)
            if DEV=="cuda":
                with torch.amp.autocast(DEV): p = net(xb)
            else: p = net(xb)
            out.append(p.float().sum(1).cpu().numpy())   # predicted cumulative forward return
    return np.nan_to_num(np.concatenate(out)/(rsc if rsc else 1.0), nan=0.0, posinf=0.0, neginf=0.0)

def boot_ci(yt, correct_mask, B=2000):
    """CI95 of selected-subset hit-rate via bootstrap over the SELECTED bets."""
    if len(correct_mask)==0: return (float("nan"), float("nan"))
    rng = np.random.default_rng(SEED); n=len(correct_mask)
    acc = np.array([correct_mask[rng.integers(0,n,n)].mean() for _ in range(B)])
    return (float(np.quantile(acc,0.025)), float(np.quantile(acc,0.975)))

def sel_metrics(p, yup, cov=COV):
    """deriv-faithful selective accuracy among top-cov |p-0.5|; returns acc,n,ci_lo,ci_hi,uprate,auc."""
    yup=yup.astype(int); auc=float(roc_auc_score(yup,p)) if len(np.unique(yup))>1 else float("nan")
    conf=np.abs(p-0.5); thr=np.quantile(conf,1-cov); sel=conf>=thr
    if sel.sum()<25: return dict(acc=float("nan"),n=int(sel.sum()),ci_lo=float("nan"),ci_hi=float("nan"),
                                 uprate=float(yup.mean()),auc=auc)
    pred=(p[sel]>0.5).astype(int); correct=(pred==yup[sel]).astype(float)
    lo,hi=boot_ci(yup[sel],correct)
    return dict(acc=float(correct.mean()),n=int(sel.sum()),ci_lo=lo,ci_hi=hi,
                uprate=float(yup.mean()),auc=auc)

# ----------------------------- run ----------------------------------------
def run_arm(arch, input_mode):
    cols = XPAIR_COLS if input_mode=="xpair" else SINGLE_COLS
    t0=time.time()
    Xtr,Rtr,ytr,_ = build_windows(SPLITS["train"], cols, STRIDE_TR)
    if len(Xtr)>MAXTR:
        sel=np.random.default_rng(SEED).choice(len(Xtr),MAXTR,replace=False); sel.sort()
        Xtr,Rtr,ytr = Xtr[sel],Rtr[sel],ytr[sel]
    Xva,Rva,yva,_ = build_windows(SPLITS["val"], cols, max(1,STRIDE_TR//2))
    years = {y: build_windows(SPLITS[y], cols, 1) for y in ["y2024","y2025","y2026"]}
    (Xtr_s,Xva_s,*yrs_s),_ = standardize(Xtr, Xva, *[years[y][0] for y in ["y2024","y2025","y2026"]])
    C=len(cols)
    net,rsc = train_forecaster(arch, Xtr_s, Rtr, Xva_s, Rva, C)
    # readout: logistic(cumret_pred -> up) fit on VAL ONLY (causal)
    cva = predict_cumret(net, Xva_s, rsc)
    lr = LogisticRegression(max_iter=1000).fit(cva.reshape(-1,1), yva.astype(int))
    def P(Xs):
        cc=predict_cumret(net,Xs,rsc); return lr.predict_proba(cc.reshape(-1,1))[:,1], cc
    out={"arch":arch,"input":input_mode,"H":H,"L":L,"cov":COV,"breakeven":BE,
         "n_train":int(len(Xtr)),"val":{}, "years":{}, "forecast_sign":{}}
    pva,_=P(Xva_s); out["val"]=sel_metrics(pva,yva); out["val_dirAUC"]=out["val"]["auc"]
    for y,key in zip(["y2024","y2025","y2026"],["2024","2025","2026"]):
        Xs=yrs_s[["y2024","y2025","y2026"].index(y)]; yy=years[y][2]
        if len(Xs)==0: out["years"][key]=None; continue
        p,cc=P(Xs); out["years"][key]=sel_metrics(p,yy)
        # literal forecast-sign readout (no logistic): conf = |cumret_pred|
        psign=0.5+0.5*np.tanh(cc/(np.abs(cc).std()+1e-12))
        out["forecast_sign"][key]=sel_metrics(psign,yy)
    # pre-registered falsifier
    yrs=[out["years"][k] for k in ["2024","2025","2026"] if out["years"][k]]
    val_ok = (out["val_dirAUC"] is not None) and (out["val_dirAUC"]>0.515)
    any_clear = any((r["ci_lo"]==r["ci_lo"]) and r["ci_lo"]>=BE for r in yrs)  # CI95-lo >= breakeven
    out["KILL"] = not (val_ok and any_clear)
    out["verdict"] = "KILLED" if out["KILL"] else "SURVIVED-escalate-to-CPCV"
    out["secs"]=round(time.time()-t0,1)
    return out

if __name__=="__main__":
    print(f"[nbeats_nhits_dir] H={H} L={L} dev={DEV} epochs={EPOCHS} stride={STRIDE_TR}", flush=True)
    results={"key":f"EURUSD.{H}m.direction","lever":"CORPUS_LEVER_INVENTORY.md:356-357 N-BEATS/N-HiTS path->sign",
             "falsifier":"KILL if VAL dirAUC<=0.515 OR no held-out year selacc CI95-lo>=0.541","arms":[]}
    for arch in ["nbeats","nhits"]:
        for inp in ["single","xpair"]:
            try:
                r=run_arm(arch,inp)
                v=r["val_dirAUC"]; ys={k:(r["years"][k]["acc"] if r["years"][k] else None) for k in ["2024","2025","2026"]}
                print(f"  {arch:6s} {inp:6s} valAUC={v:.4f} selacc@{COV:.0%} "
                      f"2024={ys['2024']} 2025={ys['2025']} 2026={ys['2026']} -> {r['verdict']} ({r['secs']}s)",flush=True)
                results["arms"].append(r)
            except Exception as e:
                import traceback; traceback.print_exc()
                results["arms"].append({"arch":arch,"input":inp,"H":H,"error":f"{type(e).__name__}: {e}","KILL":True,"verdict":"ERROR"})
            finally:
                if DEV=="cuda": torch.cuda.empty_cache()
    results["any_survived"]=any(not a["KILL"] for a in results["arms"])
    op=f"{ROOT}/nbeats_nhits_dir_{H}m_result.json"
    json.dump(results, open(op,"w"), indent=1)
    print(f"[done] {op}  any_survived={results['any_survived']}", flush=True)
