"""USDCAD 15m NY — D1 (fork of usdjpy_15m_gru.py). sequence DL (GRU on raw return path) — coverage-rule RUN of the DL class.

SCOPE: USDJPY · 15m · NY. RUN (not argue) one DL/sequence model at this key. Prior is strong (EURUSD
neural+spectral sweep KILLED 84/84 arms all tf incl 15m; info-bound) but the coverage rule says confirm here.
Input = trailing W one-minute log-returns before each NY 15m decision bar; small GRU -> P(up next 15m).
Train 2012-21 NY moved bars (subsampled), eval 2024/25/26 NY moved-AUC + cov3% win-rate. EVAL deriv-faithful.

Falsifier: VAL moved-AUC <= base .539 (GRU adds no sign over the GBM-on-TA-features) => DL class KILLED here.
Usage: ~/binary-algo-venv/bin/python usdjpy_15m_gru.py [W=32] [epochs=8]
"""
import os, sys, json, time, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import harness as H
from sessions import session_mask
from usdcad_15m_base import side_eval, boot, BE, SPL
W=int(sys.argv[1]) if len(sys.argv)>1 else 32
EPOCHS=int(sys.argv[2]) if len(sys.argv)>2 else 8
PAIR="USDCAD"; HOR=15; GAP=900; FEAT=H.FEAT_DIR; RESULT="usdcad_15m_gru_result.json"
SUB=120_000

def build_seq(years):
    """trailing W 1-min log-returns ending at decision bar t (causal), label=sign(close[t+15]-close[t]),
    NY session mask, moved, ts. Returns S(float32 [n,W,1]), y, moved, ts, ny."""
    Ss=[];ys=[];mv=[];tss=[];nys=[]
    for y in years:
        p=f"{FEAT}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p, columns=["close"]); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float); ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        r=np.zeros(n); r[1:]=np.log(c[1:]/c[:-1])
        contig=np.zeros(n,bool); contig[:n-HOR]=(ts[HOR:]-ts[:-HOR])==GAP
        fr=np.full(n,np.nan); fr[:n-HOR]=c[HOR:]/c[:-HOR]-1.0
        valid=contig & np.isfinite(fr); valid[:W]=False
        idx=np.where(valid)[0]
        if len(idx)==0: continue
        seq=np.stack([r[i-W+1:i+1] for i in idx]).astype("float32")*1e3   # scale returns
        Ss.append(seq[:,:,None]); ys.append((fr[idx]>0).astype(int)); mv.append(fr[idx]!=0.0)
        tss.append(ts[idx]); nys.append(session_mask(ts[idx],"ny"))
    return np.concatenate(Ss),np.concatenate(ys),np.concatenate(mv),np.concatenate(tss),np.concatenate(nys)

def main():
    t0=time.time()
    try:
        import torch, torch.nn as nn
    except Exception as e:
        json.dump({"error":f"torch unavailable: {e}","verdict":"SKIPPED"},open(RESULT,"w"),indent=2)
        print(f"[gru] torch unavailable: {e} -> SKIPPED",flush=True); return
    torch.manual_seed(0)
    Str,ytr,mtr,tstr,nytr=build_seq(SPL["train"]); Sva,yva,mva,tsv,nyva=build_seq(SPL["val"])
    itr=np.where(mtr&nytr)[0]
    if len(itr)>SUB: itr=np.random.default_rng(0).choice(itr,SUB,replace=False)
    iva=np.where(mva&nyva)[0]
    print(f"[gru] W={W} train(NY-moved)={len(itr):,} val={len(iva):,} build={time.time()-t0:.0f}s",flush=True)
    dev="cuda" if torch.cuda.is_available() else "cpu"
    class G(nn.Module):
        def __init__(s): super().__init__(); s.g=nn.GRU(1,24,batch_first=True); s.f=nn.Linear(24,1)
        def forward(s,x): o,_=s.g(x); return s.f(o[:,-1,:]).squeeze(-1)
    m=G().to(dev); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4); loss=nn.BCEWithLogitsLoss()
    def predict_np(arr, bs=8192):
        out=[];
        with torch.no_grad():
            for b in range(0,len(arr),bs):
                xb=torch.tensor(arr[b:b+bs]).to(dev); out.append(torch.sigmoid(m(xb)).cpu().numpy()); del xb
        return np.concatenate(out) if out else np.array([])
    Xt=torch.tensor(Str[itr]).to(dev); yt=torch.tensor(ytr[itr].astype("float32")).to(dev)
    Xv_np=Sva[iva]; yv=yva[iva]
    bs=4096; best=(1e9,None)
    for ep in range(EPOCHS):
        m.train(); perm=torch.randperm(len(Xt))
        for b in range(0,len(Xt),bs):
            j=perm[b:b+bs]; opt.zero_grad(); l=loss(m(Xt[j]),yt[j]); l.backward(); opt.step()
        m.eval(); pv=predict_np(Xv_np)
        auc=roc_auc_score(yv,pv)
        if -auc<best[0]: best=(-auc,{k:v.detach().clone() for k,v in m.state_dict().items()})
        print(f"  ep{ep} VAL moved-AUC={auc:.4f} ({time.time()-t0:.0f}s)",flush=True)
    if best[1]: m.load_state_dict(best[1])
    m.eval(); pv=predict_np(Xv_np)
    val_auc=float(roc_auc_score(yv,pv)); thr=float(np.quantile(np.abs(pv-0.5),1-0.03))
    res={"key":"USDCAD.15m.ny","model":f"GRU(24) on trailing {W} 1-min returns","val_ny_auc":round(val_auc,4),
         "base_auc":0.5433,"falsifier":{"KILL_if":"VAL-AUC<=.5433 (no lift over GBM)"},"years":{}}
    print(f"[gru] VAL-AUC(NY)={val_auc:.4f} (base .5433)",flush=True)
    for w in ("test24","test25","oos"):
        Sw,yw,mw,tsw,nyw=build_seq(SPL[w]); wi=np.where(nyw)[0]
        pr=predict_np(Sw[wi])
        auc=float(roc_auc_score(yw[wi][mw[wi]],pr[mw[wi]]))
        g=side_eval(pr,yw[wi],mw[wi],tsw[wi],thr)
        res["years"][w]={"moved_auc":round(auc,4),"cov3":{k:[g[k]["n"],round(g[k]["wr"],4)] for k in ("COMBINED","UP","DOWN")} if g else None}
        print(f"  {w}: moved-AUC={auc:.4f} | cov3 COMB {res['years'][w]['cov3']['COMBINED'] if g else None}",flush=True)
    res["verdict"]={"beats_base":bool(val_auc>0.5433),"KILLED":bool(val_auc<=0.5433)}
    json.dump(res,open(RESULT,"w"),indent=2)
    print(f"\n[gru] VERDICT: {'KILLED (no lift)' if res['verdict']['KILLED'] else 'beats base — investigate'} val-AUC {val_auc:.4f} {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
