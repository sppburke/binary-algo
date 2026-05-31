"""30m EURUSD — PATH SIGNATURE (Lévy area) + Hawkes-intensity DIRECTION features (last sign-aware orthogonal levers).

Rough-path theory: the level-2 signature's antisymmetric part = the LEVY AREA A(X,Y)=0.5∮(X dY - Y dX), the signed area
between two channels = lead-lag ROTATION (does order flow lead price?). Genuinely sign-aware and NOT captured by the 62
hand-crafted feats or the CNN. Channels: price-return, order-flow imbalance (imb), microprice-deviation. Computed
causally over trailing windows W, added to the tick feature set; LGB at HS seconds; honest selective.
Run HS=1800 (the 30m goal) and HS=5 (where microstructure signal exists) to see if signatures add anything.

  python m30_sig.py 1800
"""
import sys, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from min1_production import feats, wc_ret, nonoverlap_chrono, boot, load_split, set_pair
set_pair("EURUSD")
HS=int(sys.argv[1]) if len(sys.argv)>1 else 1800
TOL=max(2,HS//30+1); LAG=1

def levy_area(dx, dy, W):
    """Causal trailing-window Lévy area A_t = 0.5∮(X dY - Y dX) over [t-W+1, t], X=cumsum(dx),Y=cumsum(dy)."""
    P=np.cumsum(dx); Q=np.cumsum(dy)
    U=np.cumsum(P*dy - Q*dx)
    n=len(dx); Pm=np.concatenate([np.zeros(W),P[:-W]]); Qm=np.concatenate([np.zeros(W),Q[:-W]]); Um=np.concatenate([np.zeros(W),U[:-W]])
    A=0.5*((U-Um) - Pm*(Q-Qm) + Qm*(P-Pm))
    A[:W]=np.nan; return A

def sig_feats(b):
    mid=b["mid"].values.astype(float); imb=b["imb"].fillna(0).values.astype(float); micro=b["micro"].values.astype(float)
    dp=np.zeros(len(mid)); dp[1:]=np.diff(np.log(mid))          # price return increments
    dq=imb                                                       # order-flow imbalance as the 2nd channel increment
    dm=np.zeros(len(mid)); dm[1:]=np.diff((micro-mid)/mid)       # micro-dev increments
    X=pd.DataFrame(index=b.index)
    for W in (30,60,120,300):
        X[f"levy_pq{W}"]=levy_area(dp,dq,W)        # price vs order-flow rotation (sign-aware lead-lag)
        X[f"levy_pm{W}"]=levy_area(dp,dm,W)        # price vs microprice
        X[f"levy_qm{W}"]=levy_area(dq,dm,W)
        X[f"sig_dp{W}"]=pd.Series(dp,index=b.index).rolling(W).sum()   # level-1 increment
        X[f"sig_dq{W}"]=pd.Series(dq,index=b.index).rolling(W).sum()
    # normalize Levy areas by window vol scale
    for W in (30,60,120,300):
        sc=pd.Series(dp,index=b.index).rolling(W).std()*np.sqrt(W)+1e-12
        for ch in ("pq","pm"): X[f"levy_{ch}{W}"]=X[f"levy_{ch}{W}"]/sc
    return X.replace([np.inf,-np.inf],np.nan).astype("float32")

def prep(sp):
    b=load_split(sp); X=pd.concat([feats(b), sig_feats(b)], axis=1)
    mid=b["mid"].values.astype(float); ts=b.index.values.astype("datetime64[s]").astype("int64"); hour=b.index.hour.values
    ret,valid=wc_ret(ts,mid,HS,TOL,LAG); y=(ret>0).astype(int)
    return X,y,valid,ts,hour

def main():
    t0=time.time()
    Xtr,ytr,vtr,tstr,htr=prep("train"); itr=np.where(vtr)[0][::15]
    Xva,yva,vva,tsva,hva=prep("val"); iva=np.where(vva)[0]
    feat=list(Xtr.columns); nsig=sum(1 for c in feat if c.startswith(("levy","sig")))
    print(f"[sig HS={HS}] train_fit={len(itr):,} feats={len(feat)} (signature={nsig}) {time.time()-t0:.0f}s",flush=True)
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.03,num_leaves=300,min_child_samples=200,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=8,n_estimators=2500,n_jobs=20,verbosity=-1)
    L.fit(Xtr.iloc[itr],ytr[itr],eval_set=[(Xva.iloc[iva],yva[iva])],eval_metric="auc",callbacks=[lgb.early_stopping(120),lgb.log_evaluation(0)])
    imp=pd.Series(L.feature_importances_,index=feat).sort_values(ascending=False)
    nsig_top=sum(1 for c in imp.head(20).index if c.startswith(("levy","sig")))
    print(f"[sig HS={HS}] lgb iter={L.best_iteration_}; top-20 importances incl {nsig_top} signature feats. top sig: "+", ".join([c for c in imp.index if c.startswith(('levy','sig'))][:5]),flush=True)
    del Xtr
    gap=HS+TOL
    def ny(h): return (h>=12)&(h<21)
    def ind(p,y,ts,g,thr):
        m=g&(np.abs(p-0.5)>=thr)
        if m.sum()==0: return (0,np.nan,np.nan,np.nan)
        s=nonoverlap_chrono(ts,m,gap)
        if len(s)==0: return (0,np.nan,np.nan,np.nan)
        corr=((p[s]>0.5).astype(int)==y[s]).astype(float); return (len(s),corr.mean(),*boot(corr))
    pva=L.predict_proba(Xva)[:,1]
    print(f"[sig HS={HS}] AUC val={roc_auc_score(yva[vva],pva[vva]):.4f}",flush=True); del Xva
    for sp,lab in (("test","TEST"),("oos","OOS")):
        X,y,v,ts,hour=prep(sp); p=L.predict_proba(X)[:,1]
        print(f"  {lab} AUC={roc_auc_score(y[v],p[v]):.4f}",flush=True)
        for cov in (0.02,0.01,0.005):
            thr=float(np.quantile(np.abs(pva[vva]-0.5),1-cov))
            n,a,lo,hi=ind(p[v],y[v],ts[v],np.ones(int(v.sum()),bool),thr)
            ng,ag,log_,hig=ind(p[v],y[v],ts[v],ny(hour[v]),thr)
            print(f"    cov{cov:.1%} none:n{n} {a:.3f}[{lo:.2f},{hi:.2f}]  ny:n{ng} {ag:.3f}[{log_:.2f},{hig:.2f}]",flush=True)
        del X
    print(f"[sig HS={HS}] DONE {time.time()-t0:.0f}s (base tick HS={HS} AUC was ~0.51@30m / ~0.52@5s)",flush=True)

if __name__=="__main__":
    main()
