"""ADJUDICATION test: does RAW per-side bid-vol/ask-vol signed order flow (genuinely never modeled) carry
EURUSD 60s SIGN on MOVED bars, per-year, deriv-faithful? This is the single untried on-disk axis. If it lifts
2025 moved-bar sign above ~0.51 with stability, the exhaustion verdict is breakable.

Build from tick_data/raw/EURUSD: per 1s, signed flow = sum over ticks in sec of sign(dmid)*(bidvol+askvol)?? NO.
Use the cleanest informed-flow proxy: per-second net per-side volume pressure = (bidvol-askvol) aggregated, and
the up-tick/down-tick signed volume (classic tick-rule trade-sign * volume). Predict next-60s mid sign.

Deriv-faithful: mid-to-mid, entry = next 1s tick, exit = last tick <=+60s, ties LOSE, NON-OVERLAP 60s, per-year.
MOVED bars only (|fwd_ret|>0.5pip) = the real UP/DOWN question. Single-feature AND small-LGB. CI95 via boot.
"""
import glob, numpy as np, pandas as pd, time
from min1_production import boot, nonoverlap_chrono
import lightgbm as lgb

PIP=1e-4
def load_year_1s(year, months):
    """Aggregate raw EURUSD ticks to 1s: last mid, net per-side vol, tick-rule signed vol, ntick."""
    rows=[]
    for m in months:
        fs=sorted(glob.glob(f'/media/sean/CORSAIR/tick_data/raw/EURUSD/EURUSD_{year}-{m:02d}-*'))
        for f in fs:
            try: df=pd.read_parquet(f)
            except Exception: continue
            df['mid']=(df['ask'].values+df['bid'].values)*0.5
            df['ts']=df['timestamp_utc'].values.astype('int64')  # already epoch seconds (float)
            df['netps']=df['bid-vol'].values-df['ask-vol'].values
            df['totv']=df['bid-vol'].values+df['ask-vol'].values
            # tick rule: sign of mid change * total vol (signed trade volume proxy)
            dmid=np.sign(np.diff(df['mid'].values, prepend=df['mid'].values[0]))
            df['sgnv']=dmid*df['totv'].values
            g=df.groupby('ts')
            agg=pd.DataFrame({'mid':g['mid'].last(),'netps':g['netps'].sum(),
                              'sgnv':g['sgnv'].sum(),'totv':g['totv'].sum(),'nt':g.size()})
            rows.append(agg)
    if not rows: return None
    a=pd.concat(rows).groupby(level=0).agg({'mid':'last','netps':'sum','sgnv':'sum','totv':'sum','nt':'sum'})
    a=a.sort_index()
    return a

def feats60(a):
    ts=a.index.values.astype('int64'); mid=a['mid'].values.astype(float)
    # forward 60s mid-to-mid, deriv-faithful entry/exit
    entry_t=ts+1; ei=np.searchsorted(ts,entry_t,side='left')
    exit_t=entry_t+60; xi=np.searchsorted(ts,exit_t,side='right')-1
    n=len(ts); eic=np.clip(ei,0,n-1); xic=np.clip(xi,0,n-1)
    valid=(ei<n)&(xi>ei)&((ts[eic]-entry_t)<=3)&((exit_t-ts[xic])<=3)
    ret=np.where(valid, mid[xic]/np.where(mid[eic]==0,np.nan,mid[eic])-1.0, np.nan)
    # features: rolling signed flow over recent windows
    df=pd.DataFrame(index=a.index)
    for w in (5,15,30,60):
        df[f'netps{w}']=a['netps'].rolling(w,min_periods=1).sum().values
        df[f'sgnv{w}']=a['sgnv'].rolling(w,min_periods=1).sum().values
        df[f'nt{w}']=a['nt'].rolling(w,min_periods=1).sum().values
    df['flowimb']=(a['netps'].rolling(30,min_periods=1).sum()/ (a['totv'].rolling(30,min_periods=1).sum()+1)).values
    return df, ret, valid, ts

def run_year(tag, a):
    df,ret,valid,ts=feats60(a)
    yd=(ret>0).astype(float)
    moved=valid & (np.abs(ret)>0.5*PIP)
    print(f"  [{tag}] secs={len(a):,} valid={int(np.nansum(valid)):,} moved={int(np.nansum(moved)):,} up_rate_moved={np.nanmean(yd[moved]):.3f}",flush=True)
    return df,yd,ret,valid,moved,ts

def main():
    t0=time.time()
    # train on 2024 H1, test 2024 H2 / 2025 / 2026 (held out). Keep it light: a few months each.
    print("loading...",flush=True)
    a24=load_year_1s(2024,[1,2,3,9,10,11]); print(f"2024 {time.time()-t0:.0f}s",flush=True)
    a25=load_year_1s(2025,[2,3,9,10]); print(f"2025 {time.time()-t0:.0f}s",flush=True)
    a26=load_year_1s(2026,[2,3,4]); print(f"2026 {time.time()-t0:.0f}s",flush=True)
    D24,y24,r24,v24,m24,ts24=run_year("2024tr",a24)
    D25,y25,r25,v25,m25,ts25=run_year("2025",a25)
    D26,y26,r26,v26,m26,ts26=run_year("2026",a26)
    # split 2024 into train/val by time
    n=len(D24); cut=int(n*0.6)
    trm=m24.copy(); trm[cut:]=False
    vam=m24.copy(); vam[:cut]=False
    Xtr=D24.values[trm]; ytr=y24[trm]
    Xva=D24.values[vam]; yva=y24[vam]
    clf=lgb.LGBMClassifier(objective='binary',metric='auc',learning_rate=0.03,num_leaves=64,
        min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,reg_lambda=8,n_estimators=600,n_jobs=20,verbosity=-1)
    clf.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric='auc',callbacks=[lgb.early_stopping(80),lgb.log_evaluation(0)])
    from sklearn.metrics import roc_auc_score
    pva=clf.predict_proba(Xva)[:,1]
    print(f"\nVAL moved-bar dirAUC={roc_auc_score(yva,pva):.4f}",flush=True)
    confv=np.abs(pva-0.5)
    W={'2024H2':(clf.predict_proba(D24.values)[:,1],y24,m24.copy(),ts24),
       '2025':(clf.predict_proba(D25.values)[:,1],y25,m25,ts25),
       '2026':(clf.predict_proba(D26.values)[:,1],y26,m26,ts26)}
    # 2024H2 = test portion only
    W['2024H2']=(W['2024H2'][0],W['2024H2'][1],vam,ts24)
    print("\nMOVED-BAR DIRECTION (per-side raw OF), dir-conf selective:",flush=True)
    for cov in (1.0,0.5,0.25,0.1,0.05):
        dthr=float(np.quantile(confv,1-cov)); rows=[]
        for k in ('2024H2','2025','2026'):
            p_,y_,mm_,ts_=W[k]
            sel=mm_ & (np.abs(p_-0.5)>=dthr)
            if sel.sum()==0: rows.append((0,np.nan,np.nan,np.nan)); continue
            s=nonoverlap_chrono(ts_,sel,63)
            corr=((p_[s]>0.5).astype(int)==y_[s].astype(int)).astype(float)
            rows.append((len(s),corr.mean(),*boot(corr)))
        win=all(r[0]>=25 and r[2]>0.65 for r in rows)
        print(f"  cov{cov:>5}: "+" | ".join(f"{k} n{r[0]} {r[1]:.3f}[{r[2]:.2f},{r[3]:.2f}]" for k,r in zip(('24','25','26'),rows))+("  <<<WIN" if win else ""),flush=True)
    print(f"\nDONE {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__": main()
