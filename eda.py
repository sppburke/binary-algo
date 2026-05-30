"""EDA: locate conditional pockets where P(5m-up) departs from 0.5.
Uses only TRAIN+VAL years (2012-2023) to avoid peeking at TEST/OOS."""
import glob, os
import numpy as np, pandas as pd

FEAT_DIR = "/media/sean/CORSAIR/binary-algo/features"
YEARS = [str(y) for y in range(2012, 2024)]
cols = ["y","fwd_ret","valid","close","hour_sin","hour_cos",
        "1m_ret_1","5m_ret_1","15m_ret_1","1h_ret_1",
        "5m_autocorr_10","15m_autocorr_10","1h_autocorr_10",
        "mtf_trend_align","5m_rv_24","15m_rv_24","sess_overlap","dow"]
parts=[]
for y in YEARS:
    p=f"{FEAT_DIR}/EURUSD_{y}.parquet"
    if os.path.exists(p): parts.append(pd.read_parquet(p, columns=cols))
df=pd.concat(parts); df=df[df.valid==True].copy()
df["up"]=df.y.astype(int)
# recover hour from sin/cos
df["hour"]=(np.round(np.arctan2(df.hour_sin,df.hour_cos)/(2*np.pi)*24)%24).astype(int)
N=len(df); base=df.up.mean()
print(f"N={N:,}  base P(up)={base:.4f}")

def slc(name, mask):
    m=df[mask]
    if len(m)<2000: return
    print(f"  {name:38s} n={len(m):>9,} P(up)={m.up.mean():.4f}  dev={m.up.mean()-0.5:+.4f}")

print("\n== P(up) by hour (UTC) ==")
g=df.groupby("hour").up.agg(["mean","count"])
for h,r in g.iterrows():
    print(f"  h={h:02d} P(up)={r['mean']:.4f} dev={r['mean']-0.5:+.4f} n={int(r['count']):,}")

print("\n== Momentum vs reversion: P(up | last-Xm return sign) ==")
for col in ["1m_ret_1","5m_ret_1","15m_ret_1","1h_ret_1"]:
    up_after_up = df[df[col]>0].up.mean()
    up_after_dn = df[df[col]<0].up.mean()
    # continuation accuracy = P(move continues): up after up, down after down
    cont = ((df[col]>0)&(df.up==1)).sum()+((df[col]<0)&(df.up==0)).sum()
    tot = ((df[col]>0)|(df[col]<0)).sum()
    print(f"  {col}: P(up|prev_up)={up_after_up:.4f} P(up|prev_dn)={up_after_dn:.4f} "
          f"continuation_acc={cont/tot:.4f}")

print("\n== Strong-move continuation: P(continue | |5m_ret| in top decile) ==")
q=df["5m_ret_1"].abs().quantile([0.5,0.9,0.95,0.99]).to_dict()
for thr_name,thr in q.items():
    sub=df[df["5m_ret_1"].abs()>=thr]
    cont=((sub["5m_ret_1"]>0)&(sub.up==1)).sum()+((sub["5m_ret_1"]<0)&(sub.up==0)).sum()
    print(f"  |5m_ret|>=q{thr_name} ({thr:.5f}): n={len(sub):,} continuation_acc={cont/len(sub):.4f}")

print("\n== Autocorr regime x momentum ==")
for ac in ["5m_autocorr_10","15m_autocorr_10"]:
    for sgn,lab in [(1,"trending(ac>0.1)"),(-1,"revert(ac<-0.1)")]:
        if sgn>0: reg=df[df[ac]>0.1]
        else: reg=df[df[ac]<-0.1]
        if len(reg)<2000: continue
        # in this regime, does continuation of 5m move predict?
        cont=((reg["5m_ret_1"]>0)&(reg.up==1)).sum()+((reg["5m_ret_1"]<0)&(reg.up==0)).sum()
        print(f"  {ac} {lab}: n={len(reg):,} continuation_acc={cont/len(reg):.4f}")

print("\n== MTF trend alignment extremes ==")
slc("all up-aligned (align>=0.95)", df.mtf_trend_align>=0.95)
slc("all dn-aligned (align<=0.05)", df.mtf_trend_align<=0.05)

print("\n== Best single-slice hunt: hour x momentum-sign ==")
best=[]
for h in range(24):
    for col in ["5m_ret_1","15m_ret_1"]:
        for sgn in [1,-1]:
            mask=(df.hour==h)&((df[col]>0) if sgn>0 else (df[col]<0))
            m=df[mask]
            if len(m)<3000: continue
            # predict continuation
            pred_up = 1 if sgn>0 else 0
            acc=(m.up==pred_up).mean()
            best.append((max(acc,1-acc),h,col,sgn,len(m),acc))
best.sort(reverse=True)
print("  top 12 hour x momentum continuation/reversion edges:")
for sc,h,col,sgn,n,acc in best[:12]:
    print(f"   h={h:02d} {col} sgn={sgn:+d} n={n:,} pred-cont-acc={acc:.4f} best-dir-acc={sc:.4f}")
