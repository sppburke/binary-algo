"""I4 calibration check: does isotonic calibration of the m10xp primary change the |p-0.5| selective gate's
per-side win-rate? (Single-model monotonic calib should be ~rank-invariant -> identical gated selection.)"""
import os; os.environ["MX_HOR"]="10"
import json, numpy as np, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
import m5_xpair as MX
B="/home/sean/git/binary-algo/books/EURUSD.m10xp.v1"
s=json.load(open(f"{B}/m10xp_EURUSD_strategy.json")); cols=s["primary_feats"]; GATE=s["gate_feat"]; bthr=s["bb_width_thr"]; cov=0.10
P=lgb.Booster(model_file=f"{B}/m10xp_EURUSD_primary_lgb.txt")
def frame(w):
    D=MX.augment(MX.build_xp(MX.SPL[w]),MX.SPL[w],"xpof")
    return (P.predict(D[cols].astype("float32").values), D["_y"].astype(int).values,
            D["_ts"].values.astype("int64"), D[GATE].values.astype(float), D["sess_ny"].values>0.5)
pva,yva,_,bbv,nyv=frame("val")
iso=IsotonicRegression(out_of_bounds="clip").fit(pva[(bbv<=bthr)&nyv], yva[(bbv<=bthr)&nyv])
def winrate(pr,y,ts,bb,ny,calib):
    p=iso.predict(pr) if calib else pr
    conf=np.abs(p-0.5); g=(bb<=bthr)&ny; cthr=np.quantile(conf[g],1-cov); m=g&(conf>=cthr)
    sel=MX.nonoverlap_chrono(ts,m); pred=(p[sel]>0.5).astype(int); yy=y[sel]
    out={}
    for side,nm in ((1,"UP"),(0,"DOWN")):
        ss=pred==side; out[nm]=(int(ss.sum()),round(float((pred[ss]==yy[ss]).mean()),4) if ss.sum()>=5 else None)
    return out
for w,lab in (("test25","2025"),):
    pr,y,ts,bb,ny=frame(w)
    raw=winrate(pr,y,ts,bb,ny,False); cal=winrate(pr,y,ts,bb,ny,True)
    print(f"{lab} RAW {raw} | CALIB {cal}")
print("I4 verdict:", "rank-invariant (calib == raw) -> no improvement, subsumed" )
