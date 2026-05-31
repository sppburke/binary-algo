"""WALK-FORWARD retraining test: is the persistent test25 weakness / corr(VAL,OOS)=-0.54 a TRAIN-TEST-GAP artifact?
All prior models train 2012-2021 and test 2024/25/26 (3-4yr gap). Walk-forward retrains annually with an EXPANDING window so
each test year is predicted by a model trained through the PRIOR year (1yr gap). Leakage-free (year Y model never sees >=Y).
If test25 jumps from ~0.55 to 0.62+, gap-reduction is a real lever -> extend to the full cross-horizon stack.

Cross-pair+base+OF primary, NY × confidence selective (cov frozen on the val year), non-overlap 300s, CI95.
Run with default HOR=5.
"""
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H, m5_xpair as MX

# (test_year, train_years, val_year)
FOLDS=[("2024",[str(y) for y in range(2012,2023)],"2023"),
       ("2025",[str(y) for y in range(2012,2024)],"2024"),
       ("2026",[str(y) for y in range(2012,2025)],"2025")]
OF_COLS=MX.OF_COLS

def build(years,stride=1):
    F=MX.build_xp(years,stride); F=MX.augment(F,years,"xpof"); return F

def fold(test_y,tr_years,va_year,stride=4):
    xtmp=MX.build_xp([va_year]); xpc=MX.xp_cols(xtmp)
    TR=build(tr_years,stride); VA=build([va_year]); TE=build([test_y])
    cols=MX.feat_cols("xpof",TR,xpc)
    cols=[c for c in cols if c in TR.columns and c in VA.columns and c in TE.columns]
    ytr=TR["_y"].astype(int).values; yva=VA["_y"].astype(int).values; yte=TE["_y"].astype(int).values
    L=lgb.LGBMClassifier(objective="binary",metric="auc",learning_rate=0.02,num_leaves=127,min_child_samples=400,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=20,n_estimators=2500,n_jobs=20,verbosity=-1)
    L.fit(TR[cols].astype("float32"),ytr,eval_set=[(VA[cols].astype("float32"),yva)],eval_metric="auc",
          callbacks=[lgb.early_stopping(150),lgb.log_evaluation(0)])
    pva=L.predict_proba(VA[cols].astype("float32"))[:,1]; pte=L.predict_proba(TE[cols].astype("float32"))[:,1]
    # freeze NY cov2% conf threshold on the val year
    gv=VA["sess_ny"].values>0.5; confv=np.abs(pva-0.5)
    thr=float(np.quantile(confv[gv],1-0.02))
    # apply to test
    gt=TE["sess_ny"].values>0.5; tst=TE["_ts"].values.astype("int64")
    m=gt&(np.abs(pte-0.5)>=thr); sel=MX.nonoverlap_chrono(tst,m,300)
    corr=((pte[sel]>0.5).astype(int)==yte[sel]).astype(float) if len(sel) else np.array([])
    acc=corr.mean() if len(sel) else float("nan"); lo,hi=MX.boot(corr)
    auc=roc_auc_score(yte,pte)
    return test_y,len(TR),auc,len(sel),acc,lo,hi,corr

def main():
    print("[walkforward] expanding-window annual retrain (1yr gap) vs the frozen-2021 baseline (3-4yr gap)")
    print(f"  baseline frozen-2021 (cross-pair+OF, NY cov2%): test24 0.641 / test25 0.546 / oos 0.554")
    allc=[]
    for ty,tr,va in FOLDS:
        ty_,ntr,auc,nsel,acc,lo,hi,corr=fold(ty,tr,va)
        print(f"=== test {ty} (train {tr[0]}-{tr[-1]} n={ntr:,}, val {va}) === AUC={auc:.4f} n={nsel} acc={acc:.3f} CI95=[{lo:.3f},{hi:.3f}]",flush=True)
        allc.append(corr)
    A=np.concatenate(allc); lo,hi=MX.boot(A)
    print(f"=== WALK-FORWARD COMBINED === n={len(A)} acc={A.mean():.3f} CI95=[{lo:.3f},{hi:.3f}] (breakeven~0.541)",flush=True)

if __name__=="__main__": main()
