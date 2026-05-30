"""V6 — TabNet (explicitly cited by the user's reference). Final faithful method from the
Reddit author's ensemble (XGB+LGBM+CatBoost+TabNet). CPU, subsampled train for tractability.
Tests whether an attentive deep tabular model breaks the ~0.52 wall (it should not — the limit
is the data's information content, ρ₁≈−0.03)."""
import sys, time, numpy as np
from sklearn.metrics import roc_auc_score
import harness as H, dataset as D
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

STRIDE=int(sys.argv[1]) if len(sys.argv)>1 else 10
kw=dict(use_cross=True,use_of=True,use_peer_of=False)
t0=time.time()
Xtr,ytr,_=D.load_split("train",stride=STRIDE,**kw)
Xva,yva,_=D.load_split("val",stride=3,**kw)
Xte,yte,_=D.load_split("test",**kw)
Xoo,yoo,_=D.load_split("oos",**kw)
# TabNet needs no NaNs; median-impute from train
med=Xtr.median(numeric_only=True)
Xtr=Xtr.fillna(med).values.astype("float32"); Xva=Xva.fillna(med).values.astype("float32")
Xte=Xte.fillna(med).values.astype("float32"); Xoo=Xoo.fillna(med).values.astype("float32")
# standardize
mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6
Xtr=(Xtr-mu)/sd; Xva=(Xva-mu)/sd; Xte=(Xte-mu)/sd; Xoo=(Xoo-mu)/sd
print(f"shapes tr={Xtr.shape} va={Xva.shape} te={Xte.shape} oo={Xoo.shape} load={time.time()-t0:.0f}s",flush=True)

clf=TabNetClassifier(n_d=32,n_a=32,n_steps=4,gamma=1.5,n_independent=2,n_shared=2,
    seed=0,optimizer_params=dict(lr=2e-2),verbose=1)
t0=time.time()
clf.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric=["auc"],max_epochs=40,patience=8,
        batch_size=16384,virtual_batch_size=2048)
print(f"train={time.time()-t0:.0f}s",flush=True)
pv=clf.predict_proba(Xva)[:,1]; pt=clf.predict_proba(Xte)[:,1]; po=clf.predict_proba(Xoo)[:,1]
print(f"TABNET AUC val={roc_auc_score(yva,pv):.4f} test={roc_auc_score(yte,pt):.4f} oos={roc_auc_score(yoo,po):.4f}")
H.report("TEST",yte,pt); H.report("OOS ",yoo,po)
print("TABNET DONE")
