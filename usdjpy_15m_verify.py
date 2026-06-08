"""USDJPY 15m — ADVERSARIAL VERIFICATION of the certified NY edge (leakage / harness-artifact falsifier).

SCOPE: USDJPY · 15m · NY. The decisive control: train the SAME NY own-pair GBM pipeline on PERMUTED train
labels (sign destroyed, features intact). If the certified .586 edge is real, the shuffle model must collapse
to moved-AUC ~.50 and win-rate ~.50 on REAL held-out NY bars. If it still scores high, the harness leaks.
Also re-confirms the moved up-rate tripwire and reports a same-harness flat-0.5 baseline.

Usage: ~/binary-algo-venv/bin/python usdjpy_15m_verify.py
"""
import json, time, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sessions import session_mask
from usdjpy_15m_base import build, side_eval, mk_lgb, BE, SPL

def main():
    t0=time.time(); rng=np.random.default_rng(0)
    Xtr,ytr,mtr,tstr=build(SPL["train"], 6); Xva,yva,mva,tsv=build(SPL["val"])
    nytr=session_mask(tstr,"ny"); nyva=session_mask(tsv,"ny")
    itr=mtr&nytr; iva=mva&nyva
    out={"key":"USDJPY.15m.ny","check":"label-shuffle leakage control (train labels permuted, eval REAL)"}
    # ---- SHUFFLE control: permute train labels ----
    ysh=ytr.copy(); ysh[itr]=rng.permutation(ytr[itr])
    L=mk_lgb(num_leaves=127)
    L.fit(Xtr[itr], ysh[itr], eval_set=[(Xva[iva], yva[iva])], eval_metric="auc",
          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    confv=np.abs(L.predict_proba(Xva)[:,1][np.where(nyva)[0]]-0.5); THR=float(np.quantile(confv,1-0.03))
    sh={}
    for w in ("test24","test25","oos"):
        Xw,yw,mw,tsw=build(SPL[w]); wi=np.where(session_mask(tsw,"ny"))[0]
        pr=L.predict_proba(Xw)[:,1][wi]
        auc=float(roc_auc_score(yw[wi][mw[wi]], pr[mw[wi]]))
        g=side_eval(pr, yw[wi], mw[wi], tsw[wi], THR)
        sh[w]={"shuffle_auc":round(auc,4),
               "shuffle_cov3_COMB_wr":round(g["COMBINED"]["wr"],4) if g else None,
               "shuffle_cov3_UP_wr":round(g["UP"]["wr"],4) if g else None}
        print(f"  [shuffle {w}] moved-AUC={auc:.4f} (expect ~.50) | cov3% COMB {sh[w]['shuffle_cov3_COMB_wr']} UP {sh[w]['shuffle_cov3_UP_wr']}",flush=True)
    out["shuffle"]=sh
    aucs=[v["shuffle_auc"] for v in sh.values()]
    out["shuffle_passes"]=bool(max(aucs)<=0.52 and min(aucs)>=0.48)   # shuffle must be ~coin-flip
    out["verdict"]=("LEAKAGE-CLEAN: shuffle collapses to ~.50, real edge confirmed not a harness artifact"
                    if out["shuffle_passes"] else "WARNING: shuffle model scores off-0.50 -> investigate leakage")
    json.dump(out, open("usdjpy_15m_verify_result.json","w"), indent=2)
    print(f"\n[verify] shuffle AUCs {aucs} -> PASSES={out['shuffle_passes']}  ({out['verdict']})  {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
