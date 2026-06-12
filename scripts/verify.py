"""Standalone verification harness.
Loads saved out-of-sample probabilities (or a model) and reports the honest prediction
rate with strict separation of TRAIN/VAL/TEST and a fully held-out 2026 OOS set.

A threshold is chosen ONLY on VAL; the SAME threshold's accuracy + coverage is then read
off TEST and 2026-OOS. This is the unbiased estimate of the live prediction rate.

Usage: python verify.py models/v4_probs.npz
"""
import sys, numpy as np
import harness as H

f=sys.argv[1] if len(sys.argv)>1 else "models/v4_probs.npz"
d=np.load(f)
# support both v3 (pva/pte/poo) and v4 (pv/pt/po) naming
pv=d["pv"] if "pv" in d else d["pva"]; pt=d["pt"] if "pt" in d else d["pte"]; po=d["po"] if "po" in d else d["poo"]
yva=d["yva"]; yte=d["yte"]; yoo=d["yoo"]

print(f"=== VERIFICATION: {f} ===")
print("Full-coverage (predict every bar):")
H.report("VAL  2022-23", yva, pv)
H.report("TEST 2024-25", yte, pt)
H.report("OOS  2026   ", yoo, po)

print("\n=== Selective prediction rate (threshold chosen on VAL, applied unchanged) ===")
print(f"{'target':>7} {'VAL cov/acc':>16} {'TEST cov/acc':>18} {'OOS cov/acc':>18}")
for tgt in (0.75,0.70,0.65,0.60,0.58,0.56):
    bv=H.threshold_for_target(yva, pv, target=tgt, min_n=300)
    if bv is None:
        print(f"{tgt:>7.0%} {'not hit on VAL':>16}"); continue
    rt=H.apply_threshold(yte,pt,bv['conf_thr']); ro=H.apply_threshold(yoo,po,bv['conf_thr'])
    print(f"{tgt:>7.0%} {bv['coverage']:>7.3%}/{bv['accuracy']:.3f} "
          f"{rt['coverage']:>9.3%}/{rt['accuracy']:.3f} n={rt['n']:>5} "
          f"{ro['coverage']:>8.3%}/{ro['accuracy']:.3f} n={ro['n']:>4}")
print("\nInterpretation: the TEST and OOS columns are the unbiased live prediction rate at the")
print("VAL-selected confidence. A claim of X% holds only if TEST and OOS both reach X% at n>>0.")
