"""Sweep row B3a: CKS event-OFI (Cont-Kukanov-Stoikov) standalone at 120s. Retarget min1_cksofi to HS=120
(the cks 1s cache is horizon-independent; only the label horizon changes). Fast-KILL falsifier: VAL dirAUC
worst-half <= 0.515 -> null (was 0.4993 at 60s). Only do the up/down split if it passes the AUC gate."""
import json
import min1_cksofi as CK

CK.HS = 120; CK.GAP = 130  # retarget to 2-minute (cache unchanged)

s = CK.run_standalone()
va = s["val_auc_worsthalf"]
killed = va <= 0.515
out = {"row": "B3a CKS event-OFI standalone @120s", "horizon_s": 120, "breakeven": 0.541,
       "val_auc_worsthalf": va, "val_auc_full": s.get("val_auc_full"),
       "per_year_moved_auc": {k: v for k, v in s.items() if k in ("2024", "2025", "2026")},
       "falsifier": {"KILL_if_val_auc<=": 0.515, "val_auc": va, "KILLED": bool(killed),
                     "verdict": ("KILLED: VAL dirAUC<=0.515, no 120s direction signal in CKS-OFI (null, as at 60s)"
                                 if killed else "PASSES AUC gate -> proceed to per-year up/down split")}}
json.dump(out, open("min2_cksofi_result.json", "w"), indent=1)
print(f"[B3a] VAL dirAUC worst-half(120s)={va:.4f} -> {out['falsifier']['verdict']}", flush=True)
print("[B3a] -> min2_cksofi_result.json", flush=True)
