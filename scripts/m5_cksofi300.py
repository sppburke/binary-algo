"""Sweep row B3a @300s: CKS event-OFI (Cont-Kukanov-Stoikov) standalone at 300s. Retarget min1_cksofi to HS=300
(the cks 1s cache is horizon-independent; only the label horizon changes). Fast-KILL: VAL dirAUC worst-half
<=0.515 -> null (0.4993@60s / 0.4995@120s; monotone decay). Only split up/down if it passes the AUC gate."""
import json
import min1_cksofi as CK

CK.HS = 300; CK.GAP = 310

s = CK.run_standalone()
va = s["val_auc_worsthalf"]
killed = va <= 0.515
out = {"row": "B3a CKS event-OFI standalone @300s", "horizon_s": 300, "breakeven": 0.541,
       "val_auc_worsthalf": va, "val_auc_full": s.get("val_auc_full"),
       "per_year_moved_auc": {k: v for k, v in s.items() if k in ("2024", "2025", "2026")},
       "falsifier": {"KILL_if_val_auc<=": 0.515, "val_auc": va, "KILLED": bool(killed),
                     "verdict": ("KILLED: VAL dirAUC<=0.515, no 300s direction signal in CKS-OFI (null, as at 60s/120s)"
                                 if killed else "PASSES AUC gate -> proceed to per-year up/down split")}}
json.dump(out, open("m5_cksofi300_result.json", "w"), indent=1)
print(f"[B3a@300s] VAL dirAUC worst-half={va:.4f} -> {out['falsifier']['verdict']}", flush=True)
print("[B3a@300s] -> m5_cksofi300_result.json", flush=True)
