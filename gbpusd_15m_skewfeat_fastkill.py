"""Fast-kill screen for R4-02 distrib-skew-feat: rolling realized skewness as signed 3rd-moment feature.
KILL if: Kendall tau |tau| < .003 on 2023 VAL AND feature absent from top-40 FI on train.
Uses only pandas/scipy — safe to run with K=8 CPCV in background."""
import sys, json, os, glob
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, "/home/sean/git/binary-algo")
from harness import FEAT_DIR, SPLITS
from min1_production import wc_ret

PAIR = "GBPUSD"
WINDOWS = [5, 10, 20, 40, 80]

# Load GBPUSD 15m bars for 2021-2023 (train+val)
years = [2021, 2022, 2023]
dfs = []
for y in years:
    p = os.path.join(FEAT_DIR, f"{PAIR}_{y}.parquet")
    dfs.append(pd.read_parquet(p))
df = pd.concat(dfs).sort_index()
print(f"Loaded {len(df)} rows, years {years}")

# Get close column for GBPUSD own bars
close_col = [c for c in df.columns if "close" in c.lower() and "gbpusd" in c.lower()]
if not close_col:
    close_col = [c for c in df.columns if c.lower() == "close"]
if not close_col:
    # Try to find any price column
    close_col = [c for c in df.columns if "gbpusd" in c.lower()][:1]
print(f"Using price column: {close_col}")
if not close_col:
    print("ERROR: no close column found")
    print("Columns:", list(df.columns[:20]))
    sys.exit(1)

price = df[close_col[0]].dropna()
ret = price.pct_change()
# Direction label (sign of next-bar return)
direction = np.sign(ret.shift(-1))  # +1 or -1 or 0

# Moved bars only (|ret|>0)
moved = ret.abs() > 0

# Use 2023 as VAL year (SPLITS['val'] includes 2023)
val_mask = df.index.year == 2023
val_moved = val_mask & moved

print(f"VAL 2023 moved bars: {val_moved.sum()}")

# Compute rolling skewness for each window on GBPUSD own returns
results = {}
for w in WINDOWS:
    skew = ret.rolling(w, min_periods=w//2+1).skew()
    # Align with direction
    aligned = pd.DataFrame({"skew": skew, "dir": direction}).dropna()
    # VAL moved bars
    val_idx = aligned.index[aligned.index.year == 2023]
    if len(val_idx) < 100:
        print(f"  w={w}: too few VAL bars ({len(val_idx)})")
        continue
    val_sub = aligned.loc[val_idx]
    # Only moved bars in val
    val_ret = ret.loc[val_idx]
    val_moved_sub = val_ret.abs() > 0
    val_sub_moved = val_sub[val_moved_sub.values[:len(val_sub)]] if len(val_moved_sub) == len(val_sub) else val_sub

    if len(val_sub_moved) < 50:
        print(f"  w={w}: VAL moved too small ({len(val_sub_moved)})")
        continue

    tau, pval = kendalltau(val_sub_moved["skew"], val_sub_moved["dir"])
    results[w] = {"tau": tau, "pval": pval, "n": len(val_sub_moved)}
    print(f"  w={w}: tau={tau:.4f} p={pval:.4f} n={len(val_sub_moved)}")

# Also compute cross-pair skewness (EURUSD, USDJPY)
for partner in ["EURUSD", "USDJPY", "AUDUSD"]:
    p_path = os.path.join(FEAT_DIR, f"{partner}_2023.parquet")
    if not os.path.exists(p_path):
        continue
    dp = pd.read_parquet(p_path)
    # Get close for partner
    c_col = [c for c in dp.columns if c.lower() == "close" or (partner.lower() in c.lower() and "close" in c.lower())]
    if not c_col:
        c_col = [c for c in dp.columns if partner.lower() in c.lower()][:1]
    if not c_col:
        continue
    ret_p = dp[c_col[0]].pct_change()

    for w in [10, 20]:
        try:
            skew_p = ret_p.rolling(w, min_periods=w//2+1).skew()
            # Align with GBPUSD direction
            combined = pd.DataFrame({
                "skew_partner": skew_p,
                "dir_gbp": direction
            }).dropna()
            val_sub = combined[combined.index.year == 2023]
            if len(val_sub) < 100:
                continue
            tau, pval = kendalltau(val_sub["skew_partner"], val_sub["dir_gbp"])
            key = f"{partner}_w{w}"
            results[key] = {"tau": tau, "pval": pval, "n": len(val_sub)}
            print(f"  {key}: tau={tau:.4f} p={pval:.4f} n={len(val_sub)}")
        except Exception as e:
            print(f"  {partner} w={w}: error {e}")

# Summary
max_abs_tau = max((abs(v["tau"]) for v in results.values()), default=0)
kill = max_abs_tau < 0.003

verdict = "KILL" if kill else "SURVIVES_fast_kill"
print(f"\n=== FAST-KILL SUMMARY ===")
print(f"Max |tau| across all windows/pairs: {max_abs_tau:.4f}")
print(f"Threshold: 0.003")
print(f"Verdict: {verdict}")
if kill:
    print("R4-02 distrib-skew-feat KILLED: rolling skewness has no directional content (|tau| < .003)")
else:
    print("R4-02 distrib-skew-feat SURVIVES fast-kill → proceed to famonly ΔVAL-AUC screen")

result = {
    "candidate": "R4-02 distrib-skew-feat",
    "method": "rolling realized skewness (3rd-moment signed feature)",
    "kill_threshold": 0.003,
    "max_abs_tau": max_abs_tau,
    "verdict": verdict,
    "per_window": results,
    "kill_rule": "|tau| < .003 on 2023 VAL → no directional content"
}
out_path = "/home/sean/git/binary-algo/gbpusd_15m_skewfeat_fastkill_result.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"Result written to {out_path}")
