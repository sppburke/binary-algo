"""Decisive faithfulness A/B: do PANEL-derived xp features equal the certified build_xp (raw) features on identical rows?
If max-abs-diff ~ 0 across all shared feature columns, the panel construction is byte-faithful (not leaky); the lower
CPCV level (.5355 vs .567) is then purely the known weaker-reimplementation harness effect, not a panel defect.
"""
import numpy as np, pandas as pd, sys
import m5_xpair as MX
import build_panel as BP

Y = 2024
# raw certified features for year Y
import os
os.environ["MX_HOR"] = "15"
raw = MX.build_xp([str(Y)])                       # index=timestamp, includes _ts and all xp feature cols
rawcols = MX.xp_cols(raw)                          # certified model feature names (excl _y,_ts,_fwd,hour)

# panel-derived features for year Y
P = pd.read_parquet(f"{BP.FEAT}/panel_{Y}.parquet")
Xp = BP.xp_features_from_panel(P)                  # same feature names, computed from the panel
Xp.index = pd.to_datetime(P["t"].values, unit="s", utc=True)

# align on timestamp (raw index is tz-aware UTC too)
raw_idx = raw.index
common = raw_idx.intersection(Xp.index)
print(f"raw rows={len(raw):,} panel rows={len(Xp):,} common={len(common):,}")

shared = [c for c in rawcols if c in Xp.columns]
print(f"shared feature cols: {len(shared)} / raw {len(rawcols)}")
A = raw.loc[common, shared].astype(float)
B = Xp.loc[common, shared].astype(float)
both = A.notna().values & B.notna().values
worst = []
for c in shared:
    a = A[c].values; b = B[c].values; m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0: worst.append((c, float("nan"), 0)); continue
    d = np.abs(a[m] - b[m]); sc = max(np.abs(a[m]).mean(), 1e-12)
    worst.append((c, float(d.max()), float(d.max()/sc)))
worst.sort(key=lambda z: -(z[2] if np.isfinite(z[2]) else 0))
print("\ntop abs-diffs (col, max_abs, max_rel):")
for c, mx, rel in worst[:10]:
    print(f"  {c:14s} max_abs={mx:.3e} max_rel={rel:.3e}")
allmax = max(w[1] for w in worst if np.isfinite(w[1]))
print(f"\nGLOBAL max abs diff across all shared cols = {allmax:.3e}")
print("VERDICT:", "PANEL FAITHFUL (features identical to certified build_xp)" if allmax < 1e-6
      else "MISMATCH — investigate" )
# also report which certified cols are NOT in the panel feature set (gate cols expected: sess_ny/sess_ln/comp60)
missing = [c for c in rawcols if c not in Xp.columns]
print("certified cols absent from panel features (expected gate cols):", missing)
