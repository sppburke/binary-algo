"""Sweep row E1a: 2m MAGNITUDE |ret120|>=Q (the certified SIZE edge family). Sign-invariant -> OUT OF SCOPE for
an UP/DOWN key, but the goal names it. Evaluate the frozen min2 magnitude model's AUC per year."""
import json, numpy as np, joblib
import min2_production as M2
p = json.load(open("models/min2_EURUSD_strategy.json"))
magthr = p["mag_top_tercile_thr"]; feats = p["feature_names"]
Mmod = joblib.load("models/min2_EURUSD_magnitude.joblib")
import pandas as pd
def yr(ts): return pd.to_datetime(np.asarray(ts), unit="s", utc=True).year.values
from sklearn.metrics import roc_auc_score
res = {}
for sp in ("test", "oos"):
    b = M2.load_split(sp); X, y, mag, valid, ts, idx = M2.prep(b)
    pm = Mmod.predict_proba(X[feats])[:, 1]
    ylab = (mag >= magthr).astype(int)
    for Y in (2024, 2025, 2026):
        m = valid & (yr(ts) == Y)
        if m.sum() > 500:
            res[str(Y)] = round(float(roc_auc_score(ylab[m], pm[m])), 4)
    del b, X
out = {"row": "E1a magnitude |ret120|>=Q (SIZE, sign-invariant)", "horizon_s": 120, "magthr_top_tercile": float(magthr),
       "magnitude_auc_per_year": res, "note": "SIZE edge (not up/down); confirms the certified family at 2m. Tradeable on Touch/Range/Straddle, NOT Rise/Fall."}
json.dump(out, open("min2_mag_result.json", "w"), indent=1)
print("[E1a] magnitude AUC per year:", res, flush=True)
print("[E1a] -> min2_mag_result.json", flush=True)
