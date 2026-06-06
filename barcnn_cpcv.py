"""BAR-PATTERN CNN — step 3: faithful CPCV certification of the bar-image CNN predictions.

Mirrors min1_cpcv.cpcv_side (Lopez de Prado purged-combinatorial CV + block bootstrap, ties-strict) applied to
the FROZEN CNN's selected 60s trades, pooled over the held-out test+oos windows. Per the min1_cpcv note, a
frozen-model CPCV is an OPTIMISTIC bound, so a FAIL here is decisive. CERTIFY a (variant, coverage) iff
path_p10 >= 0.541 AND block-boot strict pooled CI95-lower >= 0.541.

Eligible windows in barcnn_pred_<variant>.npz are already valid+moved (mag>0) -> ties-strict holds automatically.

Run: ~/binary-algo-venv/bin/python barcnn_cpcv.py <variant>   (default loops hist,ohlc,gaf if present)
"""
import sys, os, json, numpy as np, pandas as pd
from itertools import combinations

ROOT = "/media/sean/CORSAIR/binary-algo"
HS, TOL = 60, 10; GAP = HS + TOL
N_GROUPS, K_TEST, BREAKEVEN = 8, 2, 0.541


def nonoverlap_chrono(ts, mask, gap=GAP):
    take = []; bu = -1
    for i in np.where(mask)[0]:
        if ts[i] < bu: continue
        take.append(i); bu = int(ts[i]) + gap
    return np.array(take, dtype=int)

def block_boot(win, nb=5000, block=20, seed=7):
    if len(win) < block * 2: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(win); nbk = int(np.ceil(n / block)); accs = []
    for _ in range(nb):
        starts = rng.integers(0, n, nbk)
        idx = (starts[:, None] + np.arange(block)).ravel() % n
        accs.append(win[idx[:n]].mean())
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))

def cpcv_side(ts, win):
    pooled = float(win.mean()); n = len(win); blo, bhi = block_boot(win)
    edges = np.linspace(0, n, N_GROUPS + 1).astype(int); grp = [(edges[i], edges[i + 1]) for i in range(N_GROUPS)]
    pa = []
    for testg in combinations(range(N_GROUPS), K_TEST):
        idx = np.concatenate([np.arange(grp[g][0], grp[g][1]) for g in testg])
        if len(idx) >= 30: pa.append(float(win[idx].mean()))
    pa = np.array(pa); p10 = float(np.percentile(pa, 10)) if len(pa) else float("nan")
    yr = pd.to_datetime(ts, unit="s", utc=True).year.values
    per_year = {str(Y): round(float(win[yr == Y].mean()), 4) for Y in (2024, 2025, 2026) if (yr == Y).sum() > 30}
    certified = bool(np.isfinite(p10) and np.isfinite(blo) and p10 >= BREAKEVEN and blo >= BREAKEVEN)
    return {"n_trades": n, "pooled_strict_acc": round(pooled, 4), "block_boot_ci95": [round(blo, 4), round(bhi, 4)],
            "cpcv_paths": int(len(pa)), "path_mean": round(float(pa.mean()), 4) if len(pa) else None,
            "path_p10": round(p10, 4) if np.isfinite(p10) else None,
            "path_min": round(float(pa.min()), 4) if len(pa) else None,
            "frac_paths_clear_0.541": round(float((pa >= BREAKEVEN).mean()), 3) if len(pa) else None,
            "per_year_strict": per_year, "CERTIFIED": certified}


def run(variant):
    d = np.load(f"{ROOT}/barcnn_pred_{variant}.npz")
    vts, vp = d["val_ts"], d["val_p"]; vconf = np.abs(vp - 0.5)
    # pool held-out test+oos (the moved+valid+contig windows), chronological
    ts = np.concatenate([d["test_ts"], d["oos_ts"]]).astype("int64")
    y  = np.concatenate([d["test_y"],  d["oos_y"]]).astype("int64")
    p  = np.concatenate([d["test_p"],  d["oos_p"]]).astype("float32")
    o = np.argsort(ts); ts, y, p = ts[o], y[o], p[o]
    pred = (p > 0.5).astype(int); conf = np.abs(p - 0.5)
    out = {"variant": variant, "breakeven": BREAKEVEN, "n_groups": N_GROUPS, "k_test": K_TEST,
           "note": "frozen-CNN CPCV = OPTIMISTIC bound; a FAIL is decisive. Eligible windows are moved (ties-strict).",
           "coverages": {}}
    for cov in (1.00, 0.10, 0.05, 0.02):
        thr = 0.0 if cov >= 1.0 else float(np.quantile(vconf, 1 - cov))   # gate threshold from VAL confidence
        sel = nonoverlap_chrono(ts, conf >= thr)
        if len(sel) < 60:
            out["coverages"][f"cov{cov}"] = {"val_conf_thr": round(thr, 5), "n_trades": int(len(sel)), "CERTIFIED": False}
            continue
        win = (pred[sel] == y[sel]).astype(float)
        out["coverages"][f"cov{cov}"] = {"val_conf_thr": round(thr, 5), **cpcv_side(ts[sel], win)}
    cert_any = any(v.get("CERTIFIED") for v in out["coverages"].values())
    out["ANY_COVERAGE_CERTIFIED"] = cert_any
    json.dump(out, open(f"{ROOT}/barcnn_cpcv_{variant}_result.json", "w"), indent=1)
    print(f"\n===== CPCV {variant} =====", flush=True)
    for cv, v in out["coverages"].items():
        print(f"  {cv:8s} n={v.get('n_trades')} pooled={v.get('pooled_strict_acc')} "
              f"block-boot-lo={v.get('block_boot_ci95',[None])[0]} path_p10={v.get('path_p10')} "
              f"frac_clear={v.get('frac_paths_clear_0.541')} per-year={v.get('per_year_strict')} "
              f"CERT={v.get('CERTIFIED')}", flush=True)
    print(f"  -> barcnn_cpcv_{variant}_result.json   ANY_CERT={cert_any}", flush=True)
    return out


if __name__ == "__main__":
    variants = [sys.argv[1]] if len(sys.argv) > 1 else \
        [v for v in ("hist", "ohlc", "gaf") if os.path.exists(f"{ROOT}/barcnn_pred_{v}.npz")]
    for v in variants:
        run(v)
