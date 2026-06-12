"""MULTI-TIMEFRAME Kronos direction ENSEMBLE — the user's idea: combine different-timeframe calls for one horizon
(e.g. 1m-fine + 5m-native) so higher-TF momentum informs the lower-TF direction call.

Kronos is single-series, so we run kronos_mtf.py at several (GRID,H) for the SAME target horizon H, each saving per-
decision-bar predictions to kronos_mtf_pred_<tag>.npz (tw, yw, call). This combiner INNER-JOINS them on the common
decision-bar timestamp tw (only bars that are valid decision instants in EVERY TF), soft-averages the per-TF hard
calls into a vote, thresholds at 0.5, and re-scores with the SAME deriv-faithful discipline (per-year CI95, CPCV
path_p10, pre-registered falsifier). A tie (vote==0.5) is dropped (no call).

Run: ~/binary-algo-venv/bin/python kronos_ensemble.py <out_tag> <pred_tag1> <pred_tag2> [pred_tag3 ...]
  e.g. kronos_ensemble.py ens_5m_1mfine_5mnative  mtf_zs_fine1_5m  mtf_zs_5m_all
  -> kronos_dir_mtf_ens_<out_tag>_result.json
"""
import sys, json, numpy as np, pandas as pd
from itertools import combinations
ROOT = "/home/sean/git/binary-algo"; BREAKEVEN = 0.541
OUT_TAG = sys.argv[1]; TAGS = sys.argv[2:]
assert len(TAGS) >= 2, "need >=2 prediction tags to ensemble"


def boot(c, nb=4000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def cpcv(ts, win, ng=8, k=2):
    n = len(win); e = np.linspace(0, n, ng + 1).astype(int); grp = [(e[i], e[i+1]) for i in range(ng)]; pa = []
    for tg in combinations(range(ng), k):
        ii = np.concatenate([np.arange(grp[g][0], grp[g][1]) for g in tg])
        if len(ii) >= 25: pa.append(float(win[ii].mean()))
    pa = np.array(pa)
    return (round(float(np.percentile(pa, 10)), 4), round(float((pa >= BREAKEVEN).mean()), 3)) if len(pa) else (None, None)

def per_year(ts, win):
    yr = pd.to_datetime(ts, unit="s", utc=True).year.values; out = {}
    for Y in (2024, 2025, 2026):
        m = yr == Y
        if m.sum() < 20: continue
        lo, hi = boot(win[m]); out[str(Y)] = {"n": int(m.sum()), "acc": round(float(win[m].mean()), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    return out


def main():
    # load each prediction set as {tw: (yw, call)}
    sets = []
    for tg in TAGS:
        d = np.load(f"{ROOT}/kronos_mtf_pred_{tg}.npz")
        sets.append({int(t): (int(y), float(c)) for t, y, c in zip(d["tw"], d["yw"], d["call"])})
        print(f"[{tg}] {len(sets[-1])} decision bars", flush=True)
    common = set(sets[0]); [common.intersection_update(s) for s in sets[1:]]
    common = np.array(sorted(common), dtype="int64")
    print(f"common decision bars across {len(TAGS)} TFs: {len(common)}", flush=True)
    if len(common) < 100:
        print("too few common bars — ABORT"); return
    yw = np.array([sets[0][t][0] for t in common])
    votes = np.array([[s[t][1] for s in sets] for t in common])      # [n, n_tf] of hard 0/1 calls
    mean_vote = votes.mean(axis=1)
    keep = mean_vote != 0.5                                            # drop ties (no majority)
    tw = common[keep]; yw = yw[keep]; call = (mean_vote[keep] > 0.5).astype(int)
    win = (call == yw).astype(float)
    res = {"out_tag": OUT_TAG, "tags": TAGS, "n_common": int(len(common)), "n_eval": int(len(tw)),
           "up_rate": round(float(yw.mean()), 4), "breakeven": BREAKEVEN,
           "falsifier": "KILL if no year CI95-lo>=0.541 OR CPCV path_p10<0.541",
           "per_year": per_year(tw, win)}
    p10, frac = cpcv(tw, win); res["pooled_acc"] = round(float(win.mean()), 4)
    res["cpcv_path_p10"] = p10; res["cpcv_frac_clear_0.541"] = frac
    any_clear = any(v["ci95"][0] >= BREAKEVEN for v in res["per_year"].values())
    res["VERDICT"] = "SURVIVES" if (any_clear and p10 is not None and p10 >= BREAKEVEN) else "KILLED"
    json.dump(res, open(f"{ROOT}/kronos_dir_mtf_ens_{OUT_TAG}_result.json", "w"), indent=1)
    print(f"ENSEMBLE {OUT_TAG}: pooled={res['pooled_acc']} p10={p10} frac={frac} n={len(tw)} "
          f"per-yr={ {Y:res['per_year'][Y]['acc'] for Y in res['per_year']} } -> {res['VERDICT']}", flush=True)


if __name__ == "__main__":
    main()
