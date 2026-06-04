"""Faithful certification test of the (EURUSD, 60s) side-split, to formally CLOSE the UP and DOWN deliverables.
Retarget of min2_cpcv.py (the 2m version) to 60s: purged-combinatorial path distribution + block bootstrap on the
FROZEN min1 book's UP and DOWN gated trades, TIES-STRICT (a tie is a loss). Tests whether each side clears breakeven
0.541 across held-out combinatorial time-blocks (serial-dependence-aware), not just on the lucky pooled split.

CONTEXT: the 60s UP "edge" is a one-sided pred=1 FILTER (acc = up-rate of bet-up bars), structurally identical to
the 2m UP that FAILED this exact test (min2_cpcv_result.json: pooled .5445, block-boot CI-lo .521, path p10 .524,
57% paths clear). 15m precedent under full ENSEMBLE refit: 0.647->0.5455 p10 .531 FAIL. The online-ARF keystone
(min1_online.py AUC .503-.508 every year) already says 60s direction is near-efficient. This is the formal (e)-step.

NOTE: frozen-book CPCV (no per-path refit) — an OPTIMISTIC bound; a full per-path ensemble refit would deflate
further. So a frozen-book FAIL is decisive; a frozen-book PASS would need the refit confirmation before believing.

CERTIFY a side iff path_p10 >= 0.541 AND block-boot strict pooled CI95-lower >= 0.541.
"""
import json, time, gc, numpy as np, pandas as pd
from itertools import combinations
import min1_production as MP

N_GROUPS, K_TEST = 8, 2          # C(8,2)=28 purged-combinatorial paths
BREAKEVEN = 0.541


def year_of(ts):
    return pd.to_datetime(np.asarray(ts), unit="s", utc=True).year.values


def collect_sides_strict():
    """Frozen-book gated trades (test+oos), ties-strict. Returns dict side-> (ts, win) for UP(pred=1)/DOWN(pred=0)."""
    t0 = time.time(); p, L, G, C, S = MP._load(); print(f"[cpcv60] models loaded {time.time()-t0:.0f}s", flush=True)
    acc = {"UP": ([], []), "DOWN": ([], [])}
    for sp in ("test", "oos"):
        b0 = MP.load_split(sp)
        for _, b in b0.groupby(b0.index.to_period("M")):     # month chunks cap peak RAM
            if len(b) < 4000: continue
            X, y, mag, valid, ts, idx = MP.prep(b)
            pr = MP._blend(p, L, G, C, S, X); pred = (pr > 0.5).astype(int)
            bbw = X["bbw1800"].values; rel = X["rel_ratio"].values; r300 = X["ret300"].values
            gate = valid & (bbw <= p["bbw1800_q67"]) & (rel >= p["rel_tighten"]) & (np.sign(pr - 0.5) == -np.sign(r300))
            conf = np.abs(pr - 0.5); cand = gate & (conf >= p["conf_thr"])
            tr = MP.nonoverlap_chrono(ts, cand)
            for side, sp_pred in (("UP", 1), ("DOWN", 0)):
                s = tr[pred[tr] == sp_pred]
                acc[side][0].append(ts[s])
                acc[side][1].append(((y[s] == sp_pred) & (mag[s] > 0)).astype(float))   # ties-strict (mag>0)
            del X, pr, pred, gate, conf, cand, b; gc.collect()
        del b0; gc.collect()
        print(f"[cpcv60] {sp} collected {time.time()-t0:.0f}s", flush=True)
    out = {}
    for side in ("UP", "DOWN"):
        ts = np.concatenate(acc[side][0]); win = np.concatenate(acc[side][1])
        o = np.argsort(ts); out[side] = (ts[o], win[o])
    return out


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
    yr = year_of(ts); per_year = {str(Y): round(float(win[yr == Y].mean()), 4) for Y in (2024, 2025, 2026) if (yr == Y).sum() > 30}
    certified = bool(np.isfinite(p10) and np.isfinite(blo) and p10 >= BREAKEVEN and blo >= BREAKEVEN)
    return {"n_trades": n, "pooled_strict_acc": round(pooled, 4), "block_boot_ci95": [round(blo, 4), round(bhi, 4)],
            "cpcv_paths": int(len(pa)), "path_mean": round(float(pa.mean()), 4) if len(pa) else None,
            "path_p10": round(p10, 4) if np.isfinite(p10) else None,
            "path_min": round(float(pa.min()), 4) if len(pa) else None,
            "frac_paths_clear_0.541": round(float((pa >= BREAKEVEN).mean()), 3) if len(pa) else None,
            "per_year_strict": per_year, "CERTIFIED": certified}


def main():
    sides = collect_sides_strict()
    res = {side: cpcv_side(*sides[side]) for side in ("UP", "DOWN")}
    up, dn = res["UP"], res["DOWN"]
    out = {"key": "(EURUSD,60s) UP+DOWN faithful CPCV (frozen-book, ties-strict, purged-comb + block-boot)",
           "breakeven": BREAKEVEN, "n_groups": N_GROUPS, "k_test": K_TEST,
           "note": "frozen-book (no per-path refit) = OPTIMISTIC bound; a FAIL here is decisive.",
           "UP": up, "DOWN": dn,
           "verdict": {"UP_CERTIFIED": up["CERTIFIED"], "DOWN_CERTIFIED": dn["CERTIFIED"],
                       "statement": (
                           f"UP {'CERTIFIED' if up['CERTIFIED'] else 'NOT certified'} (pooled-strict {up['pooled_strict_acc']}, "
                           f"block-boot CI-lo {up['block_boot_ci95'][0]}, path p10 {up['path_p10']}, "
                           f"{up['frac_paths_clear_0.541']} of paths clear); "
                           f"DOWN {'CERTIFIED' if dn['CERTIFIED'] else 'NOT certified'} (pooled-strict {dn['pooled_strict_acc']}, "
                           f"path p10 {dn['path_p10']}). "
                           + ("" if (up['CERTIFIED'] or dn['CERTIFIED']) else
                              "Neither 60s side survives purged-combinatorial / serial-dependence-aware scrutiny under "
                              "ties-strict settlement -> consistent with the 2m precedent (.5445 FAIL) + online-ARF "
                              "efficiency keystone. No certified 60s direction edge; UP is a regime-dependent filter, DOWN dead."))}}
    json.dump(out, open("min1_cpcv_result.json", "w"), indent=1)
    for side in ("UP", "DOWN"):
        v = res[side]; print(f"[cpcv60] {side}: n={v['n_trades']} pooled_strict={v['pooled_strict_acc']} "
                             f"block-boot-lo={v['block_boot_ci95'][0]} path_p10={v['path_p10']} "
                             f"frac_clear={v['frac_paths_clear_0.541']} per-year={v['per_year_strict']}", flush=True)
    print(f"[cpcv60] VERDICT: {out['verdict']['statement']}", flush=True)
    print("[cpcv60] -> min1_cpcv_result.json", flush=True)


if __name__ == "__main__":
    main()
