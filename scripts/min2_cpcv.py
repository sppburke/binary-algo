"""Decisive certification test of the min2 (2m,UP) side-split, prompted by the workflow's adversarial robustness
auditor (claim_holds=FALSE, major). Purged-combinatorial path distribution + block bootstrap on the FROZEN
book's UP trades, TIES-STRICT (a tie is a loss). Tests whether UP clears breakeven 0.541 across held-out
combinatorial time-blocks, not just on the lucky pooled split. (A full per-path ENSEMBLE refit would deflate
further — cf. the 15m precedent 0.647->0.5455 p10 0.531 FAIL in cpcv_certify_result.json.)

CERTIFY iff path_p10 >= 0.541 AND deflated/strict pooled CI95-lower >= 0.541."""
import json, numpy as np, pandas as pd
from itertools import combinations
import min2_production as M2

N_GROUPS, K_TEST = 8, 2   # C(8,2)=28 purged-combinatorial paths


def year_of(ts):
    return pd.to_datetime(np.asarray(ts), unit="s", utc=True).year.values


def collect_up_strict():
    """Frozen-book UP-predicted gated trades (test+oos), ties-strict win (tie=loss). Returns (ts, win)."""
    import time, gc; t0 = time.time()
    p, L, G, C, S = M2._load(); print(f"[cpcv] models loaded {time.time()-t0:.0f}s", flush=True)
    TS, WIN = [], []
    for sp in ("test", "oos"):
        b0 = M2.load_split(sp)
        months = list(b0.groupby(b0.index.to_period("M")))   # chunk by calendar month to cap peak RAM (OOM fix)
        for _, b in months:
            if len(b) < 4000:
                continue
            X, y, mag, valid, ts, idx = M2.prep(b)
            pr = M2._blend(p, L, G, C, S, X); pred = (pr > 0.5).astype(int)
            bbw = X["bbw1800"].values; rel = X["rel_ratio"].values; r300 = X["ret300"].values
            gate = valid & (bbw <= p["bbw1800_q67"]) & (rel >= p["rel_p70"]) & (np.sign(pr - 0.5) == -np.sign(r300))
            conf = np.abs(pr - 0.5); cand = gate & (conf >= p["conf_thr"])
            tr = M2.nonoverlap_chrono(ts, cand); up = tr[pred[tr] == 1]
            TS.append(ts[up]); WIN.append(((y[up] == 1) & (mag[up] > 0)).astype(float))
            del X, pr, pred, gate, conf, cand, b; gc.collect()
        del b0; gc.collect()
        print(f"[cpcv] {sp} done (chunked) {time.time()-t0:.0f}s", flush=True)
    ts = np.concatenate(TS); win = np.concatenate(WIN)
    o = np.argsort(ts); return ts[o], win[o]


def block_boot(win, nb=5000, block=20, seed=7):
    rng = np.random.default_rng(seed); n = len(win); nbk = int(np.ceil(n / block))
    accs = []
    for _ in range(nb):
        starts = rng.integers(0, n, nbk)
        idx = (starts[:, None] + np.arange(block)).ravel() % n
        accs.append(win[idx[:n]].mean())
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def main():
    ts, win = collect_up_strict()
    pooled = float(win.mean()); n = len(win)
    blo, bhi = block_boot(win)
    # purged-combinatorial path distribution
    edges = np.linspace(0, n, N_GROUPS + 1).astype(int)
    grp = [(edges[i], edges[i + 1]) for i in range(N_GROUPS)]
    path_accs = []
    for testg in combinations(range(N_GROUPS), K_TEST):
        idx = np.concatenate([np.arange(grp[g][0], grp[g][1]) for g in testg])
        if len(idx) >= 30:
            path_accs.append(float(win[idx].mean()))
    path_accs = np.array(path_accs)
    p10 = float(np.percentile(path_accs, 10)); pmean = float(path_accs.mean())
    frac_clear = float((path_accs >= 0.541).mean())
    # per-year strict
    yr = year_of(ts); per_year = {str(Y): round(float(win[yr == Y].mean()), 4) for Y in (2024, 2025, 2026) if (yr == Y).sum() > 30}
    certified = (p10 >= 0.541) and (blo >= 0.541)
    out = {"test": "min2 (2m,UP) purged-combinatorial + block-bootstrap, TIES-STRICT", "breakeven": 0.541,
           "n_up_trades": n, "pooled_strict_acc": round(pooled, 4), "block_boot_ci95": [round(blo, 4), round(bhi, 4)],
           "cpcv_paths": int(len(path_accs)), "path_mean": round(pmean, 4), "path_p10": round(p10, 4),
           "path_min": round(float(path_accs.min()), 4), "frac_paths_clear_0.541": round(frac_clear, 3),
           "per_year_strict": per_year,
           "verdict": {"CERTIFIED": bool(certified),
                       "statement": ("CERTIFIED: UP path_p10 and block-boot CI-lower both clear 0.541" if certified else
                                     "NOT CERTIFIED: UP path_p10 and/or block-boot CI95-lower below breakeven 0.541 -> the "
                                     "2m UP 0.555 is point-estimate-positive but does NOT survive purged-combinatorial / "
                                     "serial-dependence-aware scrutiny. Consistent with the 15m precedent (0.647->0.5455 "
                                     "p10 0.531 FAIL) and the online-ARF efficiency keystone. No certified 2m direction edge.")}}
    json.dump(out, open("min2_cpcv_result.json", "w"), indent=1)
    print(f"[cpcv] n={n} pooled_strict={pooled:.4f} block-boot CI95=[{blo:.4f},{bhi:.4f}]", flush=True)
    print(f"[cpcv] {len(path_accs)} paths: mean={pmean:.4f} p10={p10:.4f} min={path_accs.min():.4f} frac>=0.541={frac_clear:.2f}", flush=True)
    print(f"[cpcv] per-year strict: {per_year}", flush=True)
    print(f"[cpcv] VERDICT: {out['verdict']['statement']}", flush=True)
    print("[cpcv] -> min2_cpcv_result.json", flush=True)


if __name__ == "__main__":
    main()
