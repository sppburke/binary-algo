"""15m WALK-FORWARD robustness — can adaptive retraining lift the m15 book's weak 2025 window (frozen 0.582)?

The faithful CPCV showed the 15m book is real but regime-dependent (robust ~0.58, recent 0.647, weakest 2025). This tests whether
EXPANDING-WINDOW walk-forward (retrain the 3-model ensemble through year Y-2, tune the gate on Y-1, evaluate Y) lifts the binding
2025 window above the frozen m15_production's 0.582. Honest prior (from m5/m10 walk-forward): adaptation lifts the binding window
only ~+0.01-0.02 — so expect 2025 ~0.59-0.60, a modest robustness gain, not a regime fix. Reuses min15_cpcv loaders/ensemble +
m15_production's live-faithful non-overlap. Deriv-faithful, per-year, CI95, q-tuned gate (same protocol as m15).
"""
import numpy as np, json, time
import min15_cpcv as CP
import m15_production as M15

GAP_S = 15 * 60
SUB_FIT = 220_000

def select_gate(pv, yv, bbwv, nyv, bbw_fit):
    best = None
    for q in (10, 20, 33):
        bthr = np.nanpercentile(bbw_fit, q); g = (bbwv <= bthr) & (nyv > 0)
        if g.sum() < 100: continue
        for cov in (0.05, 0.10):
            cthr = np.quantile(np.abs(pv[g] - 0.5), 1 - cov); sel = g & (np.abs(pv - 0.5) >= cthr)
            if sel.sum() < 100: continue
            a = ((pv[sel] > 0.5).astype(int) == yv[sel]).mean()
            if best is None or a > best[0]: best = (a, bthr, cthr, q, cov)
    return best

def main():
    t0 = time.time()
    X, y, ts, bbw, ny = CP.load_pooled()
    years = (ts.astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970)
    print(f"[wf15] pooled n={len(y):,} years {years.min()}-{years.max()} load {time.time()-t0:.0f}s", flush=True)
    frozen = {2024: 0.689, 2025: 0.582, 2026: 0.663}
    out = {}
    for Y in (2024, 2025, 2026):
        fit = np.where(years < Y - 1)[0]; val = np.where(years == Y - 1)[0]; test = np.where(years == Y)[0]
        if len(fit) > SUB_FIT: fit = fit[np.linspace(0, len(fit) - 1, SUB_FIT).astype(int)]
        L, G, C = CP.train_ensemble(X[fit], y[fit])
        pv = CP.proba(L, G, C, X[val])
        b = select_gate(pv, y[val], bbw[val], ny[val], bbw[fit])
        if b is None: print(f"[wf15] {Y}: no gate"); continue
        _, bthr, cthr, q, cov = b
        pt = CP.proba(L, G, C, X[test]); tst = ts[test]
        gt = (bbw[test] <= bthr) & (ny[test] > 0) & (np.abs(pt - 0.5) >= cthr)
        sel = M15.nonoverlap_chrono(tst, gt, GAP_S)
        corr = ((pt[sel] > 0.5).astype(int) == y[test][sel]).astype(float); acc = corr.mean() if len(sel) else float("nan")
        lo, hi = CP.boot(corr)
        out[str(Y)] = {"acc": float(acc), "n": int(len(sel)), "ci": [lo, hi], "q": q, "cov": cov,
                       "fit_years": f"2012-{Y-2}", "val_year": Y - 1, "frozen_m15": frozen[Y]}
        print(f"[wf15] {Y}: WALK-FWD acc={acc:.4f} (n{len(sel)}, CI[{lo:.3f},{hi:.3f}], fit 2012-{Y-2}, gate q{q}cov{cov:.0%})  "
              f"vs frozen m15 {frozen[Y]:.3f}  delta={acc-frozen[Y]:+.3f}  {time.time()-t0:.0f}s", flush=True)
    accs = [v["acc"] for v in out.values()]
    if accs:
        print(f"\n[wf15] WALK-FORWARD floor={min(accs):.4f} mean={np.mean(accs):.4f}  vs frozen m15 floor 0.582 / combined 0.647", flush=True)
    json.dump(out, open("m15_walkforward_result.json", "w"), indent=1)
    print(f"[wf15] DONE {time.time()-t0:.0f}s -> m15_walkforward_result.json", flush=True)

if __name__ == "__main__":
    main()
