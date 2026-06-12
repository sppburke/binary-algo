"""1-MIN UP-vs-DOWN ASYMMETRY DIAGNOSTIC — does the 60s book predict one side better than the other, and would a side-specialist
(or a side-aware selective gate) beat the symmetric book? Reuses the FROZEN min1_production child (no retrain) and its production
gate (compression-release x reversion x conf). Splits the independent trades by PREDICTED direction (up vs down) and reports
per-window 2024/2025/2026 accuracy + CI95 + n for each side. Deriv-faithful (non-overlap 60s, ties LOSE). Light: inference only.
If one side is materially more accurate AND stable across windows, a single-side book is the honest deliverable.
"""
import numpy as np, pandas as pd, time
import min1_production as M

def boot(c, nb=5000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def trades(sp, p, L, G, C, S):
    """Return per-trade arrays (pred, correct, ts, year) for the production-gated, non-overlap independent trades of split sp."""
    b = M.load_split(sp); X, y, mag, valid, ts, idx = M.prep(b)
    pr = M._blend(p, L, G, C, S, X); pred = (pr > 0.5).astype(int)
    bbw = X["bbw1800"].values; rel = X["rel_ratio"].values; r300 = X["ret300"].values
    gate = valid & (bbw <= p["bbw1800_q67"]) & (rel >= p["rel_tighten"]) & (np.sign(pr - 0.5) == -np.sign(r300))
    conf = np.abs(pr - 0.5); cand = gate & (conf >= p["conf_thr"])
    tr = M.nonoverlap_chrono(ts, cand)
    correct = ((pred[tr] == y[tr]) & (mag[tr] > 0)).astype(float)
    return dict(pred=pred[tr], correct=correct, year=idx[tr].year.values, conf=conf[tr])

def main():
    t0 = time.time()
    p, L, G, C, S = M._load()
    print(f"[updown] loaded frozen child; production gate (compression x reversion x conf) {time.time()-t0:.0f}s", flush=True)
    # build per-window trade sets: split 'test' into 2024/2025; 'oos' is 2026
    T = trades("test", p, L, G, C, S); O = trades("oos", p, L, G, C, S)
    print(f"[updown] test trades={len(T['correct'])} oos trades={len(O['correct'])} {time.time()-t0:.0f}s\n", flush=True)
    WIN = [("2024", T, lambda d: d["year"] == 2024), ("2025", T, lambda d: d["year"] == 2025), ("2026", O, lambda d: d["year"] == 2026)]

    def line(name, sel):
        s = ""
        for w, D, ym in WIN:
            m = ym(D) & sel(D); c = D["correct"][m]
            if len(c) < 5: s += f"  {w}: n{int(m.sum())}(thin)"; continue
            lo, hi = boot(c); s += f"  {w}: {c.mean():.3f}(n{len(c)},CI[{lo:.3f},{hi:.3f}])"
        return s

    print("[updown] ALL trades (symmetric book):" + line("all", lambda d: np.ones(len(d["correct"]), bool)), flush=True)
    print("[updown] UP-predictions only      :" + line("up", lambda d: d["pred"] == 1), flush=True)
    print("[updown] DOWN-predictions only    :" + line("dn", lambda d: d["pred"] == 0), flush=True)
    # the implied 'trade-only-the-better-side' book per window (descriptive; the SIDE is fixed a priori per the diagnostic)
    print("\n[updown] per-window side breakdown (n_up / n_down and which side wins):", flush=True)
    for w, D, ym in WIN:
        m = ym(D)
        up = m & (D["pred"] == 1); dn = m & (D["pred"] == 0)
        au = D["correct"][up].mean() if up.sum() else float("nan")
        ad = D["correct"][dn].mean() if dn.sum() else float("nan")
        print(f"   {w}: up acc={au:.3f}(n{int(up.sum())}) | down acc={ad:.3f}(n{int(dn.sum())}) | winner={'UP' if au>ad else 'DOWN'} by {abs(au-ad):.3f}", flush=True)
    print(f"\n[updown] DONE {time.time()-t0:.0f}s  (a side is exploitable only if it wins in ALL THREE windows AND CI95 clears the bar)", flush=True)

if __name__ == "__main__":
    main()
