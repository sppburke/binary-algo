"""Confidence/coverage curve for the (5m,UP) m5xp book: as you raise the confidence gate, does the realized UP
win-rate climb (real signal) or stay flat and just shed trades (efficient/noise)? Sweeps the meta gate (the
book's confidence=P(primary correct)) AND the primary direction-confidence |pr-0.5|, over the NY-gated
UP-predicted bars, per year, deriv-faithful (nonoverlap_chrono 300s, moved-only). Marks the frozen operating
point. Output = a table: gate -> coverage% + per-year win-rate(n). Breakeven 0.541. Inference-only, frugal."""
import json, numpy as np
import m5_xpair as MX
import m5_xpair_production as XP

YEARS = (("test24", "2024"), ("test25", "2025"), ("oos", "2026"))


def boot(c, nb=2000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    p, P, M = XP._load(); cols = p["primary_feats"]; mcols = p["meta_feats"]; FROZEN = p["meta_thr"]
    # collect NY UP-predicted bars per year: (sm, dconf=|pr-.5|, y, ts), and the full NY universe size for coverage
    data = {}; ny_univ = {}
    for w, label in YEARS:
        D = MX.build_xp(XP.SPL[w]); D = MX.augment(D, XP.SPL[w], XP.MODE)
        pr = P.predict(D[cols].astype("float32")); y = D["_y"].astype(int).values
        sm = M.predict(XP._Xmeta(D, pr, mcols)); ts = D["_ts"].values.astype("int64")
        ny = D["sess_ny"].values > 0.5
        upm = ny & (pr > 0.5)
        data[label] = {"sm": sm[upm], "dconf": np.abs(pr - 0.5)[upm], "y": y[upm], "ts": ts[upm]}
        # NY universe independent-bar count (denominator for coverage), de-overlapped
        ny_univ[label] = len(MX.nonoverlap_chrono(ts, ny))
        del D
    print(f"[confcurve] frozen meta gate THR={FROZEN:.4f}; breakeven 0.541; UP=meta-gated NY up-predicted, nonoverlap 300s\n", flush=True)

    def rowstats(arr_key, thr):
        cells = {}
        for _, label in YEARS:
            d = data[label]; m = d[arr_key] >= thr
            sel = MX.nonoverlap_chrono(d["ts"], m)
            if len(sel) < 1:
                cells[label] = (0, None, (None, None)); continue
            win = (d["y"][sel] == 1).astype(float); lo, hi = boot(win)
            cells[label] = (len(sel), float(win.mean()), (lo, hi))
        covtot = sum(c[0] for c in cells.values())
        covpct = 100.0 * covtot / max(1, sum(ny_univ.values()))
        return cells, covpct

    out = {"frozen_meta_thr": FROZEN, "breakeven": 0.541, "meta_curve": [], "dconf_curve": []}
    # ---- META gate curve (the book's actual confidence gate) ----
    allsm = np.concatenate([data[l]["sm"] for _, l in YEARS])
    grid = sorted(set([float(np.quantile(allsm, q)) for q in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99)] + [FROZEN]))
    print("=== META-gate (P(primary correct)) curve ===")
    print(f"{'meta_thr':>9} {'cov%':>5} | {'2024 win(n)':>16} {'2025 win(n)':>16} {'2026 win(n)':>16}")
    for thr in grid:
        cells, covpct = rowstats("sm", thr)
        tag = " <FROZEN" if abs(thr - FROZEN) < 1e-9 else ""
        def f(c): return f"{c[1]:.3f}({c[0]})" if c[1] is not None else f"--({c[0]})"
        print(f"{thr:9.4f} {covpct:5.1f} | {f(cells['2024']):>16} {f(cells['2025']):>16} {f(cells['2026']):>16}{tag}")
        out["meta_curve"].append({"thr": round(thr, 4), "cov_pct": round(covpct, 1),
                                  **{l: {"n": cells[l][0], "win": cells[l][1], "ci": [round(cells[l][2][0],4) if cells[l][2][0] else None, round(cells[l][2][1],4) if cells[l][2][1] else None]} for _, l in YEARS}})
    # ---- PRIMARY direction-confidence |pr-0.5| curve (among the frozen meta-gated set) ----
    # restrict to meta>=FROZEN first (the book's universe), then sweep direction confidence
    for label in [l for _, l in YEARS]:
        d = data[label]; keep = d["sm"] >= FROZEN
        for k in ("sm", "dconf", "y", "ts"): d[k] = d[k][keep]
    print("\n=== PRIMARY direction-confidence |p-0.5| curve (within frozen meta gate) ===")
    print(f"{'dconf_thr':>9} {'cov%':>5} | {'2024 win(n)':>16} {'2025 win(n)':>16} {'2026 win(n)':>16}")
    alldc = np.concatenate([data[l]["dconf"] for _, l in YEARS]) if sum(len(data[l]["dconf"]) for _,l in YEARS) else np.array([0.0])
    dgrid = sorted(set(float(np.quantile(alldc, q)) for q in (0.0, 0.25, 0.5, 0.7, 0.85, 0.95)))
    for thr in dgrid:
        cells, covpct = rowstats("dconf", thr)
        def f(c): return f"{c[1]:.3f}({c[0]})" if c[1] is not None else f"--({c[0]})"
        print(f"{thr:9.4f} {covpct:5.1f} | {f(cells['2024']):>16} {f(cells['2025']):>16} {f(cells['2026']):>16}")
        out["dconf_curve"].append({"thr": round(thr, 4), "cov_pct": round(covpct, 1),
                                   **{l: {"n": cells[l][0], "win": cells[l][1]} for _, l in YEARS}})
    json.dump(out, open("m5_confcurve_result.json", "w"), indent=1)
    print("\n[confcurve] -> m5_confcurve_result.json", flush=True)


if __name__ == "__main__":
    main()
