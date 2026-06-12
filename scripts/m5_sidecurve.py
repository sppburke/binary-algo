"""SYMMETRIC confidence/coverage curve for BOTH sides of the m5xp book: does the DOWN side have a high-confidence
tail that clears breakeven the way UP does, or is it flat/dead? One build, both sides. For each meta-gate level,
per year, per side: de-overlapped trades + realized win-rate (UP: y==1 among pred-up; DOWN: y==0 among pred-down)
+ n + bootstrap CI95. Marks the frozen gate. Breakeven 0.541."""
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
    data = {}
    for w, label in YEARS:
        D = MX.build_xp(XP.SPL[w]); D = MX.augment(D, XP.SPL[w], XP.MODE)
        pr = P.predict(D[cols].astype("float32")); y = D["_y"].astype(int).values
        sm = M.predict(XP._Xmeta(D, pr, mcols)); ts = D["_ts"].values.astype("int64")
        ny = D["sess_ny"].values > 0.5
        data[label] = {"pr": pr, "sm": sm, "y": y, "ts": ts, "ny": ny}
        del D
    allsm = np.concatenate([data[l]["sm"][data[l]["ny"]] for _, l in YEARS])
    grid = sorted(set([float(np.quantile(allsm, q)) for q in (0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99)] + [FROZEN]))

    def side_rows(side):  # side: 1=UP(pred>.5,win y==1), 0=DOWN(pred<.5,win y==0)
        out = []
        for thr in grid:
            cells = {}
            for _, label in YEARS:
                d = data[label]
                if side == 1:
                    m = d["ny"] & (d["pr"] > 0.5) & (d["sm"] >= thr)
                else:
                    m = d["ny"] & (d["pr"] < 0.5) & (d["sm"] >= thr)
                sel = MX.nonoverlap_chrono(d["ts"], m)
                if len(sel) < 1: cells[label] = (0, None, (None, None)); continue
                win = (d["y"][sel] == side).astype(float); lo, hi = boot(win)
                cells[label] = (len(sel), float(win.mean()), (lo, hi))
            out.append((thr, cells))
        return out

    res = {"frozen_meta_thr": FROZEN, "breakeven": 0.541}
    for side, name in ((1, "UP"), (0, "DOWN")):
        print(f"\n=== {name} side (win = realized {'up' if side==1 else 'down'}) — meta gate sweep ===")
        print(f"{'meta_thr':>8} | {'2024 win(n)[CIlo]':>22} {'2025 win(n)[CIlo]':>22} {'2026 win(n)[CIlo]':>22}")
        rows = side_rows(side); jrows = []
        for thr, cells in rows:
            def f(c): return f"{c[1]:.3f}(n{c[0]})[{c[2][0]:.3f}]" if c[1] is not None else f"--(n{c[0]})"
            tag = " <FROZEN" if abs(thr - FROZEN) < 1e-9 else ""
            print(f"{thr:8.4f} | {f(cells['2024']):>22} {f(cells['2025']):>22} {f(cells['2026']):>22}{tag}")
            jrows.append({"thr": round(thr, 4), **{l: {"n": cells[l][0], "win": cells[l][1],
                          "ci_lo": round(cells[l][2][0], 4) if cells[l][2][0] is not None else None} for _, l in YEARS}})
        res[name] = jrows
    # DOWN summary: is there ANY gate where the binding (worst) year DOWN CI-lo clears 0.541?
    down_ok = any(all(r[l]["ci_lo"] is not None and r[l]["ci_lo"] >= 0.541 and r[l]["n"] >= 30 for _, l in YEARS) for r in res["DOWN"])
    res["down_any_gate_all3yr_CIlo_clears"] = bool(down_ok)
    json.dump(res, open("m5_sidecurve_result.json", "w"), indent=1)
    print(f"\n[sidecurve] DOWN has a gate where ALL 3 yrs CI-lo clear 0.541 (n>=30)? {down_ok}", flush=True)
    print("[sidecurve] -> m5_sidecurve_result.json", flush=True)


if __name__ == "__main__":
    main()
