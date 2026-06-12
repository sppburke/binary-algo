"""Fractional-Kelly EV-per-NY-hour curve for the (5m,UP) m5xp book. Answers: which confidence gate maximizes
COMPOUND GROWTH PER UNIT TIME, not just win-rate? Tight gates = big per-trade edge but few trades; loose gates =
small edge but many trades. Kelly growth ~ edge^2 per trade, frequency multiplies it -> there is an optimum.

HONEST sizing: set the Kelly fraction from the BINDING (worst) year's win-rate per gate (sizing on 2024's rosy
0.60 would over-bet and bleed in 2025); use QUARTER-Kelly for robustness/miscalibration. Deriv binary: stake f of
bankroll, win-> *(1+f*R), lose-> *(1-f). Log-growth/trade g(f)=p*ln(1+fR)+(1-p)*ln(1-f). Kelly f*=(p(1+R)-1)/R.
R=0.85 (breakeven p*=0.541). EV/NY-hour = g(f) * trades-per-NY-hour. CAVEAT: win-rates are forward-split; the
full-refit CPCV says the edge is regime-fragile, so these are RECENT-REGIME growth rates, not guarantees."""
import json, numpy as np
import m5_xpair as MX
import m5_xpair_production as XP

YEARS = (("test24", "2024"), ("test25", "2025"), ("oos", "2026"))
R = 0.85
KFRAC = 0.25   # quarter-Kelly


def kelly_f(p):
    return max(0.0, (p * (1 + R) - 1) / R)


def log_growth(p, f):
    if f <= 0 or f >= 1: return 0.0
    return p * np.log(1 + f * R) + (1 - p) * np.log(1 - f)


def main():
    pp, P, M = XP._load(); cols = pp["primary_feats"]; mcols = pp["meta_feats"]; FROZEN = pp["meta_thr"]
    data = {}
    for w, label in YEARS:
        D = MX.build_xp(XP.SPL[w]); D = MX.augment(D, XP.SPL[w], XP.MODE)
        pr = P.predict(D[cols].astype("float32")); y = D["_y"].astype(int).values
        sm = M.predict(XP._Xmeta(D, pr, mcols)); ts = D["_ts"].values.astype("int64")
        ny = D["sess_ny"].values > 0.5
        upm = ny & (pr > 0.5)
        ny_hours = len(np.unique(ts[ny] // 3600))   # distinct NY clock-hours available this period
        data[label] = {"sm": sm[upm], "y": y[upm], "ts": ts[upm], "ny_hours": ny_hours}
        del D
    print(f"[kelly] R={R} breakeven={1/(1+R):.4f} | {KFRAC:.0%}-Kelly | sizing on BINDING-year win | frozen meta={FROZEN:.4f}", flush=True)
    print(f"[kelly] NY-hours in data: 2024={data['2024']['ny_hours']} 2025={data['2025']['ny_hours']} 2026={data['2026']['ny_hours']}\n", flush=True)

    allsm = np.concatenate([data[l]["sm"] for _, l in YEARS])
    grid = sorted(set([float(np.quantile(allsm, q)) for q in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99)] + [FROZEN]))
    rows = []
    print(f"{'meta_thr':>8} {'cov_note':>9} | {'bind_win':>8} {'¼Kel_f%':>7} | "
          f"{'24 g/hr%':>8} {'25 g/hr%':>8} {'26 g/hr%':>8} | {'tr/hr(25)':>9}")
    for thr in grid:
        per = {}
        for _, label in YEARS:
            d = data[label]; m = d["sm"] >= thr; sel = MX.nonoverlap_chrono(d["ts"], m)
            n = len(sel)
            if n < 1: per[label] = {"n": 0, "p": None, "tph": 0.0}; continue
            p = float((d["y"][sel] == 1).mean()); tph = n / max(1, d["ny_hours"])
            per[label] = {"n": n, "p": p, "tph": tph}
        wins = [per[l]["p"] for _, l in YEARS if per[l]["p"] is not None and per[l]["n"] >= 30]
        if len(wins) < 3:
            continue
        bind = min(wins); f = KFRAC * kelly_f(bind)
        ghr = {}
        for _, label in YEARS:
            d = per[label]
            ghr[label] = (log_growth(d["p"], f) * d["tph"]) if d["p"] is not None else 0.0
        tag = "<FROZEN" if abs(thr - FROZEN) < 1e-9 else ""
        nmin = min(per[l]["n"] for _, l in YEARS)
        rows.append({"thr": round(thr, 4), "bind_win": round(bind, 4), "kelly_qf": round(f, 4),
                     "g_per_hr": {l: round(ghr[l], 6) for _, l in YEARS},
                     "n": {l: per[l]["n"] for _, l in YEARS}, "tph25": round(per["2025"]["tph"], 3),
                     "binding_g_per_hr": round(min(ghr["2024"], ghr["2025"], ghr["2026"]), 6)})
        print(f"{thr:8.4f} {('n'+str(nmin)):>9} | {bind:8.4f} {100*f:7.3f} | "
              f"{100*ghr['2024']:8.4f} {100*ghr['2025']:8.4f} {100*ghr['2026']:8.4f} | {per['2025']['tph']:9.3f} {tag}")
    best = max(rows, key=lambda r: r["binding_g_per_hr"]) if rows else None
    out = {"R": R, "kelly_fraction": KFRAC, "breakeven": round(1/(1+R), 4), "frozen_meta_thr": FROZEN,
           "ny_hours": {l: data[l]["ny_hours"] for _, l in YEARS}, "curve": rows,
           "best_by_binding_growth_per_hr": best}
    json.dump(out, open("m5_kelly_result.json", "w"), indent=1)
    if best:
        print(f"\n[kelly] PEAK binding-year growth/NY-hour at meta>={best['thr']} "
              f"(bind_win {best['bind_win']}, ¼Kelly {100*best['kelly_qf']:.2f}% stake): "
              f"binding {100*best['binding_g_per_hr']:.4f}%/hr, n(24/25/26)={best['n']}", flush=True)
    print("[kelly] -> m5_kelly_result.json", flush=True)


if __name__ == "__main__":
    main()
