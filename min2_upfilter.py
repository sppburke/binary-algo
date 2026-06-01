"""UP-ONLY FILTER refinement on the frozen 2m book (sweep row A8a). Among the gated UP-predicted trades,
tighten by confidence; select the keep-fraction on the WORST-VAL-half by UP moved-acc (NOT VAL-acc-max),
then apply that frozen confidence threshold to test/oos. Targets the (2m,UP) key directly. Deriv-faithful,
moved-bars, per-year CI95.

PRE-REGISTERED FALSIFIER: KILL unless the filtered UP beats baseline OOS(2026)=0.555 AND clears breakeven
0.541 in all 3 years AND n_moved(2026)>=50."""
import json, numpy as np, pandas as pd
import min2_production as M2

BASE_UP_2026 = 0.555


def year_of(ts):
    return pd.to_datetime(np.asarray(ts), unit="s", utc=True).year.values


def gated_up(p, L, G, C, S, sp):
    """UP-predicted gated independent trades for split sp: (ts, conf, y, mag)."""
    b = M2.load_split(sp); X, y, mag, valid, ts, idx = M2.prep(b)
    pr = M2._blend(p, L, G, C, S, X); pred = (pr > 0.5).astype(int)
    bbw = X["bbw1800"].values; rel = X["rel_ratio"].values; r300 = X["ret300"].values
    gate = valid & (bbw <= p["bbw1800_q67"]) & (rel >= p["rel_p70"]) & (np.sign(pr - 0.5) == -np.sign(r300))
    conf = np.abs(pr - 0.5); cand = gate & (conf >= p["conf_thr"])
    tr = M2.nonoverlap_chrono(ts, cand)
    up = tr[pred[tr] == 1]
    del b, X
    return ts[up], conf[up], y[up], mag[up]


def acc_ci(y, mag):
    moved = mag > 0
    if moved.sum() < 5:
        return None, [None, None], int(moved.sum())
    corr = (np.ones(moved.sum()) == y[moved]).astype(float)  # UP-pred correct iff y==1
    lo, hi = M2.boot(corr)
    return float(corr.mean()), [lo, hi], int(moved.sum())


def main():
    p, L, G, C, S = M2._load()
    vts, vcf, vy, vmag = gated_up(p, L, G, C, S, "val")
    tts, tcf, ty, tmag = gated_up(p, L, G, C, S, "test")
    ots, ocf, oy, omag = gated_up(p, L, G, C, S, "oos")
    print(f"[upf] val up-trades={len(vts)} test={len(tts)} oos={len(ots)}", flush=True)
    # worst-VAL-half = chronological 2nd half of VAL up-trades
    order = np.argsort(vts); half = order[len(order) // 2:]
    vcf_h, vy_h, vmag_h = vcf[half], vy[half], vmag[half]
    best = None
    for f in (1.0, 0.7, 0.5, 0.35, 0.25):
        thr = np.quantile(vcf, 1 - f) if f < 1.0 else -1.0   # threshold from ALL val up-trades
        sel = vcf_h >= thr
        a, ci, nm = acc_ci(vy_h[sel], vmag_h[sel])
        if nm >= 30 and a is not None and (best is None or a > best[1]):
            best = (f, a, thr, nm)
    if best is None:
        print("[upf] no valid fraction on worst-VAL-half"); return
    f, va, thr, vnm = best
    print(f"[upf] selected keep-fraction f={f} (worst-VAL-half UP acc={va:.3f} n{vnm}, conf_thr={thr:.5f})", flush=True)
    tyr = year_of(tts); oyr = year_of(ots)
    res = {}
    for label, ts_, cf, yy, mg in (("2024", tts, tcf, ty, tmag), ("2025", tts, tcf, ty, tmag), ("2026", ots, ocf, oy, omag)):
        yr = year_of(ts_)
        m = (yr == int(label)) & (cf >= thr)
        a, ci, nm = acc_ci(yy[m], mg[m])
        res[label] = {"acc": a, "ci": ci, "n_moved": nm}
    up_all = [res[Y]["acc"] for Y in ("2024", "2025", "2026")]
    up26 = res["2026"]["acc"]
    killed = not (up26 and up26 > BASE_UP_2026 and all(a and a >= 0.541 for a in up_all) and res["2026"]["n_moved"] >= 50)
    out = {"row": "A8a up-only filter refinement", "book": "EURUSD.min2.v1", "horizon_s": 120, "breakeven": 0.541,
           "keep_fraction": f, "conf_thr": float(thr), "worst_val_half_up_acc": va, "per_year_UP": res,
           "falsifier": {"BASE_UP_2026": BASE_UP_2026, "filtered_UP_2026": up26, "UP_allyears": up_all,
                         "KILLED": bool(killed),
                         "verdict": ("KILLED: filtered UP did not beat 0.555 OOS while clearing breakeven all 3 yrs (n>=50)"
                                     if killed else "SURVIVED: up-filter lifts 2m UP above baseline, robust")}}
    json.dump(out, open("min2_upfilter_result.json", "w"), indent=1)
    for Y in ("2024", "2025", "2026"):
        print(f"  {Y} UP filtered acc={res[Y]['acc']} CI={res[Y]['ci']} n_moved={res[Y]['n_moved']}", flush=True)
    print(f"[upf] FALSIFIER: {out['falsifier']['verdict']}", flush=True)
    print("[upf] -> min2_upfilter_result.json", flush=True)


if __name__ == "__main__":
    main()
