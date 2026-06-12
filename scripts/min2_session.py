"""DISCOVERY row: session-conditioning of the 2m UP side (the frozen book's gate ignores time-of-day, but
reversion dynamics differ by session). Split UP-gated trades by UTC session, select the best session set on
worst-VAL-half UP acc, apply to test/oos. Cheap (inference). Deriv-faithful, moved-bars, per-year CI95.

PRE-REGISTERED FALSIFIER: KILL unless a session-conditioned UP beats baseline OOS(2026)=0.555 AND clears
breakeven 0.541 in all 3 years AND n_moved(2026)>=50."""
import json, numpy as np, pandas as pd
import min2_production as M2

BASE_UP_2026 = 0.555
SESSIONS = {  # UTC hour windows
    "all": lambda h: np.ones_like(h, bool),
    "london": lambda h: (h >= 8) & (h < 17),
    "ny": lambda h: (h >= 13) & (h < 22),
    "overlap": lambda h: (h >= 13) & (h < 17),
    "asia": lambda h: (h >= 0) & (h < 8),
}


def year_of(ts):
    return pd.to_datetime(np.asarray(ts), unit="s", utc=True).year.values


def hour_of(ts):
    return pd.to_datetime(np.asarray(ts), unit="s", utc=True).hour.values


def gated_up(p, L, G, C, S, sp):
    b = M2.load_split(sp); X, y, mag, valid, ts, idx = M2.prep(b)
    pr = M2._blend(p, L, G, C, S, X); pred = (pr > 0.5).astype(int)
    bbw = X["bbw1800"].values; rel = X["rel_ratio"].values; r300 = X["ret300"].values
    gate = valid & (bbw <= p["bbw1800_q67"]) & (rel >= p["rel_p70"]) & (np.sign(pr - 0.5) == -np.sign(r300))
    conf = np.abs(pr - 0.5); cand = gate & (conf >= p["conf_thr"])
    tr = M2.nonoverlap_chrono(ts, cand); up = tr[pred[tr] == 1]
    del b, X
    return ts[up], y[up], mag[up]


def upacc(y, mag):
    moved = mag > 0
    if moved.sum() < 5:
        return None, [None, None], int(moved.sum())
    corr = (y[moved] == 1).astype(float); lo, hi = M2.boot(corr)
    return float(corr.mean()), [lo, hi], int(moved.sum())


def main():
    p, L, G, C, S = M2._load()
    V = gated_up(p, L, G, C, S, "val"); T = gated_up(p, L, G, C, S, "test"); O = gated_up(p, L, G, C, S, "oos")
    vts, vy, vmag = V; tts, ty, tmag = T; ots, oy, omag = O
    vh = hour_of(vts); order = np.argsort(vts); half = order[len(order) // 2:]
    # select best session on worst-VAL-half UP acc (require n>=30)
    best = None; scan = {}
    for name, fn in SESSIONS.items():
        m = fn(vh)[half]; a, ci, nm = upacc(vy[half][m], vmag[half][m])
        scan[name] = {"acc": a, "n_moved": nm}
        if nm >= 30 and a is not None and (best is None or a > best[1]):
            best = (name, a, nm)
    sess = best[0] if best else "all"
    print(f"[sess] worst-VAL-half by session: {scan}\n[sess] selected session={sess}", flush=True)
    th = hour_of(tts); oh = hour_of(ots); fn = SESSIONS[sess]
    res = {}
    for label, ts_, hh, yy, mg in (("2024", tts, th, ty, tmag), ("2025", tts, th, ty, tmag), ("2026", ots, oh, oy, omag)):
        yr = year_of(ts_); m = (yr == int(label)) & fn(hh)
        a, ci, nm = upacc(yy[m], mg[m]); res[label] = {"acc": a, "ci": ci, "n_moved": nm}
    up_all = [res[Y]["acc"] for Y in ("2024", "2025", "2026")]; up26 = res["2026"]["acc"]
    killed = not (up26 and up26 > BASE_UP_2026 and all(a and a >= 0.541 for a in up_all) and res["2026"]["n_moved"] >= 50)
    out = {"row": "discovery: session-conditioned 2m UP", "book": "EURUSD.min2.v1", "horizon_s": 120, "breakeven": 0.541,
           "selected_session": sess, "val_scan": scan, "per_year_UP": res,
           "falsifier": {"BASE_UP_2026": BASE_UP_2026, "sess_UP_2026": up26, "UP_allyears": up_all, "KILLED": bool(killed),
                         "verdict": ("KILLED: session-conditioned UP did not robustly beat 0.555 OOS" if killed
                                     else f"SURVIVED: {sess}-session UP beats baseline robustly")}}
    json.dump(out, open("min2_session_result.json", "w"), indent=1)
    for Y in ("2024", "2025", "2026"):
        print(f"  {Y} UP[{sess}] acc={res[Y]['acc']} CI={res[Y]['ci']} n_moved={res[Y]['n_moved']}", flush=True)
    print(f"[sess] FALSIFIER: {out['falsifier']['verdict']}", flush=True)
    print("[sess] -> min2_session_result.json", flush=True)


if __name__ == "__main__":
    main()
