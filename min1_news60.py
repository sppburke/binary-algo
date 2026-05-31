"""1-MIN MACRO-RELEASE IMPULSE (the one untested-at-60s directional lever).

Mechanism: a high-impact USD/EUR surprise creates a directional JUMP in EURUSD whose SIGN = sign(surprise mapped through
event_signs). The prior m5_news test was NULL — but that measured the 5-MINUTE post-release window, by which point the move is
done / mean-reverting. The 60-SECOND window is the IMPULSE timescale, where directional information demonstrably exists for a
brief moment. PRE-COMMITTED prediction = sign(eurusd_signal) from macro_calendar.parquet (built session-2, no fitting, no
sign-flip). Deriv-faithful 60s settlement (entry = next tick after release+1s, exit = last tick <= +60s, ties LOSE), per-window
2024/2025/2026 + in-sample 2021-23 reference, bootstrap CI95. Selective by volatility tier + |surprise z|. Tick data 2021-2026.

This is a SELECTIVE event-driven model (only trades in the seconds after a release) — naturally thin coverage; the bar is
whether post-release 60s accuracy is >0.65 with adequate n and a CI95 lower bound clearing it (NOT an n<30 mirage).
"""
import numpy as np, pandas as pd, time
import min1_production as M

def load_all_ticks():
    """Concatenated, time-sorted EURUSD 1s (ts_seconds, mid) across all tick splits 2021-2026."""
    tss, mids = [], []
    for sp in ("train", "val", "test", "oos"):
        b = M.load_split(sp)
        ts = b.index.values.astype("datetime64[s]").astype("int64")
        tss.append(ts); mids.append(b["mid"].values.astype(float))
        del b
    ts = np.concatenate(tss); mid = np.concatenate(mids)
    o = np.argsort(ts, kind="mergesort"); ts = ts[o]; mid = mid[o]
    u, idx = np.unique(ts, return_index=True)   # dedup identical seconds (keep first)
    return ts[idx], mid[idx]

def boot(c, nb=5000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def main():
    t0 = time.time()
    cal = pd.read_parquet("macro_calendar.parquet")
    cal = cal[cal["eurusd_signal"].astype(float) != 0.0].copy()
    _rel = pd.to_datetime(cal["ts"], utc=True)
    cal["rel_ts"] = _rel.to_numpy().astype("datetime64[s]").astype("int64")   # UTC seconds since epoch
    cal["pred_up"] = (cal["eurusd_signal"].astype(float) > 0).astype(int)
    cal["yr"] = _rel.dt.year.values
    cal["absz"] = cal["surp_z"].astype(float).abs()
    print(f"[news60] calendar events with signal={len(cal)} {time.time()-t0:.0f}s", flush=True)
    ts, mid = load_all_ticks()
    print(f"[news60] ticks {len(ts):,} span {pd.to_datetime(ts[0],unit='s')}..{pd.to_datetime(ts[-1],unit='s')} {time.time()-t0:.0f}s", flush=True)
    ret, valid = M.wc_ret(ts, mid, M.HS, M.TOL_S, M.ENTRY_LAG_S)   # deriv-faithful 60s fwd from EVERY tick
    # map each release to the first tick at/after the release second
    rel = cal["rel_ts"].values
    pos = np.searchsorted(ts, rel, side="left")
    inb = pos < len(ts)
    cal = cal.loc[inb].copy(); pos = pos[inb]
    # require the entry tick within 60s of the release (else the release fell in a data gap / weekend)
    near = (ts[pos] - cal["rel_ts"].values) <= 60
    cal = cal.loc[near].copy(); pos = pos[near]
    cal["bar"] = pos
    cal["valid"] = valid[pos]; cal["ret"] = ret[pos]
    cal = cal[cal["valid"]].copy()
    cal["actual_up"] = (cal["ret"].values > 0).astype(int); cal["mag"] = np.abs(cal["ret"].values)
    cal["correct"] = ((cal["pred_up"].values == cal["actual_up"].values) & (cal["mag"].values > 0)).astype(float)
    print(f"[news60] releases matched to a valid 60s window={len(cal)} {time.time()-t0:.0f}s\n", flush=True)

    def report(sub, label):
        print(f"  --- {label} (n_total={len(sub)}) ---", flush=True)
        print(f"  {'window':>7} {'n':>5} {'acc':>6}  {'CI95':>16}", flush=True)
        for w, yy in (("2021-23(IS)", None), ("2024", 2024), ("2025", 2025), ("2026", 2026)):
            s = sub if yy is None else sub[sub["yr"] == yy]
            if yy is None: s = sub[sub["yr"] <= 2023]
            c = s["correct"].values
            if len(c) < 5: print(f"  {w:>7} {len(c):>5}   (too few)"); continue
            lo, hi = boot(c)
            flag = "  <-- >0.65 & CI clears!" if (yy in (2024,2025,2026) and len(c) >= 25 and lo > 0.65) else ""
            print(f"  {w:>11} {len(c):>5} {c.mean():6.3f}  [{lo:.3f},{hi:.3f}]{flag}", flush=True)

    for tier in (["HIGH"], ["HIGH", "MEDIUM"]):
        sub = cal[cal["volatility"].isin(tier)]
        report(sub, f"volatility in {tier}")
        for zt in (1.0, 1.5):
            report(sub[sub["absz"] >= zt], f"volatility in {tier} & |surp_z|>={zt}")
    # also: short-horizon impulse (15s/30s) where the directional jump is freshest, HIGH-vol
    print("\n[news60] short-horizon impulse (HIGH-vol, |surp_z|>=1), entry next-tick:", flush=True)
    hv = cal_index = cal[(cal["volatility"] == "HIGH") & (cal["absz"] >= 1.0)]
    for H in (15, 30, 60):
        r2, v2 = M.wc_ret(ts, mid, H, M.TOL_S, M.ENTRY_LAG_S)
        p = hv["bar"].values; vok = v2[p]
        pu = hv["pred_up"].values[vok]; au = (r2[p][vok] > 0).astype(int); mg = np.abs(r2[p][vok])
        cor = ((pu == au) & (mg > 0)).astype(float)
        for yy in (2024, 2025, 2026):
            m = (hv["yr"].values[vok] == yy)
            c = cor[m]
            if len(c) >= 5:
                lo, hi = boot(c); print(f"    H={H}s {yy}: n={len(c)} acc={c.mean():.3f} CI[{lo:.3f},{hi:.3f}]", flush=True)
    print(f"\n[news60] DONE {time.time()-t0:.0f}s  (need a window with n>=25, acc>0.65, CI95 lower>0.65 to clear the goal)", flush=True)

if __name__ == "__main__":
    main()
