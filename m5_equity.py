"""Equity-path simulation of the DEPLOYABLE (5m,UP) config: frozen meta gate (~5% NY coverage), 1/8-Kelly stake
(~1%), trailing-win-rate KILL-SWITCH. Runs the FROZEN m5xp book's UP trades in true chronological order through
2024->2025->2026 (a legitimate forward backtest: the model trained on 2012-2023, never saw these years), and
reports the equity curve, max drawdown, longest losing streak, and how often the kill-switch fires.

Then a REGIME-DEATH STRESS (clearly synthetic): append a sub-breakeven stretch (win 0.53, the refit-deflated
level) to show the kill-switch caps the bleed vs no kill-switch. Deriv R=0.85 (breakeven 0.5405)."""
import json, numpy as np
import m5_xpair as MX
import m5_xpair_production as XP

YEARS = (("test24", "2024"), ("test25", "2025"), ("oos", "2026"))
R = 0.85
P_BELIEF = 0.58            # conservative (binding-year forward) win-rate for sizing
KFRAC = 1 / 8             # eighth-Kelly
KILL_W = 150              # trailing window (trades) for the kill-switch
KILL_LO = 0.52           # go flat if trailing win < this (clearly sub-breakeven)
RESUME_HI = 0.55         # resume betting once trailing win recovers above this (hysteresis)


def kelly_f(p): return max(0.0, (p * (1 + R) - 1) / R)
STAKE = KFRAC * kelly_f(P_BELIEF)


def collect():
    p, P, M = XP._load(); cols = p["primary_feats"]; mcols = p["meta_feats"]; THR = p["meta_thr"]
    TS, WIN, YR = [], [], []
    for w, label in YEARS:
        D = MX.build_xp(XP.SPL[w]); D = MX.augment(D, XP.SPL[w], XP.MODE)
        pr = P.predict(D[cols].astype("float32")); y = D["_y"].astype(int).values
        sm = M.predict(XP._Xmeta(D, pr, mcols)); ts = D["_ts"].values.astype("int64")
        ny = D["sess_ny"].values > 0.5
        m = ny & (sm >= THR) & (pr > 0.5); sel = MX.nonoverlap_chrono(ts, m)
        TS.append(ts[sel]); WIN.append((y[sel] == 1).astype(int)); YR.append([label] * len(sel))
        del D
    ts = np.concatenate(TS); win = np.concatenate(WIN); yr = np.concatenate(YR)
    o = np.argsort(ts); return ts[o], win[o], yr[o], THR


def simulate(win, killswitch):
    eq = 1.0; peak = 1.0; maxdd = 0.0; flat_ct = 0; kills = 0; active = True
    curve = [1.0]; streak = 0; maxstreak = 0
    for i, w in enumerate(win):
        if killswitch and i >= KILL_W:
            tw = win[i - KILL_W:i].mean()
            if active and tw < KILL_LO: active = False; kills += 1
            elif not active and tw >= RESUME_HI: active = True
        if (not killswitch) or active:
            eq *= (1 + STAKE * R) if w == 1 else (1 - STAKE)
        else:
            flat_ct += 1
        peak = max(peak, eq); maxdd = max(maxdd, (peak - eq) / peak)
        streak = 0 if w == 1 else streak + 1; maxstreak = max(maxstreak, streak)
        curve.append(eq)
    return {"end": eq, "maxdd": maxdd, "flat_pct": 100 * flat_ct / max(1, len(win)),
            "kills": kills, "max_loss_streak": maxstreak, "curve": curve}


def main():
    ts, win, yr, THR = collect()
    n = len(win); wr = win.mean()
    print(f"[equity] frozen gate meta>={THR:.4f} | trades={n} overall win={wr:.4f} | stake={100*STAKE:.2f}% (1/8-Kelly@{P_BELIEF}) | R={R}", flush=True)
    for _, label in YEARS:
        m = yr == label
        print(f"   {label}: trades={int(m.sum())} win={win[m].mean():.4f}", flush=True)
    no = simulate(win, False); ks = simulate(win, True)
    print(f"\n=== REAL FORWARD PATH 2024-2026 (model never saw these years) ===", flush=True)
    print(f"  NO kill-switch : end={no['end']:.2f}x  maxDD={100*no['maxdd']:.1f}%  longest_loss_streak={no['max_loss_streak']}", flush=True)
    print(f"  KILL-SWITCH    : end={ks['end']:.2f}x  maxDD={100*ks['maxdd']:.1f}%  flat={ks['flat_pct']:.0f}% of trades  kills={ks['kills']}", flush=True)
    # per-year end equity (no kill-switch, the realized path)
    print("\n  realized equity by year-end (no kill-switch):", flush=True)
    eq = 1.0; lastlabel = None
    yend = {}
    for i, w in enumerate(win):
        eq *= (1 + STAKE * R) if w == 1 else (1 - STAKE); yend[yr[i]] = eq
    for _, label in YEARS:
        if label in yend: print(f"    end {label}: {yend[label]:.2f}x", flush=True)
    # coarse equity sparkline (no kill-switch)
    c = np.array(no["curve"]); idx = np.linspace(0, len(c) - 1, 40).astype(int)
    samp = c[idx]; lo, hi = samp.min(), samp.max(); blocks = "▁▂▃▄▅▆▇█"
    spark = "".join(blocks[min(7, int((v - lo) / (hi - lo + 1e-9) * 7))] for v in samp)
    print(f"\n  equity curve (1.0->{no['end']:.1f}x): {spark}", flush=True)

    # ---- REGIME-DEATH STRESS (synthetic): append 600 trades at win 0.53 ----
    rng = np.random.default_rng(7)
    stress = (rng.random(600) < 0.53).astype(int)
    seq = np.concatenate([win, stress])
    s_no = simulate(seq, False); s_ks = simulate(seq, True)
    print(f"\n=== STRESS: real path THEN a 600-trade regime-death stretch @win=0.53 (synthetic, sub-breakeven) ===", flush=True)
    print(f"  NO kill-switch : end={s_no['end']:.2f}x  maxDD={100*s_no['maxdd']:.1f}%  (bleeds through the dead regime)", flush=True)
    print(f"  KILL-SWITCH    : end={s_ks['end']:.2f}x  maxDD={100*s_ks['maxdd']:.1f}%  flat={s_ks['flat_pct']:.0f}%  kills={s_ks['kills']}  (stops trading when edge dies)", flush=True)

    out = {"frozen_meta_thr": THR, "trades": int(n), "overall_win": round(float(wr), 4), "stake_pct": round(100*STAKE, 3),
           "kelly_fraction": KFRAC, "R": R, "p_belief": P_BELIEF, "killswitch": {"window": KILL_W, "kill_lo": KILL_LO, "resume_hi": RESUME_HI},
           "per_year_win": {l: round(float(win[yr == l].mean()), 4) for _, l in YEARS},
           "forward": {"no_ks": {"end_x": round(no["end"], 3), "maxdd_pct": round(100*no["maxdd"], 1), "max_loss_streak": no["max_loss_streak"]},
                       "ks": {"end_x": round(ks["end"], 3), "maxdd_pct": round(100*ks["maxdd"], 1), "flat_pct": round(ks["flat_pct"], 1), "kills": ks["kills"]}},
           "stress_regime_death": {"no_ks_end_x": round(s_no["end"], 3), "no_ks_maxdd_pct": round(100*s_no["maxdd"], 1),
                                   "ks_end_x": round(s_ks["end"], 3), "ks_maxdd_pct": round(100*s_ks["maxdd"], 1), "ks_kills": s_ks["kills"]}}
    json.dump(out, open("m5_equity_result.json", "w"), indent=1)
    print("\n[equity] -> m5_equity_result.json", flush=True)


if __name__ == "__main__":
    main()
