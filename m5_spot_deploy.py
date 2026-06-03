"""SPOT-FX DEPLOYMENT SPEC (req-5) for the certified 5m EURUSD direction edge — 2026-06-03.

Live venue verification (m5_venue_feasibility_result.json) showed NO regulated venue offers a 5m EURUSD fixed-payout
binary (deriv min 15m; Nadex closed 2025-12-20). The ONE real venue-available expression of a 5m directional edge is
SPOT FX (EURUSD spot, 24/5, NO minimum hold) — open at t, close at t+300s. This re-evaluates the certified gated trades
under SPOT settlement (P&L = signed move - round-trip spread) so the edge is deployable-spec'd at a REAL venue, with the
honest spread sensitivity (avg 5m |move| ~ 2 pips, so spread is a large fraction of the move).

Reuses the cached combo_scores_<split>.npz per-bar scores (pxp = m5xp primary P(up), mxp = m5xp meta confidence,
fwd5 = signed forward 5m return, ts, sess_ny). Applies the CERTIFIED m5xp gate (NY & meta>=THR & pr>0.5 for UP;
symmetric for DOWN), nonoverlap_chrono independent trades, then spot P&L per trade swept over realistic EURUSD spreads.

PRE-REGISTERED FALSIFIER (written before metrics): SPOT-DEPLOYABLE (a side) iff per-trade spot expectancy mean>0 with
boot CI95-lo>0 on the BINDING year (worst of 2024/25/26, n>=150) at a realistic ECN spread (<=0.2 pip). Expected: UP
positive only at tight (<=0.2 pip) spreads (avg move ~2 pips, ~.55 edge), DOWN negative after spread (marginal edge).

  ~/binary-algo-venv/bin/python m5_spot_deploy.py
"""
import os, sys; sys.argv = ["x"]
import json, time, numpy as np
import m5_xpair as MX, m5_xpair_production as XP
ROOT = "/media/sean/CORSAIR/binary-algo"
EV = (("test24", 2024), ("test25", 2025), ("oos", 2026))
SCORE_COLS = ["ts", "y5", "fwd5", "sess_ny", "pxp", "mxp"]
PIP = 0.0001 / 1.16            # ~8.62e-5 return units per pip at EURUSD~1.16
SPREADS_PIP = [0.1, 0.2, 0.4, 0.6]   # raw/ECN -> typical retail
def boot(c, nb=2500, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
def maxdd(eq):
    peak = np.maximum.accumulate(eq); return float(((eq - peak) / peak).min())
def longest_losing_streak(pnl):
    best = cur = 0
    for x in pnl:
        cur = cur + 1 if x < 0 else 0; best = max(best, cur)
    return int(best)

def gated_trades(side_up, THR):
    """Return per-year dict of (ts_sorted, fwd5_sorted) for the certified gated trades of one side."""
    out = {}
    for split, yr in EV:
        z = np.load(f"{ROOT}/combo_scores_{split}.npz")
        pxp = z["pxp"]; mxp = z["mxp"]; ts = z["ts"].astype("int64"); ny = z["sess_ny"] > 0.5; fwd = z["fwd5"].astype("float64")
        if side_up:
            g = ny & (mxp >= THR) & (pxp > 0.5)
        else:
            g = ny & (mxp >= THR) & (pxp < 0.5)
        sel = MX.nonoverlap_chrono(ts, g, 300)
        if len(sel):
            o = np.argsort(ts[sel]); out[yr] = (ts[sel][o], fwd[sel][o])
    return out

def main():
    t0 = time.time()
    THR = float(json.load(open(XP.art("strategy.json"))).get("meta_thr", 0.5738))
    json.dump({"test": "SPOT-FX deployment spec for the certified 5m EURUSD edge", "meta_thr": THR, "pip_ret": PIP,
               "spreads_pip": SPREADS_PIP, "status": "PRE-REGISTERED",
               "FALSIFIER": "spot-deployable (a side) iff per-trade expectancy mean>0 AND boot CI95-lo>0 on the binding "
                            "year (worst of 24/25/26, n>=150) at a realistic ECN spread <=0.2 pip."},
              open(f"{ROOT}/m5_spot_deploy_result.json", "w"), indent=2)
    out = {"test": "SPOT-FX deployment spec (5m EURUSD certified edge re-settled as spot)", "meta_thr": THR,
           "venue": "spot FX (EURUSD, no min duration) — the real venue-available expression; binary 5m is venue-unavailable",
           "avg_abs_move_pip": None, "sides": {}}
    for side_up, side in ((True, "UP"), (False, "DOWN")):
        tr = gated_trades(side_up, THR)
        if not tr: continue
        allmove = np.concatenate([tr[y][1] for y in tr])
        out["avg_abs_move_pip"] = round(float(np.abs(allmove).mean() / PIP), 3)
        # direction-only up-rate sanity (gated): for UP, P(move>0); for DOWN, P(move<0)
        diracc = {int(y): round(float((tr[y][1] > 0).mean() if side_up else (tr[y][1] < 0).mean()), 4) for y in tr}
        side_out = {"gated_dir_acc_per_year": diracc, "n_per_year": {int(y): int(len(tr[y][1])) for y in tr}, "by_spread": {}}
        for sp in SPREADS_PIP:
            sret = sp * PIP
            yrs = {}
            for y in tr:
                mv = tr[y][1]
                pnl = (mv - sret) if side_up else (-mv - sret)   # spot P&L in return units (long UP / short DOWN), round-trip spread paid
                if len(pnl) < 20: continue
                lo, hi = boot(pnl)
                eq = np.cumprod(1.0 + pnl)                         # full-notional equity curve (1x); fractional sizing scales DD linearly
                yrs[int(y)] = dict(n=len(pnl), exp_pip=round(float(pnl.mean() / PIP), 4),
                                   exp_ci_lo_pip=round(lo / PIP, 4), spot_winrate=round(float((pnl > 0).mean()), 4),
                                   sharpe_pertrade=round(float(pnl.mean() / (pnl.std(ddof=1) + 1e-12)), 4),
                                   maxDD_1x=round(maxdd(eq), 4), longest_losing_streak=longest_losing_streak(pnl),
                                   final_mult_1x=round(float(eq[-1]), 3))
            if not yrs: continue
            los = [yrs[y]["exp_ci_lo_pip"] for y in yrs]; ns = [yrs[y]["n"] for y in yrs]
            byear = min(yrs, key=lambda y: yrs[y]["exp_ci_lo_pip"])   # CONSERVATIVE binding = worst year by CI-lo
            side_out["by_spread"][f"{sp}pip"] = dict(per_year=yrs,
                binding=dict(year=byear, exp_pip=yrs[byear]["exp_pip"], exp_ci_lo_pip=yrs[byear]["exp_ci_lo_pip"], min_n=min(ns)),
                deployable=bool(min(los) > 0 and min(ns) >= 150))   # require ALL years' CI-lo>0
        out["sides"][side] = side_out
        print(f"[spot] {side}: avg|move|={out['avg_abs_move_pip']}pip diracc={diracc}", flush=True)
        for sp in SPREADS_PIP:
            k = f"{sp}pip"; b = side_out["by_spread"].get(k, {}).get("binding")
            if b: print(f"  spread {sp}pip: binding {b['year']} exp={b['exp_pip']}pip CI-lo={b['exp_ci_lo_pip']} n{b['min_n']} deployable={side_out['by_spread'][k]['deployable']}", flush=True)
    # verdict
    up02 = out["sides"].get("UP", {}).get("by_spread", {}).get("0.2pip", {})
    dn02 = out["sides"].get("DOWN", {}).get("by_spread", {}).get("0.2pip", {})
    out["VERDICT"] = dict(
        UP_spot_deployable_at_0p2pip=bool(up02.get("deployable")),
        DOWN_spot_deployable_at_0p2pip=bool(dn02.get("deployable")),
        statement=(f"SPOT-FX deployment of the 5m edge (avg |move| {out['avg_abs_move_pip']} pip). "
                   f"UP binding expectancy @0.2pip ECN = {up02.get('binding',{}).get('exp_pip')} pip (CI-lo {up02.get('binding',{}).get('exp_ci_lo_pip')}) -> "
                   f"{'DEPLOYABLE' if up02.get('deployable') else 'NOT deployable'}. "
                   f"DOWN @0.2pip = {dn02.get('binding',{}).get('exp_pip')} pip -> {'DEPLOYABLE' if dn02.get('deployable') else 'NOT deployable'}. "
                   "Spread-sensitive: the certified directional edge is real but the ~2-pip avg 5m move leaves thin margin "
                   "after spread; deployability requires tight ECN spreads. Binary 5m remains venue-unavailable (deriv 15m/Nadex closed)."))
    json.dump(out, open(f"{ROOT}/m5_spot_deploy_result.json", "w"), indent=2, default=str)
    print(f"\nVERDICT: {out['VERDICT']['statement']}\n-> m5_spot_deploy_result.json ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__": main()
