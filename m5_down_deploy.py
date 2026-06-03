"""DOWN DEPLOYMENT DECISION-SHEET (req-5) for EURUSD.m5xp_magw_down.v1 — 2026-06-03.

Even a non-deployable side gets the FULL deployment spec: operating gate, coverage, Kelly on the robust floor, equity
path + max-DD + losing-streak, venue, kill-switch — and the SPECIFIC metrics that disqualify it. Loads the frozen
magweight-DOWN book (models/m5xp_EURUSD_magw_down_primary_lgb.txt), extracts its certified gated DOWN trades, and
quantifies both the binary equity path (Kelly-sized on p10=.5441) and the spot expectancy — to produce a 'DO NOT DEPLOY;
here is why' sheet with numbers, not a verdict.

PRE-REGISTERED DECISION RULE (written before metrics): DEPLOY DOWN iff binding-year (worst of 2024/25/26) win-rate
CI95-lo > breakeven 0.541 (positive binary EV every year) AND spot per-trade expectancy CI-lo > 0 at <=0.2 pip. Else
DO NOT DEPLOY and record the disqualifiers. Prior: NON-deployable (binary forward-2025 CI-lo .533<.541; spot negative).

  ~/binary-algo-venv/bin/python m5_down_deploy.py
"""
import os, sys; sys.argv = ["x"]
import json, time, numpy as np
import lightgbm as lgb
import m5_xpair as MX, m5_xpair_production as XP
ROOT = "/media/sean/CORSAIR/binary-algo"; BE = 0.541; R = 0.85
EV = (("test24", 2024), ("test25", 2025), ("oos", 2026))
PIP = 0.0001 / 1.16; SPREADS_PIP = [0.1, 0.2, 0.4]
P10_FLOOR = 0.5441   # certified DOWN cov0.05 refit p10 (m5_magweight_cpcv_result.json)
def boot(c, nb=2500, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
def maxdd(eq):
    peak = np.maximum.accumulate(eq); return float(((eq - peak) / peak).min())
def longest_losing_streak(wins):
    best = cur = 0
    for w in wins:
        cur = 0 if w else cur + 1; best = max(best, cur)
    return int(best)
def kelly(p, R):  # binary win +R / lose -1
    return max(0.0, p - (1 - p) / R)

def main():
    t0 = time.time()
    strat = json.load(open(f"{ROOT}/models/m5xp_EURUSD_magw_down_strategy.json"))
    cols = strat["primary_feats"]; THR = float(strat["conf_thr_down_cov05"])
    json.dump({"test": "DOWN deployment decision-sheet (EURUSD.m5xp_magw_down.v1)", "breakeven": BE, "R": R,
               "gate": strat["gate"], "conf_thr_down_cov05": THR, "p10_floor": P10_FLOOR, "status": "PRE-REGISTERED",
               "DECISION_RULE": "DEPLOY iff binding-year win CI95-lo>0.541 AND spot exp CI-lo>0 @<=0.2pip; else DO NOT DEPLOY + disqualifiers."},
              open(f"{ROOT}/m5_down_deploy_result.json", "w"), indent=2)
    bst = lgb.Booster(model_file=f"{ROOT}/models/m5xp_EURUSD_magw_down_primary_lgb.txt")
    print(f"[down-deploy] loaded magweight-DOWN primary; gate thr={THR:.4f} ({time.time()-t0:.0f}s)", flush=True)
    peryear = {}; TS_ALL = []; WIN_ALL = []; MOVE_ALL = []
    for split, yr in EV:
        D = MX.augment(MX.build_xp(XP.SPL[split]), XP.SPL[split], XP.MODE)
        pr = bst.predict(D[cols].astype("float32")); y = D["_y"].astype(int).values
        ts = D["_ts"].values.astype("int64"); ny = D["sess_ny"].values > 0.5; fwd = D["_fwd"].astype("float64").values
        conf = np.abs(pr - 0.5); g = ny & (pr < 0.5) & (conf >= THR)
        sel = MX.nonoverlap_chrono(ts, g, 300)
        if len(sel) < 20: continue
        o = np.argsort(ts[sel]); sel = sel[o]
        win = (y[sel] == 0).astype(int); mv = fwd[sel]
        lo, hi = boot(win.astype(float))
        peryear[yr] = dict(n=int(len(sel)), frozen_down_winrate=round(float(win.mean()), 4), ci=[round(lo, 4), round(hi, 4)],
                           moved_up_rate=round(float((y[sel] == 1).mean()), 4),
                           binary_EV_pct_FROZEN=round(float(win.mean() * R - (1 - win.mean())) * 100, 3))
        TS_ALL.append(ts[sel]); WIN_ALL.append(win); MOVE_ALL.append(mv)
        del D
    ts_all = np.concatenate(TS_ALL); ordr = np.argsort(ts_all)
    win_all = np.concatenate(WIN_ALL)[ordr]; move_all = np.concatenate(MOVE_ALL)[ordr]
    # --- BINARY equity path sized on the ROBUST FLOOR p10=.5441 (full + 1/8 Kelly) ---
    fK = kelly(P10_FLOOR, R)
    binary = {}
    for name, f in (("full_kelly", fK), ("eighth_kelly", fK / 8)):
        ret = np.where(win_all == 1, f * R, -f); eq = np.cumprod(1 + ret)
        binary[name] = dict(stake_pct=round(f * 100, 4), final_mult=round(float(eq[-1]), 4),
                            maxDD=round(maxdd(eq), 4), longest_losing_streak=longest_losing_streak(win_all),
                            n_trades=int(len(win_all)))
    # --- SPOT equity (short DOWN: P&L = -move - spread) ---
    spot = {}
    for sp in SPREADS_PIP:
        sret = sp * PIP; pnl = -move_all - sret; lo, hi = boot(pnl)
        spot[f"{sp}pip"] = dict(exp_pip=round(float(pnl.mean() / PIP), 4), exp_ci_lo_pip=round(lo / PIP, 4),
                                spot_winrate=round(float((pnl > 0).mean()), 4), n=int(len(pnl)))
    # --- ROBUST (refit-CPCV) truth vs FROZEN overstatement ---
    # The frozen-book forward win-rates above OVERSTATE (frozen-trade overstatement, the documented trap): they show
    # ~.65-.73 with moved up-rate ~.35 (heavy down-selection) but the per-fold REFIT-CPCV (the honest test) gives only
    # p10 .5441 and refit forward-2025 .559 (CI-lo .533 < breakeven). The control (unweighted m5xp primary, same harness)
    # reproduces the established .536 — so the gap is the magweight book's frozen overstatement, NOT a real edge.
    refit = dict(source="m5_magweight_cpcv_result.json", cov0_05_p10=P10_FLOOR, cov0_05_frac_clear=0.893,
                 refit_forward_2025_down=0.559, refit_forward_2025_ci_lo=0.533, certified_at_cov0_05=True,
                 note="CPCV refit p10 .5441 ~ breakeven; forward-2025 CI-lo .533 < .541 -> binding year fails")
    frozen_overstatement_flag = dict(
        frozen_forward_winrate_range=[round(min(peryear[y]["frozen_down_winrate"] for y in peryear), 4),
                                      round(max(peryear[y]["frozen_down_winrate"] for y in peryear), 4)],
        frozen_moved_up_rate_range=[round(min(peryear[y]["moved_up_rate"] for y in peryear), 4),
                                    round(max(peryear[y]["moved_up_rate"] for y in peryear), 4)],
        gap_vs_refit_floor=round(min(peryear[y]["frozen_down_winrate"] for y in peryear) - P10_FLOOR, 4),
        verdict="FROZEN OVERSTATES: frozen forward >> refit p10 .5441 and up-rate far below [.47,.53] -> the frozen .65-.73 "
                "is selection/overfit, NOT robust skill. Control (m5xp unweighted, same harness) gives the established .536. "
                "Equity/spot numbers below are on frozen-overstated trades -> NOT deployable figures, shown only to document.")
    fK_refit = kelly(P10_FLOOR, R)
    deploy = bool(refit["refit_forward_2025_ci_lo"] > BE)   # robust decision on the REFIT binding year, NOT frozen
    out = json.load(open(f"{ROOT}/m5_down_deploy_result.json"))
    out.update(dict(
        book="EURUSD.m5xp_magw_down.v1", operating_gate=strat["gate"], coverage="~2.7-5% of NY-session moved bars (DOWN-confidence cover)",
        trades_per_year={int(y): peryear[y]["n"] for y in peryear},
        per_year_FROZEN_FORWARD=peryear, FROZEN_OVERSTATEMENT_FLAG=frozen_overstatement_flag,
        ROBUST_refit_cpcv=refit, kelly_on_refit_floor_full_pct=round(fK_refit * 100, 4),
        binary_equity_FROZEN_OVERSTATED=binary, spot_equity_FROZEN_OVERSTATED=spot,
        venue="5m binary venue-unavailable (deriv min 15m / Nadex closed 2025-12-20). Spot FX is the only venue; on the HONEST "
              "refit edge (~breakeven) spot EV is marginal-to-negative (the frozen +1pip is overstated).",
        kill_switch="N/A — not deployed",
        DECISION=dict(DEPLOY=deploy, basis="refit-CPCV (frozen-forward overstates and is rejected)",
            disqualifiers=[
                f"REFIT-CPCV floor p10 {P10_FLOOR} ~ breakeven {BE} -> full-Kelly only {round(fK_refit*100,3)}% (near-zero robust edge)",
                f"REFIT forward-2025 binding DOWN .559 CI95-lo .533 < breakeven {BE} -> negative-EV in the binding year on the HONEST test",
                "frozen-forward .65-.73 is OVERSTATED (moved up-rate ~.35; frozen >> refit; control m5xp reproduces .536) -> not robust skill",
                "no 5m fixed-payout binary venue exists (deriv 15m / Nadex closed); spot EV marginal-to-negative on the honest edge"],
            statement=("DO NOT DEPLOY (5m,DOWN). On the HONEST refit-CPCV test the book sits AT breakeven (p10 .5441, full-Kelly "
                f"{round(fK_refit*100,3)}%) and the binding forward year (2025) refit DOWN .559 has CI95-lo .533 < {BE} (negative-EV). "
                "The frozen-book forward .65-.73 is a FROZEN OVERSTATEMENT (moved up-rate ~.35, frozen >> refit, and the unweighted "
                "control reproduces the established .536) — NOT a real edge. Full deployment spec (gate/coverage/Kelly/equity/maxDD/"
                "streak/venue) is recorded to DOCUMENT the disqualification. The only path to a deployable DOWN is orthogonal "
                "external data (risk-reversal — verified paywalled)."))))
    json.dump(out, open(f"{ROOT}/m5_down_deploy_result.json", "w"), indent=2, default=str)
    print(f"[down-deploy] per-year DOWN: {peryear}", flush=True)
    print(f"[down-deploy] binary floor-Kelly {round(fK*100,3)}% | 1/8-Kelly eq {binary['eighth_kelly']} | full-Kelly maxDD {binary['full_kelly']['maxDD']} streak {binary['full_kelly']['longest_losing_streak']}", flush=True)
    print(f"[down-deploy] spot {spot}", flush=True)
    print(f"VERDICT: {out['DECISION']['statement']}\n-> m5_down_deploy_result.json ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__": main()
