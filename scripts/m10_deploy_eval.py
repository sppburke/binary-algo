"""10m cross-pair book DEPLOY EVAL — step (d) coverage curve + I1 ACI gate + deriv EV, per side, per year.
Inference-only on the frozen EURUSD.m10xp.v1 primary (no retrain). Three things in one light job:

(d) COVERAGE CURVE: fixed gate (bb_width<=thr & NY) + confidence cover swept cov in {2,5,10,15,20}% (conf_thr
    derived from VAL gated confidence at each cov, applied per year), per-side win-rate + n + deriv EV@R0.85.
(I1) ACI GATE (Gibbs-Candes, aci_gate replicated from m5_conformal.py): adaptive confidence threshold targeting
    win-rate w* in {.55,.57,.59} on the chronological gated per-side stream; compare per-year win+n vs the fixed
    cov10% gate. IMPROVEMENT iff ACI holds win >= target in ALL years at coverage >= ~fixed (deploy objective:
    win-rate AND coverage/EV), esp. firming the binding 2025 year.
EV = deriv ties-lose binary, NO spread (mid-to-mid): EV/bet = win*R - (1-win), R=0.85 (breakeven 0.541).

PRE-REGISTERED FALSIFIER (I1 ACI): KILL unless ACI holds per-year win-rate >= fixed-gate's at >= 0.9x its
coverage in the binding 2025 year on >=1 side (i.e. strictly better EV/time or strictly better regime-robustness).
Usage: python m10_deploy_eval.py
"""
import os
os.environ["MX_HOR"] = "10"
import json, numpy as np
import lightgbm as lgb
import m5_xpair as MX

BOOK = "/home/sean/git/binary-algo/books/EURUSD.m10xp.v1"
BE = 0.541; R = 0.85
COVS = (0.02, 0.05, 0.10, 0.15, 0.20)
WSTARS = (0.55, 0.57, 0.59)


def ev(win):
    return None if win is None else round(win * R - (1 - win), 4)


def aci_gate(conf, win, wstar, gamma=0.02):
    """Adaptive threshold on confidence targeting selective error alpha*=1-wstar (Gibbs-Candes, online, no
    look-ahead): trade bar t if conf_t>=theta_t; theta_{t+1}=theta_t+gamma*(err_t-alpha*) on traded bars only."""
    astar = 1.0 - wstar; theta = float(np.quantile(conf, 0.5))
    lo, hi = float(np.quantile(conf, 0.05)), float(np.quantile(conf, 0.995))
    traded = np.zeros(len(conf), bool)
    for t in range(len(conf)):
        if conf[t] >= theta:
            traded[t] = True
            theta = min(hi, max(lo, theta + gamma * ((1.0 - win[t]) - astar)))
    return traded


def main():
    strat = json.load(open(f"{BOOK}/m10xp_EURUSD_strategy.json"))
    cols = strat["primary_feats"]; GATE = strat.get("gate_feat", "5m_bb_width")
    bthr = strat["bb_width_thr"]; conf_thr_dep = strat["conf_thr"]; cov_dep = strat["coverage"]
    B = lgb.Booster(model_file=f"{BOOK}/m10xp_EURUSD_primary_lgb.txt")
    print(f"[deploy] book gate {GATE}<={bthr:.2e} x NY x conf>={conf_thr_dep:.4f} (deployed cov{cov_dep:.0%})", flush=True)

    # build VAL + each year; cache predictions on the NY x compression gate
    frames = {}
    for w, yr in (("val", "VAL"), ("test24", "2024"), ("test25", "2025"), ("oos", "2026")):
        D = MX.augment(MX.build_xp(MX.SPL[w]), MX.SPL[w], "xpof")
        pr = B.predict(D[cols].astype("float32").values); y = D["_y"].astype(int).values
        ts = D["_ts"].values.astype("int64"); bbw = D[GATE].values.astype(float); ny = D["sess_ny"].values > 0.5
        g = (bbw <= bthr) & ny
        frames[yr] = dict(pr=pr, y=y, ts=ts, g=g)
        del D
    confv = np.abs(frames["VAL"]["pr"] - 0.5); gv = frames["VAL"]["g"]

    # ---- (d) COVERAGE CURVE: conf_thr from VAL at each cov, applied per year, per side ----
    curve = {}
    for cov in COVS:
        cthr = float(np.quantile(confv[gv], 1 - cov))
        row = {"conf_thr": round(cthr, 4)}
        for yr in ("2024", "2025", "2026"):
            f = frames[yr]; conf = np.abs(f["pr"] - 0.5)
            m = f["g"] & (conf >= cthr); sel = MX.nonoverlap_chrono(f["ts"], m)
            pred = (f["pr"][sel] > 0.5).astype(int); yy = f["y"][sel]
            d = {}
            for side, nm in ((1, "UP"), (0, "DOWN"), (None, "ALL")):
                ss = (pred == side) if side is not None else np.ones(len(pred), bool)
                if ss.sum() < 5:
                    d[nm] = {"n": int(ss.sum()), "win": None}; continue
                win = float((pred[ss] == yy[ss]).mean())
                d[nm] = {"n": int(ss.sum()), "win": round(win, 4), "ev": ev(win)}
            row[yr] = d
        curve[f"cov{int(cov*100)}"] = row
        print(f"  cov{int(cov*100)}%: " + " ".join(f"{yr}[U {row[yr]['UP'].get('win')}/{row[yr]['UP']['n']} D {row[yr]['DOWN'].get('win')}/{row[yr]['DOWN']['n']}]" for yr in ('2024','2025','2026')), flush=True)

    # ---- (I1) ACI vs FIXED (cov10%), per side, chronological pooled stream ----
    aci_out = {}
    for side, nm in ((1, "UP"), (0, "DOWN")):
        # chronological per-side candidate stream across years, tagged by year
        TS, CONF, WIN, YR = [], [], [], []
        for yr in ("2024", "2025", "2026"):
            f = frames[yr]; conf = np.abs(f["pr"] - 0.5)
            cand = f["g"] & ((f["pr"] > 0.5) == (side == 1))
            sel = MX.nonoverlap_chrono(f["ts"], cand)
            pred = (f["pr"][sel] > 0.5).astype(int)
            TS.append(f["ts"][sel]); CONF.append(conf[sel]); WIN.append((pred == f["y"][sel]).astype(float)); YR.append([yr]*len(sel))
        ts = np.concatenate(TS); o = np.argsort(ts)
        conf = np.concatenate(CONF)[o]; win = np.concatenate(WIN)[o]; yrarr = np.concatenate(YR)[o]
        # fixed gate at deployed conf_thr
        fixed = conf >= conf_thr_dep
        def py(mask):
            r = {}
            for yr in ("2024", "2025", "2026"):
                mm = mask & (yrarr == yr)
                r[yr] = {"n": int(mm.sum()), "win": round(float(win[mm].mean()), 4) if mm.sum() >= 5 else None}
            r["TOT_n"] = int(mask.sum()); r["TOT_win"] = round(float(win[mask].mean()), 4) if mask.sum() else None
            return r
        side_out = {"FIXED_cov10": py(fixed)}
        for wstar in WSTARS:
            side_out[f"ACI_w{wstar}"] = py(aci_gate(conf, win, wstar))
        aci_out[nm] = side_out
        print(f"  [{nm}] FIXED {side_out['FIXED_cov10']} | ACI.57 {side_out['ACI_w0.57']}", flush=True)

    # ---- I1 verdict: does ACI beat fixed on binding 2025 (win>=fixed at >=0.9x cov) either side? ----
    improves = {}
    for nm in ("UP", "DOWN"):
        fx = aci_out[nm]["FIXED_cov10"]; fwin25 = fx["2025"]["win"]; fn25 = fx["2025"]["n"]
        win_any = False
        for wstar in WSTARS:
            a = aci_out[nm][f"ACI_w{wstar}"]; aw = a["2025"]["win"]; an = a["2025"]["n"]
            if aw is not None and fwin25 is not None and aw >= fwin25 and an >= 0.9 * max(fn25, 1) and an >= 20:
                win_any = True
        improves[nm] = bool(win_any)
    out = {"book": "EURUSD.m10xp.v1", "breakeven": BE, "payout_R": R, "deployed_gate": {"cov": cov_dep, "conf_thr": conf_thr_dep},
           "coverage_curve": curve, "aci_vs_fixed": aci_out,
           "I1_verdict": {"ACI_improves_UP": improves["UP"], "ACI_improves_DOWN": improves["DOWN"],
                          "note": "ACI improves a side iff it holds 2025 win>=fixed at >=0.9x coverage (deploy EV/regime objective)"},
           "PRE_REGISTERED_FALSIFIER": "KILL I1 ACI unless it holds binding-2025 win>=fixed at >=0.9x coverage on >=1 side"}
    json.dump(out, open("m10_deploy_eval_result.json", "w"), indent=1)
    print(f"[deploy] I1 ACI verdict: UP_improves={improves['UP']} DOWN_improves={improves['DOWN']} -> m10_deploy_eval_result.json", flush=True)


if __name__ == "__main__":
    main()
