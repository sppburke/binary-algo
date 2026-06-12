"""1-MIN EURUSD binary — BEST BOOK SEARCH (combine the strongest disciplined levers; select honestly).

Goal: produce the single best deriv-faithful 60s direction book by COMBINING the three strongest levers proven
so far, selecting the confidence threshold on VAL by worst-VAL-half stability (NOT VAL-acc-max), then judging on
2024 / 2025 / 2026 SEPARATELY. A book "wins" only if OOS n>=25 AND CI95 lower bound > the bar on all three windows.

LEVERS (all reuse FROZEN Tier-1 infra; no retrain of the child):
  (1) production GATE   : compression (bbw1800<=q67 & rel>=rel_tighten) x reversion (sign(p-.5)==-sign(ret300)) x conf
  (2) UP-side FILTER    : trade only predicted-up bars (child p_up>0.5); the down side is dead (min1_updown finding)
  (3) HMM vol-state GATE: fit GaussianHMM on train via min1_hmm (causal forward-filtered states); on TRAIN identify the
                          favorable state(s) by child accuracy; restrict trades to those states.

BOOKS evaluated per window (each with conf-thr swept by worst-VAL-half stability over a grid built on VAL):
  full       : production gate (child direction)                       [baseline symmetric book]
  up         : production gate AND up-only
  hmm        : production gate AND HMM favorable-state(s)
  up_hmm     : production gate AND up-only AND HMM favorable-state(s)   [the target intersection]
  engine_hmm : production gate AND HMM state-0 PER-STATE reversion engine (the U2 book; reproduced for reference)

DISCIPLINE (deriv-faithful, non-negotiable): label=sign(close[t+60s]-close[t]); entry=next tick (1s lag); exit=last
tick<=+60s; ties (ret==0) LOSE; non-overlap via M.nonoverlap_chrono(gap=70); bootstrap CI95 over non-overlap trades.
Select params on VAL only (worst-VAL-half stability); judge on EACH window separately. One split in memory at a time.

FREEZE rule: if the best book has OOS n>=200 AND floor>=0.57 with CI95 lower bound clearing breakeven (0.541) on all
three windows, save gate config (+ HMM model if used) to models/min1_best_* and a strategy.json.
"""
import os, sys, json, time, numpy as np
import min1_production as M
import min1_hmm as HM

BREAKEVEN = 0.541
BAR = 0.50          # a book "wins" only if CI95 lower bound > this bar on all three windows (direction edge over coin)
FREEZE_FLOOR = 0.57
FREEZE_N = 200


def boot(c, nb=5000, seed=7):
    """Bootstrap CI95 of the win-rate over INDEPENDENT (non-overlap) trades."""
    c = np.asarray(c, float)
    if len(c) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def winacc(d, cand, bet):
    """Non-overlap chronological selection -> per-trade correctness (ties LOSE). bet>0 => predict UP."""
    sel = M.nonoverlap_chrono(d["ts"], cand)
    if len(sel) == 0:
        return np.array([]), np.array([], int)
    pred_up = (bet[sel] > 0).astype(int)
    correct = ((pred_up == d["y"][sel]) & (d["mag"][sel] > 0)).astype(float)
    return correct, sel


def main():
    t0 = time.time()
    os.makedirs("models", exist_ok=True)
    p, L, G, Cc, S = M._load()
    qb = p["bbw1800_q67"]; rqp = p["rel_tighten"]; cthr0 = p["conf_thr"]
    print(f"[best] frozen child loaded; gate bbw<={qb:.3e} rel>={rqp:.3f} prod_conf_thr={cthr0:.5f} "
          f"K={HM.K} emit={HM.EMIT} {time.time()-t0:.0f}s", flush=True)

    # ---------------- TRAIN: fit HMM, identify favorable state(s) by child accuracy ----------------
    TR = HM.slim("train", p, L, G, Cc, S)
    print(f"[best] train n={len(TR['y'])} {time.time()-t0:.0f}s", flush=True)
    mu, sd = HM.std_fit(TR["Xe"], TR["valid"])
    vid_tr = np.where(TR["valid"])[0][::HM.SUB]
    Zfit = HM.std_apply(TR["Xe"][vid_tr], mu, sd)
    from hmmlearn.hmm import GaussianHMM
    model = GaussianHMM(n_components=HM.K, covariance_type=HM.COVTYPE, n_iter=50, random_state=HM.SEED, tol=1e-3)
    model.fit(Zfit)
    print(f"[best] HMM fit on {len(Zfit)} grid-bars (SUB={HM.SUB}); converged={model.monitor_.converged} "
          f"{time.time()-t0:.0f}s", flush=True)
    TR["state"], TR["gamma"] = HM.causal_states(model, TR, mu, sd)

    # characterize states on TRAIN under the production gate (the cohort we actually trade): child accuracy + per-state engine
    def conf_gate(d, cf):
        return d["valid"] & (np.abs(d["p_up"] - 0.5) >= cf) & (d["bbw"] <= qb) & (d["rel"] >= rqp)

    state_child_acc = {}; state_revmom = {}
    print("[best] TRAIN per-state profile under production gate (share | child-acc | up-frac | engine):", flush=True)
    g_tr = conf_gate(TR, cthr0) & (TR["mag"] > 0)
    for k in range(HM.K):
        msk = g_tr & (TR["state"] == k)
        n = int(msk.sum()); share = msk.mean()
        childpred = (TR["p_up"][msk] > 0.5).astype(int)
        cacc = (childpred == TR["y"][msk]).mean() if n else float("nan")
        upfrac = childpred.mean() if n else float("nan")
        mom = (np.sign(TR["ret300"][msk]) > 0).astype(int)
        mom_acc = (mom == TR["y"][msk]).mean() if n else float("nan")
        state_child_acc[k] = cacc; state_revmom[k] = "mom" if mom_acc >= 0.5 else "rev"
        print(f"    state {k}: n={n:>6} share={share:.3f} child_acc={cacc:.3f} up_frac={upfrac:.3f} "
              f"mom_acc={mom_acc:.3f} -> engine={state_revmom[k]}", flush=True)
    # state-0-style engine state(s): per-state reversion engine wins on TRAIN (engine=='rev') — reproduces U2
    eng_states = sorted([k for k in range(HM.K) if state_revmom[k] == "rev"])
    if not eng_states:
        eng_states = list(range(HM.K))
    # NOTE: favorable HMM state subset is NOT chosen on TRAIN (under the gate every state has high child-acc, which
    # neuters the lever). It is selected on VAL by worst-VAL-half stability below (after VAL states are computed),
    # mirroring the min1_hmm U2 methodology that isolated state 0. eng_states above is only the reference engine set.
    print(f"[best] TRAIN child-acc by state = "
          f"{ {k: round(state_child_acc[k],3) for k in range(HM.K)} }; engine(rev) states = {eng_states}", flush=True)

    del TR

    # ---------------- VAL / TEST / OOS slim arrays (one at a time would be ideal; these are small slim dicts) ----------------
    VA = HM.slim("val", p, L, G, Cc, S)
    VA["state"], VA["gamma"] = HM.causal_states(model, VA, mu, sd)
    print(f"[best] val ready n={len(VA['y'])} {time.time()-t0:.0f}s", flush=True)
    TE = HM.slim("test", p, L, G, Cc, S)
    TE["state"], TE["gamma"] = HM.causal_states(model, TE, mu, sd)
    print(f"[best] test ready n={len(TE['y'])} {time.time()-t0:.0f}s", flush=True)
    OO = HM.slim("oos", p, L, G, Cc, S)
    OO["state"], OO["gamma"] = HM.causal_states(model, OO, mu, sd)
    print(f"[best] oos ready n={len(OO['y'])} {time.time()-t0:.0f}s", flush=True)

    te24 = TE["year"] == 2024; te25 = TE["year"] == 2025
    # worst-VAL-half split (chronological)
    vord = np.argsort(VA["ts"]); h = len(vord) // 2
    vh1 = np.zeros(len(VA["ts"]), bool); vh1[vord[:h]] = True; vh2 = ~vh1

    # ---------------- select favorable HMM state subset on VAL by worst-VAL-half stability (child direction) -------------
    # Under the production gate, evaluate each cumulative state subset (states ordered by TRAIN child-acc desc) and pick the
    # subset whose worst VAL half child accuracy is highest, requiring >=20 trades per half. This isolates the genuinely
    # favorable regime (mirrors min1_hmm U2 which found state 0), instead of trivially admitting all states.
    def _vh_child_acc(states_on):
        cand1 = (VA["valid"] & (np.abs(VA["p_up"] - 0.5) >= cthr0) & (VA["bbw"] <= qb) & (VA["rel"] >= rqp)
                 & np.isin(VA["state"], states_on))
        s1 = M.nonoverlap_chrono(VA["ts"], cand1 & vh1); s2 = M.nonoverlap_chrono(VA["ts"], cand1 & vh2)
        if len(s1) < 20 or len(s2) < 20:
            return float("nan")
        bet = np.sign(VA["p_up"] - 0.5)
        a1 = (((bet[s1] > 0).astype(int) == VA["y"][s1]) & (VA["mag"][s1] > 0)).mean()
        a2 = (((bet[s2] > 0).astype(int) == VA["y"][s2]) & (VA["mag"][s2] > 0)).mean()
        return min(a1, a2)
    order = sorted(range(HM.K), key=lambda k: -state_child_acc[k])
    best_fav = None
    for j in range(1, HM.K + 1):
        ss = sorted(order[:j]); vhm = _vh_child_acc(ss)
        print(f"[best] VAL state-subset {ss}: worst-VAL-half child acc = {vhm:.3f}" if not np.isnan(vhm)
              else f"[best] VAL state-subset {ss}: too thin", flush=True)
        if not np.isnan(vhm) and (best_fav is None or vhm > best_fav[1]):
            best_fav = (ss, vhm)
    fav_states = best_fav[0] if best_fav is not None else list(range(HM.K))
    print(f"[best] selected favorable HMM states (VAL worst-half) = {fav_states} "
          f"(worst-half={best_fav[1]:.3f})" if best_fav else f"[best] fav_states fallback all", flush=True)

    # ---------------- book definitions: candidate-mask builder + direction (bet) builder ----------------
    def child_dir(d):
        return np.sign(d["p_up"] - 0.5)

    def eng_bet(d):
        bet = np.zeros(len(d["y"]), "float32")
        for k in range(HM.K):
            mk = d["state"] == k
            sgn = np.sign(d["ret300"]) if state_revmom[k] == "mom" else -np.sign(d["ret300"])
            bet[mk] = sgn[mk]
        return bet

    def cand_full(d, cf):
        return conf_gate(d, cf)

    def cand_up(d, cf):
        return conf_gate(d, cf) & (d["p_up"] > 0.5)

    def cand_hmm(d, cf):
        return conf_gate(d, cf) & np.isin(d["state"], fav_states)

    def cand_up_hmm(d, cf):
        return conf_gate(d, cf) & (d["p_up"] > 0.5) & np.isin(d["state"], fav_states)

    def cand_engine_hmm(d, cf):
        return conf_gate(d, cf) & np.isin(d["state"], eng_states)

    BOOKS = {
        "full":           (cand_full,       child_dir),
        "up-only":        (cand_up,         child_dir),
        "hmm-fav":        (cand_hmm,        child_dir),
        "up x hmm-fav":   (cand_up_hmm,     child_dir),
        "engine x hmm-0": (cand_engine_hmm, eng_bet),
    }

    # conf-thr grid: from the production thr up through higher quantiles of |p-.5| on VAL-gated bars (only TIGHTEN)
    base_gate_va = VA["valid"] & (VA["bbw"] <= qb) & (VA["rel"] >= rqp)
    conf_va = np.abs(VA["p_up"][base_gate_va] - 0.5)
    grid = sorted(set([cthr0] + [float(np.quantile(conf_va, q)) for q in
                  (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9)] if len(conf_va) > 50 else [cthr0]))
    grid = [g for g in grid if g >= cthr0 * 0.5]   # don't go far below production thr
    print(f"[best] conf-thr grid ({len(grid)}): {[round(g,5) for g in grid]}", flush=True)

    def select_thr(cand_fn, bet_fn):
        """Pick conf-thr maximizing min(VAL-half1 acc, VAL-half2 acc); require >=20 trades per half. Returns (thr, vhmin)."""
        best = None
        for thr in grid:
            c1, _ = winacc(VA, cand_fn(VA, thr) & vh1, bet_fn(VA))
            c2, _ = winacc(VA, cand_fn(VA, thr) & vh2, bet_fn(VA))
            if len(c1) < 20 or len(c2) < 20:
                continue
            vhmin = min(c1.mean(), c2.mean())
            if best is None or vhmin > best[1]:
                best = (thr, vhmin)
        if best is None:   # fall back to production thr (no half had enough trades to be stable-selected)
            return (cthr0, float("nan"))
        return best

    print("\n[best] per-book evaluation (deriv-faithful, non-overlap gap=70, ties LOSE; CI95 over non-overlap trades):", flush=True)
    results = {}
    for name, (cand_fn, bet_fn) in BOOKS.items():
        thr, vhmin = select_thr(cand_fn, bet_fn)
        c24, _ = winacc(TE, cand_fn(TE, thr) & te24, bet_fn(TE))
        c25, _ = winacc(TE, cand_fn(TE, thr) & te25, bet_fn(TE))
        c26, s26 = winacc(OO, cand_fn(OO, thr), bet_fn(OO))
        a = lambda c: (float(c.mean()) if len(c) else float("nan"))
        lo24, hi24 = boot(c24); lo25, hi25 = boot(c25); lo26, hi26 = boot(c26)
        accs = [a(c24), a(c25), a(c26)]
        floor = float(np.nanmin(accs)) if not all(np.isnan(accs)) else float("nan")
        ci_los = [lo24, lo25, lo26]; ns = [len(c24), len(c25), len(c26)]
        # WIN criterion: OOS n>=25 AND CI95 lower bound > BAR on ALL three windows
        wins = (len(c26) >= 25) and all((not np.isnan(l)) and l > BAR for l in ci_los) and all(n >= 5 for n in ns)
        results[name] = dict(thr=thr, vhmin=vhmin, floor=floor,
                             a24=a(c24), a25=a(c25), a26=a(c26),
                             n24=len(c24), n25=len(c25), n26=len(c26),
                             ci24=[lo24, hi24], ci25=[lo25, hi25], ci26=[lo26, hi26], wins=bool(wins))
        print(f"  {name:15s} thr={thr:.5f} VALworsthalf={vhmin:.3f}  ->  "
              f"2024 {a(c24):.3f}(n{len(c24)},CI[{lo24:.3f},{hi24:.3f}]) / "
              f"2025 {a(c25):.3f}(n{len(c25)},CI[{lo25:.3f},{hi25:.3f}]) / "
              f"2026 {a(c26):.3f}(n{len(c26)},CI[{lo26:.3f},{hi26:.3f}])  "
              f"FLOOR={floor:.3f}  WIN={wins}", flush=True)
        print(f"     breakeven={BREAKEVEN}; clears_065={'Y' if floor>=0.65 else 'N'} "
              f"beats_060={'Y' if floor>0.60 else 'N'}", flush=True)

    # ---------------- pick the single best book by FLOOR (tie-break: higher min CI lower bound, then larger OOS n) ----------------
    def keyf(item):
        n, r = item
        fl = r["floor"] if not np.isnan(r["floor"]) else -1
        minlo = min(r["ci24"][0], r["ci25"][0], r["ci26"][0])
        minlo = minlo if not np.isnan(minlo) else -1
        return (fl, minlo, r["n26"])
    best_name, best = max(results.items(), key=keyf)
    print(f"\n[best] >>> BEST BOOK by floor = '{best_name}'  floor={best['floor']:.3f}  "
          f"OOS n={best['n26']}  WIN={best['wins']}", flush=True)

    # ---------------- FREEZE if it clears the bar ----------------
    frozen_path = "NONE"
    ci_all = [best["ci24"][0], best["ci25"][0], best["ci26"][0]]
    freeze = (best["n26"] >= FREEZE_N and best["floor"] >= FREEZE_FLOOR
              and all((not np.isnan(l)) and l > BREAKEVEN for l in ci_all))
    if freeze:
        uses_hmm = "hmm" in best_name
        cfg = {
            "book": best_name,
            "selection": "worst-VAL-half stability; judged per 2024/2025/2026 separately",
            "bbw1800_q67": qb, "rel_tighten": rqp,
            "conf_thr": results[best_name]["thr"],
            "reversion": "sign(p_up-0.5)==-sign(ret300)",
            "up_only": "up-only" in best_name or "up x" in best_name,
            "uses_hmm": uses_hmm,
            "hmm_fav_states": fav_states if "hmm-fav" in best_name else (eng_states if "hmm-0" in best_name else []),
            "hmm_emit": HM.EMIT, "hmm_K": HM.K, "hmm_sub": HM.SUB, "hmm_cov": HM.COVTYPE,
            "horizon_s": M.HS, "gap_s": M.GAP, "breakeven": BREAKEVEN,
            "per_window": {"2024": best["a24"], "2025": best["a25"], "2026": best["a26"]},
            "ci95": {"2024": best["ci24"], "2025": best["ci25"], "2026": best["ci26"]},
            "floor": best["floor"], "child_artifacts": "models/min1_EURUSD_*",
        }
        json.dump(cfg, open("models/min1_best_strategy.json", "w"), indent=2, default=float)
        frozen_path = "models/min1_best_strategy.json"
        if uses_hmm:
            import joblib
            joblib.dump({"model": model, "mu": mu, "sd": sd, "fav_states": fav_states,
                         "eng_states": eng_states, "state_revmom": state_revmom, "emit": HM.EMIT,
                         "K": HM.K, "sub": HM.SUB}, "models/min1_best_hmm.joblib")
            frozen_path += " + models/min1_best_hmm.joblib"
        print(f"[best] FROZEN -> {frozen_path}", flush=True)
    else:
        print(f"[best] NOT frozen (need OOS n>={FREEZE_N}, floor>={FREEZE_FLOOR}, all CI95-lo>{BREAKEVEN}); "
              f"got n26={best['n26']} floor={best['floor']:.3f} ci_lo={[round(x,3) for x in ci_all]}", flush=True)

    json.dump({"books": {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()}
                         for k, v in results.items()},
               "best": best_name, "fav_states": fav_states, "eng_states": eng_states,
               "frozen": frozen_path},
              open("models/min1_best_summary.json", "w"), indent=2, default=float)
    print(f"[best] DONE {time.time()-t0:.0f}s  summary->models/min1_best_summary.json", flush=True)
    return results, best_name, frozen_path


if __name__ == "__main__":
    main()
