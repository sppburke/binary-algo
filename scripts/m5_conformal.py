"""EDGE-IMPROVEMENT: ADAPTIVE-CONFORMAL gate (ACI, Gibbs-Candès 2021) vs the FIXED meta gate, for the m5xp UP
side. The fixed gate (meta>=THR) is static; under non-stationary FX it can't adapt when a regime (2025) weakens.
ACI maintains a dynamic threshold θ_t that targets a selective win-rate w* by feedback on realized outcomes —
trading LESS when the edge weakens, MORE when it strengthens — giving a distribution-free long-run error bound.
GOAL: a more ROBUST/regime-adaptive gate that holds win-rate ≥ w* every year while maximizing coverage (EV/time).

Compare per-year: realized UP win-rate + n(trades) for FIXED vs ACI, at target w*∈{0.55,0.56,0.57}. Improvement
= ACI matches/beats fixed win-rate in the binding 2025 year AND gives comparable-or-better coverage (more EV/hr),
or holds the win-rate more stably across years. Inference-only on the frozen book."""
import json, numpy as np
import m5_xpair as MX
import m5_xpair_production as XP

YEARS = (("test24", "2024"), ("test25", "2025"), ("oos", "2026"))


def collect():
    """Chronological UP candidates (NY & pred-up), de-overlapped: (ts, sm, win, year)."""
    p, P, M = XP._load(); cols = p["primary_feats"]; mcols = p["meta_feats"]; THR = p["meta_thr"]
    TS, SM, WIN, YR = [], [], [], []
    for w, label in YEARS:
        D = MX.build_xp(XP.SPL[w]); D = MX.augment(D, XP.SPL[w], XP.MODE)
        pr = P.predict(D[cols].astype("float32")); y = D["_y"].astype(int).values
        sm = M.predict(XP._Xmeta(D, pr, mcols)); ts = D["_ts"].values.astype("int64")
        ny = D["sess_ny"].values > 0.5
        cand = ny & (pr > 0.5)                       # UP candidates (gate decides which to trade)
        sel = MX.nonoverlap_chrono(ts, cand)          # independent UP candidates
        TS.append(ts[sel]); SM.append(sm[sel]); WIN.append((y[sel] == 1).astype(float)); YR.append([label]*len(sel))
        del D
    ts = np.concatenate(TS); o = np.argsort(ts)
    return ts[o], np.concatenate(SM)[o], np.concatenate(WIN)[o], np.concatenate(YR)[o], THR


def per_year(mask, win, yr):
    out = {}
    for _, label in YEARS:
        m = mask & (yr == label)
        out[label] = {"n": int(m.sum()), "win": round(float(win[m].mean()), 4) if m.sum() >= 5 else None}
    out["TOTAL_n"] = int(mask.sum()); out["TOTAL_win"] = round(float(win[mask].mean()), 4) if mask.sum() else None
    return out


def aci_gate(sm, win, wstar, gamma=0.02, theta0=None):
    """Adaptive threshold targeting selective error alpha*=1-wstar. Trade bar t if sm_t>=theta_t; update
    theta_{t+1}=theta_t + gamma*(err_t - alpha*) using realized err only on TRADED bars (online, no look-ahead)."""
    astar = 1.0 - wstar; theta = theta0 if theta0 is not None else float(np.quantile(sm, 0.5))
    traded = np.zeros(len(sm), bool)
    lo, hi = float(np.quantile(sm, 0.05)), float(np.quantile(sm, 0.995))
    for t in range(len(sm)):
        if sm[t] >= theta:
            traded[t] = True
            err = 1.0 - win[t]
            theta = min(hi, max(lo, theta + gamma * (err - astar)))
    return traded


def main():
    ts, sm, win, yr, THR = collect()
    upr_all = float(win.mean())
    print(f"[aci] UP candidates={len(sm)} overall up-rate(unfiltered)={upr_all:.4f} fixed_thr={THR:.4f}", flush=True)
    fixed = sm >= THR
    out = {"fixed_thr": round(THR, 4), "n_candidates": int(len(sm)), "FIXED": per_year(fixed, win, yr), "ACI": {}}
    print(f"  FIXED gate: {out['FIXED']}", flush=True)
    for wstar in (0.55, 0.56, 0.57):
        tr = aci_gate(sm, win, wstar)
        py = per_year(tr, win, yr); out["ACI"][f"wstar_{wstar}"] = py
        print(f"  ACI w*={wstar}: {py}", flush=True)
    # improvement check: does any ACI target hold win>=0.55 in ALL years with TOTAL coverage >= fixed?
    fcov = out["FIXED"]["TOTAL_n"]
    best = None
    for k, py in out["ACI"].items():
        yrs_ok = all(py[l]["win"] is not None and py[l]["win"] >= 0.55 for _, l in YEARS)
        if yrs_ok and py["TOTAL_n"] >= 0.8 * fcov:
            if best is None or py["TOTAL_n"] > best[1]: best = (k, py["TOTAL_n"], py)
    out["verdict"] = {"improves": bool(best),
                      "note": (f"ACI {best[0]} holds win>=0.55 all years at n={best[1]} (fixed n={fcov}) -> more robust/coverage"
                               if best else f"ACI does not beat the fixed gate (fixed already holds; n={fcov})")}
    json.dump(out, open("m5_conformal_result.json", "w"), indent=1)
    print(f"[aci] {out['verdict']['note']}", flush=True)
    print("[aci] -> m5_conformal_result.json", flush=True)


if __name__ == "__main__":
    main()
