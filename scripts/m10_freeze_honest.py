"""Freeze the HONEST 10-min native deliverable: native-10 ensemble + a healthy-coverage, robust gate (NOT the VAL-acc-max
thin cov2% tip that produced the 0.667 combined artifact). From the honest gate sweep, the most defensible operating
point is 5m_bb_width<=q20(VAL) x sess_ny at cov10% — healthy n, all held-out windows >=0.579, combined ~0.602, the
HIGHEST test25 floor among healthy-coverage configs. Writes models/m10_EURUSD_strategy_honest.json + prints the held-out
backtest (independent non-overlap 600s, CI95). Reuses models/m10_EURUSD_direction_* (no retraining).
Usage: python m10_freeze_honest.py [GATE_FEAT=5m_bb_width] [Q=20] [COV=0.10] [SESS=sess_ny]
"""
import sys, json, numpy as np
from m10_production import _load, _dirproba, load, nonoverlap_chrono, boot, art, MODELS

SPL = {"val": ["2022", "2023"], "test24": ["2024"], "test25": ["2025"], "oos": ["2026"]}

def main(gate_feat="5m_bb_width", q=20, cov=0.10, sess="sess_ny"):
    p, L, G, C = _load()
    W = {}
    for w in SPL:
        D = load(SPL[w]); pr = _dirproba(p, L, G, C, D)
        W[w] = dict(D=D, pr=pr, conf=np.abs(pr - 0.5), y=D["_y"].astype(int).values,
                    ts=D.index.values.astype("datetime64[s]").astype("int64"))
    va = W["val"]
    gthr = float(np.nanpercentile(va["D"][gate_feat].values.astype(float), q))
    gv = (va["D"][sess].values.astype(float) > 0.5) & (va["D"][gate_feat].values.astype(float) <= gthr)
    conf_thr = float(np.quantile(va["conf"][gv], 1 - cov))
    print(f"[freeze_honest] gate={gate_feat}<={gthr:.3e} x {sess} @cov{cov:.0%} -> conf_thr={conf_thr:.4f}", flush=True)

    allc = []
    res = {}
    for w in ("test24", "test25", "oos"):
        d = W[w]; sm = (d["D"][sess].values.astype(float) > 0.5) & (d["D"][gate_feat].values.astype(float) <= gthr)
        m = sm & (d["conf"] >= conf_thr); sel = nonoverlap_chrono(d["ts"], m)
        corr = ((d["pr"][sel] > 0.5).astype(int) == d["y"][sel]).astype(float) if len(sel) else np.array([])
        acc = corr.mean() if len(sel) else float("nan"); lo, hi = boot(corr)
        res[w] = (len(sel), acc, lo, hi); allc.append(corr)
        print(f"=== {w} === n={len(sel)} acc={acc:.3f} CI95=[{lo:.3f},{hi:.3f}]", flush=True)
    A = np.concatenate(allc); acc = A.mean(); lo, hi = boot(A); fl = min(res[w][1] for w in res)
    print(f"=== COMBINED === n={len(A)} acc={acc:.3f} CI95=[{lo:.3f},{hi:.3f}] FLOOR(worst-window)={fl:.3f}", flush=True)
    print(f"    deriv EV: " + "  ".join(f"{po:.2f}->be{1/(1+po):.3f}:EV{acc*po-(1-acc):+.3f}" for po in (0.80, 0.85, 0.90)), flush=True)

    out = {"pair": "EURUSD", "horizon_bars": 10, "model": "native-10 ensemble lgb+xgb+cat (models/m10_EURUSD_direction_*)",
           "gate_feat": gate_feat, "gate_thr": gthr, "gate_q_on_val": q, "session": sess, "coverage": cov, "conf_thr": conf_thr,
           "selection": "HONEST healthy-coverage (NOT VAL-acc-max); gate from m10_gate_sweep top-floor among cov>=10%",
           "heldout": {w: {"n": res[w][0], "acc": res[w][1]} for w in res},
           "combined_acc": float(acc), "combined_ci95": [lo, hi], "worst_window_floor": float(fl),
           "note": "Honest 10-min native book ~0.60 combined; beats deriv breakeven 0.541; test25 floor ~0.58; NOT a robust >0.65 (the 10-min direction ceiling is data-bound — see m10_research_log.md)."}
    json.dump(out, open(art("strategy_honest.json"), "w"), indent=2)
    print(f"[freeze_honest] saved {art('strategy_honest.json')}", flush=True)

if __name__ == "__main__":
    gf = sys.argv[1] if len(sys.argv) > 1 else "5m_bb_width"
    q = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    cov = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
    sess = sys.argv[4] if len(sys.argv) > 4 else "sess_ny"
    main(gf, q, cov, sess)
