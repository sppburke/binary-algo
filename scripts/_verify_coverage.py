"""Independent reproduction of the auditor's coverage-transferability claim.
Computes realized coverage of the frozen conf_thr on VAL/TEST/OOS gated subsets,
mirroring the EXACT gating logic in min1_production.py / min2_production.py."""
import numpy as np, importlib, sys

def run(mod_name):
    m = importlib.import_module(mod_name)
    p, L, G, C, S = m._load()
    thr = p["conf_thr"]
    print(f"\n========== {mod_name}  conf_thr={thr:.9f}  cov(target)={p['cov']} ==========")
    is_min1 = (mod_name == "min1_production")
    rel_key = "rel_tighten" if is_min1 else "rel_p70"
    for sp in ("val", "test", "oos"):
        b = m.load_split(sp)
        X, y, mag, valid, ts, idx = m.prep(b)
        pr = m._blend(p, L, G, C, S, X)
        bbw = X["bbw1800"].values; rel = X["rel_ratio"].values; r300 = X["ret300"].values
        # gate EXACTLY as in code (test/oos path); for val replicate train-time gate
        gate = valid & (bbw <= p["bbw1800_q67"]) & (rel >= p[rel_key]) & (np.sign(pr-0.5) == -np.sign(r300))
        conf = np.abs(pr - 0.5)
        gated_n = int(gate.sum())
        if gated_n == 0:
            print(f"  {sp:5s}: gated_n=0")
            continue
        gconf = conf[gate]
        realized_cov = float((gconf >= thr).mean())
        n_pass = int((gconf >= thr).sum())
        print(f"  {sp:5s}: gated_n={gated_n:>7}  realized_cov={realized_cov:.4f}  n>=thr(raw)={n_pass}")
    sys.stdout.flush()

if __name__ == "__main__":
    run("min1_production")
    run("min2_production")
