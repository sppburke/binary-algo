"""Adversarial verification: sweep conf_thr at multiples of the frozen value
using the frozen models, reproducing backtest selection logic exactly."""
import sys, json, numpy as np
sys.argv = ["x"]  # prevent module __main__ from running anything
import importlib.util

def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

ROOT = "/home/sean/git/binary-algo"

def run(mod, gate_rel_key):
    p, L, G, C, S = mod._load()
    base_thr = p["conf_thr"]
    out = {}
    for sp in ("test", "oos"):
        b = mod.load_split(sp)
        X, y, mag, valid, ts, idx = mod.prep(b)
        pr = mod._blend(p, L, G, C, S, X)
        bbw = X["bbw1800"].values; rel = X["rel_ratio"].values; r300 = X["ret300"].values
        gate = valid & (bbw <= p["bbw1800_q67"]) & (rel >= p[gate_rel_key]) & (np.sign(pr-0.5) == -np.sign(r300))
        conf = np.abs(pr - 0.5)
        rows = []
        for mult in (0.5, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5, 2.0):
            thr = base_thr * mult
            cand = gate & (conf >= thr)
            tr = mod.nonoverlap(ts, np.where(cand, conf, -1.0), 0.0)
            tr = tr[cand[tr]]
            if len(tr):
                correct = ((pr[tr] > 0.5).astype(int) == y[tr]).astype(float)
                acc = correct.mean()
            else:
                acc = float("nan")
            rows.append((mult, len(tr), acc))
        out[sp] = rows
    return base_thr, out

m1 = load_mod(f"{ROOT}/min1_production.py", "min1_production")
m2 = load_mod(f"{ROOT}/min2_production.py", "min2_production")

for name, mod, relkey in (("MIN1", m1, "rel_tighten"), ("MIN2", m2, "rel_p70")):
    base, out = run(mod, relkey)
    print(f"\n===== {name}  base_conf_thr={base:.6f} =====")
    for sp in ("test", "oos"):
        print(f"  {sp.upper()}:")
        for mult, n, acc in out[sp]:
            print(f"    x{mult:<5} thr={base*mult:.6f}  n={n:<6} acc={acc:.4f}")
