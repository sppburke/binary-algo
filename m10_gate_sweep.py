"""10-MIN gate optimization for the native-10 ensemble (reuses models/m10_EURUSD_*; cheap inference-only).

The 15m winner only ever gated on 15m_bb_width x sess_ny. For a 10-min horizon other compression timeframes
(5m/30m_bb_width, atr) and other sessions (overlap/london) are untested and may align better. Sweep
gate=(compression_feat<=q) AND session, selective by confidence; SELECT on VAL by worst-year-half stability
(NOT VAL-acc-max), then VERIFY held-out across {test24,test25,oos} with CI95 + chronological non-overlap 600s.
Usage: python m10_gate_sweep.py
"""
import sys, numpy as np
import m10_production as M10
from m10_production import nonoverlap_chrono, boot, _load, _dirproba, load

SPL = {"val": ["2022", "2023"], "test24": ["2024"], "test25": ["2025"], "oos": ["2026"]}
COMP = ["5m_bb_width", "15m_bb_width", "30m_bb_width", "5m_atr_pct", "15m_atr_pct"]
SESS = {"ny": ["sess_ny"], "overlap": ["sess_overlap"], "london": ["sess_london"], "ny|overlap": ["sess_ny", "sess_overlap"]}

def session_mask(D, keys):
    m = np.zeros(len(D), bool)
    for k in keys: m |= D[k].values.astype(float) > 0.5
    return m
def yrof(ts): return (np.asarray(ts, dtype="datetime64[s]").astype("datetime64[Y]").astype(int) + 1970)

def main():
    p, L, G, C = _load()
    W = {}
    for w in SPL:
        D = load(SPL[w])
        pr = _dirproba(p, L, G, C, D)
        W[w] = dict(D=D, pr=pr, conf=np.abs(pr - 0.5), y=D["_y"].astype(int).values,
                    ts=D.index.values.astype("datetime64[s]").astype("int64"))
    va = W["val"]; vyr = yrof(va["ts"])
    print(f"[gate_sweep] native-10 loaded; VAL n={len(va['y']):,}", flush=True)
    qgrid = {c: {q: float(np.nanpercentile(va["D"][c].values.astype(float), q)) for q in (10, 20, 33, 50)} for c in COMP}

    rows = []
    for comp in COMP:
        for sname, skeys in SESS.items():
            sv = session_mask(va["D"], skeys)
            for q in (10, 20, 33, 50):
                gv = sv & (va["D"][comp].values.astype(float) <= qgrid[comp][q])
                if gv.sum() < 400: continue
                for cov in (0.10, 0.05, 0.03):
                    confv = va["conf"][gv]
                    thr = float(np.quantile(confv, 1 - cov))
                    # VAL worst-year-half stability
                    accs = []
                    ok = True
                    for yy in sorted(set(vyr.tolist())):
                        mm = gv & (vyr == yy) & (va["conf"] >= thr); sel = nonoverlap_chrono(va["ts"], mm)
                        if len(sel) < 25: ok = False; break
                        accs.append((va["pr"][sel] > 0.5).astype(int).__eq__(va["y"][sel]).mean())
                    if not ok: continue
                    hm = min(accs)
                    # held-out verify
                    res = {}
                    for w in ("test24", "test25", "oos"):
                        d = W[w]; sm = session_mask(d["D"], skeys) & (d["D"][comp].values.astype(float) <= qgrid[comp][q])
                        m = sm & (d["conf"] >= thr); sel = nonoverlap_chrono(d["ts"], m)
                        if len(sel) == 0: res[w] = (0, float("nan"), np.array([])); continue
                        corr = ((d["pr"][sel] > 0.5).astype(int) == d["y"][sel]).astype(float)
                        res[w] = (len(sel), float(corr.mean()), corr)
                    if any(res[w][0] < 10 for w in res): continue
                    A = np.concatenate([res[w][2] for w in res]); lo, hi = boot(A); fl = min(res[w][1] for w in res)
                    rows.append(dict(comp=comp, sess=sname, q=q, cov=cov, hm=hm, res=res, fl=fl,
                                     comb=(len(A), A.mean(), lo, hi)))

    if not rows:
        print("[gate_sweep] no eligible configs"); return
    # HONEST: pick the worst-VAL-half-stable config (require hm>=0.55), tie-break by held-out floor
    elig = [r for r in rows if r["hm"] >= 0.55]
    print(f"\n[gate_sweep] {len(rows)} configs, {len(elig)} with VAL-half-min>=0.55", flush=True)
    def show(tag, r):
        print(f"{tag} comp={r['comp']} sess={r['sess']} q{r['q']} cov{r['cov']:.0%} | VALhmin={r['hm']:.3f} -> "
              f"t24 {r['res']['test24'][1]:.3f}(n{r['res']['test24'][0]}) t25 {r['res']['test25'][1]:.3f}(n{r['res']['test25'][0]}) "
              f"oos {r['res']['oos'][1]:.3f}(n{r['res']['oos'][0]}) FLOOR={r['fl']:.3f} COMB {r['comb'][1]:.3f}"
              f"[{r['comb'][2]:.3f},{r['comb'][3]:.3f}]n{r['comb'][0]}", flush=True)
    if elig:
        b = max(elig, key=lambda r: r["hm"]); show("[HONEST worst-half-stable]", b)
    o = max(rows, key=lambda r: r["fl"]); show("[ORACLE max-floor    ]", o)
    ver = [r for r in rows if r["res"]["oos"][0] >= 80]
    if ver:
        bv = max(ver, key=lambda r: r["fl"]); show("[BEST VERIFIABLE oosn>=80]", bv)
    # top-10 by held-out floor for the log
    print("\n[gate_sweep] top-10 configs by held-out FLOOR:", flush=True)
    for r in sorted(rows, key=lambda r: -r["fl"])[:10]:
        show("  ", r)

if __name__ == "__main__":
    main()
