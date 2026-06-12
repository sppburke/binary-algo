"""1-MIN HURST / VARIANCE-RATIO PERSISTENCE SWITCH (rank-2 lever) — attack the binding 2025 wall directly.

Thesis (sofien 'Rescaled Range' + our finding that 60s is REVERSION not continuation): the 2025 regime is a price-MEMORY
inversion. A causal persistence measure (variance-ratio VR(q,W)=Var_W(r_q)/(q*Var_W(r_1)); VR>1 ⇔ Hurst H>0.5 persistent/
trending, VR<1 ⇔ anti-persistent/mean-reverting) lets us bet the MOMENTUM engine where the path is persistent, the REVERSION
engine where it is anti-persistent, and ABSTAIN near the random-walk state (VR≈1). Orthogonal to the volatility-compression
axis the book already gates (magnitude, not sign). No new direction net (V4/V6/V13/V14 proved 60s direction nets are ~0.51 AUC).

Engines (from the existing book): E_mom=sign(ret300); E_rev=-sign(ret300) (the lever production monetizes); E_child=sign(p_up-0.5).
Switch: VR>1+m -> persistent engine; VR<1-m -> anti-persistent engine; else ABSTAIN.
DISCIPLINE: VR params (q,W,m)+conf+gate picked on VAL (2024-H1) by worst-VAL-half stability; judged on EACH of {2024(test 09-12),
2025(test), 2026(oos)} SEPARATELY with deriv-faithful 60s settlement (ties LOSE) + non-overlap + CI95. Memory-safe: one split at a
time; keep only slim arrays. Reuses min1_production._load/_blend/feats/prep/wc_ret/nonoverlap_chrono UNCHANGED (child trained 2021-23).
"""
import sys, json, time, numpy as np, pandas as pd
import min1_production as M

BREAKEVEN = 0.541
QS = [10, 30, 60]; WS = [300, 600]; MS = [0.0, 0.15, 0.30]
PAIRS = [("mom", "rev"), ("mom", "child"), ("child", "rev")]

def persistence(mid, q, W):
    """Causal variance-ratio VR(q,W). >1 persistent/trending, <1 mean-reverting."""
    lm = np.log(np.asarray(mid, float)); s = pd.Series(lm)
    v1 = s.diff().rolling(W, min_periods=W // 2).var()
    vq = s.diff(q).rolling(W, min_periods=W // 2).var()
    return (vq / (q * v1 + 1e-18)).values

def slim_split(sp, p, L, G, Cc, S):
    """Process ONE split; return slim per-bar arrays + a {(q,W):VR} cache. Frees the heavy buffer/feature frame."""
    b = M.load_split(sp); X, y, mag, valid, ts, idx = M.prep(b)
    pr = M._blend(p, L, G, Cc, S, X)
    d = dict(p_up=pr.astype("float32"), ret300=X["ret300"].values.astype("float32"),
             bbw=X["bbw1800"].values.astype("float32"), rel=X["rel_ratio"].values.astype("float32"),
             y=y.astype("int8"), mag=mag.astype("float32"), valid=valid, ts=ts, year=idx.year.values.astype("int16"))
    mid = b["mid"].values.astype(float)
    vr = {(q, W): persistence(mid, q, W).astype("float32") for q in QS for W in WS}
    del b, X, mid
    return d, vr

def bet_dir(engine, d):
    if engine == "mom": return np.sign(d["ret300"])
    if engine == "rev": return -np.sign(d["ret300"])
    return np.sign(d["p_up"] - 0.5)

def gate_cand(d, vr, m, cf, gate, pe, ae):
    conf = np.abs(d["p_up"] - 0.5); reg = d["valid"].copy()
    if gate is not None:
        qb, rq_ = gate; reg &= (d["bbw"] <= qb) & (d["rel"] >= rq_)
    reg &= (conf >= cf)
    bet = np.zeros(len(vr), "float32")
    persist = vr > (1.0 + m); anti = vr < (1.0 - m)
    bet[persist] = bet_dir(pe, d)[persist]; bet[anti] = bet_dir(ae, d)[anti]
    cand = reg & np.isfinite(vr) & (bet != 0)
    return cand, bet

def select(d, vr, m, cf, gate, pe, ae, submask=None):
    """Return (correct_array, sel_original_indices). Optional submask restricts candidates (year/half)."""
    cand, bet = gate_cand(d, vr, m, cf, gate, pe, ae)
    if submask is not None: cand = cand & submask
    sel = M.nonoverlap_chrono(d["ts"], cand)
    if len(sel) == 0: return np.array([]), np.array([], int)
    pred_up = (bet[sel] > 0).astype(int)
    correct = ((pred_up == d["y"][sel]) & (d["mag"][sel] > 0)).astype(float)
    return correct, sel

def main():
    t0 = time.time()
    p, L, G, Cc, S = M._load()
    qb = p["bbw1800_q67"]; rqp = p["rel_tighten"]; cthr = p["conf_thr"]
    GATES = [("comp", (qb, rqp)), ("nocomp", None)]; CONFS = [0.0, cthr]
    print(f"[hurst] child val_auc_inregime={p.get('val_auc_inregime'):.3f}; comp bbw<={qb:.2e} rel>={rqp:.3f} conf_thr={cthr:.4f} {time.time()-t0:.0f}s", flush=True)
    VA, vrVA = slim_split("val", p, L, G, Cc, S); print(f"[hurst] val ready n={len(VA['y'])} {time.time()-t0:.0f}s", flush=True)
    TE, vrTE = slim_split("test", p, L, G, Cc, S); print(f"[hurst] test ready n={len(TE['y'])} {time.time()-t0:.0f}s", flush=True)
    OO, vrOO = slim_split("oos", p, L, G, Cc, S); print(f"[hurst] oos ready n={len(OO['y'])} {time.time()-t0:.0f}s", flush=True)
    # VAL chronological halves; TEST year masks
    vord = np.argsort(VA["ts"]); h = len(vord) // 2
    vh1 = np.zeros(len(VA["ts"]), bool); vh1[vord[:h]] = True; vh2 = ~vh1
    te24 = TE["year"] == 2024; te25 = TE["year"] == 2025
    print(f"\n[hurst] test 2024 bars={int(te24.sum())} 2025 bars={int(te25.sum())}; sweeping {len(PAIRS)*len(QS)*len(WS)*len(MS)*len(CONFS)*len(GATES)} configs", flush=True)
    print(f"{'eng':>10} {'q':>3} {'W':>4} {'m':>4} {'cf':>5} {'gate':>6}  {'vhmin':>6}  {'2024':>11} {'2025':>11} {'2026':>11}  {'FLOOR':>6}", flush=True)
    rows = []
    for (pe, ae) in PAIRS:
        for q in QS:
            for W in WS:
                vV, vT, vO = vrVA[(q, W)], vrTE[(q, W)], vrOO[(q, W)]
                for m in MS:
                    for cf in CONFS:
                        for gname, gate in GATES:
                            cV, selV = select(VA, vV, m, cf, gate, pe, ae)
                            if len(cV) < 60: continue
                            in1 = vh1[selV]; in2 = vh2[selV]
                            if in1.sum() < 20 or in2.sum() < 20: continue
                            vhmin = min(cV[in1].mean(), cV[in2].mean())
                            c24, _ = select(TE, vT, m, cf, gate, pe, ae, te24)
                            c25, _ = select(TE, vT, m, cf, gate, pe, ae, te25)
                            c26, _ = select(OO, vO, m, cf, gate, pe, ae)
                            if len(c24) < 25 or len(c25) < 25 or len(c26) < 25: continue
                            a24, a25, a26 = c24.mean(), c25.mean(), c26.mean()
                            fl = min(a24, a25, a26)
                            rows.append((pe, ae, q, W, m, cf, gname, vhmin, a24, a25, a26, fl, len(c24), len(c25), len(c26)))
    rows.sort(key=lambda r: -r[7])  # by worst-VAL-half (the honest selector)
    for r in rows[:20]:
        pe, ae, q, W, m, cf, gname, vhm, a24, a25, a26, fl, n24, n25, n26 = r
        print(f"{pe+'|'+ae:>10} {q:>3} {W:>4} {m:>4.2f} {cf:>5.3f} {gname:>6}  {vhm:6.3f}  {a24:.3f}(n{n24:>4}) {a25:.3f}(n{n25:>4}) {a26:.3f}(n{n26:>4})  {fl:6.3f}", flush=True)
    if rows:
        best = rows[0]
        pe, ae, q, W, m, cf, gname, vhm, a24, a25, a26, fl, n24, n25, n26 = best
        print(f"\n[HONEST worst-VAL-half pick] {pe}|{ae} q{q} W{W} m{m:.2f} cf{cf:.3f} {gname}: vhmin={vhm:.3f} -> "
              f"2024 {a24:.3f}(n{n24}) / 2025 {a25:.3f}(n{n25}) / 2026 {a26:.3f}(n{n26})  FLOOR={fl:.3f}", flush=True)
        print(f"[KILL check] 2025={a25:.3f} (need >=0.56 to beat the wall, >0.65 to hit goal); breakeven={BREAKEVEN}", flush=True)
        # also: the single best-FLOOR config (oracle, for ceiling reference)
        ob = max(rows, key=lambda r: r[11])
        print(f"[ORACLE max-floor] {ob[0]}|{ob[1]} q{ob[2]} W{ob[3]} m{ob[4]:.2f} cf{ob[5]:.3f} {ob[6]}: 2024 {ob[8]:.3f}/2025 {ob[9]:.3f}/2026 {ob[10]:.3f} FLOOR={ob[11]:.3f}", flush=True)
    print(f"[hurst] DONE {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
