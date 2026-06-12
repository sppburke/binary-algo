"""RFF virtue-of-complexity SDF @MX_HOR=2 (corpus lever #15, new_untried high-prior; Didisheim-Ke-Kelly-Malamud
APT/AIPT large-factor-models). The ONE genuinely-distinct remaining on-disk DIRECTION shot after the cross-pair
GBM refit-CPCV KILL (p10 .51, 0-7% paths clear). Tests whether a high-dimensional random nonlinear basis (P~T,
the virtue-of-complexity regime) extracts a 2m sign edge that the GBM / ARF / frozen-ensemble (all ~.51-.52 AUC)
underfit on the SAME xpof cross-pair features. If three flexible learners + RFF all sit at ~.51, the channel is
genuinely empty (not a model-capacity artifact).

Method: standardize xpof features; F=[sin,cos](gamma * Z @ omega), omega~N(0,I); ridge-regress signed label in
{-1,+1}; predict signed score on held-out (batched). gamma/ridge tuned on VAL by AUC. Side-split, worst-VAL-half
compression gate (1m_bb_width), nonoverlap_chrono, per-year CI95, moved bars + up-rate tripwire.
Pre-registered falsifier: KILL if VAL moved-dirAUC <= 0.515 (same bar the cross-pair GBM faced) OR no side clears
0.545 in >=2 held-out years. Usage: python min2_rff.py
"""
import os
os.environ["MX_HOR"] = "2"
import json, time, numpy as np
from sklearn.metrics import roc_auc_score
import m5_xpair as MX

HOR = 2; GAP_S = HOR * 60; BE = 0.541
GATE_FEAT = "1m_bb_width"
T_SUB = 12000     # subsample moved train rows -> virtue-of-complexity regime P ~ T
P = 8000          # total random features (P/2 sin + P/2 cos); memory- & time-safe
SEED = 7
VAL_AUC_SUB = 80000   # subsample VAL for the fast-KILL AUC (full predict only if it survives)


def yr(ts):
    return (np.asarray(ts, dtype="datetime64[s]").astype("datetime64[Y]").astype(int) + 1970)


def boot(c, nb=5000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def rff_map(Z, omega, gamma):
    proj = gamma * (Z @ omega)                       # (n, P/2)
    return np.concatenate([np.sin(proj), np.cos(proj)], axis=1).astype("float32")   # (n, P)


def predict_batched(Z, omega, gamma, w, bs=8000):
    out = np.empty(Z.shape[0], dtype="float32")
    for i in range(0, Z.shape[0], bs):
        out[i:i + bs] = rff_map(Z[i:i + bs], omega, gamma) @ w
    return out


def main():
    t0 = time.time()
    TR = MX.build_xp(MX.SPL["train"], stride=4); VA = MX.build_xp(MX.SPL["val"])
    xpc = MX.xp_cols(TR)
    TR = MX.augment(TR, MX.SPL["train"], "xpof"); VA = MX.augment(VA, MX.SPL["val"], "xpof")
    cols = MX.feat_cols("xpof", TR, xpc)
    print(f"[rff] train={len(TR):,} val={len(VA):,} feats={len(cols)} build {time.time()-t0:.0f}s", flush=True)
    Xtr = TR[cols].values.astype("float32"); ytr = TR["_y"].astype(int).values
    Xva = VA[cols].values.astype("float32"); yva = VA["_y"].astype(int).values
    # standardize (impute NaN->0 after centering); fit on train
    mu = np.nanmean(Xtr, axis=0); sd = np.nanstd(Xtr, axis=0); sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    def std(X):
        Z = (X - mu) / sd; Z[~np.isfinite(Z)] = 0.0; return Z.astype("float32")
    import gc
    rng = np.random.default_rng(SEED); D = Xtr.shape[1]
    omega = rng.standard_normal((D, P // 2)).astype("float32")
    # subsample train rows to the high-dim regime BEFORE standardizing the full frame (memory)
    idx = rng.choice(len(Xtr), size=min(T_SUB, len(Xtr)), replace=False)
    Zs = std(Xtr[idx]); ys = (2 * ytr[idx] - 1).astype("float32")     # signed label {-1,+1}
    Zva = std(Xva)
    # extract small VAL arrays for the gate, then free the big frames
    tsv = VA["_ts"].values.astype("int64"); bbv = VA[GATE_FEAT].values.astype("float32")
    nyv = VA["sess_ny"].values > 0.5
    del TR, Xtr, VA, Xva; gc.collect()
    vsub = rng.choice(len(Zva), size=min(VAL_AUC_SUB, len(Zva)), replace=False)
    Zva_s = Zva[vsub]; yva_s = yva[vsub]
    best = None
    for gamma in (1.0 / np.sqrt(D), 2.0 / np.sqrt(D)):
        F = rff_map(Zs, omega, gamma)                            # (T_SUB, P)
        G = F.T @ F                                              # (P,P)
        Fy = F.T @ ys
        for z in (10.0, 100.0):
            w = np.linalg.solve(G + z * np.eye(P, dtype="float32"), Fy).astype("float32")
            pv = predict_batched(Zva_s, omega, gamma, w)
            auc = roc_auc_score(yva_s, pv)
            print(f"[rff] gamma={gamma:.4g} ridge={z} VAL_sub AUC={auc:.4f} {time.time()-t0:.0f}s", flush=True)
            if best is None or auc > best[0]:
                best = (auc, gamma, z, w)
        del F, G, Fy; gc.collect()
    valauc, GAMMA, Z_RIDGE, W = best
    print(f"[rff] best VAL AUC={valauc:.4f} gamma={GAMMA:.4g} ridge={Z_RIDGE} (P={P}, T={T_SUB}) {time.time()-t0:.0f}s", flush=True)
    res = {"test": "RFF virtue-of-complexity SDF @MX_HOR=2 (xpof features)", "breakeven": BE, "P": P, "T_sub": T_SUB,
           "val_auc": round(float(valauc), 4), "gamma": float(GAMMA), "ridge": float(Z_RIDGE),
           "falsifier": {"KILL_if_val_auc<=": 0.515, "ESCALATE_if": "side clears 0.545 in >=2 held-out years"}}
    json.dump(res, open("min2_rff_result.json", "w"), indent=1)
    if valauc <= 0.515:
        res["verdict"] = {"KILLED": True, "reason": f"VAL moved-dirAUC {valauc:.4f} <= 0.515 — RFF extracts no 2m "
                          f"cross-pair sign edge the GBM/ARF missed; channel genuinely empty (4th flat model class)."}
        json.dump(res, open("min2_rff_result.json", "w"), indent=1)
        print(f"[rff] VERDICT KILLED (VAL AUC {valauc:.4f} <= 0.515)", flush=True)
        return
    # held-out side-split with worst-VAL-half compression gate
    pva = predict_batched(Zva, omega, GAMMA, W)
    vyr = yr(tsv); confv = np.abs(pva - np.median(pva))
    fin = np.isfinite(bbv); vyears = sorted(set(vyr.tolist())); thr0 = np.median(pva)
    best = None
    for useny in (True, False):
        for bq in (20, 33, 100):
            bthr = np.nanpercentile(bbv[fin], bq)
            base_all = fin & (bbv <= bthr) & (nyv if useny else np.ones(len(bbv), bool))
            if base_all.sum() < 200:
                continue
            for cov in (0.05, 0.10, 0.20):
                cthr = float(np.quantile(confv[base_all], 1 - cov)); accs = []; ntot = 0
                for Y in vyears:
                    m = base_all & (vyr == Y) & (confv >= cthr); sel = MX.nonoverlap_chrono(tsv, m, GAP_S)
                    if len(sel) < 40:
                        accs = None; break
                    accs.append((((pva[sel] > thr0).astype(int)) == yva[sel]).mean()); ntot += len(sel)
                if accs is None or ntot < 150:
                    continue
                wh = min(accs)
                if best is None or wh > best[0]:
                    best = (wh, useny, float(bthr), cthr, bq, cov)
    if best is None:
        res["verdict"] = {"KILLED": True, "reason": "no stable VAL gate (n>=150) — RFF too thin at 2m."}
        json.dump(res, open("min2_rff_result.json", "w"), indent=1)
        print("[rff] VERDICT KILLED (no stable VAL gate)", flush=True)
        return
    wh, USENY, BTHR, CTHR, BQ, COV = best
    res["frozen_gate"] = {"use_ny": USENY, "bb_q": BQ, "cov": COV, "val_worst_half": round(wh, 4)}
    per_year = {}
    for w, ylabel in (("test24", 2024), ("test25", 2025), ("oos", 2026)):
        D2 = MX.build_xp(MX.SPL[w]); D2 = MX.augment(D2, MX.SPL[w], "xpof")
        for c in cols:
            if c not in D2.columns:
                D2[c] = np.nan
        Zh = std(D2[cols].values.astype("float32")); pr = predict_batched(Zh, omega, GAMMA, W)
        y = D2["_y"].astype(int).values; ts = D2["_ts"].values.astype("int64"); bb = D2[GATE_FEAT].values.astype("float32")
        base = np.isfinite(bb) & (bb <= BTHR)
        if USENY:
            base = base & (D2["sess_ny"].values > 0.5)
        conf = np.abs(pr - thr0); cand = base & (conf >= CTHR); sel = MX.nonoverlap_chrono(ts, cand, GAP_S)
        rec = {"n_sel": int(len(sel)), "auc_full": round(float(roc_auc_score(y, pr)), 4)}
        if len(sel):
            pred = (pr[sel] > thr0).astype(int); yt = y[sel]; corr = (pred == yt).astype(float); lo, hi = boot(corr)
            rec["combined"] = {"acc": round(float(corr.mean()), 4), "ci": [round(lo, 4), round(hi, 4)],
                               "up_rate_moved": round(float(yt.mean()), 4)}
            for side, mask in (("UP", pred == 1), ("DOWN", pred == 0)):
                if mask.sum() >= 25:
                    c = (pred[mask] == yt[mask]).astype(float); l2, h2 = boot(c)
                    rec[side] = {"acc": round(float(c.mean()), 4), "ci": [round(l2, 4), round(h2, 4)], "n": int(mask.sum())}
        per_year[str(ylabel)] = rec
        u = rec.get("UP", {}); dn = rec.get("DOWN", {})
        print(f"=== {w} ({ylabel}) === AUC={rec['auc_full']:.4f} n={rec['n_sel']} comb={rec.get('combined',{}).get('acc','-')} "
              f"UP={u.get('acc','-')}(n{u.get('n','-')}) DOWN={dn.get('acc','-')}(n{dn.get('n','-')}) {time.time()-t0:.0f}s", flush=True)
    res["per_year"] = per_year
    def side_clears(side):
        return sum(1 for Y in ("2024", "2025", "2026") if per_year[Y].get(side, {}).get("acc", 0) >= 0.545)
    up_y, dn_y = side_clears("UP"), side_clears("DOWN")
    esc = (up_y >= 2) or (dn_y >= 2)
    res["verdict"] = {"ESCALATE": bool(esc), "up_years_ge_0.545": up_y, "down_years_ge_0.545": dn_y,
                      "reason": ("ESCALATE to refit-CPCV" if esc else
                                 "KILL: RFF carries no 2m direction edge (no side clears 0.545 in >=2 years)")}
    json.dump(res, open("min2_rff_result.json", "w"), indent=1)
    print(f"[rff] VERDICT: {'ESCALATE' if esc else 'KILL'} (UP {up_y}/3, DOWN {dn_y}/3) -> min2_rff_result.json", flush=True)


if __name__ == "__main__":
    main()
