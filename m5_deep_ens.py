"""EDGE-IMPROVEMENT EXP-2: seed-ENSEMBLED, properly-tuned (AdamW lr2e-4) cross-pair MLP, BLENDED with the m5xp
GBM on the SAME meta-gated bars — the literature-endorsed DL use-mode (decorrelated ensemble member, not
replacement). Tests whether ensembling lifts the UP-at-gate per year and the deployment-robustness tail vs the
GBM alone. Fixes the m5_deep anti-patterns (Adam lr1e-3 single-seed -> AdamW lr2e-4, M seeds, prob-averaging).

INCUMBENT (GBM m5xp UP-at-gate): .605/.577/.615. PRE-REGISTERED FALSIFIER: an IMPROVEMENT counts only if the
BLEND (or MLP-ens) UP-at-gate ≥ incumbent in the binding 2025 year (≥0.577) AND the MLP OOF-corr with GBM < 0.9
(genuine decorrelation). We WANT this to win — it targets the fragile tail, not vanity AUC."""
import sys, json, time, numpy as np
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
import m5_xpair as MX
import m5_xpair_production as XP
from m5_deep import MLP, load   # reuse loader + arch

torch.manual_seed(0); np.random.seed(0)
M = int(sys.argv[1]) if len(sys.argv) > 1 else 5   # seeds
SPL = XP.SPL


def boot(c, nb=3000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    return tuple(float(x) for x in np.percentile([c[rng.integers(0, n, n)].mean() for _ in range(nb)], [2.5, 97.5]))


def train_one(Xtr, ytr, Xva, yva, d, seed):
    torch.manual_seed(seed)
    mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0) + 1e-6
    nz = lambda X: np.nan_to_num((X - mu) / sd, nan=0.0).astype("float32")
    Xt = torch.tensor(nz(Xtr)); yt = torch.tensor(ytr.astype("float32")); Xv = torch.tensor(nz(Xva))
    m = MLP(d); opt = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss(); bs = 1024; n = len(Xt); best = (0.5, 1e9, None); bad = 0
    for ep in range(80):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; opt.zero_grad(); lossf(m(Xt[idx]), yt[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): pv = torch.sigmoid(m(Xv)).numpy()
        auc = roc_auc_score(yva, pv)
        if auc > best[0]: best = (auc, ep, m.state_dict().copy(), mu, sd); bad = 0
        else:
            bad += 1
            if bad >= 16: break
    return best  # (val_auc, ep, state, mu, sd)


def predict(state, mu, sd, X, d):
    m = MLP(d); m.load_state_dict(state); m.eval()
    nz = np.nan_to_num((X - mu) / sd, nan=0.0).astype("float32")
    with torch.no_grad(): return torch.sigmoid(m(torch.tensor(nz))).numpy()


def main():
    t0 = time.time()
    p = json.load(open(XP.art("strategy.json"))); cols = p["primary_feats"]; d = len(cols)
    Xtr, ytr, _, _ = load(SPL["train"], cols, 6); Xva, yva, _, _ = load(SPL["val"], cols)
    if len(Xtr) > 150000:
        idx = np.random.choice(len(Xtr), 150000, replace=False); Xtr, ytr = Xtr[idx], ytr[idx]
    print(f"[ens] train={len(Xtr):,} val={len(Xva):,} M={M} seeds; AdamW lr2e-4 {time.time()-t0:.0f}s", flush=True)
    models = []
    for s in range(M):
        b = train_one(Xtr, ytr, Xva, yva, d, 100 + s); models.append(b[2:])
        print(f"[ens] seed {s} VAL AUC={b[0]:.4f} ep={b[1]} {time.time()-t0:.0f}s", flush=True)
    # load GBM primary + meta for the gate + GBM probs
    pp, P, Mm = XP._load(); gcols = pp["primary_feats"]; mcols = pp["meta_feats"]; THR = pp["meta_thr"]
    inc = {"2024": 0.605, "2025": 0.577, "2026": 0.615}; res = {}; corrs = []
    for w, label in (("test24", "2024"), ("test25", "2025"), ("oos", "2026")):
        D = MX.build_xp(SPL[w]); D = MX.augment(D, SPL[w], XP.MODE)
        Xw = D[cols].astype("float32").to_numpy(); y = D["_y"].astype(int).values
        gbm = P.predict(D[gcols].astype("float32"))
        ens = np.mean([predict(st, mu, sd, Xw, d) for (st, mu, sd) in models], axis=0)
        corrs.append(np.corrcoef(gbm, ens)[0, 1])
        sm = Mm.predict(XP._Xmeta(D, gbm, mcols)); ts = D["_ts"].values.astype("int64"); ny = D["sess_ny"].values > 0.5
        sel = MX.nonoverlap_chrono(ts, ny & (sm >= THR))
        for src, name in ((gbm, "GBM"), (ens, "MLPens"), (0.5*gbm+0.5*ens, "BLEND")):
            pr = src[sel]; up = pr > 0.5
            if up.sum() < 5: res[f"{label}_{name}_UP"] = {"n": int(up.sum()), "acc": None}; continue
            corr = (y[sel][up] == 1).astype(float); lo, hi = boot(corr)
            res[f"{label}_{name}_UP"] = {"n": int(up.sum()), "acc": round(float(corr.mean()), 4), "ci": [round(lo, 4), round(hi, 4)]}
        del D
    avg_corr = float(np.mean(corrs))
    def up(label, src): return res.get(f"{label}_{src}_UP", {}).get("acc")
    blend_beats = up("2025", "BLEND") and up("2025", "BLEND") >= 0.577
    ens_beats = up("2025", "MLPens") and up("2025", "MLPens") >= 0.577
    out = {"exp": "EXP-2 seed-ensemble MLP blended with GBM", "M_seeds": M, "incumbent_up": inc,
           "mlp_gbm_corr": round(avg_corr, 3), "per_year": res,
           "verdict": {"BLEND_improves_2025": bool(blend_beats), "MLPens_improves_2025": bool(ens_beats),
                       "decorrelated": bool(avg_corr < 0.9),
                       "note": f"corr(MLP,GBM)={avg_corr:.3f}; UP2025 GBM={up('2025','GBM')} MLPens={up('2025','MLPens')} BLEND={up('2025','BLEND')} (incumbent 0.577)"}}
    json.dump(out, open("m5_deep_ens_result.json", "w"), indent=1)
    for k, v in res.items():
        if v.get("acc") is not None: print(f"  {k:18} n={v['n']} acc={v['acc']} CI={v['ci']}", flush=True)
    print(f"[ens] {out['verdict']['note']}", flush=True)
    print(f"[ens] -> m5_deep_ens_result.json ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
