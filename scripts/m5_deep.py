"""Sweep rows D1b/D-MLP: does a DEEP net (MLP + GRU/LSTM) on the m5xp CROSS-PAIR feature set beat the GBM at 5m
direction? The deep models previously run were single-pair raw-path (null); this is the missing apples-to-apples
test on the features where the certified UP edge lives. Train 2012-2021 / val 2022-23 / test 2024-26 (the m5xp
split). MLP = same per-bar features as the GBM; GRU/LSTM = windowed sequences of W bars (the temporal test).
Compare VAL AUC + per-year UP-at-gate to the GBM baseline (VAL AUC ~0.523, UP ~0.58). torch CPU.

PRE-REGISTERED FALSIFIER: KILL deep unless VAL AUC > 0.53 AND per-year UP-at-NY-gate (top-conf cov~0.10) ≥ the
GBM's .605/.577/.615 in the binding 2025 year. Prior: match-or-worse (tabular regime favors GBMs; AUC is
information-bound at ~0.52)."""
import os, json, time, numpy as np
import torch, torch.nn as nn
import m5_xpair as MX
import m5_xpair_production as XP

torch.manual_seed(0); np.random.seed(0)
DEV = "cpu"
SPL = {"train": [str(y) for y in range(2012, 2022)], "val": ["2022", "2023"],
       "test24": ["2024"], "test25": ["2025"], "oos": ["2026"]}
STRIDE_TRAIN = 6
W = 12   # GRU window (bars)


def load(years, cols, stride=1):
    import gc
    Xs, ys, tss, nys = [], [], [], []
    for yr in years:
        D = MX.build_xp([yr], stride=stride)
        if len(D) == 0: continue
        D = MX.augment(D, [yr], XP.MODE)
        Xs.append(D[cols].astype("float32").to_numpy())
        ys.append(D["_y"].astype(np.int8).values)
        tss.append(D["_ts"].values.astype("int64"))
        nys.append((D["sess_ny"].values > 0.5))
        del D; gc.collect()
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(tss), np.concatenate(nys)


def boot(c, nb=2000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.BatchNorm1d(d), nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def train_mlp(Xtr, ytr, Xva, yva, d, epochs=40):
    from sklearn.metrics import roc_auc_score
    mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0) + 1e-6
    def norm(X): return np.nan_to_num((X - mu) / sd, nan=0.0).astype("float32")
    Xt = torch.tensor(norm(Xtr)); yt = torch.tensor(ytr.astype("float32"))
    Xv = torch.tensor(norm(Xva))
    m = MLP(d).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(); bs = 4096; n = len(Xt); best = (0.5, None)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; opt.zero_grad()
            out = m(Xt[idx]); loss = lossf(out, yt[idx]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): pv = torch.sigmoid(m(Xv)).numpy()
        auc = roc_auc_score(yva, pv)
        if auc > best[0]: best = (auc, pv.copy())
    return best  # (val_auc, val_pred)


def gate_eval(pr_dict, y_dict, ny_dict, ts_dict, thr_q=0.90):
    """UP-at-gate: NY & pred-up, top (1-thr_q) confidence; per-year acc."""
    # pick conf threshold from pooled NY up-pred predictions
    allconf = np.concatenate([np.abs(pr_dict[k] - 0.5)[ny_dict[k] & (pr_dict[k] > 0.5)] for k in pr_dict])
    cthr = float(np.quantile(allconf, thr_q)) if len(allconf) else 0.0
    out = {}
    for k in pr_dict:
        pr = pr_dict[k]; y = y_dict[k]; ny = ny_dict[k]; ts = ts_dict[k]
        m = ny & (pr > 0.5) & (np.abs(pr - 0.5) >= cthr)
        sel = MX.nonoverlap_chrono(ts, m)
        if len(sel) < 5: out[k] = {"n": int(len(sel)), "up_acc": None}; continue
        out[k] = {"n": int(len(sel)), "up_acc": round(float((y[sel] == 1).mean()), 4)}
    return out, cthr


def main():
    t0 = time.time()
    p = json.load(open(XP.art("strategy.json"))); cols = p["primary_feats"]; d = len(cols)
    print(f"[deep] loading cross-pair features (d={d})...", flush=True)
    Xtr, ytr, _, _ = load(SPL["train"], cols, STRIDE_TRAIN)
    Xva, yva, _, _ = load(SPL["val"], cols)
    # subsample train
    if len(Xtr) > 150000:
        idx = np.random.choice(len(Xtr), 150000, replace=False); Xtr, ytr = Xtr[idx], ytr[idx]
    print(f"[deep] train={len(Xtr):,} val={len(Xva):,} {time.time()-t0:.0f}s; training MLP...", flush=True)
    val_auc, _ = train_mlp(Xtr, ytr, Xva, yva, d)
    print(f"[deep] MLP VAL AUC={val_auc:.4f} (GBM baseline 0.523) {time.time()-t0:.0f}s", flush=True)
    # retrain on train, predict test years at the gate
    mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0) + 1e-6
    def norm(X): return np.nan_to_num((X - mu) / sd, nan=0.0).astype("float32")
    # rebuild a final model (use the train fit; quick refit for prediction determinism)
    from sklearn.metrics import roc_auc_score
    Xt = torch.tensor(norm(Xtr)); yt = torch.tensor(ytr.astype("float32"))
    m = MLP(d); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4); lossf = nn.BCEWithLogitsLoss()
    for ep in range(30):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 4096):
            idx = perm[i:i+4096]; opt.zero_grad(); loss = lossf(m(Xt[idx]), yt[idx]); loss.backward(); opt.step()
    m.eval()
    pr_dict, y_dict, ny_dict, ts_dict = {}, {}, {}, {}
    for w, label in (("test24", "2024"), ("test25", "2025"), ("oos", "2026")):
        Xw, yw, tw, nw = load(SPL[w], cols)
        with torch.no_grad(): prw = torch.sigmoid(m(torch.tensor(norm(Xw)))).numpy()
        pr_dict[label] = prw; y_dict[label] = yw; ny_dict[label] = nw; ts_dict[label] = tw
        print(f"[deep] {label} test AUC={roc_auc_score(yw, prw):.4f} n={len(yw):,}", flush=True)
    gate, cthr = gate_eval(pr_dict, y_dict, ny_dict, ts_dict)
    gbm = {"2024": 0.605, "2025": 0.577, "2026": 0.615}
    beats = all(gate[k]["up_acc"] is not None and gate[k]["up_acc"] >= gbm[k] for k in gbm) and val_auc > 0.53
    out = {"row": "D-MLP: deep MLP on cross-pair features @5m", "val_auc_MLP": round(val_auc, 4),
           "gbm_baseline_val_auc": 0.523, "gbm_up_gate": gbm, "MLP_up_gate": gate, "gate_conf_thr": round(cthr, 4),
           "falsifier": {"DEEP_BEATS_GBM": bool(beats),
                         "verdict": ("DEEP beats GBM" if beats else
                                     f"KILLED: MLP VAL AUC {val_auc:.4f} (GBM 0.523); UP-at-gate {{{', '.join(k+':'+str(gate[k]['up_acc']) for k in gate)}}} "
                                     f"does NOT beat the GBM .605/.577/.615 -> deep on cross-pair features does NOT help 5m direction "
                                     "(information-bound; GBMs dominate this tabular regime).")}}
    json.dump(out, open("m5_deep_result.json", "w"), indent=1)
    print(f"[deep] {out['falsifier']['verdict']}", flush=True)
    print(f"[deep] -> m5_deep_result.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
