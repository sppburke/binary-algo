"""BAR-PATTERN CNN — the MAGNITUDE target (what bar geometry actually encodes; sign-invariance theorem).

Direction @60s is near-efficient (barcnn_run.py: all image variants AUC ~.50). But MAGNITUDE is the program's
ONE certified 60s edge (magAUC 0.787 vs dirAUC 0.510, MAGNITUDE_FINDINGS.md) — and move-SIZE is exactly what a
2-D bar image encodes (range expansion / compression / vol clustering are visible). So we point the SAME faithful
Sezer-style bar-image CNN at the magnitude "outcome": large-move vs small-move (balanced median split), the deriv
Touch/Range/straddle outcome. Question: does the bar-image CNN predict the (magnitude) outcome at > 65%?

Label = (|ret60| >= train-MEDIAN) over valid+contiguous-window bars (BALANCED 50/50 — so accuracy is meaningful,
not a base-rate artifact). |ret60| from the deriv-faithful wc_ret in barcnn_bars.py. Eval: AUC + balanced acc +
SELECTIVE precision of "large" calls at low coverage (the tradeable straddle prediction-rate) per year + faithful
CPCV (path AUC p10 + selective-accuracy p10). This is MAGNITUDE (sign-invariant) -> a Touch/Range product, NOT
Rise/Fall direction. Recorded in MAGNITUDE_FINDINGS.md, not the UP/DOWN keys.

Run: ~/binary-algo-venv/bin/python barcnn_mag.py [variant=ohlc] [W=30] [subsample=80000] [epochs=14]
"""
import sys, os, json, time, numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from itertools import combinations
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

ROOT = "/home/sean/git/binary-algo"; OUT = f"{ROOT}/ohlc_cache"
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "ohlc"
W = int(sys.argv[2]) if len(sys.argv) > 2 else 30
SUBSAMPLE = int(sys.argv[3]) if len(sys.argv) > 3 else 80_000
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 14
HPX, BARGAP, HS, TOL = 30, 60, 60, 10; GAP = HS + TOL
BS, VAL_ES = 1024, 15000
N_GROUPS, K_TEST = 8, 2
torch.set_num_threads(16); torch.manual_seed(7); np.random.seed(7); T0 = time.time()
def hb(m): print(f"HB[{time.time()-T0:.0f}s] {m}", flush=True)


def load(sp):
    b = pd.read_parquet(f"{OUT}/EURUSD_1m_{sp}.parquet")
    return (b["open"].values.astype("float32"), b["high"].values.astype("float32"),
            b["low"].values.astype("float32"), b["close"].values.astype("float32"),
            b["t"].values.astype("int64"), b["mag"].values.astype("float32"), b["valid"].values.astype(bool))

def mag_eligible(t, mag, valid):
    n = len(t); idx = np.arange(W - 1, n)
    contig = (t[idx] - t[idx - (W - 1)]) == (W - 1) * BARGAP
    return idx[valid[idx] & contig]                                # valid + contiguous window (mag>=0 incl ties)

def _win(arr, ends): return sliding_window_view(arr, W)[ends - (W - 1)]

PPP = 0.6e-4   # ABS scaling: 0.6 pip per pixel -> +/-9 pips over 30 rows (preserves absolute volatility = magnitude)

def render(o, h, l, c, ends, variant):
    O, H, L, C = _win(o, ends), _win(h, ends), _win(l, ends), _win(c, ends)
    if variant == "gaf":
        x = C; mn = x.min(1, keepdims=True); mx = x.max(1, keepdims=True); rng = np.maximum(mx - mn, 1e-12)
        xs = np.clip(2 * (x - mn) / rng - 1.0, -1, 1); phi = np.arccos(xs)
        gasf = np.cos(phi[:, :, None] + phi[:, None, :]); gadf = np.sin(phi[:, None, :] - phi[:, :, None])
        return np.stack([gasf, gadf], axis=1).astype("float32")
    rows = np.arange(HPX)[None, :, None]
    if variant.endswith("abs"):
        # ABSOLUTE scale: center on window mean close, fixed pips/pixel -> bar SPREAD encodes absolute vol (magnitude)
        ctr = C.mean(1, keepdims=True)
        def row(v): return np.clip(np.round(HPX / 2 - (v - ctr) / PPP), 0, HPX - 1).astype(np.int16)
    else:
        lo = L.min(1, keepdims=True); hi = H.max(1, keepdims=True); rng = np.maximum(hi - lo, 1e-12)
        def row(v): return np.clip(np.round((1 - (v - lo) / rng) * (HPX - 1)), 0, HPX - 1).astype(np.int16)
    if variant.startswith("hist"):
        Cr = row(C); return (rows >= Cr[:, None, :]).astype(np.uint8)[:, None, :, :]
    Hr, Lr, Or, Cr = row(H), row(L), row(O), row(C)
    hir = np.minimum(Or, Cr); lor = np.maximum(Or, Cr)
    wick = ((rows >= Hr[:, None, :]) & (rows <= Lr[:, None, :])).astype(np.uint8)
    body = ((rows >= hir[:, None, :]) & (rows <= lor[:, None, :])); up = (C >= O)[:, None, :]
    return np.stack([wick, (body & up).astype(np.uint8), (body & ~up).astype(np.uint8)], axis=1)

def in_ch(v): return 2 if v == "gaf" else (1 if v.startswith("hist") else 3)
def img_hw(v): return (W, W) if v == "gaf" else (HPX, W)
def to_tensor(img): return torch.from_numpy(img).float()

class CNN(nn.Module):
    def __init__(s, ch, hw):
        super().__init__()
        s.conv = nn.Sequential(nn.Conv2d(ch, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 64, 3, padding=1),
                               nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.25))
        s.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * (hw[0]//2) * (hw[1]//2), 128), nn.ReLU(),
                               nn.Dropout(0.5), nn.Linear(128, 1))
    def forward(s, x): return s.head(s.conv(x)).squeeze(-1)

def predict(model, o, h, l, c, ends, variant, chunk=16384):
    model.eval(); ps = np.empty(len(ends), "float32")
    with torch.no_grad():
        for s in range(0, len(ends), chunk):
            e = min(s + chunk, len(ends))
            ps[s:e] = torch.sigmoid(model(to_tensor(render(o, h, l, c, ends[s:e], variant)))).numpy()
    return ps

def nonoverlap_chrono(ts, mask, gap=GAP):
    take = []; bu = -1
    for i in np.where(mask)[0]:
        if ts[i] < bu: continue
        take.append(i); bu = int(ts[i]) + gap
    return np.array(take, dtype=int)

def boot(corr, nb=5000, seed=7):
    corr = np.asarray(corr, float)
    if len(corr) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(corr)
    a = np.array([corr[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def cpcv_paths(ts, score):  # score = per-trade correct(0/1); returns path p10 / frac>=0.65
    n = len(score); edges = np.linspace(0, n, N_GROUPS + 1).astype(int)
    grp = [(edges[i], edges[i+1]) for i in range(N_GROUPS)]; pa = []
    for tg in combinations(range(N_GROUPS), K_TEST):
        idx = np.concatenate([np.arange(grp[g][0], grp[g][1]) for g in tg])
        if len(idx) >= 30: pa.append(float(score[idx].mean()))
    pa = np.array(pa)
    return (round(float(np.percentile(pa, 10)), 4), round(float(pa.mean()), 4),
            round(float((pa >= 0.65).mean()), 3), round(float((pa >= 0.541).mean()), 3))


def main():
    hb(f"MAGNITUDE target | variant={VARIANT} W={W} sub={SUBSAMPLE} ep={EPOCHS}")
    tr, va, te, oo = load("train"), load("val"), load("test"), load("oos")
    etr = mag_eligible(tr[4], tr[5], tr[6]); eva = mag_eligible(va[4], va[5], va[6])
    ete = mag_eligible(te[4], te[5], te[6]); eoo = mag_eligible(oo[4], oo[5], oo[6])
    magthr = float(np.median(tr[5][etr]))                          # BALANCED median split (train-only)
    hb(f"eligible tr={len(etr):,} va={len(eva):,} te={len(ete):,} oo={len(eoo):,} | mag median thr={magthr:.2e}")

    rng = np.random.default_rng(7); sel = np.sort(rng.choice(etr, size=min(SUBSAMPLE, len(etr)), replace=False))
    Xtr = render(tr[0], tr[1], tr[2], tr[3], sel, VARIANT); ytr = (tr[5][sel] >= magthr).astype("float32")
    hb(f"train imgs {Xtr.shape} {Xtr.dtype} large_rate={ytr.mean():.3f}")
    model = CNN(in_ch(VARIANT), img_hw(VARIANT)); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(); Xt = to_tensor(Xtr); yt = torch.from_numpy(ytr); ntr = len(sel)
    eva_es = np.sort(rng.choice(eva, size=min(VAL_ES, len(eva)), replace=False))
    yva_es = (va[5][eva_es] >= magthr).astype("float32")
    best = -1; best_state = None; bad = 0
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(ntr)
        for s in range(0, ntr, BS):
            bi = perm[s:s+BS]; opt.zero_grad(); loss = lossf(model(Xt[bi]), yt[bi]); loss.backward(); opt.step()
        auc = roc_auc_score(yva_es, predict(model, va[0], va[1], va[2], va[3], eva_es, VARIANT))
        hb(f"ep{ep+1}/{EPOCHS} loss={loss.item():.4f} VAL_magAUC={auc:.4f}")
        if auc > best + 1e-4: best = auc; best_state = {k: v.clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= 3: hb(f"early stop @ep{ep+1}"); break
    model.load_state_dict(best_state); hb(f"BEST VAL magAUC={best:.4f}")

    res = {"target": "MAGNITUDE |ret60|>=train-median (balanced, sign-invariant Touch/Range outcome)",
           "variant": VARIANT, "mag_median_thr": magthr, "best_val_magAUC": round(float(best), 4),
           "note": "MAGNITUDE not direction; record in MAGNITUDE_FINDINGS.md, NOT UP/DOWN keys.", "per_year": {}}
    # pool test+oos
    pred = {}
    for nm, d, ev in (("test", te, ete), ("oos", oo, eoo)):
        pred[nm] = (d[4][ev], (d[5][ev] >= magthr).astype(int), predict(model, d[0], d[1], d[2], d[3], ev, VARIANT))
    ts = np.concatenate([pred["test"][0], pred["oos"][0]]); ytrue = np.concatenate([pred["test"][1], pred["oos"][1]])
    p = np.concatenate([pred["test"][2], pred["oos"][2]]); o = np.argsort(ts); ts, ytrue, p = ts[o], ytrue[o], p[o]
    yr = pd.to_datetime(ts, unit="s", utc=True).year.values
    res["test_magAUC"] = round(float(roc_auc_score(pred["test"][1], pred["test"][2])), 4)
    res["oos_magAUC"] = round(float(roc_auc_score(pred["oos"][1], pred["oos"][2])), 4)
    hb(f"test magAUC={res['test_magAUC']} oos magAUC={res['oos_magAUC']}")
    # SELECTIVE precision of 'large' calls = tradeable straddle prediction-rate
    res["selective_large_call_precision"] = {}
    for cov in (0.30, 0.20, 0.10, 0.05):
        thr = float(np.quantile(p, 1 - cov))                       # top-cov% highest P(large)
        cells = {}
        for Y in (2024, 2025, 2026):
            mY = (yr == Y); sel_i = nonoverlap_chrono(ts[mY], p[mY] >= thr)
            if len(sel_i) < 20: cells[str(Y)] = {"n": int(len(sel_i)), "prec": None}; continue
            corr = (ytrue[mY][sel_i] == 1).astype(float); lo, hi = boot(corr)
            cells[str(Y)] = {"n": int(len(sel_i)), "prec": round(float(corr.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)]}
        selall = nonoverlap_chrono(ts, p >= thr); corr_all = (ytrue[selall] == 1).astype(float)
        p10, pmean, fr65, fr54 = cpcv_paths(ts[selall], corr_all)
        cells["pooled_cpcv"] = {"n": int(len(selall)), "prec": round(float(corr_all.mean()), 4),
                                "path_p10": p10, "path_mean": pmean, "frac_paths>=0.65": fr65, "frac_paths>=0.541": fr54}
        res["selective_large_call_precision"][f"cov{cov}"] = cells
    json.dump(res, open(f"{ROOT}/barcnn_mag_{VARIANT}_result.json", "w"), indent=1)
    print("\n=== MAGNITUDE prediction-rate (bar-image CNN) ===", flush=True)
    print(f"VAL/test/oos magAUC = {res['best_val_magAUC']} / {res['test_magAUC']} / {res['oos_magAUC']}", flush=True)
    for cov, cells in res["selective_large_call_precision"].items():
        pc = cells["pooled_cpcv"]
        print(f"  {cov}: per-year prec={{2024:{cells['2024']['prec']},2025:{cells['2025']['prec']},2026:{cells['2026']['prec']}}} "
              f"| pooled n={pc['n']} prec={pc['prec']} CPCV p10={pc['path_p10']} frac>=0.65={pc['frac_paths>=0.65']}", flush=True)
    print(f"-> barcnn_mag_{VARIANT}_result.json", flush=True)


if __name__ == "__main__":
    main()
