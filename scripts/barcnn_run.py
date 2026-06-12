"""BAR-PATTERN CNN — step 2: render OHLC windows as 2-D images, train a small CNN, predict 60s direction.

SCOPE: (EURUSD, 60s direction). The ONE non-subsumed bar/candlestick sub-lever (commit 8bfa7f6 vetting):
a 2-D CNN over rendered OHLC bar IMAGES — local pattern detectors a per-bar GBM and a 1-D GRU (both already
null at 60s, SWEEP_MATRIX D1) cannot represent. Faithful to Sezer & Ozbayoglu CNN-BI (arXiv:1903.04610):
30-bar window, per-window min-max normalization, bottom-anchored vertical-bar histogram of normalized close
-> 30x30x1 binary image -> MNIST-class CNN (Conv32 3x3 -> Conv64 3x3 -> MaxPool2 -> Dropout .25 -> Dense128
-> Dropout .5 -> head). Their 3-class slope label LOOKS AHEAD (future days 34/45); we DROP it and use the
deriv-faithful 60s up/down label from barcnn_bars.py (no leak, ties LOSE, moved-bars only).

Variants (arg):
  hist  : Sezer CNN-BI close-histogram, 1ch 30x30 (the canonical, faithful lever)
  ohlc  : richer 3ch 30x30 -> ch0 high-low wick, ch1 up-candle body, ch2 down-candle body (adds per-bar sign)
  gaf   : GASF+GADF Gramian Angular Field of the window close series, 2ch WxW (Wang-Oates) -> 2-D CNN

MEMORY-SAFE (post-OOM): uint8 images; train subsampled to <=SUBSAMPLE; chunked eval rendering; 10 threads.

Run: ~/binary-algo-venv/bin/python barcnn_run.py <variant> [W] [SUBSAMPLE] [EPOCHS]
Writes barcnn_pred_<variant>.npz (ts,y,mag,p per eligible val/test/oos window) + barcnn_<variant>_result.json.
"""
import sys, os, json, time, gc, numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

ROOT = "/home/sean/git/binary-algo"; OUT = f"{ROOT}/ohlc_cache"
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "hist"
W       = int(sys.argv[2]) if len(sys.argv) > 2 else 30      # window length in bars
SUBSAMPLE = int(sys.argv[3]) if len(sys.argv) > 3 else 120_000
EPOCHS  = int(sys.argv[4]) if len(sys.argv) > 4 else 25
SESSION = (sys.argv[5] if len(sys.argv) > 5 else "all").lower()   # all|ny|ldn|asia — DST-correct decision-bar filter
SUF     = "" if SESSION == "all" else f"_{SESSION}"
from sessions import session_mask
HPX = 30                                                     # image height in pixels (Sezer uses 30)
BARGAP = 60                                                  # 1-min bars: contiguous window iff t_i - t_{i-W+1} == (W-1)*60
HS, TOL = 60, 10; GAP = HS + TOL                             # 60s expiry; non-overlap block 70s
torch.set_num_threads(16); torch.manual_seed(7); np.random.seed(7)
BS = 1024; VAL_ES = 15000                                    # batch size; VAL subsample for per-epoch early-stop AUC
T0 = time.time()
def hb(m): print(f"HB[{time.time()-T0:.0f}s] {m}", flush=True)


# ----------------------------- data: eligible decision bars + contiguous windows -----------------------------
def load(sp):
    b = pd.read_parquet(f"{OUT}/EURUSD_1m_{sp}.parquet")
    return (b["open"].values.astype("float32"), b["high"].values.astype("float32"),
            b["low"].values.astype("float32"), b["close"].values.astype("float32"),
            b["t"].values.astype("int64"), b["y"].values.astype("int64"),
            b["mag"].values.astype("float32"), b["valid"].values.astype(bool))

def eligible(t, y, mag, valid):
    """Decision bars i (>=W-1) that are valid & moved (mag>0) with a fully-contiguous W-bar window ending at i."""
    n = len(t); idx = np.arange(W - 1, n)
    contig = (t[idx] - t[idx - (W - 1)]) == (W - 1) * BARGAP
    ok = valid[idx] & (mag[idx] > 0) & contig
    return idx[ok]                                            # decision-bar positions in the bars array


# ----------------------------- image renderers (vectorized, uint8) -----------------------------
def _win(arr, ends):  # windows of arr ending at positions `ends` (inclusive) -> (len(ends), W)
    sw = sliding_window_view(arr, W)                          # (n-W+1, W); sw[j] = arr[j:j+W]
    return sw[ends - (W - 1)]

def render(o, h, l, c, ends, variant):
    """Return uint8 image batch for the windows ending at `ends`. Per-window min-max normalization (causal)."""
    O, H, L, C = _win(o, ends), _win(h, ends), _win(l, ends), _win(c, ends)   # each (B,W)
    B = len(ends)
    if variant == "gaf":
        x = C
        mn = x.min(1, keepdims=True); mx = x.max(1, keepdims=True); rng = np.maximum(mx - mn, 1e-12)
        xs = 2 * (x - mn) / rng - 1.0                         # scale to [-1,1]
        xs = np.clip(xs, -1, 1); phi = np.arccos(xs)          # (B,W)
        gasf = np.cos(phi[:, :, None] + phi[:, None, :])      # (B,W,W)
        gadf = np.sin(phi[:, None, :] - phi[:, :, None])      # antisymmetric (sign-carrying)
        img = np.stack([gasf, gadf], axis=1).astype("float32")  # (B,2,W,W) -- float (not uint8) for GAF
        return img
    lo = L.min(1, keepdims=True); hi = H.max(1, keepdims=True); rng = np.maximum(hi - lo, 1e-12)
    def row(v):  # normalized price -> integer pixel row (0=top, HPX-1=bottom). higher price = lower row index
        nrm = (v - lo) / rng                                  # (B,W) in [0,1]
        return np.clip(np.round((1 - nrm) * (HPX - 1)), 0, HPX - 1).astype(np.int16)
    rows = np.arange(HPX)[None, :, None]                      # (1,HPX,1)
    if variant == "hist":
        Crow = row(C)                                         # (B,W); bottom-anchored bar up to the close level
        img = (rows >= Crow[:, None, :]).astype(np.uint8)     # fill from close-row DOWN to bottom -> (B,HPX,W)
        return img[:, None, :, :]                             # (B,1,HPX,W)
    # ohlc: 3 channels — wick(L..H), up-body, down-body
    Hr, Lr, Or, Cr = row(H), row(L), row(O), row(C)
    hi_r = np.minimum(Or, Cr); lo_r = np.maximum(Or, Cr)      # body spans rows between open & close (top<bottom)
    wick = ((rows >= Hr[:, None, :]) & (rows <= Lr[:, None, :])).astype(np.uint8)
    body = ((rows >= hi_r[:, None, :]) & (rows <= lo_r[:, None, :]))
    up = (C >= O)[:, None, :]                                 # (B,1,W)
    upb = (body & up).astype(np.uint8); dnb = (body & ~up).astype(np.uint8)
    return np.stack([wick, upb, dnb], axis=1)                 # (B,3,HPX,W)

def in_ch(variant): return {"hist": 1, "ohlc": 3, "gaf": 2}[variant]
def img_hw(variant): return (W, W) if variant == "gaf" else (HPX, W)


# ----------------------------- model -----------------------------
class CNN(nn.Module):
    def __init__(s, ch, hw):
        super().__init__()
        s.conv = nn.Sequential(
            nn.Conv2d(ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25))
        fh, fw = hw[0] // 2, hw[1] // 2
        s.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * fh * fw, 128), nn.ReLU(),
                               nn.Dropout(0.5), nn.Linear(128, 1))
    def forward(s, x): return s.head(s.conv(x)).squeeze(-1)


def to_tensor(img):
    t = torch.from_numpy(img)
    return t.float() if img.dtype != np.uint8 else t.float()  # uint8->float (0/1); gaf already float


def predict(model, o, h, l, c, ends, variant, chunk=16384):
    model.eval(); ps = np.empty(len(ends), "float32")
    with torch.no_grad():
        for s in range(0, len(ends), chunk):
            e = min(s + chunk, len(ends))
            x = to_tensor(render(o, h, l, c, ends[s:e], variant))
            ps[s:e] = torch.sigmoid(model(x)).numpy()
            del x
    return ps


# ----------------------------- eval helpers (deriv-faithful) -----------------------------
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


def main():
    hb(f"variant={VARIANT} W={W} subsample={SUBSAMPLE} epochs={EPOCHS} in_ch={in_ch(VARIANT)} hw={img_hw(VARIANT)}")
    tr = load("train"); va = load("val"); te = load("test"); oo = load("oos")
    etr = eligible(*tr[4:]); eva = eligible(*va[4:]); ete = eligible(*te[4:]); eoo = eligible(*oo[4:])
    if SESSION != "all":     # DST-correct: restrict decision bars (train+val+test+oos) to the session
        etr = etr[session_mask(tr[4][etr], SESSION)]; eva = eva[session_mask(va[4][eva], SESSION)]
        ete = ete[session_mask(te[4][ete], SESSION)]; eoo = eoo[session_mask(oo[4][eoo], SESSION)]
    hb(f"[{SESSION}] eligible: train={len(etr):,} val={len(eva):,} test={len(ete):,} oos={len(eoo):,}")

    # subsample train (decorrelate 97% window overlap + cap memory); render once to uint8/float in RAM
    rng = np.random.default_rng(7)
    sel = np.sort(rng.choice(etr, size=min(SUBSAMPLE, len(etr)), replace=False))
    Xtr = render(tr[0], tr[1], tr[2], tr[3], sel, VARIANT)
    ytr = tr[5][sel].astype("float32")
    hb(f"train images {Xtr.shape} {Xtr.dtype} ~{Xtr.nbytes/1e6:.0f}MB  up_rate={ytr.mean():.4f}")

    model = CNN(in_ch(VARIANT), img_hw(VARIANT))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t = to_tensor(Xtr); ytr_t = torch.from_numpy(ytr); ntr = len(sel)
    # VAL early-stop subsample (full VAL only for final eval) — keeps per-epoch cost down
    eva_es = np.sort(rng.choice(eva, size=min(VAL_ES, len(eva)), replace=False))
    yva_es = va[5][eva_es].astype("float32")
    best_auc = -1; best_state = None; patience = 3; bad = 0
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(ntr)
        for s in range(0, ntr, BS):
            bi = perm[s:s + BS]
            opt.zero_grad(); loss = lossf(model(Xtr_t[bi]), ytr_t[bi]); loss.backward(); opt.step()
        pva = predict(model, va[0], va[1], va[2], va[3], eva_es, VARIANT)
        auc = roc_auc_score(yva_es, pva)
        hb(f"ep{ep+1}/{EPOCHS} loss={loss.item():.4f} VAL_dirAUC={auc:.4f}")
        if auc > best_auc + 1e-4:
            best_auc = auc; best_state = {k: v.clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience: hb(f"early stop @ep{ep+1}"); break
    model.load_state_dict(best_state)
    hb(f"BEST VAL dirAUC={best_auc:.4f}")

    # predictions on val/test/oos eligible windows
    out = {}
    for nm, d, ev in (("val", va, eva), ("test", te, ete), ("oos", oo, eoo)):
        p = predict(model, d[0], d[1], d[2], d[3], ev, VARIANT)
        out[nm] = {"ts": d[4][ev], "y": d[5][ev], "mag": d[6][ev], "p": p}
        hb(f"pred {nm}: n={len(ev):,} AUC={roc_auc_score(d[5][ev], p):.4f}")
    np.savez(f"{ROOT}/barcnn_pred_{VARIANT}{SUF}.npz",
             **{f"{nm}_{k}": v for nm, dd in out.items() for k, v in dd.items()})
    torch.save(model.state_dict(), f"{ROOT}/barcnn_{VARIANT}{SUF}.pt")

    # ---- VAL worst-half confidence-gate selection (NEVER val-acc-max), then per-year held-out selective ----
    pva = out["val"]["p"]; yv = out["val"]["y"]; conf_v = np.abs(pva - 0.5)
    report = {"variant": VARIANT, "session": SESSION, "W": W, "subsample": int(min(SUBSAMPLE, len(etr))), "epochs_run": ep + 1,
              "best_val_dirAUC": round(float(best_auc), 4),
              "test_AUC": round(float(roc_auc_score(out["test"]["y"], out["test"]["p"])), 4),
              "oos_AUC": round(float(roc_auc_score(out["oos"]["y"], out["oos"]["p"])), 4),
              "per_year": {}, "coverages": {}}
    # pool test+oos for per-year; choose conf threshold per coverage on VAL
    allts = np.concatenate([out["test"]["ts"], out["oos"]["ts"]])
    ally = np.concatenate([out["test"]["y"], out["oos"]["y"]])
    allp = np.concatenate([out["test"]["p"], out["oos"]["p"]])
    yr = pd.to_datetime(allts, unit="s", utc=True).year.values
    for cov in (0.10, 0.05, 0.02):
        thr = float(np.quantile(conf_v, 1 - cov))
        cells = {}
        for Y in (2024, 2025, 2026):
            m = (yr == Y) & (np.abs(allp - 0.5) >= thr)
            sel_i = nonoverlap_chrono(allts[(yr == Y)], (np.abs(allp - 0.5) >= thr)[yr == Y])
            tsY = allts[yr == Y]; pY = allp[yr == Y]; yY = ally[yr == Y]
            if len(sel_i) < 20: cells[str(Y)] = {"n": int(len(sel_i)), "acc": None}; continue
            corr = ((pY[sel_i] > 0.5).astype(int) == yY[sel_i]).astype(float)
            lo, hi = boot(corr)
            cells[str(Y)] = {"n": int(len(sel_i)), "acc": round(float(corr.mean()), 4),
                             "ci95": [round(lo, 4), round(hi, 4)]}
        report["coverages"][f"cov{cov}"] = {"val_conf_thr": round(thr, 5), "per_year": cells}
    json.dump(report, open(f"{ROOT}/barcnn_{VARIANT}{SUF}_result.json", "w"), indent=1)
    hb(f"WROTE barcnn_pred_{VARIANT}{SUF}.npz + barcnn_{VARIANT}{SUF}_result.json")
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
