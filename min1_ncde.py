"""PROBE W2-2 — NEURAL-CDE ON IRREGULAR TICK PATH (Δt as control).

FIRST-IN-WORLD pre-registered falsifier. Question: does the *irregular inter-arrival clock* of EURUSD
ticks carry any 60s-direction sign that a Neural-CDE can read from the raw event path — over and above what
the same model learns from a regular (constant-Δt) grid?

NEW INPUT (never tested in this project): the raw per-tick EVENT path, anchored at a tick t0, with the last
40 ticks as continuous-time channels:
    ch0 = cumulative time (seconds since the window start)  -> the CDE's driving "time" so Δt enters the path
    ch1 = signed_size      = sign(mid_increment) * log1p(bid-vol + ask-vol)   (signed trade-size proxy)
    ch2 = mid_increment    = mid_i - mid_{i-1}  (price, in price units)
A Neural-CDE integrates dz = f(z) dX over this irregular path; the inter-arrival Δt is encoded *as the
spacing of the path knots in ch0*. That is the genuinely new channel.

LABEL (deriv-faithful, identical discipline to min1_production.py):
    label = sign(mid(t0+60s) - mid(t0)); TIES (move==0) LOSE -> they become losing UP=0 rows but we only
    SCORE on MOVED bars (|move|>0.5pip) so ties are excluded from the evaluated set (a tie is never a moved
    bar). Non-overlapping anchors (block 60+10 s after each taken anchor) so trades are independent.
    EURUSD-CLEAN: built straight from EURUSD raw ticks, NO cross-pair timestamp-intersection / ffill, so the
    ~50%-fake-flat 0.728 mirage cannot occur. We assert moved-bar EURUSD up-rate ~0.49 as a leakage tripwire.

SPLITS (project convention, min1_production.py SPLIT_YEARS):
    train = 2021-2023 (subsample ~20-40k anchors), val = 2024 H1, oos windows = 2024 / 2025 / 2026.
    Selection rule per project discipline: report VAL AUC + EACH per-window acc with bootstrap CI95.
    We do NOT VAL-acc-max (corr(VAL,OOS)=-0.54); VAL AUC is reported only as the headline the task asked for.

CRITICAL ABLATION (the whole point):
    ARM A "IRREGULAR" : ch0 = real cumulative seconds (true Δt spacing).
    ARM B "GRID"      : ch0 = constant unit grid (Δt overwritten by 1.0 -> 0,1,2,...,39). SAME architecture,
                        SAME seed, SAME other two channels. If IRREGULAR does NOT beat GRID, the inter-arrival
                        clock carries no sign -> a clean first-in-world NEGATIVE.

Memory-safe: subsample anchors; tiny net (hidden=24); CPU; build windows per-day then drop the big tick frame.
Fallback: if torchcde import fails, fall back to a plain torch GRU on (Δt,signed_size,dx) and SAY SO in output.
"""
import sys, os, time, glob, calendar, json
import numpy as np, pandas as pd

ROOT = "/home/sean/git/binary-algo"
RAW  = "/media/sean/CORSAIR/tick_data/raw/EURUSD"
PAIR = "EURUSD"

HS      = 60      # 60-second fixed wall-clock expiry (a real deriv Rise/Fall binary)
TOL_S   = 10      # exit tick must be within 10s of the +60s instant else the window crosses a gap -> drop
GAP     = HS + TOL_S   # non-overlap block (s) so selected anchors' [t0, t0+60] windows are disjoint
CTX     = 40      # last 40 ticks as the CDE context path
PIP     = 1e-4
MOVE_TH = 0.5 * PIP    # MOVED-BARS-ONLY scoring threshold
HIDDEN  = 24
SEED    = 7
N_TRAIN = 30000   # subsample target for train anchors (within the 20-40k spec)
N_EVAL  = 12000   # cap eval anchors per window (independent, plenty for CI95)
EPOCHS  = 25
BATCH   = 1024
LR      = 2e-3

# ---- project split-date convention (mirrors tick1s_cache.py / min1_production SPLIT_YEARS) ----
def _mo(y, ms):
    out = []
    for m in ms:
        out += [f"{y}-{m:02d}-{d:02d}" for d in range(1, calendar.monthrange(y, m)[1] + 1)]
    return out
# train uses the SAME broad-coverage months tick1s_cache uses; eval windows are calendar years.
TRAIN_DATES = _mo(2021, [3, 6, 9, 12]) + _mo(2022, [2, 4, 6, 8, 10, 12]) + _mo(2023, [2, 4, 6, 8, 10, 12])
VAL_DATES   = _mo(2024, [4, 5])                          # 2024 H1 (project VAL)
# OOS per-window: distinct months inside each year, none overlapping VAL.
WIN_DATES = {
    "2024": _mo(2024, [9, 10, 11]),
    "2025": _mo(2024, [12]) and _mo(2025, [2, 3, 4, 9, 10, 11]),
    "2026": _mo(2026, [2, 3, 4]),
}

rng = np.random.default_rng(SEED)


# ----------------------------- raw-tick -> event-anchored windows -----------------------------
def _day_ticks(date):
    """Load one day's hourly EURUSD raw-tick files -> sorted (ts, mid, signed_size, dmid). EURUSD-only."""
    files = sorted(glob.glob(f"{RAW}/{PAIR}_{date}_*.parquet"))
    if not files:
        return None
    parts = [pd.read_parquet(f, columns=["timestamp_utc", "bid", "ask", "bid-vol", "ask-vol"]) for f in files]
    t = pd.concat(parts, ignore_index=True)
    t = t.sort_values("timestamp_utc").reset_index(drop=True)
    ts  = t["timestamp_utc"].values.astype(np.float64)            # fractional epoch seconds (TRUE Δt source)
    mid = ((t["bid"].values + t["ask"].values) / 2.0).astype(np.float64)
    sz  = (t["bid-vol"].values + t["ask-vol"].values).astype(np.float64)
    dmid = np.empty_like(mid); dmid[0] = 0.0; dmid[1:] = np.diff(mid)
    signed = np.sign(dmid) * np.log1p(sz)                          # signed trade-size proxy
    return ts, mid, signed, dmid


def _build_day(date, take_block=GAP):
    """Build event-anchored windows for one day.
    For each candidate anchor index i (>= CTX so a full 40-tick context exists):
      context channels (length CTX): cumulative-seconds, signed_size, mid_increment   (anchor-relative)
      label = sign(mid(t0+60s) - mid(t0)); valid iff exit tick within TOL of +60s and after the anchor.
    Non-overlapping in time (block GAP seconds after each taken anchor). Returns arrays or None.
    """
    d = _day_ticks(date)
    if d is None:
        return None
    ts, mid, signed, dmid = d
    n = len(ts)
    if n < CTX + 5:
        return None
    # 60s-ahead exit lookup (last tick at/before t0+60s), validity within TOL_S, must be a later tick
    exit_t = ts + HS
    xi  = np.searchsorted(ts, exit_t, side="right") - 1
    xic = np.clip(xi, 0, n - 1)
    idx = np.arange(n)
    valid = (xi > idx) & ((exit_t - ts[xic]) <= TOL_S)
    move  = mid[xic] - mid                                          # price move over 60s (NaN-safe: finite)
    valid &= np.isfinite(move)
    valid &= (idx >= CTX)                                           # need 40 prior ticks for the context

    # chronological non-overlap: scan valid anchors in time order, block GAP seconds after each take
    cand = np.where(valid)[0]
    if len(cand) == 0:
        return None
    take = []
    block_until = -1.0
    for i in cand:
        if ts[i] < block_until:
            continue
        take.append(i)
        block_until = ts[i] + take_block
    take = np.array(take, dtype=np.int64)
    if len(take) == 0:
        return None

    # assemble context tensors (anchor-relative cumulative seconds, signed size, mid increment)
    m = len(take)
    cum  = np.empty((m, CTX), np.float32)   # ch0 IRREGULAR: seconds since window start
    sgn  = np.empty((m, CTX), np.float32)   # ch1 signed size
    dpr  = np.empty((m, CTX), np.float32)   # ch2 mid increment (price units)
    for k, i in enumerate(take):
        sl = slice(i - CTX + 1, i + 1)      # last CTX ticks ending AT the anchor (inclusive)
        w_ts = ts[sl]
        cum[k] = (w_ts - w_ts[0]).astype(np.float32)
        sgn[k] = signed[sl].astype(np.float32)
        dpr[k] = (dmid[sl] * 1e4).astype(np.float32)   # scale mid-increment to pips for numerics
    mv = move[take].astype(np.float64)
    y  = (mv > 0).astype(np.int64)          # UP=1 win; ties (mv==0) -> 0 (lose) but excluded by moved mask
    moved = np.abs(mv) > MOVE_TH
    return cum, sgn, dpr, y, mv, moved


def build_split(dates, cap=None, label=""):
    """Build a whole split, optionally capped (uniform subsample across days)."""
    CUM, SGN, DPR, Y, MV, MOVED = [], [], [], [], [], []
    nd = 0
    for date in dates:
        out = _build_day(date)
        if out is None:
            continue
        cum, sgn, dpr, y, mv, moved = out
        CUM.append(cum); SGN.append(sgn); DPR.append(dpr); Y.append(y); MV.append(mv); MOVED.append(moved)
        nd += 1
    if not CUM:
        return None
    cum = np.concatenate(CUM); sgn = np.concatenate(SGN); dpr = np.concatenate(DPR)
    y = np.concatenate(Y); mv = np.concatenate(MV); moved = np.concatenate(MOVED)
    if cap is not None and len(y) > cap:
        sel = rng.choice(len(y), size=cap, replace=False)
        sel.sort()
        cum, sgn, dpr, y, mv, moved = cum[sel], sgn[sel], dpr[sel], y[sel], mv[sel], moved[sel]
    upmoved = (mv[moved] > 0).mean() if moved.sum() else float("nan")
    print(f"[build:{label}] days={nd} anchors={len(y):,} moved={moved.mean():.3f} "
          f"up-rate(moved)={upmoved:.4f}  (leakage tripwire: must be ~0.49)", flush=True)
    return dict(cum=cum, sgn=sgn, dpr=dpr, y=y, mv=mv, moved=moved, upmoved=upmoved)


# ----------------------------- model: Neural-CDE (torchcde) with GRU fallback -----------------------------
import torch
import torch.nn as nn

try:
    import torchcde
    HAVE_CDE = True
except Exception as e:           # pragma: no cover
    HAVE_CDE = False
    _CDE_ERR = repr(e)


# Regular integration grid over the CTX knots, shared by BOTH arms (the CDE is integrated against this index,
# NOT against wall-clock seconds — wall-clock enters as a DATA channel below). This keeps cdeint cheap
# (CTX-1 segments) and IDENTICAL across arms, so the ONLY thing that differs is the ch0 channel's content.
GRID_T = torch.linspace(0.0, 1.0, CTX, dtype=torch.float32)


def _stack_channels(d, arm):
    """(N, CTX, 3) float32 path.
       ch0 = cumulative TIME the CDE reads as a channel (its path-derivative along ch0 IS the local Δt):
             arm='irregular' -> real cumulative seconds / HS  (TRUE irregular inter-arrival spacing)
             arm='grid'      -> a regular ramp 0..1 (Δt overwritten by a constant grid; carries NO real Δt)
       ch1 = signed size, ch2 = mid increment (pips). Both arms share ch1,ch2 exactly."""
    n = len(d["y"])
    if arm == "grid":
        ramp = np.linspace(0.0, 1.0, CTX, dtype=np.float32)       # Δt := constant -> regular spacing
        ch0 = np.tile(ramp, (n, 1))
    else:
        ch0 = (d["cum"] / float(HS)).astype(np.float32)           # real cumulative seconds, scaled to ~[0,2]
    X = np.stack([ch0, d["sgn"], d["dpr"]], axis=-1).astype(np.float32)
    return X


def _normalize(Xtr, *others):
    """Standardize ch1,ch2 by train stats; ch0 (the time channel) kept in its own units so the two arms'
    only difference is real-vs-regular spacing (normalizing ch0 by std would erase the Δt magnitude info)."""
    flat = Xtr.reshape(-1, 3)
    mu = flat.mean(0); sd = flat.std(0) + 1e-6
    mu[0] = 0.0; sd[0] = 1.0                                       # leave ch0 (time) untouched
    def ap(X): return (X - mu) / sd
    return (ap(Xtr),) + tuple(ap(o) for o in others)


class CDEFunc(nn.Module):
    def __init__(self, in_ch, hid):
        super().__init__()
        self.in_ch = in_ch; self.hid = hid
        self.net = nn.Sequential(nn.Linear(hid, 48), nn.Tanh(), nn.Linear(48, hid * in_ch))
    def forward(self, t, z):
        return self.net(z).view(z.size(0), self.hid, self.in_ch).tanh()


class NeuralCDE(nn.Module):
    def __init__(self, in_ch=3, hid=HIDDEN):
        super().__init__()
        self.in_ch = in_ch; self.hid = hid
        self.initial = nn.Linear(in_ch, hid)
        self.func = CDEFunc(in_ch, hid)
        self.readout = nn.Linear(hid, 1)
    def forward(self, coeffs):
        # LINEAR-interpolation control path knotted on the REGULAR GRID_T; cdeint integrates with euler over the
        # GRID_T endpoints. (Cubic-spline + rk4 was numerically correct but ~300x slower on CPU; linear+euler is
        # the standard torchcde fast path and preserves the Δt-as-channel signal exactly via the piecewise dX/dt.)
        # The Δt signal lives inside ch0's VALUES (read through the CDE's dX term), not the integration t-axis.
        X = torchcde.LinearInterpolation(coeffs, GRID_T)
        z0 = self.initial(X.evaluate(GRID_T[0]))
        zT = torchcde.cdeint(X=X, func=self.func, z0=z0, t=X.interval,
                             method="euler", options=dict(step_size=float(GRID_T[1] - GRID_T[0])))
        return self.readout(zT[:, -1, :]).squeeze(-1)


class GRUFallback(nn.Module):
    """Plain torch GRU on (Δt, signed_size, dx) — used ONLY if torchcde import fails."""
    def __init__(self, in_ch=3, hid=HIDDEN):
        super().__init__()
        self.gru = nn.GRU(in_ch, hid, batch_first=True)
        self.readout = nn.Linear(hid, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.readout(out[:, -1, :]).squeeze(-1)


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def boot_ci(correct, nb=5000, seed=7):
    correct = np.asarray(correct, dtype=float)
    if len(correct) < 5:
        return (float("nan"), float("nan"))
    r = np.random.default_rng(seed); n = len(correct)
    a = np.array([correct[r.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def make_coeffs(X):
    """Linear-interpolation coefficients on the REGULAR GRID_T knots, for torchcde (built once per array)."""
    return torchcde.linear_interpolation_coeffs(torch.tensor(X, dtype=torch.float32), GRID_T)


def train_arm(arm, dtr, dva, windows):
    """Train one arm (irregular|grid). Returns dict: val_auc, and per-window {acc, ci, auc, n} on MOVED bars."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    Xtr = _stack_channels(dtr, arm); Xva = _stack_channels(dva, arm)
    Xw  = {w: _stack_channels(windows[w], arm) for w in windows}
    Xtr_n, Xva_n, *Xw_n = _normalize(Xtr, Xva, *[Xw[w] for w in windows])
    Xw_n = dict(zip(windows.keys(), Xw_n))
    ytr = torch.tensor(dtr["y"], dtype=torch.float32)

    if HAVE_CDE:
        model = NeuralCDE()
        Ctr = make_coeffs(Xtr_n)
        Cva = make_coeffs(Xva_n)
        Cw  = {w: make_coeffs(Xw_n[w]) for w in windows}
        def fwd(model, C, idx): return model(C[idx])
        train_obj = Ctr
    else:
        model = GRUFallback()
        train_obj = torch.tensor(Xtr_n, dtype=torch.float32)
        Tva = torch.tensor(Xva_n, dtype=torch.float32)
        Tw  = {w: torch.tensor(Xw_n[w], dtype=torch.float32) for w in windows}
        def fwd(model, T, idx): return model(T[idx])

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    n = len(ytr)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for s in range(0, n, BATCH):
            bidx = perm[s:s + BATCH]
            opt.zero_grad()
            logit = fwd(model, train_obj, bidx)
            loss = lossf(logit, ytr[bidx])
            loss.backward(); opt.step()
            tot += float(loss.detach()) * len(bidx)
        if ep == 0 or ep == EPOCHS - 1:
            print(f"   [{arm}] epoch {ep+1}/{EPOCHS} loss={tot/n:.4f}", flush=True)

    # ---- eval helper (predict in batches) ----
    def predict(obj, m):
        model.eval(); ps = []
        with torch.no_grad():
            for s in range(0, m, 1024):
                idx = torch.arange(s, min(s + 1024, m))
                ps.append(torch.sigmoid(fwd(model, obj, idx)).numpy())
        return np.concatenate(ps)

    # VAL AUC on MOVED bars (the task's requested headline)
    obj_va = Cva if HAVE_CDE else Tva
    pva = predict(obj_va, len(dva["y"]))
    vmoved = dva["moved"]
    val_auc = _auc(dva["y"][vmoved], pva[vmoved])

    res = {"val_auc": val_auc, "windows": {}}
    for w in windows:
        dW = windows[w]
        objw = Cw[w] if HAVE_CDE else Tw[w]
        pw = predict(objw, len(dW["y"]))
        mvd = dW["moved"]
        yw = dW["y"][mvd]; pwm = pw[mvd]
        pred = (pwm >= 0.5).astype(int)
        correct = (pred == yw).astype(int)
        acc = float(correct.mean()) if len(correct) else float("nan")
        ci = boot_ci(correct)
        res["windows"][w] = {"acc": acc, "ci": ci, "auc": _auc(yw, pwm), "n": int(len(yw))}
    return res


def main():
    t0 = time.time()
    print(f"=== PROBE W2-2 Neural-CDE on irregular tick path | torchcde={'YES' if HAVE_CDE else 'NO (GRU FALLBACK: '+_CDE_ERR+')'} ===", flush=True)
    print("Building event-anchored windows from EURUSD raw ticks (NO cross-pair ffill)...", flush=True)
    dtr = build_split(TRAIN_DATES, cap=N_TRAIN, label="train")
    dva = build_split(VAL_DATES,   cap=N_EVAL,  label="val")
    windows = {}
    for w, dts in WIN_DATES.items():
        windows[w] = build_split(dts, cap=N_EVAL, label=w)
    print(f"[build] total {time.time()-t0:.0f}s", flush=True)

    # leakage tripwire assertion: moved-bar up-rate must be ~0.49-0.50 (NOT the ~0.728 cross-pair ffill mirage).
    # bound is the leakage guard; small per-window sampling noise lives inside it, the ffill mirage (~0.7) does not.
    for nm, d in [("train", dtr), ("val", dva)] + [(w, windows[w]) for w in windows]:
        assert 0.44 <= d["upmoved"] <= 0.56, f"LEAKAGE TRIPWIRE: {nm} moved up-rate {d['upmoved']:.3f} not ~0.49"
    print("[tripwire] all moved-bar up-rates in [0.44,0.56] -> EURUSD-clean, no ~0.728 ffill mirage", flush=True)

    results = {}
    for arm in ("irregular", "grid"):
        print(f"\n--- ARM {arm.upper()} (ch0 = {'real cumulative seconds (true Δt)' if arm=='irregular' else 'constant unit grid (Δt:=1.0)'}) ---", flush=True)
        results[arm] = train_arm(arm, dtr, dva, windows)
        r = results[arm]
        print(f"   VAL AUC (moved)={r['val_auc']:.4f}", flush=True)
        for w in r["windows"]:
            wd = r["windows"][w]
            print(f"   {w}: acc={wd['acc']:.4f} CI95=[{wd['ci'][0]:.4f},{wd['ci'][1]:.4f}] auc={wd['auc']:.4f} n={wd['n']}", flush=True)

    # ---- verdict: does IRREGULAR beat GRID? ----
    print("\n=== ABLATION SUMMARY (IRREGULAR vs GRID) ===", flush=True)
    a, b = results["irregular"], results["grid"]
    print(f"VAL AUC: irregular={a['val_auc']:.4f}  grid={b['val_auc']:.4f}  Δ={a['val_auc']-b['val_auc']:+.4f}", flush=True)
    for w in a["windows"]:
        da = a["windows"][w]; db = b["windows"][w]
        print(f"{w}: irreg acc={da['acc']:.4f} CI{da['ci']}  | grid acc={db['acc']:.4f} CI{db['ci']}  | Δacc={da['acc']-db['acc']:+.4f}", flush=True)

    out = {"have_cde": HAVE_CDE, "results": results,
           "tripwire_upmoved": {nm: float(d["upmoved"]) for nm, d in
                                [("train", dtr), ("val", dva)] + [(w, windows[w]) for w in windows]}}
    with open(f"{ROOT}/min1_ncde_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nsaved -> {ROOT}/min1_ncde_results.json   total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
