"""KRONOS FINE-TUNE (predictor, single-process GPU) + session-only direction eval — DST-correct, deriv-faithful.

User directive: fine-tune Kronos for EURUSD 1m direction, per DST-correct session (NY/LDN/Asia), training + eval on
ONLY that session's bars. The official finetune/train_predictor.py is GPU/DDP-only (torchrun, cuda hardcoded); this is
a single-process adaptation: frozen base tokenizer, fine-tune the Kronos-small predictor on the session-only 1m stream,
then run the SAME direction eval as kronos_dir.py (slice(i-L,i) context, predict bar i, P_up=sign(pred_close-C[i-1]),
score vs precomputed deriv-faithful 60s label y) but on a session-filtered, contiguity-checked stream.

Session-only discipline (the user's correction): train/val/test/oos bars are all restricted to ONE session BEFORE
windowing; context windows must be fully 60s-contiguous (no cross-day/cross-session leakage into the L look-back).
Splits (Kronos caches): train=2021-2023, val=2024 (early-stop on next-token val LOSS), test+oos=2024-2026 (direction).
Held-out honest read = 2025/2026 (2024 mildly optimistic since val early-stop touches 2024 distribution — noted).

Run: ~/binary-algo-venv/bin/python kronos_ft.py <ny|ldn|asia> [EPOCHS=12] [BATCH=32] [L=256] [K=20] [N_PER_YR=3000]
  -> models/kronos_ft_<sess>/ (best predictor)  +  kronos_dir_ft_<sess>_result.json
"""
import sys, os, json, time, gc, numpy as np, pandas as pd
from itertools import combinations
import torch
sys.path.insert(0, "/home/sean/git/Kronos")
from model import Kronos, KronosTokenizer, KronosPredictor
from model.kronos import calc_time_stamps
from sessions import session_mask

SESSION  = (sys.argv[1] if len(sys.argv) > 1 else "ny").lower()
EPOCHS   = int(sys.argv[2]) if len(sys.argv) > 2 else 12
BATCH    = int(sys.argv[3]) if len(sys.argv) > 3 else 32
L        = int(sys.argv[4]) if len(sys.argv) > 4 else 256          # look-back context (bars) — train window & eval
K        = int(sys.argv[5]) if len(sys.argv) > 5 else 20           # eval sample_count
N_PER_YR = int(sys.argv[6]) if len(sys.argv) > 6 else 3000         # eval decision bars per year (GPU budget)
WIN      = L + 2                                                   # train window = lookback L + predict 1 + 1
STEPS_PER_EPOCH = int(os.environ.get("STEPS_PER_EPOCH", "1200")); PATIENCE = 3; EVAL_BATCH = 16
VAL_CAP = int(os.environ.get("VAL_CAP", "400"))                    # val batches cap (smoke override)
HS, TOL, GAP = 60, 10, 70; BREAKEVEN = 0.541
ROOT = "/home/sean/git/binary-algo"; OUT = f"{ROOT}/ohlc_cache"
TOK_ID = "NeoQuasar/Kronos-Tokenizer-base"; BASE = "NeoQuasar/Kronos-small"
SAVE = f"{ROOT}/models/kronos_ft_{SESSION}"
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.manual_seed(100); np.random.seed(100)
T0 = time.time()
def hb(m): print(f"HB[{time.time()-T0:.0f}s] {m}", flush=True)


# ---------- data: session-only contiguous streams ----------
def load_stream(sp):
    b = pd.read_parquet(f"{OUT}/EURUSD_1m_{sp}.parquet")
    t = b["t"].values.astype("int64")
    m = session_mask(t, SESSION)
    b = b.loc[m].reset_index(drop=True); t = t[m]
    O, Hg, Lw, C = (b[c].values.astype(np.float32) for c in ("open", "high", "low", "close"))
    V = b["vol"].values.astype(np.float32)
    feats = np.stack([O, Hg, Lw, C, V, np.zeros_like(O)], axis=1)            # OHLCV + amt(0), matches zero-shot
    stamps = calc_time_stamps(pd.Series(pd.to_datetime(t, unit="s", utc=True))).values.astype(np.float32)
    return {"feats": feats, "stamps": stamps, "t": t, "C": C,
            "y": b["y"].values.astype(int), "mag": b["mag"].values.astype(float),
            "valid": b["valid"].values.astype(bool)}

def contiguous_starts(t, win):
    """start indices s where bars [s, s+win) are fully 60s-contiguous (t[s+win-1]-t[s]==(win-1)*60)."""
    n = len(t)
    if n < win: return np.array([], dtype=int)
    s = np.arange(0, n - win + 1)
    return s[(t[s + win - 1] - t[s]) == (win - 1) * 60]


def make_batch(D, starts):
    idx = starts[:, None] + np.arange(WIN)[None, :]                          # [B, WIN]
    x = D["feats"][idx]                                                      # [B, WIN, 6]
    mu = x[:, :L, :].mean(1, keepdims=True); sd = x[:, :L, :].std(1, keepdims=True)
    x = np.clip((x - mu) / (sd + 1e-5), -5.0, 5.0)
    xs = D["stamps"][idx]                                                    # [B, WIN, 5]
    return torch.from_numpy(x).to(DEV), torch.from_numpy(xs).to(DEV)


# ---------- eval helpers (mirror kronos_dir.py) ----------
def nonoverlap_chrono(ts, idx, gap=GAP):
    take = []; bu = -1
    for i in idx:
        if ts[i] < bu: continue
        take.append(i); bu = int(ts[i]) + gap
    return np.array(take, dtype=int)

def boot(c, nb=4000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def cpcv(ts, win, ng=8, k=2):
    n = len(win); e = np.linspace(0, n, ng + 1).astype(int); grp = [(e[i], e[i+1]) for i in range(ng)]; pa = []
    for tg in combinations(range(ng), k):
        ii = np.concatenate([np.arange(grp[g][0], grp[g][1]) for g in tg])
        if len(ii) >= 25: pa.append(float(win[ii].mean()))
    pa = np.array(pa)
    return (round(float(np.percentile(pa, 10)), 4), round(float((pa >= BREAKEVEN).mean()), 3)) if len(pa) else (None, None)


# ---------- train ----------
def train(tok, mdl):
    tr = load_stream("train"); va = load_stream("val")
    s_tr = contiguous_starts(tr["t"], WIN); s_va = contiguous_starts(va["t"], WIN)
    hb(f"[{SESSION}] train stream {len(tr['t'])} bars -> {len(s_tr)} contig windows | val {len(va['t'])} -> {len(s_va)}")
    if len(s_tr) < BATCH * 10:
        hb("TOO FEW training windows — abort"); return False
    opt = torch.optim.AdamW(mdl.parameters(), lr=4e-5, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-5, steps_per_epoch=STEPS_PER_EPOCH,
                                                epochs=EPOCHS, pct_start=0.03, div_factor=10)
    rng = np.random.default_rng(100); best = float("inf"); bad = 0
    va_starts = s_va[np.linspace(0, len(s_va) - 1, min(len(s_va), VAL_CAP * BATCH)).astype(int)] if len(s_va) else s_va
    for ep in range(EPOCHS):
        mdl.train(); tl = 0.0
        for step in range(STEPS_PER_EPOCH):
            bs = rng.choice(s_tr, size=BATCH, replace=len(s_tr) < BATCH)
            x, xs = make_batch(tr, bs)
            with torch.no_grad(): t0_, t1_ = tok.encode(x, half=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = mdl(t0_[:, :-1], t1_[:, :-1], xs[:, :-1, :])
                loss, _, _ = mdl.head.compute_loss(lg[0], lg[1], t0_[:, 1:], t1_[:, 1:])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 3.0); opt.step(); sched.step()
            tl += float(loss.item())
            if (step + 1) % 400 == 0: hb(f"  ep{ep+1} step{step+1}/{STEPS_PER_EPOCH} loss={tl/(step+1):.4f} lr={opt.param_groups[0]['lr']:.2e}")
        # validation next-token loss
        mdl.eval(); vl = 0.0; nb = 0
        with torch.no_grad():
            for s in range(0, len(va_starts), BATCH):
                bs = va_starts[s:s+BATCH]
                if len(bs) < 2: continue
                x, xs = make_batch(va, bs); t0_, t1_ = tok.encode(x, half=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = mdl(t0_[:, :-1], t1_[:, :-1], xs[:, :-1, :])
                    vloss, _, _ = mdl.head.compute_loss(lg[0], lg[1], t0_[:, 1:], t1_[:, 1:])
                vl += float(vloss.item()); nb += 1
        vl = vl / max(nb, 1)
        gpu = torch.cuda.max_memory_allocated() / 1e9
        hb(f"[{SESSION}] EPOCH {ep+1}/{EPOCHS} train_loss={tl/STEPS_PER_EPOCH:.4f} val_loss={vl:.4f} (best {best:.4f}) peakVRAM={gpu:.1f}G")
        if vl < best - 1e-4:
            best = vl; bad = 0; os.makedirs(SAVE, exist_ok=True); mdl.save_pretrained(SAVE)
            hb(f"  saved best -> {SAVE}")
        else:
            bad += 1
            if bad >= PATIENCE: hb(f"early stop (patience {PATIENCE})"); break
    return os.path.exists(SAVE)


# ---------- eval (session-only context, contiguity-checked) ----------
def evaluate():
    tok = KronosTokenizer.from_pretrained(TOK_ID); mdl = Kronos.from_pretrained(SAVE)
    pred = KronosPredictor(mdl, tok, device=DEV, max_context=max(L, 256))
    frames = []
    for sp in ("test", "oos"):
        b = pd.read_parquet(f"{OUT}/EURUSD_1m_{sp}.parquet"); frames.append(b)
    B = pd.concat(frames).reset_index(drop=True)
    t = B["t"].values.astype("int64"); m = session_mask(t, SESSION)
    B = B.loc[m].reset_index(drop=True); t = t[m]
    O = B["open"].values; Hg = B["high"].values; Lw = B["low"].values; C = B["close"].values; V = B["vol"].values
    y = B["y"].values.astype(int); mag = B["mag"].values; valid = B["valid"].values.astype(bool)
    yr = pd.to_datetime(t, unit="s", utc=True).year.values
    n = len(B); ar = np.arange(n)
    # contiguity: context [i-L,i) all 60s AND bar i is true next minute -> t[i]-t[i-L]==L*60
    contig = np.zeros(n, bool)
    ok = ar >= L
    contig[ok] = (t[ar[ok]] - t[ar[ok] - L]) == L * 60
    elig = np.where(valid & (mag > 0) & contig)[0]
    hb(f"[{SESSION}] eval eligible (session+contig+valid+moved): {len(elig)}")
    pick = []
    for Y in (2024, 2025, 2026):
        ey = elig[yr[elig] == Y]
        if len(ey) > N_PER_YR: ey = ey[np.linspace(0, len(ey) - 1, N_PER_YR).astype(int)]
        pick.append(ey)
    pick = nonoverlap_chrono(t, np.sort(np.concatenate(pick)))
    hb(f"[{SESSION}] eval decision bars: {len(pick)} (nonoverlap)")
    cols = ["open", "high", "low", "close", "volume", "amount"]
    Pup = np.full(len(pick), np.nan)
    for s in range(0, len(pick), EVAL_BATCH):
        chunk = pick[s:s+EVAL_BATCH]; dfl, xtl, ytl = [], [], []
        for i in chunk:
            sl = slice(i - L, i)
            dfl.append(pd.DataFrame({"open": O[sl], "high": Hg[sl], "low": Lw[sl], "close": C[sl],
                                     "volume": V[sl], "amount": 0.0})[cols])
            xtl.append(pd.Series(pd.to_datetime(t[sl], unit="s", utc=True)))
            ytl.append(pd.Series(pd.to_datetime(t[i:i+1], unit="s", utc=True)))
        outs = pred.predict_batch(df_list=dfl, x_timestamp_list=xtl, y_timestamp_list=ytl,
                                  pred_len=1, T=1.0, top_p=0.9, sample_count=K, verbose=False)
        for j, i in enumerate(chunk):
            Pup[s+j] = 1.0 if float(outs[j]["close"].iloc[0]) > C[i-1] else 0.0
        if s % (EVAL_BATCH * 20) == 0: hb(f"  predicted {s+len(chunk)}/{len(pick)}")
    call = (Pup > 0.5).astype(int)
    tw = t[pick]; yw = y[pick]; order = np.argsort(tw); tw, yw, call = tw[order], yw[order], call[order]
    yrw = pd.to_datetime(tw, unit="s", utc=True).year.values
    res = {"tag": f"ft_{SESSION}", "model": SAVE, "session": SESSION, "L": L, "K": K, "n_eval": int(len(pick)),
           "split": "FT train=2021-23 val=2024(early-stop); eval test+oos=2024-26; honest=2025/26",
           "breakeven": BREAKEVEN, "per_year": {},
           "falsifier": "KILL if no held-out yr CI95-lo>=0.541 OR CPCV path_p10<0.541"}
    any_clear = False
    for Y in (2024, 2025, 2026):
        mm = yrw == Y
        if mm.sum() < 20: continue
        corr = (call[mm] == yw[mm]).astype(float); lo, hi = boot(corr)
        res["per_year"][str(Y)] = {"n": int(mm.sum()), "acc": round(float(corr.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)]}
        if lo >= BREAKEVEN: any_clear = True
    win = (call == yw).astype(float); p10, frac = cpcv(tw, win)
    res["pooled_acc"] = round(float(win.mean()), 4); res["cpcv_path_p10"] = p10; res["cpcv_frac_clear_0.541"] = frac
    res["VERDICT"] = "SURVIVES" if (any_clear and p10 is not None and p10 >= BREAKEVEN) else "KILLED"
    json.dump(res, open(f"{ROOT}/kronos_dir_ft_{SESSION}_result.json", "w"), indent=1)
    hb(f"[{SESSION}] DONE pooled={res['pooled_acc']} per-year={ {Y:res['per_year'][Y]['acc'] for Y in res['per_year']} } "
       f"CPCV p10={p10} frac={frac} -> {res['VERDICT']}")


def main():
    hb(f"FINE-TUNE Kronos-small session={SESSION} dev={DEV} L={L} BATCH={BATCH} EPOCHS={EPOCHS}")
    tok = KronosTokenizer.from_pretrained(TOK_ID).to(DEV).eval()
    for p in tok.parameters(): p.requires_grad_(False)
    mdl = Kronos.from_pretrained(BASE).to(DEV)
    hb(f"loaded {sum(p.numel() for p in mdl.parameters())/1e6:.1f}M predictor params (tokenizer frozen)")
    if not train(tok, mdl):
        hb("training did not save a model — abort eval"); return
    del mdl, tok; gc.collect(); torch.cuda.empty_cache()
    # NOTE: the legacy in-script eval has a 1-bar look-forward MISALIGNMENT (predicts the bar ENDING at the entry
    # instant, scored vs the label's forward window AFTER the entry — disjoint). It is superseded by the corrected
    # multi-timeframe harness kronos_mtf.py. Default: skip (just train+save the valid model); set KRONOS_FT_EVAL=1
    # only to reproduce the legacy (misaligned) number.
    if os.environ.get("KRONOS_FT_EVAL", "0") == "1":
        hb("=== eval fine-tuned model (LEGACY 1-bar-MISALIGNED harness) ===")
        evaluate()
    else:
        hb("=== legacy eval SKIPPED (1-bar misalignment); model saved — re-eval via corrected kronos_mtf.py ===")


if __name__ == "__main__":
    main()
