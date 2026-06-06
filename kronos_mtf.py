"""CORRECTED + multi-timeframe Kronos DIRECTION eval — deriv-faithful, DST per-session, CPCV.

Fixes the 1-bar look-forward MISALIGNMENT in kronos_dir.py/kronos_ft.py (user-caught 2026-06-06): the legacy harness
fed context [i-L .. i-1], predicted bar i, and scored pred_close(i) > C[i-1] — the move INTO the entry, a window
DISJOINT from the deriv label y[i] (forward [t[i]+1, t[i]+1+HS]). This harness ends context AT the decision bar i,
predicts H_steps FORWARD, and scores pred_close(+H) > C[i] (decision-bar close = entry reference) against y[i].
(Synthesis verified: forward label agrees with next-bar sign 92.3%, n=233,950 -> predicting the forward bar vs C[i]
is a faithful proxy; the legacy backward window is not.)

Multi-timeframe (user's idea "use 1m to predict 2m up the chain"): a target horizon H minutes can be predicted from a
context bar grid of GRID minutes (GRID | H) by autoregressing H_steps = H/GRID bars. mode is implicit in (GRID,H):
  - GRID==H  -> NATIVE: H-min context bars, predict 1 bar.            cache EURUSD_{H}m_{split}  (kronos_bars.py H)
  - GRID< H  -> FINE  : GRID-min context bars, predict H/GRID bars.   cache EURUSD_g{GRID}_h{H}_{split} (kronos_bars 2-arg)
The cache MUST carry the forward H-min deriv label (built by kronos_bars.py) — NOT the legacy label-less 5m year files.

Contiguity (refined vs the full-span spec, which is impossible for high-H over weekends): PREDICTED-side contiguity
t[i+H_steps]-t[i] == H_steps*step is ALWAYS required (it is what makes pred_close(+H) land on the label exit).
CONTEXT-side contiguity t[i]-t[i-L+1] == (L-1)*step is required ONLY in strict session-only-INPUT mode (session!=all),
so a context window cannot bridge a session/day gap; in all-session mode gappy context is allowed (Kronos was pretrained
on weekend-gapped bars). nonoverlap GAP = HS+TOL (horizon-correct). Up-rate tripwire [.47,.53]. Falsifier pre-registered.

Run: ~/binary-algo-venv/bin/python kronos_mtf.py <H> <session> <GRID> [L=256] [K=20] [N_PER_YR=3000] [model] [tag]
  e.g. corrected 1m all-sess zero-shot:  kronos_mtf.py 1 all 1 256 20 3000 NeoQuasar/Kronos-small mtf_zs_1m
       1m NY strict, fine-tuned:         kronos_mtf.py 1 ny  1 256 20 3000 models/kronos_ft_ny ftmtf_1m_ny
  -> kronos_dir_mtf_<tag>_result.json
"""
import sys, os, json, time, numpy as np, pandas as pd
from itertools import combinations
sys.path.insert(0, "/home/sean/git/Kronos")
from model import Kronos, KronosTokenizer, KronosPredictor
from sessions import session_mask

H        = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SESSION  = (sys.argv[2] if len(sys.argv) > 2 else "all").lower()
GRID     = int(sys.argv[3]) if len(sys.argv) > 3 else 1
L        = int(sys.argv[4]) if len(sys.argv) > 4 else 256
K        = int(sys.argv[5]) if len(sys.argv) > 5 else 20
N_PER_YR = int(sys.argv[6]) if len(sys.argv) > 6 else 3000
MODEL    = sys.argv[7] if len(sys.argv) > 7 else "NeoQuasar/Kronos-small"
TAG      = sys.argv[8] if len(sys.argv) > 8 else f"mtf_H{H}_g{GRID}_{SESSION}"
assert H % GRID == 0, f"GRID {GRID} must divide H {H}"
H_STEPS = H // GRID; STEP = GRID * 60
HS = H * 60; TOL = max(10, HS // 20); GAP = HS + TOL; BREAKEVEN = 0.541; BATCH = 16
CACHE = f"EURUSD_{H}m" if GRID == H else f"EURUSD_g{GRID}_h{H}"
TOK_ID = "NeoQuasar/Kronos-Tokenizer-base"
ROOT = "/media/sean/CORSAIR/binary-algo"; OUT = f"{ROOT}/ohlc_cache"
import torch; torch.set_num_threads(16)
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
STRICT_CTX = SESSION != "all"
T0 = time.time()
def hb(m): print(f"HB[{time.time()-T0:.0f}s] {m}", flush=True)


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

def session_of(ts):
    return {s: session_mask(ts, s) for s in ("ny", "ldn", "asia")}

def cpcv(ts, win, ng=8, k=2):
    n = len(win); e = np.linspace(0, n, ng + 1).astype(int); grp = [(e[i], e[i+1]) for i in range(ng)]; pa = []
    for tg in combinations(range(ng), k):
        ii = np.concatenate([np.arange(grp[g][0], grp[g][1]) for g in tg])
        if len(ii) >= 25: pa.append(float(win[ii].mean()))
    pa = np.array(pa)
    return (round(float(np.percentile(pa, 10)), 4), round(float((pa >= BREAKEVEN).mean()), 3)) if len(pa) else (None, None)

def per_year(ts, win):
    yr = pd.to_datetime(ts, unit="s", utc=True).year.values; out = {}
    for Y in (2024, 2025, 2026):
        m = yr == Y
        if m.sum() < 20: continue
        lo, hi = boot(win[m]); out[str(Y)] = {"n": int(m.sum()), "acc": round(float(win[m].mean()), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    return out


def main():
    hb(f"CORRECTED Kronos dir: H={H}m GRID={GRID}m H_steps={H_STEPS} model={MODEL} sess={SESSION} cache={CACHE} dev={DEV}")
    if not os.path.exists(f"{OUT}/{CACHE}_test.parquet"):
        hb(f"MISSING cache {CACHE}_test.parquet — build via kronos_bars.py (label-carrying). ABORT"); return
    tok = KronosTokenizer.from_pretrained(TOK_ID); mdl = Kronos.from_pretrained(MODEL)
    pred = KronosPredictor(mdl, tok, device=DEV, max_context=max(L, 256))
    hb(f"loaded {sum(p.numel() for p in mdl.parameters())/1e6:.1f}M params on {DEV}")

    frames = [pd.read_parquet(f"{OUT}/{CACHE}_{sp}.parquet") for sp in ("test", "oos")]
    B = pd.concat(frames).reset_index(drop=True)
    t = B["t"].values.astype("int64")
    if SESSION != "all":
        m = session_mask(t, SESSION); B = B.loc[m].reset_index(drop=True); t = t[m]
        hb(f"strict session-only INPUT: {SESSION} -> {len(B)} bars")
    O = B["open"].values; Hg = B["high"].values; Lw = B["low"].values; C = B["close"].values; V = B["vol"].values
    y = B["y"].values.astype(int); mag = B["mag"].values; valid = B["valid"].values.astype(bool)
    n = len(B); ar = np.arange(n); yr = pd.to_datetime(t, unit="s", utc=True).year.values

    lo = ar - L + 1; hi = ar + H_STEPS
    base = (lo >= 0) & (hi < n) & valid & (mag > 0)
    pred_ok = np.zeros(n, bool)                                   # predicted bars i+1..i+H_steps gap-free -> land on label exit
    pred_ok[base] = (t[hi[base]] - t[ar[base]]) == H_STEPS * STEP
    ctx_ok = np.ones(n, bool)
    if STRICT_CTX:                                                # context window fully session-internal (no cross-day jump)
        ctx_ok = np.zeros(n, bool); ctx_ok[base] = (t[ar[base]] - t[lo[base]]) == (L - 1) * STEP
    elig = np.where(base & pred_ok & ctx_ok)[0]
    if len(elig) < 100: hb(f"only {len(elig)} eligible bars — too few, ABORT"); return
    upr = y[elig][mag[elig] > 0].mean()
    flag = "" if 0.47 <= upr <= 0.53 else "  <-- UP-RATE OUT OF BAND (settlement bug?)"
    hb(f"eligible {len(elig)} (base{int(base.sum())} pred_ok{int(pred_ok.sum())} ctx_ok{int(ctx_ok.sum())}) up-rate={upr:.4f}{flag}")

    pick = []
    for Y in (2024, 2025, 2026):
        ey = elig[yr[elig] == Y]
        if len(ey) > N_PER_YR: ey = ey[np.linspace(0, len(ey) - 1, N_PER_YR).astype(int)]
        pick.append(ey)
    pick = nonoverlap_chrono(t, np.sort(np.concatenate(pick)))
    hb(f"eval decision bars: {len(pick)} (nonoverlap gap={GAP}s)")

    cols = ["open", "high", "low", "close", "volume", "amount"]; Pup = np.full(len(pick), np.nan)
    for s in range(0, len(pick), BATCH):
        chunk = pick[s:s+BATCH]; dfl, xtl, ytl = [], [], []
        for i in chunk:
            sl = slice(i - L + 1, i + 1)                          # FIX: context INCLUDES decision bar i (close = entry ref)
            dfl.append(pd.DataFrame({"open": O[sl], "high": Hg[sl], "low": Lw[sl], "close": C[sl],
                                     "volume": V[sl], "amount": 0.0})[cols])
            xtl.append(pd.Series(pd.to_datetime(t[sl], unit="s", utc=True)))
            ytl.append(pd.Series(pd.to_datetime(t[i+1:i+1+H_STEPS], unit="s", utc=True)))   # FIX: forward bars
        outs = pred.predict_batch(df_list=dfl, x_timestamp_list=xtl, y_timestamp_list=ytl,
                                  pred_len=H_STEPS, T=1.0, top_p=0.9, sample_count=K, verbose=False)
        for j, i in enumerate(chunk):
            Pup[s+j] = 1.0 if float(outs[j]["close"].iloc[-1]) > C[i] else 0.0             # FIX: pred(+H) vs C[i]
        if s % (BATCH * 25) == 0: hb(f"  predicted {s+len(chunk)}/{len(pick)}")
    hb("prediction done")

    call = (Pup > 0.5).astype(int); tw = t[pick]; yw = y[pick]
    order = np.argsort(tw); tw, yw, call = tw[order], yw[order], call[order]
    win = (call == yw).astype(float)
    np.savez(f"{ROOT}/kronos_mtf_pred_{TAG}.npz", tw=tw, yw=yw, call=call)   # for multi-TF ensemble combine
    res = {"tag": TAG, "model": MODEL, "H_min": H, "GRID_min": GRID, "H_steps": H_STEPS, "mode": ("native" if GRID == H else "fine"),
           "session": SESSION, "strict_session_input": STRICT_CTX, "L": L, "K": K, "n_eval": int(len(pick)),
           "up_rate": round(float(upr), 4), "breakeven": BREAKEVEN,
           "falsifier": "KILL if no year moved-acc CI95-lo>=0.541 OR CPCV path_p10<0.541",
           "per_year": per_year(tw, win)}
    p10, frac = cpcv(tw, win); res["pooled_acc"] = round(float(win.mean()), 4)
    res["cpcv_path_p10"] = p10; res["cpcv_frac_clear_0.541"] = frac
    res["per_session"] = {}
    for snm, sm in session_of(tw).items():
        if sm.sum() < 20: continue
        c = win[sm]; lo2, hi2 = boot(c)
        res["per_session"][snm] = {"n": int(sm.sum()), "acc": round(float(c.mean()), 4), "ci95": [round(lo2, 4), round(hi2, 4)]}
    any_clear = any(v["ci95"][0] >= BREAKEVEN for v in res["per_year"].values())
    res["VERDICT"] = "SURVIVES" if (any_clear and p10 is not None and p10 >= BREAKEVEN) else "KILLED"
    json.dump(res, open(f"{ROOT}/kronos_dir_mtf_{TAG}_result.json", "w"), indent=1)
    hb(f"DONE pooled={res['pooled_acc']} per-year={ {Y:res['per_year'][Y]['acc'] for Y in res['per_year']} } "
       f"CPCV p10={p10} frac={frac} per-sess={ {s:res['per_session'][s]['acc'] for s in res['per_session']} } -> {res['VERDICT']}")


if __name__ == "__main__":
    main()
