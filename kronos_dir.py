"""KRONOS for EURUSD 1m DIRECTION — zero-shot or fine-tuned, deriv-faithful + CPCV.

Tests the user's hypothesis: Kronos predicts the next-bar PRICE, and sign(pred_close - entry_close) is a
direction call. We score that against the deriv-faithful 60s wc_ret label (ties-strict, moved bars), per year
with CI95 + nonoverlap + purged-combinatorial CPCV, under a pre-registered falsifier.

For each eligible decision bar i (valid & moved, from ohlc_cache/EURUSD_1m_{test,oos}): take the prior L bars as
context, predict pred_len=1, sample_count=K trajectories; P_up = mean_k(sampled close_k > entry close_i).
Direction call = P_up>0.5. Eval is SUBSAMPLED per year (CPU budget) but nonoverlap + per-year CI keep it honest.

Run: ~/binary-algo-venv/bin/python kronos_dir.py <tok_id> <model_id_or_path> <tag> [L=256] [K=20] [N_PER_YR=2500]
  e.g. zero-shot:  kronos_dir.py NeoQuasar/Kronos-Tokenizer-base NeoQuasar/Kronos-small zeroshot_small
"""
import sys, os, json, time, numpy as np, pandas as pd
from itertools import combinations
sys.path.insert(0, "/home/sean/git/Kronos")
from model import Kronos, KronosTokenizer, KronosPredictor

ROOT = "/media/sean/CORSAIR/binary-algo"; OUT = f"{ROOT}/ohlc_cache"
TOK_ID  = sys.argv[1] if len(sys.argv) > 1 else "NeoQuasar/Kronos-Tokenizer-base"
MODEL   = sys.argv[2] if len(sys.argv) > 2 else "NeoQuasar/Kronos-small"
TAG     = sys.argv[3] if len(sys.argv) > 3 else "zeroshot_small"
L       = int(sys.argv[4]) if len(sys.argv) > 4 else 256          # lookback context (bars)
K       = int(sys.argv[5]) if len(sys.argv) > 5 else 20           # sample_count trajectories
N_PER_YR= int(sys.argv[6]) if len(sys.argv) > 6 else 2500         # eval windows per year (CPU budget)
SESSION = sys.argv[7] if len(sys.argv) > 7 else "all"             # all|ny|london|overlap|asia (eval filter)
HS, TOL, GAP = 60, 10, 70; BREAKEVEN = 0.541; BATCH = 16
import torch; torch.set_num_threads(16)
T0 = time.time()
def hb(m): print(f"HB[{time.time()-T0:.0f}s] {m}", flush=True)


def nonoverlap_chrono(ts, idx, gap=GAP):
    take = []; bu = -1
    for i in idx:
        if ts[i] < bu: continue
        take.append(i); bu = int(ts[i]) + gap
    return np.array(take, dtype=int)

def boot(corr, nb=4000, seed=7):
    corr = np.asarray(corr, float)
    if len(corr) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(corr)
    a = np.array([corr[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def session_of(ts):
    """DST-CORRECT sessions via the IANA tz database (handles EST/EDT + US-vs-EU DST misalignment), defined in
    EXCHANGE-LOCAL time — NOT fixed UTC. NY = 8am-5pm America/New_York; London = 8am-4pm Europe/London;
    overlap = NY & London; asia = neither. (The repo's pipeline.py uses a fixed 12-21 UTC approximation.)"""
    idx = pd.to_datetime(np.asarray(ts), unit="s", utc=True)
    ny_h = idx.tz_convert("America/New_York").hour.values
    lon_h = idx.tz_convert("Europe/London").hour.values
    ny = (ny_h >= 8) & (ny_h < 17); london = (lon_h >= 8) & (lon_h < 16)
    return {"ny": ny, "london": london, "overlap": ny & london, "asia": ~(ny | london)}

def cpcv(ts, win, n_groups=8, k=2):
    n = len(win); edges = np.linspace(0, n, n_groups + 1).astype(int)
    grp = [(edges[i], edges[i+1]) for i in range(n_groups)]; pa = []
    for tg in combinations(range(n_groups), k):
        ii = np.concatenate([np.arange(grp[g][0], grp[g][1]) for g in tg])
        if len(ii) >= 25: pa.append(float(win[ii].mean()))
    pa = np.array(pa)
    return (round(float(np.percentile(pa, 10)), 4), round(float((pa >= BREAKEVEN).mean()), 3)) if len(pa) else (None, None)


def main():
    hb(f"tok={TOK_ID} model={MODEL} tag={TAG} L={L} K={K} N/yr={N_PER_YR}")
    tok = KronosTokenizer.from_pretrained(TOK_ID); mdl = Kronos.from_pretrained(MODEL)
    pred = KronosPredictor(mdl, tok, device="cpu", max_context=max(L, 256))
    hb(f"loaded {sum(p.numel() for p in mdl.parameters())/1e6:.1f}M params on cpu")

    # gather eligible decision bars (valid & moved) across test+oos, with full-bar context arrays
    frames = []
    for sp in ("test", "oos"):
        b = pd.read_parquet(f"{OUT}/EURUSD_1m_{sp}.parquet"); b["sp"] = sp; frames.append(b)
    B = pd.concat(frames).reset_index(drop=True)
    O = B["open"].values; Hg = B["high"].values; Lw = B["low"].values; C = B["close"].values
    V = B["vol"].values; t = B["t"].values.astype("int64"); y = B["y"].values.astype(int)
    mag = B["mag"].values; valid = B["valid"].values.astype(bool)
    yr = pd.to_datetime(t, unit="s", utc=True).year.values
    elig = np.where(valid & (mag > 0) & (np.arange(len(B)) >= L))[0]
    if SESSION != "all":
        smask = session_of(t)[SESSION]; elig = elig[smask[elig]]
        hb(f"SESSION filter='{SESSION}' -> {len(elig)} eligible bars")
    # subsample per year (evenly spaced -> chronological coverage), then nonoverlap
    rng = np.random.default_rng(7); pick = []
    for Y in (2024, 2025, 2026):
        ey = elig[yr[elig] == Y]
        if len(ey) > N_PER_YR: ey = ey[np.linspace(0, len(ey) - 1, N_PER_YR).astype(int)]
        pick.append(ey)
    pick = np.sort(np.concatenate(pick))
    pick = nonoverlap_chrono(t, pick)
    hb(f"eval decision bars: {len(pick)} (nonoverlap, ~{N_PER_YR}/yr pre-overlap)")

    # batched prediction
    Pup = np.full(len(pick), np.nan)
    cols = ["open", "high", "low", "close", "volume", "amount"]
    for s in range(0, len(pick), BATCH):
        chunk = pick[s:s+BATCH]
        dfl, xtl, ytl = [], [], []
        for i in chunk:
            sl = slice(i - L, i)                       # L bars ending at i-1 (context strictly before decision close i)
            df = pd.DataFrame({"open": O[sl], "high": Hg[sl], "low": Lw[sl], "close": C[sl],
                               "volume": V[sl], "amount": 0.0})
            dfl.append(df[cols])
            xtl.append(pd.Series(pd.to_datetime(t[sl], unit="s", utc=True)))
            ytl.append(pd.Series(pd.to_datetime(t[i:i+1], unit="s", utc=True)))
        outs = pred.predict_batch(df_list=dfl, x_timestamp_list=xtl, y_timestamp_list=ytl,
                                  pred_len=1, T=1.0, top_p=0.9, sample_count=K, verbose=False)
        for j, i in enumerate(chunk):
            # predict_batch returns mean-over-samples df; recompute P_up needs per-sample -> use mean close vs entry
            pc = float(outs[j]["close"].iloc[0]); Pup[s+j] = 1.0 if pc > C[i-1] else 0.0
        if s % (BATCH*20) == 0: hb(f"  predicted {s+len(chunk)}/{len(pick)}")
    hb("prediction done")

    # NOTE: predict_batch returns the sample-MEAN path, so P_up is a hard 0/1 (mean close vs entry). For a soft
    # probability we'd need generate() per-sample; the hard call is sufficient for a direction accuracy test.
    call = (Pup > 0.5).astype(int)
    res = {"tag": TAG, "model": MODEL, "tokenizer": TOK_ID, "L": L, "K": K, "n_eval": int(len(pick)),
           "breakeven": BREAKEVEN, "per_year": {}, "falsifier": "KILL if no year moved-acc CI95-lo >= 0.541 OR CPCV path_p10 < 0.541"}
    tw = t[pick]; yw = y[pick]
    order = np.argsort(tw); tw, yw, call_o = tw[order], yw[order], call[order]
    yrw = pd.to_datetime(tw, unit="s", utc=True).year.values
    any_clear = False
    for Y in (2024, 2025, 2026):
        m = yrw == Y
        if m.sum() < 20: continue
        corr = (call_o[m] == yw[m]).astype(float); lo, hi = boot(corr)
        res["per_year"][str(Y)] = {"n": int(m.sum()), "acc": round(float(corr.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)]}
        if lo >= BREAKEVEN: any_clear = True
    win = (call_o == yw).astype(float); p10, frac = cpcv(tw, win)
    res["pooled_acc"] = round(float(win.mean()), 4); res["cpcv_path_p10"] = p10; res["cpcv_frac_clear_0.541"] = frac
    # per-session breakdown (DST-correct) on the eval set — the user's question: does direction work in a session?
    res["per_session"] = {}
    for snm, sm in session_of(tw).items():
        if sm.sum() < 20: continue
        c = (call_o[sm] == yw[sm]).astype(float); lo, hi = boot(c)
        res["per_session"][snm] = {"n": int(sm.sum()), "acc": round(float(c.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    res["VERDICT"] = "SURVIVES" if (any_clear and p10 is not None and p10 >= BREAKEVEN) else "KILLED"
    json.dump(res, open(f"{ROOT}/kronos_dir_{TAG}_result.json", "w"), indent=1)
    hb(f"DONE pooled_acc={res['pooled_acc']} per-year={ {Y:res['per_year'][Y]['acc'] for Y in res['per_year']} } "
       f"CPCV p10={p10} frac_clear={frac} -> {res['VERDICT']}")
    print(f"-> kronos_dir_{TAG}_result.json", flush=True)


if __name__ == "__main__":
    main()
