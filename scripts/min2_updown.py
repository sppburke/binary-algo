"""SIDE-SPLIT of the FROZEN 120s book (EURUSD.min2.v1) -> the measured (EURUSD, 2m, UP) and (2m, DOWN) keys.
Sweep ledger EURUSD_2m row 0. Inference-only (no training): replicate min2_production.backtest's gated
selective book, take the independent (non-overlap-chrono) trades, split by PREDICTED direction, score per
year (2024/2025/2026) moved-bars-only with bootstrap CI95. Deriv-faithful (wc_ret, ties LOSE). Mirrors the
60s min1_updown analysis. Breakeven 0.541."""
import json, numpy as np, pandas as pd
import min2_production as M2


def year_of(ts):
    return pd.to_datetime(np.asarray(ts), unit="s", utc=True).year.values


def collect():
    """Gated selective independent trades from the frozen book, pooled over test(2024-25)+oos(2026)."""
    p, L, G, C, S = M2._load()
    TS, PRED, Y, MAG = [], [], [], []
    for sp in ("test", "oos"):
        b = M2.load_split(sp); X, y, mag, valid, ts, idx = M2.prep(b)
        pr = M2._blend(p, L, G, C, S, X); pred = (pr > 0.5).astype(int)
        bbw = X["bbw1800"].values; rel = X["rel_ratio"].values; r300 = X["ret300"].values
        gate = valid & (bbw <= p["bbw1800_q67"]) & (rel >= p["rel_p70"]) & (np.sign(pr - 0.5) == -np.sign(r300))
        conf = np.abs(pr - 0.5); cand = gate & (conf >= p["conf_thr"])
        tr = M2.nonoverlap_chrono(ts, cand)
        TS.append(ts[tr]); PRED.append(pred[tr]); Y.append(y[tr]); MAG.append(mag[tr])
        del b, X;
    return (np.concatenate(TS), np.concatenate(PRED), np.concatenate(Y), np.concatenate(MAG))


def side_eval(ts, pred, y, mag):
    yrs = year_of(ts); out = {}
    for label in ("2024", "2025", "2026"):
        ym = yrs == int(label)
        for side, name in ((1, "UP"), (0, "DOWN"), (None, "COMBINED")):
            sel = ym & (pred == side) if side is not None else ym
            moved = mag[sel] > 0
            if moved.sum() < 5:
                out[f"{label}_{name}"] = {"n": int(sel.sum()), "n_moved": int(moved.sum()), "acc": None,
                                          "ci": [None, None], "out_up_rate": None}
                continue
            corr = (pred[sel][moved] == y[sel][moved]).astype(float)
            lo, hi = M2.boot(corr)
            out[f"{label}_{name}"] = {"n": int(sel.sum()), "n_moved": int(moved.sum()),
                                      "acc": float(corr.mean()), "ci": [lo, hi],
                                      "out_up_rate": float(y[sel][moved].mean())}
    return out


def main():
    ts, pred, y, mag = collect()
    upr = float(y[mag > 0].mean())
    print(f"[min2_updown] independent trades={len(ts)}  moved up-rate(all)={upr:.4f}  pred-up share={pred.mean():.3f}", flush=True)
    res = side_eval(ts, pred, y, mag)
    for k, v in res.items():
        print(f"  {k:16} n={v['n']:>4} moved={v['n_moved']:>4} acc={v['acc']} CI={v['ci']} out_up_rate={v['out_up_rate']}", flush=True)
    json.dump({"book": "EURUSD.min2.v1", "horizon_s": 120, "breakeven": 0.541,
               "settlement": "deriv-faithful wc_ret ties-LOSE; nonoverlap_chrono; moved-bars-only; per-year CI95",
               "moved_up_rate_all": upr, "per_year_side": res}, open("min2_updown_result.json", "w"), indent=1)
    print("[min2_updown] -> min2_updown_result.json", flush=True)


if __name__ == "__main__":
    main()
