"""CROSS-HORIZON STACK for 2m (sweep row A5a): front-load the 15m parent (EURUSD.m15.v1) direction onto the
frozen 2m book's (EURUSD.min2.v1) trades. Hypothesis: 2m bets are better when the 15m parent agrees on
direction. Parameter-free AGREEMENT filter (no fitting -> no VAL-max risk). Causal alignment: each 2m trade
uses the latest completed minute-bar parent prediction (merge_asof backward). Deriv-faithful, moved-bars,
per-year CI95.

PRE-REGISTERED FALSIFIER: KILL unless the AGREE filter lifts (2m,UP) OOS(2026) above the 0.555 baseline AND
UP clears breakeven 0.541 in all 3 years AND n_moved(2026 UP) >= 50. (Same bar for DOWN, separately.)"""
import json, numpy as np, pandas as pd
import min2_production as M2, m15_production as M15

BASE_UP_2026 = 0.555  # incumbent (min2_updown_result.json)


def year_of(ts):
    return pd.to_datetime(np.asarray(ts), unit="s", utc=True).year.values


def parent_p15():
    p, L, G, C = M15._load()
    parts = []
    for y in ("2024", "2025", "2026"):
        D = M15.load([y])
        if len(D) == 0:
            continue
        pr = M15._dirproba(p, L, G, C, D)
        sec = D.index.values.astype("datetime64[s]").astype("int64")
        parts.append(pd.DataFrame({"sec": sec, "p15": pr}))
        del D
    return pd.concat(parts).sort_values("sec").drop_duplicates("sec").reset_index(drop=True)


def collect2():
    p, L, G, C, S = M2._load()
    rows = []
    for sp in ("test", "oos"):
        b = M2.load_split(sp); X, y, mag, valid, ts, idx = M2.prep(b)
        pr = M2._blend(p, L, G, C, S, X); pred = (pr > 0.5).astype(int)
        bbw = X["bbw1800"].values; rel = X["rel_ratio"].values; r300 = X["ret300"].values
        gate = valid & (bbw <= p["bbw1800_q67"]) & (rel >= p["rel_p70"]) & (np.sign(pr - 0.5) == -np.sign(r300))
        conf = np.abs(pr - 0.5); cand = gate & (conf >= p["conf_thr"])
        tr = M2.nonoverlap_chrono(ts, cand)
        rows.append(pd.DataFrame({"sec": ts[tr], "p2": pr[tr], "pred": pred[tr], "y": y[tr], "mag": mag[tr]}))
        del b, X
    return pd.concat(rows).sort_values("sec").reset_index(drop=True)


def evalsplit(df):
    yrs = year_of(df["sec"].values); out = {}
    for label in ("2024", "2025", "2026"):
        ym = yrs == int(label)
        for side, name in ((1, "UP"), (0, "DOWN"), (None, "COMB")):
            sel = ym & (df["pred"].values == side) if side is not None else ym
            d = df[sel]; moved = d["mag"].values > 0
            if moved.sum() < 5:
                out[f"{label}_{name}"] = {"n": int(sel.sum()), "n_moved": int(moved.sum()), "acc": None, "ci": [None, None]}
                continue
            corr = (d["pred"].values[moved] == d["y"].values[moved]).astype(float)
            lo, hi = M2.boot(corr)
            out[f"{label}_{name}"] = {"n": int(sel.sum()), "n_moved": int(moved.sum()), "acc": float(corr.mean()), "ci": [lo, hi]}
    return out


def main():
    pa = parent_p15(); print(f"[stack] parent P15 minutes={len(pa):,}", flush=True)
    df = collect2(); print(f"[stack] 2m gated trades={len(df):,}", flush=True)
    m = pd.merge_asof(df, pa, on="sec", direction="backward").dropna(subset=["p15"]).reset_index(drop=True)
    print(f"[stack] trades aligned to parent={len(m):,}", flush=True)
    base = evalsplit(m)
    agree = m[np.sign(m["p2"].values - 0.5) == np.sign(m["p15"].values - 0.5)].reset_index(drop=True)
    ev = evalsplit(agree)
    up26 = ev.get("2026_UP", {}).get("acc")
    up_all = [ev.get(f"{Y}_UP", {}).get("acc") for Y in ("2024", "2025", "2026")]
    killed = not (up26 and up26 > BASE_UP_2026 and all(a and a >= 0.541 for a in up_all)
                  and ev.get("2026_UP", {}).get("n_moved", 0) >= 50)
    out = {"row": "A5a cross-horizon stack 15m->2m", "parent": "EURUSD.m15.v1", "child_book": "EURUSD.min2.v1",
           "horizon_s": 120, "breakeven": 0.541, "n_aligned": int(len(m)), "n_agree": int(len(agree)),
           "baseline_no_stack": base, "V1_agree": ev,
           "falsifier": {"BASE_UP_2026": BASE_UP_2026, "agree_UP_2026": up26, "agree_UP_allyears": up_all,
                         "KILLED": bool(killed),
                         "verdict": ("KILLED: AGREE-stack UP did not beat 0.555 OOS while clearing breakeven all 3 yrs (n>=50)"
                                     if killed else "SURVIVED: AGREE-stack lifts 2m UP above baseline, robust")}}
    json.dump(out, open("min2_stack_result.json", "w"), indent=1)
    for tag, r in (("BASE(no-stack)", base), ("AGREE(15m∧2m)", ev)):
        print(f"--- {tag} ---", flush=True)
        for label in ("2024", "2025", "2026"):
            u, dn, cb = r[f"{label}_UP"], r[f"{label}_DOWN"], r[f"{label}_COMB"]
            print(f"  {label}: UP {u['acc']} (n{u.get('n_moved')}) | DOWN {dn['acc']} (n{dn.get('n_moved')}) | COMB {cb['acc']}", flush=True)
    print(f"[stack] FALSIFIER: {out['falsifier']['verdict']}", flush=True)
    print("[stack] -> min2_stack_result.json", flush=True)


if __name__ == "__main__":
    main()
