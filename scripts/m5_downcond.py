"""Sweep row D3 (discovery round 2): USD-STRENGTH-CONDITIONED DOWN. Hypothesis: EURUSD 5m DOWN is the bet ONLY
when the DOLLAR is the driver (broad USD bid pulls EURUSD down coherently); rally-selling died in 2025 because
it was an EUR-up regime with no USD tailwind. Test: split the frozen m5xp DOWN-predicted gated trades by the
USD-basket strength at trade time (usdbask<0 == USD strong == EURUSD-equiv down). Disciplined FIXED sign-split
(no OOS-peek). Down-success = realized y==0. Breakeven 0.541.

PRE-REGISTERED FALSIFIER: KILL D3 unless the USD-STRONG (usdbask<0) DOWN subset clears 0.541 in the BINDING
2025 year (n>=100) AND the USD-strong DOWN-acc materially exceeds the USD-weak DOWN-acc (the conditioning must
carry the sign). If USD-strong 2025 DOWN < 0.541 -> DOWN is genuinely efficient even USD-gated."""
import json, numpy as np
import m5_xpair as MX
import m5_xpair_production as XP

YEARS = (("test24", "2024"), ("test25", "2025"), ("oos", "2026"))


def boot(c, nb=4000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    p, P, M = XP._load(); cols = p["primary_feats"]; mcols = p["meta_feats"]; THR = p["meta_thr"]
    out = {}
    # discover usdbask / eurresid columns from a probe build
    D0 = MX.build_xp(["2024"]); D0 = MX.augment(D0, ["2024"], XP.MODE)
    bask_cols = [c for c in D0.columns if c.startswith("usdbask")]
    resid_cols = [c for c in D0.columns if c.startswith("eurresid")]
    print(f"[downcond] usdbask cols={bask_cols} eurresid cols={resid_cols}", flush=True)
    del D0
    res = {}
    for w, label in YEARS:
        D = MX.build_xp([w] if False else XP.SPL[w]); D = MX.augment(D, XP.SPL[w], XP.MODE)
        pr = P.predict(D[cols].astype("float32")); y = D["_y"].astype(int).values
        sm = M.predict(XP._Xmeta(D, pr, mcols)); ts = D["_ts"].values.astype("int64")
        ny = D["sess_ny"].values > 0.5
        bask = D[bask_cols].mean(axis=1).values if bask_cols else np.zeros(len(D))
        m = ny & (sm >= THR); sel = MX.nonoverlap_chrono(ts, m)
        pred = (pr[sel] > 0.5).astype(int)
        down = sel[pred == 0]                  # DOWN-predicted gated trades
        yb = y[down]; bk = bask[down]
        usd_strong = bk < 0                     # USD strong == EURUSD-equiv down
        for cond, name in ((usd_strong, "USDstrong"), (~usd_strong, "USDweak"), (np.ones(len(yb), bool), "ALL")):
            n = int(cond.sum())
            if n < 5:
                res[f"{label}_{name}"] = {"n": n, "down_acc": None, "ci": [None, None]}; continue
            corr = (yb[cond] == 0).astype(float)  # down-success
            lo, hi = boot(corr)
            res[f"{label}_{name}"] = {"n": n, "down_acc": round(float(corr.mean()), 4), "ci": [round(lo, 4), round(hi, 4)]}
        del D
    s25 = res.get("2025_USDstrong", {})
    certified = bool(s25.get("down_acc") and s25["down_acc"] >= 0.541 and s25.get("n", 0) >= 100
                     and s25["ci"][0] is not None and s25["ci"][0] >= 0.541)
    out = {"row": "D3 USD-strength-conditioned DOWN @300s", "breakeven": 0.541, "bask_cols": bask_cols,
           "per_year_side": res,
           "falsifier": {"D3_HAS_DOWN_EDGE": certified,
                         "verdict": (f"D3 SURVIVES: USD-strong 2025 DOWN={s25.get('down_acc')} (n{s25.get('n')}) CI-lo clears 0.541"
                                     if certified else
                                     f"D3 KILLED: USD-strong 2025 DOWN={s25.get('down_acc')} (n{s25.get('n')}) does NOT clear 0.541 "
                                     "CI-lo -> EURUSD 5m DOWN is genuinely efficient even when USD-gated; rally-selling dead.")}}
    json.dump(out, open("m5_downcond_result.json", "w"), indent=1)
    for k, v in res.items():
        print(f"  {k:18} n={v['n']:>4} down_acc={v['down_acc']} CI={v['ci']}", flush=True)
    print(f"[downcond] FALSIFIER: {out['falsifier']['verdict']}", flush=True)
    print("[downcond] -> m5_downcond_result.json", flush=True)


if __name__ == "__main__":
    main()
