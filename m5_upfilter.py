"""Sweep rows A8a (up-only FILTER) + A8b (down-only FILTER): can a (5m,SIDE)-SPECIFIC meta threshold beat the
frozen combined-tuned m5xp gate? Inference-only on the frozen m5xp book. For each side, SELECT the meta
threshold that maximizes VAL worst-half side-accuracy (NY, side-predicted bars), NEVER OOS-peeking; then report
OOS per-year 2024/25/26 with bootstrap CI95 and the moved up-rate tripwire. Incumbent to beat: (5m,UP) frozen
binding 2025 = 0.577. Breakeven 0.541.

Falsifier: KILL the refinement if the VAL-selected threshold's worst held-out year (24/25/26) UP-acc CI95-lower
does not clear the frozen incumbent 0.577 (UP) / 0.541 (DOWN), OR up-rate of the gated bars leaves [0.47,0.53]."""
import json, numpy as np
import m5_xpair as MX
import m5_xpair_production as XP

WINS = {"val": ["2022", "2023"], "test24": ["2024"], "test25": ["2025"], "oos": ["2026"]}


def build_all():
    p, P, M = XP._load(); cols = p["primary_feats"]; mcols = p["meta_feats"]
    out = {}
    for w, yrs in WINS.items():
        D = MX.build_xp(yrs); D = MX.augment(D, yrs, XP.MODE)
        pr = P.predict(D[cols].astype("float32"))
        sm = M.predict(XP._Xmeta(D, pr, mcols))
        out[w] = {"pr": pr, "y": D["_y"].astype(int).values, "sm": sm,
                  "ts": D["_ts"].values.astype("int64"), "ny": D["sess_ny"].values > 0.5,
                  "yr": (np.asarray(D["_ts"].values, dtype="datetime64[s]").astype("datetime64[Y]").astype(int) + 1970)}
        del D
    return out


def side_acc(d, thr, side, year=None):
    """Independent NY trades with meta>=thr and pred==side; (n, acc, up_rate, corr-array)."""
    pred = (d["pr"] > 0.5).astype(int)
    m = d["ny"] & (d["sm"] >= thr) & (pred == side)
    if year is not None:
        m = m & (d["yr"] == year)
    sel = MX.nonoverlap_chrono(d["ts"], m)
    if len(sel) == 0:
        return 0, float("nan"), float("nan"), np.array([])
    corr = (pred[sel] == d["y"][sel]).astype(float)
    return len(sel), float(corr.mean()), float(d["y"][sel].mean()), corr


def main():
    A = build_all()
    val = A["val"]
    grid = [float(np.quantile(val["sm"][val["ny"]], q)) for q in (0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98)]
    result = {}
    for side, name in ((1, "UP"), (0, "DOWN")):
        # SELECT thr by worst-VAL-half (min over 2022/2023 of side-acc), require min n per VAL year
        best = None
        for thr in grid:
            accs, ns = [], []
            for vy in (2022, 2023):
                n, acc, _, _ = side_acc(val, thr, side, vy)
                ns.append(n); accs.append(acc)
            if min(ns) < 40 or any(np.isnan(accs)):
                continue
            wh = min(accs)
            if best is None or wh > best["val_worsthalf"]:
                best = {"thr": thr, "val_worsthalf": wh, "val_ns": ns}
        if best is None:
            result[name] = {"selected": None, "note": "no threshold met VAL min-n"}
            continue
        thr = best["thr"]; per_year = {}
        for Y in (2024, 2025, 2026):
            w = {2024: "test24", 2025: "test25", 2026: "oos"}[Y]
            n, acc, upr, corr = side_acc(A[w], thr, side, Y)
            lo, hi = MX.boot(corr) if len(corr) >= 5 else (float("nan"), float("nan"))
            per_year[str(Y)] = {"n": n, "acc": round(acc, 4) if n else None,
                                "ci": [round(lo, 4), round(hi, 4)], "out_up_rate": round(upr, 4) if n else None}
        floor = min(v["acc"] for v in per_year.values() if v["acc"] is not None)
        result[name] = {"selected_thr": round(thr, 5), "val_worsthalf": round(best["val_worsthalf"], 4),
                        "val_ns": best["val_ns"], "per_year": per_year, "worst_year_acc": round(floor, 4)}
    incU = 0.577
    out = {"test": "m5xp side-specific meta-threshold refinement (A8a UP / A8b DOWN)", "breakeven": 0.541,
           "incumbent_up_binding": incU, "frozen_gate_thr": 0.57384, "result": result}
    json.dump(out, open("m5_upfilter_result.json", "w"), indent=1)
    for name in ("UP", "DOWN"):
        r = result.get(name, {})
        print(f"=== {name} === selected_thr={r.get('selected_thr')} val_worsthalf={r.get('val_worsthalf')} worst_year_acc={r.get('worst_year_acc')}", flush=True)
        for Y, v in r.get("per_year", {}).items():
            print(f"   {Y}: n={v['n']} acc={v['acc']} CI={v['ci']} up_rate={v['out_up_rate']}", flush=True)
    print("[m5_upfilter] -> m5_upfilter_result.json", flush=True)


if __name__ == "__main__":
    main()
