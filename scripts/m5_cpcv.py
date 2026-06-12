"""Certification test of the m5xp (5m,UP) side-split. Purged-combinatorial path distribution + block bootstrap
on the FROZEN m5xp book's UP-predicted gated independent trades. Mirrors min2_cpcv.py (the 2m precedent that
FAILED at p10 0.524). Settlement = book-native (wall-clock-contiguous 300s close-to-close, _y already moved-only
so ties dropped at build). Tests whether the 5m UP side clears breakeven 0.541 across held-out combinatorial
time-blocks and under serial-dependence-aware resampling, not just the lucky pooled split.

CERTIFY iff path_p10 >= 0.541 AND block-boot CI95-lower >= 0.541.
Usage: python m5_cpcv.py [m5xp|m5stack]"""
import sys, json, numpy as np, pandas as pd
from itertools import combinations
import m5_xpair as MX

N_GROUPS, K_TEST = 8, 2   # C(8,2)=28 purged-combinatorial paths
YEARS = (("test24", "2024"), ("test25", "2025"), ("oos", "2026"))


def year_of(ts):
    return pd.to_datetime(np.asarray(ts), unit="s", utc=True).year.values


def collect_up_m5xp():
    """Frozen m5xp UP-predicted gated independent trades, pooled test+oos, time-ordered. Returns (ts, win)."""
    import os, m5_xpair_production as XP
    p, P, M = XP._load(); cols = p["primary_feats"]; mcols = p["meta_feats"]
    THR = float(os.environ.get("M5XP_THR", p["meta_thr"]))
    TS, WIN = [], []
    for w, _ in YEARS:
        D = MX.build_xp(XP.SPL[w]); D = MX.augment(D, XP.SPL[w], XP.MODE)
        pr = P.predict(D[cols].astype("float32")); y = D["_y"].astype(int).values
        sm = M.predict(XP._Xmeta(D, pr, mcols)); ts = D["_ts"].values.astype("int64")
        ny = D["sess_ny"].values > 0.5
        m = ny & (sm >= THR); sel = MX.nonoverlap_chrono(ts, m)
        pred = (pr[sel] > 0.5).astype(int); up = sel[pred == 1]
        TS.append(ts[up]); WIN.append((y[up] == 1).astype(float))
        del D
    ts = np.concatenate(TS); win = np.concatenate(WIN)
    o = np.argsort(ts); return ts[o], win[o]


def collect_up_m5stack():
    import lightgbm as lgb, json as _j, m5_stack2 as S2
    MODELS = "/home/sean/git/binary-algo/models"
    sp = _j.load(open(f"{MODELS}/m5stack_EURUSD_strategy.json")); THR = sp["meta_thr"]
    Mm = lgb.Booster(model_file=f"{MODELS}/m5stack_EURUSD_meta_lgb.txt")
    _p, L15, G15, C15 = S2.load15(); s5, P5 = S2.load5()
    TS, WIN = [], []
    for w, _ in YEARS:
        W = S2.features(w, L15, G15, C15, s5, P5)
        sm = Mm.predict(W["Xm"]); m = W["ny"] & (sm >= THR)
        sel = MX.nonoverlap_chrono(W["ts"], m, 300)
        up = sel[W["dir15"][sel] == 1]
        TS.append(W["ts"][up]); WIN.append((W["y"][up] == 1).astype(float))
    ts = np.concatenate(TS); win = np.concatenate(WIN)
    o = np.argsort(ts); return ts[o], win[o]


def block_boot(win, nb=5000, block=20, seed=7):
    rng = np.random.default_rng(seed); n = len(win); nbk = int(np.ceil(n / block))
    accs = []
    for _ in range(nb):
        starts = rng.integers(0, n, nbk)
        idx = (starts[:, None] + np.arange(block)).ravel() % n
        accs.append(win[idx[:n]].mean())
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "m5xp"
    ts, win = {"m5xp": collect_up_m5xp, "m5stack": collect_up_m5stack}[which]()
    pooled = float(win.mean()); n = len(win)
    blo, bhi = block_boot(win)
    edges = np.linspace(0, n, N_GROUPS + 1).astype(int)
    grp = [(edges[i], edges[i + 1]) for i in range(N_GROUPS)]
    path_accs = []
    for testg in combinations(range(N_GROUPS), K_TEST):
        idx = np.concatenate([np.arange(grp[g][0], grp[g][1]) for g in testg])
        if len(idx) >= 30:
            path_accs.append(float(win[idx].mean()))
    path_accs = np.array(path_accs)
    p10 = float(np.percentile(path_accs, 10)); pmean = float(path_accs.mean())
    frac_clear = float((path_accs >= 0.541).mean())
    yr = year_of(ts); per_year = {str(Y): round(float(win[yr == Y].mean()), 4) for Y in (2024, 2025, 2026) if (yr == Y).sum() > 30}
    certified = (p10 >= 0.541) and (blo >= 0.541)
    out = {"test": f"{which} (5m,UP) purged-combinatorial + block-bootstrap, book-native settlement", "breakeven": 0.541,
           "n_up_trades": n, "pooled_acc": round(pooled, 4), "block_boot_ci95": [round(blo, 4), round(bhi, 4)],
           "cpcv_paths": int(len(path_accs)), "path_mean": round(pmean, 4), "path_p10": round(p10, 4),
           "path_min": round(float(path_accs.min()), 4), "frac_paths_clear_0.541": round(frac_clear, 3),
           "per_year": per_year,
           "verdict": {"CERTIFIED": bool(certified),
                       "statement": (f"CERTIFIED: {which} 5m-UP path_p10 {p10:.4f} and block-boot CI-lower {blo:.4f} both clear 0.541"
                                     if certified else
                                     f"NOT CERTIFIED: {which} 5m-UP path_p10 {p10:.4f} and/or block-boot CI95-lower {blo:.4f} below 0.541")}}
    json.dump(out, open(f"m5_cpcv_{which}_result.json", "w"), indent=1)
    print(f"[cpcv-{which}] n={n} pooled={pooled:.4f} block-boot CI95=[{blo:.4f},{bhi:.4f}]", flush=True)
    print(f"[cpcv-{which}] {len(path_accs)} paths: mean={pmean:.4f} p10={p10:.4f} min={path_accs.min():.4f} frac>=0.541={frac_clear:.2f}", flush=True)
    print(f"[cpcv-{which}] per-year: {per_year}", flush=True)
    print(f"[cpcv-{which}] VERDICT: {out['verdict']['statement']}", flush=True)


if __name__ == "__main__":
    main()
