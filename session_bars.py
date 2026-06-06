"""SESSION campaign — bar-store GBM DIRECTION + MAGNITUDE at horizon H minutes (5/10/15/30), session-only.

MEMORY-LIGHT (precomputed 239-feature store `features/`, no tick featurization spike) -> safe to run CONCURRENTLY
with the Kronos CPU job. Per session (NY/LDN/Asia, DST-correct sessions.py): build a deriv-faithful H-horizon label
from `close` (wall-clock t+H*60s, contiguity-enforced, ties LOSE), restrict train+val+test+OOS decision bars to the
session, train a single LGBM (direction comb/UP/DOWN + magnitude |ret_H|>=session-train-median), per-year CI95 +
nonoverlap + purged-combinatorial CPCV, pre-registered falsifier.

Run: ~/binary-algo-venv/bin/python session_bars.py <H_min>   ->  session_<H>m_{dir,mag}_{ny,ldn,asia}_result.json
"""
import sys, os, json, time, gc, numpy as np, pandas as pd
from itertools import combinations
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as Hn
from sessions import session_mask, SESSIONS

H = int(sys.argv[1]) if len(sys.argv) > 1 else 5
HS = H * 60; TOL = max(60, HS // 20); GAP = HS + TOL; BREAKEVEN = 0.541
PAIR = "EURUSD"; FEAT = Hn.FEAT_DIR; FEATS = list(Hn.feature_cols(PAIR))
NTHREADS = int(os.environ.get("LGB_THREADS", "8"))            # lower default so it co-exists with Kronos threads
SUBCAP = 150_000
T0 = time.time()
def hb(m): print(f"HB[{time.time()-T0:.0f}s] {m}", flush=True)


def mk_lgb(n=2500):
    return lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=255,
        min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=8,
        n_estimators=n, n_jobs=NTHREADS, verbosity=-1)

def contig_fwd(secs, c, hs=HS, tol=TOL):
    n = len(secs); tgt = secs + hs; j = np.searchsorted(secs, tgt, side="left"); jc = np.clip(j, 0, n - 1)
    ok = (j < n) & (np.abs(secs[jc] - tgt) <= tol); ret = c[jc] / c - 1.0
    return ret, ok & np.isfinite(ret)

def load(split):
    parts = []
    for y in Hn.SPLITS[split]:
        p = f"{FEAT}/{PAIR}_{y}.parquet"
        if os.path.exists(p): parts.append(pd.read_parquet(p, columns=FEATS + ["close"]))
    df = pd.concat(parts); df = df[~df.index.duplicated(keep="last")].sort_index()
    secs = df.index.values.astype("datetime64[s]").astype("int64")
    ret, ok = contig_fwd(secs, df["close"].values.astype(float))
    X = df[FEATS].astype("float32"); y = (ret > 0).astype(int); mag = np.abs(ret)
    return X, y, mag, ok, secs

def boot(c, nb=4000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)]); return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def nonoverlap(ts, mask, gap=GAP):
    take = []; bu = -1
    for i in np.where(mask)[0]:
        if ts[i] < bu: continue
        take.append(i); bu = int(ts[i]) + gap
    return np.array(take, dtype=int)

def cpcv_side(ts, win, ng=8, k=2):
    o = np.argsort(ts); ts, win = ts[o], win[o]; n = len(win)
    if n < 60: return {"n": n, "pooled": round(float(win.mean()), 4) if n else None, "path_p10": None, "frac_clear": None}
    e = np.linspace(0, n, ng + 1).astype(int); g = [(e[i], e[i+1]) for i in range(ng)]; pa = []
    for tg in combinations(range(ng), k):
        ii = np.concatenate([np.arange(g[a][0], g[a][1]) for a in tg])
        if len(ii) >= 25: pa.append(float(win[ii].mean()))
    pa = np.array(pa)
    return {"n": n, "pooled": round(float(win.mean()), 4), "path_p10": round(float(np.percentile(pa, 10)), 4),
            "path_mean": round(float(pa.mean()), 4), "frac_clear": round(float((pa >= BREAKEVEN).mean()), 3)}

def per_year(ts, win):
    yr = pd.to_datetime(ts, unit="s", utc=True).year.values; out = {}
    for Y in (2024, 2025, 2026):
        m = yr == Y
        if m.sum() < 20: continue
        lo, hi = boot(win[m]); out[str(Y)] = {"n": int(m.sum()), "acc": round(float(win[m].mean()), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    return out


def main():
    hb(f"H={H}m HS={HS}s load train/val (mem-light bar store), threads={NTHREADS}")
    Xtr, ytr, mtr, vtr, tstr = load("train"); Xva, yva, mva, vva, tsva = load("val")
    models = {}
    for s in SESSIONS:
        smtr = session_mask(tstr, s); smva = session_mask(tsva, s)
        itr = np.where(vtr & (mtr > 0) & smtr)[0]
        if len(itr) > SUBCAP: itr = itr[np.linspace(0, len(itr) - 1, SUBCAP).astype(int)]
        iva = np.where(vva & (mva > 0) & smva)[0]
        Ld = mk_lgb(); Ld.fit(Xtr.iloc[itr], ytr[itr], eval_set=[(Xva.iloc[iva], yva[iva])], eval_metric="auc",
                              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        magthr = float(np.nanpercentile(mtr[itr], 50))
        Lm = mk_lgb(); Lm.fit(Xtr.iloc[itr], (mtr[itr] >= magthr).astype(int),
                              eval_set=[(Xva.iloc[iva], (mva[iva] >= magthr).astype(int))], eval_metric="auc",
                              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        pvd = Ld.predict_proba(Xva.iloc[iva])[:, 1]
        models[s] = {"Ld": Ld, "Lm": Lm, "magthr": magthr, "confv": np.abs(pvd - 0.5),
                     "pvm": Lm.predict_proba(Xva.iloc[iva])[:, 1], "val_auc": float(roc_auc_score(yva[iva], pvd)), "n_tr": len(itr)}
        hb(f"[{s}] fit n_tr={len(itr)} VAL dirAUC={models[s]['val_auc']:.4f}")
    del Xtr, Xva; gc.collect()

    Xte, yte, mte, vte, tste = load("test"); Xoo, yoo, moo, voo, tsoo = load("oos")
    X = pd.concat([Xte, Xoo]); y = np.concatenate([yte, yoo]); mag = np.concatenate([mte, moo])
    valid = np.concatenate([vte, voo]); ts = np.concatenate([tste, tsoo]); del Xte, Xoo; gc.collect()

    for s in SESSIONS:
        M = models[s]; sm = session_mask(ts, s); base = valid & (mag > 0) & sm
        pr = M["Ld"].predict_proba(X)[:, 1]; pred = (pr > 0.5).astype(int); conf = np.abs(pr - 0.5)
        dirres = {"key": f"(EURUSD,{H}m,{s}) session-only DST", "val_dirAUC": round(M["val_auc"], 4), "n_train": M["n_tr"],
                  "falsifier": "KILL if VAL dirAUC<=0.515 OR no yr CI-lo>=0.541 OR CPCV p10<0.541", "coverages": {}}
        for cov in (0.10, 0.05):
            thr = float(np.quantile(M["confv"], 1 - cov)) if len(M["confv"]) else 0.0
            sel = nonoverlap(ts, base & (conf >= thr))
            for side, sp in (("COMB", None), ("UP", 1), ("DOWN", 0)):
                ss = sel if sp is None else sel[pred[sel] == sp]
                if len(ss) < 20: dirres["coverages"].setdefault(f"cov{cov}", {})[side] = {"n": int(len(ss))}; continue
                win = (pred[ss] == y[ss]).astype(float)
                dirres["coverages"].setdefault(f"cov{cov}", {})[side] = {"per_year": per_year(ts[ss], win), "cpcv": cpcv_side(ts[ss], win)}
        cert = any((sd.get("cpcv") or {}).get("path_p10") and sd["cpcv"]["path_p10"] >= BREAKEVEN
                   and any(v["ci95"][0] >= BREAKEVEN for v in sd.get("per_year", {}).values())
                   for cv in dirres["coverages"].values() for sd in cv.values())
        dirres["VERDICT"] = "SURVIVES" if (M["val_auc"] > 0.515 and cert) else "KILLED"
        json.dump(dirres, open(f"session_{H}m_dir_{s}_result.json", "w"), indent=1)

        pm = M["Lm"].predict_proba(X)[:, 1]; ytrue = (mag >= M["magthr"]).astype(int)
        magres = {"key": f"(EURUSD,{H}m,{s}) MAGNITUDE session-only", "magthr": M["magthr"],
                  "test_oos_magAUC": round(float(roc_auc_score(ytrue[base], pm[base])), 4), "selective": {}}
        for cov in (0.20, 0.10, 0.05):
            thr = float(np.quantile(M["pvm"], 1 - cov)) if len(M["pvm"]) else 1.0
            sel = nonoverlap(ts, base & (pm >= thr))
            if len(sel) < 30: magres["selective"][f"cov{cov}"] = {"n": int(len(sel))}; continue
            magres["selective"][f"cov{cov}"] = {"per_year": per_year(ts[sel], (ytrue[sel] == 1).astype(float)), "cpcv": cpcv_side(ts[sel], (ytrue[sel] == 1).astype(float))}
        json.dump(magres, open(f"session_{H}m_mag_{s}_result.json", "w"), indent=1)
        hb(f"[{s}] DIR {dirres['VERDICT']} valAUC={M['val_auc']:.3f} | MAG AUC={magres['test_oos_magAUC']}")
    hb(f"DONE H={H}m -> session_{H}m_{{dir,mag}}_{{ny,ldn,asia}}_result.json")


if __name__ == "__main__":
    main()
