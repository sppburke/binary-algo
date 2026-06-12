"""SESSION re-campaign — 1m (60s) GBM DIRECTION + MAGNITUDE, session-only (NY/LDN/Asian), DST-correct.

Per session: train + select + evaluate using ONLY that session's decision bars (DST-correct, sessions.py), in
train/val/test/OOS alike. Features stay causal/continuous (min1_production.prep); only label/decision rows are
session-filtered. Deriv-faithful (wc_ret ties-LOSE), per-year CI95, nonoverlap, CPCV, pre-registered falsifier.
DIRECTION = single LGBM (mk_lgb) combined + UP-split + DOWN-split. MAGNITUDE = |ret60|>=session-train-median.

Run: ~/binary-algo-venv/bin/python session_1m.py   ->  session_1m_{dir,mag}_{ny,ldn,asia}_result.json
"""
import json, time, gc, numpy as np
from itertools import combinations
import min1_production as MP
from sessions import session_mask, SESSIONS
BREAKEVEN = 0.541; STRIDE = 5; SUBCAP = 150_000
T0 = time.time()
def hb(m): print(f"HB[{time.time()-T0:.0f}s] {m}", flush=True)


def cpcv_side(ts, win, n_groups=8, k=2):
    o = np.argsort(ts); ts, win = ts[o], win[o]; n = len(win)
    if n < 60: return {"n": n, "pooled": round(float(win.mean()), 4) if n else None, "path_p10": None, "frac_clear": None}
    edges = np.linspace(0, n, n_groups + 1).astype(int); grp = [(edges[i], edges[i+1]) for i in range(n_groups)]
    pa = []
    for tg in combinations(range(n_groups), k):
        ii = np.concatenate([np.arange(grp[g][0], grp[g][1]) for g in tg])
        if len(ii) >= 25: pa.append(float(win[ii].mean()))
    pa = np.array(pa)
    return {"n": n, "pooled": round(float(win.mean()), 4), "path_p10": round(float(np.percentile(pa, 10)), 4),
            "path_mean": round(float(pa.mean()), 4), "frac_clear": round(float((pa >= BREAKEVEN).mean()), 3)}


def per_year(ts, win):
    yr = MP.pd.to_datetime(ts, unit="s", utc=True).year.values
    out = {}
    for Y in (2024, 2025, 2026):
        m = yr == Y
        if m.sum() < 20: continue
        lo, hi = MP.boot(win[m]); out[str(Y)] = {"n": int(m.sum()), "acc": round(float(win[m].mean()), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    return out


def main():
    hb("prep train+val")
    Xtr, ytr, mtr, vtr, tstr, _ = MP.prep(MP.load_split("train"))
    Xva, yva, mva, vva, tsva, _ = MP.prep(MP.load_split("val"))
    feat = list(Xtr.columns)
    models = {}
    for s in SESSIONS:
        smtr = session_mask(tstr, s); smva = session_mask(tsva, s)
        itr = np.where(vtr & (mtr > 0) & smtr)[0][::STRIDE]
        if len(itr) > SUBCAP: itr = itr[np.linspace(0, len(itr) - 1, SUBCAP).astype(int)]
        iva = np.where(vva & (mva > 0) & smva)[0]
        Ld = MP.mk_lgb(2500); Ld.fit(Xtr.iloc[itr], ytr[itr], eval_set=[(Xva.iloc[iva], yva[iva])],
                                     eval_metric="auc", callbacks=[MP.lgb.early_stopping(100), MP.lgb.log_evaluation(0)])
        magthr = float(np.nanpercentile(mtr[itr], 50))
        Lm = MP.mk_lgb(2500); Lm.fit(Xtr.iloc[itr], (mtr[itr] >= magthr).astype(int),
                                     eval_set=[(Xva.iloc[iva], (mva[iva] >= magthr).astype(int))], eval_metric="auc",
                                     callbacks=[MP.lgb.early_stopping(100), MP.lgb.log_evaluation(0)])
        pvd = Ld.predict_proba(Xva.iloc[iva])[:, 1]; confv = np.abs(pvd - 0.5)
        pvm = Lm.predict_proba(Xva.iloc[iva])[:, 1]
        val_auc = float(MP.roc_auc_score(yva[iva], pvd))
        models[s] = {"Ld": Ld, "Lm": Lm, "magthr": magthr, "confv": confv, "pvm": pvm, "val_auc": val_auc, "n_tr": len(itr), "n_va": len(iva)}
        hb(f"[{s}] fit dir+mag n_tr={len(itr)} n_va={len(iva)} VAL dirAUC={val_auc:.4f}")
    del Xtr, Xva; gc.collect()

    hb("prep test+oos")
    Xte, yte, mte, vte, tste, idxte = MP.prep(MP.load_split("test"))
    Xoo, yoo, moo, voo, tsoo, idxoo = MP.prep(MP.load_split("oos"))
    X = MP.pd.concat([Xte, Xoo]); y = np.concatenate([yte, yoo]); mag = np.concatenate([mte, moo])
    valid = np.concatenate([vte, voo]); ts = np.concatenate([tste, tsoo]); del Xte, Xoo; gc.collect()

    for s in SESSIONS:
        M = models[s]; sm = session_mask(ts, s); base = valid & (mag > 0) & sm
        pr = M["Ld"].predict_proba(X)[:, 1]; pred = (pr > 0.5).astype(int); conf = np.abs(pr - 0.5)
        # DIRECTION: VAL worst-half confidence gate at cov 10/5%
        dirres = {"key": f"(EURUSD,60s,{s}) session-only", "val_dirAUC": round(M["val_auc"], 4), "n_train": M["n_tr"],
                  "falsifier": "KILL if VAL dirAUC<=0.515 OR no yr CI-lo>=0.541 OR CPCV p10<0.541", "coverages": {}}
        for cov in (0.10, 0.05):
            thr = float(np.quantile(M["confv"], 1 - cov)) if len(M["confv"]) else 0.0
            cand = base & (conf >= thr); sel = MP.nonoverlap_chrono(ts, cand)
            for side, sp in (("COMB", None), ("UP", 1), ("DOWN", 0)):
                ss = sel if sp is None else sel[pred[sel] == sp]
                if len(ss) < 20: dirres["coverages"].setdefault(f"cov{cov}", {})[side] = {"n": int(len(ss))}; continue
                win = (pred[ss] == y[ss]).astype(float)
                dirres["coverages"].setdefault(f"cov{cov}", {})[side] = {
                    "per_year": per_year(ts[ss], win), "cpcv": cpcv_side(ts[ss], win)}
        # verdict: any side/cov with a year CI-lo>=BE AND cpcv p10>=BE
        cert = False
        for cv in dirres["coverages"].values():
            for side in cv.values():
                cp = side.get("cpcv"); py = side.get("per_year", {})
                if cp and cp.get("path_p10") and cp["path_p10"] >= BREAKEVEN and any(v["ci95"][0] >= BREAKEVEN for v in py.values()):
                    cert = True
        dirres["VERDICT"] = "SURVIVES" if (M["val_auc"] > 0.515 and cert) else "KILLED"
        json.dump(dirres, open(f"session_1m_dir_{s}_result.json", "w"), indent=1)

        # MAGNITUDE
        pm = M["Lm"].predict_proba(X)[:, 1]; ytrue = (mag >= M["magthr"]).astype(int)
        magres = {"key": f"(EURUSD,60s,{s}) MAGNITUDE session-only", "magthr": M["magthr"],
                  "test_oos_magAUC": round(float(MP.roc_auc_score(ytrue[base], pm[base])), 4), "selective": {}}
        for cov in (0.20, 0.10, 0.05):
            thr = float(np.quantile(M["pvm"], 1 - cov)) if len(M["pvm"]) else 1.0
            cand = base & (pm >= thr); sel = MP.nonoverlap_chrono(ts, cand)
            if len(sel) < 30: magres["selective"][f"cov{cov}"] = {"n": int(len(sel))}; continue
            win = (ytrue[sel] == 1).astype(float)
            magres["selective"][f"cov{cov}"] = {"per_year": per_year(ts[sel], win), "cpcv": cpcv_side(ts[sel], win)}
        json.dump(magres, open(f"session_1m_mag_{s}_result.json", "w"), indent=1)
        hb(f"[{s}] DIR {dirres['VERDICT']} valAUC={M['val_auc']:.3f} | MAG AUC={magres['test_oos_magAUC']}")
    hb("DONE -> session_1m_{dir,mag}_{ny,ldn,asia}_result.json")


if __name__ == "__main__":
    main()
