"""Freeze the POW=0.5 magnitude-weighted primary as EURUSD.m5xp_magw_down.v1.

Trains the magweight (sample_weight=(|fwd|/median)^0.5) LightGBM primary on ALL pooled years
(2012-2026, stride=6) using the same hyperparams as the CPCV (RF.mk_lgb: n_est=700, no early stopping).
Saves artifact, derives a fixed DOWN-confidence threshold from the val split (2022-2023), then freezes
the book with manifest.

CERTIFIED: DOWN cov0.05 CPCV p10=0.5441, 89.3% paths clear (B1 m5_magweight_cpcv_result.json @ git HEAD~4).
LABELED MARGINAL: p10 barely above breakeven 0.541; regime-dependent (2025 is the weakest fold).
Gate: NY session & pr < 0.5 & |pr-0.5| >= threshold (top-5% confidence cover on val DOWN predictions).
"""
import os, json, time, gc, numpy as np
import lightgbm as lgb
import m5_xpair as MX
import m5_xpair_production as XP
import m5_cpcv_refit as RF
import manifest

MODELS = "/media/sean/CORSAIR/binary-algo/models"
STRIDE = 6
POW = 0.5
ALL_YEARS = [str(y) for y in range(2012, 2027)]
VAL_YEARS = ["2022", "2023"]
OUT_ART = os.path.join(MODELS, "m5xp_EURUSD_magw_down_primary_lgb.txt")
BOOK_ID = "EURUSD.m5xp_magw_down.v1"


def magweight(fwd, pow_=POW):
    a = np.abs(fwd).astype(float)
    med = np.median(a[a > 0]) or 1e-9
    return np.clip((a / med) ** pow_, 0.1, 10.0)


def load_years(years, stride=1):
    p = json.load(open(XP.art("strategy.json")))
    cols = p["primary_feats"]
    Xs, ys, tss, nys, fws = [], [], [], [], []
    for yr in years:
        D = MX.build_xp([yr], stride=stride)
        if len(D) == 0:
            continue
        D = MX.augment(D, [yr], XP.MODE)
        Xs.append(D[cols].astype("float32").to_numpy())
        ys.append(D["_y"].astype(np.int8).values)
        tss.append(D["_ts"].values.astype("int64"))
        nys.append((D["sess_ny"].values > 0.5))
        fws.append(D["_fwd"].astype("float32").values)
        del D; gc.collect()
    X = np.concatenate(Xs); y = np.concatenate(ys)
    ts = np.concatenate(tss); ny = np.concatenate(nys); fwd = np.concatenate(fws)
    order = np.argsort(ts, kind="stable")
    return X[order], y[order], ts[order], ny[order], fwd[order], cols


def main():
    t0 = time.time()
    print(f"[magw-freeze] loading all years stride={STRIDE}...", flush=True)
    X, y, ts, ny, fwd, cols = load_years(ALL_YEARS, stride=STRIDE)
    print(f"[magw-freeze] n={len(y):,} feats={len(cols)} up-rate={y.mean():.4f} ({time.time()-t0:.0f}s)", flush=True)

    # Train on full pool with magweight
    sw = magweight(fwd)
    model = RF.mk_lgb(n_est=700)
    model.fit(X, y, sample_weight=sw)
    os.makedirs(MODELS, exist_ok=True)
    model.booster_.save_model(OUT_ART)
    print(f"[magw-freeze] saved primary -> {OUT_ART} ({time.time()-t0:.0f}s)", flush=True)

    # Derive fixed DOWN-confidence threshold from val split (2022-2023)
    print(f"[magw-freeze] deriving DOWN threshold on val {VAL_YEARS}...", flush=True)
    Xv, yv, tsv, nyv, fwdv, _ = load_years(VAL_YEARS, stride=1)
    booster = lgb.Booster(model_file=OUT_ART)
    prv = booster.predict(Xv)
    down_gate = nyv & (prv < 0.5)
    conf_down = np.abs(prv[down_gate] - 0.5)
    # top-5% coverage threshold: 95th percentile of confidence among DOWN candidates
    cov_thr = float(np.quantile(conf_down, 0.95)) if down_gate.sum() >= 40 else 0.0
    val_down_sel = down_gate & (np.abs(prv - 0.5) >= cov_thr)
    val_down_acc = float((yv[val_down_sel] == 0).mean()) if val_down_sel.sum() > 0 else float("nan")
    val_down_n = int(val_down_sel.sum())
    print(f"[magw-freeze] val DOWN: thr={cov_thr:.4f} n={val_down_n} acc={val_down_acc:.4f}", flush=True)

    # Strategy JSON (DOWN-specific)
    strategy = {
        "pair": "EURUSD", "mode": XP.MODE,
        "primary_feats": cols,
        "gate": "sess_ny & pr<0.5 & |pr-0.5|>=conf_thr (top-5% DOWN confidence cover)",
        "conf_thr_down_cov05": cov_thr,
        "side": "DOWN",
        "cov_pct": 0.05,
        "settlement": "mid-to-mid ties lose breakeven~0.541",
        "splits": {"train": "all 2012-2026 pooled (stride=6)", "threshold_derived_on": "2022-2023 val"},
        "val_down_acc_cov05": round(val_down_acc, 4),
        "val_down_n_cov05": val_down_n,
        "cpcv_certification": {
            "result_file": "m5_magweight_cpcv_result.json (git HEAD~4, POW=0.5 B1 run)",
            "DOWN_cov0.05_p10": 0.5441,
            "DOWN_cov0.05_frac_clear": 0.893,
            "CERTIFIED": True,
            "label": "MARGINAL — p10 barely above breakeven 0.541; regime-dependent"
        }
    }
    strat_path = os.path.join(MODELS, "m5xp_EURUSD_magw_down_strategy.json")
    json.dump(strategy, open(strat_path, "w"), indent=2)
    print(f"[magw-freeze] strategy -> {strat_path}", flush=True)

    # Build and freeze manifest
    feat_dir = "/media/sean/CORSAIR/binary-algo/features"
    fp = manifest.dir_fingerprint(feat_dir, ("EURUSD_*.parquet",))
    m = manifest.build(
        book_id=BOOK_ID,
        timeframe="5m",
        side="down",
        role="direction",
        script="m5_magweight_freeze.py",
        summary=(
            "Magnitude-weighted (POW=0.5 |return|/median) LightGBM primary for the EURUSD 5m DOWN side. "
            "CERTIFIED MARGINAL: CPCV p10=0.5441 (89.3% of 28 purged paths clear 0.541 breakeven). "
            "Gate: NY session & pr<0.5 & top-5% confidence cover. Regime-dependent; 2025 is weakest fold."
        ),
        metrics={
            "cpcv_down_cov0.05_p10": 0.5441,
            "cpcv_down_cov0.05_frac_clear": 0.893,
            "cpcv_down_cov0.05_mean": 0.5569,
            "cpcv_certified": True,
            "label": "MARGINAL",
            "breakeven": 0.541,
            "val_down_acc_cov05": round(val_down_acc, 4),
        },
        artifacts=[OUT_ART],
        hyperparams={
            "model": "LightGBM",
            "n_estimators": 700,
            "learning_rate": 0.03,
            "num_leaves": 127,
            "min_child_samples": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.5,
            "reg_lambda": 20,
            "sample_weight": f"(|fwd|/median)^{POW} clipped [0.1,10]",
            "n_jobs": 20,
            "estimator_seed": "unset",
            "training_data": "all 2012-2026 pooled stride=6",
        },
        strategy_json=strat_path,
        feature_fingerprint=fp,
        depends_on=["EURUSD.m5xp.v1"],
    )
    mpath = manifest.freeze(m, artifacts_src=[OUT_ART])
    print(f"[magw-freeze] manifest frozen -> {mpath}", flush=True)
    print(f"[magw-freeze] DONE book={BOOK_ID} content_id={m['content_id']} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
