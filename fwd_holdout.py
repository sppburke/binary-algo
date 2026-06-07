"""PHASE 1B — reusable FROZEN-PAST FORWARD-HOLDOUT gate (the mandatory deployment-faithful falsifier).

WHY THIS EXISTS (leakage trap #9): pooled CPCV does NOT catch non-stationary-feature memorization. With 8 groups / k=2,
20 of 28 test folds are FLANKED by train folds, so any calendar/seasonal/slow/era-local feature memorizes structure that
has NO forward transfer — yet shows a clean pooled-CPCV lift (the deseason +tod lever: pooled +0.0140, forward decays to
-0.0544 by 2026). The ONLY faithful test of "will this feature help a model trained today, deployed forward" is to FREEZE
the past, train once on <= TRAIN_MAX, and score EACH future year separately. A feature is deployable only if it does not
decay across the forward years (delta-vs-base >= 0 every year, not just pooled).

This generalizes deseason_fwd.py into one importable function for ANY feature arm, in TWO modes:
  - mode="magnitude": label = (|fwd_ret| >= TRAIN-only Q75); metric = ROC-AUC + top/bottom-decile |ret| lift.
  - mode="direction": label = sign(fwd_ret) (ties dropped); metric = selective accuracy at confidence coverage `cov`
                       (deriv-faithful: |p-0.5| top-`cov` fraction), plus plain AUC.

Usage:
  from fwd_holdout import forward_holdout, mk_lgb
  arms = {"base": Xbase, "+feat": np.column_stack([Xbase, newfeat])}
  res  = forward_holdout(arms, target=aret, ts=ts, mode="magnitude")   # target=|ret| for magnitude
  res  = forward_holdout(arms, target=fwd_signed, ts=ts, mode="direction", cov=0.10)
  # res["deployable"][armkey] is True iff that arm's delta-vs-base >= -tol in EVERY forward year.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score


def mk_lgb(n=600):
    """The certified GBM config (matches cpcv_certify.mk_lgb / build_panel.mk_lgb). No estimator seed -> n_jobs parallel."""
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.03, num_leaves=255,
        min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10,
        n_estimators=n, n_jobs=20, verbosity=-1)


def _decile_lift(pp, a):
    """Ratio of mean |ret| in the model's top-decile vs bottom-decile by predicted score."""
    q = np.quantile(pp, [0.9, 0.1])
    return float(a[pp >= q[0]].mean() / max(a[pp <= q[1]].mean(), 1e-12))


def _sel_acc(pp, yy, cov):
    """Deriv-faithful selective accuracy: among the top-`cov` fraction by confidence |p-0.5|, hit-rate of sign(p-0.5)."""
    conf = np.abs(pp - 0.5); thr = np.quantile(conf, 1 - cov); sel = conf >= thr
    if sel.sum() < 25: return float("nan"), int(sel.sum())
    return float(((pp[sel] > 0.5).astype(int) == yy[sel]).mean()), int(sel.sum())


def forward_holdout(arms, target, ts, *, mode="magnitude", train_max=2023, test_years=(2024, 2025, 2026),
                    base_key="base", thr_q=0.75, cov=0.10, mk_model=mk_lgb, tol=0.0, verbose=True):
    """Freeze <= train_max, score each future year. `arms` = {name: X[n,d]} sharing one row order with target/ts.
    target: magnitude -> |fwd_ret| (>=0); direction -> SIGNED fwd_ret (ties==0 dropped). ts: epoch-seconds int array.
    Returns dict with per-arm per-year metrics + delta-vs-base + a `deployable` flag (delta>=-tol every forward year)."""
    target = np.asarray(target, float); ts = np.asarray(ts).astype("int64")
    yr = pd.to_datetime(ts, unit="s", utc=True).year.values
    tr = yr <= train_max
    if mode == "magnitude":
        thr = float(np.nanquantile(target[tr], thr_q)); label = (target >= thr).astype(int); keep = np.isfinite(target)
    elif mode == "direction":
        thr = None; label = (target > 0).astype(int); keep = np.isfinite(target) & (target != 0.0)
    else:
        raise ValueError(mode)
    trk = tr & keep
    if verbose:
        print(f"[fwd] mode={mode} train(<= {train_max}) n={trk.sum():,} up/big-rate={label[trk].mean():.4f}"
              + (f" thr={thr:.6g}" if thr is not None else ""))

    fitted = {}
    for nm, X in arms.items():
        X = np.asarray(X, np.float32); m = mk_model(); m.fit(X[trk], label[trk]); fitted[nm] = (m, X)
        if verbose: print(f"[fwd] fit {nm} ({X.shape[1]} feats)")

    out = {"mode": mode, "train_max": train_max, "thr_q": thr_q if mode == "magnitude" else None,
           "cov": cov if mode == "direction" else None, "by_arm": {}, "deltas": {}, "deployable": {}}
    years = list(test_years) + ["pooled"]
    for nm, (m, X) in fitted.items():
        peryr = {}
        for Y in years:
            te = (yr >= test_years[0]) & keep if Y == "pooled" else (yr == Y) & keep
            if te.sum() < 1000: continue
            pp = m.predict_proba(X[te])[:, 1]; yy = label[te]; a = target[te]
            try: auc = float(roc_auc_score(yy, pp))
            except ValueError: auc = float("nan")
            rec = {"n": int(te.sum()), "auc": round(auc, 4)}
            if mode == "magnitude":
                rec["lift"] = round(_decile_lift(pp, a), 3)
            else:
                sa, nsel = _sel_acc(pp, yy, cov); rec["selacc"] = round(sa, 4); rec["n_sel"] = nsel
            peryr[str(Y)] = rec
        out["by_arm"][nm] = peryr

    # delta vs base + deployability (must not decay in ANY forward year)
    bkey = mode == "magnitude" and "auc" or "selacc"
    basep = out["by_arm"].get(base_key, {})
    for nm, peryr in out["by_arm"].items():
        if nm == base_key: continue
        deltas = {}; ok = True
        for Y in [str(y) for y in test_years]:
            if Y in peryr and Y in basep:
                d = peryr[Y][bkey] - basep[Y][bkey]; deltas[Y] = round(d, 4)
                if d < -tol: ok = False
        out["deltas"][nm] = deltas; out["deployable"][nm] = bool(ok and deltas)

    if verbose:
        m = "auc" if mode == "magnitude" else "selacc"
        hdr = f"{'arm':>14} | " + " ".join(f"{str(y):>8}" for y in years) + " | verdict"
        print(hdr); print("-" * len(hdr))
        for nm, peryr in out["by_arm"].items():
            cells = " ".join(f"{peryr[str(y)][m]:>8.4f}" if str(y) in peryr else f"{'--':>8}" for y in years)
            verdict = "" if nm == base_key else ("DEPLOY" if out["deployable"][nm] else "DECAYS") \
                + " " + " ".join(f"{Y}{d:+.4f}" for Y, d in out["deltas"][nm].items())
            print(f"{nm:>14} | {cells} | {verdict}")
    return out


def _selfcheck():
    """The gate must DEPLOY a stationary forward-signal feature and REJECT a trap-#9 NON-STATIONARY one (informative
    in-sample, relationship reversed forward) — the exact deseason-+tod failure this gate exists to catch."""
    rng = np.random.default_rng(0); n = 60000
    ts = (np.arange(n) * 3200 + 1325376000).astype("int64")        # spread evenly across ~2012-2018
    yr = pd.to_datetime(ts, unit="s", utc=True).year.values
    sig = rng.standard_normal(n)                                   # a real driver (stationary across all years)
    fwd = 0.6 * sig + rng.standard_normal(n)                       # signed return driven by sig
    base = rng.standard_normal((n, 2)).astype(np.float32)          # uninformative base
    # non-stationary feature: in TRAIN years it predicts |fwd|; in TEST years its sign-relationship REVERSES (decays)
    z = rng.standard_normal(n); past = yr <= 2014
    decay = np.where(past, np.abs(fwd) + 0.3 * z, -np.abs(fwd) + 0.3 * z)
    arms = {"base": base, "+decay": np.column_stack([base, decay]).astype(np.float32),
            "+signal": np.column_stack([base, sig]).astype(np.float32)}
    r = forward_holdout(arms, target=np.abs(fwd), ts=ts, mode="magnitude", train_max=2014,
                        test_years=(2015, 2016), verbose=True)
    print("decay deployable (want False):", r["deployable"]["+decay"],
          "| signal deployable (want True):", r["deployable"]["+signal"])


if __name__ == "__main__":
    _selfcheck()
