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


def _persistence_prob(ts, up, keep, gap_s, win_s=86400.0, min_n=50):
    """NAIVE trailing up-rate baseline forecast, OBSERVABILITY-SAFE. A past decision bar at s predicted [s, s+gap_s], whose
    outcome is only KNOWN at s+gap_s; so at decision time t it may contribute to the baseline iff s+gap_s <= t (s <= t-gap_s).
    prob[t] = mean of those observable past outcomes within a trailing window of win_s seconds (gap_s = horizon seconds).
    Returns an array aligned to `up` (NaN off-`keep`). O(n log n) via cumsum + searchsorted on the ts-sorted kept rows."""
    ts = np.asarray(ts, "float64"); up = np.asarray(up, "float64")
    ks = np.where(keep)[0]
    order = ks[np.argsort(ts[ks], kind="mergesort")]           # kept-row indices, sorted by time
    tss = ts[order]; ups = up[order]
    cum = np.concatenate([[0.0], np.cumsum(ups)])
    right = np.searchsorted(tss, tss - gap_s, side="right")    # last bar observable by t (outcome known)
    left = np.searchsorted(tss, tss - gap_s - win_s, side="left")
    cnt = right - left; psum = cum[right] - cum[left]
    pr = np.where(cnt >= min_n, psum / np.maximum(cnt, 1), 0.5)
    out = np.full(len(up), np.nan); out[order] = pr
    return out


def _oof_isotonic(X, y, mk_model, k=3):
    """Out-of-fold isotonic calibrator fit on TRAIN ONLY (non-leaky): k positional folds -> OOF model probs -> isotonic map.
    Returned map is applied to the deployed model's forward-year probs so Brier does not penalise a well-ranked-but-
    miscalibrated model unfairly (the synthesizer's caveat). Fit on honest OOF preds, never on rows the calibrator saw."""
    from sklearn.isotonic import IsotonicRegression
    n = len(y); folds = np.array_split(np.arange(n), k); oof = np.empty(n, float)
    for j in range(k):
        te = folds[j]; tr = np.concatenate([folds[t] for t in range(k) if t != j])
        mm = mk_model(); mm.fit(X[tr], y[tr]); oof[te] = mm.predict_proba(X[te])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0); iso.fit(oof, y)
    return iso


def forward_holdout(arms, target, ts, *, mode="magnitude", train_max=2023, test_years=(2024, 2025, 2026),
                    base_key="base", thr_q=0.75, cov=0.10, mk_model=mk_lgb, tol=0.0, verbose=True,
                    brier=False, calibrate=True, cal_k=3, persist_gap_s=None, persist_win_s=86400.0, persist_min_n=50):
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

    do_brier = bool(brier) and mode == "direction"
    persist = None
    if do_brier:
        gap_s = float(persist_gap_s) if persist_gap_s is not None else 900.0
        persist = _persistence_prob(ts, label, keep, gap_s, persist_win_s, persist_min_n)
        if verbose: print(f"[fwd] brier: persistence baseline gap={gap_s:.0f}s win={persist_win_s:.0f}s "
                          f"(mean={np.nanmean(persist):.4f}) calibrate={calibrate}")

    fitted = {}
    for nm, X in arms.items():
        X = np.asarray(X, np.float32); m = mk_model(); m.fit(X[trk], label[trk])
        iso = _oof_isotonic(X[trk], label[trk], mk_model, cal_k) if (do_brier and calibrate) else None
        fitted[nm] = (m, X, iso)
        if verbose: print(f"[fwd] fit {nm} ({X.shape[1]} feats)" + (" +isotonic" if iso is not None else ""))

    out = {"mode": mode, "train_max": train_max, "thr_q": thr_q if mode == "magnitude" else None,
           "cov": cov if mode == "direction" else None, "by_arm": {}, "deltas": {}, "deployable": {}}
    years = list(test_years) + ["pooled"]
    for nm, (m, X, iso) in fitted.items():
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
            if do_brier and Y != "pooled":
                pc = iso.predict(pp) if iso is not None else pp        # calibrated probs (deployed model, train-only map)
                pb = persist[te]                                       # naive trailing up-rate baseline
                bm = float(np.mean((pc - yy) ** 2)); bmr = float(np.mean((pp - yy) ** 2))
                bf = float(np.mean((0.5 - yy) ** 2)); bp = float(np.mean((pb - yy) ** 2))
                rec["brier"] = round(bm, 5); rec["brier_raw"] = round(bmr, 5)
                rec["brier_flat"] = round(bf, 5); rec["brier_persist"] = round(bp, 5)
                rec["badv_flat"] = round(bf - bm, 5)                   # >0 => model beats flat-0.5 (resolution)
                rec["badv_persist"] = round(bp - bm, 5)               # >0 => CALIBRATED model beats persistence (skill)
                rec["badv_persist_raw"] = round(bp - bmr, 5)         # uncalibrated, conservative
                conf = np.abs(pp - 0.5); thr2 = np.quantile(conf, 1 - cov); s2 = conf >= thr2   # the bet tail
                if s2.sum() >= 25:
                    rec["badv_persist_sel"] = round(float(np.mean((pb[s2] - yy[s2]) ** 2)
                                                          - np.mean((pc[s2] - yy[s2]) ** 2)), 5)
                    rec["n_sel_brier"] = int(s2.sum())
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
        if do_brier:
            print("[brier] badv_persist (calibrated model Brier minus persistence Brier; >0 = real probabilistic skill):")
            for nm, peryr in out["by_arm"].items():
                cells = " ".join(f"{peryr[str(y)]['badv_persist']:>+8.4f}" if str(y) in peryr and "badv_persist" in peryr[str(y)]
                                 else f"{'--':>8}" for y in test_years)
                sel = " ".join(f"{peryr[str(y)].get('badv_persist_sel', float('nan')):>+8.4f}" if str(y) in peryr else f"{'--':>8}"
                               for y in test_years)
                print(f"{nm:>14} | all {cells} | bet-tail {sel}")
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


def _selfcheck_brier():
    """The Brier-advantage must be POSITIVE for a feature with genuine sign skill and ~<=0 for a noise-only base, on a
    frozen-past forward holdout — proving the metric credits real probabilistic skill over the naive persistence baseline."""
    rng = np.random.default_rng(1); n = 80000
    ts = (np.arange(n) * 2400 + 1325376000).astype("int64")        # ~6y span (2012-2017) so train_max=2014 has fwd years
    sig = rng.standard_normal(n)
    fwd = 0.5 * sig + rng.standard_normal(n)                        # signed return; its SIGN is partly driven by sig
    base = rng.standard_normal((n, 2)).astype(np.float32)          # uninformative
    arms = {"base": base, "+signal": np.column_stack([base, sig]).astype(np.float32)}
    r = forward_holdout(arms, target=fwd, ts=ts, mode="direction", train_max=2014, test_years=(2015, 2016, 2017),
                        cov=0.2, brier=True, persist_gap_s=2400.0, persist_win_s=2400.0 * 2000, verbose=True)
    yrs = [y for y in ("2015", "2016", "2017") if y in r["by_arm"]["+signal"]]
    sb = [r["by_arm"]["+signal"][y]["badv_persist"] for y in yrs]
    bb = [r["by_arm"]["base"][y]["badv_persist"] for y in yrs]
    print("brier selfcheck: +signal badv_persist", [round(x, 4) for x in sb], "(want all >0) | base",
          [round(x, 4) for x in bb], "(want ~<=0)")


if __name__ == "__main__":
    _selfcheck()
    print("\n" + "=" * 80 + "\n[brier self-check]\n")
    _selfcheck_brier()
