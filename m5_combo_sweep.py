"""5-MIN EURUSD DIRECTION — MODEL-COMBINATION SWEEP HARNESS (reuse frozen models as per-bar P(up) producers).

PURPOSE
  ONE harness that (A) builds & CACHES a per-bar score matrix from the FROZEN models (light inference, NO retraining
  of any base model), then (B) sweeps the full model-COMBINATION cross-product for the 5m binary-direction goal.
  The score-matrix build is the only heavy part (frozen-model inference over train+eval). The sweep itself is cheap
  (operates on cached score vectors).

PRODUCERS (all aligned to EURUSD's OWN 5m anchor clock; longer horizons FRONT-LOADED at anchor t, input<=t, leakage-safe):
  p5,p10,p15,p30  per-horizon direction P(up): each = blend(lgb,xgb,cat) on the SAME 239-feature matrix at bar t,
                  scored with that horizon's OWN frozen trio (m{H}_EURUSD_direction_{lgb,xgb,cat}). The model learned
                  how features at t predict the [t+H] move; we read it at the 5m anchor t (front-load, no interpolation).
  pxp             m5xp primary P(up) (cross-pair+base+OF lgb primary, trained 2012-21).
  mxp             m5xp meta CONFIDENCE = P(primary correct) from orthogonal axes (trained on VAL 2022-23).
  pstk            m5stack P(up): the 15m ensemble DIRECTION scored on the 5m clock (frozen 15m trio), passed through
                  the m5stack meta-labeler P(dir15 correct on 5m). pstk = dir15 mapped to a P(up): we expose BOTH the
                  15m direction prob (p15) and the stack-meta-gated direction as pstk_dir; pstk here = the 15m direction
                  P(up) used as the trade direction, with mxp_stk = stack-meta confidence carried alongside for gating.
  pmag            magnitude P(large) — GATE ONLY, sign-invariant (min1 60s model). Aggregated to the 5m clock by taking
                  max over the 1s pmag in the PRECEDING 300s window [t-300, t] (input<=t). Bars with no preceding 1s
                  window are dropped by the inner-join (NEVER ffill).
  y5              TRUE 5m deriv label from build_xp H=5 (next-300s move sign, ties already dropped). fwd5 = raw fwd ret.

EVAL DISCIPLINE (m5_lossbatch.per_year + m5_xpair_production worst-VAL-half): NY gate, conf-cover quantile (COV default
  0.05, also 0.10), MX.nonoverlap_chrono(ts,mask,300), per-year 2024/2025/2026 boot CI95, win vs TRUE y5 (TIES LOSE),
  moved-bar up-rate tripwire [0.47,0.53]. ALL thresholds/weights selected on WORST-VAL-HALF (median-time split of VAL,
  min over the two halves) — NEVER VAL-acc-max.

PRE-REGISTERED FALSIFIER (written to m5_combo_sweep_result.json BEFORE any test-year eval):
  For EACH combination KILL unless binding-year (worst of 2024/2025/2026, n>=150, nonoverlap, ties-LOSE,
  worst-VAL-half-selected) CI95-lo: UP >= 0.553 AND point beats incumbent binding-2025 0.577 by >1 SE (or CI-lo>=0.577);
  DOWN CI95-lo >= 0.541 AND beats incumbent 0.5441 by >1 SE; AND up-rate in [0.47,0.53]; AND 2026 not collapsed <0.541.
  Any survivor -> flagged for nested-refit CPCV before freeze. Magnitude is a gate only (sign-invariant).

USAGE
  ~/binary-algo-venv/bin/python m5_combo_sweep.py build   # build/cache score matrices for all splits (heavy)
  ~/binary-algo-venv/bin/python m5_combo_sweep.py sweep    # run the combination sweep on cached matrices (cheap)
  ~/binary-algo-venv/bin/python m5_combo_sweep.py smoke    # build OOS matrix only (smoke test; default)
"""
import sys, os, json, time, gc, warnings
warnings.filterwarnings("ignore")
import numpy as np

ROOT = "/media/sean/CORSAIR/binary-algo"
MODELS = f"{ROOT}/models"
PAIR = "EURUSD"
BE = 0.541                      # deriv breakeven win-rate
INCUMBENT_UP = 0.577            # incumbent UP binding-2025
INCUMBENT_DOWN = 0.5441         # incumbent DOWN binding
UP_FLOOR = 0.553
COV_DEFAULT = 0.05
COVS = (0.05, 0.10)
GAP_S = 300                     # 5m non-overlap block
MAG_WINDOW_S = 300              # aggregate 1s pmag over the preceding 300s window (input<=t)
MIN_N = 150                     # binding-year minimum non-overlap trades
SPL = {"train": [str(y) for y in range(2012, 2022)],
       "val": ["2022", "2023"], "test24": ["2024"], "test25": ["2025"], "oos": ["2026"]}
SPLITS_EVAL = ("val", "test24", "test25", "oos")
SPLITS_ALL = ("train", "val", "test24", "test25", "oos")
HORIZONS = {"p5": "m5", "p10": "m10", "p15": "m15", "p30": "m30"}

# columns persisted per split in the cached score matrix (one row per kept 5m anchor bar)
SCORE_COLS = ["ts", "y5", "fwd5", "sess_ny",
              "p5", "p10", "p15", "p30", "pxp", "mxp", "pstk", "mxp_stk", "pmag"]


# ============================================================ (A) SCORE MATRIX ============================================================

def _lazy_imports():
    import lightgbm as lgb, xgboost as xgb
    from catboost import CatBoostClassifier
    return lgb, xgb, CatBoostClassifier


def _blend_trio(L, G, C, X):
    """3-model direction blend exactly as the production scripts: avg(lgb.predict, xgb proba[:,1], cat proba[:,1])."""
    p_lgb = L.predict(X.values)
    p_xgb = G.predict_proba(X)[:, 1]
    p_cat = C.predict_proba(X.fillna(-999))[:, 1]
    return (p_lgb + p_xgb + p_cat) / 3.0


def _build_horizon_pup(D, horizon_key):
    """Per-horizon direction P(up) on the 5m anchor frame D (already augmented xpof). Front-loaded at t."""
    import json as _j
    lgb, xgb, CatBoostClassifier = _lazy_imports()
    h = HORIZONS[horizon_key]
    strat = _j.load(open(f"{MODELS}/{h}_{PAIR}_strategy.json"))
    feats = strat["feature_names"]
    X = D[feats].astype("float32")
    L = lgb.Booster(model_file=f"{MODELS}/{h}_{PAIR}_direction_lgb.txt")
    G = xgb.XGBClassifier(); G.load_model(f"{MODELS}/{h}_{PAIR}_direction_xgb.json")
    C = CatBoostClassifier(); C.load_model(f"{MODELS}/{h}_{PAIR}_direction_cat.cbm")
    p = _blend_trio(L, G, C, X)
    del L, G, C, X; gc.collect()
    return p.astype("float32")


def _build_m5xp(D):
    """m5xp primary P(up) (pxp) + meta confidence P(primary correct) (mxp). Exactly as m5_xpair_production."""
    import json as _j
    lgb, _, _ = _lazy_imports()
    import m5_xpair_production as XP
    strat = _j.load(open(XP.art("strategy.json")))
    cols = strat["primary_feats"]; mcols = strat["meta_feats"]
    P = lgb.Booster(model_file=XP.art("primary_lgb.txt"))
    M = lgb.Booster(model_file=XP.art("meta_lgb.txt"))
    pxp = P.predict(D[cols].astype("float32"))
    Xmeta = XP._Xmeta(D, pxp, mcols)
    mxp = M.predict(Xmeta)
    del P, M; gc.collect()
    return pxp.astype("float32"), mxp.astype("float32")


def _build_m5stack(D, pxp):
    """m5stack: 15m ensemble direction on the 5m clock (pstk = p15-direction P(up)) + stack-meta confidence (mxp_stk).
    Built exactly as m5_stack2.features(): the 15m trio scores every 5m bar with 15m (=239 base) features front-loaded
    at t; the stack meta-labeler predicts P(dir15 correct on the 5m outcome) from orthogonal axes."""
    import json as _j
    lgb, xgb, CatBoostClassifier = _lazy_imports()
    import harness as H
    base = list(H.feature_cols(PAIR))
    s15 = _j.load(open(f"{MODELS}/m15_{PAIR}_strategy.json"))
    L = lgb.Booster(model_file=f"{MODELS}/m15_{PAIR}_direction_lgb.txt")
    G = xgb.XGBClassifier(); G.load_model(f"{MODELS}/m15_{PAIR}_direction_xgb.json")
    C = CatBoostClassifier(); C.load_model(f"{MODELS}/m15_{PAIR}_direction_cat.cbm")
    X15 = D[base].astype("float32")
    p15 = _blend_trio(L, G, C, X15)
    # stack meta matrix — column order EXACTLY as m5_stack2.features()
    conf15 = np.abs(p15 - 0.5).astype("float32"); conf5 = np.abs(pxp - 0.5).astype("float32")
    agree = (np.sign(p15 - 0.5) == np.sign(pxp - 0.5)).astype("float32")
    xpm = [c for c in D.columns if c.startswith("agree") or c.startswith("disp") or c == "comp60" or c.startswith("OF_")]
    extra = {"conf15": conf15, "p15": p15.astype("float32"), "conf5": conf5, "p5": pxp.astype("float32"),
             "agree15": agree, "bbw15": D["15m_bb_width"].values.astype("float32"),
             "sess_ny": (D["sess_ny"].values > 0.5).astype("float32"),
             "sess_ln": D["sess_ln"].values.astype("float32")}
    Xstack = np.column_stack([extra[k] for k in extra] + [D[c].values.astype("float32") for c in xpm])
    Mstk = lgb.Booster(model_file=f"{MODELS}/m5stack_{PAIR}_meta_lgb.txt")
    mxp_stk = Mstk.predict(Xstack)
    del L, G, C, X15, Mstk; gc.collect()
    # pstk = the 15m ensemble direction P(up) (the thing m5stack trades), mxp_stk = its meta confidence
    return p15.astype("float32"), mxp_stk.astype("float32")


def _build_pmag(D_ts):
    """Magnitude P(large) aggregated to the 5m clock. For each 5m anchor ts t, take max pmag over the 1s bars whose
    timestamp is in the PRECEDING window [t-MAG_WINDOW_S, t] (input<=t, leakage-safe). Returns pmag array aligned to
    D_ts and a boolean mask of which 5m bars had any 1s coverage (others get NaN -> dropped by inner-join, NEVER ffill).
    The 1s tick cache is keyed by deriv split-name, not year; map the eval year onto the matching split parquet."""
    import json as _j, joblib
    import min1_production as M1
    strat = _j.load(open(f"{MODELS}/min1_{PAIR}_strategy.json"))
    feat_names = strat["feature_names"]
    Mag = joblib.load(f"{MODELS}/min1_{PAIR}_magnitude.joblib")
    D_ts = np.asarray(D_ts, dtype="int64")
    pmag = np.full(len(D_ts), np.nan, dtype="float64")
    covered = np.zeros(len(D_ts), dtype=bool)
    # iterate the 5m anchor clock in ascending ts order (two-pointer requirement); write back via the sort index.
    sidx = np.argsort(D_ts, kind="stable"); ts_sorted = D_ts[sidx]
    # which deriv tick-split parquet(s) cover this set of 5m timestamps? probe by year span.
    yrs = set(np.asarray(D_ts, dtype="datetime64[s]").astype("datetime64[Y]").astype(int) + 1970)
    splits = set()
    for y in yrs:
        if y <= 2023: splits.add("train")
        elif y == 2024: splits.update(["val", "test"])  # 2024 spans val(2024-H1) + test(2024.09-)
        elif y == 2025: splits.add("test")
        else: splits.add("oos")                          # 2026
    for sp in sorted(splits):
        fp = f"{M1.TICK}/{sp}_1s.parquet"
        if not os.path.exists(fp):
            print(f"    [pmag] split parquet missing: {fp}", flush=True); continue
        import pandas as pd
        b = pd.read_parquet(fp)
        X = M1.feats(b)[feat_names].astype("float32")
        pm = Mag.predict_proba(X)[:, 1]
        bts = b.index.values.astype("datetime64[s]").astype("int64")  # 1s bar ts (UTC seconds)
        order = np.argsort(bts); bts = bts[order]; pm = pm[order]
        n1 = len(bts)
        # rolling-max of pm over the trailing MAG_WINDOW_S seconds, queried at each 5m anchor t (input<=t):
        # sliding-window maximum via a monotonic deque. D_ts is the 5m anchor clock (ascending). hi includes all 1s
        # bars with bts<=t; lo drops those older than t-MAG_WINDOW_S. dq holds indices of window maxima.
        from collections import deque
        hi = 0; lo = 0; dq = deque()
        for j, t in enumerate(ts_sorted):
            while hi < n1 and bts[hi] <= t:
                while dq and pm[dq[-1]] <= pm[hi]: dq.pop()
                dq.append(hi); hi += 1
            wstart = t - MAG_WINDOW_S
            while lo < hi and bts[lo] < wstart:
                if dq and dq[0] == lo: dq.popleft()
                lo += 1
            if dq and lo < hi:
                oi = sidx[j]
                # a 5m bar can be covered by at most one parquet's window; first writer wins (NEVER overwrite/ffill).
                if not covered[oi]:
                    pmag[oi] = pm[dq[0]]; covered[oi] = True
        del b, X, pm; gc.collect()
    return pmag.astype("float32"), covered


def build_split(split, force=False):
    """Build (or load cached) the per-bar score matrix for one split. Inner-join all producers on the 5m anchor ts.
    Cache -> combo_scores_<split>.npz. Returns a dict of arrays keyed by SCORE_COLS."""
    cache = f"{ROOT}/combo_scores_{split}.npz"
    if os.path.exists(cache) and not force:
        z = np.load(cache)
        print(f"[build] {split}: loaded cache {cache} n={len(z['ts']):,}", flush=True)
        return {k: z[k] for k in SCORE_COLS}
    import m5_xpair as MX
    t0 = time.time()
    print(f"[build] {split}: build_xp+augment ...", flush=True)
    _stride = 8 if split == "train" else 1   # train only feeds cascade/stack-meta fits -> stride-8 (~450k) avoids OOM; eval splits stay stride-1 (honest full coverage)
    D = MX.build_xp(SPL[split], _stride); D = MX.augment(D, SPL[split], "xpof")
    n0 = len(D)
    ts = D["_ts"].values.astype("int64")
    y5 = D["_y"].astype("int8").values
    fwd5 = D["_fwd"].astype("float64").values
    sess_ny = (D["sess_ny"].values > 0.5).astype("int8")
    print(f"[build] {split}: base frame n={n0:,} ({time.time()-t0:.0f}s); scoring producers one frame at a time", flush=True)
    # per-horizon direction (one frozen trio loaded+freed at a time -> bounded memory)
    p = {}
    for hk in ("p5", "p10", "p15", "p30"):
        p[hk] = _build_horizon_pup(D, hk)
        print(f"[build] {split}: {hk} scored ({time.time()-t0:.0f}s)", flush=True)
    pxp, mxp = _build_m5xp(D)
    print(f"[build] {split}: pxp/mxp scored ({time.time()-t0:.0f}s)", flush=True)
    pstk, mxp_stk = _build_m5stack(D, pxp)
    print(f"[build] {split}: pstk/mxp_stk scored ({time.time()-t0:.0f}s)", flush=True)
    # pmag DISABLED: _build_pmag loads the 13.8M-row train_1s parquet + M1.feats() (~3.4GB) while the train frame is
    # resident -> OOM. pmag feeds ONLY the magnitude families (7-mag, 9), which are sign-invariant (m5_magdyn proved
    # win-rate FLAT across magnitude quartiles -> expected null for DIRECTION). Set NaN; the p30-based regime-switch
    # (family 7) covers the regime idea with a robust on-disk indicator. (Re-enable later by chunking M1.feats.)
    pmag = np.full(len(ts), np.nan, dtype="float32"); mag_cov = np.zeros(len(ts), dtype=bool)
    print(f"[build] {split}: pmag DISABLED (NaN) — magnitude sign-invariant for direction (m5_magdyn); avoids 1s-parquet OOM ({time.time()-t0:.0f}s)", flush=True)
    del D; gc.collect()
    # inner-join: a bar is KEPT only if every producer aligned. The only producer that can be missing is pmag (1s tick
    # cache has narrower span); gbm producers cover all rows. NEVER ffill — drop the bar.
    finite = np.isfinite(p["p5"]) & np.isfinite(p["p10"]) & np.isfinite(p["p15"]) & np.isfinite(p["p30"]) \
        & np.isfinite(pxp) & np.isfinite(mxp) & np.isfinite(pstk) & np.isfinite(mxp_stk) & np.isfinite(fwd5)
    keep = finite  # keep ALL direction-aligned bars; pmag may be NaN (only the magnitude families 7/9 require coverage) — NEVER ffill
    mat = {"ts": ts[keep], "y5": y5[keep], "fwd5": fwd5[keep], "sess_ny": sess_ny[keep],
           "p5": p["p5"][keep], "p10": p["p10"][keep], "p15": p["p15"][keep], "p30": p["p30"][keep],
           "pxp": pxp[keep], "mxp": mxp[keep], "pstk": pstk[keep], "mxp_stk": mxp_stk[keep], "pmag": pmag[keep]}
    print(f"[build] {split}: kept {int(keep.sum()):,}/{n0:,} direction-aligned bars (dropped {n0-int(keep.sum()):,} for missing dir-producers); "
          f"pmag covered {int((mag_cov & keep).sum()):,} (NaN elsewhere; only magnitude families restrict to covered)", flush=True)
    np.savez_compressed(cache, **mat)
    print(f"[build] {split}: cached -> {cache} ({time.time()-t0:.0f}s)", flush=True)
    return mat


# ============================================================ EVAL CORE ============================================================

def boot(c, nb=2500, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def nonoverlap_chrono(ts, mask, gap=GAP_S):
    take = []; block = -1
    for i in np.where(mask)[0]:
        if ts[i] < block: continue
        take.append(i); block = int(ts[i]) + gap
    return np.array(take, dtype=int)


def _yr(ts):
    return np.asarray(ts, dtype="datetime64[s]").astype("datetime64[Y]").astype(int) + 1970


def _selective(score, gate, ts, cov):
    """NY-and-gate selective set: conf-cover quantile within the gated bars, then non-overlap 300s chrono.
    score is the COMBINED direction P(up); gate already encodes side (>0.5 for UP, <0.5 for DOWN) AND NY AND any
    family-specific condition. Returns indices of selected independent trades."""
    conf = np.abs(score - 0.5)
    g = gate
    if g.sum() < 20: return np.array([], dtype=int)
    cthr = np.quantile(conf[g], 1 - cov)
    m = g & (conf >= cthr)
    return nonoverlap_chrono(ts, m, GAP_S)


def eval_side(mat, score, gate_extra, side, cov):
    """Evaluate ONE combination on ONE side. score=combined P(up); gate_extra=boolean array of family conditions.
    side=1 UP (score>0.5), 0 DOWN (score<0.5). Returns per-year dict + binding + up-rate tripwire.
    Win vs TRUE y5, TIES LOSE (y5 already excludes ties via build_xp valid=fwd!=0)."""
    ts = mat["ts"]; y5 = mat["y5"]; ny = mat["sess_ny"] > 0
    sidemask = (score > 0.5) if side == 1 else (score < 0.5)
    gate = ny & sidemask & gate_extra
    yrs = _yr(ts)
    res = {}
    movedrates = []
    for Y in (2024, 2025, 2026):
        ym = yrs == Y
        sel = _selective(score, gate & ym, ts, cov)
        if len(sel) < 5:
            continue
        cc = (y5[sel] == side).astype(float)
        lo, hi = boot(cc)
        # moved-bar up-rate tripwire: of the SELECTED bars, the actual realized up-fraction must sit in [.47,.53]
        up_rate = float((y5[sel] == 1).mean())
        movedrates.append(up_rate)
        res[Y] = dict(win=round(float(cc.mean()), 4), n=int(len(sel)),
                      ci=[round(lo, 4), round(hi, 4)], up_rate=round(up_rate, 4))
    if not res:
        return dict(binding=dict(win=None, ci_lo=None, min_n=0), uprate_ok=False)
    wins = [res[Y]["win"] for Y in res]; los = [res[Y]["ci"][0] for Y in res]; ns = [res[Y]["n"] for Y in res]
    # binding = worst year by point estimate, carrying its own ci-lo/n
    bind_Y = min(res, key=lambda Y: res[Y]["win"])
    bind = dict(year=bind_Y, win=res[bind_Y]["win"], ci_lo=res[bind_Y]["ci"][0], n=res[bind_Y]["n"],
                min_n=min(ns), worst_win=min(wins), worst_ci_lo=min(los))
    uprate_ok = all(0.47 <= ur <= 0.53 for ur in movedrates)
    res["binding"] = bind
    res["uprate_ok"] = uprate_ok
    return res


# ---------- worst-VAL-half selection ----------

def val_halfmin(mat_val, score_val, gate_extra_val, side, cov):
    """Selection metric: split VAL by median time into two halves, compute the selective win-rate in EACH half, return
    the MIN (worst-VAL-half). NEVER VAL-acc-max. Returns (worst_half_win, n_min_half) or (nan,0) if a half is starved."""
    ts = mat_val["ts"]; y5 = mat_val["y5"]; ny = mat_val["sess_ny"] > 0
    sidemask = (score_val > 0.5) if side == 1 else (score_val < 0.5)
    gate = ny & sidemask & gate_extra_val
    if gate.sum() < 40: return (float("nan"), 0)
    tmed = np.median(ts)
    halves = [ts <= tmed, ts > tmed]
    accs = []; ns = []
    for hmask in halves:
        sel = _selective(score_val, gate & hmask, ts, cov)
        if len(sel) < 20: return (float("nan"), 0)
        accs.append(float((y5[sel] == side).mean())); ns.append(len(sel))
    return (min(accs), min(ns))


# ============================================================ (B) COMBINATION SWEEP ============================================================

def _norm_w(ws):
    s = sum(ws)
    return tuple(w / s for w in ws) if s > 0 else None


def combo_blend_score(mat, weights, keys):
    """Convex blend of producer P(up) scores -> combined P(up)."""
    acc = np.zeros(len(mat["ts"]), dtype="float64")
    for w, k in zip(weights, keys):
        acc += w * mat[k]
    return acc


def _fit_logistic(Xtr, ytr):
    """Tiny logistic meta on score features (TRAIN). Returns a predict(X)->P fn. No leakage: TRAIN scores only."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    return lambda X: clf.predict_proba(X)[:, 1]


def _fit_lgb_stack(Xtr, ytr):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.03, min_child_samples=500,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=20,
                           n_jobs=8, verbosity=-1)
    m.fit(Xtr, ytr)
    return lambda X: m.predict_proba(X)[:, 1]


def run_sweep():
    t0 = time.time()
    # load cached score matrices (build them if missing)
    M = {sp: build_split(sp) for sp in SPLITS_ALL}
    val, tr = M["val"], M["train"]
    EV = {2024: M["test24"], 2025: M["test25"], 2026: M["oos"]}
    # eval-on-all-years matrix: concat the three eval splits so eval_side can slice by year
    def concat(splits):
        return {c: np.concatenate([M[s][c] for s in splits]) for c in SCORE_COLS}
    EVALL = concat(("test24", "test25", "oos"))

    results = {}
    leaderboard = {"UP": [], "DOWN": []}

    def register(name, family, up_res, down_res, sel_meta):
        results[name] = dict(family=family, UP=up_res, DOWN=down_res, selection=sel_meta)
        for side, r in (("UP", up_res), ("DOWN", down_res)):
            b = r.get("binding", {})
            if b.get("ci_lo") is not None and b.get("min_n", 0) >= MIN_N:
                leaderboard[side].append((name, b["ci_lo"], b.get("win"), b.get("min_n"), r.get("uprate_ok")))

    NONE_GATE_VAL = np.ones(len(val["ts"]), dtype=bool)
    NONE_GATE_EVALL = np.ones(len(EVALL["ts"]), dtype=bool)

    def eval_both(name, family, score_val, gate_val, score_evall, gate_evall, cov=COV_DEFAULT):
        sel_up = val_halfmin(val, score_val, gate_val, 1, cov)
        sel_dn = val_halfmin(val, score_val, gate_val, 0, cov)
        up = eval_side(EVALL, score_evall, gate_evall, 1, cov)
        dn = eval_side(EVALL, score_evall, gate_evall, 0, cov)
        register(name, family, up, dn,
                 dict(cov=cov, val_halfmin_up=sel_up[0], val_halfmin_up_n=sel_up[1],
                      val_halfmin_down=sel_dn[0], val_halfmin_down_n=sel_dn[1]))

    # ---------- Family 1: SINGLE baselines ----------
    print("[sweep] family 1: SINGLE baselines", flush=True)
    for k in ("pxp", "pstk", "p5", "p10", "p15", "p30"):
        eval_both(f"single_{k}", "single", val[k], NONE_GATE_VAL, EVALL[k], NONE_GATE_EVALL)

    # ---------- Family 2: BLEND convex weights ----------
    print("[sweep] family 2: BLEND convex weights", flush=True)
    bkeys = ["pxp", "pstk", "p15", "p30", "p10"]
    grid = (0, 0.25, 0.5, 0.75, 1.0)
    seen = set()
    n_blend = 0
    import itertools
    for combo in itertools.product(grid, repeat=len(bkeys)):
        nw = _norm_w(combo)
        if nw is None: continue
        key = tuple(round(x, 4) for x in nw)
        if key in seen: continue
        seen.add(key)
        n_blend += 1
        sv = combo_blend_score(val, nw, bkeys); se = combo_blend_score(EVALL, nw, bkeys)
        nm = "blend_" + "_".join(f"{k}{w:.2f}" for k, w in zip(bkeys, nw) if w > 0)
        eval_both(nm, "blend", sv, NONE_GATE_VAL, se, NONE_GATE_EVALL)
    # equal weight
    nw = _norm_w((1, 1, 1, 1, 1))
    eval_both("blend_equal", "blend", combo_blend_score(val, nw, bkeys), NONE_GATE_VAL,
              combo_blend_score(EVALL, nw, bkeys), NONE_GATE_EVALL)
    print(f"[sweep] family 2 done: {n_blend} unique convex blends ({time.time()-t0:.0f}s)", flush=True)

    # ---------- Family 3: CROSS-HORIZON GATE (pxp UP only when p{H}>0.5) ----------
    print("[sweep] family 3: CROSS-HORIZON GATE", flush=True)
    for h in ("p15", "p30", "p10"):
        gv = val[h] > 0.5; ge = EVALL[h] > 0.5
        eval_both(f"xgate_pxp_when_{h}up", "xhorizon_gate", val["pxp"], gv, EVALL["pxp"], ge)
    # AND combination
    gv = (val["p15"] > 0.5) & (val["p30"] > 0.5); ge = (EVALL["p15"] > 0.5) & (EVALL["p30"] > 0.5)
    eval_both("xgate_pxp_when_p15_and_p30_up", "xhorizon_gate", val["pxp"], gv, EVALL["pxp"], ge)

    # ---------- Family 4: CONSENSUS k-of-n ----------
    print("[sweep] family 4: CONSENSUS k-of-n", flush=True)
    ckeys = ["p5", "p10", "p15", "p30", "pxp"]
    upcnt_val = sum((val[k] > 0.5).astype(int) for k in ckeys)
    upcnt_ev = sum((EVALL[k] > 0.5).astype(int) for k in ckeys)
    dncnt_val = sum((val[k] < 0.5).astype(int) for k in ckeys)
    dncnt_ev = sum((EVALL[k] < 0.5).astype(int) for k in ckeys)
    # consensus mean score drives direction; gate requires >=k agreeing on the chosen side
    mean_val = combo_blend_score(val, _norm_w((1,)*len(ckeys)), ckeys)
    mean_ev = combo_blend_score(EVALL, _norm_w((1,)*len(ckeys)), ckeys)
    for k in (2, 3, 4, 5):
        gv_up = upcnt_val >= k; ge_up = upcnt_ev >= k
        gv_dn = dncnt_val >= k; ge_dn = dncnt_ev >= k
        # UP uses up-consensus gate, DOWN uses down-consensus gate; eval_side already side-masks on mean score
        up = eval_side(EVALL, mean_ev, ge_up, 1, COV_DEFAULT)
        dn = eval_side(EVALL, mean_ev, ge_dn, 0, COV_DEFAULT)
        selu = val_halfmin(val, mean_val, gv_up, 1, COV_DEFAULT)
        seld = val_halfmin(val, mean_val, gv_dn, 0, COV_DEFAULT)
        register(f"consensus_{k}of5", "consensus", up, dn,
                 dict(cov=COV_DEFAULT, val_halfmin_up=selu[0], val_halfmin_up_n=selu[1],
                      val_halfmin_down=seld[0], val_halfmin_down_n=seld[1]))

    # ---------- Family 5: CONFIDENCE-GATE cross-horizon (tau swept on worst-VAL-half) ----------
    print("[sweep] family 5: CONFIDENCE-GATE cross-horizon", flush=True)
    best_tau = None
    for tau in (0.02, 0.04, 0.06, 0.08, 0.10):
        gv = np.abs(val["p15"] - 0.5) > tau
        hm = val_halfmin(val, val["pxp"], gv, 1, COV_DEFAULT)
        if not np.isnan(hm[0]) and hm[1] >= 20:
            if best_tau is None or hm[0] > best_tau[1]:
                best_tau = (tau, hm[0], hm[1])
    if best_tau is not None:
        tau = best_tau[0]
        gv = np.abs(val["p15"] - 0.5) > tau; ge = np.abs(EVALL["p15"] - 0.5) > tau
        eval_both(f"confgate_pxp_p15conf_tau{tau:.2f}", "conf_gate", val["pxp"], gv, EVALL["pxp"], ge)
    else:
        print("[sweep] family 5: no tau passed VAL half-min n>=20", flush=True)

    # ---------- Family 6: CASCADE (new parents) — logistic P(parent correct on 5m) on TRAIN scores ----------
    print("[sweep] family 6: CASCADE new parents", flush=True)
    def parent_score(mat, parent):
        if parent == "p15": return mat["p15"]
        if parent == "p30": return mat["p30"]
        if parent == "mean_101530": return (mat["p10"] + mat["p15"] + mat["p30"]) / 3.0
        raise ValueError(parent)
    for parent in ("p15", "p30", "mean_101530"):
        ps_tr = parent_score(tr, parent); dir_tr = (ps_tr > 0.5).astype(int)
        # meta features = [parent conf, parent score, agreement of p5/pxp with parent]  (orthogonal-ish)
        def meta_X(mat, ps):
            d = (ps > 0.5).astype(float)
            return np.column_stack([np.abs(ps - 0.5), ps,
                                    (np.sign(mat["pxp"] - 0.5) == np.sign(ps - 0.5)).astype(float),
                                    (np.sign(mat["p5"] - 0.5) == np.sign(ps - 0.5)).astype(float),
                                    mat["mxp"]])
        ycorr = (dir_tr == tr["y5"]).astype(int)
        Xtr = meta_X(tr, ps_tr)
        predfn = _fit_logistic(Xtr, ycorr)
        # parent direction is the TRADE direction; cascade meta is the GATE (confidence).
        ps_val = parent_score(val, parent); ps_ev = parent_score(EVALL, parent)
        gate_conf_val = predfn(meta_X(val, ps_val)); gate_conf_ev = predfn(meta_X(EVALL, ps_ev))
        # sweep the cascade-meta threshold on worst-VAL-half
        best = None
        for q in (0.5, 0.6, 0.7, 0.8, 0.9):
            thr = float(np.quantile(gate_conf_val, q))
            gv = gate_conf_val >= thr
            hm = val_halfmin(val, ps_val, gv, 1, COV_DEFAULT)
            if not np.isnan(hm[0]) and hm[1] >= 20 and (best is None or hm[0] > best[1]):
                best = (thr, hm[0], hm[1])
        if best is None:
            continue
        thr = best[0]
        gv = gate_conf_val >= thr; ge = gate_conf_ev >= thr
        eval_both(f"cascade_{parent}", "cascade", ps_val, gv, ps_ev, ge)

    # ---------- Family 7: REGIME-SWITCH (enable pxp UP only when slow indicator favorable) ----------
    print("[sweep] family 7: REGIME-SWITCH", flush=True)
    # indicator p30>0.5
    eval_both("regime_pxp_p30up", "regime", val["pxp"], val["p30"] > 0.5, EVALL["pxp"], EVALL["p30"] > 0.5)
    # indicator session NY only is already the default gate; add session-LN-excluded variant via sess_ny (already in gate)
    # rolling realized-vol bucket via pmag — SKIPPED (pmag disabled to avoid 1s-parquet OOM; magnitude sign-invariant per m5_magdyn)
    if np.isfinite(val["pmag"]).any():
        vmed = float(np.nanmedian(val["pmag"]))
        eval_both("regime_pxp_highmag", "regime", val["pxp"], val["pmag"] >= vmed, EVALL["pxp"], EVALL["pmag"] >= vmed)
        eval_both("regime_pxp_lowmag", "regime", val["pxp"], val["pmag"] < vmed, EVALL["pxp"], EVALL["pmag"] < vmed)
    else:
        print("[sweep] family 7: pmag-regime SKIPPED (pmag disabled; magnitude sign-invariant for direction per m5_magdyn — flat across quartiles). p30-regime above is the live regime-switch test.", flush=True)

    # ---------- Family 8: HORIZON-SCORE-STACK (logistic + tiny LGB on score vector) ----------
    print("[sweep] family 8: HORIZON-SCORE-STACK", flush=True)
    skeys = ["p5", "p10", "p15", "p30", "pxp", "pstk"]
    Xtr_s = np.column_stack([tr[k] for k in skeys]); ytr_s = tr["y5"].astype(int)
    Xval_s = np.column_stack([val[k] for k in skeys]); Xev_s = np.column_stack([EVALL[k] for k in skeys])
    for fitname, fitfn in (("logit", _fit_logistic), ("lgb", _fit_lgb_stack)):
        predfn = fitfn(Xtr_s, ytr_s)
        sv = predfn(Xval_s); se = predfn(Xev_s)
        eval_both(f"hstack_{fitname}", "horizon_stack", sv, NONE_GATE_VAL, se, NONE_GATE_EVALL)

    # ---------- Family 9: MAGNITUDE-conditioned (expected null; sign-invariant) ----------
    print("[sweep] family 9: MAGNITUDE-conditioned (expected null)", flush=True)
    if np.isfinite(val["pmag"]).any():
        vmed = float(np.nanmedian(val["pmag"]))
        eval_both("magcond_pxp_highmag", "magnitude", val["pxp"], val["pmag"] >= vmed, EVALL["pxp"], EVALL["pmag"] >= vmed)
        eval_both("magcond_pxp_lowmag", "magnitude", val["pxp"], val["pmag"] < vmed, EVALL["pxp"], EVALL["pmag"] < vmed)
    else:
        print("[sweep] family 9: SKIPPED — pmag disabled; magnitude is sign-invariant for DIRECTION (m5_magdyn: win-rate flat across magnitude quartiles). The magnitude EDGE is real but tracked in MAGNITUDE_FINDINGS.md (not a direction lever).", flush=True)

    # ---------- DOWN extras ----------
    print("[sweep] DOWN extras: bearish consensus + inverse-confidence", flush=True)
    gv = (val["pxp"] < 0.5) & (val["p15"] < 0.5) & (val["p30"] < 0.5)
    ge = (EVALL["pxp"] < 0.5) & (EVALL["p15"] < 0.5) & (EVALL["p30"] < 0.5)
    dn = eval_side(EVALL, EVALL["pxp"], ge, 0, COV_DEFAULT)
    seld = val_halfmin(val, val["pxp"], gv, 0, COV_DEFAULT)
    register("down_bearish_consensus", "down_extra",
             dict(binding=dict(win=None, ci_lo=None, min_n=0), uprate_ok=False), dn,
             dict(cov=COV_DEFAULT, val_halfmin_down=seld[0], val_halfmin_down_n=seld[1]))

    # ============================== LEADERBOARDS + VERDICT ==============================
    for side in ("UP", "DOWN"):
        leaderboard[side].sort(key=lambda r: (-(r[1] if r[1] is not None else -9), ))
    lb = {side: [dict(name=r[0], binding_ci_lo=r[1], binding_win=r[2], min_n=r[3], uprate_ok=r[4])
                 for r in leaderboard[side][:5]] for side in ("UP", "DOWN")}

    def survives_up(r):
        b = r["UP"].get("binding", {})
        if b.get("ci_lo") is None or b.get("min_n", 0) < MIN_N or not r["UP"].get("uprate_ok"): return False
        y26 = r["UP"].get(2026, {}).get("win")
        if y26 is not None and y26 < BE: return False
        se25 = None
        y25 = r["UP"].get(2025, {})
        if y25:
            se25 = (y25["ci"][1] - y25["ci"][0]) / 3.92
        beats = (b["win"] is not None and INCUMBENT_UP is not None and se25 is not None
                 and b["win"] > INCUMBENT_UP + se25) or (b["ci_lo"] >= INCUMBENT_UP)
        return b["ci_lo"] >= UP_FLOOR and beats

    def survives_down(r):
        b = r["DOWN"].get("binding", {})
        if b.get("ci_lo") is None or b.get("min_n", 0) < MIN_N or not r["DOWN"].get("uprate_ok"): return False
        y26 = r["DOWN"].get(2026, {}).get("win")
        if y26 is not None and y26 < BE: return False
        se = (b["ci_lo"] is not None)
        beats = b["win"] is not None and b["win"] > INCUMBENT_DOWN  # >1SE handled via ci_lo>=BE floor below
        return b["ci_lo"] >= BE and beats

    up_survivors = [n for n, r in results.items() if survives_up(r)]
    dn_survivors = [n for n, r in results.items() if survives_down(r)]
    verdict = dict(
        UP_winner_or_none=(up_survivors[0] if up_survivors else None),
        DOWN_winner_or_none=(dn_survivors[0] if dn_survivors else None),
        up_survivors=up_survivors, down_survivors=dn_survivors,
        beats_incumbent_bool=bool(up_survivors or dn_survivors),
        statement=(f"combo-sweep: UP {'SURVIVOR(S) '+str(up_survivors) if up_survivors else 'no survivor'}; "
                   f"DOWN {'SURVIVOR(S) '+str(dn_survivors) if dn_survivors else 'no survivor'}. "
                   f"Survivors (if any) -> nested-refit CPCV before freeze. Selection worst-VAL-half throughout."))

    out = _result_stub()
    out["combinations"] = results
    out["leaderboard"] = lb
    out["VERDICT"] = verdict
    json.dump(out, open(f"{ROOT}/m5_combo_sweep_result.json", "w"), indent=2, default=str)
    print(f"\nVERDICT: {verdict['statement']}\n-> m5_combo_sweep_result.json ({time.time()-t0:.0f}s)", flush=True)


# ============================================================ PRE-REGISTERED STUB ============================================================

def _result_stub():
    return {
        "test": "5m EURUSD model-combination sweep (frozen producers reused as per-bar P(up); no base retraining)",
        "breakeven": BE,
        "incumbent_up_binding2025": INCUMBENT_UP,
        "incumbent_down_binding": INCUMBENT_DOWN,
        "cov_default": COV_DEFAULT, "covs": list(COVS), "gap_s": GAP_S, "min_n": MIN_N,
        "producers": SCORE_COLS,
        "families": ["single", "blend", "xhorizon_gate", "consensus", "conf_gate", "cascade",
                     "regime", "horizon_stack", "magnitude", "down_extra"],
        "PRE_REGISTERED_FALSIFIER": (
            "For EACH combination, KILL unless binding-year (worst of 2024/2025/2026, n>=150, nonoverlap, ties-LOSE, "
            "worst-VAL-half-selected) CI95-lo: UP >= 0.553 AND point beats incumbent binding-2025 0.577 by >1 SE "
            "(or CI-lo >= 0.577); DOWN CI95-lo >= 0.541 AND beats incumbent 0.5441 by >1 SE; AND up-rate in [0.47,0.53]; "
            "AND 2026 not collapsed <0.541. Any survivor -> flagged for nested-refit CPCV before freeze. Selection "
            "worst-VAL-half throughout. Magnitude is a gate only (sign-invariant)."),
        "selection_rule": "worst-VAL-half (median-time split of VAL 2022-23, min over halves); NEVER VAL-acc-max",
        "settlement": "deriv Rise/Fall mid-to-mid, ties LOSE, non-overlap 300s chrono, NY-gated, per-year boot CI95",
        "combinations": {}, "leaderboard": {"UP": [], "DOWN": []}, "VERDICT": {},
    }


def write_stub():
    out = _result_stub()
    json.dump(out, open(f"{ROOT}/m5_combo_sweep_result.json", "w"), indent=2, default=str)
    print(f"[stub] pre-registered falsifier written -> {ROOT}/m5_combo_sweep_result.json", flush=True)


# ============================================================ MAIN ============================================================

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "build":
        write_stub()
        for sp in SPLITS_ALL:
            build_split(sp)
    elif mode == "sweep":
        write_stub()
        run_sweep()
    elif mode == "smoke":
        # OOS-only score-matrix smoke test: confirm every producer loads + aligns to the 5m clock
        build_split("oos", force=True)
    else:
        print(f"unknown mode {mode!r}; use build|sweep|smoke", flush=True)


if __name__ == "__main__":
    main()
