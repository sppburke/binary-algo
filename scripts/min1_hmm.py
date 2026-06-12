"""1-MIN HIDDEN MARKOV REGIME MODEL (user-requested) — does a learned latent-regime switch break the 60s direction wall?

Fits a Gaussian HMM on CAUSAL emission features over the train split (2021-2023, same as the min1 child), then assigns each
held-out bar a regime using the FILTERED (forward-only) posterior P(state_t | obs_1..t) — NOT Viterbi or forward-BACKWARD
(both leak future data). Tests three honest uses of the regime, all on the deriv-faithful 60s book:
  (U1) REGIME GATE on the existing child: trade child direction only in states where it is accurate AND stable on VAL.
  (U2) PER-STATE engine switch: in each state pick momentum(sign ret300) vs reversion(-sign ret300) by VAL behaviour.
  (U3) REGIME-CONDITIONED meta: state posteriors as extra features to a worst-VAL-half-stable 'avoid-losers' meta on the child.
DISCIPLINE: HMM + per-state decisions fit on train/VAL only; frozen; judged per window {2024(test 09-12), 2025(test), 2026(oos)}
separately with non-overlap + CI95 + ties-LOSE. Reuses min1_production._load/_blend/feats/prep/nonoverlap_chrono UNCHANGED.

Config via env: HMM_K (states, default 3), HMM_EMIT (csv emission feats), HMM_COV (full|diag), HMM_SEED.
"""
import os, sys, json, time, numpy as np, pandas as pd
from scipy.special import logsumexp
from hmmlearn.hmm import GaussianHMM
import min1_production as M

BREAKEVEN = 0.541
K       = int(os.environ.get("HMM_K", "3"))
COVTYPE = os.environ.get("HMM_COV", "full")
SEED    = int(os.environ.get("HMM_SEED", "0"))
SUB     = int(os.environ.get("HMM_SUB", "30"))   # HMM runs on ~SUB-second grid (every SUB-th valid bar); regimes persist >> 1s
# emission features (must be columns produced by M.feats): signed returns + vol + flow so states CAN be directional, not only vol
EMIT = os.environ.get("HMM_EMIT", "ret60,ret300,rv300,rv900,imb_ema20,micro_dev").split(",")

def _forward_filter(model, Z):
    """Filtered posterior gamma_f[t,k]=P(state_t=k|obs_1..t), forward-only (causal; NO backward/Viterbi). Z standardized.
    Returns the (T,K) normalized filtered posterior matrix."""
    fl = model._compute_log_likelihood(Z)                      # (T,K) emission log-lik
    logA = np.log(model.transmat_ + 1e-300); logpi = np.log(model.startprob_ + 1e-300)
    T, Kk = fl.shape; la = np.empty((T, Kk))
    la[0] = logpi + fl[0]
    for t in range(1, T):
        la[t] = fl[t] + logsumexp(la[t - 1][:, None] + logA, axis=0)
    return np.exp(la - logsumexp(la, axis=1, keepdims=True))

def causal_states(model, d, mu, sd):
    """Assign every 1s bar the filtered hard state AND soft posterior of the most-recent ~SUB-second grid bar (causal)."""
    nb = len(d["y"]); vid = np.where(d["valid"])[0][::SUB]
    if len(vid) < 5:
        return np.full(nb, -1, np.int16), np.zeros((nb, K), "float32")
    gg = _forward_filter(model, std_apply(d["Xe"][vid], mu, sd))     # (G,K) per grid bar
    pos = np.searchsorted(vid, np.arange(nb), side="right") - 1      # most-recent grid bar <= i
    hard = np.full(nb, -1, np.int16); soft = np.zeros((nb, K), "float32"); ok = pos >= 0
    hard[ok] = gg[pos[ok]].argmax(1).astype(np.int16); soft[ok] = gg[pos[ok]]
    return hard, soft

def slim(sp, p, L, G, Cc, S):
    b = M.load_split(sp); X, y, mag, valid, ts, idx = M.prep(b)
    pr = M._blend(p, L, G, Cc, S, X)
    Xe = X[EMIT].astype("float32").values
    d = dict(p_up=pr.astype("float32"), ret300=X["ret300"].values.astype("float32"),
             bbw=X["bbw1800"].values.astype("float32"), rel=X["rel_ratio"].values.astype("float32"),
             y=y.astype("int8"), mag=mag.astype("float32"), valid=valid, ts=ts,
             year=idx.year.values.astype("int16"), Xe=Xe)
    del b, X
    return d

def std_fit(Xe, valid):
    m = np.nanmean(Xe[valid], 0); s = np.nanstd(Xe[valid], 0) + 1e-9
    return m, s
def std_apply(Xe, m, s):
    Z = (Xe - m) / s
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

def winacc(d, cand, bet):
    sel = M.nonoverlap_chrono(d["ts"], cand)
    if len(sel) == 0: return np.array([]), np.array([], int)
    pred_up = (bet[sel] > 0).astype(int)
    correct = ((pred_up == d["y"][sel]) & (d["mag"][sel] > 0)).astype(float)
    return correct, sel

def boot(c, nb=4000, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def main():
    t0 = time.time()
    p, L, G, Cc, S = M._load()
    qb = p["bbw1800_q67"]; rqp = p["rel_tighten"]; cthr = p["conf_thr"]
    print(f"[hmm] K={K} cov={COVTYPE} emit={EMIT} child_auc={p.get('val_auc_inregime'):.3f} {time.time()-t0:.0f}s", flush=True)
    TR = slim("train", p, L, G, Cc, S); print(f"[hmm] train n={len(TR['y'])} {time.time()-t0:.0f}s", flush=True)
    mu, sd = std_fit(TR["Xe"], TR["valid"])
    # fit HMM on the SAME ~SUB-second grid used for causal filtering (consistent transition timescale)
    vid_tr = np.where(TR["valid"])[0][::SUB]
    Zfit = std_apply(TR["Xe"][vid_tr], mu, sd)
    model = GaussianHMM(n_components=K, covariance_type=COVTYPE, n_iter=50, random_state=SEED, tol=1e-3)
    model.fit(Zfit)
    print(f"[hmm] HMM fit on {len(Zfit)} grid-bars (SUB={SUB}); converged={model.monitor_.converged} {time.time()-t0:.0f}s", flush=True)
    # characterize states on train (causal filtered, mapped to 1s bars)
    str_, _ = causal_states(model, TR, mu, sd)
    print("[hmm] train state profile (share | P(up) | child-acc-when-traded):", flush=True)
    state_child_acc = {}; state_revmom = {}
    for k in range(K):
        msk = (str_ == k) & TR["valid"] & (TR["mag"] > 0)
        share = msk.mean()
        pup = TR["y"][msk].mean() if msk.sum() else float("nan")
        childpred = (TR["p_up"][msk] > 0.5).astype(int)
        cacc = (childpred == TR["y"][msk]).mean() if msk.sum() else float("nan")
        # reversion vs momentum payoff in this state (which sign of ret300 wins over 60s)
        mom = (np.sign(TR["ret300"][msk]) > 0).astype(int)
        mom_acc = (mom == TR["y"][msk]).mean() if msk.sum() else float("nan")
        state_child_acc[k] = cacc; state_revmom[k] = "mom" if mom_acc >= 0.5 else "rev"
        print(f"    state {k}: share={share:.3f} P(up)={pup:.3f} child_acc={cacc:.3f} mom_acc={mom_acc:.3f} -> engine={state_revmom[k]}", flush=True)

    VA = slim("val", p, L, G, Cc, S); TE = slim("test", p, L, G, Cc, S); OO = slim("oos", p, L, G, Cc, S)
    print(f"[hmm] val/test/oos ready {time.time()-t0:.0f}s", flush=True)
    for D in (VA, TE, OO):
        D["state"], D["gamma"] = causal_states(model, D, mu, sd)
    te24 = TE["year"] == 2024; te25 = TE["year"] == 2025
    vord = np.argsort(VA["ts"]); h = len(vord) // 2
    vh1 = np.zeros(len(VA["ts"]), bool); vh1[vord[:h]] = True; vh2 = ~vh1

    def conf_gate(d, cf=cthr, comp=True):
        g = d["valid"] & (np.abs(d["p_up"] - 0.5) >= cf)
        if comp: g &= (d["bbw"] <= qb) & (d["rel"] >= rqp)
        return g

    def child_dir(d): return np.sign(d["p_up"] - 0.5)
    def eng_bet(d):
        bet = np.zeros(len(d["y"]), "float32")
        for k in range(K):
            mk = d["state"] == k
            bet[mk] = (np.sign(d["ret300"]) if state_revmom[k] == "mom" else -np.sign(d["ret300"]))[mk]
        return bet

    # ---- U1: regime gate on child. Pick the set of states (by VAL worst-half child-acc) where child is trusted ----
    # rank states by min(VAL-half child acc); include states cumulatively; choose by worst-VAL-half stability
    def state_va_acc(states_on, engine):
        cand = conf_gate(VA) & np.isin(VA["state"], states_on)
        bet = child_dir(VA) if engine == "child" else eng_bet(VA)
        c1, s1 = winacc(VA, cand & vh1, bet); c2, s2 = winacc(VA, cand & vh2, bet)
        if len(c1) < 20 or len(c2) < 20: return (float("nan"), 0)
        return (min(c1.mean(), c2.mean()), len(c1) + len(c2))
    print("\n[hmm] per-window eval of regime uses (deriv-faithful, non-overlap, ties lose):", flush=True)
    results = {}
    for engine in ("child", "engine"):
        # greedily order states by train child/engine acc, then pick the cumulative subset maximizing VAL worst-half
        order = sorted(range(K), key=lambda k: -(state_child_acc[k] if engine == "child" else 1.0))
        best = None
        for j in range(1, K + 1):
            ss = order[:j]; vhm, nv = state_va_acc(ss, engine)
            if np.isnan(vhm): continue
            if best is None or vhm > best[0]: best = (vhm, ss, nv)
        if best is None: continue
        vhm, ss, nv = best
        betf = child_dir if engine == "child" else (lambda d: eng_bet(d))
        def w(D, ym=None):
            cand = conf_gate(D) & np.isin(D["state"], ss)
            if ym is not None: cand = cand & ym
            return winacc(D, cand, betf(D) if engine == "child" else eng_bet(D))
        c24, _ = w(TE, te24); c25, _ = w(TE, te25); c26, _ = w(OO)
        a = lambda c: (c.mean() if len(c) else float("nan"))
        lo26, hi26 = boot(c26)
        fl = min(a(c24), a(c25), a(c26))
        results[engine] = dict(states=ss, vhmin=vhm, a24=a(c24), a25=a(c25), a26=a(c26), n24=len(c24), n25=len(c25), n26=len(c26), ci26=[lo26, hi26], floor=fl)
        print(f"  U-{engine:6s} states={ss} VALworsthalf={vhm:.3f} -> 2024 {a(c24):.3f}(n{len(c24)}) / 2025 {a(c25):.3f}(n{len(c25)}) / "
              f"2026 {a(c26):.3f}(n{len(c26)}) CI[{lo26:.3f},{hi26:.3f}]  FLOOR={fl:.3f}", flush=True)
        print(f"     [KILL] 2025={a(c25):.3f} (need >=0.56 to beat wall, >0.65 goal); breakeven={BREAKEVEN}", flush=True)

    # ---- U3: HMM soft posteriors as features into an 'avoid-losers' meta on the frozen child (research/09 idea #7; untried) ----
    import lightgbm as lgb
    def metaX(D):
        conf = np.abs(D["p_up"] - 0.5).astype("float32")
        return np.column_stack([conf, D["bbw"], D["rel"], D["gamma"]])
    gVA = conf_gate(VA); yVA = ((VA["p_up"] > 0.5).astype(int) == VA["y"]).astype(int)
    Mm = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.02, num_leaves=15, min_child_samples=500,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=20, n_estimators=300, n_jobs=20, verbosity=-1)
    Mm.fit(metaX(VA)[gVA], yVA[gVA])
    SV = Mm.predict_proba(metaX(VA))[:, 1]
    def u3sel(D, thr, ym=None):
        s = Mm.predict_proba(metaX(D))[:, 1]; cand = conf_gate(D) & (s >= thr)
        if ym is not None: cand = cand & ym
        return winacc(D, cand, child_dir(D))
    def u3half(thr):
        accs = []
        for vhx in (vh1, vh2):
            c, _ = u3sel(VA, thr, vhx)
            if len(c) < 20: return float("nan")
            accs.append(c.mean())
        return min(accs)
    grid = [float(np.quantile(SV[gVA], q)) for q in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)]
    bestu3 = None
    for thr in grid:
        hm = u3half(thr)
        if np.isnan(hm): continue
        if bestu3 is None or hm > bestu3[0]: bestu3 = (hm, thr)
    if bestu3 is not None:
        hm, thr = bestu3
        c24, _ = u3sel(TE, thr, te24); c25, _ = u3sel(TE, thr, te25); c26, _ = u3sel(OO, thr)
        a = lambda c: (c.mean() if len(c) else float("nan")); lo26, hi26 = boot(c26)
        fl = min(a(c24), a(c25), a(c26))
        results["meta_gamma"] = dict(thr=thr, vhmin=hm, a24=a(c24), a25=a(c25), a26=a(c26),
                                     n24=len(c24), n25=len(c25), n26=len(c26), ci26=[lo26, hi26], floor=fl)
        imp = dict(zip(["conf", "bbw", "rel"] + [f"gamma{k}" for k in range(K)], Mm.feature_importances_.tolist()))
        print(f"  U-meta_gamma thr={thr:.3f} VALworsthalf={hm:.3f} -> 2024 {a(c24):.3f}(n{len(c24)}) / 2025 {a(c25):.3f}(n{len(c25)}) / "
              f"2026 {a(c26):.3f}(n{len(c26)}) CI[{lo26:.3f},{hi26:.3f}]  FLOOR={fl:.3f}", flush=True)
        print(f"     [KILL] 2025={a(c25):.3f}; meta feature_importance={imp}", flush=True)
    json.dump({k: {kk: (vv if not isinstance(vv, np.ndarray) else vv.tolist()) for kk, vv in v.items()} for k, v in results.items()},
              open("models/min1_hmm_summary.json", "w"), indent=2, default=float)
    print(f"[hmm] DONE {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
