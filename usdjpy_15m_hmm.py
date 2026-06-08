"""USDJPY 15-MINUTE direction LEVER — C1 HIDDEN MARKOV REGIME (Gaussian HMM, CAUSAL FORWARD FILTER).

LEVER (method spec C1): fit a Gaussian HMM (hmmlearn; fall back to GMM, then KMeans if unavailable) on
CAUSAL emission features — trailing return + realized-vol + range-position — with K=3 states. Assign every
bar its FILTERED state via the forward-only posterior P(state_t | obs_1..t) (NO Viterbi, NO forward-BACKWARD
smoother — both leak the future; ported from min1_hmm._forward_filter). On TRAIN, map state -> next-15m
up-rate; the DIRECTION SIGNAL is the state-conditional P(up) of each bar's filtered state. We test it as a
direction predictor on held-out NY: moved-AUC of the signal vs the up/down label, plus a cov3% selective
win-rate via side_eval.

WHY this can be a no-sign mirage: HMM states are expected to gate move SIZE not SIGN (theorem arXiv:2512.15720
— state/volatility gates partition magnitude regimes). If the states carry no directional information the
state-conditional up-rate collapses to ~0.5 for every state and the moved-AUC sits at .50. We RUN the test
faithfully and let the number decide.

EVAL DISCIPLINE (non-negotiable, inherited from usdjpy_15m_base):
  * Features stay causal/continuous; only DECISION rows are restricted to the NY session
    (sessions.session_mask(ts,'ny'), America/New_York 08-17, DST-correct).
  * The EVAL label is ALWAYS the deriv-faithful fixed-15m sign = sign(close[t+15]-close[t]) with ties LOSING
    — exactly what base.build() returns via (y, moved).
  * Per held-out year (test24/test25/oos): moved-AUC of the lever signal on NY moved bars + cov3% win-rate
    via side_eval (nonoverlap gap=900 built in). Sanity tripwire: NY moved up-rate ~0.47-0.53.

The HMM is fit on TRAIN only; the state->up-rate map is learned on TRAIN only; both frozen before any
held-out read. The cov3% confidence threshold is the global 97th pct of |signal-0.5| on the VAL NY signal,
frozen before held-out.

FALSIFIER (pre-registered in the result JSON BEFORE reading held-out):
  KILL if held-out moved-AUC <= 0.51 in the (test24,test25,oos) median  (states are vol regimes, no sign).
  SURVIVE only if held-out moved-AUC exceeds the incumbent ~.539 in >=2 years, OR the NY cov3% win-rate
  CI95-lower clears breakeven 0.541 in >=2 years.

Self-contained. Run: ~/binary-algo-venv/bin/python usdjpy_15m_hmm.py [stride]
  stride defaults to 6 (RAM/runtime budget: <~3GB, <~10min). Emission feats / K / cov overridable by env.
"""
import os, sys, json, time, numpy as np

import usdjpy_15m_base as B   # build/side_eval/boot/nonoverlap_chrono/SPL/BE/FEATS
from sklearn.metrics import roc_auc_score
import sessions

KEY      = "hmm"
RESULT   = f"usdjpy_15m_{KEY}_result.json"
BE       = B.BE                                   # 0.541
INCUMBENT_AUC = 0.539                             # certified NY own-pair GBM signal AUC

K        = int(os.environ.get("HMM_K", "3"))
COVTYPE  = os.environ.get("HMM_COV", "full")
SEED     = int(os.environ.get("HMM_SEED", "0"))
# CAUSAL emission features (all in the 239 base feats, all trailing => no look-ahead):
#   trailing return + realized vol + range-position, at the native 15m clock.
EMIT     = os.environ.get("HMM_EMIT", "15m_ret_6,15m_rv_12,15m_rangepos_24").split(",")
COV3     = 0.03                                   # selective coverage for the win-rate read

def _argint(i, default):
    return int(sys.argv[i]) if len(sys.argv) > i and str(sys.argv[i]).isdigit() else default
TR_STRIDE = _argint(1, 6)


# ---------------------------------------------------------------------------
# Standardize on TRAIN moments only.
# ---------------------------------------------------------------------------
def std_fit(Z):
    m = np.nanmean(Z, 0); s = np.nanstd(Z, 0) + 1e-9
    return m, s

def std_apply(Z, m, s):
    return np.nan_to_num((Z - m) / s, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# CAUSAL FORWARD FILTER — filtered posterior gamma_f[t,k]=P(state_t=k|obs_1..t).
# Forward-only (NO backward pass, NO Viterbi). Ported from min1_hmm._forward_filter.
# ---------------------------------------------------------------------------
def forward_filter_hmm(model, Z):
    from scipy.special import logsumexp
    fl = model._compute_log_likelihood(Z)                       # (T,K) emission log-lik
    logA  = np.log(model.transmat_ + 1e-300)
    logpi = np.log(model.startprob_ + 1e-300)
    T, Kk = fl.shape; la = np.empty((T, Kk))
    la[0] = logpi + fl[0]
    for t in range(1, T):
        la[t] = fl[t] + logsumexp(la[t - 1][:, None] + logA, axis=0)
    return np.exp(la - logsumexp(la, axis=1, keepdims=True))     # (T,K) normalized filtered posterior


class _FallbackHMM:
    """Causal-filter-compatible fallback when hmmlearn is unavailable.

    Builds K states with a GMM (preferred) or KMeans, then runs a hand-rolled HMM forward filter using
    GMM/empirical Gaussian emissions and a transition matrix estimated from the TRAIN hard-state sequence.
    Still forward-only: forward_filter() implements the same causal recursion as forward_filter_hmm.
    """
    def __init__(self, K, seed):
        self.K = K; self.seed = seed; self.kind = None

    def fit(self, Z):
        try:
            from sklearn.mixture import GaussianMixture
            g = GaussianMixture(n_components=self.K, covariance_type="full",
                                random_state=self.seed, max_iter=200)
            g.fit(Z); self.kind = "gmm"; self._g = g
            hard = g.predict(Z)
            self.means_ = g.means_
        except Exception:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=self.K, random_state=self.seed, n_init=10)
            km.fit(Z); self.kind = "kmeans"; self._km = km
            hard = km.labels_
            self.means_ = km.cluster_centers_
        # per-state Gaussian emission params (diagonal) from TRAIN
        self._mu = np.zeros((self.K, Z.shape[1])); self._sd = np.ones((self.K, Z.shape[1]))
        self.startprob_ = np.full(self.K, 1.0 / self.K)
        for k in range(self.K):
            m = hard == k
            if m.sum() > 1:
                self._mu[k] = Z[m].mean(0); self._sd[k] = Z[m].std(0) + 1e-6
        # transition matrix from TRAIN hard sequence (empirical, smoothed)
        A = np.ones((self.K, self.K))
        for a, b in zip(hard[:-1], hard[1:]):
            A[a, b] += 1.0
        self.transmat_ = A / A.sum(1, keepdims=True)
        self.converged = True
        return self

    def _log_emit(self, Z):
        # diagonal-Gaussian log-likelihood per state, (T,K)
        T = Z.shape[0]; out = np.empty((T, self.K))
        for k in range(self.K):
            mu = self._mu[k]; sd = self._sd[k]
            out[:, k] = (-0.5 * (((Z - mu) / sd) ** 2).sum(1)
                         - np.log(sd).sum() - 0.5 * Z.shape[1] * np.log(2 * np.pi))
        return out

    def forward_filter(self, Z):
        from scipy.special import logsumexp
        fl = self._log_emit(Z)
        logA  = np.log(self.transmat_ + 1e-300)
        logpi = np.log(self.startprob_ + 1e-300)
        T, Kk = fl.shape; la = np.empty((T, Kk))
        la[0] = logpi + fl[0]
        for t in range(1, T):
            la[t] = fl[t] + logsumexp(la[t - 1][:, None] + logA, axis=0)
        return np.exp(la - logsumexp(la, axis=1, keepdims=True))


def fit_state_model(Zfit):
    """Try hmmlearn GaussianHMM; fall back to GMM/KMeans HMM. Returns (model, filter_fn, backend)."""
    try:
        from hmmlearn.hmm import GaussianHMM
        m = GaussianHMM(n_components=K, covariance_type=COVTYPE, n_iter=50, random_state=SEED, tol=1e-3)
        m.fit(Zfit)
        return m, (lambda Z: forward_filter_hmm(m, Z)), f"hmmlearn.GaussianHMM(cov={COVTYPE})"
    except Exception as e:
        print(f"[hmm] hmmlearn unavailable/failed ({e!r}); falling back to GMM/KMeans HMM", flush=True)
        m = _FallbackHMM(K, SEED).fit(Zfit)
        return m, (lambda Z: m.forward_filter(Z)), f"fallback.{m.kind}"


# ---------------------------------------------------------------------------
def emit_matrix(X):
    """Pull the causal emission columns out of the build() feature frame -> standardizable float array."""
    return X[EMIT].astype("float64").values


def main():
    t0 = time.time()

    # ---- PRE-REGISTER falsifier BEFORE any held-out read ----
    res = {
        "key": "USDJPY.15m.ny",
        "lever": ("C1 Gaussian HMM regime (K=%d, cov=%s) on CAUSAL emission feats %s; "
                  "filtered FORWARD-ONLY posterior -> hard state; signal = TRAIN state-conditional next-15m "
                  "up-rate. Decision rows = NY session; eval label = deriv-faithful fixed-15m sign (ties lose)."
                  % (K, COVTYPE, EMIT)),
        "falsifier": {
            "registered_utc": "pre-OOS",
            "KILL_if": ("held-out moved-AUC <= 0.51 in the (test24,test25,oos) median "
                        "(states are vol regimes, carry no sign)"),
            "SURVIVE_if": ("held-out moved-AUC > %.3f (incumbent NY own-pair GBM) in >=2 years, "
                           "OR NY cov%.0f%% win-rate CI95-lower clears %.3f in >=2 years" % (INCUMBENT_AUC, COV3*100, BE)),
            "rationale": ("HMM/vol-state gates partition move SIZE not SIGN (arXiv:2512.15720). If the states "
                          "have directional content the state up-rate departs from 0.5 and moved-AUC>.51; the "
                          "test decides. EURUSD neural+spectral DIRECTION sweep at every tf was ALL KILLED.")},
        "tr_stride": TR_STRIDE, "K": K, "cov": COVTYPE, "emit": EMIT, "cov3": COV3,
        "incumbent_auc": INCUMBENT_AUC,
    }
    json.dump(res, open(RESULT, "w"), indent=2)

    # ---- build TRAIN (strided) + held-out years. build() returns 239 feats incl. EMIT cols ----
    for c in EMIT:
        if c not in B.FEATS:
            raise SystemExit(f"emission feat {c!r} not in base FEATS (239) — fix HMM_EMIT")

    Xtr, ytr, mtr, tstr = B.build(B.SPL["train"], TR_STRIDE)
    Ztr = emit_matrix(Xtr)
    mu, sd = std_fit(Ztr[mtr])                         # standardize on TRAIN moved rows
    Zfit = std_apply(Ztr[mtr], mu, sd)                 # fit HMM on the SAME standardization
    print(f"[hmm] train rows={len(ytr):,} moved={int(mtr.sum()):,} emit={EMIT} build={time.time()-t0:.0f}s", flush=True)

    model, ffilter, backend = fit_state_model(Zfit)
    res["backend"] = backend
    print(f"[hmm] state model = {backend}  fit={time.time()-t0:.0f}s", flush=True)

    # ---- TRAIN state-conditional next-15m up-rate (the LEVER signal map). Filtered (causal) hard state. ----
    g_tr = ffilter(std_apply(Ztr, mu, sd))             # (T,K) filtered posterior over ALL train rows
    hard_tr = g_tr.argmax(1)
    state_uprate = np.full(K, 0.5)
    profile = {}
    for k in range(K):
        msk = (hard_tr == k) & mtr                     # moved train bars in state k
        n = int(msk.sum())
        pup = float(ytr[msk].mean()) if n else 0.5
        state_uprate[k] = pup
        profile[str(k)] = {"train_moved_n": n, "train_share": float((hard_tr == k).mean()),
                           "train_uprate": pup}
        print(f"    state {k}: train_share={(hard_tr==k).mean():.3f} moved_n={n} train_P(up)={pup:.4f}", flush=True)
    res["state_profile"] = profile
    res["state_uprate"] = [float(x) for x in state_uprate]

    def signal_of(X):
        """Lever direction signal in [0,1]: the TRAIN state-conditional up-rate of the bar's filtered state."""
        g = ffilter(std_apply(emit_matrix(X), mu, sd))
        return state_uprate[g.argmax(1)]

    # ---- VAL NY signal -> freeze the cov3% confidence threshold (global, on VAL NY) ----
    Xva, yva, mva, tsva = B.build(B.SPL["val"])
    nyva = sessions.session_mask(tsva, "ny")
    pva = signal_of(Xva)
    val_ny_moved = mva & nyva
    if val_ny_moved.sum() >= 20:
        val_auc = float(roc_auc_score(yva[val_ny_moved], pva[val_ny_moved]))
    else:
        val_auc = float("nan")
    confva = np.abs(pva[nyva] - 0.5)
    THR = float(np.quantile(confva, 1 - COV3)) if len(confva) else 0.0
    res["val_or_signal_auc"] = val_auc
    res["cov3_conf_thr"] = THR
    res["val_ny_moved_uprate"] = float(yva[val_ny_moved].mean()) if val_ny_moved.sum() else float("nan")
    print(f"[hmm] VAL NY moved-AUC={val_auc:.4f}  cov{COV3:.0%} conf_thr={THR:.5f}  "
          f"val_ny_uprate={res['val_ny_moved_uprate']:.4f}  {time.time()-t0:.0f}s", flush=True)

    # ---- held-out per year: moved-AUC (NY moved bars) + cov3% selective win-rate via side_eval ----
    res["years"] = {}
    for w in ("test24", "test25", "oos"):
        Xw, yw, mw, tsw = B.build(B.SPL[w])
        ny = sessions.session_mask(tsw, "ny")
        pr = signal_of(Xw)

        ny_moved = mw & ny
        up_rate = float(yw[ny_moved].mean()) if ny_moved.sum() else float("nan")
        if ny_moved.sum() >= 20 and len(np.unique(yw[ny_moved])) > 1:
            auc = float(roc_auc_score(yw[ny_moved], pr[ny_moved]))
        else:
            auc = float("nan")

        # selective win-rate at cov3% on NY DECISION rows only (features causal; decision restricted to NY).
        # Mask non-NY rows out of the candidate pool by collapsing their signal to 0.5 (no confidence).
        pr_ny = np.where(ny, pr, 0.5)
        ge = B.side_eval(pr_ny, yw, mw, tsw, THR)
        comb = ge["COMBINED"] if ge else {"n": 0, "wr": float("nan"), "ci": [float("nan")] * 2}

        res["years"][w] = {
            "auc": auc,
            "moved_up_rate": up_rate,
            "tripwire_ok": bool(0.47 <= up_rate <= 0.53) if np.isfinite(up_rate) else False,
            "cov3_wr": comb["wr"], "cov3_n": comb["n"], "cov3_ci": comb["ci"],
            "cov3_sides": {s: {"n": ge[s]["n"], "wr": ge[s]["wr"], "ci": ge[s]["ci"]}
                           for s in ("UP", "DOWN")} if ge else {},
            "beats_incumbent_auc": bool(np.isfinite(auc) and auc > INCUMBENT_AUC),
        }
        print(f"=== {w} === NY moved-AUC={auc:.4f} up-rate={up_rate:.4f} (tripwire_ok={res['years'][w]['tripwire_ok']}) "
              f"| cov{COV3:.0%}: n{comb['n']} wr={comb['wr']:.4f} CI[{comb['ci'][0]:.3f},{comb['ci'][1]:.3f}]", flush=True)

    # ---- VERDICT (apply pre-registered falsifier) ----
    yrs = ("test24", "test25", "oos")
    aucs = [res["years"][w]["auc"] for w in yrs if np.isfinite(res["years"][w]["auc"])]
    med_auc = float(np.median(aucs)) if aucs else float("nan")
    beats_auc_years = [w for w in yrs if res["years"][w]["beats_incumbent_auc"]]
    wr_clears_years = [w for w in yrs
                       if np.isfinite(res["years"][w]["cov3_ci"][0]) and res["years"][w]["cov3_ci"][0] >= BE]
    beats_base_auc = len(beats_auc_years) >= 2
    survives_wr = len(wr_clears_years) >= 2
    killed = (not (beats_base_auc or survives_wr)) or (np.isfinite(med_auc) and med_auc <= 0.51)

    res["median_heldout_auc"] = med_auc
    res["verdict"] = {
        "KILLED": bool(killed),
        "beats_base_auc": bool(beats_base_auc),
        "beats_incumbent_auc_years": beats_auc_years,
        "cov3_wr_CIlo_clears_BE_years": wr_clears_years,
        "median_heldout_auc": med_auc,
        "note": ("HMM regime as a 15m DIRECTION lever on USDJPY NY. SURVIVE needs moved-AUC>%.3f in >=2 yrs "
                 "OR cov3%% WR CI-lo>=%.3f in >=2 yrs. Expectation (arXiv:2512.15720): vol-state gates carry "
                 "SIZE not SIGN -> median AUC ~.50, falsifier fires." % (INCUMBENT_AUC, BE)),
    }
    json.dump(res, open(RESULT, "w"), indent=2)
    print(f"\n[hmm] VERDICT: {'KILLED' if killed else 'SURVIVED'}  median NY moved-AUC={med_auc:.4f} "
          f"(beats-incumbent yrs={beats_auc_years}; cov3-clears yrs={wr_clears_years}) -> {RESULT}  "
          f"total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
