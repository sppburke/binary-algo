"""USDJPY 15m DIRECTION lever — C3 RMT eigen-residual reversion (cross-pair USD-basket mode removal).

KEY: USDJPY.15m.ny    METHOD: C3 RMT eigen-residual    INCUMBENT to beat: NY own-pair GBM AUC ~.539 / WR ~.58-.60

THESIS (C3): the 7 USD-majors share a dominant common factor = the USD basket. On the rolling correlation
matrix of their 15m-decision-bar returns, Marchenko-Pastur separates that market mode (the leading eigenvalue,
which sits far above the MP upper edge lambda+) from the idiosyncratic noise bulk. Remove the leading eigen-mode
from USDJPY's standardized return; the leftover residual is USDJPY's pair-specific deviation from the USD
basket. A reversion bet says the deviation snaps back -> predict UP if residual < 0 (USDJPY sits below its
basket-implied level, so expect it to rise).

DERIV-FAITHFUL DISCIPLINE (mirrors usdjpy_15m_base.build):
  * EVAL label = sign(close[t+15]-close[t]); ties (move==0) LOSE; requires ts[t+15]-ts[t]==900 (15 clean 60s
    steps, no gap). This is the deriv Rise/Fall 15-minute binary at the deriv-FX floor expiry.
  * DECISION rows restricted to the NY session (sessions.session_mask(ts,'ny'), DST-correct 08-17 NY).
  * FEATURES (the RMT residual) stay CAUSAL: residual at bar t uses ONLY bar-t returns of the 7 majors
    (close[t] vs close[t-1]), observed at the close of bar t, to predict the t->t+15 move. The standardization
    moments (mu,sd), the rolling-correlation eigenvectors, and the residual std are all FIT ON TRAIN ONLY,
    then applied frozen out-of-sample. No look-ahead.
  * Cross-pair alignment = INNER JOIN on the USDJPY decision index with NO ffill (7-pair retention 99.95% on
    USDJPY bars, so this does not manufacture fake-flat bars).
  * Non-overlapping trades via nonoverlap_chrono gap=900 (built into side_eval); selection by |residual|.

EVAL (per held-out year test24/test25/oos):
  * moved-AUC of the lever signal (reversion score = -residual) vs the up/down label, NY MOVED bars only.
  * selective win-rate at cov3% via side_eval (COMBINED/UP/DOWN), CI95.
  * up-rate tripwire (must be ~0.47-0.53 on NY moved bars, else flag the mirage).

PRE-REGISTERED FALSIFIER (written BEFORE held-out read): KILL if signal/val AUC <= 0.51. SURVIVE only if a
held-out moved-AUC exceeds ~.539 (the incumbent NY own-pair GBM) OR NY cov3% win-rate CI-lower clears 0.541
in >=2 years.

SIGN-INVARIANCE NOTE: state/complexity/volatility gates (incl. RMT) are theorized to gate move SIZE not SIGN
(arXiv:2512.15720). We RUN the direction test faithfully and let the AUC<=.51 falsifier decide — no skip.

Usage: ~/binary-algo-venv/bin/python usdjpy_15m_rmt.py
"""
import os, sys, json, time, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

# import the template (label/eval discipline) and the session masker
import usdjpy_15m_base as B
from sessions import session_mask

KEY      = "rmt"
PAIR     = "USDJPY"
PAIRS    = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
TGT_IDX  = PAIRS.index(PAIR)
HOR, STEP, GAP, BE = B.HOR, B.STEP, B.GAP, B.BE     # 15, 60, 900, 0.541
FEAT     = B.FEAT
SPL      = B.SPL                                    # train 2012-21 / val 2022-23 / test24 / test25 / oos 2026
RESULT   = f"/home/sean/git/binary-algo/usdjpy_15m_{KEY}_result.json"
INCUMBENT_AUC = 0.539                               # certified NY own-pair GBM signal AUC


# ---------------------------------------------------------------------------
# close-only cross-pair loader (memory-light: never touches the 239 feats)
# ---------------------------------------------------------------------------
def load_panel(years):
    """Inner-join 7-pair close panel on the USDJPY decision clock (NO ffill). Returns (DataFrame[PAIRS],
    ts_epoch_s aligned to its index). Index is the common 60s bars across all 7 majors."""
    closes = {}
    for p in PAIRS:
        parts = []
        for y in years:
            fp = f"{FEAT}/{p}_{y}.parquet"
            if os.path.exists(fp):
                parts.append(pd.read_parquet(fp, columns=["close"])["close"])
        s = pd.concat(parts)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        closes[p] = s
    mat = pd.DataFrame(closes).dropna()             # INNER JOIN, no ffill
    mat = mat[PAIRS]
    ts = mat.index.values.astype("datetime64[s]").astype("int64")
    return mat, ts


# ---------------------------------------------------------------------------
# RMT fit on TRAIN ONLY: standardization moments + leading eigen-mode (USD basket) removal operator
# ---------------------------------------------------------------------------
def fit_rmt(mat_tr):
    """Fit the Marchenko-Pastur eigen-residual operator on TRAIN returns ONLY.

    Standardize bar-t log returns by TRAIN per-pair (mu,sd) so the covariance == correlation matrix.
    eigh -> descending eigenvalues. MP upper edge lambda+ = (1+sqrt(N/T))^2 (sigma^2=1 on a correlation matrix).
    The C3 spec removes the LEADING eigen-mode (the market/USD-basket factor, eigenvalue >> lambda+); the
    USDJPY residual after projecting that one mode out is the pair-specific deviation. We standardize the
    residual by its TRAIN std so |residual| is a comparable conviction across years."""
    R = np.log(mat_tr).diff().dropna()
    mu = R.mean(0).values
    sd = R.std(0, ddof=0).values + 1e-12
    Z = ((R.values - mu) / sd)
    T, N = Z.shape
    C = np.corrcoef(Z, rowvar=False)                # NxN correlation matrix
    w, V = np.linalg.eigh(C)                         # ascending
    order = np.argsort(w)[::-1]
    w = w[order]; V = V[:, order]                    # descending
    q = N / T
    lam_plus = (1 + np.sqrt(q)) ** 2                 # MP upper edge
    # leading eigen-mode (rank-1 projector onto the top eigenvector = the USD-basket / market factor)
    v1 = V[:, 0]                                     # leading eigenvector (N,)
    P1 = np.outer(v1, v1)                            # rank-1 projector onto the market mode
    # implied USDJPY std-move from the market mode = (P1 Z^T) row for USDJPY = (P1[TGT,:] . Z_t)
    proj_row = P1[TGT_IDX, :]                         # 1xN
    implied_tr = Z @ proj_row
    resid_tr = Z[:, TGT_IDX] - implied_tr
    resid_std = float(resid_tr.std(ddof=0)) + 1e-12
    sig_above = int((w > lam_plus).sum())
    return dict(mu=mu, sd=sd, proj_row=proj_row, resid_std=resid_std,
                lam_plus=float(lam_plus), lead_eig=float(w[0]), eigvals=w.tolist(),
                k_above=sig_above, q=float(q), T=int(T), N=int(N),
                lead_loadings=dict(zip(PAIRS, v1.tolist())))


# ---------------------------------------------------------------------------
# build the lever signal aligned to a deriv-faithful label on the USDJPY decision clock
# ---------------------------------------------------------------------------
def build_signal(mat, ts, rmt):
    """For an evaluation panel: CAUSAL standardized RMT residual at bar t, the deriv-faithful 15m label,
    moved/contiguity flags, and NY decision mask. Returns dict of aligned arrays over ALL panel rows.

    resid[t] uses bar-t returns -> applied frozen TRAIN operator. label/moved/contig re-derived exactly as
    usdjpy_15m_base.build (sign(close[t+15]-close[t]); ties LOSE; require ts[t+15]-ts[t]==900)."""
    R = np.log(mat).diff()                           # bar-t return at index t (NaN at row 0)
    Z = (R.values - rmt["mu"]) / rmt["sd"]
    implied = Z @ rmt["proj_row"]
    resid = (Z[:, TGT_IDX] - implied) / rmt["resid_std"]   # standardized residual (signal core)

    close = mat[PAIR].values.astype(float)
    n = len(close)
    contig = np.zeros(n, bool); contig[:n - HOR] = (ts[HOR:] - ts[:-HOR]) == GAP
    fr = np.full(n, np.nan); fr[:n - HOR] = close[HOR:] / close[:-HOR] - 1.0
    label = (fr > 0).astype(int)                     # 1=up; ties(fr==0) & down -> 0
    moved = contig & np.isfinite(fr) & (fr != 0.0)
    valid = contig & np.isfinite(fr) & np.isfinite(resid)
    ny = session_mask(ts, "ny")                       # DST-correct NY decision mask

    # lever as a probability-like score in [0,1] so side_eval's |pr-0.5| conviction works:
    # reversion bet -> predict UP when resid<0. score = sigmoid(-resid). conf grows with |resid|.
    pr = 1.0 / (1.0 + np.exp(resid))                  # = sigmoid(-resid): resid<0 -> pr>0.5 (UP bet)
    return dict(resid=resid, pr=pr, label=label, moved=moved, valid=valid, ny=ny, fr=fr)


def auc_moved_ny(sig):
    """moved-AUC of the lever (reversion score -resid) vs up-label on NY MOVED bars only."""
    m = sig["valid"] & sig["moved"] & sig["ny"]
    if m.sum() < 50:
        return float("nan"), int(m.sum()), float("nan")
    y = sig["label"][m]
    score = -sig["resid"][m]                          # higher score => more UP conviction
    up_rate = float(y.mean())
    if len(np.unique(y)) < 2:
        return float("nan"), int(m.sum()), up_rate
    return float(roc_auc_score(y, score)), int(m.sum()), up_rate


def main():
    t0 = time.time()

    # ---- PRE-REGISTER FALSIFIER (written BEFORE any held-out read) ----
    res = {
        "key": "USDJPY.15m.ny",
        "lever": "C3 RMT eigen-residual reversion: 7 USD-major close-only return panel inner-joined on the "
                 "USDJPY 15m NY decision clock; standardize on TRAIN; rolling correlation matrix; "
                 "Marchenko-Pastur isolates the leading eigen-mode (USD basket / market factor, eigenvalue "
                 ">> lambda+); remove it; USDJPY residual -> reversion direction signal (predict UP if "
                 "residual<0). Operator (mu,sd,leading eigenvector,resid-std) fit TRAIN-only, applied frozen.",
        "settlement": "deriv-faithful label sign(close[t+15]-close[t]); ties LOSE; require ts[t+15]-ts[t]==900; "
                      "NY decision rows only; gap=900 nonoverlap_chrono via side_eval; BE=0.541.",
        "splits": SPL,
        "incumbent_auc": INCUMBENT_AUC,
        "falsifier": {
            "registered_utc": "pre-OOS",
            "KILL_if": "signal/val moved-AUC <= 0.51 (direction falsifier)",
            "SURVIVE_if": "a held-out moved-AUC exceeds ~0.539 (incumbent NY own-pair GBM) OR NY cov3% "
                          "win-rate CI95-lower clears 0.541 in >=2 years",
            "rationale": "RMT is a state/volatility gate; arXiv:2512.15720 predicts it gates move SIZE not "
                         "SIGN, so direction-AUC is expected at/near 0.50. Run the direction test faithfully "
                         "and let AUC<=0.51 decide."},
        "tripwire": "NY moved up-rate must be ~0.47-0.53 (else flag the leakage/mirage).",
    }
    json.dump(res, open(RESULT, "w"), indent=2)

    # ---- TRAIN: fit MP operator on the cross-pair panel ----
    mat_tr, _ = load_panel(SPL["train"])
    rmt = fit_rmt(mat_tr)
    print(f"[rmt15m] TRAIN inner bars={len(mat_tr):,}  N={rmt['N']} T={rmt['T']}  q={rmt['q']:.2e}  "
          f"build={time.time()-t0:.0f}s", flush=True)
    print(f"[rmt15m] MP lambda+={rmt['lam_plus']:.4f}  leading eig={rmt['lead_eig']:.3f}  "
          f"#eig>lambda+={rmt['k_above']}", flush=True)
    print(f"[rmt15m] eigenvalues(desc)= " + " ".join(f"{v:.3f}" for v in rmt['eigvals']), flush=True)
    print(f"[rmt15m] leading eigenvector loadings (USD-basket mode)= "
          + " ".join(f"{p}={l:+.3f}" for p, l in rmt['lead_loadings'].items()), flush=True)
    res["rmt"] = {"lambda_plus": rmt["lam_plus"], "leading_eig": rmt["lead_eig"], "k_above": rmt["k_above"],
                  "eigvals_desc": rmt["eigvals"], "leading_eigenvector": rmt["lead_loadings"],
                  "q_NoverT": rmt["q"], "T": rmt["T"], "N": rmt["N"]}

    # ---- VAL signal AUC (NY moved) -> the pre-held-out signal_auc for the verdict gate ----
    mat_va, ts_va = load_panel(SPL["val"])
    sig_va = build_signal(mat_va, ts_va, rmt)
    val_auc, val_n, val_up = auc_moved_ny(sig_va)
    res["val_or_signal_auc"] = val_auc
    print(f"[rmt15m] VAL NY moved-AUC={val_auc:.4f}  (n={val_n}, up-rate={val_up:.4f})  "
          f"{time.time()-t0:.0f}s", flush=True)

    # ---- held-out per year: moved-AUC + cov3% win-rate (COMBINED/UP/DOWN) ----
    res["years"] = {}
    for w in ("test24", "test25", "oos"):
        mat_w, ts_w = load_panel(SPL[w])
        sig = build_signal(mat_w, ts_w, rmt)
        auc, n_moved, up_rate = auc_moved_ny(sig)

        # restrict eval rows to NY (features stay causal/continuous; decision rows = NY)
        ny = sig["ny"]
        pr_ny = sig["pr"][ny]
        y_ny = sig["label"][ny]
        moved_ny = (sig["moved"] & sig["valid"])[ny]
        ts_ny = ts_w[ny]

        # cov3% selective win-rate via the template's side_eval (gap=900 nonoverlap built in)
        conf = np.abs(pr_ny - 0.5)
        thr3 = float(np.quantile(conf[np.isfinite(conf)], 0.97)) if np.isfinite(conf).any() else 1.0
        gate3 = B.side_eval(pr_ny, y_ny, moved_ny, ts_ny, thr3)
        cc = B.covcurve(pr_ny, y_ny, moved_ny, ts_ny)

        g = gate3["COMBINED"] if gate3 else {"n": 0, "wr": float("nan"), "ci": [float("nan")] * 2}
        res["years"][w] = {
            "auc": auc, "n_moved_ny": n_moved, "moved_up_rate": up_rate,
            "tripwire_ok": bool(0.47 <= up_rate <= 0.53) if np.isfinite(up_rate) else False,
            "cov3_wr": g["wr"], "cov3_n": g["n"], "cov3_ci": g["ci"],
            "cov3_UP": gate3["UP"] if gate3 else None, "cov3_DOWN": gate3["DOWN"] if gate3 else None,
            "cov3_thr": thr3, "covcurve": cc,
        }
        print(f"=== {w} === NY moved-AUC={auc:.4f} up-rate={up_rate:.4f} (n={n_moved}) | "
              f"cov3% COMB n{g['n']} wr={g['wr']:.4f} CI[{g['ci'][0]:.3f},{g['ci'][1]:.3f}]", flush=True)
        if gate3:
            for side in ("UP", "DOWN"):
                s = gate3[side]
                print(f"        {side}: n{s['n']} wr={s['wr']:.4f} CI[{s['ci'][0]:.3f},{s['ci'][1]:.3f}]",
                      flush=True)

    # ---- VERDICT (apply pre-registered falsifier) ----
    aucs = {w: res["years"][w]["auc"] for w in ("test24", "test25", "oos")}
    auc_beats = [w for w, a in aucs.items() if np.isfinite(a) and a > INCUMBENT_AUC]
    wr_clears = [w for w in ("test24", "test25", "oos")
                 if np.isfinite(res["years"][w]["cov3_ci"][0]) and res["years"][w]["cov3_ci"][0] >= BE]
    kill_dir = (not np.isfinite(val_auc)) or (val_auc <= 0.51)
    survives = (len(auc_beats) >= 1) or (len(wr_clears) >= 2)
    killed = kill_dir or (not survives)
    res["verdict"] = {
        "KILLED": bool(killed),
        "beats_base_auc": bool(len(auc_beats) >= 1),
        "auc_beats_incumbent_years": auc_beats,
        "cov3_wr_CIlo_clears_BE_years": wr_clears,
        "val_signal_auc_le_051": bool(kill_dir),
        "note": (f"C3 RMT eigen-residual reversion lever. VAL signal AUC={val_auc:.4f}; "
                 f"held-out NY moved-AUC {{ {', '.join(f'{w}:{aucs[w]:.4f}' for w in aucs)} }}; "
                 f"AUC>incumbent(.539) in {auc_beats or 'none'}; cov3% WR CIlo>=BE in {wr_clears or 'none'}. "
                 + ("KILLED: " + ("direction falsifier AUC<=0.51" if kill_dir else
                    "no held-out AUC beats .539 and <2 years clear BE win-rate")
                    if killed else "SURVIVES the pre-registered falsifier.")),
    }
    json.dump(res, open(RESULT, "w"), indent=2)
    print(f"\n[rmt15m] VERDICT: {'KILLED' if killed else 'SURVIVED'}  "
          f"(val_auc={val_auc:.4f}; AUC-beats={auc_beats}; WR-clears={wr_clears}) -> {RESULT}  "
          f"total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
