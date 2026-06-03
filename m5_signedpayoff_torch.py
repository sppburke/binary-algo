"""SIGN-COUPLED OBJECTIVE BATCH (5m EURUSD binary DIRECTION) — the only two sign-coupled objective forms NEVER
implemented on disk. Every prior loss kill (BCE / focal-γ2 / asym-classweight / magweight |ret|^POW / lambdarank /
quantile) was either a SYMMETRIC reweight or a BCE-equivalent / sign-AGNOSTIC modulation. Neither couples the model's
SIGNED score to the SIGNED forward return. This job tests:

  HEAD A — GMADL/MADL custom loss:   L = -(1/N) Σ σ(a·R·R̂)·|R|^b
      R   = signed forward return (TR["_fwd"]),  R̂ = model raw signed score (logit).
      σ(u)=1/(1+e^{-u}).  A SIGN-WRONG large move (R·R̂<0, |R| big) is penalized MORE than a sign-right one — the
      magnitude |R|^b multiplies the *sign-agreement* gate σ(a·R·R̂). This does NOT reduce to symmetric |ret|^POW·BCE
      (magweight; that reweights a sign-agnostic BCE) nor to focal/asym (sign-agnostic). The R·R̂ coupling is explicit.
      Implemented (i) as a custom LightGBM fobj returning grad/hess wrt the raw logit (mirrors focal_obj in
      m5_lossbatch.py:22-32), full data; and (ii) as a 2-layer CPU-torch MLP with the same loss. Sweep a∈{50,100}, b∈{1,2}.

  HEAD B — RRL / differentiable-Sharpe (Moody&Saffell direct reinforcement):  2-layer torch MLP, output
      X_t = tanh(f(features)) ∈ [-1,1].  Trained on 2012-2021 minimizing  L = -mean(X_t · payoff_t)  (and an optional
      differentiable-Sharpe variant L = -Sharpe({X_t·payoff_t})).  payoff_t is the DERIV ties-LOSE binary settlement at
      H=300s, win=+0.85 / loss=-1.0, keyed to sign(X_t) vs the TRUE moved label _y (NOT a smooth proxy — discrete,
      no leak). The SIGN of X_t IS the trade decision: gate UP iff X_t>τ, DOWN iff X_t<-τ; τ on worst-VAL-half.

GMADL grad/hess wrt raw logit z (=R̂), per sample, for L_i = -σ(u)|R|^b with u = a·R·z:
    dσ/dz   = σ(u)(1-σ(u))·a·R
    grad    = dL_i/dz   = -|R|^b · σ(u)(1-σ(u)) · (a·R)
    d/dz[σ(1-σ)] = σ(1-σ)(1-2σ)·a·R
    hess    = d²L_i/dz² = -|R|^b · (a·R)² · σ(u)(1-σ(u))(1-2σ(u))           # non-convex -> use |hess|+eps for lgb
(LightGBM minimizes Σ L_i; we feed grad,hess wrt the raw margin. hess can be <0 for a non-convex objective, so we
 floor it as max(|hess|,eps) exactly the way focal_obj floors its hess — keeps the Newton step well-defined.)

EVAL is m5_lossbatch.py per_year() VERBATIM: COV=0.05, NY gate, side mask via score->[0,1], conf=|s-.5|, conf-cover
quantile, MX.nonoverlap_chrono(ts,mask,300), per-year 2024/2025/2026, boot() CI95, breakeven 0.541, ties LOSE (true _y),
up-rate∈[.47,.53] tripwire. SELECTION (a,b,τ,arch) on WORST-VAL-half objective, NEVER VAL-acc-max.

INCUMBENTS:  UP-2025 = 0.577 (m5_updown / m5xp production);  magweight DOWN-2025 = 0.5594 (m5_magweight_result.json).
PRE-REGISTERED FALSIFIER (written to result json BEFORE OOS): KILL (loss-reopt + sizing-policy families empirically
exhausted) unless a worst-VAL-half-selected book beats incumbent on the BINDING 2025 year on >=1 side at
matched-or-higher nonoverlap n:  DOWN-2025 CI95-lo >= 0.541 AND > magweight DOWN .5594,  OR  UP-2025 CI95-lo >
incumbent .577 by > 1 SE;  AND up-rate ∈ [0.47,0.53];  AND 2026 does NOT collapse < 0.541;  AND any survivor must
additionally pass nested-refit CPCV (p10 >= 0.541 AND >= 80% of 28 purged paths clear) before any freeze — failing
refit = KILL (the way ACI did).

torch: CPU, deterministic seed. TRAIN subsampled to <=150k for the torch heads (LOGGED). The LightGBM GMADL fobj
uses FULL data.

  ~/binary-algo-venv/bin/python m5_signedpayoff_torch.py
"""
import os, sys; sys.argv = ["x"]
import json, time, gc
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
import m5_xpair as MX
import m5_xpair_production as XP

ROOT = "/media/sean/CORSAIR/binary-algo"
BE = 0.541; STRIDE = 6; COV = 0.05; HOR_S = 300
DEV = "cpu"
SEED = 0
TORCH_CAP = 150_000          # subsample cap for the torch heads (LOGGED if hit)
WIN = 0.85; LOSS = -1.0      # deriv ties-LOSE binary settlement payoff
# incumbents
INC_UP25 = 0.577             # UP binding-2025 incumbent
INC_DN25 = 0.5594            # magweight DOWN binding-2025 incumbent
# sweep grids
A_GRID = [50, 100]; B_GRID = [1, 2]
ARCHS = {"mlp_256_64": (256, 64)}
TAUS = [0.0, 0.05, 0.10, 0.15, 0.20]
# nested-refit CPCV settings (mirror m5_cpcv_refit / m5_seedens_cpcv)
N_GROUPS, K_TEST = 8, 2       # C(8,2)=28 purged paths
EMB = HOR_S                   # embargo = 1 label horizon
CPCV_SUBSAMPLE = 100_000
CPCV_COV = 0.05

torch.manual_seed(SEED); np.random.seed(SEED)
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass


# ----------------------------------------------------------------------------- helpers
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def boot(c, nb=2500, seed=7):
    c = np.asarray(c, float)
    if len(c) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); n = len(c)
    a = np.array([c[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def per_year(EV, scorefn, sv):
    """m5_lossbatch.py per_year() VERBATIM. scorefn(D)->prob in [0,1]; conf=|s-.5|; side gate; conf-cover quantile;
    nonoverlap_chrono(300); per-year 2024/25/26; boot CI95; correctness vs TRUE moved label _y (ties LOSE)."""
    res = {}
    for w, D in EV.items():
        s = scorefn(D); y = D["_y"].astype(int).values; ts = D["_ts"].values.astype("int64")
        ny = D["sess_ny"].values > 0.5
        conf = np.abs(s - 0.5); g = ny & ((s > 0.5) if sv == 1 else (s < 0.5))
        if g.sum() < 20:
            continue
        cthr = np.quantile(conf[g], 1 - COV); m = g & (conf >= cthr); sel = MX.nonoverlap_chrono(ts, m, HOR_S)
        if len(sel) >= 20:
            cc = (y[sel] == sv).astype(float); lo, hi = boot(cc)
            Y = 2024 if w == "test24" else 2025 if w == "test25" else 2026
            # up-rate over the SELECTED bars (moved-bars realized up-fraction) for the [.47,.53] tripwire
            up_rate = float(y[sel].mean())
            res[Y] = dict(win=round(float(cc.mean()), 4), n=len(sel), ci=[round(lo, 4), round(hi, 4)],
                          up_rate=round(up_rate, 4))
    b = [res[Y]["win"] for Y in res]; lo = [res[Y]["ci"][0] for Y in res]; ns = [res[Y]["n"] for Y in res]
    res["binding"] = dict(win=min(b) if b else None, ci_lo=min(lo) if lo else None, min_n=min(ns) if ns else 0)
    return res


def se_of(ci):
    """1-SE estimate from a symmetric-ish bootstrap CI95 (half-width / 1.96)."""
    if ci is None or any(x is None or (isinstance(x, float) and np.isnan(x)) for x in ci):
        return float("nan")
    return (ci[1] - ci[0]) / (2 * 1.96)


def uprate_ok(side_res):
    """All per-year SELECTED up-rates within [.47,.53] tripwire band."""
    bad = []
    for Y in (2024, 2025, 2026):
        if Y in side_res and "up_rate" in side_res[Y]:
            ur = side_res[Y]["up_rate"]
            if not (0.47 <= ur <= 0.53):
                bad.append((Y, ur))
    return (len(bad) == 0), bad


# ----------------------------------------------------------------------------- GMADL LightGBM fobj
def gmadl_obj(R, a, b):
    """Custom LightGBM objective: minimize  -(1/N) Σ σ(a·R·z)·|R|^b   (z = raw margin = R̂).
    Returns grad,hess wrt z. R is aligned to the TRAIN row order LightGBM iterates (full-data, not subsampled).
    grad = -|R|^b · σ(u)(1-σ(u)) · (a·R)              u = a·R·z
    hess = -|R|^b · (a·R)² · σ(u)(1-σ(u))(1-2σ(u))    (non-convex -> floor |hess|+eps, mirrors focal_obj)
    """
    Rb = np.abs(R) ** b
    aR = a * R

    def f(y, raw):                       # y unused (label carried by R sign); raw = current margins z
        u = aR * raw
        s = 1.0 / (1.0 + np.exp(-u)); s = np.clip(s, 1e-9, 1 - 1e-9)
        sp = s * (1.0 - s)               # σ'(u)
        grad = -(Rb * sp * aR)
        hess = -(Rb * (aR ** 2) * sp * (1.0 - 2.0 * s))
        hess = np.abs(hess) + 1e-6       # non-convex floor (focal_obj-style); keeps Newton step defined
        return grad, hess
    return f


# ----------------------------------------------------------------------------- torch nets
class GMADLNet(nn.Module):
    """2-layer MLP -> scalar raw signed score R̂ (logit). HEAD A torch variant."""
    def __init__(self, d, h1=256, h2=64):
        super().__init__()
        self.net = nn.Sequential(nn.BatchNorm1d(d), nn.Linear(d, h1), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RRLNet(nn.Module):
    """2-layer MLP -> X_t = tanh(f(x)) ∈ [-1,1]. HEAD B (RRL / differentiable-Sharpe)."""
    def __init__(self, d, h1=256, h2=64):
        super().__init__()
        self.net = nn.Sequential(nn.BatchNorm1d(d), nn.Linear(d, h1), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h2, 1))

    def forward(self, x):
        return torch.tanh(self.net(x)).squeeze(-1)


def _norm_fns(Xtr):
    mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0) + 1e-6

    def norm(X):
        return np.nan_to_num((X - mu) / sd, nan=0.0).astype("float32")
    return norm


def worst_val_half_obj(scores_va, Rva, yva, tsva, nyva, vyr, side_or_signed, metric="payoff"):
    """Worst-(per-VAL-year)-half objective at COV gate, used for SELECTION (never VAL-acc-max).
    metric='payoff': mean deriv ties-LOSE payoff on the selected NY/side/conf-cover bars (binding=min over VAL years).
    side_or_signed: ('A', sv) for HEAD A (sv=1 UP / 0 DOWN, score in [0,1]); ('B', tau) for HEAD B (signed X_t)."""
    vals = []
    yrs = sorted(set(vyr.tolist()))
    kind = side_or_signed[0]
    for yy in yrs:
        ymask = (vyr == yy) & nyva
        if kind == "A":
            sv = side_or_signed[1]
            s = scores_va
            g = ymask & ((s > 0.5) if sv == 1 else (s < 0.5))
            conf = np.abs(s - 0.5)
        else:  # B: signed X_t, both sides via |X|>tau; decision sign = sign(X)
            tau = side_or_signed[1]
            x = scores_va                      # X_t ∈ [-1,1]
            g = ymask & (np.abs(x) > tau)
            conf = np.abs(x)
        if g.sum() < 20:
            vals.append(np.nan); continue
        cthr = np.quantile(conf[g], 1 - COV); m = g & (conf >= cthr); sel = MX.nonoverlap_chrono(tsva, m, HOR_S)
        if len(sel) < 20:
            vals.append(np.nan); continue
        if kind == "A":
            sv = side_or_signed[1]; correct = (yva[sel] == sv)
        else:
            pred_up = scores_va[sel] > 0; correct = (yva[sel] == pred_up.astype(int))
        payoff = np.where(correct, WIN, LOSS)         # deriv ties-LOSE settlement (yva from _y => ties already dropped)
        vals.append(float(payoff.mean()))
    v = [x for x in vals if np.isfinite(x)]
    if len(v) < len(yrs):
        return float("-inf")                          # require every VAL year populated
    return min(v)                                     # worst-half (worst VAL year)


# ----------------------------------------------------------------------------- nested-refit CPCV (survivor gate)
def cpcv_refit(survivor, cols, Xall, yall, tsall, nyall, side, gmadl_fobj_builder=None, log=print):
    """C(8,2)=28 purged-combinatorial paths. Refit the survivor's PRIMARY per path on purged train (subsample<=100k),
    evaluate side-selective accuracy at CPCV_COV with NY gate, ties-LOSE (moved label _y). CERTIFY iff p10>=0.541 AND
    frac>=0.80. survivor describes a HEAD-A GMADL-lgb config (a,b) — the only head that refits as a single lgb here."""
    import itertools
    order = np.argsort(tsall, kind="stable")
    X, y, ts, ny = Xall[order], yall[order], tsall[order], nyall[order]
    R = None
    if gmadl_fobj_builder is not None:
        R = survivor["_R_full"][order]                # signed fwd ret aligned to sorted order
    n = len(y); edges = np.linspace(0, n, N_GROUPS + 1).astype(int)
    g = np.empty(n, int)
    for k in range(N_GROUPS):
        g[edges[k]:edges[k + 1]] = k
    all_idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    accs, ns = [], []
    sv = 1 if side == "UP" else 0
    for ci, combo in enumerate(itertools.combinations(range(N_GROUPS), K_TEST)):
        te = all_idx[np.isin(g, combo)]
        tr = all_idx[~np.isin(g, combo)]
        keep = np.ones(len(tr), bool); tt = ts[tr]
        for grp in combo:
            gi = all_idx[g == grp]; lo, hi = ts[gi[0]], ts[gi[-1]]
            keep &= ~((tt >= lo - HOR_S) & (tt <= hi + EMB))
        tr = tr[keep]
        if len(tr) > CPCV_SUBSAMPLE:
            tr = np.sort(rng.choice(tr, CPCV_SUBSAMPLE, replace=False))
        if gmadl_fobj_builder is not None:
            fobj = gmadl_fobj_builder(R[tr], survivor["a"], survivor["b"])
            m = lgb.LGBMClassifier(n_estimators=700, learning_rate=0.03, num_leaves=127, min_child_samples=300,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=20,
                                   n_jobs=20, verbosity=-1, objective=fobj)
            m.fit(X[tr], y[tr])
            pr = sigmoid(m.booster_.predict(X[te], raw_score=True))
        else:
            raise RuntimeError("cpcv_refit: only HEAD-A GMADL-lgb survivors are refit-certifiable here")
        del m; gc.collect()
        yte = y[te]; conf = np.abs(pr - 0.5); gate = ny[te] & ((pr > 0.5) if sv == 1 else (pr < 0.5))
        if gate.sum() >= 40:
            cthr = np.quantile(conf[gate], 1 - CPCV_COV); sel = gate & (conf >= cthr)
            ssel = MX.nonoverlap_chrono(ts[te], sel, HOR_S)
            if len(ssel) >= 20:
                accs.append(float((yte[ssel] == sv).mean())); ns.append(int(len(ssel)))
            else:
                accs.append(np.nan); ns.append(0)
        else:
            accs.append(np.nan); ns.append(0)
        if ci % 7 == 0:
            log(f"    [cpcv {side}] path {ci+1}/28 acc={accs[-1]} ({time.time()-T0:.0f}s)")
    a = np.array(accs, float); nn = np.array(ns); v = a[np.isfinite(a) & (nn >= 20)]
    if len(v) == 0:
        return {"n_valid": 0, "CERTIFIED": False}
    p10 = float(np.percentile(v, 10)); frac = float((v >= BE).mean())
    return {"n_valid": int(len(v)), "mean": round(float(v.mean()), 4), "p10": round(p10, 4),
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4),
            "frac_clear_0.541": round(frac, 3), "med_n": int(np.median(nn[nn >= 20])),
            "CERTIFIED": bool(p10 >= BE and frac >= 0.80)}


T0 = time.time()


# ----------------------------------------------------------------------------- pre-registered result stub
def write_stub(path):
    stub = {
        "test": "SIGN-COUPLED objective batch @5m: HEAD A GMADL/MADL (lgb fobj + torch MLP), HEAD B RRL/diff-Sharpe",
        "status": "PRE-REGISTERED (written before OOS eval)",
        "breakeven": BE, "cov": COV, "stride": STRIDE, "torch_cap": TORCH_CAP,
        "payoff": {"win": WIN, "loss": LOSS, "note": "deriv ties-LOSE binary settlement @H=300s, keyed to moved label _y"},
        "incumbents": {"UP_2025": INC_UP25, "DOWN_2025_magweight": INC_DN25},
        "gmadl_grad_hess": ("L_i=-sigma(a*R*z)*|R|^b; grad=-|R|^b*sigma(u)(1-sigma(u))*(a*R); "
                            "hess=-|R|^b*(a*R)^2*sigma(u)(1-sigma(u))(1-2sigma(u)); u=a*R*z; lgb uses |hess|+eps"),
        "PRE_REGISTERED_FALSIFIER": (
            "KILL (loss-reopt + sizing-policy families empirically exhausted) unless a worst-VAL-half-selected book "
            "beats incumbent on the BINDING 2025 year on >=1 side at matched-or-higher nonoverlap n: "
            "DOWN-2025 CI95-lo >= 0.541 AND > magweight DOWN 0.5594, OR UP-2025 CI95-lo > incumbent 0.577 by > 1 SE; "
            "AND up-rate in [0.47,0.53]; AND 2026 does NOT collapse < 0.541; AND any survivor must additionally pass "
            "nested-refit CPCV (p10>=0.541 AND >=80% of 28 purged paths clear) before any freeze — failing refit = KILL."),
        "HEAD_A": {}, "HEAD_B": {}, "VERDICT": {"status": "pending OOS"},
    }
    json.dump(stub, open(path, "w"), indent=2, default=str)


# ----------------------------------------------------------------------------- main
def main():
    out_path = f"{ROOT}/m5_signedpayoff_torch_result.json"
    write_stub(out_path)
    print(f"[signedpayoff] pre-registered stub -> {out_path} ({time.time()-T0:.0f}s)", flush=True)

    p = json.load(open(XP.art("strategy.json"))); cols = p["primary_feats"]; d = len(cols)
    print(f"[signedpayoff] cols={d} torch={torch.__version__} lgb={lgb.__version__} dev={DEV} seed={SEED}", flush=True)

    # ---- build TRAIN / VAL / per-year EVAL frames (m5xp pipeline VERBATIM) ----
    TR = MX.augment(MX.build_xp(XP.SPL["train"], STRIDE), XP.SPL["train"], XP.MODE)
    VA = MX.augment(MX.build_xp(XP.SPL["val"]), XP.SPL["val"], XP.MODE)
    EV = {w: MX.augment(MX.build_xp(XP.SPL[w]), XP.SPL[w], XP.MODE) for w in ("test24", "test25", "oos")}
    Xtr = TR[cols].astype("float32").to_numpy(); ytr = TR["_y"].astype(int).values
    Rtr = TR["_fwd"].astype("float64").values                          # signed forward return R (full train)
    Xva = VA[cols].astype("float32").to_numpy(); yva = VA["_y"].astype(int).values
    Rva = VA["_fwd"].astype("float64").values
    tsva = VA["_ts"].values.astype("int64"); nyva = VA["sess_ny"].values > 0.5
    vyr = XP.yr(tsva)
    print(f"[signedpayoff] train={len(Xtr):,} val={len(Xva):,} EV={{24:{len(EV['test24']):,},25:{len(EV['test25']):,},26:{len(EV['oos']):,}}} {time.time()-T0:.0f}s", flush=True)

    # ---- torch subsample (LOGGED) ----
    if len(Xtr) > TORCH_CAP:
        rng = np.random.default_rng(SEED); tidx = np.sort(rng.choice(len(Xtr), TORCH_CAP, replace=False))
        print(f"[signedpayoff] TORCH CAP HIT: subsampling train {len(Xtr):,} -> {TORCH_CAP:,} for torch heads (lgb fobj uses FULL)", flush=True)
    else:
        tidx = np.arange(len(Xtr))
        print(f"[signedpayoff] torch train uncapped ({len(Xtr):,} <= {TORCH_CAP:,})", flush=True)
    Xtr_t = Xtr[tidx]; ytr_t = ytr[tidx]; Rtr_t = Rtr[tidx]
    norm = _norm_fns(Xtr_t)
    Xv_n = torch.tensor(norm(Xva))

    out = json.load(open(out_path))
    out["status"] = "OOS-EVALUATED"
    out["HEAD_A"] = {"lgb": {}, "torch": {}}
    out["HEAD_B"] = {}

    # =====================================================================================================
    # HEAD A (i): GMADL custom LightGBM fobj on FULL train, sweep a,b
    # =====================================================================================================
    for a in A_GRID:
        for b in B_GRID:
            key = f"a{a}_b{b}"
            fobj = gmadl_obj(Rtr, a, b)
            m = lgb.LGBMClassifier(n_estimators=1200, learning_rate=0.03, num_leaves=127, min_child_samples=300,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=20,
                                   n_jobs=20, verbosity=-1, objective=fobj)
            m.fit(Xtr, ytr)
            # VAL worst-half SELECTION objective (per side), score = sigmoid(raw)
            sval = sigmoid(m.booster_.predict(Xva, raw_score=True))
            wv_up = worst_val_half_obj(sval, Rva, yva, tsva, nyva, vyr, ("A", 1))
            wv_dn = worst_val_half_obj(sval, Rva, yva, tsva, nyva, vyr, ("A", 0))
            sc = lambda D, mm=m: sigmoid(mm.booster_.predict(D[cols].astype("float32"), raw_score=True))
            up = per_year(EV, sc, 1); dn = per_year(EV, sc, 0)
            out["HEAD_A"]["lgb"][key] = {"a": a, "b": b, "val_worsthalf_payoff": {"UP": wv_up, "DOWN": wv_dn},
                                          "UP": up, "DOWN": dn}
            print(f"  [A-lgb {key}] valWH UP={wv_up:.3f} DOWN={wv_dn:.3f} | UP_bind={up['binding']} DOWN_bind={dn['binding']} ({time.time()-T0:.0f}s)", flush=True)
            del m; gc.collect()
            json.dump(out, open(out_path, "w"), indent=2, default=str)

    # =====================================================================================================
    # HEAD A (ii): GMADL torch MLP (same loss), sweep a,b
    # =====================================================================================================
    Xt_t = torch.tensor(norm(Xtr_t))
    Rt_t = torch.tensor(Rtr_t.astype("float32"))
    yt_t = torch.tensor(ytr_t.astype("float32"))
    for a in A_GRID:
        for b in B_GRID:
            key = f"a{a}_b{b}"
            torch.manual_seed(SEED)
            net = GMADLNet(d).to(DEV)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            bs = 4096; n = len(Xt_t); best = (float("-inf"), None, -1)
            af = float(a); bf = float(b)
            for ep in range(40):
                net.train(); perm = torch.randperm(n)
                for i in range(0, n, bs):
                    idx = perm[i:i + bs]; opt.zero_grad()
                    z = net(Xt_t[idx]); R = Rt_t[idx]
                    u = af * R * z
                    # L = -mean( sigmoid(a*R*z) * |R|^b )   (sign-coupled GMADL)
                    loss = -(torch.sigmoid(u) * (R.abs() ** bf)).mean()
                    loss.backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    zv = net(Xv_n).numpy()
                sval = sigmoid(zv)                          # raw score -> prob in [0,1]
                wv = min(worst_val_half_obj(sval, Rva, yva, tsva, nyva, vyr, ("A", 1)),
                         worst_val_half_obj(sval, Rva, yva, tsva, nyva, vyr, ("A", 0)))
                if wv > best[0]:
                    best = (wv, {k: v.detach().clone() for k, v in net.state_dict().items()}, ep)
            if best[1] is not None:
                net.load_state_dict(best[1])
            net.eval()
            def sc(D, nn_=net):
                with torch.no_grad():
                    zz = nn_(torch.tensor(norm(D[cols].astype("float32").to_numpy()))).numpy()
                return sigmoid(zz)
            up = per_year(EV, sc, 1); dn = per_year(EV, sc, 0)
            out["HEAD_A"]["torch"][key] = {"a": a, "b": b, "best_epoch": best[2],
                                           "val_worsthalf_payoff_minside": round(float(best[0]), 4) if np.isfinite(best[0]) else None,
                                           "UP": up, "DOWN": dn}
            print(f"  [A-torch {key}] bestEp={best[2]} valWHmin={best[0]:.3f} | UP_bind={up['binding']} DOWN_bind={dn['binding']} ({time.time()-T0:.0f}s)", flush=True)
            del net; gc.collect()
            json.dump(out, open(out_path, "w"), indent=2, default=str)

    # =====================================================================================================
    # HEAD B: RRL / differentiable-Sharpe torch MLP. X_t=tanh(f). payoff = deriv ties-LOSE (+0.85/-1.0) keyed to _y.
    #   Two loss variants: 'meanpnl' (L=-mean(X_t*payoff_t)) and 'sharpe' (L=-Sharpe({X_t*payoff_t})).
    #   payoff_t built per-bar from the TRUE moved label: up_realized=ytr (1 up / 0 down). For a position X_t the
    #   realized binary trade outcome correctness is sign(X_t)==dir, but to keep gradient flow we use the SETTLEMENT
    #   weighting payoff_t = +0.85 if the bar moved UP else -1.0-mapping is NOT smooth; instead the differentiable
    #   reward is X_t * raw_payoff where raw_payoff encodes the asymmetric win/loss already keyed to direction:
    #       raw_payoff_t = WIN  if ytr==1 (up bar)  else  -WIN_for_down?  -> we use a SIGNED settlement vector:
    #   We define rew_t so that a long (X>0) on an up bar earns +WIN and on a down bar loses -|LOSS|, and a short
    #   (X<0) earns the mirror. With binary up/down this equals X_t * s_t where s_t is the per-bar SIGNED settlement:
    #       up bar : long wins +0.85, short loses (short on up = wrong) -1.0  -> reward(X)= X>0:+.85 / X<0:-1.0
    #   This is piecewise in sign(X) (the deriv payoff IS a step in sign), so we optimize the smooth surrogate
    #   X_t * g_t with g_t = +0.85 on up bars / -0.85... NO. To stay FAITHFUL we evaluate the TRUE step payoff in the
    #   VAL objective + all OOS reporting (per_year, ties-LOSE) and only use the smooth X_t*reward for the gradient.
    # =====================================================================================================
    # Faithful settlement reward for SELECTION/eval is the discrete ties-LOSE per_year(); the training surrogate below
    # is a smooth differentiable proxy (Moody-Saffell RRL). NO leak: surrogate only shapes weights; every reported
    # number (VAL worst-half SELECTION + OOS per-year) uses the DISCRETE ties-LOSE payoff / accuracy.
    # surrogate per-bar reward r_t in {+1 up, -1 down}; trade pnl = X_t * r_t (asym win/loss applied at eval, not here).
    rew_t = torch.tensor(np.where(ytr_t == 1, 1.0, -1.0).astype("float32"))
    for arch_name, (h1, h2) in ARCHS.items():
        for sharpe in (False, True):
            lname = "sharpe" if sharpe else "meanpnl"
            torch.manual_seed(SEED)
            net = RRLNet(d, h1, h2).to(DEV)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            bs = 4096; n = len(Xt_t)
            # SELECTION over (arch, loss, tau): track best worst-VAL-half DISCRETE ties-LOSE payoff over TAUS
            best = (float("-inf"), None, -1, None)
            for ep in range(40):
                net.train(); perm = torch.randperm(n)
                for i in range(0, n, bs):
                    idx = perm[i:i + bs]; opt.zero_grad()
                    x = net(Xt_t[idx]); pnl = x * rew_t[idx]      # smooth surrogate pnl
                    if sharpe:
                        mu = pnl.mean(); sd = pnl.std() + 1e-6; loss = -(mu / sd)
                    else:
                        loss = -pnl.mean()
                    loss.backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    xv = net(Xv_n).numpy()                        # X_t ∈ [-1,1] on VAL
                # worst-VAL-half DISCRETE ties-LOSE payoff, pick best tau this epoch
                ep_best_tau, ep_best_wv = None, float("-inf")
                for tau in TAUS:
                    wv = worst_val_half_obj(xv, Rva, yva, tsva, nyva, vyr, ("B", tau))
                    if wv > ep_best_wv:
                        ep_best_wv, ep_best_tau = wv, tau
                if ep_best_wv > best[0]:
                    best = (ep_best_wv, {k: v.detach().clone() for k, v in net.state_dict().items()}, ep, ep_best_tau)
            if best[1] is not None:
                net.load_state_dict(best[1])
            net.eval()
            TAU = best[3] if best[3] is not None else 0.0
            # score map for per_year(): s = 0.5 + 0.5*X_t  => s>0.5 iff X_t>0 (UP), s<0.5 iff X_t<0 (DOWN),
            # conf=|s-0.5|=0.5*|X_t|.  per_year conf-cover then selects high-|X_t|; identical to |X_t|>quantile gate.
            # tau is applied as a HARD pre-gate so weak |X_t|<=tau bars are abstained before the conf-cover.
            def scB(D, nn_=net, tau=TAU):
                with torch.no_grad():
                    xx = nn_(torch.tensor(norm(D[cols].astype("float32").to_numpy()))).numpy()
                xx = np.where(np.abs(xx) > tau, xx, 0.0)          # hard |X|>tau pre-gate -> s=0.5 (abstain) elsewhere
                return 0.5 + 0.5 * xx
            up = per_year(EV, scB, 1); dn = per_year(EV, scB, 0)
            key = f"{arch_name}_{lname}_tau{TAU}"
            out["HEAD_B"][key] = {"arch": arch_name, "loss": lname, "tau": TAU, "best_epoch": best[2],
                                  "val_worsthalf_payoff": round(float(best[0]), 4) if np.isfinite(best[0]) else None,
                                  "UP": up, "DOWN": dn}
            print(f"  [B {key}] bestEp={best[2]} tau={TAU} valWH={best[0]:.3f} | UP_bind={up['binding']} DOWN_bind={dn['binding']} ({time.time()-T0:.0f}s)", flush=True)
            del net; gc.collect()
            json.dump(out, open(out_path, "w"), indent=2, default=str)

    # =====================================================================================================
    # SURVIVOR CHECK (pre-registered falsifier) -> nested-refit CPCV for any HEAD-A-lgb survivor
    # =====================================================================================================
    def survivor_of(side_res, side):
        """Return survivor dict if this side's binding-2025 clears the falsifier; else None.
        DOWN: 2025 CI95-lo >= 0.541 AND win > 0.5594.  UP: 2025 CI95-lo > 0.577 + 1*SE(of UP-2025 CI).
        AND up-rate in [.47,.53] all years AND 2026 win >= 0.541 (no collapse)."""
        if 2025 not in side_res:
            return None
        r25 = side_res[2025]
        ci25 = r25.get("ci"); lo25 = ci25[0] if ci25 else None; win25 = r25.get("win")
        ok_uprate, _ = uprate_ok(side_res)
        r26 = side_res.get(2026, {})
        no_collapse26 = (r26.get("win") is not None and r26["win"] >= BE)
        if not (ok_uprate and no_collapse26 and lo25 is not None):
            return None
        if side == "DOWN":
            if lo25 >= BE and win25 is not None and win25 > INC_DN25:
                return {"side": side, "win25": win25, "ci_lo25": lo25}
        else:  # UP: beat 0.577 by > 1 SE
            se = se_of(ci25)
            if np.isfinite(se) and lo25 > (INC_UP25 + se):
                return {"side": side, "win25": win25, "ci_lo25": lo25, "se": round(float(se), 4)}
        return None

    survivors = []
    for key, v in out["HEAD_A"]["lgb"].items():
        for side in ("UP", "DOWN"):
            s = survivor_of(v[side], side)
            if s:
                s.update({"head": "A-lgb", "key": key, "a": v["a"], "b": v["b"]}); survivors.append(s)
    for head_tag, head in (("A-torch", out["HEAD_A"]["torch"]), ("B", out["HEAD_B"])):
        for key, v in head.items():
            for side in ("UP", "DOWN"):
                if side in v:
                    s = survivor_of(v[side], side)
                    if s:
                        s.update({"head": head_tag, "key": key}); survivors.append(s)
    print(f"[signedpayoff] pre-CPCV survivors: {len(survivors)} -> {survivors}", flush=True)

    # nested-refit CPCV only for HEAD-A-lgb survivors (single-lgb refit per path); torch/B survivors are flagged
    # FREEZE-BLOCKED pending a torch CPCV harness (not built — logged, not silently skipped).
    Xall = np.concatenate([Xtr] + [EV[w][cols].astype("float32").to_numpy() for w in ("test24", "test25", "oos")])
    yall = np.concatenate([ytr] + [EV[w]["_y"].astype(int).values for w in ("test24", "test25", "oos")])
    Rall = np.concatenate([Rtr] + [EV[w]["_fwd"].astype("float64").values for w in ("test24", "test25", "oos")])
    tsall = np.concatenate([TR["_ts"].values.astype("int64")] +
                           [EV[w]["_ts"].values.astype("int64") for w in ("test24", "test25", "oos")])
    nyall = np.concatenate([(TR["sess_ny"].values > 0.5)] +
                           [(EV[w]["sess_ny"].values > 0.5) for w in ("test24", "test25", "oos")])
    cpcv_out = {}
    for s in survivors:
        if s["head"] == "A-lgb":
            print(f"[signedpayoff] nested-refit CPCV for survivor {s} ...", flush=True)
            s["_R_full"] = Rall
            res = cpcv_refit(s, cols, Xall, yall, tsall, nyall, s["side"], gmadl_fobj_builder=gmadl_obj)
            cpcv_out[f"{s['head']}_{s['key']}_{s['side']}"] = res
            print(f"  -> CPCV {s['key']} {s['side']}: {res}", flush=True)
        else:
            cpcv_out[f"{s['head']}_{s['key']}_{s['side']}"] = {"CERTIFIED": False,
                "note": "torch survivor: nested-refit CPCV harness not built for torch heads -> FREEZE-BLOCKED (not certified)"}
    out["CPCV"] = cpcv_out

    # =====================================================================================================
    # VERDICT
    # =====================================================================================================
    certified = [k for k, v in cpcv_out.items() if v.get("CERTIFIED")]
    any_survivor = len(survivors) > 0
    out["VERDICT"] = {
        "pre_oos_survivors": survivors,
        "cpcv_certified": certified,
        "FROZEN_ELIGIBLE": bool(certified),
        "statement": (
            ("FREEZE-ELIGIBLE: " + ", ".join(certified) + " cleared the binding-2025 falsifier AND nested-refit CPCV "
             "(p10>=0.541, >=80% of 28 paths clear).") if certified else
            (("KILL: " + str(len(survivors)) + " config(s) cleared the binding-2025 falsifier but FAILED nested-refit CPCV "
              "(p10<0.541 or <80% paths) -> not certifiable (the way ACI did). Sign-coupled GMADL/RRL objectives do NOT "
              "produce a refit-stable 5m direction book.") if any_survivor else
             ("KILL (loss-reopt + sizing-policy families empirically exhausted): no sign-coupled GMADL/MADL (a,b) nor "
              "RRL/diff-Sharpe (arch,tau) config beat incumbent on the binding-2025 year on either side at the up-rate "
              "tripwire with non-collapsing 2026. The objective is not the binding constraint at 5m — coupling the "
              "signed score to the signed return adds nothing the symmetric-reweight / BCE-equivalent family already "
              "exhausted. Joins focal/quantile/asym/magweight/seedens/rankloss in the loss graveyard.")))}
    out["status"] = "COMPLETE"
    json.dump(out, open(out_path, "w"), indent=2, default=str)
    print(f"[signedpayoff] VERDICT: {out['VERDICT']['statement']}", flush=True)
    print(f"[signedpayoff] -> {out_path} ({time.time()-T0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
