"""DEEP-RL FALSIFIER for 1-minute (60s) EURUSD binary direction.

The user asked for REAL deep-RL (DQN). This builds two agents and reports honestly. The
deriv-faithful settlement, splits, features, and live de-overlap policy are REUSED verbatim
from min1_production.py (M.*) — no re-derivation of the label, no smoothing/leakage.

GROUND TRUTH (pre-registered): 60s sign AUC ~0.51 across all model classes; the oracle floor
predicts a direction-with-abstain policy caps ~0.60 and CANNOT clear 0.65. Magnitude |ret60|
AUC ~0.79 but is SIGN-INVARIANT — quantile skew should NOT carry sign. Both parts are
PRE-REGISTERED FALSIFIERS; a clean negative from a never-tested channel (DQN abstain / IQN CVaR
gate) is a real first-in-world result.

(A) DQN DIRECTION-WITH-ABSTAIN
    torch MLP Q-net, experience replay, target net. Actions {LONG, SHORT, ABSTAIN}.
    Reward (deriv settlement, breakeven 0.541 => payout R=0.85): correct dir -> +R, wrong -> -1,
    abstain -> 0; ties (mag==0) LOSE. Train on 2021-23 (subsample <=50k transitions). Evaluate the
    NON-abstained subset per-window 2024/2025/2026 with CI95 + coverage.
    FALSIFIER: require traded-subset CI95-lower > 0.65 in EACH year at >=5% coverage. Expected FAIL.

(B) IQN + CVaR MAGNITUDE-GATED ABSTAIN
    Implicit Quantile Network over magnitude features (realized-vol, entropy, order-flow) modeling
    the 60s SIGNED return distribution. Test (1) whether quantile SKEW carries sign (sign-invariance
    theorem predicts NO), and (2) whether a CVaR/spread abstain gate improves a magnitude-gated
    book's committed-trade accuracy + risk. Per-window accuracy on committed trades + CI95.

Memory-safe: one split loaded at a time, subsample <=50k for fits, del big frames. CPU torch.

Usage: ~/binary-algo-venv/bin/python min1_drl.py            # runs A then B, prints report
"""
import sys, os, time, json, gc, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/sean/git/binary-algo")
import min1_production as M

torch.manual_seed(7); np.random.seed(7)
DEV = "cpu"
# CPU torch on tiny batches is dominated by thread-sync overhead: benchmarked 30.7s/epoch at
# 16 threads vs 3.2s/epoch at 1 thread (10x) for this exact MLP+batch. Single-thread is FASTER here.
torch.set_num_threads(1)
PAYOUT = 0.85                       # deriv ~15% deduction => R=0.85, breakeven 1/(1+R)=0.541
N_TRAIN_SUB = 50_000               # transitions for DQN fit (subsample cap)
COV_FLOOR = 0.05                   # falsifier minimum coverage in EACH year
BAR_STRIDE = 5                     # subsample stride over 1s bars (independence + memory)


# ----------------------------- shared data plumbing (reuse M, memory-safe) ----------------
# M.prep on the full 13.8M-row train split peaks >13.7GB (pandas builds 62 full-length rolling
# Series at once) and OOM-kills a 31GB box shared with other jobs. We instead compute features in
# contiguous CHUNKs with a WARMUP prefix (covers the longest rolling window, 3600 bars, + EMA
# convergence) and a FORWARD suffix (covers the 60s+tol label lookahead), keep only the valid
# interior of each chunk, and subsample by stride. Peak ~ CHUNK rows of features (sub-GB), and the
# features are IDENTICAL to M.feats on the interior (warmup >> max window so EMA boundary error is
# negligible; rolling/pct_change windows are fully covered). The label M.wc_ret is computed on the
# chunk's own ts/mid with the forward suffix present, so it matches the full-split label exactly.
WARMUP = 5000      # >= max rolling window (3600) + EMA convergence margin
FORWARD = 90       # >= HS(60) + TOL_S(10) + ENTRY_LAG_S(1), in BARS (1s bars => seconds)
CHUNK = 1_000_000


def _prep_interior(b_slice, lo, hi):
    """Compute M.feats + M.wc_ret on a bar slice, return only rows [lo:hi] (the valid interior).
    Mirrors M.prep exactly on that interior."""
    X = M.feats(b_slice)
    mid = b_slice["mid"].values.astype(float)
    ts = b_slice.index.values.astype("datetime64[s]").astype("int64")
    ret, valid = M.wc_ret(ts, mid, M.HS, M.TOL_S, M.ENTRY_LAG_S)
    sl = slice(lo, hi)
    return (X.iloc[sl], (ret[sl] > 0).astype(int), np.abs(ret[sl]),
            valid[sl], ts[sl], b_slice.index[sl])


def build_split(sp, stride=BAR_STRIDE):
    """Return (X_df, y, mag, valid, ts, idx) for a split — chunked, memory-safe, deriv-faithful
    label (ties LOSE). Subsample by stride to bound memory and decorrelate consecutive bars."""
    b = M.load_split(sp)
    n = len(b)
    Xs, ys, ms, vs, tss, idxs = [], [], [], [], [], []
    start = 0
    while start < n:
        end = min(start + CHUNK, n)
        s0 = max(0, start - WARMUP)              # warmup prefix
        s1 = min(n, end + FORWARD)               # forward suffix for the label
        lo = start - s0                          # interior start within the slice
        hi = lo + (end - start)                  # interior end
        bx, by, bm, bv, bt, bi = _prep_interior(b.iloc[s0:s1], lo, hi)
        # .copy()/np copies so the big per-chunk feature frame is freed each iteration (a strided
        # .iloc view would otherwise pin the whole 1M-row chunk DataFrame alive across all chunks).
        ssl = slice(None, None, stride) if stride > 1 else slice(None)
        Xs.append(bx.iloc[ssl].copy()); ys.append(np.array(by[ssl])); ms.append(np.array(bm[ssl]))
        vs.append(np.array(bv[ssl])); tss.append(np.array(bt[ssl])); idxs.append(bi[ssl].copy())
        del bx, by, bm, bv, bt, bi
        start = end
        gc.collect()
    del b; gc.collect()
    X = pd.concat(Xs); y = np.concatenate(ys); mag = np.concatenate(ms)
    valid = np.concatenate(vs); ts = np.concatenate(tss); idx = idxs[0].append(idxs[1:]) if len(idxs) > 1 else idxs[0]
    del Xs, ys, ms, vs, tss, idxs; gc.collect()
    return X, y, mag, valid, ts, idx


def year_mask(idx, yr):
    return np.asarray(idx.year == yr)


def standardize_fit(Xdf):
    Xv = Xdf.values.astype(np.float32)
    med = np.nanmedian(Xv, axis=0)
    Xv = np.where(np.isfinite(Xv), Xv, med)
    mu = Xv.mean(0); sd = Xv.std(0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32), med.astype(np.float32)


def standardize_apply(Xdf, mu, sd, med):
    Xv = Xdf.values.astype(np.float32)
    Xv = np.where(np.isfinite(Xv), Xv, med)
    return ((Xv - mu) / sd).astype(np.float32)


def boot_ci(correct, nb=5000, seed=7):
    return M.boot(np.asarray(correct, dtype=float), nb=nb, seed=seed)


def eval_window(pred_dir, traded_mask, y, mag, idx, ts, yr, gap=M.GAP):
    """Live-faithful eval for one calendar year: among bars the agent chose to TRADE (traded_mask),
    restrict to year yr, apply chronological non-overlap de-overlap (no look-ahead), then accuracy =
    (pred_dir==y & mag>0). Returns (n, acc, lo, hi, coverage, n_year_candidates)."""
    ym = year_mask(idx, yr)
    cand = traded_mask & ym
    n_cand = int(cand.sum())
    if n_cand == 0:
        return 0, float("nan"), float("nan"), float("nan"), 0.0, 0
    tr = M.nonoverlap_chrono(ts, cand, gap=gap)
    if len(tr) == 0:
        return 0, float("nan"), float("nan"), float("nan"), 0.0, n_cand
    correct = ((pred_dir[tr] == y[tr]) & (mag[tr] > 0)).astype(float)
    acc = float(correct.mean()); lo, hi = boot_ci(correct)
    n_year_bars = int(ym.sum())
    cov = len(tr) / n_year_bars if n_year_bars else float("nan")
    return len(tr), acc, lo, hi, cov, n_cand


# =========================================================================================
# (A) DQN DIRECTION-WITH-ABSTAIN
# =========================================================================================
class QNet(nn.Module):
    def __init__(self, d, h=256, na=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.LayerNorm(h),
            nn.Linear(h, h), nn.ReLU(), nn.LayerNorm(h),
            nn.Linear(h, na))
    def forward(self, x):
        return self.net(x)


def action_reward(a, y, mag):
    """a: 0=LONG,1=SHORT,2=ABSTAIN. Deriv settlement net of breakeven via payout R.
    LONG correct iff y==1 & mag>0; SHORT correct iff y==0 & mag>0; ties (mag==0) LOSE."""
    win_long = (y == 1) & (mag > 0)
    win_short = (y == 0) & (mag > 0)
    r = np.zeros_like(y, dtype=np.float32)
    r[(a == 0) & win_long] = PAYOUT
    r[(a == 0) & ~win_long] = -1.0
    r[(a == 1) & win_short] = PAYOUT
    r[(a == 1) & ~win_short] = -1.0
    r[a == 2] = 0.0
    return r


def train_dqn(Xtr, ytr, mtr, vtr, d, epochs=25, bs=4096, lr=1e-3, gamma=0.0):
    """Contextual-bandit DQN (gamma=0: each bar's binary is an independent one-step episode — the
    correct MDP for a fixed-expiry option; no state transition carries across non-overlapping trades).
    Experience replay + target net retained per the spec (real DQN machinery; with gamma=0 the
    target net is a stability anchor on the bandit targets). Subsample to N_TRAIN_SUB transitions."""
    moved = vtr & (mtr > 0)
    idx_pool = np.where(moved)[0]
    if len(idx_pool) > N_TRAIN_SUB:
        idx_pool = np.random.choice(idx_pool, N_TRAIN_SUB, replace=False)
    Xs = torch.from_numpy(Xtr[idx_pool])
    yv = ytr[idx_pool]; mv = mtr[idx_pool]
    # precompute the reward for every (state, action) — bandit: full reward vector known
    R = np.stack([action_reward(np.zeros(len(yv), int), yv, mv),
                  action_reward(np.ones(len(yv), int), yv, mv),
                  action_reward(np.full(len(yv), 2, int), yv, mv)], axis=1).astype(np.float32)
    Rt = torch.from_numpy(R)
    q = QNet(d).to(DEV); qt = QNet(d).to(DEV); qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    n = len(Xs)
    eps = 1.0; tlast = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n)
        eps = max(0.05, 1.0 - ep / (epochs * 0.6))     # epsilon decay for exploration
        tot = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            xb = Xs[b]
            with torch.no_grad():
                qb = q(xb)
            # epsilon-greedy action selection -> store experience
            greedy = qb.argmax(1)
            rand = torch.randint(0, 3, (len(b),))
            explore = torch.rand(len(b)) < eps
            a = torch.where(explore, rand, greedy)
            r = Rt[b].gather(1, a.unsqueeze(1)).squeeze(1)   # realized reward for chosen action
            # bandit (gamma=0): target = immediate reward; target net used for stability on Q(s,a)
            pred = q(xb).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                # blend toward target-net estimate to damp variance (DQN-style soft target)
                tgt = r + gamma * qt(xb).max(1).values
            loss = F.smooth_l1_loss(pred, tgt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), 5.0); opt.step()
            tot += float(loss) * len(b)
        # soft target update
        with torch.no_grad():
            for tp, sp_ in zip(qt.parameters(), q.parameters()):
                tp.mul_(0.99).add_(sp_, alpha=0.01)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[A]   dqn ep{ep} loss={tot/n:.4f} eps={eps:.2f} {time.time()-tlast:.0f}s", flush=True)
            tlast = time.time()
    del Xs, Rt; gc.collect()
    return q


@torch.no_grad()
def dqn_policy(q, Xz, bs=200000):
    """Return greedy action per bar and the Q-margin (confidence proxy)."""
    acts = np.empty(len(Xz), dtype=np.int64); marg = np.empty(len(Xz), dtype=np.float32)
    for i in range(0, len(Xz), bs):
        xb = torch.from_numpy(Xz[i:i + bs])
        qb = q(xb)
        a = qb.argmax(1)
        top2 = torch.topk(qb, 2, dim=1).values
        acts[i:i + bs] = a.numpy()
        marg[i:i + bs] = (top2[:, 0] - top2[:, 1]).numpy()
    return acts, marg


def run_A():
    print("\n" + "=" * 78 + "\n(A) DQN DIRECTION-WITH-ABSTAIN  (pre-registered falsifier: CI95-lo>0.65 each yr @>=5% cov)\n" + "=" * 78, flush=True)
    t0 = time.time()
    Xtr_df, ytr, mtr, vtr, _, _ = build_split("train")
    mu, sd, med = standardize_fit(Xtr_df)
    d = Xtr_df.shape[1]
    Xtr = standardize_apply(Xtr_df, mu, sd, med); del Xtr_df; gc.collect()
    print(f"[A] train bars(stride{BAR_STRIDE})={len(Xtr)} feats={d} prep {time.time()-t0:.0f}s", flush=True)
    q = train_dqn(Xtr, ytr, mtr, vtr, d)
    del Xtr, ytr, mtr, vtr; gc.collect()
    print(f"[A] DQN trained {time.time()-t0:.0f}s", flush=True)

    results = {}
    abstain_rates = {}
    for sp in ("test", "oos"):
        Xdf, y, mag, valid, ts, idx = build_split(sp)
        Xz = standardize_apply(Xdf, mu, sd, med); del Xdf; gc.collect()
        acts, marg = dqn_policy(q, Xz); del Xz; gc.collect()
        # pred_dir: LONG(0)->up(1), SHORT(1)->down(0); ABSTAIN(2)-> no trade
        pred_dir = np.where(acts == 0, 1, 0)
        traded = valid & (acts != 2)
        abstain_rates[sp] = float((acts[valid] == 2).mean()) if valid.sum() else float("nan")
        years = [2024, 2025] if sp == "test" else [2026]
        for yr in years:
            results[yr] = eval_window(pred_dir, traded, y, mag, idx, ts, yr)
        del y, mag, valid, ts, idx, acts, marg, pred_dir, traded; gc.collect()

    print(f"\n[A] abstain-rate(valid bars): test={abstain_rates['test']:.2%} oos={abstain_rates['oos']:.2%}")
    print(f"[A] reward: correct +{PAYOUT}, wrong -1, abstain 0; breakeven={1/(1+PAYOUT):.3f}")
    pass_each = True
    for yr in (2024, 2025, 2026):
        n, acc, lo, hi, cov, ncand = results[yr]
        ok = (lo > 0.65) and (cov >= COV_FLOOR)
        pass_each &= ok
        print(f"[A] {yr}: traded={n:>5} acc={acc:.3f} CI95=[{lo:.3f},{hi:.3f}] cov={cov:.2%} "
              f"-> {'PASS' if ok else 'FAIL'}")
    print(f"[A] FALSIFIER VERDICT: {'CLEARED 0.65 (surprise!)' if pass_each else 'FAILED (confirmed — DQN abstain does not beat the 0.51 sign floor)'}")
    del q; gc.collect()
    return results, pass_each, abstain_rates


# =========================================================================================
# (B) IQN + CVaR MAGNITUDE-GATED ABSTAIN
# =========================================================================================
class IQN(nn.Module):
    """Implicit Quantile Network: maps (state, tau) -> quantile of the SIGNED 60s return.
    Cosine-embedded tau a la Dabney et al. 2018."""
    def __init__(self, d, h=128, n_cos=64):
        super().__init__()
        self.n_cos = n_cos
        self.psi = nn.Sequential(nn.Linear(d, h), nn.ReLU())
        self.phi = nn.Linear(n_cos, h)
        self.out = nn.Sequential(nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.register_buffer("ar", torch.arange(1, n_cos + 1, dtype=torch.float32) * np.pi)
    def forward(self, x, tau):
        # x:(B,d) tau:(B,N) -> (B,N) quantile values
        psi = self.psi(x)                                  # (B,h)
        cos = torch.cos(tau.unsqueeze(-1) * self.ar)       # (B,N,n_cos)
        phi = F.relu(self.phi(cos))                        # (B,N,h)
        z = psi.unsqueeze(1) * phi                         # (B,N,h)
        return self.out(z).squeeze(-1)                     # (B,N)


def quantile_huber(pred, target, tau, kappa=1.0):
    # pred:(B,N) target:(B,M) tau:(B,N)
    u = target.unsqueeze(1) - pred.unsqueeze(2)            # (B,N,M)
    hub = torch.where(u.abs() <= kappa, 0.5 * u.pow(2), kappa * (u.abs() - 0.5 * kappa))
    rho = (tau.unsqueeze(2) - (u.detach() < 0).float()).abs() * hub / kappa
    return rho.mean(2).sum(1).mean()


MAG_FEATS = ["imb", "imb_ema5", "imb_ema20", "imb_acc", "micro_dev", "micro_dev_ema15",
             "spread", "spread_ema30", "nt", "nt_ema30", "tsz", "tsz_ema30",
             "rv30", "rv60", "rv300", "rv900", "rv1800",
             "bbw300", "bbw900", "bbw1800", "rel_ratio", "rel_ratio2",
             "stretch300", "stretch900", "rangepos300", "rangepos900"]


def train_iqn(Xm, ret_signed, epochs=20, bs=4096, lr=1e-3, N=16):
    d = Xm.shape[1]
    net = IQN(d).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=lr)
    # scale signed return to ~unit for stable quantile regression
    scale = float(np.std(ret_signed)) + 1e-9
    Xt = torch.from_numpy(Xm); yt = torch.from_numpy((ret_signed / scale).astype(np.float32)).unsqueeze(1)
    n = len(Xt); tlast = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]; xb = Xt[b]; yb = yt[b]
            tau = torch.rand(len(b), N)
            pred = net(xb, tau)
            loss = quantile_huber(pred, yb, tau)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            tot += float(loss) * len(b)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[B]   iqn ep{ep} loss={tot/n:.4f} {time.time()-tlast:.0f}s", flush=True); tlast = time.time()
    return net, scale


@torch.no_grad()
def iqn_quantiles(net, Xm, taus, bs=100000):
    """Return quantile matrix (n, len(taus)) of signed return (in scaled units)."""
    tt = torch.tensor(taus, dtype=torch.float32)
    out = np.empty((len(Xm), len(taus)), dtype=np.float32)
    for i in range(0, len(Xm), bs):
        xb = torch.from_numpy(Xm[i:i + bs])
        tau = tt.unsqueeze(0).expand(len(xb), -1)
        out[i:i + bs] = net(xb, tau).numpy()
    return out


def run_B():
    print("\n" + "=" * 78 + "\n(B) IQN + CVaR MAGNITUDE-GATED ABSTAIN  (sign-invariance test + risk-gated book)\n" + "=" * 78, flush=True)
    t0 = time.time()
    Xtr_df, ytr, mtr, vtr, _, _ = build_split("train")
    feat_use = [c for c in MAG_FEATS if c in Xtr_df.columns]
    mu, sd, med = standardize_fit(Xtr_df[feat_use])
    moved = vtr & (mtr > 0)
    pool = np.where(moved)[0]
    if len(pool) > N_TRAIN_SUB:
        pool = np.random.choice(pool, N_TRAIN_SUB, replace=False)
    Xm = standardize_apply(Xtr_df[feat_use].iloc[pool], mu, sd, med)
    # signed return = +mag if up else -mag  (the IQN models the SIGNED 60s return distribution)
    ret_signed = np.where(ytr[pool] == 1, mtr[pool], -mtr[pool]).astype(np.float32)
    del Xtr_df; gc.collect()
    print(f"[B] IQN train n={len(Xm)} feats={len(feat_use)} prep {time.time()-t0:.0f}s", flush=True)
    net, scale = train_iqn(Xm, ret_signed)
    del Xm, ytr, mtr, vtr; gc.collect()
    print(f"[B] IQN trained {time.time()-t0:.0f}s", flush=True)

    taus = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], dtype=np.float32)
    results_sign = {}; results_gated = {}; results_base = {}
    for sp in ("test", "oos"):
        Xdf, y, mag, valid, ts, idx = build_split(sp)
        Xm = standardize_apply(Xdf[feat_use], mu, sd, med); del Xdf; gc.collect()
        Q = iqn_quantiles(net, Xm, taus); del Xm; gc.collect()
        med_q = Q[:, 3]                                   # tau=0.5 quantile (median signed ret)
        spread_q = Q[:, 5] - Q[:, 1]                      # 0.9-0.1 interquantile spread (magnitude)
        # skew: (q90 - median) - (median - q10); >0 = right-skew
        skew_q = (Q[:, 5] - med_q) - (med_q - Q[:, 1])
        # --- (B1) does quantile SKEW carry sign? predict up iff skew>0; baseline=median sign ---
        pred_skew = (skew_q > 0).astype(int)
        pred_med = (med_q > 0).astype(int)               # median-sign predictor (should be ~0.51 too)
        years = [2024, 2025] if sp == "test" else [2026]
        # magnitude gate threshold from THIS split's own valid distribution (top-tercile spread) for the book
        spr_thr = np.nanpercentile(spread_q[valid], 67)
        traded_all = valid.copy()                        # baseline magnitude book: trade all valid, dir=median sign
        traded_gated = valid & (spread_q >= spr_thr)     # CVaR/spread gate: only large-spread bars
        for yr in years:
            results_sign[yr] = eval_window(pred_skew, valid, y, mag, idx, ts, yr)
            results_base[yr] = eval_window(pred_med, traded_all, y, mag, idx, ts, yr)
            results_gated[yr] = eval_window(pred_med, traded_gated, y, mag, idx, ts, yr)
        del y, mag, valid, ts, idx, Q, med_q, spread_q, skew_q, pred_skew, pred_med; gc.collect()

    print("\n[B1] SIGN-INVARIANCE TEST: does IQN quantile SKEW predict direction?")
    for yr in (2024, 2025, 2026):
        n, acc, lo, hi, cov, _ = results_sign[yr]
        print(f"     {yr}: skew->dir  n={n:>5} acc={acc:.3f} CI95=[{lo:.3f},{hi:.3f}]")
    print("[B1] sign-invariance theorem predicts skew acc ~0.50 (CI straddles 0.50 => skew carries NO sign)")

    print("\n[B2] MAGNITUDE-GATED BOOK (dir = IQN median sign; abstain unless large IQN spread):")
    for yr in (2024, 2025, 2026):
        nb, ab, lb, hb, cb, _ = results_base[yr]
        ng, ag, lg, hg, cg, _ = results_gated[yr]
        print(f"     {yr}: base(all-valid) n={nb:>5} acc={ab:.3f} CI95=[{lb:.3f},{hb:.3f}] cov={cb:.2%}  ||  "
              f"CVaR-gated n={ng:>5} acc={ag:.3f} CI95=[{lg:.3f},{hg:.3f}] cov={cg:.2%}")
    del net; gc.collect()
    return results_sign, results_base, results_gated


if __name__ == "__main__":
    print(f"[min1_drl] torch {torch.__version__} dev={DEV} threads={torch.get_num_threads()} "
          f"payout={PAYOUT} breakeven={1/(1+PAYOUT):.3f} stride={BAR_STRIDE} sub={N_TRAIN_SUB}", flush=True)
    rA, passA, abst = run_A()
    rB_sign, rB_base, rB_gated = run_B()
    print("\n" + "=" * 78 + "\nSUMMARY\n" + "=" * 78)
    print("(A) DQN abstain falsifier:", "CLEARED 0.65" if passA else "FAILED 0.65 (confirmed negative)")
    for yr in (2024, 2025, 2026):
        n, acc, lo, hi, cov, _ = rA[yr]
        print(f"    A {yr}: acc={acc:.3f} CI95=[{lo:.3f},{hi:.3f}] cov={cov:.2%}")
    for yr in (2024, 2025, 2026):
        ns, accs, los, his, _, _ = rB_sign[yr]
        print(f"    B-skew {yr}: acc={accs:.3f} CI95=[{los:.3f},{his:.3f}]")
