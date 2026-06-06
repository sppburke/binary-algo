"""SESSION re-campaign — CROSS-PAIR primary (the certified >=10m direction lever), STRICT session-only, DST-correct.

The deployed EURUSD.m{10,15,30}xp.v1 books TRAIN on the full pooled cross-pair set (all sessions, 7 USD pairs) and
GATE at decision time on comp(bb_width) x NY — but with the LEGACY fixed-UTC NY window (m5_xpair build_xp:
sess_ny = hour in [13,22) UTC; pipeline uses [12,21)). The user's directive: re-test with DST-correct exchange-local
sessions AND use ONLY that session's data for train+val+test+OOS (not done before).

This is the STRICT session-only variant: pre-filter the pooled rows to ONE DST-correct session (sessions.py) BEFORE
the CPCV split, so every training fold, the VAL selection slice, and the test paths are all within that session;
the decision gate is comp(bb_width) compression only (the NY multiply is subsumed by the pre-filter). Per-side
full-refit CPCV (6 groups, C(6,2)=15 purged paths, purge+embargo=2*HOR). CERTIFY a side iff p10>=.541 & frac>=.80.
Compare to the legacy-gate certified numbers in m{HOR}_xpair_cpcv_result.json (NY-fixed-UTC).

Run: ~/binary-algo-venv/bin/python session_xpair.py <HOR=10|15|30>
  -> session_xpair_<HOR>m_{ny,ldn,asia}_result.json   (mem-light bar store; CPU lane — co-runs with GPU Kronos)
"""
import os, sys
HOR = int(sys.argv[1]) if len(sys.argv) > 1 else 10
os.environ["MX_HOR"] = str(HOR)
import json, time, numpy as np
from itertools import combinations
import lightgbm as lgb
import m5_xpair as MX
from sessions import session_mask, SESSIONS

YEARS = list(range(2012, 2027)); N_GROUPS, K_TEST = 6, 2; SUB_FIT = 150_000
GAP_S = HOR * 60; BE = 0.541; MIN_SIDE_N = 10
GATE_FEAT = {10: "5m_bb_width", 15: "15m_bb_width", 30: "1h_bb_width"}[HOR]
LGB_JOBS = int(os.environ.get("LGB_THREADS", "20"))
T0 = time.time()
def hb(m): print(f"HB[{time.time()-T0:.0f}s] {m}", flush=True)


def build_pooled():
    ref = MX.augment(MX.build_xp([2020]), [2020], "xpof")
    cols = MX.feat_cols("xpof", ref, MX.xp_cols(ref))
    assert GATE_FEAT in ref.columns, f"{GATE_FEAT} missing"
    del ref
    Xs, ys, tss, bbs = [], [], [], []
    for y in YEARS:
        F = MX.build_xp([str(y)], stride=2)
        if F is None or len(F) == 0: continue
        F = MX.augment(F, [str(y)], "xpof")
        for c in cols:
            if c not in F.columns: F[c] = np.nan
        Xs.append(F[cols].values.astype("float32")); ys.append(F["_y"].astype("int8").values)
        tss.append(F["_ts"].values.astype("int64")); bbs.append(F[GATE_FEAT].values.astype("float32"))
        del F
    X = np.concatenate(Xs); y = np.concatenate(ys); ts = np.concatenate(tss); bbw = np.concatenate(bbs)
    o = np.argsort(ts, kind="mergesort")
    return X[o], y[o], ts[o], bbw[o], cols


def summ(a):
    a = np.array(a, float)
    if len(a) == 0: return {"n_paths": 0}
    return {"n_paths": int(len(a)), "mean": round(float(a.mean()), 4), "p10": round(float(np.percentile(a, 10)), 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
            "frac_clear_BE": round(float((a >= BE).mean()), 3)}


def run_session(X0, y0, ts0, bbw0, cols, sess):
    m = session_mask(ts0, sess)
    X, y, ts, bbw = X0[m], y0[m], ts0[m], bbw0[m]            # STRICT: train+val+test all session-only
    n = len(y)
    hb(f"[{sess}] session-only pooled n={n:,} up-rate={y.mean():.4f}")
    if n < 30_000:
        return {"session": sess, "n": int(n), "note": "too few session rows", "combined": summ([]), "UP": summ([]), "DOWN": summ([])}
    edges = np.linspace(0, n, N_GROUPS + 1).astype(int); grp = [(edges[i], edges[i + 1]) for i in range(N_GROUPS)]
    embargo = HOR * 60; purge = HOR * 60 + embargo
    accs_c, accs_u, accs_d = [], [], []
    for pi, testg in enumerate(combinations(range(N_GROUPS), K_TEST)):
        te = np.zeros(n, bool)
        for gi in testg: te[grp[gi][0]:grp[gi][1]] = True
        tr = ~te
        for gi in testg:
            a, b = ts[grp[gi][0]], ts[grp[gi][1] - 1]
            tr &= ~((ts >= a - purge) & (ts <= b + embargo))
        tri = np.where(tr)[0]
        if len(tri) < 5000: continue
        cut = tri[int(len(tri) * 0.8)]; ct = ts[cut]
        fit = tri[ts[tri] < ct]; val = tri[ts[tri] >= ct]
        if len(fit) > SUB_FIT: fit = fit[np.linspace(0, len(fit) - 1, SUB_FIT).astype(int)]
        if len(val) < 200: continue
        P = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.03, num_leaves=127,
                               min_child_samples=400, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                               reg_lambda=20, n_estimators=800, n_jobs=LGB_JOBS, verbosity=-1).fit(X[fit], y[fit])
        pv = P.predict_proba(X[val])[:, 1]
        best = None
        for q in (10, 20, 33):
            bthr = np.nanpercentile(bbw[fit], q); gmask = (bbw[val] <= bthr)        # NY subsumed by session pre-filter
            if gmask.sum() < 100: continue
            for cov in (0.05, 0.10):
                cthr = np.quantile(np.abs(pv[gmask] - 0.5), 1 - cov); sel = gmask & (np.abs(pv - 0.5) >= cthr)
                if sel.sum() < 100: continue
                acc = ((pv[sel] > 0.5).astype(int) == y[val][sel]).mean()
                if best is None or acc > best[0]: best = (acc, bthr, cthr, q, cov)
        if best is None: continue
        _, bthr, cthr, q, cov = best
        pt = P.predict_proba(X[te])[:, 1]; tst = ts[te]
        gt = (bbw[te] <= bthr) & (np.abs(pt - 0.5) >= cthr)
        sel = MX.nonoverlap_chrono(tst, gt, GAP_S)
        if len(sel) < 10: continue
        pred = (pt[sel] > 0.5).astype(int); yt = y[te][sel]
        accs_c.append((pred == yt).mean())
        su = pred == 1; sd = pred == 0
        if su.sum() >= MIN_SIDE_N: accs_u.append(float((pred[su] == yt[su]).mean()))
        if sd.sum() >= MIN_SIDE_N: accs_d.append(float((pred[sd] == yt[sd]).mean()))
    out = {"session": sess, "n": int(n), "gate_feat": GATE_FEAT, "breakeven": BE,
           "combined": summ(accs_c), "UP": summ(accs_u), "DOWN": summ(accs_d),
           "all_paths": {"combined": accs_c, "UP": accs_u, "DOWN": accs_d}}
    for s in ("UP", "DOWN"):
        r = out[s]
        out[s]["CERT"] = bool(r.get("n_paths", 0) >= 5 and r.get("p10", 0) >= BE and r.get("frac_clear_BE", 0) >= 0.80)
    hb(f"[{sess}] UP {out['UP']} | DOWN {out['DOWN']}")
    return out


def main():
    hb(f"CROSS-PAIR session-only @MX_HOR={HOR} gate={GATE_FEAT} jobs={LGB_JOBS}")
    X, y, ts, bbw, cols = build_pooled()
    hb(f"pooled built n={len(y):,} feats={len(cols)}")
    for sess in SESSIONS:
        out = run_session(X, y, ts, bbw, cols, sess)
        json.dump(out, open(f"session_xpair_{HOR}m_{sess}_result.json", "w"), indent=1)
        hb(f"[{sess}] -> session_xpair_{HOR}m_{sess}_result.json "
           f"UP_p10={out['UP'].get('p10')} cert={out['UP'].get('CERT')} | DOWN_p10={out['DOWN'].get('p10')} cert={out['DOWN'].get('CERT')}")
    hb(f"DONE HOR={HOR}")


if __name__ == "__main__":
    main()
