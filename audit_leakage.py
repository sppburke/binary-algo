"""Leakage / methodology self-test for the production binary pipelines. RUN BEFORE ANY DEPLOY:

    ~/binary-algo-venv/bin/python audit_leakage.py

Exits non-zero if any invariant fails. These are the guards established by the 2026-05 bias audit
(see research_log "BIAS AUDIT"). They assert that the numbers reported by the production backtests are
not artifacts of look-ahead, a mislabeled horizon, split contamination, or overlapping (non-independent)
trades. Cheap to run; treat a FAIL as a release blocker.
"""
import sys, numpy as np, pandas as pd
import min1_production as m1
import min2_production as m2

FAILS = []
def check(name, cond, detail=""):
    ok = bool(cond)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   -- {detail}" if detail else ""), flush=True)
    if not ok: FAILS.append(name)

def span(sp):
    t = pd.read_parquet(f"{m1.TICK}/{sp}_1s.parquet", columns=["mid"]).index
    t = t.values.astype("datetime64[s]").astype("int64")
    return int(t.min()), int(t.max())

# ----------------------------------------------------------------------------------
# 1) FEATURE CAUSALITY — features at row i must not depend on any row j>i.
#    Perturb the entire tail of a buffer; confirm head-row features are bit-identical.
# ----------------------------------------------------------------------------------
b = m1.load_split("val").iloc[:200_000].copy()
X0 = m1.feats(b)
cut = len(b)//2
b2 = b.copy()
for col, fac in (("mid",1.05), ("micro",1.05), ("imb",-1.0), ("spread",3.0)):
    b2.iloc[cut:, b2.columns.get_loc(col)] = b2.iloc[cut:, b2.columns.get_loc(col)].values * fac
X1 = m1.feats(b2)
head = slice(3600, cut-2)   # rows safely before the perturbation, past the longest (3600) window
same = np.array_equal(np.nan_to_num(X0.iloc[head].values, nan=0.0),
                      np.nan_to_num(X1.iloc[head].values, nan=0.0))
check("tick feats(): future rows never change past features (no look-ahead)", same)
# every transform is trailing (ewm/rolling/pct_change/shift(+)/groupby-cumcount); the only forward op is the label
check("tick feats(): same NaN mask in head under perturbation",
      np.array_equal(X0.iloc[head].isna().values, X1.iloc[head].isna().values))

# ----------------------------------------------------------------------------------
# 2) DERIV LABEL ALIGNMENT — entry = next tick after order; exit = last tick AT/BEFORE expiry; mid-to-mid;
#    true fixed wall-clock expiry (NOT a bar-count shift). Verified vs deriv T&C 2.2.1.3 / 2.2.3.1.
# ----------------------------------------------------------------------------------
ts  = b.index.values.astype("datetime64[s]").astype("int64")
mid = b["mid"].values.astype(float); n = len(ts)
for mod, HS in ((m1,60), (m2,120)):
    ret, valid = mod.wc_ret(ts, mid, HS, mod.TOL_S, 0)        # lag=0 here for a clean horizon check
    ei = np.searchsorted(ts, ts,    side="left")              # entry = first tick at/after order
    xi = np.searchsorted(ts, ts+HS, side="right") - 1         # exit  = last tick AT/BEFORE expiry (deriv)
    eic = np.clip(ei,0,n-1); xic = np.clip(xi,0,n-1); v = valid
    exit_gap = (ts+HS) - ts[xic]                              # how far the exit tick sits BEFORE the expiry instant
    check(f"{mod.__name__}: exit tick within (expiry-{mod.TOL_S}s, expiry] for ALL valid rows (deriv: latest tick at/before end)",
          bool(v.any()) and bool(np.all((exit_gap[v] >= 0) & (exit_gap[v] <= mod.TOL_S))),
          f"max_exit_gap={int(exit_gap[v].max())}s over {int(v.sum()):,} valid rows")
    check(f"{mod.__name__}: label == sign(mid(exit) - mid(entry)) on valid rows (ties allowed as losses)",
          bool(np.all((ret[v] > 0) == (mid[xic][v] > mid[eic][v]))))
    wc = (ts[xic] - ts[eic])[v]
    check(f"{mod.__name__}: median realized horizon == {HS}s (TRUE fixed expiry — NOT the old ~2x bar-count horizon)",
          abs(float(np.median(wc)) - HS) <= mod.TOL_S, f"median={float(np.median(wc)):.0f}s p90={float(np.percentile(wc,90)):.0f}s")
    # the old bug, for contrast: bar-count shift spans ~2x wall-clock
    old_wc = (np.roll(ts, -HS) - ts)[:n-HS]
    check(f"{mod.__name__}: (regression guard) old bar-count shift would span > {HS}s — confirms the bug existed",
          float(np.median(old_wc)) > HS, f"old_barcount_median={float(np.median(old_wc)):.0f}s")

# ----------------------------------------------------------------------------------
# 3) SPLIT TEMPORAL DISJOINTNESS — train < val < test < oos, no overlap (tick pipeline).
# ----------------------------------------------------------------------------------
tr, va, te, oo = span("train"), span("val"), span("test"), span("oos")
check("tick splits are time-ordered and disjoint: train < val < test < oos",
      tr[1] < va[0] and va[1] < te[0] and te[1] < oo[0],
      f"train_end={np.datetime64(tr[1],'s')} val=[{np.datetime64(va[0],'s')}..] test=[{np.datetime64(te[0],'s')}..] oos=[{np.datetime64(oo[0],'s')}..]")

# ----------------------------------------------------------------------------------
# 4) NON-OVERLAP INDEPENDENCE — selected trades' [t, t+HS] outcome windows are disjoint.
# ----------------------------------------------------------------------------------
rng = np.random.default_rng(0)
for mod, HS in ((m1,60), (m2,120)):
    tt = ts[:50_000]; conf = rng.random(len(tt))
    sel = mod.nonoverlap(tt, conf, 0.0)
    d = np.diff(np.sort(tt[sel]))
    check(f"{mod.__name__}: non-overlap selection spaces trades >= horizon ({HS}s) apart -> independent windows",
          bool(len(d) == 0 or np.all(d >= HS)), f"min_spacing={int(d.min()) if len(d) else 'NA'}s gap_const={mod.GAP}s")

# ----------------------------------------------------------------------------------
# 5) 15m pipeline — label horizon is exact wall-clock (contiguity enforced) & eval is independence-aware.
# ----------------------------------------------------------------------------------
try:
    import m15_production as m15
    D = m15.load(["2026"])
    tsd = D.index.values.astype("datetime64[s]").astype("int64")
    sel = m15.nonoverlap(tsd, np.ones(len(tsd)), 0.0)
    d = np.diff(np.sort(tsd[sel]))
    check("m15: non-overlap spaces trades >= 15min apart (independent windows)",
          bool(len(d) == 0 or np.all(d >= m15.GAP_S)), f"gap_const={m15.GAP_S}s")
    check("m15: GAP_S == HOR*60 (label horizon in seconds)", m15.GAP_S == m15.HOR*60)
except Exception as e:
    check("m15 pipeline self-test ran", False, f"exception: {e}")

print("\n" + ("ALL CHECKS PASSED ✅" if not FAILS else f"❌ FAILURES ({len(FAILS)}): " + "; ".join(FAILS)), flush=True)
sys.exit(1 if FAILS else 0)
