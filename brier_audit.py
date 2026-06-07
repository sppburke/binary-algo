"""BRIER-ADVANTAGE AUDIT of the certified cross-pair DIRECTION books (EURUSD.m5xp/m10xp/m15xp/m30xp.v1).

WHY: our certifications rank on AUC + deriv-faithful selective accuracy (win-rate) — both reward RANKING, not
CALIBRATION. A book can clear the 0.541 win-rate yet still be a WORSE probability forecast than a naive trailing
up-rate. The one metric that catches that is the Brier-advantage = baseline_Brier - model_Brier (>0 = the model's
probabilities carry genuine skill on top of the baseline). Imported lever from evan-kolberg/prediction-market-
backtesting (the only transferable idea from the 11-repo Polymarket scour, 2026-06-07); verified absent in-repo
(`brier` token appears nowhere; fwd_holdout emitted only AUC+selacc).

DESIGN (deployment-faithful, mirrors frac_direction.py's anchor):
  - model   = the certified gate config mk_lgb(600), trained ONCE on <= 2023, NY session, the SAME proxy the
              campaign's forward-holdout deployability calls used (NOT the exact frozen 3000-tree artifact — caveat).
  - base    = certified xp book features (m5_xpair.build_xp) at MX.HOR = {5,10,15,30}.
  - calibrate = OOF isotonic fit on TRAIN ONLY (so Brier doesn't unfairly punish a well-ranked-but-miscalibrated model).
  - baselines = flat-0.5 and an OBSERVABILITY-SAFE trailing persistence up-rate (a past bar contributes only once its
                outcome is known, i.e. gap = horizon). Reported per forward year 2024/25/26, all-bars AND the bet-tail.

FALSIFIER (pre-registered): a certified book carries genuine probabilistic skill iff badv_persist > 0 in EVERY forward
year. If any forward year is <= 0, that book's AUC/selacc edge is RANKING-ONLY (no probabilistic skill over a naive
baseline) and its certification downgrades to ranking/calibration-dependent. This audit can only ever TIGHTEN a cert.

Run: ~/binary-algo-venv/bin/python brier_audit.py [HORS=5,10,15,30]   ->  brier_audit_result.json
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from fwd_holdout import forward_holdout
import m5_xpair as MX

FEAT = MX.FEAT; PAIRS = MX.PAIRS
YEARS = list(range(2012, 2027)); TRAIN_MAX = 2023
T0 = time.time()
def hb(m): print(f"[{time.time()-T0:.0f}s] {m}", flush=True)


def build_base(HOR):
    """Build the certified cross-pair xp book matrix for one horizon (NY decision bars), exactly as frac_direction.py."""
    MX.HOR = HOR; MX.GAP_S = HOR * 60                              # build_xp binds HOR at import -> override module attrs
    parts = []
    for y in YEARS:
        try:
            bx = MX.build_xp([str(y)])
        except Exception as e:
            hb(f"  {y}: skip ({type(e).__name__}: {e})"); continue
        if bx is None or len(bx) == 0:
            hb(f"  {y}: empty"); continue
        parts.append(bx); hb(f"  {y}: base {len(bx):,}")
    B = pd.concat(parts); B = B[~B.index.duplicated(keep="last")].sort_index()
    xpc = [c for c in MX.xp_cols(B) if c in B.columns]
    ts = B["_ts"].values.astype("int64"); fwd = B["_fwd"].values; ny = B["sess_ny"].values > 0.5
    Xb = B[xpc].astype(np.float32).values
    fin = np.isfinite(Xb).all(1) & np.isfinite(fwd) & (fwd != 0) & ny
    return Xb[fin], fwd[fin], ts[fin], len(xpc)


KEEP = ("n", "auc", "selacc", "n_sel", "brier", "brier_raw", "brier_flat", "brier_persist",
        "badv_flat", "badv_persist", "badv_persist_raw", "badv_persist_sel", "n_sel_brier")


def main():
    hors = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["5", "10", "15", "30"])]
    allres = {}
    for HOR in hors:
        hb(f"===== HOR={HOR}m : building certified base xp book =====")
        Xb, fwd, ts, nf = build_base(HOR)
        hb(f"HOR={HOR}: NY decision bars n={len(fwd):,} up-rate={(fwd>0).mean():.4f} feats={nf}; running forward holdout + brier ...")
        res = forward_holdout({"base": Xb}, target=fwd, ts=ts, mode="direction", train_max=TRAIN_MAX,
                              test_years=(2024, 2025, 2026), cov=0.10, brier=True, persist_gap_s=HOR * 60.0, verbose=True)
        ba = res["by_arm"]["base"]
        yrs = [y for y in ("2024", "2025", "2026") if y in ba and "badv_persist" in ba[y]]
        badv = {y: ba[y]["badv_persist"] for y in yrs}
        badv_sel = {y: ba[y].get("badv_persist_sel") for y in yrs}
        skill = bool(len(yrs) == 3 and all(v > 0 for v in badv.values()))
        skill_sel = bool(len(yrs) == 3 and all((badv_sel[y] is not None and badv_sel[y] > 0) for y in yrs))
        allres[f"{HOR}m"] = {
            "book": f"EURUSD.m{HOR}xp.v1", "n_feats": nf, "n_bars": int(len(fwd)),
            "by_year": {y: {k: ba[y][k] for k in KEEP if k in ba[y]} for y in yrs},
            "badv_persist_by_year": badv, "badv_persist_sel_by_year": badv_sel,
            "probabilistic_skill_vs_persistence_allbars": skill,
            "probabilistic_skill_vs_persistence_bettail": skill_sel}
        hb(f"HOR={HOR}: badv_persist(all)={ {y: round(v,5) for y,v in badv.items()} } skill_all={skill} | "
           f"badv_persist(bet-tail)={ {y: (round(v,5) if v is not None else None) for y,v in badv_sel.items()} } skill_tail={skill_sel}")
    out = {
        "design": "Brier-advantage audit of certified cross-pair direction books. model=certified gate config mk_lgb(600) "
                  "trained once <=2023, NY, cov0.10; calibration=OOF isotonic fit on train only; baselines=flat-0.5 & "
                  "observability-safe trailing persistence up-rate (gap=horizon). POSITIVE badv = model Brier beats baseline.",
        "falsifier": "a certified book carries genuine probabilistic skill iff badv_persist>0 in EVERY forward year "
                     "2024/25/26; any year <=0 => the AUC/selacc edge is ranking-only and the cert downgrades to "
                     "ranking/calibration-dependent. Audit can only tighten, never inflate, a certification.",
        "caveat": "uses the forward-holdout gate model mk_lgb(600) — the same proxy the campaign's deployability calls "
                  "used — NOT the exact frozen 3000-tree book artifact; persistence baseline is a naive trailing up-rate; "
                  "bet-tail Brier-advantage selects bars by MODEL confidence so it is informative but mildly model-favoring.",
        "results": allres}
    json.dump(out, open("/media/sean/CORSAIR/binary-algo/brier_audit_result.json", "w"), indent=1)
    hb("DONE -> brier_audit_result.json")


if __name__ == "__main__":
    main()
