"""Pipeline step (d): CONFIDENCE/COVERAGE curve for the frozen 30m cross-pair book (EURUSD.m30xp.v1), per side,
per forward year. Maps how per-side accuracy + independent-trade count trade off against coverage (conf threshold),
so the deploy operating point is chosen with the tradeoff visible (not a single cov). Frozen-book (no refit) — this
describes the DEPLOYED book's forward behaviour; the CERTIFICATION remains the refit-CPCV (m30_xpair_cpcv.py).

Gate = 1h_bb_width<=bthr (frozen comp_q) AND sess_ny, then top-(cov) by |p-0.5| among gated, nonoverlap 1800s,
split by predicted side. Reports per (year, side, cov) acc + n + CI95.
Usage: python m30_xpair_covcurve.py
"""
import os
os.environ["MX_HOR"] = "30"
import json, numpy as np
import lightgbm as lgb
import m5_xpair as MX

MODELS = "/media/sean/CORSAIR/binary-algo/models"
GATE_FEAT = "1h_bb_width"
BE = 0.541
COVS = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]


def main():
    strat = json.load(open(f"{MODELS}/m30xp_EURUSD_strategy.json"))
    cols = strat["primary_feats"]; bthr = strat["bb_width_thr"]
    B = lgb.Booster(model_file=f"{MODELS}/m30xp_EURUSD_primary_lgb.txt")
    out = {"book": "EURUSD.m30xp.v1", "gate_feat": GATE_FEAT, "bb_width_thr": bthr, "breakeven": BE, "covs": COVS, "curve": {}}
    for w, yr in (("test24", "2024"), ("test25", "2025"), ("oos", "2026")):
        D = MX.augment(MX.build_xp(MX.SPL[w]), MX.SPL[w], "xpof")
        pr = B.predict(D[cols].astype("float32").values); y = D["_y"].astype(int).values
        ts = D["_ts"].values.astype("int64"); bbw = D[GATE_FEAT].values.astype(float); ny = D["sess_ny"].values > 0.5
        gate = (bbw <= bthr) & ny; conf = np.abs(pr - 0.5)
        out["curve"][yr] = {}
        for cov in COVS:
            if gate.sum() < 50:
                continue
            cthr = float(np.quantile(conf[gate], 1 - cov)); m = gate & (conf >= cthr)
            sel = MX.nonoverlap_chrono(ts, m); pred = (pr[sel] > 0.5).astype(int); yy = y[sel]
            row = {}
            for side, nm in ((1, "UP"), (0, "DOWN"), (None, "COMB")):
                ss = (pred == side) if side is not None else np.ones(len(pred), bool)
                if ss.sum() < 5:
                    row[nm] = {"n": int(ss.sum()), "acc": None}; continue
                corr = (pred[ss] == yy[ss]).astype(float); lo, hi = MX.boot(corr)
                row[nm] = {"n": int(ss.sum()), "acc": round(float(corr.mean()), 4), "ci_lo": round(lo, 4)}
            out["curve"][yr][f"cov{cov}"] = row
            u = row["UP"]; d = row["DOWN"]
            print(f"  {yr} cov{cov:.0%}: UP n{u['n']} acc={u['acc']} | DOWN n{d['n']} acc={d['acc']}", flush=True)
        del D
    json.dump(out, open("m30_xpair_covcurve_result.json", "w"), indent=1)
    print("[covcurve30] DONE -> m30_xpair_covcurve_result.json", flush=True)


if __name__ == "__main__":
    main()
