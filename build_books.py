"""build_books.py — ONE-TIME retrofit: freeze the existing deliverable books into the version-controlled
books/ registry with provenance manifests. Run: ~/binary-algo-venv/bin/python build_books.py

Scope = the FROZEN DELIVERABLE books referenced in EURUSD_RESULTS.md (the survivors), NOT the ~50 killed/null
experiments (those are adequately captured by their *_result.json + pre-registered falsifiers). Going forward,
new production scripts should call manifest.freeze(...) at train time instead of being retrofitted here.

Each book: copy its trained-model artifacts into books/<id>/ + write books/<id>.manifest.json (sha256s, git sha,
hyperparams, feature/data fingerprint, captured env versions, metrics). The persisted artifact is the verbatim
model (GBMs are not bit-for-bit retrainable here); the manifest is its birth certificate.
"""
import os, json
import manifest as M

ROOT = M.ROOT
MODELS = os.path.join(ROOT, "models")
BOOKS = os.path.join(ROOT, "books")
FEAT = os.path.join(ROOT, "features")          # bar features (EURUSD_<year>.parquet)
FEAT_TICK = os.path.join(ROOT, "features_tick")  # 1s micro cache (<split>_1s.parquet)
CREATED = "2026-06-01"  # fixed so manifests are stable across regenerations

FP_BAR = M.dir_fingerprint(FEAT, ("EURUSD_*.parquet",))
FP_TICK = M.dir_fingerprint(FEAT_TICK, ("*_1s.parquet",))


def m(name):
    return os.path.join(MODELS, name)


# ---- the registry: one entry per frozen deliverable book (keyed exactly as EURUSD_RESULTS.md cells) ----
REGISTRY = [
    dict(id="EURUSD.tick3.v1", timeframe="1-5s", side="combined", role="direction", script="m_tick_prod.py",
         summary="Tick microstructure 3s GBM ensemble (lgb+xgb+cat) — the one genuine >0.65 directional edge (needs a tick venue).",
         metrics={"oos_3s_selective": 0.657, "alt_3s": 0.667, "note": "TEST+OOS robust; venue floor deriv=15m"},
         arts=[m("mtick3_EURUSD_lgb.txt"), m("mtick3_EURUSD_xgb.json"), m("mtick3_EURUSD_cat.cbm"), m("mtick3_EURUSD_strategy.json")],
         strategy="models/mtick3_EURUSD_strategy.json", fp=FP_TICK,
         hyper=dict(ensemble="lgb+xgb+cat", lgb="lr0.03,num_leaves300,n_est3000", xgb="lr0.03,max_depth8,n_est2000",
                    cat="lr0.03,depth8,iter2000", horizon_s=3, entry_lag_s=1)),
    dict(id="EURUSD.tick5.v1", timeframe="1-5s", side="combined", role="direction", script="m_tick_prod.py",
         summary="Tick microstructure 5s GBM ensemble (lgb+xgb+cat); HS=5s chosen for OOS robustness (n=377-1124 cov0.5-2%).",
         metrics={"oos_5s_selective": "~0.65-0.66"},
         arts=[m("mtick5_EURUSD_lgb.txt"), m("mtick5_EURUSD_xgb.json"), m("mtick5_EURUSD_cat.cbm"), m("mtick5_EURUSD_strategy.json")],
         strategy="models/mtick5_EURUSD_strategy.json", fp=FP_TICK,
         hyper=dict(ensemble="lgb+xgb+cat", lgb="lr0.03,num_leaves300,n_est3000", xgb="lr0.03,max_depth8,n_est2000",
                    cat="lr0.03,depth8,iter2000", horizon_s=5, entry_lag_s=1)),
    dict(id="EURUSD.min1.v1", timeframe="60s", side="combined", role="direction", script="min1_production.py",
         summary="Frozen 60s book: 3-model GBM ensemble (lgb+xgb+cat) + compression-release LGBM specialist (50/50), reversion-filtered, comp-release gated. Up-only FILTER on this = the (60s,UP) key.",
         metrics={"test_2024_25": 0.539, "oos_2026": 0.550, "up_filter_2026": 0.613, "up_filter_2025": 0.584,
                  "down_2026": 0.516, "breakeven": 0.541},
         arts=[m("min1_EURUSD_direction_lgb.txt"), m("min1_EURUSD_direction_xgb.json"), m("min1_EURUSD_direction_cat.cbm"),
               m("min1_EURUSD_direction_spec_lgb.txt"), m("min1_EURUSD_magnitude.joblib"), m("min1_EURUSD_strategy.json")],
         strategy="models/min1_EURUSD_strategy.json", fp=FP_TICK,
         hyper=dict(ensemble="lgb+xgb+cat + compression-release lgb specialist (w_spec0.5)",
                    lgb="lr0.02,num_leaves350,n_est4000", spec_lgb="lr0.01,num_leaves512,n_est6000",
                    xgb="lr0.02,max_depth9", cat="lr0.02,depth9", horizon_s=60, gap_s=70, entry_lag_s=1,
                    gate="bbw1800<=q67 & rel_ratio>=p80 & reversion(-sign ret300) & conf>=thr(5%cov)")),
    dict(id="EURUSD.min2.v1", timeframe="120s", side="combined", role="direction", script="min2_production.py",
         summary="Frozen 120s book (analog of min1 at HS=120): 3-model ensemble + specialist, reversion-filtered.",
         metrics={"test": 0.528, "oos_2026": 0.539, "breakeven": 0.541, "note": "not a row in EURUSD_RESULTS; included for completeness"},
         arts=[m("min2_EURUSD_dir_v1_lgb.txt"), m("min2_EURUSD_dir_v1_xgb.json"), m("min2_EURUSD_dir_v1_cat.cbm"),
               m("min2_EURUSD_dir_spec_lgb.txt"), m("min2_EURUSD_magnitude.joblib"), m("min2_EURUSD_strategy.json")],
         strategy="models/min2_EURUSD_strategy.json", fp=FP_TICK,
         hyper=dict(ensemble="lgb+xgb+cat + specialist", lgb="lr0.02,num_leaves350", spec_lgb="lr0.01,num_leaves512",
                    horizon_s=120, gap_s=130, entry_lag_s=1)),
    dict(id="EURUSD.m5.v1", timeframe="5m", side="combined", role="direction", script="m5_production.py",
         summary="5m base OHLCV 3-model GBM ensemble (lgb+xgb+cat) on 239 multi-TF features.",
         metrics={"val_auc": 0.523, "oos_auc": 0.518, "book_combined": "0.586-0.594"},
         arts=[m("m5_EURUSD_direction_lgb.txt"), m("m5_EURUSD_direction_xgb.json"), m("m5_EURUSD_direction_cat.cbm"),
               m("m5_EURUSD_strategy.json")],
         strategy="models/m5_EURUSD_strategy.json", fp=FP_BAR,
         hyper=dict(ensemble="lgb+xgb+cat", horizon_min=5, features=239)),
    dict(id="EURUSD.m5xp.v1", timeframe="5m", side="combined", role="direction", script="m5_xpair_production.py",
         summary="5m PRODUCTION freeze: cross-pair USD-residual + order-flow, primary lgb + meta-labeler lgb.",
         metrics={"combined": 0.583, "oos_2026": 0.606, "ci": [0.570, 0.595], "ev_R0.85": 0.078, "breakeven": 0.541},
         arts=[m("m5xp_EURUSD_primary_lgb.txt"), m("m5xp_EURUSD_meta_lgb.txt"), m("m5xp_EURUSD_strategy.json")],
         strategy="models/m5xp_EURUSD_strategy.json", fp=FP_BAR,
         hyper=dict(primary_lgb="lr0.02,num_leaves127,n_est3000", meta_lgb="lr0.02,num_leaves15,n_est400",
                    horizon_min=5, features="239 base + cross-pair + 18 order-flow")),
    dict(id="EURUSD.m5stack.v1", timeframe="5m", side="combined", role="stack-meta", script="m5_stack2.py",
         summary="5m BEST: cross-horizon soft stack — front-load the 15m-parent ensemble's direction into the 5m outcome, meta-gated.",
         metrics={"combined_verifiable": 0.613, "oos_2026_frozen_q0.98": 0.571, "thin_q0.99": 0.648, "breakeven": 0.541},
         arts=[m("m5stack_EURUSD_meta_lgb.txt"), m("m5stack_EURUSD_strategy.json")],
         strategy="models/m5stack_EURUSD_strategy.json", fp=FP_BAR,
         hyper=dict(meta_lgb="lr0.02,num_leaves15,n_est400", horizon_min=5),
         depends_on=["EURUSD.m15.v1 (parent ensemble front-loaded)", "EURUSD.m5.v1 (5m base)"]),
    dict(id="EURUSD.m10.v1", timeframe="10m", side="combined", role="direction", script="m10_freeze_honest.py",
         summary="10m HONEST freeze: native-10 3-model GBM ensemble, gated by 5m_bb_width x NY (cov10%).",
         metrics={"combined": 0.602, "oos_2026": 0.594, "floor_2025": 0.579, "ci": [0.582, 0.621], "ev_R0.85": 0.113, "breakeven": 0.541},
         arts=[m("m10_EURUSD_direction_lgb.txt"), m("m10_EURUSD_direction_xgb.json"), m("m10_EURUSD_direction_cat.cbm"),
               m("m10_EURUSD_strategy.json"), m("m10_EURUSD_strategy_honest.json")],
         strategy="models/m10_EURUSD_strategy_honest.json", fp=FP_BAR,
         hyper=dict(ensemble="lgb+xgb+cat", lgb="lr0.02,num_leaves255", horizon_min=10,
                    gate="5m_bb_width<=q20 & sess_ny, cov10%")),
    dict(id="EURUSD.m15.v1", timeframe="15m", side="combined", role="direction", script="m15_production.py",
         summary="15m DELIVERABLE (best combined book in program): 3-model GBM ensemble (lgb+xgb+cat) gated by 15m compression(bb_width) x NY, conf-selective.",
         metrics={"combined_recent": 0.647, "ci": [0.612, 0.684], "oos_2026": 0.663, "test_2025": 0.582, "test_2024": 0.689,
                  "cpcv_faithful_mean": 0.579, "cpcv_p10": 0.557, "breakeven": 0.541},
         arts=[m("m15_EURUSD_direction_lgb.txt"), m("m15_EURUSD_direction_xgb.json"), m("m15_EURUSD_direction_cat.cbm"),
               m("m15_EURUSD_strategy.json")],
         strategy="models/m15_EURUSD_strategy.json", fp=FP_BAR,
         hyper=dict(ensemble="lgb+xgb+cat", lgb="lr0.02,num_leaves255,n_est3000", xgb="lr0.02,max_depth8,n_est2000",
                    cat="lr0.02,depth8,iter2000", horizon_min=15, gate="15m_bb_width<=q{10,20,33} & sess_ny, VAL-tuned cov")),
    dict(id="EURUSD.m30.v1", timeframe="30m", side="combined", role="direction", script="m30_production.py",
         summary="30m DELIVERABLE: 3-model GBM ensemble gated by 1h-compression x NY, conf-selective.",
         metrics={"combined": 0.591, "ci": [0.556, 0.625], "oos_2026": 0.546, "test_2025": 0.589, "test_2024": 0.623,
                  "ev_R0.85": 0.064, "breakeven": 0.541},
         arts=[m("m30_EURUSD_direction_lgb.txt"), m("m30_EURUSD_direction_xgb.json"), m("m30_EURUSD_direction_cat.cbm"),
               m("m30_EURUSD_strategy.json")],
         strategy="models/m30_EURUSD_strategy.json", fp=FP_BAR,
         hyper=dict(ensemble="lgb+xgb+cat", lgb="lr0.02,num_leaves255,n_est3000", xgb="lr0.02,max_depth8,n_est2000",
                    horizon_min=30, gate="1h-compression x sess_ny, cov5%")),
]


def main():
    os.makedirs(BOOKS, exist_ok=True)
    index = []
    for b in REGISTRY:
        missing = [a for a in b["arts"] if not os.path.exists(a)]
        if missing:
            print(f"[skip] {b['id']}: missing artifacts {[os.path.basename(x) for x in missing]}", flush=True)
            continue
        man = M.build(b["id"], timeframe=b["timeframe"], side=b["side"], role=b["role"], script=b["script"],
                      summary=b["summary"], metrics=b["metrics"], artifacts=b["arts"], hyperparams=b["hyper"],
                      strategy_json=b.get("strategy"), feature_fingerprint=b["fp"], depends_on=b.get("depends_on"),
                      created_utc=CREATED)
        path = M.freeze(man, artifacts_src=b["arts"], books_dir=BOOKS)
        nbytes = sum(a["bytes"] for a in man["artifacts"])
        print(f"[ok] {b['id']:<20} {b['timeframe']:<5} {len(man['artifacts'])} arts {nbytes/1e6:.1f}MB -> {os.path.relpath(path, ROOT)}", flush=True)
        index.append({"id": b["id"], "currency": "EURUSD", "timeframe": b["timeframe"], "side": b["side"],
                      "role": b["role"], "script": b["script"], "metrics": b["metrics"],
                      "manifest": f"books/{b['id']}.manifest.json", "summary": b["summary"],
                      "depends_on": b.get("depends_on")})
    with open(os.path.join(BOOKS, "INDEX.json"), "w") as f:
        json.dump({"schema": "book-index/v1", "git_sha": M.git_sha(), "created_utc": CREATED,
                   "env": M.env_snapshot(), "books": index}, f, indent=2)
    print(f"\n[done] {len(index)} books frozen -> books/INDEX.json (git_sha {M.git_sha()[:8]})", flush=True)


if __name__ == "__main__":
    main()
