"""Freeze the ACI adaptive-conformal-gate improvement as a registry book: EURUSD.m5xp_aci.v1.
It is the m5xp UP book (frozen primary+meta) PLUS the online adaptive-conformal gate policy that targets a
selective win-rate w*, trading more in-regime and less off-regime (causal; no look-ahead). Deployable
improvement over the fixed gate: more coverage + tighter cross-regime spread at a stable-or-better binding-year
win-rate. Metrics recorded honestly (within-experiment vs fixed; pending CPCV-validation of the policy)."""
import json, os, time
import manifest

ROOT = "/home/sean/git/binary-algo"
m5xp = json.load(open(f"{ROOT}/models/m5xp_EURUSD_strategy.json"))
conf = json.load(open(f"{ROOT}/m5_conformal_result.json"))
aci57 = conf["ACI"]["wstar_0.57"]; fixed = conf["FIXED"]

# --- the ACI policy spec (the "strategy.json" for this book) ---
aci_strategy = {
    "book": "EURUSD.m5xp_aci.v1", "base_book": "EURUSD.m5xp.v1", "horizon_s": 300, "side": "up",
    "candidate_filter": "sess_ny & primary_pred>0.5 (m5xp primary)",
    "gate": "ADAPTIVE-CONFORMAL (ACI, Gibbs-Candes 2021): trade UP candidate t iff meta_sm_t >= theta_t; "
            "online update theta_{t+1}=theta_t+gamma*(err_t - (1-wstar)) on TRADED bars only (causal).",
    "aci_params": {"wstar": 0.57, "gamma": 0.02, "theta0": "median(meta_sm)",
                   "theta_bounds": "[quantile(meta_sm,0.05), quantile(meta_sm,0.995)]"},
    "fixed_gate_meta_thr": m5xp["meta_thr"], "breakeven": 0.541,
    "meta_score": "m5xp meta-labeler P(primary correct); same artifacts as EURUSD.m5xp.v1",
    "settlement": "deriv-faithful book-native (contiguous-300s, _y moved-only, nonoverlap_chrono 300s)",
}
strat_path = f"{ROOT}/models/m5xp_aci_EURUSD_strategy.json"
json.dump(aci_strategy, open(strat_path, "w"), indent=2)

metrics = {
    "deployable": "UP, adaptive-conformal gate w*=0.57",
    "aci_up_per_year_2024_2025_2026": [aci57["2024"]["win"], aci57["2025"]["win"], aci57["2026"]["win"]],
    "aci_up_n_per_year": [aci57["2024"]["n"], aci57["2025"]["n"], aci57["2026"]["n"]],
    "fixed_up_per_year": [fixed["2024"]["win"], fixed["2025"]["win"], fixed["2026"]["win"]],
    "fixed_up_n_per_year": [fixed["2024"]["n"], fixed["2025"]["n"], fixed["2026"]["n"]],
    "binding_2025": {"aci": aci57["2025"]["win"], "aci_n": aci57["2025"]["n"],
                     "fixed": fixed["2025"]["win"], "fixed_n": fixed["2025"]["n"]},
    "coverage_total": {"aci": aci57["TOTAL_n"], "fixed": fixed["TOTAL_n"]},
    "breakeven": 0.541,
    "status": "IMPROVEMENT vs fixed gate (more coverage + regime-stable; binding 2025 .584 at n764 vs fixed "
              ".579 at n618). Within-experiment comparison (m5_conformal.py); CPCV-validation of the adaptive "
              "policy is the open follow-up.",
}

arts = [f"{ROOT}/models/m5xp_EURUSD_primary_lgb.txt", f"{ROOT}/models/m5xp_EURUSD_meta_lgb.txt",
        f"{ROOT}/models/m5xp_EURUSD_strategy.json", strat_path]
man = manifest.build(
    book_id="EURUSD.m5xp_aci.v1", timeframe="5m", side="up", role="gate",
    script="m5_conformal.py (policy) + m5_xpair_production.py (base book)",
    summary="m5xp UP book + ONLINE ADAPTIVE-CONFORMAL (ACI) gate targeting win-rate w*=0.57. Deployable "
            "improvement over the fixed meta gate: ~36% more trades + tighter cross-regime spread at a "
            "stable/slightly-better binding-year win-rate. Causal (past-outcome feedback, no look-ahead).",
    metrics=metrics, artifacts=arts,
    hyperparams={"aci_wstar": 0.57, "aci_gamma": 0.02, "base": "m5xp primary lgb + meta lgb (unchanged)"},
    strategy_json="models/m5xp_aci_EURUSD_strategy.json",
    feature_fingerprint=manifest.dir_fingerprint(f"{ROOT}/features_of"),
    depends_on=["EURUSD.m5xp.v1"],
    notes="The improvement is in the GATE POLICY (coverage + regime-robustness), not new model weights; the "
          "primary+meta bytes are identical to EURUSD.m5xp.v1. Found by the post-sweep edge-improvement loop "
          "(lit-review toolkit). EXP-1 magweight rebalanced (no UP win); EXP-2 seed-ensemble decorrelated but "
          "blend=GBM. ACI is the lever that worked.",
    created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
)
mpath = manifest.freeze(man, arts)
print(f"[register] froze EURUSD.m5xp_aci.v1 -> {mpath}")
print(f"[register] content_id={man['content_id']} artifacts={[a['file'] for a in man['artifacts']]}")

# update books/INDEX.json
idxp = f"{ROOT}/books/INDEX.json"
idx = json.load(open(idxp))
books = idx["books"] if isinstance(idx, dict) and "books" in idx else idx
entry = {"id": "EURUSD.m5xp_aci.v1", "currency": "EURUSD", "timeframe": "5m", "side": "up", "role": "gate",
         "script": "m5_conformal.py", "metrics": metrics, "manifest": "books/EURUSD.m5xp_aci.v1.manifest.json",
         "summary": man["summary"], "depends_on": ["EURUSD.m5xp.v1"]}
books.append(entry)
json.dump(idx, open(idxp, "w"), indent=1)
print(f"[register] appended to INDEX.json ({len(books)} books)")
