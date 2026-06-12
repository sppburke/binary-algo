"""Sweep row F1a: macro-release directional impulse at 120s. Retarget min1_news60 to HS=120 (monkeypatch).
News direction was null/sign-unstable at 60s (HIGH-vol .373/.526/.517; bigger surprise = more wrong) -> 120s is
even more priced-in. Fast-KILL falsifier: no vol-tier x |surp_z| window clears breakeven 0.541 stably."""
import min1_production as MP
MP.HS = 120
import min1_news60
min1_news60.main()
