"""Comprehensive positive & negative safety test suite for the SOTA Deriv VPS runtime.

Verifies every DO-NOT-SHIP invariant before any book is frozen or traded:
1. Quote parser & dual floor + calibrated-EV gate (`NaN`, `±inf`, `<=0`, `bool`, missing fields).
2. Demo (`VRTC*`) vs. Live (`CR*` + `DERIV_ALLOW_LIVE_REAL_MONEY=1`) fail-closed boundary.
3. Session, `21:00-23:59 UTC` sub-5h blackout, `16:50-H` NY rollover (`15m` & `30m`, EDT & EST), and `16h` NY haircut gates.
4. Currency-leg concentration caps (`USD` AND `JPY` simultaneously) + persisted SQLite daily-loss, drawdown, stake, open-count, and cooldown circuit breakers.
5. Pure-NumPy `nystroem` & `logit-hourly` book freeze/reload parity (`< 1e-11` vs. `sklearn`), `allowed_sides` enforcement, lifecycle gating (`inactive_candidate` / `demo_venue_settled_canary` / `active`), SHA-256 tamper detection, and symlink rejection.
6. Strict tick-level feature causality (`ts < decision_epoch`) and stale-tick rejection (`deriv_sota_tick_stream.py`).
7. End-to-end async `SotaTradeExecutor` signal -> proposal -> EV-gated buy -> settlement reconciliation -> `deriv_live_pnl_ledger.py` exact-cent promotion gate + 5-error `KILL` tripwire.
"""

from __future__ import annotations

import asyncio
import json
import math
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, SGDClassifier

from deriv_live_pnl_ledger import summarize_settled_queue
from deriv_runtime_core_v2 import (
    SOTA_DERIV_SYMBOLS,
    ExecutorError,
    after_last_start_cutoff_v2,
    assert_non_us_deriv_host,
    contract_type_for,
    currency_leg_exposure,
    evaluate_sota_multiplier_gate,
    evaluate_sota_payout_gate,
    format_barrier_levels,
    is_ny_16h_haircut_window,
    is_venue_sub5h_open,
    parse_quote_v2,
    validate_account_mode,
)
from deriv_sota_executor import (
    RiskBreakerConfig,
    SotaTradeExecutor,
    check_risk_and_exposure,
    connect_sota_queue,
    enqueue_sota_signal,
)
from deriv_sota_tick_stream import (
    MAJORS,
    SotaFeatureError,
    extract_decision_feature_row,
)
from sota_book_freeze import (
    extract_hourly_logit_arrays,
    extract_nystroem_arrays,
    extract_nystroem_ens_arrays,
    freeze_sota_book,
)
from sota_book_runtime import (
    ACTIVE_LIFECYCLE,
    DEMO_CANARY_LIFECYCLE,
    INACTIVE_LIFECYCLE,
    SotaBookError,
    load_enabled_sota_books,
    load_sota_book,
    parse_book_ids_spec,
)


def _expect_raises(exc_type: type[BaseException], fn: Any, *args: Any, **kwargs: Any) -> None:
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    raise AssertionError(f"Expected {exc_type.__name__} from {fn.__name__}, but no exception was raised")


def test_quote_parser_and_dual_ev_gate() -> None:
    for bad_val in (None, 0, -1.0, float("nan"), float("inf"), float("-inf"), True, False, "nan", "inf", "-inf", "abc", ""):
        ask, pay, be, reason = parse_quote_v2({"ask_price": bad_val, "payout": 1.85})
        assert reason == "missing_or_invalid_ask", f"failed to reject bad ask {bad_val!r}: {reason}"
        ask2, pay2, be2, reason2 = parse_quote_v2({"ask_price": 1.0, "payout": bad_val})
        assert reason2 == "missing_or_invalid_payout", f"failed to reject bad payout {bad_val!r}: {reason2}"

    # Valid quote passing floor AND calibrated EV
    dec_ok = evaluate_sota_payout_gate(
        {"ask_price": 1.0, "payout": 1.80},
        effective_floor=0.570,
        absolute_breakeven_ceiling=0.60,
        calibrated_win_prob=0.575,
        ev_margin=0.01,
        min_payout_ratio=1.75,
    )
    assert dec_ok.passed and dec_ok.reason is None
    assert abs((dec_ok.calibrated_ev or 0.0) - (0.575 * 1.80 - 1.0)) < 1e-12

    # Fails when 16:00 NY payout haircut drops payout to 1.58 (breakeven = 0.6329 > floor 0.570)
    dec_haircut = evaluate_sota_payout_gate(
        {"ask_price": 1.0, "payout": 1.58},
        effective_floor=0.570,
        absolute_breakeven_ceiling=0.60,
        calibrated_win_prob=0.59,
        ev_margin=0.005,
    )
    assert not dec_haircut.passed and dec_haircut.reason == "edge_not_positive"

    # Passes floor (0.570 > 1/1.76 = 0.56818) but fails calibrated EV margin (0.569 * 1.76 - 1 = +0.00144 < 0.005)
    dec_ev_low = evaluate_sota_payout_gate(
        {"ask_price": 1.0, "payout": 1.76},
        effective_floor=0.570,
        absolute_breakeven_ceiling=0.60,
        calibrated_win_prob=0.569,
        ev_margin=0.005,
    )
    assert not dec_ev_low.passed and dec_ev_low.reason == "calibrated_ev_below_margin"

    # Fails min_payout_ratio
    dec_min_pay = evaluate_sota_payout_gate(
        {"ask_price": 1.0, "payout": 1.74},
        effective_floor=0.585,
        absolute_breakeven_ceiling=0.60,
        calibrated_win_prob=0.59,
        ev_margin=0.005,
        min_payout_ratio=1.75,
    )
    assert not dec_min_pay.passed and dec_min_pay.reason == "payout_below_floor"

    # Fails closed when calibrated_win_prob is None (MAJOR-1) or proposal currency != USD (CRITICAL-3)
    dec_no_cal = evaluate_sota_payout_gate(
        {"ask_price": 1.0, "payout": 1.82},
        effective_floor=0.585,
        absolute_breakeven_ceiling=0.60,
        calibrated_win_prob=None,
        ev_margin=0.005,
    )
    assert not dec_no_cal.passed and dec_no_cal.reason == "missing_calibrated_probability"
    dec_eur = evaluate_sota_payout_gate(
        {"ask_price": 1.0, "payout": 1.82, "currency": "EUR"},
        effective_floor=0.585,
        absolute_breakeven_ceiling=0.60,
        calibrated_win_prob=0.60,
        ev_margin=0.005,
    )
    assert not dec_eur.passed and dec_eur.reason == "non_usd_proposal_currency"


def test_account_mode_isolation() -> None:
    # Valid Demo
    validate_account_mode(
        account_mode="demo",
        account_id="VRTC991122",
        ws_url="wss://api.derivws.com/trading/v1/options/ws/demo",
        env_live_flag="0",
    )
    # Valid Live (Options OTP /live and Classic v3)
    validate_account_mode(
        account_mode="live",
        account_id="CR887766",
        ws_url="wss://api.derivws.com/trading/v1/options/ws/live",
        env_live_flag="1",
    )
    validate_account_mode(
        account_mode="live",
        account_id="CR887766",
        ws_url="wss://ws.derivws.com/websockets/v3?app_id=1089",
        env_live_flag="1",
    )

    # Demo rejections
    _expect_raises(ExecutorError, validate_account_mode, account_mode="demo", account_id="CR887766", ws_url="wss://api.derivws.com/trading/v1/options/ws/demo", env_live_flag="0")
    _expect_raises(ExecutorError, validate_account_mode, account_mode="demo", account_id="VRTC991122", ws_url="wss://api.derivws.com/trading/v1/options/ws/live", env_live_flag="0")
    _expect_raises(ExecutorError, validate_account_mode, account_mode="demo", account_id="VRTC991122", ws_url="wss://api.derivws.com/trading/v1/options/ws/demo", env_live_flag="1")

    # Live & Demo spoofed URL rejections (CRITICAL-3: urlsplit exact host/path pinning)
    _expect_raises(ExecutorError, validate_account_mode, account_mode="live", account_id="CR887766", ws_url="wss://api.derivws.com/trading/v1/options/ws/live", env_live_flag="0")
    _expect_raises(ExecutorError, validate_account_mode, account_mode="live", account_id="VRTC991122", ws_url="wss://api.derivws.com/trading/v1/options/ws/live", env_live_flag="1")
    _expect_raises(ExecutorError, validate_account_mode, account_mode="live", account_id="CR887766", ws_url="wss://api.derivws.com/trading/v1/options/ws/demo", env_live_flag="1")
    _expect_raises(ExecutorError, validate_account_mode, account_mode="live", account_id="CR887766", ws_url="wss://evil.example/ws", env_live_flag="1")
    _expect_raises(ExecutorError, validate_account_mode, account_mode="demo", account_id="VRTC991122", ws_url="wss://evil.example/trading/v1/options/ws/demo", env_live_flag="0")
    _expect_raises(ExecutorError, validate_account_mode, account_mode="live", account_id="CR887766", ws_url="wss://evil.example/?x=ws.derivws.com", env_live_flag="1")

    # Jurisdiction fence: US workstation (`quantum` or missing mexico_vps flag) MUST fail closed!
    _expect_raises(ExecutorError, assert_non_us_deriv_host, hostname="quantum", jurisdiction_flag="mexico_vps", us_lab_marker=Path("/nonexistent_marker"))
    _expect_raises(ExecutorError, assert_non_us_deriv_host, hostname="mx-vps-01", jurisdiction_flag="", us_lab_marker=Path("/nonexistent_marker"))
    assert_non_us_deriv_host(hostname="mx-vps-01", jurisdiction_flag="mexico_vps", us_lab_marker=Path("/nonexistent_marker"))


def test_session_cutoffs_blackout_and_haircut() -> None:
    # Summer EDT (UTC-4): 16:34:40 NY = 20:34:40 UTC; 16:35:00 NY = 20:35:00 UTC
    assert not after_last_start_cutoff_v2(datetime(2026, 7, 15, 20, 34, 40, tzinfo=timezone.utc), 15)
    assert after_last_start_cutoff_v2(datetime(2026, 7, 15, 20, 35, 0, tzinfo=timezone.utc), 15)
    # 30m cutoff in EDT: 16:19:40 NY allowed, 16:20:00 NY blocked
    assert not after_last_start_cutoff_v2(datetime(2026, 7, 15, 20, 19, 40, tzinfo=timezone.utc), 30)
    assert after_last_start_cutoff_v2(datetime(2026, 7, 15, 20, 20, 0, tzinfo=timezone.utc), 30)

    # Winter EST (UTC-5): 16:00 NY = 21:00 UTC -> blocked by 21:00-23:59 UTC sub-5h venue blackout!
    assert is_venue_sub5h_open(datetime(2026, 11, 18, 20, 59, 0, tzinfo=timezone.utc), 15)
    assert not is_venue_sub5h_open(datetime(2026, 11, 18, 21, 0, 0, tzinfo=timezone.utc), 15)
    assert not is_venue_sub5h_open(datetime(2026, 11, 18, 23, 59, 59, tzinfo=timezone.utc), 30)
    assert is_venue_sub5h_open(datetime(2026, 11, 18, 22, 0, 0, tzinfo=timezone.utc), 300)

    # NY 16:00-16:35 haircut window
    assert not is_ny_16h_haircut_window(datetime(2026, 7, 15, 19, 59, 0, tzinfo=timezone.utc))
    assert is_ny_16h_haircut_window(datetime(2026, 7, 15, 20, 0, 0, tzinfo=timezone.utc))
    assert is_ny_16h_haircut_window(datetime(2026, 7, 15, 20, 34, 0, tzinfo=timezone.utc))


def test_currency_legs_and_risk_circuit_breakers() -> None:
    assert currency_leg_exposure("USDJPY", "UP") == {"USD": 1, "JPY": -1}
    assert currency_leg_exposure("EURJPY", "UP") == {"EUR": 1, "JPY": -1}
    assert currency_leg_exposure("EURUSD", "UP") == {"EUR": 1, "USD": -1}

    with tempfile.TemporaryDirectory() as td:
        conn = connect_sota_queue(Path(td) / "q.sqlite")
        now = datetime(2026, 7, 15, 14, 0, 0, tzinfo=timezone.utc)
        cfg = RiskBreakerConfig(
            max_stake_usd=5.0,
            max_daily_loss_usd=10.0,
            max_drawdown_usd=15.0,
            max_open_contracts=4,
            max_usd_factor_open=2,
            max_jpy_factor_open=2,
        )

        # Stake cap
        assert check_risk_and_exposure(conn, book_id="b1", pair="USDJPY", horizon_minutes=15, side="UP", stake_usd=6.0, cfg=cfg, now_utc=now) == "stake_exceeds_max_stake_usd"

        # Insert 2 short-JPY open contracts: USDJPY UP and EURJPY UP (opened 2.4s after bar close)
        for idx, (p, bid) in enumerate([("USDJPY", "b1"), ("EURJPY", "b2")]):
            sid = f"sig-{idx}"
            conn.execute(
                "INSERT INTO sota_signals VALUES (?,?,?,?,?,'wall',0,?,0.57,0.58,'{}','bought',NULL,?,?)",
                (sid, bid, p, 15, "UP", now.isoformat(), now.isoformat(), now.isoformat()),
            )
            conn.execute(
                "INSERT INTO sota_contracts VALUES (?,?,?,?,?,?,?,100,180,0,0,'open',?,?,NULL,'h',?,'risefall',NULL,'demo',NULL)",
                (f"c-{idx}", sid, bid, p, 15, "UP", f"p-{idx}", (now + timedelta(seconds=2.4)).isoformat(), (now + timedelta(minutes=15)).isoformat(), now.isoformat()),
            )

        # 3rd short-JPY (GBPJPY UP -> JPY:-1) MUST be blocked by max_jpy_factor_open=2!
        assert check_risk_and_exposure(conn, book_id="b3", pair="GBPJPY", horizon_minutes=15, side="UP", stake_usd=1.0, cfg=cfg, now_utc=now) == "max_jpy_factor_open_reached"
        # Opposite JPY direction (GBPJPY DOWN -> JPY:+1) is NOT blocked by JPY cap!
        assert check_risk_and_exposure(conn, book_id="b3", pair="GBPJPY", horizon_minutes=15, side="DOWN", stake_usd=1.0, cfg=cfg, now_utc=now) is None
        # Same (pair, horizon) cooldown blocks USDJPY 15m at +5m, while exact +15m bar close is NOT falsely blocked by 2.4s open lag (MAJOR-2)!
        assert check_risk_and_exposure(conn, book_id="b1", pair="USDJPY", horizon_minutes=15, side="DOWN", stake_usd=1.0, cfg=cfg, now_utc=now + timedelta(minutes=5), signal_close_utc=(now + timedelta(minutes=5)).isoformat()) == "same_pair_horizon_cooldown"
        assert check_risk_and_exposure(conn, book_id="b1", pair="USDJPY", horizon_minutes=15, side="DOWN", stake_usd=1.0, cfg=cfg, now_utc=now + timedelta(minutes=15, seconds=1), signal_close_utc=(now + timedelta(minutes=15)).isoformat()) is None
        assert check_risk_and_exposure(conn, book_id="b1_30", pair="USDJPY", horizon_minutes=30, side="DOWN", stake_usd=1.0, cfg=cfg, now_utc=now + timedelta(minutes=5), signal_close_utc=(now + timedelta(minutes=5)).isoformat()) is None

        # Settle two losing contracts (-$9.00 realized today) + $2.00 open stake + $1.00 new stake = $12.00 > $10.00 daily loss cap!
        conn.execute("INSERT INTO sota_signals VALUES ('s-loss','b4','AUDUSD',15,'UP','wall',0,?,0.57,0.58,'{}','contract_closed',NULL,?,?)", (now.isoformat(), now.isoformat(), now.isoformat()))
        conn.execute(
            "INSERT INTO sota_contracts VALUES ('c-loss','s-loss','b4','AUDUSD',15,'UP','p-loss',900,1620,0,-900,'lost',?,?,?,'h',?,'risefall',NULL,'demo',NULL)",
            (now.isoformat(), now.isoformat(), now.isoformat(), now.isoformat()),
        )
        assert check_risk_and_exposure(conn, book_id="b5", pair="NZDUSD", horizon_minutes=15, side="UP", stake_usd=1.0, cfg=cfg, now_utc=now + timedelta(minutes=20)) == "daily_loss_circuit_breaker"

        # CRITICAL-2: any buy_unknown signal immediately halts all new buys!
        conn.execute("INSERT INTO sota_signals VALUES ('s-unk','b5','NZDUSD',15,'UP','wall',0,?,0.57,0.58,'{}','buy_unknown','timeout',?,?)", (now.isoformat(), now.isoformat(), now.isoformat()))
        assert check_risk_and_exposure(conn, book_id="b5", pair="NZDUSD", horizon_minutes=15, side="UP", stake_usd=0.5, cfg=cfg, now_utc=now + timedelta(days=1)) == "unresolved_buy_uncertainty_halt"
        conn.close()


def test_book_freeze_parity_and_certificate_gates() -> None:
    rng = np.random.default_rng(42)
    fcols = [f"f_{i}" for i in range(12)]
    X = rng.normal(size=(600, 12))
    y = (X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.5, size=600) > 0).astype(int)
    df = pd.DataFrame(X, columns=fcols)

    # 1. Nystroem parity (< 1e-12)
    med = np.median(X, axis=0)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-12
    Z = np.clip((X - mu) / sd, -5.0, 5.0)
    ny = Nystroem(kernel="rbf", gamma=1.0 / X.shape[1], n_components=64, random_state=0).fit(Z)
    sgd = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=50, tol=1e-5, random_state=0).fit(ny.transform(Z), y)
    expected_nys = sgd.predict_proba(ny.transform(Z[:50]))[:, 1]
    nys_arrays = extract_nystroem_arrays(med, mu, sd, ny, sgd)

    dev_certs = {
        "G1_cpcv": {"passed": True, "p10_wr": 0.576},
        "G2_placebo": {"passed": True, "mean_excess": 0.063},
        "G3_entry_delay": {"passed": True, "d4_p10": 0.575},
        "G4_sign_null": {"passed": True, "null_max": 0.548},
        "G5_payout": {"passed": True, "p10_ev": 0.0143},
    }

    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        (repo / "books").mkdir()
        # Freezing as ACTIVE without G6 MUST fail closed!
        _expect_raises(
            SotaBookError,
            freeze_sota_book,
            book_id="USDJPY.test_nys.v1",
            pair="USDJPY",
            horizon_minutes=15,
            allowed_sides=["UP"],
            lifecycle_status=ACTIVE_LIFECYCLE,
            model_kind="nystroem",
            feature_cols=fcols,
            coverage=0.03,
            confidence_threshold=0.05,
            refit_cpcv_p10_floors={"UP": 0.576},
            platt_calibration={"UP": {"coef": 1.0, "intercept": 0.0}},
            gate_certificates=dev_certs,
            summary="test",
            metrics={"p10_wr": 0.576},
            source_script="test",
            weights_arrays=nys_arrays,
            repo_root=repo,
        )

        # Freezing as DEMO_CANARY_LIFECYCLE with G1-G5 succeeds and verifies < 1e-11 parity!
        freeze_sota_book(
            book_id="USDJPY.test_nys.v1",
            pair="USDJPY",
            horizon_minutes=15,
            allowed_sides=["UP"],
            lifecycle_status=DEMO_CANARY_LIFECYCLE,
            model_kind="nystroem",
            feature_cols=fcols,
            coverage=0.03,
            confidence_threshold=0.02,
            refit_cpcv_p10_floors={"UP": 0.576},
            platt_calibration={"UP": {"coef": 1.0, "intercept": 0.0}},
            gate_certificates=dev_certs,
            summary="test canary",
            metrics={"p10_wr": 0.576},
            source_script="test",
            weights_arrays=nys_arrays,
            drop_ny_16h=True,
            parity_check_df=df.iloc[:50],
            parity_expected_probs=expected_nys,
            repo_root=repo,
        )

        # Demo mode loads canary; Live mode refuses canary!
        demo_books = load_enabled_sota_books(repo_root=repo, account_mode="demo")
        assert len(demo_books) == 1
        live_books = load_enabled_sota_books(repo_root=repo, account_mode="live")
        assert len(live_books) == 0
        _expect_raises(SotaBookError, load_sota_book, "USDJPY.test_nys.v1", repo_root=repo, allow_demo_canary=False)

        # DOWN prediction on UP-only book fails gate!
        b = demo_books[0]
        low_row = df.iloc[int(np.argmin(expected_nys))]
        sc_dn = b.score_row(low_row, int(datetime(2026, 7, 15, 14, 0, tzinfo=timezone.utc).timestamp()))
        assert sc_dn.direction == "DOWN" and not sc_dn.gate_passed

        # 2. Hourly logit parity (< 1e-12) + G6 active book
        epochs = np.array([int(datetime(2026, 7, 15, 12 + (i % 3), 0, tzinfo=timezone.utc).timestamp()) for i in range(600)], dtype=np.int64)
        ny_h = pd.to_datetime(epochs, unit="s", utc=True).tz_convert("America/New_York").hour.to_numpy()
        h_models = {}
        expected_hl = np.zeros(60, dtype=np.float64)
        for h in sorted(set(ny_h.tolist())):
            m = ny_h == h
            Xh = X[m]
            med_h = np.median(Xh, axis=0)
            mu_h = Xh.mean(axis=0)
            sd_h = np.where(Xh.std(axis=0) > 0, Xh.std(axis=0), 1.0)
            Zh = np.clip((Xh - mu_h) / sd_h, -5.0, 5.0)
            clf = LogisticRegression(C=0.05, max_iter=1000).fit(Zh, y[m])
            h_models[int(h)] = (med_h, mu_h, sd_h, clf)
            m60 = ny_h[:60] == h
            if m60.any():
                expected_hl[m60] = clf.predict_proba(np.clip((X[:60][m60] - mu_h) / sd_h, -5.0, 5.0))[:, 1]

        full_certs = {**dev_certs, "G6_holdout_or_prospective": {"passed": True, "holdout_ev_ci95_lo": 0.012}}
        freeze_sota_book(
            book_id="EURJPY.test_hlogit.v1",
            pair="EURJPY",
            horizon_minutes=15,
            allowed_sides=["UP"],
            lifecycle_status=ACTIVE_LIFECYCLE,
            model_kind="logit-hourly",
            feature_cols=fcols,
            coverage=0.03,
            confidence_threshold=0.02,
            refit_cpcv_p10_floors={"UP": 0.585},
            platt_calibration={"UP": {"coef": 1.0, "intercept": 0.0}},
            gate_certificates=full_certs,
            summary="test active",
            metrics={"p10_wr": 0.585},
            source_script="test",
            weights_arrays=extract_hourly_logit_arrays(h_models),
            parity_check_df=df.iloc[:60],
            parity_check_epochs=epochs[:60],
            parity_expected_probs=expected_hl,
            repo_root=repo,
        )
        assert len(load_enabled_sota_books(repo_root=repo, account_mode="live")) == 1

        # Tampering with strategy JSON must fail SHA-256 verification!
        strat_p = repo / "books" / "EURJPY.test_hlogit.v1" / "EURJPY.test_hlogit.v1_strategy.json"
        strat_p.write_text(strat_p.read_text() + " ")
        _expect_raises(SotaBookError, load_sota_book, "EURJPY.test_hlogit.v1", repo_root=repo)


def test_tick_feature_strict_causality() -> None:
    # Synthesize 30 hours of 1-second ticks across all 7 majors
    t0 = int(datetime(2026, 7, 14, 8, 0, tzinfo=timezone.utc).timestamp())
    dec_t = t0 + 26 * 3600  # 26 hours in -> all 1440m lookbacks finite
    ts = np.arange(t0, dec_t + 600, 5, dtype=np.int64)
    rng = np.random.default_rng(7)
    ticks_a = {}
    ticks_b = {}
    for p in MAJORS:
        px = 100.0 + np.cumsum(rng.normal(0.0, 0.01, size=len(ts)))
        ticks_a[p] = (ts.copy(), px.copy())
        px_b = px.copy()
        # Wildly corrupt every tick AT or AFTER decision_epoch: causal features MUST NOT change by a single bit!
        px_b[ts >= dec_t] *= 50.0
        ticks_b[p] = (ts.copy(), px_b)

    cols = ["ret_1m", "ret_15m", "ret_1440m", "rv_60m", "activity_15m", "ticks_per_bar_15m", "x_EURUSD_ret_15m", "x_usd_weak_15m"]
    fa = extract_decision_feature_row("USDJPY", ticks_a, dec_t, cols)
    fb = extract_decision_feature_row("USDJPY", ticks_b, dec_t, cols)
    assert np.array_equal(fa.row.to_numpy(), fb.row.to_numpy()), "future tick leaked into causal decision row!"

    # Stale feed (> 120s before dec_t) must fail closed
    stale_ticks = dict(ticks_a)
    ts_u, px_u = stale_ticks["USDJPY"]
    keep = ts_u < (dec_t - 180)
    stale_ticks["USDJPY"] = (ts_u[keep], px_u[keep])
    _expect_raises(SotaFeatureError, extract_decision_feature_row, "USDJPY", stale_ticks, dec_t, cols)


class _FakeAsyncDerivClient:
    def __init__(self, *, ask: float = 1.0, payout: float = 1.82, win: bool = True, fail_reconcile: bool = False):
        self.ask = ask
        self.payout = payout
        self.win = win
        self.fail_reconcile = fail_reconcile
        self.seq = 0

    async def request(self, payload: dict[str, Any], timeout: float = 5.0) -> dict[str, Any]:
        self.seq += 1
        if "proposal" in payload:
            return {"proposal": {"id": f"prop-{self.seq}", "ask_price": self.ask, "payout": self.payout}}
        if "buy" in payload:
            return {"buy": {"contract_id": f"cid-{self.seq}", "buy_price": self.ask, "payout": self.payout}}
        if "proposal_open_contract" in payload:
            if self.fail_reconcile:
                raise RuntimeError("simulated transport failure")
            cid = payload["contract_id"]
            profit = round(self.payout - self.ask, 2) if self.win else -self.ask
            sell = self.payout if self.win else 0.0
            return {
                "proposal_open_contract": {
                    "contract_id": cid,
                    "is_sold": 1,
                    "status": "won" if self.win else "lost",
                    "sell_price": sell,
                    "profit": profit,
                }
            }
        raise AssertionError(f"unexpected payload {payload}")


def test_async_executor_and_live_pnl_promotion_gate() -> None:
    async def _scenario() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_path = root / "sota_demo.sqlite"
            log_dir = root / "logs"
            conn = connect_sota_queue(db_path)
            ex = SotaTradeExecutor(
                account_mode="demo",
                stake_usd=1.0,
                db_path=db_path,
                log_dir=log_dir,
                risk_cfg=RiskBreakerConfig(
                    max_stake_usd=5.0,
                    max_daily_loss_usd=100.0,
                    max_drawdown_usd=100.0,
                    max_open_contracts=10,
                    max_usd_factor_open=10,
                    max_jpy_factor_open=10,
                    same_book_min_gap_seconds=0,
                ),
                ev_margin=0.005,
            )
            ex.client = _FakeAsyncDerivClient(ask=1.0, payout=1.82, win=True)  # type: ignore[assignment]

            base_t = datetime(2026, 7, 13, 13, 0, 0, tzinfo=timezone.utc)  # Monday
            # Simulate 55 settled trades across distinct weekday NY-session hours
            for i in range(55):
                day_idx = i // 3
                weeks, wd = divmod(day_idx, 5)
                t_i = base_t + timedelta(days=weeks * 7 + wd, hours=(i % 3))
                ex.client.win = (i % 11) not in {0, 1, 2}  # type: ignore[attr-defined]
                sig = {
                    "book_id": "USDJPY.c2nysabs.v1",
                    "pair": "USDJPY",
                    "horizon_minutes": 15,
                    "side": "UP",
                    "lane": "wall",
                    "offset_id": i,
                    "signal_close_utc": t_i.isoformat(),
                    "effective_floor": 0.571,
                    "calibrated_win_prob": 0.580,
                    "drop_ny_16h": True,
                    "contract_family": "risefall",
                    "book_content_id": "deadbeef12345678",
                }
                ok_ins, sid = enqueue_sota_signal(conn, sig)
                assert ok_ins
                out = await ex.process_one_signal(conn, sid, sig, now_utc=t_i + timedelta(seconds=2))
                assert out["status"] == "bought", f"expected bought, got {out}"
                await ex.reconcile_open_contracts(conn)

                if i == 20:
                    early = summarize_settled_queue(db_path, min_settled_for_promotion=50, book_id="USDJPY.c2nysabs.v1")
                    assert not early["promotion_eligible"] and early["settled_contracts"] == 21

            # CRITICAL-4: Add 5 losing trades on a second book (`USDJPY.other_30m.v1`) and verify per-book non-pooling!
            for j in range(5):
                t_j = base_t + timedelta(days=42 + j)
                ex.client.win = False  # type: ignore[attr-defined]
                sig_b = {
                    "book_id": "USDJPY.other_30m.v1",
                    "pair": "USDJPY",
                    "horizon_minutes": 30,
                    "side": "UP",
                    "lane": "wall",
                    "offset_id": j,
                    "signal_close_utc": t_j.isoformat(),
                    "effective_floor": 0.571,
                    "calibrated_win_prob": 0.580,
                    "book_content_id": "cafebabe87654321",
                }
                _, sid_b = enqueue_sota_signal(conn, sig_b)
                await ex.process_one_signal(conn, sid_b, sig_b, now_utc=t_j + timedelta(seconds=2))
                await ex.reconcile_open_contracts(conn)

            final_a = summarize_settled_queue(db_path, min_settled_for_promotion=50, book_id="USDJPY.c2nysabs.v1")
            final_b = summarize_settled_queue(db_path, min_settled_for_promotion=50, book_id="USDJPY.other_30m.v1")
            assert final_a["settled_contracts"] == 60
            assert final_a["promotion_eligible"] is True
            assert final_a["by_book"]["USDJPY.c2nysabs.v1"]["ev_ci95"][0] > 0.0
            assert final_b["promotion_eligible"] is False
            assert final_a["promotion_eligible_books"] == ["USDJPY.c2nysabs.v1"]

            # Test 5 consecutive reconciliation errors trip automatic KILL switch
            t_k = datetime(2026, 9, 16, 14, 0, 0, tzinfo=timezone.utc)
            sig_k = {
                "book_id": "USDJPY.c2nysabs.v1",
                "pair": "USDJPY",
                "horizon_minutes": 15,
                "side": "UP",
                "lane": "wall",
                "offset_id": 999,
                "signal_close_utc": t_k.isoformat(),
                "effective_floor": 0.571,
                "calibrated_win_prob": 0.580,
            }
            _, sid_k = enqueue_sota_signal(conn, sig_k)
            await ex.process_one_signal(conn, sid_k, sig_k, now_utc=t_k + timedelta(seconds=2))
            ex.client = _FakeAsyncDerivClient(fail_reconcile=True)  # type: ignore[assignment]
            for _ in range(5):
                await ex.reconcile_open_contracts(conn)
            assert ex.kill_path.exists(), "KILL switch was not tripped after 5 consecutive reconcile errors"
            conn.close()

    asyncio.run(_scenario())


def test_real_frozen_sota_books_end_to_end() -> None:
    repo = Path(__file__).resolve().parents[1]
    demo_books = load_enabled_sota_books(repo_root=repo, account_mode="demo")
    assert {
        "USDJPY.sota_ens5_m30_up.v1",
        "USDJPY.sota_ens5_m15_up.v1",
        "USDJPY.sota_c2nysabs_m15_up.v1",
        "USDJPY.sota_e49nys30_m30_up.v1",
    }.issubset({b.book_id for b in demo_books})
    # Rank #1 30m ensemble precedes legacy 1-seed 30m; 15m ensemble precedes legacy 1-seed 15m
    demo_ids = [b.book_id for b in demo_books]
    assert demo_ids.index("USDJPY.sota_ens5_m30_up.v1") < demo_ids.index("USDJPY.sota_e49nys30_m30_up.v1")
    assert demo_ids.index("USDJPY.sota_ens5_m15_up.v1") < demo_ids.index("USDJPY.sota_c2nysabs_m15_up.v1")
    sel_books = load_enabled_sota_books(
        repo_root=repo,
        account_mode="demo",
        book_ids=parse_book_ids_spec("USDJPY.sota_ens5_m30_up.v1,USDJPY.sota_ens5_m15_up.v1"),
    )
    assert [b.book_id for b in sel_books] == ["USDJPY.sota_ens5_m30_up.v1", "USDJPY.sota_ens5_m15_up.v1"]
    _expect_raises(SotaBookError, parse_book_ids_spec, "USDJPY.sota_ens5_m30_up.v1,USDJPY.sota_ens5_m30_up.v1")
    assert load_enabled_sota_books(repo_root=repo, account_mode="live") == []

    t0 = int(datetime(2026, 7, 14, 8, 0, tzinfo=timezone.utc).timestamp())
    dec_t = t0 + 28 * 3600  # 2026-07-15 12:00 UTC (08:00 EDT)
    ts = np.arange(t0, dec_t + 60, 5, dtype=np.int64)
    rng = np.random.default_rng(123)
    ticks = {p: (ts.copy(), 100.0 + np.cumsum(rng.normal(0.0, 0.01, size=len(ts)))) for p in MAJORS}

    for b in demo_books:
        frow = extract_decision_feature_row(b.pair, ticks, dec_t, b.feature_cols)
        sc = b.score_row(frow.row, dec_t)
        assert math.isfinite(sc.proba_up_raw) and 0.0 < sc.proba_up_raw < 1.0
        cal_up = b.calibrate_side_prob(sc.proba_up_raw, "UP")
        assert cal_up is not None and 0.0 < cal_up < 1.0
        if sc.direction in b.allowed_sides:
            assert sc.calibrated_win_prob is not None and 0.0 < sc.calibrated_win_prob < 1.0
        else:
            assert not sc.gate_passed and sc.calibrated_win_prob is None

        # MAJOR-6: < 50% NaN imputes via saved median cleanly; >= 50% NaN or inf fails closed!
        nan_row = frow.row.copy()
        nan_row.iloc[0] = float("nan")
        sc_nan1 = b.score_row(nan_row, dec_t)
        assert math.isfinite(sc_nan1.proba_up_raw)
        half_nan = frow.row.copy()
        half_nan.iloc[:35] = float("nan")
        _expect_raises(SotaBookError, b.score_row, half_nan, dec_t)
        inf_row = frow.row.copy()
        inf_row.iloc[0] = float("inf")
        _expect_raises(SotaBookError, b.score_row, inf_row, dec_t)


def test_nystroem_ens_multiplier_and_barrier_extensions() -> None:
    # 1. Pure-NumPy 3-seed `nystroem-ens` parity (< 1e-12 vs sklearn)
    rng = np.random.default_rng(99)
    fcols = [f"f_{i}" for i in range(10)]
    X = rng.normal(size=(400, 10))
    y = (X[:, 0] + 0.6 * X[:, 2] + rng.normal(scale=0.5, size=400) > 0).astype(int)
    df = pd.DataFrame(X, columns=fcols)
    med = np.median(X, axis=0)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-12
    Z = np.clip((X - mu) / sd, -5.0, 5.0)
    ny_list, sgd_list, prob_list = [], [], []
    for s_i in range(3):
        ny_s = Nystroem(kernel="rbf", gamma=1.0 / X.shape[1], n_components=48, random_state=s_i).fit(Z)
        sgd_s = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=40, tol=1e-5, random_state=s_i).fit(ny_s.transform(Z), y)
        ny_list.append(ny_s)
        sgd_list.append(sgd_s)
        prob_list.append(sgd_s.predict_proba(ny_s.transform(Z[:40]))[:, 1])
    expected_ens = np.mean(prob_list, axis=0)
    ens_arrays = extract_nystroem_ens_arrays(med, mu, sd, ny_list, sgd_list)

    dev_certs = {
        "G1_cpcv": {"passed": True, "p10_wr": 0.59},
        "G2_placebo": {"passed": True, "mean_excess": 0.07},
        "G3_entry_delay": {"passed": True, "d4_p10": 0.58},
        "G4_sign_null": {"passed": True, "null_max": 0.53},
        "G5_payout": {"passed": True, "p10_ev": 0.08},
    }
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        (repo / "books").mkdir()
        freeze_sota_book(
            book_id="USDJPY.test_ens3_m60.v1",
            pair="USDJPY",
            horizon_minutes=60,
            allowed_sides=["UP"],
            lifecycle_status=DEMO_CANARY_LIFECYCLE,
            model_kind="nystroem-ens",
            feature_cols=fcols,
            coverage=0.03,
            confidence_threshold=0.02,
            refit_cpcv_p10_floors={"UP": 0.59},
            platt_calibration={"UP": {"coef": 1.2, "intercept": 0.0}},
            gate_certificates=dev_certs,
            summary="test nystroem-ens",
            metrics={"p10_wr": 0.59},
            source_script="test",
            weights_arrays=ens_arrays,
            parity_check_df=df.iloc[:40],
            parity_expected_probs=expected_ens,
            repo_root=repo,
        )
        loaded = load_sota_book("USDJPY.test_ens3_m60.v1", repo_root=repo, allow_demo_canary=True)
        got = loaded.predict_probabilities(df.iloc[:40])
        assert float(np.max(np.abs(got - expected_ens))) < 1e-12

    # 2. Multi-asset symbols, exposure legs, barrier formatting, and multiplier gate
    assert SOTA_DERIV_SYMBOLS["XAUUSD"] == "frxXAUUSD"
    assert SOTA_DERIV_SYMBOLS["BTCUSD"] == "cryBTCUSD"
    assert SOTA_DERIV_SYMBOLS["WLDUSD"] == "WLDUSD"
    assert SOTA_DERIV_SYMBOLS["OTC_SPC"] == "OTC_SPC"
    assert currency_leg_exposure("WLDUSD", "UP") == {"USD": 1}
    assert currency_leg_exposure("OTC_SPC", "DOWN") == {"OTC_SPC": -1}
    assert currency_leg_exposure("EURGBP", "NEUTRAL") == {"VOL_EURGBP": 1}
    assert contract_type_for("UP", "multiplier") == "MULTUP"
    assert contract_type_for("NEUTRAL", "barrier_1d", barrier_contract_type="EXPIRYRANGE") == "EXPIRYRANGE"
    hi_jpy, lo_jpy = format_barrier_levels("USDJPY", 155.000, 0.005, "RANGE", side="NEUTRAL")
    assert hi_jpy == "155.775" and lo_jpy == "154.225"
    b1_eur, b2_eur = format_barrier_levels("EURUSD", 1.10000, 0.005, "ONETOUCH", side="DOWN")
    assert b1_eur == "1.09450" and b2_eur is None

    ok_mult = evaluate_sota_multiplier_gate(
        {"ask_price": 10.0, "commission": 0.30, "spot": 158.0, "currency": "USD", "limit_order": {"stop_out": {"value": 156.46}}},
        expected_ev=0.04,
        ev_margin=0.005,
    )
    assert ok_mult.passed and ok_mult.reason is None
    bad_comm = evaluate_sota_multiplier_gate(
        {"ask_price": 10.0, "commission": 2.50, "spot": 158.0, "currency": "USD", "limit_order": {"stop_out": {"value": 156.46}}},
        expected_ev=0.04,
        max_commission_frac=0.20,
    )
    assert not bad_comm.passed and bad_comm.reason == "commission_above_ceiling"

    # 3. Executor end-to-end timed `sell` on Multiplier & 1d Barrier proposal/buy
    class _MultBarrierClient:
        def __init__(self) -> None:
            self.sold = False
            self.last_prop: dict[str, Any] = {}

        async def request(self, payload: dict[str, Any], timeout: float = 5.0) -> dict[str, Any]:
            if "proposal" in payload:
                self.last_prop = dict(payload)
                if payload["contract_type"] == "MULTUP":
                    return {
                        "proposal": {
                            "id": "p-mult-1",
                            "ask_price": 10.0,
                            "commission": 0.30,
                            "spot": 158.0,
                            "currency": "USD",
                            "limit_order": {"stop_out": {"value": 156.46}},
                        }
                    }
                return {"proposal": {"id": "p-bar-1", "ask_price": 10.0, "payout": 18.5, "currency": "USD"}}
            if "buy" in payload:
                is_m = payload["buy"] == "p-mult-1"
                return {"buy": {"contract_id": "c-mult" if is_m else "c-bar", "buy_price": 10.0, "payout": 0.0 if is_m else 18.5}}
            if "sell" in payload:
                self.sold = True
                return {"sell": {"sold_for": 11.40, "contract_id": payload["sell"]}}
            if "proposal_open_contract" in payload:
                if self.sold:
                    return {
                        "proposal_open_contract": {
                            "contract_id": payload["contract_id"],
                            "is_sold": 1,
                            "status": "sold",
                            "sell_price": 11.40,
                            "profit": 1.40,
                        }
                    }
                return {
                    "proposal_open_contract": {
                        "contract_id": payload["contract_id"],
                        "is_sold": 0,
                        "status": "open",
                        "is_valid_to_sell": 1,
                    }
                }
            raise AssertionError(f"unexpected {payload}")

    async def _run_mb() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_p = root / "mb.sqlite"
            conn = connect_sota_queue(db_p)
            ex = SotaTradeExecutor(
                account_mode="demo",
                stake_usd=10.0,
                db_path=db_p,
                log_dir=root / "logs",
                risk_cfg=RiskBreakerConfig(max_stake_usd=20.0, max_daily_loss_usd=100.0, max_drawdown_usd=100.0),
                ev_margin=0.005,
            )
            fake = _MultBarrierClient()
            ex.client = fake  # type: ignore[assignment]
            t_now = datetime(2026, 7, 15, 14, 0, 0, tzinfo=timezone.utc)

            # Multiplier buy -> before expiry stays open -> at expiry triggers timed `sell`!
            sig_m = {
                "book_id": "USDJPY.mult100.v1",
                "pair": "USDJPY",
                "horizon_minutes": 30,
                "side": "UP",
                "signal_close_utc": t_now.isoformat(),
                "effective_floor": 0.0,
                "expected_ev": 0.035,
                "multiplier": 100,
                "contract_family": "multiplier",
            }
            _, sid_m = enqueue_sota_signal(conn, sig_m)
            r_m = await ex.process_one_signal(conn, sid_m, sig_m, now_utc=t_now + timedelta(seconds=2))
            assert r_m["status"] == "bought" and fake.last_prop["multiplier"] == 100
            await ex.reconcile_open_contracts(conn, now_utc=t_now + timedelta(minutes=10))
            assert not fake.sold
            await ex.reconcile_open_contracts(conn, now_utc=t_now + timedelta(minutes=31))
            assert fake.sold
            st_row = conn.execute("SELECT status, profit_cents FROM sota_contracts WHERE contract_id='c-mult'").fetchone()
            assert st_row["status"] == "won" and int(st_row["profit_cents"]) == 140

            # 1d Barrier two-barrier proposal & buy
            sig_b = {
                "book_id": "EURGBP.range1d.v1",
                "pair": "EURGBP",
                "horizon_minutes": 1440,
                "side": "NEUTRAL",
                "signal_close_utc": t_now.isoformat(),
                "effective_floor": 0.56,
                "calibrated_win_prob": 0.60,
                "contract_family": "barrier_1d",
                "barrier_contract_type": "RANGE",
                "spot_ref": 0.85000,
                "barrier_rel_half_width": 0.0030,
            }
            _, sid_b = enqueue_sota_signal(conn, sig_b)
            r_b = await ex.process_one_signal(conn, sid_b, sig_b, now_utc=t_now + timedelta(seconds=3))
            assert r_b["status"] == "bought"
            assert fake.last_prop["duration_unit"] == "d" and "barrier" in fake.last_prop and "barrier2" in fake.last_prop
            conn.close()

    asyncio.run(_run_mb())

    # 4. Adversarial edge cases: legacy SQLite migration, wire integer IDs, incomplete post-sell poc,
    #    dry-run sell suppression, malformed limit_order/NPZ rejection, and empty unrelated tick tolerance.
    _expect_raises(ExecutorError, format_barrier_levels, "USDJPY", 158.000, 1e-7, "ONETOUCH", side="UP")
    _expect_raises(ExecutorError, format_barrier_levels, "USDJPY", 158.000, 0.005, "ONETOUCH", side="NEUTRAL")
    bad_lo = evaluate_sota_multiplier_gate(
        {"ask_price": 5.0, "commission": 0.10, "spot": 158.0, "limit_order": "malformed"},
        expected_ev=0.03,
        side="UP",
    )
    assert not bad_lo.passed and bad_lo.reason == "missing_or_invalid_stop_out"
    wrong_side_so = evaluate_sota_multiplier_gate(
        {"ask_price": 5.0, "commission": 0.10, "spot": 158.0, "limit_order": {"stop_out": {"value": 159.5}}},
        expected_ev=0.03,
        side="UP",
    )
    assert not wrong_side_so.passed and wrong_side_so.reason == "invalid_stop_out_side"
    nan_policy = evaluate_sota_multiplier_gate(
        {"ask_price": 5.0, "commission": 0.10, "spot": 158.0, "limit_order": {"stop_out": {"value": 156.5}}},
        expected_ev=0.03,
        ev_margin=float("nan"),
    )
    assert not nan_policy.passed and nan_policy.reason == "invalid_policy_bounds"

    # Malformed NPZ rejection (object array or zero sd) before writing any artifact
    bad_arrays = dict(ens_arrays)
    bad_arrays["sd"] = np.zeros_like(ens_arrays["sd"])
    with tempfile.TemporaryDirectory() as td_bad:
        repo_b = Path(td_bad)
        (repo_b / "books").mkdir()
        _expect_raises(
            SotaBookError,
            freeze_sota_book,
            book_id="USDJPY.bad_sd.v1",
            pair="USDJPY",
            horizon_minutes=30,
            allowed_sides=["UP"],
            lifecycle_status=DEMO_CANARY_LIFECYCLE,
            model_kind="nystroem-ens",
            feature_cols=fcols,
            coverage=0.03,
            confidence_threshold=0.02,
            refit_cpcv_p10_floors={"UP": 0.59},
            platt_calibration={"UP": {"coef": 1.0, "intercept": 0.0}},
            gate_certificates=dev_certs,
            summary="bad",
            metrics={},
            source_script="test",
            weights_arrays=bad_arrays,
            parity_check_df=df.iloc[:10],
            parity_expected_probs=expected_ens[:10],
            repo_root=repo_b,
        )
        assert not (repo_b / "books" / "SOTA_INDEX.json").exists()

    # Empty unrelated XAUUSD tick array in snapshot must NOT break USDJPY feature extraction!
    t0_tk = int(datetime(2026, 7, 14, 8, 0, tzinfo=timezone.utc).timestamp())
    dec_tk = t0_tk + 26 * 3600
    ts_tk = np.arange(t0_tk, dec_tk + 60, 5, dtype=np.int64)
    snap_with_empty_xau = {p: (ts_tk.copy(), np.full(len(ts_tk), 100.0) + np.arange(len(ts_tk)) * 1e-4) for p in MAJORS}
    snap_with_empty_xau["XAUUSD"] = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64))
    frow_ok = extract_decision_feature_row("USDJPY", snap_with_empty_xau, dec_tk, ["ret_15m", "rv_60m", "x_usd_weak_15m"])
    assert frow_ok.spot_ref > 0.0

    # Legacy SQLite queue migration + numeric wire ID + incomplete post-sell response safety
    async def _run_migration_and_wire_safety() -> None:
        import sqlite3
        with tempfile.TemporaryDirectory() as td_mig:
            db_old = Path(td_mig) / "legacy.sqlite"
            raw_c = sqlite3.connect(str(db_old))
            raw_c.executescript(
                """
                CREATE TABLE sota_signals (
                    signal_id TEXT PRIMARY KEY,
                    book_id TEXT NOT NULL,
                    pair TEXT NOT NULL,
                    horizon_minutes INTEGER NOT NULL CHECK (horizon_minutes IN (15,30)),
                    side TEXT NOT NULL CHECK (side IN ('UP','DOWN')),
                    lane TEXT NOT NULL DEFAULT 'wall',
                    offset_id INTEGER NOT NULL DEFAULT 0,
                    signal_close_utc TEXT NOT NULL,
                    effective_floor REAL NOT NULL,
                    calibrated_win_prob REAL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN (
                        'queued','claimed','buy_intent','bought','contract_closed',
                        'terminal_skip','signal_expired','buy_blocked_risk','buy_unknown'
                    )),
                    reason TEXT,
                    created_utc TEXT NOT NULL,
                    updated_utc TEXT NOT NULL,
                    UNIQUE(pair, horizon_minutes, side, lane, offset_id, signal_close_utc)
                );
                CREATE TABLE sota_contracts (
                    contract_id TEXT PRIMARY KEY,
                    signal_id TEXT NOT NULL UNIQUE REFERENCES sota_signals(signal_id),
                    book_id TEXT NOT NULL,
                    pair TEXT NOT NULL,
                    horizon_minutes INTEGER NOT NULL,
                    side TEXT NOT NULL CHECK (side IN ('UP','DOWN')),
                    proposal_id TEXT NOT NULL,
                    buy_price_cents INTEGER NOT NULL,
                    payout_cents INTEGER NOT NULL,
                    sell_price_cents INTEGER NOT NULL DEFAULT 0,
                    profit_cents INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL CHECK (status IN ('open','won','lost','sold','expired','cancelled','unknown')),
                    opened_utc TEXT NOT NULL,
                    expected_expiry_utc TEXT NOT NULL,
                    closed_utc TEXT,
                    raw_hash TEXT NOT NULL
                );
                """
            )
            raw_c.close()
            conn_m = connect_sota_queue(db_old)
            ins_n, _ = enqueue_sota_signal(
                conn_m,
                {
                    "book_id": "EURGBP.range1d.v1",
                    "pair": "EURGBP",
                    "horizon_minutes": 1440,
                    "side": "NEUTRAL",
                    "signal_close_utc": "2026-07-15T14:00:00+00:00",
                    "effective_floor": 0.58,
                },
            )
            assert ins_n, "migrated legacy SQLite table failed to insert NEUTRAL 1440m signal!"

            # Test integer wire contract_id AND incomplete post-sell follow-up NOT booking a fake loss
            wire_seen: list[Any] = []
            poll_step = 0

            class _WireCheckClient:
                async def request(self, payload: dict[str, Any], timeout: float = 5.0) -> dict[str, Any]:
                    nonlocal poll_step
                    if "proposal_open_contract" in payload:
                        wire_seen.append(payload["contract_id"])
                        poll_step += 1
                        if poll_step == 1:
                            return {"proposal_open_contract": {"contract_id": 987654, "is_sold": 0, "status": "open", "is_valid_to_sell": 1}}
                        if poll_step == 2:
                            # Incomplete non-terminal follow-up right after sell: must stay open, NEVER book -$5 loss!
                            return {"proposal_open_contract": {}}
                        return {"proposal_open_contract": {"contract_id": 987654, "is_sold": 1, "status": "sold", "sell_price": 6.40, "profit": 1.40}}
                    if "sell" in payload:
                        wire_seen.append(payload["sell"])
                        return {"sell": {"sold_for": 6.40, "contract_id": 987654}}
                    raise AssertionError(f"unexpected {payload}")

            conn_m.execute(
                "INSERT INTO sota_signals VALUES ('sig-w','USDJPY.m.v1','USDJPY',30,'UP','wall',0,'2026-07-15T14:00:00+00:00',0.0,NULL,'{}','bought',NULL,'2026-07-15T14:00:00+00:00','2026-07-15T14:00:00+00:00')"
            )
            conn_m.execute(
                "INSERT INTO sota_contracts VALUES ('987654','sig-w','USDJPY.m.v1','USDJPY',30,'UP','p1',500,0,0,0,'open','2026-07-15T14:00:00+00:00','2026-07-15T14:30:00+00:00',NULL,'h','2026-07-15T14:00:00+00:00','multiplier',NULL,'demo',NULL)"
            )
            ex_w = SotaTradeExecutor(
                account_mode="demo",
                stake_usd=5.0,
                db_path=db_old,
                log_dir=Path(td_mig) / "logs",
                risk_cfg=RiskBreakerConfig(),
                dry_run_proposal_only=True,
            )
            ex_w.client = _WireCheckClient()  # type: ignore[assignment]
            # Dry-run mode MUST NOT send sell!
            await ex_w.reconcile_open_contracts(conn_m, now_utc=datetime(2026, 7, 15, 14, 31, tzinfo=timezone.utc))
            assert wire_seen == [987654] and all(isinstance(x, int) for x in wire_seen)
            # Live mode sends sell; incomplete immediate poc keeps contract open (not -$5 loss!); next poll settles +$1.40!
            ex_w.dry_run_proposal_only = False
            poll_step = 0
            await ex_w.reconcile_open_contracts(conn_m, now_utc=datetime(2026, 7, 15, 14, 31, tzinfo=timezone.utc))
            r_mid = conn_m.execute("SELECT status, profit_cents FROM sota_contracts WHERE contract_id='987654'").fetchone()
            assert r_mid["status"] == "open" and int(r_mid["profit_cents"]) == 0
            await ex_w.reconcile_open_contracts(conn_m, now_utc=datetime(2026, 7, 15, 14, 31, 5, tzinfo=timezone.utc))
            r_fin = conn_m.execute("SELECT status, profit_cents FROM sota_contracts WHERE contract_id='987654'").fetchone()
            assert r_fin["status"] == "won" and int(r_fin["profit_cents"]) == 140
            led = summarize_settled_queue(db_old, min_settled_for_promotion=1, book_id="USDJPY.m.v1")
            assert led["by_book"]["USDJPY.m.v1"]["mean_payout_ratio"] is None
            assert led["promotion_eligible"] is False
            assert any(r.startswith("min_trades_arg_below_hard_floor:") for r in led["reasons"])
            conn_m.close()

    asyncio.run(_run_migration_and_wire_safety())


def test_da2_adversarial_hardening_regressions() -> None:
    from build_certified_sota_books import freeze_from_v2_dossier
    from vps_preflight import run_network_proposal_smoke

    # #8: Reject empty elements in explicit book_ids spec while keeping blank/omitted as None
    assert parse_book_ids_spec(None) is None
    assert parse_book_ids_spec("") is None
    assert parse_book_ids_spec("   ") is None
    _expect_raises(SotaBookError, parse_book_ids_spec, " , , ")
    _expect_raises(SotaBookError, parse_book_ids_spec, "USDJPY.sota_ens5_m30_up.v1,")
    _expect_raises(SotaBookError, parse_book_ids_spec, ",USDJPY.sota_ens5_m30_up.v1")

    # #3 & #4: No-replace book publication + strict v2 dossier validation
    rng = np.random.default_rng(77)
    fcols = [f"f_{i}" for i in range(6)]
    X = rng.normal(size=(120, 6))
    y = (X[:, 0] > 0).astype(int)
    med, mu, sd = np.median(X, axis=0), X.mean(axis=0), X.std(axis=0) + 1e-12
    Z = np.clip((X - mu) / sd, -5.0, 5.0)
    ny1 = Nystroem(kernel="rbf", gamma=1.0 / 6, n_components=16, random_state=1).fit(Z)
    sgd1 = SGDClassifier(loss="log_loss", max_iter=30, random_state=1).fit(ny1.transform(Z), y)
    ny2 = Nystroem(kernel="rbf", gamma=1.0 / 6, n_components=16, random_state=2).fit(Z)
    sgd2 = SGDClassifier(loss="log_loss", max_iter=30, random_state=2).fit(ny2.transform(Z), y)
    dev_certs = {
        "G1_cpcv": {"passed": True, "p10_wr": 0.58},
        "G2_placebo": {"passed": True, "mean_excess": 0.05},
        "G3_entry_delay": {"passed": True, "d4_p10": 0.57},
        "G4_sign_null": {"passed": True, "null_max": 0.52},
        "G5_payout": {"passed": True, "p10_ev": 0.04},
    }
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        (repo / "books").mkdir()
        df_p = pd.DataFrame(X[:20], columns=fcols)
        p1 = sgd1.predict_proba(ny1.transform(Z[:20]))[:, 1]
        freeze_sota_book(
            book_id="USDJPY.noreplace.v1",
            pair="USDJPY",
            horizon_minutes=30,
            allowed_sides=["UP"],
            lifecycle_status=DEMO_CANARY_LIFECYCLE,
            model_kind="nystroem",
            feature_cols=fcols,
            coverage=0.03,
            confidence_threshold=0.02,
            refit_cpcv_p10_floors={"UP": 0.58},
            platt_calibration={"UP": {"coef": 1.0, "intercept": 0.0}},
            gate_certificates=dev_certs,
            summary="v1",
            metrics={"p10_wr": 0.58},
            source_script="scripts/test_deriv_sota_runtime.py",
            weights_arrays=extract_nystroem_arrays(med, mu, sd, ny1, sgd1),
            parity_check_df=df_p,
            parity_expected_probs=p1,
            repo_root=repo,
        )
        assert not (repo / "books" / ".sota_index.lock").exists()
        # Same content_id + same lifecycle retry succeeds idempotently; lifecycle change or different weights fail closed unless allow_overwrite=True
        freeze_sota_book(
            book_id="USDJPY.noreplace.v1",
            pair="USDJPY",
            horizon_minutes=30,
            allowed_sides=["UP"],
            lifecycle_status=DEMO_CANARY_LIFECYCLE,
            model_kind="nystroem",
            feature_cols=fcols,
            coverage=0.03,
            confidence_threshold=0.02,
            refit_cpcv_p10_floors={"UP": 0.58},
            platt_calibration={"UP": {"coef": 1.0, "intercept": 0.0}},
            gate_certificates=dev_certs,
            summary="v1 retry",
            metrics={"p10_wr": 0.58},
            source_script="scripts/test_deriv_sota_runtime.py",
            weights_arrays=extract_nystroem_arrays(med, mu, sd, ny1, sgd1),
            parity_check_df=df_p,
            parity_expected_probs=p1,
            repo_root=repo,
        )
        _expect_raises(
            SotaBookError,
            freeze_sota_book,
            book_id="USDJPY.noreplace.v1",
            pair="USDJPY",
            horizon_minutes=30,
            allowed_sides=["UP"],
            lifecycle_status=ACTIVE_LIFECYCLE,
            model_kind="nystroem",
            feature_cols=fcols,
            coverage=0.03,
            confidence_threshold=0.02,
            refit_cpcv_p10_floors={"UP": 0.58},
            platt_calibration={"UP": {"coef": 1.0, "intercept": 0.0}},
            gate_certificates={**dev_certs, "G6_holdout_or_prospective": {"passed": True}},
            summary="v1 lifecycle mutation without overwrite",
            metrics={"p10_wr": 0.58},
            source_script="scripts/test_deriv_sota_runtime.py",
            weights_arrays=extract_nystroem_arrays(med, mu, sd, ny1, sgd1),
            parity_check_df=df_p,
            parity_expected_probs=p1,
            repo_root=repo,
        )
        p2 = sgd2.predict_proba(ny2.transform(Z[:20]))[:, 1]
        _expect_raises(
            SotaBookError,
            freeze_sota_book,
            book_id="USDJPY.noreplace.v1",
            pair="USDJPY",
            horizon_minutes=30,
            allowed_sides=["UP"],
            lifecycle_status=DEMO_CANARY_LIFECYCLE,
            model_kind="nystroem",
            feature_cols=fcols,
            coverage=0.03,
            confidence_threshold=0.02,
            refit_cpcv_p10_floors={"UP": 0.58},
            platt_calibration={"UP": {"coef": 1.0, "intercept": 0.0}},
            gate_certificates=dev_certs,
            summary="v1 overwrite attempt",
            metrics={"p10_wr": 0.58},
            source_script="scripts/test_deriv_sota_runtime.py",
            weights_arrays=extract_nystroem_arrays(med, mu, sd, ny2, sgd2),
            parity_check_df=df_p,
            parity_expected_probs=p2,
            repo_root=repo,
        )

        # Synthetic tampered dossier with false individual gates or non-USDJPY out-of-band drift MUST fail before fit!
        (repo / "results" / "json").mkdir(parents=True)
        bad_dos = {
            "schema": "sota-5gate-certification-dossier/v2",
            "book_id": "EURJPY.bad.v1",
            "contract_family": "risefall",
            "shared_calendar_aligned": True,
            "cutoff_utc": 1785542400,
            "all_5_gates_passed": True,
            "drift_check_passed": True,
            "pair": "EURJPY",
            "horizon_min": 30,
            "side": "UP",
            "cov": 0.03,
            "model_variant": "core66_nys_ens5",
            "feature_count": 6,
            "moved_up_rate": 0.555,
            "tie_rate": 0.01,
            "frozen_2025_to_2026_oos": {"ev": 0.05},
            "G1_cpcv": {"passed": True, "paths_counted": 15, "paths_clear_0541": 15, "p10_wr": 0.59, "mean_wr": 0.61},
            "G2_placebo": {"passed": True, "paths_positive": 14, "mean_excess_hour_vol": 0.05, "verdict": "SKILL BEYOND DRIFT TIMING"},
            "G3_entry_delay": {"passed": True, "d4_paths_clear": 14, "d4_p10_wr": 0.58, "verdict_delay": "OPERATIONALLY ROBUST", "verdict_staleness": "OK"},
            "G4_sign_null": {"passed": True, "null_passes": 0, "null_max_p10": 0.52, "real_p10_rank": "16/16", "verdict": "SIGNIFICANT vs sign-null replay"},
            "G5_payout": {"passed": True, "paths_ev_positive": 14, "p10_ev": 0.05, "mean_ev": 0.10, "annual_floor_log_growth_path_p10": 0.2, "est_trades_per_year": 200, "verdict": "PAYOUT-CLEARED"},
            "quoted_payout_ratio": 1.82,
        }
        (repo / "results" / "json" / "bad.json").write_text(json.dumps(bad_dos), encoding="utf-8")
        dummy_dev = pd.DataFrame(X, columns=fcols)
        dummy_dev["decision_ts"] = np.arange(1755000000, 1755000000 + len(X) * 60, 60, dtype=np.int64)
        _expect_raises(
            SotaBookError,
            freeze_from_v2_dossier,
            book_id="EURJPY.bad.v1",
            dossier_json_rel="results/json/bad.json",
            df_dev=dummy_dev,
            ret_fwd_true=np.ones(len(X)) * 0.001,
            valid_true=np.ones(len(X), dtype=bool),
            min_payout_ratio=1.78,
            repo_root=repo,
        )

    # #1, #2, #5, #6, #7: Post-quote KILL/expiry, malformed start_time persistence, correlated uncertain-buy recovery,
    # zero-proceeds sell_ack_cents=0 settlement + no duplicate sell, and Demo+content_id+50-trade promotion floor
    async def _exec_hardening() -> None:
        with tempfile.TemporaryDirectory() as td_ex:
            root = Path(td_ex)
            db_p = root / "harden.sqlite"
            log_d = root / "logs"
            conn = connect_sota_queue(db_p)
            ex = SotaTradeExecutor(
                account_mode="demo",
                stake_usd=5.0,
                db_path=db_p,
                log_dir=log_d,
                risk_cfg=RiskBreakerConfig(max_stake_usd=20.0, max_daily_loss_usd=200.0, max_drawdown_usd=200.0, same_book_min_gap_seconds=0),
                ev_margin=0.005,
            )
            t0 = datetime(2026, 7, 15, 14, 0, 0, tzinfo=timezone.utc)

            # #2a: Proposal trips KILL mid-flight -> MUST NOT send buy!
            buy_calls = 0

            class _KillDuringQuoteClient:
                async def request(self, payload: dict[str, Any], timeout: float = 5.0) -> dict[str, Any]:
                    nonlocal buy_calls
                    if "proposal" in payload:
                        ex.trip_kill_switch("mid_quote_kill")
                        return {"proposal": {"id": "p-kill", "ask_price": 5.0, "payout": 9.10, "currency": "USD"}}
                    if "buy" in payload:
                        buy_calls += 1
                        return {"buy": {"contract_id": "c-never"}}
                    raise AssertionError(payload)

            ex.client = _KillDuringQuoteClient()  # type: ignore[assignment]
            _, sid_kq = enqueue_sota_signal(
                conn,
                {
                    "book_id": "USDJPY.sota_ens5_m30_up.v1",
                    "pair": "USDJPY",
                    "horizon_minutes": 30,
                    "side": "UP",
                    "signal_close_utc": t0.isoformat(),
                    "effective_floor": 0.58,
                    "calibrated_win_prob": 0.60,
                    "book_content_id": "c8d0b13768a4c649",
                },
            )
            r_kq = await ex.process_one_signal(conn, sid_kq, json.loads(conn.execute("SELECT payload_json FROM sota_signals WHERE signal_id=?", (sid_kq,)).fetchone()[0]), now_utc=t0 + timedelta(seconds=10))
            assert buy_calls == 0 and r_kq["reason"] == "kill_switch_triggered"
            ex.kill_path.unlink(missing_ok=True)

            # #2b: Proposal starts at age 44s, finishes at age 46s -> signal_expired, zero buy!
            ex._post_quote_clock_fn = lambda: t0 + timedelta(seconds=46)  # type: ignore[attr-defined]

            class _SlowQuoteClient:
                async def request(self, payload: dict[str, Any], timeout: float = 5.0) -> dict[str, Any]:
                    nonlocal buy_calls
                    if "proposal" in payload:
                        return {"proposal": {"id": "p-slow", "ask_price": 5.0, "payout": 9.10, "currency": "USD"}}
                    if "buy" in payload:
                        buy_calls += 1
                        return {"buy": {"contract_id": "c-never2"}}
                    raise AssertionError(payload)

            ex.client = _SlowQuoteClient()  # type: ignore[assignment]
            _, sid_exp = enqueue_sota_signal(
                conn,
                {
                    "book_id": "USDJPY.sota_ens5_m30_up.v1",
                    "pair": "USDJPY",
                    "horizon_minutes": 30,
                    "side": "UP",
                    "offset_id": 1,
                    "signal_close_utc": t0.isoformat(),
                    "effective_floor": 0.58,
                    "calibrated_win_prob": 0.60,
                },
            )
            r_exp = await ex.process_one_signal(conn, sid_exp, json.loads(conn.execute("SELECT payload_json FROM sota_signals WHERE signal_id=?", (sid_exp,)).fetchone()[0]), now_utc=t0 + timedelta(seconds=44))
            assert buy_calls == 0 and r_exp["status"] == "signal_expired"
            ex._post_quote_clock_fn = None  # type: ignore[attr-defined]

            # #5: Confirmed buy with malformed start_time='bad' MUST persist contract_id=12345 cleanly!
            class _BadStartBuyClient:
                async def request(self, payload: dict[str, Any], timeout: float = 5.0) -> dict[str, Any]:
                    if "proposal" in payload:
                        return {"proposal": {"id": "p-bads", "ask_price": 5.0, "payout": 9.10, "currency": "USD"}}
                    if "buy" in payload:
                        assert payload.get("passthrough", {}).get("sota_signal_id") is not None
                        return {"buy": {"contract_id": 12345, "buy_price": 5.0, "payout": 9.10, "start_time": "bad", "expiry_time": "1e20"}}
                    raise AssertionError(payload)

            ex.client = _BadStartBuyClient()  # type: ignore[assignment]
            _, sid_bs = enqueue_sota_signal(
                conn,
                {
                    "book_id": "USDJPY.sota_ens5_m30_up.v1",
                    "pair": "USDJPY",
                    "horizon_minutes": 30,
                    "side": "UP",
                    "offset_id": 2,
                    "signal_close_utc": t0.isoformat(),
                    "effective_floor": 0.58,
                    "calibrated_win_prob": 0.60,
                    "book_content_id": "c8d0b13768a4c649",
                },
            )
            r_bs = await ex.process_one_signal(conn, sid_bs, json.loads(conn.execute("SELECT payload_json FROM sota_signals WHERE signal_id=?", (sid_bs,)).fetchone()[0]), now_utc=t0 + timedelta(seconds=2))
            assert r_bs["status"] == "bought" and r_bs["contract_id"] == "12345"
            conn.execute("UPDATE sota_contracts SET status='won', profit_cents=410, closed_utc=? WHERE contract_id='12345'", (t0.isoformat(),))

            # #1: Uncertain buy MUST NOT adopt unrelated 800x contract 987 by proximity, MUST adopt correlated passthrough match!
            _, sid_u = enqueue_sota_signal(
                conn,
                {
                    "book_id": "USDJPY.mult100.v1",
                    "pair": "USDJPY",
                    "horizon_minutes": 30,
                    "side": "UP",
                    "offset_id": 3,
                    "signal_close_utc": t0.isoformat(),
                    "effective_floor": 0.0,
                    "expected_ev": 0.04,
                    "multiplier": 100,
                    "contract_family": "multiplier",
                },
            )
            conn.execute("UPDATE sota_signals SET status='buy_unknown', reason='proposal_id:prop-want-1|buy_transport_uncertain' WHERE signal_id=?", (sid_u,))

            class _UnrelatedStatementClient:
                def __init__(self, include_match: bool = False) -> None:
                    self.include_match = include_match

                async def request(self, payload: dict[str, Any], timeout: float = 5.0) -> dict[str, Any]:
                    if "portfolio" in payload:
                        cs = [{"contract_id": 987, "symbol": "frxUSDJPY", "contract_type": "MULTUP", "date_start": int(t0.timestamp()) + 2}]
                        if self.include_match:
                            cs.append({"contract_id": 988, "symbol": "frxUSDJPY", "contract_type": "MULTUP", "buy_price": 5.0, "passthrough": {"sota_signal_id": sid_u}})
                        return {"portfolio": {"contracts": cs}}
                    if "statement" in payload:
                        return {
                            "statement": {
                                "transactions": [
                                    {"action_type": "buy", "contract_id": 987, "shortcode": "MULTUP_FRXUSDJPY_5.00_800_1", "amount": -5.0, "transaction_time": int(t0.timestamp()) + 2}
                                ]
                            }
                        }
                    if "proposal_open_contract" in payload:
                        return {"proposal_open_contract": {"contract_id": payload["contract_id"], "is_sold": 0, "status": "open"}}
                    raise AssertionError(payload)

            ex.client = _UnrelatedStatementClient(include_match=False)  # type: ignore[assignment]
            await ex.reconcile_open_contracts(conn, now_utc=t0 + timedelta(seconds=10))
            assert conn.execute("SELECT COUNT(*) FROM sota_contracts WHERE contract_id='987'").fetchone()[0] == 0
            assert conn.execute("SELECT status FROM sota_signals WHERE signal_id=?", (sid_u,)).fetchone()[0] == "buy_unknown"

            ex.client = _UnrelatedStatementClient(include_match=True)  # type: ignore[assignment]
            await ex.reconcile_open_contracts(conn, now_utc=t0 + timedelta(seconds=15))
            assert conn.execute("SELECT COUNT(*) FROM sota_contracts WHERE contract_id='988'").fetchone()[0] == 1
            assert conn.execute("SELECT status FROM sota_signals WHERE signal_id=?", (sid_u,)).fetchone()[0] == "bought"

            # #6: sold_for=0.0 sets sell_ack_cents=0, blocks duplicate sell on stale open poll, and settles -500 cents on terminal status-only poll!
            sell_count = 0
            poc_call = 0

            class _ZeroProceedsSellClient:
                async def request(self, payload: dict[str, Any], timeout: float = 5.0) -> dict[str, Any]:
                    nonlocal sell_count, poc_call
                    if "sell" in payload:
                        sell_count += 1
                        return {"sell": {"sold_for": 0.0, "contract_id": payload["sell"]}}
                    if "proposal_open_contract" in payload:
                        poc_call += 1
                        if poc_call in {1, 2, 3}:
                            # Call 1 triggers sell; Call 2 (immediate follow-up) & Call 3 (next loop) return stale open+valid_to_sell
                            return {"proposal_open_contract": {"contract_id": 988, "is_sold": 0, "status": "open", "is_valid_to_sell": 1}}
                        # Call 4 returns terminal sold status WITHOUT sell_price/profit fields
                        return {"proposal_open_contract": {"contract_id": 988, "is_sold": 1, "status": "sold"}}
                    raise AssertionError(payload)

            ex.client = _ZeroProceedsSellClient()  # type: ignore[assignment]
            await ex.reconcile_open_contracts(conn, now_utc=t0 + timedelta(minutes=31))
            await ex.reconcile_open_contracts(conn, now_utc=t0 + timedelta(minutes=31, seconds=5))
            assert sell_count == 1, f"expected exactly 1 sell call after sell_ack_cents=0, got {sell_count}"
            await ex.reconcile_open_contracts(conn, now_utc=t0 + timedelta(minutes=31, seconds=10))
            c988 = conn.execute("SELECT status, sell_price_cents, profit_cents FROM sota_contracts WHERE contract_id='988'").fetchone()
            assert c988["status"] == "lost" and int(c988["sell_price_cents"]) == 0 and int(c988["profit_cents"]) == -500

            # #7: Live account_mode or wrong book_content_id MUST block promotion even at n >= 50!
            for idx_l in range(55):
                conn.execute(
                    "INSERT INTO sota_signals VALUES (?,?,?,?,?,'wall',?,?,'0.58','0.60','{}','contract_closed',NULL,?,?)",
                    (f"sig-lv-{idx_l}", "USDJPY.sota_ens5_m30_up.v1", "USDJPY", 30, "UP", 100 + idx_l, t0.isoformat(), t0.isoformat(), t0.isoformat()),
                )
                conn.execute(
                    """
                    INSERT INTO sota_contracts (
                        contract_id, signal_id, book_id, pair, horizon_minutes, side,
                        proposal_id, buy_price_cents, payout_cents, sell_price_cents, profit_cents,
                        status, opened_utc, expected_expiry_utc, closed_utc, raw_hash,
                        signal_close_utc, contract_family, account_mode, book_content_id
                    ) VALUES (?,?,?,?,?,?,?,100,182,182,82,'won',?,?,?,'h',?,'risefall','live','wrong_cid_000000')
                    """,
                    (f"c-lv-{idx_l}", f"sig-lv-{idx_l}", "USDJPY.sota_ens5_m30_up.v1", "USDJPY", 30, "UP", f"p-lv-{idx_l}", t0.isoformat(), t0.isoformat(), t0.isoformat(), t0.isoformat()),
                )
            rep_bad = summarize_settled_queue(db_p, min_settled_for_promotion=50, book_id="USDJPY.sota_ens5_m30_up.v1")
            assert rep_bad["promotion_eligible"] is False
            assert any(r.startswith("non_demo_provenance_rows:") for r in rep_bad["reasons"])
            assert any(r.startswith("mixed_book_content_ids:") or r.startswith("book_content_id_mismatch:") for r in rep_bad["reasons"])
            conn.close()

            # #10: Preflight smoke rejects empty `{}` proposal response
            import vps_preflight as vp_mod

            orig_assert = vp_mod.assert_non_us_deriv_host
            orig_env = vp_mod.DerivEnv.from_env
            orig_req_ws = vp_mod.request_authenticated_ws_url
            orig_val = vp_mod.validate_account_mode
            orig_client = vp_mod.DerivAsyncClient

            class _EmptyPropClient:
                def __init__(self, *a: Any, **kw: Any) -> None:
                    pass

                async def __aenter__(self) -> "_EmptyPropClient":
                    return self

                async def __aexit__(self, *a: Any) -> None:
                    return None

                async def request(self, payload: dict[str, Any], timeout: float = 8.0) -> dict[str, Any]:
                    if "authorize" in payload:
                        return {"authorize": {"loginid": "VRTC123", "currency": "USD", "is_virtual": 1}}
                    return {}

            try:
                vp_mod.assert_non_us_deriv_host = lambda *a, **kw: None  # type: ignore[assignment]
                vp_mod.DerivEnv.from_env = classmethod(lambda cls: type("E", (), {"account_id": "VRTC123", "pat": "tok", "app_id": "1089"})())  # type: ignore[assignment]
                vp_mod.request_authenticated_ws_url = lambda env, mode: ("wss://ws.derivws.com/websockets/v3?app_id=1089", "v3_authorize")  # type: ignore[assignment]
                vp_mod.validate_account_mode = lambda **kw: None  # type: ignore[assignment]
                vp_mod.DerivAsyncClient = _EmptyPropClient  # type: ignore[assignment]
                repo_real = Path(__file__).resolve().parents[1]
                b1 = load_enabled_sota_books(repo_root=repo_real, account_mode="demo", book_ids=["USDJPY.sota_ens5_m30_up.v1"])
                smoke_res = await run_network_proposal_smoke("demo", b1, 1.0)
                assert smoke_res[0]["proposal_received"] is False

                class _BadRatioPropClient(_EmptyPropClient):
                    async def request(self, payload: dict[str, Any], timeout: float = 8.0) -> dict[str, Any]:
                        if "authorize" in payload:
                            return {"authorize": {"loginid": "VRTC123", "currency": "USD", "is_virtual": 1}}
                        return {"proposal": {"id": "p1", "ask_price": 5.0, "payout": 4.5, "currency": "USD"}}

                vp_mod.DerivAsyncClient = _BadRatioPropClient  # type: ignore[assignment]
                smoke_bad_ratio = await run_network_proposal_smoke("demo", b1, 1.0)
                assert smoke_bad_ratio[0]["proposal_received"] is False
            finally:
                vp_mod.assert_non_us_deriv_host = orig_assert  # type: ignore[assignment]
                vp_mod.DerivEnv.from_env = orig_env  # type: ignore[assignment]
                vp_mod.request_authenticated_ws_url = orig_req_ws  # type: ignore[assignment]
                vp_mod.validate_account_mode = orig_val  # type: ignore[assignment]
                vp_mod.DerivAsyncClient = orig_client  # type: ignore[assignment]

    asyncio.run(_exec_hardening())


def main() -> int:
    tests = [
        test_quote_parser_and_dual_ev_gate,
        test_account_mode_isolation,
        test_session_cutoffs_blackout_and_haircut,
        test_currency_legs_and_risk_circuit_breakers,
        test_book_freeze_parity_and_certificate_gates,
        test_tick_feature_strict_causality,
        test_async_executor_and_live_pnl_promotion_gate,
        test_real_frozen_sota_books_end_to_end,
        test_nystroem_ens_multiplier_and_barrier_extensions,
        test_da2_adversarial_hardening_regressions,
    ]
    for fn in tests:
        fn()
        print(f"PASS: {fn.__name__}")
    print(f"ALL {len(tests)} SOTA RUNTIME SAFETY SUITES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
