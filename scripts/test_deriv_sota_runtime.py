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
    # Valid Demo (WS v3 VRTC*/VRW* and Options v1 D[A-Z]{2}[0-9]+)
    validate_account_mode(
        account_mode="demo",
        account_id="VRTC991122",
        ws_url="wss://api.derivws.com/trading/v1/options/ws/demo",
        env_live_flag="0",
    )
    validate_account_mode(
        account_mode="demo",
        account_id="VRW12345678",
        ws_url="wss://ws.derivws.com/websockets/v3?app_id=1089",
        env_live_flag="0",
    )
    validate_account_mode(
        account_mode="demo",
        account_id="DMO12345678",
        ws_url="wss://api.derivws.com/trading/v1/options/ws/demo",
        env_live_flag="0",
    )
    # Options v1 3-letter D-prefixed ID MUST be refused on WS v3 and in live mode!
    _expect_raises(ExecutorError, validate_account_mode, account_mode="demo", account_id="DMO12345678", ws_url="wss://ws.derivws.com/websockets/v3?app_id=1089", env_live_flag="0")
    _expect_raises(ExecutorError, validate_account_mode, account_mode="demo", account_id="DMO12345678", ws_url="wss://api.derivws.com/trading/v1/options/ws/live", env_live_flag="0")
    _expect_raises(ExecutorError, validate_account_mode, account_mode="live", account_id="DMO12345678", ws_url="wss://api.derivws.com/trading/v1/options/ws/live", env_live_flag="1")
    # options_otp transport MUST refuse WS v3 URL even for VRTC/VRW, and MUST NOT leak ?otp=... in error messages!
    try:
        validate_account_mode(account_mode="demo", account_id="VRTC991122", ws_url="wss://ws.derivws.com/websockets/v3?otp=SECRET_OTP_TOK", env_live_flag="0", expected_transport="options_otp")
        raise AssertionError("expected options_otp refusal for v3 URL")
    except ExecutorError as exc:
        assert "SECRET_OTP_TOK" not in str(exc)
    for bad_demo_acct in ("CR887766", "MF123456", "MLT123456", "MX123456", "REAL123456", "VRTC_FROM_ENV", "DMO", "D12345678"):
        _expect_raises(ExecutorError, validate_account_mode, account_mode="demo", account_id=bad_demo_acct, ws_url="wss://api.derivws.com/trading/v1/options/ws/demo", env_live_flag="0")
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
            orig_client = vp_mod.SotaDerivAsyncClient

            class _EmptyPropClient:
                def __init__(self, *a: Any, **kw: Any) -> None:
                    self.allow_buy = False
                    self.expected_account_id = "VRTC123"
                    self.url = "wss://ws.derivws.com/websockets/v3?app_id=1089"

                async def connect(self) -> None:
                    return None

                async def close(self) -> None:
                    return None

                def is_identity_verified(self) -> bool:
                    return False

                async def establish_identity(self, **kw: Any) -> int:
                    return 1780000000

                async def request_proposal(self, payload: dict[str, Any], timeout: float = 8.0) -> dict[str, Any]:
                    return {}

            try:
                vp_mod.assert_non_us_deriv_host = lambda *a, **kw: None  # type: ignore[assignment]
                vp_mod.DerivEnv.from_env = classmethod(lambda cls: type("E", (), {"account_id": "VRTC123", "pat": "tok", "app_id": "1089"})())  # type: ignore[assignment]
                vp_mod.request_authenticated_ws_url = lambda env, mode: ("wss://ws.derivws.com/websockets/v3?app_id=1089", "v3_authorize")  # type: ignore[assignment]
                vp_mod.validate_account_mode = lambda **kw: None  # type: ignore[assignment]
                vp_mod.SotaDerivAsyncClient = _EmptyPropClient  # type: ignore[assignment]
                repo_real = Path(__file__).resolve().parents[1]
                b1 = load_enabled_sota_books(repo_root=repo_real, account_mode="demo", book_ids=["USDJPY.sota_ens5_m30_up.v1"])
                smoke_res = await run_network_proposal_smoke("demo", b1, 1.0, pace_s=0.0)
                assert smoke_res[0]["proposal_received"] is False and smoke_res[0]["smoke_passed"] is False

                class _BadRatioPropClient(_EmptyPropClient):
                    async def request_proposal(self, payload: dict[str, Any], timeout: float = 8.0) -> dict[str, Any]:
                        return {"proposal": {"id": "p1", "ask_price": 5.0, "payout": 4.5, "currency": "USD"}}

                vp_mod.SotaDerivAsyncClient = _BadRatioPropClient  # type: ignore[assignment]
                smoke_bad_ratio = await run_network_proposal_smoke("demo", b1, 1.0, pace_s=0.0)
                assert smoke_bad_ratio[0]["proposal_received"] is False and smoke_bad_ratio[0]["smoke_passed"] is False
            finally:
                vp_mod.assert_non_us_deriv_host = orig_assert  # type: ignore[assignment]
                vp_mod.DerivEnv.from_env = orig_env  # type: ignore[assignment]
                vp_mod.request_authenticated_ws_url = orig_req_ws  # type: ignore[assignment]
                vp_mod.validate_account_mode = orig_val  # type: ignore[assignment]
                vp_mod.SotaDerivAsyncClient = orig_client  # type: ignore[assignment]

    asyncio.run(_exec_hardening())


# ---------------------------------------------------------------------------
# Phase 1 Comprehensive Unit, Negative-Control & Consumer Proof Suites (§6 #1)
# ---------------------------------------------------------------------------


def _fit_synthetic_stack05_bundle(n_rows: int = 240, n_feat: int = 8, seed: int = 11) -> dict[str, Any]:
    import lightgbm as lgb
    from sota_book_freeze import (
        extract_nystroem_lgb_stack_arrays,
        serialize_sota_lgb_txt_bytes,
        serialize_sota_strategy_bytes,
        serialize_sota_weights_npz_bytes,
    )
    from sota_book_runtime import compute_sota_content_id_from_buffers

    rng = np.random.default_rng(seed)
    fcols = [f"f_{i}" for i in range(n_feat)]
    X = rng.normal(size=(n_rows, n_feat)).astype(np.float64)
    X[1, 0] = np.nan
    X[2, 1] = np.nan
    y = ((np.nan_to_num(X[:, 0]) + 0.7 * np.nan_to_num(X[:, 1]) + rng.normal(scale=0.4, size=n_rows)) > 0).astype(int)
    med = np.nanmedian(X, axis=0)
    X_imp = np.where(np.isfinite(X), X, med)
    mu = X_imp.mean(axis=0)
    sd = X_imp.std(axis=0) + 1e-12
    Z = np.clip((X_imp - mu) / sd, -5.0, 5.0)

    ny_list, sgd_list, p_nys_list = [], [], []
    for s_i in range(5):
        ny_s = Nystroem(kernel="rbf", gamma=1.0 / n_feat, n_components=24, random_state=s_i).fit(Z)
        sgd_s = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=30, tol=1e-4, random_state=s_i).fit(ny_s.transform(Z), y)
        ny_list.append(ny_s)
        sgd_list.append(sgd_s)
        p_nys_list.append(sgd_s.predict_proba(ny_s.transform(Z))[:, 1])
    p_nys = np.mean(p_nys_list, axis=0)

    clf = lgb.LGBMClassifier(
        n_estimators=20,
        learning_rate=0.05,
        num_leaves=7,
        min_child_samples=10,
        random_state=42,
        deterministic=True,
        force_col_wise=True,
        n_jobs=1,
        verbose=-1,
    )
    clf.fit(X, y, feature_name=fcols)
    p_lgb = clf.booster_.predict(X, num_iteration=20)
    p_raw = 0.5 * p_nys + 0.5 * p_lgb

    # Construct 7-input parity rows covering finite UP, <50%-NaN, UP boundary, and disallowed DOWN!
    order = np.argsort(p_raw)
    pick_idx = [int(order[-1]), 1, int(order[len(order) // 2]), int(order[0])]
    p_sub = p_raw[pick_idx]
    conf_thr = float(max(0.01, abs(p_sub[2] - 0.5) * 0.5))
    coef, intercept = 1.2, 0.05
    logits = np.clip(np.log(np.clip(p_sub, 1e-6, 1.0 - 1e-6) / (1.0 - np.clip(p_sub, 1e-6, 1.0 - 1e-6))), -15.0, 15.0)
    p_up_cal = 1.0 / (1.0 + np.exp(-(coef * logits + intercept)))
    dirs = ["UP" if p >= 0.5 else "DOWN" for p in p_sub]
    gates = [bool(d == "UP" and abs(p - 0.5) >= conf_thr) for d, p in zip(dirs, p_sub)]
    cal_present = [bool(d == "UP") for d in dirs]
    epochs = np.full(len(pick_idx), int(datetime(2026, 7, 15, 14, 0, tzinfo=timezone.utc).timestamp()), dtype=np.int64)

    weights_arrays = extract_nystroem_lgb_stack_arrays(med, mu, sd, ny_list, sgd_list, blend_weight=0.5)
    lgb_bytes = serialize_sota_lgb_txt_bytes(clf, fcols, num_iteration=20)
    npz_bytes = serialize_sota_weights_npz_bytes(weights_arrays)
    strat = {
        "book_id": "USDJPY.test_stack05.v1",
        "pair": "USDJPY",
        "horizon_minutes": 30,
        "allowed_sides": ["UP"],
        "model_kind": "nystroem-lgb-stack",
        "contract_family": "risefall",
        "feature_cols": fcols,
        "coverage": 0.05,
        "confidence_threshold": conf_thr,
        "refit_cpcv_p10_floors": {"UP": 0.58},
        "payout_edge_margin": 0.005,
        "min_payout_ratio": 1.78,
        "drop_ny_16h": False,
        "platt_calibration": {"UP": {"coef": coef, "intercept": intercept}},
    }
    strat_bytes = serialize_sota_strategy_bytes(strat)
    cid = compute_sota_content_id_from_buffers(
        [
            ("USDJPY.test_stack05.v1_strategy.json", strat_bytes),
            ("USDJPY.test_stack05.v1_weights.npz", npz_bytes),
            ("USDJPY.test_stack05.v1_lgb_s0.txt", lgb_bytes),
        ]
    )
    return {
        "fcols": fcols,
        "df_parity": pd.DataFrame(X[pick_idx], columns=fcols),
        "epochs": epochs,
        "p_raw": p_sub,
        "p_up_cal": p_up_cal,
        "dirs": dirs,
        "gates": gates,
        "cal_present": cal_present,
        "conf_thr": conf_thr,
        "platt": {"UP": {"coef": coef, "intercept": intercept}},
        "weights_arrays": weights_arrays,
        "lgb_clf": clf,
        "lgb_bytes": lgb_bytes,
        "npz_bytes": npz_bytes,
        "strat_bytes": strat_bytes,
        "content_id": cid,
        "X": X,
        "y": y,
    }


def test_phase1_stack05_runtime_and_freezer() -> None:
    import hashlib
    import os
    import subprocess
    import sys
    import lightgbm as lgb
    import certify_sota_stack05 as cert_mod
    import sota_book_freeze as freeze_mod
    from build_certified_sota_books import (
        install_staged_stack05,
        stage_stack05_in_lab_dir,
        verify_staged_stack05_in_dir,
    )
    from sota_book_freeze import (
        canonical_deterministic_manifest_sha256,
        complete_interrupted_stack_index_install,
        serialize_sota_lgb_txt_bytes,
        serialize_sota_weights_npz_bytes,
    )
    from sota_book_runtime import check_pipeline_features_dir_exists

    syn = _fit_synthetic_stack05_bundle()
    # 1. Byte determinism + feature_name enforcement (unnamed Column_0.. booster MUST be rejected!)
    assert serialize_sota_weights_npz_bytes(syn["weights_arrays"]) == syn["npz_bytes"]
    assert serialize_sota_lgb_txt_bytes(syn["lgb_clf"], syn["fcols"], num_iteration=20) == syn["lgb_bytes"]
    unnamed_clf = lgb.LGBMClassifier(n_estimators=10, num_leaves=7, min_child_samples=10, random_state=42, verbose=-1)
    unnamed_clf.fit(syn["X"], syn["y"])  # omits feature_name -> emits Column_0..
    _expect_raises(SotaBookError, serialize_sota_lgb_txt_bytes, unnamed_clf, syn["fcols"], num_iteration=10)

    # 2. Pre-import /home/sean/git/binary-algo/features existence check & sys.path / 15-file closure
    check_pipeline_features_dir_exists(Path("/home/sean/git/binary-algo/features"))
    _expect_raises(SotaBookError, check_pipeline_features_dir_exists, Path("/tmp/definitely_missing_features_dir_xyz"))
    # Calling inside test_deriv_sota_runtime.py MUST reject non-closure module (__main__)!
    _expect_raises(cert_mod.OperationalHoldError, cert_mod.verify_source_closure_and_loaded_modules, Path(__file__).resolve().parents[1])
    # Clean subprocess importing ONLY certify_sota_stack05 MUST pass with exact 15-file closure!
    clo_res = subprocess.run(
        [
            sys.executable, "-B", "-E", "-s", "-c",
            "import sys, json; sys.path.insert(0, 'scripts'); import certify_sota_stack05 as c; s, m = c.verify_source_closure_and_loaded_modules(); print(json.dumps({'sha': s, 'n': len(m)}))",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        check=False,
    )
    assert clo_res.returncode == 0, f"clean closure check failed: {clo_res.stdout} {clo_res.stderr}"
    clo_obj = json.loads(clo_res.stdout)
    assert len(clo_obj["sha"]) == 64 and clo_obj["n"] == 15

    # 3. O_NOFOLLOW DEV loader symlink & hash mismatch negative controls
    with tempfile.TemporaryDirectory() as td_nf:
        tdp = Path(td_nf)
        real_f = tdp / "a.bin"
        real_f.write_bytes(b"hello-world")
        sym_f = tdp / "sym.bin"
        sym_f.symlink_to(real_f)
        assert cert_mod.read_verified_file_nofollow(tdp, ["a.bin"], expected_bytes=11, expected_sha256=hashlib.sha256(b"hello-world").hexdigest()) == b"hello-world"
        _expect_raises(cert_mod.OperationalHoldError, cert_mod.read_verified_file_nofollow, tdp, ["sym.bin"])
        _expect_raises(cert_mod.OperationalHoldError, cert_mod.read_verified_file_nofollow, tdp, ["a.bin"], expected_sha256="0" * 64)

    # 4. 7-input staged parity, D2 staged fsync, B4 dual books/ fsync, pre-rename expected install + 0444 builder_suffix,
    #    and B2 zero-fit interrupted-index recovery after book rename before first books/ barrier!
    dev_certs = {
        "G1_cpcv": {"passed": True, "p10_wr": 0.58},
        "G2_placebo": {"passed": True, "mean_excess": 0.05},
        "G3_entry_delay": {"passed": True, "d4_p10": 0.57},
        "G4_sign_null": {"passed": True, "null_max": 0.52},
        "G5_payout": {"passed": True, "p10_ev": 0.05},
    }
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "books").mkdir()
        (root / "logs").mkdir()
        arch_dir = root / "lab" / "archive" / "stack05_m30_up_v1"
        arch_dir.mkdir(parents=True)
        exp_install_p = root / "logs" / "sota_stack05_expected_install.json"
        bsuf_p = arch_dir / "builder_suffix.json"

        # Spy on fsync_file & fsync_dir during interrupted install (interrupt right after rename_noreplace, before first books/ fsync!)
        fsync_events: list[tuple[str, str]] = []
        orig_ff = freeze_mod.fsync_file
        orig_fd = freeze_mod.fsync_dir

        def _spy_ff(p: Path) -> None:
            fsync_events.append(("file", str(p.resolve())))
            orig_ff(p)

        def _spy_fd(p: Path) -> None:
            fsync_events.append(("dir", str(p.resolve())))
            orig_fd(p)

        class _InjectedCrash(RuntimeError):
            pass

        def _crash_after_rename() -> None:
            raise _InjectedCrash("crash_right_after_book_rename_before_books_fsync")

        try:
            freeze_mod.fsync_file = _spy_ff  # type: ignore[assignment]
            freeze_mod.fsync_dir = _spy_fd  # type: ignore[assignment]
            try:
                freeze_mod.freeze_sota_book(
                    book_id="USDJPY.test_stack05.v1",
                    pair="USDJPY",
                    horizon_minutes=30,
                    allowed_sides=["UP"],
                    lifecycle_status=DEMO_CANARY_LIFECYCLE,
                    model_kind="nystroem-lgb-stack",
                    feature_cols=syn["fcols"],
                    coverage=0.05,
                    confidence_threshold=syn["conf_thr"],
                    refit_cpcv_p10_floors={"UP": 0.58},
                    platt_calibration=syn["platt"],
                    gate_certificates=dev_certs,
                    summary="test stack05",
                    metrics={"p10_wr": 0.58},
                    source_script="scripts/test_deriv_sota_runtime.py",
                    weights_arrays=syn["weights_arrays"],
                    lgb_model_bytes=("USDJPY.test_stack05.v1_lgb_s0.txt", syn["lgb_bytes"]),
                    min_payout_ratio=1.78,
                    parity_check_df=syn["df_parity"],
                    parity_check_epochs=syn["epochs"],
                    parity_expected_probs=syn["p_raw"],
                    parity_expected_up_cal_probs=syn["p_up_cal"],
                    parity_expected_directions=syn["dirs"],
                    parity_expected_gates=syn["gates"],
                    parity_expected_row_cal_present=syn["cal_present"],
                    expected_content_id=syn["content_id"],
                    expected_install_path=exp_install_p,
                    builder_suffix_path=bsuf_p,
                    builder_suffix_metadata={"schema": "sota-stack05-builder-suffix/v1", "book_id": "USDJPY.test_stack05.v1"},
                    archive_sync_stop_at=root / "lab",
                    pre_index_hook=_crash_after_rename,
                    repo_root=root,
                )
                raise AssertionError("expected _InjectedCrash")
            except _InjectedCrash:
                pass

            # Verify D2 + MAJOR #5: staged files & staged dir were fsynced BEFORE descriptor/suffix publish,
            # AND pre_index_hook fired right after rename_noreplace BEFORE the first books/ fsync!
            books_dir_res = str((root / "books").resolve())
            assert any(kind == "file" and "_lgb_s0.txt" in path for kind, path in fsync_events)
            assert any(kind == "dir" and ".stage_USDJPY.test_stack05.v1_" in path for kind, path in fsync_events)
            assert books_dir_res not in [p for k, p in fsync_events if k == "dir"], (
                f"pre_index_hook must fire BEFORE first books/ fsync, saw: {fsync_events}"
            )
            # Verify pre-rename durable files exist, builder_suffix is 0444, book dir exists, SOTA_INDEX.json does NOT yet exist!
            assert exp_install_p.is_file() and bsuf_p.is_file()
            assert (os.stat(bsuf_p).st_mode & 0o777) == 0o444
            assert (root / "books" / "USDJPY.test_stack05.v1").is_dir()
            assert not (root / "books" / "SOTA_INDEX.json").exists()

            # B2 Zero-fit interrupted-index recovery: MUST fsync books/ BEFORE AND AFTER index replacement, then retire marker!
            fsync_events.clear()
            complete_interrupted_stack_index_install(
                book_id="USDJPY.test_stack05.v1",
                repo_root=root,
                expected_install_path=exp_install_p,
            )
            books_sync_count = sum(1 for k, p in fsync_events if k == "dir" and p == books_dir_res)
            assert books_sync_count >= 2, f"expected >=2 books/ fsyncs (before & after index), got {books_sync_count}"
            assert not exp_install_p.exists(), "expected_install marker was not retired after index publish"
            assert (root / "books" / "SOTA_INDEX.json").is_file()
            loaded = load_sota_book("USDJPY.test_stack05.v1", repo_root=root, allow_demo_canary=True)
            assert loaded.content_id == syn["content_id"]
            assert loaded.boosters is not None and loaded.boosters[0].feature_name() == syn["fcols"]

            # Tampering booster text to unnamed Column_0.. MUST fail reload!
            man_path = root / "books" / "USDJPY.test_stack05.v1" / "USDJPY.test_stack05.v1.manifest.json"
            assert len(canonical_deterministic_manifest_sha256(json.loads(man_path.read_text()))) == 64

            # Verify check_book_manifests.py accepts sota-book-manifest/v1 with pair & rejects conflicting currency!
            chk = subprocess.run(
                [sys.executable, "-B", str(Path(__file__).resolve().parent / "check_book_manifests.py")],
                cwd=str(Path(__file__).resolve().parents[1]),
                capture_output=True,
                text=True,
                check=False,
            )
            assert chk.returncode == 0, f"check_book_manifests failed: {chk.stdout} {chk.stderr}"
        finally:
            freeze_mod.fsync_file = orig_ff  # type: ignore[assignment]
            freeze_mod.fsync_dir = orig_fd  # type: ignore[assignment]

    # 5. B1: Fresh archive tree & custody retry sync $A, $LAB/archive, and $LAB bottom-up!
    with tempfile.TemporaryDirectory() as td_b1:
        root_b1 = Path(td_b1)
        src_t = root_b1 / "src_t"
        src_t.mkdir()
        ck_root = root_b1 / "checkout"
        (ck_root / "results" / "json").mkdir(parents=True)
        (ck_root / "logs").mkdir(parents=True)
        lab_root = root_b1 / "lab"
        lab_root.mkdir()
        arch_a = lab_root / "archive" / "stack05_m30_up_v1"
        for fname in (
            cert_mod.CANONICAL_SCORES_NAME,
            cert_mod.CANONICAL_LEDGER_NAME,
            cert_mod.CANONICAL_DOSSIER_NAME,
            cert_mod.CANONICAL_RECEIPT_NAME,
        ):
            (src_t / fname).write_bytes(
                b'{"arm_name":"stack05_m30_up_v1"}\n' if fname == cert_mod.CANONICAL_RECEIPT_NAME else f"payload:{fname}\n".encode("utf-8")
            )
        synced_dirs: list[str] = []
        orig_cfd = cert_mod.fsync_dir
        orig_ffd = freeze_mod.fsync_dir
        try:
            def _spy_cfd(p: Path) -> None:
                synced_dirs.append(str(p.resolve()))
                orig_ffd(p)

            cert_mod.fsync_dir = _spy_cfd  # type: ignore[assignment]
            freeze_mod.fsync_dir = _spy_cfd  # type: ignore[assignment]
            # Fresh publish
            cert_mod.publish_certifier_prefix_bundle(src_dir=src_t, checkout_root=ck_root, archive_dir=arch_a, lab_root=lab_root)
            want_order = [str(arch_a.resolve()), str((lab_root / "archive").resolve()), str(lab_root.resolve())]
            # Assert bottom-up subsequence $A -> $LAB/archive -> $LAB on fresh publish
            pos = [synced_dirs.index(d) for d in want_order]
            assert pos == sorted(pos), f"fresh archive ancestry order wrong: {synced_dirs}"
            for fname in (cert_mod.CANONICAL_SCORES_NAME, cert_mod.CANONICAL_LEDGER_NAME, cert_mod.CANONICAL_DOSSIER_NAME, cert_mod.CANONICAL_RECEIPT_NAME):
                assert (os.stat(arch_a / fname).st_mode & 0o777) == 0o444

            # Retry publish MUST re-sync $A -> $LAB/archive -> $LAB bottom-up idempotently!
            synced_dirs.clear()
            cert_mod.publish_certifier_prefix_bundle(src_dir=src_t, checkout_root=ck_root, archive_dir=arch_a, lab_root=lab_root)
            pos2 = [synced_dirs.index(d) for d in want_order]
            assert pos2 == sorted(pos2), f"retry archive ancestry order wrong: {synced_dirs}"
        finally:
            cert_mod.fsync_dir = orig_cfd  # type: ignore[assignment]
            freeze_mod.fsync_dir = orig_ffd  # type: ignore[assignment]

    # 6. B3: Interrupted/uncertain capped launch (absent debit vs spent debit with incomplete stage evidence) -> HOLD!
    with tempfile.TemporaryDirectory() as td_b3:
        root_b3 = Path(td_b3)
        ck = root_b3 / "ck"
        (ck / "results" / "json").mkdir(parents=True)
        (ck / "logs").mkdir(parents=True)
        stage_t = root_b3 / "stage_t"
        stage_t.mkdir()
        prereg_sha = "a" * 64
        dos_stub = {
            "schema": "sota-5gate-certification-dossier/v3",
            "book_id": "USDJPY.sota_stack05_m30_up.v1",
            "verdict": "PASS",
            "all_5_gates_passed": True,
            "drift_check_passed": True,
            "preregistration_sha256": prereg_sha,
            "G1_cpcv": {"passed": True, "p10_wr": 0.59},
            "G2_placebo": {"passed": True},
            "G3_entry_delay": {"passed": True},
            "G4_sign_null": {"passed": True, "verdict": "DESCRIPTIVE_RANK_DIAGNOSTIC_PASS"},
            "G5_payout": {"passed": True, "classification": "fixed_ratio_dev_sensitivity_only_not_decision5"},
            "final_fit_expected_content_id": syn["content_id"],
        }
        dos_b = cert_mod.pretty_json_bytes(dos_stub)
        (ck / "results" / "json" / cert_mod.CANONICAL_DOSSIER_NAME).write_bytes(dos_b)
        led_0 = {"schema": cert_mod.FIT_LEDGER_SCHEMA, "preregistration_sha256": prereg_sha, "max_fit_units": 263, "attempts": []}
        led_0_b = cert_mod.canonical_json_bytes(led_0)
        (ck / "logs" / cert_mod.CANONICAL_LEDGER_NAME).write_bytes(led_0_b)
        rcpt_obj = {
            "schema": cert_mod.RECEIPT_SCHEMA,
            "verdict": "PASS",
            "dossier_sha256": hashlib.sha256(dos_b).hexdigest(),
            "certifier_attempt_count": 0,
            "certifier_fit_ledger_prefix_sha256": hashlib.sha256(led_0_b).hexdigest(),
            "preregistration_sha256": prereg_sha,
            "final_fit_expected_content_id": syn["content_id"],
            "final_fit_expected_artifacts": [
                {"filename": "USDJPY.sota_stack05_m30_up.v1_strategy.json", "bytes": len(syn["strat_bytes"]), "sha256": hashlib.sha256(syn["strat_bytes"]).hexdigest()},
            ],
        }
        rcpt_b = cert_mod.pretty_json_bytes(rcpt_obj)
        rcpt_sha = hashlib.sha256(rcpt_b).hexdigest()
        (ck / "logs" / cert_mod.CANONICAL_RECEIPT_NAME).write_bytes(rcpt_b)

        # Case 6a: Absent debit + missing staged_install_manifest.json -> verify_staged_stack05_in_dir HOLDS specifically on incomplete stage!
        try:
            verify_staged_stack05_in_dir(stage_dir=stage_t, expected_receipt_sha256=rcpt_sha, repo_root=ck)
            raise AssertionError("expected SotaBookError on incomplete stage")
        except SotaBookError as exc:
            assert "incomplete staged evidence" in str(exc), f"unexpected error: {exc}"

        # Case 6b: Spent builder debit in $T ledger + incomplete stage evidence -> HOLDS specifically on debit check AND ledger stays intact!
        led_1 = {
            "schema": cert_mod.FIT_LEDGER_SCHEMA,
            "preregistration_sha256": prereg_sha,
            "max_fit_units": 263,
            "attempts": [{"attempt_id": 1, "stage": "builder", "role": "builder_final_fit", "preregistration_sha256": prereg_sha, "started_utc": "2026-11-25T00:00:00Z"}],
        }
        led_1_b = cert_mod.canonical_json_bytes(led_1)
        (stage_t / cert_mod.CANONICAL_LEDGER_NAME).write_bytes(led_1_b)
        try:
            verify_staged_stack05_in_dir(stage_dir=stage_t, expected_receipt_sha256=rcpt_sha, repo_root=ck)
            raise AssertionError("expected SotaBookError on spent debit with incomplete stage")
        except SotaBookError as exc:
            assert "incomplete staged evidence" in str(exc), f"unexpected error: {exc}"
        assert (stage_t / cert_mod.CANONICAL_LEDGER_NAME).read_bytes() == led_1_b
        # Re-invoking stage_stack05_in_lab_dir on spent debit with incomplete stage MUST reach debit check & refuse without overwriting ledger!
        prog_p = root_b3 / "PROGRESS.md"
        prog_p.write_text("# progress\n", encoding="utf-8")
        try:
            stage_stack05_in_lab_dir(
                out_dir=stage_t,
                expected_receipt_sha256=rcpt_sha,
                progress_md=prog_p,
                repo_root=ck,
                allow_fixture_digest_override=True,
                fixture_root=root_b3,
            )
            raise AssertionError("expected SotaBookError on spent builder debit")
        except SotaBookError as exc:
            assert "builder debit already spent" in str(exc), f"expected debit refusal, got: {exc}"
        assert (stage_t / cert_mod.CANONICAL_LEDGER_NAME).read_bytes() == led_1_b
        import build_certified_sota_books as bcsb_mod
        assert hasattr(bcsb_mod, "os") and callable(bcsb_mod.os.fsync), "build_certified_sota_books must import os for stage fsync"
        assert callable(install_staged_stack05)


def test_phase1_certifier_reducers_and_verify_bundle() -> None:
    import copy
    import hashlib
    import certificate as cert_check_mod
    import certify_sota_stack05 as cert_mod
    import deriv_dataset as dd_mod

    # 1. settle_with_intervals bit-identical to deriv_dataset.settle() at d=0 + [0,1]->(inf,False) + empty ticks safe!
    tick_ts = np.array([1000, 1001, 1061, 2801, 2860, 4000, 5800], dtype=np.int64)
    tick_px = np.array([150.0, 0.0, 150.1, 1.0, 150.5, 151.0, 151.0], dtype=np.float64)
    dec_ts = np.array([1000, 1060, 3999, 9000], dtype=np.int64)  # includes zero-price [0,1]->(inf,False), valid win, tie, out-of-range
    r_ref, v_ref = dd_mod.settle(dec_ts, tick_ts, tick_px, horizon_s=1800, tol_s=60)
    r_got, v_got, ei_got, xi_got = cert_mod.settle_with_intervals(dec_ts, tick_ts, tick_px, horizon_s=1800, delay_s=0, tol_s=60)
    assert np.array_equal(v_got, v_ref)
    assert np.array_equal(np.isnan(r_got), np.isnan(r_ref))
    assert np.array_equal(np.isinf(r_got), np.isinf(r_ref))
    fin = np.isfinite(r_ref)
    assert np.array_equal(r_got[fin], r_ref[fin])
    assert not v_got[0] and np.isinf(r_got[0])  # [0, 1] -> (inf, False)!
    assert ei_got[0] == -1 and xi_got[0] == -1
    # Empty ticks -> safe no IndexError!
    r_e, v_e, ei_e, xi_e = cert_mod.settle_with_intervals(dec_ts, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64))
    assert len(r_e) == len(dec_ts) and not v_e.any() and np.all(ei_e == -1) and np.all(xi_e == -1)

    # Empty registered DEV tick file -> OperationalHoldError before any fit!
    with tempfile.TemporaryDirectory() as td_emp:
        tdp = Path(td_emp)
        ddir = tdp / "dev"
        tdir = tdp / "ticks" / "USDJPY"
        ddir.mkdir()
        tdir.mkdir(parents=True)
        empty_pq = tdir / "USDJPY_2025-08-11_ticks.parquet"
        pd.DataFrame({"datetime_utc": pd.Series([], dtype="datetime64[ns, UTC]"), "price": pd.Series([], dtype="float64")}).to_parquet(empty_pq, index=False)
        raw_pq = empty_pq.read_bytes()
        man = {"inputs": {"USDJPY": [{"file": empty_pq.name, "bytes": len(raw_pq), "sha256": hashlib.sha256(raw_pq).hexdigest()}]}}
        man_b = json.dumps(man).encode("utf-8")
        (ddir / "manifest.json").write_bytes(man_b)
        _expect_raises(
            cert_mod.OperationalHoldError,
            cert_mod.load_pinned_dev_ticks,
            "USDJPY",
            dev_dir=ddir,
            ticks_root=tdp / "ticks",
            expected_manifest_sha256=hashlib.sha256(man_b).hexdigest(),
        )

    # 2. split_train_val support guards + null receiver-cut/donor-purge precedence!
    n_p = 4000
    ts_s = np.arange(1755000000, 1755000000 + n_p * 600, 600, dtype=np.int64)
    ent_s = ts_s + 1
    ext_s = ent_s + 1800
    ret_s = np.where(np.arange(n_p) % 2 == 0, 0.001, -0.001)
    sp_ok = cert_mod.split_train_val(np.arange(n_p, dtype=np.int64), ts_s, ent_s, ext_s, ret_s)
    assert sp_ok["support_ok"] is True
    # Zero overlap across (fit_all, val), (val_h1, val_h2), (tr_sub, es_sub)
    assert not cert_mod.intervals_overlap_any(ent_s[sp_ok["fit_all_idx"]], ext_s[sp_ok["fit_all_idx"]], ent_s[sp_ok["val_idx"]], ext_s[sp_ok["val_idx"]]).any()
    assert not cert_mod.intervals_overlap_any(ent_s[sp_ok["val_h1_idx"]], ext_s[sp_ok["val_h1_idx"]], ent_s[sp_ok["val_h2_idx"]], ext_s[sp_ok["val_h2_idx"]]).any()
    assert not cert_mod.intervals_overlap_any(ent_s[sp_ok["tr_sub_idx"]], ext_s[sp_ok["tr_sub_idx"]], ent_s[sp_ok["es_sub_idx"]], ext_s[sp_ok["es_sub_idx"]]).any()
    # Too-small pool -> exact ordered failure reason
    assert cert_mod.split_train_val(np.arange(100, dtype=np.int64), ts_s, ent_s, ext_s, ret_s)["failure_reason"] == "pool_below_min_val"
    # Donor-purge precedence: make all donor intervals of earlier half collide with val_h2 -> val_h1 purged -> val_support_failed!
    donor_ent = np.full(n_p, int(ent_s[-1]), dtype=np.int64)
    donor_ext = donor_ent + 1800
    sp_null_collide = cert_mod.split_train_val(
        np.arange(n_p, dtype=np.int64),
        ts_s,
        ent_s,
        ext_s,
        ret_s,
        donor_entry_ts=donor_ent,
        donor_exit_ts=donor_ext,
    )
    assert sp_null_collide["support_ok"] is False and sp_null_collide["failure_reason"] == "val_support_failed"

    # 3. Linear percentiles, G5 unavailable branch, strict JSON allow_nan=False, certificate._reread_result (exact=2)
    assert abs((cert_mod._linear_p10([0.50, 0.60, 0.70, 0.80, 0.90]) or 0.0) - 0.54) < 1e-12
    assert cert_mod._linear_p10([]) is None

    # 4. Full synthetic 262-role bundle + verify_bundle PASS + B3 D1 trade_wins swap refusal + missing builder_suffix + false KILL-STACK!
    # Build a compact synthetic DEV dataframe spanning all 6 shared CPCV groups with valid NY weekday rows so all 262 roles exist!
    group_bounds = cert_mod.SHARED_GROUP_BOUNDS_UTC
    rows_ts: list[int] = []
    # Pick valid NY weekday bars spaced 32m (1920s > 1805+2s) apart strictly inside each of the 6 shared CPCV groups!
    for g in range(6):
        dt_g = datetime.fromtimestamp(group_bounds[g] + 86400, tz=timezone.utc).replace(hour=13, minute=0, second=0)
        end_ep = group_bounds[g + 1] - 86400
        while int(dt_g.timestamp()) < end_ep:
            if dt_g.weekday() < 5:
                for m_step in range(0, 448, 32):
                    cand_ep = int((dt_g + timedelta(minutes=m_step)).timestamp())
                    if cand_ep < end_ep:
                        rows_ts.append(cand_ep)
            dt_g += timedelta(days=1)
    ts_all = np.array(sorted(set(rows_ts)), dtype=np.int64)
    n_syn = len(ts_all)
    # Create ticks covering entry (ts+1..ts+30) and expiry (ts+1801..ts+1830) for every synthetic bar, plus fresh prior ticks (ts-1)!
    tk_ts_list: list[int] = []
    tk_px_list: list[float] = []
    for i, t_i in enumerate(ts_all):
        # 70% win rate on positive-score rows, ~2% ties, ~50% overall moved_up_rate
        is_tie = (i % 50 == 0)
        is_up = (i % 2 == 0)
        p0 = 150.0
        p_end = 150.0 if is_tie else (150.2 if is_up else 149.8)
        # Strictly prior tick at t_i - 1 with price change so fresh_quote_le2s and fresh_midchg_le2s are nonempty!
        tk_ts_list.extend([int(t_i - 2), int(t_i - 1), int(t_i + 1), int(t_i + 5), int(t_i + 1801), int(t_i + 1805)])
        tk_px_list.extend([149.99, p0, p0, p0, p_end, p_end])
    order_tk = np.argsort(tk_ts_list, kind="mergesort")
    tk_ts = np.asarray(tk_ts_list, dtype=np.int64)[order_tk]
    tk_px = np.asarray(tk_px_list, dtype=np.float64)[order_tk]
    # Deduplicate timestamps keeping last
    _, uniq_inv = np.unique(tk_ts[::-1], return_index=True)
    keep_idx = (len(tk_ts) - 1) - uniq_inv
    keep_idx.sort()
    tk_ts = tk_ts[keep_idx]
    tk_px = tk_px[keep_idx]

    ret_t, val_t, ent_t, ext_t = cert_mod.settle_with_intervals(ts_all, tk_ts, tk_px, horizon_s=1800, delay_s=0, tol_s=60)
    fcols_syn = ["rv_60m"] + [f"f_{i}" for i in range(65)]
    rng_df = np.random.default_rng(1234)
    df_syn = pd.DataFrame(rng_df.normal(size=(n_syn, 66)), columns=fcols_syn)
    df_syn["rv_60m"] = np.linspace(0.001, 0.010, n_syn)
    df_syn["decision_ts"] = ts_all
    df_syn["bar_start_ts"] = ts_all - 60

    elig_syn = cert_mod.build_eligible_indices(df_syn, fcols_syn, val_t, ret_t)
    # Build synthetic scores favoring even-indexed winning bars on real/fwd/frozen_past/final_fit and noisy on nulls!
    # Use a compact split helper with small min_val so synthetic 4,200 rows support all real/fwd/frozen/final roles cleanly!
    orig_split = cert_mod.split_train_val
    try:
        def _fast_split(pool_idx, ts, entry_ts, exit_ts, ret_fwd, **kw):
            n_pool = len(pool_idx)
            if n_pool < 620:
                return {"support_ok": False, "failure_reason": "pool_below_min_val", "fit_all_idx": np.empty(0, dtype=np.int64), "val_h1_idx": np.empty(0, dtype=np.int64), "val_h2_idx": np.empty(0, dtype=np.int64), "val_idx": np.empty(0, dtype=np.int64), "tr_sub_idx": np.empty(0, dtype=np.int64), "es_sub_idx": np.empty(0, dtype=np.int64)}
            n_val = max(560, int(round(n_pool * 0.25)))
            val_raw = pool_idx[-n_val:]
            mid = len(val_raw) // 2
            val_h2 = val_raw[mid:]
            val_h1 = val_raw[:mid][(ts[val_raw[:mid]] < int(ts[val_h2[0]]) - 1800) & (exit_ts[val_raw[:mid]] < int(entry_ts[val_h2[0]]))]
            val_idx = np.concatenate([val_h1, val_h2])
            fit_all = pool_idx[(ts[pool_idx] < int(ts[val_idx[0]]) - 1800) & (exit_ts[pool_idx] < int(entry_ts[val_idx[0]]))]
            mv_fit = fit_all[ret_fwd[fit_all] != 0.0]
            if len(val_h1) < 1 or len(val_h2) < 1 or len(val_idx) < 500 or len(fit_all) < 20 or len(mv_fit) < 20:
                return {"support_ok": False, "failure_reason": "val_support_failed", "fit_all_idx": np.empty(0, dtype=np.int64), "val_h1_idx": np.empty(0, dtype=np.int64), "val_h2_idx": np.empty(0, dtype=np.int64), "val_idx": np.empty(0, dtype=np.int64), "tr_sub_idx": np.empty(0, dtype=np.int64), "es_sub_idx": np.empty(0, dtype=np.int64)}
            n_es = max(10, len(mv_fit) // 5)
            es_sub = mv_fit[-n_es:]
            tr_sub = mv_fit[(ts[mv_fit] < int(ts[es_sub[0]]) - 1800) & (exit_ts[mv_fit] < int(entry_ts[es_sub[0]]))]
            return {"support_ok": True, "failure_reason": None, "fit_all_idx": fit_all, "val_h1_idx": val_h1, "val_h2_idx": val_h2, "val_idx": val_idx, "tr_sub_idx": tr_sub, "es_sub_idx": es_sub}

        cert_mod.split_train_val = _fast_split  # type: ignore[assignment]
        role_specs = cert_mod.build_all_role_specs(ts_all, ret_t, val_t, ent_t, ext_t, elig_syn)
        scores_dict: dict[str, np.ndarray] = {}
        cal_by_role: dict[str, dict[str, Any]] = {}
        rng_sc = np.random.default_rng(77)
        for rs in role_specs:
            rname = rs["role"]
            if not rs["support_ok"]:
                scores_dict[f"{rname}_val"] = np.empty(0, dtype=np.float64)
                if rname != "final_fit":
                    scores_dict[f"{rname}_test"] = np.empty(0, dtype=np.float64)
                cal_by_role[rname] = {"support_ok": False, "cal_admissible": False, "failure_reason": rs["failure_reason"], "n_val_oos": 0, "conf_thr": None, "platt_coef": None, "platt_intercept": None}
                continue
            v_idx = rs["val_idx"]
            t_idx = rs["test_idx"]
            is_null = rname.startswith("null_")
            if is_null:
                v_p = rng_sc.uniform(0.35, 0.65, size=len(v_idx))
                t_p = rng_sc.uniform(0.35, 0.65, size=len(t_idx))
            else:
                # High score on winning rows + a subset of losing rows well above conf_thr -> strong PASS with both wins & losses!
                v_win = rs["ret_arr"][v_idx] > 0.0
                v_p = np.where(v_win, 0.72, 0.35) + rng_sc.uniform(-0.02, 0.02, size=len(v_idx))
                if len(t_idx) > 0:
                    t_win = rs["ret_arr"][t_idx] > 0.0
                    t_p = np.where(t_win, 0.85, 0.30)
                    lose_pos = np.where(~t_win)[0]
                    if len(lose_pos) >= 10:
                        t_p[lose_pos[::10]] = 0.90
                    t_p = np.clip(t_p + rng_sc.uniform(-0.005, 0.005, size=len(t_idx)), 0.05, 0.95)
                else:
                    t_p = np.empty(0, dtype=np.float64)
            scores_dict[f"{rname}_val"] = v_p.astype(np.float64)
            if rname != "final_fit":
                scores_dict[f"{rname}_test"] = t_p.astype(np.float64)
            conf_thr = float(np.quantile(np.abs(v_p - 0.5), 1.0 - float(rs["cov"]), method="linear"))
            cal_by_role[rname] = {
                "support_ok": True,
                "cal_admissible": True,
                "failure_reason": None,
                "n_val_oos": int(len(v_idx)),
                "conf_thr": conf_thr,
                "platt_coef": 2.0,
                "platt_intercept": 0.2,
            }

        prereg_payload = {"schema": "test", "version": 1}
        prereg_sha = cert_mod.sha256_bytes(cert_mod.canonical_json_bytes(prereg_payload))
        peer_tie_info = {"max_ties": 1, "max_total": 20, "max_tie_rate": 0.05, "pairs": {}}
        exp_id = {"content_id": "0123456789abcdef", "artifacts": []}
        dossier = cert_mod.evaluate_all_gates_from_scores(
            df_dev=df_syn,
            fcols=fcols_syn,
            tick_ts=tk_ts,
            tick_px=tk_px,
            ret_fwd_true=ret_t,
            valid_true=val_t,
            entry_true=ent_t,
            exit_true=ext_t,
            eligible_idx=elig_syn,
            role_specs=role_specs,
            scores_dict=scores_dict,
            cal_by_role=cal_by_role,
            peer_tie_info=peer_tie_info,
            preregistration_payload=prereg_payload,
            preregistration_sha256=prereg_sha,
            local_source_closure_sha256="b" * 64,
            final_fit_expected_identity=exp_id,
        )
        # Verify strict allow_nan=False JSON + certificate._reread_result == (2, [], [], ...)
        dos_bytes = cert_mod.pretty_json_bytes(dossier)
        reread = cert_check_mod._reread_result(cert_mod.CANONICAL_DOSSIER_NAME, json.loads(dos_bytes.decode("utf-8")))
        assert reread[0] == 2 and reread[1] == [] and reread[2] == [], f"certificate._reread_result mismatch: {reread}"
        assert dossier["G4_sign_null"]["verdict"].startswith("DESCRIPTIVE_RANK_DIAGNOSTIC_")
        assert "SIGNIFICANT" not in dossier["G4_sign_null"]["verdict"]
        # Verify G3 empty subgroup & G5 unavailable branch on zero-counted-path KILL-STACK (calibration refusal: _val full, _test empty!)
        zero_scores = {
            k: (scores_dict[k] if k.endswith("_val") else np.empty(0, dtype=np.float64))
            for k in scores_dict
        }
        zero_cal = {
            r["role"]: {
                "support_ok": bool(r["support_ok"]),
                "cal_admissible": False,
                "failure_reason": "up_calibration_inadmissible" if r["support_ok"] else r["failure_reason"],
                "n_val_oos": len(r["val_idx"]),
                "conf_thr": None,
                "platt_coef": None,
                "platt_intercept": None,
            }
            for r in role_specs
        }
        dos_no_prior = cert_mod.evaluate_all_gates_from_scores(
            df_dev=df_syn,
            fcols=fcols_syn,
            tick_ts=tk_ts[tk_ts >= ts_all[0] + 1],
            tick_px=tk_px[tk_ts >= ts_all[0] + 1],
            ret_fwd_true=ret_t,
            valid_true=val_t,
            entry_true=ent_t,
            exit_true=ext_t,
            eligible_idx=elig_syn,
            role_specs=role_specs,
            scores_dict=zero_scores,
            cal_by_role=zero_cal,
            peer_tie_info=peer_tie_info,
            preregistration_payload=prereg_payload,
            preregistration_sha256=prereg_sha,
            local_source_closure_sha256="b" * 64,
        )
        assert dos_no_prior["verdict"] == "KILL-STACK"
        assert dos_no_prior["G5_payout"]["passed"] is False and dos_no_prior["G5_payout"]["p10_ev"] is None
        assert dos_no_prior["G3_entry_delay"]["freshness_subgroups"]["fresh_quote_le2s"]["excess"] is None
        rr_zero = cert_check_mod._reread_result(cert_mod.CANONICAL_DOSSIER_NAME, json.loads(cert_mod.pretty_json_bytes(dos_no_prior)))
        assert rr_zero[0] == 2 and rr_zero[1] == [] and rr_zero[2] == []

        # Write bundle fixture to temp dir and test verify_bundle:
        # (a) Unchanged fixture PASSES!
        # (b) Missing builder_suffix.json under --allow-builder-suffix FAILS!
        # (c) Digest-consistent false KILL-STACK FAILS!
        # (d) D1/B3 digest-consistent G1_cpcv.paths.trade_wins win/loss swap FAILS specifically at trade_wins comparison!
        with tempfile.TemporaryDirectory() as td_vb:
            bdir = Path(td_vb)
            ordered_keys = cert_mod.canonical_scores_npz_keys()
            npz_bytes = cert_mod.write_deterministic_uncompressed_npz(bdir / cert_mod.CANONICAL_SCORES_NAME, [(k, scores_dict[k]) for k in ordered_keys])
            ledger_obj = {
                "schema": cert_mod.FIT_LEDGER_SCHEMA,
                "preregistration_sha256": prereg_sha,
                "max_fit_units": 263,
                "attempts": [{"attempt_id": 1, "stage": "certifier", "role": "real_0_1", "preregistration_sha256": prereg_sha, "started_utc": "2026-11-25T00:00:00Z"}],
            }
            led_bytes = cert_mod.canonical_json_bytes(ledger_obj)
            (bdir / cert_mod.CANONICAL_LEDGER_NAME).write_bytes(led_bytes)
            (bdir / cert_mod.CANONICAL_DOSSIER_NAME).write_bytes(dos_bytes)

            def _write_receipt(d_bytes: bytes, verdict_v: str, failed_p: list[str]) -> str:
                r_obj = {
                    "schema": cert_mod.RECEIPT_SCHEMA,
                    "book_id": cert_mod.BOOK_ID,
                    "dossier_sha256": cert_mod.sha256_bytes(d_bytes),
                    "scores_npz_sha256": cert_mod.sha256_bytes(npz_bytes),
                    "scores_npz_bytes": len(npz_bytes),
                    "certifier_attempt_count": 1,
                    "certifier_fit_ledger_prefix_sha256": cert_mod.sha256_bytes(led_bytes),
                    "preregistration_sha256": prereg_sha,
                    "local_source_closure_sha256": "b" * 64,
                    "verdict": verdict_v,
                    "failed_predicates": failed_p,
                    "final_fit_expected_content_id": exp_id["content_id"] if verdict_v == "PASS" else None,
                    "final_fit_expected_artifacts": exp_id["artifacts"] if verdict_v == "PASS" else None,
                }
                rb = cert_mod.pretty_json_bytes(r_obj)
                (bdir / cert_mod.CANONICAL_RECEIPT_NAME).write_bytes(rb)
                return cert_mod.sha256_bytes(rb)

            injected = {
                "df_dev": df_syn,
                "fcols": fcols_syn,
                "tick_ts": tk_ts,
                "tick_px": tk_px,
                "ret_fwd_true": ret_t,
                "valid_true": val_t,
                "entry_true": ent_t,
                "exit_true": ext_t,
                "eligible_idx": elig_syn,
                "role_specs": role_specs,
                "peer_tie_info": peer_tie_info,
            }
            rcpt_sha_ok = _write_receipt(dos_bytes, dossier["verdict"], dossier["failed_predicates"])
            res_ok = cert_mod.verify_bundle(
                expected_receipt_sha256=rcpt_sha_ok,
                bundle_dir=bdir,
                expected_verdict=dossier["verdict"],
                skip_closure_check=True,
                _injected_data=injected,
            )
            assert res_ok["verified"] is True

            # (b) Missing progress_md / builder_suffix.json / non-builder_final_fit role under allow_builder_suffix=True MUST FAIL!
            _expect_raises(
                cert_mod.OperationalHoldError,
                cert_mod.verify_bundle,
                expected_receipt_sha256=rcpt_sha_ok,
                bundle_dir=bdir,
                allow_builder_suffix=True,
                progress_md=None,
                skip_closure_check=True,
                _injected_data=injected,
            )
            prog_vb = bdir / "PROGRESS.md"
            prog_vb.write_text("# empty\n", encoding="utf-8")
            _expect_raises(
                cert_mod.OperationalHoldError,
                cert_mod.verify_bundle,
                expected_receipt_sha256=rcpt_sha_ok,
                bundle_dir=bdir,
                allow_builder_suffix=True,
                progress_md=prog_vb,
                skip_closure_check=True,
                _injected_data=injected,
            )
            # Non-builder_final_fit role in ledger MUST be refused!
            bad_role_ldg = {
                "schema": cert_mod.FIT_LEDGER_SCHEMA,
                "preregistration_sha256": prereg_sha,
                "max_fit_units": 263,
                "attempts": ledger_obj["attempts"] + [
                    {"attempt_id": 2, "stage": "builder", "role": "rogue_role", "preregistration_sha256": prereg_sha, "started_utc": "2026-11-25T00:00:00Z"}
                ],
            }
            _expect_raises(
                cert_mod.OperationalHoldError,
                cert_mod.validate_fit_ledger_obj,
                bad_role_ldg,
                expected_prereg_sha=prereg_sha,
                allow_builder_attempt=True,
            )

            # (c) Digest-consistent false KILL-STACK (mutated verdict with recomputed dossier & receipt SHA-256) MUST FAIL!
            if dossier["verdict"] == "PASS":
                dos_fk = copy.deepcopy(dossier)
                dos_fk["verdict"] = "KILL-STACK"
                dos_fk["failed_predicates"] = ["fake_kill"]
                dos_fk_b = cert_mod.pretty_json_bytes(dos_fk)
                (bdir / cert_mod.CANONICAL_DOSSIER_NAME).write_bytes(dos_fk_b)
                rcpt_fk = _write_receipt(dos_fk_b, "KILL-STACK", ["fake_kill"])
                _expect_raises(
                    cert_mod.OperationalHoldError,
                    cert_mod.verify_bundle,
                    expected_receipt_sha256=rcpt_fk,
                    bundle_dir=bdir,
                    expected_verdict="KILL-STACK",
                    skip_closure_check=True,
                    _injected_data=injected,
                )

            # (d) D1 / B3 Negative Fixture: swap an early win & early loss in G1_cpcv.paths.trade_wins while preserving
            #     trade_ts, n, wins, scores.npz, trade_ts_sha256, role_indices_sha256; recompute dossier & receipt SHA-256!
            dos_swap = copy.deepcopy(dossier)
            swapped = False
            for p_ent in dos_swap["G1_cpcv"]["paths"]:
                tw = p_ent["trade_wins"]
                if 1 in tw and 0 in tw:
                    i1 = tw.index(1)
                    i0 = tw.index(0)
                    tw[i1], tw[i0] = tw[i0], tw[i1]
                    swapped = True
                    break
            assert swapped, "synthetic fixture must have at least one path with both a win and a loss"
            dos_swap_b = cert_mod.pretty_json_bytes(dos_swap)
            (bdir / cert_mod.CANONICAL_DOSSIER_NAME).write_bytes(dos_swap_b)
            rcpt_swap = _write_receipt(dos_swap_b, dossier["verdict"], dossier["failed_predicates"])
            try:
                cert_mod.verify_bundle(
                    expected_receipt_sha256=rcpt_swap,
                    bundle_dir=bdir,
                    expected_verdict=dossier["verdict"],
                    skip_closure_check=True,
                    _injected_data=injected,
                )
                raise AssertionError("D1 trade_wins swap was NOT rejected by verify_bundle!")
            except cert_mod.OperationalHoldError as exc:
                assert "trade_wins mismatch" in str(exc), f"expected refusal specifically at trade_wins comparison, got: {exc}"

            # (e) stack05_m30_up_v2 raw_top5 G4 mode: null roles with cal_admissible=False & full _test scores remain readable & verify_bundle PASSES!
            cal_v2 = copy.deepcopy(cal_by_role)
            scores_v2 = dict(scores_dict)
            rng_v2 = np.random.default_rng(2026)
            for rs in role_specs:
                if rs["kind"] == "null" and rs["support_ok"]:
                    r_n = rs["role"]
                    cal_v2[r_n]["cal_admissible"] = False
                    cal_v2[r_n]["failure_reason"] = "up_calibration_inadmissible"
                    cal_v2[r_n]["platt_coef"] = -0.1
                    # Keep conf_thr consistent with val_scores while ensuring >= 25 uncorrelated UP tail trades on compact synthetic test fold
                    v_n = scores_v2[f"{r_n}_val"]
                    thr_n = float(np.quantile(np.abs(v_n - 0.5), 1.0 - float(rs["cov"]), method="linear"))
                    cal_v2[r_n]["conf_thr"] = thr_n
                    t_n = rng_v2.uniform(0.50 + thr_n + 0.01, 0.70, size=len(rs["test_idx"]))
                    scores_v2[f"{r_n}_test"] = t_n.astype(np.float64)
            prereg_v2 = {"schema": "test", "version": 2, "arm_name": "stack05_m30_up_v2", "g4_selection_mode": "raw_top5_apples_to_apples"}
            prereg_v2_sha = cert_mod.sha256_bytes(cert_mod.canonical_json_bytes(prereg_v2))
            dos_v2 = cert_mod.evaluate_all_gates_from_scores(
                df_dev=df_syn,
                fcols=fcols_syn,
                tick_ts=tk_ts,
                tick_px=tk_px,
                ret_fwd_true=ret_t,
                valid_true=val_t,
                entry_true=ent_t,
                exit_true=ext_t,
                eligible_idx=elig_syn,
                role_specs=role_specs,
                scores_dict=scores_v2,
                cal_by_role=cal_v2,
                peer_tie_info=peer_tie_info,
                preregistration_payload=prereg_v2,
                preregistration_sha256=prereg_v2_sha,
                local_source_closure_sha256="b" * 64,
                final_fit_expected_identity=exp_id,
                arm_name="stack05_m30_up_v2",
                g4_mode="raw_top5",
            )
            assert dos_v2["arm_name"] == "stack05_m30_up_v2"
            assert dos_v2["G4_sign_null"]["g4_selection_mode"] == "raw_top5_apples_to_apples"
            assert dos_v2["G4_sign_null"]["real_raw_top5_paths_counted"] == 15
            assert dos_v2["G4_sign_null"]["readable_null_shifts"] == 8  # all 8 support-ok synthetic shifts readable despite cal_admissible=False!
            assert dos_v2["G4_sign_null"]["n_null_ge_real_raw"] == 0 and dos_v2["G4_sign_null"]["n_null_ge_real_platt"] == 0
            with tempfile.TemporaryDirectory() as td_v2:
                b2 = Path(td_v2)
                nm2 = cert_mod.artifact_names_for_arm("stack05_m30_up_v2")
                npz2 = cert_mod.write_deterministic_uncompressed_npz(b2 / nm2["scores"], [(k, scores_v2[k]) for k in ordered_keys])
                ldg2 = cert_mod.canonical_json_bytes({
                    "schema": cert_mod.FIT_LEDGER_SCHEMA,
                    "preregistration_sha256": prereg_v2_sha,
                    "max_fit_units": 263,
                    "attempts": [{"attempt_id": 1, "stage": "certifier", "role": "real_0_1", "preregistration_sha256": prereg_v2_sha, "started_utc": "2026-11-26T00:00:00Z"}],
                })
                (b2 / nm2["ledger"]).write_bytes(ldg2)
                dos2_b = cert_mod.pretty_json_bytes(dos_v2)
                (b2 / nm2["dossier"]).write_bytes(dos2_b)
                rcpt2_b = cert_mod.pretty_json_bytes({
                    "schema": cert_mod.RECEIPT_SCHEMA,
                    "book_id": cert_mod.BOOK_ID,
                    "arm_name": "stack05_m30_up_v2",
                    "dossier_sha256": cert_mod.sha256_bytes(dos2_b),
                    "scores_npz_sha256": cert_mod.sha256_bytes(npz2),
                    "scores_npz_bytes": len(npz2),
                    "certifier_attempt_count": 1,
                    "certifier_fit_ledger_prefix_sha256": cert_mod.sha256_bytes(ldg2),
                    "preregistration_sha256": prereg_v2_sha,
                    "local_source_closure_sha256": "b" * 64,
                    "verdict": dos_v2["verdict"],
                    "failed_predicates": dos_v2["failed_predicates"],
                    "final_fit_expected_content_id": exp_id["content_id"] if dos_v2["verdict"] == "PASS" else None,
                    "final_fit_expected_artifacts": exp_id["artifacts"] if dos_v2["verdict"] == "PASS" else None,
                })
                (b2 / nm2["receipt"]).write_bytes(rcpt2_b)
                res_v2 = cert_mod.verify_bundle(
                    expected_receipt_sha256=cert_mod.sha256_bytes(rcpt2_b),
                    bundle_dir=b2,
                    expected_verdict=dos_v2["verdict"],
                    skip_closure_check=True,
                    arm_name="stack05_m30_up_v2",
                    _injected_data=injected,
                )
                assert res_v2["verified"] is True

                # Cross-arm identity & protocol binding refusals (review MAJOR #2 & #3 + delta review edge cases)
                for bad_arm in ("stack05", "..", "../escape", "a.b"):
                    try:
                        cert_mod.artifact_names_for_arm(bad_arm)
                        raise AssertionError(f"expected invalid arm_name refusal for {bad_arm!r}")
                    except cert_mod.OperationalHoldError:
                        pass
                try:
                    cert_mod.validate_arm_g4_mode("stack05_m30_up_v1", "raw_top5")
                    raise AssertionError("expected refusal for v1 + raw_top5")
                except cert_mod.OperationalHoldError:
                    pass
                assert cert_mod.default_out_dir_for_arm("stack05_m30_up_v2").name == "stack05_m30_up_v2"
                assert cert_mod.default_archive_dir_for_arm("stack05_m30_up_v2").name == "stack05_m30_up_v2"
                # Verify --verify-bundle CLI dispatch when --bundle-dir is omitted (with & without --allow-builder-suffix)
                cli_calls: list[dict[str, Any]] = []
                orig_vb = cert_mod.verify_bundle
                try:
                    def _spy_vb(**kw: Any) -> dict[str, Any]:
                        cli_calls.append(kw)
                        return {"verified": True}
                    cert_mod.verify_bundle = _spy_vb  # type: ignore[assignment]
                    rc_chk = cert_mod.main(["--verify-bundle", "--expected-receipt-sha256", "a" * 64, "--arm-name", "stack05_m30_up_v2"])
                    rc_arc = cert_mod.main(["--verify-bundle", "--expected-receipt-sha256", "a" * 64, "--arm-name", "stack05_m30_up_v2", "--allow-builder-suffix"])
                    assert rc_chk == 0 and rc_arc == 0
                    assert cli_calls[0]["bundle_dir"] is None and cli_calls[0]["allow_builder_suffix"] is False
                    assert cli_calls[1]["bundle_dir"] == cert_mod.default_archive_dir_for_arm("stack05_m30_up_v2") and cli_calls[1]["allow_builder_suffix"] is True
                finally:
                    cert_mod.verify_bundle = orig_vb  # type: ignore[assignment]
                try:
                    cert_mod.publish_certifier_prefix_bundle(
                        src_dir=b2,
                        checkout_root=b2,
                        archive_dir=b2 / "stack05_m30_up_v1",
                        arm_name="stack05_m30_up_v2",
                    )
                    raise AssertionError("expected refusal on cross-arm archive target")
                except cert_mod.OperationalHoldError as exc:
                    assert "cross-arm" in str(exc)
                # Custom archive dir containing foreign v1 scores file must also be refused before writing!
                custom_arc = b2 / "custom_arc"
                custom_arc.mkdir()
                (custom_arc / cert_mod.CANONICAL_SCORES_NAME).write_bytes(b"v1-scores")
                try:
                    cert_mod.publish_certifier_prefix_bundle(
                        src_dir=b2,
                        checkout_root=b2,
                        archive_dir=custom_arc,
                        arm_name="stack05_m30_up_v2",
                    )
                    raise AssertionError("expected refusal on foreign scores file in custom archive")
                except cert_mod.OperationalHoldError as exc:
                    assert "unexpected file" in str(exc)
                # Builder must also reject a hash-consistent v2 PASS bundle carrying platt_ev instead of raw_top5!
                from build_certified_sota_books import _authenticate_stack05_pass_inputs
                b_bad_proto = b2 / "bad_v2_proto"
                b_bad_proto.mkdir()
                dos_bad = copy.deepcopy(dos_v2)
                dos_bad["verdict"] = "PASS"
                dos_bad["all_5_gates_passed"] = True
                dos_bad["drift_check_passed"] = True
                for g_k in ("G1_cpcv", "G2_placebo", "G3_entry_delay", "G4_sign_null", "G5_payout"):
                    dos_bad[g_k]["passed"] = True
                dos_bad["G4_sign_null"]["verdict"] = "DESCRIPTIVE_RANK_DIAGNOSTIC_PASS"
                dos_bad["preregistration_payload"] = {"schema": "test", "arm_name": "stack05_m30_up_v2"}  # missing raw_top5!
                dos_bad_b = cert_mod.pretty_json_bytes(dos_bad)
                rcpt_bad_b = cert_mod.pretty_json_bytes({
                    "schema": cert_mod.RECEIPT_SCHEMA,
                    "book_id": cert_mod.BOOK_ID,
                    "arm_name": "stack05_m30_up_v2",
                    "dossier_sha256": cert_mod.sha256_bytes(dos_bad_b),
                    "certifier_fit_ledger_prefix_sha256": cert_mod.sha256_bytes(ldg2),
                    "preregistration_sha256": prereg_v2_sha,
                    "certifier_attempt_count": 1,
                    "verdict": "PASS",
                })
                (b_bad_proto / nm2["dossier"]).write_bytes(dos_bad_b)
                (b_bad_proto / nm2["receipt"]).write_bytes(rcpt_bad_b)
                (b_bad_proto / nm2["ledger"]).write_bytes(ldg2)
                try:
                    _authenticate_stack05_pass_inputs(
                        expected_receipt_sha256=cert_mod.sha256_bytes(rcpt_bad_b),
                        repo_root=b2,
                        bundle_dir=b_bad_proto,
                        arm_name="stack05_m30_up_v2",
                    )
                    raise AssertionError("expected builder refusal for v2 PASS bundle with platt_ev")
                except SotaBookError as exc:
                    assert "requires g4_mode='raw_top5'" in str(exc)
                # Copy v2 bundle under v1 filenames to verify verify_bundle rejects arm_name mismatch
                b_cross = b2 / "cross_arm"
                b_cross.mkdir()
                nm1 = cert_mod.artifact_names_for_arm("stack05_m30_up_v1")
                for k_n in ("scores", "ledger", "dossier", "receipt"):
                    (b_cross / nm1[k_n]).write_bytes((b2 / nm2[k_n]).read_bytes())
                try:
                    cert_mod.verify_bundle(
                        expected_receipt_sha256=cert_mod.sha256_bytes(rcpt2_b),
                        bundle_dir=b_cross,
                        skip_closure_check=True,
                        arm_name="stack05_m30_up_v1",
                        _injected_data=injected,
                    )
                    raise AssertionError("expected arm_name mismatch refusal in verify_bundle")
                except cert_mod.OperationalHoldError as exc:
                    assert "arm_name mismatch" in str(exc)

            # (f) G4 raw_top5 PASS case, BLOCKER real-path Platt-refusal symmetry test, & independent comparator refusals!
            # Construct full 16-shift support-ok synthetic specs so all 16 shifts are readable (>=12).
            all_sup_specs = copy.deepcopy(role_specs)
            real0_spec = next(s for s in all_sup_specs if s["role"] == "real_0_1")
            for sp in all_sup_specs:
                if sp["kind"] == "null" and not sp["support_ok"]:
                    sp["support_ok"] = True
                    sp["failure_reason"] = None
                    sp["val_idx"] = real0_spec["val_idx"].copy()
                    sp["test_idx"] = real0_spec["test_idx"].copy()
                    sp["role_indices_sha256"] = real0_spec["role_indices_sha256"]

            ts_syn = df_syn["decision_ts"].to_numpy(dtype=np.int64)
            def _make_scores_for_wr(spec_obj: dict[str, Any], target_wr: float) -> tuple[np.ndarray, np.ndarray]:
                n_v = len(spec_obj["val_idx"])
                v_s = np.linspace(0.40, 0.70, n_v, dtype=np.float64)
                t_idx = spec_obj["test_idx"]
                r_t = spec_obj["ret_arr"][t_idx]
                pos_l = np.flatnonzero(r_t > 0.0)
                neg_l = np.flatnonzero(r_t <= 0.0)
                t_s = np.full(len(t_idx), 0.45, dtype=np.float64)
                # Place 36 well-spaced (>=1800s) candidate bars at score 0.90 matching target_wr
                used_ts: list[int] = []
                cand_w: list[int] = []
                n_tot = 36
                n_w = int(round(n_tot * target_wr))
                for idx_c in pos_l:
                    t_c = int(ts_syn[t_idx[idx_c]])
                    if all(abs(t_c - u) >= 1800 for u in used_ts):
                        used_ts.append(t_c)
                        cand_w.append(int(idx_c))
                        if len(cand_w) >= n_w:
                            break
                cand_l: list[int] = []
                for idx_c in neg_l:
                    t_c = int(ts_syn[t_idx[idx_c]])
                    if all(abs(t_c - u) >= 1800 for u in used_ts):
                        used_ts.append(t_c)
                        cand_l.append(int(idx_c))
                        if len(cand_l) >= (n_tot - len(cand_w)):
                            break
                for i_c in cand_w + cand_l:
                    t_s[i_c] = 0.90
                return v_s, t_s

            def _run_g4_matrix(real_wrs: list[float], real_cal_ok: list[bool], null_wrs_by_w: dict[int, float], *, platt_high_subset: bool = False) -> dict[str, Any]:
                sc_m = copy.deepcopy(scores_v2)
                cal_m = copy.deepcopy(cal_v2)
                real_only = [s for s in all_sup_specs if s["kind"] == "real"]
                for idx_rp, sp in enumerate(real_only):
                    rname = sp["role"]
                    v_s, t_s = _make_scores_for_wr(sp, real_wrs[idx_rp])
                    if platt_high_subset:
                        # Give winning bars score 0.95 and losing bars score 0.72 so Platt EV>0 filters out losing bars -> high Platt WR, low raw WR!
                        t_idx = sp["test_idx"]
                        r_t = sp["ret_arr"][t_idx]
                        high = t_s > 0.80
                        t_s = np.where(high & (r_t > 0.0), 0.95, np.where(high, 0.72, 0.45))
                    sc_m[f"{rname}_val"] = v_s
                    sc_m[f"{rname}_test"] = t_s
                    thr_v = float(np.percentile(np.abs(v_s - 0.5), 95.0, method="linear"))
                    if real_cal_ok[idx_rp]:
                        cal_m[rname] = {
                            "cal_admissible": True,
                            "failure_reason": None,
                            "n_val_oos": len(v_s),
                            "conf_thr": thr_v,
                            "platt_coef": 2.0 if platt_high_subset else 1.5,
                            "platt_intercept": -3.5 if platt_high_subset else 1.2,
                        }
                    else:
                        cal_m[rname] = {
                            "cal_admissible": False,
                            "failure_reason": "nonpositive_platt_slope",
                            "n_val_oos": len(v_s),
                            "conf_thr": thr_v,
                            "platt_coef": -0.2,
                            "platt_intercept": 0.1,
                        }
                for sp in all_sup_specs:
                    if sp["kind"] == "null":
                        rname = sp["role"]
                        w_sh = int(sp["week_shift"])
                        v_s, t_s = _make_scores_for_wr(sp, null_wrs_by_w[w_sh])
                        sc_m[f"{rname}_val"] = v_s
                        sc_m[f"{rname}_test"] = t_s
                        cal_m[rname] = {
                            "cal_admissible": False,
                            "failure_reason": "nonpositive_platt_slope",
                            "n_val_oos": len(v_s),
                            "conf_thr": float(np.percentile(np.abs(v_s - 0.5), 95.0, method="linear")),
                            "platt_coef": -0.1,
                            "platt_intercept": 0.0,
                        }
                return cert_mod.evaluate_all_gates_from_scores(
                    df_dev=df_syn,
                    fcols=fcols_syn,
                    tick_ts=tk_ts,
                    tick_px=tk_px,
                    ret_fwd_true=ret_t,
                    valid_true=val_t,
                    entry_true=ent_t,
                    exit_true=ext_t,
                    eligible_idx=elig_syn,
                    role_specs=all_sup_specs,
                    scores_dict=sc_m,
                    cal_by_role=cal_m,
                    peer_tie_info=peer_tie_info,
                    preregistration_payload=prereg_v2,
                    preregistration_sha256=prereg_v2_sha,
                    local_source_closure_sha256="b" * 64,
                    final_fit_expected_identity=exp_id,
                    arm_name="stack05_m30_up_v2",
                    g4_mode="raw_top5",
                )

            # (f1) Passing 16/16 readable raw_top5 G4 case
            dos_pass_g4 = _run_g4_matrix([0.70] * 15, [True] * 15, {w: 0.48 for w in cert_mod.NULL_SHIFTS_WEEKS})
            assert dos_pass_g4["G4_sign_null"]["passed"] is True
            assert dos_pass_g4["G4_sign_null"]["verdict"] == "DESCRIPTIVE_RANK_DIAGNOSTIC_PASS"
            assert dos_pass_g4["G4_sign_null"]["readable_null_shifts"] == 16
            assert dos_pass_g4["G4_sign_null"]["real_raw_top5_paths_counted"] == 15

            # (f2) BLOCKER regression: 13 Platt-admissible real paths @ 0.70 + 2 Platt-inadmissible raw real paths @ 0.35 vs 16 null shifts @ 0.50 -> MUST include the 2 raw real paths & FAIL G4!
            dos_sym_fail = _run_g4_matrix([0.70] * 13 + [0.35, 0.35], [True] * 13 + [False, False], {w: 0.50 for w in cert_mod.NULL_SHIFTS_WEEKS})
            assert dos_sym_fail["G1_cpcv"]["paths_counted"] == 13  # Platt eligibility for G1 unchanged
            assert dos_sym_fail["G4_sign_null"]["real_raw_top5_paths_counted"] == 15  # Raw comparator includes all 15!
            assert dos_sym_fail["G4_sign_null"]["n_null_ge_real_raw"] > 1
            assert dos_sym_fail["G4_sign_null"]["passed"] is False

            # (f3) Independent comparator refusal: n_null_ge_real_raw > 1 while n_null_ge_real_platt == 0 -> FAIL!
            dos_raw_only_fail = _run_g4_matrix([0.45] * 15, [True] * 15, {w: 0.52 for w in cert_mod.NULL_SHIFTS_WEEKS}, platt_high_subset=True)
            assert dos_raw_only_fail["G4_sign_null"]["n_null_ge_real_raw"] > 1
            assert dos_raw_only_fail["G4_sign_null"]["n_null_ge_real_platt"] == 0
            assert dos_raw_only_fail["G4_sign_null"]["passed"] is False

            # (f3b) Opposite independent comparator refusal: n_null_ge_real_platt > 1 while n_null_ge_real_raw == 0 -> FAIL!
            # Real raw top-5% has 70% WR, but Platt keeps only the losing top subset (score 0.95) -> Platt p10 < 0.50 < null 0.52 < raw 0.70!
            sc_rev = copy.deepcopy(scores_v2)
            cal_rev = copy.deepcopy(cal_v2)
            for sp in all_sup_specs:
                rname = sp["role"]
                if sp["kind"] == "real":
                    v_s, t_s = _make_scores_for_wr(sp, 0.70)
                    r_t = sp["ret_arr"][sp["test_idx"]]
                    high = t_s > 0.80
                    # Wins sit at 0.72 (above raw conf_thr ~0.185, rejected by Platt intercept -3.5), losses + a few wins sit at 0.95 (admitted by Platt -> ~0.30 Platt WR)
                    win_hi = np.flatnonzero(high & (r_t > 0.0))
                    loss_hi = np.flatnonzero(high & (r_t <= 0.0))
                    t_s2 = np.full(len(t_s), 0.45, dtype=np.float64)
                    t_s2[win_hi] = 0.72
                    t_s2[loss_hi] = 0.95
                    # Add enough wins at 0.95 only to stay < 0.40 Platt WR while keeping >=25 Platt trades (need >=25 losses or high subset)
                    # Easier: use target_wr=0.60 with 60 candidate bars if available, or place 26 losses at 0.95 and 34 wins at 0.72 -> raw 34/60=0.567 > 0.52, Platt 0/26=0.0 < 0.52!
                    pos_l = np.flatnonzero(r_t > 0.0)
                    neg_l = np.flatnonzero(r_t <= 0.0)
                    used_t: list[int] = []
                    w_i: list[int] = []
                    for ic in pos_l:
                        tc = int(ts_syn[sp["test_idx"][ic]])
                        if all(abs(tc - u) >= 1800 for u in used_t):
                            used_t.append(tc)
                            w_i.append(int(ic))
                            if len(w_i) >= 36:
                                break
                    l_i: list[int] = []
                    for ic in neg_l:
                        tc = int(ts_syn[sp["test_idx"][ic]])
                        if all(abs(tc - u) >= 1800 for u in used_t):
                            used_t.append(tc)
                            l_i.append(int(ic))
                            if len(l_i) >= 26:
                                break
                    t_s2 = np.full(len(t_s), 0.45, dtype=np.float64)
                    for ic in w_i:
                        t_s2[ic] = 0.72
                    for ic in l_i:
                        t_s2[ic] = 0.95
                    sc_rev[f"{rname}_val"] = v_s
                    sc_rev[f"{rname}_test"] = t_s2
                    cal_rev[rname] = {
                        "cal_admissible": True,
                        "failure_reason": None,
                        "n_val_oos": len(v_s),
                        "conf_thr": float(np.percentile(np.abs(v_s - 0.5), 95.0, method="linear")),
                        "platt_coef": 2.0,
                        "platt_intercept": -3.5,
                    }
                elif sp["kind"] == "null":
                    v_s, t_s = _make_scores_for_wr(sp, 0.50)
                    sc_rev[f"{rname}_val"] = v_s
                    sc_rev[f"{rname}_test"] = t_s
                    cal_rev[rname] = {
                        "cal_admissible": False,
                        "failure_reason": "nonpositive_platt_slope",
                        "n_val_oos": len(v_s),
                        "conf_thr": float(np.percentile(np.abs(v_s - 0.5), 95.0, method="linear")),
                        "platt_coef": -0.1,
                        "platt_intercept": 0.0,
                    }
            dos_platt_only_fail = cert_mod.evaluate_all_gates_from_scores(
                df_dev=df_syn,
                fcols=fcols_syn,
                tick_ts=tk_ts,
                tick_px=tk_px,
                ret_fwd_true=ret_t,
                valid_true=val_t,
                entry_true=ent_t,
                exit_true=ext_t,
                eligible_idx=elig_syn,
                role_specs=all_sup_specs,
                scores_dict=sc_rev,
                cal_by_role=cal_rev,
                peer_tie_info=peer_tie_info,
                preregistration_payload=prereg_v2,
                preregistration_sha256=prereg_v2_sha,
                local_source_closure_sha256="b" * 64,
                final_fit_expected_identity=exp_id,
                arm_name="stack05_m30_up_v2",
                g4_mode="raw_top5",
            )
            assert dos_platt_only_fail["G4_sign_null"]["n_null_ge_real_raw"] == 0
            assert dos_platt_only_fail["G4_sign_null"]["n_null_ge_real_platt"] > 1
            assert dos_platt_only_fail["G4_sign_null"]["passed"] is False

            # (f4) Replay O_NOFOLLOW & dossier_path allowlist refusal (review MAJOR #4)
            import os
            import replay_sota_portfolio as rsp_mod
            with tempfile.TemporaryDirectory() as td_rp:
                r_root = Path(td_rp)
                (r_root / "books" / "USDJPY.sota_stack05_m30_up.v1").mkdir(parents=True)
                (r_root / "books" / "USDJPY.sota_stack05_m30_up.v1" / "USDJPY.sota_stack05_m30_up.v1.manifest.json").write_text(
                    json.dumps({"evidence_binding": {"dossier_sha256": "a" * 64, "dossier_path": "/etc/passwd"}}), encoding="utf-8"
                )
                try:
                    rsp_mod._load_and_validate_dossiers(r_root)
                    raise AssertionError("expected refusal on unapproved manifest dossier_path")
                except RuntimeError as exc:
                    assert "unapproved stack05 manifest dossier_path" in str(exc)
                (r_root / "results" / "json").mkdir(parents=True)
                sym_target = r_root / "outside.json"
                sym_target.write_text("{}", encoding="utf-8")
                os.symlink(sym_target, r_root / "results" / "json" / "deriv_5gate_USDJPY_sota_stack05_m30_up_v2_result.json")
                try:
                    rsp_mod._read_repo_file_nofollow(r_root, "results/json/deriv_5gate_USDJPY_sota_stack05_m30_up_v2_result.json")
                    raise AssertionError("expected O_NOFOLLOW refusal on symlink")
                except RuntimeError as exc:
                    assert "failed to safely read" in str(exc)
    finally:
        cert_mod.split_train_val = orig_split  # type: ignore[assignment]


def test_phase1_tick_warmup_and_standalone_consumer() -> None:
    from deriv_sota_tick_stream import (
        LiveTickBuffer,
        build_pair_decision_snapshot,
        read_tick_store_watermark,
    )
    from deriv_sota_executor import SotaDerivAsyncClient
    import certify_sota_stack05 as cert_mod

    # Load pinned DEV USDJPY parquet & real ticks for all 7 majors over a slice covering all 4 spec DEV epochs:
    # 1756123200, 1755698400, 1766757600, 1767362400
    target_epochs = [1755698400, 1756123200, 1766757600, 1767362400]
    dev_dir = Path("/mnt/data/deriv-sota-lab/cache/dataset_dev_v0.1")
    ticks_root = Path("/mnt/data/deriv/ticks")
    if not (dev_dir / "manifest.json").is_file():
        return  # Skip DEV archive check only if running in an environment without /mnt/data
    man = cert_mod.load_pinned_dev_manifest(dev_dir)
    df_usdjpy = cert_mod.load_pinned_dev_parquet("USDJPY", dev_dir)
    ts_col = df_usdjpy["decision_ts"].to_numpy(np.int64)
    repo = Path(__file__).resolve().parents[1]
    demo_books = load_enabled_sota_books(repo_root=repo, account_mode="demo")
    check_books = [
        b
        for b in demo_books
        if b.book_id in {"USDJPY.sota_ens5_m30_up.v1", "USDJPY.sota_ens5_m15_up.v1", "USDJPY.sota_stack05_m30_up.v1"}
    ]
    assert len(check_books) >= 2
    fcols = check_books[0].feature_cols

    # Build a helper that reads only the exact DEV daily tick files needed around each epoch (4-day window <= epoch)
    # via the O_NOFOLLOW loader so this test runs in ~3 seconds while exercising REAL pinned DEV ticks!
    def _load_window_ticks(epoch_s: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        lo_s = epoch_s - 5 * 86400
        d_lo = datetime.fromtimestamp(lo_s, tz=timezone.utc).strftime("%Y-%m-%d")
        d_hi = datetime.fromtimestamp(epoch_s, tz=timezone.utc).strftime("%Y-%m-%d")
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for p in MAJORS:
            ts_parts, px_parts = [], []
            for ent in man["inputs"][p]:
                day_str = ent["file"][len(p) + 1 : len(p) + 11]
                if d_lo <= day_str <= d_hi:
                    raw = cert_mod.read_verified_file_nofollow(
                        ticks_root, [p, ent["file"]], expected_bytes=int(ent["bytes"]), expected_sha256=str(ent["sha256"])
                    )
                    import io
                    tdf = pd.read_parquet(io.BytesIO(raw), columns=["datetime_utc", "price"])
                    t_ns = tdf["datetime_utc"].astype("int64").to_numpy()
                    t_s = (t_ns // 1_000_000_000).astype(np.int64)
                    px = tdf["price"].to_numpy(np.float64)
                    m = (t_s >= lo_s) & (t_s < epoch_s)
                    if m.any():
                        ts_parts.append(t_s[m])
                        px_parts.append(px[m])
            if ts_parts:
                t_cat = np.concatenate(ts_parts)
                p_cat = np.concatenate(px_parts)
                ord_i = np.argsort(t_cat, kind="mergesort")
                out[p] = (t_cat[ord_i], p_cat[ord_i])
            else:
                out[p] = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64))
        return out

    # Serve 1,000s ticks_history pages through REAL SotaDerivAsyncClient.request_public_read over a fake WS!
    class _HistWS:
        def __init__(self, sym_to_ticks: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
            self.sym_to_ticks = sym_to_ticks
            self.inbox: asyncio.Queue[str] = asyncio.Queue()
            self.sent: list[dict[str, Any]] = []

        async def send(self, text: str) -> None:
            req = json.loads(text)
            self.sent.append(req)
            rid = req["req_id"]
            if "ticks_history" in req:
                sym = req["ticks_history"]
                s_ep = int(req["start"])
                e_ep = int(req["end"])
                ts_a, px_a = self.sym_to_ticks[sym]
                m = (ts_a >= s_ep) & (ts_a <= e_ep)
                resp = {"req_id": rid, "history": {"times": ts_a[m].tolist(), "prices": px_a[m].tolist()}}
                await self.inbox.put(json.dumps(resp))
            else:
                await self.inbox.put(json.dumps({"req_id": rid, "echo_req": req}))

        async def recv(self) -> str:
            return await self.inbox.get()

        async def close(self) -> None:
            return None

    async def _run_warmup_proofs() -> None:
        sym_map_inv = {SOTA_DERIV_SYMBOLS[p]: p for p in MAJORS}
        for ep in target_epochs:
            win_ticks = _load_window_ticks(ep)
            sym_ticks = {SOTA_DERIV_SYMBOLS[p]: win_ticks[p] for p in MAJORS}
            fws = _HistWS(sym_ticks)
            with tempfile.TemporaryDirectory() as td_w:
                buf = LiveTickBuffer(pairs=["USDJPY"], store_dir=Path(td_w) / "store", ring_seconds=345600)
                async with SotaDerivAsyncClient(
                    "wss://api.derivws.com/trading/v1/options/ws/public",
                    allow_buy=False,
                    ws_opener=lambda _u: fws,
                ) as scli:
                    # Warmup via SotaDerivAsyncClient.request_public_read (1,000s chunks, 36h/1600 active bars/5d!)
                    await buf.warmup_from_public_ws(
                        scli,
                        lookback_hours=36,
                        min_bars=1600,
                        request_spacing_s=0.0,
                        history_request_fn=scli.request_public_read,
                        now_epoch=ep,
                    )

                assert any("ticks_history" in m for m in fws.sent)
                snap = buf.snapshot_ticks()
                frow = build_pair_decision_snapshot("USDJPY", snap, ep, fcols, min_active_bars=1600)
                # Compare against reference DEV parquet row at exact epoch ep!
                idx_pos = int(np.searchsorted(ts_col, ep))
                assert idx_pos < len(ts_col) and int(ts_col[idx_pos]) == ep
                ref_row = df_usdjpy.iloc[idx_pos][fcols].astype(np.float64)
                got_row = frow.row[fcols].astype(np.float64)
                assert np.array_equal(np.isnan(got_row.to_numpy()), np.isnan(ref_row.to_numpy())), f"NaN mask drift at {ep}"
                fin_m = np.isfinite(ref_row.to_numpy())
                max_diff = float(np.max(np.abs(got_row.to_numpy()[fin_m] - ref_row.to_numpy()[fin_m])))
                assert max_diff <= 1e-5, f"feature drift {max_diff} > 1e-5 at epoch {ep}"
                # Exact score_row() direction & gate agreement across incumbents (+ candidate when frozen)!
                for b in check_books:
                    sc_ref = b.score_row(ref_row, ep)
                    sc_got = b.score_row(got_row, ep)
                    assert sc_got.direction == sc_ref.direction and sc_got.gate_passed == sc_ref.gate_passed, (
                        f"score_row mismatch on {b.book_id} at {ep}: {sc_got} vs {sc_ref}"
                    )

                # Restart persistence & standalone watermark consumer check (on first epoch)!
                if ep == target_epochs[0]:
                    buf.persist_snapshot(minute_boundary_epoch=ep)
                    wm = read_tick_store_watermark(Path(td_w) / "store")
                    assert int(wm["minute_boundary_epoch"]) == ep
                    buf_reloaded = LiveTickBuffer(pairs=["USDJPY"], store_dir=Path(td_w) / "store")
                    buf_reloaded.load_persisted()
                    frow2 = build_pair_decision_snapshot("USDJPY", buf_reloaded.snapshot_ticks(), ep, fcols, min_active_bars=1600)
                    assert np.allclose(frow2.row[fcols].fillna(-999.0), got_row.fillna(-999.0), atol=1e-12)

                    # Missing-peer scenario -> fails closed with SotaFeatureError!
                    snap_no_peer = dict(snap)
                    snap_no_peer["EURUSD"] = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64))
                    _expect_raises(SotaFeatureError, build_pair_decision_snapshot, "USDJPY", snap_no_peer, ep, fcols, min_active_bars=1600)

                    # Insufficient active bars (< 1600) -> fails closed SotaFeatureError!
                    short_snap = {p: (t[-50:], px[-50:]) for p, (t, px) in snap.items()}
                    _expect_raises(SotaFeatureError, build_pair_decision_snapshot, "USDJPY", short_snap, ep, fcols, min_active_bars=1600)

        # Standalone run_stream_daemon compatibility: plain public client.request callable (NOT SotaDerivAsyncClient)!
        ep0 = target_epochs[0]
        win0 = _load_window_ticks(ep0)
        sym0 = {SOTA_DERIV_SYMBOLS[p]: win0[p] for p in MAJORS}
        pub_calls = 0

        class _StandalonePubClient:
            async def request(self, payload: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
                nonlocal pub_calls
                pub_calls += 1
                sym = payload["ticks_history"]
                ts_a, px_a = sym0[sym]
                m = (ts_a >= int(payload["start"])) & (ts_a <= int(payload["end"]))
                return {"history": {"times": ts_a[m].tolist(), "prices": px_a[m].tolist()}}

        with tempfile.TemporaryDirectory() as td_st:
            st_dir = Path(td_st) / "ext_store"
            buf_ext = LiveTickBuffer(pairs=["USDJPY"], store_dir=st_dir)
            # Omitting history_request_fn defaults to standalone client.request!
            await buf_ext.warmup_from_public_ws(
                _StandalonePubClient(),
                lookback_hours=36,
                min_bars=1600,
                request_spacing_s=0.0,
                now_epoch=ep0,
            )
            assert pub_calls > 0
            buf_ext.persist_snapshot(minute_boundary_epoch=ep0)
            wm_ext = read_tick_store_watermark(st_dir)
            assert int(wm_ext["minute_boundary_epoch"]) == ep0
            # External executor consumes persisted store + watermark!
            ext_reader = LiveTickBuffer(pairs=["USDJPY"], store_dir=st_dir)
            ext_reader.load_persisted()
            frow_ext = build_pair_decision_snapshot("USDJPY", ext_reader.snapshot_ticks(), ep0, fcols, min_active_bars=1600)
            assert frow_ext.decision_epoch == ep0

    asyncio.run(_run_warmup_proofs())


def test_phase1_sota_client_and_buy_intent_recovery() -> None:
    from deriv_client import DerivEnv
    from deriv_sota_executor import (
        SotaDerivAsyncClient,
        _WireDecimal,
        check_exact_rational_quote_floor,
        connect_sota_queue_v3,
        create_v3_queue_schema_atomic,
        is_safety_latch_set,
        strict_json_loads_wire,
    )

    # 1. Exact wire JSON decoding: _WireDecimal raw token, duplicate-key & non-finite rejection, exact rational floor!
    from deriv_client import DerivAPIError
    parsed_w = strict_json_loads_wire('{"ask_price": 1.0, "payout": 1.7799999999}')
    assert isinstance(parsed_w["payout"], _WireDecimal) and str(parsed_w["payout"]) == "1.7799999999"
    _expect_raises(DerivAPIError, strict_json_loads_wire, '{"a": 1, "a": 2}')
    _expect_raises(DerivAPIError, strict_json_loads_wire, '{"a": NaN}')
    _expect_raises(DerivAPIError, strict_json_loads_wire, '{"a": Infinity}')
    _, _, _, err_1779 = check_exact_rational_quote_floor(parsed_w, min_payout_ratio=1.78)
    assert err_1779 == "payout_below_exact_rational_floor"
    _, _, _, err_1780 = check_exact_rational_quote_floor(strict_json_loads_wire('{"ask_price": 1.0, "payout": 1.78}'), min_payout_ratio=1.78)
    assert err_1780 is None

    # 2. Fake WS testing ALL 5 capability classes, both options_otp bootstrap balance AND v3_authorize,
    #    generation invalidation, allow_buy=False blocking buy & sell (0 bytes!), durable buy_intent & recovery!
    class _FakeWS:
        def __init__(self, handler: Any) -> None:
            self.handler = handler
            self.inbox: asyncio.Queue[str] = asyncio.Queue()
            self.raw_sent: list[str] = []

        async def send(self, text: str) -> None:
            self.raw_sent.append(text)
            req = json.loads(text)
            resp = self.handler(req)
            if isinstance(resp, BaseException):
                raise resp
            if isinstance(resp, str):
                await self.inbox.put(resp.replace("__REQ_ID__", str(req["req_id"])))
            elif resp is not None:
                resp["req_id"] = req["req_id"]
                await self.inbox.put(json.dumps(resp))

        async def recv(self) -> str:
            return await self.inbox.get()

        async def close(self) -> None:
            return None

    async def _run_client_tests() -> None:
        # (a) options_otp bootstrap balance + venue time + allow_buy=False blocking BOTH buy & sell (0 bytes!)
        def _otp_handler(req: dict[str, Any]) -> dict[str, Any]:
            if "balance" in req:
                return {"balance": {"loginid": "VRTC123456", "currency": "USD", "balance": 1000.0}}
            if "time" in req:
                return {"time": 1784124000}
            if "portfolio" in req:
                return {"portfolio": {"contracts": []}}
            if "proposal" in req:
                return {"proposal": {"id": "p-otp-1", "ask_price": 1.0, "payout": 1.82, "currency": "USD"}}
            return {"echo_req": req}

        fws_otp = _FakeWS(_otp_handler)
        async with SotaDerivAsyncClient(
            "wss://api.derivws.com/trading/v1/options/ws/demo",
            allow_buy=False,
            expected_account_id="VRTC123456",
            ws_opener=lambda _u: fws_otp,
        ) as cli:
            # Generic request() MUST refuse!
            try:
                await cli.request({"balance": 1})
                raise AssertionError("generic request() must be refused")
            except ExecutorError:
                pass
            # Unverified account_read/proposal MUST refuse with 0 bytes!
            sent_before = len(fws_otp.raw_sent)
            try:
                await cli.request_account_read({"portfolio": 1})
                raise AssertionError("unverified account_read must fail")
            except ExecutorError:
                pass
            assert len(fws_otp.raw_sent) == sent_before

            # Establish options_otp identity (bootstrap balance + time)
            v_ep = await cli.establish_identity(account_id="VRTC123456", account_mode="demo", transport_kind="options_otp")
            assert v_ep == 1784124000 and cli.is_identity_verified()
            # Second bootstrap balance on same generation MUST refuse!
            try:
                await cli.request_identity({"balance": 1})
                raise AssertionError("second bootstrap balance on same generation must fail")
            except ExecutorError:
                pass
            # Verified account_read & proposal succeed!
            await cli.request_account_read({"portfolio": 1})
            await cli.request_proposal({"proposal": 1, "amount": 1.0})
            # allow_buy=False MUST refuse BOTH buy AND sell with ZERO bytes sent!
            n_before_mut = len(fws_otp.raw_sent)
            for mut_payload in ({"buy": "p-otp-1", "price": 1.0}, {"sell": 12345, "price": 0}):
                try:
                    await cli.request_trade_mutation(mut_payload)
                    raise AssertionError(f"allow_buy=False failed to block {mut_payload}")
                except ExecutorError:
                    pass
            assert len(fws_otp.raw_sent) == n_before_mut, "allow_buy=False sent bytes on buy/sell!"
            # Stale generation refused with 0 bytes!
            try:
                await cli.request_proposal({"proposal": 1}, expected_generation=cli.generation + 99)
                raise AssertionError("stale generation must fail")
            except ExecutorError:
                pass
            assert len(fws_otp.raw_sent) == n_before_mut

        # (b) Reject conflicting/missing OTP bootstrap loginid, conflicting is_virtual=0, AND failed time clearing verification!
        fws_bad_otp = _FakeWS(lambda req: {"balance": {"loginid": "VRTC999999", "currency": "USD"}} if "balance" in req else {"time": 1784124000})
        async with SotaDerivAsyncClient(
            "wss://api.derivws.com/trading/v1/options/ws/demo",
            allow_buy=False,
            expected_account_id="VRTC123456",
            ws_opener=lambda _u: fws_bad_otp,
        ) as cli_bad:
            try:
                await cli_bad.establish_identity(account_id="VRTC123456", account_mode="demo", transport_kind="options_otp")
                raise AssertionError("mismatched OTP loginid must fail")
            except ExecutorError:
                pass
            assert not cli_bad.is_identity_verified()

        fws_bad_virt = _FakeWS(lambda req: {"balance": {"loginid": "VRTC123456", "currency": "USD", "is_virtual": 0}} if "balance" in req else {"time": 1784124000})
        async with SotaDerivAsyncClient(
            "wss://api.derivws.com/trading/v1/options/ws/demo",
            allow_buy=False,
            expected_account_id="VRTC123456",
            ws_opener=lambda _u: fws_bad_virt,
        ) as cli_bv:
            try:
                await cli_bv.establish_identity(account_id="VRTC123456", account_mode="demo", transport_kind="options_otp")
                raise AssertionError("OTP conflicting is_virtual=0 on demo must fail")
            except ExecutorError:
                pass
            assert not cli_bv.is_identity_verified()

        fws_bad_time = _FakeWS(lambda req: {"balance": {"loginid": "VRTC123456", "currency": "USD", "is_virtual": 1}} if "balance" in req else {"time": "not-an-int"})
        async with SotaDerivAsyncClient(
            "wss://api.derivws.com/trading/v1/options/ws/demo",
            allow_buy=False,
            expected_account_id="VRTC123456",
            ws_opener=lambda _u: fws_bad_time,
        ) as cli_bt:
            try:
                await cli_bt.establish_identity(account_id="VRTC123456", account_mode="demo", transport_kind="options_otp")
                raise AssertionError("invalid venue time during bootstrap must fail")
            except ExecutorError:
                pass
            assert not cli_bt.is_identity_verified(), "failed venue time MUST clear identity verification!"

        # (c) v3_authorize + SotaTradeExecutor on v3 queue:
        #     - 1.7799999999 payout & 1.0000000000000001 ask rejected with 0 buy bytes
        #     - pre-handoff refusal & stop latch set AT handoff-lock acquisition -> 0 buy bytes
        #     - post-handoff failure -> buy_unknown + blocks new buys + fails closed if POC unavailable + recovers via portfolio + authoritative POC!
        #     - live same-generation venue clock refresh clears cooldown without reconnect!
        #     - two 30m books open at same bar close without false same_pair_horizon_cooldown!
        #     - unwritable 0500 log_dir still allows SotaTradeExecutor construction & reconciliation!
        #     - differing venue buy_price trips phase1_safety_latch atomically!
        with tempfile.TemporaryDirectory() as td_ex:
            import os
            import sqlite3
            from deriv_sota_executor import commit_safety_latch
            root = Path(td_ex)
            db_p = root / "sota_trade_queue_demo.sqlite"
            raw_c = sqlite3.connect(str(db_p), isolation_level=None)
            raw_c.execute("PRAGMA foreign_keys=ON")
            raw_c.execute("PRAGMA journal_mode=WAL")
            create_v3_queue_schema_atomic(raw_c, account_id="VRTC123456", account_mode="demo")
            raw_c.close()
            conn = connect_sota_queue_v3(db_p, expected_account_id="VRTC123456", expected_account_mode="demo")
            t0 = datetime(2026, 7, 15, 14, 0, 0, tzinfo=timezone.utc)
            ep0 = int(t0.timestamp())

            mode_state: dict[str, Any] = {
                "ask_token": "1.0",
                "payout_token": "1.7799999999",
                "venue_time": ep0 + 2,
                "fail_buy": False,
                "fail_poc": False,
                "omit_poc_money": False,
                "buy_price": 1.0,
                "poc_buy_price": 1.0,
                "contract_seq": 7000,
                "portfolio": [],
            }

            def _v3_handler(req: dict[str, Any]) -> Any:
                if "authorize" in req:
                    return {"authorize": {"loginid": "VRTC123456", "currency": "USD", "is_virtual": 1}}
                if "time" in req:
                    return {"time": mode_state["venue_time"]}
                if "proposal" in req:
                    return (
                        f'{{"req_id": __REQ_ID__, "proposal": {{"id": "prop-{req["req_id"]}", '
                        f'"ask_price": {mode_state["ask_token"]}, "payout": {mode_state["payout_token"]}, "currency": "USD"}}}}'
                    )
                if "buy" in req:
                    if mode_state["fail_buy"]:
                        raise RuntimeError("simulated post-handoff transport loss")
                    mode_state["contract_seq"] += 1
                    cid_n = mode_state["contract_seq"]
                    return {
                        "buy": {
                            "contract_id": cid_n,
                            "buy_price": mode_state["buy_price"],
                            "payout": float(mode_state["payout_token"]),
                            "start_time": mode_state["venue_time"],
                            "expiry_time": mode_state["venue_time"] + 1800,
                        }
                    }
                if "portfolio" in req:
                    return {"portfolio": {"contracts": list(mode_state["portfolio"])}}
                if "statement" in req:
                    return {"statement": {"transactions": []}}
                if "proposal_open_contract" in req:
                    if mode_state["fail_poc"]:
                        raise RuntimeError("simulated POC read failure")
                    poc_obj: dict[str, Any] = {
                        "contract_id": req["contract_id"],
                        "is_sold": 1,
                        "status": "won",
                        "buy_price": mode_state["poc_buy_price"],
                        "payout": 1.82,
                        "date_start": ep0 + 2,
                        "date_expiry": ep0 + 1802,
                        "underlying_symbol": "frxUSDJPY",
                        "contract_type": "CALL",
                    }
                    if not mode_state["omit_poc_money"]:
                        poc_obj["sell_price"] = 1.82
                        poc_obj["profit"] = 0.82
                    return {"proposal_open_contract": poc_obj}
                return {"echo_req": req}

            fws_v3 = _FakeWS(_v3_handler)
            env_t = DerivEnv(app_id="1089", pat="tok", account_id="VRTC123456")
            ex = SotaTradeExecutor(
                account_mode="demo",
                stake_usd=1.0,
                db_path=db_p,
                log_dir=root / "logs",
                risk_cfg=RiskBreakerConfig(max_stake_usd=5.0, max_daily_loss_usd=25.0, max_drawdown_usd=50.0, max_open_contracts=3, max_usd_factor_open=3, max_jpy_factor_open=3),
                ev_margin=0.005,
                env=env_t,
            )
            ex.transport_kind = "v3_authorize"
            handoff_lp = root / "sota_handoff.lock"
            async with SotaDerivAsyncClient(
                "wss://ws.derivws.com/websockets/v3?app_id=1089",
                allow_buy=True,
                expected_account_id="VRTC123456",
                handoff_lock_path=handoff_lp,
                ws_opener=lambda _u: fws_v3,
            ) as cli_v3:
                ex.client = cli_v3
                # 1.7799999999 MUST be rejected with 0 buy bytes!
                sig1 = {
                    "book_id": "USDJPY.sota_ens5_m30_up.v1",
                    "pair": "USDJPY",
                    "horizon_minutes": 30,
                    "side": "UP",
                    "lane": "wall",
                    "offset_id": 0,
                    "signal_close_utc": t0.isoformat(),
                    "effective_floor": 0.57,
                    "calibrated_win_prob": 0.59,
                    "min_payout_ratio": 1.78,
                    "book_content_id": "c8d0b13768a4c649",
                }
                _, sid1 = enqueue_sota_signal(conn, sig1)
                r1 = await ex.process_one_signal(conn, sid1, sig1, now_utc=t0 + timedelta(seconds=2))
                assert r1["reason"] == "payout_below_exact_rational_floor"
                assert not any('"buy"' in s for s in fws_v3.raw_sent)

                # Raw ask 1.0000000000000001 MUST be rejected with 0 buy bytes (BLOCKER #2)!
                mode_state["ask_token"] = "1.0000000000000001"
                mode_state["payout_token"] = "1.82"
                sig_ask = {**sig1, "offset_id": 99}
                _, sid_ask = enqueue_sota_signal(conn, sig_ask)
                r_ask = await ex.process_one_signal(conn, sid_ask, sig_ask, now_utc=t0 + timedelta(seconds=2))
                assert r_ask["reason"] == "ask_price_not_exact_whole_cents", f"unexpected ask rejection: {r_ask}"
                assert not any('"buy"' in s for s in fws_v3.raw_sent)
                mode_state["ask_token"] = "1.0"

                # Pre-handoff refusal when allow_buy=False -> buy_refused_pre_send & 0 buy bytes!
                cli_v3.allow_buy = False
                sig2 = {**sig1, "offset_id": 1}
                _, sid2 = enqueue_sota_signal(conn, sig2)
                r2 = await ex.process_one_signal(conn, sid2, sig2, now_utc=t0 + timedelta(seconds=2))
                assert r2["status"] == "buy_refused_pre_send"
                assert not any('"buy"' in s for s in fws_v3.raw_sent)

                # Stop latch committed AT handoff lock acquisition MUST abort inside sota_handoff lock with 0 buy bytes (BLOCKER #2)!
                cli_v3.allow_buy = True
                import deriv_sota_executor as ex_mod
                orig_flock = ex_mod.acquire_sota_flock
                from contextlib import contextmanager

                @contextmanager
                def _trip_latch_at_handoff(path_l: Path, *a: Any, **kw: Any):
                    with orig_flock(path_l, *a, **kw):
                        if path_l == handoff_lp:
                            handoff_hits[0] += 1
                            # Trip latch on the 2nd handoff lock acquisition (the BUY frame, AFTER proposal succeeded!)
                            if handoff_hits[0] == 2 and is_safety_latch_set(conn) is None:
                                commit_safety_latch(conn, "mid_handoff_stop_latch")
                        yield

                handoff_hits = [0]
                ex_mod.acquire_sota_flock = _trip_latch_at_handoff  # type: ignore[assignment]
                try:
                    sig_lat = {**sig1, "offset_id": 98}
                    _, sid_lat = enqueue_sota_signal(conn, sig_lat)
                    r_lat = await ex.process_one_signal(conn, sid_lat, sig_lat, now_utc=t0 + timedelta(seconds=2))
                    db_lat_reason = str(conn.execute("SELECT reason FROM sota_signals WHERE signal_id=?", (sid_lat,)).fetchone()[0])
                    assert handoff_hits[0] == 2, f"expected proposal+buy handoff hits, got {handoff_hits[0]}"
                    assert r_lat["status"] == "buy_refused_pre_send" and r_lat["reason"] == "pre_send_refused", f"unexpected: {r_lat}"
                    assert "latch_or_kill_detected_under_handoff_lock" in db_lat_reason, f"unexpected db reason: {db_lat_reason}"
                    assert not any('"buy"' in s for s in fws_v3.raw_sent), "buy frame crossed handoff after latch committed!"
                finally:
                    ex_mod.acquire_sota_flock = orig_flock  # type: ignore[assignment]
                    conn.execute("DELETE FROM sota_risk_state WHERE key='phase1_safety_latch'")

                # Post-handoff failure -> buy_unknown, blocks new buy until authoritative POC confirms!
                mode_state["fail_buy"] = True
                sig3 = {**sig1, "offset_id": 2}
                _, sid3 = enqueue_sota_signal(conn, sig3)
                r3 = await ex.process_one_signal(conn, sid3, sig3, now_utc=t0 + timedelta(seconds=2))
                assert r3["status"] == "buy_unknown"
                # Next signal blocked by unresolved_buy_uncertainty_halt
                sig_blk = {**sig1, "book_id": "USDJPY.sota_stack05_m30_up.v1", "offset_id": 2}
                _, sid_blk = enqueue_sota_signal(conn, sig_blk)
                r_blk = await ex.process_one_signal(conn, sid_blk, sig_blk, now_utc=t0 + timedelta(seconds=2))
                assert r_blk["reason"] == "unresolved_buy_uncertainty_halt"
                # BLOCKER #3: Ambiguous duplicate candidates MUST refuse & set safety latch (leaving buy_unknown)!
                mode_state["portfolio"] = [
                    {"contract_id": 7999, "buy_price": 1.0, "payout": 1.82, "date_start": ep0 + 2, "expiry_time": ep0 + 1802, "passthrough": {"sota_signal_id": sid3}},
                    {"contract_id": 7998, "buy_price": 1.0, "payout": 1.82, "date_start": ep0 + 2, "expiry_time": ep0 + 1802, "passthrough": {"sota_signal_id": sid3}},
                ]
                await ex.reconcile_open_contracts(conn, now_utc=t0 + timedelta(minutes=31))
                assert conn.execute("SELECT status FROM sota_signals WHERE signal_id=?", (sid3,)).fetchone()[0] == "buy_unknown"
                assert is_safety_latch_set(conn) is not None
                conn.execute("DELETE FROM sota_risk_state WHERE key='phase1_safety_latch'")

                # BLOCKER #3: If POC fails during recovery, signal MUST remain buy_unknown (never clear uncertainty on inferred cost)!
                mode_state["portfolio"] = [{"contract_id": 7999, "buy_price": 1.0, "payout": 1.82, "date_start": ep0 + 2, "expiry_time": ep0 + 1802, "passthrough": {"sota_signal_id": sid3}}]
                mode_state["fail_poc"] = True
                await ex.reconcile_open_contracts(conn, now_utc=t0 + timedelta(minutes=31))
                assert conn.execute("SELECT status FROM sota_signals WHERE signal_id=?", (sid3,)).fetchone()[0] == "buy_unknown"
                # Now allow POC, but omit sell_price/profit AND buy_price first -> adopts as cost_unknown=1 + latched, stays open!
                mode_state["fail_poc"] = False
                mode_state["omit_poc_money"] = True
                mode_state["poc_buy_price"] = None
                await ex.reconcile_open_contracts(conn, now_utc=t0 + timedelta(minutes=31))
                c7999_u = conn.execute("SELECT status, cost_unknown, buy_price_cents FROM sota_contracts WHERE contract_id='7999'").fetchone()
                assert c7999_u["status"] == "open" and int(c7999_u["cost_unknown"]) == 1 and c7999_u["buy_price_cents"] is None
                assert is_safety_latch_set(conn) is not None
                conn.execute("DELETE FROM sota_risk_state WHERE key='phase1_safety_latch'")
                # Now supply authoritative POC buy_price + sell_price + profit -> settles cleanly & resolves cost_unknown=0!
                mode_state["poc_buy_price"] = 1.0
                mode_state["omit_poc_money"] = False
                await ex.reconcile_open_contracts(conn, now_utc=t0 + timedelta(minutes=31, seconds=5))
                c7999 = conn.execute("SELECT status, cost_unknown, buy_price_cents, venue_underlying_symbol, venue_contract_type, profit_cents FROM sota_contracts WHERE contract_id='7999'").fetchone()
                assert c7999["status"] == "won" and int(c7999["cost_unknown"]) == 0 and int(c7999["buy_price_cents"]) == 100 and c7999["venue_underlying_symbol"] == "frxUSDJPY" and c7999["venue_contract_type"] == "CALL" and int(c7999["profit_cents"]) == 82

                # Two distinct 30m books (ens5_m30 & stack05_m30) BOTH open at same bar close t1 using FRESH same-generation venue clock (MAJOR #1)!
                mode_state["fail_buy"] = False
                mode_state["portfolio"] = []
                t1 = t0 + timedelta(hours=1)
                # Do NOT touch ex.last_venue_time_epoch manually; advance server venue_time to prove live clock refresh clears cooldown!
                mode_state["venue_time"] = int(t1.timestamp()) + 2
                sig_a = {**sig1, "signal_close_utc": t1.isoformat(), "offset_id": 0}
                sig_b = {**sig1, "book_id": "USDJPY.sota_stack05_m30_up.v1", "signal_close_utc": t1.isoformat(), "offset_id": 0}
                ok_a, sid_a = enqueue_sota_signal(conn, sig_a)
                ok_b, sid_b = enqueue_sota_signal(conn, sig_b)
                assert ok_a and ok_b  # UNIQUE(book_id, ...) allows both!
                ra = await ex.process_one_signal(conn, sid_a, sig_a, now_utc=t1 + timedelta(seconds=2))
                rb = await ex.process_one_signal(conn, sid_b, sig_b, now_utc=t1 + timedelta(seconds=2))
                assert ra["status"] == "bought" and rb["status"] == "bought", f"both 30m books should open: {ra}, {rb}"

                # Differing venue buy_price (1.50 != 1.00) trips phase1_safety_latch atomically!
                mode_state["buy_price"] = 1.50
                sig_m = {**sig1, "book_id": "USDJPY.sota_ens5_m15_up.v1", "horizon_minutes": 15, "min_payout_ratio": 1.76, "signal_close_utc": t1.isoformat(), "offset_id": 0}
                _, sid_m = enqueue_sota_signal(conn, sig_m)
                rm = await ex.process_one_signal(conn, sid_m, sig_m, now_utc=t1 + timedelta(seconds=2))
                assert rm["status"] == "bought"
                assert is_safety_latch_set(conn) is not None

                # MAJOR #2: Unwritable 0500 log_dir MUST NOT break SotaTradeExecutor construction or reconcile_open_contracts!
                ro_log_d = root / "ro_exec_logs"
                ro_log_d.mkdir(mode=0o500)
                try:
                    ex_ro = SotaTradeExecutor(
                        account_mode="demo",
                        stake_usd=1.0,
                        db_path=db_p,
                        log_dir=ro_log_d,
                        risk_cfg=RiskBreakerConfig(max_stake_usd=5.0, max_daily_loss_usd=25.0, max_drawdown_usd=50.0),
                        env=env_t,
                    )
                    ex_ro.client = cli_v3
                    await ex_ro.reconcile_open_contracts(conn, now_utc=t1 + timedelta(minutes=35))
                finally:
                    os.chmod(ro_log_d, 0o700)
            conn.close()

    asyncio.run(_run_client_tests())


def test_phase1_v3_queue_and_6shape_adoption() -> None:
    import hashlib
    import os
    import sqlite3
    import deriv_sota_executor as ex_mod
    from deriv_sota_executor import (
        adopt_legacy_account,
        canonical_sota_lock_paths,
        clear_stale_legacy_lock,
        connect_sota_queue_v3,
        detect_predecessor_shape,
        export_legacy_ownership_preimage,
        init_v3_queue_first_use,
        is_safety_latch_set,
        prepare_coherent_predecessor_backup,
        request_stop_latch,
    )

    def _make_receipt(p: Path, acct: str = "VRTC123456") -> Path:
        p.write_text(
            json.dumps(
                {
                    "schema": "sota-credential-isolation-receipt/v1",
                    "account_id": acct,
                    "account_mode": "demo",
                    "legacy_credentials_revoked": True,
                    "dedicated_sota_credentials_provisioned": True,
                    "attested_by": "owner",
                    "attested_utc": "2026-11-25T00:00:00Z",
                }
            ),
            encoding="utf-8",
        )
        return p

    # 1. --init-queue --reason first_use (absent, empty v0, idempotent empty v3, PRAGMA assertions)
    with tempfile.TemporaryDirectory() as td_i:
        root = Path(td_i)
        pw_home = root / "home"
        pw_home.mkdir()
        rcpt = _make_receipt(root / "rcpt.json")
        fur = root / "fur.json"
        fur.write_text(json.dumps({"schema": "sota-first-use-record/v1", "account_id": "VRTC123456", "account_mode": "demo"}), encoding="utf-8")
        q_db = root / "runtime" / "sota_trade_queue_demo.sqlite"
        r1 = init_v3_queue_first_use(queue_db=q_db, account_id="VRTC123456", account_mode="demo", first_use_record_path=fur, receipt_path=rcpt, pw_home=pw_home)
        assert r1["status"] == "initialized_v3"
        # Idempotent retry on empty v3
        r2 = init_v3_queue_first_use(queue_db=q_db, account_id="VRTC123456", account_mode="demo", first_use_record_path=fur, receipt_path=rcpt, pw_home=pw_home)
        assert r2["status"] == "idempotent_empty_v3"

        # 1b. Demo-only single_host_units_masked credential mode (positive + 5 adversarial refusals)
        import os as _os_m
        from deriv_sota_executor import (
            CREDENTIAL_ISOLATION_MODE_SINGLE_HOST_MASKED,
            DEMO_ENV_ACCOUNT_SENTINEL,
            LEGACY_MASKED_UNIT_NAMES,
            validate_credential_isolation_receipt,
        )
        m_root = root / "masked_mode"
        m_home = m_root / "home"
        m_systemd = m_home / ".config" / "systemd" / "user"
        m_systemd.mkdir(parents=True)
        for u_name in LEGACY_MASKED_UNIT_NAMES:
            _os_m.symlink("/dev/null", m_systemd / u_name)
        m_rcpt = m_root / "rcpt_masked.json"
        m_rcpt.write_text(
            json.dumps({
                "schema": "sota-credential-isolation-receipt/v1",
                "credential_isolation_mode": CREDENTIAL_ISOLATION_MODE_SINGLE_HOST_MASKED,
                "account_id": DEMO_ENV_ACCOUNT_SENTINEL,
                "account_mode": "demo",
                "legacy_credentials_revoked": False,
                "legacy_units_masked": list(LEGACY_MASKED_UNIT_NAMES),
                "attested_by": "owner_demo",
                "attested_utc": "2026-11-26T18:22:00Z",
            }),
            encoding="utf-8",
        )
        m_fur = m_root / "fur_masked.json"
        m_fur.write_text(
            json.dumps({"schema": "sota-first-use-record/v1", "account_id": DEMO_ENV_ACCOUNT_SENTINEL, "account_mode": "demo"}),
            encoding="utf-8",
        )
        m_db = m_root / "runtime" / "sota_trade_queue_demo.sqlite"
        # Positive: Demo VRTC* + all 4 /dev/null mask symlinks + systemd probe passes receipt & first_use init!
        ok_states = {u: {"Id": u, "LoadState": "masked", "UnitFileState": "masked", "ActiveState": "inactive"} for u in LEGACY_MASKED_UNIT_NAMES}
        validate_credential_isolation_receipt(
            m_rcpt, expected_account_id="VRTC999999", expected_account_mode="demo", pw_home=m_home,
            systemd_state_probe=lambda _units: ok_states,
        )
        rm1 = init_v3_queue_first_use(queue_db=m_db, account_id="VRTC999999", account_mode="demo", first_use_record_path=m_fur, receipt_path=m_rcpt, pw_home=m_home)
        assert rm1["status"] == "initialized_v3"
        # Negative 1: Live / non-VRTC account unconditionally refused (including live receipt hitting Demo-only guard)!
        live_rcpt = m_root / "rcpt_masked_live.json"
        live_rcpt.write_text(
            json.dumps({
                "schema": "sota-credential-isolation-receipt/v1",
                "credential_isolation_mode": CREDENTIAL_ISOLATION_MODE_SINGLE_HOST_MASKED,
                "account_id": "CR123456",
                "account_mode": "live",
                "legacy_credentials_revoked": False,
                "legacy_units_masked": list(LEGACY_MASKED_UNIT_NAMES),
                "attested_by": "owner_demo",
                "attested_utc": "2026-11-26T18:22:00Z",
            }),
            encoding="utf-8",
        )
        try:
            validate_credential_isolation_receipt(live_rcpt, expected_account_id="CR123456", expected_account_mode="live", pw_home=m_home)
            raise AssertionError("expected Demo-only refusal for live receipt")
        except ExecutorError as exc:
            assert "ONLY for Demo accounts" in str(exc) and "CR123456" not in str(exc)
        # Positive 1b: Deriv Options v1 D[A-Z]{2}[0-9]+ Demo ID passes receipt, first_use init, options_otp identity, and scrubbing!
        validate_credential_isolation_receipt(
            m_rcpt, expected_account_id="DMO12345678", expected_account_mode="demo", pw_home=m_home,
            systemd_state_probe=lambda _units: ok_states,
        )
        m_db_opt = m_root / "runtime" / "sota_trade_queue_demo_opt.sqlite"
        rm_opt = init_v3_queue_first_use(queue_db=m_db_opt, account_id="DMO12345678", account_mode="demo", first_use_record_path=m_fur, receipt_path=m_rcpt, pw_home=m_home)
        assert rm_opt["status"] == "initialized_v3"
        import deriv_sota_executor as _ex_scr
        assert "DMO12345678" not in _ex_scr._scrub_secret_text("err DMO12345678 VRW87654321 MF12345")
        assert "VRW87654321" not in _ex_scr._scrub_secret_text("err DMO12345678 VRW87654321 MF12345")
        assert "MF12345" not in _ex_scr._scrub_secret_text("err DMO12345678 VRW87654321 MF12345")
        # Standalone revoked receipt MUST refuse live-prefix MF* in demo mode!
        mf_rcpt = m_root / "rcpt_mf_demo.json"
        mf_rcpt.write_text(json.dumps({"schema": "sota-credential-isolation-receipt/v1", "account_id": "MF123456", "account_mode": "demo", "legacy_credentials_revoked": True, "attested_by": "owner_demo", "attested_utc": "2026-11-26T18:22:00Z"}), encoding="utf-8")
        try:
            validate_credential_isolation_receipt(mf_rcpt, expected_account_id="MF123456", expected_account_mode="demo", pw_home=m_home)
            raise AssertionError("expected refusal for MF* in demo revoked receipt")
        except ExecutorError:
            pass
        # Offline config validator MUST reject Options-only DMO* when DERIV_WS_TRANSPORT=v3_authorize!
        fx_repo_dmo = m_root / "fx_repo"
        fx_repo_dmo.mkdir(parents=True, exist_ok=True)
        orig_tr = _os_m.environ.get("DERIV_WS_TRANSPORT")
        try:
            _os_m.environ["DERIV_WS_TRANSPORT"] = "v3_authorize"
            try:
                _ex_scr.resolve_and_validate_sota_runtime_config(
                    account_mode="demo", account_id="DMO12345678",
                    book_ids_raw="USDJPY.sota_ens5_m30_up.v1",
                    stake_raw="1.0", max_stake_usd=5.0, max_daily_loss_usd=25.0, max_drawdown_usd=50.0,
                    max_open_contracts=3, max_usd_factor_open=3, max_jpy_factor_open=3,
                    absolute_breakeven_ceiling=0.60, ev_margin=0.005, max_buy_dispatch_delay_s=4.0,
                    queue_db_raw="deriv_data/runtime/sota_trade_queue_demo.sqlite", log_dir_raw="logs/sota_runtime",
                    credential_isolation_receipt_raw=str(m_rcpt),
                    fixture_home=m_home, fixture_repo_root=fx_repo_dmo, allow_fixture=True,
                )
                raise AssertionError("expected refusal for DMO* with v3_authorize")
            except ExecutorError as exc:
                assert "DERIV_WS_TRANSPORT" in str(exc)
            # And options_otp with relative queue_db_raw MUST pass!
            _os_m.environ["DERIV_WS_TRANSPORT"] = "options_otp"
            cfg_rel = _ex_scr.resolve_and_validate_sota_runtime_config(
                account_mode="demo", account_id="DMO12345678",
                book_ids_raw="USDJPY.sota_ens5_m30_up.v1",
                stake_raw="1.0", max_stake_usd=5.0, max_daily_loss_usd=25.0, max_drawdown_usd=50.0,
                max_open_contracts=3, max_usd_factor_open=3, max_jpy_factor_open=3,
                absolute_breakeven_ceiling=0.60, ev_margin=0.005, max_buy_dispatch_delay_s=4.0,
                queue_db_raw="deriv_data/runtime/sota_trade_queue_demo.sqlite", log_dir_raw="logs/sota_runtime",
                credential_isolation_receipt_raw=str(m_rcpt),
                fixture_home=m_home, fixture_repo_root=fx_repo_dmo, allow_fixture=True,
            )
            assert cfg_rel.queue_db == (fx_repo_dmo / "deriv_data" / "runtime" / "sota_trade_queue_demo.sqlite")
        finally:
            if orig_tr is None:
                _os_m.environ.pop("DERIV_WS_TRANSPORT", None)
            else:
                _os_m.environ["DERIV_WS_TRANSPORT"] = orig_tr
        # Mocked options_otp identity with DMO12345678 passes, while v3_authorize with DMO12345678 refuses!
        async def _check_dmo_identity() -> None:
            class _MiniWS:
                def __init__(self, responder: Any) -> None:
                    self._r = responder
                    self.inbox: asyncio.Queue[str] = asyncio.Queue()
                async def send(self, text: str) -> None:
                    req = json.loads(text)
                    res = dict(self._r(req))
                    res["req_id"] = req["req_id"]
                    await self.inbox.put(json.dumps(res))
                async def recv(self) -> str:
                    return await self.inbox.get()
                async def close(self) -> None:
                    return None
            ws_dmo = _MiniWS(lambda req: {"balance": {"loginid": "DMO12345678", "currency": "USD", "is_virtual": 1}} if "balance" in req else {"time": 1784124000})
            async with _ex_scr.SotaDerivAsyncClient("wss://api.derivws.com/trading/v1/options/ws/demo", allow_buy=False, expected_account_id="DMO12345678", ws_opener=lambda _u: ws_dmo) as c_ok:
                vt = await c_ok.establish_identity(account_id="DMO12345678", account_mode="demo", transport_kind="options_otp")
                assert vt == 1784124000
            ws_v3_bad = _MiniWS(lambda req: {"authorize": {"loginid": "DMO12345678", "currency": "USD", "is_virtual": 1}} if "authorize" in req else {"time": 1784124000})
            async with _ex_scr.SotaDerivAsyncClient("wss://ws.derivws.com/websockets/v3?app_id=1089", allow_buy=False, expected_account_id="DMO12345678", ws_opener=lambda _u: ws_v3_bad) as c_bad:
                try:
                    await c_bad.establish_identity(account_id="DMO12345678", account_mode="demo", transport_kind="v3_authorize", pat="tok")
                    raise AssertionError("expected v3_authorize refusal for DMO12345678")
                except ExecutorError:
                    pass
        asyncio.run(_check_dmo_identity())
        for bad_acct, bad_mode in (("CR123456", "demo"), ("VRTC999999", "live")):
            try:
                validate_credential_isolation_receipt(m_rcpt, expected_account_id=bad_acct, expected_account_mode=bad_mode, pw_home=m_home)
                raise AssertionError(f"expected refusal for {bad_acct}/{bad_mode}")
            except ExecutorError as exc:
                assert bad_acct not in str(exc)
        # Negative 1c: VRTC_FROM_ENV as expected_account_id refused everywhere (revoked, masked, config, first_use)!
        for r_path in (rcpt, m_rcpt):
            try:
                validate_credential_isolation_receipt(r_path, expected_account_id=DEMO_ENV_ACCOUNT_SENTINEL, expected_account_mode="demo", pw_home=m_home)
                raise AssertionError("expected refusal when expected_account_id is sentinel VRTC_FROM_ENV")
            except ExecutorError:
                pass
        try:
            init_v3_queue_first_use(queue_db=m_root / "runtime" / "s2.sqlite", account_id=DEMO_ENV_ACCOUNT_SENTINEL, account_mode="demo", first_use_record_path=m_fur, receipt_path=m_rcpt, pw_home=m_home)
            raise AssertionError("expected refusal for sentinel expected account in init_v3_queue_first_use")
        except ExecutorError:
            pass
        # Negative 1d: systemd user manager overrides (LoadState=loaded, ActiveState=active, UnitFileState=masked-runtime) refused!
        for field_k, bad_v in (("LoadState", "loaded"), ("ActiveState", "active"), ("UnitFileState", "masked-runtime")):
            bad_st = {u: dict(v) for u, v in ok_states.items()}
            bad_st[LEGACY_MASKED_UNIT_NAMES[0]][field_k] = bad_v
            try:
                validate_credential_isolation_receipt(
                    m_rcpt, expected_account_id="VRTC999999", expected_account_mode="demo", pw_home=m_home,
                    systemd_state_probe=lambda _units, _st=bad_st: _st,
                )
                raise AssertionError(f"expected systemd probe refusal for {field_k}={bad_v}")
            except ExecutorError as exc:
                assert "LoadState=masked" in str(exc)
        # Negative 2: Contradictory legacy_credentials_revoked=True refused!
        bad_rev = m_root / "rcpt_bad_rev.json"
        bad_rev.write_text(
            json.dumps({
                "schema": "sota-credential-isolation-receipt/v1",
                "credential_isolation_mode": CREDENTIAL_ISOLATION_MODE_SINGLE_HOST_MASKED,
                "account_id": DEMO_ENV_ACCOUNT_SENTINEL,
                "account_mode": "demo",
                "legacy_credentials_revoked": True,
                "legacy_units_masked": list(LEGACY_MASKED_UNIT_NAMES),
                "attested_by": "owner_demo",
                "attested_utc": "2026-11-26T18:22:00Z",
            }),
            encoding="utf-8",
        )
        try:
            validate_credential_isolation_receipt(bad_rev, expected_account_id="VRTC999999", expected_account_mode="demo", pw_home=m_home)
            raise AssertionError("expected refusal when legacy_credentials_revoked=true in single_host_units_masked")
        except ExecutorError as exc:
            assert "legacy_credentials_revoked=false" in str(exc)
        # Negative 3: VRTC_FROM_ENV refused under default revoked mode!
        bad_sent = m_root / "rcpt_bad_sent.json"
        bad_sent.write_text(
            json.dumps({
                "schema": "sota-credential-isolation-receipt/v1",
                "account_id": DEMO_ENV_ACCOUNT_SENTINEL,
                "account_mode": "demo",
                "legacy_credentials_revoked": True,
                "attested_by": "owner_demo",
                "attested_utc": "2026-11-26T18:22:00Z",
            }),
            encoding="utf-8",
        )
        try:
            validate_credential_isolation_receipt(bad_sent, expected_account_id="VRTC999999", expected_account_mode="demo", pw_home=m_home)
            raise AssertionError("expected refusal for VRTC_FROM_ENV under revoked mode")
        except ExecutorError as exc:
            assert "VRTC999999" not in str(exc)
        # Negative 3b: Non-string & _WireDecimal attested_by, explicit null mode, duplicate unit names,
        # duplicate-key receipt, symlinked receipt parent & symlink+.. traversal, secret in first_use_record!
        for bad_name, bad_obj in (
            ("bad_att", {"schema": "sota-credential-isolation-receipt/v1", "account_id": "VRTC999999", "account_mode": "demo", "legacy_credentials_revoked": True, "attested_by": {"user": "sean"}, "attested_utc": "2026-11-26T18:22:00Z"}),
            ("dec_att", {"schema": "sota-credential-isolation-receipt/v1", "account_id": "VRTC999999", "account_mode": "demo", "legacy_credentials_revoked": True, "attested_by": 1.25, "attested_utc": "2026-11-26T18:22:00Z"}),
            ("null_mode", {"schema": "sota-credential-isolation-receipt/v1", "credential_isolation_mode": None, "account_id": "VRTC999999", "account_mode": "demo", "legacy_credentials_revoked": True, "attested_by": "owner_demo", "attested_utc": "2026-11-26T18:22:00Z"}),
            ("dup_units", {"schema": "sota-credential-isolation-receipt/v1", "credential_isolation_mode": CREDENTIAL_ISOLATION_MODE_SINGLE_HOST_MASKED, "account_id": DEMO_ENV_ACCOUNT_SENTINEL, "account_mode": "demo", "legacy_credentials_revoked": False, "legacy_units_masked": [LEGACY_MASKED_UNIT_NAMES[0], LEGACY_MASKED_UNIT_NAMES[0], LEGACY_MASKED_UNIT_NAMES[1], LEGACY_MASKED_UNIT_NAMES[2]], "attested_by": "owner_demo", "attested_utc": "2026-11-26T18:22:00Z"}),
        ):
            bp = m_root / f"{bad_name}.json"
            bp.write_text(json.dumps(bad_obj), encoding="utf-8")
            try:
                validate_credential_isolation_receipt(bp, expected_account_id="VRTC999999", expected_account_mode="demo", pw_home=m_home)
                raise AssertionError(f"expected refusal for {bad_name}")
            except ExecutorError:
                pass
        dup_key_rcpt = m_root / "dup_key_rcpt.json"
        dup_key_rcpt.write_text('{"LEAK_KEY_NAME": 1, "LEAK_KEY_NAME": 2}', encoding="utf-8")
        try:
            validate_credential_isolation_receipt(dup_key_rcpt, expected_account_id="VRTC999999", expected_account_mode="demo", pw_home=m_home)
            raise AssertionError("expected refusal for duplicate-key receipt")
        except ExecutorError as exc:
            assert "LEAK_KEY_NAME" not in str(exc)
        sym_parent = m_root / "sym_rcpt_dir"
        _os_m.symlink(m_root, sym_parent)
        for bad_sym_p in (sym_parent / "rcpt_masked.json", sym_parent / ".." / "masked_mode" / "rcpt_masked.json"):
            try:
                validate_credential_isolation_receipt(bad_sym_p, expected_account_id="VRTC999999", expected_account_mode="demo", pw_home=m_home)
                raise AssertionError(f"expected refusal for symlink path {bad_sym_p}")
            except ExecutorError:
                pass
        # Negative 3c: Redirected XDG_CONFIG_HOME (including trailing space!) refused on real-host check!
        import deriv_sota_executor as _ex_xdg
        orig_pw_fn = _ex_xdg.canonical_pw_home
        orig_xdg = _os_m.environ.get("XDG_CONFIG_HOME")
        try:
            _ex_xdg.canonical_pw_home = lambda: m_home  # type: ignore[assignment]
            for bad_xdg in (f"{m_home / '.config'} ", str(m_root / "alt_cfg")):
                _os_m.environ["XDG_CONFIG_HOME"] = bad_xdg
                try:
                    validate_credential_isolation_receipt(
                        m_rcpt, expected_account_id="VRTC999999", expected_account_mode="demo",
                        pw_home=m_home, systemd_state_probe=lambda _u: ok_states,
                    )
                    raise AssertionError(f"expected refusal for XDG_CONFIG_HOME={bad_xdg!r}")
                except ExecutorError as exc:
                    assert "XDG_CONFIG_HOME" in str(exc)
        finally:
            _ex_xdg.canonical_pw_home = orig_pw_fn  # type: ignore[assignment]
            if orig_xdg is None:
                _os_m.environ.pop("XDG_CONFIG_HOME", None)
            else:
                _os_m.environ["XDG_CONFIG_HOME"] = orig_xdg
        bad_fur_sec = m_root / "fur_secret.json"
        bad_fur_sec.write_text(json.dumps({"schema": "sota-first-use-record/v1", "account_id": "VRTC999999", "account_mode": "demo", "token": "TOPSECRET_SHOULD_NOT_ECHO"}), encoding="utf-8")
        try:
            init_v3_queue_first_use(queue_db=m_root / "runtime" / "s3.sqlite", account_id="VRTC999999", account_mode="demo", first_use_record_path=bad_fur_sec, receipt_path=m_rcpt, pw_home=m_home)
            raise AssertionError("expected refusal for secret key in first_use_record")
        except ExecutorError as exc:
            assert "TOPSECRET_SHOULD_NOT_ECHO" not in str(exc) and "VRTC999999" not in str(exc)
        # Negative 4: Relative mask symlink (../../dev/null), unmasked regular file, and symlinked systemd dir refused!
        (m_systemd / LEGACY_MASKED_UNIT_NAMES[0]).unlink()
        _os_m.symlink("../../../../../dev/null", m_systemd / LEGACY_MASKED_UNIT_NAMES[0])
        try:
            validate_credential_isolation_receipt(m_rcpt, expected_account_id="VRTC999999", expected_account_mode="demo", pw_home=m_home)
            raise AssertionError("expected refusal for relative /dev/null symlink")
        except ExecutorError as exc:
            assert "/dev/null" in str(exc)
        (m_systemd / LEGACY_MASKED_UNIT_NAMES[0]).unlink()
        (m_systemd / LEGACY_MASKED_UNIT_NAMES[0]).write_text("[Unit]\n", encoding="utf-8")
        try:
            validate_credential_isolation_receipt(m_rcpt, expected_account_id="VRTC999999", expected_account_mode="demo", pw_home=m_home)
            raise AssertionError("expected refusal when one legacy unit is unmasked")
        except ExecutorError as exc:
            assert "/dev/null" in str(exc)
        sym_home = m_root / "sym_home"
        (sym_home / ".config" / "systemd").mkdir(parents=True)
        _os_m.symlink(m_systemd, sym_home / ".config" / "systemd" / "user")
        try:
            validate_credential_isolation_receipt(m_rcpt, expected_account_id="VRTC999999", expected_account_mode="demo", pw_home=sym_home)
            raise AssertionError("expected refusal for symlinked .config/systemd/user directory")
        except ExecutorError as exc:
            assert "non-symlink directory" in str(exc)
        # Negative 5: Unknown credential_isolation_mode refused!
        bad_mode_p = m_root / "rcpt_bad_mode.json"
        bad_mode_p.write_text(
            json.dumps({
                "schema": "sota-credential-isolation-receipt/v1",
                "credential_isolation_mode": "bogus_mode",
                "account_id": "VRTC999999",
                "account_mode": "demo",
                "legacy_credentials_revoked": False,
                "attested_by": "owner_demo",
                "attested_utc": "2026-11-26T18:22:00Z",
            }),
            encoding="utf-8",
        )
        try:
            validate_credential_isolation_receipt(bad_mode_p, expected_account_id="VRTC999999", expected_account_mode="demo", pw_home=m_home)
            raise AssertionError("expected refusal for unknown credential_isolation_mode")
        except ExecutorError as exc:
            assert "unsupported credential_isolation_mode" in str(exc) and "bogus_mode" not in str(exc)
        # Negative 6: Maintenance CLIs (--init-queue, --adopt-legacy-account, --register-cohort-anchor)
        # validate BEFORE any network call (for non-demo account, bad receipt, AND bad first_use record),
        # and CLI stdout redacts raw account_id!
        import contextlib as _ctx_cli
        import deriv_sota_executor as _ex_pre
        import io as _io_cli
        net_calls: list[str] = []
        orig_env_fn = _ex_pre.DerivEnv.from_env
        orig_ws_fn = _ex_pre.request_authenticated_ws_url
        try:
            _ex_pre.request_authenticated_ws_url = lambda *a, **kw: (net_calls.append("called") or ("wss://x", "options_otp"))  # type: ignore[assignment]
            # 6a. CR* account refused pre-network across ALL 3 maintenance CLIs
            _ex_pre.DerivEnv.from_env = classmethod(lambda cls: type("E", (), {"account_id": "CR999999", "pat": "tok", "app_id": "1089"})())  # type: ignore[assignment]
            for bad_argv in (
                ["--init-queue", "--reason", "first_use", "--first-use-record-json", str(m_fur), "--credential-isolation-receipt", str(m_rcpt), "--account-mode", "demo"],
                ["--adopt-legacy-account", "--maintenance-quarantine-dir", str(m_root / "q"), "--staged-v3-dir", str(m_root / "s"), "--ownership-proof-json", str(m_root / "p.json"), "--credential-isolation-receipt", str(m_rcpt), "--account-mode", "demo"],
                ["--register-cohort-anchor", "--book-id", "USDJPY.sota_ens5_m30_up.v1", "--cohort-start-utc", "2026-12-01T00:00:00Z", "--credential-isolation-receipt", str(m_rcpt), "--account-mode", "demo"],
            ):
                try:
                    _ex_pre.main(bad_argv)
                    raise AssertionError(f"expected pre-network refusal for {bad_argv}")
                except ExecutorError:
                    pass
            # 6b. Concrete VRTC* account with bad receipt OR bad first_use_record refused pre-network!
            cli_rcpt = m_root / "rcpt_cli_ok.json"
            cli_rcpt.write_text(
                json.dumps({
                    "schema": "sota-credential-isolation-receipt/v1",
                    "account_id": "VRTC999999",
                    "account_mode": "demo",
                    "legacy_credentials_revoked": True,
                    "attested_by": "owner_demo",
                    "attested_utc": "2026-11-26T18:22:00Z",
                }),
                encoding="utf-8",
            )
            _ex_pre.DerivEnv.from_env = classmethod(lambda cls: type("E", (), {"account_id": "VRTC999999", "pat": "tok", "app_id": "1089"})())  # type: ignore[assignment]
            for bad_argv, exp_sub in (
                (["--init-queue", "--reason", "first_use", "--first-use-record-json", str(m_fur), "--credential-isolation-receipt", str(bad_mode_p), "--account-mode", "demo"], "unsupported credential_isolation_mode"),
                (["--init-queue", "--reason", "first_use", "--first-use-record-json", str(bad_fur_sec), "--credential-isolation-receipt", str(cli_rcpt), "--account-mode", "demo"], "first_use record"),
                (["--adopt-legacy-account", "--maintenance-quarantine-dir", str(m_root / "q"), "--staged-v3-dir", str(m_root / "s"), "--ownership-proof-json", str(m_root / "p.json"), "--credential-isolation-receipt", str(bad_mode_p), "--account-mode", "demo"], "unsupported credential_isolation_mode"),
                (["--register-cohort-anchor", "--book-id", "USDJPY.sota_ens5_m30_up.v1", "--cohort-start-utc", "2026-12-01T00:00:00Z", "--credential-isolation-receipt", str(bad_mode_p), "--account-mode", "demo"], "unsupported credential_isolation_mode"),
            ):
                try:
                    _ex_pre.main(bad_argv)
                    raise AssertionError(f"expected pre-network refusal for {bad_argv}")
                except ExecutorError as exc:
                    assert exp_sub in str(exc), f"expected {exp_sub!r} in {exc!r}"
            assert len(net_calls) == 0, f"network call happened before validation: {net_calls}"
            # 6c. Redacted CLI stdout for --init-queue and --register-cohort-anchor (mocked venue + isolated queue/pw_home)
            cli_fur = m_root / "fur_cli_ok.json"
            cli_fur.write_text(
                json.dumps({"schema": "sota-first-use-record/v1", "account_id": "VRTC999999", "account_mode": "demo"}),
                encoding="utf-8",
            )
            cli_q = m_root / "runtime" / "cli_redacted_demo.sqlite"
            orig_canon_q = _ex_pre.canonical_sota_queue_path
            orig_pw_cli = _ex_pre.canonical_pw_home
            orig_cli_cls = _ex_pre.SotaDerivAsyncClient
            class _FakeCli:
                def __init__(self, *a: Any, **kw: Any) -> None:
                    pass
                async def __aenter__(self) -> "_FakeCli":
                    return self
                async def __aexit__(self, *a: Any) -> None:
                    return None
                async def establish_identity(self, **kw: Any) -> int:
                    return 1784124000
            try:
                _ex_pre.canonical_sota_queue_path = lambda _m, repo_root=None: cli_q  # type: ignore[assignment]
                _ex_pre.canonical_pw_home = lambda: m_home  # type: ignore[assignment]
                _ex_pre.request_authenticated_ws_url = lambda *a, **kw: ("wss://demo.test", "options_otp")  # type: ignore[assignment]
                _ex_pre.SotaDerivAsyncClient = _FakeCli  # type: ignore[assignment]
                b_init = _io_cli.StringIO()
                with _ctx_cli.redirect_stdout(b_init):
                    assert _ex_pre.main(["--init-queue", "--reason", "first_use", "--first-use-record-json", str(cli_fur), "--credential-isolation-receipt", str(cli_rcpt), "--account-mode", "demo"]) == 0
                assert "VRTC999999" not in b_init.getvalue() and "account_id_sha256" in b_init.getvalue()
                b_anc = _io_cli.StringIO()
                with _ctx_cli.redirect_stdout(b_anc):
                    assert _ex_pre.main(["--register-cohort-anchor", "--book-id", "USDJPY.sota_ens5_m30_up.v1", "--cohort-start-utc", "2026-12-01T00:00:00Z", "--credential-isolation-receipt", str(cli_rcpt), "--account-mode", "demo"]) == 0
                assert "VRTC999999" not in b_anc.getvalue() and "account_id_sha256" in b_anc.getvalue()
                # Queue binding mismatch error must also redact both raw account IDs!
                try:
                    connect_sota_queue_v3(cli_q, expected_account_id="VRTC888888", expected_account_mode="demo")
                    raise AssertionError("expected binding mismatch")
                except ExecutorError as exc:
                    assert "VRTC999999" not in str(exc) and "VRTC888888" not in str(exc) and "sha256:" in str(exc)
            finally:
                _ex_pre.canonical_sota_queue_path = orig_canon_q  # type: ignore[assignment]
                _ex_pre.canonical_pw_home = orig_pw_cli  # type: ignore[assignment]
                _ex_pre.SotaDerivAsyncClient = orig_cli_cls  # type: ignore[assignment]
        finally:
            _ex_pre.DerivEnv.from_env = orig_env_fn  # type: ignore[assignment]
            _ex_pre.request_authenticated_ws_url = orig_ws_fn  # type: ignore[assignment]
        conn_v3 = connect_sota_queue_v3(q_db, expected_account_id="VRTC123456", expected_account_mode="demo")
        assert int(conn_v3.execute("PRAGMA foreign_keys").fetchone()[0]) == 1
        assert str(conn_v3.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"

        # Unbounded Python int risk sum (> 2**62 cents!) + direct venue-time cooldown!
        t_c = datetime(2026, 7, 15, 14, 0, 0, tzinfo=timezone.utc)
        conn_v3.execute(
            "INSERT INTO sota_signals VALUES ('s-big','b1','USDJPY',30,'UP','wall',0,?,0.57,0.60,'{}','bought',NULL,'c-big',NULL,?,?)",
            (t_c.isoformat(), t_c.isoformat(), t_c.isoformat()),
        )
        # Direct venue-time cooldown: venue_end_epoch = 1784125800 -> venue_time 1784125800 blocked, 1784125801 allowed!
        conn_v3.execute(
            """
            INSERT INTO sota_contracts (
                contract_id, signal_id, book_id, book_content_id, account_id, account_mode,
                pair, horizon_minutes, side, contract_family, is_migrated_legacy, cost_unknown,
                buy_price_cents, payout_cents, sell_price_cents, profit_cents, stake_authoritative,
                venue_start_epoch, venue_end_epoch, signal_close_utc, status, opened_utc, closed_utc, updated_utc
            ) VALUES ('c-big','s-big','b1','cid1','VRTC123456','demo','USDJPY',30,'UP','risefall',0,0,100,182,182,82,1,1784124000,1784125800,?,'won',?,?,?)
            """,
            (t_c.isoformat(), t_c.isoformat(), t_c.isoformat(), t_c.isoformat()),
        )
        cfg_r = RiskBreakerConfig(max_stake_usd=5.0, max_daily_loss_usd=25.0, max_drawdown_usd=50.0, same_book_min_gap_seconds=0)
        assert check_risk_and_exposure(conn_v3, book_id="b1", pair="USDJPY", horizon_minutes=30, side="UP", stake_usd=1.0, cfg=cfg_r, now_utc=t_c + timedelta(hours=2), venue_time_epoch=1784125800) == "venue_clock_cooldown_active"
        assert check_risk_and_exposure(conn_v3, book_id="b1", pair="USDJPY", horizon_minutes=30, side="UP", stake_usd=1.0, cfg=cfg_r, now_utc=t_c + timedelta(hours=2), venue_time_epoch=1784125801) is None

        # Unbounded Python int risk sum (> 2**62 cents across two huge losses without SQLite 64-bit SUM overflow!)
        big_loss = -(2**62)
        for k_idx in range(2):
            conn_v3.execute(
                "INSERT INTO sota_signals VALUES (?,?,?,?,?,'wall',?, ?,0.57,0.60,'{}','bought',NULL,?,NULL,?,?)",
                (f"s-ov-{k_idx}", "b2", "EURUSD", 15, "UP", k_idx, t_c.isoformat(), f"c-ov-{k_idx}", t_c.isoformat(), t_c.isoformat()),
            )
            conn_v3.execute(
                """
                INSERT INTO sota_contracts (
                    contract_id, signal_id, book_id, book_content_id, account_id, account_mode,
                    pair, horizon_minutes, side, contract_family, is_migrated_legacy, cost_unknown,
                    buy_price_cents, payout_cents, sell_price_cents, profit_cents, stake_authoritative,
                    signal_close_utc, status, opened_utc, closed_utc, updated_utc
                ) VALUES (?,?,'b2','cid2','VRTC123456','demo','EURUSD',15,'UP','risefall',0,0,100,182,0,?,0,?,'cancelled',?,?,?)
                """,
                (f"c-ov-{k_idx}", f"s-ov-{k_idx}", big_loss, t_c.isoformat(), t_c.isoformat(), t_c.isoformat(), t_c.isoformat()),
            )
        assert check_risk_and_exposure(conn_v3, book_id="b3", pair="GBPUSD", horizon_minutes=15, side="UP", stake_usd=1.0, cfg=cfg_r, now_utc=t_c) == "daily_loss_circuit_breaker"
        conn_v3.close()

        # --request-stop-latch under unwritable 0500 log_dir MUST succeed & commit phase1_safety_latch!
        ro_logs = root / "ro_logs"
        ro_logs.mkdir(mode=0o500)
        try:
            lat = request_stop_latch(queue_db=q_db, account_id="VRTC123456", account_mode="demo", reason="operator_halt", log_dir=ro_logs, pw_home=pw_home)
            assert lat["reason"] == "operator_halt"
            c_chk = connect_sota_queue_v3(q_db, expected_account_id="VRTC123456", expected_account_mode="demo")
            assert is_safety_latch_set(c_chk) is not None
            c_chk.close()
        finally:
            os.chmod(ro_logs, 0o700)

        # --clear-stale-legacy-lock: clears dead PID, refuses live PID (current os.getpid())!
        from deriv_runtime_core_v2 import executor_lock_path, sota_account_lock_key
        lroot = root / "legacy_locks"
        lroot.mkdir(mode=0o700)
        lp = executor_lock_path(lroot, sota_account_lock_key("VRTC123456", "demo"))
        lp.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
        _expect_raises(ExecutorError, clear_stale_legacy_lock, account_id="VRTC123456", account_mode="demo", lock_root=lroot)
        lp.write_text(json.dumps({"pid": 9999999}), encoding="utf-8")
        clr = clear_stale_legacy_lock(account_id="VRTC123456", account_mode="demo", lock_root=lroot)
        assert clr["cleared"] is True and not lp.exists()

    # 2. ALL 6 source-derived predecessor shapes + B6 multiplier payout_cents=0->NULL & absent/null signal_close_utc backfill
    #    + uncheckpointed -wal coherent backup (0 bytes written to canonical!) + real `--adopt-legacy-account` command crash-recovery!
    def _build_predecessor_db(db_path: Path, shape_idx: int, *, leave_uncheckpointed_wal: bool = False) -> None:
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        neutral_chk = "('UP','DOWN','NEUTRAL')" if shape_idx >= 3 else "('UP','DOWN')"
        conn.execute(
            f"""
            CREATE TABLE sota_signals (
                signal_id TEXT PRIMARY KEY, book_id TEXT NOT NULL, pair TEXT NOT NULL,
                horizon_minutes INTEGER NOT NULL, side TEXT NOT NULL CHECK (side IN {neutral_chk}),
                lane TEXT NOT NULL DEFAULT 'wall', offset_id INTEGER NOT NULL DEFAULT 0,
                signal_close_utc TEXT NOT NULL, effective_floor REAL NOT NULL, calibrated_win_prob REAL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('queued','claimed','buy_intent','bought','contract_closed','terminal_skip','signal_expired','buy_blocked_risk','buy_unknown')),
                reason TEXT, created_utc TEXT NOT NULL, updated_utc TEXT NOT NULL
            )
            """
        )
        extra_cols = []
        if shape_idx >= 2:
            extra_cols.append("signal_close_utc TEXT")
        if shape_idx >= 3:
            extra_cols.append("contract_family TEXT NOT NULL DEFAULT 'risefall'")
        if shape_idx >= 4:
            extra_cols.append("sell_ack_cents INTEGER DEFAULT NULL")
        if shape_idx >= 5:
            extra_cols.append("account_mode TEXT NOT NULL DEFAULT 'demo'")
        if shape_idx >= 6:
            extra_cols.append("book_content_id TEXT DEFAULT NULL")
        extra_ddl = (",\n" + ",\n".join(extra_cols)) if extra_cols else ""
        conn.execute(
            f"""
            CREATE TABLE sota_contracts (
                contract_id TEXT PRIMARY KEY, signal_id TEXT NOT NULL UNIQUE REFERENCES sota_signals(signal_id),
                book_id TEXT NOT NULL, pair TEXT NOT NULL, horizon_minutes INTEGER NOT NULL,
                side TEXT NOT NULL CHECK (side IN {neutral_chk}), proposal_id TEXT NOT NULL,
                buy_price_cents INTEGER NOT NULL, payout_cents INTEGER NOT NULL,
                sell_price_cents INTEGER NOT NULL DEFAULT 0, profit_cents INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL CHECK (status IN ('open','won','lost','sold','expired','cancelled','unknown')),
                opened_utc TEXT NOT NULL, expected_expiry_utc TEXT NOT NULL, closed_utc TEXT, raw_hash TEXT NOT NULL
                {extra_ddl}
            )
            """
        )
        t_s = "2026-07-15T14:00:00+00:00"
        conn.execute("INSERT INTO sota_signals VALUES ('s1','USDJPY.sota_ens5_m30_up.v1','USDJPY',30,'UP','wall',0,?,0.58,0.60,'{}','contract_closed',NULL,?,?)", (t_s, t_s, t_s))
        if shape_idx >= 3:
            # Also include a legacy NEUTRAL expired signal (allowed in signals!) and a settled multiplier row (500/640/140, payout_cents=0 -> NULL!)
            conn.execute("INSERT INTO sota_signals VALUES ('s-neu','EURGBP.r.v1','EURGBP',1440,'NEUTRAL','wall',0,?,0.58,0.60,'{}','signal_expired',NULL,?,?)", (t_s, t_s, t_s))
            conn.execute("INSERT INTO sota_signals VALUES ('s-m','USDJPY.mult100.v1','USDJPY',30,'UP','wall',1,?,0.0,NULL,'{}','contract_closed',NULL,?,?)", (t_s, t_s, t_s))

        # Insert risefall contract (with NULL signal_close_utc on shape2 to test backfill!)
        base_vals = ["c1", "s1", "USDJPY.sota_ens5_m30_up.v1", "USDJPY", 30, "UP", "p1", 100, 182, 182, 82, "won", t_s, t_s, t_s, "h1"]
        if shape_idx >= 2:
            base_vals.append(None if shape_idx == 2 else t_s)
        if shape_idx >= 3:
            base_vals.append("risefall")
        if shape_idx >= 4:
            base_vals.append(None)
        if shape_idx >= 5:
            base_vals.append("demo")
        if shape_idx >= 6:
            base_vals.append("c8d0b13768a4c649")
        conn.execute(f"INSERT INTO sota_contracts VALUES ({','.join('?' * len(base_vals))})", tuple(base_vals))

        if shape_idx >= 3:
            # Settled multiplier row with payout_cents=0, buy=500, sell=640, profit=140, NULL signal_close_utc!
            m_vals: list[Any] = ["c-mult", "s-m", "USDJPY.mult100.v1", "USDJPY", 30, "UP", "pm", 500, 0, 640, 140, "won", t_s, t_s, t_s, "hm", None, "multiplier"]
            if shape_idx >= 4:
                m_vals.append(640)
            if shape_idx >= 5:
                m_vals.append("demo")
            if shape_idx >= 6:
                m_vals.append(None)
            conn.execute(f"INSERT INTO sota_contracts VALUES ({','.join('?' * len(m_vals))})", tuple(m_vals))

        if leave_uncheckpointed_wal:
            # MAJOR #5: Produce a REAL non-empty uncheckpointed committed WAL-only row ('c-wal') on disk with 0 open connections!
            import shutil as _sh
            src_tmp = db_path.parent / f".src_wal_{db_path.name}"
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.close()
            _sh.copy2(db_path, src_tmp)
            r_hold = sqlite3.connect(str(src_tmp), isolation_level=None)
            r_hold.execute("PRAGMA journal_mode=WAL")
            r_hold.execute("BEGIN")
            r_hold.execute("SELECT COUNT(*) FROM sota_contracts").fetchone()
            w_wal = sqlite3.connect(str(src_tmp), isolation_level=None)
            w_wal.execute("PRAGMA wal_autocheckpoint=0")
            w_wal.execute("INSERT INTO sota_signals VALUES ('s-wal','USDJPY.sota_ens5_m30_up.v1','USDJPY',30,'UP','wall',9,?,0.58,0.60,'{}','contract_closed',NULL,?,?)", (t_s, t_s, t_s))
            w_vals: list[Any] = ["c-wal", "s-wal", "USDJPY.sota_ens5_m30_up.v1", "USDJPY", 30, "UP", "pw", 100, 182, 182, 82, "won", t_s, t_s, t_s, "hw"]
            if shape_idx >= 2:
                w_vals.append(t_s)
            if shape_idx >= 3:
                w_vals.append("risefall")
            if shape_idx >= 4:
                w_vals.append(None)
            if shape_idx >= 5:
                w_vals.append("demo")
            if shape_idx >= 6:
                w_vals.append("c8d0b13768a4c649")
            w_wal.execute(f"INSERT INTO sota_contracts VALUES ({','.join('?' * len(w_vals))})", tuple(w_vals))
            # Snapshot exact db + non-empty -wal + -shm while r_hold keeps checkpoint blocked!
            for sfx in ("", "-wal", "-shm"):
                sp = Path(f"{src_tmp}{sfx}")
                dp = Path(f"{db_path}{sfx}")
                if sp.exists():
                    _sh.copy2(sp, dp)
            w_wal.close()
            r_hold.close()
            for sfx in ("", "-wal", "-shm"):
                Path(f"{src_tmp}{sfx}").unlink(missing_ok=True)
        else:
            conn.close()

    # Test ALL 6 shapes detect cleanly and adopt cleanly!
    for shape_i in range(1, 7):
        with tempfile.TemporaryDirectory() as td_s:
            root = Path(td_s)
            pw_h = root / "home"
            pw_h.mkdir()
            rt_dir = root / "repo" / "deriv_data" / "runtime"
            (rt_dir / "locks").mkdir(parents=True)
            idx_lock = rt_dir / "locks" / "sota_index.lock"
            idx_lock.write_text("preserve-me\n", encoding="utf-8")
            cdb = rt_dir / "sota_trade_queue_demo.sqlite"
            _build_predecessor_db(cdb, shape_i)
            ro_chk = ex_mod.open_sota_queue_readonly(cdb)
            got_shape = detect_predecessor_shape(ro_chk)
            ro_chk.close()
            assert got_shape.startswith(f"shape{shape_i}_"), f"wrong shape {got_shape} for {shape_i}"

            qdir = root / "new_a" / "new_b" / "quarantine"
            sdir = root / "new_c" / "new_d" / "staged"
            rcpt = _make_receipt(root / "rcpt.json")
            bak_info = prepare_coherent_predecessor_backup(cdb, qdir, durable_ancestor=root)
            preimg = export_legacy_ownership_preimage(qdir)
            assert preimg["backup_sha256"] == bak_info["backup_sha256"]
            proof_p = root / "proof.json"
            proof_p.write_text(
                json.dumps(
                    {
                        "backup_sha256": preimg["backup_sha256"],
                        "account_id": "VRTC123456",
                        "contract_ownership_records": [
                            {"contract_id": c["contract_id"], "account_id": "VRTC123456", "verified_against_venue": True}
                            for c in preimg["contracts"]
                        ],
                    }
                ),
                encoding="utf-8",
            )
            res_ad = adopt_legacy_account(
                canonical_db=cdb,
                quarantine_dir=qdir,
                staged_v3_dir=sdir,
                ownership_proof_path=proof_p,
                receipt_path=rcpt,
                account_id="VRTC123456",
                pw_home=pw_h,
                durable_ancestor=root,
            )
            assert res_ad["status"] == "adopted_v3_installed"
            assert idx_lock.read_text(encoding="utf-8") == "preserve-me\n"
            c_new = connect_sota_queue_v3(cdb, expected_account_id="VRTC123456", expected_account_mode="demo")
            c1 = c_new.execute("SELECT signal_close_utc, is_migrated_legacy FROM sota_contracts WHERE contract_id='c1'").fetchone()
            assert c1["signal_close_utc"] == "2026-07-15T14:00:00+00:00" and int(c1["is_migrated_legacy"]) == 1
            if shape_i >= 3:
                cm = c_new.execute("SELECT contract_family, buy_price_cents, payout_cents, sell_price_cents, profit_cents, signal_close_utc FROM sota_contracts WHERE contract_id='c-mult'").fetchone()
                assert cm["contract_family"] == "multiplier" and int(cm["buy_price_cents"]) == 500 and cm["payout_cents"] is None and int(cm["sell_price_cents"]) == 640 and int(cm["profit_cents"]) == 140 and cm["signal_close_utc"] == "2026-07-15T14:00:00+00:00"
            c_new.close()

    # 3. REAL `--adopt-legacy-account` command crash-recovery across EVERY crash point + newly created nested dirs fsync
    #    + single-file v3 dual-dir re-sync + unexpected sidecar hold without unlinking v3!
    for crash_pt in ("after_intent_before_unlink", "mid_canonical_removal", "after_sync_predecessor_removals", "after_rename_before_dir_fsync"):
        with tempfile.TemporaryDirectory() as td_cr:
            root = Path(td_cr)
            pw_h = root / "home"
            pw_h.mkdir()
            rt_dir = root / "repo" / "deriv_data" / "runtime"
            (rt_dir / "locks").mkdir(parents=True)
            idx_lock = rt_dir / "locks" / "sota_index.lock"
            idx_lock.write_text("lock-untouched\n", encoding="utf-8")
            cdb = rt_dir / "sota_trade_queue_demo.sqlite"
            _build_predecessor_db(cdb, 6, leave_uncheckpointed_wal=True)
            # Authentic non-empty uncheckpointed -wal containing committed row 'c-wal'!
            wal_p = Path(f"{cdb}-wal")
            shm_p = Path(f"{cdb}-shm")
            assert wal_p.is_file() and wal_p.stat().st_size > 0, "expected real non-empty uncheckpointed -wal file"
            pre_cdb_sha = hashlib.sha256(cdb.read_bytes()).hexdigest()
            pre_wal_sha = hashlib.sha256(wal_p.read_bytes()).hexdigest()

            qdir = root / "nested_q1" / "nested_q2" / "quarantine"
            sdir = root / "nested_s1" / "nested_s2" / "staged"
            rcpt = _make_receipt(root / "rcpt.json")
            bak_info = prepare_coherent_predecessor_backup(cdb, qdir, durable_ancestor=root)
            # Backup MUST NOT mutate canonical predecessor (.sqlite OR -wal)!
            assert hashlib.sha256(cdb.read_bytes()).hexdigest() == pre_cdb_sha
            assert hashlib.sha256(wal_p.read_bytes()).hexdigest() == pre_wal_sha
            preimg = export_legacy_ownership_preimage(qdir)
            # Assert real uncheckpointed WAL-only row 'c-wal' was captured in coherent backup!
            assert any(c["contract_id"] == "c-wal" for c in preimg["contracts"]), "coherent backup missed uncheckpointed WAL-only row 'c-wal'!"
            proof_p = root / "proof.json"
            proof_p.write_text(
                json.dumps(
                    {
                        "backup_sha256": preimg["backup_sha256"],
                        "account_id": "VRTC123456",
                        "contract_ownership_records": [
                            {"contract_id": c["contract_id"], "account_id": "VRTC123456", "verified_against_venue": True}
                            for c in preimg["contracts"]
                        ],
                    }
                ),
                encoding="utf-8",
            )

            class _SimCrash(RuntimeError):
                pass

            def _hook(pt: str) -> None:
                if pt == crash_pt:
                    raise _SimCrash(pt)

            # Trigger simulated crash in first pass
            try:
                adopt_legacy_account(
                    canonical_db=cdb,
                    quarantine_dir=qdir,
                    staged_v3_dir=sdir,
                    ownership_proof_path=proof_p,
                    receipt_path=rcpt,
                    account_id="VRTC123456",
                    pw_home=pw_h,
                    durable_ancestor=root,
                    crash_hook=_hook,
                )
                raise AssertionError(f"expected _SimCrash at {crash_pt}")
            except _SimCrash:
                pass

            # Now invoke REAL `deriv_sota_executor.main(["--adopt-legacy-account", ...])` to recover from install_intent.json!
            import sota_book_freeze as sbf_mod
            synced_dirs_r: list[str] = []
            orig_qpath = ex_mod.canonical_sota_queue_path
            orig_pwhome = ex_mod.canonical_pw_home
            orig_env = ex_mod.DerivEnv.from_env
            orig_ws_url = ex_mod.request_authenticated_ws_url
            orig_scli = ex_mod.SotaDerivAsyncClient
            orig_fd = ex_mod.fsync_dir
            orig_sbf_fd = sbf_mod.fsync_dir
            orig_proc = ex_mod._check_no_legacy_buyers_proc

            class _OkIdCli:
                def __init__(self, *a: Any, **kw: Any) -> None:
                    pass
                async def __aenter__(self) -> "_OkIdCli":
                    return self
                async def __aexit__(self, *a: Any) -> None:
                    return None
                async def establish_identity(self, **kw: Any) -> int:
                    return 1784124000

            try:
                ex_mod.canonical_sota_queue_path = lambda mode, repo_root=None: cdb  # type: ignore[assignment]
                ex_mod.canonical_pw_home = lambda **kw: pw_h  # type: ignore[assignment]
                ex_mod.DerivEnv.from_env = classmethod(lambda cls: type("E", (), {"account_id": "VRTC123456", "pat": "tok", "app_id": "1089"})())  # type: ignore[assignment]
                ex_mod.request_authenticated_ws_url = lambda env, mode: ("wss://api.derivws.com/trading/v1/options/ws/demo", "options_otp")  # type: ignore[assignment]
                ex_mod.SotaDerivAsyncClient = _OkIdCli  # type: ignore[assignment]
                ex_mod._check_no_legacy_buyers_proc = lambda: None  # type: ignore[assignment]
                def _spy_ex_fd(p: Path) -> None:
                    synced_dirs_r.append(str(p.resolve()))
                    orig_sbf_fd(p)
                ex_mod.fsync_dir = _spy_ex_fd  # type: ignore[assignment]
                sbf_mod.fsync_dir = _spy_ex_fd  # type: ignore[assignment]

                rc_ad = ex_mod.main(
                    [
                        "--adopt-legacy-account",
                        "--account-mode", "demo",
                        "--maintenance-quarantine-dir", str(qdir),
                        "--staged-v3-dir", str(sdir),
                        "--ownership-proof-json", str(proof_p),
                        "--credential-isolation-receipt", str(rcpt),
                    ]
                )
                assert rc_ad == 0
                # Assert BOTH rename directories synced on completion (and nested dirs synced on pre-install retry!)
                assert str(sdir.resolve()) in synced_dirs_r and str(rt_dir.resolve()) in synced_dirs_r
                if crash_pt != "after_rename_before_dir_fsync":
                    assert str((root / "nested_q1").resolve()) in synced_dirs_r
                    assert str((root / "nested_s1").resolve()) in synced_dirs_r
                assert idx_lock.read_text(encoding="utf-8") == "lock-untouched\n"
                assert hashlib.sha256((qdir / "predecessor_coherent_backup.sqlite").read_bytes()).hexdigest() == bak_info["backup_sha256"]

                # On after_rename_before_dir_fsync: also verify unexpected -wal sidecar HOLDS without unlinking installed v3!
                if crash_pt == "after_rename_before_dir_fsync":
                    v3_sha = hashlib.sha256(cdb.read_bytes()).hexdigest()
                    wal_p.write_bytes(b"rogue-wal")
                    _expect_raises(
                        ExecutorError,
                        ex_mod.main,
                        [
                            "--adopt-legacy-account",
                            "--account-mode", "demo",
                            "--maintenance-quarantine-dir", str(qdir),
                            "--staged-v3-dir", str(sdir),
                            "--ownership-proof-json", str(proof_p),
                            "--credential-isolation-receipt", str(rcpt),
                        ],
                    )
                    assert cdb.is_file() and hashlib.sha256(cdb.read_bytes()).hexdigest() == v3_sha
            finally:
                ex_mod.canonical_sota_queue_path = orig_qpath  # type: ignore[assignment]
                ex_mod.canonical_pw_home = orig_pwhome  # type: ignore[assignment]
                ex_mod.DerivEnv.from_env = orig_env  # type: ignore[assignment]
                ex_mod.request_authenticated_ws_url = orig_ws_url  # type: ignore[assignment]
                ex_mod.SotaDerivAsyncClient = orig_scli  # type: ignore[assignment]
                ex_mod.fsync_dir = orig_fd  # type: ignore[assignment]
                sbf_mod.fsync_dir = orig_sbf_fd  # type: ignore[assignment]
                ex_mod._check_no_legacy_buyers_proc = orig_proc  # type: ignore[assignment]


    # 4. BLOCKER #4 extra checks: unexpected extra table rejected by detect_predecessor_shape,
    #    and cross-FS staging (st_dev mismatch) refused BEFORE unlinking canonical DB!
    with tempfile.TemporaryDirectory() as td_ex4:
        root4 = Path(td_ex4)
        extra_db = root4 / "extra_tbl.sqlite"
        _build_predecessor_db(extra_db, 6)
        ec = sqlite3.connect(str(extra_db))
        ec.execute("CREATE TABLE rogue_table (id INTEGER PRIMARY KEY)")
        ec.commit()
        ec.close()
        ro_e = ex_mod.open_sota_queue_readonly(extra_db)
        try:
            _expect_raises(ExecutorError, detect_predecessor_shape, ro_e)
        finally:
            ro_e.close()

        # Cross-FS staging refusal before any unlink
        pw4 = root4 / "home"
        pw4.mkdir()
        rt4 = root4 / "repo" / "deriv_data" / "runtime"
        rt4.mkdir(parents=True)
        cdb4 = rt4 / "sota_trade_queue_demo.sqlite"
        _build_predecessor_db(cdb4, 6)
        qdir4 = root4 / "q4"
        sdir4 = root4 / "s4"
        rcpt4 = _make_receipt(root4 / "rcpt4.json")
        prepare_coherent_predecessor_backup(cdb4, qdir4, durable_ancestor=root4)
        preimg4 = export_legacy_ownership_preimage(qdir4)
        proof4 = root4 / "proof4.json"
        proof4.write_text(
            json.dumps(
                {
                    "backup_sha256": preimg4["backup_sha256"],
                    "account_id": "VRTC123456",
                    "contract_ownership_records": [
                        {"contract_id": c["contract_id"], "account_id": "VRTC123456", "verified_against_venue": True}
                        for c in preimg4["contracts"]
                    ],
                }
            ),
            encoding="utf-8",
        )
        orig_stat = ex_mod.os.stat
        sdir4_str = str(sdir4.resolve())

        def _fake_cross_dev_stat(path_s: Any, *a: Any, **kw: Any):
            st = orig_stat(path_s, *a, **kw)
            if isinstance(path_s, (str, Path)) and str(path_s) == sdir4_str:
                return type("_FakeStat", (), {"st_dev": st.st_dev + 999, "st_size": st.st_size, "st_mode": st.st_mode})()
            return st

        ex_mod.os.stat = _fake_cross_dev_stat  # type: ignore[assignment]
        try:
            _expect_raises(
                ExecutorError,
                adopt_legacy_account,
                canonical_db=cdb4,
                quarantine_dir=qdir4,
                staged_v3_dir=sdir4,
                ownership_proof_path=proof4,
                receipt_path=rcpt4,
                account_id="VRTC123456",
                pw_home=pw4,
                durable_ancestor=root4,
            )
            assert cdb4.is_file(), "canonical DB was unlinked despite cross-FS st_dev refusal!"
        finally:
            ex_mod.os.stat = orig_stat  # type: ignore[assignment]


def test_phase1_prospective_g6_and_preflight() -> None:
    import shutil
    import sqlite3
    from fractions import Fraction
    from deriv_live_pnl_ledger import exact_fraction_bootstrap_ci95
    from deriv_sota_executor import (
        SotaDerivAsyncClient,
        canonical_sota_lock_paths,
        connect_sota_queue_v3,
        create_v3_queue_schema_atomic,
        register_cohort_anchor,
    )
    import vps_preflight as vp_mod

    repo_real = Path(__file__).resolve().parents[1]
    ens30 = load_sota_book("USDJPY.sota_ens5_m30_up.v1", repo_root=repo_real, allow_demo_canary=True)

    # 1. Exact Fraction bootstrap (s[124] + 39*s[125]) / (40 * n * 100) + huge-integer (>2**61) overflow safety (MAJOR #4)!
    profits = [82] * 35 + [-100] * 15
    lo_f, hi_f = exact_fraction_bootstrap_ci95(profits, stake_cents=100, n_boot=5000, seed=7)
    rng_v = np.random.default_rng(7)
    arr_p = np.asarray(profits, dtype=np.int64)
    sums_v = sorted(int(arr_p[rng_v.integers(0, 50, 50)].sum()) for _ in range(5000))
    assert lo_f == Fraction(sums_v[124] + 39 * sums_v[125], 40 * 50 * 100)
    assert hi_f == Fraction(39 * sums_v[4874] + sums_v[4875], 40 * 50 * 100)
    assert lo_f > 0
    big_lo, big_hi = exact_fraction_bootstrap_ci95([2**61] * 10, stake_cents=100, n_boot=100, seed=1)
    assert big_lo == Fraction(2**61, 100) and big_hi == Fraction(2**61, 100)

    # 2. Prospective G6 cohort anchor, boundary-crossing exclusion, contract-spec & POC venue-field checks, canonical DB enforcement!
    with tempfile.TemporaryDirectory() as td_g6:
        root = Path(td_g6)
        pw_h = root / "home"
        pw_h.mkdir()
        rt_dir = root / "repo" / "deriv_data" / "runtime"
        rt_dir.mkdir(parents=True)
        q_db = rt_dir / "sota_trade_queue_demo.sqlite"
        raw_c = sqlite3.connect(str(q_db), isolation_level=None)
        raw_c.execute("PRAGMA foreign_keys=ON")
        raw_c.execute("PRAGMA journal_mode=WAL")
        create_v3_queue_schema_atomic(raw_c, account_id="VRTC123456", account_mode="demo")
        raw_c.close()
        rcpt = root / "rcpt.json"
        rcpt.write_text(
            json.dumps(
                {
                    "schema": "sota-credential-isolation-receipt/v1",
                    "account_id": "VRTC123456",
                    "account_mode": "demo",
                    "legacy_credentials_revoked": True,
                    "dedicated_sota_credentials_provisioned": True,
                    "attested_by": "owner",
                    "attested_utc": "2026-11-25T00:00:00Z",
                }
            ),
            encoding="utf-8",
        )
        v_reg = 1785000000
        coh_start_ep = v_reg + 3600
        coh_start_iso = datetime.fromtimestamp(coh_start_ep, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        # Past/equal start refused; future whole-second accepted; exact retry idempotent!
        _expect_raises(
            ExecutorError,
            register_cohort_anchor,
            queue_db=q_db, account_id="VRTC123456", account_mode="demo", book_id=ens30.book_id,
            cohort_start_utc=datetime.fromtimestamp(v_reg, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            venue_time_epoch=v_reg, receipt_path=rcpt, repo_root=repo_real, pw_home=pw_h,
        )
        ra1 = register_cohort_anchor(
            queue_db=q_db, account_id="VRTC123456", account_mode="demo", book_id=ens30.book_id,
            cohort_start_utc=coh_start_iso, venue_time_epoch=v_reg, receipt_path=rcpt, repo_root=repo_real, pw_home=pw_h,
        )
        assert ra1["status"] == "registered"
        ra2 = register_cohort_anchor(
            queue_db=q_db, account_id="VRTC123456", account_mode="demo", book_id=ens30.book_id,
            cohort_start_utc=coh_start_iso, venue_time_epoch=v_reg + 10, receipt_path=rcpt, repo_root=repo_real, pw_home=pw_h,
        )
        assert ra2["status"] == "idempotent_existing_anchor"

        conn = connect_sota_queue_v3(q_db, expected_account_id="VRTC123456", expected_account_mode="demo")
        # Insert 1 boundary-crossing contract (started BEFORE cohort_start_ep, closed AFTER!) -> MUST be excluded!
        bc_start = coh_start_ep - 600
        bc_close = datetime.fromtimestamp(coh_start_ep + 1200, tz=timezone.utc).isoformat()
        bc_sig = datetime.fromtimestamp(bc_start - 2, tz=timezone.utc).isoformat()
        conn.execute("INSERT INTO sota_signals VALUES ('sig-bc',?,'USDJPY',30,'UP','wall',0,?,0.58,0.60,'{}','bought',NULL,'c-bc',NULL,?,?)", (ens30.book_id, bc_sig, bc_sig, bc_sig))
        conn.execute(
            """
            INSERT INTO sota_contracts (
                contract_id, signal_id, book_id, book_content_id, account_id, account_mode,
                pair, horizon_minutes, side, contract_family, is_migrated_legacy, cost_unknown,
                buy_price_cents, payout_cents, sell_price_cents, profit_cents, stake_authoritative,
                venue_start_epoch, venue_end_epoch, venue_terminal_status, venue_underlying_symbol, venue_contract_type,
                signal_close_utc, status, opened_utc, closed_utc, updated_utc
            ) VALUES ('c-bc','sig-bc',?,?,'VRTC123456','demo','USDJPY',30,'UP','risefall',0,0,100,182,0,-100,1,?,?,'lost','frxUSDJPY','CALL',?,'lost',?,?,?)
            """,
            (ens30.book_id, ens30.content_id, bc_start, bc_start + 1800, bc_sig, bc_sig, bc_close, bc_close),
        )
        # Insert 52 clean prospective settled contracts (38 wins, 14 losses -> strong positive CI95!)
        for k in range(52):
            s_ep = coh_start_ep + k * 3600 + 2
            e_ep = s_ep + 1800
            sig_iso = datetime.fromtimestamp(s_ep - 2, tz=timezone.utc).isoformat()
            cl_iso = datetime.fromtimestamp(e_ep, tz=timezone.utc).isoformat()
            is_win = (k % 4) != 0
            st_k = "won" if is_win else "lost"
            sp_k = 182 if is_win else 0
            pf_k = 82 if is_win else -100
            conn.execute("INSERT INTO sota_signals VALUES (?,?,?,?,?,'wall',?, ?,0.58,0.60,'{}','bought',NULL,?,NULL,?,?)", (f"sg-{k}", ens30.book_id, "USDJPY", 30, "UP", k, sig_iso, f"cg-{k}", sig_iso, sig_iso))
            conn.execute(
                """
                INSERT INTO sota_contracts (
                    contract_id, signal_id, book_id, book_content_id, account_id, account_mode,
                    pair, horizon_minutes, side, contract_family, is_migrated_legacy, cost_unknown,
                    buy_price_cents, payout_cents, sell_price_cents, profit_cents, stake_authoritative,
                    venue_start_epoch, venue_end_epoch, venue_terminal_status, venue_underlying_symbol, venue_contract_type,
                    signal_close_utc, status, opened_utc, closed_utc, updated_utc
                ) VALUES (?,?,?,?,'VRTC123456','demo','USDJPY',30,'UP','risefall',0,0,100,182,?,?,1,?,?,?,'frxUSDJPY','CALL',?,?,?, ?,?)
                """,
                (f"cg-{k}", f"sg-{k}", ens30.book_id, ens30.content_id, sp_k, pf_k, s_ep, e_ep, st_k, sig_iso, st_k, sig_iso, cl_iso, cl_iso),
            )
        eval_iso = datetime.fromtimestamp(coh_start_ep + 60 * 3600, tz=timezone.utc).isoformat()

        rep_ok = summarize_settled_queue(
            q_db,
            min_settled_for_promotion=50,
            book_id=ens30.book_id,
            repo_root=repo_real,
            require_registered_book=True,
            cohort_start_utc=coh_start_iso,
            expected_account_id="VRTC123456",
            expected_stake_usd="1.00",
            eval_now_utc=eval_iso,
            canonical_db_path=q_db,
            enforce_canonical_db=True,
        )
        assert rep_ok["promotion_eligible"] is True and rep_ok["settled_contracts"] == 52 and rep_ok["by_book"][ens30.book_id]["n"] == 52
        assert rep_ok["by_book"][ens30.book_id]["excluded_pre_anchor"] == 1

        # Non-canonical diagnostic DB path MUST return promotion_eligible: False!
        rep_noncanon = summarize_settled_queue(
            q_db,
            min_settled_for_promotion=50,
            book_id=ens30.book_id,
            repo_root=repo_real,
            cohort_start_utc=coh_start_iso,
            expected_account_id="VRTC123456",
            expected_stake_usd="1.00",
            eval_now_utc=eval_iso,
            canonical_db_path=root / "other.sqlite",
            enforce_canonical_db=True,
        )
        assert rep_noncanon["promotion_eligible"] is False and "non_canonical_db_diagnostic_only" in rep_noncanon["reasons"]

        # Corrupt ONE contract's venue_contract_type ('PUT' != 'CALL') -> MUST refuse promotion!
        conn.execute("UPDATE sota_contracts SET venue_contract_type='PUT' WHERE contract_id='cg-0'")
        rep_bad_poc = summarize_settled_queue(
            q_db,
            min_settled_for_promotion=50,
            book_id=ens30.book_id,
            repo_root=repo_real,
            cohort_start_utc=coh_start_iso,
            expected_account_id="VRTC123456",
            expected_stake_usd="1.00",
            eval_now_utc=eval_iso,
            canonical_db_path=q_db,
            enforce_canonical_db=True,
        )
        assert rep_bad_poc["promotion_eligible"] is False
        assert any("venue_symbol_or_contract_type_mismatch" in r for r in rep_bad_poc["reasons"])
        conn.close()

        # BLOCKER #5: CLI --db --book-id WITHOUT cohort args MUST NOT promote retrospective data!
        import deriv_live_pnl_ledger as pnl_mod
        import io as _io
        from contextlib import redirect_stdout
        buf_cli = _io.StringIO()
        with redirect_stdout(buf_cli):
            rc_pnl = pnl_mod.main(["--db", str(q_db), "--book-id", ens30.book_id])
        assert rc_pnl == 0
        cli_rep = json.loads(buf_cli.getvalue())
        assert cli_rep["promotion_eligible"] is False

        # 3. vps_preflight.py negative & positive controls:
        #    - Mutually exclusive --offline-only + --live-smoke -> RC 2
        assert vp_mod.main(["--offline-only", "--live-smoke"]) == 2
        #    - Fixture flags refused on --live-smoke and on deriv_sota_executor.py!
        assert vp_mod.main(["--live-smoke", "--fixture-home", str(pw_h), "--fixture-repo-root", str(root / "repo")]) == 2
        import deriv_sota_executor as ex_mod
        _expect_raises(ExecutorError, ex_mod.main, ["--run", "--fixture-home", str(pw_h), "--fixture-repo-root", str(root / "repo")])
        #    - Fixture root pointing at real repo -> refused!
        #    - BLOCKER #1: Omitting --phase1-policy STILL enforces Phase 1 admission, AND empty --fixture-home "" is refused!
        import os
        orig_acct = os.environ.get("DERIV_ACCOUNT_ID")
        try:
            os.environ["DERIV_ACCOUNT_ID"] = "VRTC123456"
            assert vp_mod.main(["--offline-only", "--account-mode", "demo", "--book-ids", "USDJPY.sota_ens5_m30_up.v1"]) != 0
            assert vp_mod.main(
                [
                    "--offline-only", "--fixture-home", "", "--fixture-repo-root", str(root / "repo"),
                    "--allow-unprovisioned-queue", "--account-mode", "demo", "--book-ids", "USDJPY.sota_ens5_m30_up.v1",
                ]
            ) != 0
            rc_real_ref = vp_mod.main(
                [
                    "--offline-only", "--phase1-policy", "--require-explicit-book-ids",
                    "--fixture-home", str(pw_h), "--fixture-repo-root", str(repo_real),
                    "--account-mode", "demo", "--book-ids", "USDJPY.sota_ens5_m30_up.v1",
                    "--queue-db", str(repo_real / "deriv_data" / "runtime" / "sota_trade_queue_demo.sqlite"),
                    "--credential-isolation-receipt", str(rcpt),
                ]
            )
            assert rc_real_ref != 0

            # Valid disposable fixture root with copied books & bound v3 DB passes --offline-only;
            # sota_activation.hold present fails; missing DB with --allow-unprovisioned-queue passes!
            fix_repo = root / "repo"
            (fix_repo / "books").mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo_real / "books" / "SOTA_INDEX.json", fix_repo / "books" / "SOTA_INDEX.json")
            for b_id in ("USDJPY.sota_ens5_m30_up.v1", "USDJPY.sota_ens5_m15_up.v1"):
                shutil.copytree(repo_real / "books" / b_id, fix_repo / "books" / b_id, symlinks=False)
            base_argv = [
                "--offline-only", "--phase1-policy", "--require-explicit-book-ids",
                "--fixture-home", str(pw_h), "--fixture-repo-root", str(fix_repo),
                "--account-mode", "demo",
                "--book-ids", "USDJPY.sota_ens5_m30_up.v1,USDJPY.sota_ens5_m15_up.v1",
                "--stake", "1.0", "--max-stake-usd", "5.0", "--max-daily-loss-usd", "25.0", "--max-drawdown-usd", "50.0",
                "--max-open-contracts", "3", "--max-usd-factor-open", "3", "--max-jpy-factor-open", "3",
                "--absolute-breakeven-ceiling", "0.60", "--ev-margin", "0.005", "--max-buy-dispatch-delay-s", "4.0",
                "--queue-db", str(q_db), "--log-dir", str(root / "logs"),
                "--credential-isolation-receipt", str(rcpt),
            ]
            import contextlib as _ctx_m
            import io as _io_m
            buf_bound = _io_m.StringIO()
            with _ctx_m.redirect_stdout(buf_bound):
                assert vp_mod.main(base_argv) == 0
            out_bound = buf_bound.getvalue()
            assert "VRTC123456" not in out_bound and "account_id_sha256" in out_bound and "bound_account_sha256" in out_bound
            # sota_activation.hold present MUST refuse!
            locks_fx = canonical_sota_lock_paths(q_db, pw_home=pw_h)
            locks_fx["hold"].parent.mkdir(parents=True, exist_ok=True)
            locks_fx["hold"].write_text("hold\n", encoding="utf-8")
            assert vp_mod.main(base_argv) != 0
            locks_fx["hold"].unlink()
            # Missing DB with --allow-unprovisioned-queue passes inside validated fixture root!
            q_db.unlink()
            assert vp_mod.main(base_argv) != 0
            assert vp_mod.main(base_argv + ["--allow-unprovisioned-queue"]) == 0

            # End-to-end 3-book --offline-only preflight under single_host_units_masked + VRTC_FROM_ENV
            shutil.copytree(
                repo_real / "books" / "USDJPY.sota_stack05_m30_up.v1",
                fix_repo / "books" / "USDJPY.sota_stack05_m30_up.v1",
                symlinks=False,
            )
            masked_dir = pw_h / ".config" / "systemd" / "user"
            masked_dir.mkdir(parents=True, exist_ok=True)
            for u_name in ex_mod.LEGACY_MASKED_UNIT_NAMES:
                (masked_dir / u_name).symlink_to("/dev/null")
            rcpt_masked = root / "rcpt_masked_preflight.json"
            rcpt_masked.write_text(
                json.dumps({
                    "schema": "sota-credential-isolation-receipt/v1",
                    "credential_isolation_mode": ex_mod.CREDENTIAL_ISOLATION_MODE_SINGLE_HOST_MASKED,
                    "account_id": ex_mod.DEMO_ENV_ACCOUNT_SENTINEL,
                    "account_mode": "demo",
                    "legacy_credentials_revoked": False,
                    "legacy_units_masked": list(ex_mod.LEGACY_MASKED_UNIT_NAMES),
                    "attested_by": "owner_demo",
                    "attested_utc": "2026-11-26T18:22:00Z",
                }),
                encoding="utf-8",
            )
            masked_argv = [
                "--offline-only", "--phase1-policy", "--require-explicit-book-ids", "--enforce-3book-cutover",
                "--allow-unprovisioned-queue",
                "--fixture-home", str(pw_h), "--fixture-repo-root", str(fix_repo),
                "--account-mode", "demo",
                "--book-ids", "USDJPY.sota_ens5_m30_up.v1,USDJPY.sota_stack05_m30_up.v1,USDJPY.sota_ens5_m15_up.v1",
                "--stake", "1.0", "--max-stake-usd", "5.0", "--max-daily-loss-usd", "25.0", "--max-drawdown-usd", "50.0",
                "--max-open-contracts", "3", "--max-usd-factor-open", "3", "--max-jpy-factor-open", "3",
                "--absolute-breakeven-ceiling", "0.60", "--ev-margin", "0.005", "--max-buy-dispatch-delay-s", "4.0",
                "--queue-db", str(q_db), "--log-dir", str(root / "logs"),
                "--credential-isolation-receipt", str(rcpt_masked),
            ]
            import contextlib as _ctx_m
            import io as _io_m
            buf_masked = _io_m.StringIO()
            with _ctx_m.redirect_stdout(buf_masked):
                rc_m = vp_mod.main(masked_argv)
            assert rc_m == 0
            out_m = buf_masked.getvalue()
            assert "VRTC123456" not in out_m and "account_id_sha256" in out_m
            # Removing one /dev/null mask symlink MUST fail preflight immediately!
            (masked_dir / ex_mod.LEGACY_MASKED_UNIT_NAMES[0]).unlink()
            assert vp_mod.main(masked_argv) != 0
        finally:
            if orig_acct is None:
                os.environ.pop("DERIV_ACCOUNT_ID", None)
            else:
                os.environ["DERIV_ACCOUNT_ID"] = orig_acct

    # 4. --live-smoke negative test: valid-shaped m30 proposal at 1.7799999999 FAILS with 0 buy/sell frames + wall timeout!
    async def _run_smoke_1779_and_timeout() -> None:
        class _SmokeWS:
            def __init__(self, delay_s: float = 0.0) -> None:
                self.delay_s = delay_s
                self.inbox: asyncio.Queue[str] = asyncio.Queue()
                self.raw_sent: list[str] = []

            async def send(self, text: str) -> None:
                self.raw_sent.append(text)
                req = json.loads(text)
                if self.delay_s > 0 and "proposal" in req:
                    await asyncio.sleep(self.delay_s)
                rid = req["req_id"]
                if "balance" in req:
                    await self.inbox.put(json.dumps({"req_id": rid, "balance": {"loginid": "VRTC123456", "currency": "USD"}}))
                elif "time" in req:
                    await self.inbox.put(json.dumps({"req_id": rid, "time": 1784124000}))
                elif "proposal" in req:
                    # Emit exact literal JSON token 1.7799999999 (below 1.78!)
                    await self.inbox.put(f'{{"req_id": {rid}, "proposal": {{"id": "p-1779", "ask_price": 1.0, "payout": 1.7799999999, "currency": "USD"}}}}')

            async def recv(self) -> str:
                return await self.inbox.get()

            async def close(self) -> None:
                return None

        ws_1779 = _SmokeWS(delay_s=0.0)
        async with SotaDerivAsyncClient(
            "wss://api.derivws.com/trading/v1/options/ws/demo",
            allow_buy=False,
            expected_account_id="VRTC123456",
            ws_opener=lambda _u: ws_1779,
        ) as cli_sm:
            sm_res = await vp_mod.run_network_proposal_smoke(
                "demo", [ens30], 1.0, client_override=cli_sm, transport_kind_override="options_otp", pace_s=0.0
            )
            assert len(sm_res) == 1
            assert sm_res[0]["proposal_received"] is True
            assert sm_res[0]["gate_passed_at_floor"] is False
            assert sm_res[0]["smoke_passed"] is False
            assert sm_res[0]["gate_reason"] == "payout_below_exact_rational_floor"
            assert sm_res[0]["row_calibrated_evidence"] is False
            assert sm_res[0]["decision5_satisfied"] is False
            assert not any('"buy"' in s or '"sell"' in s for s in ws_1779.raw_sent)

        # MAJOR #3: Missing "currency" in proposal MUST be rejected!
        class _NoCcyWS(_SmokeWS):
            async def send(self, text: str) -> None:
                self.raw_sent.append(text)
                req = json.loads(text)
                rid = req["req_id"]
                if "balance" in req:
                    await self.inbox.put(json.dumps({"req_id": rid, "balance": {"loginid": "VRTC123456", "currency": "USD"}}))
                elif "time" in req:
                    await self.inbox.put(json.dumps({"req_id": rid, "time": 1784124000}))
                elif "proposal" in req:
                    await self.inbox.put(f'{{"req_id": {rid}, "proposal": {{"id": "p-noccy", "ask_price": 1.0, "payout": 1.82}}}}')

        ws_noccy = _NoCcyWS()
        async with SotaDerivAsyncClient(
            "wss://api.derivws.com/trading/v1/options/ws/demo",
            allow_buy=False,
            expected_account_id="VRTC123456",
            ws_opener=lambda _u: ws_noccy,
        ) as cli_nc:
            sm_nc = await vp_mod.run_network_proposal_smoke(
                "demo", [ens30], 1.0, client_override=cli_nc, transport_kind_override="options_otp", pace_s=0.0
            )
            assert sm_nc[0]["smoke_passed"] is False and sm_nc[0]["gate_reason"] == "missing_or_non_usd_proposal_currency"

        # Bounded wall timeout (--smoke-timeout-s) covering BOTH slow proposal AND slow OTP!
        ws_slow = _SmokeWS(delay_s=2.0)
        async with SotaDerivAsyncClient(
            "wss://api.derivws.com/trading/v1/options/ws/demo",
            allow_buy=False,
            expected_account_id="VRTC123456",
            ws_opener=lambda _u: ws_slow,
        ) as cli_sl:
            try:
                await vp_mod.run_network_proposal_smoke(
                    "demo", [ens30], 1.0, smoke_timeout_s=0.15, client_override=cli_sl, transport_kind_override="options_otp", pace_s=0.0
                )
                raise AssertionError("expected asyncio.TimeoutError on bounded smoke_timeout_s")
            except asyncio.TimeoutError:
                pass

        # OTP hang timeout check (verify outer wait_for bounds request_authenticated_ws_url!)
        orig_non_us = vp_mod.assert_non_us_deriv_host
        orig_req_ws2 = vp_mod.request_authenticated_ws_url
        orig_val2 = vp_mod.validate_account_mode
        import time as _tm
        try:
            vp_mod.assert_non_us_deriv_host = lambda *a, **kw: None  # type: ignore[assignment]
            vp_mod.validate_account_mode = lambda **kw: None  # type: ignore[assignment]
            def _slow_otp(env_o: Any, mode_o: str):
                _tm.sleep(0.5)
                return ("wss://api.derivws.com/trading/v1/options/ws/demo", "options_otp")
            vp_mod.request_authenticated_ws_url = _slow_otp  # type: ignore[assignment]
            from deriv_client import DerivEnv as _DE
            try:
                await vp_mod.run_network_proposal_smoke(
                    "demo", [ens30], 1.0, smoke_timeout_s=0.1, env_override=_DE(app_id="1089", pat="tok", account_id="VRTC123456"), pace_s=0.0
                )
                raise AssertionError("expected asyncio.TimeoutError on hung OTP")
            except asyncio.TimeoutError:
                pass
        finally:
            vp_mod.assert_non_us_deriv_host = orig_non_us  # type: ignore[assignment]
            vp_mod.request_authenticated_ws_url = orig_req_ws2  # type: ignore[assignment]
            vp_mod.validate_account_mode = orig_val2  # type: ignore[assignment]

    asyncio.run(_run_smoke_1779_and_timeout())


def test_phase1_portfolio_replay_fixtures_a_to_f() -> None:
    from replay_sota_portfolio import (
        PATH_YEARS,
        PAYOUT_POLICY,
        compute_pareto_flags,
        reduce_scenario_paths,
        simulate_path_events,
    )

    # Fixture A: initial_cash=2.0, cap=1, flat_1usd, R=1.82
    fa = simulate_path_events(
        [(0, 0, 1800, 0, 1.82), (0, 2, 900, 1, 1.82), (1800, 2, 900, 1, 1.82)],
        max_open_cap=1,
        sizing_family="flat",
        initial_cash=2.0,
        path_years=PATH_YEARS,
    )
    assert fa["trades"] == 2 and fa["wins"] == 1 and fa["skipped_cap"] == 1 and fa["skipped_cash"] == 0
    assert abs(fa["final_cash"] - 1.82) < 1e-12 and fa["ruined"] is False and abs(fa["path_max_dd"] - 0.5) < 1e-12
    assert abs(fa["path_log_growth"] - (-0.2911012029031342)) < 1e-12

    # Fixture B: 3rd event win=0 -> ruined
    fb = simulate_path_events(
        [(0, 0, 1800, 0, 1.82), (0, 2, 900, 1, 1.82), (1800, 2, 900, 0, 1.82)],
        max_open_cap=1,
        sizing_family="flat",
        initial_cash=2.0,
        path_years=PATH_YEARS,
    )
    assert fb["final_cash"] == 0.0 and fb["ruined"] is True and fb["path_max_dd"] == 1.0 and fb["path_log_growth"] is None
    rb = reduce_scenario_paths(["0_1"], [fb])
    assert rb["status"] == "ruined" and rb["any_ruined"] is True and rb["pareto_efficient"] is False

    # Fixture C: initial_cash=1.0, cap=2, flat_1usd -> cash debited at admission, 2nd event skipped_cash
    fc = simulate_path_events(
        [(0, 0, 1800, 0, 1.82), (60, 2, 900, 1, 1.82)],
        max_open_cap=2,
        sizing_family="flat",
        initial_cash=1.0,
        path_years=PATH_YEARS,
    )
    assert fc["trades"] == 1 and fc["skipped_cash"] == 1 and fc["final_cash"] == 0.0 and fc["ruined"] is True

    # Fixture D: empty intersection -> status="empty_intersection", any_ruined=False
    rd = reduce_scenario_paths([], [])
    assert rd["status"] == "empty_intersection" and rd["any_ruined"] is False and rd["n_paths"] == 0 and rd["pareto_efficient"] is False

    # Fixture E: direct (p10_g, p90_dd) Pareto + cross-multiplier dominance in "kelly" + cross-family isolation ("flat" vs "kelly")!
    def _sc(p10: float, dd90: float, fam: str = "kelly", mult: float | None = 0.20) -> dict[str, Any]:
        return {
            "status": "ok",
            "path_ids": ["0_1"],
            "payout_policy": [list(x) for x in PAYOUT_POLICY],
            "sizing_family": fam,
            "kelly_multiplier": mult,
            "scope": "certified_phase1",
            "p10_g": p10,
            "p90_dd": dd90,
        }

    fe_sc = [
        _sc(0.20, 0.10, "kelly", 0.15),
        _sc(0.15, 0.12, "kelly", 0.20),  # dominated by kelly 0.15 across multipliers!
        _sc(0.25, 0.15, "kelly", 0.25),
        _sc(0.20, 0.10, "kelly", 0.20),  # tie -> stays True!
        _sc(0.05, 0.50, "flat", None),   # isolated in "flat" -> NOT dominated by "kelly"!
    ]
    assert compute_pareto_flags(fe_sc) == [True, False, True, True, True]

    # Fixture F: 5-path aggregation rejecting mean-growth or worst-DD substitutes!
    s1_paths = [
        {"ruined": False, "path_log_growth": g, "path_max_dd": d}
        for g, d in zip([0.05, 0.50, 0.50, 0.50, 0.50], [0.10, 0.10, 0.10, 0.10, 0.50])
    ]
    s2_paths = [
        {"ruined": False, "path_log_growth": g, "path_max_dd": d}
        for g, d in zip([0.25, 0.30, 0.30, 0.30, 0.30], [0.35, 0.35, 0.35, 0.35, 0.35])
    ]
    pids5 = ["0_1", "0_2", "0_3", "0_4", "0_5"]
    rs1 = reduce_scenario_paths(pids5, s1_paths)
    rs2 = reduce_scenario_paths(pids5, s2_paths)
    assert abs(rs1["p10_g"] - 0.23) < 1e-12 and abs(rs1["mean_g"] - 0.41) < 1e-12 and abs(rs1["p90_dd"] - 0.34) < 1e-12 and abs(rs1["worst_path_max_dd"] - 0.50) < 1e-12
    assert abs(rs2["p10_g"] - 0.27) < 1e-12 and abs(rs2["mean_g"] - 0.29) < 1e-12 and abs(rs2["p90_dd"] - 0.35) < 1e-12 and abs(rs2["worst_path_max_dd"] - 0.35) < 1e-12
    for obj in (rs1, rs2):
        obj["path_ids"] = pids5
        obj["payout_policy"] = [list(x) for x in PAYOUT_POLICY]
        obj["sizing_family"] = "kelly"
        obj["scope"] = "certified_phase1"
    assert compute_pareto_flags([rs1, rs2]) == [True, True]


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
        test_phase1_stack05_runtime_and_freezer,
        test_phase1_certifier_reducers_and_verify_bundle,
        test_phase1_tick_warmup_and_standalone_consumer,
        test_phase1_sota_client_and_buy_intent_recovery,
        test_phase1_v3_queue_and_6shape_adoption,
        test_phase1_prospective_g6_and_preflight,
        test_phase1_portfolio_replay_fixtures_a_to_f,
    ]
    for fn in tests:
        fn()
        print(f"PASS: {fn.__name__}")
    print(f"ALL {len(tests)} SOTA RUNTIME SAFETY SUITES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
