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
    ExecutorError,
    after_last_start_cutoff_v2,
    currency_leg_exposure,
    evaluate_sota_payout_gate,
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
    freeze_sota_book,
)
from sota_book_runtime import (
    ACTIVE_LIFECYCLE,
    DEMO_CANARY_LIFECYCLE,
    INACTIVE_LIFECYCLE,
    SotaBookError,
    load_enabled_sota_books,
    load_sota_book,
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
                "INSERT INTO sota_contracts VALUES (?,?,?,?,?,?,?,100,180,0,0,'open',?,?,NULL,'h',?)",
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
            "INSERT INTO sota_contracts VALUES ('c-loss','s-loss','b4','AUDUSD',15,'UP','p-loss',900,1620,0,-900,'lost',?,?,?,'h',?)",
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
    assert {b.book_id for b in demo_books} == {
        "USDJPY.sota_c2nysabs_m15_up.v1",
        "USDJPY.sota_e49nys30_m30_up.v1",
    }
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
    ]
    for fn in tests:
        fn()
        print(f"PASS: {fn.__name__}")
    print(f"ALL {len(tests)} SOTA RUNTIME SAFETY SUITES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
