#!/usr/bin/env python3
"""Focused synthetic gate for the ROS-3A Deriv economics ledger."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sqlite3
import stat
import sys
import tempfile
import threading
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import deriv_economics_ledger as frozen_v1
import deriv_economics_ledger_v2 as ledger
import evidence_store as evidence


class LedgerFixture:
    def __init__(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        (self.root / "evidence/objects").mkdir(parents=True)
        for name in (
            "deriv_economics_ledger.py", "deriv_economics_ledger_v2.py",
            "evidence_store.py", "manifest.py",
        ):
            shutil.copy2(Path(__file__).parent / name, self.root / "scripts" / name)
        self.db = self.root / ledger.DB_LOGICAL_PATH
        self.log = self.root / ledger.LOG_LOGICAL_PATH
        self.output = self.root / "deriv_data/venue_evidence"
        self.db.parent.mkdir(parents=True)
        self.log.parent.mkdir(parents=True)
        self._create_db()
        self.predecessor = frozen_v1.seal_acquisition(self.root)
        if self.predecessor.object_id != ledger.PREDECESSOR_SPEC_ID:
            raise AssertionError("synthetic predecessor identity differs")
        self.spec = ledger.seal_acquisition(self.root)

    def close(self) -> None:
        self.temp.cleanup()

    def _create_db(self) -> None:
        conn = sqlite3.connect(self.db)
        conn.executescript(
            """
            CREATE TABLE signals (
                signal_id TEXT PRIMARY KEY, pair TEXT, side TEXT,
                signal_close_utc TEXT
            );
            CREATE TABLE contracts (
                contract_id TEXT PRIMARY KEY, signal_id TEXT, contract_status TEXT,
                proposal_id TEXT, buy_price REAL, stake REAL, sell_price REAL,
                profit REAL, payout REAL, terminal_utc TEXT, created_utc TEXT,
                raw_hash TEXT
            );
            """
        )
        conn.commit()
        conn.close()

    def add_chain(
        self,
        suffix: str,
        *,
        created: str,
        won: bool,
        proposal_payout: float = 1.86,
        terminal_payout: float = 1.85,
    ) -> list[dict]:
        signal = f"signal-{suffix}"
        proposal = f"proposal-{suffix}"
        contract = f"contract-{suffix}"
        base_minute = 10 if suffix == "a" else 20
        close = f"2026-07-06T{base_minute:02d}:00:00+00:00"
        terminal = f"2026-07-06T{base_minute:02d}:04:00+00:00"
        sell = terminal_payout if won else 0.0
        profit = 0.85 if won else -1.0
        status = "won" if won else "lost"
        conn = sqlite3.connect(self.db)
        conn.execute("INSERT INTO signals VALUES (?,?,?,?)", (signal, "USDJPY", "UP", close))
        conn.execute(
            "INSERT INTO contracts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                contract, signal, status, proposal, 1.0, 1.0, sell, profit,
                terminal_payout, terminal, created, "0123456789abcdef",
            ),
        )
        conn.commit()
        conn.close()
        startup = self._event(
            "executor_startup",
            timestamp_utc=f"2026-07-06T{base_minute - 1:02d}:00:00+00:00",
        )
        connected = self._event(
            "exec_client_connected",
            timestamp_utc=f"2026-07-06T{base_minute - 1:02d}:30:00+00:00",
            authenticated=True,
            endpoint="wss://example.invalid/demo",
        )
        common = {"signal_id": signal, "pair": "USDJPY", "side": "UP", "signal_close_utc": close}
        selected = self._event(
            "proposal_selected",
            timestamp_utc=f"2026-07-06T{base_minute:02d}:01:00+00:00",
            proposal_id=proposal,
            ask=1.0,
            payout=proposal_payout,
            proposal_source="fresh",
            **common,
        )
        submitted = self._event(
            "buy_submitted",
            timestamp_utc=f"2026-07-06T{base_minute:02d}:02:00+00:00",
            proposal_id=proposal,
            **common,
        )
        confirmed = self._event(
            "buy_confirmed",
            timestamp_utc=f"2026-07-06T{base_minute:02d}:03:00+00:00",
            contract_id=contract,
            **common,
        )
        update = self._event(
            "contract_update",
            timestamp_utc=f"2026-07-06T{base_minute:02d}:05:00+00:00",
            signal_id=signal,
            pair="USDJPY",
            side="UP",
            contract_id=contract,
            contract_status=status,
            raw_hash="0123456789abcdef",
            terminal=True,
        )
        closed = self._event(
            "contract_closed",
            timestamp_utc=f"2026-07-06T{base_minute:02d}:06:00+00:00",
            signal_id=signal,
            pair="USDJPY",
            side="UP",
            contract_id=contract,
            terminal_status=status,
            raw_hash="0123456789abcdef",
            sell_price=sell,
            profit=profit,
        )
        return [startup, connected, selected, submitted, confirmed, update, closed]

    @staticmethod
    def _event(event: str, **overrides: object) -> dict:
        defaults = {
            "executor_startup": {
                "absolute_breakeven_ceiling": 0.99, "cutoff": "x", "event": event,
                "fixed_defaults": {}, "lock_path": "x", "lock_root": "x",
                "queue_db": "x", "timestamp_utc": "2026-07-06T09:00:00+00:00",
            },
            "exec_client_connected": {
                "authenticated": True, "endpoint": "wss://example.invalid/demo",
                "event": event, "timestamp_utc": "2026-07-06T09:01:00+00:00",
            },
            "proposal_selected": {
                "ask": 1.0, "bar_age_s": 0.1, "book_id": "book", "effective_floor": 0.5,
                "event": event, "is_latest_bar": True, "lane": "wall",
                "live_breakeven": 0.5, "offset_id": 0, "pair": "USDJPY",
                "payout": 1.86, "proposal_id": "proposal", "proposal_source": "fresh",
                "side": "UP", "signal_close_utc": "2026-07-06T10:00:00+00:00",
                "signal_id": "signal", "timestamp_utc": "2026-07-06T10:01:00+00:00",
                "worker_id": "worker",
            },
            "buy_submitted": {
                "bar_age_s": 0.1, "book_id": "book", "buy_send_utc": "misleading",
                "buy_transport": "async", "claim_to_buy_submitted_ms": 1.0,
                "effective_floor": 0.5, "event": event, "is_latest_bar": True,
                "lane": "wall", "offset_id": 0, "pair": "USDJPY",
                "proposal_id": "proposal", "side": "UP",
                "signal_close_utc": "2026-07-06T10:00:00+00:00",
                "signal_id": "signal", "timestamp_utc": "2026-07-06T10:02:00+00:00",
            },
            "buy_confirmed": {
                "bar_age_s": 0.1, "book_id": "book", "contract_id": "contract",
                "effective_floor": 0.5, "event": event, "is_latest_bar": True,
                "lane": "wall", "offset_id": 0, "pair": "USDJPY", "side": "UP",
                "signal_close_utc": "2026-07-06T10:00:00+00:00",
                "signal_id": "signal", "timestamp_utc": "2026-07-06T10:03:00+00:00",
            },
            "contract_update": {
                "contract_id": "contract", "contract_status": "won", "event": event,
                "pair": "USDJPY", "raw_hash": "0123456789abcdef", "side": "UP",
                "signal_id": "signal", "terminal": True,
                "timestamp_utc": "2026-07-06T10:05:00+00:00",
            },
            "contract_closed": {
                "contract_id": "contract", "event": event, "pair": "USDJPY",
                "profit": 0.85, "raw_hash": "0123456789abcdef", "sell_price": 1.85,
                "side": "UP", "signal_id": "signal", "terminal_status": "won",
                "timestamp_utc": "2026-07-06T10:06:00+00:00",
            },
        }[event]
        defaults.update(overrides)
        return defaults

    def write_log(self, rows: list[dict]) -> None:
        self.log.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def export(self, output: Path | None = None) -> dict:
        return ledger.export_packet(
            spec_id=self.spec.object_id,
            authority_root=self.root,
            db_path=self.db,
            log_path=self.log,
            output_dir=output or self.output,
        )


class DerivEconomicsLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = LedgerFixture()

    def tearDown(self) -> None:
        self.fx.close()

    def test_end_to_end_win_one_read_and_zero_read_resume(self) -> None:
        self.fx.write_log(self.fx.add_chain("a", created="2026-07-06T10:00:00+00:00", won=True))
        descriptor = self.fx.export()
        packet_path = self.fx.root / descriptor["relative_path"]
        self.assertEqual(stat.S_IMODE(packet_path.stat().st_mode), 0o600)
        source = evidence.ArtifactRef(
            path=descriptor["relative_path"], bytes=descriptor["bytes"], sha256=descriptor["sha256"]
        )
        normalized = ledger.reduce_packet(
            repo_root=self.fx.root, spec_id=self.fx.spec.object_id,
            source_ref=source, descriptor=descriptor,
        )
        self.assertEqual(normalized.kind, ledger.NORMALIZED_KIND)
        self.assertEqual(normalized.payload["evidence_class"], "VENUE_SETTLED")
        self.assertEqual(normalized.payload["derivation"]["realized_profit"], "0.85")
        self.assertEqual(normalized.payload["derivation"]["quoted_win_profit"], "0.86")
        self.assertEqual(len(evidence.verify_store(repo_root=self.fx.root)), 5)
        original = evidence.ArtifactRef.read_verified

        def no_packet_read(ref: evidence.ArtifactRef, *, repo_root: Path = evidence.REPO_ROOT) -> bytes:
            if ref.path == source.path:
                raise AssertionError("completed resume reread the supplied packet")
            return original(ref, repo_root=repo_root)

        with mock.patch.object(evidence.ArtifactRef, "read_verified", no_packet_read):
            resumed = ledger.reduce_packet(
                repo_root=self.fx.root, spec_id=self.fx.spec.object_id,
                source_ref=source, descriptor=descriptor,
            )
        self.assertEqual(resumed.object_id, normalized.object_id)

    def test_loss_and_distinct_payouts_reconcile(self) -> None:
        self.fx.write_log(
            self.fx.add_chain(
                "a", created="2026-07-06T10:00:00+00:00", won=False,
                proposal_payout=1.90, terminal_payout=1.85,
            )
        )
        descriptor = self.fx.export()
        source = evidence.ArtifactRef(
            descriptor["relative_path"], descriptor["bytes"], descriptor["sha256"]
        )
        normalized = ledger.reduce_packet(
            repo_root=self.fx.root, spec_id=self.fx.spec.object_id,
            source_ref=source, descriptor=descriptor,
        )
        self.assertEqual(normalized.payload["derivation"]["realized_profit"], "-1.00")
        observation = normalized.payload["observation"]
        self.assertEqual(observation["proposal"]["payout"], "1.90")
        self.assertEqual(observation["terminal"]["payout"], "1.85")

    def test_selection_is_outcome_independent_and_log_order_independent(self) -> None:
        rows = self.fx.add_chain("a", created="2026-07-06T10:00:00+00:00", won=False)
        rows += self.fx.add_chain("b", created="2026-07-06T20:00:00+00:00", won=True)
        random.Random(7).shuffle(rows)
        self.fx.write_log(rows)
        first = self.fx.export(self.fx.root / "first")
        packet = evidence.decode_canonical_json((self.fx.root / first["relative_path"]).read_bytes())
        self.assertEqual(packet["linkage"]["signal_id"], "signal-a")
        conn = sqlite3.connect(self.fx.db)
        conn.execute(
            "UPDATE contracts SET contract_status='won', sell_price=1.85, profit=.85 WHERE signal_id='signal-a'"
        )
        conn.commit()
        conn.close()
        for row in rows:
            if row.get("signal_id") == "signal-a" and row["event"] == "contract_update":
                row["contract_status"] = "won"
            if row.get("signal_id") == "signal-a" and row["event"] == "contract_closed":
                row.update(terminal_status="won", sell_price=1.85, profit=.85)
        random.Random(11).shuffle(rows)
        self.fx.write_log(rows)
        second = self.fx.export(self.fx.root / "second")
        changed = evidence.decode_canonical_json((self.fx.root / second["relative_path"]).read_bytes())
        self.assertEqual(changed["linkage"]["signal_id"], "signal-a")

    def test_authority_mismatch_blocks_before_database_open(self) -> None:
        with mock.patch.object(sqlite3, "connect") as connect:
            with self.assertRaises(ledger.DerivEconomicsError):
                ledger.export_packet(
                    spec_id="0" * 64, authority_root=self.fx.root, db_path=self.fx.db,
                    log_path=self.fx.log, output_dir=self.fx.output,
                )
            connect.assert_not_called()

    def test_bound_artifact_and_remote_path_mismatches_block_before_source(self) -> None:
        exporter = self.fx.root / ledger.EXPORTER_PATH
        exporter.write_bytes(exporter.read_bytes() + b"\n# drift\n")
        with mock.patch.object(sqlite3, "connect") as connect:
            with self.assertRaises(ledger.DerivEconomicsError):
                self.fx.export()
            connect.assert_not_called()
        shutil.copy2(
            Path(__file__).parent / "deriv_economics_ledger_v2.py", exporter
        )
        with mock.patch.object(sqlite3, "connect") as connect:
            with self.assertRaises(ledger.DerivEconomicsError):
                ledger.export_packet(
                    spec_id=self.fx.spec.object_id,
                    authority_root=self.fx.root,
                    db_path=self.fx.db,
                    log_path=self.fx.log,
                    output_dir=self.fx.output,
                    enforce_remote_paths=True,
                )
            connect.assert_not_called()

    def test_v1_is_frozen_and_v2_admits_only_observed_exact_variants(self) -> None:
        frozen_path = self.fx.root / frozen_v1.EXPORTER_PATH
        self.assertEqual(hashlib.sha256(frozen_path.read_bytes()).hexdigest(), ledger.FROZEN_V1_SHA256)
        base = self.fx.add_chain("a", created="2026-07-06T10:00:00+00:00", won=True)
        variant = self.fx.add_chain("b", created="2026-07-06T20:00:00+00:00", won=False)
        variant[0]["allocation"] = {"observed": True}
        variant[0]["reconciliation"] = {"observed": True}
        variant[2]["net_edge"] = 0.01
        self.fx.write_log(base + variant)
        descriptor = self.fx.export(self.fx.root / "variant-output")
        self.assertTrue((self.fx.root / descriptor["relative_path"]).exists())
        with self.assertRaises(frozen_v1.DerivEconomicsError):
            frozen_v1._decode_log(self.fx.log)
        variant[2]["third_variant"] = True
        self.fx.write_log(base + variant)
        with self.assertRaises(ledger.DerivEconomicsError):
            self.fx.export(self.fx.root / "third-variant-output")

    def test_unknown_url_duplicate_and_clock_conflicts_fail(self) -> None:
        cases = []
        base = self.fx.add_chain("a", created="2026-07-06T10:00:00+00:00", won=True)
        unknown = [dict(row) for row in base]
        unknown[2]["unexpected"] = 1
        cases.append(unknown)
        url = [dict(row) for row in base]
        url[2]["url"] = "wss://secret.invalid/?token=x"
        cases.append(url)
        duplicate = [dict(row) for row in base] + [dict(base[2])]
        cases.append(duplicate)
        clock = [dict(row) for row in base]
        clock[3]["timestamp_utc"] = "2026-07-06T10:00:30+00:00"
        cases.append(clock)
        for rows in cases:
            with self.subTest(case=len(rows)):
                self.fx.write_log(rows)
                with self.assertRaises(ledger.DerivEconomicsError):
                    self.fx.export(self.fx.root / f"bad-{cases.index(rows)}")

    def test_non_demo_open_early_sale_and_economic_mismatches_fail(self) -> None:
        rows = self.fx.add_chain("a", created="2026-07-06T10:00:00+00:00", won=True)
        rows[1]["authenticated"] = False
        self.fx.write_log(rows)
        with self.assertRaises(ledger.DerivEconomicsError):
            self.fx.export()

    def test_conflicting_database_and_log_terminal_money_fails(self) -> None:
        rows = self.fx.add_chain("a", created="2026-07-06T10:00:00+00:00", won=True)
        rows[-1]["sell_price"] = 9.99
        rows[-1]["profit"] = 9.99
        self.fx.write_log(rows)
        with self.assertRaises(ledger.DerivEconomicsError):
            self.fx.export()
        rows[1]["authenticated"] = True
        conn = sqlite3.connect(self.fx.db)
        conn.execute("UPDATE contracts SET buy_price=1.001 WHERE signal_id='signal-a'")
        conn.commit()
        conn.close()
        self.fx.write_log(rows)
        with self.assertRaises(ledger.DerivEconomicsError):
            self.fx.export()

    def test_packet_numeric_money_and_secret_fields_reject_after_spend(self) -> None:
        self.fx.write_log(self.fx.add_chain("a", created="2026-07-06T10:00:00+00:00", won=True))
        descriptor = self.fx.export()
        path = self.fx.root / descriptor["relative_path"]
        packet = evidence.decode_canonical_json(path.read_bytes())
        packet["proposal"]["ask"] = 1.0
        bad = evidence.canonical_json_bytes(packet)
        bad_path = self.fx.output / f"{hashlib.sha256(bad).hexdigest()}.json"
        bad_path.write_bytes(bad)
        os.chmod(bad_path, 0o600)
        ref = evidence.ArtifactRef(
            path=bad_path.relative_to(self.fx.root).as_posix(), bytes=len(bad),
            sha256=hashlib.sha256(bad).hexdigest(),
        )
        bad_descriptor = dict(descriptor, relative_path=ref.path, bytes=ref.bytes, sha256=ref.sha256)
        with self.assertRaises(ledger.DerivEconomicsError):
            ledger.reduce_packet(
                repo_root=self.fx.root, spec_id=self.fx.spec.object_id,
                source_ref=ref, descriptor=bad_descriptor,
            )
        groups = ledger._domain_objects(self.fx.root, self.fx.spec.object_id)
        self.assertEqual(len(groups[ledger.ACCESS_KIND]), 1)
        self.assertEqual(len(groups[ledger.NORMALIZED_KIND]), 0)

    def test_settlement_goldens_and_ties(self) -> None:
        self.assertTrue(ledger.settle_direction("CALL", Decimal("1"), Decimal("2")))
        self.assertFalse(ledger.settle_direction("CALL", Decimal("1"), Decimal("1")))
        self.assertTrue(ledger.settle_direction("PUT", Decimal("2"), Decimal("1")))
        self.assertFalse(ledger.settle_direction("PUT", Decimal("1"), Decimal("1")))
        cost = Decimal("1.00")
        sell = Decimal("1.00")
        self.assertEqual(sell - cost, Decimal("0.00"))
        with self.assertRaises(ledger.DerivEconomicsError):
            ledger.settle_direction("UNKNOWN", Decimal("1"), Decimal("2"))

    def test_evidence_classes_and_decimal_forms_fail_closed(self) -> None:
        for name in ledger.EVIDENCE_CLASSES[:-1]:
            self.assertEqual(
                ledger.evidence_class_result(name, realized_profit=None),
                {"evidence_class": name, "realized_profit": None},
            )
            with self.assertRaises(ledger.DerivEconomicsError):
                ledger.evidence_class_result(name, realized_profit="0.85")
        with self.assertRaises(ledger.DerivEconomicsError):
            ledger.evidence_class_result("VENUE_SETTLED", realized_profit=None)
        with self.assertRaises(ledger.DerivEconomicsError):
            ledger.evidence_class_result("FRAMEWORK", realized_profit=None)
        for bad in (1.0, "1e0", "1.000", "1.001", "NaN", "Infinity", "+1.00"):
            with self.subTest(value=bad):
                with self.assertRaises(ledger.DerivEconomicsError):
                    ledger._decimal(bad, "money")

    def test_packet_contains_only_sanitized_whitelist(self) -> None:
        self.fx.write_log(self.fx.add_chain("a", created="2026-07-06T10:00:00+00:00", won=True))
        descriptor = self.fx.export()
        packet = evidence.decode_canonical_json(
            (self.fx.root / descriptor["relative_path"]).read_bytes()
        )
        self.assertEqual(set(packet), ledger.PACKET_FIELDS)
        rendered = json.dumps(packet, sort_keys=True).lower()
        for forbidden in (
            "endpoint", "wss://", "buy_send_utc", "book_id", "effective_floor",
            "live_breakeven", "worker_id", "account_id", "authorization",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertFalse(packet["account"]["real_money"])

    def test_conflicting_packet_and_spent_ambiguity_fail_closed(self) -> None:
        self.fx.write_log(self.fx.add_chain("a", created="2026-07-06T10:00:00+00:00", won=True))
        descriptor = self.fx.export()
        packet_path = self.fx.root / descriptor["relative_path"]
        self.assertEqual(self.fx.export(), descriptor)
        conflict = self.fx.output / ("f" * 64 + ".json")
        conflict.write_text("{}", encoding="utf-8")
        os.chmod(conflict, 0o600)
        with self.assertRaises(ledger.DerivEconomicsError):
            self.fx.export()
        conflict.unlink()
        source = evidence.ArtifactRef(
            descriptor["relative_path"], descriptor["bytes"], descriptor["sha256"]
        )
        sealed = evidence.EvidenceEnvelope.create(
            kind=ledger.SEALED_KIND,
            payload={
                "schema": ledger.SEALED_KIND,
                "acquisition_spec_id": self.fx.spec.object_id,
                "source_artifact": source.as_dict(), "descriptor": descriptor,
            },
            dependencies=[self.fx.spec.object_id],
        )
        receipt = evidence.EvidenceEnvelope.create(
            kind=ledger.ACCESS_KIND,
            payload={
                "schema": ledger.ACCESS_KIND,
                "acquisition_spec_id": self.fx.spec.object_id,
                "sealed_source_id": sealed.object_id,
                "source_artifact": source.as_dict(),
                "access_state": ledger.ACCESS_STATE,
            },
            dependencies=[sealed.object_id],
        )
        evidence.publish(sealed, repo_root=self.fx.root)
        evidence.publish(receipt, repo_root=self.fx.root)
        with self.assertRaisesRegex(ledger.DerivEconomicsError, "BLOCKED"):
            ledger.reduce_packet(
                repo_root=self.fx.root, spec_id=self.fx.spec.object_id,
                source_ref=source, descriptor=descriptor,
            )
        self.assertTrue(packet_path.exists())

    def test_concurrent_duplicate_allows_only_one_packet_read(self) -> None:
        self.fx.write_log(self.fx.add_chain("a", created="2026-07-06T10:00:00+00:00", won=True))
        descriptor = self.fx.export()
        source = evidence.ArtifactRef(
            descriptor["relative_path"], descriptor["bytes"], descriptor["sha256"]
        )
        original = evidence.ArtifactRef.read_verified
        entered = threading.Event()
        release = threading.Event()
        reads = 0
        lock = threading.Lock()

        def controlled(ref: evidence.ArtifactRef, *, repo_root: Path = evidence.REPO_ROOT) -> bytes:
            nonlocal reads
            if ref.path == source.path:
                with lock:
                    reads += 1
                entered.set()
                if not release.wait(5):
                    raise AssertionError("concurrency test timed out")
            return original(ref, repo_root=repo_root)

        outcomes: list[object] = []

        def run() -> None:
            try:
                outcomes.append(
                    ledger.reduce_packet(
                        repo_root=self.fx.root, spec_id=self.fx.spec.object_id,
                        source_ref=source, descriptor=descriptor,
                    )
                )
            except Exception as exc:  # test records the fail-closed loser
                outcomes.append(exc)

        with mock.patch.object(evidence.ArtifactRef, "read_verified", controlled):
            first = threading.Thread(target=run)
            first.start()
            self.assertTrue(entered.wait(5))
            second = threading.Thread(target=run)
            second.start()
            second.join(5)
            self.assertFalse(second.is_alive())
            release.set()
            first.join(5)
        self.assertEqual(reads, 1)
        self.assertEqual(sum(isinstance(item, evidence.EvidenceEnvelope) for item in outcomes), 1)
        errors = [item for item in outcomes if isinstance(item, Exception)]
        self.assertEqual(len(errors), 1)
        self.assertIn("BLOCKED", str(errors[0]))

    def test_malformed_receipt_ancestry_cannot_hide_spent_state(self) -> None:
        self.fx.write_log(self.fx.add_chain("a", created="2026-07-06T10:00:00+00:00", won=True))
        descriptor = self.fx.export()
        source = evidence.ArtifactRef(
            descriptor["relative_path"], descriptor["bytes"], descriptor["sha256"]
        )
        sealed = evidence.EvidenceEnvelope.create(
            kind=ledger.SEALED_KIND,
            payload={
                "schema": ledger.SEALED_KIND,
                "acquisition_spec_id": self.fx.spec.object_id,
                "source_artifact": source.as_dict(), "descriptor": descriptor,
            },
            dependencies=[self.fx.spec.object_id],
        )
        malformed_receipt = evidence.EvidenceEnvelope.create(
            kind=ledger.ACCESS_KIND,
            payload={
                "schema": ledger.ACCESS_KIND,
                "acquisition_spec_id": "f" * 64,
                "sealed_source_id": sealed.object_id,
                "source_artifact": source.as_dict(),
                "access_state": ledger.ACCESS_STATE,
            },
            dependencies=[self.fx.spec.object_id],
        )
        evidence.publish(sealed, repo_root=self.fx.root)
        evidence.publish(malformed_receipt, repo_root=self.fx.root)
        original = evidence.ArtifactRef.read_verified

        def no_packet_read(ref: evidence.ArtifactRef, *, repo_root: Path = evidence.REPO_ROOT) -> bytes:
            if ref.path == source.path:
                raise AssertionError("malformed spent state permitted another source read")
            return original(ref, repo_root=repo_root)

        with mock.patch.object(evidence.ArtifactRef, "read_verified", no_packet_read):
            with self.assertRaises(ledger.DerivEconomicsError):
                ledger.reduce_packet(
                    repo_root=self.fx.root, spec_id=self.fx.spec.object_id,
                    source_ref=source, descriptor=descriptor,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
