"""Seven-day public payout-quote survey (Deriv program, owner decisions 9 and 10 of 2026-11-17).

One PUBLIC connection per process, no login. Every API request this program sends leaves through one gate
(RequestGate, enforced at GatedSocket, the only thing the client sends on). The gate

  - lets through only the survey's own requests - a price request for CALL, PUT, CALLE or PUTE at the survey's
    one stake, with or without a subscription; a contract list; the trading times - and stops the run on
    anything else, so nothing here can buy or log in. (The WebSocket library's own ping, pong and close frames
    are transport control, not API requests; they do not pass the gate.);
  - lets a request leave no sooner than SEND_SPACING_S after the one before: 24 a minute at most, the pace and
    the rule of the tick backfill. Nothing is saved up and nothing is exempt. A process's first request leaves
    no sooner than SEND_SPACING_S after the process took the connection lock, and at its end a process keeps the
    lock until its next request could have left: the pace holds from one process, and one tool, to the next;
  - lets nothing leave once the survey is over. It is over at --end-utc by the wall clock, read at the moment a
    request would leave, so a machine that was suspended or a clock that was set forward cannot make the survey
    run on; and it is over at --end-boot, the same end as a reading of the boot clock, which goes on through a
    suspension and which nobody can set. That reading is taken once, before the first process of the survey is
    started (--print-end-boot), and every process of the survey is given it: neither a clock that was set back
    nor a new process can make the survey run longer than it was started for. It holds for one boot of the
    machine: after a restart of the machine a new reading is taken, from the wall clock as it then is;
  - lets nothing leave for RATE_LIMIT_REST_S after the venue has answered with a rate-limit rejection, whatever
    that rejection answers: a request that is still waiting, one that has timed out or that a stop has
    cancelled, or a stream. The rest begins as the rejection is read, so nothing can come between the two.

Requests are sent one at a time: the next leaves when the one before has its answer or has timed out.

What is recorded:

  stream lanes  15-minute CALL and PUT of every pair as price subscriptions, sampled every SAMPLE_S seconds into
                `quote_snapshot` rows by the code of `deriv_quote_workers.py --audit` (the method of the July 2026
                archive), so the two samples can be compared row for row. The sampler reads memory only and keeps
                its schedule whatever the network does. A row repeats the lane's last quote; `quote_age_ms` says
                how old that quote is;
  poll units    one (pair, minutes, family) at a time, its two sides one after the other, unsubscribed, at a
                stake of 1: Rise/Fall (CALL, PUT) and Rise/Fall Equal (CALLE, PUTE) at 15, 30, 60, 120 and 240
                minutes. One `quote_poll` row per side. The 15-minute Rise/Fall unit asks for a stream lane's own
                contract on purpose: it shows whether a polled quote equals a streamed one, and it is what keeps
                the lane (next paragraph);
  metadata      `contracts_for` of every pair and `trading_times`, at the end of the first cycle and then every
                META_S seconds.

In both kinds of row a quote with a number that is not finite is no quote (`non_finite_quote`): the shared
parser alone takes a payout of infinity for a price. A row is the record of an outcome: a request that was in
flight when a process was killed has none.

One connection, never repaired. A process connects once. It ends with exit code 1, for a supervisor to start a
new process, when

  - the connection cannot be opened or is lost;
  - a subscription request ends without a clear answer: the venue may then hold a stream that nothing here reads;
  - a lane is dead: it has had no update for STALE_S seconds although the poll that has just priced its contract
    shows a later tick than the lane's last update. (A lane in a market without ticks is quiet, not dead: the
    poll's quote is then of the same tick. When one of the two has no spot time, silence alone decides. An error
    frame on a lane's stream is counted and does not end the run by itself.);
  - UNANSWERED_MAX requests in a row get no answer.

The new process subscribes each lane when a poll has just priced its contract, as the first did, and never
before. No lane is ended or restarted on its own and no subscription is carried from one connection to the next,
so none can be left behind or owned twice.

Closed mode. A cycle without one priced side (the market is closed) switches to one unit every CLOSED_PROBE_S
seconds, going through every length and family in turn, until a side is priced again. A refusal is an answer:
it does not end the run.

Rows go to <log-dir>/YYYY-MM-DD.jsonl by the UTC date of the row; every line is JSON. Exit code: 0 when the
survey is over or on SIGTERM/SIGINT (the run ends within CLOSE_S seconds of either; the process then keeps the
connection lock until its next request could have left); 1 when the connection can no longer be trusted or after
an error, so that a supervisor can start a new process (--end-utc and --end-boot are fixed: that never extends
the survey); 2 for a command line that cannot work, an --end-boot of another boot of the machine included; 75
while another tool holds the connection lock.

One connection to Deriv at a time. CONNECTION_LOCK is the one lock that says which tool has it; the tick backfill
takes the same file. No option names another.

Usage rule of the sample (owner decision 9): payout levels and availability only. No candidate is scored on these
dates and nothing is selected or tuned on them.

Run from the repository root:
    ~/binary-algo-venv/bin/python scripts/deriv_quote_survey.py --smoke
    ~/binary-algo-venv/bin/python scripts/deriv_quote_survey.py --end-utc 2026-11-24T16:00:00Z --print-end-boot
    ~/binary-algo-venv/bin/python scripts/deriv_quote_survey.py --log-dir DIR --end-utc 2026-11-24T16:00:00Z
        --end-boot BOOT_ID:SECONDS          (what the line before printed)
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import math
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from deriv_async_client import DerivAsyncClient
from deriv_client import PUBLIC_WS_URL, DerivAPIError
from deriv_quote_workers import SIDES, QuoteWorkers, resolve_pairs, sample_rows
from deriv_runtime_core import HORIZON_MINUTES, parse_quote
from live_features import DERIV_SYMBOLS

TOOL = "deriv_quote_survey/0.7"
MINUTES = (15, 30, 60, 120, 240)
FAMILIES = {"risefall": ("CALL", "PUT"), "equal": ("CALLE", "PUTE")}
STREAM_STAKE = 1.0                  # the July archive's stake, and the stake of every poll
LANE_KIND = (HORIZON_MINUTES, "risefall")       # the poll unit (minutes, family) that asks for a lane's own contract
MAX_DAYS = 7.0                      # the owner approved seven days; a later --end-utc is refused
SEND_SPACING_S = 2.5                # between two requests: 24 a minute, the pace and the rule of the tick backfill
RATE_LIMIT_REST_S = 75.0            # nothing is sent for this long after a rate-limit rejection (the backfill's pause)
SAMPLE_S = 30.0                     # the July sampler's interval
CLOSED_PROBE_S = 60.0               # closed mode: one unit this often
META_S = 6 * 3600.0                 # contract lists and trading times this often
STALE_S = 300.0                     # a lane this long without an update, while a poll shows a later tick, is dead
UNANSWERED_MAX = 3                  # this many requests in a row without an answer: the connection is given up
CLOSE_S = 10.0                      # the longest the end of a run waits for the connection to close
EVENT_ROWS_PER_MINUTE = 30          # rows of one client or lane event in a minute; the rest is counted only
CLOCK_PAIR_S = 0.05                 # the wall clock is read between two readings of the boot clock this close
BOOT_ID = Path("/proc/sys/kernel/random/boot_id")              # the name of this boot of the machine
EXIT_USAGE, EXIT_BUSY = 2, 75
CONNECTION_LOCK = Path("/mnt/data/deriv/.connection.lock")     # says which one tool has a connection to Deriv
QUOTE_FIELDS = ("ask", "payout", "live_breakeven")

Unit = tuple[str, int, str]         # pair, minutes, family


def _named(value: Any) -> Any:
    """A copy of `value` in which a number that JSON cannot hold (inf, nan) is its name as text."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: _named(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_named(item) for item in value]
    return value


class DailyJsonl:
    """Rows in <log_dir>/YYYY-MM-DD.jsonl, the file chosen by the UTC date of each row (JsonlLogger keeps writing
    to the file of the day it was created on). Every line is JSON. A file that a killed process left without a
    final newline gets one before this process's first row, so that the torn row stays a line of its own, and a
    `survey_torn_line` row says so."""

    def __init__(self, log_dir: Path, utcnow: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._utcnow = utcnow
        self._seen: set[Path] = set()

    @staticmethod
    def _torn(path: Path) -> bool:
        try:
            with path.open("rb") as fh:
                fh.seek(-1, os.SEEK_END)
                return fh.read(1) != b"\n"
        except OSError:                 # no such file, or an empty one
            return False

    def write(self, event: str, *, at: datetime | None = None, **fields: Any) -> None:
        """One row. Its time is `at` when the caller has taken one, and the time of the writing otherwise."""
        ts = at or self._utcnow()
        path = self.log_dir / f"{ts.date().isoformat()}.jsonl"
        torn = path not in self._seen and self._torn(path)
        self._seen.add(path)
        rows = [{"event": event, "timestamp_utc": ts.isoformat(), **fields}]
        if torn:
            rows.insert(0, {"event": "survey_torn_line", "timestamp_utc": ts.isoformat(), "file": path.name})
        text = "\n" if torn else ""
        for row in rows:
            try:
                text += json.dumps(row, sort_keys=True, default=str, allow_nan=False) + "\n"
            except ValueError:          # inf or nan in what the venue sent: kept as text, the line stays JSON
                text += json.dumps(_named(row), sort_keys=True, default=str, allow_nan=False) + "\n"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(text)


class EventRows:
    """The events of the client and of the lanes as rows, at most EVENT_ROWS_PER_MINUTE of one name in a minute:
    a venue that repeats one error every second must not fill the disk. `counts` has every event, written or
    not; the cycle rows carry it."""

    def __init__(self, logger: DailyJsonl) -> None:
        self._logger = logger
        self.counts: dict[str, int] = {}
        self._minute: dict[str, tuple[float, int]] = {}

    def __call__(self, event: str, fields: dict[str, Any]) -> None:
        self.counts[event] = self.counts.get(event, 0) + 1
        now = time.monotonic()
        started, written = self._minute.get(event, (now, 0))
        if now - started >= 60.0:
            started, written = now, 0
        self._minute[event] = (started, written + 1)
        if written < EVENT_ROWS_PER_MINUTE:
            self._logger.write(event, **fields)


# ---------------------------------------------------------------- the gate every request leaves through


class ForbiddenRequest(RuntimeError):
    """A request the survey may not send. Not a DerivAPIError on purpose: nothing catches it, so the run stops."""


class SurveyOver(Exception):
    """The survey is over: nothing more is sent or sampled. Not a failure: the run ends with exit code 0."""


class ConnectionSuspect(RuntimeError):
    """The one connection can no longer be trusted. The run ends with exit code 1; a supervisor starts a new
    process, which opens a new connection and subscribes its lanes afresh."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


ALLOWED_REQUESTS = {
    "proposal": frozenset({"proposal", "amount", "basis", "contract_type", "currency", "duration", "duration_unit",
                           "underlying_symbol", "subscribe", "req_id"}),
    "contracts_for": frozenset({"contracts_for", "req_id"}),
    "trading_times": frozenset({"trading_times", "req_id"}),
}
CONTRACT_TYPES = frozenset(ct for types in FAMILIES.values() for ct in types)


def own_contract(frame: dict[str, Any]) -> bool:
    """A price request for one of the survey's four contract types at its one stake."""
    amount = frame.get("amount")
    return (frame.get("contract_type") in CONTRACT_TYPES and frame.get("basis") == "stake"
            and type(amount) in (int, float) and amount == STREAM_STAKE)


def request_kind(frame: Any) -> str:
    """The kind of a request the survey may send: a price request for one of its four contract types at its one
    stake (with or without a subscription), a contract list or the trading times. Anything else - a purchase, a
    login, the end of a subscription, any other call, any other field, any other stake - raises
    ForbiddenRequest."""
    if isinstance(frame, dict):
        kinds = [kind for kind in ALLOWED_REQUESTS if kind in frame]
        if len(kinds) == 1 and set(frame) <= ALLOWED_REQUESTS[kinds[0]] and (
                kinds[0] != "proposal" or own_contract(frame)):
            return kinds[0]
        raise ForbiddenRequest(f"not a survey request: fields {sorted(map(str, frame))}")
    raise ForbiddenRequest(f"not a survey request: {type(frame).__name__}")


def rate_limit_text(text: str) -> bool:
    """The venue says it is asked too often (the tick downloader's test of the same thing)."""
    text = text.lower()
    return "ratelimit" in text or "rate limit" in text or "too many" in text


def boot_seconds() -> float:
    """Seconds on the boot clock: it goes on while the machine is suspended, and nobody can set it."""
    return time.clock_gettime(time.CLOCK_BOOTTIME)


def boot_id() -> str:
    """The name of this boot of the machine. The boot clock starts anew with the next one."""
    return BOOT_ID.read_text().strip()


def boot_deadline(end_utc: datetime, utcnow: Callable[[], datetime],
                  elapsed: Callable[[], float] = boot_seconds) -> float:
    """end_utc as a reading of the `elapsed` clock: where that clock stands now, plus the time that the wall
    clock says is left. The wall clock is read between two readings of the other; when those are more than
    CLOCK_PAIR_S apart - the process stood still between them, or the machine slept - all three are read again.
    The earlier reading counts, so the deadline is at most CLOCK_PAIR_S early and never late."""
    for _ in range(100):
        before = elapsed()
        left = (end_utc - utcnow()).total_seconds()
        if elapsed() - before <= CLOCK_PAIR_S:
            return before + left
    raise RuntimeError("the wall clock and the boot clock could not be read together in 100 tries")


def end_boot_text(end_utc: datetime, utcnow: Callable[[], datetime],
                  elapsed: Callable[[], float] = boot_seconds) -> str:
    """--end-boot for end_utc: `<boot id>:<seconds>`, the seconds of boot_deadline() with three decimals, cut
    and never rounded up: the deadline that a process reads from this text is not later than that one."""
    seconds = boot_deadline(end_utc, utcnow, elapsed)
    cut = math.floor(seconds * 1000.0)
    while cut / 1000.0 > seconds:
        cut -= 1
    return f"{boot_id()}:{cut / 1000.0:.3f}"


class RequestGate:
    """The pace of the one connection and the end of the survey, at the point where a request leaves.

    A request waits for its turn (turn()): no sooner than `spacing` seconds after the request before it left -
    strict, as in the tick downloader: nothing is saved up - not during a rest and, for the first request, no
    sooner than `spacing` after the gate was made. A process makes its gate when it has the connection lock, so
    its first request is that far from the last request of whatever held the lock before it. As a request
    leaves (leave(), which the socket calls) it gives the turn back. Without a turn it is refused
    (ForbiddenRequest), and once the survey is over it does not leave either (SurveyOver). One turn is one
    request: requests go one at a time, and a request that ends without leaving gives its turn back.

    The survey is over (over()) at end_utc by the wall clock, or when the `elapsed` clock reads end_boot,
    whichever comes first. Both ends are given to the gate: it pairs no clocks itself, so every process of one
    survey is held to the same two."""

    def __init__(self, end_utc: datetime, utcnow: Callable[[], datetime], end_boot: float,
                 spacing: float = SEND_SPACING_S, elapsed: Callable[[], float] = boot_seconds) -> None:
        self.end_utc, self.utcnow, self.end_boot = end_utc, utcnow, end_boot
        self._spacing, self._elapsed = spacing, elapsed
        self._not_before = time.monotonic() + spacing
        self._turn = False
        self.sent = 0                           # requests that left
        self.sent_at = time.monotonic()         # when the last one left
        self.least_gap: float | None = None     # the smallest time between two requests that left

    def over(self, now: datetime | None = None) -> str | None:
        """Why the survey is over, or None while it is not; by `now`, a reading of the wall clock, when given."""
        if (now or self.utcnow()) >= self.end_utc:
            return "end_utc"
        # A billionth of a second is a rounding error here too (see turn()).
        return "elapsed_limit" if self.end_boot - self._elapsed() <= 1e-9 else None

    def time_of_row(self) -> datetime | None:
        """The time for a row that holds what the venue said, or None when the survey is over at that time. It
        is taken when the venue's words have been read, and the row is written with it and with no other time:
        no such row is of a time at or after the end, however long the writing then takes."""
        now = self.utcnow()
        return None if self.over(now) else now

    def left(self) -> float:
        """Seconds until the survey is over, by the clock that says less."""
        return min((self.end_utc - self.utcnow()).total_seconds(), self.end_boot - self._elapsed())

    def wait(self) -> float:
        """Seconds until a request could leave."""
        return self._not_before - time.monotonic()

    async def turn(self) -> None:
        # A billionth of a second is a rounding error, not a wait: it could be too short for the clock.
        while (wait := self.wait()) > 1e-9:
            await asyncio.sleep(wait)
        self._turn = True

    def leave(self) -> None:
        if not self._turn:
            raise ForbiddenRequest("a request without its turn")
        self._turn = False
        if self.over():
            raise SurveyOver
        now = time.monotonic()
        if self.sent:
            gap = now - self.sent_at
            self.least_gap = gap if self.least_gap is None else min(self.least_gap, gap)
        self.sent_at, self._not_before = now, now + self._spacing
        self.sent += 1

    def give_back(self) -> None:
        """The turn of a request that ended without leaving is not kept for another."""
        self._turn = False

    def rest(self, seconds: float) -> None:
        """Nothing leaves for `seconds` from now."""
        self._not_before = max(self._not_before, time.monotonic() + seconds)


class GatedSocket:
    """What the client sends on. A request is checked against the allowlist and must have its turn at the gate
    before it leaves; `left`, when there is one, is then told the request. Everything else (recv, close) is the
    socket's own."""

    def __init__(self, ws: Any, gate: RequestGate, left: Callable[[dict[str, Any]], None] | None = None) -> None:
        self._ws, self._gate, self._left = ws, gate, left

    async def send(self, text: str) -> None:
        frame = json.loads(text)
        request_kind(frame)
        self._gate.leave()
        if self._left is not None:
            self._left(frame)
        await self._ws.send(text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ws, name)


class SurveyClient(DerivAsyncClient):
    """DerivAsyncClient on the public endpoint, sending through one GatedSocket, with one connection for its
    whole life: a lost connection is not repaired here (reconnect_max_attempts=0); the client is then dead.

    Every request waits for its turn at the gate before the client takes it up, so a request that is still
    waiting when the connection is lost never leaves. `unanswered` counts the requests in a row that ended
    without an answer from the venue; an answer of any kind, a refusal included, sets it back to 0. `unowned`
    is set when a subscription request left and ended without a clear answer: the venue may then hold a stream
    that nothing here reads. A rate-limit rejection rests the gate as it is read, whatever it answers.
    `late_answers` counts the answers to unsubscribed requests that came when their request was no longer
    waiting (it had timed out, a stop had cancelled it, or it had been answered before); they are not kept,
    however old the request. The client knows such an answer by the id of its request: one of the numbers it
    has given out, and not that of a subscription request. The ids of the subscription requests that left are
    all it remembers."""

    def __init__(self, gate: RequestGate, on_event: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        super().__init__(PUBLIC_WS_URL, reconnect_max_attempts=0, on_event=on_event)
        self.gate = gate
        self.unanswered = 0
        self.unowned: str | None = None
        self.rate_limited = 0
        self.late_answers = 0
        self._streams: set[Any] = set()         # ids of the subscription requests that left

    async def _dial(self) -> Any:
        return await super()._open()

    async def _open(self, *, refresh_url: bool = False) -> Any:
        return GatedSocket(await self._dial(), self.gate, self._request_left)

    def _request_left(self, frame: dict[str, Any]) -> None:
        """A request has left. The id of one that asks for a subscription is remembered: the frames of that
        subscription carry it for as long as the connection lives."""
        if "subscribe" in frame:
            self._streams.add(frame.get("req_id"))

    def _single_answer(self, rid: Any) -> bool:
        """`rid` is the id of a request of this client that asked for one answer: the shared client numbers its
        requests 1, 2, 3 and so on, and the subscription requests among them are remembered."""
        return type(rid) is int and 0 < rid <= self._req_id and rid not in self._streams

    def _rest_for_rate_limit(self) -> None:
        self.rate_limited += 1
        self.gate.rest(RATE_LIMIT_REST_S)

    async def request(self, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        if self.is_connected():         # otherwise the client refuses the request itself, and nothing leaves
            await self.gate.turn()
        try:
            response = await super().request(payload, timeout)
        except DerivAPIError as exc:
            self.unanswered = self.unanswered + 1 if exc.code is None else 0
            raise
        finally:
            self.gate.give_back()
        self.unanswered = 0
        return response

    def _dispatch(self, frame: dict[str, Any]) -> None:
        """Every frame, before the shared client sorts it. A rate-limit rejection rests the gate here, as it is
        read, whether a request is waiting for it or not: a stop that cancels the waiting request cannot come
        between the rejection and the rest. The answer to an unsubscribed request that is no longer waiting for
        it is counted and dropped, however old the request: the shared client would keep it, for a subscription
        that is not coming."""
        rid = frame.get("req_id")
        if "error" in frame and rate_limit_text(str(frame["error"])):
            self._rest_for_rate_limit()
        if self._single_answer(rid):
            waiting = self._pending.get(rid)
            if waiting is None or waiting.done():
                self.late_answers += 1
                return
        super()._dispatch(frame)

    async def subscribe(self, payload: dict[str, Any], queue_maxsize: int | None = None) -> Any:
        before = self.gate.sent
        try:
            return await super().subscribe(payload, queue_maxsize)
        except DerivAPIError as exc:
            if exc.code is None and self.gate.sent > before:
                self.unowned = f"{payload.get('underlying_symbol')} {payload.get('contract_type')}: {exc}"[:300]
            raise

    def trusted(self) -> None:
        """Raise ConnectionSuspect when this connection can no longer be trusted."""
        dead = self.connection_state()["dead"]
        if dead:
            raise ConnectionSuspect("connection_lost", str(dead)[:300])
        if self.unowned:
            raise ConnectionSuspect("subscription_unanswered", self.unowned)
        if self.unanswered >= UNANSWERED_MAX:
            raise ConnectionSuspect("venue_unanswering", f"{self.unanswered} requests in a row without an answer")


# ---------------------------------------------------------------- polls, metadata, lanes


def poll_units(pairs: list[str]) -> list[Unit]:
    """Polling order: per length and family, all pairs in a row; the lanes' own contracts come first."""
    return [(pair, minutes, family) for minutes in MINUTES for family in FAMILIES for pair in pairs]


def closed_probes(pairs: list[str]) -> list[Unit]:
    """Probing order of closed mode: every length and family in turn, the pair moving on by one each time, so
    that ten probes in a row cover the ten kinds of contract and ten times len(pairs) cover the grid."""
    kinds = [(minutes, family) for minutes in MINUTES for family in FAMILIES]
    return [(pairs[(i + turn) % len(pairs)], minutes, family)
            for turn in range(len(pairs)) for i, (minutes, family) in enumerate(kinds)]


def proposal_payload(pair: str, minutes: int, contract_type: str) -> dict[str, Any]:
    return {"proposal": 1, "amount": STREAM_STAKE, "basis": "stake", "contract_type": contract_type,
            "currency": "USD", "duration": minutes, "duration_unit": "m", "underlying_symbol": DERIV_SYMBOLS[pair]}


def finite_quote(row: dict[str, Any]) -> dict[str, Any]:
    """`row` as it is, unless its ask, payout or breakeven is a number that is not finite: then it has no quote."""
    if all(row[field] is None or math.isfinite(row[field]) for field in QUOTE_FIELDS):
        return row
    return {**row, **dict.fromkeys(QUOTE_FIELDS), "invalid_reason": row["invalid_reason"] or "non_finite_quote"}


def poll_row(unit: Unit, contract_type: str, seq: int, result: Any, latency_ms: float | None) -> dict[str, Any]:
    """One `quote_poll` row; ask, payout and live_breakeven by the audit sampler's parse_quote, failing closed.
    `result` is the venue's answer, the DerivAPIError the request ended with (its code is None when there was no
    answer), or None for a request that had left when the run was ended."""
    pair, minutes, family = unit
    row = {"pair": pair, "minutes": minutes, "family": family, "stake": STREAM_STAKE, "contract_type": contract_type,
           "seq": seq, "latency_ms": latency_ms, "source": "quote_survey", **dict.fromkeys(QUOTE_FIELDS)}
    if result is None:
        return {**row, "invalid_reason": "run_ended"}
    if isinstance(result, DerivAPIError):
        return {**row, "invalid_reason": "request_failed", "error_code": result.code, "error": str(result)[:300]}
    prop = result.get("proposal") if isinstance(result, dict) else None
    if not isinstance(prop, dict):
        return {**row, "invalid_reason": "no_proposal_in_response"}
    ask, payout, live_breakeven, invalid_reason = parse_quote(prop)
    return finite_quote({
        **row, "ask": ask, "payout": payout, "live_breakeven": live_breakeven, "invalid_reason": invalid_reason,
        "spot": prop.get("spot"), "spot_time": prop.get("spot_time"), "date_start": prop.get("date_start"),
        "date_expiry": prop.get("date_expiry"), "proposal_id_present": bool(prop.get("id"))})


async def poll_side(client: SurveyClient, unit: Unit, contract_type: str, seq: int,
                    logger: DailyJsonl) -> dict[str, Any]:
    """Ask for one side of a unit, log its row and return it. `latency_ms` is the time from the request leaving
    to the end of the request (its answer, or its timeout); it is null for a request that never left. A request
    that had left when the run was ended gets a row as well (`run_ended`); one that had not left gets none. An
    answer that comes when the survey is over is not a sample: its row is `run_ended` too. The row of an answer
    carries the time at which the answer was taken up (the gate's time_of_row())."""
    pair, minutes, _family = unit
    gate = client.gate
    before = gate.sent

    def latency_ms() -> float | None:
        return round((time.monotonic() - gate.sent_at) * 1000.0, 1) if gate.sent > before else None

    try:
        result: Any = await client.request(proposal_payload(pair, minutes, contract_type))
    except DerivAPIError as exc:
        result = exc
    except asyncio.CancelledError:
        if gate.sent > before:
            logger.write("quote_poll", **poll_row(unit, contract_type, seq, None, latency_ms()))
        raise
    at = gate.time_of_row()
    row = poll_row(unit, contract_type, seq, result if at else None, latency_ms())
    logger.write("quote_poll", at=at, **row)
    return row


async def fetch_metadata(client: SurveyClient, pairs: list[str], logger: DailyJsonl) -> None:
    """The contract list of every pair and the day's trading times, each stored as the venue answered - the
    list, or its refusal - with the time at which that was taken up (the gate's time_of_row()). A request that
    had left when the run was ended, or whose outcome - an answer, a refusal, a timeout - came when the survey
    was over, has a row that says so (`run_ended`) and holds nothing of the venue's."""
    gate = client.gate
    day = gate.utcnow().date().isoformat()
    requests = [("contracts_for", {"contracts_for": DERIV_SYMBOLS[pair]}, {"pair": pair}) for pair in pairs]
    requests.append(("trading_times", {"trading_times": day}, {"day": day}))
    for event, payload, fields in requests:
        client.trusted()
        before = gate.sent
        try:
            response = await client.request(payload)
        except DerivAPIError as exc:
            outcome: dict[str, Any] = {"ok": False, "error_code": exc.code, "error": str(exc)[:300]}
        except asyncio.CancelledError:
            if gate.sent > before:
                logger.write(event, **fields, ok=False, error_code=None, error="run_ended")
            raise
        else:
            outcome = {"ok": True, "response": response.get(event)}
        at = gate.time_of_row()
        if at is None:
            outcome = {"ok": False, "error_code": None, "error": "run_ended"}
        logger.write(event, at=at, **fields, **outcome)


def lane_behind(lane_spot: Any, poll_spot: Any) -> bool:
    """The poll has seen a tick that the lane has not shown: the spot time of its quote is later than that of
    the lane's last update. True as well when that cannot be told, because one of the two has no spot time."""
    known = all(type(t) in (int, float) and math.isfinite(t) for t in (lane_spot, poll_spot))
    return not known or poll_spot > lane_spot


async def keep_lanes(client: SurveyClient, workers: QuoteWorkers, pair: str, priced: dict[str, Any]) -> None:
    """A poll has just priced the 15-minute contract of this pair on the sides in `priced` (side: the spot time
    of that quote). A side without a lane is subscribed; a refusal is tried again at the next such poll. A lane
    is dead when it has had no update for STALE_S although the poll's quote is of a later tick than the lane's
    last one. (In the commissioning runs a lane had an update about once a second, with the ticks; a lane in a
    market without ticks is quiet, and as old as the poll's quote.) A subscription request without a clear
    answer, or a dead lane, ends the run: the next process starts every lane afresh on a new connection."""
    for side in SIDES:
        if side not in priced:
            continue
        if not workers.has_lane(pair, side):
            await workers.ensure_lane(pair, side)
            client.trusted()
            continue
        rec = workers.state.get(pair, side)
        if rec is None:
            raise ConnectionSuspect("lanes_silent", f"{pair} {side}: subscribed, and never had a quote")
        if rec["age_s"] >= STALE_S and lane_behind(rec["spot_time"], priced[side]):
            raise ConnectionSuspect("lanes_silent", f"{pair} {side}: no update for {rec['age_s']} s; spot time of "
                                    f"its last update {rec['spot_time']}, of the poll's quote {priced[side]}")


def lane_count(workers: QuoteWorkers, pairs: list[str]) -> int:
    return sum(workers.has_lane(pair, side) for pair in pairs for side in SIDES)


def counters(client: SurveyClient) -> dict[str, Any]:
    """What the gate and the client have counted in this process, for the state rows and the last row.
    `least_gap_seconds` is the smallest time between two requests, measured where they leave."""
    gate = client.gate
    return {"requests_sent": gate.sent,
            "least_gap_seconds": None if gate.least_gap is None else round(gate.least_gap, 3),
            "rate_limited": client.rate_limited, "late_answers": client.late_answers,
            "pre_sub_buffered": client.stats["pre_sub_buffered"]}


async def poll_loop(client: SurveyClient, workers: QuoteWorkers, pairs: list[str], logger: DailyJsonl,
                    events: EventRows) -> None:
    """Poll one unit after the other, keep the lanes and fetch the metadata, until cancelled, until the end
    (SurveyOver) or until the connection can no longer be trusted (ConnectionSuspect)."""
    units, probes = poll_units(pairs), closed_probes(pairs)
    seq = index = probe_index = priced_in_cycle = 0
    closed = False
    not_before = next_meta = cycle_started = time.monotonic()
    while True:
        unit = probes[probe_index % len(probes)] if closed else units[index]
        await asyncio.sleep(max(0.0, not_before - time.monotonic()))
        started = time.monotonic()
        seq += 1
        priced: dict[str, Any] = {}             # the sides that are priced: the spot time of each one's quote
        for contract_type in FAMILIES[unit[2]]:
            client.trusted()
            row = await poll_side(client, unit, contract_type, seq, logger)
            if row["invalid_reason"] is None:
                priced[contract_type] = row.get("spot_time")
        client.trusted()
        if priced and unit[1:] == LANE_KIND:
            await keep_lanes(client, workers, unit[0], priced)
        state = {"seq": seq, "lanes": lane_count(workers, pairs), "stream_updates": workers.updates,
                 "stream_errors": workers.errors, **counters(client), "events": dict(events.counts)}
        between_cycles = closed
        if closed:
            probe_index += 1
            if priced:
                closed, index, priced_in_cycle, cycle_started = False, 0, 0, time.monotonic()
                logger.write("survey_open", **state)
        else:
            priced_in_cycle += len(priced)
            index += 1
            if index == len(units):
                between_cycles = True
                logger.write("survey_cycle", units=len(units), priced_sides=priced_in_cycle,
                             seconds=round(time.monotonic() - cycle_started, 1), **state)
                if priced_in_cycle == 0:
                    closed = True
                    logger.write("survey_closed", **state)
                index, priced_in_cycle, cycle_started = 0, 0, time.monotonic()
        if between_cycles and time.monotonic() >= next_meta:
            await fetch_metadata(client, pairs, logger)
            next_meta = time.monotonic() + META_S
        not_before = started + CLOSED_PROBE_S if closed else started


async def sample_loop(workers: QuoteWorkers, pairs: list[str], logger: DailyJsonl,
                      time_of_row: Callable[[], datetime | None]) -> None:
    """One quote_snapshot row per lane every SAMPLE_S, on a fixed schedule, and none once the survey is over.
    Every row gets its time from `time_of_row` (the gate's) when the lanes have been read, and is written with
    that time; when there is none, the survey is over. So no row is of a time at or after the end, also when a
    row is slow to write or the process stands still in the middle of a sample. It reads memory only, so
    nothing the network does can delay it."""
    due = time.monotonic() + SAMPLE_S
    while True:
        await asyncio.sleep(max(0.0, due - time.monotonic()))
        for row in sample_rows(workers, pairs):
            at = time_of_row()
            if at is None:
                raise SurveyOver
            logger.write("quote_snapshot", at=at, **finite_quote(row))
        due += SAMPLE_S
        while due <= time.monotonic():      # after a stall (a stopped process) the missed samples are not made up
            due += SAMPLE_S


async def until_over(gate: RequestGate) -> None:
    """Return when the survey is over. Looks at the clocks at least every 30 s, and not more often than every
    thousandth of a second."""
    while not gate.over():
        await asyncio.sleep(min(max(gate.left(), 0.001), 30.0))


async def run_until_done(client: SurveyClient, workers: QuoteWorkers, pairs: list[str], logger: DailyJsonl,
                         events: EventRows, stop: asyncio.Event) -> tuple[int, str]:
    """Run the sampler and the polls until the survey is over, a stop request or a failure, whichever is first;
    whatever is in flight then is cancelled. Returns the exit code and the reason."""
    gate = client.gate

    async def polls() -> None:
        try:
            await client.connect()
        except Exception as exc:        # whatever keeps the one connection from opening: the next process tries
            raise ConnectionSuspect("connect_failed", f"{type(exc).__name__}: {exc}"[:300]) from exc
        await poll_loop(client, workers, pairs, logger, events)

    work = {"polls": asyncio.create_task(polls()),
            "sampler": asyncio.create_task(sample_loop(workers, pairs, logger, gate.time_of_row))}
    ends = {"stop_requested": asyncio.create_task(stop.wait()), "over": asyncio.create_task(until_over(gate))}
    try:
        await asyncio.wait([*work.values(), *ends.values()], return_when=asyncio.FIRST_COMPLETED)
        for name, task in work.items():
            if task.done():
                exc = task.exception()
                if isinstance(exc, SurveyOver):
                    return 0, gate.over() or "end_utc"
                reason = exc.reason if isinstance(exc, ConnectionSuspect) else "task_failed"
                logger.write("survey_failed", task=name, reason=reason, error=f"{type(exc).__name__}: {exc}"[:300])
                return 1, reason
        return 0, "stop_requested" if ends["stop_requested"].done() else gate.over() or "end_utc"
    finally:
        for task in (*work.values(), *ends.values()):
            task.cancel()
        await asyncio.gather(*work.values(), *ends.values(), return_exceptions=True)


async def shut_down(client: SurveyClient, workers: QuoteWorkers, logger: DailyJsonl) -> None:
    """Close the connection, waiting at most CLOSE_S, then end the lanes. A closed connection has no
    subscriptions left at the venue, and a closed client sends nothing: the end of a run sends nothing."""
    try:
        await asyncio.wait_for(client.close(), CLOSE_S)
    except asyncio.TimeoutError:
        logger.write("survey_close_timeout", seconds=CLOSE_S)
    await workers.stop()


def code_checksums() -> dict[str, str]:
    """sha256 of this file and of every loaded module that lives beside it: the project code that is running."""
    here = Path(__file__).resolve().parent
    files = {Path(f).resolve() for m in list(sys.modules.values()) if (f := getattr(m, "__file__", None))}
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files) if p.parent == here}


async def survey(logger: DailyJsonl, pairs: list[str], gate: RequestGate, stop: asyncio.Event,
                 client_of: Callable[..., SurveyClient], note: str = "") -> int:
    """One run on one connection: the start row, the work until its end, the close, the stop row; then the wait
    until this run's next request could have left, so that whoever takes the connection lock next keeps the
    pace (and a rest) without knowing of this run. `lock_kept_seconds` in the stop row is that wait as it stood
    when the row was made: what the run is about to do, not a record of it. Returns the exit code."""
    events = EventRows(logger)
    client = client_of(gate, lambda e, f: events("client_" + e, f))
    workers = QuoteWorkers(client, pairs, STREAM_STAKE, on_event=events)
    logger.write("survey_started", tool=TOOL, pid=os.getpid(), pairs=pairs, minutes=list(MINUTES),
                 families=FAMILIES, units=len(poll_units(pairs)), end_utc=gate.end_utc.isoformat(),
                 seconds_left=round(gate.left(), 1), boot_id=boot_id(), end_boot_seconds=round(gate.end_boot, 3),
                 send_spacing_seconds=SEND_SPACING_S,
                 rate_limit_rest_seconds=RATE_LIMIT_REST_S, sample_seconds=SAMPLE_S,
                 closed_probe_seconds=CLOSED_PROBE_S, meta_seconds=META_S, stale_seconds=STALE_S,
                 unanswered_max=UNANSWERED_MAX, url=client.url, note=note, code_sha256=code_checksums())
    code, reason = 1, "error"
    try:
        code, reason = await run_until_done(client, workers, pairs, logger, events, stop)
    finally:
        lanes = lane_count(workers, pairs)
        await shut_down(client, workers, logger)
        keep = max(0.0, gate.wait()) if gate.sent else 0.0
        logger.write("survey_stopped", exit_code=code, reason=reason, lanes=lanes, stream_updates=workers.updates,
                     stream_errors=workers.errors, **counters(client), overflow=client.stats["queue_overflow"],
                     unmatched=client.stats["unmatched_frames"], lock_kept_seconds=round(keep, 1),
                     events=dict(events.counts))
        if gate.sent:                   # what is left of that wait now that the row is written, and no more
            await asyncio.sleep(max(0.0, gate.wait()))
    return code


async def run_survey(args: argparse.Namespace, client_of: Callable[..., SurveyClient] = SurveyClient) -> int:
    """The survey of this process. The gate is made first: this process has just taken the connection lock. Its
    two ends are the command line's: no process of a survey makes an end of its own."""
    gate = RequestGate(args.end_utc, lambda: datetime.now(timezone.utc), args.end_boot)
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, stop.set)
    return await survey(DailyJsonl(Path(args.log_dir)), args.pairs, gate, stop, client_of, args.note)


def take_connection_lock(path: Path) -> int | None:
    """The lock that says which one tool has a connection to Deriv (the tick backfill takes the same file). It is
    held until this process ends, however it ends. Returns its descriptor, or None while another tool holds it."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def parse_end_utc(text: str, now: datetime) -> datetime:
    """--end-utc as an aware UTC time, at most MAX_DAYS ahead."""
    try:
        end = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"--end-utc {text!r} is not an ISO time") from None
    if end.tzinfo is None:
        raise ValueError("--end-utc needs a UTC offset, for example 2026-11-24T16:00:00Z")
    end = end.astimezone(timezone.utc)
    if end - now > timedelta(days=MAX_DAYS):
        raise ValueError(f"--end-utc is more than {MAX_DAYS:g} days ahead")
    return end


def parse_end_boot(text: str, boot: str, now: float) -> float:
    """--end-boot, `<boot id>:<seconds>` as --print-end-boot gives it, as a reading of the boot clock: of this
    boot of the machine (`boot`), and at most MAX_DAYS after `now`, where that clock stands."""
    name, _, seconds = text.rpartition(":")
    try:
        end = float(seconds)
    except ValueError:
        end = math.nan
    if not name or not math.isfinite(end):
        raise ValueError(f"--end-boot {text!r} is not <boot id>:<seconds>")
    if name != boot:
        raise ValueError("--end-boot is of another boot of this machine, whose clock is gone: take a new one with "
                         "--print-end-boot")
    if end - now > MAX_DAYS * 86400.0:
        raise ValueError(f"--end-boot is more than {MAX_DAYS:g} days ahead")
    return end


# ---------------------------------------------------------------- smoke (deterministic, no network)


_VIRTUAL_NOW = [0.0]        # the clock of _run_virtual. A check moves it forward to imitate a process that stood still


def _run_virtual(main: Callable[[], Any], unread: list[str]) -> Any:
    """Run main() on an event loop whose clock jumps to the next timer instead of waiting for it. time.monotonic
    is that clock while it runs, so the client, the lanes and the survey all read the same time. Whatever the
    loop itself has to report - a failure nobody read, an error in a callback - is added to `unread`."""
    import selectors

    now = _VIRTUAL_NOW
    now[0] = 1000.0

    def report(loop: Any, context: dict[str, Any]) -> None:
        unread.append(f"{context.get('message')}: {context.get('exception')!r}")

    class Jump(selectors.DefaultSelector):
        def select(self, timeout: float | None = None) -> list[Any]:
            if timeout is None:
                raise RuntimeError("virtual clock: every task is waiting and no timer is set")
            ready = super().select(0)
            if timeout > 0 and not ready:
                now[0] += timeout
            return ready

    real = time.monotonic
    time.monotonic = lambda: now[0]
    try:
        with asyncio.Runner(loop_factory=lambda: asyncio.SelectorEventLoop(Jump())) as runner:
            runner.get_loop().set_exception_handler(report)
            return runner.run(main())
    finally:
        time.monotonic = real


class _Venue:
    """An in-memory Deriv: one connection at a time, a record of every request that reached it with its time,
    and answers by script. What a request asks for is done when it arrives - a subscription exists from then on -
    whether or not its answer ever comes back."""

    def __init__(self) -> None:
        self.frames: list[tuple[float, dict[str, Any]]] = []
        self.opens: list[float] = []
        self.priced: Callable[[dict[str, Any]], bool] = lambda frame: True
        self.refuse: Callable[[dict[str, Any]], str | None] = lambda frame: None    # the error code to answer with
        self.mute: Callable[[dict[str, Any]], bool] = lambda frame: False           # the answer never comes back
        self.delay: Callable[[dict[str, Any]], float] = lambda frame: 0.0           # seconds until it comes back
        self.quote: Callable[[dict[str, Any]], dict[str, Any]] = lambda frame: {
            "ask_price": frame["amount"], "payout": round(frame["amount"] * 1.8, 2)}
        self.lane_quote: dict[str, Any] = {"ask_price": 1.0, "payout": 1.85}       # what a lane's update carries
        self.close_hangs = False
        self.down = False                       # a connection attempt fails
        self.silent = False                     # subscriptions get no updates
        self.wall = 0.0                         # seconds the wall clock is ahead of the process's own clock
        self.spot_time: Callable[[], Any] | None = None     # the time of the venue's last tick, when it tells one
        self.client: Any = None                 # the client of the last run against this venue
        self.stop: Callable[[], Any] = lambda: None         # asks the run against this venue to stop
        self.socket: _Socket | None = None
        self.subs: dict[str, _Socket] = {}
        self.ended: set[str] = set()            # subscriptions the venue has ended
        self._n = 0

    async def dial(self) -> "_Socket":
        self.opens.append(time.monotonic())
        if self.down:
            raise OSError("unreachable")
        self.socket = _Socket(self)
        return self.socket

    def live_subs(self) -> list[str]:
        return [sid for sid, socket in self.subs.items()
                if socket is self.socket and not socket.closed and sid not in self.ended]

    def of_tick(self) -> dict[str, Any]:
        """What a quote says of the tick it was made on."""
        return {} if self.spot_time is None else {"spot_time": self.spot_time()}

    def tick(self) -> None:
        for sid in () if self.silent else self.live_subs():
            self.socket.deliver({"msg_type": "proposal", "subscription": {"id": sid},
                                 "proposal": {"id": sid, **self.lane_quote, **self.of_tick()}})

    async def stream(self, every: float = 1.0) -> None:
        """An update of every live subscription each `every` seconds, 0.3 s off the second: never at the instant
        at which a request leaves or an answer comes, so the order of the two is not left to chance."""
        await asyncio.sleep(0.3)
        while True:
            self.tick()
            await asyncio.sleep(every)

    def stream_error(self, sid: str) -> None:
        """The venue ends one stream with an error frame."""
        self.socket.deliver({"msg_type": "proposal", "subscription": {"id": sid},
                             "error": {"code": "StreamEnded", "message": "this stream has ended"}})
        self.ended.add(sid)

    def take(self, socket: "_Socket", frame: dict[str, Any]) -> dict[str, Any] | None:
        """Do what the request asks and return its answer (None: no answer comes back). The venue decides by
        itself what it is asked: a call it does not know gets an error, whatever the survey's allowlist says."""
        kind = next((k for k in ("proposal", "contracts_for", "trading_times") if k in frame), "unknown")
        head = {"msg_type": kind, "req_id": frame.get("req_id")}
        code = self.refuse(frame) or ("UnrecognisedRequest" if kind == "unknown" else None) or (
            None if kind != "proposal" or self.priced(frame) else "MarketIsClosed")
        if code:
            reply = {**head, "error": {"code": code, "message": f"refused: {code}"}}
        elif kind != "proposal":
            reply = {**head, kind: {"listed": True}}
        else:
            self._n += 1
            reply = {**head, "proposal": {"id": f"p{self._n}", **self.quote(frame), **self.of_tick()}}
            if "subscribe" in frame:
                self.subs[f"p{self._n}"] = socket
                reply["subscription"] = {"id": f"p{self._n}"}
        return None if self.mute(frame) else reply


class _Socket:
    def __init__(self, venue: _Venue) -> None:
        self.venue = venue
        self.inbox: asyncio.Queue[str | None | Exception] = asyncio.Queue()
        self.closed = False

    def deliver(self, frame: dict[str, Any] | None) -> None:
        if frame is not None and not self.closed:
            self.inbox.put_nowait(json.dumps(frame))

    def end(self, error: Exception | None = None) -> None:
        self.closed = True
        self.inbox.put_nowait(error)

    async def send(self, text: str) -> None:
        import websockets.exceptions
        if self.closed:
            raise websockets.exceptions.ConnectionClosedOK(None, None)
        frame = json.loads(text)
        self.venue.frames.append((time.monotonic(), frame))
        reply = self.venue.take(self, frame)
        asyncio.get_running_loop().call_later(self.venue.delay(frame), lambda: self.deliver(reply))

    async def recv(self) -> str:
        import websockets.exceptions
        item = await self.inbox.get()
        if isinstance(item, str):
            return item
        self.inbox.put_nowait(item)             # a closed socket stays closed for every later reader
        raise item if item is not None else websockets.exceptions.ConnectionClosedOK(None, None)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        if self.venue.close_hangs:
            await asyncio.sleep(1e9)
        self.end()


def run_smoke() -> int:
    """The checks, in a temporary directory that is removed afterwards, in a process that from here on cannot
    reach the network and cannot open the archive's connection lock: the lock of the checks is a file in that
    directory."""
    global CONNECTION_LOCK
    import tempfile

    archive_lock = CONNECTION_LOCK
    shut = {str(archive_lock)}          # files that this process must not open

    def guard(event: str, args: Any) -> None:
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise RuntimeError("network access is forbidden in the smoke")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)) and os.fsdecode(args[0]) in shut:
            raise RuntimeError("this file is not to be opened in the smoke")

    sys.addaudithook(guard)
    with tempfile.TemporaryDirectory(prefix="quote_survey_smoke_") as tmp:
        CONNECTION_LOCK = Path(tmp) / ".connection.lock"
        try:
            return _smoke(Path(tmp), archive_lock, shut)
        finally:
            CONNECTION_LOCK = archive_lock


def _smoke(tmp: Path, archive_lock: Path, shut: set[str]) -> int:
    global META_S
    import collections
    import contextlib
    import inspect
    import io
    import socket
    from types import SimpleNamespace

    import websockets.exceptions

    failures: list[str] = []
    unread: list[str] = []
    runs: list[SimpleNamespace] = []    # every run of the survey that these checks make
    day0 = datetime(2026, 11, 17, 6, 0, tzinfo=timezone.utc)
    far = day0 + timedelta(days=1)

    def virtual(main: Callable[[], Any]) -> Any:
        return _run_virtual(main, unread)

    def check(name: str, ok: bool, detail: Any = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f" - {detail}" if detail != "" else ""))
        if not ok:
            failures.append(name)

    def strict(line: str) -> dict[str, Any]:
        """The row of one line, which must be JSON as the standard has it: no NaN, no Infinity."""
        def constant(name: str) -> Any:
            raise ValueError(f"not JSON: {name}")
        return json.loads(line, parse_constant=constant)

    def rows(log_dir: Path) -> list[dict[str, Any]]:
        return [strict(line) for f in sorted(log_dir.glob("*.jsonl")) for line in f.read_text().splitlines()]

    def at(row: dict[str, Any]) -> float:
        """Seconds from day0 to this row."""
        return round((datetime.fromisoformat(row["timestamp_utc"]) - day0).total_seconds(), 3)

    def of(s: SimpleNamespace, *events: str) -> list[dict[str, Any]]:
        """The rows of these events in the run of s."""
        return [r for r in s.run if r["event"] in events]

    def busiest(times: list[float], window: float) -> int:
        """The most requests in any `window` seconds."""
        best = lo = 0
        for hi, t in enumerate(times):
            while t - times[lo] >= window:
                lo += 1
            best = max(best, hi - lo + 1)
        return best

    def paced(times: list[float]) -> bool:
        """No two requests closer than the spacing (a millionth of a second is the rounding of the times)."""
        return all(b - a >= SEND_SPACING_S - 1e-6 for a, b in zip(times, times[1:], strict=False))

    class VenueClient(SurveyClient):
        def __init__(self, venue: _Venue, gate: RequestGate, on_event: Any = None) -> None:
            super().__init__(gate, on_event)
            self._venue = venue

        async def _dial(self) -> Any:
            return await self._venue.dial()

    class Timed(DailyJsonl):
        """The rows of a run, and when the row of its end was written, on the process's own clock. Writing that
        row takes `stop_row_takes` seconds, in which the process stands still (a disk that stalls)."""

        stopped_at: float | None = None
        stop_row_takes = 0.0

        def write(self, event: str, **fields: Any) -> None:
            if event == "survey_stopped":
                self.stopped_at = time.monotonic()
            super().write(event, **fields)
            if event == "survey_stopped":
                _VIRTUAL_NOW[0] += self.stop_row_takes

    async def session(name: str, venue: _Venue, pairs: list[str], seconds: float, script: Any = (),
                      stop_at: float | None = None, start: float = 0.0, lead: float = SEND_SPACING_S,
                      end_boot: float | None = None, stop_row_takes: float = 0.0) -> SimpleNamespace:
        """One run of the survey itself - survey(), as a process runs it - against the venue, in the directory
        `name`. It starts `start` seconds after day0 and its end time is `seconds` after day0. Its gate is made
        `lead` seconds before the start: one spacing unless a check says otherwise, so that the first request can
        leave at the start and the times of a run are those of its requests' own pace. (A process makes its gate
        as it takes the connection lock; the runs with lead 0 are such processes.) The gate's end on the
        process's own clock is read as --print-end-boot reads it, when the gate is made, unless `end_boot` is
        the reading that an earlier process of the same survey was given. script is [(seconds after day0, what
        happens)]; stop_at is when a stop is requested; stop_row_takes is how long the writing of the run's last
        row stalls. `end_boot` in what is returned is the gate's. Every other time in it is in seconds
        after day0 on the process's own clock: `took` is when the row of the run's end was written, `released`
        when the run returned (a process then ends, and lets go of the connection lock), `last` when its last
        request left (None: it sent nothing)."""
        t0 = time.monotonic() + lead - start

        def utcnow() -> datetime:
            return day0 + timedelta(seconds=time.monotonic() - t0 + venue.wall)

        def client_of(gate: RequestGate, on_event: Any) -> SurveyClient:
            venue.client = VenueClient(venue, gate, on_event)
            return venue.client

        end = day0 + timedelta(seconds=seconds)
        gate = RequestGate(end, utcnow, boot_deadline(end, utcnow, time.monotonic) if end_boot is None else end_boot,
                           elapsed=time.monotonic)
        await asyncio.sleep(lead)
        stop = asyncio.Event()
        venue.stop = stop.set
        loop = asyncio.get_running_loop()
        timers = [loop.call_later(when - start, what) for when, what in script]
        if stop_at is not None:
            timers.append(loop.call_later(stop_at - start, stop.set))
        stream = asyncio.create_task(venue.stream())
        logger, before = Timed(tmp / name, utcnow), len(venue.frames)
        logger.stop_row_takes = stop_row_takes
        code: int | None
        try:
            code = await asyncio.wait_for(survey(logger, pairs, gate, stop, client_of, name),
                                          seconds - start + 300.0)
        except asyncio.TimeoutError:
            code = None                         # 300 s after its end time: the checks on it then fail
        released = time.monotonic()
        stream.cancel()
        for timer in timers:
            timer.cancel()
        every = rows(tmp / name)
        run = every[max(i for i, r in enumerate(every) if r["event"] == "survey_started"):]
        own = [round(t - t0, 3) for t, _ in venue.frames[before:]]
        took = released if logger.stopped_at is None else logger.stopped_at
        runs.append(SimpleNamespace(
            name=name, code=code, took=round(took - t0, 3), released=round(released - t0, 3), end_boot=gate.end_boot,
            last=own[-1] if own else None, rows=every, run=run, stopped=run[-1], venue=venue,
            sent=[(round(t - t0, 3), f) for t, f in venue.frames],
            times=[round(t - t0, 3) for t, _ in venue.frames],
            subs=[(round(t - t0, 3), f["underlying_symbol"], f["contract_type"])
                  for t, f in venue.frames if "subscribe" in f],
            opens=[round(t - t0, 3) for t in venue.opens]))
        return runs[-1]

    def ended(s: SimpleNamespace) -> tuple[Any, ...]:
        """How a run ended: exit code, reason, and when the row of its end was written (the close included)."""
        return s.code, s.stopped.get("reason"), s.took

    def turn_kept(gate: RequestGate) -> bool:
        """A request could leave on a turn that an earlier request left behind."""
        try:
            gate.leave()
        except ForbiddenRequest:
            return False
        except SurveyOver:
            pass
        return True

    seven = list(DERIV_SYMBOLS)
    two = seven[:2]
    frx = DERIV_SYMBOLS

    # 0. nothing in this process can reach the network (the addresses are this machine's own in any case)
    def attempt(what: Callable[[], Any]) -> str:
        try:
            what()
        except (RuntimeError, OSError) as exc:
            return str(exc)
        return "went through"

    def connect() -> None:
        with socket.socket() as sock:
            sock.connect(("127.0.0.1", 9))

    tried = [attempt(lambda: socket.getaddrinfo("127.0.0.1", 9)), attempt(connect)]
    check("this process cannot reach the network: a name lookup and a connection are refused before they start",
          tried == ["network access is forbidden in the smoke"] * 2, tried)

    # 1. the allowlist: what may leave, and what may not
    allowed = [proposal_payload("EURUSD", 30, "CALLE"),
               {**proposal_payload("USDJPY", 15, "PUT"), "subscribe": 1, "req_id": 7},
               {"contracts_for": "frxEURUSD"}, {"trading_times": "2026-11-17", "req_id": 2}]
    check("the survey's own three kinds of request pass the allowlist",
          [request_kind(f) for f in allowed] == ["proposal", "proposal", "contracts_for", "trading_times"])
    forbidden = [{"buy": "id", "price": 1}, {"authorize": "token"}, {"sell": 1}, {"ticks_history": "frxEURUSD"},
                 {"proposal": 1, "buy": 1}, {**proposal_payload("EURUSD", 15, "CALL"), "contract_type": "MULTUP"},
                 {**proposal_payload("EURUSD", 15, "CALL"), "passthrough": {}}, {"forget": "abc"},
                 {"forget_all": "proposal"}, {"contracts_for": "frxEURUSD", "trading_times": "2026-11-17"},
                 {}, ["buy"], "buy",
                 *({**proposal_payload("EURUSD", 15, "CALL"), **other}
                   for other in ({"amount": 10.0}, {"amount": 50}, {"amount": "1"}, {"amount": True},
                                 {"amount": None}, {"basis": "payout"})),
                 {k: v for k, v in proposal_payload("EURUSD", 15, "CALL").items() if k != "amount"}]
    refused = 0
    for frame in forbidden:
        try:
            request_kind(frame)
        except ForbiddenRequest:
            refused += 1
    check("a purchase, a login, the end of a subscription, any other call, an unknown contract type, an extra "
          "field, another stake than 1 (10, 50, a stake that is not a number, none at all) or a price asked by "
          "payout does not", refused == len(forbidden) == 20, f"{refused} of {len(forbidden)} refused")

    # 2. the units and the closed-mode order
    units = poll_units(seven)
    grid = {(p, m, f) for p in seven for m in MINUTES for f in FAMILIES}
    check("7 pairs: 70 units, each once, the lanes' own contracts first",
          len(seven) == 7 and len(units) == 70 and set(units) == grid
          and [u for u in units if u[1:] == LANE_KIND] == units[:7] and LANE_KIND == (15, "risefall"))
    probes = closed_probes(seven)
    kinds = {(m, f) for m in MINUTES for f in FAMILIES}
    check("closed mode: any ten probes in a row cover the ten kinds of contract, and 70 cover the grid",
          sorted(probes) == sorted(units) and all({u[1:] for u in (probes + probes)[i:i + 10]} == kinds
                                                  for i in range(70))
          and all(len({u[0] for u in (probes + probes)[i:i + 10]}) == 7 for i in range(70)))
    check("the same holds for two pairs", sorted(closed_probes(two)) == sorted(poll_units(two))
          and all({u[1:] for u in closed_probes(two)[i:i + 10]} == kinds for i in (0, 10)))

    # 3. the gate on its own clocks
    async def gate_alone() -> dict[str, Any]:
        t0, wall, out = time.monotonic(), [0.0], {}
        gate = RequestGate(day0 + timedelta(seconds=1000),
                           lambda: day0 + timedelta(seconds=time.monotonic() - t0 + wall[0]), t0 + 1000.0,
                           elapsed=time.monotonic)

        async def go() -> float:
            await gate.turn()
            gate.leave()
            return time.monotonic() - t0

        out["row"] = [await go() for _ in range(5)]
        await asyncio.sleep(100.0)
        out["after_idling"] = [await go() for _ in range(3)]
        gate.rest(75.0)
        out["after_rest"] = await go()
        out["least_gap"] = gate.least_gap
        try:
            gate.leave()
            out["without_turn"] = "left"
        except ForbiddenRequest:
            out["without_turn"] = "refused"
        await gate.turn()
        gate.give_back()
        try:
            gate.leave()
            out["given_back"] = "left"
        except ForbiddenRequest:
            out["given_back"] = "refused"
        await gate.turn()
        wall[0] = 7200.0                # the wall clock is past the end between the turn and the leaving
        try:
            gate.leave()
            out["after_end"] = "left"
        except SurveyOver:
            out["after_end"] = "over"
        out["sent"], out["why"] = gate.sent, gate.over()
        return out

    async def gate_set_back() -> list[Any]:
        """A gate for 1000 s whose wall clock is set back by an hour after 400 s."""
        t0, wall, seen = time.monotonic(), [0.0], []
        gate = RequestGate(day0 + timedelta(seconds=1000),
                           lambda: day0 + timedelta(seconds=time.monotonic() - t0 + wall[0]), t0 + 1000.0,
                           elapsed=time.monotonic)
        for nap in (400.0, 599.0, 1.0):
            seen.append((gate.over(), round(gate.left(), 3)))
            await asyncio.sleep(nap)
            wall[0] = -3600.0
        seen.append((gate.over(), round(gate.left(), 3)))
        await gate.turn()
        try:
            gate.leave()
            seen.append("left")
        except SurveyOver:
            seen.append("over")
        return seen

    async def gate_slept() -> list[Any]:
        """A gate for 1000 s on a machine that sleeps for 700 s after 400 s, and whose wall clock - set by hand
        when it wakes - shows none of that sleep. The boot clock went on; the process's own clock did not."""
        t0, slept, seen = time.monotonic(), [0.0], []
        gate = RequestGate(day0 + timedelta(seconds=1000), lambda: day0 + timedelta(seconds=time.monotonic() - t0),
                           t0 + 1000.0, elapsed=lambda: time.monotonic() + slept[0])
        await asyncio.sleep(400.0)
        seen.append((gate.over(), round(gate.left(), 3)))
        slept[0] = 700.0
        seen.append((gate.over(), round(gate.left(), 3)))
        await gate.turn()
        try:
            gate.leave()
            seen.append("left")
        except SurveyOver:
            seen.append("over")
        return seen

    def paired(naps: list[float]) -> tuple[float, int]:
        """boot_deadline() for an end 100 s away, on clocks that stand still while they are read - except that
        the machine sleeps for naps[i] seconds during the i-th reading of the wall clock. Returns the deadline
        and how often the wall clock was read."""
        clock, reads = [0.0], [0]

        def wall_clock() -> datetime:
            clock[0] += naps[reads[0]] if reads[0] < len(naps) else 0.0
            reads[0] += 1
            return day0 + timedelta(seconds=clock[0])

        return boot_deadline(day0 + timedelta(seconds=100.0), wall_clock, lambda: clock[0]), reads[0]

    g = virtual(gate_alone)
    check("the gate: the first request leaves 2.5 s after the gate was made and the next ones 2.5 s apart; after "
          "100 idle seconds one leaves at once and the next 2.5 s later - nothing was saved up; the smallest gap "
          "the gate measured is 2.5 s", g["row"] == [2.5, 5.0, 7.5, 10.0, 12.5]
          and g["after_idling"] == [112.5, 115.0, 117.5] and g["least_gap"] == 2.5, g)
    check("the gate: after a rest of 75 s the next request leaves 75 s later; a request without a turn is refused, "
          "and so is one whose turn was given back; one whose turn came before the end and that would leave after "
          "it does not leave", (g["after_rest"], g["without_turn"], g["given_back"], g["after_end"], g["sent"],
                                g["why"]) == (192.5, "refused", "refused", "over", 9, "end_utc"), g)
    g = virtual(gate_set_back)
    check("the gate: a wall clock that is set back gives the survey no more time - the end is also a reading of "
          "a clock that nobody sets, and nothing leaves once that clock has reached it",
          g == [(None, 1000.0), (None, 600.0), (None, 1.0), ("elapsed_limit", 0.0), "over"], g)
    g = virtual(gate_slept)
    check("the gate: a machine that sleeps past the end, and whose wall clock then shows none of that sleep - the "
          "clock that went on while it slept ends the survey, and nothing leaves",
          g == [(None, 600.0), ("elapsed_limit", -100.0), "over"], g)
    asked: list[Any] = []
    system_clock = time.clock_gettime

    def two_clocks(clock: Any) -> float:
        """The clocks of a machine that has slept for an hour: its boot clock is that hour ahead of the other."""
        asked.append(clock)
        return 3700.0 if clock == time.CLOCK_BOOTTIME else 100.0

    time.clock_gettime = two_clocks
    try:
        reading = boot_seconds()
    finally:
        time.clock_gettime = system_clock
    check("outside these checks that clock is the boot clock, which nobody sets and which goes on through a "
          "suspension: the gate, and the reading that --print-end-boot takes, ask the system for that clock and "
          "for no other - on a machine that has slept for an hour they read the hour that its other clock lacks",
          inspect.signature(RequestGate).parameters["elapsed"].default is boot_seconds
          and inspect.signature(boot_deadline).parameters["elapsed"].default is boot_seconds
          and (reading, asked) == (3700.0, [time.CLOCK_BOOTTIME]), (reading, asked))
    try:
        paired([1.0] * 200)
        never = "a deadline"
    except RuntimeError:
        never = "none"
    pairings = [paired([]), paired([40.0]), paired([0.04])]
    check("the end as a reading of the boot clock is taken with the wall clock read between two readings of the "
          "boot clock. A machine that sleeps for 40 s between the readings has them taken again: the deadline is "
          "where the boot clock will stand at the end, not 40 s before it. 0.04 s between the readings make the "
          "deadline that much early, never late. Clocks that cannot be read together in 100 tries give none",
          pairings == [(100.0, 1), (100.0, 2), (99.96, 1)] and never == "none", (pairings, never))

    # 4. an open market, all seven pairs, twenty minutes
    async def open_market() -> SimpleNamespace:
        venue = _Venue()
        venue.delay = lambda f: 0.5 if f.get("contract_type") in ("PUT", "PUTE") else 0.25
        return await session("open", venue, seven, 1200.0)

    s = virtual(open_market)
    check("open market, 7 pairs, 20 minutes: the run ends at its end time with exit code 0, on one connection",
          ended(s) == (0, "end_utc", 1200.0) and s.opens == [0.0], (ended(s), s.opens))
    check("the pace is strict: no two requests closer than 2.5 s, 24 in the busiest minute, 480 in the 20 minutes",
          paced(s.times) and busiest(s.times, 60.0) == 24 and len(s.times) == 480,
          (busiest(s.times, 60.0), len(s.times)))
    check("a cycle of 70 units takes 350 s; the first is longer by its 14 subscriptions, the second by the "
          "metadata",
          [(at(r), r["seconds"], r["priced_sides"], r["lanes"], r["requests_sent"]) for r in of(s, "survey_cycle")]
          == [(383.0, 383.0, 140, 14, 154), (753.0, 370.0, 140, 14, 302), (1103.0, 350.0, 140, 14, 442)],
          [(at(r), r["seconds"], r["priced_sides"], r["lanes"], r["requests_sent"]) for r in of(s, "survey_cycle")])
    polled = of(s, "quote_poll")
    sides = {(p, m, f, ct) for p, m, f in grid for ct in FAMILIES[f]}
    check("every side of the 70 units is asked once a cycle, the units in their order, and all are priced",
          len(polled) == 458 and all(r["invalid_reason"] is None and r["stake"] == 1.0 and r["ask"] == 1.0
                                     and r["payout"] == 1.8 for r in polled)
          and all({(r["pair"], r["minutes"], r["family"], r["contract_type"]) for r in polled[i:i + 140]} == sides
                  for i in (0, 140, 280))
          and [(r["pair"], r["minutes"], r["family"]) for r in polled[:140:2]] == units, len(polled))
    check("each row has its own request's time, from leaving to its answer: 250 ms for a Rise side, 500 ms for a "
          "Fall side", {(r["contract_type"], r["latency_ms"]) for r in polled}
          == {("CALL", 250.0), ("CALLE", 250.0), ("PUT", 500.0), ("PUTE", 500.0)})
    check("each lane is subscribed once, right after a poll priced its contract: 14 subscriptions in the first "
          "70 s", s.subs == [(10.0 * i + d, frx[p], ct) for i, p in enumerate(seven)
                             for d, ct in ((5.0, "CALL"), (7.5, "PUT"))], s.subs[:4])
    meta = of(s, "contracts_for", "trading_times")
    check("the contract lists and the trading times are fetched once, after the first cycle",
          [r["event"] for r in meta] == ["contracts_for"] * 7 + ["trading_times"] and all(r["ok"] for r in meta)
          and [r.get("pair") for r in meta[:7]] == seven and [at(r) for r in meta] == [385.25 + 2.5 * i
                                                                                       for i in range(8)])
    shots = of(s, "quote_snapshot")
    check("the sampler writes one row per lane every 30 s and none at or after the end: 39 samples of 14 rows",
          sorted({at(r) for r in shots}) == [30.0 * i for i in range(1, 40)] and len(shots) == 39 * 14, len(shots))
    check("a lane has a row from the start: 'no_quote_received' until it is subscribed, then its latest quote",
          [r["invalid_reason"] for r in shots[:14]].count(None) == 6
          and [r["invalid_reason"] for r in shots[:14]].count("no_quote_received") == 8
          and all(r["invalid_reason"] is None and r["quote_age_ms"] <= 1000.0 and r["source"] == "quote_audit"
                  and r["payout"] == 1.85 for r in shots[28:]))
    first, last = s.run[0], s.stopped
    check("the first row says what runs (the tool, 70 units, the time left, the end as a reading of the boot "
          "clock and the boot it is of, the pace, the rest after a rate-limit rejection, six hours between "
          "metadata, five minutes for a lane, the code's checksums), the last how it ended and what was counted: "
          "here 480 requests, never closer than 2.5 s, and nothing else",
          (first["event"], first["tool"], first["units"], first["seconds_left"], first["send_spacing_seconds"],
           first["rate_limit_rest_seconds"], first["meta_seconds"], first["stale_seconds"])
          == ("survey_started", TOOL, 70, 1200.0, 2.5, 75.0, 21600.0, 300.0)
          and (first.get("boot_id"), first.get("end_boot_seconds")) == (boot_id(), round(s.end_boot, 3))
          and "deriv_quote_survey.py" in first["code_sha256"]
          and [last[k] for k in ("event", "exit_code", "lanes", "requests_sent", "least_gap_seconds",
                                 "stream_errors", "rate_limited", "late_answers", "pre_sub_buffered", "overflow",
                                 "unmatched", "lock_kept_seconds")]
          == ["survey_stopped", 0, 14, 480, 2.5, 0, 0, 0, 0, 0, 0, 0.0],
          {k: v for k, v in last.items() if k != "events"})
    check("what reached the venue is what the gate counted, all of it the survey's own kinds, none of it at or "
          "after the end", {request_kind(f) for _, f in s.sent} == {"proposal", "contracts_for", "trading_times"}
          and max(s.times) == 1197.5)

    # 5. one request at a time
    async def slow_venue() -> SimpleNamespace:
        venue = _Venue()
        venue.delay = lambda f: 4.0
        return await session("slow", venue, two, 101.0)

    s = virtual(slow_venue)
    cut = of(s, "quote_poll")[-1]
    check("requests go one at a time: when an answer takes 4 s the next request leaves as that answer comes",
          s.times == [4.0 * i for i in range(26)]
          and {r["latency_ms"] for r in of(s, "quote_poll") if r["invalid_reason"] is None} == {4000.0}, s.times[:6])
    check("the run ends at its end time, not at the next request: a request in flight then gets a 'run_ended' row; "
          "the run returns 1.5 s later, when that request's successor could have left, and says so; the smallest "
          "gap it measured is the 4 s",
          ended(s) == (0, "end_utc", 101.0) and (cut["invalid_reason"], cut["latency_ms"]) == ("run_ended", 1000.0)
          and (s.released, s.stopped["lock_kept_seconds"], s.stopped["least_gap_seconds"]) == (102.5, 1.5, 4.0),
          (ended(s), cut["invalid_reason"], cut["latency_ms"], s.released, s.stopped["lock_kept_seconds"]))

    # 6. the connection is lost, or cannot be opened
    async def lost_connection() -> SimpleNamespace:
        venue = _Venue()
        return await session("lost", venue, two, 300.0, script=[
            (51.0, lambda: venue.socket.end(websockets.exceptions.ConnectionClosedError(None, None)))])

    s = virtual(lost_connection)
    cut = of(s, "quote_poll")[-1]
    check("a lost connection is not repaired: the run ends with exit code 1 (connection_lost) at its next request, "
          "which never leaves and whose turn is not kept for another; there was one connection attempt in all",
          ended(s) == (1, "connection_lost", 52.5) and s.opens == [0.0] and max(s.times) == 50.0
          and not turn_kept(s.venue.client.gate)
          and (cut["invalid_reason"], cut.get("error_code"), cut["latency_ms"]) == ("request_failed", None, None)
          and [r["reason"] for r in of(s, "survey_failed")] == ["connection_lost"]
          and {"client_disconnected", "client_reconnect_exhausted"} <= set(s.stopped["events"]),
          (ended(s), s.opens, max(s.times)))

    async def unreachable() -> SimpleNamespace:
        venue = _Venue()
        venue.down = True
        return await session("unreachable", venue, two, 300.0)

    s = virtual(unreachable)
    check("a connection that cannot be opened: exit code 1 (connect_failed) after one attempt, and nothing is sent",
          ended(s) == (1, "connect_failed", 0.0) and s.opens == [0.0] and s.sent == []
          and [r["event"] for r in s.run] == ["survey_started", "survey_failed", "survey_stopped"],
          (ended(s), s.opens))

    # 7. a subscription without a clear answer; a subscription the venue refuses
    async def answer_lost() -> SimpleNamespace:
        venue = _Venue()
        venue.mute = lambda f: "subscribe" in f
        return await session("answer_lost", venue, two, 1800.0)

    async def answer_late() -> SimpleNamespace:
        venue = _Venue()
        venue.delay = lambda f: 16.0 if "subscribe" in f else 0.0
        return await session("answer_late", venue, two, 1800.0)

    for how, s in (("is lost", virtual(answer_lost)), ("comes a second after its time", virtual(answer_late))):
        check(f"a subscription whose answer {how}: the venue has a stream that nothing here reads, so the run "
              "ends with exit code 1 (subscription_unanswered) after that one request, and the closed connection "
              "takes the stream with it",
              ended(s) == (1, "subscription_unanswered", 20.0) and [t for t, _, _ in s.subs] == [5.0]
              and s.stopped["lanes"] == 0 and len(s.venue.subs) == 1 and s.venue.live_subs() == []
              and s.venue.socket.closed, (ended(s), s.subs, len(s.venue.subs)))

    async def subscription_refused() -> SimpleNamespace:
        venue = _Venue()
        venue.refuse = lambda f: "ContractCreationFailure" if "subscribe" in f else None
        return await session("sub_refused", venue, two, 300.0)

    s = virtual(subscription_refused)
    check("a subscription the venue refuses is an answer: the run goes on and asks again once a cycle",
          ended(s) == (0, "end_utc", 300.0) and s.stopped["lanes"] == 0
          and [t for t, _, _ in s.subs] == [5.0, 7.5, 15.0, 17.5, 122.5, 125.0, 132.5, 135.0, 232.5, 235.0, 242.5,
                                           245.0]
          and {r["invalid_reason"] for r in of(s, "quote_snapshot")} == {"no_quote_received"}
          and s.stopped["events"].get("quote_resubscribe_failed") == 12, (ended(s), [t for t, _, _ in s.subs]))

    # 8. a venue that stops answering; one answer that comes too late
    async def venue_mute() -> SimpleNamespace:
        venue = _Venue()
        return await session("mute", venue, two, 600.0,
                             script=[(99.0, lambda: setattr(venue, "mute", lambda f: True))])

    s = virtual(venue_mute)
    silent = [r for r in of(s, "quote_poll") if r["invalid_reason"] == "request_failed"]
    check("a venue that stops answering on an open connection: after 3 requests in a row without an answer the "
          "run ends with exit code 1 (venue_unanswering)",
          ended(s) == (1, "venue_unanswering", 145.0) and [t for t in s.times if t > 99.0] == [100.0, 115.0, 130.0]
          and [(r["error_code"], r["latency_ms"]) for r in silent] == [(None, 15000.0)] * 3,
          (ended(s), [t for t in s.times if t > 99.0]))

    async def one_answer_late() -> SimpleNamespace:
        venue = _Venue()
        venue.delay = lambda f: 16.0 if f.get("req_id") == 10 else 0.0
        return await session("one_late", venue, two, 300.0)

    s = virtual(one_answer_late)
    silent = [r for r in of(s, "quote_poll") if r["invalid_reason"] == "request_failed"]
    check("one request without an answer does not end the run; its answer, a second after its time, is counted "
          "and not kept",
          ended(s) == (0, "end_utc", 300.0) and [(r["error_code"], r["latency_ms"]) for r in silent]
          == [(None, 15000.0)] and s.stopped["lanes"] == 4
          and [s.stopped[k] for k in ("late_answers", "pre_sub_buffered", "unmatched", "rate_limited")]
          == [1, 0, 0, 0], (ended(s), len(silent), s.stopped["late_answers"], s.stopped["pre_sub_buffered"]))

    async def answers_at_the_limit() -> SimpleNamespace:
        """Every second unsubscribed request gets its answer 15 s later: in the instant in which it times out."""
        venue = _Venue()
        venue.delay = lambda f: 15.0 if "subscribe" not in f and f.get("req_id") % 2 == 0 else 0.0
        return await session("at_the_limit", venue, two, 600.0)

    s = virtual(answers_at_the_limit)
    silent = [r for r in of(s, "quote_poll") if r["invalid_reason"] == "request_failed"]
    lists = [(r["event"], r["error_code"]) for r in of(s, "contracts_for", "trading_times") if not r["ok"]]
    check("an answer that comes in the instant in which its request times out is late as well: with every second "
          "request answered so - 32 price requests and a contract list in this run - each such answer is counted "
          "and dropped and none is kept",
          ended(s) == (0, "end_utc", 600.0) and (len(silent), lists) == (32, [("contracts_for", None)])
          and {(r["error_code"], r["latency_ms"]) for r in silent} == {(None, 15000.0)}
          and [s.stopped[k] for k in ("late_answers", "pre_sub_buffered", "unmatched")] == [33, 0, 0]
          and not s.venue.client._pre_sub and of(s, "quote_poll")[-1]["invalid_reason"] == "run_ended",
          (ended(s), len(silent), lists, s.stopped["late_answers"], s.stopped["pre_sub_buffered"],
           s.stopped["unmatched"], len(s.venue.client._pre_sub)))

    async def nothing_waits() -> tuple[Any, ...]:
        """A client that has made 6000 requests, one of them (number 5000) for a subscription, and has no
        request waiting: what becomes of the frames that then come."""
        client = VenueClient(_Venue(), RequestGate(far, lambda: day0, math.inf))
        client._req_id = 6000
        client._request_left({**proposal_payload("EURUSD", 15, "CALL"), "subscribe": 1, "req_id": 5000})
        for rid in range(1, 3001):              # the answers to its first 3000 requests, every one of them late
            client._dispatch({"msg_type": "proposal", "req_id": rid, "proposal": {"id": f"p{rid}"}})
        old = (client.late_answers, len(client._pre_sub), client.stats["unmatched_frames"], len(client._streams))
        for frame in ({"msg_type": "proposal", "req_id": 10, "proposal": {"id": "again"}},      # late once more
                      {"msg_type": "trading_times", "req_id": 3, "trading_times": {}},          # late: not a price
                      {"msg_type": "proposal", "req_id": 5000, "proposal": {"id": "stream"}},   # a subscription's
                      {"msg_type": "proposal", "req_id": 7000, "proposal": {"id": "unknown"}},  # no request of its own
                      {"msg_type": "proposal", "req_id": "10", "proposal": {"id": "text"}},     # nor is this one
                      {"msg_type": "proposal", "req_id": True, "proposal": {"id": "truth"}},    # nor this
                      {"msg_type": "tick"}):                                                    # no id at all
            client._dispatch(frame)
        return (old, client.late_answers, sorted(client._pre_sub), client.stats["unmatched_frames"])

    kept = virtual(nothing_waits)
    check("an answer to an unsubscribed request that no longer waits is late however old the request, and none is "
          "kept: 3000 such answers are counted and dropped, and so is one that comes twice. The client remembers "
          "the ids of its subscription requests only; a frame with such an id, or with an id that is no request "
          "of this client, is left to the shared client, which keeps it for a subscription or counts it as "
          "unmatched", kept == ((3000, 0, 0, 1), 3002, ["stream", "text", "truth", "unknown"], 1), kept)

    # 9. lanes without updates: dead, or quiet
    async def lanes_without_updates(name: str, spot_time: Callable[[], Any] | None) -> SimpleNamespace:
        venue = _Venue()
        venue.silent, venue.spot_time = True, spot_time
        return await session(name, venue, two, 600.0)

    tick_0 = 1_800_000_000
    s = virtual(lambda: lanes_without_updates("silent", None))
    dead = virtual(lambda: lanes_without_updates("silent_dead", lambda: tick_0 + int(time.monotonic())))
    check("lanes without updates: when a poll prices a lane's contract and the lane has been silent for 5 minutes "
          "the run ends with exit code 1 (lanes_silent) - when the venue tells no tick times, and when the poll's "
          "quote is of a later tick than the lane's last update",
          ended(s) == ended(dead) == (1, "lanes_silent", 320.0) and s.opens == dead.opens == [0.0]
          and max(s.times) == max(dead.times) == 320.0, (ended(s), ended(dead), max(s.times)))
    quiet = virtual(lambda: lanes_without_updates("silent_quiet", lambda: tick_0))
    check("a market without ticks: the poll's quote is of the same tick as the lane's last update, so the lane is "
          "quiet and not dead, and the run goes on to its end with its lanes",
          ended(quiet) == (0, "end_utc", 600.0) and quiet.stopped["lanes"] == 4 and len(quiet.subs) == 4
          and not of(quiet, "survey_failed"), (ended(quiet), quiet.stopped["lanes"]))
    cases = [(100.0, 101), (100.0, 100), (100.0, 99.5), (None, 100), (100.0, None), (float("nan"), 100),
             (100.0, "101"), (True, 100)]
    check("a lane is behind a poll when the poll's spot time is the later one, and also when one of the two is "
          "not a time: then silence alone decides",
          [lane_behind(*case) for case in cases] == [True, False, False, True, True, True, True, True],
          [lane_behind(*case) for case in cases])

    async def stream_error() -> SimpleNamespace:
        venue = _Venue()
        return await session("stream_error", venue, two, 600.0, script=[(50.5, lambda: venue.stream_error("p3"))])

    s = virtual(stream_error)
    check("an error frame on a lane's stream is counted and does not end the run by itself; the lane that then "
          "stays silent does, 5 minutes later, when a poll prices its contract",
          ended(s) == (1, "lanes_silent", 420.0) and s.stopped["stream_errors"] == 1
          and s.stopped["events"].get("quote_stream_error") == 1 and s.stopped["lanes"] == 4,
          (ended(s), s.stopped["stream_errors"]))

    # 10. refusals are answers: one side, a closed market, a reopening, a market that is open in part
    async def one_side_refused() -> SimpleNamespace:
        venue = _Venue()
        venue.priced = lambda f: not (f["underlying_symbol"] == frx[two[0]] and f["contract_type"] == "PUT"
                                      and f["duration"] == 15)
        return await session("one_side", venue, two, 300.0)

    s = virtual(one_side_refused)
    refusals = [r for r in of(s, "quote_poll") if r["invalid_reason"] is not None]
    check("one side refused: its row has the venue's error code, the other side is priced and gets its lane, and "
          "the run goes on", ended(s) == (0, "end_utc", 300.0) and s.stopped["lanes"] == 3
          and [(sym, ct) for _, sym, ct in s.subs] == [(frx[two[0]], "CALL"), (frx[two[1]], "CALL"),
                                                       (frx[two[1]], "PUT")]
          and [(r["pair"], r["contract_type"], r["invalid_reason"], r.get("error_code")) for r in refusals]
          == [(two[0], "PUT", "request_failed", "MarketIsClosed")] * 3, (ended(s), s.subs, len(refusals)))

    async def closed_then_open() -> SimpleNamespace:
        venue = _Venue()
        venue.priced = lambda f: False
        return await session("closed", venue, seven, 800.0,
                             script=[(700.7, lambda: setattr(venue, "priced", lambda f: True))])

    s = virtual(closed_then_open)
    probed = [(r["pair"], r["minutes"], r["family"]) for r in of(s, "quote_poll") if 76 >= r["seq"] > 70][::2]
    check("a closed market: after one cycle of refusals the survey asks one unit a minute, a different kind each "
          "time", [(r["event"], at(r)) for r in of(s, "survey_closed", "survey_open")]
          == [("survey_closed", 347.5), ("survey_open", 705.0)]
          and [t for t in s.times if 370.0 < t < 700.0] == [402.5 + 60.0 * i + d for i in range(5) for d in (0.0, 2.5)]
          and probed == closed_probes(seven)[:6], ([t for t in s.times if 370.0 < t < 700.0], probed))
    check("when a probe is priced the survey is back at its pace at once, and subscribes the lanes as their "
          "contracts are priced", [t for t in s.times if t > 705.0][:4] == [707.5, 710.0, 712.5, 715.0]
          and [t for t, _, _ in s.subs][:2] == [712.5, 715.0] and ended(s) == (0, "end_utc", 800.0),
          ([t for t in s.times if t > 705.0][:4], s.subs[:2]))

    async def weekend() -> tuple[SimpleNamespace, SimpleNamespace]:
        venue = _Venue()

        def market_closes() -> None:
            venue.priced = lambda f: False
            venue.ended |= set(venue.subs)      # the venue ends its streams at the close and says nothing

        before = await session("weekend", venue, two, 1400.0, script=[
            (150.7, market_closes), (900.7, lambda: setattr(venue, "priced", lambda f: True))])
        await asyncio.sleep(90.0)               # what the unit does after exit code 1: a new process 90 s later
        return before, await session("weekend", venue, two, 1400.0, start=before.released + 90.0, lead=0.0)

    s, again = virtual(weekend)
    check("a weekend: the close is seen in one cycle of refusals and the reopening by a probe",
          [(r["event"], at(r)) for r in of(s, "survey_closed", "survey_open")]
          == [("survey_closed", 315.0), ("survey_open", 912.5)],
          [(r["event"], at(r)) for r in of(s, "survey_closed", "survey_open")])
    check("the lanes the venue ended at the close are found silent when a poll prices their contract again: the "
          "run ends with exit code 1 (lanes_silent)", ended(s) == (1, "lanes_silent", 917.5), ended(s))
    check("the next process, 90 s after the first returned, in the same directory, appends its rows, counts its "
          "units from 1 again, opens a new connection, sends its first request 2.5 s after it started and "
          "subscribes every lane afresh",
          s.released == 920.0 and ended(again) == (0, "end_utc", 1400.0) and again.opens == [0.0, 1010.0]
          and [r["event"] for r in again.rows].count("survey_started") == 2 and again.rows[:len(s.rows)] == s.rows
          and of(again, "quote_poll")[0]["seq"] == 1 and [t for t in again.times if t > 920.0][0] == 1012.5
          and [t for t, _, _ in again.subs if t > 920.0] == [1017.5, 1020.0, 1027.5, 1030.0]
          and again.stopped["lanes"] == 4
          and all(r["invalid_reason"] is None and r["quote_age_ms"] <= 1000.0
                  for r in of(again, "quote_snapshot")[-4:]),
          (ended(again), again.opens, [t for t, _, _ in again.subs if t > 920.0]))

    async def open_in_part() -> SimpleNamespace:
        venue = _Venue()
        venue.priced = lambda f: False
        return await session("in_part", venue, two, 500.0,
                             script=[(200.7, lambda: setattr(venue, "priced", lambda f: f["duration"] >= 30))])

    s = virtual(open_in_part)
    check("a market that opens for the longer contracts only is seen by the probe of such a contract, and the "
          "survey then runs without lanes", [(r["event"], at(r)) for r in of(s, "survey_closed", "survey_open")]
          == [("survey_closed", 97.5), ("survey_open", 275.0)] and s.subs == []
          and [r["priced_sides"] for r in of(s, "survey_cycle")] == [0, 32, 32]
          and ended(s) == (0, "end_utc", 500.0),
          ([(r["event"], at(r)) for r in of(s, "survey_closed", "survey_open")],
           [r["priced_sides"] for r in of(s, "survey_cycle")], ended(s)))

    # 11. a rate-limit rejection, in time or late
    async def rate_limited(name: str, late: float = 0.0) -> SimpleNamespace:
        venue = _Venue()
        venue.refuse = lambda f: "RateLimit" if f.get("req_id") == 12 else None
        venue.delay = lambda f: late if f.get("req_id") == 12 else 0.0
        return await session(name, venue, two, 300.0)

    s = virtual(lambda: rate_limited("rate_limit"))
    check("a rate-limit rejection: nothing is sent for 75 s, then the survey goes on at its pace",
          s.times[10:14] == [25.0, 27.5, 102.5, 105.0] and s.stopped["rate_limited"] == 1
          and [r.get("error_code") for r in of(s, "quote_poll") if r["invalid_reason"] is not None] == ["RateLimit"]
          and ended(s) == (0, "end_utc", 300.0), (s.times[10:14], s.stopped["rate_limited"]))
    s = virtual(lambda: rate_limited("rate_limit_late", late=16.0))
    check("a rate-limit rejection that comes a second after its request has timed out: one more request has left "
          "by then; from the rejection on nothing is sent for 75 s, and it is counted as a rejection and as a "
          "late answer",
          s.times[10:15] == [25.0, 27.5, 42.5, 118.5, 121.0]
          and [s.stopped[k] for k in ("rate_limited", "late_answers", "pre_sub_buffered", "unmatched")]
          == [1, 1, 0, 0]
          and [(r.get("error_code"), r["latency_ms"]) for r in of(s, "quote_poll") if r["invalid_reason"] is not None]
          == [(None, 15000.0)] and ended(s) == (0, "end_utc", 300.0),
          (s.times[10:15], s.stopped["rate_limited"], s.stopped["late_answers"]))

    async def rejection_then_cancel() -> tuple[Any, ...]:
        """A rate-limit rejection is read while its request waits, and the request is cancelled - as a stop
        cancels it - before it has seen the rejection."""
        venue = _Venue()
        venue.mute = lambda f: True
        client = VenueClient(venue, RequestGate(far, lambda: day0, math.inf))
        await client.connect()
        asking = asyncio.create_task(client.request(proposal_payload("EURUSD", 15, "CALL")))
        await asyncio.sleep(3.0)                # the request left 2.5 s after the gate was made, and waits
        client._dispatch({"msg_type": "proposal", "req_id": 1,
                          "error": {"code": "RateLimit", "message": "refused: RateLimit"}})
        asking.cancel()
        await asyncio.gather(asking, return_exceptions=True)
        out = (asking.cancelled(), client.rate_limited, round(client.gate.wait(), 3), len(venue.frames))
        await client.close()
        return out

    out = virtual(rejection_then_cancel)
    check("a rate-limit rejection rests the gate as it is read: a request that is cancelled before it has seen "
          "its own rejection leaves the 75 s rest behind all the same", out == (True, 1, 75.0, 1), out)

    async def stop_as_rejection_comes() -> SimpleNamespace:
        """The 12th request has no answer yet when, 3 s after it left, a stop is asked for; the venue's
        rate-limit rejection of that request is read right behind the stop, before the run has ended it."""
        venue = _Venue()
        venue.mute = lambda f: f.get("req_id") == 12

        def stop_then_reject() -> None:
            venue.stop()
            asyncio.get_running_loop().call_soon(venue.socket.deliver, {
                "msg_type": "proposal", "req_id": 12, "error": {"code": "RateLimit", "message": "refused: RateLimit"}})

        return await session("stop_rejected", venue, two, 300.0, script=[(30.5, stop_then_reject)])

    s = virtual(stop_as_rejection_comes)
    cut = of(s, "quote_poll")[-1]
    check("and so in a whole run: a stop, and right behind it the rejection of the request in flight - the run "
          "ends at once, counts the rejection, and keeps the connection lock for the 75 s of the rest",
          ended(s) == (0, "stop_requested", 30.5) and s.last == 27.5
          and (cut["invalid_reason"], cut["latency_ms"]) == ("run_ended", 3000.0)
          and (s.stopped["rate_limited"], s.stopped["lock_kept_seconds"], s.released) == (1, 75.0, 105.5),
          (ended(s), s.last, cut["invalid_reason"], s.stopped["rate_limited"], s.stopped["lock_kept_seconds"],
           s.released))

    # 12. the end of a run: a stop request, the end time, clocks that jump, a connection that does not close,
    #     and the next process
    async def stop_in_flight() -> SimpleNamespace:
        venue = _Venue()
        venue.delay = lambda f: 4.0 if f.get("req_id") == 2 else 0.0
        return await session("stop_in_flight", venue, two, 300.0, stop_at=5.0)

    async def stop_while_waiting() -> SimpleNamespace:
        return await session("stop_waiting", _Venue(), two, 300.0, stop_at=1.0)

    s, waiting = virtual(stop_in_flight), virtual(stop_while_waiting)
    check("a stop request ends the run at once with exit code 0 and sends nothing; a request that had left gets "
          "a 'run_ended' row", ended(s) == (0, "stop_requested", 5.0) and s.times == [0.0, 2.5]
          and [(r["contract_type"], r["invalid_reason"], r["latency_ms"]) for r in of(s, "quote_poll")]
          == [("CALL", None, 0.0), ("PUT", "run_ended", 2500.0)], (ended(s), s.times))
    check("a request that was still waiting for its turn when the stop came never leaves and has no row; the run "
          "returns when that request could have left, 1.5 s later",
          ended(waiting) == (0, "stop_requested", 1.0) and waiting.times == [0.0]
          and [r["contract_type"] for r in of(waiting, "quote_poll")] == ["CALL"]
          and (waiting.released, waiting.stopped["lock_kept_seconds"]) == (2.5, 1.5),
          (ended(waiting), waiting.times, waiting.released))

    async def handoff(name: str, stop_at: float, limited: bool) -> tuple[SimpleNamespace, SimpleNamespace]:
        """A run that is stopped, and a process that takes the connection lock the moment the first lets go."""
        venue = _Venue()
        if limited:
            venue.refuse = lambda f: "RateLimit" if len(venue.frames) == 12 else None    # the 12th request, once
        first = await session(name, venue, two, 200.0, stop_at=stop_at)
        return first, await session(name, venue, two, 200.0, start=first.released, lead=0.0)

    s, again = virtual(lambda: handoff("handoff", 61.0, False))
    check("from one process to the next the pace holds: a run stopped a second after its 25th request returns "
          "1.5 s later; the process that then takes over sends its first request 2.5 s after that - 24 requests "
          "in the busiest minute of the two together, and no two closer than 2.5 s",
          ended(s) == (0, "stop_requested", 61.0) and (s.released, s.stopped["lock_kept_seconds"]) == (62.5, 1.5)
          and again.opens == [0.0, 62.5] and again.times[23:27] == [57.5, 60.0, 65.0, 67.5] and paced(again.times)
          and busiest(again.times, 60.0) == 24 and ended(again) == (0, "end_utc", 200.0),
          (ended(s), s.released, again.opens, again.times[23:27], busiest(again.times, 60.0)))
    s, again = virtual(lambda: handoff("handoff_limited", 40.0, True))
    check("and so does a rest: a run stopped 12.5 s after a rate-limit rejection returns when the 75 s are over, "
          "62.5 s later, so the process that then takes over sends nothing before",
          ended(s) == (0, "stop_requested", 40.0) and s.last == 27.5
          and (s.released, s.stopped["lock_kept_seconds"], s.stopped["rate_limited"]) == (102.5, 62.5, 1)
          and again.opens == [0.0, 102.5] and [t for t in again.times if t > 27.5][:2] == [105.0, 107.5]
          and ended(again) == (0, "end_utc", 200.0),
          (ended(s), s.released, s.stopped["lock_kept_seconds"], [t for t in again.times if t > 27.5][:2]))

    async def stop_row_stalls(name: str, takes: float) -> SimpleNamespace:
        """A run that is stopped 12.5 s after a rate-limit rejection, on a disk that takes `takes` seconds to
        write the run's last row."""
        venue = _Venue()
        venue.refuse = lambda f: "RateLimit" if len(venue.frames) == 12 else None
        return await session(name, venue, two, 200.0, stop_at=40.0, stop_row_takes=takes)

    s, longer = virtual(lambda: stop_row_stalls("stall_60", 60.0)), virtual(lambda: stop_row_stalls("stall_100", 100.0))
    check("the wait at the end of a run is until its next request could have left, and what the writing of the "
          "last row takes is part of it: when that row takes 60 s the run returns as the rest ends, 62.5 s after "
          "the stop and not 60 s later; when it takes 100 s the run returns as the row is written. The row has "
          "the wait as it stood before it was written",
          [(ended(r), r.stopped["lock_kept_seconds"], r.released) for r in (s, longer)]
          == [((0, "stop_requested", 40.0), 62.5, 102.5), ((0, "stop_requested", 40.0), 62.5, 140.0)],
          [(ended(r), r.stopped["lock_kept_seconds"], r.released) for r in (s, longer)])

    async def suspended() -> SimpleNamespace:
        venue = _Venue()
        return await session("suspended", venue, two, 1000.0, script=[(100.7, lambda: setattr(venue, "wall", 7200.0))])

    s = virtual(suspended)
    check("a machine suspended across the end time (the wall clock is 2 hours on, the process's own clock is not): "
          "the next request does not leave, no further sample is taken, exit code 0",
          ended(s) == (0, "end_utc", 102.5) and max(s.times) == 100.0
          and sorted({at(r) for r in of(s, "quote_snapshot")}) == [30.0, 60.0, 90.0], (ended(s), max(s.times)))

    async def set_back() -> SimpleNamespace:
        venue = _Venue()
        return await session("set_back", venue, two, 300.0, script=[(100.7, lambda: setattr(venue, "wall", -3600.0))])

    s = virtual(set_back)
    check("a wall clock set back by an hour during the run: the run still ends when the 300 s it was started for "
          "are over (elapsed_limit, exit code 0); no request leaves and no sample is taken at or after that",
          ended(s) == (0, "elapsed_limit", 300.0) and max(s.times) == 297.5
          and len({at(r) for r in of(s, "quote_snapshot")}) == 9 and s.run[0]["seconds_left"] == 300.0,
          (ended(s), max(s.times), len({at(r) for r in of(s, "quote_snapshot")})))

    async def set_back_and_replaced() -> tuple[SimpleNamespace, SimpleNamespace]:
        """A survey of 300 s. After 100 s its wall clock is set back by an hour; after 150 s its connection is
        lost. The process that a supervisor starts 90 s after the first has ended is given the first one's
        reading of the boot clock, as every process of one survey is."""
        venue = _Venue()
        first = await session("replaced_first", venue, two, 300.0, script=[
            (100.7, lambda: setattr(venue, "wall", -3600.0)),
            (150.2, lambda: venue.socket.end(websockets.exceptions.ConnectionClosedError(None, None)))])
        await asyncio.sleep(90.0)
        return first, await session("replaced_second", venue, two, 300.0, start=first.released + 90.0, lead=0.0,
                                    end_boot=first.end_boot)

    s, again = virtual(set_back_and_replaced)
    check("a wall clock set back by an hour, and then a new process: the survey still ends when the 300 s it was "
          "started for are over. The process that replaces the first has the same reading of the boot clock; by "
          "its wall clock more than an hour is left, and it ends at 300 s all the same (elapsed_limit, exit code "
          "0), its last request before that",
          ended(s) == (1, "connection_lost", 152.5) and again.opens == [0.0, 242.5]
          and (again.run[0]["seconds_left"], again.run[0].get("end_boot_seconds"))
          == (57.5, round(s.end_boot, 3)) and ended(again) == (0, "elapsed_limit", 300.0)
          and max(again.times) == 297.5 and not [t for t in again.times if 150.0 < t < 245.0],
          (ended(s), again.opens, again.run[0]["seconds_left"], ended(again), max(again.times)))

    async def answer_after_end() -> SimpleNamespace:
        venue = _Venue()
        venue.delay = lambda f: 4.0 if f.get("req_id") == 5 else 0.0            # a poll, sent at 10.0
        return await session("after_end", venue, two, 300.0, script=[(12.0, lambda: setattr(venue, "wall", 7200.0))])

    s = virtual(answer_after_end)
    check("an answer that comes when the survey is over - the machine was suspended across the end time while "
          "the request was in flight - is not a sample: its row is 'run_ended' and has no quote",
          ended(s) == (0, "end_utc", 14.0) and max(s.times) == 10.0
          and [(r["contract_type"], r["invalid_reason"], r["latency_ms"], r["payout"]) for r in of(s, "quote_poll")]
          == [("CALL", None, 0.0, 1.8), ("PUT", None, 0.0, 1.8), ("CALL", "run_ended", 4000.0, None)],
          (ended(s), [(r["contract_type"], r["invalid_reason"]) for r in of(s, "quote_poll")]))

    async def metadata_cut(name: str, by_stop: bool, refused: bool = False) -> SimpleNamespace:
        venue = _Venue()
        venue.delay = lambda f: 4.0 if f.get("req_id") == 46 else 0.0       # the second contract list, sent at 112.5
        if refused:
            venue.refuse = lambda f: "InvalidSymbol" if f.get("req_id") == 46 else None
        if by_stop:
            return await session(name, venue, two, 300.0, stop_at=114.0)
        return await session(name, venue, two, 300.0, script=[(114.0, lambda: setattr(venue, "wall", 7200.0))])

    s, lapsed = virtual(lambda: metadata_cut("meta_stopped", True)), virtual(lambda: metadata_cut("meta_over", False))
    refused = virtual(lambda: metadata_cut("meta_refused_over", False, refused=True))
    lists = [[(r["event"], r["ok"], r.get("error_code"), r.get("error"))
              for r in of(run, "contracts_for", "trading_times")] for run in (s, lapsed, refused)]
    check("the same holds for the contract lists and the trading times: a request that had left when a stop came, "
          "one whose answer came when the survey was over, and one that the venue refused when the survey was "
          "over, each have a row that says 'run_ended' and holds nothing of the venue's",
          lists == [[("contracts_for", True, None, None), ("contracts_for", False, None, "run_ended")]] * 3
          and (ended(s), s.released) == ((0, "stop_requested", 114.0), 115.0)
          and ended(lapsed) == ended(refused) == (0, "end_utc", 116.5)
          and max(lapsed.times) == max(refused.times) == 112.5,
          (lists, ended(s), ended(lapsed), ended(refused)))

    async def metadata_again() -> SimpleNamespace:
        return await session("meta_again", _Venue(), two, 500.0)

    kept, META_S = META_S, 250.0
    try:
        s = virtual(metadata_again)
    finally:
        META_S = kept
    meta = of(s, "contracts_for", "trading_times")
    check("the contract lists and the trading times are fetched again at the end of the first cycle that ends "
          "META_S or more after the last fetch (in this run META_S is 250 s: the fourth cycle)",
          [r["event"] for r in meta] == ["contracts_for", "contracts_for", "trading_times"] * 2
          and [at(r) for r in meta] == [110.0, 112.5, 115.0, 417.5, 420.0, 422.5] and all(r["ok"] for r in meta)
          and s.run[0]["meta_seconds"] == 250.0 and META_S == 21600.0 and ended(s) == (0, "end_utc", 500.0),
          ([at(r) for r in meta], ended(s)))

    async def stood_still() -> SimpleNamespace:
        def stall() -> None:
            _VIRTUAL_NOW[0] += 95.0

        return await session("stood_still", _Venue(), two, 400.0, script=[(100.7, stall)])

    s = virtual(stood_still)
    per_sample = sorted(collections.Counter(at(r) for r in of(s, "quote_snapshot")).items())
    check("a process that stood still for 95 s: the samples it missed are not made up - one sample of 4 rows when "
          "it goes on, the next on the old schedule - and no request was saved up: one leaves at once and the next "
          "2.5 s later",
          per_sample[:6] == [(30.0, 4), (60.0, 4), (90.0, 4), (195.7, 4), (210.0, 4), (240.0, 4)]
          and [t for t in s.times if t > 100.7][:3] == [195.7, 198.2, 200.7] and paced(s.times)
          and ended(s) == (0, "end_utc", 400.0), (per_sample[:6], [t for t in s.times if t > 100.7][:3], ended(s)))

    async def close_hangs() -> SimpleNamespace:
        venue = _Venue()
        venue.close_hangs = True
        return await session("close_hangs", venue, two, 50.0)

    s = virtual(close_hangs)
    check("a connection that does not close: the end of the run waits 10 s for it and no longer",
          ended(s) == (0, "end_utc", 60.0) and [r["event"] for r in s.run[-2:]]
          == ["survey_close_timeout", "survey_stopped"], ended(s))

    async def sampler_alone(jump_at: float | None, jump: float = 7200.0, row_takes: float = 0.0,
                            writer_ahead: float = 0.0) -> list[float]:
        """The sampler for 100 s; at `jump_at` the wall clock jumps by `jump` seconds. Writing a row takes
        `row_takes` seconds, in which the process stands still. The writer's own clock reads `writer_ahead`
        seconds more than the gate's: a process that stood still between the two readings."""
        t0, wall, name = time.monotonic(), [0.0], f"sampler_{jump_at}_{jump}_{row_takes}_{writer_ahead}"

        def utcnow() -> datetime:
            return day0 + timedelta(seconds=time.monotonic() - t0 + wall[0])

        class Slow(DailyJsonl):
            def write(self, event: str, **fields: Any) -> None:
                super().write(event, **fields)
                _VIRTUAL_NOW[0] += row_takes

        gate = RequestGate(day0 + timedelta(seconds=100.0), utcnow, t0 + 100.0, elapsed=time.monotonic)
        if jump_at is not None:
            asyncio.get_running_loop().call_later(jump_at, lambda: wall.__setitem__(0, jump))
        with contextlib.suppress(SurveyOver, asyncio.TimeoutError):
            await asyncio.wait_for(sample_loop(
                QuoteWorkers(None, two, STREAM_STAKE), two,
                Slow(tmp / name, lambda: utcnow() + timedelta(seconds=writer_ahead)), gate.time_of_row), 200.0)
        return sorted({at(r) for r in rows(tmp / name)})

    sampled = [virtual(lambda: sampler_alone(None)), virtual(lambda: sampler_alone(40.7)),
               virtual(lambda: sampler_alone(40.7, -3600.0))]
    check("the sampler on its own takes no sample at or after the end time, also when the wall clock jumps past "
          "it; and when the wall clock is set back by an hour it takes none after the 100 s it was started for "
          "(its rows then carry the times of the clock that was set back)",
          sampled == [[30.0, 60.0, 90.0], [30.0], [-3540.0, -3510.0, 30.0]], sampled)
    slow = virtual(lambda: sampler_alone(None, row_takes=4.0))
    check("the sampler looks at the end before every row: when a row takes 4 s to write, the sample of four rows "
          "that begins 10 s before the end stops after its third, and no row is of the end time or later",
          slow == [30.0, 34.0, 38.0, 42.0, 60.0, 64.0, 68.0, 72.0, 90.0, 94.0, 98.0], slow)
    ahead = virtual(lambda: sampler_alone(None, writer_ahead=15.0))
    check("the time of a sample row is the one the gate gave it when the lanes had been read, not the time of the "
          "writing: with a writer whose own clock reads 15 s more - a process that stood still before it wrote - "
          "the rows are of 30, 60 and 90 s, and none is of the end time or later",
          ahead == [30.0, 60.0, 90.0], ahead)

    async def taken_up() -> list[tuple[Any, ...]]:
        """A price, a contract list and the trading times, asked for by a client whose rows are written by a
        writer with a clock that reads 500 s more than the gate's: past the end, which is 100 s away."""
        t0 = time.monotonic()

        def utcnow() -> datetime:
            return day0 + timedelta(seconds=time.monotonic() - t0)

        client = VenueClient(_Venue(), RequestGate(day0 + timedelta(seconds=100.0), utcnow, math.inf))
        logger = DailyJsonl(tmp / "taken_up", lambda: utcnow() + timedelta(seconds=500.0))
        await client.connect()
        await poll_side(client, ("EURUSD", 15, "risefall"), "CALL", 1, logger)
        await fetch_metadata(client, ["EURUSD"], logger)
        await client.close()
        return [(r["event"], at(r), r.get("invalid_reason"), r.get("ok")) for r in rows(tmp / "taken_up")]

    taken = virtual(taken_up)
    check("a row that holds what the venue said - a price, a contract list, the trading times - carries the time "
          "at which the answer was taken up, not the time of the writing: with a writer whose own clock is past "
          "the end, the three rows are of 2.5, 5 and 7.5 s",
          taken == [("quote_poll", 2.5, None, None), ("contracts_for", 5.0, None, True),
                    ("trading_times", 7.5, None, True)], taken)

    readings = [0]

    def stepping() -> datetime:
        """A wall clock that goes on a second at every reading."""
        readings[0] += 1
        return day0 + timedelta(seconds=readings[0] - 1.0)

    stepped, given = RequestGate(day0 + timedelta(seconds=10.5), stepping, math.inf), []
    for _ in range(20):
        moment = stepped.time_of_row()
        if moment is None:
            break
        given.append((moment - day0).total_seconds())
    check("the time that the gate gives a row is the reading by which it judged the row, and it reads the wall "
          "clock once for it: with a clock that goes on a second at every reading and an end 10.5 s away, eleven "
          "rows get the times 0 to 10 s, the twelfth reading is past the end, and no time is of the end or later",
          given == [float(k) for k in range(11)] and readings[0] == 12, (given, readings[0]))

    async def event_rows() -> tuple[dict[str, int], int]:
        events = EventRows(DailyJsonl(tmp / "events", lambda: day0))
        for _ in range(100):
            events("quote_stream_error", {"pair": "EURUSD"})
            await asyncio.sleep(1.0)
        return events.counts, len(rows(tmp / "events"))

    counted = virtual(event_rows)
    check("an event that repeats every second is written 30 times a minute and counted every time",
          counted == ({"quote_stream_error": 100}, 60), counted)

    # 13. forbidden requests through the client itself
    async def forbidden_frames() -> tuple[list[str], int]:
        venue, out = _Venue(), []
        client = VenueClient(venue, RequestGate(far, lambda: day0, math.inf))
        await client.connect()
        for frame in ({"forget": "p1"}, {"buy": "p1", "price": 1}, {"authorize": "token"},
                      {**proposal_payload("EURUSD", 15, "CALL"), "amount": 10.0}):
            try:
                await client.request(frame)
                out.append("left")
            except ForbiddenRequest:
                out.append("refused")
            except DerivAPIError:
                out.append("left, and the venue refused it")
        bare = GatedSocket(await venue.dial(), RequestGate(far, lambda: day0, math.inf))
        try:
            await bare.send(json.dumps(proposal_payload("EURUSD", 15, "CALL")))
            out.append("left")
        except ForbiddenRequest:
            out.append("refused")
        await client.close()
        return out, len(venue.frames)

    check("through the client itself: the end of a subscription, a purchase, a login and a price request at "
          "another stake are refused before they leave, and so is one of the survey's own requests that has not "
          "had its turn", virtual(forbidden_frames) == (["refused"] * 5, 0))

    # 14. the rows and the files
    async def not_finite() -> SimpleNamespace:
        venue = _Venue()
        venue.quote = lambda f: {"ask_price": 1.0, "payout": float("inf") if f["contract_type"] == "PUT" else 1.8}
        return await session("not_finite", venue, two, 100.0, script=[
            (40.7, lambda: setattr(venue, "lane_quote", {"ask_price": "Infinity", "payout": 1.85}))])

    def quote(row: dict[str, Any]) -> tuple[Any, ...]:
        return (*(row[field] for field in QUOTE_FIELDS), row["invalid_reason"])

    s = virtual(not_finite)
    no_quote = (None, None, None, "non_finite_quote")
    put_polls = [quote(r) for r in of(s, "quote_poll") if r["contract_type"] == "PUT"]
    lane_rows = {(at(r), r["contract_type"]): quote(r) for r in of(s, "quote_snapshot")}
    check("a quote with a number that is not finite is no quote: a poll row with an infinite payout has none, so "
          "its side gets no lane; a lane's row with an infinite ask has none; and every line is still JSON",
          len(put_polls) == 10 and set(put_polls) == {no_quote} and s.stopped["lanes"] == 2
          and [ct for _, _, ct in s.subs] == ["CALL", "CALL"] and len(lane_rows) == 6
          and lane_rows.get((30.0, "CALL")) == (1.0, 1.85, 1.0 / 1.85, None)
          and lane_rows.get((30.0, "PUT")) == (None, None, None, "no_quote_received")
          and lane_rows.get((60.0, "CALL")) == lane_rows.get((90.0, "CALL")) == no_quote
          and ended(s) == (0, "end_utc", 100.0),
          (len(put_polls), set(put_polls), s.stopped["lanes"], sorted(lane_rows.items())[:3]))
    unit = ("EURUSD", 15, "risefall")
    fine = poll_row(unit, "CALL", 1, {"proposal": {"ask_price": 1.0, "payout": 1.85, "id": "x"}}, 1.0)
    check("a priced poll row has the ask, the payout and the breakeven, ask / payout",
          (fine["ask"], fine["payout"], fine["invalid_reason"], fine["proposal_id_present"])
          == (1.0, 1.85, None, True) and abs(fine["live_breakeven"] - 1.0 / 1.85) < 1e-12)
    others = [poll_row(unit, "PUT", 2, None, 12.0), poll_row(unit, "PUT", 2, {"proposal": None}, 1.0),
              poll_row(unit, "PUT", 2, DerivAPIError("refused", code="MarketIsClosed"), 1.0),
              poll_row(unit, "PUT", 2, DerivAPIError("timeout"), 15000.0)]
    check("a poll row says why it has no quote: the run ended, no proposal in the answer, a refusal with the "
          "venue's code, or no answer (no code)",
          [(r["invalid_reason"], r.get("error_code")) for r in others]
          == [("run_ended", None), ("no_proposal_in_response", None), ("request_failed", "MarketIsClosed"),
              ("request_failed", None)] and all(r["payout"] is None for r in others))

    log, clock = tmp / "files", [datetime(2026, 11, 17, 23, 59, 58, tzinfo=timezone.utc)]
    log.mkdir()
    (log / "2026-11-17.jsonl").write_text('{"event": "kept"}\n{"event": "quote_poll", "seq": 3')
    writer = DailyJsonl(log, lambda: clock[0])
    writer.write("first", n=1)
    writer.write("second", worth=float("inf"), nested={"a": [float("nan"), 1.5]})
    clock[0] += timedelta(seconds=4)
    writer.write("third")
    DailyJsonl(log, lambda: clock[0]).write("fourth")
    lines = (log / "2026-11-17.jsonl").read_text().split("\n")
    try:
        whole = [strict(line) for i, line in enumerate(lines) if line and i != 1]
    except ValueError as exc:
        whole = [{"event": str(exc)}]
    second = next((r for r in whole if r.get("event") == "second"), {})
    check("a torn last line that a killed process left stays a line of its own; the rows after it are whole, and "
          "the first of them says so", lines[1] == '{"event": "quote_poll", "seq": 3' and lines[-1] == ""
          and [r.get("event") for r in whole] == ["kept", "survey_torn_line", "first", "second"]
          and whole[1].get("file") == "2026-11-17.jsonl", lines[:3])
    check("every line is JSON: a number that JSON cannot hold is written as its name",
          (second.get("worth"), second.get("nested")) == ("inf", {"a": ["nan", 1.5]}), [r.get("event") for r in whole])
    next_day = log / "2026-11-18.jsonl"
    check("a row goes to the file of its own UTC day, and a file that ends in a whole line gets no repair",
          next_day.exists() and [strict(line)["event"] for line in next_day.read_text().splitlines()]
          == ["third", "fourth"])

    # 15. the command line: the end time, the lock, and their order
    now = datetime(2026, 11, 17, 16, 0, tzinfo=timezone.utc)
    ok = parse_end_utc("2026-11-24T16:00:00Z", now) == now + timedelta(days=7)
    for bad in ("2026-11-24T16:00:00", "next week", "2026-11-24T16:00:01Z"):
        try:
            parse_end_utc(bad, now)
            ok = False
        except ValueError:
            pass
    check("--end-utc: a UTC time at most 7 days ahead; one second more, a naive or an unreadable time is refused", ok)
    lock_path, closed_file = tmp / ".connection.lock", tmp / "shut.lock"
    shut.add(str(closed_file))
    tried = [attempt(lambda: take_connection_lock(closed_file)),
             attempt(lambda: sys.audit("open", str(archive_lock), None, os.O_RDWR))]
    shut.discard(str(closed_file))
    check("the connection lock is the archive's one file; in these checks it is a file of their own, and this "
          "process cannot open the archive's: the lock function is refused a file that is shut to the process, "
          "before that file exists, and the archive's lock is shut",
          archive_lock == Path("/mnt/data/deriv/.connection.lock") and CONNECTION_LOCK == lock_path
          and tried == ["this file is not to be opened in the smoke"] * 2 and not closed_file.exists(),
          (str(archive_lock), str(CONNECTION_LOCK), tried))
    held = take_connection_lock(lock_path)
    check("the connection lock is taken once: a second taker is refused while it is held",
          held is not None and take_connection_lock(lock_path) is None)

    def run_main(*argv: str, what: Callable[..., Any] = cleared_to_run) -> tuple[Any, str]:
        """What is decided before the first connection for this command line: the exit code ("would run" when
        there is none) and what was printed. The survey itself is never started from here."""
        said = io.StringIO()
        p = parser()
        with contextlib.redirect_stdout(said), contextlib.redirect_stderr(said):
            try:
                code: Any = what(p, p.parse_args(["--log-dir", str(tmp / "never"), *argv]))
            except SystemExit as exc:
                code = exc.code
        return "would run" if code is None else code, said.getvalue()

    def utc(**ahead: float) -> str:
        return (datetime.now(timezone.utc) + timedelta(**ahead)).isoformat()

    def boot(**ahead: float) -> str:
        """An --end-boot of this boot of the machine, this far ahead."""
        return f"{boot_id()}:{boot_seconds() + timedelta(**ahead).total_seconds():.3f}"

    code, said = run_main("--end-utc", utc(seconds=-1), "--end-boot", boot(days=1))
    check("an end time that has passed ends with exit code 0 even while another tool holds the lock",
          code == 0 and "has passed" in said, (code, said))
    code, said = run_main("--end-utc", utc(days=1), "--end-boot", boot(seconds=-1))
    check("and so does an end that has passed on the boot clock while the wall clock says that a day is left: "
          "the process that a supervisor starts after the clock was set back",
          code == 0 and "has passed" in said, (code, said))
    day = ("--end-utc", utc(days=1), "--end-boot", boot(days=1))
    code, said = run_main(*day)
    check("with time left and the lock held elsewhere: exit code 75, the lock is named, and nothing is written",
          code == EXIT_BUSY and str(lock_path) in said and not (tmp / "never").exists(), (code, said))
    other_boot = f"{'0' * 8}-0000-0000-0000-{'0' * 12}:900.0"
    usage = [run_main(*bad) for bad in (("--end-utc", utc(days=7, hours=1), "--end-boot", boot(days=1)),
                                        (*day, "--pairs", "EURUSD,XXXYYY"),
                                        (*day, "--ladder", "10,50"),
                                        (*day, "--connection-lock", str(tmp / "other.lock")),
                                        (),
                                        ("--end-utc", utc(days=1)),
                                        ("--end-utc", utc(days=1), "--end-boot", boot(days=7, hours=1)),
                                        ("--end-utc", utc(days=1), "--end-boot", other_boot),
                                        ("--end-utc", utc(days=1), "--end-boot", "tomorrow"),
                                        ("--end-utc", utc(days=1), "--end-boot", f"{boot_id()}:nan"))]
    whys = ("--end-utc is more than 7 days ahead", "XXXYYY", "--ladder", "--connection-lock", "are required",
            "are required", "--end-boot is more than 7 days ahead", "another boot", "is not <boot id>:<seconds>",
            "is not <boot id>:<seconds>")
    check("a command line that cannot work ends with exit code 2 and says why, before the lock is looked at; "
          "there is no stake ladder to ask for, and no other lock to name; the end on the boot clock must be "
          "given, be of this boot of the machine, be a number, and be no more than 7 days ahead",
          [code for code, _ in usage] == [EXIT_USAGE] * 10 and not (tmp / "other.lock").exists()
          and all(why in said for (_, said), why in zip(usage, whys, strict=True)),
          [(code, said.replace(boot_id(), "<this boot>")[-90:]) for code, said in usage])
    said_io = io.StringIO()
    with contextlib.redirect_stdout(said_io), contextlib.redirect_stderr(said_io):
        try:
            code = main(["--end-utc", utc(days=1), "--print-end-boot"])
        except SystemExit as exc:
            code = exc.code
    said = said_io.getvalue().strip()
    name, _, reading = said.rpartition(":")
    try:
        printed = float(reading)
    except ValueError:
        printed = math.nan
    check("--print-end-boot prints the end as a reading of the boot clock, with the name of this boot before it: "
          "a day from now on that clock, and what --end-boot then takes as it stands",
          code == 0 and name == boot_id() and abs(printed - boot_seconds() - 86400.0) < 60.0
          and parse_end_boot(said, boot_id(), boot_seconds()) == printed,
          (code, said.replace(boot_id(), "<this boot>")[-200:]))
    def printed_for(seconds: float) -> str:
        """What --print-end-boot prints after the boot's name for a deadline `seconds` after the boot."""
        return end_boot_text(day0 + timedelta(seconds=seconds), lambda: day0, lambda: 0.0).rpartition(":")[2]

    exact = [timedelta(seconds=100.0 + k * 0.000037).total_seconds() for k in range(3000)]
    texts = [printed_for(x) for x in exact]
    # A reading just under a whole thousandth, whose thousandths the multiplication rounds up to the whole
    # number: cut by floor() alone it would be printed as that thousandth, which is later than the reading.
    edge = next(x for x in (math.nextafter(n / 1000.0, 0.0) for n in range(100001, 200000))
                if math.floor(x * 1000.0) / 1000.0 > x)
    edge_text = end_boot_text(day0, lambda: day0, lambda: edge).rpartition(":")[2]
    check("the reading is printed with three decimals, cut and never rounded up: a deadline of 100.0006 s is "
          "printed as 100.000, and of 3000 deadlines none is later when it is read again, and none is a "
          "thousandth of a second earlier; the same holds for a reading whose thousandths round up to a whole "
          "number when they are multiplied out",
          (printed_for(100.0006), printed_for(100.000999), printed_for(100.001)) == ("100.000", "100.000", "100.001")
          and all(0.0 <= x - float(text) < 0.001 for x, text in zip([*exact, edge], [*texts, edge_text], strict=True)),
          (printed_for(100.0006), printed_for(100.000999), printed_for(100.001), repr(edge), edge_text))
    refused_ends = [run_main(*bad, what=print_end_boot) for bad in ((), ("--end-utc", utc(seconds=-1)),
                                                                    ("--end-utc", utc(days=7, hours=1)))]
    check("it needs an --end-utc that is ahead, and no more than 7 days: otherwise exit code 2",
          [code for code, _ in refused_ends] == [EXIT_USAGE] * 3
          and all(why in said for (_, said), why in zip(refused_ends, ("needs --end-utc", "has passed",
                                                                       "7 days ahead"), strict=True)),
          [(code, said[-60:]) for code, said in refused_ends])
    defaults = parser().parse_args([])
    check("unless asked otherwise the survey is the seven pairs: 70 units a cycle",
          resolve_pairs(defaults.pairs) == seven and len(poll_units(resolve_pairs(defaults.pairs))) == 70)
    os.close(held)
    again_fd = take_connection_lock(lock_path)
    check("once the holder is gone the lock can be taken again", again_fd is not None)
    os.close(again_fd)
    code, said = run_main(*day)
    check("with time left and the lock free the survey may run, and this process then holds the lock",
          code == "would run" and take_connection_lock(lock_path) is None and not (tmp / "never").exists(),
          (code, said))

    async def as_process() -> tuple[Any, ...]:
        """run_survey(), which is what main() runs, for a command line whose end on the boot clock passed a
        second ago while its --end-utc is a day away."""
        venue = _Venue()
        args = SimpleNamespace(log_dir=str(tmp / "process"), end_utc=datetime.now(timezone.utc) + timedelta(days=1),
                               end_boot=boot_seconds() - 1.0, pairs=two, note="as a process")
        try:
            code: Any = await asyncio.wait_for(
                run_survey(args, lambda gate, on_event: VenueClient(venue, gate, on_event)), 600.0)
        except asyncio.TimeoutError:
            code = None                         # it ran on for 600 s: the checks on it then fail
        return code, rows(tmp / "process"), len(venue.frames), round(args.end_boot, 3)

    code, made, reached, end_boot = virtual(as_process)
    check("a process is held to the two ends of its command line and makes none of its own: run_survey() with an "
          "end on the boot clock that has just passed, and a day left by the wall clock, writes its start row - "
          "with that end and the boot it is of - sends nothing, and ends at once with exit code 0 (elapsed_limit)",
          code == 0 and reached == 0 and (made[0]["event"], made[0].get("boot_id"), made[0].get("end_boot_seconds"),
                                          made[0]["note"]) == ("survey_started", boot_id(), end_boot, "as a process")
          and made[0]["seconds_left"] < 0.0 and (made[-1]["event"], made[-1]["reason"], made[-1]["requests_sent"])
          == ("survey_stopped", "elapsed_limit", 0), (code, reached, [r["event"] for r in made]))
    sums = code_checksums()
    check("the code checksums cover this file and the project modules it runs on",
          {"deriv_quote_survey.py", "deriv_quote_workers.py", "deriv_async_client.py", "deriv_client.py",
           "deriv_runtime_core.py", "live_features.py"} <= set(sums) and all(len(v) == 64 for v in sums.values()))

    # 16. all the runs together
    early = [(run.name, run.last, run.released) for run in runs
             if run.last is not None and run.released - run.last < SEND_SPACING_S - 1e-6]
    check("in all 42 runs of the survey that these checks made, the run returned no sooner than 2.5 s after its "
          "last request: a process keeps the connection lock until its next request could have left",
          len(runs) == 42 and not early, (len(runs), early[:3]))
    check("in all these runs the event loop had nothing to report: no failure went unread, no callback failed",
          unread == [], unread[:3])

    print(f"\n{'SMOKE ALL PASS' if not failures else f'SMOKE {len(failures)} FAILURES: {failures}'}")
    return 1 if failures else 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true", help="deterministic checks, no network")
    p.add_argument("--log-dir", help="directory of the daily JSONL files")
    p.add_argument("--end-utc", help="when the survey ends, for example 2026-11-24T16:00:00Z")
    p.add_argument("--end-boot", help="the same end on the boot clock, as --print-end-boot gave it before the "
                                      "first process of the survey was started")
    p.add_argument("--print-end-boot", action="store_true",
                   help="print the --end-boot of --end-utc on this boot of the machine, and do nothing else")
    p.add_argument("--pairs", default=",".join(DERIV_SYMBOLS))
    p.add_argument("--note", default="", help="free text stored in the survey_started row")
    return p


def cleared_to_run(p: argparse.ArgumentParser, args: argparse.Namespace) -> int | None:
    """Everything that is decided before the first connection, in this order: a command line that cannot work
    ends with exit code 2; a survey whose end has passed, on either clock, returns 0; a connection lock that
    another tool holds returns 75. None means the survey may run: `args` then holds the two ends and the pairs
    as values, and this process holds the lock until it ends."""
    if not (args.log_dir and args.end_utc and args.end_boot):
        p.error("--log-dir, --end-utc and --end-boot are required")
    now = datetime.now(timezone.utc)
    try:
        args.end_utc = parse_end_utc(args.end_utc, now)
        args.end_boot = parse_end_boot(args.end_boot, boot_id(), boot_seconds())
        args.pairs = resolve_pairs(args.pairs)
    except (ValueError, SystemExit) as exc:
        p.error(str(exc))
    if args.end_utc <= now or args.end_boot <= boot_seconds():
        print("the end of the survey has passed: nothing to do")
        return 0
    if take_connection_lock(CONNECTION_LOCK) is None:
        print(f"another tool has the connection to Deriv (lock held: {CONNECTION_LOCK})", file=sys.stderr)
        return EXIT_BUSY
    return None


def print_end_boot(p: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """--print-end-boot: the --end-boot of --end-utc, for the command line of every process of one survey."""
    def utcnow() -> datetime:
        return datetime.now(timezone.utc)

    if not args.end_utc:
        p.error("--print-end-boot needs --end-utc")
    try:
        end = parse_end_utc(args.end_utc, utcnow())
        if end <= utcnow():
            raise ValueError("--end-utc has passed")
        print(end_boot_text(end, utcnow))
    except (ValueError, RuntimeError) as exc:
        p.error(str(exc))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if args.smoke:
        return run_smoke()
    if args.print_end_boot:
        return print_end_boot(p, args)
    code = cleared_to_run(p, args)
    return asyncio.run(run_survey(args)) if code is None else code


if __name__ == "__main__":
    sys.exit(main())
