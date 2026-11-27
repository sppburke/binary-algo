#!/usr/bin/env python3
"""
Deriv tick archive downloader (public market-data WebSocket, no authentication).

Extends the archive under <data-dir>/ticks/{SYMBOL}/ with one Parquet file per UTC calendar
day plus a JSON completeness sidecar, in exactly the layout and schema of the original archive:

    ticks/{SYMBOL}/{SYMBOL}_{YYYY-MM-DD}_ticks.parquet      datetime_utc (timestamp[ns, UTC]), price (double)
    ticks/{SYMBOL}/{SYMBOL}_{YYYY-MM-DD}_ticks.meta.json    how the day was fetched and whether it is whole

PROVENANCE
    Fork of sppburke/binary_alpha@b4176ff scripts/download_deriv_ticks.py, which built the original
    archive (2025-08-11 .. 2026-08-07). The completeness model, FX-week rule, window planning and
    sidecar keys are kept. What changed, and why (all measured 2026-11-14 unless noted):

    1. ENDPOINT. The legacy host (wss://ws.derivws.com/websockets/v3) answered HTTP 520. This tool
       uses the documented public endpoint wss://api.derivws.com/trading/v1/options/ws/public
       (developers.deriv.com/llms/api-overview.md: "no authentication, no request body").
       It needs no app_id and no token. Transport is the `websockets` library; the original's
       python_deriv_api dependency (and its unbounded response cache) is gone.

    2. PAGE CAP IS 1000, NOT 5000. A {start, end} request returns at most 1000 ticks and drops the
       HEAD (oldest) when the window holds more. FX majors print one tick per second, so a
       1000-second window that is whole returns exactly 1000 ticks beginning at `start`; anything
       else that comes back full-size is re-requested for its missing head.

    3. OUT-OF-RANGE WINDOWS RETURN THE MOST RECENT TICKS, NOT AN EMPTY RESPONSE. A window in the
       future, or older than the ~365-day retention floor, silently returns the newest 1000 ticks.
       The original trusted whatever came back. Every tick is now checked against the requested
       [start, end]; a response with no tick inside the window is classified `unservable` and is
       never written.

    4. `end` IS IGNORED UNLESS `start` IS ALSO SENT (original finding, re-confirmed on this
       endpoint). Every request sends both.

    5. NEVER LOSE ARCHIVED TICKS. Re-fetching a day unions the new ticks with the file already on
       disk. If a re-fetch disagrees with an archived price, the archived value is kept and the
       disagreement is recorded in the sidecar. (The original replaced the file, so a transiently
       empty window could shrink a day that can no longer be re-downloaded.)

    6. DAY-MAJOR ORDER. All symbols are fetched for one day before the next day starts, so an
       interrupted run leaves a clean front instead of some symbols finished and others untouched.

RATE LIMITS
    Published (developers.deriv.com/docs/limits, fetched 2026-11-14): WebSocket "All other calls"
    share 220 per minute and 14,400 per hour. Best practices: one connection, paced requests,
    exponential backoff after a rejection.
    Measured (2026-11-14, twice, each on a fresh connection): tick history is held to a stricter
    class the limits page does not list. The server accepted exactly 30 requests, rejected the
    31st with "RateLimit: You have reached the rate limit for ticks_history_expensive." and kept
    rejecting until about 60 s after the first request.
    This tool therefore uses one connection and a strict spacing of one request every 2.5 s
    (24 per minute, 1,440 per hour). If the server still rejects a request, all sending stops for
    a cooldown that doubles on repeats (75 s up to 15 min) and the pace drops by a fifth for the
    rest of the run. It never opens a second connection to widen the budget, and it fetches only
    while it holds the lock that says which one tool has a connection to Deriv (one file,
    /mnt/data/deriv/.connection.lock, whatever --data-dir is; no option names another). A run
    stops, instead of sending more requests, once the server has rejected its pace on six separate
    occasions, once three symbol-days in a row have failed, or once fifteen in a row (weekends not
    counted) were refused for what their answers held or got not a single tick from inside their
    windows; a symbol-day stops at its first failed request. A run stopped by any of these exits 1.

EMPTY ANSWERS
    A closed market and a window the server cannot produce at that moment both come back as an
    empty answer. An empty window inside trading hours is therefore asked for three times, and
    then believed only when a second fetch of the day, at least six hours after the first, finds
    it empty again. Until then the day is not complete (the sidecar keeps the time of the
    first fetch in `gaps_first_seen`) and the run ends with exit code 3. After that the sidecar
    has gaps_reconfirmed=true; a weekday on which every window was empty both times is recorded
    as a closed market, any other day stays complete=false with its empty windows listed.
    Either kind of day is fetched one last time by the first default run a week or more after
    its empty windows were first seen and after they were confirmed, in case the server was at
    fault for longer than six hours (`gaps_final_check` records that fetch); after that only an
    explicit --start-date asks for it again.
    Empty windows listed by a sidecar from before version 1.3 count as seen once, at the time
    that sidecar was written, and a day such a sidecar records as closed is not trusted as complete.
    A day that leaves the server's retention while it waits cannot have its second fetch; it
    stays complete=false. A settled day whose last look comes when it has begun to leave
    retention is fetched as far as it is still served: its record is kept (with the last look
    noted) unless ticks the archive lacked come back, and then it says complete=false.

WHAT IS REFUSED
    An answer is used only if it is a history answer: two lists of one length, times in whole
    seconds each later than the one before, prices finite and above zero. Anything else is asked
    for again, and the symbol-day fails if it keeps coming. Nothing is written for a symbol-day
    either when the tick file on disk is not the one its sidecar recorded a checksum for, is
    missing although its sidecar records ticks, cannot be read or holds a value no tick can have
    (the file is left as it is), or when the server returns a window empty in which the archive
    already holds ticks (it cannot then be believed about the other windows it returned empty).
    Just before a tick file is replaced, its sidecar names the file on disk as the current one and
    the new file's checksum as the next (`parquet_sha256_next`), so that a run killed between the
    two replacements, even twice over, leaves an update the next default run completes rather
    than a file it refuses.

EXIT CODES
    0 nothing left to fetch, 1 symbol-days failed or are incomplete (run again), 2 bad arguments,
    3 nothing failed but symbol-days wait for their second fetch (run again six hours later),
    75 another tool holds the connection lock, 76 final persistence unconfirmed after quiet drain,
    130 stopped on request. Exit 76 follows a boot-clock rest under exclusion; it does not
    promise a durable exit-plus-900 deadline when storage is unavailable.
    Final persistence never starts a write later than the drain deadline minus W = 30 s
    (DRAIN_WRITE_WINDOW_SECONDS, on CLOCK_BOOTTIME). After that cutoff it only waits for
    the deadline, then releases. A stop before the cutoff attempts at most one drain write;
    a later stop starts none. A storage call that started in time can overrun the deadline
    by its own duration; release follows its return with no further attempt. A call that
    never returns holds the process (and the lock) until it returns or the process is
    killed, which is the safe direction.

WHAT THE DATA IS
    Deriv's FX feed is a fixed 1-second sampling grid of the MID price: no bid, no ask, no volume.
    Timestamps are UTC; files are partitioned on the UTC calendar day. Deriv's FX week runs
    Monday 00:00 UTC to Friday 21:00 UTC (last tick 20:55:00).

RETENTION
    About 365 days, rolling. History that was not captured is gone; the archive is the only copy
    of anything older than the floor.

USAGE
    python deriv_ticks_download.py --dry-run                      # show the plan and request budget
    python deriv_ticks_download.py                                # resume from the first day not complete
    python deriv_ticks_download.py --start-date 2026-08-08 --end-date 2026-11-13
    python deriv_ticks_download.py --symbols frxEURUSD --start-date 2026-11-02
"""

import argparse
import asyncio
import bisect
import fcntl
import hashlib
import json
import logging
import math
import os
import random
import re
import signal
import sys
import tempfile
import time
from datetime import date as date_cls, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import websockets

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'scripts'))
from deriv_runtime_core_v2 import assert_non_us_deriv_host
from sota_rolling_runtime import canonical_vps_connection_paths

TOOL_VERSION = 'deriv_ticks_download_vps/1.7'

# One connection to Deriv at a time, whichever archive a run writes: every tool that opens one holds
# this file's lock while it runs (this downloader and the quote survey). It is one
# file for all of them and for every run: no option names another.
_DEFAULT_LOCK_PATH, _DEFAULT_ADMISSION_PATH = canonical_vps_connection_paths()
CONNECTION_LOCK = str(_DEFAULT_LOCK_PATH)
CONNECTION_ADMISSION = str(_DEFAULT_ADMISSION_PATH)
UNKNOWN_HISTORY_SECONDS = 60.0

PUBLIC_WS_URL = 'wss://api.derivws.com/trading/v1/options/ws/public'

# Measured: a {start, end} ticks_history response never exceeds this many ticks.
PAGE_CAP = 1000

# One FX tick per second, so a whole window of this many seconds returns exactly PAGE_CAP ticks
# starting at `start`. A full-size response that starts later than `start` had its head dropped.
WINDOW_SECONDS = 1000

SECONDS_PER_DAY = 86400

RETENTION_DAYS = 365
RETENTION_SAFETY_HOURS = 6

# MEASURED 2026-11-14, twice, each on a fresh connection sending 2.5 requests/s: the server accepted
# exactly 30 ticks_history requests, rejected the 31st with
#     RateLimit: You have reached the rate limit for ticks_history_expensive.
# and kept rejecting until about 60 s after the first request. That class is NOT on the published
# limits page, which puts ticks_history under "All other calls" at 220 per minute. The published row
# with the same per-minute figure allows 30 per minute and 1,500 per hour, so the default pace is
# chosen to sit under both numbers.
DEFAULT_REQUESTS_PER_MINUTE = 20.0     # conservative VPS default (hard ceiling 24/min)
MAX_REQUESTS_PER_MINUTE = 24.0         # every message, including time, across tools

# After a rate-limit rejection: stop sending entirely (the pause doubles while rejections keep
# coming) and lower the pace for the rest of the run. The pace never climbs back -- a run that was
# told to slow down stays slowed down -- and a run told so this many separate times stops: by
# then it has paused 75 + 150 + 300 + 600 + 900 s and is down to 7.9 requests per minute.
RATE_LIMIT_COOLDOWN_SECONDS = 75.0
RATE_LIMIT_COOLDOWN_MAX_SECONDS = 900.0
DRAIN_WRITE_WINDOW_SECONDS = 30.0
RATE_LIMIT_SLOWDOWN = 0.8
RATE_LIMIT_MAX_EVENTS = 6

# Requests in flight at once. This only hides latency; the limiter still spaces every send.
MAX_IN_FLIGHT = 2

REQUEST_TIMEOUT_SECONDS = 45
REQUEST_ATTEMPTS = 5
EMPTY_WINDOW_RETRIES = 2

# An empty answer inside trading hours is believed only when a second fetch of the day, at least
# this long after the first, finds the same windows empty.
GAP_RECHECK_SECONDS = 6 * 3600
# ... and the day is fetched one last time once this long has passed since the first, in case the
# server was at fault for longer than those hours.
GAP_FINAL_CHECK_SECONDS = 7 * 86400

# Errors that will not fix themselves on retry.
FATAL_ERROR_CODES = {'InputValidationFailed', 'InvalidSymbol', 'UnknownMethod', 'PermissionDenied'}

# Symbol-days failing one after another mean the server is refusing the requests, the link is down
# or nothing can be saved. The run stops there instead of sending more; the next run resumes.
MAX_CONSECUTIVE_FAILED_DAYS = 3
# A symbol-day refused for what its answer or its file held, or one whose windows gave not a single
# tick (all empty, or ticks from elsewhere), does not count towards that stop: one odd calendar day
# must not block the days after it, and a holiday is all empty for the seven default symbols. Two
# calendar days of that in a row are not one odd day: Christmas and New Year's Day together make
# fourteen when nothing between them needs fetching, so the run stops at the fifteenth. A day that
# needs no request (a weekend) neither adds to that count nor resets it.
MAX_CONSECUTIVE_UNUSABLE_DAYS = 15

DEFAULT_SYMBOLS = [
    'frxEURUSD', 'frxGBPUSD', 'frxUSDJPY', 'frxUSDCHF', 'frxUSDCAD', 'frxAUDUSD', 'frxNZDUSD',
]

# Deriv's FX week closes at 21:00 UTC on Friday (last tick 20:55:00 UTC on every Friday in the
# archive) and reopens Monday 00:00 UTC. The boundary is UTC-fixed; the ET wall time moves.
FRIDAY_CLOSE_HOUR_UTC = 21

TICK_SCHEMA = pa.schema([
    ('datetime_utc', pa.timestamp('ns', tz='UTC')),
    ('price', pa.float64()),
])

log = logging.getLogger('deriv_ticks')


class DerivError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(f'{code}: {message}')
        self.code = code
        self.message = message or ''

    @property
    def is_rate_limit(self) -> bool:
        if self.code:
            return self.code == 'RateLimit'
        text = self.message.lower()
        return 'ratelimit' in text or 'rate limit' in text or 'too many' in text


class UnservableWindow(Exception):
    """The server answered with ticks from outside the requested window (future / beyond retention)."""


class DayRefused(RuntimeError):
    """This day cannot be written as things stand. Only that day fails; the run goes on."""


class UnusableAnswer(DayRefused):
    """The server answered, but with content this tool refuses to write."""


class ArchiveDamaged(DayRefused):
    """The tick file on disk is not the one its sidecar recorded, or holds values no tick can have."""


class ServerContradicted(RuntimeError):
    """The server returned a window empty in which the archive holds ticks. The day is not written,
    and like any request that was not served it counts towards stopping the run."""


class RateLimitExhausted(RuntimeError):
    """The server kept rejecting the pace of this run. The run stops; a later run starts afresh."""


def boot_seconds() -> float:
    return time.clock_gettime(time.CLOCK_BOOTTIME)


def utc_seconds() -> float:
    return time.time()


class AdmissionState:
    """Durable admission history, used only inside this process's connection-lock lifetime.

    Deadlines of this boot use CLOCK_BOOTTIME, including time in suspension. UTC is for other
    boots and other tools. Across boots an implausible wall-clock wait is capped at 900 + 60 s.
    """

    def __init__(self, connection_lock):
        self.connection_lock = connection_lock       # keep our own exclusion alive
        self.path = Path(CONNECTION_ADMISSION)
        self.boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        now, boot = utc_seconds(), boot_seconds()
        self.last_request = None
        self.not_before = connection_lock.admission_taken_boot + UNKNOWN_HISTORY_SECONDS
        try:
            state = json.loads(self.path.read_text())
            for key in ('recorded_utc', 'boottime', 'not_before_utc', 'not_before_boot'):
                if type(state[key]) not in (int, float) or not math.isfinite(state[key]):
                    raise ValueError(f'invalid {key}')
            for key in ('last_request_utc', 'last_request_boot'):
                if state[key] is not None and (type(state[key]) not in (int, float) or not math.isfinite(state[key])):
                    raise ValueError(f'invalid {key}')
            for key in ('recorded_utc', 'not_before_utc', 'last_request_utc'):
                # A broad fixed UTC range (2000 through 2100), independent of wall-clock steps.
                if state[key] is not None and not 946684800 <= state[key] <= 4102444800:
                    raise ValueError(f'implausible {key}')
            if (state['last_request_utc'] is None) != (state['last_request_boot'] is None):
                raise ValueError('inconsistent last request')
            if (not isinstance(state['boot_id'], str)
                    or re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', state['boot_id']) is None
                    or not isinstance(state['writer'], dict)):
                raise ValueError('missing boot or writer')
            for key in ('tool', 'version'):
                if not isinstance(state['writer'][key], str) or not state['writer'][key].strip():
                    raise ValueError(f'invalid writer {key}')
            pid = state['writer']['pid']
            if type(pid) is not int or not 0 < pid <= 4194304:
                raise ValueError('invalid writer pid')
            if (not 0 <= state['boottime'] <= 3155760000 or state['not_before_boot'] > state['boottime'] + 960.0
                    or state['not_before_boot'] < 0
                    or (state['last_request_boot'] is not None
                        and not 0 <= state['last_request_boot'] <= state['boottime'])):
                raise ValueError('impossible boot-clock history')
            if (state['not_before_utc'] > state['recorded_utc'] + 960.0
                    or (state['last_request_utc'] is not None
                        and state['last_request_utc'] > state['recorded_utc'])):
                raise ValueError('impossible wall-clock history')
            if state['boot_id'] == self.boot_id:
                if state['boottime'] > boot:
                    raise ValueError('current-boot history is from the future')
                self.last_request = state['last_request_boot']
                self.not_before = state['not_before_boot']
                if abs((now - state['recorded_utc']) - (boot - state['boottime'])) > 1.0:
                    log.warning('admission: wall-clock step detected; using this boot\'s deadlines')
            else:
                bound = RATE_LIMIT_COOLDOWN_MAX_SECONDS + UNKNOWN_HISTORY_SECONDS
                self.last_request = (None if state['last_request_utc'] is None else
                                     boot + min(bound - 60.0 / MAX_REQUESTS_PER_MINUTE,
                                                state['last_request_utc'] - now))
                self.not_before = boot + min(bound, max(0.0, state['not_before_utc'] - now))
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            log.warning('admission: unknown history (%s); first request waits 60 s from lock acquisition', exc)
            self.write()                         # replace unreadable history, even if stopped before a send

    def wait(self, interval: float) -> float:
        deadline = self.not_before
        if self.last_request is not None:
            deadline = max(deadline, self.last_request + interval)
        return deadline - boot_seconds()

    def write(self):
        now, boot = utc_seconds(), boot_seconds()
        state = dict(last_request_utc=None if self.last_request is None else now + self.last_request - boot,
                     not_before_utc=now + self.not_before - boot,
                     writer=dict(tool=TOOL_VERSION.split('/')[0], version=TOOL_VERSION.split('/')[1], pid=os.getpid()),
                     boot_id=self.boot_id, recorded_utc=now, boottime=boot,
                     last_request_boot=self.last_request, not_before_boot=self.not_before)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=self.path.parent,
                                             prefix=self.path.name + '.', suffix='.tmp', delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(state, handle, allow_nan=False)
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def sent(self):
        self.last_request = boot_seconds()
        self.write()

    def rest(self, seconds: float):
        self.not_before = max(self.not_before, boot_seconds() + seconds)
        self.write()

    async def finish_rest(self, stop):
        # Freeze the quiet-drain bound before final-write reservations extend not_before.
        # It covers the last send and the rejection-driven deadline, even after wall steps.
        drain_until = max(self.last_request or 0.0, self.not_before) + RATE_LIMIT_COOLDOWN_MAX_SECONDS

        def persist():
            # Synchronous storage cannot be interrupted safely. Never start it near expiry,
            # including another initial attempt after an earlier call consumed the deadline.
            if boot_seconds() > drain_until - DRAIN_WRITE_WINDOW_SECONDS:
                return False
            try:
                self.rest(RATE_LIMIT_COOLDOWN_MAX_SECONDS + UNKNOWN_HISTORY_SECONDS)
                saved = json.loads(self.path.read_text())
            except OSError as exc:
                log.error('admission: final persistence failed: %s', exc)
                return False
            return (saved['not_before_boot'] - boot_seconds() >= RATE_LIMIT_COOLDOWN_MAX_SECONDS
                    and saved['not_before_utc'] - utc_seconds() >= RATE_LIMIT_COOLDOWN_MAX_SECONDS)

        for _ in range(3):
            if persist():
                return True
            await asyncio.sleep(0)       # let installed signal callbacks run between writes
        log.critical('admission: QUIET DRAIN; holding connection lock without sending until boot %.3f', drain_until)
        retry_at = boot_seconds() + 30.0
        stopped = False
        while boot_seconds() < drain_until:
            if stop.is_set() and not stopped:
                stopped = True
                if boot_seconds() <= drain_until - DRAIN_WRITE_WINDOW_SECONDS:
                    log.warning('admission: QUIET DRAIN stop requested; one final durable write')
                    if persist():
                        log.info('admission: QUIET DRAIN final write confirmed; normal release')
                        return True
                else:
                    log.warning('admission: QUIET DRAIN stop requested inside write window; no write')
                log.warning('admission: QUIET DRAIN rest unconfirmed; no more retries; retaining lock to deadline')
            if (not stopped and boot_seconds() >= retry_at
                    and boot_seconds() <= drain_until - DRAIN_WRITE_WINDOW_SECONDS):
                log.info('admission: QUIET DRAIN retrying durable write')
                if persist():
                    log.info('admission: QUIET DRAIN durable rest recovered; normal release')
                    return True
                retry_at = boot_seconds() + 30.0
            # Short asynchronous waits keep SIGTERM/SIGINT responsive. A stop never cuts rest.
            await asyncio.sleep(min(1.0, max(0.0, drain_until - boot_seconds())))
        log.critical('admission: QUIET DRAIN elapsed; releasing lock with unconfirmed persistence, exit 76')
        return False


class RateLimiter:
    """
    Strict minimum spacing between requests.

    A rate-limit rejection is the server saying "wait", so it pauses ALL sending for a cooldown
    (doubling while rejections keep coming) and lowers the pace for the rest of the run.
    """

    def __init__(self, per_minute: float):
        self._interval = 60.0 / min(per_minute, MAX_REQUESTS_PER_MINUTE)
        self._next_allowed = 0.0
        self._lock = asyncio.Lock()
        self._cooldown = RATE_LIMIT_COOLDOWN_SECONDS
        self._last_rejection = None
        self._clean_streak = 0
        self.rejections = 0
        self.slowdowns = 0
        self.admission = None

    async def send(self, ws, payload):
        # The lock makes the spacing strict even with several requests waiting at once. The loop
        # re-checks after sleeping because a rejection may have pushed _next_allowed further out.
        async with self._lock:
            loop = asyncio.get_running_loop()
            while True:
                wait = max(self._next_allowed - loop.time(),
                           self.admission.wait(60.0 / MAX_REQUESTS_PER_MINUTE)
                           if self.admission is not None else 0.0)
                if wait <= 0:
                    break
                await asyncio.sleep(wait)
            if self.exhausted:
                raise RateLimitExhausted(
                    f'the server rejected the request pace on {self.slowdowns} separate occasions')
            if self.admission is not None:
                self.admission.sent()            # durable before any bytes can leave
            try:
                await ws.send(payload)
            finally:
                # Include time spent writing state and sending: the next tool cannot count from
                # an earlier reservation, even when send fails after possibly sending bytes.
                self._next_allowed = max(self._next_allowed, loop.time() + self._interval)
                if self.admission is not None:
                    self.admission.sent()

    def rejected(self):
        """Record a rate-limit rejection: pause everything, then continue at a lower pace."""
        now = asyncio.get_running_loop().time()
        self.rejections += 1
        self._clean_streak = 0
        # Rejections arriving together (requests that were already in flight) are one event.
        burst = self._last_rejection is not None and now - self._last_rejection < 5.0
        self._last_rejection = now
        if burst:
            if self.admission is not None:
                self.admission.rest(max(0.0, self._next_allowed - now))
            return
        self.slowdowns += 1
        self._interval /= RATE_LIMIT_SLOWDOWN
        pause = self._cooldown
        self._cooldown = min(self._cooldown * 2, RATE_LIMIT_COOLDOWN_MAX_SECONDS)
        self._next_allowed = max(self._next_allowed, now + pause)
        if self.admission is not None:
            self.admission.rest(pause)           # synchronous, before any sleep or cancellation
        log.warning('rate-limit rejection #%d: pausing all requests %.0f s, then continuing at %.1f requests/min',
                    self.rejections, pause, self.per_minute)

    def accepted(self):
        # A long clean stretch resets the cooldown ladder. It never restores the pace.
        self._clean_streak += 1
        if self._clean_streak >= 200:
            self._cooldown = RATE_LIMIT_COOLDOWN_SECONDS

    @property
    def per_minute(self) -> float:
        return 60.0 / self._interval

    @property
    def exhausted(self) -> bool:
        return self.slowdowns >= RATE_LIMIT_MAX_EVENTS


class Connection:
    """One WebSocket shared by all requests, multiplexed on req_id."""

    def __init__(self, url: str):
        self.url = url
        self.ws = None
        self._reader = None
        self._pending = {}
        self._req_id = 0
        self.limiter = None

    async def open(self):
        self.ws = await websockets.connect(
            self.url, max_size=None, open_timeout=30, ping_interval=20, ping_timeout=30,
            user_agent_header='deriv-tick-archive/1.0',
        )
        self._reader = asyncio.create_task(self._read_loop())

    async def _read_loop(self):
        error = ConnectionError('connection closed')
        try:
            async for raw in self.ws:
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                venue_error = msg.get('error')
                if venue_error:
                    rejection = DerivError(venue_error.get('code'), venue_error.get('message'))
                    log.warning('venue rejection code=%r: %s', rejection.code, rejection.message)
                    if rejection.is_rate_limit:
                        self.limiter.rejected()  # record even late answers, before waking a waiter
                fut = self._pending.pop(msg.get('req_id'), None)
                if fut is not None and not fut.done():
                    fut.set_result(msg)
        except Exception as exc:  # connection dropped mid-stream
            error = ConnectionError(f'connection lost: {type(exc).__name__}: {exc}')
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(error)
            self._pending.clear()

    async def request(self, payload: dict, timeout: float) -> dict:
        if self.ws is None:
            raise ConnectionError('not connected')
        self._req_id += 1
        req_id = self._req_id
        fut = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        try:
            if self.limiter is None:
                raise RuntimeError('connection has no admission limiter')
            await self.limiter.send(self.ws, json.dumps(dict(payload, req_id=req_id)))
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(req_id, None)

    async def close(self):
        if self._reader is not None:
            self._reader.cancel()
            try:
                await self._reader
            except (asyncio.CancelledError, Exception):
                pass
            self._reader = None
        if self.ws is not None:
            try:
                await asyncio.wait_for(self.ws.close(), 10)
            except Exception:
                pass
            self.ws = None


def is_fx_closed(win_start: int, win_end: int) -> bool:
    """
    True if the window lies entirely outside Deriv's FX trading week (both ends measured, UTC):
      OPEN  -- Monday 00:00 UTC.   CLOSE -- Friday 21:00 UTC.
    Testing the window START against the Friday close keeps this conservative: a window that
    straddles 21:00 UTC is still fetched, so no real tick is ever skipped.
    """
    start = datetime.fromtimestamp(win_start, timezone.utc)
    end = datetime.fromtimestamp(win_end, timezone.utc)
    if start.weekday() >= 5 and end.weekday() >= 5:
        return True
    if start.weekday() == 4 and start.hour >= FRIDAY_CLOSE_HOUR_UTC:
        return True
    return False


def clean_symbol(symbol: str) -> str:
    """frxEURUSD -> EURUSD (the on-disk folder convention)."""
    return symbol[3:] if symbol.startswith('frx') else symbol


def atomic_write(path: Path, write_fn):
    """Temp file + os.replace: an interrupted run can never leave a truncated file at the final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_day_ticks(path: Path) -> dict:
    """Existing day file -> {epoch_second: price}. Missing file -> {} (strictly via O_NOFOLLOW regular-file read)."""
    import io as _io_r
    import stat as _st_pq
    if path.is_symlink():
        raise ArchiveDamaged(f'refusing symlinked tick parquet: {path}')
    try:
        fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return {}
    try:
        if not _st_pq.S_ISREG(os.fstat(fd).st_mode):
            raise ArchiveDamaged(f'tick parquet is not a regular file: {path}')
        chunks: list[bytes] = []
        while True:
            b = os.read(fd, 1 << 20)
            if not b:
                break
            chunks.append(b)
        raw = b''.join(chunks)
    finally:
        os.close(fd)
    table = pq.read_table(_io_r.BytesIO(raw), columns=['datetime_utc', 'price'])
    ns = table.column('datetime_utc').cast(pa.int64()).to_numpy()
    if (ns % 1_000_000_000 != 0).any():
        raise RuntimeError(f'{path}: sub-second timestamps are not expected in this archive')
    secs = (ns // 1_000_000_000).tolist()
    return dict(zip(secs, table.column('price').to_numpy().tolist()))


def write_day_ticks(path: Path, rows: dict, before_replace=None):
    """Write the day's ticks atomically. `before_replace(tmp)` runs once the new file is complete
    under its temporary name and before it replaces the old one."""
    epochs = np.array(sorted(rows), dtype=np.int64)
    prices = np.array([rows[e] for e in epochs.tolist()], dtype=np.float64)
    table = pa.Table.from_arrays(
        [pa.array(epochs * 1_000_000_000, type=pa.timestamp('ns', tz='UTC')), pa.array(prices, type=pa.float64())],
        schema=TICK_SCHEMA,
    )

    def write(tmp):
        pq.write_table(table, tmp, compression='snappy')
        if before_replace is not None:
            before_replace(tmp)
    atomic_write(path, write)


def file_sha256(path: Path) -> str:
    import stat as _st_sh
    if path.is_symlink():
        raise ArchiveDamaged(f'refusing symlinked file for sha256: {path}')
    digest = hashlib.sha256()
    fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not _st_sh.S_ISREG(os.fstat(fd).st_mode):
            raise ArchiveDamaged(f'not a regular file for sha256: {path}')
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            digest.update(block)
    finally:
        os.close(fd)
    return digest.hexdigest()


def parse_history(response: dict) -> tuple:
    """
    (times, prices) of a tick-history answer. Raises KeyError, TypeError or ValueError for anything
    that is not one: a `history` object holding two lists of one length, `times` in whole seconds,
    each later than the one before, and `prices` finite and above zero. Measured 2026-11-15
    (checks/reply_shape_result.json): times arrive as JSON integers and prices as JSON numbers,
    and a closed-market window has `"prices": []` and `"times": []`.
    """
    history = response['history']
    times, prices = history['times'], history['prices']
    if not isinstance(times, list) or not isinstance(prices, list) or len(times) != len(prices):
        raise ValueError('times and prices are not two lists of one length')
    for moment in times:
        if isinstance(moment, bool) or not isinstance(moment, int):
            raise ValueError(f'a time that is not a whole second: {moment!r}')
    for price in prices:
        if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
            raise ValueError(f'a price that is not a finite number above zero: {price!r}')
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise ValueError('times that are not in increasing order')
    return times, [float(price) for price in prices]


def gaps_first_seen(meta: dict):
    """When the empty windows listed in a sidecar were first seen, or None if it does not say."""
    try:
        seen = datetime.fromisoformat(meta.get('gaps_first_seen') or meta['written_at'])
    except (KeyError, TypeError, ValueError):
        return None
    return seen if seen.tzinfo else seen.replace(tzinfo=timezone.utc)


def gaps_confirmed(meta: dict) -> bool:
    """
    True when a second fetch, at least six hours after the first, found the listed windows empty
    again. Only version 1.3 on applies that rule, and it records gaps_first_seen with it; the same
    flag in an older sidecar came from two fetches minutes apart and is not taken at its word.
    """
    return meta.get('gaps_reconfirmed') is True and bool(meta.get('gaps_first_seen'))


def recheck_due(meta) -> 'datetime | None':
    """When a day with empty windows may be fetched the second time; None if it does not wait for that."""
    if (not isinstance(meta, dict) or meta.get('complete') or 'tool' not in meta or not meta.get('gaps')
            or gaps_confirmed(meta) or meta.get('clipped_by_retention')
            or meta.get('clipped_by_now') or meta.get('unservable_windows')):
        return None
    seen = gaps_first_seen(meta)
    return seen + timedelta(seconds=GAP_RECHECK_SECONDS) if seen else None


def final_check_due(meta) -> 'datetime | None':
    """When a day whose empty windows were confirmed is to be fetched one last time; None once
    that fetch has been made (`gaps_final_check`) or if no confirmed empty windows are listed."""
    if (not isinstance(meta, dict) or 'tool' not in meta or not meta.get('gaps') or not gaps_confirmed(meta)
            or meta.get('gaps_final_check')):
        return None
    seen = gaps_first_seen(meta)
    return seen + timedelta(seconds=GAP_FINAL_CHECK_SECONDS) if seen else None


def trusted_complete(meta) -> bool:
    """complete=true in a sidecar, except a closed day whose empty windows only a version before
    1.3 called confirmed (two fetches minutes apart): that day still waits for its second fetch."""
    return (isinstance(meta, dict) and bool(meta.get('complete'))
            and not ('tool' in meta and meta.get('market_closed') and meta.get('gaps') and not gaps_confirmed(meta)))


def merge_ticks(existing: dict, fetched: dict) -> tuple:
    """
    Union of what is on disk and what was just fetched. Archived values win on disagreement.
    Returns (merged, conflicts) where conflicts is [[epoch, archived_price, fetched_price], ...].
    """
    merged = dict(fetched)
    conflicts = []
    for epoch, price in existing.items():
        new = merged.get(epoch)
        if new is not None and new != price:
            conflicts.append([epoch, price, new])
        merged[epoch] = price
    return merged, conflicts


class TickDownloader:
    def __init__(self, ticks_dir: Path, url: str, per_minute: float, overwrite: bool = False,
                 window_seconds: int = WINDOW_SECONDS):
        self.ticks_dir = ticks_dir
        self.url = url
        self.overwrite = overwrite
        self.window_seconds = window_seconds
        self.limiter = RateLimiter(per_minute)
        self.conn = None
        self.requests_made = 0
        self.reconnects = 0
        self.rate_limit_hits = 0
        self.stopped_early = False      # a stop rule ended the run with symbol-days not attempted
        self._conn_lock = asyncio.Lock()
        self._generation = 0
        self._started = time.monotonic()

    # ------------------------------------------------------------------ connection

    async def connect(self, max_retries: int = 8):
        for attempt in range(max_retries):
            conn = Connection(self.url)
            conn.limiter = self.limiter
            try:
                await conn.open()
                reply = await conn.request({'time': 1}, REQUEST_TIMEOUT_SECONDS)
                if reply.get('error'):
                    raise DerivError(reply['error'].get('code'), reply['error'].get('message'))
                log.info('connected to %s (server time %s)', self.url,
                         datetime.fromtimestamp(reply['time'], timezone.utc).isoformat())
                self.conn = conn
                return
            except BaseException as exc:
                await conn.close()
                if not isinstance(exc, Exception):
                    raise
                log.warning('connect attempt %d/%d failed: %s: %s', attempt + 1, max_retries, type(exc).__name__, exc)
                if attempt == max_retries - 1:
                    raise
                # Exponential backoff with jitter (Deriv best practices).
                await asyncio.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))

    async def _reconnect(self, seen_generation: int):
        """Rebuild the connection unless a concurrent task already did."""
        async with self._conn_lock:
            if seen_generation != self._generation:
                return
            self.reconnects += 1
            log.warning('reconnecting (reconnect #%d)', self.reconnects)
            if self.conn is not None:
                await self.conn.close()
                self.conn = None
            await self.connect()
            self._generation += 1

    async def close(self):
        if self.conn is not None:
            await self.conn.close()
            self.conn = None

    # ------------------------------------------------------------------- retention

    @staticmethod
    def retention_floor() -> datetime:
        """Oldest timestamp worth requesting right now, with a safety margin."""
        return (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
                + timedelta(hours=RETENTION_SAFETY_HOURS))

    # ----------------------------------------------------------------------- paths

    def _assert_safe_leaf(self, path: Path) -> Path:
        if self.ticks_dir.is_symlink():
            raise ArchiveDamaged(f'refusing symlinked ticks directory: {self.ticks_dir}')
        if path.parent.is_symlink():
            raise ArchiveDamaged(f'refusing symlinked symbol directory: {path.parent}')
        if path.is_symlink():
            raise ArchiveDamaged(f'refusing symlinked archive file: {path}')
        return path

    def _read_regular_text_nofollow(self, path: Path) -> str:
        self._assert_safe_leaf(path)
        import stat as _st_r
        fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
        try:
            if not _st_r.S_ISREG(os.fstat(fd).st_mode):
                raise ArchiveDamaged(f'not a regular archive file: {path}')
            chunks: list[bytes] = []
            while True:
                b = os.read(fd, 1 << 20)
                if not b:
                    break
                chunks.append(b)
            return b''.join(chunks).decode('utf-8')
        finally:
            os.close(fd)

    def file_for(self, symbol: str, date_str: str) -> Path:
        clean = clean_symbol(symbol)
        return self._assert_safe_leaf(self.ticks_dir / clean / f'{clean}_{date_str}_ticks.parquet')

    def meta_for(self, symbol: str, date_str: str) -> Path:
        return self._assert_safe_leaf(self.file_for(symbol, date_str).with_suffix('.meta.json'))

    def day_is_complete(self, symbol: str, date_str: str) -> bool:
        """A day counts as done only if its sidecar says so -- never on file existence."""
        meta_path = self.meta_for(symbol, date_str)
        if not meta_path.exists():
            return False
        try:
            return trusted_complete(json.loads(self._read_regular_text_nofollow(meta_path)))
        except Exception:
            return False

    def day_needs_fetch(self, symbol: str, date_str: str) -> bool:
        """Default resume: every day that is not complete (same rule as day_is_complete), except
        incomplete days from the original archive (sidecar without a 'tool' field), days whose
        empty windows a second fetch has confirmed, and days whose second fetch is not due yet.
        A day with confirmed empty windows, closed days included, is fetched one last time a week
        after those windows were first seen. A day whose last update stopped before its sidecar was
        written is always fetched, so that the update is completed."""
        try:
            meta = json.loads(self._read_regular_text_nofollow(self.meta_for(symbol, date_str)))
        except (OSError, ValueError):
            return True
        if not isinstance(meta, dict):
            return True
        if isinstance(meta.get('parquet_sha256_next'), str):
            return True
        if 'tool' not in meta:
            return False
        now = datetime.now(timezone.utc)
        last_look = final_check_due(meta)
        if last_look is not None:
            # Due as long as part of the day is still served: what is left can still hold ticks.
            day_end = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc) + timedelta(days=1)
            return last_look <= now and day_end > self.retention_floor()
        if trusted_complete(meta) or gaps_confirmed(meta):
            return False
        due = recheck_due(meta)
        return due is None or due <= now

    # -------------------------------------------------------------------- fetching

    async def _request(self, symbol: str, start: int, end: int) -> tuple:
        """One ticks_history call with pacing, timeout, retry, backoff and reconnect."""
        last_exc = None
        attempt = 0            # counts real failures only; a rate-limit rejection is a wait, not a failure
        rate_limit_waits = 0

        async def failed_attempt():
            log.warning('%s [%d,%d] attempt %d/%d: %s', symbol, start, end, attempt, REQUEST_ATTEMPTS, last_exc)
            await asyncio.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))

        while attempt < REQUEST_ATTEMPTS:
            if self.limiter.exhausted:           # nothing more is sent in a run the server has had enough of
                raise RateLimitExhausted(
                    f'the server rejected the request pace on {self.limiter.slowdowns} separate occasions')
            self.requests_made += 1
            generation = self._generation
            try:
                if self.conn is None:
                    raise ConnectionError('not connected')
                response = await self.conn.request({
                    'ticks_history': symbol,
                    'start': int(start),        # mandatory: `end` is ignored without it
                    'end': int(end),
                    'count': PAGE_CAP,
                    'style': 'ticks',
                }, REQUEST_TIMEOUT_SECONDS)
            except (asyncio.TimeoutError, ConnectionError, OSError, websockets.exceptions.WebSocketException) as exc:
                last_exc = exc
                attempt += 1
                log.warning('%s [%d,%d] attempt %d/%d: %s: %s', symbol, start, end, attempt,
                            REQUEST_ATTEMPTS, type(exc).__name__, exc)
                if attempt < REQUEST_ATTEMPTS:
                    try:
                        await self._reconnect(generation)
                    except Exception as rexc:
                        log.error('reconnect failed: %s', rexc)
                    await asyncio.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))
                continue

            error = response.get('error')
            if error:
                last_exc = DerivError(error.get('code'), error.get('message'))
                if last_exc.code in FATAL_ERROR_CODES:
                    raise last_exc
                if last_exc.is_rate_limit:
                    self.rate_limit_hits += 1
                    rate_limit_waits += 1
                    log.warning('%s [%d,%d]: %s', symbol, start, end, last_exc)
                    continue                     # the retry waits out the cooldown inside the send limiter
                attempt += 1
                await failed_attempt()
                continue

            self.limiter.accepted()
            # Anything that is not a history answer is not an answer to the question asked, so it
            # is asked again instead of being read as "no ticks" or written to the archive.
            try:
                return parse_history(response)
            except (KeyError, TypeError, ValueError) as exc:
                last_exc = RuntimeError(f'malformed reply ({type(exc).__name__}: {exc})')
                attempt += 1
                await failed_attempt()

        raise RuntimeError(f'{symbol}: request [{start},{end}] gave up after {attempt} failed attempt(s) and '
                           f'{rate_limit_waits} rate-limit wait(s): {last_exc}')

    async def fetch_window(self, symbol: str, start: int, end: int) -> dict:
        """
        Every tick in [start, end], working around head truncation.

        Raises UnservableWindow when the server substitutes its most recent ticks for a window it
        cannot serve (future, or older than retention). Returns {} for a genuinely empty window.
        """
        collected = {}
        cursor_end = end
        while cursor_end >= start:
            times, prices = await self._request(symbol, start, cursor_end)
            if not times:
                break
            inside = [(t, p) for t, p in zip(times, prices) if start <= t <= cursor_end]
            if len(inside) != len(times):
                if inside:
                    raise UnusableAnswer(
                        f'{symbol} [{start},{cursor_end}]: response mixes in-window and out-of-window ticks '
                        f'({len(inside)}/{len(times)} inside) -- refusing to guess')
                raise UnservableWindow(
                    f'{symbol} [{start},{cursor_end}]: server returned ticks from {times[0]}..{times[-1]} instead')
            collected.update(inside)
            if len(times) < PAGE_CAP:
                break                      # window fully covered
            oldest = min(times)
            if oldest <= start:
                break                      # full-size response that starts at `start`: whole window
            cursor_end = oldest - 1        # head was dropped; go back for it
        return collected

    async def download_day(self, symbol: str, date_str: str) -> dict:
        """Fetch one UTC calendar day, merge with what is on disk, write ticks + sidecar."""
        try:
            previous_meta = json.loads(self._read_regular_text_nofollow(self.meta_for(symbol, date_str)))
        except (OSError, ValueError):
            previous_meta = {}
        if not isinstance(previous_meta, dict):
            previous_meta = {}
        day = datetime.strptime(date_str, '%Y-%m-%d').date()
        day_start = int(datetime.combine(day, datetime.min.time()).replace(tzinfo=timezone.utc).timestamp())
        day_end = day_start + SECONDS_PER_DAY - 1          # UTC days are always exactly 24h

        floor_epoch = int(self.retention_floor().timestamp())
        now_epoch = int(datetime.now(timezone.utc).timestamp())

        # A day clipped by retention or still in progress is NOT complete. Its data is kept, and
        # the sidecar says complete=false so a later run repairs it.
        clipped_start = day_start < floor_epoch
        clipped_end = day_end > now_epoch
        eff_start = max(day_start, floor_epoch)
        eff_end = min(day_end, now_epoch)
        if eff_start >= eff_end:
            return {'ticks': 0, 'complete': False, 'skipped': True}
        # A settled day that has begun to leave the server's retention: what is left of it can
        # still hold ticks the archive lacks, but cannot confirm anything. Its record is replaced
        # only if this fetch finds such ticks (an update left unfinished is always completed).
        keep_record_unless_improved = (clipped_start and not isinstance(previous_meta.get('parquet_sha256_next'), str)
                                       and (trusted_complete(previous_meta) or gaps_confirmed(previous_meta)))

        # The file on disk is read before anything is asked for. New ticks are merged only into a
        # file that is the one its sidecar recorded and that holds nothing a tick cannot be: a
        # changed file, merged and given a fresh checksum, would pass for sound from then on; a
        # missing one would let empty answers turn a day with ticks into a closed day.
        path = self.file_for(symbol, date_str)
        recorded = previous_meta.get('parquet_sha256')
        recorded_ticks = previous_meta.get('ticks')
        try:
            on_disk = file_sha256(path) if path.exists() else None
            existing = read_day_ticks(path)
        except Exception as exc:
            raise ArchiveDamaged(f'{path.name} cannot be read ({type(exc).__name__}: {exc}) -- left as it is') from exc
        if on_disk is None and (isinstance(recorded, str) or (isinstance(recorded_ticks, int)
                                                              and not isinstance(recorded_ticks, bool) and recorded_ticks > 0)):
            raise ArchiveDamaged(f'{path.name} is missing although its sidecar records {recorded_ticks} tick(s) -- '
                                 f'nothing written')
        if isinstance(recorded, str) and on_disk is not None and on_disk != recorded:
            if on_disk != previous_meta.get('parquet_sha256_next'):
                raise ArchiveDamaged(f'{path.name} is not the file its sidecar recorded a checksum for -- left as it is')
            # The sidecar named this checksum just before the file was replaced: an update that
            # was cut off before its sidecar was written. This fetch completes it.
            log.warning('%s: completing an update that stopped after the tick file was replaced', path.name)
        unusable = sum(1 for second, price in existing.items()
                       if not day_start <= second <= day_end or not (math.isfinite(price) and price > 0))
        if unusable:
            raise ArchiveDamaged(f'{path.name} holds {unusable} tick(s) outside its day or without a usable '
                                 f'price -- left as it is')

        is_fx = symbol.startswith('frx')
        planned, closed_windows = [], 0
        cursor = eff_start
        while cursor <= eff_end:
            win_end = min(cursor + self.window_seconds - 1, eff_end)
            if is_fx and is_fx_closed(cursor, win_end):
                closed_windows += 1
            else:
                planned.append((cursor, win_end))
            cursor = win_end + 1

        sem = asyncio.Semaphore(MAX_IN_FLIGHT)
        abandoned = False          # a window failed for good: the day fails, so nothing more is sent for it

        async def fetch_one(win_start: int, win_end: int):
            nonlocal abandoned
            async with sem:
                if abandoned:
                    return None
                try:
                    ticks = await self.fetch_window(symbol, win_start, win_end)
                    # An empty answer inside trading hours is a holiday or a transient failure.
                    # Ask again before believing it.
                    for _ in range(EMPTY_WINDOW_RETRIES if not ticks else 0):
                        await asyncio.sleep(1.0)
                        ticks = await self.fetch_window(symbol, win_start, win_end)
                        if ticks:
                            break
                    return (win_start, win_end), ticks, 'data' if ticks else 'empty'
                except UnservableWindow as exc:
                    log.warning('%s', exc)
                    return (win_start, win_end), {}, 'unservable'
                except Exception:
                    abandoned = True
                    raise

        tasks = [asyncio.create_task(fetch_one(a, b)) for a, b in planned]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            # A window failed for good (or a stop was requested): nothing is written for the day.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        fetched, gaps, unservable = {}, [], []
        for (win_start, win_end), ticks, status in results:
            if status == 'data':
                fetched.update(ticks)
            elif status == 'empty':
                gaps.append([win_start, win_end])
            else:
                unservable.append([win_start, win_end])
        gaps.sort()
        unservable.sort()

        stray = [e for e in fetched if not day_start <= e <= day_end]
        if stray:
            raise UnusableAnswer(f'{symbol} {date_str}: {len(stray)} fetched tick(s) fall outside the day -- refusing to write')

        if gaps and existing:
            # An empty answer for a window in which the archive holds ticks is a wrong answer, and a
            # server giving one cannot be believed about the other windows it returned empty.
            have = sorted(existing)
            denied = [gap for gap in gaps if bisect.bisect_left(have, gap[0]) != bisect.bisect_right(have, gap[1])]
            if denied:
                raise ServerContradicted(
                    f'{symbol} {date_str}: {len(denied)} window(s) came back empty in which the archive holds '
                    f'ticks -- nothing written')
        rows, conflicts = merge_ticks(existing, fetched)
        kept_from_disk = len(rows) - len(fetched)
        if keep_record_unless_improved and rows == existing:
            if final_check_due(previous_meta) is not None:
                # This fetch was the day's last look: that is recorded, and nothing else changes.
                looked = dict(previous_meta, gaps_final_check=datetime.now(timezone.utc).isoformat())
                atomic_write(self.meta_for(symbol, date_str), lambda p: p.write_text(json.dumps(looked, indent=2)))
            log.info('%s %s: begins before the retention floor, is already settled and nothing new came back -- '
                     'its record is kept', symbol, date_str)
            return {'ticks': len(rows), 'complete': False, 'skipped': True}

        windows = len(planned)
        now = datetime.now(timezone.utc)
        # A clipped or unservable range cannot confirm anything about the windows inside it.
        settled = not clipped_start and not clipped_end and not unservable
        # Seen before: every window empty now was already listed as empty by the previous fetch.
        earlier = previous_meta.get('gaps') if isinstance(previous_meta.get('gaps'), list) else []
        same_gaps = bool(gaps and settled and 'tool' in previous_meta and all(gap in earlier for gap in gaps))
        first_seen = (gaps_first_seen(previous_meta) if same_gaps else None) or now
        gaps_reconfirmed = same_gaps and (gaps_confirmed(previous_meta)
                                          or (now - first_seen).total_seconds() >= GAP_RECHECK_SECONDS)
        # The last look a week on is this fetch when the windows were confirmed before it and are
        # still empty; it is recorded, so a late second fetch is not taken for it.
        if same_gaps and gaps_confirmed(previous_meta) and (now - first_seen).total_seconds() >= GAP_FINAL_CHECK_SECONDS:
            final_check = now.isoformat()
        else:
            final_check = previous_meta.get('gaps_final_check') if same_gaps and gaps_reconfirmed else None
        # No window was requested (weekend), or every window was empty on two fetches hours apart
        # (holiday): the market was shut. One all-empty fetch alone does not establish that.
        market_shut = windows == 0 or (not rows and len(gaps) == windows and gaps_reconfirmed)
        complete = settled and (market_shut or not gaps)
        recheck_after = (first_seen + timedelta(seconds=GAP_RECHECK_SECONDS)
                         if gaps and settled and not gaps_reconfirmed else None)

        meta = {
            'symbol': symbol,
            'date': date_str,
            'calendar': 'UTC',
            'ticks': len(rows),
            'windows_requested': windows,
            'windows_skipped_closed': closed_windows,
            'gaps': gaps,
            'gaps_first_seen': first_seen.isoformat() if gaps else None,
            'gaps_reconfirmed': gaps_reconfirmed,
            'gaps_final_check': final_check,
            'clipped_by_retention': clipped_start or bool(unservable and not clipped_end),
            'clipped_by_now': clipped_end,
            'market_closed': market_shut,
            'complete': complete,
            'written_at': now.isoformat(),
            # additive keys (absent from the original archive's sidecars)
            'tool': TOOL_VERSION,
            'source_endpoint': self.url,
            'window_seconds': self.window_seconds,
            'unservable_windows': unservable,
            'ticks_fetched': len(fetched),
            'ticks_kept_from_disk_only': kept_from_disk,
            'conflicts_kept_archived': conflicts[:50],
            'conflict_count': len(conflicts),
        }

        meta_path = self.meta_for(symbol, date_str)
        if rows and (rows != existing):
            def announce(tmp):
                # Just before the tick file is replaced, its sidecar names the checksum of the new
                # file. A run killed between the two replacements thus leaves a state the next run
                # recognises as an update to complete, not as a damaged file.
                # It names the file now on disk (checked against the sidecar above) as the current
                # one, so a second interruption leaves both files recognised.
                if isinstance(previous_meta.get('parquet_sha256'), str):
                    pending = dict(previous_meta, parquet_sha256=on_disk, parquet_sha256_next=file_sha256(tmp))
                    atomic_write(meta_path, lambda p: p.write_text(json.dumps(pending, indent=2)))
            write_day_ticks(path, rows, before_replace=announce)
        # The checksum of the tick file as it now stands on disk, for deriv_archive_verify.py.
        meta['parquet_sha256'] = file_sha256(path) if rows else None
        atomic_write(meta_path, lambda p: p.write_text(json.dumps(meta, indent=2)))

        if conflicts:
            log.warning('%s %s: %d re-fetched price(s) disagree with the archive; archived values kept',
                        symbol, date_str, len(conflicts))
        if recheck_after:
            log.warning('%s %s: %d window(s) empty inside trading hours -- to be fetched again after %s UTC',
                        symbol, date_str, len(gaps), recheck_after.strftime('%Y-%m-%d %H:%M'))
        elif gaps and not market_shut:
            log.warning('%s %s: %d window(s) inside trading hours %s -- day stays marked incomplete', symbol, date_str,
                        len(gaps), 'empty again on a second fetch' if gaps_reconfirmed else 'empty in a range cut short')
        return {'ticks': len(rows), 'complete': complete, 'skipped': False, 'gaps': gaps,
                'market_closed': market_shut, 'windows': windows,
                'awaits_recheck': recheck_after is not None, 'gaps_reconfirmed': gaps_reconfirmed,
                # windows were asked for, and not one tick came back from inside them
                'nothing_usable': windows > 0 and not fetched}

    # ------------------------------------------------------------------------ plan

    def pending_days(self, symbols: list, start_day: date_cls, end_day: date_cls,
                     explicit_start: bool = False) -> list:
        """[(date_str, [symbols needing a fetch]), ...] in day-major order."""
        plan = []
        cursor = start_day
        while cursor <= end_day:
            date_str = cursor.strftime('%Y-%m-%d')
            todo = [s for s in symbols if self.overwrite or
                    (not self.day_is_complete(s, date_str) if explicit_start
                     else self.day_needs_fetch(s, date_str))]
            if todo:
                plan.append((date_str, todo))
            cursor += timedelta(days=1)
        return plan

    def estimate_requests(self, plan: list) -> int:
        total = 0
        for date_str, todo in plan:
            day = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            start = int(day.timestamp())
            wins = sum(
                0 if is_fx_closed(a, min(a + self.window_seconds - 1, start + SECONDS_PER_DAY - 1)) else 1
                for a in range(start, start + SECONDS_PER_DAY, self.window_seconds)
            )
            total += wins * len(todo)
        return total

    async def run(self, plan: list) -> dict:
        summary = {s: {'written': 0, 'closed': 0, 'incomplete': 0, 'failed': 0, 'ticks': 0,
                       'awaiting_recheck': 0, 'confirmed_gaps': 0} for s in
                   sorted({s for _, todo in plan for s in todo})}
        total_days = sum(len(todo) for _, todo in plan)
        done = 0
        failed_in_a_row = 0
        unusable_in_a_row = 0

        def stop_for_unusable() -> bool:
            if unusable_in_a_row < MAX_CONSECUTIVE_UNUSABLE_DAYS:
                return False
            log.error('%d symbol-days in a row were refused or got no tick at all from inside their windows: '
                      'stopping this run with %d symbol-day(s) not attempted', unusable_in_a_row, total_days - done)
            self.stopped_early = True
            return True

        for date_str, todo in plan:
            for symbol in todo:
                done += 1
                stats = summary[symbol]
                try:
                    res = await self.download_day(symbol, date_str)
                except Exception as exc:
                    stats['failed'] += 1
                    log.error('%s %s: FAILED: %s: %s', symbol, date_str, type(exc).__name__, exc)
                    if isinstance(exc, RateLimitExhausted):
                        log.error('stopping this run with %d symbol-day(s) not attempted', total_days - done)
                        self.stopped_early = True
                        return summary
                    # A day refused for what its answer or its file on disk contained fails alone and
                    # the run goes on, until that has happened too many times in a row. Anything else
                    # (requests not served, a full disk, a defect) counts towards the earlier stop.
                    if isinstance(exc, DayRefused):
                        failed_in_a_row = 0
                        unusable_in_a_row += 1
                    else:
                        failed_in_a_row += 1
                    if failed_in_a_row >= MAX_CONSECUTIVE_FAILED_DAYS:
                        log.error('%d symbol-days failed in a row: stopping this run with %d symbol-day(s) not attempted',
                                  failed_in_a_row, total_days - done)
                        self.stopped_early = True
                        return summary
                    if stop_for_unusable():
                        return summary
                    continue
                failed_in_a_row = 0
                if res.get('skipped'):
                    continue
                if res.get('windows', 0):          # a day that needed no request says nothing about the server
                    unusable_in_a_row = unusable_in_a_row + 1 if res.get('nothing_usable') else 0
                if res['ticks']:
                    stats['written'] += 1
                    stats['ticks'] += res['ticks']
                elif res.get('market_closed'):
                    stats['closed'] += 1
                if not res['complete']:
                    # Empty windows are not a failure: the day is saved and either waits for its
                    # second fetch or has had it. Anything else incomplete is fetched again next run.
                    stats['awaiting_recheck' if res.get('awaits_recheck') else
                          'confirmed_gaps' if res.get('gaps_reconfirmed') else 'incomplete'] += 1
                elapsed = time.monotonic() - self._started
                log.info('%s %s: %6d ticks  %3d windows  complete=%s  [%d/%d symbol-days, %d requests, %.1f/min avg, %.0f min]',
                         clean_symbol(symbol), date_str, res['ticks'], res.get('windows', 0), res['complete'],
                         done, total_days, self.requests_made, self.requests_made / max(elapsed, 1e-9) * 60, elapsed / 60)
                if stop_for_unusable():
                    return summary
        return summary


def acquire_connection_lock(lock_path: Path):
    import stat as _stat
    if lock_path.parent.is_symlink():
        raise SystemExit(f'connection lock parent symlink refused: {lock_path.parent}')
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if lock_path.parent.is_symlink():
        raise SystemExit(f'connection lock parent symlink refused: {lock_path.parent}')
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        st_open = os.fstat(fd)
        if not _stat.S_ISREG(st_open.st_mode):
            raise OSError(f'connection lock is not a regular file: {lock_path}')
        os.fchmod(fd, 0o600)
        os.set_inheritable(fd, False)
        handle = os.fdopen(fd, 'r+', encoding='utf-8')
        fd = -1
    except BaseException:
        if fd >= 0:
            os.close(fd)
        raise
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        st_live = os.lstat(str(lock_path))
        if (int(st_open.st_dev), int(st_open.st_ino)) != (int(st_live.st_dev), int(st_live.st_ino)):
            raise OSError(f'connection lock inode replaced: {lock_path}')
    except BaseException as exc:
        handle.close()
        if not isinstance(exc, BlockingIOError):
            raise
        log.info('another tool has the connection to Deriv (lock held: %s)', lock_path)
        raise SystemExit(75)
    handle.admission_taken_boot = boot_seconds()
    return handle


def acquire_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    handle = os.fdopen(fd, 'r+', encoding='utf-8')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        log.info('another download or offline refit holds the archive lock (lock held: %s)', lock_path)
        raise SystemExit(75)
    handle.seek(0)
    handle.truncate()
    handle.write(f'{os.getpid()}\n')
    handle.flush()
    return handle


def first_incomplete_day(dl: TickDownloader, symbols: list, floor_day: date_cls, end_day: date_cls) -> date_cls:
    """Earliest day needing a fetch within retention, or the day after the range."""
    first = end_day + timedelta(days=1)
    originals = 0
    cursor = floor_day
    while cursor <= end_day:
        date_str = cursor.isoformat()
        for symbol in symbols:
            if dl.day_needs_fetch(symbol, date_str):
                first = min(first, cursor)
            else:
                try:
                    meta = json.loads(dl._read_regular_text_nofollow(dl.meta_for(symbol, date_str)))
                    originals += int(isinstance(meta, dict) and not meta.get('complete') and 'tool' not in meta)
                except (OSError, ValueError):
                    pass
        cursor += timedelta(days=1)
    if originals:
        log.info('left %d original incomplete symbol-days alone; retry explicitly with --start-date YYYY-MM-DD',
                 originals)
    return first


def awaiting_recheck(dl: TickDownloader, symbols: list, first_day: date_cls, last_day: date_cls) -> list:
    """Due times of the symbol-days in the range that wait for the second fetch of their empty windows."""
    due = []
    cursor = first_day
    while cursor <= last_day:
        for symbol in symbols:
            try:
                when = recheck_due(json.loads(dl._read_regular_text_nofollow(dl.meta_for(symbol, cursor.isoformat()))))
            except (OSError, ValueError):
                continue
            if when is not None:
                due.append(when)
        cursor += timedelta(days=1)
    return due


async def amain(args) -> int:
    data_dir = Path(args.data_dir)
    ticks_dir = data_dir / 'ticks'
    symbols = args.symbols or DEFAULT_SYMBOLS
    dl = TickDownloader(ticks_dir, args.url, args.per_minute, overwrite=args.overwrite, window_seconds=args.window)

    today = datetime.now(timezone.utc).date()
    floor_day = dl.retention_floor().date()
    end_day = (datetime.strptime(args.end_date, '%Y-%m-%d').date() if args.end_date
               else today - timedelta(days=1))          # last UTC day that has fully elapsed
    explicit_start = datetime.strptime(args.start_date, '%Y-%m-%d').date() if args.start_date else None
    if end_day > today or (explicit_start is not None and explicit_start > today):
        print('error: --start-date and --end-date must not be after the current UTC date', file=sys.stderr)
        return 2
    if args.start_date:
        start_day = explicit_start
    else:
        start_day = first_incomplete_day(dl, symbols, floor_day, end_day)
    if start_day < floor_day:
        log.warning('start %s predates retention; clamping to %s', start_day, floor_day)
        start_day = floor_day

    def finished() -> int:
        """Exit code of a run in which nothing failed: 3 while days wait for their second fetch."""
        due = awaiting_recheck(dl, symbols, explicit_start or floor_day, end_day)
        if not due:
            return 0
        log.info('%d symbol-day(s) with empty windows inside trading hours wait for a second fetch, due by %s UTC',
                 len(due), max(due).strftime('%Y-%m-%d %H:%M'))
        return 3

    if start_day > end_day:
        log.info('nothing to do: start %s is after end %s', start_day, end_day)
        return 0 if args.dry_run else finished()

    plan = dl.pending_days(
        symbols,
        start_day,
        end_day,
        explicit_start=bool(getattr(args, 'explicit_start_arg', bool(args.start_date and args.rolling_days is None))),
    )
    est = dl.estimate_requests(plan)
    log.info('UTC day range %s -> %s | %d day(s) with work | %d symbol-days | ~%d requests | ~%.1f h at %.0f requests/min',
             start_day, end_day, len(plan), sum(len(t) for _, t in plan), est, est / args.per_minute / 60, args.per_minute)
    if args.dry_run:
        for date_str, todo in plan[:400]:
            log.info('  %s: %s', date_str, ' '.join(clean_symbol(s) for s in todo))
        return 0
    if not plan:
        return finished()

    # Mandatory non-US jurisdiction fence: refuse ANY Deriv socket/lock on US workstation (quantum)!
    assert_non_us_deriv_host()
    # Keep our connection-lock handle alive through the rest of the run, including its summary.
    connection_lock = acquire_connection_lock(Path(CONNECTION_LOCK))
    stop = asyncio.Event()
    try:
        dl.limiter.admission = AdmissionState(connection_lock)
        lock = acquire_lock(data_dir / '.download.lock')
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)

        async def connect_and_run() -> dict:
            await dl.connect()
            return await dl.run(plan)

        # Connecting is part of what a stop request interrupts: it can take minutes of retries.
        runner = asyncio.create_task(connect_and_run())
        stopper = asyncio.create_task(stop.wait())
        try:
            await asyncio.wait({runner, stopper}, return_when=asyncio.FIRST_COMPLETED)
            interrupted = not runner.done()
            if interrupted:
                log.warning('stop requested: cancelling (files already written are whole; re-run to resume)')
                runner.cancel()
                try:
                    await runner
                except (asyncio.CancelledError, Exception):
                    pass
                summary = {}
            else:
                summary = runner.result()
        except Exception as exc:
            log.error('the run ended early: %s: %s', type(exc).__name__, exc)
            return 1
        finally:
            stopper.cancel()
            runner.cancel()
            try:
                try:
                    await runner
                except (asyncio.CancelledError, Exception):
                    pass
            finally:
                try:
                    await dl.close()
                finally:
                    lock.close()

        log.info('=' * 78)
        for symbol, s in summary.items():
            log.info('  %-10s %4d written  %4d closed  %4d incomplete  %4d failed  %4d awaiting a second fetch  '
                     '%4d with confirmed empty windows  %12s ticks', symbol, s['written'], s['closed'], s['incomplete'],
                     s['failed'], s['awaiting_recheck'], s['confirmed_gaps'], f"{s['ticks']:,}")
        log.info('requests: %d | reconnects: %d | rate-limit rejections: %d | final pace: %.1f/min | elapsed: %.1f min',
                 dl.requests_made, dl.reconnects, dl.rate_limit_hits, dl.limiter.per_minute,
                 (time.monotonic() - dl._started) / 60)
        problems = sum(s['failed'] + s['incomplete'] for s in summary.values())
        if interrupted:
            return 130
        if problems:
            log.warning('%d symbol-day(s) incomplete or failed -- re-run to repair', problems)
            return 1
        if dl.stopped_early:
            log.warning('the run was stopped with symbol-days not attempted -- re-run to continue')
            return 1
        return finished()
    finally:
        try:
            if dl.limiter.rejections and dl.limiter.admission is not None:
                if not await dl.limiter.admission.finish_rest(stop):
                    return 76
        finally:
            connection_lock.close()


_VALID_FX_SYMBOL_RE = re.compile(r'^frx[A-Z]{6}$')


def validate_vps_archive_dir(data_dir: Path, *, symbols: list[str] | None = None) -> Path:
    """Refuse research archive/lab roots (/mnt/data/deriv, /mnt/data/deriv-sota-lab) and symlink escapes unconditionally."""
    p_raw = Path(data_dir)
    if '..' in p_raw.parts:
        raise ValueError(f'--data-dir refuses parent traversal: {data_dir!r}')
    if p_raw.is_symlink():
        raise ValueError(f'--data-dir refuses symlink root: {data_dir!r}')
    dd_res = p_raw.resolve()
    forbidden_roots = (Path('/mnt/data/deriv').resolve(), Path('/mnt/data/deriv-sota-lab').resolve())
    for forbidden in forbidden_roots:
        try:
            dd_res.relative_to(forbidden)
            raise ValueError(f'--data-dir refuses research root {forbidden}: {data_dir!r}')
        except ValueError as exc:
            if 'refuses research root' in str(exc):
                raise
    for sub_name in ('ticks', 'logs'):
        sub_p = p_raw / sub_name
        if sub_p.is_symlink():
            raise ValueError(f'--data-dir refuses symlinked {sub_name}/ directory: {sub_p}')
        if sub_p.exists():
            sub_res = sub_p.resolve()
            for forbidden in forbidden_roots:
                try:
                    sub_res.relative_to(forbidden)
                    raise ValueError(f'--data-dir/{sub_name} refuses research root {forbidden}: {sub_p}')
                except ValueError as exc:
                    if 'refuses research root' in str(exc):
                        raise
            if not sub_res.is_relative_to(dd_res):
                raise ValueError(f'--data-dir/{sub_name} escaped archive root: {sub_p}')
    for sym in (symbols or DEFAULT_SYMBOLS):
        if not isinstance(sym, str) or not _VALID_FX_SYMBOL_RE.fullmatch(sym):
            raise ValueError(f'invalid --symbols identifier refused: {sym!r}')
        sym_dir = p_raw / 'ticks' / clean_symbol(sym)
        if sym_dir.is_symlink():
            raise ValueError(f'refusing symlinked symbol directory: {sym_dir}')
        if sym_dir.exists() and not sym_dir.resolve().is_relative_to(dd_res):
            raise ValueError(f'symbol directory escaped archive root: {sym_dir}')
    return dd_res


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Extend the Deriv tick archive (daily UTC Parquet + sidecar)')
    parser.add_argument('--data-dir', default=str(REPO_ROOT / 'deriv_data' / 'rolling_ticks_archive'), help='operational VPS archive root (ticks/ lives under it)')
    parser.add_argument('--symbols', nargs='+', help=f'Deriv symbols (default: {" ".join(DEFAULT_SYMBOLS)})')
    parser.add_argument('--start-date', help='first UTC date YYYY-MM-DD (default: earliest day needing a fetch)')
    parser.add_argument('--rolling-days', type=int, default=None, help='convenience: fetch trailing N UTC calendar days up to --end-date (mutually exclusive with --start-date)')
    parser.add_argument('--end-date', help='last UTC date YYYY-MM-DD inclusive (default: yesterday UTC)')
    parser.add_argument('--per-minute', type=float, default=DEFAULT_REQUESTS_PER_MINUTE,
                        help=f'requests per minute (default {DEFAULT_REQUESTS_PER_MINUTE:g}, max {MAX_REQUESTS_PER_MINUTE:g})')
    parser.add_argument('--window', type=int, default=WINDOW_SECONDS, help=f'request window seconds (default {WINDOW_SECONDS})')
    parser.add_argument('--url', default=PUBLIC_WS_URL, help='public WebSocket endpoint')
    parser.add_argument('--overwrite', action='store_true', help='re-fetch days already complete (still merges, never drops ticks)')
    parser.add_argument('--dry-run', action='store_true', help='print the plan and request budget, fetch nothing')
    parser.add_argument('--log-file', default=None, help='log file (default: <data-dir>/logs/download_ticks.log)')
    args = parser.parse_args(argv)

    # parser.error() ends with exit code 2: arguments that can never work are not worth a retry.
    if not 0 < args.per_minute <= MAX_REQUESTS_PER_MINUTE:
        parser.error(f'--per-minute must be in (0, {MAX_REQUESTS_PER_MINUTE:g}]: the server rejects the 31st '
                     f'tick-history request inside a minute')
    if not 1 <= args.window <= PAGE_CAP:
        parser.error(f'--window must be in [1, {PAGE_CAP}]: a larger window can exceed the page cap silently')
    if args.url != PUBLIC_WS_URL:
        parser.error(f'--url is strictly pinned to {PUBLIC_WS_URL!r} (refusing {args.url!r})')
    args.explicit_start_arg = bool(args.start_date is not None)
    if args.rolling_days is not None:
        if args.start_date is not None:
            parser.error('--rolling-days and --start-date are mutually exclusive')
        if not 1 <= args.rolling_days <= 360:
            parser.error('--rolling-days must be in [1, 360]')
        end_ref = datetime.strptime(args.end_date, '%Y-%m-%d').date() if args.end_date else (datetime.now(timezone.utc).date() - timedelta(days=1))
        args.start_date = (end_ref - timedelta(days=args.rolling_days - 1)).isoformat()
    try:
        validate_vps_archive_dir(Path(args.data_dir), symbols=args.symbols)
    except ValueError as exc:
        parser.error(str(exc))
    for option, text in (('--start-date', args.start_date), ('--end-date', args.end_date)):
        try:
            datetime.strptime(text or '2000-01-01', '%Y-%m-%d')
        except ValueError:
            parser.error(f'{option} must be a date YYYY-MM-DD, not {text!r}')

    log_file = Path(args.log_file) if args.log_file else Path(args.data_dir) / 'logs' / 'download_ticks.log'
    if '..' in log_file.parts or log_file.is_symlink():
        parser.error(f'--log-file refuses symlink or traversal: {str(log_file)!r}')
    # Refuse symlinks on every existing ancestor of BOTH explicit AND default log_file!
    # (Skip only an outer repo-root alias matching REPO_ROOT.resolve(), e.g. ~/git/binary-algo on vultr.)
    repo_res = REPO_ROOT.resolve()
    cur_p = log_file.parent
    while cur_p != cur_p.parent:
        if cur_p.is_symlink():
            try:
                if cur_p.resolve() == repo_res:
                    cur_p = cur_p.parent
                    continue
            except OSError:
                pass
            parser.error(f'--log-file refuses symlinked ancestor {cur_p}: {str(log_file)!r}')
        cur_p = cur_p.parent
    try:
        lp_res = log_file.resolve()
        for fb in (Path('/mnt/data/deriv').resolve(), Path('/mnt/data/deriv-sota-lab').resolve()):
            try:
                lp_res.relative_to(fb)
                parser.error(f'--log-file refuses research root {fb}: {str(log_file)!r}')
            except ValueError:
                pass
    except OSError:
        pass

    log_file.parent.mkdir(parents=True, exist_ok=True)
    if not log.handlers:
        import stat as _st_log
        log_fd = os.open(str(log_file), os.O_CREAT | os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        try:
            if not _st_log.S_ISREG(os.fstat(log_fd).st_mode):
                raise ValueError(f'log file is not regular: {log_file}')
            os.fchmod(log_fd, 0o600)
            log_stream = os.fdopen(log_fd, 'a', encoding='utf-8')
            log_fd = -1
        finally:
            if log_fd >= 0:
                os.close(log_fd)
        fmt = logging.Formatter('%(asctime)s +0000 | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fmt.converter = time.gmtime
        for handler in (logging.StreamHandler(sys.stderr), logging.StreamHandler(log_stream)):
            handler.setFormatter(fmt)
            log.addHandler(handler)
    log.setLevel(logging.INFO)

    return asyncio.run(amain(args))


if __name__ == '__main__':
    raise SystemExit(main())
