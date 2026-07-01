"""Synchronous Deriv Options API client for demo execution.

The client keeps the transport small and explicit: public market-data sockets use
Deriv's public endpoint, authenticated demo sockets are obtained through the
Options OTP REST endpoint, and every request carries a req_id for correlation.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import websocket


API_BASE = "https://api.derivws.com"
PUBLIC_WS_URL = "wss://api.derivws.com/trading/v1/options/ws/public"


class DerivAPIError(RuntimeError):
    """Raised when Deriv returns an error payload or a malformed response."""


@dataclass(frozen=True)
class DerivEnv:
    app_id: str
    pat: str
    account_id: str

    @classmethod
    def from_env(cls) -> "DerivEnv":
        missing = [k for k in ("DERIV_APP_ID", "DERIV_PAT", "DERIV_ACCOUNT_ID") if not os.environ.get(k)]
        if missing:
            raise DerivAPIError(f"missing required env vars: {', '.join(missing)}")
        return cls(
            app_id=os.environ["DERIV_APP_ID"],
            pat=os.environ["DERIV_PAT"],
            account_id=os.environ["DERIV_ACCOUNT_ID"],
        )


def request_demo_ws_url(env: DerivEnv, timeout: float = 10.0) -> str:
    """Return the authenticated demo WebSocket URL from the Options OTP endpoint."""

    url = f"{API_BASE}/trading/v1/options/accounts/{env.account_id}/otp"
    req = urllib.request.Request(
        url,
        method="POST",
        headers={
            "Deriv-App-ID": env.app_id,
            "Authorization": f"Bearer {env.pat}",
            "Content-Type": "application/json",
        },
        data=b"{}",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise DerivAPIError(f"OTP request failed: HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise DerivAPIError(f"OTP request failed: {exc}") from exc

    data = payload.get("data", payload)
    ws_url = data.get("url") or data.get("websocket_url")
    if not isinstance(ws_url, str) or not ws_url.startswith("wss://"):
        raise DerivAPIError(f"OTP response did not contain a WebSocket URL: {payload!r}")
    if "/demo" not in ws_url:
        raise DerivAPIError("OTP response was not a demo WebSocket URL; refusing buy-capable connection")
    return ws_url


class DerivOptionsClient:
    def __init__(self, url: str, timeout: float = 10.0):
        self.url = url
        self.timeout = timeout
        self.ws: websocket.WebSocket | None = None
        self._req_id = 0

    @classmethod
    def public(cls, timeout: float = 10.0) -> "DerivOptionsClient":
        return cls(PUBLIC_WS_URL, timeout=timeout)

    @classmethod
    def demo_from_env(cls, timeout: float = 10.0) -> "DerivOptionsClient":
        return cls(request_demo_ws_url(DerivEnv.from_env(), timeout=timeout), timeout=timeout)

    @property
    def is_demo(self) -> bool:
        return "/demo" in self.url

    def connect(self) -> None:
        if self.ws is not None:
            return
        self.ws = websocket.create_connection(self.url, timeout=self.timeout)
        self.ws.settimeout(self.timeout)

    def close(self) -> None:
        if self.ws is not None:
            self.ws.close()
            self.ws = None

    def __enter__(self) -> "DerivOptionsClient":
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def request(self, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        if self.ws is None:
            self.connect()
        assert self.ws is not None
        self._req_id += 1
        req_id = self._req_id
        msg = dict(payload)
        msg["req_id"] = req_id
        self.ws.send(json.dumps(msg))
        deadline = time.monotonic() + (timeout or self.timeout)
        while True:
            if time.monotonic() > deadline:
                raise DerivAPIError(f"timeout waiting for req_id={req_id} payload={payload}")
            raw = self.ws.recv()
            resp = json.loads(raw)
            if resp.get("req_id") != req_id:
                continue
            if "error" in resp:
                raise DerivAPIError(f"Deriv error for {payload}: {resp['error']}")
            return resp

    def recv(self, timeout: float | None = None) -> dict[str, Any]:
        if self.ws is None:
            self.connect()
        assert self.ws is not None
        old_timeout = self.ws.gettimeout()
        self.ws.settimeout(timeout or self.timeout)
        try:
            return json.loads(self.ws.recv())
        finally:
            self.ws.settimeout(old_timeout)

    def ping(self) -> dict[str, Any]:
        return self.request({"ping": 1})

    def active_symbols(self) -> dict[str, Any]:
        return self.request({"active_symbols": "brief", "product_type": "basic"})

    def contracts_for(self, symbol: str) -> dict[str, Any]:
        return self.request({"contracts_for": symbol})

    def ticks_history(
        self,
        symbol: str,
        *,
        style: str = "candles",
        count: int = 5000,
        granularity: int | None = 60,
        end: str = "latest",
        adjust_start_time: int = 1,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ticks_history": symbol,
            "style": style,
            "count": int(count),
            "end": end,
            "adjust_start_time": int(adjust_start_time),
        }
        if granularity is not None:
            payload["granularity"] = int(granularity)
        return self.request(payload, timeout=max(self.timeout, 20.0))

    def ticks(self, symbol: str, subscribe: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {"ticks": symbol}
        if subscribe:
            payload["subscribe"] = 1
        return self.request(payload)

    def proposal(
        self,
        *,
        symbol: str,
        contract_type: str,
        amount: float,
        basis: str = "stake",
        currency: str = "USD",
        duration: int = 15,
        duration_unit: str = "m",
    ) -> dict[str, Any]:
        return self.request(
            {
                "proposal": 1,
                "amount": float(amount),
                "basis": basis,
                "contract_type": contract_type,
                "currency": currency,
                "duration": int(duration),
                "duration_unit": duration_unit,
                # Options endpoint schema: required underlying_symbol, additionalProperties:false
                "underlying_symbol": symbol,
            }
        )

    def buy(self, proposal_id: str, price: float) -> dict[str, Any]:
        if not self.is_demo:
            raise DerivAPIError("refusing buy on a non-demo WebSocket URL")
        return self.request({"buy": proposal_id, "price": float(price)})

    def proposal_open_contract(self, contract_id: str | int, subscribe: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {"proposal_open_contract": 1, "contract_id": contract_id}
        if subscribe:
            payload["subscribe"] = 1
        return self.request(payload)

    def forget(self, subscription_id: str) -> dict[str, Any]:
        return self.request({"forget": subscription_id})
