#!/usr/bin/env python3
"""Collect contiguous historical ticks from Deriv synthetic indices via the PUBLIC WebSocket API.

No auth required (ticks_history has auth_required=0). Pages `ticks_history` backwards via the
`end` epoch, 5000 ticks/request (server hard cap), throttled under the 220 req/min general limit.
The assembled series is the actual realized output sequence of Deriv's price generator for that
symbol -- exactly what we want to test for randomness / structure.

Verified live this session (2026-06-09): wss://ws.derivws.com works with app_id=1089 (no auth),
ticks_history returns history.{prices,times}, count capped at 5000, paging via `end` works.

Usage:
    python3 syn_collect.py <outdir> SYM=COUNT [SYM=COUNT ...]
    python3 syn_collect.py syn_data R_100=1500000 1HZ100V=1500000
"""
import asyncio, json, os, sys, time
import numpy as np
import websockets

APP_ID = os.environ.get("DERIV_APP_ID", "1089")
URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
SLEEP = 0.30  # ~200 req/min, under the 220/min general cap


async def collect_symbol(ws, symbol, target):
    batches_p, batches_t = [], []
    end = "latest"
    got, req_id, last_oldest = 0, 0, None
    t0 = time.time()
    while got < target:
        req_id += 1
        await ws.send(json.dumps({
            "ticks_history": symbol, "end": end, "count": 5000,
            "style": "ticks", "req_id": req_id,
        }))
        while True:
            msg = json.loads(await ws.recv())
            mt = msg.get("msg_type")
            if mt == "history":
                break
            if "error" in msg:
                print(f"  ERROR [{symbol}]:", msg["error"].get("message"), flush=True)
                return None
            # ignore pings / other frames
        hist = msg.get("history", {})
        p, t = hist.get("prices", []), hist.get("times", [])
        if not t:
            print(f"  [{symbol}] empty batch -> history floor reached", flush=True)
            break
        oldest = t[0]  # batch is chronological ascending
        if last_oldest is not None and oldest >= last_oldest:
            print(f"  [{symbol}] no older data (oldest={oldest}) -> floor reached", flush=True)
            break
        last_oldest = oldest
        batches_p.append(np.asarray(p, dtype=np.float64))
        batches_t.append(np.asarray(t, dtype=np.int64))
        got += len(t)
        end = str(int(oldest) - 1)  # page strictly backwards
        if (got // 5000) % 10 == 0:
            print(f"  [{symbol}] {got:,} ticks  oldest_epoch={oldest}  ({time.time()-t0:.0f}s)", flush=True)
        await asyncio.sleep(SLEEP)
        if len(t) < 5000:  # short batch usually = floor; loop will catch it next round
            pass
    if not batches_t:
        return None
    P = np.concatenate(batches_p[::-1])  # batches collected newest->oldest; reverse to ascending
    T = np.concatenate(batches_t[::-1])
    order = np.argsort(T, kind="stable")
    T, P = T[order], P[order]
    keep = np.concatenate(([True], np.diff(T) != 0))  # dedup identical epochs
    T, P = T[keep], P[keep]
    return P, T


async def main():
    outdir = sys.argv[1]
    jobs = [a.split("=") for a in sys.argv[2:]]
    os.makedirs(outdir, exist_ok=True)
    async with websockets.connect(URL, max_size=2**24, ping_interval=20, ping_timeout=30) as ws:
        for sym, cnt in jobs:
            print(f"=== collecting {sym} target={int(cnt):,} ===", flush=True)
            res = await collect_symbol(ws, sym, int(cnt))
            if res is None:
                print(f"  {sym}: NO DATA", flush=True)
                continue
            P, T = res
            out = os.path.join(outdir, f"{sym}_ticks.npz")
            np.savez_compressed(out, prices=P, times=T)
            dt = np.diff(T)
            span = (T[-1] - T[0]) / 86400.0
            print(f"  SAVED {out}: {len(P):,} ticks | span {span:.2f} days | "
                  f"dt median={np.median(dt):.0f}s min={dt.min()}s max={dt.max()}s | "
                  f"price [{P.min():.4f}, {P.max():.4f}]", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
