"""DERIV LEAD-PROBE — the $0 gating experiment for the Deriv latency-arb thesis.

QUESTION IT ANSWERS: does an INDEPENDENT EUR/USD feed LEAD Deriv's published frxEURUSD mid by a margin larger than
Deriv's +1-tick (~1s) entry lag? If not, there is no latency-arb to chase and you spend nothing further (see memory
`deriv-fx-data-source`: Deriv self-MAKES the binary tick at <=1 update/s, (bid+ask)/2, spike-filtered; its WS is
Cloudflare anycast so 'colocate near Deriv' is moot; T&C 1.16.1.2/1.16.3 also make a detectable latency-arb
clawback-able). This tool MEASURES the lead before you risk a cent.

METHOD (deriv-faithful, surrogate-null-gated like the rest of this repo):
  capture  : live-capture Deriv frxEURUSD, stamping EVERY message with local time.time_ns() AT RECEIPT (Deriv's own
             `epoch` is whole-Unix-second-quantized -> useless for sub-second lead; we use RECEIPT time). Writes JSONL.
  dukascopy: download free Dukascopy EURUSD tick .bi5 files for the captured UTC window (NO signup) -> parquet of
             (market_ns, mid). Dukascopy ticks carry millisecond MARKET timestamps.
  analyze  : align both mids onto a common UTC grid (default 100ms), log-return them, cross-correlate over +-max-lag,
             find the peak-correlation lag, and run a CIRCULAR-SHIFT SURROGATE NULL (rotation preserves each series'
             autocorrelation exactly while destroying cross-correlation -- same philosophy as surrogate_null.py) to
             get a significance band + p-value. Reports the lead in ms and a GO/NO-GO vs the entry-lag threshold.
  selfcheck: offline validation -- recovers a known injected lead, calibrates the null on independent walks, and
             decodes a real historical Dukascopy hour. Run this first; it needs no open market.

INTERPRETATION: the measured "reference leads Deriv by L ms" = how stale Deriv's published mid is *as you receive it*
relative to a fine-grained market reference (= Deriv publish/filter delay + YOUR network leg). That is exactly the
arb-relevant quantity for you (you act on what you receive). L includes your own receive latency, so it is the lead
YOU experience, an UPPER bound on the structural lead a co-located actor would see. A real, significant L still may be
UN-tradeable after the +1-tick entry, the spike filter (which strips the very stray ticks a fast feed leads on), and
the anti-arb clause -- gate on net-of-frictions exploitability, not on L>0 alone.

Run (markets must be OPEN for capture; FX closes ~Fri 21:00 UTC -> Sun 21:00 UTC):
  ~/binary-algo-venv/bin/python deriv_lead_probe.py selfcheck
  ~/binary-algo-venv/bin/python deriv_lead_probe.py capture   --minutes 120 --out captures/deriv_$(date +%Y%m%d).jsonl
  ~/binary-algo-venv/bin/python deriv_lead_probe.py dukascopy --from-capture captures/deriv_YYYYMMDD.jsonl --out captures/duka_YYYYMMDD.parquet
  ~/binary-algo-venv/bin/python deriv_lead_probe.py analyze   --target captures/deriv_YYYYMMDD.jsonl --reference captures/duka_YYYYMMDD.parquet --grid-ms 100 --max-lag-ms 3000
You may instead pass ANY live reference to `analyze --reference X.{parquet,csv}` (cols: ts_ns,mid) captured on the SAME
box (e.g. an OANDA-practice or cTrader-FIX stream) for a clean live-vs-live cross-check.
"""
import sys, os, json, time, argparse, struct, lzma, urllib.request
import numpy as np, pandas as pd

ROOT = "/home/sean/git/binary-algo"
WS_URL = "wss://ws.derivws.com/websockets/v3?app_id=1089"      # app_id 1089 = no auth/account needed (verified)
SYMBOL = "frxEURUSD"
DUKA_SYMBOL = "EURUSD"; DUKA_POINT = 1e5                        # EURUSD is 5-decimal -> raw int price / 1e5
T0 = time.time()
def hb(m): print(f"[{time.time()-T0:.0f}s] {m}", flush=True)


# ----------------------------------------------------------------------------- capture (live Deriv)
async def _capture(minutes, out):
    import websockets
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    deadline = time.time() + minutes * 60
    n = 0; f = open(out, "a", buffering=1)
    hb(f"capture -> {out} for {minutes}m (compression OFF; recv-time stamped)")
    while time.time() < deadline:
        try:
            async with websockets.connect(WS_URL, compression=None, open_timeout=20, ping_interval=15) as ws:
                await ws.send(json.dumps({"ticks": SYMBOL, "subscribe": 1}))
                while time.time() < deadline:
                    raw = await _recv_timeout(ws, 30)
                    recv_ns = time.time_ns()                    # STAMP AT RECEIPT (the whole point)
                    msg = json.loads(raw)
                    if msg.get("msg_type") == "tick" and "tick" in msg:
                        t = msg["tick"]
                        q = t.get("quote"); b = t.get("bid"); a = t.get("ask")
                        mid = q if q is not None else (None if (b is None or a is None) else (b + a) / 2)
                        if mid is None: continue
                        f.write(json.dumps({"recv_ns": recv_ns, "epoch": t.get("epoch"),
                                            "quote": q, "bid": b, "ask": a, "mid": mid}) + "\n")
                        n += 1
                        if n % 200 == 0: hb(f"  {n} ticks (last mid {mid})")
                    elif "error" in msg:
                        code = msg["error"].get("code")
                        hb(f"  ERROR {code}: {msg['error'].get('message')}")
                        if code == "MarketIsClosed":
                            hb("  market is CLOSED -> run during open FX hours (Sun 21:00 - Fri 21:00 UTC). stopping.")
                            f.close(); return n
                        await asyncio.sleep(2)
        except Exception as e:
            hb(f"  reconnect after {type(e).__name__}: {e}"); await asyncio.sleep(2)
    f.close(); hb(f"capture done: {n} ticks -> {out}"); return n


async def _recv_timeout(ws, t):
    import asyncio as _a
    return await _a.wait_for(ws.recv(), timeout=t)


# ----------------------------------------------------------------------------- dukascopy (free reference)
_TICK = struct.Struct(">3i2f")                                 # ms_offset, ask_pts, bid_pts, ask_vol, bid_vol (big-endian)

def _decompress_bi5(raw):
    if not raw: return b""
    try: return lzma.decompress(raw)                            # FORMAT_AUTO handles .lzma(alone)
    except lzma.LZMAError:
        return lzma.LZMADecompressor(format=lzma.FORMAT_ALONE).decompress(raw)

def _duka_hour(dt_utc):
    """Download + decode ONE Dukascopy EURUSD hour. dt_utc = pandas Timestamp truncated to the hour (UTC).
    Returns (market_ns[int64], mid[float]) arrays. Dukascopy MONTHS ARE 0-INDEXED."""
    url = (f"https://datafeed.dukascopy.com/datafeed/{DUKA_SYMBOL}/"
           f"{dt_utc.year}/{dt_utc.month-1:02d}/{dt_utc.day:02d}/{dt_utc.hour:02d}h_ticks.bi5")
    try:
        with urllib.request.urlopen(url, timeout=30) as r: raw = r.read()
    except Exception as e:
        hb(f"  duka {url} -> {type(e).__name__}: {e}"); return np.array([], "int64"), np.array([], float)
    data = _decompress_bi5(raw)
    if len(data) < _TICK.size: return np.array([], "int64"), np.array([], float)
    hour_ns = int(dt_utc.value)                                 # ns at hour start (UTC)
    ms, ask, bid = [], [], []
    for off in range(0, len(data) - len(data) % _TICK.size, _TICK.size):
        t_ms, a_pts, b_pts, _, _ = _TICK.unpack_from(data, off)
        ms.append(t_ms); ask.append(a_pts); bid.append(b_pts)
    ms = np.asarray(ms, "int64"); ask = np.asarray(ask, float); bid = np.asarray(bid, float)
    market_ns = hour_ns + ms * 1_000_000
    mid = (ask + bid) / 2.0 / DUKA_POINT
    return market_ns, mid

def dukascopy(window_ns, out):
    t0, t1 = window_ns
    hrs = pd.date_range(pd.to_datetime(t0, unit="ns", utc=True).floor("h"),
                        pd.to_datetime(t1, unit="ns", utc=True).ceil("h"), freq="h")
    hb(f"dukascopy: {len(hrs)} hour-file(s) {hrs[0]} .. {hrs[-1]}")
    parts_t, parts_m = [], []
    for h in hrs:
        mt, md = _duka_hour(h)
        if len(mt): parts_t.append(mt); parts_m.append(md); hb(f"  {h:%Y-%m-%d %Hh} -> {len(mt)} ticks")
    if not parts_t: hb("dukascopy: NO ticks (closed window?)"); return 0
    mt = np.concatenate(parts_t); md = np.concatenate(parts_m); o = np.argsort(mt, kind="mergesort")
    df = pd.DataFrame({"ts_ns": mt[o], "mid": md[o]})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True); df.to_parquet(out)
    hb(f"dukascopy done: {len(df)} ticks -> {out}"); return len(df)


# ----------------------------------------------------------------------------- analyze (align + xcorr + null)
def _load(path, kind):
    if kind == "deriv":
        df = pd.read_json(path, lines=True)[["recv_ns", "mid"]].rename(columns={"recv_ns": "ts_ns"})
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "ts_ns" not in df or "mid" not in df:
        raise ValueError(f"{path}: need columns ts_ns,mid (got {list(df.columns)})")
    return df[["ts_ns", "mid"]].dropna().astype({"ts_ns": "int64", "mid": float}).sort_values("ts_ns")

def _grid_returns(df, grid_ms, t0, t1):
    """Last-value resample onto the common UTC grid, then log-returns aligned to that grid."""
    idx = pd.date_range(pd.to_datetime(t0, unit="ns", utc=True).ceil(f"{grid_ms}ms"),
                        pd.to_datetime(t1, unit="ns", utc=True).floor(f"{grid_ms}ms"), freq=f"{grid_ms}ms")
    s = pd.Series(df["mid"].values, index=pd.to_datetime(df["ts_ns"].values, unit="ns", utc=True))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s.reindex(s.index.union(idx)).sort_index().ffill().reindex(idx)
    return np.log(s.values), idx

def _xcorr(zt, zr, K):
    """r(k)=corr(target[t], ref[t+k]) for k in [-K,K]. peak at k<0 => reference LEADS target by |k| steps."""
    out = np.full(2 * K + 1, np.nan)
    for i, k in enumerate(range(-K, K + 1)):
        if k < 0:   a, b = zt[-k:], zr[:len(zr) + k]
        elif k > 0: a, b = zt[:len(zt) - k], zr[k:]
        else:       a, b = zt, zr
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 30: out[i] = float(np.corrcoef(a[m], b[m])[0, 1])
    return out

def analyze(target, reference, grid_ms, max_lag_ms, n_null, entry_lag_ms, out, seed=0):
    dt = _load(target, "deriv"); dr = _load(reference, "ref" if reference.endswith((".parquet", ".csv")) else "deriv")
    t0 = max(dt["ts_ns"].min(), dr["ts_ns"].min()); t1 = min(dt["ts_ns"].max(), dr["ts_ns"].max())
    if t1 <= t0: hb("analyze: NO time overlap between target and reference"); return None
    hb(f"analyze: overlap {pd.to_datetime(t0,unit='ns',utc=True)} .. {pd.to_datetime(t1,unit='ns',utc=True)} "
       f"({(t1-t0)/1e9/60:.1f} min); target {len(dt):,} ref {len(dr):,}")
    lt, idx = _grid_returns(dt, grid_ms, t0, t1); lr, _ = _grid_returns(dr, grid_ms, t0, t1)
    rt = np.diff(lt); rr = np.diff(lr)                          # grid log-returns
    fin = np.isfinite(rt) & np.isfinite(rr)
    if fin.sum() < 200: hb(f"analyze: too few overlapping grid returns ({fin.sum()})"); return None
    zt = (rt - np.nanmean(rt[fin])) / (np.nanstd(rt[fin]) + 1e-12)
    zr = (rr - np.nanmean(rr[fin])) / (np.nanstd(rr[fin]) + 1e-12)
    zt[~fin] = np.nan; zr[~fin] = np.nan
    K = max(1, int(round(max_lag_ms / grid_ms)))
    r = _xcorr(zt, zr, K)
    lags_ms = np.arange(-K, K + 1) * grid_ms
    pk = int(np.nanargmax(np.abs(r))); peak_r = float(r[pk]); peak_lag_ms = int(lags_ms[pk])
    lead_ms = -peak_lag_ms                                      # +ve => REFERENCE leads Deriv

    # circular-shift surrogate null: rotate ref returns, recompute max|xcorr| over the lag window
    rng = np.random.default_rng(seed); zrf = np.where(np.isfinite(zr), zr, 0.0); n = len(zrf)
    null = np.empty(n_null)
    for j in range(n_null):
        sh = int(rng.integers(K + 1, n - K - 1))
        null[j] = np.nanmax(np.abs(_xcorr(zt, np.roll(zrf, sh), K)))
    obs = abs(peak_r)
    p = float((1 + np.sum(null >= obs)) / (n_null + 1))
    p95, p99 = float(np.quantile(null, 0.95)), float(np.quantile(null, 0.99))

    ref_leads = lead_ms > 0 and peak_r > 0
    significant = p < 0.01 and obs > p99
    verdict =("NO-GO: no significant lead (peak within surrogate null)" if not significant else
               ("NO-GO: significant co-movement but Deriv does NOT lag the reference (lead<=0)" if not ref_leads else
                (f"WEAK: reference leads by {lead_ms}ms but that is BELOW the +{entry_lag_ms}ms entry window — "
                 f"you cannot act before Deriv's next tick settles" if lead_ms < entry_lag_ms else
                 f"GO (gross): reference leads Deriv by {lead_ms}ms (> {entry_lag_ms}ms entry window) — a gross "
                 f"lead exists; now net it against spike-filter + mid-basis + the anti-arb clawback clause before risking capital")))

    res = {"grid_ms": grid_ms, "max_lag_ms": max_lag_ms, "n_grid_returns": int(fin.sum()),
           "overlap_min": round((t1 - t0) / 1e9 / 60, 2), "entry_lag_ms": entry_lag_ms,
           "peak_corr": round(peak_r, 4), "peak_lag_ms": peak_lag_ms,
           "reference_leads_deriv_ms": lead_ms, "reference_leads_deriv": bool(ref_leads),
           "null_p95": round(p95, 4), "null_p99": round(p99, 4), "p_value": round(p, 5),
           "significant": bool(significant), "verdict": verdict,
           "xcorr": {int(l): (round(float(v), 4) if np.isfinite(v) else None) for l, v in zip(lags_ms, r)}}
    if out:
        json.dump(res, open(out, "w"), indent=1)
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.axhspan(-p99, p99, color="0.85", label="surrogate null |r|<p99"); ax.axhline(0, color="0.6", lw=.7)
            ax.axvline(0, color="0.6", lw=.7, ls=":"); ax.plot(lags_ms, r, lw=1.4)
            ax.plot(peak_lag_ms, peak_r, "o", color="crimson",
                    label=f"peak r={peak_r:.3f} @ lag {peak_lag_ms}ms  (ref leads {lead_ms}ms)")
            ax.set_xlabel("lag k (ms)  —  r(k)=corr(Deriv[t], ref[t+k]); k<0 ⇒ reference leads")
            ax.set_ylabel("cross-correlation of log-returns"); ax.set_title(f"Deriv frxEURUSD vs reference — {verdict[:60]}")
            ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out.replace(".json", ".png"), dpi=110); plt.close(fig)
            hb(f"  plot -> {out.replace('.json', '.png')}")
        except Exception as e:
            hb(f"  (plot skipped: {e})")
        hb(f"analyze -> {out}")
    hb(f"VERDICT: {verdict}")
    hb(f"  peak r={peak_r:+.3f} @ lag {peak_lag_ms}ms | reference leads Deriv by {lead_ms}ms | "
       f"p={p:.4f} (null p99={p99:.3f}) | n={fin.sum()} grid-returns")
    return res


# ----------------------------------------------------------------------------- selfcheck (offline)
def selfcheck():
    hb("SELF-CHECK 1/3: recover a known injected lead (reference leads Deriv by 350ms; Deriv = ref delayed + mid-basis noise)")
    rng = np.random.default_rng(7); fs_ms = 20; n = 60 * 60 * 1000 // fs_ms      # 1h @ 20ms market grid
    base_ns = int(pd.Timestamp("2026-05-04T10:00:00Z").value)
    mt = base_ns + np.arange(n) * fs_ms * 1_000_000
    walk = np.cumsum(rng.standard_normal(n) * 2e-5); ref_mid = 1.10 + walk        # fine reference mid
    LEAD_MS = 350; shift = LEAD_MS // fs_ms
    der = np.empty(n); der[shift:] = ref_mid[:-shift]; der[:shift] = ref_mid[0]   # Deriv = ref delayed 350ms ...
    der += rng.standard_normal(n) * 3e-6                                          # ... + small idiosyncratic noise (mid-basis)
    os.makedirs(f"{ROOT}/captures", exist_ok=True)
    pd.DataFrame({"recv_ns": mt, "mid": der}).to_json(p1 := f"{ROOT}/captures/_sc_deriv.jsonl", orient="records", lines=True)
    pd.DataFrame({"ts_ns": mt, "mid": ref_mid}).to_parquet(p2 := f"{ROOT}/captures/_sc_ref.parquet")
    r = analyze(p1, p2, grid_ms=100, max_lag_ms=2000, n_null=500, entry_lag_ms=1000, out=None, seed=1)
    ok1 = r and r["reference_leads_deriv"] and abs(r["reference_leads_deriv_ms"] - LEAD_MS) <= 150 and r["significant"]
    print(f"  -> detected lead {r['reference_leads_deriv_ms']}ms (want ~{LEAD_MS}), p={r['p_value']}, sig={r['significant']} : {'PASS' if ok1 else 'FAIL'}")

    hb("SELF-CHECK 2/3: null calibration — two INDEPENDENT walks must NOT be significant")
    a = 1.10 + np.cumsum(rng.standard_normal(n) * 2e-5); b = 1.10 + np.cumsum(rng.standard_normal(n) * 2e-5)
    pd.DataFrame({"recv_ns": mt, "mid": a}).to_json(p3 := f"{ROOT}/captures/_sc_a.jsonl", orient="records", lines=True)
    pd.DataFrame({"ts_ns": mt, "mid": b}).to_parquet(p4 := f"{ROOT}/captures/_sc_b.parquet")
    r2 = analyze(p3, p4, grid_ms=100, max_lag_ms=2000, n_null=500, entry_lag_ms=1000, out=None, seed=2)
    ok2 = r2 and not r2["significant"]
    print(f"  -> independent walks p={r2['p_value']} sig={r2['significant']} (want NOT sig) : {'PASS' if ok2 else 'FAIL'}")

    hb("SELF-CHECK 3/3: decode a real historical Dukascopy hour (2026-06-04 10:00 UTC, a Thursday)")
    mtk, md = _duka_hour(pd.Timestamp("2026-06-04T10:00:00Z"))
    ok3 = len(mtk) > 100 and np.all(np.diff(mtk) >= 0) and 0.9 < np.median(md) < 1.3
    print(f"  -> {len(mtk)} ticks, mid~{np.median(md):.5f}, monotonic={bool(np.all(np.diff(mtk)>=0))} : {'PASS' if ok3 else 'FAIL'}")
    for p in (p1, p2, p3, p4):
        try: os.remove(p)
        except OSError: pass
    print(f"\nSELF-CHECK {'ALL PASS' if (ok1 and ok2 and ok3) else 'FAILED'}")
    return ok1 and ok2 and ok3


# ----------------------------------------------------------------------------- cli
def _window_from_capture(path):
    ns = pd.read_json(path, lines=True)["recv_ns"]; return int(ns.min()), int(ns.max())

def main():
    ap = argparse.ArgumentParser(description="Deriv lead-probe: the $0 gating experiment for the latency-arb thesis")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture"); c.add_argument("--minutes", type=float, default=120); c.add_argument("--out", default=f"{ROOT}/captures/deriv.jsonl")
    d = sub.add_parser("dukascopy"); d.add_argument("--from-capture"); d.add_argument("--from-ns", nargs=2, type=int); d.add_argument("--out", default=f"{ROOT}/captures/duka.parquet")
    a = sub.add_parser("analyze"); a.add_argument("--target", required=True); a.add_argument("--reference", required=True)
    a.add_argument("--grid-ms", type=int, default=100); a.add_argument("--max-lag-ms", type=int, default=3000)
    a.add_argument("--n-null", type=int, default=1000); a.add_argument("--entry-lag-ms", type=int, default=1000)
    a.add_argument("--out", default=f"{ROOT}/deriv_lead_probe_result.json")
    sub.add_parser("selfcheck")
    args = ap.parse_args()

    if args.cmd == "capture":
        import asyncio; asyncio.run(_capture(args.minutes, args.out))
    elif args.cmd == "dukascopy":
        win = _window_from_capture(args.__dict__["from_capture"]) if args.__dict__.get("from_capture") else tuple(args.from_ns)
        dukascopy(win, args.out)
    elif args.cmd == "analyze":
        analyze(args.target, args.reference, args.grid_ms, args.max_lag_ms, args.n_null, args.entry_lag_ms, args.out)
    elif args.cmd == "selfcheck":
        sys.exit(0 if selfcheck() else 1)

import asyncio  # noqa: E402  (used inside _recv_timeout / capture)
if __name__ == "__main__":
    main()
