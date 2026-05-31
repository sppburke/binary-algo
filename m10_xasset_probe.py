"""10-MIN cross-asset feasibility probe: does ES (S&P500 e-mini, CME minute futures) LEAD EURUSD's next 10-min direction?
Risk-on channel: ES up -> risk-on -> USD weakness -> EURUSD up. The 30m session found ES 30m lead-corr~0; this re-tests at
10m where the equity-FX risk link is tighter and inside NY/overlap hours. DECISIVE before building a full feature model:
if the LAGGED ES->future-EURUSD corr is ~0 and/or sign-FLIPS across 2023/24/25/26, the lever is null (only contemporaneous
co-movement, which is untradeable). We also print the CONTEMPORANEOUS corr as a timezone-alignment sanity check (must be
clearly positive). Read-only on the external LEAN repo. Usage: python m10_xasset_probe.py
"""
import os, io, zipfile, glob, numpy as np, pandas as pd

ESDIR = "/home/sean/git/reverse-engineered-trading/lean/data/future/cme/minute/es"
FEAT = "/media/sean/CORSAIR/binary-algo/features"
YEARS = ["2023", "2024", "2025", "2026"]
LB = [1, 3, 5, 10]   # ES lookback minutes
HOR = 10

def load_es_year(year):
    """Front-month (max-volume contract per day) ES 1-min close series, UTC. Returns DataFrame[ts, es_close]."""
    files = sorted(glob.glob(f"{ESDIR}/{year}*_trade.zip"))
    rows = []
    for fp in files:
        day = os.path.basename(fp)[:8]
        base = pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:8]}", tz="UTC")
        best = None; bestvol = -1
        try:
            with zipfile.ZipFile(fp) as z:
                for name in z.namelist():
                    if not name.endswith(".csv"): continue
                    df = pd.read_csv(io.BytesIO(z.read(name)), header=None,
                                     names=["ms", "o", "h", "l", "c", "v"])
                    tv = df["v"].sum()
                    if tv > bestvol: bestvol = tv; best = df
        except Exception:
            continue
        if best is None or len(best) == 0: continue
        ts = base + pd.to_timedelta(best["ms"].values, unit="ms")
        rows.append(pd.DataFrame({"ts": ts, "es_close": best["c"].values}))
    if not rows: return None
    out = pd.concat(rows).drop_duplicates("ts").sort_values("ts").set_index("ts")
    return out

def load_eur(year):
    p = f"{FEAT}/EURUSD_{year}.parquet"
    d = pd.read_parquet(p, columns=["close"]); d = d[~d.index.duplicated(keep="last")]
    if d.index.tz is None: d.index = d.index.tz_localize("UTC")
    return d

def main():
    print(f"[xasset] ES->EURUSD lead-lag @ {HOR}m; front-month max-vol; NY-gated. years={YEARS}", flush=True)
    for year in YEARS:
        es = load_es_year(year)
        if es is None: print(f"  [{year}] no ES data"); continue
        eur = load_eur(year)
        # align on minute
        es1 = es["es_close"].reindex(eur.index, method=None)  # exact-minute match only
        df = pd.DataFrame({"eur": eur["close"].values, "es": es1.values}, index=eur.index).dropna()
        if len(df) < 5000: print(f"  [{year}] only {len(df)} aligned minutes — skip"); continue
        idx = df.index; secs = idx.values.astype("datetime64[s]").astype("int64"); n = len(df)
        le = np.log(df["eur"].values); ls = np.log(df["es"].values)
        # forward 10-min EURUSD return, contiguous
        fwd = np.full(n, np.nan)
        contig = (secs[HOR:] - secs[:-HOR]) == HOR * 60
        fwd[:n - HOR] = np.where(contig, le[HOR:] - le[:-HOR], np.nan)
        ysign = np.sign(fwd)
        hours = idx.hour.values + idx.minute.values / 60.0
        ny = (hours >= 13.0) & (hours < 22.0)
        # contemporaneous sanity (es_ret_1 vs eur_ret_1, same bar)
        er1 = np.concatenate([[np.nan], le[1:] - le[:-1]])
        sr1 = np.concatenate([[np.nan], ls[1:] - ls[:-1]])
        cm = np.isfinite(er1) & np.isfinite(sr1) & ny
        contemp = np.corrcoef(er1[cm], sr1[cm])[0, 1] if cm.sum() > 100 else float("nan")
        print(f"  [{year}] aligned={n:,} NYfrac={ny.mean():.2f}  CONTEMP corr(es_r1,eur_r1|NY)={contemp:+.3f}  (sanity: want >>0)", flush=True)
        # lagged ES return -> future EURUSD
        for k in LB:
            esr = np.concatenate([[np.nan] * k, ls[k:] - ls[:-k]])
            m = np.isfinite(esr) & np.isfinite(fwd) & (fwd != 0) & ny
            if m.sum() < 500: continue
            c = np.corrcoef(esr[m], fwd[m])[0, 1]
            # directional: predict EUR up iff past ES return > 0
            hit = (np.sign(esr[m]) == ysign[m]).mean()
            print(f"      ES_ret_{k:>2}m -> fwd{HOR}m: corr={c:+.4f}  dir_hit={hit:.4f} (n{m.sum()})", flush=True)

if __name__ == "__main__":
    main()
