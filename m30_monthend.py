"""Discovery R2 lever: month-end / quarter-end rebalancing-flow directional bias (on-disk calendar).
FAST-KILL stats check (no GBM, no cross-pair): does the last-3-trading-days-of-month / last-5-of-quarter window
carry a STABLE significant directional up-rate bias on 30m EURUSD, across 2024/2025/2026? If the moved-bar
period-end up-rate does NOT exit [.47,.53] in a STABLE direction (or chi-sq vs non-period-end is n.s.), there is
no signed gate to build -> KILL before any refit (per the lever's own pre-registered falsifier; agent expected KILL
since hour-of-day seasonality is already the certified book's #1/#3 feature).

Usage: python m30_monthend.py
"""
import os, json, numpy as np, pandas as pd
from scipy import stats
FEAT = "/home/sean/git/binary-algo/features"
HOR = 30
STRIDE = 3  # cap memory; estimating an up-rate bias does not need every bar


def load(year):
    p = f"{FEAT}/EURUSD_{year}.parquet"
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p, columns=["close"]); df = df[~df.index.duplicated(keep="last")].sort_index()
    c = df["close"].values; n = len(c)
    secs = df.index.values.astype("datetime64[s]").astype("int64")
    contig = np.zeros(n, bool); contig[:n - HOR] = (secs[HOR:] - secs[:-HOR]) == HOR * 60
    fwd = np.full(n, np.nan); fwd[:n - HOR] = c[HOR:]; ret = fwd / c - 1.0
    valid = contig & np.isfinite(ret) & (ret != 0)
    idx = df.index[valid]
    up = (ret[valid] > 0).astype(int)
    # last 3 trading days of month / last 5 of quarter: approximate trading-day rank within month
    d = pd.DataFrame({"up": up}, index=idx)
    d["date"] = d.index.normalize()
    days = d["date"].drop_duplicates().sort_values()
    dd = pd.DataFrame({"date": days})
    dd["ym"] = dd["date"].dt.to_period("M")
    dd["rank_from_end_m"] = dd.groupby("ym")["date"].rank(ascending=False, method="dense")
    dd["q"] = dd["date"].dt.to_period("Q")
    dd["rank_from_end_q"] = dd.groupby("q")["date"].rank(ascending=False, method="dense")
    dd["me"] = dd["rank_from_end_m"] <= 3
    dd["qe"] = dd["rank_from_end_q"] <= 5
    m = dd.set_index("date")[["me", "qe"]]
    d = d.join(m, on="date")
    d = d.iloc[::STRIDE]
    return d


def updiff(d, col):
    a = d.loc[d[col], "up"]; b = d.loc[~d[col], "up"]
    if len(a) < 30 or len(b) < 30:
        return None
    # 2x2 chi-sq: period-end vs not, up vs down
    ct = np.array([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    return {"pe_n": int(len(a)), "pe_uprate": round(float(a.mean()), 4), "rest_uprate": round(float(b.mean()), 4),
            "chi2_p": round(float(p), 4)}


def main():
    res = {"lever": "month-end/quarter-end rebalancing-flow directional bias", "HOR": HOR, "stride": STRIDE,
           "tripwire": "[0.47,0.53] moved up-rate; need STABLE significant directional bias across 2024/25/26"}
    per = {}
    for yr in (2024, 2025, 2026):
        d = load(yr)
        if d is None:
            continue
        per[yr] = {"month_end": updiff(d, "me"), "quarter_end": updiff(d, "qe")}
        print(f"{yr}: ME {per[yr]['month_end']} | QE {per[yr]['quarter_end']}", flush=True)
    res["per_year"] = per
    # verdict: stable direction (pe_uprate-0.5 same sign all yrs) AND any year exits tripwire AND chi-sq sig
    def stable(kind):
        rows = [per[y][kind] for y in per if per[y][kind]]
        if len(rows) < 3:
            return False, "insufficient years"
        signs = [np.sign(r["pe_uprate"] - 0.5) for r in rows]
        same = len(set(signs)) == 1 and signs[0] != 0
        exits = any((r["pe_uprate"] < 0.47 or r["pe_uprate"] > 0.53) for r in rows)
        sig = any(r["chi2_p"] < 0.05 for r in rows)
        return bool(same and exits and sig), f"same_dir={same} exits_tripwire={exits} any_sig={sig}"
    me_ok, me_why = stable("month_end"); qe_ok, qe_why = stable("quarter_end")
    res["verdict"] = {"month_end_survives": me_ok, "month_end_why": me_why,
                      "quarter_end_survives": qe_ok, "quarter_end_why": qe_why,
                      "KILLED": (not me_ok) and (not qe_ok)}
    json.dump(res, open("m30_monthend_result.json", "w"), indent=1)
    print(f"[monthend] VERDICT: ME survives={me_ok} ({me_why}); QE survives={qe_ok} ({qe_why}); "
          f"KILLED={res['verdict']['KILLED']} -> m30_monthend_result.json", flush=True)


if __name__ == "__main__":
    main()
