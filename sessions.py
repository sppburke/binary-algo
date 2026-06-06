"""DST-CORRECT trading-session masks — the SINGLE SOURCE OF TRUTH for the session re-campaign.

Sessions are defined in EXCHANGE-LOCAL time via the IANA tz database (pandas tz_convert), so EST/EDT, GMT/BST,
and the US-vs-EU DST-transition misalignment are all handled automatically — NOT a fixed UTC window (the repo's
legacy `pipeline.py` `sess_ny = 12<=UTC_hour<21` is a DST-APPROXIMATE shortcut we are replacing).

  ny    = 08:00-17:00  America/New_York   (US session; UTC-5 winter / UTC-4 summer)
  ldn   = 08:00-16:00  Europe/London      (GMT winter / BST summer)
  asia  = 09:00-18:00  Asia/Tokyo         (UTC+9, no DST)
  overlap = ny & ldn   (London-NY overlap, highest liquidity)

CAMPAIGN RULE: a session test restricts the DECISION bars (where labels are trained/selected/evaluated, in
train/val/test/OOS alike) to the session. FEATURES stay causal/continuous (rolling stats legitimately span the
prior session) — only the prediction/label rows are filtered. Use session_mask(ts_epoch_s, name)->bool array.
"""
import numpy as np, pandas as pd

LOCAL = {"ny": ("America/New_York", 8, 17), "ldn": ("Europe/London", 8, 16), "asia": ("Asia/Tokyo", 9, 18)}
SESSIONS = ["ny", "ldn", "asia"]            # the three the campaign splits separately


def session_mask(ts, name):
    """ts = epoch SECONDS (int array or scalar). Returns a boolean mask for `name` in {ny,ldn,asia,overlap,all}."""
    name = str(name).lower()
    idx = pd.to_datetime(np.asarray(ts), unit="s", utc=True)
    if name in ("all", "any", ""):
        return np.ones(len(idx), dtype=bool)
    if name == "overlap":
        return session_mask(ts, "ny") & session_mask(ts, "ldn")
    tz, lo, hi = LOCAL[name]
    h = idx.tz_convert(tz).hour.values
    return (h >= lo) & (h < hi)


def session_of_index(idx, name):
    """Same, for a tz-aware pandas DatetimeIndex."""
    return session_mask(idx.view("int64") // 1_000_000_000, name)


if __name__ == "__main__":
    # DST self-check: same UTC instant, winter vs summer, must shift NY local hour by 1
    for s in ("2025-01-15 18:00", "2025-07-15 18:00"):
        u = pd.Timestamp(s, tz="UTC")
        print(s, "UTC -> NY", u.tz_convert("America/New_York").strftime("%H %Z"),
              "| in NY session:", bool(session_mask([int(u.timestamp())], "ny")[0]))
