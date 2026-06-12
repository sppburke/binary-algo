"""Event -> currency-direction sign map for macro-surprise conditioning (format-independent domain knowledge).

surprise = actual - forecast (consensus). For most metrics a POSITIVE surprise STRENGTHENS the currency (higher growth/
inflation => hawkish). Exceptions invert: jobless claims, unemployment rate (higher = weaker currency).

eurusd_signal = surprise_sign * bullish_sign * ccy_sign, where:
  bullish_sign = +1 if 'higher actual strengthens the currency' else -1
  ccy_sign     = +1 for EUR events (stronger EUR -> EURUSD UP), -1 for USD events (stronger USD -> EURUSD DOWN)
So a hawkish USD surprise (e.g. hot CPI) => eurusd_signal < 0 (predict DOWN); a hawkish EUR surprise => > 0 (UP).
"""
import re

# (regex keyword, currency, bullish_when_higher)  — first match wins; order matters (specific before generic).
# ccy=None means "take currency from the hint" (for metrics whose sign is country-independent, e.g. unemployment).
RULES = [
    # ---- country-independent (sign same for any ccy); ccy resolved from hint ----
    (r"unemployment rate|unemployment change|jobless|claimant count", None, False),  # higher unemployment = weaker ccy
    # ---- USD (higher actual => stronger USD unless noted) ----
    (r"non.?farm|nfp|nonfarm payroll", "USD", True),
    (r"core pce|pce price", "USD", True),
    (r"core cpi|cpi|consumer price|inflation rate", "USD", True),
    (r"\bppi\b|producer price", "USD", True),
    (r"gdp", "USD", True),
    (r"retail sales", "USD", True),
    (r"ism.*(manufactur|services|non.?manufactur)|ism pmi", "USD", True),
    (r"durable goods", "USD", True),
    (r"fomc|fed funds|federal funds|interest rate decision.*(fed|united states|u\.?s\.?)", "USD", True),
    (r"building permits|housing starts|new home sales|existing home", "USD", True),
    (r"consumer confidence|michigan|consumer sentiment", "USD", True),
    (r"adp", "USD", True),
    (r"factory orders", "USD", True),
    (r"industrial production", "USD", True),
    (r"trade balance", "USD", True),
    # ---- EUR (higher actual => stronger EUR unless noted) ----
    (r"ecb|main refinancing|deposit facility rate|interest rate decision.*(ecb|euro)", "EUR", True),
    (r"unemployment.*(euro|german|germany|france|spain|italy)", "EUR", False),
    (r"(hicp|cpi|consumer price|inflation).*(euro|german|germany|france|spain|italy|ez)", "EUR", True),
    (r"(german|germany|france|french|euro|ez|spain|italy).*(cpi|hicp|inflation)", "EUR", True),
    (r"(pmi|purchasing managers).*(euro|german|france|ez|composite|manufactur|services)", "EUR", True),
    (r"(euro|german|france|ez).*(pmi|purchasing managers)", "EUR", True),
    (r"zew", "EUR", True),
    (r"ifo", "EUR", True),
    (r"(gdp).*(euro|german|germany|france|ez)|(euro|german|germany|france|ez).*gdp", "EUR", True),
    (r"(retail sales).*(euro|german|ez)|(german|euro|ez).*retail sales", "EUR", True),
    (r"(industrial production|factory orders).*(euro|german|ez)|(german|euro|ez).*(industrial production|factory orders)", "EUR", True),
    (r"(sentix|economic sentiment).*(euro|ez)", "EUR", True),
]

def classify(event_title, currency_hint=None):
    """Return (ccy in {USD,EUR,None}, bullish_when_higher in {+1,-1,None}).
    currency_hint (e.g. the calendar's country/ccy column) disambiguates generic titles like 'CPI'."""
    t = (event_title or "").lower()
    hint = _norm_ccy(currency_hint) if currency_hint else None
    for pat, ccy, bull in RULES:
        if re.search(pat, t):
            if ccy is None:                       # country-independent metric: resolve ccy from the hint
                if hint not in ("USD", "EUR"): continue
                return hint, (1 if bull else -1)
            if currency_hint and ccy != hint:     # title says one ccy but hint says another -> trust the hint
                continue
            return ccy, (1 if bull else -1)
    # fall back to the hint for generic high-impact titles (NOTE: 'employment' excluded - sign is ambiguous)
    if hint in ("USD", "EUR") and re.search(r"cpi|gdp|pmi|\brate\b|retail sales|production|sentiment|confidence|inflation", t):
        return hint, 1
    return None, None

def _norm_ccy(h):
    h = (h or "").upper()
    if h in ("USD", "US", "USA", "UNITED STATES"): return "USD"
    if h in ("EUR", "EZ", "EA", "EUROZONE", "EURO AREA", "GERMANY", "DE", "FRANCE", "FR", "ITALY", "SPAIN"): return "EUR"
    return h

def eurusd_signal(ccy, bullish_when_higher, surprise):
    """+ => predict EURUSD UP, - => DOWN, scaled by surprise magnitude. surprise = actual - forecast (any units;
    caller should z-score per event for cross-event comparability)."""
    if ccy is None or bullish_when_higher is None or surprise is None:
        return 0.0
    ccy_sign = 1.0 if ccy == "EUR" else -1.0  # USD event flips EURUSD
    return float(surprise) * bullish_when_higher * ccy_sign

if __name__ == "__main__":
    tests = [("Nonfarm Payrolls", "USD", +200), ("Core CPI", "USD", +0.3), ("Initial Jobless Claims", "USD", +50),
             ("ECB Interest Rate Decision", "EUR", +0.25), ("German ZEW Economic Sentiment", "EUR", +5),
             ("Eurozone Unemployment Rate", "EUR", +0.2)]
    for title, ccy, surp in tests:
        c, b = classify(title, ccy)
        print(f"{title:<32} ccy={c} bull={b} surprise={surp:+} -> EURUSD signal={eurusd_signal(c,b,surp):+.2f}")
