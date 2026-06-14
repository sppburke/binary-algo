# Book Deriv Replication

`scripts/book_deriv_replicate.py` is the Deriv-native replication attempt for
the certified 15m book schemas. It is separate from the original Dukascopy
feature store and writes model artifacts under gitignored `deriv_data/models/`.
It is a candle-close proxy replication, not final Deriv tick-settlement
certification.

Run from the repo root:

```bash
~/binary-algo-venv/bin/python scripts/book_deriv_replicate.py --pairs all --cpcv candidates
```

The script:

- uses each incumbent book's feature-column schema where Deriv 1m candles can
  reproduce it;
- adds cross-pair features through the existing live feature recipe;
- holds out the latest two calendar months of Deriv data;
- purges train/validation rows whose label horizon crosses a split;
- counts Rise/Fall ties as losses;
- reports first-come non-overlapping selected-trade metrics;
- writes tracked result evidence to `results/json/book_deriv_replicate_*_result.json`.

EURUSD exact replication is expected to fail from candles alone while its
incumbent schema requires `OF_*` order-flow columns. Those columns must come from
a real order-flow feed, not zero-filled stubs.

Certification remains blocked until the label path uses Deriv-faithful tick
settlement: next tick entry, last tick before expiry, and ties charged as
losses. The candle-close proxy can kill weak replications and identify
candidates, but cannot certify a deployable book by itself.
