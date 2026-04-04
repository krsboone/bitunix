# Bitunix Perpetual Futures Trader

Algorithmic trading system for BTC, ETH, and RIVER perpetual futures on [Bitunix](https://www.bitunix.com).
Two independent strategies, shared infrastructure.

---

## Setup

**1. Install dependencies**
```bash
pip install requests
```

**2. Create `config.py`** (not committed — stays local)
```python
API_KEY    = "your-api-key"
SECRET_KEY = "your-secret-key"
```

**3. Verify connection**
```bash
python3 balance.py
```

---

## Scripts

### `balance.py`
Checks account balance and position mode. Good first test after setup.
```bash
python3 balance.py
```

### `market.py`
Fetches live candles and computes volatility (sigma) and Brownian motion price range estimates for each symbol. Useful for understanding current market conditions before running a strategy.
```bash
python3 market.py                        # BTC + ETH
python3 market.py RIVERUSDT              # single symbol
python3 market.py BTCUSDT --interval 5m  # different candle size
```

### `trader.py` — Brownian Motion Strategy
Monitors 1-minute candles and enters trades when cumulative price drift over a short window exceeds a Z-score threshold, indicating directional momentum. No trend bias — takes long or short signals equally.

**Entry logic:**
- Computes rolling volatility (sigma) over `SIGMA_CANDLES` recent candles
- Measures cumulative log-return drift over the last `SIGNAL_CANDLES` candles
- Normalises drift by `sigma × sqrt(SIGNAL_CANDLES)` to produce a Z-score
- Enters LONG if `z ≥ Z_ENTRY`, SHORT if `z ≤ -Z_ENTRY`

**Trade management:**
- Take profit: `TP_MULT × sigma × sqrt(HOLD_INTERVALS)` from entry
- Stop loss: `SL_MULT × sigma × sqrt(HOLD_INTERVALS)` from entry
- Fee gate: skips trade if expected profit ≤ round-trip fee cost
- Time exit: closes position if held longer than `MAX_HOLD_MINS`
- Circuit breaker: halts if balance drops below `MIN_BALANCE_PCT` of starting balance
- Reference balance resets when flat, so position sizing tracks actual balance

```bash
python3 trader.py --debug    # simulate — no real orders
python3 trader.py            # live trading
```

### `trend_trader.py` — Trend-Filtered Brownian Motion Strategy
Adds a daily trend line filter on top of the Brownian motion signal. Only takes trades whose direction aligns with the broader trend — longs when price is below the trend line, shorts when above.

**Trend line methodology** (matches Pine Script: *Historical Price Analysis & Projection Line*):
- Fetches `LOOKBACK_DAYS` daily candles
- **P1:** average open of the first 5 candles (range start)
- **P2:** average close of the last 5 candles (range end)
- **P3:** average `(high + low) / 2` of the middle 5 candles (midpoint anchor)
- Slope computed from P1 → P2
- Intercept averaged from both ends, then adjusted 30% toward the P3 deviation
- Trend line refreshes every ~10 minutes (`TREND_REFRESH_CYCLES` poll cycles)

**Trade gate:**
- Price below trend line → LONG signals only
- Price above trend line → SHORT signals only
- Z-score must still exceed `Z_ENTRY` threshold in the bias direction

```bash
python3 trend_trader.py --debug    # simulate — no real orders
python3 trend_trader.py            # live trading
```

### `backtest.py`
Replays recorded trade signals against historical 1-minute candles to determine whether TP, SL, or the time limit would have been hit first. Both traders write to their own CSV automatically — no manual steps required.

```bash
python3 backtest.py log/trades.csv                  # trader.py results
python3 backtest.py log/trend_trades.csv            # trend_trader.py results
python3 backtest.py log/trades.csv --hold 45        # test wider hold window
python3 backtest.py log/archive/would-1.csv         # historical archive
```

---

## Configuration

Key constants at the top of each strategy file:

| Constant | Description |
|---|---|
| `SYMBOLS` | Symbols to trade — add/remove to change coverage |
| `LEVERAGE` | Position leverage (default 2×) |
| `Z_ENTRY` | Z-score threshold to trigger entry |
| `TP_MULT` | Take profit distance in σ units |
| `SL_MULT` | Stop loss distance in σ units |
| `MAX_HOLD_MINS` | Time-based exit threshold |
| `MAX_TRADE_PCT` | Max % of balance per trade |
| `FEE_TAKER` | Market order fee (update as tier improves) |
| `FEE_MAKER` | Limit order fee (update as tier improves) |
| `MIN_BALANCE_PCT` | Circuit breaker floor |
| `LOOKBACK_DAYS` | Daily candles for trend line *(trend_trader only)* |

---

## Logs

Both traders log to `log/` (gitignored):

| File | Contents |
|---|---|
| `log/trader.log` | Full trader.py session log |
| `log/trend_trader.log` | Full trend_trader.py session log |
| `log/trades.csv` | Signals from trader.py — backtest input |
| `log/trend_trades.csv` | Signals from trend_trader.py — backtest input |

Archive CSVs periodically to `log/archive/` for historical comparison.

---

## Authentication

Bitunix uses a double SHA256 signature scheme. The key detail — **query parameters are serialised as key+value concatenated with no separator, sorted by key name** (e.g. `{marginCoin: "USDT"}` → `"marginCoinUSDT"`). Standard `urlencode` format does not work. See [Cairn](https://github.com/krsboone/cairn) for full documentation of this quirk.

REST timestamp: milliseconds UTC
WebSocket timestamp: seconds UTC (different — watch for this if extending to WebSocket feeds)

---

## Notes

- Account must be in **HEDGE mode** (verified on setup)
- Leverage is set to `LEVERAGE` on startup for all symbols
- On restart, live positions are detected and hydrated automatically — no duplicate entries
- `debug_auth.py` is a diagnostic tool for troubleshooting signature failures
