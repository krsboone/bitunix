"""
market.py — Bitunix market data + volatility analysis

Fetches recent OHLC candles for a symbol and computes:
  - Rolling log-return volatility (sigma) per candle interval
  - Annualized and per-period sigma
  - Brownian motion price range estimate for a given holding period
  - Current price context (last, mark, bid/ask spread from depth)

Usage:
    python3 market.py                     # BTC + ETH summary
    python3 market.py BTCUSDT             # single symbol
    python3 market.py BTCUSDT --interval 5m --candles 60
"""

import argparse
import math
import sys
from auth import BitunixClient
from config import API_KEY, SECRET_KEY

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_SYMBOLS  = ["BTCUSDT", "ETHUSDT"]
DEFAULT_INTERVAL = "1m"     # candle size
DEFAULT_CANDLES  = 30       # number of candles for sigma calculation
HOLD_MINUTES     = 15       # expected holding period for range estimate


# ── Candle fetch ──────────────────────────────────────────────────────────────

def fetch_candles(client: BitunixClient, symbol: str,
                  interval: str = "1m", limit: int = 50) -> list[dict]:
    """Return list of OHLC candles, newest last."""
    resp = client.get("/api/v1/futures/market/kline", {
        "symbol": symbol,
        "interval": interval,
        "limit": str(limit),
    })
    if resp.get("code") != 0:
        raise RuntimeError(f"Kline error {resp.get('code')}: {resp.get('msg')}")
    # Each candle: [timestamp, open, high, low, close, volume, ...]
    candles = resp.get("data", [])
    return candles


# ── Volatility ────────────────────────────────────────────────────────────────

def compute_sigma(candles: list, price_type: str = "close") -> float:
    """
    Compute rolling volatility as std dev of log returns.
    Returns sigma per candle interval (not annualized).
    Candles are dicts with keys: open, high, low, close.
    """
    prices = [float(c[price_type]) for c in candles if float(c[price_type]) > 0]
    if len(prices) < 2:
        return 0.0
    log_returns = [math.log(prices[i] / prices[i - 1])
                   for i in range(1, len(prices))]
    n    = len(log_returns)
    mean = sum(log_returns) / n
    var  = sum((r - mean) ** 2 for r in log_returns) / (n - 1)
    return math.sqrt(var)


def brownian_range(price: float, sigma_per_interval: float,
                   hold_intervals: int, z: float = 1.645) -> tuple[float, float]:
    """
    Estimate price range after hold_intervals candles at 90% confidence (z=1.645).
    Returns (lower_bound, upper_bound).
    """
    sigma_hold = sigma_per_interval * math.sqrt(hold_intervals)
    move       = price * sigma_hold * z
    return price - move, price + move


# ── Ticker / depth ────────────────────────────────────────────────────────────

def fetch_ticker(client: BitunixClient, symbol: str) -> dict:
    resp = client.get("/api/v1/futures/market/tickers", {"symbols": symbol})
    if resp.get("code") != 0:
        raise RuntimeError(f"Ticker error {resp.get('code')}: {resp.get('msg')}")
    data = resp.get("data", [])
    return data[0] if data else {}


def fetch_depth(client: BitunixClient, symbol: str, limit: str = "5") -> dict:
    resp = client.get("/api/v1/futures/market/depth", {
        "symbol": symbol,
        "limit":  limit,
    })
    if resp.get("code") != 0:
        return {}
    return resp.get("data", {})


# ── Display ───────────────────────────────────────────────────────────────────

def analyze(client: BitunixClient, symbol: str,
            interval: str = DEFAULT_INTERVAL,
            candle_count: int = DEFAULT_CANDLES,
            hold_minutes: int = HOLD_MINUTES) -> dict:
    """Full market analysis for one symbol. Returns result dict."""

    candles = fetch_candles(client, symbol, interval, candle_count + 1)
    ticker  = fetch_ticker(client, symbol)
    depth   = fetch_depth(client, symbol)

    last_price  = float(ticker.get("lastPrice",  0))
    mark_price  = float(ticker.get("markPrice",  0))
    high_24h    = float(ticker.get("high",        0))
    low_24h     = float(ticker.get("low",         0))

    # Best bid/ask from depth
    bids = depth.get("bids", [])
    asks = depth.get("asks", [])
    best_bid = float(bids[0][0]) if bids else 0.0
    best_ask = float(asks[0][0]) if asks else 0.0
    spread   = best_ask - best_bid if best_bid and best_ask else 0.0

    sigma    = compute_sigma(candles)
    sigma_pct = sigma * 100

    # Interval size in minutes
    interval_mins = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240,
    }.get(interval, 1)

    hold_intervals = max(1, hold_minutes // interval_mins)
    low_est, high_est = brownian_range(last_price, sigma, hold_intervals)

    # Stop-loss at 2× sigma from entry (configurable later)
    sl_distance = last_price * sigma * math.sqrt(hold_intervals) * 2.0
    tp_distance = last_price * sigma * math.sqrt(hold_intervals) * 1.5

    return {
        "symbol":         symbol,
        "last_price":     last_price,
        "mark_price":     mark_price,
        "high_24h":       high_24h,
        "low_24h":        low_24h,
        "best_bid":       best_bid,
        "best_ask":       best_ask,
        "spread":         spread,
        "sigma_per_int":  sigma,
        "sigma_pct":      sigma_pct,
        "hold_intervals": hold_intervals,
        "range_low":      low_est,
        "range_high":     high_est,
        "sl_distance":    sl_distance,
        "tp_distance":    tp_distance,
        "interval":       interval,
        "candle_count":   len(candles),
    }


def print_analysis(r: dict) -> None:
    sym   = r["symbol"]
    intv  = r["interval"]
    print(f"\n{'━' * 56}")
    print(f"  {sym}  ({r['candle_count']} × {intv} candles)")
    print(f"{'━' * 56}")
    print(f"  Last price    : {r['last_price']:>12.4f} USDT")
    print(f"  Mark price    : {r['mark_price']:>12.4f} USDT")
    print(f"  24h high/low  : {r['high_24h']:>12.4f} / {r['low_24h']:.4f}")
    print(f"  Bid / Ask     : {r['best_bid']:>12.4f} / {r['best_ask']:.4f}  "
          f"(spread {r['spread']:.4f})")
    print()
    print(f"  Sigma ({intv})   : {r['sigma_pct']:>10.4f}%  per candle")
    print(f"  Hold period   : {r['hold_intervals']} candle(s)")
    print(f"  90% range     : {r['range_low']:>12.4f} – {r['range_high']:.4f}")
    print()
    print(f"  Suggested TP  : ±{r['tp_distance']:>8.4f}  (1.5σ√t)")
    print(f"  Suggested SL  : ∓{r['sl_distance']:>8.4f}  (2.0σ√t)")
    if r['last_price'] > 0:
        tp_pct = r['tp_distance'] / r['last_price'] * 100
        sl_pct = r['sl_distance'] / r['last_price'] * 100
        print(f"  TP / SL %     :  {tp_pct:.3f}% / {sl_pct:.3f}%")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Bitunix market volatility analysis")
    parser.add_argument("symbol",   nargs="?", default=None)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--candles",  type=int, default=DEFAULT_CANDLES)
    parser.add_argument("--hold",     type=int, default=HOLD_MINUTES,
                        help="Expected holding period in minutes")
    args = parser.parse_args()

    client  = BitunixClient(API_KEY, SECRET_KEY)
    symbols = [args.symbol.upper()] if args.symbol else DEFAULT_SYMBOLS

    for sym in symbols:
        try:
            result = analyze(client, sym, args.interval, args.candles, args.hold)
            print_analysis(result)
        except Exception as e:
            print(f"\n  {sym}: error — {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
