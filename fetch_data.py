"""
fetch_data.py — Historical candle data fetcher and local cache manager

Downloads OHLC candle data from Bitunix and stores it locally as CSV.
On the first run, fetches the full requested history. On subsequent runs,
only fetches candles newer than the last stored timestamp — keeping the
dataset current without redundant API calls.

Data is stored in data/{SYMBOL}_{interval}.csv, one file per symbol/interval.

Usage:
    python3 fetch_data.py                            # update all default symbols
    python3 fetch_data.py --symbol BTCUSDT           # single symbol
    python3 fetch_data.py --symbol BTCUSDT --days 60 # initial fetch, 60 days back
    python3 fetch_data.py --interval 1m --days 30    # explicit interval and depth
"""

import argparse
import csv
import os
import time
from datetime import datetime, timezone

from auth import BitunixClient
from config import API_KEY, SECRET_KEY

# ── Configuration ─────────────────────────────────────────────────────────────

#SYMBOLS  = ["BTCUSDT", "ETHUSDT", "RIVERUSDT"]
SYMBOLS  = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "1m"
DATA_DIR = "data"
BATCH    = 1000   # candles per API request


# ── File helpers ──────────────────────────────────────────────────────────────

def data_path(symbol: str, interval: str) -> str:
    return os.path.join(DATA_DIR, f"{symbol}_{interval}.csv")

def load_stored_timestamps(path: str) -> set[int]:
    """Return set of all stored timestamp_ms values."""
    if not os.path.exists(path):
        return set()
    timestamps = set()
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)   # skip header
        for row in reader:
            if row:
                timestamps.add(int(row[0]))
    return timestamps

def latest_ts(path: str) -> int | None:
    ts_set = load_stored_timestamps(path)
    return max(ts_set) if ts_set else None

def fmt_ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ── API fetch ─────────────────────────────────────────────────────────────────

def fetch_batch(client: BitunixClient, symbol: str,
                interval: str, end_ms: int) -> list[dict]:
    """Fetch up to BATCH candles ending at (and including) end_ms."""
    resp = client.get("/api/v1/futures/market/kline", {
        "symbol":   symbol,
        "interval": interval,
        "limit":    str(BATCH),
        "endTime":  str(end_ms),
    })
    if resp.get("code") != 0:
        raise RuntimeError(f"Kline error: {resp.get('msg')}")
    return resp.get("data", [])

def candle_row(c: dict) -> list:
    return [
        int(c["time"]),
        float(c["open"]),
        float(c["high"]),
        float(c["low"]),
        float(c["close"]),
        float(c.get("baseVol", 0)),   # baseVol = USDT value of trades
    ]


# ── Fetch logic ───────────────────────────────────────────────────────────────

def fetch_and_store(client: BitunixClient, symbol: str,
                    interval: str, days: int) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    path      = data_path(symbol, interval)
    now_ms    = int(datetime.now(timezone.utc).timestamp() * 1000)
    stored_ts = load_stored_timestamps(path)
    last_ts   = max(stored_ts) if stored_ts else None

    if last_ts:
        # ── Incremental update ─────────────────────────────────────────────
        fetch_from = last_ts + 60_000   # one 1m candle past last stored
        print(f"  {symbol}: updating from {fmt_ts(fetch_from)}")

        new_candles = []
        end_ms = now_ms

        while True:
            batch = fetch_batch(client, symbol, interval, end_ms)
            if not batch:
                break
            # Keep only candles newer than what we have
            fresh = [c for c in batch if int(c["time"]) >= fetch_from]
            new_candles.extend(fresh)
            oldest_in_batch = min(int(c["time"]) for c in batch)
            if oldest_in_batch < fetch_from or len(fresh) < len(batch):
                break
            end_ms = oldest_in_batch - 1
            time.sleep(0.1)

        # Deduplicate against stored
        new_candles = [c for c in new_candles
                       if int(c["time"]) not in stored_ts]
        new_candles.sort(key=lambda c: int(c["time"]))

        if not new_candles:
            print(f"  {symbol}: already up to date ({fmt_ts(last_ts)})")
            return

        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            for c in new_candles:
                writer.writerow(candle_row(c))

        new_last = max(int(c["time"]) for c in new_candles)
        print(f"  {symbol}: +{len(new_candles)} candles  "
              f"→ now through {fmt_ts(new_last)}")

    else:
        # ── Initial fetch ──────────────────────────────────────────────────
        target_ms = now_ms - days * 24 * 60 * 60 * 1000
        print(f"  {symbol}: initial fetch — {days} days "
              f"from {fmt_ts(target_ms)}")

        all_candles = []
        end_ms = now_ms

        while end_ms > target_ms:
            batch = fetch_batch(client, symbol, interval, end_ms)
            if not batch:
                break
            all_candles.extend(batch)
            oldest = min(int(c["time"]) for c in batch)
            if oldest <= target_ms:
                break
            end_ms = oldest - 1
            time.sleep(0.2)

        # Filter, deduplicate, sort
        seen, unique = set(), []
        for c in all_candles:
            ts = int(c["time"])
            if ts >= target_ms and ts not in seen:
                seen.add(ts)
                unique.append(c)
        unique.sort(key=lambda c: int(c["time"]))

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_ms", "open", "high", "low",
                             "close", "volume"])
            for c in unique:
                writer.writerow(candle_row(c))

        if unique:
            span_from = fmt_ts(min(int(c["time"]) for c in unique))
            span_to   = fmt_ts(max(int(c["time"]) for c in unique))
            print(f"  {symbol}: {len(unique)} candles stored  "
                  f"({span_from} → {span_to})")
        else:
            print(f"  {symbol}: no data returned")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and update local Bitunix candle cache")
    parser.add_argument("--symbol", nargs="+", metavar="SYMBOL",
                        help=f"Symbol(s) to fetch (default: {', '.join(SYMBOLS)})")
    parser.add_argument("--interval", default=INTERVAL,
                        help=f"Candle interval (default: {INTERVAL})")
    parser.add_argument("--days", type=int, default=30,
                        help="Days of history for initial fetch (default: 30)")
    args = parser.parse_args()

    client  = BitunixClient(API_KEY, SECRET_KEY)
    symbols = args.symbol or SYMBOLS

    print(f"Candle cache — {args.interval} interval")
    print(f"Symbols: {', '.join(symbols)}")
    print()
    for sym in symbols:
        try:
            fetch_and_store(client, sym, args.interval, args.days)
        except Exception as e:
            print(f"  {sym}: error — {e}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
