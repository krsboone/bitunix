"""
backtest.py — Replay would-be trades against historical candles

For each trade in the CSV, fetches 1-minute candles from entry time
forward (up to MAX_HOLD_MINS) and determines whether TP or SL would
have been hit first, or if the position would have timed out.

Usage:
    python3 backtest.py log/archive/would-1.csv
    python3 backtest.py log/archive/would-1.csv --hold 45
"""

import argparse
import csv
import sys
import time
from datetime import datetime, timezone, timedelta

from auth import BitunixClient
from config import API_KEY, SECRET_KEY

MAX_HOLD_MINS = 30      # match trader.py default (override with --hold)
CANDLES_AHEAD = 35      # fetch slightly more than MAX_HOLD_MINS to cover slippage


def parse_row(row: list[str]) -> dict | None:
    """Parse a CSV row into a trade dict. Returns None if unparseable."""
    try:
        # Format: timestamp, timestamp, symbol, SYM, qty, N, side, BUY/SELL, ..., tpPrice, N, slPrice, N, ...
        fields = [f.strip() for f in row]
        def get(key):
            idx = fields.index(key)
            return fields[idx + 1]

        raw_ts = fields[0].strip()  # e.g. "2026-04-0318:28:22"
        # Insert missing space between date and time if absent
        if len(raw_ts) == 18 and " " not in raw_ts:
            raw_ts = raw_ts[:10] + " " + raw_ts[10:]
        entry_time = datetime.strptime(raw_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

        return {
            "entry_time": entry_time,
            "symbol":     get("symbol"),
            "side":       get("side"),      # BUY or SELL
            "qty":        float(get("qty")),
            "tp_price":   float(get("tpPrice")),
            "sl_price":   float(get("slPrice")),
        }
    except Exception:
        return None


def fetch_candles_from(client: BitunixClient, symbol: str,
                       from_time: datetime, count: int) -> list[dict]:
    """
    Fetch `count` 1-minute candles starting from from_time.
    Bitunix kline returns candles newest-first ending at endTime,
    so we set endTime = from_time + count minutes and filter.
    """
    end_ms = int((from_time.timestamp() + count * 60) * 1000)
    resp = client.get("/api/v1/futures/market/kline", {
        "symbol":  symbol,
        "interval": "1m",
        "limit":   str(count),
        "endTime": str(end_ms),
    })
    if resp.get("code") != 0:
        return []
    candles = resp.get("data", [])
    # Sort ascending and filter to only candles at or after entry time
    entry_ms = int(from_time.timestamp() * 1000)
    candles  = [c for c in candles if int(c.get("time", 0)) >= entry_ms]
    return sorted(candles, key=lambda c: int(c.get("time", 0)))


def evaluate_trade(trade: dict, candles: list[dict], max_hold_mins: int) -> dict:
    """
    Walk candles forward from entry, check if high/low touches TP or SL.
    Returns result dict with outcome, minutes_to_outcome, pnl_per_unit.
    """
    tp = trade["tp_price"]
    sl = trade["sl_price"]
    is_long = trade["side"] == "BUY"
    entry_time = trade["entry_time"]

    for i, candle in enumerate(candles):
        candle_time = datetime.fromtimestamp(int(candle["time"]) / 1000, tz=timezone.utc)
        mins_elapsed = (candle_time - entry_time).total_seconds() / 60

        if mins_elapsed > max_hold_mins:
            close_price = float(candle["close"])
            pnl = (close_price - trade["tp_price"]) if is_long else (trade["tp_price"] - close_price)
            # Use entry approximation: midpoint of TP/SL range as entry
            entry_est = (tp + sl) / 2  # rough; actual entry is entry candle close
            close_pnl = (close_price - entry_est) if is_long else (entry_est - close_price)
            return {
                "outcome":     "TIME_EXIT",
                "mins":        mins_elapsed,
                "close_price": close_price,
                "pnl_dir":     close_pnl,
            }

        high = float(candle["high"])
        low  = float(candle["low"])

        if is_long:
            # Check SL first (worse outcome) then TP — conservative
            if low <= sl:
                return {"outcome": "SL_HIT", "mins": mins_elapsed,
                        "close_price": sl, "pnl_dir": sl - tp}
            if high >= tp:
                return {"outcome": "TP_HIT", "mins": mins_elapsed,
                        "close_price": tp, "pnl_dir": tp - sl}
        else:
            if high >= sl:
                return {"outcome": "SL_HIT", "mins": mins_elapsed,
                        "close_price": sl, "pnl_dir": sl - tp}
            if low <= tp:
                return {"outcome": "TP_HIT", "mins": mins_elapsed,
                        "close_price": tp, "pnl_dir": tp - sl}

    return {"outcome": "NO_DATA", "mins": 0, "close_price": 0, "pnl_dir": 0}


def run(csv_path: str, max_hold_mins: int) -> None:
    client = BitunixClient(API_KEY, SECRET_KEY)

    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))

    trades = [parse_row(r) for r in rows if r]
    trades = [t for t in trades if t]

    print(f"\nBacktest — {csv_path}")
    print(f"Trades parsed : {len(trades)}  |  Max hold : {max_hold_mins}min")
    print("─" * 80)
    print(f"{'Time':19}  {'Sym':10} {'Side':5} {'TP':>10} {'SL':>10} "
          f"{'Outcome':10} {'Mins':>5} {'Close':>10}")
    print("─" * 80)

    results = {"TP_HIT": 0, "SL_HIT": 0, "TIME_EXIT": 0, "NO_DATA": 0}

    for trade in trades:
        # Rate limit courtesy pause
        time.sleep(0.2)

        candles = fetch_candles_from(client, trade["symbol"],
                                     trade["entry_time"], CANDLES_AHEAD)
        result  = evaluate_trade(trade, candles, max_hold_mins)
        results[result["outcome"]] = results.get(result["outcome"], 0) + 1

        ts  = trade["entry_time"].strftime("%Y-%m-%d %H:%M:%S")
        sym = trade["symbol"]
        print(f"{ts}  {sym:10} {trade['side']:5} "
              f"{trade['tp_price']:>10.3f} {trade['sl_price']:>10.3f}  "
              f"{result['outcome']:10} {result['mins']:>5.1f} "
              f"{result['close_price']:>10.3f}")

    total = len(trades)
    print("─" * 80)
    print(f"\nSummary ({total} trades)")
    print(f"  TP hit    : {results['TP_HIT']:>3}  ({results['TP_HIT']/total*100:.1f}%)")
    print(f"  SL hit    : {results['SL_HIT']:>3}  ({results['SL_HIT']/total*100:.1f}%)")
    print(f"  Time exit : {results['TIME_EXIT']:>3}  ({results['TIME_EXIT']/total*100:.1f}%)")
    if results["NO_DATA"]:
        print(f"  No data   : {results['NO_DATA']:>3}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest would-be trades from CSV")
    parser.add_argument("csv", help="Path to would-be trades CSV")
    parser.add_argument("--hold", type=int, default=MAX_HOLD_MINS,
                        help=f"Max hold minutes (default {MAX_HOLD_MINS})")
    args = parser.parse_args()
    run(args.csv, args.hold)
