"""
backtest.py — Replay would-be trades against locally-cached candle data

Reads a trades CSV (produced by trader.py, sr_trader.py, bb_trader.py, etc.)
and walks through the local 1m candle cache to determine whether each trade
would have hit TP, SL, or timed out. Reports PnL per trade including fees.

Fee model:
  Entry:            taker fee (market order)
  TP exit:          maker fee (limit order filled by exchange)
  SL / Time exit:   taker fee (market order)
  Configurable via --fee-taker and --fee-maker

Hold behaviour:
  --hold N   : scan up to N minutes, then time-exit
  (omitted)  : scan until TP or SL is hit; no time cap

Usage:
    python3 backtest.py log/sr_trades.csv
    python3 backtest.py log/bb_trades.csv --hold 33
    python3 backtest.py log/trades.csv --hold 45 --fee-taker 0.0006
"""

import argparse
import csv
import os
from datetime import datetime, timezone

DATA_DIR = "data"

# ── Fee defaults ───────────────────────────────────────────────────────────────

FEE_TAKER = 0.00060   # market order (entry, SL, time exit)
FEE_MAKER = 0.00020   # limit order  (TP hit via inline limit on exchange)


# ── Local candle cache ─────────────────────────────────────────────────────────

_candle_cache: dict[str, list[dict]] = {}

def _load_symbol(symbol: str) -> list[dict]:
    """Load and cache local 1m candles for a symbol."""
    if symbol in _candle_cache:
        return _candle_cache[symbol]
    path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No local data for {symbol} at {path}. "
            f"Run: python3 fetch_data.py --symbol {symbol}"
        )
    candles = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles.append({
                "time":  int(row["timestamp_ms"]),
                "open":  float(row["open"]),
                "high":  float(row["high"]),
                "low":   float(row["low"]),
                "close": float(row["close"]),
            })
    candles.sort(key=lambda c: c["time"])
    _candle_cache[symbol] = candles
    return candles


def candles_from(symbol: str, from_ts_ms: int,
                 max_mins: int | None) -> list[dict]:
    """
    Return local candles for symbol starting at from_ts_ms.
    If max_mins is given, cap at from_ts_ms + max_mins minutes.
    """
    all_c  = _load_symbol(symbol)
    end_ms = (from_ts_ms + max_mins * 60_000) if max_mins is not None else None
    result = []
    for c in all_c:
        if c["time"] < from_ts_ms:
            continue
        if end_ms is not None and c["time"] > end_ms:
            break
        result.append(c)
    return result


# ── Trade CSV parsing ──────────────────────────────────────────────────────────

def parse_row(row: list[str]) -> dict | None:
    """Parse a CSV row into a trade dict. Returns None if unparseable."""
    try:
        fields = [f.strip() for f in row]

        def get(key):
            idx = fields.index(key)
            return fields[idx + 1]

        raw_ts = fields[0].strip()
        # Handle compact "2026-04-0318:28:22" format (no space between date/time)
        if len(raw_ts) == 18 and " " not in raw_ts:
            raw_ts = raw_ts[:10] + " " + raw_ts[10:]
        entry_time = datetime.strptime(raw_ts, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc)

        return {
            "entry_time": entry_time,
            "entry_ts":   int(entry_time.timestamp() * 1000),
            "symbol":     get("symbol"),
            "side":       get("side"),        # BUY or SELL
            "qty":        float(get("qty")),
            "tp_price":   float(get("tpPrice")),
            "sl_price":   float(get("slPrice")),
        }
    except Exception:
        return None


# ── Trade evaluation ───────────────────────────────────────────────────────────

def evaluate_trade(trade: dict, candles: list[dict],
                   max_hold_mins: int | None,
                   fee_taker: float, fee_maker: float) -> dict:
    """
    Walk candles forward from entry. Returns result dict:
      outcome, mins, close_price, pnl (after fees), entry_price_est
    """
    tp      = trade["tp_price"]
    sl      = trade["sl_price"]
    is_long = trade["side"] == "BUY"
    qty     = trade["qty"]
    entry_time = trade["entry_time"]

    # Estimate entry price as the open of the first candle at/after entry
    entry_price = candles[0]["open"] if candles else (tp + sl) / 2

    def calc_pnl(close_price: float, exit_fee_rate: float) -> float:
        direction = 1 if is_long else -1
        gross = direction * (close_price - entry_price) * qty
        fees  = (entry_price * fee_taker + close_price * exit_fee_rate) * qty
        return gross - fees

    for candle in candles:
        candle_time  = datetime.fromtimestamp(candle["time"] / 1000, tz=timezone.utc)
        mins_elapsed = (candle_time - entry_time).total_seconds() / 60

        # Time exit cap (only when --hold is set)
        if max_hold_mins is not None and mins_elapsed > max_hold_mins:
            close_price = candle["close"]
            return {
                "outcome":      "TIME_EXIT",
                "mins":         mins_elapsed,
                "close_price":  close_price,
                "pnl":          calc_pnl(close_price, fee_taker),
                "entry_price":  entry_price,
            }

        high = candle["high"]
        low  = candle["low"]

        # Check both TP and SL within the same candle
        if is_long:
            sl_hit = low  <= sl
            tp_hit = high >= tp
        else:
            sl_hit = high >= sl
            tp_hit = low  <= tp

        if sl_hit and tp_hit:
            # Both triggered — use whichever is closer to entry
            if abs(tp - entry_price) <= abs(sl - entry_price):
                return {"outcome": "TP_HIT", "mins": mins_elapsed,
                        "close_price": tp,
                        "pnl": calc_pnl(tp, fee_maker),
                        "entry_price": entry_price}
            else:
                return {"outcome": "SL_HIT", "mins": mins_elapsed,
                        "close_price": sl,
                        "pnl": calc_pnl(sl, fee_taker),
                        "entry_price": entry_price}
        elif tp_hit:
            return {"outcome": "TP_HIT", "mins": mins_elapsed,
                    "close_price": tp,
                    "pnl": calc_pnl(tp, fee_maker),
                    "entry_price": entry_price}
        elif sl_hit:
            return {"outcome": "SL_HIT", "mins": mins_elapsed,
                    "close_price": sl,
                    "pnl": calc_pnl(sl, fee_taker),
                    "entry_price": entry_price}

    # Ran out of candle data without resolution
    last_close = candles[-1]["close"] if candles else entry_price
    last_mins  = (datetime.fromtimestamp(candles[-1]["time"] / 1000,
                  tz=timezone.utc) - entry_time).total_seconds() / 60 if candles else 0

    # If a hold limit was set, exhausting the candle window = time exit
    # (candles_from already clipped to max_hold_mins; TP/SL simply didn't fire)
    outcome    = "TIME_EXIT" if max_hold_mins is not None else "NO_DATA"
    exit_fee   = fee_taker
    return {
        "outcome":     outcome,
        "mins":        last_mins,
        "close_price": last_close,
        "pnl":         calc_pnl(last_close, exit_fee),
        "entry_price": entry_price,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run(csv_path: str, max_hold_mins: int | None,
        fee_taker: float, fee_maker: float) -> None:

    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))

    trades = [parse_row(r) for r in rows if r]
    trades = [t for t in trades if t]

    hold_label = f"{max_hold_mins}min" if max_hold_mins is not None else "unlimited"
    print(f"\nBacktest — {csv_path}")
    print(f"Trades parsed : {len(trades)}  |  Hold : {hold_label}")
    print(f"Fees          : taker {fee_taker*100:.3f}%  maker {fee_maker*100:.3f}%")
    print("─" * 122)
    print(f"{'Time':19}  {'Sym':10} {'Side':4}  {'Qty':>8}  {'Entry':>10} {'TP':>10} {'SL':>10} "
          f"{'Outcome':10} {'Mins':>5}  {'Close':>10}  {'PnL':>10}")
    print("─" * 122)

    results   = {"TP_HIT": 0, "SL_HIT": 0, "TIME_EXIT": 0, "NO_DATA": 0}
    total_pnl = 0.0
    skipped   = 0

    for trade in trades:
        sym = trade["symbol"]
        try:
            candles = candles_from(sym, trade["entry_ts"], max_hold_mins)
        except FileNotFoundError as e:
            print(f"  SKIP {sym}: {e}")
            skipped += 1
            continue

        if not candles:
            print(f"  {trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}  "
                  f"{sym:10} {trade['side']:4}  — no candles at entry time —")
            skipped += 1
            continue

        result = evaluate_trade(trade, candles, max_hold_mins, fee_taker, fee_maker)
        results[result["outcome"]] = results.get(result["outcome"], 0) + 1
        total_pnl += result["pnl"]

        ts  = trade["entry_time"].strftime("%Y-%m-%d %H:%M:%S")
        pnl_str = f"{result['pnl']:+.4f}"
        print(f"{ts}  {sym:10} {trade['side']:4}  {trade['qty']:>8}  "
              f"{result['entry_price']:>10.4f} {trade['tp_price']:>10.4f} "
              f"{trade['sl_price']:>10.4f}  "
              f"{result['outcome']:10} {result['mins']:>5.1f}  "
              f"{result['close_price']:>10.4f}  {pnl_str:>10}")

    total = len(trades) - skipped
    print("─" * 122)

    if total == 0:
        print(f"\nSummary (0 trades)")
        return

    print(f"\nSummary ({total} trades)")
    print(f"  TP hit    : {results['TP_HIT']:>4}  ({results['TP_HIT']/total*100:.1f}%)")
    print(f"  SL hit    : {results['SL_HIT']:>4}  ({results['SL_HIT']/total*100:.1f}%)")
    print(f"  Time exit : {results['TIME_EXIT']:>4}  ({results['TIME_EXIT']/total*100:.1f}%)")
    if results["NO_DATA"]:
        print(f"  No data   : {results['NO_DATA']:>4}")
    if skipped:
        print(f"  Skipped   : {skipped:>4}")
    print(f"  Total PnL : {total_pnl:+.4f} USDT  (after fees)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest would-be trades against local candle cache")
    parser.add_argument("csv",
                        help="Path to trades CSV")
    parser.add_argument("--hold", type=int, default=None,
                        help="Max hold minutes (default: unlimited — scan until TP/SL)")
    parser.add_argument("--fee-taker", type=float, default=FEE_TAKER,
                        dest="fee_taker",
                        help=f"Taker fee rate (default: {FEE_TAKER})")
    parser.add_argument("--fee-maker", type=float, default=FEE_MAKER,
                        dest="fee_maker",
                        help=f"Maker fee rate (default: {FEE_MAKER})")
    args = parser.parse_args()
    run(args.csv, args.hold, args.fee_taker, args.fee_maker)
