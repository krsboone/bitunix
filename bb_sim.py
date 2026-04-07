"""
bb_sim.py — Bollinger Band mean-reversion walk-forward simulator

Strategy:
  1. Resample 1m candles to 5m internally (no separate data file needed)
  2. Compute Bollinger Bands on completed 5m candles using only prior data
  3. Entry signal: 5m candle CLOSES outside a band
       - Close below lower band → LONG (expect reversion to midline)
       - Close above upper band → SHORT (expect reversion to midline)
  4. Filters applied before entry:
       - Band-width squeeze: only enter when bands are narrow relative to their
         recent rolling average (blocks entries during strong momentum swings)
       - Directional cooldown: after an SL hit, block re-entry in that same
         direction for N 5m candles (prevents successive losses in trending markets)
  5. Enter at open of next 1m candle after signal fires
  6. Exit:
       - TP: price touches midline (SMA at entry time — fixed target)
       - SL: price extends sl_mult × half_band_width beyond the entry band
       - Time exit: MAX_HOLD_MINS

Run fetch_data.py first to build the local candle cache.

Usage:
    python3 bb_sim.py                              # all symbols, defaults
    python3 bb_sim.py --symbol BTCUSDT             # single symbol
    python3 bb_sim.py --period 20 --mult 2.0       # custom BB shape
    python3 bb_sim.py --squeeze 0.85               # tighter squeeze filter
    python3 bb_sim.py --sl-mult 1.5                # wider stop loss
    python3 bb_sim.py --cooldown 10                # 10 candle directional block
    python3 bb_sim.py --start 2026-04-01           # simulate from date
    python3 bb_sim.py --quiet                      # summary only (for sweeps)
"""

import argparse
import csv
import math
import os
from datetime import date, datetime, timezone

# ── Configuration ─────────────────────────────────────────────────────────────

SYMBOLS      = ["BTCUSDT", "ETHUSDT"]
DATA_DIR     = "data"
FIVE_MIN_MS  = 5 * 60 * 1000

# Bollinger Band parameters
BB_PERIOD        = 20     # 5m candles for BB calculation
BB_MULT          = 2.0    # std dev multiplier

# Band-width squeeze filter
SQUEEZE_THRESH   = 1.0    # enter only when bw < bw_rolling_avg × threshold
SQUEEZE_LOOKBACK = 50     # 5m candles for rolling band-width average

# Trade management
SL_MULT          = 1.0    # SL = sl_mult × half_band_width beyond the entry band
COOLDOWN_5M      = 10     # 5m candles to block same-direction entry after SL hit
MAX_HOLD_MINS    = 33

# Minimum completed 5m candles before simulation starts
MIN_HISTORY_5M   = max(BB_PERIOD, SQUEEZE_LOOKBACK) + 10


# ── Data loading ───────────────────────────────────────────────────────────────

def load_candles(symbol: str) -> list[dict]:
    """Load locally-cached 1m candles. Returns list sorted by time."""
    path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No data file for {symbol} at {path}. "
            f"Run: python3 fetch_data.py --symbol {symbol}"
        )
    candles = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles.append({
                "time":   int(row["timestamp_ms"]),
                "open":   float(row["open"]),
                "high":   float(row["high"]),
                "low":    float(row["low"]),
                "close":  float(row["close"]),
                "volume": float(row["volume"]),
            })
    candles.sort(key=lambda c: c["time"])
    return candles


# ── 5m resampling ──────────────────────────────────────────────────────────────

def start_5m(c: dict) -> dict:
    return {"time": c["time"], "open": c["open"], "high": c["high"],
            "low": c["low"], "close": c["close"], "volume": c["volume"]}

def update_5m(c5: dict, c1: dict) -> None:
    c5["high"]    = max(c5["high"], c1["high"])
    c5["low"]     = min(c5["low"],  c1["low"])
    c5["close"]   = c1["close"]
    c5["volume"] += c1["volume"]


# ── Bollinger Band computation ─────────────────────────────────────────────────

def bollinger(buf5: list[dict], period: int, mult: float) -> tuple:
    """
    Compute BB from the last `period` completed 5m candles.
    Returns (sma, upper, lower, half_width) or (None,)*4 if insufficient data.
    """
    if len(buf5) < period:
        return None, None, None, 0.0
    window = [c["close"] for c in buf5[-period:]]
    sma    = sum(window) / period
    var    = sum((p - sma) ** 2 for p in window) / (period - 1)
    std    = math.sqrt(var)
    hw     = mult * std          # half_width = distance from SMA to each band
    return sma, sma + hw, sma - hw, hw


def band_width(sma: float, hw: float) -> float:
    """Relative band width: (upper - lower) / sma = 2 × hw / sma."""
    return 2 * hw / sma if sma > 0 else 0.0


# ── Reporting ──────────────────────────────────────────────────────────────────

def utc_str(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def utc_date(ts_ms: int) -> date:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date()


def print_results(all_trades: list[dict], symbols: list[str],
                  quiet: bool = False) -> None:
    if not all_trades:
        print("No trades recorded.")
        return

    if not quiet:
        col_w = 86
        print("─" * col_w)
        print(f"{'Time':<22} {'Sym':<10} {'Side':<6} {'Entry':>10} "
              f"{'TP':>10} {'SL':>10} {'Outcome':<12} {'Mins':>6} {'Close':>10}")
        print("─" * col_w)
        for t in sorted(all_trades, key=lambda x: x["time"]):
            print(f"{t['time']:<22} {t['symbol']:<10} {t['side']:<6} "
                  f"{t['entry_price']:>10.4f} {t['tp']:>10.4f} {t['sl']:>10.4f} "
                  f"{t['outcome']:<12} {t['mins']:>6.1f} {t['close_price']:>10.4f}")
        print("─" * col_w)

    for sym in symbols:
        sym_trades = [t for t in all_trades if t["symbol"] == sym]
        if sym_trades:
            _print_summary(sym_trades, sym)

    if len(symbols) > 1:
        _print_summary(all_trades, "ALL SYMBOLS")


def _print_summary(trades: list[dict], label: str) -> None:
    n      = len(trades)
    tp     = sum(1 for t in trades if t["outcome"] == "TP_HIT")
    sl     = sum(1 for t in trades if t["outcome"] == "SL_HIT")
    tx     = sum(1 for t in trades if t["outcome"] == "TIME_EXIT")
    longs  = sum(1 for t in trades if t["side"] == "LONG")
    shorts = sum(1 for t in trades if t["side"] == "SHORT")
    print(f"\n  {label} — {n} trades  (LONG {longs}  SHORT {shorts})")
    print(f"    TP hit    : {tp:>4}  ({tp/n*100:.1f}%)")
    print(f"    SL hit    : {sl:>4}  ({sl/n*100:.1f}%)")
    print(f"    Time exit : {tx:>4}  ({tx/n*100:.1f}%)")


# ── Simulation ─────────────────────────────────────────────────────────────────

def run_symbol(symbol: str, candles_1m: list[dict],
               start_date: date | None, end_date: date | None,
               args, quiet: bool = False) -> list[dict]:
    """
    Walk forward through 1m candles, resampling to 5m on the fly.
    Fires BB signals on completed 5m candles; monitors trades on 1m candles.
    """
    trades = []

    # Date filter applied to 1m candles
    if start_date:
        candles_1m = [c for c in candles_1m if utc_date(c["time"]) >= start_date]
    if end_date:
        candles_1m = [c for c in candles_1m if utc_date(c["time"]) <= end_date]

    if not candles_1m:
        print(f"  {symbol}: no candles in date range")
        return []

    # ── State ─────────────────────────────────────────────────────────────────
    buf_5m        = []           # completed 5m candles
    bw_history    = []           # rolling band-width values
    current_5m    = None         # 5m candle in progress
    current_bucket = None        # 5m timestamp of candle in progress

    state         = "WATCHING"
    long_blocked  = 0            # 5m candle count to block LONG entries
    short_blocked = 0            # 5m candle count to block SHORT entries

    # Pending entry parameters (set when signal fires, entered on next 1m open)
    pending_side  = None
    pending_tp    = None
    pending_sl    = None

    # Active trade parameters
    entry_price   = None
    tp_price      = None
    sl_price      = None
    trade_side    = None
    trade_open_ts = None         # 1m timestamp of entry candle

    if not quiet:
        total = len(candles_1m)
        step  = max(1, total // 20)
        print(f"  {symbol}: walking {total:,} 1m candles...", flush=True)

    for i, c1 in enumerate(candles_1m):
        if not quiet and i % step == 0:
            pct = i / len(candles_1m) * 100
            dt  = utc_str(c1["time"])[:10]
            print(f"  {symbol}: {pct:5.1f}%  {dt}  trades: {len(trades)}", flush=True)

        ts     = c1["time"]
        bucket = (ts // FIVE_MIN_MS) * FIVE_MIN_MS

        # ── Update 5m candle ───────────────────────────────────────────────────
        completed_5m = None
        if current_bucket is None:
            current_bucket = bucket
            current_5m     = start_5m(c1)
        elif bucket != current_bucket:
            # Previous 5m candle completed
            completed_5m   = current_5m
            buf_5m.append(completed_5m)
            current_bucket = bucket
            current_5m     = start_5m(c1)
            if long_blocked  > 0: long_blocked  -= 1
            if short_blocked > 0: short_blocked -= 1
        else:
            update_5m(current_5m, c1)

        # ── IN_TRADE: monitor on this 1m candle ───────────────────────────────
        if state == "IN_TRADE":
            held_mins = (ts - trade_open_ts) / 60_000

            tp_hit = (c1["high"] >= tp_price) if trade_side == "LONG" else (c1["low"] <= tp_price)
            sl_hit = (c1["low"]  <= sl_price) if trade_side == "LONG" else (c1["high"] >= sl_price)

            outcome     = None
            close_price = None

            if tp_hit and sl_hit:
                entry_p = entry_price
                if abs(tp_price - entry_p) <= abs(sl_price - entry_p):
                    outcome, close_price = "TP_HIT", tp_price
                else:
                    outcome, close_price = "SL_HIT", sl_price
            elif tp_hit:
                outcome, close_price = "TP_HIT", tp_price
            elif sl_hit:
                outcome, close_price = "SL_HIT", sl_price
            elif held_mins >= args.hold:
                outcome, close_price = "TIME_EXIT", c1["close"]

            if outcome:
                duration = held_mins
                trades.append({
                    "time":        utc_str(trade_open_ts),
                    "symbol":      symbol,
                    "side":        trade_side,
                    "entry_price": entry_price,
                    "tp":          tp_price,
                    "sl":          sl_price,
                    "outcome":     outcome,
                    "mins":        round(duration, 1),
                    "close_price": close_price,
                })
                if outcome == "SL_HIT":
                    if trade_side == "LONG":
                        long_blocked  = args.cooldown
                    else:
                        short_blocked = args.cooldown
                state = "WATCHING"
            continue

        # ── ENTRY_PENDING: enter at open of this 1m candle ────────────────────
        if state == "ENTRY_PENDING":
            entry_price   = c1["open"]
            tp_price      = pending_tp
            sl_price      = pending_sl
            trade_side    = pending_side
            trade_open_ts = ts
            state         = "IN_TRADE"
            pending_side  = pending_tp = pending_sl = None
            continue

        # ── WATCHING: check for BB signal on newly completed 5m candle ────────
        if completed_5m is None or len(buf_5m) < MIN_HISTORY_5M:
            continue

        sma, upper, lower, hw = bollinger(buf_5m, args.period, args.mult)
        if sma is None:
            continue

        # Update rolling band-width history
        bw = band_width(sma, hw)
        bw_history.append(bw)
        if len(bw_history) > args.squeeze_lookback:
            bw_history = bw_history[-args.squeeze_lookback:]

        # Squeeze filter: only enter when bands are narrow relative to recent avg
        bw_avg     = sum(bw_history) / len(bw_history)
        squeeze_ok = bw <= bw_avg * args.squeeze

        price = completed_5m["close"]

        if price < lower and squeeze_ok and long_blocked == 0:
            # LONG signal — enter at next 1m open
            sl_dist    = hw * args.sl_mult
            pending_side = "LONG"
            pending_tp   = sma                # midline reversion target
            pending_sl   = lower - sl_dist    # beyond the lower band
            state        = "ENTRY_PENDING"

        elif price > upper and squeeze_ok and short_blocked == 0:
            # SHORT signal — enter at next 1m open
            sl_dist    = hw * args.sl_mult
            pending_side = "SHORT"
            pending_tp   = sma                # midline reversion target
            pending_sl   = upper + sl_dist    # beyond the upper band
            state        = "ENTRY_PENDING"

    return trades


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bollinger Band mean-reversion walk-forward simulator")

    parser.add_argument("--symbol", nargs="+", metavar="SYMBOL",
                        help=f"Symbol(s) to simulate (default: {', '.join(SYMBOLS)})")
    parser.add_argument("--start", metavar="YYYY-MM-DD",
                        help="Simulate from this date")
    parser.add_argument("--end",   metavar="YYYY-MM-DD",
                        help="Simulate up to this date")

    # BB shape
    parser.add_argument("--period", type=int, default=BB_PERIOD,
                        help=f"BB period in 5m candles (default: {BB_PERIOD})")
    parser.add_argument("--mult",   type=float, default=BB_MULT,
                        help=f"BB std dev multiplier (default: {BB_MULT})")

    # Squeeze filter
    parser.add_argument("--squeeze", type=float, default=SQUEEZE_THRESH,
                        help=f"Band-width squeeze threshold relative to rolling avg "
                             f"(default: {SQUEEZE_THRESH}; lower = tighter filter)")
    parser.add_argument("--squeeze-lookback", dest="squeeze_lookback",
                        type=int, default=SQUEEZE_LOOKBACK,
                        help=f"5m candles for band-width average (default: {SQUEEZE_LOOKBACK})")

    # Trade management
    parser.add_argument("--sl-mult", dest="sl_mult", type=float, default=SL_MULT,
                        help=f"SL = sl_mult × half_band_width beyond band (default: {SL_MULT})")
    parser.add_argument("--cooldown", type=int, default=COOLDOWN_5M,
                        help=f"5m candles to block re-entry after SL hit (default: {COOLDOWN_5M})")
    parser.add_argument("--hold", type=int, default=MAX_HOLD_MINS,
                        help=f"Max hold minutes (default: {MAX_HOLD_MINS})")

    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress and trade table (summary only)")

    args    = parser.parse_args()
    symbols = args.symbol or SYMBOLS
    quiet   = args.quiet

    start_date = date.fromisoformat(args.start) if args.start else None
    end_date   = date.fromisoformat(args.end)   if args.end   else None

    if not quiet:
        print(f"\nBollinger Band Simulator")
        print(f"  Symbols     : {', '.join(symbols)}")
        if start_date or end_date:
            print(f"  Date range  : {args.start or 'start'} → {args.end or 'now'}")
        print(f"  BB          : period={args.period}  mult={args.mult}×")
        print(f"  Squeeze     : threshold={args.squeeze}  lookback={args.squeeze_lookback}")
        print(f"  SL mult     : {args.sl_mult}×  cooldown={args.cooldown} 5m-candles")
        print(f"  Hold        : ≤{args.hold}min")
        print()

    all_trades = []
    for sym in symbols:
        try:
            candles = load_candles(sym)
            if not quiet:
                print(f"  {sym}: {len(candles)} candles loaded  "
                      f"({utc_str(candles[0]['time'])[:10]} "
                      f"→ {utc_str(candles[-1]['time'])[:10]})")
            sym_trades = run_symbol(sym, candles, start_date, end_date, args, quiet=quiet)
            if not quiet:
                print(f"  {sym}: {len(sym_trades)} trades simulated")
            all_trades.extend(sym_trades)
        except FileNotFoundError as e:
            print(f"  {sym}: {e}")
        except Exception as e:
            print(f"  {sym}: error — {e}")

    if not quiet:
        print()
    print_results(all_trades, symbols, quiet=quiet)
    print()


if __name__ == "__main__":
    main()
