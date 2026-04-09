"""
sr_sim.py — Support/Resistance breakout walk-forward simulator

Tests the S/R breakout strategy against locally-cached candle data (from
fetch_data.py) with no lookahead bias. Walks candle-by-candle, deriving
all signals only from data available at that point in time.

Strategy:
  1. Identify S/R levels at each candle using only prior data:
       - Current session high/low (from UTC midnight to this candle)
       - 4-hour rolling high/low (last 240 candles)
       - Previous session high/low (prior UTC calendar day)
  2. ARM when price closes within ARM_DISTANCE of any level
  3. CONFIRM when the next candle closes beyond the level by BREAKOUT_PCT,
     with volume above VOL_MULT × rolling average and Z-score confirming direction
  4. ENTER at the open of the candle following confirmation
  5. Exit via TP, SL, or MAX_HOLD_MINS time exit

Run fetch_data.py first to build the local candle cache.

Usage:
    python3 sr_sim.py                              # all symbols, defaults
    python3 sr_sim.py --symbol BTCUSDT             # single symbol
    python3 sr_sim.py --symbol BTCUSDT ETHUSDT     # subset
    python3 sr_sim.py --tp 2.0 --sl 1.5           # custom TP/SL multipliers
    python3 sr_sim.py --vol-mult 2.0 --z 1.5      # tighter entry confirmation
    python3 sr_sim.py --breakout 0.0015            # wider breakout threshold
    python3 sr_sim.py --start 2026-04-01           # simulate from specific date
    python3 sr_sim.py --start 2026-04-01 --end 2026-04-05
"""

import argparse
import csv
import math
import os
from datetime import datetime, date, timezone

# ── Configuration (tunable via args) ─────────────────────────────────────────

#SYMBOLS        = ["BTCUSDT", "ETHUSDT", "RIVERUSDT"]
SYMBOLS        = ["BTCUSDT", "ETHUSDT"]
DATA_DIR       = "data"
INTERVAL       = "1m"

# S/R level detection
# Per-symbol sweep winners: BTC arm=0.0015 break=0.001 vm=2.0 | ETH arm=0.003 break=0.0015 vm=3.0
ARM_DISTANCE   = 0.0020   # arm when price within 0.20% of a level
BREAKOUT_PCT   = 0.0010   # confirm when candle close is 0.10% beyond level
ROLLING_4H     = 240      # 1m candles in 4 hours

# Volume confirmation
VOL_MULT       = 2.0      # volume must be VOL_MULT × rolling average to confirm
VOL_LOOKBACK   = 20       # candles for rolling volume average

# Entry signal (Z-score)
SIGMA_CANDLES  = 20
SIGNAL_CANDLES = 5
Z_ENTRY        = 1.2

# Trade management — tp/sl sweep winners: both symbols agree tp=1.0 sl=3.0
TP_MULT        = 1.0
SL_MULT        = 3.0
HOLD_INTERVALS = 15       # for sigma_hold calculation (1.5σ√15)
MAX_HOLD_MINS  = 33

# Level cooldown — don't re-trigger same level for N candles after a breakout
LEVEL_COOLDOWN = 30

# Minimum history required before simulation begins
MIN_HISTORY    = max(ROLLING_4H, SIGMA_CANDLES + SIGNAL_CANDLES, VOL_LOOKBACK) + 10


# ── Data loading ──────────────────────────────────────────────────────────────

def load_candles(symbol: str, interval: str = INTERVAL) -> list[dict]:
    """Load locally-cached candles for a symbol. Returns list sorted by time."""
    path = os.path.join(DATA_DIR, f"{symbol}_{interval}.csv")
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


# ── Signal computation ────────────────────────────────────────────────────────

def compute_sigma(candles: list) -> float:
    prices = [c["close"] for c in candles]
    if len(prices) < 2:
        return 0.0
    log_returns = [math.log(prices[i] / prices[i - 1])
                   for i in range(1, len(prices))]
    n    = len(log_returns)
    mean = sum(log_returns) / n
    var  = sum((r - mean) ** 2 for r in log_returns) / (n - 1)
    return math.sqrt(var)

def compute_z(candles: list, sigma: float) -> float:
    if sigma == 0 or len(candles) < 2:
        return 0.0
    prices = [c["close"] for c in candles]
    drift  = math.log(prices[-1] / prices[0])
    return drift / (sigma * math.sqrt(len(candles) - 1))

def rolling_vol_avg(candles: list) -> float:
    vols = [c["volume"] for c in candles if c["volume"] > 0]
    return sum(vols) / len(vols) if vols else 0.0


# ── S/R level computation ─────────────────────────────────────────────────────

def utc_date(ts_ms: int) -> date:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date()

def get_levels(candles: list, i: int) -> list[float]:
    """
    Compute S/R levels at candle i using only candles[0..i-1].
    Returns list of distinct price levels.
    """
    prior = candles[:i]
    if not prior:
        return []

    levels = set()
    current_day = utc_date(candles[i]["time"])

    # ── Current session high/low ───────────────────────────────────────────
    session = [c for c in prior if utc_date(c["time"]) == current_day]
    if session:
        levels.add(max(c["high"] for c in session))
        levels.add(min(c["low"]  for c in session))

    # ── 4-hour rolling high/low ────────────────────────────────────────────
    window_4h = prior[-ROLLING_4H:]
    if window_4h:
        levels.add(max(c["high"] for c in window_4h))
        levels.add(min(c["low"]  for c in window_4h))

    # ── Previous session high/low ──────────────────────────────────────────
    prev_day = [c for c in prior if utc_date(c["time"]) < current_day]
    if prev_day:
        prev_session_day = max(utc_date(c["time"]) for c in prev_day)
        prev_session = [c for c in prev_day
                        if utc_date(c["time"]) == prev_session_day]
        if prev_session:
            levels.add(max(c["high"] for c in prev_session))
            levels.add(min(c["low"]  for c in prev_session))

    return sorted(levels)


# ── Simulation state ──────────────────────────────────────────────────────────

class State:
    WATCHING = "WATCHING"
    ARMED    = "ARMED"
    IN_TRADE = "IN_TRADE"


def run_symbol(symbol: str, candles: list,
               start_date: date | None, end_date: date | None,
               args, quiet: bool = False) -> list[dict]:
    """
    Walk forward through candles, simulate the S/R breakout strategy.
    Returns list of completed trade records.
    """
    trades = []

    # Apply date filters
    if start_date:
        candles = [c for c in candles if utc_date(c["time"]) >= start_date]
    if end_date:
        candles = [c for c in candles if utc_date(c["time"]) <= end_date]

    if len(candles) < MIN_HISTORY + 1:
        print(f"  {symbol}: insufficient data ({len(candles)} candles, "
              f"need {MIN_HISTORY + 1})")
        return []

    state         = State.WATCHING
    armed_level   = None
    armed_dir     = None    # "UP" or "DOWN"
    entry_pending = False   # enter on next candle open
    entry_data    = {}
    trade_entry_candle = None

    total   = len(candles) - MIN_HISTORY
    step    = max(1, total // 20)   # print ~20 progress updates
    if not quiet:
        print(f"  {symbol}: walking {total:,} candles...", flush=True)

    # Track recently broken levels to avoid re-triggering
    broken_levels: dict[float, int] = {}   # level → candle index when broken

    # ── Incremental S/R session tracking (replaces O(n²) get_levels) ──────────
    # Build initial session state from the warm-up period (candles[:MIN_HISTORY])
    _sess_high = _sess_low = None
    _prev_sess_high = _prev_sess_low = None
    _sess_day = None
    for _c in candles[:MIN_HISTORY]:
        _d = utc_date(_c["time"])
        if _d != _sess_day:
            if _sess_high is not None:
                _prev_sess_high = _sess_high
                _prev_sess_low  = _sess_low
            _sess_day  = _d
            _sess_high = _c["high"]
            _sess_low  = _c["low"]
        else:
            _sess_high = max(_sess_high, _c["high"])
            _sess_low  = min(_sess_low,  _c["low"])

    for i in range(MIN_HISTORY, len(candles) - 1):
        c = candles[i]
        price = c["close"]
        ts    = c["time"]

        # ── Update session state with previous candle (so levels reflect candles[:i]) ──
        if i > MIN_HISTORY:
            pc   = candles[i - 1]
            pc_d = utc_date(pc["time"])
            c_d  = utc_date(c["time"])
            if c_d != pc_d:                      # day rolled over
                _prev_sess_high = _sess_high
                _prev_sess_low  = _sess_low
                _sess_high = _sess_low = None
            _sess_high = max(_sess_high, pc["high"]) if _sess_high is not None else pc["high"]
            _sess_low  = min(_sess_low,  pc["low"])  if _sess_low  is not None else pc["low"]

        # ── Pre-compute levels for this candle (O(240) not O(n)) ──────────────
        _lvl = set()
        if _sess_high is not None:
            _lvl.add(_sess_high)
            _lvl.add(_sess_low)
        _w4h = candles[max(0, i - ROLLING_4H):i]
        if _w4h:
            _lvl.add(max(c2["high"] for c2 in _w4h))
            _lvl.add(min(c2["low"]  for c2 in _w4h))
        if _prev_sess_high is not None:
            _lvl.add(_prev_sess_high)
            _lvl.add(_prev_sess_low)
        _precomputed_levels = sorted(_lvl)

        # ── Progress indicator ─────────────────────────────────────────────
        if not quiet:
            elapsed = i - MIN_HISTORY
            if elapsed % step == 0:
                pct  = elapsed / total * 100
                dt   = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                print(f"  {symbol}: {pct:5.1f}%  {dt}  trades so far: {len(trades)}",
                      flush=True)

        # ── Expire broken level cooldowns ──────────────────────────────────
        broken_levels = {lvl: idx for lvl, idx in broken_levels.items()
                         if i - idx < args.cooldown}

        # ── Compute signals ────────────────────────────────────────────────
        sigma_candles  = candles[i - SIGMA_CANDLES - SIGNAL_CANDLES: i - SIGNAL_CANDLES]
        signal_candles = candles[i - SIGNAL_CANDLES: i]
        vol_candles    = candles[i - args.vol_lookback: i]

        sigma    = compute_sigma(sigma_candles)
        z        = compute_z(signal_candles, sigma) if sigma > 0 else 0.0
        vol_avg  = rolling_vol_avg(vol_candles)
        vol_now  = c["volume"]

        # ── IN_TRADE — monitor position ────────────────────────────────────
        if state == State.IN_TRADE:
            held_mins = (ts - trade_entry_candle["time"]) / 60_000

            tp = entry_data["tp"]
            sl = entry_data["sl"]
            side = entry_data["side"]

            tp_hit = (c["high"] >= tp) if side == "LONG" else (c["low"] <= tp)
            sl_hit = (c["low"]  <= sl) if side == "LONG" else (c["high"] >= sl)

            outcome    = None
            close_price = None

            if tp_hit and sl_hit:
                # Both triggered — use whichever is closer to entry
                entry_p = entry_data["entry_price"]
                if abs(tp - entry_p) <= abs(sl - entry_p):
                    outcome, close_price = "TP_HIT", tp
                else:
                    outcome, close_price = "SL_HIT", sl
            elif tp_hit:
                outcome, close_price = "TP_HIT", tp
            elif sl_hit:
                outcome, close_price = "SL_HIT", sl
            elif held_mins >= args.hold:
                outcome = "TIME_EXIT"
                close_price = price

            if outcome:
                duration = (ts - trade_entry_candle["time"]) / 60_000
                trades.append({
                    "time":        datetime.fromtimestamp(
                                       trade_entry_candle["time"] / 1000,
                                       tz=timezone.utc
                                   ).strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol":      symbol,
                    "side":        side,
                    "entry_price": entry_data["entry_price"],
                    "tp":          tp,
                    "sl":          sl,
                    "outcome":     outcome,
                    "mins":        round(duration, 1),
                    "close_price": close_price,
                    "level":       entry_data["level"],
                })
                state = State.WATCHING
                entry_pending = False
            continue

        # ── Entry pending — enter at this candle's open ────────────────────
        if entry_pending:
            entry_price = candles[i]["open"]
            sigma_hold  = entry_data["sigma"] * math.sqrt(args.hold_intervals)
            tp_move     = entry_price * sigma_hold * args.tp
            sl_move     = entry_price * sigma_hold * args.sl

            if entry_data["side"] == "LONG":
                tp = entry_price + tp_move
                sl = entry_price - sl_move
            else:
                tp = entry_price - tp_move
                sl = entry_price + sl_move

            entry_data["entry_price"] = entry_price
            entry_data["tp"]          = tp
            entry_data["sl"]          = sl
            trade_entry_candle        = candles[i]
            state                     = State.IN_TRADE
            entry_pending             = False
            # Mark this level as broken
            broken_levels[entry_data["level"]] = i
            continue

        # ── ARMED — watching for breakout confirmation ─────────────────────
        if state == State.ARMED:
            level = armed_level
            direction = armed_dir

            # Disarm if price has moved away from the level
            dist_pct = abs(price - level) / level
            if dist_pct > args.arm_distance * 3:
                state = State.WATCHING
                armed_level = armed_dir = None
                continue

            # Check breakout: candle body must close convincingly beyond level
            if direction == "UP":
                body_close = price > c["open"]   # bullish candle
                broke = price > level * (1 + args.breakout)
                z_ok  = z >= args.z_entry
            else:
                body_close = price < c["open"]   # bearish candle
                broke = price < level * (1 - args.breakout)
                z_ok  = z <= -args.z_entry

            vol_ok = vol_avg > 0 and vol_now >= vol_avg * args.vol_mult

            if broke and z_ok and vol_ok:
                side = "LONG" if direction == "UP" else "SHORT"
                entry_data = {
                    "side":  side,
                    "sigma": sigma,
                    "level": level,
                }
                entry_pending = True
                state         = State.WATCHING
                armed_level   = armed_dir = None
            continue

        # ── WATCHING — scan for levels to arm ─────────────────────────────
        # Time-of-day and day-of-week filter
        candle_dt   = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        candle_hour = candle_dt.hour
        candle_dow  = candle_dt.weekday()  # 0=Mon … 6=Sun
        if hasattr(args, "start_hour") and args.start_hour is not None:
            if hasattr(args, "end_hour") and args.end_hour is not None:
                if args.start_hour <= args.end_hour:
                    if not (args.start_hour <= candle_hour < args.end_hour):
                        continue
                else:
                    if not (candle_hour >= args.start_hour or candle_hour < args.end_hour):
                        continue
        if hasattr(args, "skip_days") and args.skip_days:
            if candle_dow in args.skip_days:
                continue

        levels = _precomputed_levels
        if not levels:
            continue

        # Skip levels that were recently broken
        levels = [lvl for lvl in levels if lvl not in broken_levels]
        if not levels:
            continue

        # Find nearest level to current price
        nearest = min(levels, key=lambda lvl: abs(price - lvl))
        dist_pct = abs(price - nearest) / nearest

        if dist_pct <= args.arm_distance:
            # Determine expected breakout direction
            direction = "UP" if price < nearest else "DOWN"
            armed_level = nearest
            armed_dir   = direction
            state       = State.ARMED

    return trades


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_results(all_trades: list[dict], symbols: list[str],
                  quiet: bool = False) -> None:
    if not all_trades:
        print("No trades recorded.")
        return

    if not quiet:
        # ── Trade table ────────────────────────────────────────────────────
        col_w = 80
        print("─" * col_w)
        print(f"{'Time':<22} {'Sym':<10} {'Side':<6} {'Entry':>10} "
              f"{'TP':>10} {'SL':>10} {'Outcome':<12} {'Mins':>6} {'Close':>10}")
        print("─" * col_w)

        for t in sorted(all_trades, key=lambda x: x["time"]):
            print(f"{t['time']:<22} {t['symbol']:<10} {t['side']:<6} "
                  f"{t['entry_price']:>10.4f} {t['tp']:>10.4f} {t['sl']:>10.4f} "
                  f"{t['outcome']:<12} {t['mins']:>6.1f} {t['close_price']:>10.4f}")

        print("─" * col_w)

    # ── Per-symbol summary ─────────────────────────────────────────────────
    for sym in symbols:
        sym_trades = [t for t in all_trades if t["symbol"] == sym]
        if not sym_trades:
            continue
        _print_summary(sym_trades, sym)

    # ── Overall summary ────────────────────────────────────────────────────
    if len(symbols) > 1:
        _print_summary(all_trades, "ALL SYMBOLS")


def _print_summary(trades: list[dict], label: str) -> None:
    n      = len(trades)
    tp     = sum(1 for t in trades if t["outcome"] == "TP_HIT")
    sl     = sum(1 for t in trades if t["outcome"] == "SL_HIT")
    tx     = sum(1 for t in trades if t["outcome"] == "TIME_EXIT")
    longs  = sum(1 for t in trades if t["side"] == "LONG")
    shorts = sum(1 for t in trades if t["side"] == "SHORT")

    print(f"\n  {label} — {n} trades  "
          f"(LONG {longs}  SHORT {shorts})")
    print(f"    TP hit    : {tp:>4}  ({tp/n*100:.1f}%)")
    print(f"    SL hit    : {sl:>4}  ({sl/n*100:.1f}%)")
    print(f"    Time exit : {tx:>4}  ({tx/n*100:.1f}%)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="S/R breakout walk-forward simulator")

    parser.add_argument("--symbol", nargs="+", metavar="SYMBOL",
                        help=f"Symbol(s) to simulate (default: {', '.join(SYMBOLS)})")
    parser.add_argument("--start",  metavar="YYYY-MM-DD",
                        help="Simulate from this date (default: all available data)")
    parser.add_argument("--end",    metavar="YYYY-MM-DD",
                        help="Simulate up to this date (default: all available data)")

    # Strategy parameters
    parser.add_argument("--arm-distance", type=float, default=ARM_DISTANCE,
                        help=f"Arm when price within X%% of level (default: {ARM_DISTANCE})")
    parser.add_argument("--breakout", type=float, default=BREAKOUT_PCT,
                        help=f"Confirm when close X%% beyond level (default: {BREAKOUT_PCT})")
    parser.add_argument("--vol-mult", type=float, default=VOL_MULT,
                        help=f"Volume multiplier for confirmation (default: {VOL_MULT})")
    parser.add_argument("--vol-lookback", type=int, default=VOL_LOOKBACK,
                        help=f"Candles for volume average (default: {VOL_LOOKBACK})")
    parser.add_argument("--z", dest="z_entry", type=float, default=Z_ENTRY,
                        help=f"Z-score entry threshold (default: {Z_ENTRY})")
    parser.add_argument("--tp", type=float, default=TP_MULT,
                        help=f"TP multiplier in sigma units (default: {TP_MULT})")
    parser.add_argument("--sl", type=float, default=SL_MULT,
                        help=f"SL multiplier in sigma units (default: {SL_MULT})")
    parser.add_argument("--hold", type=int, default=MAX_HOLD_MINS,
                        help=f"Max hold minutes (default: {MAX_HOLD_MINS})")
    parser.add_argument("--hold-intervals", type=int, default=HOLD_INTERVALS,
                        help=f"Hold intervals for sigma scaling (default: {HOLD_INTERVALS})")
    parser.add_argument("--cooldown", type=int, default=LEVEL_COOLDOWN,
                        help=f"Candles before re-triggering same level (default: {LEVEL_COOLDOWN})")
    parser.add_argument("--start-hour", dest="start_hour", type=int, default=None,
                        metavar="H", help="Only enter trades at/after this UTC hour (0-23)")
    parser.add_argument("--end-hour", dest="end_hour", type=int, default=None,
                        metavar="H", help="Only enter trades before this UTC hour (0-23)")
    parser.add_argument("--skip-days", dest="skip_days_str", type=str, default=None,
                        metavar="DAYS",
                        help="Comma-separated days to skip entries (mon,tue,wed,thu,fri,sat,sun)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress and trade table (summary only)")

    args    = parser.parse_args()
    symbols = args.symbol or SYMBOLS
    quiet   = args.quiet

    start_date = (date.fromisoformat(args.start) if args.start else None)
    end_date   = (date.fromisoformat(args.end)   if args.end   else None)

    # Parse --skip-days into set of weekday ints (0=Mon … 6=Sun)
    _day_map = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    args.skip_days = set()
    if args.skip_days_str:
        for d in args.skip_days_str.lower().split(","):
            d = d.strip()
            if d in _day_map:
                args.skip_days.add(_day_map[d])
            else:
                print(f"  Warning: unknown day '{d}' — use mon,tue,wed,thu,fri,sat,sun")

    if not quiet:
        print(f"\nS/R Breakout Simulator")
        print(f"  Symbols     : {', '.join(symbols)}")
        if start_date or end_date:
            print(f"  Date range  : {args.start or 'start'} → {args.end or 'now'}")
        print(f"  Arm dist    : {args.arm_distance*100:.2f}%  "
              f"Breakout: {args.breakout*100:.3f}%  "
              f"Vol mult: {args.vol_mult}×")
        print(f"  Z-entry     : {args.z_entry}  "
              f"TP/SL: {args.tp}/{args.sl}σ  "
              f"Hold: {args.hold}min")
        if args.start_hour is not None or args.end_hour is not None:
            sh = f"{args.start_hour:02d}:00" if args.start_hour is not None else "00:00"
            eh = f"{args.end_hour:02d}:00"   if args.end_hour   is not None else "24:00"
            print(f"  Trade hours : {sh} → {eh} UTC")
        if args.skip_days:
            _day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            skipped = ", ".join(_day_names[d] for d in sorted(args.skip_days))
            print(f"  Skip days   : {skipped}")
        print()

    all_trades = []

    for sym in symbols:
        try:
            candles = load_candles(sym)
            if not quiet:
                print(f"  {sym}: {len(candles)} candles loaded  "
                      f"({datetime.fromtimestamp(candles[0]['time']/1000, tz=timezone.utc).strftime('%Y-%m-%d')} "
                      f"→ {datetime.fromtimestamp(candles[-1]['time']/1000, tz=timezone.utc).strftime('%Y-%m-%d')})")
            trades = run_symbol(sym, candles, start_date, end_date, args, quiet=quiet)
            if not quiet:
                print(f"  {sym}: {len(trades)} trades simulated")
            all_trades.extend(trades)
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
