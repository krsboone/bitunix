"""
vwap_sim.py — VWAP deviation walk-forward simulator

Strategy:
  1. Compute session VWAP (resets at UTC midnight — crypto runs 24/7)
     VWAP = Σ(typical_price × volume) / Σ(volume)
     typical_price = (high + low + close) / 3
  2. Signal when price deviates more than DEV_PCT from VWAP:
       - Momentum mode (default): enter in the direction of deviation
         (price far above VWAP → LONG, price far below VWAP → SHORT)
       - Reversion mode (--revert): enter back toward VWAP
         (price far above VWAP → SHORT, price far below VWAP → LONG)
  3. TP/SL sized as multiples of the deviation distance at signal time
  4. Enter at open of next 1m candle after signal fires
  5. Exit via TP, SL, or MAX_HOLD_MINS time exit
  6. Cooldown: after SL hit, block same-direction entries for N candles

Run fetch_data.py first to build the local candle cache.

Usage:
    python3 vwap_sim.py                          # all symbols, momentum mode
    python3 vwap_sim.py --revert                 # reversion mode
    python3 vwap_sim.py --dev 0.003              # entry at 0.3% deviation
    python3 vwap_sim.py --tp-mult 1.5 --sl-mult 1.0
    python3 vwap_sim.py --symbol BTCUSDT
    python3 vwap_sim.py --start 2026-04-01
    python3 vwap_sim.py --quiet                  # summary only (for sweeps)
"""

import argparse
import csv
import os
from datetime import date, datetime, timezone

# ── Configuration ─────────────────────────────────────────────────────────────

SYMBOLS      = ["BTCUSDT", "ETHUSDT"]
DATA_DIR     = "data"

DEV_PCT      = 0.002    # enter when price deviates ≥ 0.2% from VWAP
TP_MULT      = 1.0      # TP = tp_mult × deviation distance
SL_MULT      = 1.0      # SL = sl_mult × deviation distance
MAX_HOLD_MINS = 33
COOLDOWN     = 10       # candles to block same-direction re-entry after SL

MIN_HISTORY  = 60       # candles before trading begins (warm-up VWAP)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_candles(symbol: str) -> list[dict]:
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def utc_str(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def utc_date(ts_ms: int) -> date:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date()


# ── Simulation ─────────────────────────────────────────────────────────────────

def run_symbol(symbol: str, candles: list[dict],
               start_date: date | None, end_date: date | None,
               args, quiet: bool = False) -> list[dict]:
    trades = []

    if start_date:
        candles = [c for c in candles if utc_date(c["time"]) >= start_date]
    if end_date:
        candles = [c for c in candles if utc_date(c["time"]) <= end_date]

    if len(candles) < MIN_HISTORY + 1:
        if not quiet:
            print(f"  {symbol}: insufficient data ({len(candles)} candles)")
        return []

    if not quiet:
        print(f"  {symbol}: walking {len(candles):,} 1m candles...", flush=True)
        step = max(1, len(candles) // 20)

    # ── State ──────────────────────────────────────────────────────────────────
    state          = "WATCHING"
    long_blocked   = 0
    short_blocked  = 0

    pending_side   = None
    pending_tp     = None
    pending_sl     = None

    entry_price    = None
    tp_price       = None
    sl_price       = None
    trade_side     = None
    trade_open_ts  = None

    # VWAP accumulators — reset at UTC midnight
    vwap_cum_tp_vol = 0.0   # Σ(typical_price × volume)
    vwap_cum_vol    = 0.0   # Σ(volume)
    vwap_day        = None

    for i, c in enumerate(candles):
        if not quiet and i % step == 0:
            pct = i / len(candles) * 100
            print(f"  {symbol}: {pct:5.1f}%  {utc_str(c['time'])[:10]}  trades: {len(trades)}",
                  flush=True)

        ts    = c["time"]
        day   = utc_date(ts)
        price = c["close"]

        # ── Reset VWAP at session open (UTC midnight) ──────────────────────────
        if day != vwap_day:
            vwap_cum_tp_vol = 0.0
            vwap_cum_vol    = 0.0
            vwap_day        = day

        typical = (c["high"] + c["low"] + c["close"]) / 3
        vwap_cum_tp_vol += typical * c["volume"]
        vwap_cum_vol    += c["volume"]
        vwap = vwap_cum_tp_vol / vwap_cum_vol if vwap_cum_vol > 0 else price

        # Need warm-up before trading
        if i < MIN_HISTORY:
            continue

        # ── IN_TRADE: monitor ──────────────────────────────────────────────────
        if state == "IN_TRADE":
            held_mins = (ts - trade_open_ts) / 60_000
            tp_hit = (c["high"] >= tp_price) if trade_side == "LONG" else (c["low"] <= tp_price)
            sl_hit = (c["low"]  <= sl_price) if trade_side == "LONG" else (c["high"] >= sl_price)

            outcome = close_p = None
            if tp_hit and sl_hit:
                if abs(tp_price - entry_price) <= abs(sl_price - entry_price):
                    outcome, close_p = "TP_HIT", tp_price
                else:
                    outcome, close_p = "SL_HIT", sl_price
            elif tp_hit:
                outcome, close_p = "TP_HIT", tp_price
            elif sl_hit:
                outcome, close_p = "SL_HIT", sl_price
            elif held_mins >= args.hold:
                outcome, close_p = "TIME_EXIT", price

            if outcome:
                trades.append({
                    "time":        utc_str(trade_open_ts),
                    "symbol":      symbol,
                    "side":        trade_side,
                    "entry_price": entry_price,
                    "tp":          tp_price,
                    "sl":          sl_price,
                    "outcome":     outcome,
                    "mins":        round((ts - trade_open_ts) / 60_000, 1),
                    "close_price": close_p,
                })
                if outcome == "SL_HIT":
                    if trade_side == "LONG": long_blocked  = args.cooldown
                    else:                    short_blocked = args.cooldown
                state = "WATCHING"
            continue

        # ── ENTRY_PENDING: enter at open of this candle ────────────────────────
        if state == "ENTRY_PENDING":
            entry_price   = c["open"]
            tp_price      = pending_tp
            sl_price      = pending_sl
            trade_side    = pending_side
            trade_open_ts = ts
            state         = "IN_TRADE"
            pending_side  = pending_tp = pending_sl = None
            # Decrement cooldowns
            if long_blocked  > 0: long_blocked  -= 1
            if short_blocked > 0: short_blocked -= 1
            continue

        # Decrement cooldowns
        if long_blocked  > 0: long_blocked  -= 1
        if short_blocked > 0: short_blocked -= 1

        # ── WATCHING: check for VWAP deviation signal ──────────────────────────
        dev = (price - vwap) / vwap   # positive = above VWAP, negative = below

        if abs(dev) < args.dev:
            continue

        dev_dist = abs(price - vwap)

        if dev > 0:
            # Price above VWAP
            if not args.revert:
                # Momentum: price pushing higher → LONG
                if long_blocked > 0:
                    continue
                pending_side = "LONG"
                pending_tp   = price + dev_dist * args.tp_mult
                pending_sl   = price - dev_dist * args.sl_mult
            else:
                # Reversion: price stretched above → SHORT back toward VWAP
                if short_blocked > 0:
                    continue
                pending_side = "SHORT"
                pending_tp   = price - dev_dist * args.tp_mult
                pending_sl   = price + dev_dist * args.sl_mult
        else:
            # Price below VWAP
            if not args.revert:
                # Momentum: price pushing lower → SHORT
                if short_blocked > 0:
                    continue
                pending_side = "SHORT"
                pending_tp   = price - dev_dist * args.tp_mult
                pending_sl   = price + dev_dist * args.sl_mult
            else:
                # Reversion: price stretched below → LONG back toward VWAP
                if long_blocked > 0:
                    continue
                pending_side = "LONG"
                pending_tp   = price + dev_dist * args.tp_mult
                pending_sl   = price - dev_dist * args.sl_mult

        state = "ENTRY_PENDING"

    return trades


# ── Reporting ──────────────────────────────────────────────────────────────────

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


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VWAP deviation walk-forward simulator")
    parser.add_argument("--symbol", nargs="+", metavar="SYMBOL",
                        help=f"Symbol(s) to simulate (default: {', '.join(SYMBOLS)})")
    parser.add_argument("--start", metavar="YYYY-MM-DD")
    parser.add_argument("--end",   metavar="YYYY-MM-DD")
    parser.add_argument("--dev", type=float, default=DEV_PCT,
                        help=f"Min deviation from VWAP to signal (default: {DEV_PCT})")
    parser.add_argument("--tp-mult", dest="tp_mult", type=float, default=TP_MULT,
                        help=f"TP = tp_mult × deviation distance (default: {TP_MULT})")
    parser.add_argument("--sl-mult", dest="sl_mult", type=float, default=SL_MULT,
                        help=f"SL = sl_mult × deviation distance (default: {SL_MULT})")
    parser.add_argument("--hold", type=int, default=MAX_HOLD_MINS,
                        help=f"Max hold minutes (default: {MAX_HOLD_MINS})")
    parser.add_argument("--cooldown", type=int, default=COOLDOWN,
                        help=f"Candles to block re-entry after SL (default: {COOLDOWN})")
    parser.add_argument("--revert", action="store_true",
                        help="Reversion mode: enter back toward VWAP instead of with momentum")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress and trade table (summary only)")

    args    = parser.parse_args()
    symbols = args.symbol or SYMBOLS
    quiet   = args.quiet

    start_date = date.fromisoformat(args.start) if args.start else None
    end_date   = date.fromisoformat(args.end)   if args.end   else None

    if not quiet:
        mode = "REVERSION" if args.revert else "MOMENTUM"
        print(f"\nVWAP Deviation Simulator  [{mode}]")
        print(f"  Symbols   : {', '.join(symbols)}")
        if start_date or end_date:
            print(f"  Date range: {args.start or 'start'} → {args.end or 'now'}")
        print(f"  Dev thresh: {args.dev*100:.2f}%  TP mult: {args.tp_mult}×  SL mult: {args.sl_mult}×")
        print(f"  Hold      : ≤{args.hold}min  Cooldown: {args.cooldown} candles")
        print()

    all_trades = []
    for sym in symbols:
        try:
            candles = load_candles(sym)
            if not quiet:
                print(f"  {sym}: {len(candles)} candles loaded  "
                      f"({utc_str(candles[0]['time'])[:10]} → {utc_str(candles[-1]['time'])[:10]})")
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
