"""
exhaustion_sim.py — Momentum exhaustion walk-forward simulator

Strategy:
  A streak of N consecutive same-direction candles represents sustained
  momentum. When the FINAL candle of that streak has elevated volume
  (≥ vol_mult × rolling average), it signals climactic exhaustion —
  everyone who wanted to join the move just did. Enter in the OPPOSITE
  direction expecting reversal.

  Signal conditions (all must be true):
    1. Last STREAK_LEN candles all close in the same direction
    2. The final candle's volume ≥ VOL_MULT × rolling VOL_LOOKBACK average
    3. No directional cooldown active

  Entry: open of the candle immediately after signal fires
  TP/SL: sized in ATR units (Average True Range over ATR_PERIOD candles)
    - TP = entry ± ATR × TP_MULT
    - SL = entry ∓ ATR × SL_MULT
  Exit: TP hit, SL hit, or MAX_HOLD_MINS time exit
  Cooldown: after SL hit, block same-direction re-entry for COOLDOWN candles

Data-derived rationale:
  Streak alone: ~53% reversal rate
  Vol exhaustion alone: ~54% reversal rate
  Combined (streak + high-vol finale): ~55-56% — consistent across both
  symbols and streak lengths of 3-5. The combination outperforms either
  signal alone.

Run fetch_data.py first to build the local candle cache.

Usage:
    python3 exhaustion_sim.py                        # all symbols, defaults
    python3 exhaustion_sim.py --streak 5             # require 5-candle streak
    python3 exhaustion_sim.py --vol-mult 2.0         # stricter volume threshold
    python3 exhaustion_sim.py --tp-mult 2.0 --sl-mult 1.0
    python3 exhaustion_sim.py --symbol BTCUSDT
    python3 exhaustion_sim.py --start 2026-04-01
    python3 exhaustion_sim.py --quiet                # summary only (for sweeps)
"""

import argparse
import csv
import os
from datetime import date, datetime, timezone

# ── Configuration ─────────────────────────────────────────────────────────────

SYMBOLS      = ["BTCUSDT", "ETHUSDT"]
DATA_DIR     = "data"

STREAK_LEN   = 4       # consecutive same-direction candles required
VOL_MULT     = 1.5     # final candle volume must be ≥ vol_mult × rolling avg
VOL_LOOKBACK = 20      # candles for rolling volume average
ATR_PERIOD   = 14      # candles for ATR
TP_MULT      = 1.5     # TP = entry ± atr × tp_mult
SL_MULT      = 1.0     # SL = entry ∓ atr × sl_mult
MAX_HOLD_MINS = 33
COOLDOWN     = 10      # candles to block same-direction re-entry after SL

MIN_HISTORY  = max(VOL_LOOKBACK, ATR_PERIOD, STREAK_LEN) + 5


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


def compute_atr(candles: list[dict], period: int) -> float:
    """Average True Range over the last `period` candles."""
    if len(candles) < period + 1:
        return 0.0
    window = candles[-(period + 1):]
    true_ranges = []
    for j in range(1, len(window)):
        prev_close = window[j - 1]["close"]
        high = window[j]["high"]
        low  = window[j]["low"]
        tr   = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0


# ── Simulation ─────────────────────────────────────────────────────────────────

def run_symbol(symbol: str, candles: list[dict],
               start_date: date | None, end_date: date | None,
               args, quiet: bool = False) -> list[dict]:
    trades = []

    if start_date:
        candles = [c for c in candles if utc_date(c["time"]) >= start_date]
    if end_date:
        candles = [c for c in candles if utc_date(c["time"]) <= end_date]

    min_hist = max(args.vol_lookback, args.atr_period, args.streak) + 5
    if len(candles) < min_hist + 1:
        if not quiet:
            print(f"  {symbol}: insufficient data ({len(candles)} candles, need {min_hist + 1})")
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

    # Incremental rolling windows — avoids O(n²) candles[:i] slices
    vol_window = []   # rolling volume window (length = vol_lookback)
    atr_window = []   # rolling candle window for ATR (length = atr_period + 1)

    for i, c in enumerate(candles):
        if not quiet and i % step == 0:
            pct = i / len(candles) * 100
            print(f"  {symbol}: {pct:5.1f}%  {utc_str(c['time'])[:10]}  trades: {len(trades)}",
                  flush=True)

        ts    = c["time"]
        price = c["close"]

        # Maintain incremental windows
        vol_window.append(c["volume"])
        if len(vol_window) > args.vol_lookback:
            vol_window.pop(0)
        atr_window.append(c)
        if len(atr_window) > args.atr_period + 2:
            atr_window.pop(0)

        if i < min_hist:
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
            if long_blocked  > 0: long_blocked  -= 1
            if short_blocked > 0: short_blocked -= 1
            continue

        if long_blocked  > 0: long_blocked  -= 1
        if short_blocked > 0: short_blocked -= 1

        # ── WATCHING: check for exhaustion signal ──────────────────────────────
        # Condition 1: streak of N same-direction candles ending at candle i-1
        streak_window = candles[i - args.streak: i]
        dirs = [1 if c2["close"] > c2["open"] else (-1 if c2["close"] < c2["open"] else 0)
                for c2 in streak_window]
        if 0 in dirs: continue          # doji in streak — skip
        if len(set(dirs)) != 1: continue  # not all same direction
        streak_dir = dirs[-1]           # +1 = up streak, -1 = down streak

        # Condition 2: final candle of streak has elevated volume
        prior_vols = [v for v in vol_window[:-1] if v > 0]
        avg_vol    = sum(prior_vols) / len(prior_vols) if prior_vols else 0
        final_vol  = streak_window[-1]["volume"]
        if avg_vol == 0 or final_vol < avg_vol * args.vol_mult:
            continue

        # Condition 3: ATR available for sizing
        atr = compute_atr(atr_window[:-1], args.atr_period)
        if atr == 0:
            continue

        # Signal: enter OPPOSITE to streak direction
        tp_dist = atr * args.tp_mult
        sl_dist = atr * args.sl_mult

        if streak_dir == 1:
            # Up streak exhausted → SHORT
            if short_blocked > 0: continue
            pending_side = "SHORT"
            pending_tp   = price - tp_dist
            pending_sl   = price + sl_dist
        else:
            # Down streak exhausted → LONG
            if long_blocked > 0: continue
            pending_side = "LONG"
            pending_tp   = price + tp_dist
            pending_sl   = price - sl_dist

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
        description="Momentum exhaustion walk-forward simulator")
    parser.add_argument("--symbol", nargs="+", metavar="SYMBOL",
                        help=f"Symbol(s) to simulate (default: {', '.join(SYMBOLS)})")
    parser.add_argument("--start", metavar="YYYY-MM-DD")
    parser.add_argument("--end",   metavar="YYYY-MM-DD")
    parser.add_argument("--streak", type=int, default=STREAK_LEN,
                        help=f"Consecutive same-direction candles required (default: {STREAK_LEN})")
    parser.add_argument("--vol-mult", dest="vol_mult", type=float, default=VOL_MULT,
                        help=f"Final candle volume ≥ vol_mult × rolling avg (default: {VOL_MULT})")
    parser.add_argument("--vol-lookback", dest="vol_lookback", type=int, default=VOL_LOOKBACK,
                        help=f"Candles for rolling volume average (default: {VOL_LOOKBACK})")
    parser.add_argument("--atr-period", dest="atr_period", type=int, default=ATR_PERIOD,
                        help=f"Candles for ATR calculation (default: {ATR_PERIOD})")
    parser.add_argument("--tp-mult", dest="tp_mult", type=float, default=TP_MULT,
                        help=f"TP = entry ± atr × tp_mult (default: {TP_MULT})")
    parser.add_argument("--sl-mult", dest="sl_mult", type=float, default=SL_MULT,
                        help=f"SL = entry ∓ atr × sl_mult (default: {SL_MULT})")
    parser.add_argument("--hold", type=int, default=MAX_HOLD_MINS,
                        help=f"Max hold minutes (default: {MAX_HOLD_MINS})")
    parser.add_argument("--cooldown", type=int, default=COOLDOWN,
                        help=f"Candles to block re-entry after SL (default: {COOLDOWN})")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress and trade table (summary only)")

    args    = parser.parse_args()
    symbols = args.symbol or SYMBOLS
    quiet   = args.quiet

    start_date = date.fromisoformat(args.start) if args.start else None
    end_date   = date.fromisoformat(args.end)   if args.end   else None

    if not quiet:
        print(f"\nMomentum Exhaustion Simulator")
        print(f"  Symbols    : {', '.join(symbols)}")
        if start_date or end_date:
            print(f"  Date range : {args.start or 'start'} → {args.end or 'now'}")
        print(f"  Streak     : {args.streak} candles  Vol mult: {args.vol_mult}×  "
              f"Vol lookback: {args.vol_lookback}")
        print(f"  ATR period : {args.atr_period}  TP mult: {args.tp_mult}×  "
              f"SL mult: {args.sl_mult}×")
        print(f"  Hold       : ≤{args.hold}min  Cooldown: {args.cooldown} candles")
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
