"""
sweep_hours.py — Sweep bb_sim.py over hour windows and day combinations

For each (start_hour, end_hour) window and weekend inclusion/exclusion,
runs bb_sim.py in quiet mode and collects TP%, SL%, trade count, and score.

Usage:
    python3 sweep_hours.py                          # BTC+ETH, all window sizes
    python3 sweep_hours.py --symbol BTCUSDT         # single symbol
    python3 sweep_hours.py --min-window 4           # only windows ≥ 4h wide
    python3 sweep_hours.py --max-window 12          # only windows ≤ 12h wide
    python3 sweep_hours.py --start 2026-01-01       # date range passed to sim
    python3 sweep_hours.py --top 20                 # show top N results
"""

import argparse
import itertools
import subprocess
import sys
import re

SYMBOLS_DEFAULT = ["BTCUSDT", "ETHUSDT"]

# Sim presets: name → (script, fixed_args, tp_mult, sl_mult)
# tp_mult and sl_mult are used for EV calculation and must match what's passed in fixed_args.
SIM_PRESETS = {
    "bb": {
        "script":    "bb_sim.py",
        "args":      ["--flip", "--mult", "1.5", "--tp-mult", "1.0", "--sl-mult", "2.0"],
        "tp_mult":   1.0,
        "sl_mult":   2.0,
    },
    "vol_spike": {
        "script":    "vol_spike_sim.py",
        "args":      ["--flip", "--tp-mult", "1.5", "--sl-mult", "1.0"],
        "tp_mult":   1.5,
        "sl_mult":   1.0,
    },
    "exhaustion": {
        "script":    "exhaustion_sim.py",
        "args":      ["--tp-mult", "1.0", "--sl-mult", "1.0"],
        "tp_mult":   1.0,
        "sl_mult":   1.0,
    },
    "sr": {
        "script":    "sr_sim.py",
        "args":      ["--tp", "1.0", "--sl", "3.0", "--vol-mult", "2.0",
                      "--breakout", "0.001", "--arm-distance", "0.0015"],
        "tp_mult":   1.0,
        "sl_mult":   3.0,
    },
    "sr-eth": {
        "script":    "sr_sim.py",
        "args":      ["--tp", "1.0", "--sl", "3.0", "--vol-mult", "3.0",
                      "--breakout", "0.0015", "--arm-distance", "0.003"],
        "tp_mult":   1.0,
        "sl_mult":   3.0,
    },
}


def run_sim(symbol: str, start_hour: int, end_hour: int,
            skip_weekends: bool, extra_args: list[str],
            preset: dict) -> dict | None:
    """Run a sim script for one combination. Returns result dict or None on failure."""
    cmd = [
        sys.executable, preset["script"],
        "--symbol",  symbol,
        "--start-hour", str(start_hour),
        "--end-hour",   str(end_hour),
        "--quiet",
    ] + preset["args"]
    if skip_weekends:
        cmd += ["--skip-days", "sat,sun"]
    cmd += extra_args

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return None

    # Parse summary line: "  SYMBOL — N trades  (LONG x  SHORT y)"
    m_trades = re.search(r"(\w+) — (\d+) trades", output)
    m_tp     = re.search(r"TP hit\s+:\s+\d+\s+\((\d+\.\d+)%\)", output)
    m_sl     = re.search(r"SL hit\s+:\s+\d+\s+\((\d+\.\d+)%\)", output)
    m_tx     = re.search(r"Time exit\s+:\s+\d+\s+\((\d+\.\d+)%\)", output)

    if not (m_trades and m_tp and m_sl):
        return None

    trades = int(m_trades.group(2))
    tp_pct = float(m_tp.group(1))
    sl_pct = float(m_sl.group(1))
    tx_pct = float(m_tx.group(1)) if m_tx else 0.0

    # EV: tp_rate * tp_mult - sl_rate * sl_mult
    ev = (tp_pct / 100) * preset["tp_mult"] - (sl_pct / 100) * preset["sl_mult"]

    return {
        "symbol":   symbol,
        "start":    start_hour,
        "end":      end_hour,
        "window":   end_hour - start_hour if end_hour > start_hour else (24 - start_hour + end_hour),
        "weekends": not skip_weekends,
        "trades":   trades,
        "tp_pct":   tp_pct,
        "sl_pct":   sl_pct,
        "tx_pct":   tx_pct,
        "ev":       ev,
        "score":    tp_pct - sl_pct,
    }


def fmt_window(start: int, end: int) -> str:
    return f"{start:02d}:00–{end:02d}:00"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep bb_sim.py over UTC hour windows and day combinations")
    parser.add_argument("--sim", choices=list(SIM_PRESETS.keys()), default="bb",
                        help="Simulator to sweep: bb, vol_spike, exhaustion, sr, sr-eth (default: bb)")
    parser.add_argument("--symbol", nargs="+", metavar="SYMBOL",
                        default=SYMBOLS_DEFAULT,
                        help=f"Symbol(s) to test (default: {' '.join(SYMBOLS_DEFAULT)})")
    parser.add_argument("--min-window", dest="min_window", type=int, default=1,
                        help="Minimum window width in hours (default: 1)")
    parser.add_argument("--max-window", dest="max_window", type=int, default=12,
                        help="Maximum window width in hours (default: 12)")
    parser.add_argument("--min-trades", dest="min_trades", type=int, default=50,
                        help="Minimum trades to include result (default: 50)")
    parser.add_argument("--top", type=int, default=30,
                        help="Show top N results by EV (default: 30)")
    parser.add_argument("--sort", choices=["ev", "tp", "score", "trades"],
                        default="ev", help="Sort by: ev, tp, score, trades (default: ev)")
    parser.add_argument("--start", metavar="YYYY-MM-DD",
                        help="Pass --start to bb_sim.py")
    parser.add_argument("--end", metavar="YYYY-MM-DD",
                        help="Pass --end to bb_sim.py")
    args = parser.parse_args()

    preset = SIM_PRESETS[args.sim]
    extra = []
    if args.start: extra += ["--start", args.start]
    if args.end:   extra += ["--end",   args.end]

    # Build all (start_hour, end_hour, skip_weekends) combinations
    combos = []
    for start in range(24):
        for window in range(args.min_window, args.max_window + 1):
            end = (start + window) % 24
            combos.append((start, end))

    day_modes = [
        (True,  "Mon–Fri"),
        (False, "All days"),
    ]

    total = len(args.symbol) * len(combos) * len(day_modes)
    print(f"\nSweep: {len(combos)} windows × {len(day_modes)} day modes × "
          f"{len(args.symbol)} symbol(s) = {total} combinations\n")

    results = []
    n = 0
    for sym in args.symbol:
        for (start, end), (skip_wknd, day_label) in itertools.product(combos, day_modes):
            n += 1
            window = end - start if end > start else (24 - start + end)
            print(f"  [{n:>4}/{total}]  {sym}  {fmt_window(start, end)}  "
                  f"({window}h)  {day_label} ...", end=" ", flush=True)

            r = run_sim(sym, start, end, skip_wknd, extra, preset)
            if r is None or r["trades"] < args.min_trades:
                print(f"skip (trades={r['trades'] if r else 0})")
                continue

            print(f"TP {r['tp_pct']:.1f}%  SL {r['sl_pct']:.1f}%  "
                  f"trades {r['trades']:,}  EV {r['ev']:+.4f}")
            r["day_label"] = day_label
            results.append(r)

    if not results:
        print("\nNo results above min-trades threshold.")
        return

    # Sort
    key_map = {
        "ev":     lambda r: r["ev"],
        "tp":     lambda r: r["tp_pct"],
        "score":  lambda r: r["score"],
        "trades": lambda r: r["trades"],
    }
    results.sort(key=key_map[args.sort], reverse=True)

    # Print table
    top = results[:args.top]
    print(f"\n{'─' * 85}")
    print(f"  Top {args.top} by {args.sort.upper()}  |  "
          f"sim={args.sim}  tp={preset['tp_mult']}  sl={preset['sl_mult']}")
    print(f"{'─' * 85}")
    print(f"  {'Rank':>4}  {'Sym':<8} {'Window':<14} {'Days':<10} "
          f"{'Trades':>7} {'TP%':>7} {'SL%':>6} {'TX%':>6}  {'EV':>8}")
    print(f"{'─' * 85}")
    for i, r in enumerate(top, 1):
        marker = " ◀" if i == 1 else ""
        print(f"  {i:>4}  {r['symbol']:<8} {fmt_window(r['start'], r['end']):<14} "
              f"{r['day_label']:<10} {r['trades']:>7,} {r['tp_pct']:>6.1f}% "
              f"{r['sl_pct']:>5.1f}% {r['tx_pct']:>5.1f}%  {r['ev']:>+.4f}{marker}")
    print(f"{'─' * 85}")

    best = top[0]
    print(f"\n  Best: {best['symbol']}  {fmt_window(best['start'], best['end'])}  "
          f"{best['day_label']}  →  TP {best['tp_pct']:.1f}%  "
          f"EV {best['ev']:+.4f}  trades {best['trades']:,}\n")


if __name__ == "__main__":
    main()
