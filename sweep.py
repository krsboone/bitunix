"""
sweep.py — Parameter sweep for sr_sim.py

Runs sr_sim.py across a grid of parameter combinations and ranks results
by TP rate. Prints a sorted summary table — no manual runs needed.

Two sweep modes:
  --mode vol-break  : sweep vol-mult × breakout (default)
  --mode tp-sl      : sweep tp × sl multipliers
  --mode arm        : sweep arm-distance × breakout
  --mode river      : RIVER-specific sweep (wider arm/breakout)

Each mode fixes non-swept parameters at their best-known defaults
so results are comparable. Run vol-break first, then use the winner
as defaults when running tp-sl or arm.

Usage:
    python3 sweep.py                          # vol-mult × breakout sweep
    python3 sweep.py --mode tp-sl             # TP/SL multiplier sweep
    python3 sweep.py --mode arm               # arm-distance × breakout sweep
    python3 sweep.py --mode river             # RIVER-specific tuning
    python3 sweep.py --symbol BTCUSDT         # limit to one symbol
    python3 sweep.py --start 2026-04-01       # limit date range
    python3 sweep.py --vol-mult 2.0           # fix vol-mult, sweep others
"""

import argparse
import subprocess
import sys
import re

# ── Sweep grids ───────────────────────────────────────────────────────────────

GRIDS = {
    "vol-break": {
        "description": "Volume multiplier × breakout threshold",
        "params": [
            {"--vol-mult": vm, "--breakout": bp}
            for vm in [1.5, 2.0, 2.5, 3.0]
            for bp in [0.0005, 0.0010, 0.0015, 0.0020]
        ],
    },
    "tp-sl": {
        "description": "TP × SL multipliers",
        "params": [
            {"--tp": tp, "--sl": sl}
            for tp in [1.0, 1.5, 2.0, 2.5]
            for sl in [1.5, 2.0, 2.5, 3.0]
        ],
    },
    "arm": {
        "description": "Arm distance × breakout threshold",
        "params": [
            {"--arm-distance": arm, "--breakout": bp}
            for arm in [0.0010, 0.0015, 0.0020, 0.0030, 0.0040]
            for bp in [0.0005, 0.0010, 0.0015, 0.0020]
        ],
    },
    "river": {
        "description": "RIVER-specific: arm distance × breakout (wider ranges)",
        "symbol": ["RIVERUSDT"],
        "params": [
            {"--arm-distance": arm, "--breakout": bp, "--vol-mult": vm}
            for arm in [0.0020, 0.0030, 0.0040, 0.0060]
            for bp in [0.0010, 0.0015, 0.0020, 0.0030]
            for vm in [1.5, 2.0]
        ],
    },
}

# Fixed defaults used when a parameter is not being swept
DEFAULTS = {
    "--vol-mult":     2.0,
    "--breakout":     0.0010,
    "--arm-distance": 0.0020,
    "--tp":           1.0,
    "--sl":           3.0,
    "--z":            1.2,
    "--hold":         33,
}


# ── Run one simulation ────────────────────────────────────────────────────────

def run_sim(fixed_args: list[str], sweep_params: dict) -> dict | None:
    """Run sr_sim.py with given parameters. Returns parsed result or None."""
    cmd = [sys.executable, "sr_sim.py", "--quiet"] + fixed_args

    # Apply defaults, then override with sweep params
    merged = {**DEFAULTS, **sweep_params}
    for k, v in merged.items():
        cmd += [k, str(v)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout
    except subprocess.TimeoutExpired:
        print("  [TIMEOUT]", end="")
        return None

    if result.returncode != 0:
        print(f"  [EXIT {result.returncode}]", end="")
        return None

    parsed = parse_output(output, sweep_params)
    if parsed is None:
        # Show last few lines of output for diagnosis
        tail = [l for l in output.split('\n') if l.strip()][-5:]
        print(f"\n  [DEBUG last lines]: {tail}", end="")
    return parsed


def parse_output(output: str, sweep_params: dict) -> dict | None:
    """Extract summary stats from sr_sim.py output."""
    lines = output.split('\n')

    # Find the ALL SYMBOLS summary line, or the last single-symbol summary line
    target = None
    for i, line in enumerate(lines):
        if 'ALL SYMBOLS —' in line and 'trades' in line:
            target = i
            break  # prefer ALL SYMBOLS — stop at first match
    if target is None:
        for i, line in enumerate(lines):
            if '—' in line and 'trades' in line and 'LONG' in line:
                target = i  # keep updating to get the last one

    if target is None:
        return None

    m = re.search(r'— (\d+) trades', lines[target])
    if not m:
        return None
    trades = int(m.group(1))

    tp = sl = tx = None
    tp_pct = sl_pct = tx_pct = None
    for line in lines[target + 1 : target + 8]:
        if 'TP hit' in line:
            m2 = re.search(r':\s+(\d+)\s+\(([0-9.]+)%\)', line)
            if m2:
                tp, tp_pct = int(m2.group(1)), float(m2.group(2))
        elif 'SL hit' in line:
            m2 = re.search(r':\s+(\d+)\s+\(([0-9.]+)%\)', line)
            if m2:
                sl, sl_pct = int(m2.group(1)), float(m2.group(2))
        elif 'Time exit' in line:
            m2 = re.search(r':\s+(\d+)\s+\(([0-9.]+)%\)', line)
            if m2:
                tx, tx_pct = int(m2.group(1)), float(m2.group(2))

    if None in (tp_pct, sl_pct, tx_pct):
        return None

    return {
        "params":   sweep_params,
        "trades":   trades,
        "tp":       tp,
        "tp_pct":   tp_pct,
        "sl":       sl,
        "sl_pct":   sl_pct,
        "tx":       tx,
        "tx_pct":   tx_pct,
        "score":    tp_pct - sl_pct,
    }


# ── Display ───────────────────────────────────────────────────────────────────

def print_results(results: list[dict], param_keys: list[str]) -> None:
    if not results:
        print("No results.")
        return

    results = sorted(results, key=lambda r: r["score"], reverse=True)

    # Header
    param_cols = "  ".join(f"{k:<16}" for k in param_keys)
    print(f"\n{'Rank':<5} {param_cols} {'Trades':>6}  {'TP%':>6}  {'SL%':>6}  {'TX%':>6}  {'Score':>7}")
    print("─" * (5 + 16 * len(param_keys) + 40))

    for rank, r in enumerate(results, 1):
        param_vals = "  ".join(
            f"{str(r['params'].get(k, DEFAULTS.get(k, '?'))):<16}"
            for k in param_keys
        )
        marker = " ◀" if rank == 1 else ""
        print(f"{rank:<5} {param_vals} {r['trades']:>6}  "
              f"{r['tp_pct']:>6.1f}  {r['sl_pct']:>6.1f}  "
              f"{r['tx_pct']:>6.1f}  {r['score']:>7.1f}{marker}")

    print()
    best = results[0]
    print("  Best combination:")
    for k, v in best["params"].items():
        print(f"    {k} {v}")
    print(f"  TP: {best['tp_pct']:.1f}%  SL: {best['sl_pct']:.1f}%  "
          f"Score (TP-SL): {best['score']:.1f}  Trades: {best['trades']}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parameter sweep for sr_sim.py")
    parser.add_argument("--mode", default="vol-break",
                        choices=list(GRIDS.keys()),
                        help="Sweep mode (default: vol-break)")
    parser.add_argument("--symbol", nargs="+", metavar="SYMBOL",
                        help="Symbol(s) to simulate (overrides mode default)")
    parser.add_argument("--start",  metavar="YYYY-MM-DD")
    parser.add_argument("--end",    metavar="YYYY-MM-DD")

    # Allow fixing individual params when sweeping others
    parser.add_argument("--vol-mult",     type=float)
    parser.add_argument("--breakout",     type=float)
    parser.add_argument("--arm-distance", type=float)
    parser.add_argument("--tp",           type=float)
    parser.add_argument("--sl",           type=float)

    args = parser.parse_args()

    grid = GRIDS[args.mode]
    print(f"\nSweep mode : {args.mode} — {grid['description']}")

    # Build fixed args (symbol, date range)
    symbols = args.symbol or grid.get("symbol")
    fixed = []
    if symbols:
        fixed += ["--symbol"] + symbols
    if args.start:
        fixed += ["--start", args.start]
    if args.end:
        fixed += ["--end", args.end]

    # Apply any manually fixed params over DEFAULTS
    overrides = {}
    for k, attr in [("--vol-mult", "vol_mult"), ("--breakout", "breakout"),
                    ("--arm-distance", "arm_distance"), ("--tp", "tp"),
                    ("--sl", "sl")]:
        v = getattr(args, attr, None)
        if v is not None:
            overrides[k] = v

    param_keys = list(grid["params"][0].keys())
    total = len(grid["params"])
    print(f"Combinations: {total}")
    if symbols:
        print(f"Symbols    : {', '.join(symbols)}")
    print()

    results = []
    for idx, sweep_params in enumerate(grid["params"], 1):
        merged_params = {**overrides, **sweep_params}
        param_str = "  ".join(f"{k} {v}" for k, v in sweep_params.items())
        print(f"  [{idx:>3}/{total}]  {param_str} ...", end="", flush=True)
        result = run_sim(fixed, merged_params)
        if result:
            results.append(result)
            print(f"  TP {result['tp_pct']:.1f}%  SL {result['sl_pct']:.1f}%  "
                  f"trades {result['trades']}")
        else:
            print("  (no result)")

    print_results(results, param_keys)


if __name__ == "__main__":
    main()
