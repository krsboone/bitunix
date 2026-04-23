"""
reconcile_pnl.py — Fetch actual realized PnL from Bitunix for closed positions.

Reads close_events.csv, queries Bitunix position history by positionId, and
writes actual fill data to log/reconciled_pnl.csv. Idempotent — skips any
order_id already present in the output file.

Usage:
    python3 reconcile_pnl.py          # reconcile all unprocessed positions
    python3 reconcile_pnl.py --show   # also print full per-trade table

Columns in reconciled_pnl.csv:
    arm_id, order_id, strategy, symbol, side, outcome, close_time
    est_exit_price  — price logged at close time (unreliable for EXCHANGE_CLOSED)
    est_pnl         — PnL estimated by the trader script
    actual_entry_price, actual_close_price
    realized_pnl    — Bitunix realizedPNL field: already net of fees and funding
    fee             — fees paid (positive number, already deducted from realized_pnl)
    funding         — funding paid (positive number, already deducted from realized_pnl)
    net_pnl         — same as realized_pnl (confirmed: Bitunix deducts fee+funding before reporting)
"""

import argparse
import csv
import os
import time

from auth import BitunixClient
from config import API_KEY, SECRET_KEY

CLOSE_EVENTS_CSV = os.path.join("log", "close_events.csv")
RECONCILED_CSV   = os.path.join("log", "reconciled_pnl.csv")

_HEADER = [
    "arm_id", "order_id", "strategy", "symbol", "side", "outcome", "close_time",
    "est_exit_price", "est_pnl",
    "actual_entry_price", "actual_close_price",
    "realized_pnl", "fee", "funding", "net_pnl",
]


def _load_done() -> set[str]:
    """Return set of order_ids already written to reconciled_pnl.csv."""
    if not os.path.exists(RECONCILED_CSV):
        return set()
    with open(RECONCILED_CSV, newline="") as f:
        return {row["order_id"] for row in csv.DictReader(f) if row.get("order_id")}


def _fetch_symbol_history(client: BitunixClient, symbol: str) -> dict[str, dict]:
    """Fetch all closed positions for a symbol and return a {positionId: data} lookup."""
    try:
        resp = client.get("/api/v1/futures/position/get_history_positions",
                          {"symbol": symbol, "limit": "200"})
    except Exception as exc:
        print(f"  Request error fetching {symbol} history: {exc}")
        return {}

    if resp.get("code") != 0:
        print(f"  API error fetching {symbol} history: {resp.get('msg')}")
        return {}

    positions = resp.get("data", {}).get("positionList", [])
    return {p["positionId"]: p for p in positions if p.get("positionId")}


def reconcile(show: bool = False) -> None:
    client = BitunixClient(API_KEY, SECRET_KEY)
    done   = _load_done()

    if not os.path.exists(CLOSE_EVENTS_CSV):
        print("No close_events.csv found.")
        return

    with open(CLOSE_EVENTS_CSV, newline="") as f:
        all_rows = list(csv.DictReader(f))

    pending = [
        r for r in all_rows
        if r.get("order_id")
        and not r["order_id"].startswith("DEBUG")
        and r["order_id"] not in done
    ]

    if not pending:
        print("Nothing new to reconcile.")
        _print_summary(show)
        return

    print(f"Reconciling {len(pending)} position(s)...\n")

    # Fetch history once per symbol — much cheaper than one API call per position
    symbols  = {r["symbol"] for r in pending if r.get("symbol")}
    history  = {}   # symbol → {positionId: data}
    for sym in sorted(symbols):
        history[sym] = _fetch_symbol_history(client, sym)
        print(f"  Fetched {len(history[sym])} closed positions for {sym}")
        time.sleep(0.3)
    print()

    new_rows = []
    for row in pending:
        order_id = row["order_id"]
        symbol   = row.get("symbol", "")
        label    = f"{row.get('strategy','?')} {symbol} {row.get('side','?')}"
        print(f"  {label}  {row.get('close_time','')}  [{order_id}]")

        p = history.get(symbol, {}).get(order_id)
        if p is None:
            print(f"    positionId not found in {symbol} history "
                  f"(may be a pre-fix orderId or not yet settled)")
            continue

        realized  = _f(p.get("realizedPNL"))
        fee       = _f(p.get("fee"))
        funding   = _f(p.get("funding"))
        net       = realized   # realizedPNL is already net of fee and funding

        print(f"    entry={p.get('entryPrice')}  close={p.get('closePrice')}"
              f"  realized={realized:+.6f}  fee={fee:.6f}  funding={funding:.6f}"
              f"  net={net:+.6f}  (est was {row.get('realized_pnl') or '—'})")

        new_rows.append({
            "arm_id":              row.get("arm_id", ""),
            "order_id":            order_id,
            "strategy":            row.get("strategy", ""),
            "symbol":              row.get("symbol", ""),
            "side":                row.get("side", ""),
            "outcome":             row.get("outcome", ""),
            "close_time":          row.get("close_time", ""),
            "est_exit_price":      row.get("exit_price", ""),
            "est_pnl":             row.get("realized_pnl", ""),
            "actual_entry_price":  p.get("entryPrice", ""),
            "actual_close_price":  p.get("closePrice", ""),
            "realized_pnl":        f"{realized:.6f}",
            "fee":                 f"{fee:.6f}",
            "funding":             f"{funding:.6f}",
            "net_pnl":             f"{net:.6f}",
        })

    if new_rows:
        write_header = not os.path.exists(RECONCILED_CSV)
        with open(RECONCILED_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_HEADER)
            if write_header:
                w.writeheader()
            w.writerows(new_rows)
        print(f"\nWrote {len(new_rows)} row(s) to {RECONCILED_CSV}")
    else:
        print("\nNo new data written.")

    _print_summary(show)


def _print_summary(show: bool) -> None:
    if not os.path.exists(RECONCILED_CSV):
        return

    with open(RECONCILED_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    print(f"\n{'─'*70}")
    print(f"  Reconciled positions: {len(rows)}")

    if show:
        fmt = "  {:<12} {:<8} {:<7} {:<16} {:<10} {:>10} {:>10} {:>10}"
        print(fmt.format("strategy", "symbol", "side", "outcome",
                         "close_time"[:10], "est_pnl", "net_pnl", "diff"))
        print("  " + "─" * 68)
        for r in rows:
            est = _f(r.get("est_pnl"))
            net = _f(r.get("net_pnl"))
            print(fmt.format(
                r.get("strategy", ""), r.get("symbol", ""), r.get("side", ""),
                r.get("outcome", ""), r.get("close_time", "")[:10],
                f"{est:+.4f}", f"{net:+.4f}", f"{net-est:+.4f}",
            ))
        print()

    total_est = sum(_f(r.get("est_pnl")) for r in rows)
    total_net = sum(_f(r.get("net_pnl")) for r in rows)
    print(f"  Total estimated PnL : {total_est:+.4f}")
    print(f"  Total actual net PnL: {total_net:+.4f}")
    print(f"  Difference          : {total_net - total_est:+.4f}")
    print(f"{'─'*70}\n")


def _f(val) -> float:
    """Safe float parse, returns 0.0 on empty/None."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconcile PnL against Bitunix position history")
    parser.add_argument("--show", action="store_true", help="Print full per-trade table")
    args = parser.parse_args()
    reconcile(show=args.show)
