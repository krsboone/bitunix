#!/usr/bin/env python3
"""
resolve_shadows.py — Resolve PENDING shadow arm events against local candle data.

Reads:   log/arm_events.csv          (shadow rows with outcome=PENDING)
         data/BTCUSDT_1m.csv
         data/ETHUSDT_1m.csv
Writes:  log/arm_events.csv          (in-place update of resolved rows)

For each PENDING shadow event, scans 1m candles from the candle immediately
after arm_time up to MAX_HOLD_MINS later and determines:
  LONG:  TP if candle.high >= would_be_tp
         SL if candle.low  <= would_be_sl
  SHORT: TP if candle.low  <= would_be_tp
         SL if candle.high >= would_be_sl
  If both triggered in the same candle, the level closer to arm_price wins.
  If neither within MAX_HOLD_MINS: TIME

Usage:
    python3 resolve_shadows.py           # resolve all PENDING shadow rows
    python3 resolve_shadows.py --dry-run # show what would be resolved, no writes
    python3 resolve_shadows.py --report  # print summary after resolving
"""

import argparse
import bisect
import csv
import os
from datetime import datetime, timezone

ARM_CSV      = os.path.join("log", "arm_events.csv")
DATA_DIR     = "data"
MAX_HOLD_MINS = 33   # all strategies use 33 min hold

CANDLE_FILES = {
    "BTCUSDT": os.path.join(DATA_DIR, "BTCUSDT_1m.csv"),
    "ETHUSDT": os.path.join(DATA_DIR, "ETHUSDT_1m.csv"),
}

_ARM_HEADER = [
    "arm_id", "strategy", "symbol",
    "arm_time", "arm_price", "direction",
    "outcome", "disarm_time", "disarm_price", "no_fire_reason",
    "shadow", "atr", "would_be_tp", "would_be_sl",
]


def _to_utc_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def _dt_to_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ── Candle store ──────────────────────────────────────────────────────────────

class CandleStore:
    """
    In-memory sorted candle store for one symbol.
    Candles are stored as dicts; lookup by timestamp_ms using bisect.
    """
    def __init__(self, path: str):
        self.timestamps: list[int] = []
        self.candles:    list[dict] = []
        self._load(path)

    def _load(self, path: str) -> None:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.timestamps.append(int(row["timestamp_ms"]))
                self.candles.append({
                    "ts":     int(row["timestamp_ms"]),
                    "open":   float(row["open"]),
                    "high":   float(row["high"]),
                    "low":    float(row["low"]),
                    "close":  float(row["close"]),
                })
        print(f"  Loaded {len(self.candles):,} candles from {path}")

    def candles_in_window(self, start_ms: int, end_ms: int) -> list[dict]:
        """Return candles with timestamp_ms in [start_ms, end_ms]."""
        lo = bisect.bisect_left(self.timestamps, start_ms)
        hi = bisect.bisect_right(self.timestamps, end_ms)
        return self.candles[lo:hi]


# ── Resolution logic ──────────────────────────────────────────────────────────

def resolve_shadow(row: dict, store: CandleStore) -> dict | None:
    """
    Attempt to resolve a PENDING shadow row.
    Returns updated row dict, or None if candle data is unavailable.
    """
    arm_time  = _to_utc_dt(row["arm_time"])
    arm_ms    = int(arm_time.timestamp() * 1000)
    end_ms    = arm_ms + MAX_HOLD_MINS * 60 * 1000
    direction = row["direction"]

    try:
        wtp = float(row["would_be_tp"])
        wsl = float(row["would_be_sl"])
        arm_price = float(row["arm_price"])
    except (ValueError, KeyError):
        return None

    # Start scanning from the candle *after* the signal candle
    start_ms = arm_ms + 60_000
    candles  = store.candles_in_window(start_ms, end_ms)

    if not candles:
        return None   # no candle data for this window — skip

    outcome      = None
    disarm_ms    = None
    disarm_price = None

    for c in candles:
        if direction == "LONG":
            tp_hit = c["high"] >= wtp
            sl_hit = c["low"]  <= wsl
        else:  # SHORT
            tp_hit = c["low"]  <= wtp
            sl_hit = c["high"] >= wsl

        if tp_hit and sl_hit:
            # Both in same candle — whichever level is closer to arm_price wins
            if abs(wtp - arm_price) <= abs(wsl - arm_price):
                outcome, disarm_price = "TP", wtp
            else:
                outcome, disarm_price = "SL", wsl
            disarm_ms = c["ts"]
            break
        elif tp_hit:
            outcome, disarm_price, disarm_ms = "TP", wtp, c["ts"]
            break
        elif sl_hit:
            outcome, disarm_price, disarm_ms = "SL", wsl, c["ts"]
            break

    if outcome is None:
        # Only call TIME if the full hold window has elapsed.
        # If the last candle is more than 2 minutes short of end_ms,
        # the data isn't complete yet — leave as PENDING.
        last = candles[-1]
        if last["ts"] < end_ms - 2 * 60_000:
            return None   # window not fully elapsed — skip, stays PENDING
        outcome      = "TIME"
        disarm_price = last["close"]
        disarm_ms    = last["ts"]

    disarm_dt = datetime.fromtimestamp(disarm_ms / 1000, tz=timezone.utc)

    updated = dict(row)
    updated["outcome"]      = outcome
    updated["disarm_time"]  = _dt_to_str(disarm_dt)
    updated["disarm_price"] = f"{disarm_price:.6f}"
    return updated


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve PENDING shadow arm events")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show resolutions without writing to CSV")
    parser.add_argument("--report", action="store_true",
                        help="Print outcome summary after resolving")
    args = parser.parse_args()

    # Load arm events
    with open(ARM_CSV, newline="") as f:
        all_rows = list(csv.DictReader(f))

    pending = [r for r in all_rows
               if r["shadow"] == "1" and r["outcome"] == "PENDING"]
    print(f"\n  {len(all_rows)} total arm events  |  {len(pending)} PENDING shadows to resolve\n")

    if not pending:
        print("  Nothing to resolve.")
        return

    # Load candle stores (once per symbol needed)
    needed_symbols = {r["symbol"] for r in pending}
    stores: dict[str, CandleStore] = {}
    for sym in needed_symbols:
        path = CANDLE_FILES.get(sym)
        if not path or not os.path.exists(path):
            print(f"  WARNING: no candle file for {sym} — those rows will be skipped")
            continue
        stores[sym] = CandleStore(path)
    print()

    # Resolve
    resolved_ids: dict[str, dict] = {}
    skipped = 0
    for row in pending:
        store = stores.get(row["symbol"])
        if store is None:
            skipped += 1
            continue
        updated = resolve_shadow(row, store)
        if updated is None:
            skipped += 1
            print(f"  SKIP  {row['arm_id']}  {row['symbol']}  {row['arm_time']} "
                  f"— no candle data in window")
            continue
        resolved_ids[row["arm_id"]] = updated
        if args.dry_run:
            print(f"  {updated['outcome']:<4}  {row['arm_id']}  {row['strategy']:<11} "
                  f"{row['symbol']}  {row['direction']:<5}  "
                  f"arm={float(row['arm_price']):,.2f}  "
                  f"tp={float(row['would_be_tp']):,.2f}  "
                  f"sl={float(row['would_be_sl']):,.2f}  "
                  f"→ disarm={updated['disarm_time']}  price={float(updated['disarm_price']):,.4f}")

    print(f"\n  Resolved: {len(resolved_ids)}  |  Skipped: {skipped}")

    if args.dry_run:
        print("  [dry-run] No changes written.")
        return

    # Write updated CSV in place
    updated_rows = []
    for row in all_rows:
        if row["arm_id"] in resolved_ids:
            updated_rows.append(resolved_ids[row["arm_id"]])
        else:
            updated_rows.append(row)

    with open(ARM_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_ARM_HEADER)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"  ✓ arm_events.csv updated")

    # Summary report
    if args.report or True:  # always show summary
        from collections import Counter
        outcomes = Counter(r["outcome"] for r in resolved_ids.values())
        strat_outcomes: dict[str, Counter] = {}
        for r in resolved_ids.values():
            s = r["strategy"]
            if s not in strat_outcomes:
                strat_outcomes[s] = Counter()
            strat_outcomes[s][r["outcome"]] += 1

        total = len(resolved_ids)
        print(f"\n{'─' * 55}")
        print(f"  Shadow Resolution Summary")
        print(f"{'─' * 55}")
        print(f"  {'Outcome':<8}  {'Count':>6}  {'Rate':>7}")
        print(f"{'─' * 55}")
        for outcome in ("TP", "SL", "TIME"):
            n = outcomes.get(outcome, 0)
            print(f"  {outcome:<8}  {n:>6}  {n/total*100:>6.1f}%")
        print(f"{'─' * 55}")
        print(f"\n  By strategy:")
        for strat, ctr in sorted(strat_outcomes.items()):
            n = sum(ctr.values())
            tp = ctr.get("TP", 0)
            sl = ctr.get("SL", 0)
            tx = ctr.get("TIME", 0)
            print(f"    {strat:<12}  n={n:>3}  "
                  f"TP={tp} ({tp/n*100:.0f}%)  "
                  f"SL={sl} ({sl/n*100:.0f}%)  "
                  f"TIME={tx} ({tx/n*100:.0f}%)")
        print()


if __name__ == "__main__":
    main()
