"""
arm_log.py — Shared arm-event logger for all Bitunix traders.

Writes one row per signal lifecycle event to log/arm_events.csv,
capturing the full arm → fire/no-fire funnel for dashboard analysis.

Columns:
  arm_id          — short UUID linking arm event to trade CSV row
  strategy        — bb | sr | vol_spike | exhaustion
  symbol          — BTCUSDT | ETHUSDT
  arm_time        — UTC ISO when signal/arm was detected
  arm_price       — close price of the signal candle at arm time
  direction       — LONG | SHORT (empty if unknown at arm time)
  outcome         — FIRED | NO_FIRE
  disarm_time     — UTC ISO when resolved
  disarm_price    — price at resolution (entry price if FIRED)
  no_fire_reason  — pipe-delimited: ATR_FILTER|COOLDOWN|FEE_GATE|MIN_QTY|
                    ORDER_FAILED|PRICE_MOVED_AWAY (empty if FIRED)
"""

import csv
import os
import uuid
from datetime import datetime, timezone

ARM_EVENTS_CSV = os.path.join("log", "arm_events.csv")

_HEADER = [
    "arm_id", "strategy", "symbol",
    "arm_time", "arm_price", "direction",
    "outcome", "disarm_time", "disarm_price", "no_fire_reason",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def new_arm_id() -> str:
    """Generate a short unique ID for one arm event."""
    return uuid.uuid4().hex[:12]


def _ensure_header() -> None:
    os.makedirs("log", exist_ok=True)
    if not os.path.exists(ARM_EVENTS_CSV):
        with open(ARM_EVENTS_CSV, "w", newline="") as f:
            csv.writer(f).writerow(_HEADER)


def log_arm_event(
    arm_id: str,
    strategy: str,
    symbol: str,
    arm_time: str,
    arm_price: float,
    direction: str | None,
    outcome: str,
    disarm_time: str | None = None,
    disarm_price: float | None = None,
    no_fire_reason: str | None = None,
) -> None:
    """Append one row to arm_events.csv."""
    _ensure_header()
    with open(ARM_EVENTS_CSV, "a", newline="") as f:
        csv.writer(f).writerow([
            arm_id,
            strategy,
            symbol,
            arm_time,
            f"{arm_price:.6f}",
            direction or "",
            outcome,
            disarm_time or _now_iso(),
            f"{disarm_price:.6f}" if disarm_price is not None else "",
            no_fire_reason or "",
        ])
