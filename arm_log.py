"""
arm_log.py — Shared event logger for all Bitunix traders.

Writes to two CSV files in log/:

  arm_events.csv  — one row per signal lifecycle event
    arm_id          short UUID linking to trade CSV and close_events
    strategy        bb | sr | vol_spike | exhaustion
    symbol          BTCUSDT | ETHUSDT
    arm_time        UTC ISO when signal was detected
    arm_price       close price of the signal candle
    direction       LONG | SHORT (empty if unknown at arm time)
    outcome         FIRED | NO_FIRE | PENDING (shadow events, resolved by dashboard)
    disarm_time     UTC ISO when resolved
    disarm_price    price at resolution (entry price if FIRED)
    no_fire_reason  pipe-delimited: ATR_FILTER|COOLDOWN|FEE_GATE|MIN_QTY|
                    ORDER_FAILED|PRICE_MOVED_AWAY  (empty if FIRED/PENDING)
    shadow          1 = out-of-window signal logged for observability; 0 = real
    atr             ATR (or equivalent volatility measure) at signal time
    would_be_tp     estimated TP price if trade had fired (shadow events)
    would_be_sl     estimated SL price if trade had fired (shadow events)

  close_events.csv — one row per trade close
    arm_id          links to arm_events.csv and trade CSV
    strategy
    symbol
    side            LONG | SHORT
    outcome         TP | SL | TIME | EXCHANGE_CLOSED
    close_time      UTC ISO
    exit_price      price at close (estimated for exchange-closed positions)
    hold_mins       minutes held
    realized_pnl    net PnL after round-trip fee estimate
"""

import csv
import os
import uuid
from datetime import datetime, timezone

ARM_EVENTS_CSV   = os.path.join("log", "arm_events.csv")
CLOSE_EVENTS_CSV = os.path.join("log", "close_events.csv")

_ARM_HEADER = [
    "arm_id", "strategy", "symbol",
    "arm_time", "arm_price", "direction",
    "outcome", "disarm_time", "disarm_price", "no_fire_reason",
    "shadow", "atr", "would_be_tp", "would_be_sl",
]

_CLOSE_HEADER = [
    "arm_id", "strategy", "symbol", "side",
    "outcome", "close_time", "exit_price", "hold_mins", "realized_pnl",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def new_arm_id() -> str:
    """Generate a short unique ID for one arm event."""
    return uuid.uuid4().hex[:12]


def _ensure_file(path: str, header: list) -> None:
    os.makedirs("log", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


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
    shadow: bool = False,
    atr: float | None = None,
    would_be_tp: float | None = None,
    would_be_sl: float | None = None,
) -> None:
    """Append one row to arm_events.csv."""
    _ensure_file(ARM_EVENTS_CSV, _ARM_HEADER)
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
            "1" if shadow else "0",
            f"{atr:.6f}" if atr is not None else "",
            f"{would_be_tp:.4f}" if would_be_tp is not None else "",
            f"{would_be_sl:.4f}" if would_be_sl is not None else "",
        ])


def log_close_event(
    arm_id: str,
    strategy: str,
    symbol: str,
    side: str,
    outcome: str,
    exit_price: float | None,
    hold_mins: float,
    realized_pnl: float | None,
) -> None:
    """Append one row to close_events.csv."""
    _ensure_file(CLOSE_EVENTS_CSV, _CLOSE_HEADER)
    with open(CLOSE_EVENTS_CSV, "a", newline="") as f:
        csv.writer(f).writerow([
            arm_id,
            strategy,
            symbol,
            side,
            outcome,
            _now_iso(),
            f"{exit_price:.6f}" if exit_price is not None else "",
            f"{hold_mins:.1f}",
            f"{realized_pnl:.6f}" if realized_pnl is not None else "",
        ])


def calc_pnl(
    side: str,
    entry_price: float,
    exit_price: float,
    qty: float,
    round_trip_fee_rate: float,
) -> float:
    """Estimate realized PnL after round-trip fees (entry fee approximated at entry price)."""
    gross = (exit_price - entry_price) * qty if side == "LONG" else (entry_price - exit_price) * qty
    fee   = entry_price * qty * round_trip_fee_rate
    return gross - fee
