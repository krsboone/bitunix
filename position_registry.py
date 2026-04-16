"""
position_registry.py — Shared position ownership registry.

Tracks which strategy opened each active position so scripts can safely
skip positions they didn't open (manual trades, other scripts' positions).

Registry file: log/open_positions.csv

Usage:
    On open:      position_registry.register(position_id, strategy, ...)
    On hydration: if not position_registry.owns(position_id, strategy): skip
    On close:     position_registry.release(position_id)
"""

import csv
import os
from datetime import datetime, timezone

REGISTRY_CSV = os.path.join("log", "open_positions.csv")

_HEADER = ["position_id", "strategy", "symbol", "side",
           "entry_price", "open_time", "arm_id"]


def _load() -> dict[str, dict]:
    """Return {position_id: row} for all registered positions."""
    if not os.path.exists(REGISTRY_CSV):
        return {}
    with open(REGISTRY_CSV, newline="") as f:
        return {
            row["position_id"]: row
            for row in csv.DictReader(f)
            if row.get("position_id")
        }


def _save(registry: dict[str, dict]) -> None:
    os.makedirs("log", exist_ok=True)
    with open(REGISTRY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_HEADER)
        w.writeheader()
        w.writerows(registry.values())


def register(position_id: str, strategy: str, symbol: str, side: str,
             entry_price: float, arm_id: str = "") -> None:
    """Record that this strategy owns this position."""
    if not position_id or str(position_id).startswith("DEBUG"):
        return
    registry = _load()
    registry[position_id] = {
        "position_id": position_id,
        "strategy":    strategy,
        "symbol":      symbol,
        "side":        side,
        "entry_price": f"{entry_price:.6f}",
        "open_time":   datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "arm_id":      arm_id or "",
    }
    _save(registry)


def owns(position_id: str, strategy: str) -> bool:
    """Return True only if this strategy owns this position.

    - Registered to this strategy  → True  (claim it)
    - Registered to other strategy → False (another script owns it)
    - Not in registry              → False (manual trade or pre-registry position)
    """
    if not position_id:
        return False
    registry = _load()
    if position_id not in registry:
        return False
    return registry[position_id]["strategy"] == strategy


def release(position_id: str) -> None:
    """Remove a closed position from the registry."""
    if not position_id or str(position_id).startswith("DEBUG"):
        return
    registry = _load()
    if position_id in registry:
        del registry[position_id]
        _save(registry)
