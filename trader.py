"""
trader.py — Bitunix perpetual futures trader

Strategy:
  - Monitor BTCUSDT and ETHUSDT on 1-minute candles
  - Entry signal: Brownian motion Z-score — enter when cumulative drift
    over recent candles exceeds Z_ENTRY standard deviations
  - TP: 1.5 × sigma × sqrt(hold_intervals) from entry
  - SL: 2.0 × sigma × sqrt(hold_intervals) from entry
  - Time-based exit: close position if open longer than MAX_HOLD_MINUTES
  - Leverage: 2× (set on startup)
  - Max position size: MAX_TRADE_PCT of available balance per trade
  - One position per symbol at a time

Usage:
    python3 trader.py              # live trading
    python3 trader.py --debug      # DEBUG mode (no real orders)
"""

import argparse
import json
import logging
import math
import time
from datetime import datetime, timezone

from auth import BitunixClient
from config import API_KEY, SECRET_KEY
from market import fetch_candles, fetch_ticker, compute_sigma
from log_cap import start_logging

start_logging("trader")

# ── Configuration ─────────────────────────────────────────────────────────────

SYMBOLS        = ["BTCUSDT", "ETHUSDT"]
LEVERAGE       = 2
MARGIN_COIN    = "USDT"
INTERVAL       = "1m"          # candle interval
SIGMA_CANDLES  = 20            # candles used to compute sigma
SIGNAL_CANDLES = 5             # recent candles for directional drift signal
Z_ENTRY        = 1.2           # Z-score threshold to trigger entry
TP_MULT        = 1.5           # TP distance = TP_MULT × sigma × sqrt(hold)
SL_MULT        = 2.0           # SL distance = SL_MULT × sigma × sqrt(hold)
HOLD_INTERVALS = 15            # expected hold in candle-lengths (15 min at 1m)
MAX_HOLD_MINS  = 30            # time-based exit after this many minutes
MAX_TRADE_PCT  = 0.20          # max 20% of available balance per trade
POLL_SECS      = 30            # seconds between scan cycles

# ── Fee configuration (update as your tier improves) ──────────────────────────
FEE_TAKER      = 0.00060       # market order entry fee (0.060%)
FEE_MAKER      = 0.00020       # limit order exit fee  (0.020%)
ROUND_TRIP_FEE = FEE_TAKER + FEE_MAKER   # 0.080% total

# ── Circuit breaker ───────────────────────────────────────────────────────────
MIN_BALANCE_PCT = 0.70         # halt trading if balance drops below 70% of start

# Symbol precision (from trading_pairs endpoint)
PRECISION = {
    "BTCUSDT": {"qty": 4, "price": 1},
    "ETHUSDT": {"qty": 3, "price": 2},
}

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def round_qty(symbol: str, qty: float) -> float:
    prec = PRECISION.get(symbol, {}).get("qty", 4)
    return round(qty, prec)


def round_price(symbol: str, price: float) -> float:
    prec = PRECISION.get(symbol, {}).get("price", 2)
    return round(price, prec)


def get_balance(client: BitunixClient) -> float:
    resp = client.get("/api/v1/futures/account", {"marginCoin": MARGIN_COIN})
    if resp.get("code") != 0:
        raise RuntimeError(f"Balance error: {resp.get('msg')}")
    data = resp["data"]
    return float(data.get("available", 0)) + float(data.get("crossUnrealizedPNL", 0))


def get_open_positions(client: BitunixClient, symbol: str = None) -> list[dict]:
    params = {}
    if symbol:
        params["symbol"] = symbol
    resp = client.get("/api/v1/futures/position/get_pending_positions", params)
    if resp.get("code") != 0:
        raise RuntimeError(f"Positions error: {resp.get('msg')}")
    data = resp.get("data", [])
    return data if isinstance(data, list) else []


def set_leverage(client: BitunixClient, symbol: str, debug: bool) -> None:
    if debug:
        log.info(f"  [DEBUG] Would set {symbol} leverage → {LEVERAGE}×")
        return
    resp = client.post("/api/v1/futures/account/change_leverage", {
        "symbol":     symbol,
        "leverage":   LEVERAGE,
        "marginCoin": MARGIN_COIN,
    })
    if resp.get("code") != 0:
        log.warning(f"  Leverage set failed for {symbol}: {resp.get('msg')}")
    else:
        log.info(f"  {symbol} leverage set to {LEVERAGE}×")


# ── Entry signal ──────────────────────────────────────────────────────────────

def entry_signal(candles: list) -> tuple[str | None, float, float]:
    """
    Compute Brownian motion directional signal.

    Uses the most recent SIGNAL_CANDLES log returns normalised by sigma
    over the full SIGMA_CANDLES window to produce a Z-score.

    Returns:
        (side, z_score, sigma)
        side = "LONG" | "SHORT" | None (no signal)
    """
    if len(candles) < SIGMA_CANDLES + SIGNAL_CANDLES:
        return None, 0.0, 0.0

    sigma = compute_sigma(candles[:-SIGNAL_CANDLES])
    if sigma == 0:
        return None, 0.0, 0.0

    # Cumulative drift over signal window
    recent = candles[-SIGNAL_CANDLES:]
    prices = [float(c["close"]) for c in recent]
    drift  = math.log(prices[-1] / prices[0])
    z      = drift / (sigma * math.sqrt(SIGNAL_CANDLES))

    if z >= Z_ENTRY:
        return "LONG", z, sigma
    elif z <= -Z_ENTRY:
        return "SHORT", z, sigma
    return None, z, sigma


# ── Order placement ───────────────────────────────────────────────────────────

def place_order(client: BitunixClient, symbol: str, side: str,
                qty: float, tp_price: float, sl_price: float,
                debug: bool) -> str | None:
    """
    Place a market order (hedge mode: tradeSide=OPEN).
    Inline TP/SL on MARK_PRICE.
    Returns orderId or None.
    """
    buy_sell = "BUY" if side == "LONG" else "SELL"
    body = {
        "symbol":      symbol,
        "qty":         str(qty),
        "side":        buy_sell,
        "orderType":   "MARKET",
        "tradeSide":   "OPEN",
        "tpPrice":     str(round_price(symbol, tp_price)),
        "slPrice":     str(round_price(symbol, sl_price)),
        "tpStopType":  "MARK_PRICE",
        "slStopType":  "MARK_PRICE",
    }

    if debug:
        log.info(f"  [DEBUG] Would place order: {json.dumps(body)}")
        return "DEBUG-ORDER-ID"

    resp = client.post("/api/v1/futures/trade/place_order", body)
    if resp.get("code") != 0:
        log.error(f"  Order failed: {resp.get('msg')} (code {resp.get('code')})")
        return None
    order_id = resp.get("data", {}).get("orderId")
    log.info(f"  Order placed: {order_id}")
    return order_id


def close_position(client: BitunixClient, position_id: str,
                   symbol: str, debug: bool) -> bool:
    """Flash close a position at market."""
    if debug:
        log.info(f"  [DEBUG] Would flash-close position {position_id}")
        return True
    resp = client.post("/api/v1/futures/trade/flash_close_position",
                       {"positionId": position_id})
    if resp.get("code") != 0:
        log.error(f"  Close failed: {resp.get('msg')}")
        return False
    log.info(f"  Position {position_id} closed")
    return True


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(debug: bool) -> None:
    client = BitunixClient(API_KEY, SECRET_KEY)

    log.info("━" * 60)
    log.info("  Bitunix Perpetual Futures Trader")
    log.info(f"  Symbols   : {', '.join(SYMBOLS)}")
    log.info(f"  Leverage  : {LEVERAGE}×  |  Max trade : {MAX_TRADE_PCT:.0%}")
    log.info(f"  Z-entry   : {Z_ENTRY}  |  TP/SL mult: {TP_MULT}/{SL_MULT}σ")
    log.info(f"  Hold      : ≤{MAX_HOLD_MINS}min  |  Candles  : {SIGMA_CANDLES}×{INTERVAL}")
    log.info(f"  Fees      : taker {FEE_TAKER*100:.3f}%  maker {FEE_MAKER*100:.3f}%  "
             f"round-trip {ROUND_TRIP_FEE*100:.3f}%")
    log.info(f"  Circuit   : halt if balance < {MIN_BALANCE_PCT:.0%} of start")
    if debug:
        log.info("  MODE      : DEBUG — no real orders will be placed")
    log.info("━" * 60)

    # Set leverage on startup
    for sym in SYMBOLS:
        set_leverage(client, sym, debug)

    # Record starting balance for circuit breaker
    start_balance = get_balance(client)
    min_balance   = start_balance * MIN_BALANCE_PCT
    log.info(f"  Starting balance: {start_balance:.4f} USDT  |  "
             f"Circuit floor: {min_balance:.4f} USDT")

    # Track open positions: symbol → {position_id, side, entry_price, opened_at, qty, debug}
    tracked: dict[str, dict] = {}

    # ── Hydrate from any positions already open on the exchange ───────────────
    if not debug:
        existing = get_open_positions(client)
        for p in existing:
            sym = p.get("symbol")
            if sym not in SYMBOLS:
                continue
            side = "LONG" if p.get("side", "").upper() == "LONG" else "SHORT"
            tracked[sym] = {
                "position_id": p.get("positionId"),
                "side":        side,
                "entry_price": float(p.get("avgOpenPrice", 0)),
                "qty":         float(p.get("qty", 0)),
                "opened_at":   now_utc(),   # exact open time unknown; use now conservatively
                "debug":       False,
            }
            log.info(f"  Hydrated {sym} [{side}]  "
                     f"pos={p.get('positionId')}  "
                     f"entry={tracked[sym]['entry_price']:.4f}  "
                     f"qty={tracked[sym]['qty']}")

    while True:
        try:
            cycle_start = now_utc()
            balance = get_balance(client)
            log.info(f"  Balance: {balance:.4f} USDT  |  {cycle_start.strftime('%H:%M:%S')} UTC")

            # ── Circuit breaker ────────────────────────────────────────────────
            if balance < min_balance:
                log.warning(f"  CIRCUIT BREAKER: balance {balance:.4f} < floor "
                            f"{min_balance:.4f} — halting")
                break

            # ── Monitor existing positions ─────────────────────────────────────
            for sym in list(tracked.keys()):
                pos_info  = tracked[sym]
                held_mins = (now_utc() - pos_info["opened_at"]).total_seconds() / 60

                if pos_info.get("debug"):
                    # Debug position — no real API state, track by time only
                    log.info(f"  {sym} [DEBUG {pos_info['side']}]  "
                             f"qty={pos_info['qty']}  "
                             f"entry={pos_info['entry_price']:.4f}  "
                             f"held={held_mins:.1f}min")
                    if held_mins >= MAX_HOLD_MINS:
                        log.info(f"  {sym} [DEBUG] time exit after {held_mins:.1f}min")
                        del tracked[sym]
                    continue

                # Live position — refresh from API
                live = [p for p in get_open_positions(client, sym)
                        if p.get("positionId") == pos_info["position_id"]]

                if not live:
                    log.info(f"  {sym} position {pos_info['position_id']} closed "
                             f"(TP/SL hit or filled)")
                    del tracked[sym]
                    continue

                p = live[0]
                upnl = float(p.get("unrealizedPNL", 0))
                log.info(f"  {sym} [{pos_info['side']}]  "
                         f"qty={pos_info['qty']}  "
                         f"entry={pos_info['entry_price']:.4f}  "
                         f"uPnL={upnl:+.4f}  "
                         f"held={held_mins:.1f}min")

                # Time-based exit
                if held_mins >= MAX_HOLD_MINS:
                    log.info(f"  {sym} time exit after {held_mins:.1f}min")
                    if close_position(client, pos_info["position_id"], sym, debug):
                        del tracked[sym]

            # ── Scan for entries ───────────────────────────────────────────────
            for sym in SYMBOLS:
                if sym in tracked:
                    continue  # already in a position

                try:
                    candles = fetch_candles(client, sym, INTERVAL,
                                           SIGMA_CANDLES + SIGNAL_CANDLES + 1)
                    ticker  = fetch_ticker(client, sym)
                    price   = float(ticker.get("lastPrice", 0))
                    sigma   = compute_sigma(candles[:-SIGNAL_CANDLES])
                    side, z, sig = entry_signal(candles)

                    log.info(f"  {sym}  price={price:.4f}  "
                             f"sigma={sigma*100:.4f}%  z={z:+.3f}")

                    if side is None:
                        log.info(f"  {sym}  no signal (z={z:+.3f} < threshold ±{Z_ENTRY})")
                        continue

                    # Position sizing
                    notional = balance * MAX_TRADE_PCT * LEVERAGE
                    qty = round_qty(sym, notional / price)
                    min_qty = {"BTCUSDT": 0.0001, "ETHUSDT": 0.003}.get(sym, 0.001)
                    if qty < min_qty:
                        log.info(f"  {sym}  qty {qty} below minimum {min_qty}, skip")
                        continue

                    # TP / SL prices — set purely from signal, no fee inflation
                    sigma_hold   = sig * math.sqrt(HOLD_INTERVALS)
                    tp_move      = price * sigma_hold * TP_MULT
                    fee_cost     = price * ROUND_TRIP_FEE

                    # Fee gate: skip if expected profit doesn't cover round-trip cost
                    if tp_move <= fee_cost:
                        log.info(f"  {sym}  SKIP — expected profit {tp_move:.4f} "
                                 f"≤ fee cost {fee_cost:.4f}")
                        continue

                    if side == "LONG":
                        tp_price = price + tp_move
                        sl_price = price - price * sigma_hold * SL_MULT
                    else:
                        tp_price = price - tp_move
                        sl_price = price + price * sigma_hold * SL_MULT

                    log.info(f"  {sym}  SIGNAL {side}  z={z:+.3f}  "
                             f"qty={qty}  TP={tp_price:.4f}  SL={sl_price:.4f}  "
                             f"expected={tp_move:.4f}  fee={fee_cost:.4f}")

                    order_id = place_order(client, sym, side, qty,
                                           tp_price, sl_price, debug)
                    if not order_id:
                        continue

                    # In debug mode, record a simulated position
                    if debug:
                        tracked[sym] = {
                            "position_id": "DEBUG-POS-" + sym,
                            "side":        side,
                            "entry_price": price,
                            "qty":         qty,
                            "opened_at":   now_utc(),
                            "debug":       True,
                        }
                        continue

                    # Wait briefly for fill then fetch real position
                    time.sleep(2)
                    positions = get_open_positions(client, sym)
                    if positions:
                        p = positions[0]
                        tracked[sym] = {
                            "position_id": p.get("positionId"),
                            "side":        side,
                            "entry_price": float(p.get("avgOpenPrice", price)),
                            "qty":         float(p.get("qty", qty)),
                            "opened_at":   now_utc(),
                        }
                        log.info(f"  {sym} position opened: "
                                 f"{tracked[sym]['position_id']}  "
                                 f"entry={tracked[sym]['entry_price']:.4f}")

                except Exception as e:
                    log.error(f"  {sym} scan error: {e}")

            # ── Refresh reference balance when flat ───────────────────────────
            if not tracked:
                new_balance = get_balance(client)
                if new_balance != start_balance:
                    start_balance = new_balance
                    min_balance   = start_balance * MIN_BALANCE_PCT
                    log.info(f"  Flat — reference balance updated: "
                             f"{start_balance:.4f} USDT  |  "
                             f"Circuit floor: {min_balance:.4f} USDT")

        except Exception as e:
            log.error(f"  Cycle error: {e}")

        time.sleep(POLL_SECS)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bitunix perpetual futures trader")
    parser.add_argument("--debug", action="store_true",
                        help="Run in DEBUG mode — no real orders placed")
    args = parser.parse_args()
    run(debug=args.debug)
