"""
follow_trader.py — Bitunix trend-following perpetual futures trader

Strategy:
  1. Compute a projection trend line from 2-hour candles over LOOKBACK periods:
       - Point 1: avg open of first 5 candles  (start of range)
       - Point 2: avg close of last 5 candles  (end of range)
       - Point 3: avg (high+low)/2 of middle 5 candles (midpoint)
     Line slope from P1→P2, intercept adjusted 30% toward P3 deviation.
     (Matches Pine Script: "Historical Price Analysis & Projection Line")

  2. Trend-following bias (differs from trend_trader.py):
       - Slope direction sets the bias: rising slope → LONG only, falling → SHORT only
       - Price must be on the favorable side of the line to enter:
           LONG  entries: price ≤ trend line  (pulled back into uptrend)
           SHORT entries: price ≥ trend line  (bounced up into downtrend)
     This trades mean-reversion TO the trend, not momentum AWAY from price-vs-line.

  3. Entry confirmation: Brownian motion Z-score on 1-min candles
     must exceed Z_ENTRY threshold in the bias direction.

  4. TP: TP_MULT × sigma × sqrt(HOLD_INTERVALS) from entry
     SL: SL_MULT × sigma × sqrt(HOLD_INTERVALS) from entry
     Fee gate: skip if expected profit ≤ round-trip fee cost
     Time exit: close if held > MAX_HOLD_MINS

Usage:
    python3 follow_trader.py              # live trading
    python3 follow_trader.py --debug      # DEBUG mode (no real orders)
"""

import argparse
import csv
import json
import logging
import math
import os
import time
from datetime import datetime, timezone

from auth import BitunixClient
from config import API_KEY, SECRET_KEY
from market import fetch_candles, fetch_ticker, compute_sigma
from log_cap import start_logging

start_logging("follow_trader")

# ── Configuration ─────────────────────────────────────────────────────────────

SYMBOLS        = ["BTCUSDT", "ETHUSDT", "RIVERUSDT"]
LEVERAGE       = 2
MARGIN_COIN    = "USDT"

# Trend line (2-hour candles)
TREND_INTERVAL = "2h"          # candle size for trend line
LOOKBACK       = 15            # candles to analyse
TREND_CANDLES  = 5             # candles at each anchor point

# Entry signal (1-min candles)
INTERVAL       = "1m"
SIGMA_CANDLES  = 20
SIGNAL_CANDLES = 5
Z_ENTRY        = 1.2

# Trade management
TP_MULT        = 1.5
SL_MULT        = 2.0
HOLD_INTERVALS = 15
MAX_HOLD_MINS  = 33
MAX_TRADE_PCT  = 0.20
POLL_SECS      = 30

# Fees
FEE_TAKER      = 0.00060
FEE_MAKER      = 0.00060
ROUND_TRIP_FEE = FEE_TAKER + FEE_MAKER

# Circuit breaker
MIN_BALANCE_PCT = 0.70

# Symbol precision
PRECISION = {
    "BTCUSDT":   {"qty": 4, "price": 1},
    "ETHUSDT":   {"qty": 3, "price": 2},
    "RIVERUSDT": {"qty": 2, "price": 3},
}

MIN_QTY = {"BTCUSDT": 0.0001, "ETHUSDT": 0.003, "RIVERUSDT": 0.5}

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Trade CSV ─────────────────────────────────────────────────────────────────

TRADE_CSV = os.path.join("log", "follow_trades.csv")

def log_trade_csv(body: dict) -> None:
    os.makedirs("log", exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    row = [
        ts, ts,
        "symbol",    body["symbol"],
        "qty",       body["qty"],
        "side",      body["side"],
        "orderType", body["orderType"],
        "tradeSide", body["tradeSide"],
        "tpPrice",   body["tpPrice"],
        "slPrice",   body["slPrice"],
        "tpStopType", body["tpStopType"],
        "slStopType", body["slStopType"],
    ]
    with open(TRADE_CSV, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ── Helpers ───────────────────────────────────────────────────────────────────

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def round_qty(symbol: str, qty: float) -> float:
    return round(qty, PRECISION.get(symbol, {}).get("qty", 4))

def round_price(symbol: str, price: float) -> float:
    return round(price, PRECISION.get(symbol, {}).get("price", 2))

def get_balance(client: BitunixClient) -> float:
    resp = client.get("/api/v1/futures/account", {"marginCoin": MARGIN_COIN})
    if resp.get("code") != 0:
        raise RuntimeError(f"Balance error: {resp.get('msg')}")
    data = resp["data"]
    return float(data.get("available", 0)) + float(data.get("crossUnrealizedPNL", 0))

def get_open_positions(client: BitunixClient, symbol: str = None) -> list[dict]:
    params = {"symbol": symbol} if symbol else {}
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
        "symbol": symbol, "leverage": LEVERAGE, "marginCoin": MARGIN_COIN,
    })
    if resp.get("code") != 0:
        log.warning(f"  Leverage set failed for {symbol}: {resp.get('msg')}")
    else:
        log.info(f"  {symbol} leverage set to {LEVERAGE}×")


# ── Trend line ────────────────────────────────────────────────────────────────

def compute_trend_line(client: BitunixClient, symbol: str) -> dict | None:
    """
    Fetch LOOKBACK 2-hour candles and compute the projection trend line
    matching the Pine Script methodology:
      - P1: avg open of first TREND_CANDLES candles
      - P2: avg close of last TREND_CANDLES candles
      - P3: avg (high+low)/2 of middle TREND_CANDLES candles
      - slope = (P2 - P1) / LOOKBACK
      - intercept averaged from P1 and P2 ends, then adjusted 30% toward P3

    Returns dict with slope, intercept_adjusted, and projected current value.
    """
    resp = client.get("/api/v1/futures/market/kline", {
        "symbol":   symbol,
        "interval": TREND_INTERVAL,
        "limit":    str(LOOKBACK + 2),
    })
    if resp.get("code") != 0:
        return None

    candles = sorted(resp.get("data", []), key=lambda c: int(c.get("time", 0)))
    if len(candles) < LOOKBACK:
        return None

    candles = candles[-LOOKBACK:]
    n       = len(candles)

    p1_price = sum(float(candles[i]["open"]) for i in range(TREND_CANDLES)) / TREND_CANDLES
    p1_idx   = TREND_CANDLES // 2

    p2_price = sum(float(candles[n - TREND_CANDLES + i]["close"]) for i in range(TREND_CANDLES)) / TREND_CANDLES
    p2_idx   = n - TREND_CANDLES // 2

    mid      = n // 2
    p3_price = sum(
        (float(candles[mid - 2 + i]["high"]) + float(candles[mid - 2 + i]["low"])) / 2
        for i in range(TREND_CANDLES)
    ) / TREND_CANDLES
    p3_idx   = mid

    slope = (p2_price - p1_price) / (p2_idx - p1_idx)

    intercept_p1 = p1_price - slope * p1_idx
    intercept_p2 = p2_price - slope * p2_idx
    intercept    = (intercept_p1 + intercept_p2) / 2

    predicted_p3  = slope * p3_idx + intercept
    middle_dev    = p3_price - predicted_p3
    intercept_adj = intercept + middle_dev * 0.3

    trend_now = slope * n + intercept_adj

    return {
        "p1_price":      p1_price,
        "p2_price":      p2_price,
        "p3_price":      p3_price,
        "slope":         slope,
        "intercept_adj": intercept_adj,
        "trend_now":     trend_now,
        "slope_pct":     (slope / p1_price) * 100 * n,
    }


# ── Entry signal ──────────────────────────────────────────────────────────────

def entry_signal(candles: list) -> tuple[str | None, float, float]:
    if len(candles) < SIGMA_CANDLES + SIGNAL_CANDLES:
        return None, 0.0, 0.0
    sigma = compute_sigma(candles[:-SIGNAL_CANDLES])
    if sigma == 0:
        return None, 0.0, 0.0
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
    log_trade_csv(body)
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
    if debug:
        log.info(f"  [DEBUG] Would flash-close {position_id}")
        return True
    resp = client.post("/api/v1/futures/trade/flash_close_position",
                       {"positionId": position_id})
    if resp.get("code") != 0:
        log.error(f"  Close failed: {resp.get('msg')}")
        return False
    log.info(f"  Position {position_id} closed")
    return True


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(debug: bool, symbols: list[str] = None) -> None:
    client = BitunixClient(API_KEY, SECRET_KEY)
    active_symbols = symbols or SYMBOLS

    log.info("━" * 62)
    log.info("  Bitunix Follow-the-Trend Trader")
    log.info(f"  Symbols    : {', '.join(active_symbols)}")
    log.info(f"  Leverage   : {LEVERAGE}×  |  Max trade : {MAX_TRADE_PCT:.0%}")
    log.info(f"  Trend line : {LOOKBACK}×{TREND_INTERVAL} lookback  |  Z-entry : {Z_ENTRY}")
    log.info(f"  TP/SL      : {TP_MULT}/{SL_MULT}σ  |  Hold : ≤{MAX_HOLD_MINS}min")
    log.info(f"  Fees       : taker {FEE_TAKER*100:.3f}%  maker {FEE_MAKER*100:.3f}%  "
             f"round-trip {ROUND_TRIP_FEE*100:.3f}%")
    log.info(f"  Circuit    : halt if balance < {MIN_BALANCE_PCT:.0%} of start")
    log.info(f"  Gate       : slope sets bias · price must be at/through line to enter")
    if debug:
        log.info("  MODE       : DEBUG — no real orders will be placed")
    log.info("━" * 62)

    for sym in active_symbols:
        set_leverage(client, sym, debug)

    start_balance = get_balance(client)
    min_balance   = start_balance * MIN_BALANCE_PCT
    log.info(f"  Starting balance: {start_balance:.4f} USDT  |  "
             f"Circuit floor: {min_balance:.4f} USDT")

    tracked: dict[str, dict] = {}

    # Hydrate from existing live positions
    if not debug:
        for p in get_open_positions(client):
            sym = p.get("symbol")
            if sym not in active_symbols:
                continue
            side = "LONG" if p.get("side", "").upper() == "BUY" else "SHORT"
            tracked[sym] = {
                "position_id": p.get("positionId"),
                "side":        side,
                "entry_price": float(p.get("avgOpenPrice", 0)),
                "qty":         float(p.get("qty", 0)),
                "opened_at":   now_utc(),
                "debug":       False,
            }
            log.info(f"  Hydrated {sym} [{side}]  "
                     f"pos={tracked[sym]['position_id']}  "
                     f"entry={tracked[sym]['entry_price']:.4f}  "
                     f"qty={tracked[sym]['qty']}")

    TREND_REFRESH_CYCLES = 20
    trend_cache: dict[str, dict] = {}
    cycle_count = 0

    while True:
        try:
            cycle_start = now_utc()
            balance = get_balance(client)
            log.info(f"  Balance: {balance:.4f} USDT  |  {cycle_start.strftime('%H:%M:%S')} UTC")

            # ── Circuit breaker ────────────────────────────────────────────────
            if balance < min_balance:
                log.warning(f"  CIRCUIT BREAKER: balance {balance:.4f} < "
                            f"floor {min_balance:.4f} — halting")
                break

            # ── Refresh trend lines periodically ──────────────────────────────
            if cycle_count % TREND_REFRESH_CYCLES == 0:
                for sym in active_symbols:
                    trend = compute_trend_line(client, sym)
                    if trend:
                        trend_cache[sym] = trend
                        direction = "↑ uptrend  bias=LONG" if trend["slope"] > 0 else "↓ downtrend  bias=SHORT"
                        log.info(f"  {sym} trend {direction}  "
                                 f"line={trend['trend_now']:.4f}  "
                                 f"slope={trend['slope_pct']:+.2f}%")

            cycle_count += 1

            # ── Monitor existing positions ─────────────────────────────────────
            for sym in list(tracked.keys()):
                pos_info  = tracked[sym]
                held_mins = (now_utc() - pos_info["opened_at"]).total_seconds() / 60

                if pos_info.get("debug"):
                    ticker_now  = fetch_ticker(client, sym)
                    close_price = float(ticker_now.get("lastPrice", 0))
                    pnl_est     = (close_price - pos_info["entry_price"]) * pos_info["qty"]
                    if pos_info["side"] == "SHORT":
                        pnl_est = -pnl_est
                    log.info(f"  {sym} [DEBUG {pos_info['side']}]  "
                             f"entry={pos_info['entry_price']:.4f}  "
                             f"now={close_price:.4f}  "
                             f"uPnL≈{pnl_est:+.4f}  "
                             f"held={held_mins:.1f}min")
                    if held_mins >= MAX_HOLD_MINS:
                        log.info(f"  {sym} [DEBUG] time exit after {held_mins:.1f}min  "
                                 f"entry={pos_info['entry_price']:.4f}  "
                                 f"close≈{close_price:.4f}  "
                                 f"PnL≈{pnl_est:+.4f}")
                        del tracked[sym]
                    continue

                live = [p for p in get_open_positions(client, sym)
                        if p.get("positionId") == pos_info["position_id"]]
                if not live:
                    log.info(f"  {sym} position closed (TP/SL hit)")
                    del tracked[sym]
                    continue

                p     = live[0]
                upnl  = float(p.get("unrealizedPNL", 0))
                log.info(f"  {sym} [{pos_info['side']}]  "
                         f"entry={pos_info['entry_price']:.4f}  "
                         f"uPnL={upnl:+.4f}  held={held_mins:.1f}min")

                if held_mins >= MAX_HOLD_MINS:
                    mark = float(p.get("markPrice", pos_info["entry_price"]))
                    log.info(f"  {sym} time exit after {held_mins:.1f}min  "
                             f"entry={pos_info['entry_price']:.4f}  "
                             f"close≈{mark:.4f}  uPnL={upnl:+.4f}")
                    if close_position(client, pos_info["position_id"], sym, debug):
                        del tracked[sym]

            # ── Scan for entries ───────────────────────────────────────────────
            for sym in active_symbols:
                if sym in tracked:
                    continue

                trend = trend_cache.get(sym)
                if not trend:
                    log.info(f"  {sym}  no trend data yet, skipping")
                    continue

                try:
                    candles = fetch_candles(client, sym, INTERVAL,
                                           SIGMA_CANDLES + SIGNAL_CANDLES + 1)
                    ticker  = fetch_ticker(client, sym)
                    price   = float(ticker.get("lastPrice", 0))
                    sigma   = compute_sigma(candles[:-SIGNAL_CANDLES])
                    side, z, sig = entry_signal(candles)

                    trend_val  = trend["trend_now"]
                    # Slope determines bias — follow the trend direction
                    bias       = "LONG" if trend["slope"] > 0 else "SHORT"
                    # Price must have pulled back to/through the line
                    price_ok   = (price <= trend_val) if bias == "LONG" else (price >= trend_val)
                    price_rel  = f"{'≤' if price <= trend_val else '>'} line" if bias == "LONG" \
                                 else f"{'≥' if price >= trend_val else '<'} line"

                    log.info(f"  {sym}  price={price:.4f}  "
                             f"trend={trend_val:.4f}  "
                             f"bias={bias}  price {price_rel}  "
                             f"sigma={sigma*100:.4f}%  z={z:+.3f}")

                    if side is None:
                        log.info(f"  {sym}  no signal (z={z:+.3f} < ±{Z_ENTRY})")
                        continue

                    # Gate 1 — Z-score must confirm bias direction
                    if side != bias:
                        log.info(f"  {sym}  BLOCKED — signal {side} conflicts "
                                 f"with trend bias {bias}")
                        continue

                    # Gate 2 — price must be at/through the line (favorable entry zone)
                    if not price_ok:
                        log.info(f"  {sym}  BLOCKED — price not at entry zone "
                                 f"(bias={bias}  need price {'≤' if bias == 'LONG' else '≥'} "
                                 f"{trend_val:.4f}  got {price:.4f})")
                        continue

                    # Position sizing
                    notional = balance * MAX_TRADE_PCT * LEVERAGE
                    qty      = round_qty(sym, notional / price)
                    min_qty  = MIN_QTY.get(sym, 0.001)
                    if qty < min_qty:
                        log.info(f"  {sym}  qty {qty} below minimum {min_qty}, skip")
                        continue

                    # TP / SL — fee gate
                    sigma_hold = sig * math.sqrt(HOLD_INTERVALS)
                    tp_move    = price * sigma_hold * TP_MULT
                    fee_cost   = price * ROUND_TRIP_FEE
                    if tp_move <= fee_cost:
                        log.info(f"  {sym}  SKIP fee gate — "
                                 f"profit {tp_move:.4f} ≤ fee {fee_cost:.4f}")
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
                            "debug":       False,
                        }
                        log.info(f"  {sym} position opened: "
                                 f"{tracked[sym]['position_id']}  "
                                 f"entry={tracked[sym]['entry_price']:.4f}")

                except Exception as e:
                    log.error(f"  {sym} scan error: {e}")

            # ── Flat balance refresh ───────────────────────────────────────────
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
    parser = argparse.ArgumentParser(
        description="Bitunix follow-the-trend Brownian motion trader")
    parser.add_argument("--debug", action="store_true",
                        help="Run in DEBUG mode — no real orders placed")
    parser.add_argument("--symbol", nargs="+", metavar="SYMBOL",
                        help=f"Symbol(s) to trade (default: {', '.join(SYMBOLS)})")
    args = parser.parse_args()
    run(debug=args.debug, symbols=args.symbol)
