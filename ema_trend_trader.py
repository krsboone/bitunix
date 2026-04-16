"""
ema_trend_trader.py — Bitunix EMA crossover trend-following trader

Strategy: EMA crossover on 15m candles
  - Resample live 1m candles to 15m internally
  - When fast EMA crosses above slow EMA on completed 15m candle → LONG
  - When fast EMA crosses below slow EMA on completed 15m candle → SHORT
  - SL:   SL_MULT × ATR from entry — exchange-managed safety net
  - TP:   TP_MULT × ATR from entry — wide backstop (rarely hits via exchange)
  - Trend exit: close when EMA re-crosses (fast flips relative to slow on 15m close)
  - Time exit:  close if held > MAX_HOLD_MINS (6h default)

Runs 24/7 — no trading window restrictions. Trends develop at any hour;
the longer hold window makes fee drag minor relative to expected move.

Default parameters (sweep candidates):
  Both symbols: fast_period=9  slow_period=21  atr_period=14
  Shared:       sl_mult=2.0    tp_mult=5.0     hold=360min

Usage:
    python3 ema_trend_trader.py              # live trading
    python3 ema_trend_trader.py --debug      # DEBUG mode — no real orders
    python3 ema_trend_trader.py --symbol BTCUSDT
"""

import argparse
import csv
import json
import logging
import os
import time
from datetime import datetime, timezone

from auth import BitunixClient
from config import API_KEY, SECRET_KEY
from market import fetch_ticker
from log_cap import start_logging
import arm_log

start_logging("ema_trend_trader")

STRATEGY = "ema_trend"

# ── Per-symbol config ──────────────────────────────────────────────────────────

SYMBOL_CONFIGS = {
    "BTCUSDT": {
        "fast_period": 9,
        "slow_period": 21,
    },
    "ETHUSDT": {
        "fast_period": 9,
        "slow_period": 21,
    },
}
SYMBOLS = list(SYMBOL_CONFIGS.keys())

# ── Shared strategy parameters ─────────────────────────────────────────────────

ATR_PERIOD      = 14
SL_MULT         = 2.0    # ATR multiples for exchange-managed safety stop
TP_MULT         = 5.0    # ATR multiples for wide backstop (EMA re-cross is primary exit)
MAX_HOLD_MINS   = 360    # 6-hour time exit backstop
BREAKEVEN_ATR   = 1.0    # profit must reach 1× ATR to activate breakeven stop (sweep candidate)
LOCK_ATR        = 0.25   # new SL locks in 0.25× ATR above entry (sweep candidate)

FIFTEEN_MIN_MS  = 15 * 60 * 1000

# ── Trading parameters ─────────────────────────────────────────────────────────

LEVERAGE        = 2
MARGIN_COIN     = "USDT"
INTERVAL        = "1m"
MAX_TRADE_PCT   = 0.10
POLL_SECS       = 60

FEE_TAKER       = 0.00060
FEE_MAKER       = 0.00060
ROUND_TRIP_FEE  = FEE_TAKER + FEE_MAKER
MIN_BALANCE_PCT = 0.70

PRECISION = {
    "BTCUSDT": {"qty": 4, "price": 1},
    "ETHUSDT": {"qty": 3, "price": 2},
}

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

TRADE_CSV = os.path.join("log", "ema_trend_trades.csv")


# ── Candle helpers ─────────────────────────────────────────────────────────────

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def normalize(c: dict) -> dict:
    return {
        "time":   int(c["time"]),
        "open":   float(c["open"]),
        "high":   float(c["high"]),
        "low":    float(c["low"]),
        "close":  float(c["close"]),
        "volume": float(c.get("baseVol", c.get("vol", 0))),
    }


def fetch_latest_candles(client: BitunixClient, symbol: str, n: int = 20) -> list[dict]:
    resp = client.get("/api/v1/futures/market/kline", {
        "symbol": symbol, "interval": INTERVAL, "limit": str(n),
    })
    if resp.get("code") != 0:
        raise RuntimeError(f"Kline error: {resp.get('msg')}")
    return [normalize(c) for c in resp.get("data", [])]


def load_candles_csv(symbol: str) -> list[dict]:
    path = os.path.join("data", f"{symbol}_1m.csv")
    if not os.path.exists(path):
        return []
    candles = []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
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


def fetch_gap_candles(client: BitunixClient, symbol: str, after_ms: int) -> list[dict]:
    now_ms = int(now_utc().timestamp() * 1000)
    end_ms = now_ms
    raw    = []
    while True:
        resp = client.get("/api/v1/futures/market/kline", {
            "symbol": symbol, "interval": INTERVAL,
            "limit": "1000", "endTime": str(end_ms),
        })
        if resp.get("code") != 0:
            raise RuntimeError(f"Kline error: {resp.get('msg')}")
        batch = resp.get("data", [])
        if not batch:
            break
        batch = [c for c in batch if int(c["time"]) > after_ms]
        raw.extend(batch)
        oldest = min(int(c["time"]) for c in resp["data"])
        if oldest <= after_ms:
            break
        end_ms = oldest - 1

    seen, candles = set(), []
    for c in raw:
        ts = int(c["time"])
        if ts not in seen:
            seen.add(ts)
            candles.append(normalize(c))
    candles.sort(key=lambda c: c["time"])
    return candles


def fetch_candles_paged(client: BitunixClient, symbol: str, pages: int = 4) -> list[dict]:
    now_ms = int(now_utc().timestamp() * 1000)
    end_ms = now_ms
    raw    = []
    for _ in range(pages):
        resp = client.get("/api/v1/futures/market/kline", {
            "symbol": symbol, "interval": INTERVAL,
            "limit": "1000", "endTime": str(end_ms),
        })
        if resp.get("code") != 0:
            raise RuntimeError(f"Kline error: {resp.get('msg')}")
        batch = resp.get("data", [])
        if not batch:
            break
        raw.extend(batch)
        end_ms = min(int(c["time"]) for c in batch) - 1

    seen, candles = set(), []
    for c in raw:
        ts = int(c["time"])
        if ts not in seen:
            seen.add(ts)
            candles.append(normalize(c))
    candles.sort(key=lambda c: c["time"])
    return candles


# ── 15m resampling ─────────────────────────────────────────────────────────────

def build_15m_buf(candles_1m: list[dict]) -> tuple[list[dict], dict | None, int | None]:
    buf_15m        = []
    current_15m    = None
    current_bucket = None

    for c in candles_1m:
        bucket = (c["time"] // FIFTEEN_MIN_MS) * FIFTEEN_MIN_MS
        if current_bucket is None:
            current_bucket = bucket
            current_15m    = _start_15m(c)
        elif bucket != current_bucket:
            buf_15m.append(current_15m)
            current_bucket = bucket
            current_15m    = _start_15m(c)
        else:
            _update_15m(current_15m, c)

    return buf_15m, current_15m, current_bucket


def _start_15m(c: dict) -> dict:
    return {"time": c["time"], "open": c["open"], "high": c["high"],
            "low": c["low"], "close": c["close"], "volume": c["volume"]}


def _update_15m(c15: dict, c: dict) -> None:
    c15["high"]    = max(c15["high"], c["high"])
    c15["low"]     = min(c15["low"],  c["low"])
    c15["close"]   = c["close"]
    c15["volume"] += c["volume"]


# ── EMA and ATR ────────────────────────────────────────────────────────────────

def init_emas(buf_15m: list[dict], fast_period: int, slow_period: int) -> tuple[float | None, float | None]:
    """Seed EMAs from historical 15m buffer. Returns (ema_fast, ema_slow)."""
    closes = [c["close"] for c in buf_15m]
    if len(closes) < slow_period + 1:
        return None, None

    fast_alpha = 2.0 / (fast_period + 1)
    slow_alpha = 2.0 / (slow_period + 1)

    ema_f = sum(closes[:fast_period]) / fast_period
    for c in closes[fast_period:]:
        ema_f = fast_alpha * c + (1 - fast_alpha) * ema_f

    ema_s = sum(closes[:slow_period]) / slow_period
    for c in closes[slow_period:]:
        ema_s = slow_alpha * c + (1 - slow_alpha) * ema_s

    return ema_f, ema_s


def compute_atr(buf_15m: list[dict], period: int = ATR_PERIOD) -> float | None:
    if len(buf_15m) < period + 1:
        return None
    recent = buf_15m[-(period + 1):]
    trs = [
        max(recent[i]["high"] - recent[i]["low"],
            abs(recent[i]["high"] - recent[i - 1]["close"]),
            abs(recent[i]["low"]  - recent[i - 1]["close"]))
        for i in range(1, len(recent))
    ]
    return sum(trs) / len(trs)


# ── API helpers ────────────────────────────────────────────────────────────────

def round_qty(symbol: str, qty: float) -> float:
    return round(qty, PRECISION.get(symbol, {}).get("qty", 4))


def round_price(symbol: str, price: float) -> float:
    return round(price, PRECISION.get(symbol, {}).get("price", 2))


def get_balance(client: BitunixClient) -> float:
    resp = client.get("/api/v1/futures/account", {"marginCoin": MARGIN_COIN})
    if resp.get("code") != 0:
        raise RuntimeError(f"Balance error: {resp.get('msg')}")
    d = resp["data"]
    return float(d.get("available", 0)) + float(d.get("crossUnrealizedPNL", 0))


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
        log.warning(f"  Leverage set failed {symbol}: {resp.get('msg')}")
    else:
        log.info(f"  {symbol} leverage → {LEVERAGE}×")


def log_trade(body: dict) -> None:
    os.makedirs("log", exist_ok=True)
    ts       = now_utc().strftime("%Y-%m-%d %H:%M:%S")
    buy_sell = "BUY" if body.get("side") == "LONG" else "SELL"
    row = [
        ts, ts,
        "symbol",      body["symbol"],
        "qty",         body["qty"],
        "side",        buy_sell,
        "orderType",   "MARKET",
        "tradeSide",   "OPEN",
        "tpPrice",     body["tp_price"],
        "slPrice",     body["sl_price"],
        "tpStopType",  "MARK_PRICE",
        "slStopType",  "MARK_PRICE",
        "entryPrice",  body["entry_price"],
        "armId",       body.get("arm_id", ""),
        "armPrice",    body.get("arm_price", ""),
        "signalPrice", body.get("signal_price", ""),
    ]
    with open(TRADE_CSV, "a", newline="") as f:
        csv.writer(f).writerow(row)


def place_order(client: BitunixClient, symbol: str, side: str,
                qty: float, tp_price: float, sl_price: float,
                debug: bool) -> str | None:
    buy_sell = "BUY" if side == "LONG" else "SELL"
    body = {
        "symbol":     symbol,
        "qty":        str(qty),
        "side":       buy_sell,
        "orderType":  "MARKET",
        "tradeSide":  "OPEN",
        "tpPrice":    str(round_price(symbol, tp_price)),
        "slPrice":    str(round_price(symbol, sl_price)),
        "tpStopType": "MARK_PRICE",
        "slStopType": "MARK_PRICE",
    }
    if debug:
        log.info(f"  [DEBUG] Would place: {json.dumps(body)}")
        return "DEBUG-ID"
    resp = client.post("/api/v1/futures/trade/place_order", body)
    if resp.get("code") != 0:
        log.error(f"  Order failed: {resp.get('msg')}")
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


def update_sl(client: BitunixClient, symbol: str,
              position_id: str, new_sl: float) -> bool:
    """Update the exchange-managed SL for an open position."""
    resp = client.post("/api/v1/futures/tpsl/position/modify_order", {
        "symbol":     symbol,
        "positionId": position_id,
        "slPrice":    str(new_sl),
        "slStopType": "MARK_PRICE",
    })
    if resp.get("code") != 0:
        log.error(f"  SL update failed: {resp.get('msg')}")
        return False
    return True


def _check_breakeven(client: BitunixClient, sym: str, pos: dict,
                     price: float, debug: bool) -> None:
    """Move SL to entry + LOCK_ATR × ATR once profit reaches BREAKEVEN_ATR × ATR."""
    if pos.get("breakeven_triggered") or not pos.get("atr"):
        return

    side   = pos["side"]
    atr    = pos["atr"]
    profit = (price - pos["entry_price"]) if side == "LONG" else (pos["entry_price"] - price)

    target = BREAKEVEN_ATR * atr
    log.debug(f"  {sym}: breakeven check — profit={profit:.4f}  threshold={target:.4f}  triggered={pos.get('breakeven_triggered')}")
    if profit < target:
        return

    if side == "LONG":
        new_sl = round_price(sym, pos["entry_price"] + LOCK_ATR * atr)
        if pos["sl_price"] is not None and new_sl <= pos["sl_price"]:
            return
    else:
        new_sl = round_price(sym, pos["entry_price"] - LOCK_ATR * atr)
        if pos["sl_price"] is not None and new_sl >= pos["sl_price"]:
            return

    old_sl = pos["sl_price"]

    if debug:
        log.info(f"  {sym} [DEBUG]: BREAKEVEN activated — "
                 f"SL {old_sl:.4f} → {new_sl:.4f}  (entry + {LOCK_ATR}×ATR)")
    else:
        if not update_sl(client, sym, pos["position_id"], new_sl):
            return
        log.info(f"  {sym}: BREAKEVEN activated — "
                 f"SL {old_sl:.4f} → {new_sl:.4f}  (entry + {LOCK_ATR}×ATR)")

    pos["sl_price"]           = new_sl
    pos["breakeven_triggered"] = True


# ── Per-symbol processing ──────────────────────────────────────────────────────

def _process_symbol(client: BitunixClient, sym: str, s: dict,
                    balance: float, debug: bool) -> None:
    cfg        = SYMBOL_CONFIGS[sym]
    fast_alpha = 2.0 / (cfg["fast_period"] + 1)
    slow_alpha = 2.0 / (cfg["slow_period"] + 1)
    min_buf    = cfg["slow_period"] + 10

    # ── 1. Fetch new 1m candles ────────────────────────────────────────────────
    recent = fetch_latest_candles(client, sym, n=20)
    new    = [c for c in recent if c["time"] > s["last_ts"]]
    new.sort(key=lambda c: c["time"])

    if not new:
        log.info(f"  {sym}: no new candle yet")
        return

    # ── 2. Update 15m buffer ───────────────────────────────────────────────────
    completed_15m = None
    for c1 in new:
        bucket = (c1["time"] // FIFTEEN_MIN_MS) * FIFTEEN_MIN_MS
        if s["current_bucket"] is None:
            s["current_bucket"] = bucket
            s["current_15m"]    = _start_15m(c1)
        elif bucket != s["current_bucket"]:
            completed_15m = s["current_15m"]
            s["buf_15m"].append(completed_15m)
            if len(s["buf_15m"]) > 300:
                s["buf_15m"] = s["buf_15m"][-300:]
            s["current_bucket"] = bucket
            s["current_15m"]    = _start_15m(c1)
        else:
            _update_15m(s["current_15m"], c1)
        s["last_ts"] = c1["time"]

    last_1m = new[-1]

    # ── 3. IN_TRADE: monitor on 1m tick + EMA re-cross on 15m close ───────────
    if s["state"] == "IN_TRADE":
        _monitor_trade(client, sym, s, last_1m, completed_15m,
                       fast_alpha, slow_alpha, debug)
        return

    # ── 4. ENTRY_PENDING: enter at open of next 1m candle ─────────────────────
    if s["state"] == "ENTRY_PENDING":
        _enter_pending(client, sym, s, last_1m, balance, debug)
        return

    # ── 5. WATCHING: check for EMA crossover on newly completed 15m candle ────
    if completed_15m is None:
        log.info(f"  {sym}: waiting for 15m close  buf={len(s['buf_15m'])}")
        return

    if len(s["buf_15m"]) < min_buf or s["ema_fast"] is None:
        log.info(f"  {sym}: warming up ({len(s['buf_15m'])}/{min_buf} 15m candles)")
        # Try to init EMAs if we now have enough history
        if len(s["buf_15m"]) >= min_buf and s["ema_fast"] is None:
            s["ema_fast"], s["ema_slow"] = init_emas(
                s["buf_15m"], cfg["fast_period"], cfg["slow_period"])
        return

    # Update EMA on newly completed candle
    close      = completed_15m["close"]
    prev_fast  = s["ema_fast"]
    prev_slow  = s["ema_slow"]
    s["ema_fast"] = fast_alpha * close + (1 - fast_alpha) * prev_fast
    s["ema_slow"] = slow_alpha * close + (1 - slow_alpha) * prev_slow

    atr     = compute_atr(s["buf_15m"])
    atr_str = f"{atr:.4f}" if atr is not None else "n/a"

    # Detect crossover
    signal_side = None
    if prev_fast <= prev_slow and s["ema_fast"] > s["ema_slow"]:
        signal_side = "LONG"
    elif prev_fast >= prev_slow and s["ema_fast"] < s["ema_slow"]:
        signal_side = "SHORT"

    log.info(
        f"  {sym}  15m_close={close:.4f}  "
        f"ema_fast={s['ema_fast']:.4f}  ema_slow={s['ema_slow']:.4f}  "
        f"atr={atr_str}"
        + (f"  ← CROSS {signal_side}" if signal_side else "")
    )

    if signal_side is None or atr is None:
        return

    s["pending_side"] = signal_side
    s["pending_atr"]  = atr
    s["state"]        = "ENTRY_PENDING"
    s["arm_id"]       = arm_log.new_arm_id()
    s["arm_time"]     = now_utc().strftime("%Y-%m-%d %H:%M:%S")
    s["arm_price"]    = close

    if signal_side == "LONG":
        wtp, wsl = close + atr * TP_MULT, close - atr * SL_MULT
    else:
        wtp, wsl = close - atr * TP_MULT, close + atr * SL_MULT
    log.info(f"  {sym}: EMA CROSS {signal_side}  wtp≈{wtp:.4f}  wsl≈{wsl:.4f}")


def _enter_pending(client: BitunixClient, sym: str, s: dict,
                   c1: dict, balance: float, debug: bool) -> None:
    side = s["pending_side"]
    atr  = s["pending_atr"]

    ticker      = fetch_ticker(client, sym)
    entry_price = float(ticker.get("lastPrice", c1["open"]))

    if side == "LONG":
        tp_price = entry_price + atr * TP_MULT
        sl_price = entry_price - atr * SL_MULT
    else:
        tp_price = entry_price - atr * TP_MULT
        sl_price = entry_price + atr * SL_MULT

    # Fee gate
    tp_dist  = abs(tp_price - entry_price)
    fee_cost = entry_price * ROUND_TRIP_FEE
    if tp_dist <= fee_cost:
        log.info(f"  {sym}: SKIP entry — TP dist {tp_dist:.4f} ≤ fee {fee_cost:.4f}")
        arm_log.log_arm_event(s["arm_id"], STRATEGY, sym, s["arm_time"], s["arm_price"],
                              side, "NO_FIRE", disarm_price=entry_price,
                              no_fire_reason="FEE_GATE", atr=atr)
        s["state"] = "WATCHING"
        _clear_pending(s)
        return

    notional = balance * MAX_TRADE_PCT * LEVERAGE
    qty      = round_qty(sym, notional / entry_price)
    min_qty  = {"BTCUSDT": 0.0001, "ETHUSDT": 0.003}.get(sym, 0.001)
    if qty < min_qty:
        log.info(f"  {sym}: qty {qty} < min {min_qty}, skip")
        arm_log.log_arm_event(s["arm_id"], STRATEGY, sym, s["arm_time"], s["arm_price"],
                              side, "NO_FIRE", disarm_price=entry_price,
                              no_fire_reason="MIN_QTY", atr=atr)
        s["state"] = "WATCHING"
        _clear_pending(s)
        return

    log.info(f"  {sym}: ENTER {side}  entry≈{entry_price:.4f}  "
             f"tp={tp_price:.4f} (backstop)  sl={sl_price:.4f}  qty={qty}")

    order_id = place_order(client, sym, side, qty, tp_price, sl_price, debug)
    if order_id is None:
        arm_log.log_arm_event(s["arm_id"], STRATEGY, sym, s["arm_time"], s["arm_price"],
                              side, "NO_FIRE", disarm_price=entry_price,
                              no_fire_reason="ORDER_FAILED", atr=atr)
        s["state"] = "WATCHING"
        _clear_pending(s)
        return

    arm_log.log_arm_event(s["arm_id"], STRATEGY, sym, s["arm_time"], s["arm_price"],
                          side, "FIRED", disarm_price=entry_price, atr=atr)
    log_trade({
        "symbol":       sym,
        "side":         side,
        "qty":          qty,
        "tp_price":     round_price(sym, tp_price),
        "sl_price":     round_price(sym, sl_price),
        "entry_price":  entry_price,
        "arm_id":       s["arm_id"],
        "arm_price":    s["arm_price"],
        "signal_price": entry_price,
    })

    s["state"]    = "IN_TRADE"
    s["position"] = {
        "position_id":         order_id,
        "side":                side,
        "entry_price":         entry_price,
        "tp_price":            tp_price,
        "sl_price":            sl_price,
        "qty":                 qty,
        "opened_at":           now_utc(),
        "debug":               debug,
        "arm_id":              s["arm_id"],
        "atr":                 atr,
        "breakeven_triggered": False,
    }
    _clear_pending(s)


def _monitor_trade(client: BitunixClient, sym: str, s: dict,
                   c1: dict, completed_15m: dict | None,
                   fast_alpha: float, slow_alpha: float, debug: bool) -> None:
    pos       = s["position"]
    held_mins = (now_utc() - pos["opened_at"]).total_seconds() / 60
    side      = pos["side"]
    tp        = pos["tp_price"]
    sl        = pos["sl_price"]

    # Update EMAs and check for reversal on each completed 15m candle
    ema_reversed = False
    if completed_15m is not None:
        close         = completed_15m["close"]
        s["ema_fast"] = fast_alpha * close + (1 - fast_alpha) * s["ema_fast"]
        s["ema_slow"] = slow_alpha * close + (1 - slow_alpha) * s["ema_slow"]
        if side == "LONG"  and s["ema_fast"] < s["ema_slow"]:
            ema_reversed = True
        elif side == "SHORT" and s["ema_fast"] > s["ema_slow"]:
            ema_reversed = True
        if ema_reversed:
            log.info(f"  {sym}: EMA reversal — fast={s['ema_fast']:.4f}  slow={s['ema_slow']:.4f}")

    # Breakeven stop — may update pos["sl_price"] in-place
    _check_breakeven(client, sym, pos, c1["close"], debug)
    sl = pos["sl_price"]  # re-read after potential update

    if debug:
        price  = c1["close"]
        tp_hit = (c1["high"] >= tp) if side == "LONG" else (c1["low"]  <= tp)
        sl_hit = (c1["low"]  <= sl) if side == "LONG" else (c1["high"] >= sl)
        pnl    = (price - pos["entry_price"]) * pos["qty"]
        if side == "SHORT":
            pnl = -pnl

        log.info(
            f"  {sym} [DEBUG {side}]  entry={pos['entry_price']:.4f}  "
            f"now={price:.4f}  uPnL≈{pnl:+.4f}  held={held_mins:.1f}min  "
            f"ema_fast={s['ema_fast']:.4f}  ema_slow={s['ema_slow']:.4f}"
            + ("  ← EMA REVERSED" if ema_reversed else "")
        )

        if tp_hit and sl_hit:
            outcome = "TP" if abs(tp - pos["entry_price"]) <= abs(sl - pos["entry_price"]) else "SL"
        elif tp_hit:
            outcome = "TP"
        elif sl_hit:
            outcome = "SL"
        elif ema_reversed:
            outcome = "EMA_EXIT"
        elif held_mins >= MAX_HOLD_MINS:
            outcome = "TIME"
        else:
            outcome = None

        if outcome:
            exit_price = tp if outcome == "TP" else (sl if outcome == "SL" else price)
            log.info(f"  {sym} [DEBUG]: {outcome} @ {exit_price:.4f}  held={held_mins:.1f}min")
            arm_log.log_close_event(
                pos.get("arm_id", ""), STRATEGY, sym, side, outcome,
                exit_price, held_mins,
                arm_log.calc_pnl(side, pos["entry_price"], exit_price, pos["qty"], ROUND_TRIP_FEE),
                order_id=pos.get("position_id"),
            )
            s["state"]    = "WATCHING"
            s["position"] = None
        return

    # Live: check if exchange already closed (TP/SL hit inline)
    live = [p for p in get_open_positions(client, sym)
            if p.get("positionId") == pos["position_id"]]

    if not live:
        cur_price = c1["close"]
        tp_px, sl_px = pos.get("tp_price"), pos.get("sl_price")
        if tp_px is not None and sl_px is not None:
            if side == "LONG":
                outcome_str = "TP" if cur_price >= tp_px else ("SL" if cur_price <= sl_px else "EXCHANGE_CLOSED")
            else:
                outcome_str = "TP" if cur_price <= tp_px else ("SL" if cur_price >= sl_px else "EXCHANGE_CLOSED")
            exit_px = tp_px if outcome_str == "TP" else (sl_px if outcome_str == "SL" else cur_price)
        else:
            outcome_str, exit_px = "EXCHANGE_CLOSED", cur_price
        log.info(f"  {sym}: position closed by exchange ({outcome_str})")
        arm_log.log_close_event(
            pos.get("arm_id", ""), STRATEGY, sym, side, outcome_str,
            exit_px, held_mins,
            arm_log.calc_pnl(side, pos["entry_price"], exit_px, pos["qty"], ROUND_TRIP_FEE),
            order_id=pos.get("position_id"),
        )
        s["state"]    = "WATCHING"
        s["position"] = None
        return

    p    = live[0]
    upnl = float(p.get("unrealizedPNL", 0))
    log.info(
        f"  {sym} [{side}]  entry={pos['entry_price']:.4f}  "
        f"uPnL={upnl:+.4f}  held={held_mins:.1f}min  "
        f"ema_fast={s['ema_fast']:.4f}  ema_slow={s['ema_slow']:.4f}"
        + ("  ← EMA REVERSED" if ema_reversed else "")
    )

    exit_reason = None
    if ema_reversed:
        exit_reason = "EMA_EXIT"
    elif held_mins >= MAX_HOLD_MINS:
        exit_reason = "TIME"

    if exit_reason:
        log.info(f"  {sym}: {exit_reason} after {held_mins:.1f}min")
        if close_position(client, pos["position_id"], sym, debug):
            arm_log.log_close_event(
                pos.get("arm_id", ""), STRATEGY, sym, side, exit_reason,
                c1["close"], held_mins,
                arm_log.calc_pnl(side, pos["entry_price"], c1["close"], pos["qty"], ROUND_TRIP_FEE),
                order_id=pos.get("position_id"),
            )
            s["state"]    = "WATCHING"
            s["position"] = None


def _clear_pending(s: dict) -> None:
    s["pending_side"] = None
    s["pending_atr"]  = None
    s["arm_id"]       = None
    s["arm_time"]     = None
    s["arm_price"]    = None


# ── Main loop ──────────────────────────────────────────────────────────────────

def run(debug: bool, symbols: list[str] = None) -> None:
    client = BitunixClient(API_KEY, SECRET_KEY)
    active = symbols or SYMBOLS

    log.info("━" * 60)
    log.info("  EMA Trend Trader")
    log.info(f"  Symbols     : {', '.join(active)}")
    log.info(f"  Windows     : 24/7 (no time restriction)")
    for sym in active:
        cfg = SYMBOL_CONFIGS[sym]
        log.info(f"  {sym}  fast={cfg['fast_period']}  slow={cfg['slow_period']}")
    log.info(f"  ATR period  : {ATR_PERIOD}")
    log.info(f"  SL/TP       : {SL_MULT}/{TP_MULT}× ATR  (primary exit: EMA re-cross)")
    log.info(f"  Hold        : ≤{MAX_HOLD_MINS}min ({MAX_HOLD_MINS // 60}h)  |  Leverage: {LEVERAGE}×")
    log.info(f"  Max trade   : {MAX_TRADE_PCT:.0%}  |  Fees: {ROUND_TRIP_FEE * 100:.3f}%")
    if debug:
        log.info("  MODE        : DEBUG — no real orders")
    log.info("━" * 60)

    for sym in active:
        set_leverage(client, sym, debug)

    start_balance = get_balance(client)
    min_balance   = start_balance * MIN_BALANCE_PCT
    log.info(f"  Balance: {start_balance:.4f} USDT  |  Floor: {min_balance:.4f}")

    # ── Initialise per-symbol state ────────────────────────────────────────────
    sym_state: dict[str, dict] = {}
    for sym in active:
        cfg = SYMBOL_CONFIGS[sym]

        csv_candles = load_candles_csv(sym)
        if csv_candles:
            log.info(f"  {sym}: {len(csv_candles):,} candles from local CSV  "
                     f"({datetime.fromtimestamp(csv_candles[0]['time']/1000, tz=timezone.utc).strftime('%Y-%m-%d')} "
                     f"→ {datetime.fromtimestamp(csv_candles[-1]['time']/1000, tz=timezone.utc).strftime('%Y-%m-%d')})")
            log.info(f"  {sym}: fetching gap candles from API...")
            gap_candles = fetch_gap_candles(client, sym, csv_candles[-1]["time"])
            log.info(f"  {sym}: {len(gap_candles)} gap candles fetched")
            candles_1m = csv_candles + gap_candles
        else:
            log.info(f"  {sym}: no local CSV, fetching from API...")
            candles_1m = fetch_candles_paged(client, sym, pages=4)
            log.info(f"  {sym}: {len(candles_1m)} candles from API")

        buf_15m, current_15m, current_bucket = build_15m_buf(candles_1m)
        ema_fast, ema_slow = init_emas(buf_15m, cfg["fast_period"], cfg["slow_period"])

        if ema_fast is None:
            log.warning(f"  {sym}: insufficient 15m history — "
                        f"need {cfg['slow_period'] + 1}, have {len(buf_15m)}")
        else:
            log.info(f"  {sym}: {len(buf_15m)} 15m candles  "
                     f"ema_fast={ema_fast:.4f}  ema_slow={ema_slow:.4f}  — ready")

        sym_state[sym] = {
            "buf_15m":        buf_15m[-(cfg["slow_period"] + 50):],
            "current_15m":    current_15m,
            "current_bucket": current_bucket,
            "last_ts":        candles_1m[-1]["time"] if candles_1m else 0,
            "ema_fast":       ema_fast,
            "ema_slow":       ema_slow,
            # State machine
            "state":          "WATCHING",
            "pending_side":   None,
            "pending_atr":    None,
            "position":       None,
            # Arm event tracking
            "arm_id":         None,
            "arm_time":       None,
            "arm_price":      None,
        }

    # ── Hydrate open positions from exchange ───────────────────────────────────
    if not debug:
        for p in get_open_positions(client):
            sym = p.get("symbol")
            if sym not in active:
                continue
            side = "LONG" if p.get("side", "").upper() == "BUY" else "SHORT"
            sym_state[sym]["state"]    = "IN_TRADE"
            sym_state[sym]["position"] = {
                "position_id":         p.get("positionId"),
                "side":                side,
                "entry_price":         float(p.get("avgOpenPrice", 0)),
                "tp_price":            None,
                "sl_price":            None,
                "qty":                 float(p.get("qty", 0)),
                "opened_at":           now_utc(),
                "debug":               False,
                "arm_id":              "",
                "atr":                 None,
                "breakeven_triggered": False,
            }
            log.info(f"  Hydrated {sym} [{side}] @ "
                     f"{sym_state[sym]['position']['entry_price']:.4f}")

    # ── Main cycle ─────────────────────────────────────────────────────────────
    while True:
        cycle_start = now_utc()
        try:
            balance = get_balance(client)
            log.info(f"  Balance: {balance:.4f}  |  {cycle_start.strftime('%H:%M:%S')} UTC")

            if balance < min_balance:
                log.warning(f"  CIRCUIT BREAKER: {balance:.4f} < floor {min_balance:.4f} — halting")
                break

            for sym in active:
                try:
                    _process_symbol(client, sym, sym_state[sym], balance, debug)
                except Exception as e:
                    log.error(f"  {sym}: error — {e}")

        except Exception as e:
            log.error(f"  Cycle error: {e}")

        elapsed = (now_utc() - cycle_start).total_seconds()
        sleep_s = max(0, POLL_SECS - elapsed)
        log.info(f"  Sleeping {sleep_s:.0f}s")
        time.sleep(sleep_s)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMA Trend Trader")
    parser.add_argument("--debug",  action="store_true", help="DEBUG mode — no real orders")
    parser.add_argument("--symbol", help="Single symbol (default: both)")
    args = parser.parse_args()
    run(args.debug, [args.symbol] if args.symbol else None)
