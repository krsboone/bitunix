"""
bb_trader.py — Bitunix Bollinger Band momentum perpetual futures trader

Strategy: BB band-touch momentum (--flip mode from bb_sim.py)
  - Resample live 1m candles to 5m internally
  - When 5m candle closes outside a Bollinger Band → enter in that direction
    (momentum continuation, NOT mean-reversion)
  - TP:   0.5 × half_band_width beyond the band (tight, high hit-rate target)
  - SL:   2.0 × half_band_width back inside the band (wide stop, rarely hit)
  - Band-width squeeze filter: skip entries when bands are expanding rapidly
  - Directional cooldown: block same-direction re-entry for N 5m candles after SL
  - Time exit: close if held > MAX_HOLD_MINS

Per-symbol sweep-optimised parameters (30-day walk-forward + 7-day holdout):
  BTCUSDT: period=30  mult=1.5  (holdout TP 80.1%  SL 5.2%  score 74.9)
  ETHUSDT: period=20  mult=1.5  (holdout TP 82.3%  SL 5.6%  score 76.7)
  Shared:  tp_mult=1.0  sl_mult=2.0  squeeze=1.0  cooldown=5

Usage:
    python3 bb_trader.py              # live trading
    python3 bb_trader.py --debug      # DEBUG mode — no real orders
    python3 bb_trader.py --symbol BTCUSDT
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
from market import fetch_ticker
from log_cap import start_logging
import arm_log

start_logging("bb_trader")

STRATEGY = "bb"

# ── Per-symbol sweep-optimised config ─────────────────────────────────────────

SYMBOL_CONFIGS = {
    "BTCUSDT": {
        "period":    30,
        "mult":      1.5,
        "windows":   [(11, 13)],   # 11:00–13:00 UTC Mon-Fri
        "skip_days": {5, 6},       # Sat, Sun
    },
    "ETHUSDT": {
        "period":    20,
        "mult":      1.5,
        "windows":   [(9, 11), (14, 17)],  # 09:00–11:00 and 14:00–17:00 UTC Mon-Fri
        "skip_days": {5, 6},               # Sat, Sun
    },
}
SYMBOLS = list(SYMBOL_CONFIGS.keys())

# ── Shared strategy parameters ─────────────────────────────────────────────────

TP_MULT          = 1.0    # TP = tp_mult × half_band_width beyond the band
SL_MULT          = 2.0    # SL = sl_mult × half_band_width back inside the band
SQUEEZE_THRESH   = 1.0    # enter only when bw ≤ bw_rolling_avg × threshold
SQUEEZE_LOOKBACK = 50     # 5m candles for rolling band-width average
COOLDOWN_5M      = 5      # 5m candles to block same-direction re-entry after SL
MAX_HOLD_MINS    = 33

FIVE_MIN_MS      = 5 * 60 * 1000
MIN_HISTORY_5M   = max(50, SQUEEZE_LOOKBACK) + 10   # 5m candles before trading

# ── Trading parameters ─────────────────────────────────────────────────────────

LEVERAGE         = 2
MARGIN_COIN      = "USDT"
INTERVAL         = "1m"
MAX_TRADE_PCT    = 0.10    # 10% of balance per trade
POLL_SECS        = 60      # 1-minute candle rhythm

FEE_TAKER        = 0.00060
FEE_MAKER        = 0.00060
ROUND_TRIP_FEE   = FEE_TAKER + FEE_MAKER
MIN_BALANCE_PCT  = 0.70

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

TRADE_CSV = os.path.join("log", "bb_trades.csv")


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


def fetch_candles_paged(client: BitunixClient, symbol: str,
                        pages: int = 2) -> list[dict]:
    """Fetch pages × 1000 historical 1m candles, sorted ascending."""
    now_ms = int(now_utc().timestamp() * 1000)
    end_ms = now_ms
    raw    = []

    for _ in range(pages):
        resp = client.get("/api/v1/futures/market/kline", {
            "symbol":   symbol,
            "interval": INTERVAL,
            "limit":    "1000",
            "endTime":  str(end_ms),
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


def fetch_latest_candles(client: BitunixClient, symbol: str,
                         n: int = 10) -> list[dict]:
    resp = client.get("/api/v1/futures/market/kline", {
        "symbol":   symbol,
        "interval": INTERVAL,
        "limit":    str(n),
    })
    if resp.get("code") != 0:
        raise RuntimeError(f"Kline error: {resp.get('msg')}")
    return [normalize(c) for c in resp.get("data", [])]


def load_candles_csv(symbol: str) -> list[dict]:
    """Load locally-cached 1m candles from fetch_data.py CSV, sorted ascending."""
    import csv as _csv
    path = os.path.join("data", f"{symbol}_1m.csv")
    if not os.path.exists(path):
        return []
    candles = []
    with open(path, "r") as f:
        reader = _csv.DictReader(f)
        for row in reader:
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


def fetch_gap_candles(client: BitunixClient, symbol: str,
                      after_ms: int) -> list[dict]:
    """Fetch all 1m candles from after_ms up to now via paged API calls."""
    now_ms = int(now_utc().timestamp() * 1000)
    end_ms = now_ms
    raw    = []

    while True:
        resp = client.get("/api/v1/futures/market/kline", {
            "symbol":    symbol,
            "interval":  INTERVAL,
            "limit":     "1000",
            "endTime":   str(end_ms),
        })
        if resp.get("code") != 0:
            raise RuntimeError(f"Kline error: {resp.get('msg')}")
        batch = resp.get("data", [])
        if not batch:
            break
        # Keep only candles newer than the CSV
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


# ── 5m resampling ──────────────────────────────────────────────────────────────

def build_5m_buf(candles_1m: list[dict]) -> tuple[list[dict], dict | None, int | None]:
    """
    Build a completed 5m candle buffer from 1m history.
    Returns (buf_5m, current_5m_in_progress, current_bucket).
    """
    buf_5m         = []
    current_5m     = None
    current_bucket = None

    for c in candles_1m:
        bucket = (c["time"] // FIVE_MIN_MS) * FIVE_MIN_MS
        if current_bucket is None:
            current_bucket = bucket
            current_5m = _start_5m(c)
        elif bucket != current_bucket:
            buf_5m.append(current_5m)
            current_bucket = bucket
            current_5m = _start_5m(c)
        else:
            _update_5m(current_5m, c)

    return buf_5m, current_5m, current_bucket


def _start_5m(c: dict) -> dict:
    return {"time": c["time"], "open": c["open"], "high": c["high"],
            "low": c["low"], "close": c["close"], "volume": c["volume"]}


def _update_5m(c5: dict, c: dict) -> None:
    c5["high"]    = max(c5["high"],  c["high"])
    c5["low"]     = min(c5["low"],   c["low"])
    c5["close"]   = c["close"]
    c5["volume"] += c["volume"]


# ── Bollinger Band computation ─────────────────────────────────────────────────

def bollinger(buf_5m: list[dict], period: int, mult: float) -> tuple:
    """
    Returns (sma, upper, lower, half_width) from the last `period` candles.
    Returns (None, None, None, 0) if insufficient data.
    """
    if len(buf_5m) < period:
        return None, None, None, 0.0
    closes = [c["close"] for c in buf_5m[-period:]]
    sma    = sum(closes) / period
    var    = sum((p - sma) ** 2 for p in closes) / (period - 1)
    hw     = mult * math.sqrt(var)
    return sma, sma + hw, sma - hw, hw


def band_width(sma: float, hw: float) -> float:
    return 2 * hw / sma if sma > 0 else 0.0


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
    """Append entry in backtest.py-compatible key-value format."""
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


# ── Per-symbol processing ──────────────────────────────────────────────────────

def _process_symbol(client: BitunixClient, sym: str, s: dict,
                    balance: float, debug: bool) -> None:
    """Fetch new 1m candles, update 5m state, run signal/trade logic."""
    cfg = SYMBOL_CONFIGS.get(sym, {"period": 20, "mult": 1.5})

    # ── 1. Fetch new 1m candles ────────────────────────────────────────────────
    recent = fetch_latest_candles(client, sym, n=10)
    new    = [c for c in recent if c["time"] > s["last_ts"]]
    new.sort(key=lambda c: c["time"])

    if not new:
        log.info(f"  {sym}: no new candle yet")
        _log_state(sym, s)
        return

    # ── 2. Append to 1m buffer and update 5m state ────────────────────────────
    completed_5m = None
    for c1 in new:
        bucket = (c1["time"] // FIVE_MIN_MS) * FIVE_MIN_MS

        if s["current_bucket"] is None:
            s["current_bucket"] = bucket
            s["current_5m"]     = _start_5m(c1)
        elif bucket != s["current_bucket"]:
            # Previous 5m candle just completed
            completed_5m = s["current_5m"]
            s["buf_5m"].append(completed_5m)
            if len(s["buf_5m"]) > 500:
                s["buf_5m"] = s["buf_5m"][-500:]
            s["current_bucket"] = bucket
            s["current_5m"]     = _start_5m(c1)
            # Decrement directional cooldowns
            if s["long_blocked"]  > 0: s["long_blocked"]  -= 1
            if s["short_blocked"] > 0: s["short_blocked"] -= 1
        else:
            _update_5m(s["current_5m"], c1)

        s["last_ts"] = c1["time"]

    last_1m = new[-1]

    # ── 3. IN_TRADE: monitor on latest 1m candle ──────────────────────────────
    if s["state"] == "IN_TRADE":
        _monitor_trade(client, sym, s, last_1m, debug)
        return

    # ── 4. ENTRY_PENDING: enter at open of next 1m candle ─────────────────────
    if s["state"] == "ENTRY_PENDING":
        _enter_pending(client, sym, s, last_1m, balance, debug)
        return

    # ── 5. WATCHING: check for BB signal on newly completed 5m candle ─────────
    if completed_5m is None or len(s["buf_5m"]) < MIN_HISTORY_5M:
        log.info(f"  {sym}: warming up ({len(s['buf_5m'])}/{MIN_HISTORY_5M} 5m candles)")
        return

    # Time-of-day and day-of-week filter (only blocks new entries, not open trades)
    candle_dt   = datetime.fromtimestamp(completed_5m["time"] / 1000, tz=timezone.utc)
    candle_hour = candle_dt.hour
    candle_dow  = candle_dt.weekday()
    windows = cfg.get("windows")
    if windows:
        in_window = False
        for sh, eh in windows:
            if sh <= eh:
                if sh <= candle_hour < eh:
                    in_window = True; break
            else:  # wraps midnight
                if candle_hour >= sh or candle_hour < eh:
                    in_window = True; break
        if not in_window:
            log.info(f"  {sym}: outside trade window ({candle_dt.strftime('%a %H:%M')} UTC) — skip")
            return
    if candle_dow in cfg.get("skip_days", set()):
        log.info(f"  {sym}: skip day ({candle_dt.strftime('%a')} UTC) — skip")
        return

    sma, upper, lower, hw = bollinger(s["buf_5m"], cfg["period"], cfg["mult"])
    if sma is None:
        return

    bw = band_width(sma, hw)
    s["bw_history"].append(bw)
    if len(s["bw_history"]) > SQUEEZE_LOOKBACK:
        s["bw_history"] = s["bw_history"][-SQUEEZE_LOOKBACK:]

    bw_avg     = sum(s["bw_history"]) / len(s["bw_history"])
    squeeze_ok = bw <= bw_avg * SQUEEZE_THRESH
    price      = completed_5m["close"]

    log.info(f"  {sym}  5m_close={price:.4f}  sma={sma:.4f}  "
             f"upper={upper:.4f}  lower={lower:.4f}  "
             f"bw={bw*100:.3f}%  squeeze={'OK' if squeeze_ok else 'SKIP'}  "
             f"state={s['state']}")

    if price > upper and squeeze_ok and s["long_blocked"] == 0:
        # Momentum LONG: price broke above upper band
        s["pending_side"] = "LONG"
        s["pending_hw"]   = hw
        s["state"]        = "ENTRY_PENDING"
        s["arm_id"]       = arm_log.new_arm_id()
        s["arm_time"]     = now_utc().strftime("%Y-%m-%d %H:%M:%S")
        s["arm_price"]    = price
        log.info(f"  {sym}: LONG signal  upper={upper:.4f}  hw={hw:.4f}")

    elif price < lower and squeeze_ok and s["short_blocked"] == 0:
        # Momentum SHORT: price broke below lower band
        s["pending_side"] = "SHORT"
        s["pending_hw"]   = hw
        s["state"]        = "ENTRY_PENDING"
        s["arm_id"]       = arm_log.new_arm_id()
        s["arm_time"]     = now_utc().strftime("%Y-%m-%d %H:%M:%S")
        s["arm_price"]    = price
        log.info(f"  {sym}: SHORT signal  lower={lower:.4f}  hw={hw:.4f}")

    else:
        blocked = []
        if s["long_blocked"]:  blocked.append(f"LONG blocked({s['long_blocked']})")
        if s["short_blocked"]: blocked.append(f"SHORT blocked({s['short_blocked']})")
        if not squeeze_ok:     blocked.append("squeeze")
        log.info(f"  {sym}: no signal" + (f"  [{', '.join(blocked)}]" if blocked else ""))


def _enter_pending(client: BitunixClient, sym: str, s: dict,
                   c1: dict, balance: float, debug: bool) -> None:
    """Execute the pending entry at the open of this 1m candle."""
    side = s["pending_side"]
    hw   = s["pending_hw"]

    # Use live ticker for a tighter entry price estimate
    ticker      = fetch_ticker(client, sym)
    entry_price = float(ticker.get("lastPrice", c1["open"]))

    # Anchor TP/SL to actual entry price (not stale band levels from signal candle)
    if side == "LONG":
        tp_price = entry_price + hw * TP_MULT
        sl_price = entry_price - hw * SL_MULT
    else:
        tp_price = entry_price - hw * TP_MULT
        sl_price = entry_price + hw * SL_MULT

    # Fee gate: skip if the TP distance doesn't cover round-trip fees
    tp_dist  = abs(tp_price - entry_price)
    fee_cost = entry_price * ROUND_TRIP_FEE
    if tp_dist <= fee_cost:
        log.info(f"  {sym}: SKIP entry — TP dist {tp_dist:.4f} ≤ fee {fee_cost:.4f}")
        arm_log.log_arm_event(s["arm_id"], STRATEGY, sym, s["arm_time"], s["arm_price"],
                              side, "NO_FIRE", disarm_price=entry_price,
                              no_fire_reason="FEE_GATE")
        s["state"]        = "WATCHING"
        s["pending_side"] = s["pending_hw"] = None
        s["arm_id"] = s["arm_time"] = s["arm_price"] = None
        return

    notional = balance * MAX_TRADE_PCT * LEVERAGE
    qty      = round_qty(sym, notional / entry_price)
    min_qty  = {"BTCUSDT": 0.0001, "ETHUSDT": 0.003}.get(sym, 0.001)
    if qty < min_qty:
        log.info(f"  {sym}: qty {qty} < min {min_qty}, skip")
        arm_log.log_arm_event(s["arm_id"], STRATEGY, sym, s["arm_time"], s["arm_price"],
                              side, "NO_FIRE", disarm_price=entry_price,
                              no_fire_reason="MIN_QTY")
        s["state"]        = "WATCHING"
        s["pending_side"] = s["pending_hw"] = None
        s["arm_id"] = s["arm_time"] = s["arm_price"] = None
        return

    log.info(f"  {sym}: ENTER {side}  entry≈{entry_price:.4f}  "
             f"tp={tp_price:.4f}  sl={sl_price:.4f}  qty={qty}")

    order_id = place_order(client, sym, side, qty, tp_price, sl_price, debug)
    if order_id is None:
        arm_log.log_arm_event(s["arm_id"], STRATEGY, sym, s["arm_time"], s["arm_price"],
                              side, "NO_FIRE", disarm_price=entry_price,
                              no_fire_reason="ORDER_FAILED")
        s["state"]        = "WATCHING"
        s["pending_side"] = s["pending_hw"] = None
        s["arm_id"] = s["arm_time"] = s["arm_price"] = None
        return

    arm_log.log_arm_event(s["arm_id"], STRATEGY, sym, s["arm_time"], s["arm_price"],
                          side, "FIRED", disarm_price=entry_price)
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
        "position_id": order_id,
        "side":        side,
        "entry_price": entry_price,
        "tp_price":    tp_price,
        "sl_price":    sl_price,
        "qty":         qty,
        "opened_at":   now_utc(),
        "debug":       debug,
    }
    s["pending_side"] = s["pending_hw"] = None
    s["arm_id"] = s["arm_time"] = s["arm_price"] = None


def _monitor_trade(client: BitunixClient, sym: str, s: dict,
                   c1: dict, debug: bool) -> None:
    """Check if the open position has resolved; handle time exit."""
    pos       = s["position"]
    held_mins = (now_utc() - pos["opened_at"]).total_seconds() / 60
    side      = pos["side"]
    tp        = pos["tp_price"]
    sl        = pos["sl_price"]

    if debug:
        price  = c1["close"]
        tp_hit = (c1["high"] >= tp) if side == "LONG" else (c1["low"]  <= tp)
        sl_hit = (c1["low"]  <= sl) if side == "LONG" else (c1["high"] >= sl)
        pnl    = (price - pos["entry_price"]) * pos["qty"]
        if side == "SHORT":
            pnl = -pnl

        log.info(f"  {sym} [DEBUG {side}]  entry={pos['entry_price']:.4f}  "
                 f"now={price:.4f}  uPnL≈{pnl:+.4f}  held={held_mins:.1f}min  "
                 f"tp={tp:.4f}  sl={sl:.4f}")

        if tp_hit and sl_hit:
            outcome = "TP" if abs(tp - pos["entry_price"]) <= abs(sl - pos["entry_price"]) else "SL"
        elif tp_hit:
            outcome = "TP"
        elif sl_hit:
            outcome = "SL"
        elif held_mins >= MAX_HOLD_MINS:
            outcome = "TIME"
        else:
            outcome = None

        if outcome:
            log.info(f"  {sym} [DEBUG]: {outcome} exit  held={held_mins:.1f}min")
            if outcome == "SL":
                if side == "LONG":  s["long_blocked"]  = COOLDOWN_5M
                else:               s["short_blocked"] = COOLDOWN_5M
            s["state"]    = "WATCHING"
            s["position"] = None
        return

    # Live: check exchange (TP/SL are inline on the order)
    live = [p for p in get_open_positions(client, sym)
            if p.get("positionId") == pos["position_id"]]

    if not live:
        log.info(f"  {sym}: position closed by exchange (TP/SL hit)")
        s["state"]    = "WATCHING"
        s["position"] = None
        return

    p    = live[0]
    upnl = float(p.get("unrealizedPNL", 0))
    log.info(f"  {sym} [{side}]  entry={pos['entry_price']:.4f}  "
             f"uPnL={upnl:+.4f}  held={held_mins:.1f}min  "
             f"tp={tp:.4f}  sl={sl:.4f}")

    if held_mins >= MAX_HOLD_MINS:
        log.info(f"  {sym}: time exit after {held_mins:.1f}min")
        if close_position(client, pos["position_id"], sym, debug):
            s["state"]    = "WATCHING"
            s["position"] = None


def _log_state(sym: str, s: dict) -> None:
    log.info(f"  {sym}  state={s['state']}  "
             f"5m_buf={len(s['buf_5m'])}  "
             f"long_blocked={s['long_blocked']}  "
             f"short_blocked={s['short_blocked']}")


# ── Main loop ──────────────────────────────────────────────────────────────────

def run(debug: bool, symbols: list[str] = None) -> None:
    client = BitunixClient(API_KEY, SECRET_KEY)
    active = symbols or SYMBOLS

    log.info("━" * 60)
    log.info("  BB Momentum Trader")
    log.info(f"  Symbols   : {', '.join(active)}")
    log.info(f"  TP/SL     : {TP_MULT}/{SL_MULT}× half-band-width")
    log.info(f"  Squeeze   : threshold={SQUEEZE_THRESH}  lookback={SQUEEZE_LOOKBACK}")
    log.info(f"  Cooldown  : {COOLDOWN_5M} 5m-candles after SL")
    log.info(f"  Hold      : ≤{MAX_HOLD_MINS}min  |  Leverage: {LEVERAGE}×")
    log.info(f"  Max trade : {MAX_TRADE_PCT:.0%}  |  Fees: {ROUND_TRIP_FEE*100:.3f}%")
    if debug:
        log.info("  MODE      : DEBUG — no real orders")
    log.info("━" * 60)

    _day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    for sym in active:
        cfg     = SYMBOL_CONFIGS.get(sym, {})
        windows = cfg.get("windows")
        if windows:
            hrs = "  ".join(f"{sh:02d}:00–{eh:02d}:00" for sh, eh in windows) + " UTC"
        else:
            hrs = "all hours"
        skip = cfg.get("skip_days", set())
        days = ", ".join(_day_names[d] for d in sorted(skip)) if skip else "all days"
        log.info(f"  {sym}  period={cfg.get('period')}  mult={cfg.get('mult')}×  "
                 f"hours={hrs}  skip={days}")

    for sym in active:
        set_leverage(client, sym, debug)

    start_balance = get_balance(client)
    min_balance   = start_balance * MIN_BALANCE_PCT
    log.info(f"  Balance: {start_balance:.4f} USDT  |  Floor: {min_balance:.4f}")

    # ── Initialise per-symbol state ────────────────────────────────────────────
    sym_state: dict[str, dict] = {}
    for sym in active:
        cfg = SYMBOL_CONFIGS.get(sym, {"period": 20, "mult": 1.5})

        # 1. Load full local CSV history for accurate bw_avg calibration
        csv_candles = load_candles_csv(sym)
        if csv_candles:
            log.info(f"  {sym}: {len(csv_candles):,} candles from local CSV  "
                     f"({datetime.fromtimestamp(csv_candles[0]['time']/1000, tz=timezone.utc).strftime('%Y-%m-%d')} "
                     f"→ {datetime.fromtimestamp(csv_candles[-1]['time']/1000, tz=timezone.utc).strftime('%Y-%m-%d')})")
            # 2. Gap-fill from CSV end to now via API
            log.info(f"  {sym}: fetching gap candles from API...")
            gap_candles = fetch_gap_candles(client, sym, csv_candles[-1]["time"])
            log.info(f"  {sym}: {len(gap_candles)} gap candles fetched")
            candles_1m = csv_candles + gap_candles
        else:
            # No local CSV — fall back to API pages
            log.info(f"  {sym}: no local CSV found, fetching from API...")
            candles_1m = fetch_candles_paged(client, sym, pages=2)
            log.info(f"  {sym}: {len(candles_1m)} candles fetched from API")

        # 3. Build 5m buffer and bw_history from full dataset
        buf_5m, current_5m, current_bucket = build_5m_buf(candles_1m)

        bw_history = []
        for i in range(cfg["period"], len(buf_5m)):
            sma, _, _, hw = bollinger(buf_5m[:i+1], cfg["period"], cfg["mult"])
            if sma:
                bw_history.append(band_width(sma, hw))
        bw_history = bw_history[-SQUEEZE_LOOKBACK:]

        # 4. Trim buf_5m for memory — keep enough for BB calculation
        buf_5m = buf_5m[-(max(cfg["period"], SQUEEZE_LOOKBACK) + 50):]

        sym_state[sym] = {
            "buf_5m":         buf_5m,
            "bw_history":     bw_history,
            "current_5m":     current_5m,
            "current_bucket": current_bucket,
            "last_ts":        candles_1m[-1]["time"] if candles_1m else 0,
            # State machine
            "state":          "WATCHING",
            "pending_side":   None,
            "pending_hw":     None,
            "position":       None,
            # Directional cooldowns (in 5m candle counts)
            "long_blocked":   0,
            "short_blocked":  0,
            # Arm event tracking
            "arm_id":         None,
            "arm_time":       None,
            "arm_price":      None,
        }
        log.info(f"  {sym}: {len(buf_5m)} 5m candles in buf  "
                 f"bw_history={len(bw_history)} entries — ready")

    # ── Hydrate open positions from exchange ───────────────────────────────────
    if not debug:
        for p in get_open_positions(client):
            sym = p.get("symbol")
            if sym not in active:
                continue
            side = "LONG" if p.get("side", "").upper() == "BUY" else "SHORT"
            sym_state[sym]["state"]    = "IN_TRADE"
            sym_state[sym]["position"] = {
                "position_id": p.get("positionId"),
                "side":        side,
                "entry_price": float(p.get("avgOpenPrice", 0)),
                "tp_price":    None,
                "sl_price":    None,
                "qty":         float(p.get("qty", 0)),
                "opened_at":   now_utc(),
                "debug":       False,
            }
            log.info(f"  Hydrated {sym} [{side}] @ "
                     f"{sym_state[sym]['position']['entry_price']:.4f}")

    # ── Main cycle ─────────────────────────────────────────────────────────────
    while True:
        try:
            cycle_start = now_utc()
            balance     = get_balance(client)
            log.info(f"  Balance: {balance:.4f}  |  {cycle_start.strftime('%H:%M:%S')} UTC")

            if balance < min_balance:
                log.warning(f"  CIRCUIT BREAKER: {balance:.4f} < floor "
                            f"{min_balance:.4f} — halting")
                break

            for sym in active:
                try:
                    _process_symbol(client, sym, sym_state[sym], balance, debug)
                except Exception as e:
                    log.error(f"  {sym} error: {e}")

        except Exception as e:
            log.error(f"  Cycle error: {e}")

        elapsed = (now_utc() - cycle_start).total_seconds()
        time.sleep(max(5, POLL_SECS - elapsed))


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bollinger Band momentum perpetual futures trader")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode — no real orders placed")
    parser.add_argument("--symbol", nargs="+", metavar="SYMBOL",
                        help=f"Symbol(s) to trade (default: {', '.join(SYMBOLS)})")
    args = parser.parse_args()
    run(debug=args.debug, symbols=args.symbol)


if __name__ == "__main__":
    main()
