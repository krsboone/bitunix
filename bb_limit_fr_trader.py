"""
bb_limit_fr_trader.py — BB Momentum trader with funding-rate filter

Same as bb_limit_trader.py with one addition:
  - Funding rate is fetched each cycle and used as a directional gate.
  - Positive funding (longs paying) → LONG signals skipped, SHORT allowed.
  - Negative funding (shorts paying) → SHORT signals skipped, LONG allowed.
  - Neutral (within ±FUNDING_THRESH) → both directions trade normally.

Rationale: 205 days of data show that when funding is positive, BTC/ETH
drift down over the next 15m–4h (avg −0.033% to −0.170%). Filtering
counter-funding signals is intended to lift the TP rate above the
bb_limit baseline (~55.5%).

State machine:
  WATCHING → ENTRY_PENDING → LIMIT_OPEN → IN_TRADE → WATCHING

Usage:
    python3 bb_limit_fr_trader.py              # live trading
    python3 bb_limit_fr_trader.py --debug      # DEBUG mode — no real orders
    python3 bb_limit_fr_trader.py --symbol BTCUSDT
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
import position_registry

start_logging("bb_limit_fr")

STRATEGY = "bb_limit_fr"

# ── Per-symbol sweep-optimised config (same as bb_trader.py) ──────────────────

SYMBOL_CONFIGS = {
    "BTCUSDT": {
        "period":    30,
        "mult":      1.5,
        #"windows":   [(11, 13)],
        "windows":   [],
        "skip_days": {5, 6},
    },
    "ETHUSDT": {
        "period":    20,
        "mult":      1.5,
        #"windows":   [(9, 11), (14, 17)],
        "windows":   [],
        "skip_days": {5, 6},
    },
}
SYMBOLS = list(SYMBOL_CONFIGS.keys())

# ── Shared strategy parameters ─────────────────────────────────────────────────

TP_MULT          = 2.0
SL_MULT          = 2.0
SQUEEZE_THRESH   = 1.0
SQUEEZE_LOOKBACK = 50
COOLDOWN_5M      = 5
MAX_HOLD_MINS    = 60

# ── Limit order parameters ─────────────────────────────────────────────────────

LIMIT_ENTRY_TIMEOUT_MINS = 5      # cancel entry limit if unfilled after this long
LIMIT_CANCEL_DRIFT_PCT   = 0.003  # cancel if price drifts 0.3% against signal direction

FIVE_MIN_MS      = 5 * 60 * 1000
MIN_HISTORY_5M   = max(50, SQUEEZE_LOOKBACK) + 10

# ── Trading parameters ─────────────────────────────────────────────────────────

# ── Funding rate filter ────────────────────────────────────────────────────────

# Signals whose direction opposes the funding bias are skipped.
# Set to 0.0 to disable (trade all signals regardless of funding).
FUNDING_THRESH        = 0.00005   # 0.005% — below this is treated as neutral
FUNDING_REFRESH_SECS  = 900       # re-fetch funding every 15 minutes

# ── Trading parameters ─────────────────────────────────────────────────────────

LEVERAGE         = 2
MARGIN_COIN      = "USDT"
INTERVAL         = "1m"
MAX_TRADE_PCT    = 0.10
POLL_SECS        = 60

FEE_TAKER        = 0.00060   # taker rate — SL exits (confirm with Bitunix fee schedule)
FEE_MAKER        = 0.00020   # maker rate — limit entry + limit TP (confirm with Bitunix fee schedule)
ROUND_TRIP_FEE   = FEE_MAKER * 2   # fee gate uses best case: both legs maker
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

TRADE_CSV = os.path.join("log", "bb_limit_fr_trades.csv")


# ── Funding rate helpers ───────────────────────────────────────────────────────

def fetch_funding_rates(client: BitunixClient) -> dict[str, float]:
    """Return {symbol: most_recently_settled_funding_rate}.

    Uses the history endpoint (settled rates) rather than the batch endpoint
    (live pre-settlement rate) — the backtest was built on settled rates and
    the live rate fluctuates too wildly to be a reliable filter signal.
    """
    rates = {}
    for symbol in SYMBOLS:
        try:
            resp = client.get("/api/v1/futures/market/get_funding_rate_history",
                              {"symbol": symbol, "limit": "1"})
            if resp.get("code") != 0:
                log.warning(f"  Funding history fetch failed {symbol}: {resp.get('msg')}")
                continue
            data = resp.get("data") or []
            if data:
                rates[symbol] = float(data[0]["fundingRate"])
        except Exception as e:
            log.warning(f"  Funding rate fetch error {symbol}: {e}")
    return rates


def funding_allows(signal_side: str, funding_rate: float | None) -> tuple[bool, str]:
    """
    Return (allowed, reason_str).
    Positive funding → longs pay shorts → skip LONG, allow SHORT.
    Negative funding → shorts pay longs → skip SHORT, allow LONG.
    Neutral or unknown → allow both.
    """
    if funding_rate is None:
        return True, "unknown"
    if funding_rate > FUNDING_THRESH and signal_side == "LONG":
        return False, f"funding={funding_rate*100:+.4f}% (positive — skip LONG)"
    if funding_rate < -FUNDING_THRESH and signal_side == "SHORT":
        return False, f"funding={funding_rate*100:+.4f}% (negative — skip SHORT)"
    bias = "neutral" if abs(funding_rate) <= FUNDING_THRESH else (
        "aligned-SHORT" if funding_rate > 0 else "aligned-LONG")
    return True, f"funding={funding_rate*100:+.4f}% ({bias})"


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


def get_closed_position(client: BitunixClient, symbol: str, position_id: str) -> dict | None:
    try:
        resp = client.get("/api/v1/futures/position/get_history_positions",
                          {"symbol": symbol, "limit": "20"})
        if resp.get("code") != 0:
            return None
        for p in resp.get("data", {}).get("positionList", []):
            if p.get("positionId") == position_id:
                return p
    except Exception:
        pass
    return None


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
        "orderType",   "LIMIT",
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


def place_limit_entry(client: BitunixClient, symbol: str, side: str,
                      qty: float, limit_price: float, sl_price: float,
                      debug: bool) -> str | None:
    """Limit OPEN order with inline stop-market SL. No inline TP — managed separately."""
    buy_sell = "BUY" if side == "LONG" else "SELL"
    body = {
        "symbol":     symbol,
        "qty":        str(qty),
        "side":       buy_sell,
        "orderType":  "LIMIT",
        "price":      str(round_price(symbol, limit_price)),
        "tradeSide":  "OPEN",
        "slPrice":    str(round_price(symbol, sl_price)),
        "slStopType": "MARK_PRICE",
    }
    if debug:
        log.info(f"  [DEBUG] Would place limit entry: {json.dumps(body)}")
        return "DEBUG-LIMIT-ID"
    resp = client.post("/api/v1/futures/trade/place_order", body)
    if resp.get("code") != 0:
        log.error(f"  Limit entry failed: {resp.get('msg')}")
        return None
    order_id = resp.get("data", {}).get("orderId")
    log.info(f"  Limit entry placed: {order_id} @ {limit_price}")
    return order_id


def cancel_order(client: BitunixClient, order_id: str, symbol: str,
                 debug: bool) -> bool:
    if debug:
        log.info(f"  [DEBUG] Would cancel order {order_id}")
        return True
    resp = client.post("/api/v1/futures/trade/cancel_orders", {
        "symbol":      symbol,
        "orderIdList": [order_id],
    })
    if resp.get("code") != 0:
        log.error(f"  Cancel failed {order_id}: {resp.get('msg')}")
        return False
    log.info(f"  Order {order_id} cancelled")
    return True


def get_order_status(client: BitunixClient, order_id: str, symbol: str) -> str | None:
    """Returns order status string or None on error."""
    try:
        resp = client.get("/api/v1/futures/trade/get_order_detail", {
            "symbol":  symbol,
            "orderId": order_id,
        })
        if resp.get("code") != 0:
            return None
        return resp.get("data", {}).get("status")
    except Exception:
        return None


def place_limit_tp(client: BitunixClient, symbol: str, side: str,
                   qty: float, tp_price: float, debug: bool) -> str | None:
    """Limit CLOSE order at the TP price."""
    close_side = "SELL" if side == "LONG" else "BUY"
    body = {
        "symbol":    symbol,
        "qty":       str(qty),
        "side":      close_side,
        "orderType": "LIMIT",
        "price":     str(round_price(symbol, tp_price)),
        "tradeSide": "CLOSE",
    }
    if debug:
        log.info(f"  [DEBUG] Would place limit TP: {json.dumps(body)}")
        return "DEBUG-TP-ID"
    resp = client.post("/api/v1/futures/trade/place_order", body)
    if resp.get("code") != 0:
        log.error(f"  Limit TP failed: {resp.get('msg')}")
        return None
    order_id = resp.get("data", {}).get("orderId")
    log.info(f"  Limit TP placed: {order_id} @ {tp_price}")
    return order_id


def resolve_position_id(client: BitunixClient, symbol: str, side: str,
                        timeout: int = 10) -> str | None:
    api_side = "BUY" if side == "LONG" else "SELL"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            for p in get_open_positions(client, symbol):
                if (p.get("symbol") == symbol and
                        p.get("side", "").upper() == api_side):
                    return p.get("positionId")
        except Exception:
            pass
        time.sleep(1)
    return None


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


def _clear_pending(s: dict) -> None:
    """Clear all pending/limit fields after fill or cancel."""
    s["pending_side"]    = None
    s["pending_hw"]      = None
    s["pending_qty"]     = None
    s["limit_order_id"]  = None
    s["limit_price"]     = None
    s["limit_placed_at"] = None
    s["arm_id"]          = None
    s["arm_time"]        = None
    s["arm_price"]       = None


# ── Per-symbol processing ──────────────────────────────────────────────────────

def _process_symbol(client: BitunixClient, sym: str, s: dict,
                    balance: float, debug: bool) -> None:
    cfg = SYMBOL_CONFIGS.get(sym, {"period": 20, "mult": 1.5})

    # ── 1. Fetch new 1m candles ────────────────────────────────────────────────
    recent = fetch_latest_candles(client, sym, n=10)
    new    = [c for c in recent if c["time"] > s["last_ts"]]
    new.sort(key=lambda c: c["time"])

    if not new:
        log.info(f"  {sym}: no new candle yet")
        _log_state(sym, s)
        return

    # ── 2. Update 5m state ────────────────────────────────────────────────────
    completed_5m = None
    for c1 in new:
        bucket = (c1["time"] // FIVE_MIN_MS) * FIVE_MIN_MS

        if s["current_bucket"] is None:
            s["current_bucket"] = bucket
            s["current_5m"]     = _start_5m(c1)
        elif bucket != s["current_bucket"]:
            completed_5m = s["current_5m"]
            s["buf_5m"].append(completed_5m)
            if len(s["buf_5m"]) > 500:
                s["buf_5m"] = s["buf_5m"][-500:]
            s["current_bucket"] = bucket
            s["current_5m"]     = _start_5m(c1)
            if s["long_blocked"]  > 0: s["long_blocked"]  -= 1
            if s["short_blocked"] > 0: s["short_blocked"] -= 1
        else:
            _update_5m(s["current_5m"], c1)

        s["last_ts"] = c1["time"]

    last_1m = new[-1]

    # ── 3. IN_TRADE: monitor open position ────────────────────────────────────
    if s["state"] == "IN_TRADE":
        _monitor_trade(client, sym, s, last_1m, debug)
        return

    # ── 4. LIMIT_OPEN: monitor pending limit entry ────────────────────────────
    if s["state"] == "LIMIT_OPEN":
        _monitor_limit_order(client, sym, s, last_1m, debug)
        return

    # ── 5. ENTRY_PENDING: place limit order at open of next 1m candle ─────────
    if s["state"] == "ENTRY_PENDING":
        _enter_pending(client, sym, s, last_1m, balance, debug)
        return

    # ── 6. WATCHING: check for BB signal on newly completed 5m candle ─────────
    if completed_5m is None or len(s["buf_5m"]) < MIN_HISTORY_5M:
        log.info(f"  {sym}: warming up ({len(s['buf_5m'])}/{MIN_HISTORY_5M} 5m candles)")
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

    signal_side = None
    if price > upper and squeeze_ok and s["long_blocked"] == 0:
        signal_side = "LONG"
    elif price < lower and squeeze_ok and s["short_blocked"] == 0:
        signal_side = "SHORT"

    candle_dt   = datetime.fromtimestamp(completed_5m["time"] / 1000, tz=timezone.utc)
    candle_hour = candle_dt.hour
    candle_dow  = candle_dt.weekday()
    windows     = cfg.get("windows")
    in_window   = True
    if windows:
        in_window = False
        for sh, eh in windows:
            if sh <= eh:
                if sh <= candle_hour < eh:
                    in_window = True; break
            else:
                if candle_hour >= sh or candle_hour < eh:
                    in_window = True; break
    skip_day = candle_dow in cfg.get("skip_days", set())

    if not in_window or skip_day:
        if signal_side:
            if signal_side == "LONG":
                wtp = price + hw * TP_MULT
                wsl = price - hw * SL_MULT
            else:
                wtp = price - hw * TP_MULT
                wsl = price + hw * SL_MULT
            arm_log.log_arm_event(
                arm_log.new_arm_id(), STRATEGY, sym,
                now_utc().strftime("%Y-%m-%d %H:%M:%S"), price, signal_side,
                "PENDING", shadow=True, atr=hw, would_be_tp=wtp, would_be_sl=wsl,
            )
            log.info(f"  {sym}: shadow {signal_side} signal "
                     f"({candle_dt.strftime('%a %H:%M')} UTC outside window)")
        if not in_window:
            log.info(f"  {sym}: outside trade window ({candle_dt.strftime('%a %H:%M')} UTC) — skip")
        else:
            log.info(f"  {sym}: skip day ({candle_dt.strftime('%a')} UTC) — skip")
        return

    log.info(f"  {sym}  5m_close={price:.4f}  sma={sma:.4f}  "
             f"upper={upper:.4f}  lower={lower:.4f}  "
             f"bw={bw*100:.3f}%  squeeze={'OK' if squeeze_ok else 'SKIP'}  "
             f"state={s['state']}")

    if signal_side == "LONG":
        fr       = s.get("funding_rate")
        allowed, fr_reason = funding_allows("LONG", fr)
        if not allowed:
            log.info(f"  {sym}: LONG signal FILTERED — {fr_reason}")
            arm_log.log_arm_event(
                arm_log.new_arm_id(), STRATEGY, sym,
                now_utc().strftime("%Y-%m-%d %H:%M:%S"), price, "LONG",
                "NO_FIRE", no_fire_reason="FUNDING_FILTER", atr=hw,
            )
        else:
            log.info(f"  {sym}: LONG signal  upper={upper:.4f}  hw={hw:.4f}  {fr_reason}")
            s["pending_side"] = "LONG"
            s["pending_hw"]   = hw
            s["state"]        = "ENTRY_PENDING"
            s["arm_id"]       = arm_log.new_arm_id()
            s["arm_time"]     = now_utc().strftime("%Y-%m-%d %H:%M:%S")
            s["arm_price"]    = price

    elif signal_side == "SHORT":
        fr       = s.get("funding_rate")
        allowed, fr_reason = funding_allows("SHORT", fr)
        if not allowed:
            log.info(f"  {sym}: SHORT signal FILTERED — {fr_reason}")
            arm_log.log_arm_event(
                arm_log.new_arm_id(), STRATEGY, sym,
                now_utc().strftime("%Y-%m-%d %H:%M:%S"), price, "SHORT",
                "NO_FIRE", no_fire_reason="FUNDING_FILTER", atr=hw,
            )
        else:
            log.info(f"  {sym}: SHORT signal  lower={lower:.4f}  hw={hw:.4f}  {fr_reason}")
            s["pending_side"] = "SHORT"
            s["pending_hw"]   = hw
            s["state"]        = "ENTRY_PENDING"
            s["arm_id"]       = arm_log.new_arm_id()
            s["arm_time"]     = now_utc().strftime("%Y-%m-%d %H:%M:%S")
            s["arm_price"]    = price

    else:
        blocked = []
        if s["long_blocked"]:  blocked.append(f"LONG blocked({s['long_blocked']})")
        if s["short_blocked"]: blocked.append(f"SHORT blocked({s['short_blocked']})")
        if not squeeze_ok:     blocked.append("squeeze")
        log.info(f"  {sym}: no signal" + (f"  [{', '.join(blocked)}]" if blocked else ""))


def _enter_pending(client: BitunixClient, sym: str, s: dict,
                   c1: dict, balance: float, debug: bool) -> None:
    """Place a limit entry order at the current decision price."""
    side = s["pending_side"]
    hw   = s["pending_hw"]

    ticker      = fetch_ticker(client, sym)
    limit_price = float(ticker.get("lastPrice", c1["open"]))

    if side == "LONG":
        tp_price = limit_price + hw * TP_MULT
        sl_price = limit_price - hw * SL_MULT
    else:
        tp_price = limit_price - hw * TP_MULT
        sl_price = limit_price + hw * SL_MULT

    tp_dist  = abs(tp_price - limit_price)
    fee_cost = limit_price * ROUND_TRIP_FEE
    if tp_dist <= fee_cost:
        log.info(f"  {sym}: SKIP entry — TP dist {tp_dist:.4f} ≤ fee {fee_cost:.4f}")
        arm_log.log_arm_event(s["arm_id"], STRATEGY, sym, s["arm_time"], s["arm_price"],
                              side, "NO_FIRE", disarm_price=limit_price,
                              no_fire_reason="FEE_GATE", atr=s["pending_hw"])
        s["state"] = "WATCHING"
        _clear_pending(s)
        return

    notional = balance * MAX_TRADE_PCT * LEVERAGE
    qty      = round_qty(sym, notional / limit_price)
    min_qty  = {"BTCUSDT": 0.0001, "ETHUSDT": 0.003}.get(sym, 0.001)
    if qty < min_qty:
        log.info(f"  {sym}: qty {qty} < min {min_qty}, skip")
        arm_log.log_arm_event(s["arm_id"], STRATEGY, sym, s["arm_time"], s["arm_price"],
                              side, "NO_FIRE", disarm_price=limit_price,
                              no_fire_reason="MIN_QTY", atr=s["pending_hw"])
        s["state"] = "WATCHING"
        _clear_pending(s)
        return

    log.info(f"  {sym}: placing LIMIT entry [{side}] @ {limit_price:.4f}  "
             f"sl={sl_price:.4f}  tp_target={tp_price:.4f}  qty={qty}  "
             f"timeout={LIMIT_ENTRY_TIMEOUT_MINS}min  drift={LIMIT_CANCEL_DRIFT_PCT*100:.1f}%")

    order_id = place_limit_entry(client, sym, side, qty, limit_price, sl_price, debug)
    if order_id is None:
        arm_log.log_arm_event(s["arm_id"], STRATEGY, sym, s["arm_time"], s["arm_price"],
                              side, "NO_FIRE", disarm_price=limit_price,
                              no_fire_reason="ORDER_FAILED", atr=s["pending_hw"])
        s["state"] = "WATCHING"
        _clear_pending(s)
        return

    s["state"]           = "LIMIT_OPEN"
    s["pending_qty"]     = qty
    s["pending_hw"]      = hw
    s["limit_order_id"]  = order_id
    s["limit_price"]     = limit_price
    s["limit_placed_at"] = now_utc()
    # pending_side, arm_id, arm_time, arm_price preserved for use in _monitor_limit_order


def _monitor_limit_order(client: BitunixClient, sym: str, s: dict,
                         last_1m: dict, debug: bool) -> None:
    """Poll the pending limit entry: cancel on timeout/drift, transition on fill."""
    side          = s["pending_side"]
    hw            = s["pending_hw"]
    qty           = s["pending_qty"]
    limit_price   = s["limit_price"]
    limit_order_id = s["limit_order_id"]
    elapsed_mins  = (now_utc() - s["limit_placed_at"]).total_seconds() / 60

    try:
        ticker        = fetch_ticker(client, sym)
        current_price = float(ticker.get("lastPrice", last_1m["close"]))
    except Exception:
        current_price = last_1m["close"]

    # Drift check: if price moves against signal direction beyond threshold, cancel
    drift_cancel = False
    if side == "LONG":
        if current_price < limit_price * (1 - LIMIT_CANCEL_DRIFT_PCT):
            drift_cancel = True
    else:
        if current_price > limit_price * (1 + LIMIT_CANCEL_DRIFT_PCT):
            drift_cancel = True

    timeout_cancel = elapsed_mins >= LIMIT_ENTRY_TIMEOUT_MINS

    if drift_cancel or timeout_cancel:
        reason = "DRIFT" if drift_cancel else "TIMEOUT"
        log.info(f"  {sym}: limit entry cancelled ({reason})  "
                 f"[{side}] limit={limit_price:.4f}  current={current_price:.4f}  "
                 f"elapsed={elapsed_mins:.1f}min")
        cancel_order(client, limit_order_id, sym, debug)
        arm_log.log_arm_event(
            s["arm_id"], STRATEGY, sym, s["arm_time"], s["arm_price"],
            side, "NO_FIRE", disarm_price=current_price,
            no_fire_reason=f"LIMIT_{reason}", atr=hw,
        )
        s["state"] = "WATCHING"
        _clear_pending(s)
        return

    # Check fill status
    if debug:
        # In debug mode: simulate fill immediately (first cycle after placement)
        filled = True
    else:
        status = get_order_status(client, limit_order_id, sym)
        filled = status in ("FILLED", "COMPLETELY_FILLED")

    if not filled:
        log.info(f"  {sym}: limit entry pending [{side}] @ {limit_price:.4f}  "
                 f"current={current_price:.4f}  elapsed={elapsed_mins:.1f}min")
        return

    # Filled — compute TP anchored to actual fill price (= limit_price for a limit order)
    entry_price = limit_price
    if side == "LONG":
        tp_price = entry_price + hw * TP_MULT
        sl_price = entry_price - hw * SL_MULT
    else:
        tp_price = entry_price - hw * TP_MULT
        sl_price = entry_price + hw * SL_MULT

    # Resolve positionId
    if debug:
        position_id = limit_order_id
    else:
        position_id = resolve_position_id(client, sym, side)
        if position_id is None:
            log.warning(f"  {sym}: could not resolve positionId — falling back to orderId")
            position_id = limit_order_id
        else:
            log.info(f"  {sym}: resolved positionId={position_id}")

    # Place limit TP close order
    tp_order_id = place_limit_tp(client, sym, side, qty, tp_price, debug)

    log.info(f"  {sym}: limit FILLED [{side}] @ {entry_price:.4f}  "
             f"tp_order={tp_order_id} @ {tp_price:.4f}  sl={sl_price:.4f}")

    arm_log.log_arm_event(
        s["arm_id"], STRATEGY, sym, s["arm_time"], s["arm_price"],
        side, "FIRED", disarm_price=entry_price, atr=hw,
    )
    log_trade({
        "symbol":       sym,
        "side":         side,
        "qty":          qty,
        "tp_price":     round_price(sym, tp_price),
        "sl_price":     round_price(sym, sl_price),
        "entry_price":  entry_price,
        "arm_id":       s["arm_id"],
        "arm_price":    s["arm_price"],
        "signal_price": s["arm_price"],
    })

    s["state"]    = "IN_TRADE"
    s["position"] = {
        "position_id": position_id,
        "side":        side,
        "entry_price": entry_price,
        "tp_price":    tp_price,
        "sl_price":    sl_price,
        "qty":         qty,
        "opened_at":   now_utc(),
        "debug":       debug,
        "arm_id":      s["arm_id"],
        "tp_order_id": tp_order_id,
    }
    position_registry.register(position_id, STRATEGY, sym, side, entry_price, s["arm_id"])
    _clear_pending(s)


def _monitor_trade(client: BitunixClient, sym: str, s: dict,
                   c1: dict, debug: bool) -> None:
    """Monitor open position; cancel TP limit on any non-TP close."""
    pos          = s["position"]
    held_mins    = (now_utc() - pos["opened_at"]).total_seconds() / 60
    side         = pos["side"]
    tp           = pos["tp_price"]
    sl           = pos["sl_price"]
    tp_order_id  = pos.get("tp_order_id")

    if debug:
        price  = c1["close"]
        tp_hit = (c1["high"] >= tp) if (side == "LONG"  and tp is not None) else \
                 (c1["low"]  <= tp) if (side == "SHORT" and tp is not None) else False
        sl_hit = (c1["low"]  <= sl) if (side == "LONG"  and sl is not None) else \
                 (c1["high"] >= sl) if (side == "SHORT" and sl is not None) else False
        pnl    = (price - pos["entry_price"]) * pos["qty"]
        if side == "SHORT":
            pnl = -pnl
        tp_str = f"{tp:.4f}" if tp is not None else "?"
        sl_str = f"{sl:.4f}" if sl is not None else "?"
        log.info(f"  {sym} [DEBUG {side}]  entry={pos['entry_price']:.4f}  "
                 f"now={price:.4f}  uPnL≈{pnl:+.4f}  held={held_mins:.1f}min  "
                 f"tp={tp_str}  sl={sl_str}")

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
            exit_price = tp if outcome == "TP" else (sl if outcome == "SL" else price)
            log.info(f"  {sym} [DEBUG]: {outcome} exit @ {exit_price:.4f}  held={held_mins:.1f}min")
            arm_log.log_close_event(
                pos.get("arm_id", ""), STRATEGY, sym, side,
                outcome, exit_price, held_mins,
                arm_log.calc_pnl(side, pos["entry_price"], exit_price, pos["qty"],
                                 ROUND_TRIP_FEE if outcome == "TP" else FEE_MAKER + FEE_TAKER),
                order_id=pos.get("position_id"),
            )
            position_registry.release(pos.get("position_id", ""))
            if outcome == "SL":
                if side == "LONG":  s["long_blocked"]  = COOLDOWN_5M
                else:               s["short_blocked"] = COOLDOWN_5M
            s["state"]    = "WATCHING"
            s["position"] = None
        return

    # ── Live: check exchange ───────────────────────────────────────────────────
    live = [p for p in get_open_positions(client, sym)
            if p.get("positionId") == pos["position_id"]]

    if not live:
        # Position is gone — determine outcome via TP order status
        tp_filled = False
        if tp_order_id:
            tp_status = get_order_status(client, tp_order_id, sym)
            tp_filled = tp_status in ("FILLED", "COMPLETELY_FILLED")

        hist = get_closed_position(client, sym, pos["position_id"])
        close_px = float(hist.get("closePrice", c1["close"])) if hist else c1["close"]
        if not hist:
            log.warning(f"  {sym}: could not fetch close history, inferring from candle price")

        if tp_filled:
            outcome_str = "TP"
            exit_px     = tp if tp is not None else close_px
        elif tp is not None and sl is not None:
            if side == "LONG":
                outcome_str = "SL" if close_px <= sl else "EXCHANGE_CLOSED"
            else:
                outcome_str = "SL" if close_px >= sl else "EXCHANGE_CLOSED"
            exit_px = sl if outcome_str == "SL" else close_px
        else:
            outcome_str, exit_px = "EXCHANGE_CLOSED", close_px

        # Cancel TP limit order if position closed by SL or exchange (not by TP fill)
        if tp_order_id and not tp_filled:
            cancel_order(client, tp_order_id, sym, debug)

        log.info(f"  {sym}: position closed ({outcome_str}) @ {exit_px:.4f}")
        fee = ROUND_TRIP_FEE if outcome_str == "TP" else FEE_MAKER + FEE_TAKER
        arm_log.log_close_event(
            pos.get("arm_id", ""), STRATEGY, sym, side,
            outcome_str, exit_px, held_mins,
            arm_log.calc_pnl(side, pos["entry_price"], exit_px, pos["qty"], fee),
            order_id=pos.get("position_id"),
        )
        position_registry.release(pos.get("position_id", ""))
        if outcome_str == "SL":
            if side == "LONG":  s["long_blocked"]  = COOLDOWN_5M
            else:               s["short_blocked"] = COOLDOWN_5M
        s["state"]    = "WATCHING"
        s["position"] = None
        return

    p      = live[0]
    upnl   = float(p.get("unrealizedPNL", 0))
    tp_str = f"{tp:.4f}" if tp is not None else "?"
    sl_str = f"{sl:.4f}" if sl is not None else "?"
    log.info(f"  {sym} [{side}]  entry={pos['entry_price']:.4f}  "
             f"uPnL={upnl:+.4f}  held={held_mins:.1f}min  "
             f"tp={tp_str}  sl={sl_str}  tp_order={tp_order_id}")

    if held_mins >= MAX_HOLD_MINS:
        log.info(f"  {sym}: time exit after {held_mins:.1f}min")
        # Cancel TP limit before flash-close to avoid competing close orders
        if tp_order_id:
            cancel_order(client, tp_order_id, sym, debug)
        if close_position(client, pos["position_id"], sym, debug):
            arm_log.log_close_event(
                pos.get("arm_id", ""), STRATEGY, sym, side,
                "TIME", c1["close"], held_mins,
                arm_log.calc_pnl(side, pos["entry_price"], c1["close"], pos["qty"],
                                 FEE_MAKER + FEE_TAKER),
                order_id=pos.get("position_id"),
            )
            position_registry.release(pos.get("position_id", ""))
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
    log.info("  BB Momentum Trader — Limit Orders + Funding Filter")
    log.info(f"  Symbols   : {', '.join(active)}")
    log.info(f"  TP/SL     : {TP_MULT}/{SL_MULT}× half-band-width")
    log.info(f"  Funding   : threshold=±{FUNDING_THRESH*100:.3f}%  refresh={FUNDING_REFRESH_SECS}s")
    log.info(f"  Squeeze   : threshold={SQUEEZE_THRESH}  lookback={SQUEEZE_LOOKBACK}")
    log.info(f"  Cooldown  : {COOLDOWN_5M} 5m-candles after SL")
    log.info(f"  Hold      : ≤{MAX_HOLD_MINS}min  |  Leverage: {LEVERAGE}×")
    log.info(f"  Max trade : {MAX_TRADE_PCT:.0%}  |  Fees: maker={FEE_MAKER*100:.3f}%  taker={FEE_TAKER*100:.3f}%")
    log.info(f"  Limit     : timeout={LIMIT_ENTRY_TIMEOUT_MINS}min  drift={LIMIT_CANCEL_DRIFT_PCT*100:.1f}%")
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
            log.info(f"  {sym}: no local CSV found, fetching from API...")
            candles_1m = fetch_candles_paged(client, sym, pages=2)
            log.info(f"  {sym}: {len(candles_1m)} candles fetched from API")

        buf_5m, current_5m, current_bucket = build_5m_buf(candles_1m)

        bw_history = []
        for i in range(cfg["period"], len(buf_5m)):
            sma, _, _, hw = bollinger(buf_5m[:i+1], cfg["period"], cfg["mult"])
            if sma:
                bw_history.append(band_width(sma, hw))
        bw_history = bw_history[-SQUEEZE_LOOKBACK:]

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
            "pending_qty":    None,
            "position":       None,
            # Limit order tracking
            "limit_order_id":  None,
            "limit_price":     None,
            "limit_placed_at": None,
            # Directional cooldowns
            "long_blocked":   0,
            "short_blocked":  0,
            # Arm event tracking
            "arm_id":         None,
            "arm_time":       None,
            "arm_price":      None,
            # Funding rate cache (refreshed every FUNDING_REFRESH_SECS)
            "funding_rate":   None,
        }
        log.info(f"  {sym}: {len(buf_5m)} 5m candles in buf  "
                 f"bw_history={len(bw_history)} entries — ready")

    # ── Hydrate open positions from exchange ───────────────────────────────────
    if not debug:
        for p in get_open_positions(client):
            sym = p.get("symbol")
            if sym not in active:
                continue
            pid  = p.get("positionId")
            side = "LONG" if p.get("side", "").upper() == "BUY" else "SHORT"
            if not position_registry.owns(pid, STRATEGY):
                log.info(f"  Skipping {sym} position {pid} — not owned by {STRATEGY}")
                continue
            sym_state[sym]["state"]    = "IN_TRADE"
            sym_state[sym]["position"] = {
                "position_id": pid,
                "side":        side,
                "entry_price": float(p.get("avgOpenPrice", 0)),
                "tp_price":    None,
                "sl_price":    None,
                "qty":         float(p.get("qty", 0)),
                "opened_at":   now_utc(),
                "debug":       False,
                "arm_id":      "",
                "tp_order_id": None,
            }
            log.info(f"  Hydrated {sym} [{side}] @ "
                     f"{sym_state[sym]['position']['entry_price']:.4f}  "
                     f"(note: TP limit order unknown after restart — time exit only)")

    # ── Initial funding rate fetch ─────────────────────────────────────────────
    funding_last_fetched = 0.0
    log.info("  Fetching initial funding rates...")
    rates = fetch_funding_rates(client)
    for sym in active:
        sym_state[sym]["funding_rate"] = rates.get(sym)
        fr = sym_state[sym]["funding_rate"]
        log.info(f"  {sym} funding rate: {fr*100:+.4f}%" if fr is not None else
                 f"  {sym} funding rate: unavailable")
    funding_last_fetched = time.time()

    # ── Main cycle ─────────────────────────────────────────────────────────────
    while True:
        try:
            cycle_start = now_utc()
            balance     = get_balance(client)
            log.info(f"  Balance: {balance:.4f}  |  {cycle_start.strftime('%H:%M:%S')} UTC")

            # Refresh funding rates every FUNDING_REFRESH_SECS
            if time.time() - funding_last_fetched >= FUNDING_REFRESH_SECS:
                rates = fetch_funding_rates(client)
                if rates:
                    for sym in active:
                        old = sym_state[sym].get("funding_rate")
                        new = rates.get(sym)
                        sym_state[sym]["funding_rate"] = new
                        if new is not None and old != new:
                            log.info(f"  {sym} funding rate updated: {new*100:+.4f}%")
                    funding_last_fetched = time.time()

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
        description="BB Momentum trader — limit entry + limit TP")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode — no real orders placed")
    parser.add_argument("--symbol", nargs="+", metavar="SYMBOL",
                        help=f"Symbol(s) to trade (default: {', '.join(SYMBOLS)})")
    args = parser.parse_args()
    run(debug=args.debug, symbols=args.symbol)


if __name__ == "__main__":
    main()
