"""
sr_limit_trader.py — S/R Breakout trader: limit entry + limit TP

Same strategy as sr_trader.py. Differences:
  - Entry:  limit order at breakout price  → maker fee on open
  - Cancel: if unfilled after LIMIT_ENTRY_TIMEOUT_MINS, or price drifts
            LIMIT_CANCEL_DRIFT_PCT against the signal direction
  - TP:     separate limit close order placed after entry fills → maker fee on close
  - SL:     stop-market inline on entry order (execution guaranteed)

State machine:
  WATCHING → ARMED → LIMIT_OPEN → IN_TRADE → WATCHING

Usage:
    python3 sr_limit_trader.py              # live trading
    python3 sr_limit_trader.py --debug      # DEBUG mode — no real orders
    python3 sr_limit_trader.py --symbol BTCUSDT
"""

import argparse
import csv
import json
import logging
import math
import os
import time
from datetime import date, datetime, timezone

from auth import BitunixClient
from config import API_KEY, SECRET_KEY
from market import fetch_ticker, compute_sigma
from log_cap import start_logging
import arm_log
import position_registry

start_logging("sr_limit_trader")

STRATEGY = "sr_limit"

# ── Per-symbol sweep-optimised config (same as sr_trader.py) ──────────────────

SYMBOL_CONFIGS = {
    "BTCUSDT": {
        "vol_mult":     2.0,
        "breakout":     0.0010,
        "arm_distance": 0.0015,
        "windows":      [],
        "skip_days":    set(),
    },
    "ETHUSDT": {
        "vol_mult":     3.0,
        "breakout":     0.0015,
        "arm_distance": 0.0030,
        "windows":      [],
        "skip_days":    set(),
    },
}
SYMBOLS = list(SYMBOL_CONFIGS.keys())

# ── Shared strategy parameters ─────────────────────────────────────────────────

TP_MULT        = 2.0
SL_MULT        = 1.0
Z_ENTRY        = 1.2
VOL_LOOKBACK   = 20
SIGMA_CANDLES  = 20
SIGNAL_CANDLES = 5
ROLLING_4H     = 240
HOLD_INTERVALS = 15
MAX_HOLD_MINS  = 33
LEVEL_COOLDOWN = 10

# ── Limit order parameters ─────────────────────────────────────────────────────

LIMIT_ENTRY_TIMEOUT_MINS = 5
LIMIT_CANCEL_DRIFT_PCT   = 0.003

# ── Trading parameters ─────────────────────────────────────────────────────────

LEVERAGE        = 2
MARGIN_COIN     = "USDT"
INTERVAL        = "1m"
MAX_TRADE_PCT   = 0.10
POLL_SECS       = 60
INIT_PAGES      = 2

FEE_TAKER       = 0.00060   # taker rate — SL exits
FEE_MAKER       = 0.00020   # maker rate — limit entry + limit TP
ROUND_TRIP_FEE  = FEE_MAKER * 2   # fee gate: both legs maker
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

TRADE_CSV = os.path.join("log", "sr_limit_trades.csv")


# ── Candle helpers ─────────────────────────────────────────────────────────────

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
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
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


# ── S/R level computation ──────────────────────────────────────────────────────

def utc_date_of(ts_ms: int) -> date:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date()


def update_session(s: dict, c: dict) -> None:
    d = utc_date_of(c["time"])
    if d != s["sess_day"]:
        if s["sess_high"] is not None:
            s["prev_sess_high"] = s["sess_high"]
            s["prev_sess_low"]  = s["sess_low"]
        s["sess_day"]  = d
        s["sess_high"] = c["high"]
        s["sess_low"]  = c["low"]
    else:
        if s["sess_high"] is None:
            s["sess_high"] = c["high"]
            s["sess_low"]  = c["low"]
        else:
            s["sess_high"] = max(s["sess_high"], c["high"])
            s["sess_low"]  = min(s["sess_low"],  c["low"])


def init_session(candles: list[dict]) -> dict:
    s = {"sess_high": None, "sess_low": None,
         "prev_sess_high": None, "prev_sess_low": None, "sess_day": None}
    for c in candles[:-1]:
        update_session(s, c)
    return s


def compute_levels(s: dict, buf: list[dict]) -> list[float]:
    lvl = set()
    if s["sess_high"] is not None:
        lvl.add(s["sess_high"])
        lvl.add(s["sess_low"])
    w4h = buf[-ROLLING_4H:]
    if w4h:
        lvl.add(max(c["high"] for c in w4h))
        lvl.add(min(c["low"]  for c in w4h))
    if s["prev_sess_high"] is not None:
        lvl.add(s["prev_sess_high"])
        lvl.add(s["prev_sess_low"])
    return sorted(lvl)


# ── Signal computation ─────────────────────────────────────────────────────────

def signals(buf: list[dict]) -> tuple[float, float, float]:
    min_len = SIGMA_CANDLES + SIGNAL_CANDLES
    if len(buf) < min_len + VOL_LOOKBACK:
        return 0.0, 0.0, 0.0
    sigma_w  = buf[-(SIGMA_CANDLES + SIGNAL_CANDLES):-SIGNAL_CANDLES]
    signal_w = buf[-SIGNAL_CANDLES:]
    vol_w    = buf[-VOL_LOOKBACK:]
    sigma = compute_sigma(sigma_w)
    if sigma == 0:
        return 0.0, 0.0, 0.0
    prices = [c["close"] for c in signal_w]
    drift  = math.log(prices[-1] / prices[0])
    z      = drift / (sigma * math.sqrt(SIGNAL_CANDLES))
    vols    = [c["volume"] for c in vol_w if c["volume"] > 0]
    vol_avg = sum(vols) / len(vols) if vols else 0.0
    return sigma, z, vol_avg


# ── API helpers ────────────────────────────────────────────────────────────────

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
        "symbol",      body.get("symbol"),
        "qty",         body.get("qty"),
        "side",        buy_sell,
        "orderType",   "LIMIT",
        "tradeSide",   "OPEN",
        "tpPrice",     body.get("tp_price"),
        "slPrice",     body.get("sl_price"),
        "tpStopType",  "MARK_PRICE",
        "slStopType",  "MARK_PRICE",
        "entryPrice",  body.get("entry_price"),
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


def _clear_limit(s: dict) -> None:
    """Clear limit order tracking and arm fields after fill or cancel."""
    s["limit_order_id"]  = None
    s["limit_price"]     = None
    s["limit_placed_at"] = None
    s["limit_side"]      = None
    s["limit_qty"]       = None
    s["limit_sigma"]     = None
    s["limit_tp_price"]  = None
    s["limit_sl_price"]  = None
    s["armed_level"]     = None
    s["armed_dir"]       = None
    s["arm_id"]          = None
    s["arm_time"]        = None
    s["arm_price"]       = None


# ── Main loop ──────────────────────────────────────────────────────────────────

def run(debug: bool, symbols: list[str] = None) -> None:
    client = BitunixClient(API_KEY, SECRET_KEY)
    active = symbols or SYMBOLS

    log.info("━" * 60)
    log.info("  S/R Breakout Trader — Limit Orders")
    log.info(f"  Symbols   : {', '.join(active)}")
    log.info(f"  TP/SL     : {TP_MULT}/{SL_MULT}σ  |  Hold: ≤{MAX_HOLD_MINS}min")
    log.info(f"  Z-entry   : {Z_ENTRY}  |  Leverage: {LEVERAGE}×")
    log.info(f"  Max trade : {MAX_TRADE_PCT:.0%}  |  Fees: maker={FEE_MAKER*100:.3f}%  taker={FEE_TAKER*100:.3f}%")
    log.info(f"  Limit     : timeout={LIMIT_ENTRY_TIMEOUT_MINS}min  drift={LIMIT_CANCEL_DRIFT_PCT*100:.1f}%")
    if debug:
        log.info("  MODE      : DEBUG — no real orders")
    log.info("━" * 60)

    _day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    for sym in active:
        cfg     = SYMBOL_CONFIGS.get(sym, {})
        windows = cfg.get("windows")
        hrs     = ("  ".join(f"{sh:02d}:00–{eh:02d}:00" for sh, eh in windows) + " UTC") if windows else "all hours"
        skip    = cfg.get("skip_days", set())
        days    = ", ".join(_day_names[d] for d in sorted(skip)) if skip else "all days"
        log.info(f"  {sym}  vol-mult {cfg.get('vol_mult')}×  "
                 f"breakout {cfg.get('breakout')*100:.3f}%  "
                 f"arm {cfg.get('arm_distance')*100:.3f}%  "
                 f"hours={hrs}  skip={days}")

    for sym in active:
        set_leverage(client, sym, debug)

    start_balance = get_balance(client)
    min_balance   = start_balance * MIN_BALANCE_PCT
    log.info(f"  Balance: {start_balance:.4f} USDT  |  Floor: {min_balance:.4f}")

    # ── Initialise per-symbol state ────────────────────────────────────────────
    sym_state: dict[str, dict] = {}
    for sym in active:
        log.info(f"  {sym}: fetching {INIT_PAGES * 1000} candles for init...")
        buf  = fetch_candles_paged(client, sym, pages=INIT_PAGES)
        sess = init_session(buf)
        sym_state[sym] = {
            "buf":              buf,
            "last_ts":          buf[-1]["time"] if buf else 0,
            **sess,
            "state":            "WATCHING",
            "armed_level":      None,
            "armed_dir":        None,
            "broken_levels":    {},
            "position":         None,
            # Limit order tracking
            "limit_order_id":   None,
            "limit_price":      None,
            "limit_placed_at":  None,
            "limit_side":       None,
            "limit_qty":        None,
            "limit_sigma":      None,
            "limit_tp_price":   None,
            "limit_sl_price":   None,
            # Arm event tracking
            "arm_id":           None,
            "arm_time":         None,
            "arm_price":        None,
        }
        log.info(f"  {sym}: {len(buf)} candles loaded — ready")

    # ── Hydrate open positions ─────────────────────────────────────────────────
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
                "qty":         float(p.get("qty", 0)),
                "tp_price":    None,
                "sl_price":    None,
                "opened_at":   now_utc(),
                "debug":       False,
                "arm_id":      "",
                "tp_order_id": None,
            }
            log.info(f"  Hydrated {sym} [{side}] @ "
                     f"{sym_state[sym]['position']['entry_price']:.4f}  "
                     f"(note: TP limit order unknown after restart — time exit only)")

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
                s   = sym_state[sym]
                cfg = SYMBOL_CONFIGS.get(sym, {})
                try:
                    _process_symbol(client, sym, s, cfg, balance, debug)
                except Exception as e:
                    log.error(f"  {sym} error: {e}")

        except Exception as e:
            log.error(f"  Cycle error: {e}")

        elapsed = (now_utc() - cycle_start).total_seconds()
        time.sleep(max(5, POLL_SECS - elapsed))


def _process_symbol(client: BitunixClient, sym: str, s: dict, cfg: dict,
                    balance: float, debug: bool) -> None:

    # ── 1. Fetch and append new candles ───────────────────────────────────────
    recent = fetch_latest_candles(client, sym, n=5)
    new    = [c for c in recent if c["time"] > s["last_ts"]]
    new.sort(key=lambda c: c["time"])

    if not new:
        log.info(f"  {sym}: no new candle yet")
        return

    for nc in new:
        if s["buf"]:
            update_session(s, s["buf"][-1])
        s["buf"].append(nc)
        if len(s["buf"]) > 2000:
            s["buf"] = s["buf"][-2000:]
        s["last_ts"] = nc["time"]

    buf   = s["buf"]
    c     = buf[-1]
    price = c["close"]
    ts    = c["time"]

    cooldown_ms = LEVEL_COOLDOWN * 60_000
    s["broken_levels"] = {
        lvl: t for lvl, t in s["broken_levels"].items()
        if ts - t < cooldown_ms
    }

    sigma, z, vol_avg = signals(buf)
    vol_now = c["volume"]

    log.info(f"  {sym}  price={price:.4f}  σ={sigma*100:.4f}%  "
             f"z={z:+.3f}  state={s['state']}")

    # ── 2. State machine ───────────────────────────────────────────────────────

    if s["state"] == "IN_TRADE":
        _monitor_trade(client, sym, s, c, debug)
        return

    if s["state"] == "LIMIT_OPEN":
        _monitor_limit_order(client, sym, s, c, debug)
        return

    if s["state"] == "ARMED":
        level     = s["armed_level"]
        direction = s["armed_dir"]
        dist_pct  = abs(price - level) / level

        if dist_pct > cfg["arm_distance"] * 3:
            log.info(f"  {sym}: disarmed (price moved away from {level:.4f})")
            arm_log.log_arm_event(s["arm_id"], STRATEGY, sym, s["arm_time"], s["arm_price"],
                                  "LONG" if direction == "UP" else "SHORT",
                                  "NO_FIRE", disarm_price=price,
                                  no_fire_reason="PRICE_MOVED_AWAY")
            s["state"] = "WATCHING"
            s["armed_level"] = s["armed_dir"] = None
            s["arm_id"] = s["arm_time"] = s["arm_price"] = None
            return

        if direction == "UP":
            broke  = price > level * (1 + cfg["breakout"])
            z_ok   = z >= Z_ENTRY
        else:
            broke  = price < level * (1 - cfg["breakout"])
            z_ok   = z <= -Z_ENTRY

        vol_ok = vol_avg > 0 and vol_now >= vol_avg * cfg["vol_mult"]

        if broke and z_ok and vol_ok:
            side = "LONG" if direction == "UP" else "SHORT"
            log.info(f"  {sym}: BREAKOUT confirmed  "
                     f"level={level:.4f}  dir={direction}  → {side}")
            _enter_trade(client, sym, s, cfg, c, side, level, sigma, balance, debug)
        else:
            reasons = []
            if not broke:  reasons.append(f"no-break(need {level*(1+cfg['breakout']) if direction=='UP' else level*(1-cfg['breakout']):.4f})")
            if not z_ok:   reasons.append(f"z={z:+.3f}")
            if not vol_ok: reasons.append(f"vol={vol_now:.0f}<{vol_avg*cfg['vol_mult']:.0f}")
            log.info(f"  {sym}: armed @ {level:.4f} — waiting [{', '.join(reasons)}]")
        return

    # WATCHING
    levels = compute_levels(s, buf)
    levels = [lvl for lvl in levels if lvl not in s["broken_levels"]]

    near_level = None
    near_dir   = None
    if levels:
        nearest  = min(levels, key=lambda lvl: abs(price - lvl))
        dist_pct = abs(price - nearest) / nearest
        log.info(f"  {sym}: nearest level {nearest:.4f}  dist {dist_pct*100:.3f}%")
        if dist_pct <= cfg["arm_distance"]:
            near_level = nearest
            near_dir   = "UP" if price < nearest else "DOWN"
    else:
        log.info(f"  {sym}: no levels available")

    candle_dt   = datetime.fromtimestamp(c["time"] / 1000, tz=timezone.utc)
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
        if near_level is not None:
            side_est   = "LONG" if near_dir == "UP" else "SHORT"
            wtp = wsl  = None
            if sigma > 0:
                sigma_hold = sigma * math.sqrt(HOLD_INTERVALS)
                move = price * sigma_hold
                wtp = (price + move * TP_MULT) if side_est == "LONG" else (price - move * TP_MULT)
                wsl = (price - move * SL_MULT) if side_est == "LONG" else (price + move * SL_MULT)
            arm_log.log_arm_event(
                arm_log.new_arm_id(), STRATEGY, sym,
                now_utc().strftime("%Y-%m-%d %H:%M:%S"), price, side_est,
                "PENDING", shadow=True, atr=sigma if sigma > 0 else None,
                would_be_tp=wtp, would_be_sl=wsl,
            )
            log.info(f"  {sym}: shadow ARMED @ {near_level:.4f} dir={near_dir} "
                     f"({candle_dt.strftime('%a %H:%M')} UTC outside window)")
        if not in_window:
            log.info(f"  {sym}: outside trade window ({candle_dt.strftime('%a %H:%M')} UTC) — skip")
        else:
            log.info(f"  {sym}: skip day ({candle_dt.strftime('%a')} UTC) — skip")
        return

    if near_level is not None:
        s["state"]       = "ARMED"
        s["armed_level"] = near_level
        s["armed_dir"]   = near_dir
        s["arm_id"]      = arm_log.new_arm_id()
        s["arm_time"]    = now_utc().strftime("%Y-%m-%d %H:%M:%S")
        s["arm_price"]   = price
        log.info(f"  {sym}: ARMED  level={near_level:.4f}  dir={near_dir}")


def _enter_trade(client: BitunixClient, sym: str, s: dict, cfg: dict,
                 c: dict, side: str, level: float,
                 sigma: float, balance: float, debug: bool) -> None:
    """Place a limit entry order at the breakout price."""
    if sigma == 0:
        log.warning(f"  {sym}: sigma=0, skip entry")
        return

    ticker      = fetch_ticker(client, sym)
    limit_price = float(ticker.get("lastPrice", c["close"]))
    sigma_hold  = sigma * math.sqrt(HOLD_INTERVALS)
    tp_move     = limit_price * sigma_hold * TP_MULT
    sl_move     = limit_price * sigma_hold * SL_MULT
    fee_cost    = limit_price * ROUND_TRIP_FEE

    if tp_move <= fee_cost:
        log.info(f"  {sym}: SKIP — TP move {tp_move:.4f} ≤ fee {fee_cost:.4f}")
        arm_log.log_arm_event(s["arm_id"], STRATEGY, sym, s["arm_time"], s["arm_price"],
                              side, "NO_FIRE", disarm_price=limit_price,
                              no_fire_reason="FEE_GATE", atr=sigma)
        s["state"] = "WATCHING"
        s["armed_level"] = s["armed_dir"] = None
        s["arm_id"] = s["arm_time"] = s["arm_price"] = None
        return

    tp_price = (limit_price + tp_move) if side == "LONG" else (limit_price - tp_move)
    sl_price = (limit_price - sl_move) if side == "LONG" else (limit_price + sl_move)

    notional = balance * MAX_TRADE_PCT * LEVERAGE
    qty      = round_qty(sym, notional / limit_price)
    min_qty  = {"BTCUSDT": 0.0001, "ETHUSDT": 0.003}.get(sym, 0.001)
    if qty < min_qty:
        log.info(f"  {sym}: qty {qty} < min {min_qty}, skip")
        arm_log.log_arm_event(s["arm_id"], STRATEGY, sym, s["arm_time"], s["arm_price"],
                              side, "NO_FIRE", disarm_price=limit_price,
                              no_fire_reason="MIN_QTY", atr=sigma)
        s["state"] = "WATCHING"
        s["armed_level"] = s["armed_dir"] = None
        s["arm_id"] = s["arm_time"] = s["arm_price"] = None
        return

    log.info(f"  {sym}: placing LIMIT entry [{side}] @ {limit_price:.4f}  "
             f"sl={sl_price:.4f}  tp_target={tp_price:.4f}  qty={qty}  "
             f"timeout={LIMIT_ENTRY_TIMEOUT_MINS}min  drift={LIMIT_CANCEL_DRIFT_PCT*100:.1f}%")

    order_id = place_limit_entry(client, sym, side, qty, limit_price, sl_price, debug)
    if order_id is None:
        arm_log.log_arm_event(s["arm_id"], STRATEGY, sym, s["arm_time"], s["arm_price"],
                              side, "NO_FIRE", disarm_price=limit_price,
                              no_fire_reason="ORDER_FAILED", atr=sigma)
        s["state"] = "WATCHING"
        s["armed_level"] = s["armed_dir"] = None
        s["arm_id"] = s["arm_time"] = s["arm_price"] = None
        return

    s["state"]           = "LIMIT_OPEN"
    s["limit_order_id"]  = order_id
    s["limit_price"]     = limit_price
    s["limit_placed_at"] = now_utc()
    s["limit_side"]      = side
    s["limit_qty"]       = qty
    s["limit_sigma"]     = sigma
    s["limit_tp_price"]  = tp_price
    s["limit_sl_price"]  = sl_price
    # armed_level preserved for broken_levels update on fill
    # arm_id/arm_time/arm_price preserved for FIRED log on fill


def _monitor_limit_order(client: BitunixClient, sym: str, s: dict,
                         c: dict, debug: bool) -> None:
    """Poll the pending limit entry: cancel on timeout/drift, transition on fill."""
    side          = s["limit_side"]
    qty           = s["limit_qty"]
    limit_price   = s["limit_price"]
    limit_order_id = s["limit_order_id"]
    sigma         = s["limit_sigma"]
    tp_price      = s["limit_tp_price"]
    sl_price      = s["limit_sl_price"]
    elapsed_mins  = (now_utc() - s["limit_placed_at"]).total_seconds() / 60

    try:
        ticker        = fetch_ticker(client, sym)
        current_price = float(ticker.get("lastPrice", c["close"]))
    except Exception:
        current_price = c["close"]

    drift_cancel = (
        (side == "LONG"  and current_price < limit_price * (1 - LIMIT_CANCEL_DRIFT_PCT)) or
        (side == "SHORT" and current_price > limit_price * (1 + LIMIT_CANCEL_DRIFT_PCT))
    )
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
            no_fire_reason=f"LIMIT_{reason}", atr=sigma,
        )
        s["state"] = "WATCHING"
        _clear_limit(s)
        return

    if debug:
        filled = True
    else:
        status = get_order_status(client, limit_order_id, sym)
        filled = status in ("FILLED", "COMPLETELY_FILLED")

    if not filled:
        log.info(f"  {sym}: limit entry pending [{side}] @ {limit_price:.4f}  "
                 f"current={current_price:.4f}  elapsed={elapsed_mins:.1f}min")
        return

    # Filled — resolve position and place TP
    if debug:
        position_id = limit_order_id
    else:
        position_id = resolve_position_id(client, sym, side)
        if position_id is None:
            log.warning(f"  {sym}: could not resolve positionId — falling back to orderId")
            position_id = limit_order_id
        else:
            log.info(f"  {sym}: resolved positionId={position_id}")

    tp_order_id = place_limit_tp(client, sym, side, qty, tp_price, debug)

    log.info(f"  {sym}: limit FILLED [{side}] @ {limit_price:.4f}  "
             f"tp_order={tp_order_id} @ {tp_price:.4f}  sl={sl_price:.4f}")

    arm_log.log_arm_event(
        s["arm_id"], STRATEGY, sym, s["arm_time"], s["arm_price"],
        side, "FIRED", disarm_price=limit_price, atr=sigma,
    )
    log_trade({
        "symbol":       sym,
        "side":         side,
        "qty":          qty,
        "tp_price":     tp_price,
        "sl_price":     sl_price,
        "entry_price":  limit_price,
        "arm_id":       s["arm_id"],
        "arm_price":    s["arm_price"],
        "signal_price": s["arm_price"],
    })

    saved_arm_id   = s["arm_id"]
    broken_level   = s["armed_level"]
    s["state"]     = "IN_TRADE"
    s["broken_levels"][broken_level] = c["time"]
    _clear_limit(s)
    s["position"] = {
        "position_id": position_id,
        "side":        side,
        "entry_price": limit_price,
        "tp_price":    tp_price,
        "sl_price":    sl_price,
        "qty":         qty,
        "opened_at":   now_utc(),
        "debug":       debug,
        "arm_id":      saved_arm_id,
        "tp_order_id": tp_order_id,
    }
    position_registry.register(position_id, STRATEGY, sym, side, limit_price, saved_arm_id)


def _monitor_trade(client: BitunixClient, sym: str, s: dict,
                   c: dict, debug: bool) -> None:
    pos          = s["position"]
    side         = pos["side"]
    held_mins    = (now_utc() - pos["opened_at"]).total_seconds() / 60
    tp           = pos["tp_price"]
    sl           = pos["sl_price"]
    tp_order_id  = pos.get("tp_order_id")

    if debug:
        price  = c["close"]
        tp_hit = (c["high"] >= tp) if (side == "LONG" and tp is not None) else \
                 (c["low"]  <= tp) if (side == "SHORT" and tp is not None) else False
        sl_hit = (c["low"]  <= sl) if (side == "LONG" and sl is not None) else \
                 (c["high"] >= sl) if (side == "SHORT" and sl is not None) else False
        pnl    = (price - pos["entry_price"]) * pos["qty"]
        if side == "SHORT":
            pnl = -pnl
        log.info(f"  {sym} [DEBUG {side}]  entry={pos['entry_price']:.4f}  "
                 f"now={price:.4f}  uPnL≈{pnl:+.4f}  held={held_mins:.1f}min")

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
            fee = ROUND_TRIP_FEE if outcome == "TP" else FEE_MAKER + FEE_TAKER
            arm_log.log_close_event(
                pos.get("arm_id", ""), STRATEGY, sym, side,
                outcome, exit_price, held_mins,
                arm_log.calc_pnl(side, pos["entry_price"], exit_price, pos["qty"], fee),
                order_id=pos.get("position_id"),
            )
            position_registry.release(pos.get("position_id", ""))
            s["state"]    = "WATCHING"
            s["position"] = None
        return

    # Live
    live = [p for p in get_open_positions(client, sym)
            if p.get("positionId") == pos["position_id"]]

    if not live:
        tp_filled = False
        if tp_order_id:
            tp_status = get_order_status(client, tp_order_id, sym)
            tp_filled = tp_status in ("FILLED", "COMPLETELY_FILLED")

        hist     = get_closed_position(client, sym, pos["position_id"])
        close_px = float(hist.get("closePrice", c["close"])) if hist else c["close"]
        if not hist:
            log.warning(f"  {sym}: could not fetch close history, inferring from candle price")

        if tp_filled:
            outcome_str, exit_px = "TP", (tp if tp is not None else close_px)
        elif tp is not None and sl is not None:
            if side == "LONG":
                outcome_str = "SL" if close_px <= sl else "EXCHANGE_CLOSED"
            else:
                outcome_str = "SL" if close_px >= sl else "EXCHANGE_CLOSED"
            exit_px = sl if outcome_str == "SL" else close_px
        else:
            outcome_str, exit_px = "EXCHANGE_CLOSED", close_px

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
        s["state"]    = "WATCHING"
        s["position"] = None
        return

    p    = live[0]
    if p.get("positionId") and p["positionId"] != pos["position_id"]:
        log.info(f"  {sym}: syncing positionId {pos['position_id']} → {p['positionId']}")
        pos["position_id"] = p["positionId"]
    upnl = float(p.get("unrealizedPNL", 0))
    tp_str = f"{tp:.4f}" if tp is not None else "?"
    sl_str = f"{sl:.4f}" if sl is not None else "?"
    log.info(f"  {sym} [{side}]  entry={pos['entry_price']:.4f}  "
             f"uPnL={upnl:+.4f}  held={held_mins:.1f}min  "
             f"tp={tp_str}  sl={sl_str}  tp_order={tp_order_id}")

    if held_mins >= MAX_HOLD_MINS:
        log.info(f"  {sym}: time exit after {held_mins:.1f}min")
        if tp_order_id:
            cancel_order(client, tp_order_id, sym, debug)
        if close_position(client, pos["position_id"], sym, debug):
            arm_log.log_close_event(
                pos.get("arm_id", ""), STRATEGY, sym, side,
                "TIME", c["close"], held_mins,
                arm_log.calc_pnl(side, pos["entry_price"], c["close"], pos["qty"],
                                 FEE_MAKER + FEE_TAKER),
                order_id=pos.get("position_id"),
            )
            position_registry.release(pos.get("position_id", ""))
            s["state"]    = "WATCHING"
            s["position"] = None


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="S/R Breakout trader — limit entry + limit TP")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode — no real orders placed")
    parser.add_argument("--symbol", nargs="+", metavar="SYMBOL",
                        help=f"Symbol(s) to trade (default: {', '.join(SYMBOLS)})")
    args = parser.parse_args()
    run(debug=args.debug, symbols=args.symbol)


if __name__ == "__main__":
    main()
