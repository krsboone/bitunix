"""
sr_trader.py — Bitunix S/R Breakout perpetual futures trader

Strategy: Support/Resistance breakout with state machine
  WATCHING → ARMED → IN_TRADE

  S/R levels:  current session high/low | 4-hour rolling high/low |
               previous session high/low
  Arm:         price approaches a level within arm_distance %
  Enter:       price breaks convincingly through the armed level:
                 breakout % + volume confirmation + Z-score gate
  TP/SL:       1.0σ / 3.0σ × sqrt(hold_intervals) from entry
  Time exit:   close if held > MAX_HOLD_MINS

Per-symbol sweep-optimised parameters (30-day walk-forward + 7-day holdout):
  BTCUSDT: vol-mult 2.0  breakout 0.10%  arm-distance 0.15%  (holdout score 36.2)
  ETHUSDT: vol-mult 3.0  breakout 0.15%  arm-distance 0.30%  (holdout score 42.4)

Usage:
    python3 sr_trader.py              # live trading
    python3 sr_trader.py --debug      # DEBUG mode — no real orders
    python3 sr_trader.py --symbol BTCUSDT
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

start_logging("sr_trader")

# ── Per-symbol sweep-optimised config ─────────────────────────────────────────

SYMBOL_CONFIGS = {
    "BTCUSDT": {
        "vol_mult":    2.0,
        "breakout":    0.0010,
        "arm_distance": 0.0015,
        "windows":     [(6, 9)],   # 06:00–09:00 UTC Mon-Fri
        "skip_days":   {5, 6},     # Sat, Sun
    },
    "ETHUSDT": {
        "vol_mult":    3.0,
        "breakout":    0.0015,
        "arm_distance": 0.0030,
        "windows":     [(7, 9)],   # 07:00–09:00 UTC Mon-Fri
        "skip_days":   {5, 6},     # Sat, Sun
    },
}
SYMBOLS = list(SYMBOL_CONFIGS.keys())

# ── Shared strategy parameters ─────────────────────────────────────────────────

TP_MULT        = 1.0    # sweep winner: both symbols
SL_MULT        = 3.0    # sweep winner: both symbols
Z_ENTRY        = 1.2
VOL_LOOKBACK   = 20
SIGMA_CANDLES  = 20
SIGNAL_CANDLES = 5
ROLLING_4H     = 240    # 1m candles in 4 hours
HOLD_INTERVALS = 15
MAX_HOLD_MINS  = 33
LEVEL_COOLDOWN = 10     # minutes before re-triggering same level

# ── Trading parameters ─────────────────────────────────────────────────────────

LEVERAGE        = 2
MARGIN_COIN     = "USDT"
INTERVAL        = "1m"
MAX_TRADE_PCT   = 0.10   # 10% of balance per trade (conservative)
POLL_SECS       = 60     # 1-minute candle rhythm
INIT_PAGES      = 2      # pages of 1000 candles for S/R session init (~33h)

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

TRADE_CSV = os.path.join("log", "sr_trades.csv")


# ── Candle helpers ─────────────────────────────────────────────────────────────

def normalize(c: dict) -> dict:
    """Convert raw API candle (string values) to float dict."""
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
    """Fetch up to pages × 1000 historical 1m candles, newest last."""
    now_ms  = int(datetime.now(timezone.utc).timestamp() * 1000)
    end_ms  = now_ms
    raw     = []

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

    # Deduplicate, sort ascending, normalize
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
    """Fetch the n most recent 1m candles."""
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
    """Add candle c into the rolling session state."""
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
    """
    Build initial session state by walking through all but the last candle.
    Returns state dict ready for the main loop (session reflects candles[:-1]).
    """
    s = {"sess_high": None, "sess_low": None,
         "prev_sess_high": None, "prev_sess_low": None, "sess_day": None}
    for c in candles[:-1]:
        update_session(s, c)
    return s


def compute_levels(s: dict, buf: list[dict]) -> list[float]:
    """O(ROLLING_4H) S/R level list from incremental session state."""
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
    """Returns (sigma, z_score, vol_avg). Returns zeros if buffer too short."""
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
        "symbol": symbol, "leverage": LEVERAGE, "marginCoin": MARGIN_COIN,
    })
    if resp.get("code") != 0:
        log.warning(f"  Leverage set failed {symbol}: {resp.get('msg')}")
    else:
        log.info(f"  {symbol} leverage → {LEVERAGE}×")


def log_trade(body: dict) -> None:
    """Write a row in backtest.py-compatible key-value format."""
    os.makedirs("log", exist_ok=True)
    ts      = now_utc().strftime("%Y-%m-%d %H:%M:%S")
    side    = body.get("side")
    buy_sell = "BUY" if side == "LONG" else "SELL"
    row = [
        ts, ts,
        "symbol",     body.get("symbol"),
        "qty",        body.get("qty"),
        "side",       buy_sell,
        "orderType",  "MARKET",
        "tradeSide",  "OPEN",
        "tpPrice",    body.get("tp_price"),
        "slPrice",    body.get("sl_price"),
        "tpStopType", "MARK_PRICE",
        "slStopType", "MARK_PRICE",
        "entryPrice", body.get("entry_price"),
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


# ── Main loop ──────────────────────────────────────────────────────────────────

def run(debug: bool, symbols: list[str] = None) -> None:
    client  = BitunixClient(API_KEY, SECRET_KEY)
    active  = symbols or SYMBOLS

    log.info("━" * 60)
    log.info("  S/R Breakout Trader")
    log.info(f"  Symbols   : {', '.join(active)}")
    log.info(f"  TP/SL     : {TP_MULT}/{SL_MULT}σ  |  Hold: ≤{MAX_HOLD_MINS}min")
    log.info(f"  Z-entry   : {Z_ENTRY}  |  Leverage: {LEVERAGE}×")
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
        buf = fetch_candles_paged(client, sym, pages=INIT_PAGES)
        sess = init_session(buf)
        sym_state[sym] = {
            # Candle buffer
            "buf":              buf,
            "last_ts":          buf[-1]["time"] if buf else 0,
            # S/R session state (reflects buf[:-1])
            **sess,
            # State machine
            "state":            "WATCHING",
            "armed_level":      None,
            "armed_dir":        None,
            # Broken level cooldown: level → timestamp_ms when triggered
            "broken_levels":    {},
            # Open position (None when WATCHING/ARMED)
            "position":         None,
        }
        log.info(f"  {sym}: {len(buf)} candles loaded — ready")

    # ── Hydrate open positions ─────────────────────────────────────────────────
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
                "qty":         float(p.get("qty", 0)),
                "tp_price":    None,  # set by exchange; not tracked locally
                "sl_price":    None,
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
                s   = sym_state[sym]
                cfg = SYMBOL_CONFIGS.get(sym, {})
                try:
                    _process_symbol(client, sym, s, cfg, balance, debug)
                except Exception as e:
                    log.error(f"  {sym} error: {e}")

        except Exception as e:
            log.error(f"  Cycle error: {e}")

        # Sleep until next 60s boundary
        elapsed = (now_utc() - cycle_start).total_seconds()
        sleep   = max(5, POLL_SECS - elapsed)
        time.sleep(sleep)


def _process_symbol(client: BitunixClient, sym: str, s: dict, cfg: dict,
                    balance: float, debug: bool) -> None:
    """Fetch new candles, update state, run state machine for one symbol."""

    # ── 1. Fetch and append new candles ───────────────────────────────────────
    recent = fetch_latest_candles(client, sym, n=5)
    new    = [c for c in recent if c["time"] > s["last_ts"]]
    new.sort(key=lambda c: c["time"])

    if not new:
        log.info(f"  {sym}: no new candle yet")
        return

    for nc in new:
        # Update session with the candle BEFORE appending (so session reflects buf[:-1])
        if s["buf"]:
            update_session(s, s["buf"][-1])
        s["buf"].append(nc)
        if len(s["buf"]) > 2000:
            s["buf"] = s["buf"][-2000:]
        s["last_ts"] = nc["time"]

    buf = s["buf"]
    c   = buf[-1]   # most recent complete candle
    price = c["close"]
    ts    = c["time"]

    # Expire broken level cooldowns
    cooldown_ms = LEVEL_COOLDOWN * 60_000
    s["broken_levels"] = {
        lvl: t for lvl, t in s["broken_levels"].items()
        if ts - t < cooldown_ms
    }

    # Compute signals
    sigma, z, vol_avg = signals(buf)
    vol_now = c["volume"]

    log.info(f"  {sym}  price={price:.4f}  σ={sigma*100:.4f}%  "
             f"z={z:+.3f}  state={s['state']}")

    # ── 2. State machine ───────────────────────────────────────────────────────

    if s["state"] == "IN_TRADE":
        _monitor_trade(client, sym, s, c, debug)
        return

    if s["state"] == "ARMED":
        level     = s["armed_level"]
        direction = s["armed_dir"]
        dist_pct  = abs(price - level) / level

        # Disarm if price has moved too far away
        if dist_pct > cfg["arm_distance"] * 3:
            log.info(f"  {sym}: disarmed (price moved away from {level:.4f})")
            s["state"] = "WATCHING"
            s["armed_level"] = s["armed_dir"] = None
            return

        # Check breakout
        if direction == "UP":
            broke    = price > level * (1 + cfg["breakout"])
            body_ok  = price > c["open"]
            z_ok     = z >= Z_ENTRY
        else:
            broke    = price < level * (1 - cfg["breakout"])
            body_ok  = price < c["open"]
            z_ok     = z <= -Z_ENTRY

        vol_ok = vol_avg > 0 and vol_now >= vol_avg * cfg["vol_mult"]

        if broke and z_ok and vol_ok:
            side = "LONG" if direction == "UP" else "SHORT"
            log.info(f"  {sym}: BREAKOUT confirmed  "
                     f"level={level:.4f}  dir={direction}  → {side}")
            _enter_trade(client, sym, s, cfg, c, side, level,
                         sigma, balance, debug)
        else:
            reasons = []
            if not broke:   reasons.append(f"no-break(price={price:.4f} need {level*(1+cfg['breakout'] if direction=='UP' else 1-cfg['breakout']):.4f})")
            if not z_ok:    reasons.append(f"z={z:+.3f}")
            if not vol_ok:  reasons.append(f"vol={vol_now:.0f}<{vol_avg*cfg['vol_mult']:.0f}")
            log.info(f"  {sym}: armed @ {level:.4f} — waiting [{', '.join(reasons)}]")
        return

    # Time-of-day and day-of-week filter (only blocks new entries, not open trades)
    candle_dt   = datetime.fromtimestamp(c["time"] / 1000, tz=timezone.utc)
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

    # WATCHING — compute levels and check for arming
    levels = compute_levels(s, buf)
    levels = [lvl for lvl in levels if lvl not in s["broken_levels"]]

    if not levels:
        log.info(f"  {sym}: no levels available")
        return

    nearest  = min(levels, key=lambda lvl: abs(price - lvl))
    dist_pct = abs(price - nearest) / nearest

    log.info(f"  {sym}: nearest level {nearest:.4f}  dist {dist_pct*100:.3f}%")

    if dist_pct <= cfg["arm_distance"]:
        direction = "UP" if price < nearest else "DOWN"
        s["state"]       = "ARMED"
        s["armed_level"] = nearest
        s["armed_dir"]   = direction
        log.info(f"  {sym}: ARMED  level={nearest:.4f}  dir={direction}")


def _enter_trade(client: BitunixClient, sym: str, s: dict, cfg: dict,
                 c: dict, side: str, level: float,
                 sigma: float, balance: float, debug: bool) -> None:
    """Size and place a market entry order."""
    if sigma == 0:
        log.warning(f"  {sym}: sigma=0, skip entry")
        return

    ticker     = fetch_ticker(client, sym)
    entry      = float(ticker.get("lastPrice", c["close"]))
    sigma_hold = sigma * math.sqrt(HOLD_INTERVALS)
    tp_move    = entry * sigma_hold * TP_MULT
    sl_move    = entry * sigma_hold * SL_MULT
    fee_cost   = entry * ROUND_TRIP_FEE

    if tp_move <= fee_cost:
        log.info(f"  {sym}: SKIP — TP move {tp_move:.4f} ≤ fee {fee_cost:.4f}")
        return

    tp_price = (entry + tp_move) if side == "LONG" else (entry - tp_move)
    sl_price = (entry - sl_move) if side == "LONG" else (entry + sl_move)

    notional = balance * MAX_TRADE_PCT * LEVERAGE
    qty      = round_qty(sym, notional / entry)
    min_qty  = {"BTCUSDT": 0.0001, "ETHUSDT": 0.003}.get(sym, 0.001)
    if qty < min_qty:
        log.info(f"  {sym}: qty {qty} < min {min_qty}, skip")
        return

    log.info(f"  {sym}: ENTER {side}  entry≈{entry:.4f}  "
             f"TP={tp_price:.4f}  SL={sl_price:.4f}  qty={qty}")

    order_id = place_order(client, sym, side, qty, tp_price, sl_price, debug)
    if order_id is None:
        return

    log_trade({
        "symbol":      sym,
        "side":        side,
        "qty":         qty,
        "tp_price":    tp_price,
        "sl_price":    sl_price,
        "entry_price": entry,
        "sigma":       sigma,
    })

    s["broken_levels"][level] = c["time"]
    s["state"]       = "IN_TRADE"
    s["armed_level"] = s["armed_dir"] = None
    s["position"]    = {
        "position_id": order_id,
        "side":        side,
        "entry_price": entry,
        "tp_price":    tp_price,
        "sl_price":    sl_price,
        "qty":         qty,
        "opened_at":   now_utc(),
        "debug":       debug,
    }


def _monitor_trade(client: BitunixClient, sym: str, s: dict,
                   c: dict, debug: bool) -> None:
    """Check if the open position has resolved (TP/SL/time exit)."""
    pos       = s["position"]
    held_mins = (now_utc() - pos["opened_at"]).total_seconds() / 60

    if debug:
        price  = c["close"]
        tp, sl = pos["tp_price"], pos["sl_price"]
        side   = pos["side"]
        tp_hit = (c["high"] >= tp) if side == "LONG" else (c["low"] <= tp)
        sl_hit = (c["low"]  <= sl) if side == "LONG" else (c["high"] >= sl)
        pnl    = (price - pos["entry_price"]) * pos["qty"]
        if side == "SHORT":
            pnl = -pnl

        log.info(f"  {sym} [DEBUG {side}]  entry={pos['entry_price']:.4f}  "
                 f"now={price:.4f}  uPnL≈{pnl:+.4f}  held={held_mins:.1f}min")

        if tp_hit:
            log.info(f"  {sym} [DEBUG]: TP hit @ {tp:.4f}")
            s["state"] = "WATCHING"
            s["position"] = None
        elif sl_hit:
            log.info(f"  {sym} [DEBUG]: SL hit @ {sl:.4f}")
            s["state"] = "WATCHING"
            s["position"] = None
        elif held_mins >= MAX_HOLD_MINS:
            log.info(f"  {sym} [DEBUG]: time exit @ {held_mins:.1f}min")
            s["state"] = "WATCHING"
            s["position"] = None
        return

    # Live: check if position still open on exchange (TP/SL set inline)
    live = [p for p in get_open_positions(client, sym)
            if p.get("positionId") == pos["position_id"]]

    if not live:
        log.info(f"  {sym}: position closed by exchange (TP/SL hit)")
        s["state"]    = "WATCHING"
        s["position"] = None
        return

    p    = live[0]
    upnl = float(p.get("unrealizedPNL", 0))
    log.info(f"  {sym} [{pos['side']}]  entry={pos['entry_price']:.4f}  "
             f"uPnL={upnl:+.4f}  held={held_mins:.1f}min")

    if held_mins >= MAX_HOLD_MINS:
        log.info(f"  {sym}: time exit after {held_mins:.1f}min")
        if close_position(client, pos["position_id"], sym, debug):
            s["state"]    = "WATCHING"
            s["position"] = None


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="S/R Breakout perpetual futures trader")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode — no real orders placed")
    parser.add_argument("--symbol", nargs="+", metavar="SYMBOL",
                        help=f"Symbol(s) to trade (default: {', '.join(SYMBOLS)})")
    args = parser.parse_args()
    run(debug=args.debug, symbols=args.symbol)


if __name__ == "__main__":
    main()
