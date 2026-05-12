"""
sr_bounce_trader.py — S/R Bounce trader: market entry on reversal candle

Strategy:
  Arms when price approaches a known support/resistance level from the correct side:
    LONG  — price above support, approaching from above (expects bounce up)
    SHORT — price below resistance, approaching from below (expects bounce down)
  Fires on the first reversal candle (close > prev for LONG, close < prev for SHORT)
  while still within max_bounce of the level.

Levels: current session H/L + previous session H/L + weekly (7-day) H/L.
        Rolling 4h window removed — too unstable to serve as support/resistance.

SL: structural — just below support (LONG) / just above resistance (SHORT)
TP: next significant level in trade direction. Fallback: 1.2% fixed target.
R/R gate: skip if TP distance < 1.5× SL distance.

State machine:
  WATCHING → ARMED → IN_TRADE → WATCHING

Usage:
    python3 sr_bounce_trader.py                        # live trading
    python3 sr_bounce_trader.py --debug                # DEBUG mode — no real orders
    python3 sr_bounce_trader.py --debug --dir LONG     # long-only in debug
    python3 sr_bounce_trader.py --debug --dir SHORT    # short-only in debug
    python3 sr_bounce_trader.py --symbol BTCUSDT
"""

import argparse
import csv
import json
import logging
import os
import time
from datetime import date, datetime, timezone

from auth import BitunixClient
from config import API_KEY, SECRET_KEY
from market import fetch_ticker
from log_cap import start_logging
import arm_log
import position_registry

start_logging("sr_bounce")

STRATEGY = "sr_bounce"

# ── Per-symbol config ──────────────────────────────────────────────────────────

SYMBOL_CONFIGS = {
    "BTCUSDT": {
        "arm_distance": 0.0010,   # arm when price is within 0.10% of level
        "sl_buffer":    0.0020,   # SL placed 0.20% below support / above resistance
        "max_bounce":   0.0060,   # disarm if price bounces > 0.60% from level
    },
    "ETHUSDT": {
        "arm_distance": 0.0015,
        "sl_buffer":    0.0030,
        "max_bounce":   0.0090,
    },
}
SYMBOLS = list(SYMBOL_CONFIGS.keys())

# ── Strategy parameters ────────────────────────────────────────────────────────

TP_FIXED        = 0.012      # 1.2% fallback TP when no next level exists
MIN_RR          = 1.5        # skip if TP dist / SL dist < this
MAX_HOLD_MINS   = 240
BREAK_TOLERANCE    = 0.0003     # candle must close > 0.03% through level to count as break
WEEKLY_LOOKBACK    = 7 * 24 * 60   # 10,080 1m candles
PRIOR_TEST_LOOKBACK = 24 * 60      # 1,440 1m candles — 24h window for prior level test
REVERSAL_REQUIRED  = 2             # consecutive reversal candles before entry
BUF_SIZE        = 11_500     # max candles to keep in memory (~8 days)

# ── Trading parameters ─────────────────────────────────────────────────────────

LEVERAGE        = 2
MARGIN_COIN     = "USDT"
INTERVAL        = "1m"
MAX_TRADE_PCT   = 0.10
POLL_SECS       = 60
INIT_PAGES      = 2

FEE_TAKER       = 0.00060   # market entry and SL/TIME exits
FEE_MAKER       = 0.00020   # limit TP exit
ROUND_TRIP_FEE  = FEE_TAKER + FEE_MAKER
MIN_BALANCE_PCT = 0.70

PRECISION = {
    "BTCUSDT": {"qty": 4, "price": 1},
    "ETHUSDT": {"qty": 3, "price": 2},
}

DATA_DIR = "data"

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

TRADE_CSV = os.path.join("log", "sr_bounce_trades.csv")


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


def load_candles_csv(symbol: str) -> list[dict]:
    path = os.path.join(DATA_DIR, f"{symbol}_1m.csv")
    if not os.path.exists(path):
        return []
    candles = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles.append({
                "time":   int(row["timestamp_ms"]),
                "open":   float(row["open"]),
                "high":   float(row["high"]),
                "low":    float(row["low"]),
                "close":  float(row["close"]),
                "volume": float(row.get("volume", 0)),
            })
    candles.sort(key=lambda c: c["time"])
    return candles


def fetch_gap_candles(client: BitunixClient, symbol: str,
                      after_ms: int) -> list[dict]:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    end_ms = now_ms
    raw    = []
    while True:
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


def compute_weekly_hl(buf: list[dict]) -> tuple[float | None, float | None]:
    window = buf[-WEEKLY_LOOKBACK:] if len(buf) >= WEEKLY_LOOKBACK else buf
    if not window:
        return None, None
    return max(c["high"] for c in window), min(c["low"] for c in window)


def compute_levels(s: dict, buf: list[dict]) -> list[float]:
    lvl = set()
    if s["sess_high"] is not None:
        lvl.add(s["sess_high"])
        lvl.add(s["sess_low"])
    if s["prev_sess_high"] is not None:
        lvl.add(s["prev_sess_high"])
        lvl.add(s["prev_sess_low"])
    wh, wl = compute_weekly_hl(buf)
    if wh is not None:
        lvl.add(wh)
    if wl is not None:
        lvl.add(wl)
    return sorted(lvl)


def find_tp_level(entry_price: float, side: str, levels: list[float]) -> float:
    """Return nearest level beyond entry in trade direction, or fixed fallback."""
    if side == "LONG":
        candidates = [l for l in levels if l > entry_price * (1 + 0.001)]
        if candidates:
            return min(candidates)
        return entry_price * (1 + TP_FIXED)
    else:
        candidates = [l for l in levels if l < entry_price * (1 - 0.001)]
        if candidates:
            return max(candidates)
        return entry_price * (1 - TP_FIXED)


def level_has_prior_test(level: float, arm_distance: float, buf: list[dict]) -> bool:
    """Return True if price touched within arm_distance of level at least once in
    the past PRIOR_TEST_LOOKBACK candles (excluding the current candle).
    A level touched before is proven S/R; one that hasn't been tested has no track record.
    """
    window = buf[max(0, len(buf) - PRIOR_TEST_LOOKBACK - 1):-1]
    for c in window:
        if (abs(c["high"] - level) / level <= arm_distance or
                abs(c["low"]  - level) / level <= arm_distance):
            return True
    return False


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
        "armId",      body.get("arm_id", ""),
        "armPrice",   body.get("arm_price", ""),
    ]
    with open(TRADE_CSV, "a", newline="") as f:
        csv.writer(f).writerow(row)


def place_market_entry(client: BitunixClient, symbol: str, side: str,
                       qty: float, sl_price: float, debug: bool) -> str | None:
    """Market OPEN order with inline stop-market SL. TP placed separately after fill."""
    buy_sell = "BUY" if side == "LONG" else "SELL"
    body = {
        "symbol":     symbol,
        "qty":        str(qty),
        "side":       buy_sell,
        "orderType":  "MARKET",
        "tradeSide":  "OPEN",
        "slPrice":    str(round_price(symbol, sl_price)),
        "slStopType": "MARK_PRICE",
    }
    if debug:
        log.info(f"  [DEBUG] Would place market entry: {json.dumps(body)}")
        return "DEBUG-MARKET-ID"
    resp = client.post("/api/v1/futures/trade/place_order", body)
    if resp.get("code") != 0:
        log.error(f"  Market entry failed: {resp.get('msg')}")
        return None
    order_id = resp.get("data", {}).get("orderId")
    log.info(f"  Market entry placed: {order_id}")
    return order_id


def place_limit_tp(client: BitunixClient, symbol: str, side: str,
                   qty: float, tp_price: float, position_id: str,
                   debug: bool) -> str | None:
    """Limit CLOSE order at TP price."""
    close_side = "SELL" if side == "LONG" else "BUY"
    body = {
        "symbol":     symbol,
        "qty":        str(qty),
        "side":       close_side,
        "orderType":  "LIMIT",
        "price":      str(round_price(symbol, tp_price)),
        "tradeSide":  "CLOSE",
        "positionId": position_id,
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


# ── Main loop ──────────────────────────────────────────────────────────────────

def run(debug: bool, symbols: list[str] = None, dir_filter: str = "BOTH") -> None:
    client = BitunixClient(API_KEY, SECRET_KEY)
    active = symbols or SYMBOLS

    log.info("━" * 60)
    log.info("  S/R Bounce Trader — Market Entry on Reversal Candle")
    log.info(f"  Symbols     : {', '.join(active)}")
    log.info(f"  TP target   : next level (fallback {TP_FIXED*100:.1f}%)  |  Hold: ≤{MAX_HOLD_MINS}min")
    log.info(f"  Min R/R     : {MIN_RR}×  |  Leverage: {LEVERAGE}×  |  Max trade: {MAX_TRADE_PCT:.0%}")
    log.info(f"  Direction   : {dir_filter}")
    log.info(f"  Fees        : maker={FEE_MAKER*100:.3f}%  taker={FEE_TAKER*100:.3f}%")
    if debug:
        log.info("  MODE        : DEBUG — no real orders")
    log.info("━" * 60)

    for sym in active:
        cfg = SYMBOL_CONFIGS.get(sym, {})
        log.info(f"  {sym}  arm={cfg.get('arm_distance',0)*100:.2f}%  "
                 f"sl_buf={cfg.get('sl_buffer',0)*100:.2f}%  "
                 f"max_bounce={cfg.get('max_bounce',0)*100:.2f}%")

    for sym in active:
        set_leverage(client, sym, debug)

    start_balance = get_balance(client)
    min_balance   = start_balance * MIN_BALANCE_PCT
    log.info(f"  Balance: {start_balance:.4f} USDT  |  Floor: {min_balance:.4f}")

    # ── Initialise per-symbol state ────────────────────────────────────────────
    sym_state: dict[str, dict] = {}
    for sym in active:
        csv_candles = load_candles_csv(sym)
        if csv_candles:
            log.info(f"  {sym}: {len(csv_candles):,} candles from local CSV "
                     f"({datetime.fromtimestamp(csv_candles[0]['time']/1000, tz=timezone.utc).strftime('%Y-%m-%d')} "
                     f"→ {datetime.fromtimestamp(csv_candles[-1]['time']/1000, tz=timezone.utc).strftime('%Y-%m-%d')})")
            log.info(f"  {sym}: fetching gap candles from API...")
            gap_candles = fetch_gap_candles(client, sym, csv_candles[-1]["time"])
            log.info(f"  {sym}: {len(gap_candles)} gap candles fetched")
            all_candles = csv_candles + gap_candles
        else:
            log.info(f"  {sym}: no local CSV — fetching {INIT_PAGES * 1000} candles from API...")
            all_candles = fetch_candles_paged(client, sym, pages=INIT_PAGES)
            log.info(f"  {sym}: {len(all_candles)} candles fetched from API")

        buf  = all_candles[-BUF_SIZE:]
        sess = init_session(buf)
        wh, wl = compute_weekly_hl(buf)
        sym_state[sym] = {
            "buf":         buf,
            "last_ts":     buf[-1]["time"] if buf else 0,
            **sess,
            "state":          "WATCHING",
            "armed_level":    None,
            "armed_side":     None,
            "arm_id":         None,
            "arm_time":       None,
            "arm_price":      None,
            "position":       None,
            "reversal_count": 0,
        }
        weekly_str = f"weekly H/L {wh:.2f}/{wl:.2f}" if wh else "weekly H/L unavailable (<7d data)"
        log.info(f"  {sym}: {len(buf)} candles loaded — {weekly_str}")

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
                     f"(note: TP order unknown after restart — time exit only)")

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
                    _process_symbol(client, sym, s, cfg, balance, debug, dir_filter)
                except Exception as e:
                    log.error(f"  {sym} error: {e}")

        except Exception as e:
            log.error(f"  Cycle error: {e}")

        elapsed = (now_utc() - cycle_start).total_seconds()
        time.sleep(max(5, POLL_SECS - elapsed))


def _process_symbol(client: BitunixClient, sym: str, s: dict, cfg: dict,
                    balance: float, debug: bool, dir_filter: str) -> None:

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
        if len(s["buf"]) > BUF_SIZE:
            s["buf"] = s["buf"][-BUF_SIZE:]
        s["last_ts"] = nc["time"]

    buf   = s["buf"]
    c     = buf[-1]
    price = c["close"]

    log.info(f"  {sym}  price={price:.4f}  state={s['state']}")

    # ── 2. State machine ───────────────────────────────────────────────────────

    if s["state"] == "IN_TRADE":
        _monitor_trade(client, sym, s, c, debug)
        return

    if s["state"] == "ARMED":
        _monitor_armed(client, sym, s, cfg, c, buf, balance, debug)
        return

    # ── WATCHING: scan levels for arm condition ────────────────────────────────
    levels = compute_levels(s, buf)
    if not levels:
        log.info(f"  {sym}: no levels computed")
        return

    # Collect candidates: LONG = price above level approaching from above
    #                     SHORT = price below level approaching from below
    candidates = []
    for lvl in levels:
        dist_pct = abs(price - lvl) / lvl
        if dist_pct > cfg["arm_distance"]:
            continue
        if price > lvl:
            candidates.append((lvl, "LONG", dist_pct))
        elif price < lvl:
            candidates.append((lvl, "SHORT", dist_pct))

    candidates.sort(key=lambda x: x[2])  # nearest first

    arm_level = None
    arm_side  = None
    for lvl, side, dist in candidates:
        if dir_filter != "BOTH" and side != dir_filter:
            log.info(f"  {sym}: {side} level={lvl:.4f} filtered — dir={dir_filter}")
            arm_log.log_arm_event(
                arm_log.new_arm_id(), STRATEGY, sym,
                now_utc().strftime("%Y-%m-%d %H:%M:%S"), price, side,
                "NO_FIRE", no_fire_reason="DIR_FILTER",
            )
            continue
        cfg_sym = SYMBOL_CONFIGS.get(sym, {})
        if not level_has_prior_test(lvl, cfg_sym.get("arm_distance", 0.001), buf):
            log.info(f"  {sym}: {side} level={lvl:.4f} skipped — no prior test in 24h")
            continue
        arm_level = lvl
        arm_side  = side
        break

    if arm_level is None:
        nearest  = min(levels, key=lambda lvl: abs(price - lvl))
        dist_pct = abs(price - nearest) / nearest * 100
        log.info(f"  {sym}: WATCHING — nearest level {nearest:.4f} ({dist_pct:.3f}% away)  "
                 f"levels=[{', '.join(f'{l:.2f}' for l in levels)}]")
        return

    s["state"]       = "ARMED"
    s["armed_level"] = arm_level
    s["armed_side"]  = arm_side
    s["arm_id"]      = arm_log.new_arm_id()
    s["arm_time"]    = now_utc().strftime("%Y-%m-%d %H:%M:%S")
    s["arm_price"]   = price
    log.info(f"  {sym}: ARMED  level={arm_level:.4f}  side={arm_side}  "
             f"dist={abs(price - arm_level)/arm_level*100:.3f}%")


def _monitor_armed(client: BitunixClient, sym: str, s: dict, cfg: dict,
                   c: dict, buf: list[dict], balance: float, debug: bool) -> None:
    """Check disarm conditions then fire condition each candle."""
    level = s["armed_level"]
    side  = s["armed_side"]
    price = c["close"]

    # Disarm: candle closed through support/resistance
    broke = (
        (side == "LONG"  and price < level * (1 - BREAK_TOLERANCE)) or
        (side == "SHORT" and price > level * (1 + BREAK_TOLERANCE))
    )
    if broke:
        log.info(f"  {sym}: DISARMED — broke through {level:.4f}  "
                 f"close={price:.4f}  [{side}]")
        arm_log.log_arm_event(
            s["arm_id"], STRATEGY, sym, s["arm_time"], s["arm_price"],
            side, "NO_FIRE", disarm_price=price, no_fire_reason="BROKE_THROUGH",
        )
        s["state"]         = "WATCHING"
        s["armed_level"]   = None
        s["armed_side"]    = None
        s["reversal_count"] = 0
        s["arm_id"] = s["arm_time"] = s["arm_price"] = None
        return

    # Disarm: price bounced too far from level without reversing
    dist_from_level = abs(price - level) / level
    drifted = (
        (side == "LONG"  and price > level and dist_from_level > cfg["max_bounce"]) or
        (side == "SHORT" and price < level and dist_from_level > cfg["max_bounce"])
    )
    if drifted:
        log.info(f"  {sym}: DISARMED — drifted {dist_from_level*100:.3f}% from "
                 f"{level:.4f} (max={cfg['max_bounce']*100:.2f}%)  [{side}]")
        arm_log.log_arm_event(
            s["arm_id"], STRATEGY, sym, s["arm_time"], s["arm_price"],
            side, "NO_FIRE", disarm_price=price, no_fire_reason="DRIFTED_AWAY",
        )
        s["state"]          = "WATCHING"
        s["armed_level"]    = None
        s["armed_side"]     = None
        s["reversal_count"] = 0
        s["arm_id"] = s["arm_time"] = s["arm_price"] = None
        return

    # Fire: require REVERSAL_REQUIRED consecutive reversal candles
    if len(buf) < 2:
        return
    prev_close = buf[-2]["close"]
    is_reversal = (
        (side == "LONG"  and price > prev_close and dist_from_level <= cfg["max_bounce"]) or
        (side == "SHORT" and price < prev_close and dist_from_level <= cfg["max_bounce"])
    )
    if is_reversal:
        s["reversal_count"] += 1
        log.info(f"  {sym}: reversal candle {s['reversal_count']}/{REVERSAL_REQUIRED} "
                 f"[{side}]  level={level:.4f}  close={price:.4f}  "
                 f"prev={prev_close:.4f}  dist={dist_from_level*100:.3f}%")
        if s["reversal_count"] >= REVERSAL_REQUIRED:
            _enter_trade(client, sym, s, cfg, c, buf, side, level, balance, debug)
        return
    else:
        if s["reversal_count"] > 0:
            log.info(f"  {sym}: reversal streak reset (was {s['reversal_count']})")
        s["reversal_count"] = 0

    log.info(f"  {sym}: armed @ {level:.4f}  close={price:.4f}  "
             f"dist={dist_from_level*100:.3f}%  [{side}]  waiting for reversal")


def _enter_trade(client: BitunixClient, sym: str, s: dict, cfg: dict,
                 c: dict, buf: list[dict], side: str, armed_level: float,
                 balance: float, debug: bool) -> None:
    """Place market entry + limit TP."""
    ticker      = fetch_ticker(client, sym)
    entry_price = float(ticker.get("lastPrice", c["close"]))

    sl_price = (armed_level * (1 - cfg["sl_buffer"])) if side == "LONG" \
               else (armed_level * (1 + cfg["sl_buffer"]))
    sl_dist  = abs(entry_price - sl_price)

    if sl_dist == 0:
        log.warning(f"  {sym}: SL distance is zero, skip")
        _abort_arm(s, sym, side, entry_price, "ZERO_SL")
        return

    # TP: next significant level beyond entry, excluding the armed level
    levels   = compute_levels(s, buf)
    tp_price = find_tp_level(entry_price, side, [l for l in levels if l != armed_level])
    tp_dist  = abs(tp_price - entry_price)

    rr = tp_dist / sl_dist
    if rr < MIN_RR:
        log.info(f"  {sym}: SKIP — R/R {rr:.2f} < {MIN_RR}  "
                 f"tp={tp_price:.4f}  sl={sl_price:.4f}")
        _abort_arm(s, sym, side, entry_price, "RR_GATE")
        return

    fee_cost = entry_price * ROUND_TRIP_FEE
    if tp_dist <= fee_cost:
        log.info(f"  {sym}: SKIP — TP dist {tp_dist:.4f} ≤ fee {fee_cost:.4f}")
        _abort_arm(s, sym, side, entry_price, "FEE_GATE")
        return

    notional = balance * MAX_TRADE_PCT * LEVERAGE
    qty      = round_qty(sym, notional / entry_price)
    min_qty  = {"BTCUSDT": 0.0001, "ETHUSDT": 0.003}.get(sym, 0.001)
    if qty < min_qty:
        log.info(f"  {sym}: qty {qty} < min {min_qty}, skip")
        _abort_arm(s, sym, side, entry_price, "MIN_QTY")
        return

    log.info(f"  {sym}: ENTER [{side}] @ ~{entry_price:.4f}  "
             f"level={armed_level:.4f}  sl={sl_price:.4f}  "
             f"tp={tp_price:.4f}  R/R={rr:.2f}  qty={qty}")

    order_id = place_market_entry(client, sym, side, qty, sl_price, debug)
    if order_id is None:
        _abort_arm(s, sym, side, entry_price, "ORDER_FAILED")
        return

    if debug:
        position_id = order_id
    else:
        position_id = resolve_position_id(client, sym, side)
        if position_id is None:
            log.warning(f"  {sym}: could not resolve positionId — using orderId")
            position_id = order_id
        else:
            log.info(f"  {sym}: positionId={position_id}")

    tp_order_id = place_limit_tp(client, sym, side, qty, tp_price, position_id, debug)

    arm_log.log_arm_event(
        s["arm_id"], STRATEGY, sym, s["arm_time"], s["arm_price"],
        side, "FIRED", disarm_price=entry_price, atr=cfg["sl_buffer"],
    )
    log_trade({
        "symbol":      sym,   "side":       side,  "qty":        qty,
        "tp_price":    tp_price, "sl_price": sl_price,
        "entry_price": entry_price,
        "arm_id":      s["arm_id"], "arm_price": s["arm_price"],
    })

    saved_arm_id     = s["arm_id"]
    s["state"]       = "IN_TRADE"
    s["armed_level"] = None
    s["armed_side"]  = None
    s["arm_id"] = s["arm_time"] = s["arm_price"] = None
    s["position"] = {
        "position_id": position_id,
        "side":        side,
        "entry_price": entry_price,
        "tp_price":    tp_price,
        "sl_price":    sl_price,
        "qty":         qty,
        "opened_at":   now_utc(),
        "debug":       debug,
        "arm_id":      saved_arm_id,
        "tp_order_id": tp_order_id,
    }
    position_registry.register(position_id, STRATEGY, sym, side, entry_price, saved_arm_id)


def _abort_arm(s: dict, sym: str, side: str, disarm_price: float, reason: str) -> None:
    arm_log.log_arm_event(
        s["arm_id"], STRATEGY, sym, s["arm_time"], s["arm_price"],
        side, "NO_FIRE", disarm_price=disarm_price, no_fire_reason=reason,
    )
    s["state"]          = "WATCHING"
    s["armed_level"]    = None
    s["armed_side"]     = None
    s["reversal_count"] = 0
    s["arm_id"] = s["arm_time"] = s["arm_price"] = None


def _monitor_trade(client: BitunixClient, sym: str, s: dict,
                   c: dict, debug: bool) -> None:
    pos         = s["position"]
    side        = pos["side"]
    held_mins   = (now_utc() - pos["opened_at"]).total_seconds() / 60
    tp          = pos["tp_price"]
    sl          = pos["sl_price"]
    tp_order_id = pos.get("tp_order_id")

    if debug:
        price  = c["close"]
        tp_hit = (c["high"] >= tp) if (side == "LONG" and tp is not None) else \
                 (c["low"]  <= tp) if (side == "SHORT" and tp is not None) else False
        sl_hit = (c["low"]  <= sl) if (side == "LONG" and sl is not None) else \
                 (c["high"] >= sl) if (side == "SHORT" and sl is not None) else False
        pnl = (price - pos["entry_price"]) * pos["qty"]
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
            exit_price = tp if outcome == "TP" else (sl if outcome == "SL" else price)
            log.info(f"  {sym} [DEBUG]: {outcome} exit @ {exit_price:.4f}  held={held_mins:.1f}min")
            fee = FEE_MAKER * 2 if outcome == "TP" else FEE_TAKER + FEE_MAKER
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
            log.warning(f"  {sym}: could not fetch close history, using candle price")

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
        fee = FEE_MAKER * 2 if outcome_str == "TP" else FEE_TAKER + FEE_MAKER
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

    p = live[0]
    if p.get("positionId") and p["positionId"] != pos["position_id"]:
        log.info(f"  {sym}: syncing positionId {pos['position_id']} → {p['positionId']}")
        pos["position_id"] = p["positionId"]
    upnl   = float(p.get("unrealizedPNL", 0))
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
                                 FEE_TAKER + FEE_MAKER),
                order_id=pos.get("position_id"),
            )
            position_registry.release(pos.get("position_id", ""))
            s["state"]    = "WATCHING"
            s["position"] = None


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="S/R Bounce trader — market entry on reversal candle at S/R levels")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode — no real orders placed")
    parser.add_argument("--symbol", nargs="+", metavar="SYMBOL",
                        help=f"Symbol(s) to trade (default: {', '.join(SYMBOLS)})")
    parser.add_argument("--dir", choices=["LONG", "SHORT", "BOTH"], default="BOTH",
                        metavar="DIR", help="Direction filter: LONG, SHORT, or BOTH (default)")
    args = parser.parse_args()
    run(debug=args.debug, symbols=args.symbol, dir_filter=args.dir)


if __name__ == "__main__":
    main()
