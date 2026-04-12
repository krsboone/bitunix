"""
exhaustion_trader.py — Bitunix momentum exhaustion reversion perpetual futures trader

Strategy: Momentum exhaustion reversal (from exhaustion_sim.py)
  - Monitor live 1m candles for N consecutive same-direction candles (streak)
    whose FINAL candle has elevated volume (≥ VOL_MULT × rolling average).
  - Signal means the momentum is climactically exhausted — enter in the OPPOSITE
    direction expecting reversal:
      Up streak + high-vol finale   → SHORT
      Down streak + high-vol finale → LONG
  - TP/SL sized in ATR units (Average True Range over ATR_PERIOD candles):
      TP = entry ± ATR × TP_MULT
      SL = entry ∓ ATR × SL_MULT
  - Directional cooldown: block same-direction re-entry for N candles after SL
  - Time exit: close if held > MAX_HOLD_MINS

Data-derived signal: sweep showed 54.7% TP at equal TP/SL sizing (1:1 ATR)
at 5,620 trades. Positive EV signal from exhaustion_sim.py analysis on 90
days of 1m OHLCV data.

Sweep-validated parameters (180d + 161d + 90d cross-check):
  Shared: streak=4  vol_mult=1.5  vol_lookback=20  atr_period=14
          tp_mult=4.5  sl_mult=0.2  cooldown=10
  ETHUSDT: 22:00–23:00 UTC all days
  BTCUSDT: windows=[] — commented out pending further sweep analysis

Usage:
    python3 exhaustion_trader.py          # live trading
    python3 exhaustion_trader.py --debug  # DEBUG mode — no real orders
    python3 exhaustion_trader.py --symbol BTCUSDT
"""

import argparse
import csv
import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone

from auth import BitunixClient
from config import API_KEY, SECRET_KEY
from market import fetch_ticker
from log_cap import start_logging

start_logging("exhaustion_trader")

# ── Strategy parameters ────────────────────────────────────────────────────────

SYMBOLS      = ["BTCUSDT", "ETHUSDT"]

STREAK_LEN    = 4       # consecutive same-direction candles required
VOL_MULT      = 1.5    # final candle volume must be ≥ vol_mult × rolling avg
VOL_LOOKBACK  = 20     # candles for rolling volume average
ATR_PERIOD    = 14     # candles for ATR calculation
TP_MULT       = 4.5    # TP = entry ± atr × tp_mult
SL_MULT       = 0.2    # SL = entry ∓ atr × sl_mult
MAX_HOLD_MINS = 33
COOLDOWN      = 10     # candles to block same-direction re-entry after SL

# ATR filter: only enter when current ATR ≥ ATR_FILTER_THRESH × rolling ATR average
# Set to None to disable
ATR_FILTER_THRESH   = 1.5
ATR_FILTER_LOOKBACK = 50   # candles of ATR history to average

MIN_HISTORY  = max(VOL_LOOKBACK, ATR_PERIOD, STREAK_LEN) + 5

# Per-symbol time windows (UTC). windows = list of (start_hour, end_hour) tuples.
# skip_days: 0=Mon … 6=Sun. Empty windows list = trade all hours.
SYMBOL_CONFIGS = {
    "BTCUSDT": {
        "windows":   [],        # no validated window yet — trades all hours until confirmed
        "skip_days": set(),
    },
    "ETHUSDT": {
        "windows":   [(22, 23), (23, 0)],  # 22:00–00:00 UTC all days
        "skip_days": set(),
    },
}

# ── Trading parameters ─────────────────────────────────────────────────────────

LEVERAGE         = 2
MARGIN_COIN      = "USDT"
INTERVAL         = "1m"
MAX_TRADE_PCT    = 0.10
POLL_SECS        = 60
INIT_PAGES       = 2       # pages × 1000 1m candles on startup (~33h)

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

TRADE_CSV = os.path.join("log", "exhaustion_trades.csv")


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


# ── Signal computation ─────────────────────────────────────────────────────────

def compute_atr(window: list[dict]) -> float:
    """ATR over the last ATR_PERIOD candles in window (needs ATR_PERIOD+1 items)."""
    if len(window) < ATR_PERIOD + 1:
        return 0.0
    relevant = window[-(ATR_PERIOD + 1):]
    true_ranges = []
    for j in range(1, len(relevant)):
        prev_close = relevant[j - 1]["close"]
        high = relevant[j]["high"]
        low  = relevant[j]["low"]
        tr   = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0


def check_exhaustion_signal(streak_window: list[dict],
                             vol_window: deque) -> int | None:
    """
    Check whether the last STREAK_LEN candles form an exhaustion signal.
    Returns +1 (up streak exhausted → SHORT), -1 (down streak → LONG), or None.
    """
    if len(streak_window) < STREAK_LEN:
        return None

    recent = streak_window[-STREAK_LEN:]
    dirs = []
    for c in recent:
        body = c["close"] - c["open"]
        if body > 0:
            dirs.append(1)
        elif body < 0:
            dirs.append(-1)
        else:
            return None   # doji breaks streak
    if len(set(dirs)) != 1:
        return None   # mixed directions

    streak_dir = dirs[-1]

    # Final candle volume must be elevated
    prior_vols = [v for v in list(vol_window)[:-1] if v > 0]
    if not prior_vols:
        return None
    avg_vol   = sum(prior_vols) / len(prior_vols)
    final_vol = recent[-1]["volume"]
    if avg_vol == 0 or final_vol < avg_vol * VOL_MULT:
        return None

    return streak_dir   # +1 = up streak done, -1 = down streak done


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
        "symbol",     body["symbol"],
        "qty",        body["qty"],
        "side",       buy_sell,
        "orderType",  "MARKET",
        "tradeSide",  "OPEN",
        "tpPrice",    body["tp_price"],
        "slPrice",    body["sl_price"],
        "tpStopType", "MARK_PRICE",
        "slStopType", "MARK_PRICE",
        "entryPrice", body["entry_price"],
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
    # ── 1. Fetch new 1m candles ────────────────────────────────────────────────
    recent = fetch_latest_candles(client, sym, n=10)
    new    = [c for c in recent if c["time"] > s["last_ts"]]
    new.sort(key=lambda c: c["time"])

    if not new:
        log.info(f"  {sym}: no new candle yet")
        return

    for c1 in new:
        s["vol_window"].append(c1["volume"])
        s["atr_window"].append(c1)
        s["atr_filter_window"].append(c1)
        s["streak_window"].append(c1)
        s["last_ts"] = c1["time"]

    candle_count = len(s["vol_window"]) + s["candles_seen"]

    # ── 2. IN_TRADE: monitor on latest 1m candle ──────────────────────────────
    if s["state"] == "IN_TRADE":
        _monitor_trade(client, sym, s, new[-1], debug)
        return

    # ── 3. ENTRY_PENDING: enter at open of next candle ─────────────────────────
    if s["state"] == "ENTRY_PENDING":
        _enter_pending(client, sym, s, new[-1], balance, debug)
        return

    # ── 4. Need warm-up ────────────────────────────────────────────────────────
    if candle_count < MIN_HISTORY:
        log.info(f"  {sym}: warming up ({candle_count}/{MIN_HISTORY} candles)")
        return

    # Decrement cooldowns on each new candle
    for _ in new:
        if s["long_blocked"]  > 0: s["long_blocked"]  -= 1
        if s["short_blocked"] > 0: s["short_blocked"] -= 1

    # ── 5. WATCHING: check signal on each new candle ───────────────────────────
    cfg = SYMBOL_CONFIGS.get(sym, {})
    for c1 in new:
        if s["state"] != "WATCHING":
            break

        # Time window filter
        candle_dt   = datetime.fromtimestamp(c1["time"] / 1000, tz=timezone.utc)
        candle_hour = candle_dt.hour
        candle_dow  = candle_dt.weekday()
        windows = cfg.get("windows")
        if windows:
            in_window = False
            for sh, eh in windows:
                if sh <= eh:
                    if sh <= candle_hour < eh:
                        in_window = True; break
                else:
                    if candle_hour >= sh or candle_hour < eh:
                        in_window = True; break
            if not in_window:
                log.info(f"  {sym}: outside trade window "
                         f"({candle_dt.strftime('%a %H:%M')} UTC) — skip")
                continue
        if candle_dow in cfg.get("skip_days", set()):
            log.info(f"  {sym}: skip day ({candle_dt.strftime('%a')} UTC) — skip")
            continue

        # ATR sizing
        atr_list = list(s["atr_window"])[:-1]
        atr = compute_atr(atr_list)
        if atr == 0:
            continue

        # ATR filter: skip if volatility is below threshold
        if ATR_FILTER_THRESH is not None:
            fw = list(s["atr_filter_window"])
            needed = ATR_PERIOD + ATR_FILTER_LOOKBACK + 1
            if len(fw) >= needed:
                atr_series = []
                for k in range(1, ATR_FILTER_LOOKBACK + 1):
                    end = len(fw) - k
                    sub = fw[max(0, end - ATR_PERIOD - 1): end]
                    a = compute_atr(sub)
                    if a > 0:
                        atr_series.append(a)
                if atr_series:
                    atr_avg = sum(atr_series) / len(atr_series)
                    if atr < atr_avg * ATR_FILTER_THRESH:
                        log.info(f"  {sym}: ATR filter — atr={atr:.4f} < "
                                 f"{ATR_FILTER_THRESH}× avg={atr_avg:.4f} — skip")
                        continue

        streak_dir = check_exhaustion_signal(list(s["streak_window"]), s["vol_window"])
        if streak_dir is None:
            continue

        if streak_dir == 1:
            # Up streak exhausted → SHORT
            if s["short_blocked"] > 0:
                log.info(f"  {sym}: exhaustion signal blocked [SHORT cooldown]")
                continue
            s["pending_side"] = "SHORT"
            s["pending_atr"]  = atr
        else:
            # Down streak exhausted → LONG
            if s["long_blocked"] > 0:
                log.info(f"  {sym}: exhaustion signal blocked [LONG cooldown]")
                continue
            s["pending_side"] = "LONG"
            s["pending_atr"]  = atr

        final_vol   = list(s["streak_window"])[-1]["volume"]
        prior_vols  = [v for v in list(s["vol_window"])[:-1] if v > 0]
        vol_ratio   = final_vol / (sum(prior_vols) / len(prior_vols)) if prior_vols else 0
        direction   = "UP" if streak_dir == 1 else "DOWN"
        log.info(f"  {sym}: {STREAK_LEN}-candle {direction} streak exhausted  "
                 f"vol={vol_ratio:.1f}x avg  atr={atr:.4f}  "
                 f"→ ENTRY_PENDING {s['pending_side']}")
        s["state"] = "ENTRY_PENDING"


def _enter_pending(client: BitunixClient, sym: str, s: dict,
                   c1: dict, balance: float, debug: bool) -> None:
    side = s["pending_side"]
    atr  = s["pending_atr"]

    # Use live ticker for entry price
    ticker      = fetch_ticker(client, sym)
    entry_price = float(ticker.get("lastPrice", c1["open"]))

    # TP/SL anchored to actual entry price
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
        log.info(f"  {sym}: SKIP — TP dist {tp_dist:.4f} ≤ fee {fee_cost:.4f}")
        s["state"] = "WATCHING"
        s["pending_side"] = s["pending_atr"] = None
        return

    notional = balance * MAX_TRADE_PCT * LEVERAGE
    qty      = round_qty(sym, notional / entry_price)
    min_qty  = {"BTCUSDT": 0.0001, "ETHUSDT": 0.003}.get(sym, 0.001)
    if qty < min_qty:
        log.info(f"  {sym}: qty {qty} < min {min_qty}, skip")
        s["state"] = "WATCHING"
        s["pending_side"] = s["pending_atr"] = None
        return

    log.info(f"  {sym}: ENTER {side}  entry≈{entry_price:.4f}  "
             f"tp={tp_price:.4f}  sl={sl_price:.4f}  qty={qty}")

    order_id = place_order(client, sym, side, qty, tp_price, sl_price, debug)
    if order_id is None:
        s["state"] = "WATCHING"
        s["pending_side"] = s["pending_atr"] = None
        return

    log_trade({
        "symbol":      sym,
        "side":        side,
        "qty":         qty,
        "tp_price":    round_price(sym, tp_price),
        "sl_price":    round_price(sym, sl_price),
        "entry_price": entry_price,
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
    s["pending_side"] = s["pending_atr"] = None


def _monitor_trade(client: BitunixClient, sym: str, s: dict,
                   c1: dict, debug: bool) -> None:
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
                if side == "LONG": s["long_blocked"]  = COOLDOWN
                else:              s["short_blocked"] = COOLDOWN
            s["state"]    = "WATCHING"
            s["position"] = None
        return

    # Live: check exchange
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


# ── Main loop ──────────────────────────────────────────────────────────────────

def run(debug: bool, symbols: list[str] = None) -> None:
    client = BitunixClient(API_KEY, SECRET_KEY)
    active = symbols or SYMBOLS

    log.info("━" * 60)
    log.info("  Momentum Exhaustion Reversion Trader")
    log.info(f"  Symbols    : {', '.join(active)}")
    log.info(f"  Streak     : {STREAK_LEN} candles  Vol mult: {VOL_MULT}×  Lookback: {VOL_LOOKBACK}")
    log.info(f"  ATR period : {ATR_PERIOD}  TP/SL: {TP_MULT}/{SL_MULT}×ATR")
    if ATR_FILTER_THRESH is not None:
        log.info(f"  ATR filter : ≥ {ATR_FILTER_THRESH}× rolling avg  "
                 f"(lookback: {ATR_FILTER_LOOKBACK} candles)")
    log.info(f"  Cooldown   : {COOLDOWN} candles after SL")
    log.info(f"  Hold       : ≤{MAX_HOLD_MINS}min  |  Leverage: {LEVERAGE}×")
    log.info(f"  Max trade  : {MAX_TRADE_PCT:.0%}  |  Fees: {ROUND_TRIP_FEE*100:.3f}%")
    for sym in active:
        cfg = SYMBOL_CONFIGS.get(sym, {})
        windows = cfg.get("windows", [])
        skip    = cfg.get("skip_days", set())
        day_str = "all days" if not skip else f"skip {', '.join(str(d) for d in skip)}"
        win_str = ", ".join(f"{sh:02d}:00–{eh:02d}:00" for sh, eh in windows) if windows else "all hours"
        log.info(f"  {sym:<10}: {win_str} UTC  ({day_str})")
    if debug:
        log.info("  MODE       : DEBUG — no real orders")
    log.info("━" * 60)

    for sym in active:
        set_leverage(client, sym, debug)

    start_balance = get_balance(client)
    min_balance   = start_balance * MIN_BALANCE_PCT
    log.info(f"  Balance: {start_balance:.4f} USDT  |  Floor: {min_balance:.4f}")

    # ── Initialise per-symbol state ────────────────────────────────────────────
    sym_state: dict[str, dict] = {}
    for sym in active:
        log.info(f"  {sym}: fetching history for warm-up...")
        candles_1m = fetch_candles_paged(client, sym, pages=INIT_PAGES)

        # Pre-fill rolling windows from history
        vol_window        = deque(maxlen=VOL_LOOKBACK)
        atr_window        = deque(maxlen=ATR_PERIOD + 2)
        atr_filter_window = deque(maxlen=ATR_PERIOD + ATR_FILTER_LOOKBACK + 2)
        streak_window     = deque(maxlen=STREAK_LEN)
        for c in candles_1m:
            vol_window.append(c["volume"])
            atr_window.append(c)
            atr_filter_window.append(c)
            streak_window.append(c)

        sym_state[sym] = {
            "vol_window":        vol_window,
            "atr_window":        atr_window,
            "atr_filter_window": atr_filter_window,
            "streak_window":     streak_window,
            "candles_seen":  len(candles_1m),
            "last_ts":       candles_1m[-1]["time"] if candles_1m else 0,
            # State machine
            "state":         "WATCHING",
            "pending_side":  None,
            "pending_atr":   None,
            "position":      None,
            "long_blocked":  0,
            "short_blocked": 0,
        }
        log.info(f"  {sym}: {len(candles_1m)} candles loaded — ready")

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
        description="Momentum exhaustion reversion perpetual futures trader")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode — no real orders placed")
    parser.add_argument("--symbol", nargs="+", metavar="SYMBOL",
                        help=f"Symbol(s) to trade (default: {', '.join(SYMBOLS)})")
    args = parser.parse_args()
    run(debug=args.debug, symbols=args.symbol)


if __name__ == "__main__":
    main()
