"""
funding_analysis.py — Does extreme funding rate predict short-term price direction?

For each historical funding settlement on BTCUSDT / ETHUSDT:
  - Records the funding rate and mark price at settlement
  - Measures forward price returns at +15m, +30m, +1h, +4h using candle data
  - Groups results by funding rate bucket and reports average return + win rate

Also produces three chart panels:
  1. Price over time with funding settlements colour-coded by magnitude
  2. Scatter — funding rate vs 1h forward return
  3. Bar — average forward return by bucket across all horizons

Usage:
    python3 funding_analysis.py               # BTC + ETH
    python3 funding_analysis.py --symbol BTC  # one symbol
    python3 funding_analysis.py --no-chart    # table only
"""

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_URL   = "https://fapi.bitunix.com"
DATA_DIR   = Path("data")
SYMBOLS    = ["BTCUSDT", "ETHUSDT"]

HORIZONS   = [15, 30, 60, 240]          # minutes ahead to measure
HOR_LABELS = ["15m", "30m", "1h", "4h"]

# Funding rate bucket edges (as fractions, e.g. 0.0003 = 0.03%)
BUCKET_EDGES = [-0.0005, -0.0002, -0.00005, 0.00005, 0.0002, 0.0005]
BUCKET_LABELS = [
    "≤ −0.05%",
    "−0.05% to −0.02%",
    "−0.02% to −0.005%",
    "±0.005% (neutral)",
    "+0.005% to +0.02%",
    "+0.02% to +0.05%",
    "≥ +0.05%",
]


# ── Funding history fetch ───────────────────────────────────────────────────────

def fetch_funding_history(symbol: str) -> list[dict]:
    """Page through the funding history endpoint, oldest-first, all available."""
    records  = []
    end_time = int(time.time() * 1000)

    while True:
        params = {"symbol": symbol, "endTime": end_time, "limit": 200}
        try:
            r = requests.get(
                f"{BASE_URL}/api/v1/futures/market/get_funding_rate_history",
                params=params, timeout=10,
            )
            data = r.json()
        except Exception as e:
            print(f"  [{symbol}] fetch error: {e}")
            break

        if data.get("code") != 0:
            print(f"  [{symbol}] API error: {data.get('msg')}")
            break

        batch = data.get("data") or []
        if not batch:
            break

        records.extend(batch)
        oldest = min(int(rec["fundingTime"]) for rec in batch)

        # Stop if this batch is older than our candle data or we got a partial page
        if len(batch) < 200:
            break

        end_time = oldest - 1
        time.sleep(0.15)   # stay well within 10 req/s

    # Sort oldest → newest, deduplicate
    seen, unique = set(), []
    for rec in sorted(records, key=lambda r: int(r["fundingTime"])):
        ft = int(rec["fundingTime"])
        if ft not in seen:
            seen.add(ft)
            unique.append({
                "funding_time": ft,
                "funding_rate": float(rec["fundingRate"]),
                "mark_price":   float(rec["markPrice"]),
            })

    return unique


# ── Candle loading ─────────────────────────────────────────────────────────────

def load_candles(symbol: str) -> dict[int, float]:
    """Return {timestamp_ms: close_price} for every 1m candle."""
    path = DATA_DIR / f"{symbol}_1m.csv"
    if not path.exists():
        raise FileNotFoundError(f"No candle file: {path}")
    candles = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles[int(row["timestamp_ms"])] = float(row["close"])
    return candles


def price_at(candles: dict[int, float], target_ms: int,
             tolerance_ms: int = 90_000) -> float | None:
    """Find the close price of the 1m candle nearest to target_ms."""
    minute_ms = 60_000
    # Try exact then ±1 candle
    for offset in range(0, tolerance_ms // minute_ms + 1):
        for sign in (0, 1, -1):
            ts = ((target_ms // minute_ms) * minute_ms) + sign * offset * minute_ms
            if ts in candles:
                return candles[ts]
    return None


# ── Bucketing ──────────────────────────────────────────────────────────────────

def bucket(rate: float) -> int:
    for i, edge in enumerate(BUCKET_EDGES):
        if rate < edge:
            return i
    return len(BUCKET_EDGES)


# ── Analysis ───────────────────────────────────────────────────────────────────

def analyse(symbol: str) -> tuple[list[dict], dict]:
    print(f"\n{'━'*60}")
    print(f"  {symbol}")
    print(f"{'━'*60}")

    print("  Fetching funding history...")
    funding = fetch_funding_history(symbol)
    print(f"  {len(funding)} settlement records fetched")

    print("  Loading candle data...")
    candles = load_candles(symbol)
    candle_start = min(candles)
    candle_end   = max(candles)
    print(f"  {len(candles):,} candles  "
          f"({datetime.fromtimestamp(candle_start/1000, tz=timezone.utc).strftime('%Y-%m-%d')} → "
          f"{datetime.fromtimestamp(candle_end/1000, tz=timezone.utc).strftime('%Y-%m-%d')})")

    # Filter funding records to candle range (leave room for 4h forward)
    max_horizon_ms = max(HORIZONS) * 60_000
    rows = []
    skipped = 0
    for rec in funding:
        ft = rec["funding_time"]
        if ft < candle_start or ft + max_horizon_ms > candle_end:
            skipped += 1
            continue
        p0 = price_at(candles, ft)
        if p0 is None:
            skipped += 1
            continue

        fwd = {}
        for mins, label in zip(HORIZONS, HOR_LABELS):
            p1 = price_at(candles, ft + mins * 60_000)
            fwd[label] = (p1 / p0 - 1) * 100 if p1 else None   # % return

        rows.append({
            "funding_time": ft,
            "funding_rate": rec["funding_rate"],
            "mark_price":   rec["mark_price"],
            "bucket":       bucket(rec["funding_rate"]),
            **fwd,
        })

    print(f"  {len(rows)} settlements with full forward data  ({skipped} skipped)")

    # ── Per-bucket summary ────────────────────────────────────────────────────
    buckets: dict[int, list[dict]] = {i: [] for i in range(len(BUCKET_LABELS))}
    for row in rows:
        buckets[row["bucket"]].append(row)

    print(f"\n  {'Bucket':<28} {'N':>5}  " +
          "  ".join(f"{'avg ' + l:>9}  {'win%':>5}" for l in HOR_LABELS))
    print("  " + "─" * 100)

    summary = {}
    for i, label in enumerate(BUCKET_LABELS):
        grp = buckets[i]
        if not grp:
            continue
        n = len(grp)
        stats = {}
        parts = []
        for l in HOR_LABELS:
            vals = [r[l] for r in grp if r[l] is not None]
            if vals:
                avg  = sum(vals) / len(vals)
                wins = sum(1 for v in vals if v > 0)
                win_pct = wins / len(vals) * 100
                stats[l] = {"avg": avg, "win_pct": win_pct, "n": len(vals)}
                parts.append(f"{avg:+8.3f}%  {win_pct:5.1f}%")
            else:
                parts.append(f"{'—':>9}  {'—':>5}")
        summary[label] = stats
        print(f"  {label:<28} {n:>5}  " + "  ".join(parts))

    return rows, summary


# ── Charts ─────────────────────────────────────────────────────────────────────

def plot(symbol: str, rows: list[dict], candles: dict[int, float]) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import numpy as np
    except ImportError:
        print("  matplotlib/numpy not installed — skipping chart")
        return

    rates  = [r["funding_rate"] for r in rows]
    times  = [datetime.fromtimestamp(r["funding_time"] / 1000, tz=timezone.utc) for r in rows]
    fwd_1h = [r["1h"] for r in rows]

    # Colour map: blue = negative funding, red = positive funding
    rate_arr = np.array(rates)
    max_abs  = max(abs(rate_arr.max()), abs(rate_arr.min()), 1e-8)
    norm     = rate_arr / max_abs   # −1 … +1

    def rate_colour(n: float) -> tuple:
        # negative → blue, zero → grey, positive → red
        if n < 0:
            return (0.2, 0.4, 1.0, min(0.9, 0.3 + abs(n) * 0.7))
        else:
            return (1.0, 0.2, 0.2, min(0.9, 0.3 + abs(n) * 0.7))

    colours = [rate_colour(n) for n in norm]

    fig = plt.figure(figsize=(16, 12), facecolor="#0c0e14")
    fig.suptitle(f"{symbol} — Funding Rate Edge Analysis",
                 color="#f1f5f9", fontsize=14, fontweight="bold", y=0.98)

    ax_style = dict(facecolor="#151922", labelcolor="#cbd5e1",
                    titlecolor="#f1f5f9")

    # ── Panel 1: price + funding dots ────────────────────────────────────────
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_facecolor(ax_style["facecolor"])

    # Thin price line — downsample to hourly for speed
    sorted_ts = sorted(candles)
    hour_ms   = 3_600_000
    hourly_ts = sorted_ts[::60]
    px_times  = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts in hourly_ts]
    px_prices = [candles[ts] for ts in hourly_ts]
    ax1.plot(px_times, px_prices, color="#475569", linewidth=0.6, alpha=0.8, zorder=1)

    # Funding dots
    for t, rate, col in zip(times, rates, colours):
        p = price_at(candles, int(t.timestamp() * 1000))
        if p:
            ax1.scatter(t, p, color=col, s=30, zorder=2, linewidths=0)

    ax1.set_title("Price with funding settlements  (blue = negative, red = positive)",
                  color="#f1f5f9", fontsize=9, pad=6)
    ax1.tick_params(colors="#64748b", labelsize=8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    for spine in ax1.spines.values():
        spine.set_edgecolor("#1e2533")

    # ── Panel 2: scatter — funding rate vs 1h return ─────────────────────────
    ax2 = fig.add_subplot(3, 2, 3)
    ax2.set_facecolor(ax_style["facecolor"])

    valid = [(r * 100, f) for r, f in zip(rates, fwd_1h) if f is not None]
    if valid:
        xv, yv = zip(*valid)
        xv_arr, yv_arr = np.array(xv), np.array(yv)
        pt_col = ["#ef4444" if x > 0 else "#3b82f6" for x in xv_arr]
        ax2.scatter(xv_arr, yv_arr, c=pt_col, s=12, alpha=0.5, linewidths=0)
        ax2.axhline(0, color="#475569", linewidth=0.8, linestyle="--")
        ax2.axvline(0, color="#475569", linewidth=0.8, linestyle="--")

        # Linear trend line
        z = np.polyfit(xv_arr, yv_arr, 1)
        p = np.poly1d(z)
        xs = np.linspace(xv_arr.min(), xv_arr.max(), 100)
        ax2.plot(xs, p(xs), color="#a78bfa", linewidth=1.5, label=f"slope {z[0]:.2f}")
        ax2.legend(fontsize=8, facecolor="#1e2533", labelcolor="#cbd5e1")

    ax2.set_xlabel("Funding rate (%)", color="#94a3b8", fontsize=8)
    ax2.set_ylabel("1h forward return (%)", color="#94a3b8", fontsize=8)
    ax2.set_title("Funding rate vs 1h forward return", color="#f1f5f9", fontsize=9, pad=6)
    ax2.tick_params(colors="#64748b", labelsize=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#1e2533")

    # ── Panel 3: bar — avg return by bucket ──────────────────────────────────
    ax3 = fig.add_subplot(3, 2, 4)
    ax3.set_facecolor(ax_style["facecolor"])

    bucket_avgs = {l: [] for l in HOR_LABELS}
    bucket_ns   = []
    xlabels     = []
    for i, label in enumerate(BUCKET_LABELS):
        grp = [r for r in rows if r["bucket"] == i]
        if not grp:
            continue
        xlabels.append(label)
        bucket_ns.append(len(grp))
        for l in HOR_LABELS:
            vals = [r[l] for r in grp if r[l] is not None]
            bucket_avgs[l].append(sum(vals) / len(vals) if vals else 0)

    x     = np.arange(len(xlabels))
    width = 0.2
    hor_cols = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444"]
    for j, (l, col) in enumerate(zip(HOR_LABELS, hor_cols)):
        ax3.bar(x + j * width, bucket_avgs[l], width, label=l,
                color=col, alpha=0.75)

    ax3.axhline(0, color="#475569", linewidth=0.8)
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=7, color="#64748b")
    ax3.set_ylabel("Avg forward return (%)", color="#94a3b8", fontsize=8)
    ax3.set_title("Avg return by funding bucket & horizon",
                  color="#f1f5f9", fontsize=9, pad=6)
    ax3.legend(fontsize=8, facecolor="#1e2533", labelcolor="#cbd5e1")
    ax3.tick_params(colors="#64748b", labelsize=8)
    for spine in ax3.spines.values():
        spine.set_edgecolor("#1e2533")

    # ── Panel 4: funding rate over time ──────────────────────────────────────
    ax4 = fig.add_subplot(3, 1, 3)
    ax4.set_facecolor(ax_style["facecolor"])

    rate_pct = [r * 100 for r in rates]
    bar_cols = ["#ef4444" if r > 0 else "#3b82f6" for r in rates]
    ax4.bar(times, rate_pct, color=bar_cols, alpha=0.7,
            width=0.28, linewidth=0)
    ax4.axhline(0, color="#475569", linewidth=0.8)
    ax4.set_ylabel("Funding rate (%)", color="#94a3b8", fontsize=8)
    ax4.set_title("Funding rate over time  (red = longs pay, blue = shorts pay)",
                  color="#f1f5f9", fontsize=9, pad=6)
    ax4.tick_params(colors="#64748b", labelsize=8)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    for spine in ax4.spines.values():
        spine.set_edgecolor("#1e2533")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path("log") / f"funding_analysis_{symbol}.png"
    plt.savefig(out, dpi=140, facecolor=fig.get_facecolor())
    plt.show()
    print(f"  Chart saved → {out}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Funding rate edge analysis")
    parser.add_argument("--symbol", metavar="SYM",
                        help="Single symbol base, e.g. BTC or ETH (default: both)")
    parser.add_argument("--no-chart", action="store_true",
                        help="Skip chart output")
    args = parser.parse_args()

    if args.symbol:
        sym = args.symbol.upper()
        if not sym.endswith("USDT"):
            sym += "USDT"
        symbols = [sym]
    else:
        symbols = SYMBOLS

    for symbol in symbols:
        rows, summary = analyse(symbol)
        if not args.no_chart and rows:
            candles = load_candles(symbol)
            plot(symbol, rows, candles)


if __name__ == "__main__":
    main()
