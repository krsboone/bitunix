#!/usr/bin/env python3
"""
dashboard.py — Generate browser-based signal dashboard

Reads:  log/arm_events.csv
        log/close_events.csv  (optional)
Writes: log/dashboard.html

Usage:
    python3 dashboard.py
    python3 dashboard.py --open     # also opens in browser
    python3 dashboard.py --out path/to/custom.html
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict, Counter
from datetime import datetime, timezone

ARM_CSV   = os.path.join("log", "arm_events.csv")
CLOSE_CSV = os.path.join("log", "close_events.csv")
OUT_HTML  = os.path.join("log", "dashboard.html")

STRATEGY_COLORS = {
    "bb":         "#3b82f6",   # blue
    "sr":         "#10b981",   # emerald
    "sr_rt":      "#06b6d4",   # cyan
    "ema_trend":  "#8b5cf6",   # violet
    "vol_spike":  "#f59e0b",   # amber
    "exhaustion": "#ec4899",   # pink
}
STRATEGY_ORDER = ["bb", "sr", "sr_rt", "ema_trend", "vol_spike", "exhaustion"]


# ── helpers ───────────────────────────────────────────────────────────────────

def read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def safe_float(v, default=None):
    try:
        return float(v) if v else default
    except (ValueError, TypeError):
        return default


def pct(n: int, d: int) -> float:
    return round(n / d * 100, 1) if d else 0.0


def median(vals: list[float]) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def color(strat: str) -> str:
    return STRATEGY_COLORS.get(strat, "#94a3b8")


# ── data computation ──────────────────────────────────────────────────────────

def compute(arms: list[dict], closes: list[dict]) -> dict:
    strats = [s for s in STRATEGY_ORDER if any(r["strategy"] == s for r in arms)]

    shadow_rows = [r for r in arms if r["shadow"] == "1"]
    real_rows   = [r for r in arms if r["shadow"] == "0"]
    fired_rows  = [r for r in real_rows if r["outcome"] == "FIRED"]
    nofire_rows = [r for r in real_rows if r["outcome"] == "NO_FIRE"]

    # ── Strategy funnel ────────────────────────────────────────────────────────
    funnel = {}
    for s in strats:
        s_all    = [r for r in arms      if r["strategy"] == s]
        s_shadow = [r for r in shadow_rows if r["strategy"] == s]
        s_real   = [r for r in real_rows   if r["strategy"] == s]
        s_fired  = [r for r in fired_rows  if r["strategy"] == s]
        s_nofire = [r for r in nofire_rows if r["strategy"] == s]
        funnel[s] = {
            "total":  len(s_all),
            "shadow": len(s_shadow),
            "real":   len(s_real),
            "fired":  len(s_fired),
            "nofire": len(s_nofire),
        }

    # ── No-fire reasons ────────────────────────────────────────────────────────
    reasons_by_strat = defaultdict(Counter)
    for r in nofire_rows:
        for reason in r["no_fire_reason"].split("|"):
            if reason:
                reasons_by_strat[r["strategy"]][reason] += 1
    all_reasons = sorted({reason for ctr in reasons_by_strat.values() for reason in ctr})

    # ── Hourly signal counts ───────────────────────────────────────────────────
    hour_strat = defaultdict(lambda: defaultdict(int))
    for r in arms:
        try:
            dt = datetime.strptime(r["arm_time"], "%Y-%m-%d %H:%M:%S")
            hour_strat[dt.hour][r["strategy"]] += 1
        except ValueError:
            pass
    # Fill all 24 hours
    hours_data = {h: dict(hour_strat.get(h, {})) for h in range(24)}

    # ── Direction balance ──────────────────────────────────────────────────────
    direction_by_strat = {}
    for s in strats:
        s_rows = [r for r in arms if r["strategy"] == s]
        longs  = sum(1 for r in s_rows if r["direction"] == "LONG")
        shorts = sum(1 for r in s_rows if r["direction"] == "SHORT")
        direction_by_strat[s] = {"LONG": longs, "SHORT": shorts}

    # ── ATR stats by strategy ─────────────────────────────────────────────────
    atr_stats = {}
    for s in strats:
        vals = [safe_float(r["atr"]) for r in arms
                if r["strategy"] == s and safe_float(r["atr"])]
        if vals:
            s_sorted = sorted(vals)
            n = len(s_sorted)
            atr_stats[s] = {
                "min":    round(s_sorted[0], 6),
                "p25":    round(s_sorted[max(0, n // 4)], 6),
                "median": round(s_sorted[n // 2], 6),
                "p75":    round(s_sorted[min(n - 1, 3 * n // 4)], 6),
                "max":    round(s_sorted[-1], 6),
                "count":  n,
            }

    # ── Shadow event TP/SL distance (as % of arm_price) ───────────────────────
    shadow_stats = {}
    for s in strats:
        tp_pcts, sl_pcts = [], []
        s_shadow = [r for r in shadow_rows if r["strategy"] == s]
        for r in s_shadow:
            price = safe_float(r["arm_price"])
            wtp   = safe_float(r["would_be_tp"])
            wsl   = safe_float(r["would_be_sl"])
            if price and price > 0:
                if wtp:
                    tp_pcts.append(abs(wtp - price) / price * 100)
                if wsl:
                    sl_pcts.append(abs(wsl - price) / price * 100)
        resolved = [r for r in s_shadow if r["outcome"] in ("TP", "SL", "TIME")]
        tp_n     = sum(1 for r in resolved if r["outcome"] == "TP")
        sl_n     = sum(1 for r in resolved if r["outcome"] == "SL")
        tx_n     = sum(1 for r in resolved if r["outcome"] == "TIME")
        pending_n = sum(1 for r in s_shadow if r["outcome"] == "PENDING")
        if s_shadow:
            shadow_stats[s] = {
                "count":      len(s_shadow),
                "resolved":   len(resolved),
                "pending":    pending_n,
                "tp_n":       tp_n,
                "sl_n":       sl_n,
                "tx_n":       tx_n,
                "tp_pct":     pct(tp_n, len(resolved)),
                "sl_pct":     pct(sl_n, len(resolved)),
                "tx_pct":     pct(tx_n, len(resolved)),
                "avg_tp_pct": round(sum(tp_pcts) / len(tp_pcts), 4) if tp_pcts else None,
                "avg_sl_pct": round(sum(sl_pcts) / len(sl_pcts), 4) if sl_pcts else None,
            }

    # ── Price slippage for FIRED events ───────────────────────────────────────
    slippage = []
    for r in fired_rows:
        arm_p  = safe_float(r["arm_price"])
        fill_p = safe_float(r["disarm_price"])
        if arm_p and fill_p and arm_p > 0:
            slippage.append(round((fill_p - arm_p) / arm_p * 100, 4))

    # ── Close event stats ─────────────────────────────────────────────────────
    close_stats = {}
    if closes:
        all_close_strats = [s for s in STRATEGY_ORDER
                            if any(c["strategy"] == s for c in closes)]
        for s in all_close_strats:
            s_cl = [c for c in closes if c["strategy"] == s]
            tp_n  = sum(1 for c in s_cl if c["outcome"] == "TP")
            sl_n  = sum(1 for c in s_cl if c["outcome"] == "SL")
            tx_n  = sum(1 for c in s_cl if c["outcome"] == "TIME")
            ex_n  = sum(1 for c in s_cl if c["outcome"] == "EXCHANGE_CLOSED")
            ema_n = sum(1 for c in s_cl if c["outcome"] == "EMA_EXIT")
            pnls  = [p for p in (safe_float(c["realized_pnl"]) for c in s_cl)
                     if p is not None]
            holds = [h for h in (safe_float(c["hold_mins"]) for c in s_cl)
                     if h is not None]
            close_stats[s] = {
                "total":     len(s_cl),
                "tp":        tp_n,  "sl":       sl_n,
                "time":      tx_n,  "ex":       ex_n,  "ema":      ema_n,
                "tp_pct":    pct(tp_n,  len(s_cl)),
                "sl_pct":    pct(sl_n,  len(s_cl)),
                "tx_pct":    pct(tx_n,  len(s_cl)),
                "ema_pct":   pct(ema_n, len(s_cl)),
                "total_pnl": round(sum(pnls), 4) if pnls else None,
                "avg_hold":  round(sum(holds) / len(holds), 1) if holds else None,
            }
        # per-close outcome distribution for chart
        all_cl_outcomes = [c["outcome"] for c in closes]

    # ── Recent events (last 30, newest first) ─────────────────────────────────
    recent = sorted(arms, key=lambda r: r["arm_time"], reverse=True)[:30]

    # ── Summary ────────────────────────────────────────────────────────────────
    times      = sorted(r["arm_time"] for r in arms if r["arm_time"])
    data_range = f"{times[0]} → {times[-1]}" if times else "—"
    runtime_h  = None
    if len(times) >= 2:
        try:
            t0 = datetime.strptime(times[0],  "%Y-%m-%d %H:%M:%S")
            t1 = datetime.strptime(times[-1], "%Y-%m-%d %H:%M:%S")
            runtime_h = round((t1 - t0).total_seconds() / 3600, 1)
        except ValueError:
            pass

    return {
        "generated":        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "data_range":       data_range,
        "runtime_h":        runtime_h,
        "total":            len(arms),
        "shadow_count":     len(shadow_rows),
        "real_count":       len(real_rows),
        "fired_count":      len(fired_rows),
        "nofire_count":     len(nofire_rows),
        "strats":           strats,
        "funnel":           funnel,
        "reasons_by_strat": {k: dict(v) for k, v in reasons_by_strat.items()},
        "all_reasons":      all_reasons,
        "hours_data":       hours_data,
        "direction_by_strat": direction_by_strat,
        "atr_stats":        atr_stats,
        "shadow_stats":     shadow_stats,
        "slippage":         slippage,
        "close_stats":      close_stats,
        "recent":           recent,
        "has_fires":        len(fired_rows) > 0,
        "has_closes":       bool(closes),
        "colors":           STRATEGY_COLORS,
    }


# ── HTML generation ───────────────────────────────────────────────────────────

def badge(strat: str) -> str:
    c = STRATEGY_COLORS.get(strat, "#94a3b8")
    return (f'<span style="background:{c}22;color:{c};border:1px solid {c}44;'
            f'border-radius:4px;padding:1px 7px;font-size:0.78em">{strat}</span>')


def outcome_badge(outcome: str) -> str:
    palette = {
        "FIRED":          ("#22c55e", "#14532d"),
        "NO_FIRE":        ("#ef4444", "#450a0a"),
        "PENDING":        ("#a78bfa", "#2e1065"),
        "TP":             ("#22c55e", "#14532d"),
        "SL":             ("#ef4444", "#450a0a"),
        "TIME":           ("#94a3b8", "#1e293b"),
        "EMA_EXIT":       ("#8b5cf6", "#2e1065"),
        "EXCHANGE_CLOSED":("#f59e0b", "#451a03"),
    }
    fg, bg = palette.get(outcome, ("#94a3b8", "#1e293b"))
    return (f'<span style="background:{bg};color:{fg};border:1px solid {fg}44;'
            f'border-radius:4px;padding:1px 7px;font-size:0.78em">{outcome}</span>')


def build_html(d: dict) -> str:
    strats    = d["strats"]
    funnel    = d["funnel"]
    has_fires = d["has_fires"]
    has_closes = d["has_closes"]

    strat_colors_js = json.dumps(d["colors"])
    data_js         = json.dumps(d)

    # ── Funnel table rows ──────────────────────────────────────────────────────
    funnel_rows = ""
    for s in strats:
        f = funnel[s]
        c = color(s)
        fire_pct = pct(f["fired"], f["real"]) if f["real"] else 0
        funnel_rows += f"""
        <tr>
          <td>{badge(s)}</td>
          <td class="num">{f['total']}</td>
          <td class="num">{f['shadow']}</td>
          <td class="num">{f['real']}</td>
          <td class="num">{f['nofire']}</td>
          <td class="num bold" style="color:{c}">{f['fired']}</td>
          <td class="num">{fire_pct:.0f}%</td>
        </tr>"""

    # ── No-fire table rows ────────────────────────────────────────────────────
    nofire_rows_html = ""
    for s, ctr in d["reasons_by_strat"].items():
        for reason, cnt in sorted(ctr.items(), key=lambda x: -x[1]):
            nofire_rows_html += f"""
        <tr>
          <td>{badge(s)}</td>
          <td><code style="color:#fbbf24">{reason}</code></td>
          <td class="num">{cnt}</td>
        </tr>"""
    if not nofire_rows_html:
        nofire_rows_html = '<tr><td colspan="3" class="muted center">No no-fire events yet</td></tr>'

    # ── Shadow stats table rows ───────────────────────────────────────────────
    shadow_rows_html = ""
    for s in strats:
        ss = d["shadow_stats"].get(s)
        if ss:
            tp_dist = f"{ss['avg_tp_pct']:.3f}%" if ss["avg_tp_pct"] else "—"
            sl_dist = f"{ss['avg_sl_pct']:.3f}%" if ss["avg_sl_pct"] else "—"
            pending_str = f' <span class="muted" style="font-size:0.8em">+{ss["pending"]} pending</span>' if ss["pending"] else ""
            shadow_rows_html += f"""
        <tr>
          <td>{badge(s)}</td>
          <td class="num">{ss['count']}{pending_str}</td>
          <td class="num" style="color:#22c55e">{ss['tp_n']} ({ss['tp_pct']:.0f}%)</td>
          <td class="num" style="color:#ef4444">{ss['sl_n']} ({ss['sl_pct']:.0f}%)</td>
          <td class="num" style="color:#94a3b8">{ss['tx_n']} ({ss['tx_pct']:.0f}%)</td>
          <td class="num muted">{tp_dist}</td>
          <td class="num muted">{sl_dist}</td>
        </tr>"""
    if not shadow_rows_html:
        shadow_rows_html = '<tr><td colspan="7" class="muted center">No shadow events</td></tr>'

    # ── ATR stats table rows ───────────────────────────────────────────────────
    atr_rows_html = ""
    for s in strats:
        a = d["atr_stats"].get(s)
        if a:
            note = " (sigma)" if s == "sr" else ""
            atr_rows_html += f"""
        <tr>
          <td>{badge(s)}</td>
          <td class="num muted">{a['count']}</td>
          <td class="num">{a['min']}</td>
          <td class="num bold">{a['median']}</td>
          <td class="num">{a['max']}</td>
          <td class="num muted">{note}</td>
        </tr>"""

    # ── Close stats table rows ────────────────────────────────────────────────
    close_rows_html = ""
    if has_closes:
        for s, cs in d["close_stats"].items():
            pnl_str = f"{cs['total_pnl']:+.4f}" if cs["total_pnl"] is not None else "—"
            hold_str = f"{cs['avg_hold']}m" if cs["avg_hold"] else "—"
            pnl_color = "#22c55e" if (cs["total_pnl"] or 0) >= 0 else "#ef4444"
            close_rows_html += f"""
        <tr>
          <td>{badge(s)}</td>
          <td class="num">{cs['total']}</td>
          <td class="num" style="color:#22c55e">{cs['tp']} ({cs['tp_pct']}%)</td>
          <td class="num" style="color:#ef4444">{cs['sl']} ({cs['sl_pct']}%)</td>
          <td class="num" style="color:#94a3b8">{cs['time']} ({cs['tx_pct']}%)</td>
          <td class="num" style="color:#8b5cf6">{cs['ema']} ({cs['ema_pct']}%)</td>
          <td class="num" style="color:#94a3b8">{cs['ex']}</td>
          <td class="num" style="color:{pnl_color}">{pnl_str}</td>
          <td class="num">{hold_str}</td>
        </tr>"""
    else:
        close_rows_html = '<tr><td colspan="8" class="muted center">No closed trades yet</td></tr>'

    # ── Recent events table ───────────────────────────────────────────────────
    recent_rows_html = ""
    for r in d["recent"]:
        shadow_tag = ' <span class="muted" style="font-size:0.75em">[shadow]</span>' if r["shadow"] == "1" else ""
        atr_val = safe_float(r["atr"])
        atr_str = f"{atr_val:.4f}" if atr_val else "—"
        reason  = f'<code style="color:#fbbf24;font-size:0.85em">{r["no_fire_reason"]}</code>' \
                  if r["no_fire_reason"] else "—"
        dir_color = "#22c55e" if r["direction"] == "LONG" else ("#ef4444" if r["direction"] == "SHORT" else "#94a3b8")
        recent_rows_html += f"""
        <tr>
          <td class="mono muted">{r['arm_time'][:16]}</td>
          <td>{badge(r['strategy'])}{shadow_tag}</td>
          <td><span style="color:{dir_color}">{r['direction'] or '—'}</span></td>
          <td>{r['symbol']}</td>
          <td class="num">{float(r['arm_price']):,.2f}</td>
          <td class="num">{atr_str}</td>
          <td>{outcome_badge(r['outcome'])}</td>
          <td>{reason}</td>
        </tr>"""

    # ── KPI cards ─────────────────────────────────────────────────────────────
    fired_color = "#22c55e" if d["fired_count"] > 0 else "#475569"
    shadow_pct  = pct(d["shadow_count"], d["total"])
    runtime_str = f"{d['runtime_h']}h" if d["runtime_h"] else "—"

    kpi_cards = f"""
    <div class="kpi-grid">
      <div class="card kpi">
        <div class="kpi-label">Signals detected</div>
        <div class="kpi-value">{d['total']}</div>
        <div class="kpi-sub">over {runtime_str}</div>
      </div>
      <div class="card kpi">
        <div class="kpi-label">Shadow (out-of-window)</div>
        <div class="kpi-value" style="color:#a78bfa">{d['shadow_count']}</div>
        <div class="kpi-sub">{shadow_pct:.0f}% of all signals</div>
      </div>
      <div class="card kpi">
        <div class="kpi-label">In-window signals</div>
        <div class="kpi-value">{d['real_count']}</div>
        <div class="kpi-sub">{d['nofire_count']} no-fire &nbsp;·&nbsp; {d['fired_count']} fired</div>
      </div>
      <div class="card kpi">
        <div class="kpi-label">Trades fired</div>
        <div class="kpi-value" style="color:{fired_color}">{d['fired_count']}</div>
        <div class="kpi-sub">{"— awaiting first trade" if d['fired_count'] == 0 else ""}</div>
      </div>
    </div>"""

    # ── Slippage section ──────────────────────────────────────────────────────
    slippage_html = ""
    if d["slippage"]:
        avg_slip = sum(d["slippage"]) / len(d["slippage"])
        slippage_html = f"""
    <div class="card" style="margin-top:16px">
      <div class="section-title">Decision → Fill Price Slippage</div>
      <p class="muted" style="margin:0 0 12px">
        Difference between arm_price (signal candle close) and disarm_price (actual fill).
      </p>
      <table class="data-table" style="max-width:400px">
        <tr><td>Samples</td><td class="num">{len(d['slippage'])}</td></tr>
        <tr><td>Avg slippage</td><td class="num">{avg_slip:+.4f}%</td></tr>
        <tr><td>Min</td><td class="num">{min(d['slippage']):+.4f}%</td></tr>
        <tr><td>Max</td><td class="num">{max(d['slippage']):+.4f}%</td></tr>
      </table>
    </div>"""
    elif has_fires:
        slippage_html = '<div class="card muted" style="margin-top:16px">Slippage data available after first FIRED trade.</div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Bitunix Signal Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    html {{ font-size: 15px; }}
    body {{
      margin: 0; padding: 20px 24px 40px;
      background: #0c0e14;
      color: #cbd5e1;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
      line-height: 1.5;
    }}
    h1 {{ color: #f1f5f9; font-size: 1.4rem; margin: 0 0 4px; font-weight: 600; }}
    h2 {{ color: #f1f5f9; font-size: 1rem; margin: 0 0 14px; font-weight: 600;
          letter-spacing: 0.03em; text-transform: uppercase; }}
    .meta {{ color: #475569; font-size: 0.8rem; margin-bottom: 24px; }}
    .card {{
      background: #151922;
      border: 1px solid #1e2533;
      border-radius: 10px;
      padding: 20px 22px;
    }}
    .section-title {{
      font-size: 0.8rem; font-weight: 600; letter-spacing: 0.06em;
      text-transform: uppercase; color: #64748b; margin-bottom: 14px;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 14px;
      margin-bottom: 16px;
    }}
    .kpi {{ text-align: center; }}
    .kpi-label {{ font-size: 0.78rem; color: #64748b; text-transform: uppercase;
                  letter-spacing: 0.05em; margin-bottom: 6px; }}
    .kpi-value {{ font-size: 2.2rem; font-weight: 700; color: #f1f5f9;
                  line-height: 1.1; }}
    .kpi-sub {{ font-size: 0.78rem; color: #475569; margin-top: 4px; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
                margin-bottom: 16px; }}
    .three-col {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px;
                  margin-bottom: 16px; }}
    .full {{ margin-bottom: 16px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    .data-table th, .data-table td {{
      padding: 7px 10px;
      border-bottom: 1px solid #1e2533;
      font-size: 0.85rem;
    }}
    .data-table th {{
      color: #64748b; font-weight: 600;
      text-align: left; font-size: 0.75rem;
      text-transform: uppercase; letter-spacing: 0.04em;
    }}
    .data-table tr:last-child td {{ border-bottom: none; }}
    .data-table tr:hover td {{ background: #1a2030; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .data-table th.num {{ text-align: right; }}
    .bold {{ font-weight: 600; color: #f1f5f9; }}
    .muted {{ color: #475569; }}
    .center {{ text-align: center; }}
    .mono {{ font-family: "SF Mono", "Cascadia Code", monospace; font-size: 0.82em; }}
    .chart-wrap {{ position: relative; height: 220px; }}
    .chart-wrap-tall {{ position: relative; height: 280px; }}
    .note {{
      background: #1a1f2e; border: 1px solid #2d3748;
      border-radius: 6px; padding: 10px 14px;
      font-size: 0.82rem; color: #94a3b8; margin-top: 12px;
    }}
    code {{ background: #1e2533; border-radius: 4px; padding: 1px 5px; }}
    @media (max-width: 900px) {{
      .kpi-grid {{ grid-template-columns: 1fr 1fr; }}
      .two-col, .three-col {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>

<h1>Bitunix Signal Dashboard</h1>
<div class="meta">
  Generated {d['generated']} &nbsp;·&nbsp; {d['data_range']}
</div>

{kpi_cards}

<!-- ── Strategy Funnel ─────────────────────────────────────────────────── -->
<div class="full">
  <div class="card">
    <div class="section-title">Signal Funnel by Strategy</div>
    <table class="data-table">
      <thead><tr>
        <th>Strategy</th><th class="num">Total signals</th>
        <th class="num">Shadow</th><th class="num">In-window</th>
        <th class="num">No-fire</th><th class="num">Fired</th>
        <th class="num">Fire rate</th>
      </tr></thead>
      <tbody>{funnel_rows}</tbody>
    </table>
  </div>
</div>

<!-- ── Charts row 1 ────────────────────────────────────────────────────── -->
<div class="two-col">
  <div class="card">
    <div class="section-title">Signals by UTC Hour</div>
    <div class="chart-wrap"><canvas id="hourChart"></canvas></div>
    <div class="note">All signals (shadow + in-window). Useful for confirming
      out-of-window hours are being captured as shadow events.</div>
  </div>
  <div class="card">
    <div class="section-title">Direction Balance (LONG vs SHORT)</div>
    <div class="chart-wrap"><canvas id="dirChart"></canvas></div>
  </div>
</div>

<!-- ── No-fire + Shadow ────────────────────────────────────────────────── -->
<div class="two-col">
  <div class="card">
    <div class="section-title">No-Fire Reasons</div>
    <table class="data-table">
      <thead><tr>
        <th>Strategy</th><th>Reason</th><th class="num">Count</th>
      </tr></thead>
      <tbody>{nofire_rows_html}</tbody>
    </table>
  </div>
  <div class="card">
    <div class="section-title">Shadow Events — Would-Be Trade Sizes</div>
    <p class="muted" style="font-size:0.82rem;margin:0 0 12px">
      Outcomes resolved against local candle data via resolve_shadows.py.
      TP/SL dist = avg would-be distance as % of entry price.
    </p>
    <table class="data-table">
      <thead><tr>
        <th>Strategy</th><th class="num">Signals</th>
        <th class="num">TP</th><th class="num">SL</th><th class="num">TIME</th>
        <th class="num">Avg TP dist</th><th class="num">Avg SL dist</th>
      </tr></thead>
      <tbody>{shadow_rows_html}</tbody>
    </table>
  </div>
</div>

<!-- ── ATR stats ──────────────────────────────────────────────────────── -->
<div class="full">
  <div class="card">
    <div class="section-title">ATR at Signal Time (min / median / max)</div>
    <table class="data-table">
      <thead><tr>
        <th>Strategy</th><th class="num">N</th>
        <th class="num">Min</th><th class="num">Median</th>
        <th class="num">Max</th><th>Note</th>
      </tr></thead>
      <tbody>{atr_rows_html}</tbody>
    </table>
    <div class="note">
      sr uses <code>sigma</code> (return std-dev) rather than ATR — values
      are not comparable to other strategies.
    </div>
  </div>
</div>

<!-- ── Trade outcomes ─────────────────────────────────────────────────── -->
<div class="full">
  <div class="card">
    <div class="section-title">Trade Outcomes (from close_events.csv)</div>
    <table class="data-table">
      <thead><tr>
        <th>Strategy</th><th class="num">Trades</th>
        <th class="num">TP</th><th class="num">SL</th>
        <th class="num">Time exit</th>
        <th class="num">EMA exit</th>
        <th class="num">Exch closed</th>
        <th class="num">Net PnL (USDT)</th>
        <th class="num">Avg hold</th>
      </tr></thead>
      <tbody>{close_rows_html}</tbody>
    </table>
  </div>
</div>

<!-- ── Slippage ──────────────────────────────────────────────────────── -->
{slippage_html}

<!-- ── Recent events ─────────────────────────────────────────────────── -->
<div class="full" style="margin-top:16px">
  <div class="card">
    <div class="section-title">Recent Events (last 30)</div>
    <div style="overflow-x:auto">
    <table class="data-table">
      <thead><tr>
        <th>Time (UTC)</th><th>Strategy</th><th>Dir</th><th>Symbol</th>
        <th class="num">Price</th><th class="num">ATR</th>
        <th>Outcome</th><th>No-fire reason</th>
      </tr></thead>
      <tbody>{recent_rows_html}</tbody>
    </table>
    </div>
  </div>
</div>

<script>
const D      = {data_js};
const COLORS = {strat_colors_js};
const STRATS = D.strats;

// Hourly chart
(function() {{
  const ctx = document.getElementById("hourChart");
  const datasets = STRATS.map(s => ({{
    label: s,
    data: Array.from({{length: 24}}, (_, h) => (D.hours_data[h] || {{}})[s] || 0),
    backgroundColor: COLORS[s] || "#94a3b8",
    stack: "h",
  }}));
  new Chart(ctx, {{
    type: "bar",
    data: {{
      labels: Array.from({{length: 24}}, (_, h) => h + ":00"),
      datasets,
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ labels: {{ color: "#94a3b8", boxWidth: 12, font: {{ size: 11 }} }} }} }},
      scales: {{
        x: {{ stacked: true, ticks: {{ color: "#475569", font: {{ size: 10 }} }},
               grid: {{ color: "#1e2533" }} }},
        y: {{ stacked: true, ticks: {{ color: "#475569", font: {{ size: 10 }} }},
               grid: {{ color: "#1e2533" }} }},
      }},
    }},
  }});
}})();

// Direction balance chart
(function() {{
  const ctx = document.getElementById("dirChart");
  const longData  = STRATS.map(s => (D.direction_by_strat[s] || {{}}).LONG  || 0);
  const shortData = STRATS.map(s => (D.direction_by_strat[s] || {{}}).SHORT || 0);
  new Chart(ctx, {{
    type: "bar",
    data: {{
      labels: STRATS,
      datasets: [
        {{ label: "LONG",  data: longData,  backgroundColor: "#22c55e88", borderColor: "#22c55e", borderWidth: 1 }},
        {{ label: "SHORT", data: shortData, backgroundColor: "#ef444488", borderColor: "#ef4444", borderWidth: 1 }},
      ],
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ labels: {{ color: "#94a3b8", boxWidth: 12, font: {{ size: 11 }} }} }} }},
      scales: {{
        x: {{ ticks: {{ color: "#94a3b8" }}, grid: {{ color: "#1e2533" }} }},
        y: {{ ticks: {{ color: "#475569" }}, grid: {{ color: "#1e2533" }} }},
      }},
    }},
  }});
}})();
</script>
</body>
</html>"""


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Bitunix signal dashboard")
    parser.add_argument("--open", action="store_true", help="Open in browser after generating")
    parser.add_argument("--out", default=OUT_HTML, help=f"Output path (default: {OUT_HTML})")
    args = parser.parse_args()

    arms   = read_csv(ARM_CSV)
    closes = read_csv(CLOSE_CSV)

    if not arms:
        print(f"No data found in {ARM_CSV}", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(arms)} arm events  |  {len(closes)} close events")

    d    = compute(arms, closes)
    html = build_html(d)

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)

    print(f"  → {args.out}")

    if args.open:
        subprocess.run(["open", args.out] if sys.platform == "darwin"
                       else ["xdg-open", args.out], check=False)


if __name__ == "__main__":
    main()
