"""Non-LLM research tool fanout (stage 2), after the paper's "10 conditional
research tools" (FRED, code, Wiki, Kalshi, orderbook, articles, web, Congress,
courts, earnings).

v1 implements the market-native tools — the ones the poster's example forecast
leans on ("Kalshi 54%, +18% (7d)", "Sister market 34%"):

  market_snapshot   bid/ask/last, spread, momentum (1d/7d/30d where the venue
                    exposes it), volume, open interest
  sister_markets    prices of other markets in the same event/series — often the
                    strongest single signal for relative-value reads

Each tool is conditional (runs only where it applies), isolated (one failure
doesn't kill the stage), and returns plain text for the dossier. Web/news/wiki
evidence comes from the LLM web-search call in pipeline.research(); FRED,
Congress, courts, earnings are future slots — add a function and list it in
TOOLS.
"""

from __future__ import annotations

import json
from typing import Callable, Optional

import httpx

from .markets import GAMMA_BASE, KALSHI_BASE, Market, _f

_TIMEOUT = httpx.Timeout(20.0)
_MAX_SISTERS = 12


def _get(url: str, params: dict | None = None) -> dict | list:
    with httpx.Client(timeout=_TIMEOUT) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        return r.json()


# ------------------------------------------------------------- formatters
# Pure functions over raw venue JSON — unit-testable offline.

def format_kalshi_snapshot(raw: dict) -> str:
    bid, ask = _f(raw.get("yes_bid_dollars")), _f(raw.get("yes_ask_dollars"))
    last, prev = _f(raw.get("last_price_dollars")), _f(raw.get("previous_price_dollars"))
    lines = [f"YES bid/ask: {bid:.2f}/{ask:.2f} (spread {ask - bid:.2f}), last {last:.2f}"]
    if prev > 0:
        lines.append(f"Change vs previous close: {last - prev:+.2f} (from {prev:.2f})")
    lines.append(
        f"Volume: {_f(raw.get('volume_fp')):,.0f} total, {_f(raw.get('volume_24h_fp')):,.0f} 24h; "
        f"open interest {_f(raw.get('open_interest_fp')):,.0f}; "
        f"liquidity ${_f(raw.get('liquidity_dollars')):,.0f}"
    )
    return "\n".join(lines)


def format_polymarket_snapshot(raw: dict) -> str:
    bid, ask = _f(raw.get("bestBid")), _f(raw.get("bestAsk"))
    lines = [
        f"YES bid/ask: {bid:.2f}/{ask:.2f} (spread {_f(raw.get('spread')):.2f}), "
        f"last trade {_f(raw.get('lastTradePrice')):.2f}"
    ]
    momentum = []
    for label, key in (("1d", "oneDayPriceChange"), ("7d", "oneWeekPriceChange"), ("30d", "oneMonthPriceChange")):
        if raw.get(key) is not None:
            momentum.append(f"{label} {_f(raw.get(key)):+.2f}")
    if momentum:
        lines.append("Price change: " + ", ".join(momentum))
    lines.append(
        f"Volume: {_f(raw.get('volumeNum')):,.0f} total, {_f(raw.get('volume24hr')):,.0f} 24h; "
        f"liquidity {_f(raw.get('liquidityNum')):,.0f}"
    )
    return "\n".join(lines)


def format_sisters(pairs: list[tuple[str, float]], own_ticker: str, event_label: str) -> Optional[str]:
    rows = [f"- {q}: {p:.2f}" for q, p in pairs[:_MAX_SISTERS] if q and own_ticker not in q]
    if not rows:
        return None
    return f"Other markets in the same event ({event_label}):\n" + "\n".join(rows)


# ------------------------------------------------------------- tools

def market_snapshot(market: Market) -> Optional[str]:
    if market.source == "kalshi":
        raw = _get(f"{KALSHI_BASE}/markets/{market.ticker}")["market"]
        return format_kalshi_snapshot(raw)
    if market.source == "polymarket":
        raw = _get(f"{GAMMA_BASE}/markets/{market.ticker}")
        return format_polymarket_snapshot(raw)
    return None


def sister_markets(market: Market) -> Optional[str]:
    if market.source == "kalshi":
        raw = _get(f"{KALSHI_BASE}/markets/{market.ticker}")["market"]
        event = raw.get("event_ticker")
        if not event:
            return None
        siblings = _get(f"{KALSHI_BASE}/markets", params={"event_ticker": event, "limit": 50})
        pairs = []
        for m in siblings.get("markets", []):
            if m.get("ticker") == market.ticker:
                continue
            label = m.get("yes_sub_title") or m.get("title") or m.get("ticker")
            pairs.append((label, _f(m.get("last_price_dollars"))))
        return format_sisters(pairs, market.ticker, event)
    if market.source == "polymarket":
        # The by-id endpoint omits `events`; the list form includes it.
        rows = _get(f"{GAMMA_BASE}/markets", params={"id": market.ticker})
        events = (rows[0].get("events") or []) if rows else []
        if not events:
            return None
        event = _get(f"{GAMMA_BASE}/events/{events[0]['id']}")
        pairs = []
        for m in event.get("markets", []):
            if str(m.get("id")) == market.ticker:
                continue
            try:
                price = float(json.loads(m.get("outcomePrices") or "[0]")[0])
            except (json.JSONDecodeError, ValueError, IndexError):
                continue
            pairs.append((m.get("question", ""), price))
        return format_sisters(pairs, market.ticker, event.get("title", str(events[0]["id"])))
    return None


TOOLS: list[tuple[str, Callable[[Market], Optional[str]]]] = [
    ("market_snapshot", market_snapshot),
    ("sister_markets", sister_markets),
]


def run_tools(market: Market) -> dict[str, str]:
    """Run every applicable tool; failures degrade to a note instead of raising.

    Returns {tool_name: text}. The caller decides which sections the price-blind
    stages may see: `market_snapshot` reveals the market's own price/momentum, so
    the pipeline withholds it until the critic (stage 5); everything else is
    ordinary evidence.
    """
    out: dict[str, str] = {}
    for name, fn in TOOLS:
        try:
            text = fn(market)
        except Exception as exc:
            text = f"(tool failed: {exc})"
        if text:
            out[name] = text
    return out
