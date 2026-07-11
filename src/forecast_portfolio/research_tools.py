"""Non-LLM research tool fanout (stage 2), after the paper's ten conditional
tools (Table 5). The paper's per-tool ablation (Table 3) drives what we build:

  implemented here      paper ΔIC    notes
  wikipedia_lookup      +0.25        the single most helpful conditional tool
  kalshi_orderbook      +0.13        bid/ask depth
  market_snapshot        n/a         ≈ kalshi_data (price history/trends);
                                     always-invoked in the paper, no contrast
  sister_markets         n/a         relative-value signal used in its example
  web_search             n/a         runs as the LLM call in pipeline.research()

  deliberately omitted (negative ΔIC in the paper):
  code_execution -0.24, fred_data -0.26, congress_bills -0.27, court_docket -0.12
  (the paper reads this as tool selection/usage misalignment, not useless data —
  they stay out until there's a selection policy that earns them back)

Each tool is conditional (runs only where it applies), isolated (one failure
doesn't kill the stage), and returns plain text for the dossier. PRICE_TOOLS
reveal the market's own price and are withheld from blind stages until the
stage-5 critic.
"""

from __future__ import annotations

import json
from typing import Callable, Optional

import httpx

from .markets import GAMMA_BASE, KALSHI_BASE, Market, _f

_TIMEOUT = httpx.Timeout(20.0)
_MAX_SISTERS = 12


# Wikipedia's API policy rejects default client user agents.
_HEADERS = {"User-Agent": "forecast-portfolio/0.1 (research; github.com/francohtlin/base-agent)"}


def _get(url: str, params: dict | None = None) -> dict | list:
    with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS, follow_redirects=True) as client:
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


def format_kalshi_orderbook(raw: dict) -> Optional[str]:
    ob = raw.get("orderbook_fp") or raw.get("orderbook") or {}
    yes = sorted(((_f(p), _f(s)) for p, s in ob.get("yes_dollars") or []), reverse=True)
    no = sorted(((_f(p), _f(s)) for p, s in ob.get("no_dollars") or []), reverse=True)
    if not yes and not no:
        return None
    lines = []
    best_yes_bid = yes[0][0] if yes else None
    best_no_bid = no[0][0] if no else None
    if best_yes_bid is not None and best_no_bid is not None:
        lines.append(
            f"Best YES bid {best_yes_bid:.2f} / implied YES ask {1 - best_no_bid:.2f} "
            f"(spread {1 - best_no_bid - best_yes_bid:.2f})"
        )

    def depth_near_top(levels):
        if not levels:
            return 0.0
        top = levels[0][0]
        return sum(s for p, s in levels if p >= top - 0.05)

    lines.append(
        f"Depth within 5c of top: {depth_near_top(yes):,.0f} contracts bid YES, "
        f"{depth_near_top(no):,.0f} bid NO; "
        f"total resting {sum(s for _, s in yes):,.0f} YES / {sum(s for _, s in no):,.0f} NO"
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


def kalshi_orderbook(market: Market) -> Optional[str]:
    if market.source != "kalshi":
        return None
    raw = _get(f"{KALSHI_BASE}/markets/{market.ticker}/orderbook")
    return format_kalshi_orderbook(raw)


WIKI_SEARCH = "https://en.wikipedia.org/w/rest.php/v1/search/page"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary"


def wikipedia_lookup(topics: list[str]) -> Optional[str]:
    """Base rates and background facts — the paper's highest-ΔIC tool (+0.25)."""
    sections = []
    for topic in topics[:3]:
        hits = _get(WIKI_SEARCH, params={"q": topic, "limit": 1}).get("pages", [])
        if not hits:
            continue
        summary = _get(f"{WIKI_SUMMARY}/{hits[0]['key']}")
        extract = summary.get("extract", "")
        if extract:
            sections.append(f"[{summary.get('title', topic)}] {extract[:1200]}")
    return "\n\n".join(sections) or None


# Tools whose output reveals the market's own price — withheld from the blind
# stages and shown first to the stage-5 critic.
PRICE_TOOLS = ("market_snapshot", "kalshi_orderbook")


def run_tools(market: Market, wiki_topics: list[str] | None = None) -> dict[str, str]:
    """Run every applicable tool; failures degrade to a note instead of raising.

    Returns {tool_name: text}. The caller splits it on PRICE_TOOLS: those
    sections go to the critic/final stages only; the rest is ordinary evidence.
    """
    jobs: list[tuple[str, Callable[[], Optional[str]]]] = [
        ("market_snapshot", lambda: market_snapshot(market)),
        ("kalshi_orderbook", lambda: kalshi_orderbook(market)),
        ("sister_markets", lambda: sister_markets(market)),
        ("wikipedia_lookup", lambda: wikipedia_lookup(wiki_topics or [])),
    ]
    out: dict[str, str] = {}
    for name, fn in jobs:
        try:
            text = fn()
        except Exception as exc:
            text = f"(tool failed: {exc})"
        if text:
            out[name] = text
    return out
