"""Kalshi and Polymarket public market-data clients with a unified Market model.

Read-only, unauthenticated endpoints:
  Kalshi:      https://api.elections.kalshi.com/trade-api/v2/markets
  Polymarket:  https://gamma-api.polymarket.com/markets
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx

from .config import Settings

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA_BASE = "https://gamma-api.polymarket.com"

_TIMEOUT = httpx.Timeout(20.0)


@dataclass
class Market:
    id: str  # "<source>:<ticker>"
    source: str  # "kalshi" | "polymarket"
    ticker: str
    question: str
    description: str
    yes_price: float  # current probability-price of YES, in [0, 1]
    close_time: Optional[datetime]
    liquidity: float
    volume: float
    url: str
    status: str = "open"  # open | closed | settled
    resolution: Optional[str] = None  # "yes" | "no" once settled
    yes_bid: Optional[float] = None  # best bid/ask when the venue exposes them —
    yes_ask: Optional[float] = None  # paper trades fill by crossing this spread

    @property
    def days_to_close(self) -> Optional[float]:
        if self.close_time is None:
            return None
        return (self.close_time - datetime.now(timezone.utc)).total_seconds() / 86400


def _parse_dt(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _f(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- Kalshi

def parse_kalshi(m: dict) -> Optional[Market]:
    if m.get("market_type") != "binary":
        return None
    if m.get("mve_collection_ticker"):  # auto-generated multivariate parlay combos
        return None
    bid, ask = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
    has_book = 0 < bid <= ask < 1
    price = (bid + ask) / 2 if has_book else _f(m.get("last_price_dollars"))
    if not 0 < price < 1:
        return None
    status = m.get("status", "open")
    result = m.get("result") or None
    question = m.get("title", "")
    sub = m.get("yes_sub_title") or ""
    if sub and sub not in question:
        question = f"{question} — {sub}"
    return Market(
        id=f"kalshi:{m['ticker']}",
        source="kalshi",
        ticker=m["ticker"],
        question=question,
        description=(m.get("rules_primary") or "")[:4000],
        yes_price=price,
        close_time=_parse_dt(m.get("close_time")),
        liquidity=_f(m.get("liquidity_dollars")),
        volume=_f(m.get("volume_fp")),
        url=f"https://kalshi.com/markets/{m['ticker']}",
        status="settled" if status == "settled" else ("closed" if status in ("closed", "finalized") else "open"),
        resolution=result if result in ("yes", "no") else None,
        yes_bid=bid if has_book else None,
        yes_ask=ask if has_book else None,
    )


def fetch_kalshi(limit: int = 100) -> list[Market]:
    """Fetch via /events with nested markets: the raw /markets feed is dominated
    by thousands of zero-priced auto-generated parlay combos (newest-first), so
    the curated events surface is the usable one. `limit` = events requested;
    each event carries several markets."""
    with httpx.Client(timeout=_TIMEOUT) as client:
        r = client.get(
            f"{KALSHI_BASE}/events",
            params={"status": "open", "limit": min(limit, 200), "with_nested_markets": "true"},
        )
        r.raise_for_status()
    out = []
    for event in r.json().get("events", []):
        for raw in event.get("markets") or []:
            parsed = parse_kalshi(raw)
            if parsed:
                out.append(parsed)
    return out


def fetch_kalshi_market(ticker: str) -> Optional[Market]:
    with httpx.Client(timeout=_TIMEOUT) as client:
        r = client.get(f"{KALSHI_BASE}/markets/{ticker}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
    return parse_kalshi(r.json()["market"])


# ---------------------------------------------------------------- Polymarket

def parse_polymarket(m: dict) -> Optional[Market]:
    try:
        outcomes = json.loads(m.get("outcomes") or "[]")
        prices = json.loads(m.get("outcomePrices") or "[]")
    except json.JSONDecodeError:
        return None
    if [o.lower() for o in outcomes] != ["yes", "no"] or len(prices) != 2:
        return None
    bid, ask = _f(m.get("bestBid")), _f(m.get("bestAsk"))
    has_book = 0 < bid <= ask < 1
    price = (bid + ask) / 2 if has_book else _f(prices[0], _f(m.get("lastTradePrice")))
    closed = bool(m.get("closed"))
    resolution = None
    if closed and prices[0] in ("1", "0"):
        resolution = "yes" if prices[0] == "1" else "no"
    if not closed and not 0 < price < 1:
        return None
    return Market(
        id=f"polymarket:{m['id']}",
        source="polymarket",
        ticker=str(m["id"]),
        question=m.get("question", ""),
        description=(m.get("description") or "")[:4000],
        yes_price=min(max(price, 0.0), 1.0),
        close_time=_parse_dt(m.get("endDate")),
        liquidity=_f(m.get("liquidityNum")),
        volume=_f(m.get("volumeNum")),
        url=f"https://polymarket.com/market/{m.get('slug', m['id'])}",
        status="settled" if resolution else ("closed" if closed else "open"),
        resolution=resolution,
        yes_bid=bid if has_book else None,
        yes_ask=ask if has_book else None,
    )


def fetch_polymarket(limit: int = 100) -> list[Market]:
    with httpx.Client(timeout=_TIMEOUT) as client:
        r = client.get(
            f"{GAMMA_BASE}/markets",
            params={
                "active": "true", "closed": "false", "limit": min(limit, 500),
                "order": "volumeNum", "ascending": "false",
            },
        )
        r.raise_for_status()
    out = []
    for raw in r.json():
        parsed = parse_polymarket(raw)
        if parsed:
            out.append(parsed)
    return out


def fetch_polymarket_market(market_id: str) -> Optional[Market]:
    with httpx.Client(timeout=_TIMEOUT) as client:
        r = client.get(f"{GAMMA_BASE}/markets/{market_id}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
    return parse_polymarket(r.json())


# ---------------------------------------------------------------- unified

def fetch_all(sources: list[str], limit: int = 100) -> list[Market]:
    markets: list[Market] = []
    if "kalshi" in sources:
        markets += fetch_kalshi(limit)
    if "polymarket" in sources:
        markets += fetch_polymarket(limit)
    return markets


def refresh(market_id: str) -> Optional[Market]:
    """Re-fetch one market by its unified id for marking/resolution."""
    source, _, ticker = market_id.partition(":")
    if source == "kalshi":
        return fetch_kalshi_market(ticker)
    if source == "polymarket":
        return fetch_polymarket_market(ticker)
    raise ValueError(f"unknown source in market id: {market_id}")


def screen(markets: list[Market], settings: Settings) -> list[Market]:
    """Filter to tradeable candidates, best (most liquid) first."""
    picks = []
    for m in markets:
        days = m.days_to_close
        if m.status != "open" or days is None:
            continue
        if not settings.min_days <= days <= settings.max_days:
            continue
        if not settings.min_price <= m.yes_price <= settings.max_price:
            continue
        if max(m.liquidity, m.volume) < settings.min_liquidity:
            continue
        picks.append(m)
    picks.sort(key=lambda m: max(m.liquidity, m.volume), reverse=True)
    return picks
