from datetime import datetime, timedelta, timezone

from forecast_portfolio.config import Settings
from forecast_portfolio.markets import Market, parse_kalshi, parse_polymarket, screen

KALSHI_FIXTURE = {
    "ticker": "KXTEST-26DEC31",
    "title": "Will X happen in 2026?",
    "yes_sub_title": "X happens",
    "rules_primary": "Resolves YES if X happens by Dec 31.",
    "market_type": "binary",
    "status": "open",
    "close_time": "2026-12-31T23:59:00Z",
    "yes_bid_dollars": "0.4200",
    "yes_ask_dollars": "0.4600",
    "last_price_dollars": "0.4300",
    "liquidity_dollars": "15000.00",
    "volume_fp": "84000",
    "result": "",
}

POLYMARKET_FIXTURE = {
    "id": "540817",
    "question": "Will Y happen before 2027?",
    "description": "Resolves YES if Y happens.",
    "slug": "will-y-happen",
    "outcomes": '["Yes", "No"]',
    "outcomePrices": '["0.35", "0.65"]',
    "bestBid": "0.34",
    "bestAsk": "0.36",
    "lastTradePrice": "0.35",
    "endDate": "2026-08-31T12:00:00Z",
    "liquidityNum": 26725.76,
    "volumeNum": 500000.0,
    "closed": False,
    "active": True,
}


def test_parse_kalshi_uses_bid_ask_mid():
    m = parse_kalshi(KALSHI_FIXTURE)
    assert m is not None
    assert m.id == "kalshi:KXTEST-26DEC31"
    assert m.yes_price == 0.44  # mid of 0.42/0.46
    assert m.status == "open" and m.resolution is None
    assert "X happens" in m.question


def test_parse_kalshi_settled():
    settled = dict(KALSHI_FIXTURE, status="settled", result="yes")
    m = parse_kalshi(settled)
    assert m.status == "settled" and m.resolution == "yes"


def test_parse_kalshi_rejects_non_binary():
    assert parse_kalshi(dict(KALSHI_FIXTURE, market_type="scalar")) is None


def test_parse_polymarket_binary():
    m = parse_polymarket(POLYMARKET_FIXTURE)
    assert m is not None
    assert m.id == "polymarket:540817"
    assert m.yes_price == 0.35  # mid of 0.34/0.36
    assert m.close_time.tzinfo is not None


def test_parse_polymarket_rejects_multi_outcome():
    multi = dict(POLYMARKET_FIXTURE, outcomes='["A", "B"]')
    assert parse_polymarket(multi) is None


def test_parse_polymarket_resolved():
    resolved = dict(POLYMARKET_FIXTURE, closed=True, outcomePrices='["1", "0"]')
    m = parse_polymarket(resolved)
    assert m.status == "settled" and m.resolution == "yes"


def test_screen_filters():
    settings = Settings()
    settings.min_liquidity = 1000
    good = Market(
        id="kalshi:A", source="kalshi", ticker="A", question="q", description="",
        yes_price=0.5, close_time=datetime.now(timezone.utc) + timedelta(days=30),
        liquidity=5000, volume=0, url="",
    )
    extreme_price = Market(**{**good.__dict__, "id": "kalshi:B", "yes_price": 0.99})
    too_soon = Market(**{**good.__dict__, "id": "kalshi:C",
                         "close_time": datetime.now(timezone.utc) + timedelta(hours=2)})
    illiquid = Market(**{**good.__dict__, "id": "kalshi:D", "liquidity": 10})
    picks = screen([good, extreme_price, too_soon, illiquid], settings)
    assert [m.id for m in picks] == ["kalshi:A"]
