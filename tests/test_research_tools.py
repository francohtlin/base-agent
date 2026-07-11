from forecast_portfolio.research_tools import (
    PRICE_TOOLS, format_kalshi_orderbook, format_kalshi_snapshot,
    format_polymarket_snapshot, format_sisters,
)

KALSHI_RAW = {
    "yes_bid_dollars": "0.4200", "yes_ask_dollars": "0.4600",
    "last_price_dollars": "0.4400", "previous_price_dollars": "0.4000",
    "volume_fp": "84000", "volume_24h_fp": "1200",
    "open_interest_fp": "5000", "liquidity_dollars": "15000.00",
}

POLY_RAW = {
    "bestBid": "0.34", "bestAsk": "0.36", "spread": "0.02", "lastTradePrice": "0.35",
    "oneDayPriceChange": 0.01, "oneWeekPriceChange": 0.18, "oneMonthPriceChange": -0.02,
    "volumeNum": 500000.0, "volume24hr": 20000.0, "liquidityNum": 26725.76,
}


def test_kalshi_snapshot_includes_momentum_and_depth():
    text = format_kalshi_snapshot(KALSHI_RAW)
    assert "0.42/0.46" in text
    assert "+0.04" in text  # change vs previous close
    assert "open interest 5,000" in text


def test_polymarket_snapshot_momentum():
    text = format_polymarket_snapshot(POLY_RAW)
    assert "7d +0.18" in text  # the "spike with no catalyst" signal
    assert "spread 0.02" in text


def test_kalshi_orderbook_implied_ask_and_depth():
    raw = {"orderbook_fp": {
        "yes_dollars": [["0.0400", "878.89"], ["0.0600", "48.00"], ["0.0100", "169.00"]],
        "no_dollars": [["0.9300", "397.94"], ["0.9200", "361.10"], ["0.0100", "1002.00"]],
    }}
    text = format_kalshi_orderbook(raw)
    # best yes bid 0.06; best no bid 0.93 -> implied yes ask 0.07
    assert "Best YES bid 0.06" in text
    assert "implied YES ask 0.07" in text
    assert "spread 0.01" in text
    # depth within 5c of top (inclusive): 48 + 878.89 + 169 = 1095.89 -> "1,096"
    assert "1,096 contracts bid YES" in text
    assert "759 bid NO" in text

    assert format_kalshi_orderbook({"orderbook_fp": {}}) is None


def test_price_tools_cover_own_price_leaks():
    # Anything that reveals the market's own price must be on the withheld list.
    assert "market_snapshot" in PRICE_TOOLS
    assert "kalshi_orderbook" in PRICE_TOOLS


def test_sisters_excludes_self_and_caps():
    pairs = [(f"Team {i}", i / 100) for i in range(20)]
    text = format_sisters(pairs, own_ticker="XYZ", event_label="EVT")
    assert text.count("\n- ") + text.count("\n") >= 1
    assert "EVT" in text
    assert text.count("- Team") == 12  # capped

    assert format_sisters([], "XYZ", "EVT") is None
