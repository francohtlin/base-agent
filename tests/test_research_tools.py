from forecast_portfolio.research_tools import (
    format_kalshi_snapshot, format_polymarket_snapshot, format_sisters,
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


def test_sisters_excludes_self_and_caps():
    pairs = [(f"Team {i}", i / 100) for i in range(20)]
    text = format_sisters(pairs, own_ticker="XYZ", event_label="EVT")
    assert text.count("\n- ") + text.count("\n") >= 1
    assert "EVT" in text
    assert text.count("- Team") == 12  # capped

    assert format_sisters([], "XYZ", "EVT") is None
