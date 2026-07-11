from datetime import datetime, timedelta, timezone

import pytest

from forecast_portfolio.config import Settings
from forecast_portfolio.markets import Market
from forecast_portfolio.pipeline import PipelineResult
from forecast_portfolio.portfolio import Ledger


@pytest.fixture
def settings(tmp_path):
    s = Settings()
    s.db_path = tmp_path / "test.db"
    s.stake_usd = 100.0
    s.edge_threshold = 0.05
    return s


def make_market(price=0.40):
    return Market(
        id="kalshi:TEST-1", source="kalshi", ticker="TEST-1",
        question="Will the thing happen?", description="rules",
        yes_price=price,
        close_time=datetime.now(timezone.utc) + timedelta(days=10),
        liquidity=5000, volume=9000, url="https://example.com",
    )


def make_result(p_final=0.55, price=0.40):
    return PipelineResult(
        market_id="kalshi:TEST-1", market_price=price,
        p_forecasters=[0.5, 0.55, 0.6], p_critic=0.52,
        p_final=p_final, p_baseline=0.45,
    )


def test_trade_lifecycle_yes_win(settings):
    ledger = Ledger(settings.db_path)
    market, result = make_market(0.40), make_result(0.55, 0.40)

    scan_id = ledger.record_scan(market, result)
    trade = ledger.maybe_open_trade(market, result, settings, scan_id)
    assert trade is not None and trade.side == "yes"
    assert trade.contracts == pytest.approx(100 / 0.40)

    # mark-to-market: price moves to 0.50 -> unrealized = 250 * 0.50 - 100 = +25
    assert trade.unrealized(0.50) == pytest.approx(25.0)

    settled = ledger.settle(market.id, "yes")
    assert len(settled) == 1
    # payout 250 contracts * $1 - $100 stake = +150
    assert settled[0].pnl == pytest.approx(150.0)
    assert not ledger.open_trades()


def test_trade_no_side_and_loss(settings):
    ledger = Ledger(settings.db_path)
    market = make_market(0.70)
    result = make_result(p_final=0.55, price=0.70)  # edge -0.15 -> buy NO at 0.30

    scan_id = ledger.record_scan(market, result)
    trade = ledger.maybe_open_trade(market, result, settings, scan_id)
    assert trade.side == "no"
    assert trade.entry_price == pytest.approx(0.30)

    settled = ledger.settle(market.id, "yes")  # NO side loses
    assert settled[0].pnl == pytest.approx(-100.0)


def test_no_trade_below_threshold_or_duplicate(settings):
    ledger = Ledger(settings.db_path)
    market = make_market(0.40)

    small_edge = make_result(p_final=0.42, price=0.40)
    scan_id = ledger.record_scan(market, small_edge)
    assert ledger.maybe_open_trade(market, small_edge, settings, scan_id) is None

    big_edge = make_result(p_final=0.60, price=0.40)
    scan_id = ledger.record_scan(market, big_edge)
    assert ledger.maybe_open_trade(market, big_edge, settings, scan_id) is not None
    # second signal on same market: no doubling up
    assert ledger.maybe_open_trade(market, big_edge, settings, scan_id) is None


def test_scan_records_mark_and_resolution_dataset(settings):
    ledger = Ledger(settings.db_path)
    market, result = make_market(0.40), make_result()
    ledger.record_scan(market, result)

    assert ledger.latest_mark(market.id) == pytest.approx(0.40)
    assert ledger.tracked_market_ids() == [market.id]

    ledger.settle(market.id, "no")
    rows = ledger.resolved_scans()
    assert len(rows) == 1 and rows[0]["outcome"] == "no"
    assert ledger.tracked_market_ids() == []  # resolved markets drop out of marking
