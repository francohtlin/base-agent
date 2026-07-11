from datetime import datetime, timedelta, timezone

from forecast_portfolio.config import Settings
from forecast_portfolio.markets import Market
from forecast_portfolio.pipeline import MockPipeline


def make_market(mid="kalshi:MOCK-1", price=0.30):
    return Market(
        id=mid, source="kalshi", ticker=mid.split(":")[1], question="q", description="",
        yes_price=price, close_time=datetime.now(timezone.utc) + timedelta(days=20),
        liquidity=5000, volume=0, url="",
    )


def test_mock_pipeline_is_deterministic_and_bounded():
    pipe = MockPipeline(Settings())
    m = make_market()
    r1, r2 = pipe.run(m), pipe.run(m)
    assert r1.p_final == r2.p_final  # deterministic
    for p in [*r1.p_forecasters, r1.p_critic, r1.p_final, r1.p_baseline]:
        assert 0.01 <= p <= 0.99
    assert r1.edge == r1.p_final - m.yes_price


def test_mock_pipeline_varies_by_market():
    pipe = MockPipeline(Settings())
    a = pipe.run(make_market("kalshi:AAA"))
    b = pipe.run(make_market("kalshi:BBB"))
    assert a.p_final != b.p_final
