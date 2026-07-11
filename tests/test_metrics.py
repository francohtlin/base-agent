import math
from datetime import datetime, timedelta, timezone

from forecast_portfolio.metrics import (
    ScanPoint, brier, calibration, directional_accuracy, edge_move_pairs,
    information_coefficient, log_loss,
)


def test_brier_perfect_and_coin():
    assert brier([(1.0, 1), (0.0, 0)]) == 0.0
    assert brier([(0.5, 1), (0.5, 0)]) == 0.25


def test_log_loss_orders_correctly():
    confident_right = log_loss([(0.9, 1)])
    hedged = log_loss([(0.6, 1)])
    assert confident_right < hedged


def test_calibration_bins():
    pairs = [(0.15, 0), (0.17, 0), (0.85, 1), (0.88, 1)]
    table = calibration(pairs, bins=10)
    assert len(table) == 2
    low = table[0]
    assert low["n"] == 2 and low["freq"] == 0.0


def test_information_coefficient_signs():
    aligned = [(0.1, 0.05), (0.2, 0.10), (-0.1, -0.04), (-0.3, -0.12)]
    assert information_coefficient(aligned) > 0.9
    inverted = [(e, -m) for e, m in aligned]
    assert information_coefficient(inverted) < -0.9


def test_directional_accuracy():
    pairs = [(0.1, 0.02), (0.1, -0.02), (-0.2, -0.05), (-0.2, -0.01)]
    assert directional_accuracy(pairs) == 0.75


def test_directional_accuracy_empty_is_nan():
    assert math.isnan(directional_accuracy([]))


def test_edge_move_pairs_picks_nearest_mark_within_tolerance():
    t0 = datetime(2026, 7, 1, tzinfo=timezone.utc)
    scans = [ScanPoint(market_id="m", ts=t0, price=0.40, edge=0.10)]
    marks = {
        "m": [
            (t0, 0.40),                                # t=0 mark: ignored (dt <= 0)
            (t0 + timedelta(days=1.1), 0.46),           # nearest to t+1d
            (t0 + timedelta(days=6), 0.55),             # outside t+1d tolerance
        ]
    }
    pairs = edge_move_pairs(scans, marks, horizon_days=1)
    assert pairs == [(0.10, 0.46 - 0.40)]
    # No mark near t+14d -> no pair
    assert edge_move_pairs(scans, marks, horizon_days=14) == []
