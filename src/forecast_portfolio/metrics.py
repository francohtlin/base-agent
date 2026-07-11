"""Forecast-skill metrics: Brier, log loss, calibration, information coefficient,
directional accuracy. Pure functions over (probability, outcome) and (edge, move)
pairs — no numpy dependency."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime


def brier(pairs: list[tuple[float, int]]) -> float:
    """Mean squared error of probability vs binary outcome (0 = perfect, 0.25 = coin)."""
    if not pairs:
        return float("nan")
    return sum((p - o) ** 2 for p, o in pairs) / len(pairs)


def log_loss(pairs: list[tuple[float, int]], eps: float = 1e-6) -> float:
    if not pairs:
        return float("nan")
    total = 0.0
    for p, o in pairs:
        p = min(max(p, eps), 1 - eps)
        total += -(o * math.log(p) + (1 - o) * math.log(1 - p))
    return total / len(pairs)


def calibration(pairs: list[tuple[float, int]], bins: int = 10) -> list[dict]:
    """Per-bin predicted probability vs realized frequency."""
    table = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [(p, o) for p, o in pairs if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if not members:
            continue
        table.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": len(members),
            "mean_p": sum(p for p, _ in members) / len(members),
            "freq": sum(o for _, o in members) / len(members),
        })
    return table


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3 or n != len(ys):
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def information_coefficient(pairs: list[tuple[float, float]]) -> float:
    """Correlation between edge at scan time and subsequent price move.

    The field-standard skill measure used by the reference system: magnitude-aware
    (being right by 20pp counts more than by 2pp)."""
    return pearson([e for e, _ in pairs], [m for _, m in pairs])


def directional_accuracy(pairs: list[tuple[float, float]]) -> float:
    """Fraction of scans where the market subsequently moved in the predicted direction."""
    decided = [(e, m) for e, m in pairs if e != 0 and m != 0]
    if not decided:
        return float("nan")
    return sum(1 for e, m in decided if (e > 0) == (m > 0)) / len(decided)


# ------------------------------------------------------------- IC assembly

@dataclass
class ScanPoint:
    market_id: str
    ts: datetime
    price: float
    edge: float


def edge_move_pairs(
    scans: list[ScanPoint],
    marks: dict[str, list[tuple[datetime, float]]],
    horizon_days: float,
    tolerance: float = 0.5,
) -> list[tuple[float, float]]:
    """For each scan, pair its edge with the price move to the mark nearest
    (scan_ts + horizon), accepted within ±tolerance×horizon."""
    pairs = []
    for s in scans:
        series = marks.get(s.market_id, [])
        target = horizon_days * 86400
        best, best_err = None, None
        for ts, price in series:
            dt = (ts - s.ts).total_seconds()
            if dt <= 0:
                continue
            err = abs(dt - target)
            if err <= tolerance * target and (best_err is None or err < best_err):
                best, best_err = price, err
        if best is not None:
            pairs.append((s.edge, best - s.price))
    return pairs
