import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    heavy_model: str = field(default_factory=lambda: _env("FP_HEAVY_MODEL", "claude-opus-4-8"))
    light_model: str = field(default_factory=lambda: _env("FP_LIGHT_MODEL", "claude-sonnet-5"))
    stake_usd: float = field(default_factory=lambda: float(_env("FP_STAKE_USD", "100")))
    # The paper's conviction analysis: only edges above ~15pp hold a clearly
    # positive information coefficient, and those are what its agent trades on.
    edge_threshold: float = field(default_factory=lambda: float(_env("FP_EDGE_THRESHOLD", "0.15")))
    min_liquidity: float = field(default_factory=lambda: float(_env("FP_MIN_LIQUIDITY", "1000")))
    min_days: float = field(default_factory=lambda: float(_env("FP_MIN_DAYS", "1")))
    max_days: float = field(default_factory=lambda: float(_env("FP_MAX_DAYS", "120")))
    db_path: Path = field(default_factory=lambda: Path(_env("FP_DB_PATH", "data/portfolio.db")))
    web_search_max_uses: int = field(default_factory=lambda: int(_env("FP_WEB_SEARCH_MAX_USES", "6")))
    n_forecasters: int = 3
    # Screen out near-certain markets where there is no information left to trade.
    min_price: float = 0.05
    max_price: float = 0.95
