"""SQLite ledger: scans (all stage probabilities), paper trades, price marks,
and resolutions.

Trade mechanics (mirrors exchange payoffs, no fees in v1):
  buy YES at price p:      contracts = stake / p        wins $1/contract if YES
  buy NO  at price 1 - p:  contracts = stake / (1 - p)  wins $1/contract if NO
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import Settings
from .markets import Market
from .pipeline import PipelineResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  market_id TEXT NOT NULL,
  source TEXT,
  question TEXT,
  market_price REAL NOT NULL,
  close_time TEXT,
  p_f1 REAL, p_f2 REAL, p_f3 REAL,
  p_critic REAL, p_final REAL, p_baseline REAL,
  edge REAL,
  confidence REAL,
  rationale TEXT,
  dossier TEXT
);
CREATE TABLE IF NOT EXISTS trades(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  scan_id INTEGER,
  ts TEXT NOT NULL,
  market_id TEXT NOT NULL,
  question TEXT,
  side TEXT NOT NULL,
  entry_price REAL NOT NULL,
  stake REAL NOT NULL,
  contracts REAL NOT NULL,
  edge REAL,
  status TEXT NOT NULL DEFAULT 'open',
  resolution TEXT,
  pnl REAL,
  settled_ts TEXT
);
CREATE TABLE IF NOT EXISTS marks(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  market_id TEXT NOT NULL,
  yes_price REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS resolutions(
  market_id TEXT PRIMARY KEY,
  outcome TEXT NOT NULL,
  ts TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Trade:
    id: int
    market_id: str
    question: str
    side: str
    entry_price: float
    stake: float
    contracts: float
    status: str
    resolution: Optional[str]
    pnl: Optional[float]

    def unrealized(self, current_yes_price: float) -> float:
        value = current_yes_price if self.side == "yes" else 1 - current_yes_price
        return self.contracts * value - self.stake


class Ledger:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self):
        self.conn.close()

    # ------------------------------------------------------------- scans

    def record_scan(self, market: Market, result: PipelineResult) -> int:
        pf = (result.p_forecasters + [None, None, None])[:3]
        cur = self.conn.execute(
            """INSERT INTO scans(ts, market_id, source, question, market_price, close_time,
                                 p_f1, p_f2, p_f3, p_critic, p_final, p_baseline, edge,
                                 confidence, rationale, dossier)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                _now(), market.id, market.source, market.question, market.yes_price,
                market.close_time.isoformat() if market.close_time else None,
                pf[0], pf[1], pf[2], result.p_critic, result.p_final, result.p_baseline,
                result.edge, result.confidence, result.rationale, result.dossier,
            ),
        )
        self.conn.commit()
        # Every scan doubles as a t=0 price mark for the IC series.
        self.mark(market.id, market.yes_price)
        return cur.lastrowid

    # ------------------------------------------------------------- trades

    def has_open_trade(self, market_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM trades WHERE market_id=? AND status='open' LIMIT 1", (market_id,)
        ).fetchone()
        return row is not None

    def maybe_open_trade(
        self, market: Market, result: PipelineResult, settings: Settings, scan_id: int
    ) -> Optional[Trade]:
        edge = result.edge
        if abs(edge) < settings.edge_threshold or self.has_open_trade(market.id):
            return None
        side = "yes" if edge > 0 else "no"
        entry = market.yes_price if side == "yes" else 1 - market.yes_price
        if not 0 < entry < 1:
            return None
        contracts = settings.stake_usd / entry
        cur = self.conn.execute(
            """INSERT INTO trades(scan_id, ts, market_id, question, side, entry_price,
                                  stake, contracts, edge)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (scan_id, _now(), market.id, market.question, side, entry,
             settings.stake_usd, contracts, edge),
        )
        self.conn.commit()
        return self.get_trade(cur.lastrowid)

    def get_trade(self, trade_id: int) -> Trade:
        return self._trade(self.conn.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone())

    @staticmethod
    def _trade(row: sqlite3.Row) -> Trade:
        return Trade(
            id=row["id"], market_id=row["market_id"], question=row["question"],
            side=row["side"], entry_price=row["entry_price"], stake=row["stake"],
            contracts=row["contracts"], status=row["status"],
            resolution=row["resolution"], pnl=row["pnl"],
        )

    def open_trades(self) -> list[Trade]:
        rows = self.conn.execute("SELECT * FROM trades WHERE status='open' ORDER BY ts").fetchall()
        return [self._trade(r) for r in rows]

    def settled_trades(self) -> list[Trade]:
        rows = self.conn.execute("SELECT * FROM trades WHERE status='settled' ORDER BY ts").fetchall()
        return [self._trade(r) for r in rows]

    # ------------------------------------------------------------- marks

    def mark(self, market_id: str, yes_price: float):
        self.conn.execute(
            "INSERT INTO marks(ts, market_id, yes_price) VALUES(?,?,?)",
            (_now(), market_id, yes_price),
        )
        self.conn.commit()

    def latest_mark(self, market_id: str) -> Optional[float]:
        row = self.conn.execute(
            "SELECT yes_price FROM marks WHERE market_id=? ORDER BY ts DESC LIMIT 1", (market_id,)
        ).fetchone()
        return row["yes_price"] if row else None

    def tracked_market_ids(self) -> list[str]:
        rows = self.conn.execute(
            """SELECT DISTINCT market_id FROM scans
               WHERE market_id NOT IN (SELECT market_id FROM resolutions)"""
        ).fetchall()
        return [r["market_id"] for r in rows]

    # ------------------------------------------------------------- settle

    def settle(self, market_id: str, outcome: str) -> list[Trade]:
        assert outcome in ("yes", "no")
        self.conn.execute(
            "INSERT OR REPLACE INTO resolutions(market_id, outcome, ts) VALUES(?,?,?)",
            (market_id, outcome, _now()),
        )
        settled = []
        for t in self.open_trades():
            if t.market_id != market_id:
                continue
            won = t.side == outcome
            pnl = (t.contracts * 1.0 - t.stake) if won else -t.stake
            self.conn.execute(
                """UPDATE trades SET status='settled', resolution=?, pnl=?, settled_ts=?
                   WHERE id=?""",
                (outcome, pnl, _now(), t.id),
            )
            settled.append(self.get_trade(t.id))
        self.conn.commit()
        return settled

    # ------------------------------------------------------------- report queries

    def resolved_scans(self) -> list[sqlite3.Row]:
        """Scans joined with the eventual outcome — the accuracy dataset."""
        return self.conn.execute(
            """SELECT s.*, r.outcome FROM scans s
               JOIN resolutions r ON r.market_id = s.market_id
               ORDER BY s.ts"""
        ).fetchall()

    def all_scans(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM scans ORDER BY ts").fetchall()

    def marks_for(self, market_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT ts, yes_price FROM marks WHERE market_id=? ORDER BY ts", (market_id,)
        ).fetchall()
