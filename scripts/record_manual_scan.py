"""Record a manually-produced forecast (analyst-in-the-loop) into the ledger.

Used when the stage probabilities come from a human or an interactive Claude
session instead of the API pipeline. Refreshes the market for an up-to-date
entry price, records the scan (which also writes the t=0 mark), and opens a
paper trade if the edge clears the threshold.

Usage:
  uv run python scripts/record_manual_scan.py --market kalshi:TICKER \
      --p-base-rate 0.3 --p-evidence 0.35 --p-contrarian 0.25 \
      --p-critic 0.3 --p-final 0.32 --p-baseline 0.4 \
      --confidence 0.6 --rationale "..." [--no-trade]
"""

import argparse

from forecast_portfolio import markets as mkt
from forecast_portfolio.config import Settings
from forecast_portfolio.pipeline import PipelineResult
from forecast_portfolio.portfolio import Ledger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True)
    for name in ("p-base-rate", "p-evidence", "p-contrarian", "p-critic", "p-final", "p-baseline"):
        ap.add_argument(f"--{name}", type=float, required=True)
    ap.add_argument("--confidence", type=float, default=0.5)
    ap.add_argument("--rationale", default="")
    ap.add_argument("--no-trade", action="store_true")
    args = ap.parse_args()

    settings = Settings()
    market = mkt.refresh(args.market)
    if market is None or market.status != "open":
        raise SystemExit(f"market not open/found: {args.market}")

    result = PipelineResult(
        market_id=market.id,
        market_price=market.yes_price,
        p_forecasters=[args.p_base_rate, args.p_evidence, args.p_contrarian],
        p_critic=args.p_critic,
        p_final=args.p_final,
        p_baseline=args.p_baseline,
        confidence=args.confidence,
        rationale=f"[manual scan] {args.rationale}",
        dossier="(manual scan - evidence gathered interactively)",
    )
    ledger = Ledger(settings.db_path)
    scan_id = ledger.record_scan(market, result)
    print(f"scan #{scan_id}: {market.id} price={market.yes_price:.2f} "
          f"final={result.p_final:.2f} edge={result.edge:+.2f}")
    if not args.no_trade:
        trade = ledger.maybe_open_trade(market, result, settings, scan_id)
        if trade:
            print(f"PAPER TRADE #{trade.id}: {trade.side.upper()} @ {trade.entry_price:.2f} "
                  f"(${trade.stake:.0f} -> {trade.contracts:.1f} contracts)")
        else:
            print(f"no trade (|edge| < {settings.edge_threshold:.2f})")
    ledger.close()


if __name__ == "__main__":
    main()
