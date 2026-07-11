"""`fp` command-line interface: scan, forecast, mark, resolve, report."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from . import markets as mkt
from .config import Settings
from .metrics import (
    ScanPoint, brier, calibration, directional_accuracy, edge_move_pairs,
    information_coefficient, log_loss,
)
from .pipeline import ForecastPipeline, MockPipeline
from .portfolio import Ledger


def _fmt(x, width=6, digits=3):
    if x is None:
        return " " * width
    try:
        if x != x:  # NaN
            return "  n/a ".rjust(width)
        return f"{x:+.{digits}f}".rjust(width) if x < 0 or digits else f"{x:.{digits}f}".rjust(width)
    except TypeError:
        return str(x).rjust(width)


def _print_market_row(m: mkt.Market):
    days = f"{m.days_to_close:5.1f}d" if m.days_to_close is not None else "     ?"
    print(f"  {m.yes_price:5.2f}  {days}  vol {max(m.liquidity, m.volume):>12,.0f}  "
          f"{m.id[:44]:44}  {m.question[:70]}")


# ------------------------------------------------------------------ commands

def cmd_scan(args, settings: Settings):
    found = mkt.fetch_all(args.source.split(","), limit=args.limit)
    picks = mkt.screen(found, settings)
    print(f"{len(found)} binary markets fetched, {len(picks)} pass the screen "
          f"(liq>={settings.min_liquidity:,.0f}, {settings.min_days}-{settings.max_days}d, "
          f"price {settings.min_price}-{settings.max_price}):\n")
    print("  price   close  volume/liquidity  market id                                     question")
    for m in picks[: args.top]:
        _print_market_row(m)


def cmd_forecast(args, settings: Settings):
    ledger = Ledger(settings.db_path)
    pipeline = (MockPipeline if args.mock else ForecastPipeline)(settings)

    if args.market:
        m = mkt.refresh(args.market)
        if m is None:
            sys.exit(f"market not found: {args.market}")
        targets = [m]
    else:
        picks = mkt.screen(mkt.fetch_all(args.source.split(","), limit=args.limit), settings)
        targets = [m for m in picks if not ledger.has_open_trade(m.id)][: args.top]
        if not targets:
            sys.exit("no screened markets without an existing open position")

    for m in targets:
        print(f"\n=== {m.id}\n    {m.question}\n    market price {m.yes_price:.2f}", flush=True)
        result = pipeline.run(m)
        scan_id = ledger.record_scan(m, result)
        pf = ", ".join(f"{p:.2f}" for p in result.p_forecasters)
        print(f"    ensemble [base-rate, evidence, contrarian] = [{pf}]  critic {result.p_critic:.2f}")
        print(f"    final {result.p_final:.2f}  baseline {result.p_baseline:.2f}  "
              f"edge {result.edge:+.2f}  confidence {result.confidence:.2f}")
        if args.no_trade:
            continue
        trade = ledger.maybe_open_trade(m, result, settings, scan_id)
        if trade:
            print(f"    PAPER TRADE: buy {trade.side.upper()} @ {trade.entry_price:.2f} "
                  f"(${trade.stake:.0f} -> {trade.contracts:.1f} contracts)")
        else:
            print(f"    no trade (|edge| < {settings.edge_threshold} or position already open)")
    ledger.close()


def cmd_mark(args, settings: Settings):
    ledger = Ledger(settings.db_path)
    ids = ledger.tracked_market_ids()
    if not ids:
        sys.exit("nothing to mark - run `fp forecast` first")
    settled_hint = 0
    for market_id in ids:
        try:
            m = mkt.refresh(market_id)
        except Exception as exc:
            print(f"  {market_id}: fetch failed ({exc})")
            continue
        if m is None:
            print(f"  {market_id}: gone from venue")
            continue
        ledger.mark(market_id, m.yes_price)
        note = ""
        if m.resolution:
            note = f"  <- SETTLED {m.resolution.upper()} (run `fp resolve --auto`)"
            settled_hint += 1
        print(f"  {market_id}: {m.yes_price:.2f}{note}")
    print(f"\nmarked {len(ids)} markets" + (f", {settled_hint} awaiting resolution" if settled_hint else ""))
    ledger.close()


def cmd_resolve(args, settings: Settings):
    ledger = Ledger(settings.db_path)
    if args.market and args.outcome:
        settled = ledger.settle(args.market, args.outcome)
        for t in settled:
            print(f"  settled trade #{t.id} {t.side.upper()} -> {t.resolution}: pnl {t.pnl:+.2f}")
        if not settled:
            print(f"  recorded resolution {args.outcome} for {args.market} (no open trades)")
    elif args.auto:
        n = 0
        for market_id in ledger.tracked_market_ids():
            try:
                m = mkt.refresh(market_id)
            except Exception:
                continue
            if m and m.resolution:
                settled = ledger.settle(market_id, m.resolution)
                n += 1
                print(f"  {market_id} -> {m.resolution.upper()} ({len(settled)} trades settled)")
        print(f"\n{n} markets resolved")
    else:
        sys.exit("use --auto, or --market ID --outcome yes|no")
    ledger.close()


def cmd_report(args, settings: Settings):
    ledger = Ledger(settings.db_path)

    # ---- portfolio
    open_trades = ledger.open_trades()
    settled = ledger.settled_trades()
    realized = sum(t.pnl or 0 for t in settled)
    wins = sum(1 for t in settled if (t.pnl or 0) > 0)
    unrealized = 0.0
    print("== Portfolio ==")
    for t in open_trades:
        price = ledger.latest_mark(t.market_id)
        u = t.unrealized(price) if price is not None else None
        unrealized += u or 0
        print(f"  open  #{t.id:<3} {t.side.upper():3} @ {t.entry_price:.2f}  "
              f"${t.stake:.0f}  unrlzd {_fmt(u, 8, 2)}  {t.question[:60]}")
    print(f"  open positions: {len(open_trades)}  unrealized: {unrealized:+.2f}")
    print(f"  settled trades: {len(settled)}  realized P&L: {realized:+.2f}"
          + (f"  win rate: {wins / len(settled):.0%}" if settled else ""))

    # ---- accuracy on resolved scans (ablation across stages)
    rows = ledger.resolved_scans()
    print(f"\n== Accuracy ({len(rows)} resolved scans) ==")
    if rows:
        def pairs(col):
            out = []
            for r in rows:
                if r[col] is not None:
                    out.append((r[col], 1 if r["outcome"] == "yes" else 0))
            return out

        fmean = [((r["p_f1"] + r["p_f2"] + r["p_f3"]) / 3, 1 if r["outcome"] == "yes" else 0)
                 for r in rows if r["p_f1"] is not None]
        market = [(r["market_price"], 1 if r["outcome"] == "yes" else 0) for r in rows]
        print("  stage            brier   logloss")
        for name, pp in [("market price", market), ("baseline", pairs("p_baseline")),
                         ("forecaster mean", fmean), ("critic", pairs("p_critic")),
                         ("final", pairs("p_final"))]:
            print(f"  {name:15} {_fmt(brier(pp), 7)} {_fmt(log_loss(pp), 8)}")
        print("\n  calibration (final):  bin        n   mean_p   freq")
        for row in calibration(pairs("p_final")):
            print(f"                        {row['bin']}  {row['n']:3}   {row['mean_p']:.2f}    {row['freq']:.2f}")
    else:
        print("  (no resolved scans yet)")

    # ---- per-stage IC vs subsequent market movement (the paper's headline chart)
    all_rows = ledger.all_scans()
    marks = {
        r["market_id"]: [(datetime.fromisoformat(m["ts"]), m["yes_price"])
                         for m in ledger.marks_for(r["market_id"])]
        for r in all_rows
    }
    horizons = (1, 3, 7, 14)

    def _ensemble(r):
        if r["p_f1"] is None:
            return None
        return (r["p_f1"] + r["p_f2"] + r["p_f3"]) / 3

    stages = [
        ("zero-shot", lambda r: r["p_baseline"]),
        ("base-rate", lambda r: r["p_f1"]),
        ("evidence-driven", lambda r: r["p_f2"]),
        ("contrarian", lambda r: r["p_f3"]),
        ("ensemble (avg)", _ensemble),
        ("critic", lambda r: r["p_critic"]),
        ("final (post-critic)", lambda r: r["p_final"]),
    ]

    def stage_pairs(prob_fn, horizon):
        points = []
        for r in all_rows:
            p = prob_fn(r)
            if p is None:
                continue
            points.append(ScanPoint(
                market_id=r["market_id"], ts=datetime.fromisoformat(r["ts"]),
                price=r["market_price"], edge=p - r["market_price"],
            ))
        return edge_move_pairs(points, marks, horizon_days=horizon)

    final_counts = {h: len(stage_pairs(stages[-1][1], h)) for h in horizons}
    print("\n== Information coefficient by stage (edge vs subsequent move) ==")
    print("  stage                " + "".join(f"   t+{h:<2}d " for h in horizons))
    print("  n =                  " + "".join(f"  {final_counts[h]:5}  " for h in horizons))
    for name, fn in stages:
        cells = "".join(f" {_fmt(information_coefficient(stage_pairs(fn, h)), 7)}" for h in horizons)
        print(f"  {name:20}{cells}")

    # ---- conviction: IC by |edge| bucket (paper: only 15+pp edges carry signal)
    buckets = [("<5pp", 0.0, 0.05), ("5-15pp", 0.05, 0.15), ("15+pp", 0.15, 1.01)]
    print("\n== Conviction: final-stage IC by |edge| bucket ==")
    print("  bucket   " + "".join(f"      t+{h:<2}d " for h in horizons))
    for label, lo, hi in buckets:
        cells = ""
        for h in horizons:
            pp = [(e, m) for e, m in stage_pairs(stages[-1][1], h) if lo <= abs(e) < hi]
            cells += f" {_fmt(information_coefficient(pp), 7)}({len(pp):2})"
        print(f"  {label:8}{cells}")

    da = {h: directional_accuracy(stage_pairs(stages[-1][1], h)) for h in horizons}
    print("\n  directional accuracy (final): "
          + "  ".join(f"t+{h}d {_fmt(da[h], 5, 2)}" for h in horizons))
    print("\n(IC needs repeated `fp mark` runs to accumulate price follow-ups.)")
    ledger.close()


# ------------------------------------------------------------------ entry

def main(argv=None):
    parser = argparse.ArgumentParser(prog="fp", description="Paper portfolio forecasting Kalshi/Polymarket questions")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scan", help="fetch + screen open binary markets")
    p.add_argument("--source", default="kalshi,polymarket")
    p.add_argument("--limit", type=int, default=100, help="markets to fetch per venue")
    p.add_argument("--top", type=int, default=25, help="rows to display")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("forecast", help="run the pipeline and paper-trade edges")
    p.add_argument("--market", help="single market id, e.g. kalshi:KXTICKER or polymarket:12345")
    p.add_argument("--top", type=int, default=3, help="how many screened markets to forecast")
    p.add_argument("--source", default="kalshi,polymarket")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--no-trade", action="store_true", help="record scans only")
    p.add_argument("--mock", action="store_true", help="offline stub pipeline (no API calls)")
    p.set_defaults(func=cmd_forecast)

    p = sub.add_parser("mark", help="refresh prices for tracked markets")
    p.set_defaults(func=cmd_mark)

    p = sub.add_parser("resolve", help="settle finished markets")
    p.add_argument("--auto", action="store_true", help="detect settled markets from venue APIs")
    p.add_argument("--market")
    p.add_argument("--outcome", choices=["yes", "no"])
    p.set_defaults(func=cmd_resolve)

    p = sub.add_parser("report", help="P&L, accuracy, calibration, IC")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    args.func(args, Settings())


if __name__ == "__main__":
    main()
