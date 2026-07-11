# forecast-portfolio

A paper-trading portfolio that forecasts live **Kalshi** and **Polymarket** binary
questions with a multi-stage Claude pipeline, records simulated positions, and scores
itself with the same metrics quant forecasters use (information coefficient, Brier,
calibration, P&L).

Architecture is modeled on NYU Agentic Learning AI Lab's live agent at
[forecast.agenticlearning.ai](https://forecast.agenticlearning.ai) — see
[PLAN.md](PLAN.md) for the full design, the mapping to their seven-stage pipeline, and
the roadmap.

**Paper only.** No orders are ever placed; market data comes from public read-only
endpoints. Not financial advice.

## Quickstart

```sh
uv sync

# Credentials: either
export ANTHROPIC_API_KEY=sk-ant-...
# or, with the Anthropic CLI: ant auth login   (the SDK picks up the profile)

# 1. See what's tradeable right now (no API key needed)
uv run fp scan --limit 15

# 2. Run the full pipeline on the top 3 screened markets and paper-trade any edges
uv run fp forecast --top 3

# 3. Periodically refresh prices (marks open positions + feeds the IC metric)
uv run fp mark

# 4. Settle finished markets and view the scoreboard
uv run fp resolve --auto
uv run fp report
```

Offline / no-key dry run: `uv run fp forecast --top 2 --mock` exercises the whole
scan → forecast → trade → report loop with a deterministic stub pipeline.

## Commands

| Command | What it does |
|---|---|
| `fp scan` | Fetch + screen open binary markets from Kalshi and Polymarket |
| `fp forecast --top N` | Run the 7-stage pipeline, record all stage probabilities, open paper trades where \|edge\| ≥ threshold |
| `fp forecast --market kalshi:TICKER` | Forecast one specific market |
| `fp mark` | Re-fetch current prices: mark-to-market + price-followup series for IC |
| `fp resolve --auto` | Detect settled markets and realize P&L (or `--market … --outcome yes\|no`) |
| `fp report` | P&L, open positions, Brier/log-loss/calibration, IC and directional accuracy |

## Configuration (env vars)

| Var | Default | Meaning |
|---|---|---|
| `FP_HEAVY_MODEL` | `claude-opus-4-8` | Forecasters, critic, integrator |
| `FP_LIGHT_MODEL` | `claude-sonnet-5` | Plan, research, synthesis, baseline |
| `FP_STAKE_USD` | `100` | Fixed stake per paper position |
| `FP_EDGE_THRESHOLD` | `0.05` | Minimum \|p_final − price\| to open a trade |
| `FP_MIN_LIQUIDITY` | `1000` | Screen floor (USD-ish, per venue's own measure) |
| `FP_MIN_DAYS` / `FP_MAX_DAYS` | `1` / `120` | Time-to-close window |
| `FP_DB_PATH` | `data/portfolio.db` | SQLite ledger location |
| `FP_WEB_SEARCH_MAX_USES` | `6` | Web searches allowed in the research stage |

## Layout

```
src/forecast_portfolio/
  markets.py    Kalshi + Polymarket clients, unified Market model, screening
  pipeline.py   7-stage Claude pipeline (plan → research → synthesize →
                3× blind forecast → critique → integrate → baseline)
  portfolio.py  SQLite ledger: scans, trades, marks, resolution, P&L
  metrics.py    Brier, log loss, calibration, IC, directional accuracy
  cli.py        fp entrypoint
tests/          offline unit tests (mock pipeline, fixture payloads)
```

## Development

```sh
uv run pytest
```
