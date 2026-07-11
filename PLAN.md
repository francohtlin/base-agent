# Build Plan — Paper Portfolio for Kalshi/Polymarket Forecasting

Modeled on NYU Agentic Learning AI Lab's live forecasting agent
([forecast.agenticlearning.ai](https://forecast.agenticlearning.ai)), which runs a
seven-stage LLM pipeline over live U.S. prediction markets, paper-trades the edges it
finds, and evaluates itself with information coefficient (IC) against subsequent market
movement. This repo reimplements that architecture as a self-hosted paper portfolio.

## What the reference system does (from its /about and /data pages)

1. **Planning** — Sonnet creates a structured analysis plan.
2. **Research** — no-LLM fanout to ~10 conditional tools gathers evidence.
3. **Evidence synthesis** — Sonnet condenses findings into a dossier.
4. **Independent forecasting** — three parallel Opus forecasters estimate the
   probability *without seeing the market price* (avoids anchoring).
5. **Adversarial review** — an Opus critic introduces the market price and attacks the
   reasoning.
6. **Final integration** — Opus synthesizes all perspectives into one probability.
7. **Baseline** — a zero-shot Sonnet call provides an ablation control.

Six probabilities are recorded per scan (3 forecasters, critic, integrated, baseline) so
you can measure where the analytical lift comes from. Paper trades are simulated fixed-
dollar positions entered when the model's probability diverges from the market price
("edge"); performance is scored with **IC** (correlation between edge and subsequent
price movement, measured at horizons t+1…t+30 days), **directional accuracy**, win rate,
and P&L. Their reported result: multi-step IC ≈ +0.17 over t+1…t+14 vs ≈ +0.02 for
zero-shot and ≈ +0.04 for mean reversion — i.e. the pipeline, not the base model, is
where the signal lives.

## Architecture of this repo

```
                 ┌─────────────────────────────────────────────────┐
  Kalshi API ───►│ markets.py   fetch + screen binary markets      │
  Polymarket ───►│              (liquidity, time-to-close, price)  │
                 └────────────────────┬────────────────────────────┘
                                      ▼
                 ┌─────────────────────────────────────────────────┐
                 │ pipeline.py  7-stage Claude pipeline            │
                 │  1 plan        (light model)                    │
                 │  2 research    (light model + web_search tool)  │
                 │  3 synthesize  (light model)                    │
                 │  4 forecast ×3 (heavy model, price-blind, ∥)    │
                 │  5 critique    (heavy model, sees price)        │
                 │  6 integrate   (heavy model → final p)          │
                 │  7 baseline    (light model, zero-shot control) │
                 └────────────────────┬────────────────────────────┘
                                      ▼
                 ┌─────────────────────────────────────────────────┐
                 │ portfolio.py  SQLite ledger                     │
                 │  scans (all 6 probabilities per market)         │
                 │  trades (paper positions, entry/exit/P&L)       │
                 │  marks (price follow-ups for IC)                │
                 └────────────────────┬────────────────────────────┘
                                      ▼
                 ┌─────────────────────────────────────────────────┐
                 │ metrics.py  Brier · log loss · calibration ·    │
                 │             IC · directional accuracy · P&L     │
                 └─────────────────────────────────────────────────┘
```

### Deviations from the reference, and why

- **Research stage uses Claude's server-side `web_search` tool** instead of a bespoke
  10-tool fanout. It's one API call, needs no scrapers/keys, and is pluggable — the
  stage returns a plain-text evidence dossier, so a custom fanout can replace it later
  without touching other stages.
- **Models**: heavy stages default to `claude-opus-4-8`, light stages to
  `claude-sonnet-5` (both configurable via env). The reference used the then-current
  Sonnet/Opus split for the same reason: planning/synthesis are cheap glue, the
  forecast/critique/integrate steps are where capability pays.
- **Forecaster diversity** comes from three analyst personas (outside view / inside
  view / skeptic) rather than sampling temperature — current Opus models don't accept
  sampling parameters, and persona framing decorrelates errors more reliably anyway.

### Paper-trading rules (v1)

- **Universe screen**: binary markets only; yes-price in [0.05, 0.95]; liquidity/volume
  above floor; closes between 1 and 120 days out.
- **Entry**: run the pipeline; `edge = p_final − market_yes_price`. If `edge ≥ +τ` buy
  YES at the ask-ish price; if `edge ≤ −τ` buy NO at `1 − price`. Default τ = 5pp,
  fixed stake $100 per position (both configurable). One open position per market.
- **Marking**: `fp mark` re-fetches prices for every scanned market — this both marks
  open positions to market and accumulates the price-followup series that IC needs.
- **Exit**: hold to resolution (payout $1/contract) by default; `fp resolve --auto`
  detects settled markets from the exchange APIs.
- No fees/slippage modeled in v1 (flagged below as future work).

### Evaluation

- **IC (primary, matches the reference)**: Pearson correlation between `edge` at scan
  time and subsequent price movement at each available horizon. Magnitude-aware.
- **Directional accuracy**: sign agreement between edge and subsequent move.
- **Brier score & log loss** on resolved markets, for the final probability *and* every
  ablation stage (forecaster mean, critic, baseline) — this reproduces the reference's
  key chart: does the pipeline beat zero-shot?
- **Calibration table**: 10 probability bins, predicted vs realized frequency.
- **P&L**: realized + mark-to-market unrealized, win rate.

### Cost model

Per full forecast: ~3 light calls (plan, synthesis, baseline) + 1 light call with web
search + 5 heavy calls (3 forecasters, critic, integrator). At current list prices
(Opus 4.8 $5/$25 per MTok, Sonnet 5 $3/$15) a typical run lands around **$0.50–$1.50
per market** depending on dossier size and thinking depth. The zero-shot baseline is
nearly free — which is exactly why IC-vs-baseline is the honest test.

## Milestones

- **M0 — scaffold (this commit)**: repo, market clients, full pipeline, ledger,
  metrics, CLI, tests, mock mode for offline runs.
- **M1 — live operation**: run `fp forecast --top N` daily + `fp mark` a few times a
  day (cron/launchd), accumulate ≥ 50 scans before reading anything into the metrics.
- **M2 — evaluation hardening**: per-stage IC ablations, mean-reversion baseline,
  category breakdowns (politics/econ/sci), fee model, Kelly-fraction sizing option.
- **M3 — surface**: small dashboard (static HTML report or Artifact) mirroring the
  reference site's scans/trades/P&L/forecast-evolution views.

## Risks / honesty notes

- **Three forecasters share one base model** — errors correlate; personas mitigate but
  don't eliminate. The critic stage exists precisely to counter groupthink.
- **IC needs marks**: without regular `fp mark` runs the IC table will be empty.
  Automate it early.
- **Paper only.** Prices are read from public endpoints; no orders are ever placed.
  This is a research instrument, not trading advice.
- **Small-N delusion**: with < 50 resolved markets every metric here is noise. The
  reference site's 55% win rate is over hundreds of trades.
