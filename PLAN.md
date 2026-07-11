# Build Plan — Paper Portfolio for Kalshi/Polymarket Forecasting

Modeled on NYU Agentic Learning AI Lab's live forecasting agent
([forecast.agenticlearning.ai](https://forecast.agenticlearning.ai)) and the paper
behind it: **"Alive and Predicting: A Live Evaluation of Multi-Step Forecasting
Agents"** — Will Wu, Hui Dai, Mengye Ren (NYU / UChicago),
[OpenReview SXVjN9VLeJ](https://openreview.net/pdf?id=SXVjN9VLeJ). The paper's framing:
LLM forecasters are usually scored only on final probabilities for already-resolved
questions; instead, deploy the agent live, record every intermediate forecast, evidence
item, and tool call, and ask which stages and tools actually contribute accuracy. This
repo reimplements that architecture as a self-hosted paper portfolio.

## The reference pipeline (paper, Fig. "Agent Pipeline")

| Stage | Name | What it does | Sees market price? |
|---|---|---|---|
| 0 | Zero-shot baseline | Single LLM call, no tools → p̂₀; runs in parallel on the same question with the same evidence cutoff | no |
| 1 | Plan | Decompose the question, pick tools | no |
| 2 | Research | **10 conditional tools**: FRED, code, Wiki, Kalshi, orderbook, articles, web, Congress, courts, earnings | no (tools may read prices; blind stages don't see the primary market's) |
| 3 | Synthesise | Score the evidence | no |
| 4 | Ensemble | Three price-blind perspectives — **base-rate, evidence-driven, contrarian** → p̂₄ | **no** |
| 5 | Critic | Devil's advocate; market price revealed *here* for the first time | **yes** |
| 6 | Final | Integrate everything → p̂₆ + confidence | yes |

Every stage probability is recorded, which enables the paper's two headline analyses:

- **Per-stage IC over days 1–30 after forecast**: final (post-critic) tops the chart at
  ≈ 0.15–0.20; ensemble average and individual perspectives sit below it; zero-shot is
  lowest (≈ 0.02–0.08). The pipeline, not the base model, is where the signal lives.
- **Conviction**: grouping scans by |edge|, **only edges above ~15 percentage points
  hold a clearly positive IC** (≈ 0.2–0.3); sub-5pp and 5–15pp edges hover at zero or
  negative. The trading agent typically trades only on 15+pp calls.

Trades are simulated fixed-dollar positions; performance is IC (magnitude-aware,
field-standard), directional accuracy, win rate, and P&L.

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

- **Research fanout is 2 native tools + web search, not 10.** `research_tools.py`
  implements the market-native tools the paper's example forecast leans on —
  `market_snapshot` (bid/ask, spread, 1d/7d/30d momentum, volume, open interest ≈ the
  paper's Kalshi/orderbook tools) and `sister_markets` (prices of other markets in the
  same event — the "sister market 34%" signal). Web/news/wiki evidence comes from
  Claude's server-side `web_search` tool. FRED, Congress, courts, earnings are declared
  future slots: each is one function added to `TOOLS`.
- **Price-blind boundary**: the paper reveals the market price at stage 5. We enforce
  that by withholding the *primary market's own snapshot* (level and momentum) from
  research/synthesis/ensemble and handing it to the critic and final stages only;
  sister-market prices count as ordinary evidence. (The paper's example shows momentum
  surfacing pre-critic; we hold back both level and momentum — strictly blinder.)
- **Models**: heavy stages default to `claude-opus-4-8`, light stages to
  `claude-sonnet-5` (both configurable via env). Planning/synthesis are cheap glue; the
  ensemble/critic/final steps are where capability pays. The paper's ensemble mixes
  model families (Opus + a non-Claude model); ours uses one family with three
  perspective prompts — the perspective names and order match the paper exactly
  (base-rate, evidence-driven, contrarian → ledger columns p_f1/p_f2/p_f3).
- **Perspective diversity via prompts, not temperature** — current Opus models don't
  accept sampling parameters, and the paper's own decorrelation mechanism is the
  perspective framing.

### Paper-trading rules (v1)

- **Universe screen**: binary markets only; yes-price in [0.05, 0.95]; liquidity/volume
  above floor; closes between 1 and 120 days out.
- **Entry**: run the pipeline; `edge = p_final − market_yes_price`. If `edge ≥ +τ` buy
  YES at the ask-ish price; if `edge ≤ −τ` buy NO at `1 − price`. **Default τ = 15pp**,
  straight from the paper's conviction result — only 15+pp edges carry positive IC.
  Fixed stake $100 per position (both configurable). One open position per market.
- **Marking**: `fp mark` re-fetches prices for every scanned market — this both marks
  open positions to market and accumulates the price-followup series that IC needs.
- **Exit**: hold to resolution (payout $1/contract) by default; `fp resolve --auto`
  detects settled markets from the exchange APIs.
- No fees/slippage modeled in v1 (flagged below as future work).

### Evaluation

- **Per-stage IC (primary — reproduces the paper's headline chart)**: Pearson
  correlation between each stage's edge (p̂_stage − price) and subsequent price
  movement, at horizons t+1/3/7/14, for all seven lines: zero-shot, base-rate,
  evidence-driven, contrarian, ensemble average, critic, final (post-critic).
- **Conviction buckets (the paper's second chart)**: final-stage IC split by |edge|
  bucket — <5pp, 5–15pp, 15+pp — to verify the trade threshold is earning its keep.
- **Directional accuracy**: sign agreement between edge and subsequent move.
- **Brier score & log loss** on resolved markets, per stage.
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
