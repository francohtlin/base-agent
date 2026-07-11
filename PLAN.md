# Build Plan — Paper Portfolio for Kalshi/Polymarket Forecasting

Reimplements the system from **"Alive and Predicting: A Live Evaluation of Multi-Step
Forecasting Agents"** — Will Wu, Hui Dai, Mengye Ren (NYU / UChicago), ICML 2026
Workshop on Forecasting as a New Frontier of Intelligence
([OpenReview SXVjN9VLeJ](https://openreview.net/pdf?id=SXVjN9VLeJ)) — live at
[forecast.agenticlearning.ai](https://forecast.agenticlearning.ai). The paper's
framing: LLM forecasters are usually scored only on final probabilities for
already-resolved questions; instead, deploy the agent live, record every intermediate
forecast, evidence item, and tool call, and ask which stages and tools actually
contribute accuracy. Their headline sample: 335 scans over 269 unique Kalshi markets
across a 4-week window.

## The reference pipeline (paper §2 + Appendix B)

| Stage | Name | Tier | What it does | Sees price? |
|---|---|---|---|---|
| 0 | Zero-shot baseline | frontier | Single call on title/rules/dates, no tools → p̂₀; runs in parallel with the same evidence cutoff | no |
| 1 | Plan | fast | Classify the question, decompose into **2–5 weighted binary sub-questions**, select tools | no |
| 2 | Research | no LLM | Execute the selected tools in parallel (10-tool inventory, mean 5.4 used/scan) | tools read prices; blind stages never see the primary market's |
| 3 | Synthesis | fast | Label each evidence item: **strength, credibility 1–100, direction, priced-in** | no |
| 4 | Ensemble | 3× frontier | **base-rate / evidence-driven / contrarian** (verbatim prompts in Appendix B) → p̂₄ = simple average | **no** |
| 5 | Critic | frontier | Devil's advocate challenges the ensemble for reasoning flaws and math errors; price revealed *here* | **yes** |
| 6 | Final | frontier | Integrate critique + price → p̂₆ **+ confidence** | yes |

Six forecasts per scan: {p̂₀, p̂₄ₐ, p̂₄ᵦ, p̂₄𝒸, p̂₄, p̂₆}. Paper models: Claude Sonnet 4.6
(fast tier), Claude Opus 4.7 (frontier tier).

### The paper's results (what this repo's report is built to reproduce)

- **Mean IC over t+1…t+14**: final (post-critic) **+0.17** [0.00, +0.33], ensemble
  average **+0.14**, zero-shot **+0.05**, mean-reversion (p=0.5) **+0.05**. Final is
  the only stage whose 95% bootstrap CI has a non-negative lower bound, and it tops
  every horizon t+1…t+30. The margin is largest in the first two weeks — the market
  then converges toward the agent's earlier view.
- **The critic's gain looks like calibration, not new reasoning**: the trace shows it
  mainly pulling the ensemble toward the market price (+0.14 → +0.17).
- **Diversity collapse**: on 38% of scans the three perspectives differ by < 0.02
  (mean spread 4.3pp, 12% fully degenerate) — the ensemble average adds little over
  any single perspective (all within ±0.03 IC).
- **Conviction is everything**: splitting by |edge| — <5pp (n=141), 5–15pp (n=115),
  15+pp (n=79) — **only 15+pp calls hold a clearly positive IC at every horizon**.
  The trading agent takes positions only on those.
- **Per-tool ΔIC** (IC with tool − IC without): wikipedia_lookup **+0.25**,
  kalshi_orderbook **+0.13**, earnings +0.05; court_docket −0.12, code_execution
  −0.24, fred_data −0.26, congress_bills −0.27. The paper reads negative values as
  selection/usage misalignment, not proof the data is useless.
- **Paper trading**: positions sized by the final forecast at the prevailing price,
  simulated with **realistic bid/ask spreads from the orderbook**; no real capital.

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

- **Tool inventory is evidence-driven, not complete.** `research_tools.py` implements
  the tools the paper's own ablation rewards: `wikipedia_lookup` (ΔIC +0.25, its best
  conditional tool), `kalshi_orderbook` (+0.13), `market_snapshot` (≈ its
  always-invoked kalshi_data), and `sister_markets` (the relative-value signal in its
  example forecast). Web/news evidence comes from Claude's server-side `web_search`
  tool. We **deliberately omit** code_execution, FRED, Congress bills, and court
  docket — all negative ΔIC in the paper — until a better selection policy earns them
  back. Each is one function away (`TOOLS` in research_tools.py).
- **Price-blind boundary**: the paper reveals the market price at stage 5. We enforce
  that by withholding the PRICE_TOOLS sections (own-market snapshot *and* orderbook)
  from research/synthesis/ensemble and handing them to the critic and final stages
  only; sister-market prices count as ordinary evidence.
- **Models**: paper = Sonnet 4.6 fast tier (stages 1, 3) + Opus 4.7 frontier tier
  (stages 0, 4, 5, 6). Ours = the current equivalents, `claude-sonnet-5` and
  `claude-opus-4-8`, same stage assignment — including the zero-shot baseline on the
  frontier model, so the multi-step gain can't be explained by model capability.
- **Ensemble prompts are the paper's, verbatim** (Appendix B), in fixed order →
  ledger columns p_f1/p_f2/p_f3. Expect the paper's diversity collapse (38% of scans
  within 2pp) — the critic stage, not the ensemble spread, is where the extra IC
  comes from. We additionally record the critic's own revised probability (a 7th
  data point the paper doesn't log) for finer ablation.
- **Perspective diversity via prompts, not temperature** — current Opus models don't
  accept sampling parameters, and the paper's own decorrelation mechanism is the
  perspective framing.

### Paper-trading rules (v1)

- **Universe screen**: binary markets only; yes-price in [0.05, 0.95]; liquidity/volume
  above floor; closes between 1 and 120 days out.
- **Entry**: run the pipeline; `edge = p_final − market_yes_price` (mid). If
  `edge ≥ +τ` buy YES **at the ask**; if `edge ≤ −τ` buy NO **at 1 − bid** — fills
  cross the spread like the paper's simulated trades (mid-price fallback when the
  venue exposes no book). **Default τ = 15pp**, straight from the paper's conviction
  result — only 15+pp edges carry positive IC. Fixed stake $100 per position (both
  configurable). One open position per market.
- **Marking**: `fp mark` re-fetches prices for every scanned market — this both marks
  open positions to market and accumulates the price-followup series that IC needs.
- **Exit**: hold to resolution (payout $1/contract) by default; `fp resolve --auto`
  detects settled markets from the exchange APIs.
- Spreads are modeled (entries cross bid/ask); exchange fees are not yet.

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
