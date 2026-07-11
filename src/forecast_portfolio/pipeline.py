"""Seven-stage forecasting pipeline, modeled on forecast.agenticlearning.ai:

  1. plan        (light model)  structured analysis plan
  2. research    (light model + server-side web_search) evidence dossier
  3. synthesize  (light model)  condensed dossier
  4. forecast ×3 (heavy model)  independent, price-BLIND probabilities, in parallel
  5. critique    (heavy model)  adversarial review that introduces the market price
  6. integrate   (heavy model)  final probability
  7. baseline    (light model)  zero-shot control for ablation

All six probabilities are recorded so the ledger can answer "where does the lift
come from?" the same way the reference system does.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pydantic import BaseModel

from .config import Settings
from .markets import Market


class PlanOut(BaseModel):
    key_questions: list[str]
    research_queries: list[str]


class ForecastOut(BaseModel):
    probability: float
    rationale: str


class CritiqueOut(BaseModel):
    revised_probability: float
    critique: str


class FinalOut(BaseModel):
    probability: float
    rationale: str


class BaselineOut(BaseModel):
    probability: float


PERSONAS = [
    "an outside-view forecaster: anchor on reference classes and base rates before any story-specific detail",
    "an inside-view analyst: weigh the latest concrete evidence and causal mechanics most heavily",
    "a skeptic: assume the consensus narrative is wrong and stress-test how this resolves the unpopular way",
]


def _clamp(p: float) -> float:
    return min(max(float(p), 0.01), 0.99)


def _market_brief(market: Market, with_price: bool) -> str:
    close = market.close_time.date().isoformat() if market.close_time else "unknown"
    lines = [
        f"Question: {market.question}",
        f"Resolution rules: {market.description or '(none provided)'}",
        f"Market closes: {close}",
        f"Today's date: {datetime.now(timezone.utc).date().isoformat()}",
        f"Venue: {market.source}",
    ]
    if with_price:
        lines.append(f"Current market price of YES: {market.yes_price:.2f}")
    return "\n".join(lines)


@dataclass
class PipelineResult:
    market_id: str
    market_price: float
    p_forecasters: list[float]
    p_critic: float
    p_final: float
    p_baseline: float
    dossier: str = ""
    rationale: str = ""

    @property
    def edge(self) -> float:
        return self.p_final - self.market_price


class ForecastPipeline:
    def __init__(self, settings: Settings, client=None):
        self.settings = settings
        if client is None:
            import anthropic

            # Zero-arg client: resolves ANTHROPIC_API_KEY or an `ant auth login` profile.
            client = anthropic.Anthropic()
        self.client = client

    # ------------------------------------------------------------- stages

    def _parse(self, model: str, prompt: str, out_type, max_tokens: int = 16000, thinking: bool = False):
        kwargs = {}
        if thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        resp = self.client.messages.parse(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            output_format=out_type,
            **kwargs,
        )
        return resp.parsed_output

    def plan(self, market: Market) -> PlanOut:
        prompt = (
            "You are planning research for a probabilistic forecast on a prediction-market question.\n\n"
            f"{_market_brief(market, with_price=False)}\n\n"
            "List the 3-5 key sub-questions that determine the outcome, and 3-5 concrete web "
            "search queries that would surface the most decision-relevant, recent evidence."
        )
        return self._parse(self.settings.light_model, prompt, PlanOut, max_tokens=4000)

    def research(self, market: Market, plan: PlanOut) -> str:
        queries = "\n".join(f"- {q}" for q in plan.research_queries)
        prompt = (
            "Gather evidence for a probabilistic forecast. Search the web for the most recent, "
            "decision-relevant information, then write an evidence dossier: dated facts, base rates, "
            "expert/insider signals, and anything that cuts against the obvious answer. "
            "Cite dates. Do NOT state a probability.\n\n"
            f"{_market_brief(market, with_price=False)}\n\nSuggested searches:\n{queries}"
        )
        messages = [{"role": "user", "content": prompt}]
        tools = [{
            "type": "web_search_20260209",
            "name": "web_search",
            "max_uses": self.settings.web_search_max_uses,
        }]
        resp = None
        for _ in range(4):  # pause_turn continuation cap
            resp = self.client.messages.create(
                model=self.settings.light_model,
                max_tokens=16000,
                tools=tools,
                messages=messages,
            )
            if resp.stop_reason != "pause_turn":
                break
            messages = messages + [{"role": "assistant", "content": resp.content}]
        return "\n".join(b.text for b in resp.content if b.type == "text").strip()

    def synthesize(self, market: Market, dossier: str) -> str:
        prompt = (
            "Condense this research dossier to the 10-15 facts most relevant to forecasting the "
            "question below. Keep dates and numbers. Preserve disconfirming evidence.\n\n"
            f"{_market_brief(market, with_price=False)}\n\nDossier:\n{dossier}"
        )
        resp = self.client.messages.create(
            model=self.settings.light_model,
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt}],
        )
        return "\n".join(b.text for b in resp.content if b.type == "text").strip()

    def forecast_one(self, market: Market, evidence: str, persona: str) -> ForecastOut:
        prompt = (
            f"You are {persona}.\n\n"
            "Estimate the probability that the following prediction-market question resolves YES. "
            "You are deliberately NOT shown the market price - form your estimate independently. "
            "Reason from base rates, the resolution rules, time remaining, and the evidence.\n\n"
            f"{_market_brief(market, with_price=False)}\n\nEvidence:\n{evidence}\n\n"
            "Give your probability (0-1) and a concise rationale."
        )
        out = self._parse(self.settings.heavy_model, prompt, ForecastOut, thinking=True)
        out.probability = _clamp(out.probability)
        return out

    def critique(self, market: Market, evidence: str, forecasts: list[ForecastOut]) -> CritiqueOut:
        panel = "\n\n".join(
            f"Forecaster {i + 1}: p={f.probability:.2f}\n{f.rationale}" for i, f in enumerate(forecasts)
        )
        prompt = (
            "You are an adversarial reviewer of a forecasting panel. The panel did NOT see the "
            "market price; you do. The market aggregates real-money opinion - if the panel "
            "diverges from it, either the panel found real edge or it is missing something. "
            "Attack the panel's reasoning: stale evidence, ignored base rates, misread resolution "
            "rules, wishful thinking. Then give your revised probability of YES.\n\n"
            f"{_market_brief(market, with_price=True)}\n\nEvidence:\n{evidence}\n\nPanel:\n{panel}"
        )
        out = self._parse(self.settings.heavy_model, prompt, CritiqueOut, thinking=True)
        out.revised_probability = _clamp(out.revised_probability)
        return out

    def integrate(self, market: Market, evidence: str, forecasts: list[ForecastOut], crit: CritiqueOut) -> FinalOut:
        panel = "\n".join(f"Forecaster {i + 1}: {f.probability:.2f}" for i, f in enumerate(forecasts))
        prompt = (
            "Produce the final probability that this question resolves YES, integrating the "
            "independent panel, the adversarial critique, and the market price. Deviate from the "
            "market only where the evidence justifies it; stay calibrated.\n\n"
            f"{_market_brief(market, with_price=True)}\n\nEvidence:\n{evidence}\n\n"
            f"Panel probabilities:\n{panel}\n\n"
            f"Critique (revised p={crit.revised_probability:.2f}):\n{crit.critique}\n\n"
            "Give the final probability (0-1) and a 3-5 sentence rationale."
        )
        out = self._parse(self.settings.heavy_model, prompt, FinalOut, thinking=True)
        out.probability = _clamp(out.probability)
        return out

    def baseline(self, market: Market) -> float:
        prompt = (
            "Without any research, estimate the probability (0-1) that this prediction-market "
            f"question resolves YES.\n\n{_market_brief(market, with_price=False)}"
        )
        out = self._parse(self.settings.light_model, prompt, BaselineOut, max_tokens=4000)
        return _clamp(out.probability)

    # ------------------------------------------------------------- driver

    def run(self, market: Market) -> PipelineResult:
        plan = self.plan(market)
        try:
            dossier = self.research(market, plan)
        except Exception as exc:  # web search may be unavailable on some orgs/plans
            dossier = f"(research stage unavailable: {exc})"
        evidence = self.synthesize(market, dossier) if len(dossier) > 2000 else dossier

        with ThreadPoolExecutor(max_workers=self.settings.n_forecasters) as pool:
            forecasts = list(pool.map(
                lambda persona: self.forecast_one(market, evidence, persona),
                PERSONAS[: self.settings.n_forecasters],
            ))

        crit = self.critique(market, evidence, forecasts)
        final = self.integrate(market, evidence, forecasts, crit)
        p_baseline = self.baseline(market)

        return PipelineResult(
            market_id=market.id,
            market_price=market.yes_price,
            p_forecasters=[f.probability for f in forecasts],
            p_critic=crit.revised_probability,
            p_final=final.probability,
            p_baseline=p_baseline,
            dossier=evidence,
            rationale=final.rationale,
        )


class MockPipeline:
    """Deterministic offline stand-in: exercises the full scan->trade->report loop
    without API calls. Probability is a stable hash of the market id, mildly pulled
    toward the market price so mock edges look plausible."""

    def __init__(self, settings: Settings, client=None):
        self.settings = settings

    def run(self, market: Market) -> PipelineResult:
        seed = int(hashlib.sha256(market.id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        p = _clamp(0.5 * market.yes_price + 0.5 * seed)
        spread = 0.06 * (seed - 0.5)
        return PipelineResult(
            market_id=market.id,
            market_price=market.yes_price,
            p_forecasters=[_clamp(p - spread), _clamp(p), _clamp(p + spread)],
            p_critic=_clamp((p + market.yes_price) / 2),
            p_final=p,
            p_baseline=_clamp(seed),
            dossier="(mock run - no research performed)",
            rationale="(mock run)",
        )
