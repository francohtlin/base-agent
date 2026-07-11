"""Multi-step forecasting pipeline after "Alive and Predicting: A Live Evaluation
of Multi-Step Forecasting Agents" (Wu, Dai & Ren, NYU/UChicago — the system live
at forecast.agenticlearning.ai). Stage numbering follows the paper:

  0. zero-shot baseline (light model)  single call, no tools, no price -> p0
  1. plan        (light model)  decompose the question, pick tools
  2. research    (non-LLM tool fanout + light model with web_search)
  3. synthesize  (light model)  score/condense the evidence
  4. ensemble    (heavy model)  three price-BLIND perspectives, in parallel:
                                base-rate / evidence-driven / contrarian -> p4
  5. critic      (heavy model)  devil's advocate; the market price is HIDDEN
                                from every stage until here
  6. final       (heavy model)  integrate, output probability + confidence -> p6

Every stage probability is recorded so the report can reproduce the paper's
per-stage information-coefficient ablation.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pydantic import BaseModel

from .config import Settings
from .markets import Market


class SubQuestion(BaseModel):
    question: str
    weight: float


class PlanOut(BaseModel):
    category: str
    sub_questions: list[SubQuestion]  # 2-5 weighted binary sub-questions (paper, stage 1)
    research_queries: list[str]
    wikipedia_topics: list[str]  # 0-3 topics for the base-rate lookup


class EvidenceItem(BaseModel):
    fact: str
    strength: str  # strong | moderate | weak
    credibility: int  # 1-100 (paper, stage 3)
    direction: str  # yes | no | neutral
    priced_in: bool


class SynthesisOut(BaseModel):
    items: list[EvidenceItem]


class ForecastOut(BaseModel):
    probability: float
    rationale: str


class CritiqueOut(BaseModel):
    revised_probability: float
    critique: str


class FinalOut(BaseModel):
    probability: float
    confidence: float
    rationale: str


class BaselineOut(BaseModel):
    probability: float


# The paper's three ensemble perspectives, verbatim from Appendix B, in this
# fixed order (the ledger maps them to columns p_f1/p_f2/p_f3):
PERSONAS = [
    # base-rate
    "You anchor heavily on historical frequencies and reference classes before considering "
    "specific evidence. Start from 'how often does this type of event happen?' and only "
    "deviate with strong evidence.",
    # evidence-driven
    "You weight recent, specific evidence most heavily. Focus on what has CHANGED recently: "
    "new information, breaking developments, and shifts that make this situation different "
    "from historical base rates.",
    # contrarian
    "You actively look for reasons the consensus might be wrong. What are people overlooking? "
    "What scenario would surprise most observers? Weight evidence against the popular "
    "narrative more heavily.",
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
    p_forecasters: list[float]  # [base-rate, evidence-driven, contrarian]
    p_critic: float
    p_final: float
    p_baseline: float
    confidence: float = 0.5
    dossier: str = ""
    rationale: str = ""

    @property
    def edge(self) -> float:
        return self.p_final - self.market_price

    @property
    def p_ensemble(self) -> float:
        return sum(self.p_forecasters) / len(self.p_forecasters)


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
            "Produce:\n"
            "- category: one of politics, economics, science, geopolitics, tech, legal, culture, general\n"
            "- sub_questions: 2-5 weighted binary sub-questions that determine the outcome "
            "(weights sum to ~1.0)\n"
            "- research_queries: 3-5 concrete web search queries for the most decision-relevant, "
            "recent evidence\n"
            "- wikipedia_topics: 0-3 Wikipedia article topics that would give base rates or key "
            "background facts (e.g. 'List of FIFA World Cup finals', a person, a treaty)"
        )
        return self._parse(self.settings.light_model, prompt, PlanOut, max_tokens=4000)

    def research(self, market: Market, plan: PlanOut, evidence_notes: str) -> str:
        subqs = "\n".join(f"- ({q.weight:.0%}) {q.question}" for q in plan.sub_questions)
        queries = "\n".join(f"- {q}" for q in plan.research_queries)
        tools_section = (
            f"\n\nStructured data already gathered by non-LLM tools:\n{evidence_notes}"
            if evidence_notes else ""
        )
        prompt = (
            "Gather evidence for a probabilistic forecast. Search the web for the most recent, "
            "decision-relevant information, then write an evidence dossier: dated facts, base rates, "
            "expert/insider signals, and anything that cuts against the obvious answer. "
            "Cite dates. Do NOT state a probability, and do NOT report or guess this market's "
            "own current price - downstream forecasters must stay price-blind.\n\n"
            f"{_market_brief(market, with_price=False)}\n\n"
            f"Weighted sub-questions to resolve:\n{subqs}\n\n"
            f"Suggested searches:\n{queries}"
            f"{tools_section}"
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
        """Stage 3: label each evidence item with strength, credibility (1-100),
        direction, and whether it is already priced in (paper, Appendix B)."""
        prompt = (
            "Extract the distinct evidence items from this research dossier for the question "
            "below (10-15 items max, keep dates and numbers, preserve disconfirming evidence). "
            "Label each with: strength (strong/moderate/weak), credibility (1-100), direction "
            "(yes/no/neutral - which resolution it supports), and priced_in (would an attentive "
            "market participant already know this?).\n\n"
            f"{_market_brief(market, with_price=False)}\n\nDossier:\n{dossier}"
        )
        try:
            out = self._parse(self.settings.light_model, prompt, SynthesisOut, max_tokens=8000)
            items = out.items
        except Exception:
            return dossier  # labeling is an enhancement, not a gate
        if not items:
            return dossier
        return "\n".join(
            f"- [{i.strength} | cred {i.credibility}/100 | supports {i.direction.upper()}"
            f"{' | likely priced in' if i.priced_in else ''}] {i.fact}"
            for i in items
        )

    def forecast_one(self, market: Market, evidence: str, persona: str) -> ForecastOut:
        prompt = (
            f"{persona}\n\n"
            "Estimate the probability that the following prediction-market question resolves YES. "
            "You are deliberately NOT shown the market price - form your estimate independently "
            "from the resolution rules, time remaining, and the labeled evidence.\n\n"
            f"{_market_brief(market, with_price=False)}\n\nEvidence:\n{evidence}\n\n"
            "Give your probability (0-1) and a concise rationale."
        )
        out = self._parse(self.settings.heavy_model, prompt, ForecastOut, thinking=True)
        out.probability = _clamp(out.probability)
        return out

    def critique(
        self, market: Market, evidence: str, forecasts: list[ForecastOut], price_notes: str
    ) -> CritiqueOut:
        panel = "\n\n".join(
            f"Forecaster {i + 1}: p={f.probability:.2f}\n{f.rationale}" for i, f in enumerate(forecasts)
        )
        snapshot = f"\n\nMarket microstructure (revealed only at this stage):\n{price_notes}" if price_notes else ""
        prompt = (
            "You are an adversarial reviewer (devil's advocate) of a forecasting panel. The panel "
            "did NOT see the market price; you do. The market aggregates real-money opinion - if "
            "the panel diverges from it, either the panel found real edge or it is missing "
            "something. Attack the panel's reasoning: stale evidence, ignored base rates, math "
            "errors, misread resolution rules, wishful thinking. Flag price moves with no visible "
            "catalyst and thin markets whose price deserves less deference. Then give your revised "
            "probability of YES.\n\n"
            f"{_market_brief(market, with_price=True)}{snapshot}\n\n"
            f"Evidence:\n{evidence}\n\nPanel:\n{panel}"
        )
        out = self._parse(self.settings.heavy_model, prompt, CritiqueOut, thinking=True)
        out.revised_probability = _clamp(out.revised_probability)
        return out

    def integrate(
        self, market: Market, evidence: str, forecasts: list[ForecastOut],
        crit: CritiqueOut, price_notes: str,
    ) -> FinalOut:
        panel = "\n".join(f"Forecaster {i + 1}: {f.probability:.2f}" for i, f in enumerate(forecasts))
        snapshot = f"\n\nMarket microstructure:\n{price_notes}" if price_notes else ""
        prompt = (
            "Produce the final probability that this question resolves YES, integrating the "
            "independent panel, the adversarial critique, and the market price. Deviate from the "
            "market only where the evidence justifies it; a thin or spiking market deserves less "
            "deference than a deep one. Stay calibrated.\n\n"
            f"{_market_brief(market, with_price=True)}{snapshot}\n\n"
            f"Evidence:\n{evidence}\n\nPanel probabilities:\n{panel}\n\n"
            f"Critique (revised p={crit.revised_probability:.2f}):\n{crit.critique}\n\n"
            "Give: the final probability (0-1); your confidence in that probability (0-1, how "
            "much you would stake on this estimate vs the market's); and a 3-5 sentence rationale."
        )
        out = self._parse(self.settings.heavy_model, prompt, FinalOut, thinking=True)
        out.probability = _clamp(out.probability)
        out.confidence = min(max(out.confidence, 0.0), 1.0)
        return out

    def baseline(self, market: Market) -> float:
        """Stage 0: frontier-tier single call, no tools, no price (the paper runs
        the baseline on the same model class as the forecasters, so the multi-step
        gain can't be explained by model capability)."""
        prompt = (
            "Without any research, estimate the probability (0-1) that this prediction-market "
            f"question resolves YES.\n\n{_market_brief(market, with_price=False)}"
        )
        out = self._parse(self.settings.heavy_model, prompt, BaselineOut, thinking=True)
        return _clamp(out.probability)

    # ------------------------------------------------------------- driver

    def run(self, market: Market) -> PipelineResult:
        from . import research_tools

        plan = self.plan(market)

        # Stage 2a: non-LLM tool fanout. PRICE_TOOLS (own-market snapshot,
        # orderbook) are held back from every price-blind stage and revealed to
        # the critic at stage 5; the rest is ordinary evidence.
        tool_sections = research_tools.run_tools(market, wiki_topics=plan.wikipedia_topics)
        price_notes = "\n\n".join(
            f"[{name}]\n{tool_sections.pop(name)}"
            for name in research_tools.PRICE_TOOLS if name in tool_sections
        )
        evidence_notes = "\n\n".join(f"[{k}]\n{v}" for k, v in tool_sections.items())

        # Stage 2b: LLM web research on top of the tool output.
        try:
            dossier = self.research(market, plan, evidence_notes)
        except Exception as exc:  # web search may be unavailable on some orgs/plans
            dossier = (evidence_notes + f"\n\n(web research unavailable: {exc})").strip()

        # Stage 3: label every evidence item (strength / credibility / direction /
        # priced-in) — falls back to the raw dossier if labeling fails.
        evidence = self.synthesize(market, dossier)

        with ThreadPoolExecutor(max_workers=self.settings.n_forecasters) as pool:
            forecasts = list(pool.map(
                lambda persona: self.forecast_one(market, evidence, persona),
                PERSONAS[: self.settings.n_forecasters],
            ))

        crit = self.critique(market, evidence, forecasts, price_notes)
        final = self.integrate(market, evidence, forecasts, crit, price_notes)
        p_baseline = self.baseline(market)

        return PipelineResult(
            market_id=market.id,
            market_price=market.yes_price,
            p_forecasters=[f.probability for f in forecasts],
            p_critic=crit.revised_probability,
            p_final=final.probability,
            p_baseline=p_baseline,
            confidence=final.confidence,
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
