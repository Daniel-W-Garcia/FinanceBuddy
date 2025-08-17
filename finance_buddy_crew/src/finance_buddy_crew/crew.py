# crew.py -- Crew definition using pydantic v2 models for task outputs
import os
from typing import List, Optional

# Load .env if available
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

import logging

logger = logging.getLogger(__name__)

# Ensure SEC_USER_AGENT is present and well-formed for SEC access
# SEC prefers a user-agent with contact information, e.g. "FinanceBuddy/1.0 (contact@example.com)"
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT")
if not SEC_USER_AGENT:
    # Provide a safe default (non-personal) to avoid immediate hard failures; prefer explicit env in production
    default_contact = os.getenv("FINBUDDY_CONTACT", "contact@example.com")
    SEC_USER_AGENT = f"FinanceBuddy/1.0 ({default_contact})"
    os.environ["SEC_USER_AGENT"] = SEC_USER_AGENT
    logger.warning(
        "SEC_USER_AGENT not set; using fallback value. Set SEC_USER_AGENT in your environment with a contact email."
    )
else:
    # Quick format check; warn if it doesn't look like it contains contact info
    if "@" not in SEC_USER_AGENT and "(" not in SEC_USER_AGENT:
        logger.warning(
            "SEC_USER_AGENT does not appear to include contact info. SEC recommends including an email/contact in the User-Agent."
        )

# CrewAI imports
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from pydantic import BaseModel, Field, ConfigDict

# Tools (keep your current imports; update paths if you moved modules)
from .tools.extraction_tools import (
    get_sec_info,
    get_recent_filings,
    get_company_facts,
    get_news,
    get_prices,
)

# ---- Pydantic base schema to enforce strictness (v2) ----
# crew.py (excerpt showing BaseSchema update)

class BaseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)
    # Diagnostics fields for partial-output behavior (populated by validator decorator)
    warnings: List[str] = Field(default_factory=list)
    raw_input: Optional[dict] = None


# ---- Task output schemas (Pydantic v2) ----
class ScopeRequestSchema(BaseSchema):
    company: Optional[str] = None
    ticker: Optional[str] = None
    exchange: Optional[str] = None
    timeframe: Optional[str] = None
    focus_areas: List[str] = Field(default_factory=list)
    output_format: Optional[str] = None
    constraints: List[str] = Field(default_factory=list)
    peers: List[str] = Field(default_factory=list)


class FilingSchema(BaseSchema):
    form: Optional[str] = None
    filing_date: Optional[str] = None
    report_date: Optional[str] = None
    accession: Optional[str] = None
    primary_doc: Optional[str] = None
    items: Optional[str] = None
    doc_url: Optional[str] = None
    index_url: Optional[str] = None


class EdgarBasicsSchema(BaseSchema):
    cik10: Optional[str] = None
    company_name: Optional[str] = None
    ceo: Optional[str] = None
    sic: Optional[str] = None
    filings: List[FilingSchema] = Field(default_factory=list)


class MetricPoint(BaseSchema):
    period: Optional[str] = None  # e.g., "2024-12"
    value: Optional[float] = None
    unit: Optional[str] = None


class CompanyFactsSnapshotSchema(BaseSchema):
    currency: Optional[str] = None
    ceo: Optional[str] = None
    revenue: List[MetricPoint] = Field(default_factory=list)
    operating_income: List[MetricPoint] = Field(default_factory=list)
    net_income: List[MetricPoint] = Field(default_factory=list)
    diluted_eps: List[MetricPoint] = Field(default_factory=list)
    shares_diluted: List[MetricPoint] = Field(default_factory=list)
    cash_or_fcf_proxy: List[MetricPoint] = Field(default_factory=list)
    debt: List[MetricPoint] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    raw_facts: Optional[dict] = None


class NewsItemSchema(BaseSchema):
    date: Optional[str] = None
    source: Optional[str] = None
    title: Optional[str] = None
    link: Optional[str] = None
    relevance: Optional[str] = None


class NewsAndCatalystsSchema(BaseSchema):
    items: List[NewsItemSchema] = Field(default_factory=list)
    top_catalysts: List[str] = Field(default_factory=list)
    top_risks: List[str] = Field(default_factory=list)


class ReturnsSchema(BaseSchema):
    d1m: Optional[float] = None
    d3m: Optional[float] = None
    d6m: Optional[float] = None
    d12m: Optional[float] = None
    vol_proxy: Optional[float] = None


class PriceContextSchema(BaseSchema):
    returns: Optional[ReturnsSchema] = None
    notes: List[str] = Field(default_factory=list)


class SynthesisBriefSchema(BaseSchema):
    # Make TL;DR a list defaulting to empty list so missing TL;DR doesn't break validation
    tldr: List[str] = Field(default_factory=list)
    positives: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    notable_filings: List[str] = Field(default_factory=list)
    fundamentals_takeaways: List[str] = Field(default_factory=list)
    recent_catalysts: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    disclaimer: Optional[str] = None
    # Consider adding a warnings field to capture validation warnings / partial-output notices
    warnings: List[str] = Field(default_factory=list)


# LLM/model selection helpers (lazy)
def _llm_high() -> LLM:
    return LLM(model="gpt-5-2025-08-07")


def _llm_med() -> LLM:
    return LLM(model="gpt-5-mini-2025-08-07")


def _llm_low() -> LLM:
    return LLM(model="gpt-5-nano-2025-08-07")


# Output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)


@CrewBase
class FinanceBuddy:
    """FinanceBuddy crew for automated financial research"""

    # Provide defaults so instances are safe to create during tests
    agents: List[BaseAgent] = []
    tasks: List[Task] = []

    @agent
    def intake_coordinator(self) -> Agent:
        # Lightweight coordinator model
        return Agent(
            config=self.agents_config["intake_coordinator"],
            verbose=True,
            llm=_llm_low(),
            max_iter=3,
        )

    @agent
    def research_analyst(self) -> Agent:
        # Provide tools (ensure the get_* symbols match your tool implementations)
        return Agent(
            config=self.agents_config["research_analyst"],
            verbose=True,
            tools=[get_sec_info, get_recent_filings, get_news],
            llm=_llm_med(),
            max_iter=5,
        )

    @agent
    def financial_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["financial_analyst"],
            verbose=True,
            tools=[get_company_facts, get_prices],
            llm=_llm_med(),
            max_iter=5,
        )

    @agent
    def lead_synthesizer(self) -> Agent:
        return Agent(
            config=self.agents_config["lead_synthesizer"],
            verbose=True,
            llm=_llm_high(),
            max_iter=3,
        )

    # ---- Tasks ----
    @task
    def scope_request_task(self) -> Task:
        """Initial task to clarify and scope the research request"""
        return Task(
            config=self.tasks_config["scope_request_task"],
            output_pydantic=ScopeRequestSchema,
            output_file=f"{output_dir}/research_scope.json",
        )

    @task
    def edgar_basics_task(self) -> Task:
        return Task(
            config=self.tasks_config["edgar_basics_task"],
            output_pydantic=EdgarBasicsSchema,
            output_file=f"{output_dir}/edgar_basics.json",
        )

    @task
    def company_facts_snapshot_task(self) -> Task:
        return Task(
            config=self.tasks_config["company_facts_snapshot_task"],
            output_pydantic=CompanyFactsSnapshotSchema,
            output_file=f"{output_dir}/company_facts_snapshot.json",
        )

    @task
    def news_and_catalysts_task(self) -> Task:
        return Task(
            config=self.tasks_config["news_and_catalysts_task"],
            output_pydantic=NewsAndCatalystsSchema,
            output_file=f"{output_dir}/news_and_catalysts.json",
        )

    @task
    def price_context_task(self) -> Task:
        return Task(
            config=self.tasks_config["price_context_task"],
            output_pydantic=PriceContextSchema,
            output_file=f"{output_dir}/price_context.json",
        )

    @task
    def synthesis_brief_task(self) -> Task:
        # Synthesis outputs are structured JSON; render to .json so downstream Python tools can easily consume
        return Task(
            config=self.tasks_config["synthesis_brief_task"],
            output_pydantic=SynthesisBriefSchema,
            output_file=f"{output_dir}/synthesis_brief.json",
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=False,  # Disabled memory for now
        )
