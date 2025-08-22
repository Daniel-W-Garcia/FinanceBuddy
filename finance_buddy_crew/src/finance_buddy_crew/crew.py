import os
from typing import List, Optional, Dict

from crewai import Agent, Crew, Process, Task, LLM

from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from pydantic import BaseModel, Field, ConfigDict


try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

import logging

logger = logging.getLogger(__name__)

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

from crewai_tools import SerperDevTool
search_tool = SerperDevTool()

from .tools.extraction_tools import (
    get_sec_info,
    get_recent_filings,
    get_company_facts,
    get_news,
)

focus_areas = ["Fundamental Analysis", "Qualitative Analysis", "Quantitative Analysis","Investors",]

# ---- Pydantic base schema to enforce strictness (v2) ----

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


class CalculationDetail(BaseSchema):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    start_price: Optional[float] = None
    end_price: Optional[float] = None
    calculated_return: Optional[float] = None
    data_points_used: Optional[int] = None


class ReturnsSchema(BaseSchema):
    d1m: Optional[float] = Field(None, description="1-month return percentage")
    d3m: Optional[float] = Field(None, description="3-month return percentage")
    d6m: Optional[float] = Field(None, description="6-month return percentage")
    d12m: Optional[float] = Field(None, description="12-month return percentage")
    vol_proxy: Optional[float] = Field(None, description="Annualized volatility")

class FundamentalAnalysisSchema(BaseSchema):
    fundamental_takeaways: List[str] = Field(
        default_factory=list,
        description="A list of bullet points summarizing fundamental analysis."
    )

class QualitativeAnalysisSchema(BaseSchema):
    qualitative_takeaways: List[str] = Field(
        default_factory=list,
        description="A list of bullet points summarizing qualitative analysis."
    )

    # Add validation to catch unrealistic returns
    def model_post_init(self, __context):
        # check each period and add warnings to the BaseSchema.warnings list
        checks = [
            ("d1m", self.d1m),
            ("d3m", self.d3m),
            ("d6m", self.d6m),
            ("d12m", self.d12m),
        ]
        for period, value in checks:
            if value is not None and (value < -95 or value > 500):
                self.warnings.append(f"WARNING: {period} return {value}% seems unrealistic")


class PriceContextSchema(BaseSchema):
    returns: Optional[ReturnsSchema] = None
    calculation_details: Dict[str, CalculationDetail] = Field(default_factory=dict)
    methodology_notes: List[str] = Field(default_factory=list)
    data_quality_score: Optional[float] = Field(None, description="0-1 score for data completeness")
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
    warnings: List[str] = Field(default_factory=list)


def _llm_high() -> LLM:
    return LLM(model="gpt-5-2025-08-07")

def _llm_med() -> LLM:
    return LLM(
        model="gpt-5-mini-2025-08-07",
    )


def _llm_low() -> LLM:
    return LLM(
        model="gpt-5-mini-2025-08-07",
    )

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)


@CrewBase
class FinanceBuddy:

    # Provide defaults so instances are safe to create during tests
    agents: List[BaseAgent] = []
    tasks: List[Task] = []

    @agent
    def intake_coordinator(self) -> Agent:
        return Agent(
            config=self.agents_config["intake_coordinator"],
            verbose=True,
            llm=_llm_high(),
            max_iter=3,
            allow_delegation=True
        )

    @agent
    def edgar_research_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["edgar_research_analyst"],
            verbose=True,
            tools=[get_sec_info, get_recent_filings],
            llm=_llm_med(),
            max_iter=5,
        )

    @agent
    def facts_reporter(self) -> Agent:
        return Agent(
            config=self.agents_config["facts_reporter"],
            verbose=True,
            llm=_llm_low(),
            max_iter=5,
        )

    @agent
    def price_reporter(self) -> Agent:
        return Agent(
            config=self.agents_config["price_reporter"],
            verbose=True,
            llm=_llm_low(),
            max_iter=5,
        )


    @agent
    def news_research_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["news_research_analyst"],
            verbose=True,
            llm=_llm_med(),
            max_iter=5,
            tools=[get_news]
        )

    @agent
    def fundamental_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["fundamental_analyst"],
            verbose=True,
            llm=_llm_med(),
            max_iter=3,
        )

    @agent
    def qualitative_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["qualitative_analyst"],
            verbose=True,
            llm=_llm_med(),
            max_iter=3,
        )

    @agent
    def lead_synthesizer(self) -> Agent:
        return Agent(
            config=self.agents_config["lead_synthesizer"],
            verbose=True,
            llm=_llm_med(),
            max_iter=3,
        )


    # ---- Tasks ----
    @task
    def scope_request_task(self) -> Task:
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
            tools=[get_sec_info, get_recent_filings]
        )

    @task
    def company_facts_snapshot_task(self) -> Task:
        return Task(
            config=self.tasks_config["company_facts_snapshot_task"],
            output_pydantic=CompanyFactsSnapshotSchema,
            output_file=f"{output_dir}/company_facts_snapshot.json",
            tools=[get_company_facts]
        )

    @task
    def news_and_catalysts_task(self) -> Task:
        return Task(
            config=self.tasks_config["news_and_catalysts_task"],
            output_pydantic=NewsAndCatalystsSchema,
            output_file=f"{output_dir}/news_and_catalysts.json",
            tools=[get_news]
        )

    @task
    def price_context_task(self) -> Task:
        return Task(
            config=self.tasks_config["price_context_task"],
            output_pydantic=PriceContextSchema,
            output_file=f"{output_dir}/price_context.json",
        )

    @task
    def fundamental_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["fundamental_analysis_task"],
            output_pydantic=FundamentalAnalysisSchema,
            output_file=f"{output_dir}/fundamental_analysis.json",
        )

    @task
    def qualitative_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["qualitative_analysis_task"],
            output_pydantic=QualitativeAnalysisSchema,
            output_file=f"{output_dir}/qualitative_analysis.json"
        )

    @task
    def synthesis_brief_task(self) -> Task:
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
            memory=True,
        )