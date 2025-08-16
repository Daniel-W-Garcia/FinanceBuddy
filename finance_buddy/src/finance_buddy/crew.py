import os
import sys
from datetime import datetime
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

if not os.environ.get("SEC_USER_AGENT"):
    os.environ["SEC_USER_AGENT"] = "FinanceBuddy/1.0daniel@danielwgarcia.com"


from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from pydantic import BaseModel, Field

from .tools.custom_extraction_tools import get_sec_info, get_recent_filings, get_company_facts, get_news, get_prices

# ---- Minimal Pydantic Schemas for task outputs ----
class ScopeRequestSchema(BaseModel):
    company: Optional[str] = None
    ticker: Optional[str] = None
    exchange: Optional[str] = None
    timeframe: str
    focus_areas: List[str]
    output_format: Optional[str] = None
    constraints: Optional[List[str]] = None
    peers: Optional[List[str]] = None


class FilingSchema(BaseModel):
    form: str
    filing_date: Optional[str] = None
    report_date: Optional[str] = None
    accession: Optional[str] = None
    primary_doc: Optional[str] = None
    items: Optional[str] = None
    doc_url: Optional[str] = None
    index_url: Optional[str] = None

class EdgarBasicsSchema(BaseModel):
    cik10: Optional[str] = None
    company_name: Optional[str] = None
    ceo: Optional[str] = None
    sic: Optional[str] = None
    filings: List[FilingSchema] = Field(default_factory=list)

class MetricPoint(BaseModel):
    period: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None

class CompanyFactsSnapshotSchema(BaseModel):
    currency: Optional[str] = None
    ceo: Optional[str] = None
    revenue: Optional[List[MetricPoint]] = None
    operating_income: Optional[List[MetricPoint]] = None
    net_income: Optional[List[MetricPoint]] = None
    diluted_eps: Optional[List[MetricPoint]] = None
    shares_diluted: Optional[List[MetricPoint]] = None
    cash_or_fcf_proxy: Optional[List[MetricPoint]] = None
    debt: Optional[List[MetricPoint]] = None
    notes: Optional[List[str]] = None
    raw_facts: Optional[dict] = None


class NewsItemSchema(BaseModel):
    date: Optional[str] = None
    source: Optional[str] = None
    title: Optional[str] = None
    link: Optional[str] = None
    relevance: Optional[str] = None

class NewsAndCatalystsSchema(BaseModel):
    items: List[NewsItemSchema] = Field(default_factory=list)
    top_catalysts: Optional[List[str]] = None
    top_risks: Optional[List[str]] = None

class ReturnsSchema(BaseModel):
    d1m: Optional[float] = None
    d3m: Optional[float] = None
    d6m: Optional[float] = None
    d12m: Optional[float] = None
    vol_proxy: Optional[float] = None

class PriceContextSchema(BaseModel):
    returns: Optional[ReturnsSchema] = None
    notes: Optional[List[str]] = None

class SynthesisBriefSchema(BaseModel):
    tldr: List[str]
    positives: Optional[List[str]] = None
    risks: Optional[List[str]] = None
    notable_filings: Optional[List[str]] = None
    fundamentals_takeaways: Optional[List[str]] = None
    recent_catalysts: Optional[List[str]] = None
    next_steps: Optional[List[str]] = None
    disclaimer: Optional[str] = None


# Use appropriate LLM models based on task complexity
llm_high = LLM(model='gpt-5-2025-08-07')
llm_med = LLM(model='gpt-5-mini-2025-08-07')
llm_low = LLM(model='gpt-5-nano-2025-08-07')

output_dir = 'output'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

@CrewBase
class FinanceBuddy():
    """FinanceBuddy crew for automated financial research"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def intake_coordinator(self) -> Agent:
        return Agent(
            config=self.agents_config['intake_coordinator'],
            verbose=True,
            llm=llm_low,  # Simple coordination doesn't need expensive model
            max_iter=3
        )

    @agent
    def research_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['research_analyst'],
            verbose=True,
            tools=[get_sec_info, get_recent_filings, get_news],  # Updated to use simple tools
            llm=llm_med,
            max_iter=5
        )

    @agent
    def financial_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['financial_analyst'],
            verbose=True,
            tools=[get_company_facts, get_prices],  # Updated to use simple tools
            llm=llm_med,
            max_iter=5
        )

    @agent
    def lead_synthesizer(self) -> Agent:
        return Agent(
            config=self.agents_config['lead_synthesizer'],
            verbose=True,
            llm=llm_high,  # Synthesis needs the best model
            max_iter=3
        )

    @task
    def scope_request_task(self) -> Task:
        """Initial task to clarify and scope the research request"""
        return Task(
            config=self.tasks_config['scope_request_task'],
            output_json=ScopeRequestSchema,
            output_file=f'{output_dir}/research_scope.json'
        )

    @task
    def edgar_basics_task(self) -> Task:
        return Task(
            config=self.tasks_config['edgar_basics_task'],
            output_json=EdgarBasicsSchema,
            output_file=f'{output_dir}/edgar_basics.json'
        )

    @task
    def company_facts_snapshot_task(self) -> Task:
        return Task(
            config=self.tasks_config['company_facts_snapshot_task'],
            output_json=CompanyFactsSnapshotSchema,
            output_file=f'{output_dir}/company_facts_snapshot.json'
        )

    @task
    def news_and_catalysts_task(self) -> Task:
        return Task(
            config=self.tasks_config['news_and_catalysts_task'],
            output_json=NewsAndCatalystsSchema,
            output_file=f'{output_dir}/news_and_catalysts.json'
        )

    @task
    def price_context_task(self) -> Task:
        return Task(
            config=self.tasks_config['price_context_task'],
            output_json=PriceContextSchema,
            output_file=f'{output_dir}/price_context.json'
        )

    @task
    def synthesis_brief_task(self) -> Task:
        return Task(
            config=self.tasks_config['synthesis_brief_task'],
            output_json=SynthesisBriefSchema,
            output_file=f'{output_dir}/synthesis_brief.md'
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=False,  # Disabled memory to avoid the API error for now
        )
