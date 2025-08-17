# tools_free_feeds.py
# CrewAI Tool wrappers around free_feeds.py functions (refactored)
# Prereqs: free_feeds.py (refactored) in the same package
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field, ConfigDict, ValidationError

from crewai.tools import BaseTool

from feeds import (
    sec_get_cik_record,
    sec_recent_filings,
    sec_company_facts,
    gdelt_news,
    stooq_prices,
    RequestsClient,
    FeedError,
    NotFoundError,
    HTTPFeedError,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SecGetCikTool",
    "SecRecentFilingsTool",
    "SecCompanyFactsTool",
    "GdeltNewsTool",
    "StooqPricesTool",
]

# ---------------------
# Input schemas (pydantic)
# ---------------------
class StrictBaseModel(BaseModel):
    """
    BaseModel configured to forbid extra fields so that unexpected inputs are rejected early.
    Use model_validate(...) to parse incoming data.
    """
    model_config = ConfigDict(extra="forbid")


class SecGetCikArgs(StrictBaseModel):
    ticker: str = Field(..., min_length=1, description="Stock ticker symbol (e.g., 'AAPL')")


class SecRecentFilingsArgs(StrictBaseModel):
    cik10: str = Field(..., min_length=10, max_length=10, description="10-digit zero-padded CIK")
    forms: Optional[List[str]] = Field(default=None, description="List of form types to include (e.g., 10-K)")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of filings to return")


class SecCompanyFactsArgs(StrictBaseModel):
    cik10: str = Field(..., min_length=10, max_length=10, description="10-digit zero-padded CIK")


class GdeltNewsArgs(StrictBaseModel):
    query: str = Field(..., min_length=1, description='Search string, e.g., "Apple Inc" OR AAPL')
    days: int = Field(default=14, ge=1, le=60)
    limit: int = Field(default=20, ge=1, le=50)


class StooqPricesArgs(StrictBaseModel):
    symbol: str = Field(..., min_length=1, description="Ticker symbol (e.g., 'AAPL')")
    start_date: Optional[str] = Field(default=None, description="YYYY-MM-DD inclusive")
    end_date: Optional[str] = Field(default=None, description="YYYY-MM-DD inclusive")


# ---------------------
# Helpers
# ---------------------
def _parse_possible_input(payload: Any, key: str) -> Any:
    """
    Safely attempt to extract a key or value from various CrewAI input shapes:
      - plain string (return as-is if looking for the main scalar)
      - JSON string containing dict or list
      - dict with the key
      - list of dicts (find first dict with key)
    Returns the extracted value or raises ValueError if not found.
    """
    # If caller passed a raw string and key is the main scalar name, return it
    if isinstance(payload, str):
        # attempt to parse JSON string first (common in LLM tool calls)
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            # Not JSON: treat as plain scalar (common for 'ticker' or 'symbol')
            return payload
        else:
            payload = parsed

    # If it's a dict, try direct extraction
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        # allow direct scalar passed as {"value": "AAPL"} or {"ticker":"AAPL"}? Not assumed; require explicit key
        raise ValueError(f"Expected key '{key}' in dict payload: keys={list(payload.keys())}")

    # If list, find first dict that contains key
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and key in item:
                return item[key]
        raise ValueError(f"Could not find key '{key}' in list payload")

    # If it's already the scalar expected
    return payload


def _ensure_serializable(o: Any) -> Any:
    """
    Convert objects to JSON-serializable primitives where needed.
    We primarily handle pydantic models (convert to dict), but fallback gracefully.
    """
    # If pydantic BaseModel
    try:
        if isinstance(o, BaseModel):
            return o.model_dump()
    except Exception:
        pass
    return o


# ---------------------
# Tool wrappers
# ---------------------
class SecGetCikTool(BaseTool):
    name: str = "sec_get_cik"
    description: str = (
        "Get SEC company information by ticker symbol. "
        "Input: ticker string (e.g., 'AAPL')."
    )
    args_schema: Type[BaseModel] = SecGetCikArgs

    def __init__(self, client: Optional[RequestsClient] = None):
        super().__init__()
        self.client = client or free_feeds._default_client

    def _run(self, ticker: Any) -> Dict[str, Any]:
        try:
            # Accept various CrewAI payload shapes
            val = _parse_possible_input(ticker, "ticker")
            args = SecGetCikArgs.model_validate({"ticker": val})
        except (ValidationError, ValueError) as exc:
            raise ValueError(f"Invalid arguments for sec_get_cik: {exc}") from exc

        try:
            out = sec_get_cik_record(args.ticker, client=self.client)
            return _ensure_serializable(out)
        except NotFoundError as exc:
            raise ValueError(str(exc)) from exc
        except FeedError as exc:
            logger.exception("Error fetching CIK for %s", args.ticker)
            raise RuntimeError(f"SEC feed error: {exc}") from exc


class SecRecentFilingsTool(BaseTool):
    name: str = "sec_recent_filings"
    description: str = (
        "List recent SEC filings for a company by CIK. Input: {'cik10': '0000320193', 'forms': [...], 'limit': 10}"
    )
    args_schema: Type[BaseModel] = SecRecentFilingsArgs

    def __init__(self, client: Optional[RequestsClient] = None):
        super().__init__()
        self.client = client or free_feeds._default_client

    def _run(self, cik10: Any, forms: Any = None, limit: Any = 10) -> List[Dict[str, Any]]:
        try:
            payload = {"cik10": cik10, "forms": forms, "limit": limit}
            args = SecRecentFilingsArgs.model_validate(payload)
        except (ValidationError, ValueError) as exc:
            raise ValueError(f"Invalid arguments for sec_recent_filings: {exc}") from exc

        try:
            out = sec_recent_filings(args.cik10, forms=args.forms, limit=args.limit, client=self.client)
            return _ensure_serializable(out)
        except FeedError as exc:
            logger.exception("Error fetching recent filings for %s", args.cik10)
            raise RuntimeError(f"SEC feed error: {exc}") from exc


class SecCompanyFactsTool(BaseTool):
    name: str = "sec_company_facts"
    description: str = "Fetch SEC XBRL company facts JSON by 10-digit CIK."
    args_schema: Type[BaseModel] = SecCompanyFactsArgs

    def __init__(self, client: Optional[RequestsClient] = None):
        super().__init__()
        self.client = client or free_feeds._default_client

    def _run(self, cik10: Any) -> Dict[str, Any]:
        try:
            args = SecCompanyFactsArgs.model_validate({"cik10": cik10})
        except (ValidationError, ValueError) as exc:
            raise ValueError(f"Invalid arguments for sec_company_facts: {exc}") from exc

        try:
            out = sec_company_facts(args.cik10, client=self.client)
            return _ensure_serializable(out)
        except FeedError as exc:
            logger.exception("Error fetching company facts for %s", args.cik10)
            raise RuntimeError(f"SEC feed error: {exc}") from exc


class GdeltNewsTool(BaseTool):
    name: str = "gdelt_news"
    description: str = "Fetch recent news via GDELT Doc API. Input: query string."
    args_schema: Type[BaseModel] = GdeltNewsArgs

    def __init__(self, client: Optional[RequestsClient] = None):
        super().__init__()
        self.client = client or free_feeds._default_client

    def _run(self, query: Any, days: Any = 14, limit: Any = 20) -> List[Dict[str, Any]]:
        try:
            payload = {"query": query, "days": days, "limit": limit}
            args = GdeltNewsArgs.model_validate(payload)
        except (ValidationError, ValueError) as exc:
            raise ValueError(f"Invalid arguments for gdelt_news: {exc}") from exc

        try:
            out = gdelt_news(args.query, days=args.days, limit=args.limit, client=self.client)
            return _ensure_serializable(out)
        except FeedError as exc:
            logger.exception("GDELT error for query=%s", args.query)
            raise RuntimeError(f"GDELT feed error: {exc}") from exc


class StooqPricesTool(BaseTool):
    name: str = "stooq_prices"
    description: str = "Retrieve daily OHLC price history from Stooq for a given symbol."
    args_schema: Type[BaseModel] = StooqPricesArgs

    def __init__(self, client: Optional[RequestsClient] = None):
        super().__init__()
        self.client = client or free_feeds._default_client

    def _run(self, symbol: Any, start_date: Any = None, end_date: Any = None) -> List[Dict[str, Any]]:
        try:
            payload = {"symbol": symbol, "start_date": start_date, "end_date": end_date}
            args = StooqPricesArgs.model_validate(payload)
        except (ValidationError, ValueError) as exc:
            raise ValueError(f"Invalid arguments for stooq_prices: {exc}") from exc

        try:
            out = stooq_prices(args.symbol, start_date=args.start_date, end_date=args.end_date, client=self.client)
            return _ensure_serializable(out)
        except NotFoundError as exc:
            raise ValueError(str(exc)) from exc
        except FeedError as exc:
            logger.exception("Stooq error for symbol=%s", args.symbol)
            raise RuntimeError(f"Stooq feed error: {exc}") from exc
