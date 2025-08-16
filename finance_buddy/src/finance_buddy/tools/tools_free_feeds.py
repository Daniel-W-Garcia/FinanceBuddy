# CrewAI Tool wrappers around free_feeds.py functions
# Prereqs: free_feeds.py in the same directory

from typing import Type, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from typing import Any
import json
from crewai.tools import BaseTool

from .free_feeds import (
    sec_get_cik_record,
    sec_recent_filings,
    sec_company_facts,
    gdelt_news,
    stooq_prices,
)

__all__ = [
    "SecGetCikTool",
    "SecRecentFilingsTool",
    "SecCompanyFactsTool",
    "GdeltNewsTool",
    "StooqPricesTool",
]


# ---- Input validation helper ----
def safe_extract_value(input_data: Any, key: str, fallback: Any = None) -> Any:
    """
    Safely extract a value from various input formats that CrewAI might pass
    """
    try:
        # If it's already the right type, return it
        if isinstance(input_data, str) and key == 'ticker':
            return input_data.strip().upper()

        # Try to parse as JSON if it's a string
        if isinstance(input_data, str):
            try:
                parsed = json.loads(input_data)
                if isinstance(parsed, dict) and key in parsed:
                    return parsed[key]
                elif isinstance(parsed, list) and len(parsed) > 0:
                    # Handle arrays - look for the key in first dict
                    for item in parsed:
                        if isinstance(item, dict) and key in item:
                            return item[key]
            except (json.JSONDecodeError, KeyError):
                pass

        # If it's a dict, extract the key
        if isinstance(input_data, dict) and key in input_data:
            return input_data[key]

        # If it's a list, look through items
        if isinstance(input_data, list):
            for item in input_data:
                if isinstance(item, dict) and key in item:
                    return item[key]

        return fallback
    except Exception:
        return fallback


# ---- SEC: get CIK ----
class SecGetCikArgs(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'GME')")


class SecGetCikTool(BaseTool):
    name: str = "sec_get_cik"
    description: str = (
        "Get SEC company information by ticker symbol. "
        "IMPORTANT: Only pass the ticker symbol as a string. "
        "Example: Use 'GME' not {'ticker': 'GME'} or complex objects. "
        "Input parameter: ticker (string) - just the ticker symbol like 'AAPL' or 'GME'"
    )
    args_schema: Type[BaseModel] = SecGetCikArgs

    def _run(self, ticker: str) -> Dict[str, Any]:
        # Try to extract ticker from whatever format was passed
        extracted_ticker = safe_extract_value(ticker, 'ticker', ticker)

        if not isinstance(extracted_ticker, str) or not extracted_ticker.strip():
            raise ValueError(f"Could not extract valid ticker from input: {ticker}")

        clean_ticker = extracted_ticker.strip().upper()
        return sec_get_cik_record(clean_ticker)


# ---- SEC: recent filings ----
class SecRecentFilingsArgs(BaseModel):
    cik10: str = Field(..., description="10-digit zero-padded CIK (e.g., '0000320193')")
    forms: Optional[List[str]] = Field(
        default=["10-K", "10-Q", "8-K"],
        description="Form types to include (e.g., 10-K, 10-Q, 8-K)",
    )
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of filings")


class SecRecentFilingsTool(BaseTool):
    name: str = "sec_recent_filings"
    description: str = (
        "List recent SEC filings for a company by CIK with direct document and index links. "
        "IMPORTANT: Only pass the 10-digit CIK as a string. "
        "Example: Use '0001326380' not {'cik10': '0001326380'} or complex objects. "
        "Defaults to forms: 10-K, 10-Q, 8-K."
    )
    args_schema: Type[BaseModel] = SecRecentFilingsArgs

    def _run(self, cik10: str, forms: Optional[List[str]] = None, limit: int = 10) -> List[Dict[str, Any]]:
        # Extract and validate CIK
        extracted_cik = safe_extract_value(cik10, 'cik10', cik10)

        if not isinstance(extracted_cik, str):
            raise ValueError(f"Expected string CIK, got {type(extracted_cik)}: {extracted_cik}")

        clean_cik = str(extracted_cik).strip()
        if len(clean_cik) != 10 or not clean_cik.isdigit():
            raise ValueError(f"CIK must be 10-digit string, got: {clean_cik}")

        return sec_recent_filings(clean_cik, forms=forms, limit=limit)


# ---- SEC: company facts (XBRL JSON) ----
class SecCompanyFactsArgs(BaseModel):
    cik10: str = Field(..., description="10-digit zero-padded CIK (e.g., '0000320193')")


class SecCompanyFactsTool(BaseTool):
    name: str = "sec_company_facts"
    description: str = (
        "Get SEC XBRL company facts by CIK. "
        "IMPORTANT: Only pass the 10-digit CIK as a string. "
        "Example: Use '0001326380' not {'cik10': '0001326380'} or complex objects. "
        "Input parameter: cik10 (string) - the 10-digit zero-padded CIK"
    )
    args_schema: Type[BaseModel] = SecCompanyFactsArgs

    def _run(self, cik10: str) -> Dict[str, Any]:
        # Extract and validate CIK
        extracted_cik = safe_extract_value(cik10, 'cik10', cik10)

        if not isinstance(extracted_cik, str):
            raise ValueError(f"Expected string CIK, got {type(extracted_cik)}: {extracted_cik}")

        clean_cik = str(extracted_cik).strip()
        if len(clean_cik) != 10 or not clean_cik.isdigit():
            raise ValueError(f"CIK must be 10-digit string, got: {clean_cik}")

        return sec_company_facts(clean_cik)


# ---- GDELT: recent news ----
class GdeltNewsArgs(BaseModel):
    query: str = Field(..., description='Search string, e.g., "Apple Inc" OR AAPL')
    days: int = Field(default=14, ge=1, le=60, description="Timespan in days")
    limit: int = Field(default=20, ge=1, le=50, description="Max records")


class GdeltNewsTool(BaseTool):
    name: str = "gdelt_news"
    description: str = (
        "Fetch recent news mentions via the GDELT Doc API. "
        "IMPORTANT: Only pass the search query as a string. "
        "Example: Use 'GameStop OR GME' not {'query': 'GameStop OR GME'}"
    )
    args_schema: Type[BaseModel] = GdeltNewsArgs

    def _run(self, query: str, days: int = 14, limit: int = 20) -> List[Dict[str, Any]]:
        # Extract query from whatever format was passed
        extracted_query = safe_extract_value(query, 'query', query)

        if not isinstance(extracted_query, str) or not extracted_query.strip():
            raise ValueError(f"Could not extract valid query from input: {query}")

        clean_query = extracted_query.strip()
        return gdelt_news(clean_query, days=days, limit=limit)


# ---- Stooq: daily OHLC prices ----
class StooqPricesArgs(BaseModel):
    symbol: str = Field(..., description="Ticker symbol (e.g., 'AAPL')")


class StooqPricesTool(BaseTool):
    name: str = "stooq_prices"
    description: str = (
        "Retrieve daily OHLC price history from Stooq for a given symbol. "
        "IMPORTANT: Only pass the ticker symbol as a string. "
        "Example: Use 'AAPL' not {'symbol': 'AAPL'}"
    )
    args_schema: Type[BaseModel] = StooqPricesArgs

    def _run(self, symbol: str) -> List[Dict[str, Any]]:
        # Extract symbol from whatever format was passed
        extracted_symbol = safe_extract_value(symbol, 'symbol', symbol)

        if not isinstance(extracted_symbol, str) or not extracted_symbol.strip():
            raise ValueError(f"Could not extract valid symbol from input: {symbol}")

        clean_symbol = extracted_symbol.strip()
        return stooq_prices(clean_symbol)
