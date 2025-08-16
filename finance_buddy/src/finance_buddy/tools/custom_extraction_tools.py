from crewai.tools import tool
from .free_feeds import sec_get_cik_record, sec_recent_filings, sec_company_facts, gdelt_news, stooq_prices

@tool("Get company SEC info by ticker")
def get_sec_info(ticker: str) -> dict:
    """Get SEC company information. Pass only the ticker symbol as string."""
    return sec_get_cik_record(ticker.strip().upper())

@tool("Get recent SEC filings by CIK")
def get_recent_filings(cik: str) -> list:
    """Get recent SEC filings. Pass only the 10-digit CIK as string."""
    return sec_recent_filings(cik.strip())

@tool("Get company financial facts by CIK")
def get_company_facts(cik: str) -> dict:
    """Get SEC XBRL facts. Pass only the 10-digit CIK as string."""
    return sec_company_facts(cik.strip())

@tool("Get recent news by ticker")
def get_news(ticker: str) -> list:
    """Get recent news. Pass only the ticker symbol."""
    return gdelt_news(f"{ticker} OR {ticker.strip().upper()}", days=30, limit=20)

@tool("Get stock prices by ticker")
def get_prices(ticker: str) -> list:
    """Get stock prices. Pass only the ticker symbol."""
    return stooq_prices(ticker.strip().upper())

