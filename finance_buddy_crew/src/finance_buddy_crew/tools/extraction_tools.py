# custom_extraction_tools.py
from typing import Any, Callable, Dict, List, Optional, Type
import functools
import logging
import importlib
import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np
from datetime import datetime
import math


from crewai.tools import tool
from pydantic import ValidationError, BaseModel, create_model, Field

from .feeds import (
    sec_get_cik_record,
    sec_recent_filings,
    sec_company_facts,
    gdelt_news,
    yahoo_prices,
)

logger = logging.getLogger(__name__)


# extraction_tools.py (Replace the whole function with this)

def _process_sec_facts(raw_facts: dict, timeframe: str = "LTM") -> dict:
    """
    Parses the raw, complex JSON from the SEC companyfacts API into a clean,
    structured dictionary that matches our Pydantic schema, and derives key metrics.
    """
    processed_data = {
        "currency": "USD",
        "notes": [],
        "revenue": [],
        "operating_income": [],
        "net_income": [],
        "diluted_eps": [],
    }

    if not raw_facts.get("facts"):
        processed_data["notes"].append("No 'facts' found in raw SEC data.")
        return processed_data

    facts = raw_facts["facts"].get("us-gaap", {})

    # Helper to pull the raw data points for a given SEC tag
    def _get_fact_values(tag: str) -> dict:
        try:
            points = facts[tag]["units"]["USD"]
            # We only care about quarterly (10-Q) and annual (10-K) filings
            relevant_points = [p for p in points if p.get("form") in ["10-Q", "10-K"]]
            # Return a dictionary mapping the period end-date to the value
            return {p['end']: p['val'] for p in relevant_points}
        except KeyError:
            processed_data["notes"].append(f"Metric tag not found or has no USD data: {tag}")
            return {}

    # --- 1. Fetch the fundamental building blocks ---
    revenues = _get_fact_values("Revenues")
    costs_of_revenue = _get_fact_values("CostOfGoodsAndServicesSold")  # GME is a retailer, so this tag is key
    net_income = _get_fact_values("NetIncomeLoss")
    eps_diluted = _get_fact_values("EarningsPerShareDiluted")

    if not revenues:
        processed_data["notes"].append("Could not process facts without Revenue data.")
        return processed_data

    # --- 2. Iterate through periods and build the snapshot ---
    # Get all unique reporting dates from the revenue data, and sort them
    all_periods = sorted(revenues.keys(), reverse=True)

    for period in all_periods[:8]:  # Look at the last 8 periods (2 years)
        rev = revenues.get(period)
        cost = costs_of_revenue.get(period)

        # We can only calculate profit if both revenue and cost are present for a period
        op_income = None
        if rev is not None and cost is not None:
            # This is a simplification. Real operating income also subtracts SG&A.
            # But for this purpose, Gross Profit is a much better proxy than what we had.
            op_income = rev - cost

        processed_data["revenue"].append({"period": period, "value": rev, "unit": "USD"})
        processed_data["operating_income"].append({"period": period, "value": op_income, "unit": "USD"})
        processed_data["net_income"].append({"period": period, "value": net_income.get(period), "unit": "USD"})
        processed_data["diluted_eps"].append({"period": period, "value": eps_diluted.get(period), "unit": "USD/Share"})

    processed_data['raw_facts'] = raw_facts
    return processed_data


# extraction_tools.py (ADD THIS NEW FUNCTION)

def get_financial_statements(ticker: str) -> dict:
    """
    Gets key financial statement data from yfinance for the last 4 years
    and returns it in a clean, structured format for reporting.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is not installed. Please install it with 'pip install yfinance'")

    print(f"Fetching financial statements for {ticker} from yfinance...")
    stock = yf.Ticker(ticker)

    # Fetch the income statement, balance sheet, and stock info
    income_stmt = stock.income_stmt
    balance_sheet = stock.balance_sheet
    stock_info = stock.info

    # Prepare our clean output dictionary
    processed_data = {
        "notes": ["Data sourced from yfinance."],
        "revenue": [], "operating_income": [], "net_income": [],
        "debt": [], "cash": [], "shares_diluted": []
    }

    # Process Income Statement (last 4 periods)
    income_stmt_t = income_stmt.transpose().head(4)
    for period, data in income_stmt_t.iterrows():
        period_str = period.strftime('%Y-%m-%d')
        processed_data["revenue"].append({"period": period_str, "value": data.get("Total Revenue")})
        processed_data["operating_income"].append({"period": period_str, "value": data.get("Operating Income")})
        processed_data["net_income"].append({"period": period_str, "value": data.get("Net Income")})

    # Process Balance Sheet (last 4 periods)
    balance_sheet_t = balance_sheet.transpose().head(4)
    for period, data in balance_sheet_t.iterrows():
        period_str = period.strftime('%Y-%m-%d')
        processed_data["debt"].append({"period": period_str, "value": data.get("Total Debt")})
        processed_data["cash"].append({"period": period_str, "value": data.get("Cash And Cash Equivalents")})

    # Add shares outstanding as a single, current value
    if stock_info.get("sharesOutstanding"):
        latest_period = income_stmt_t.index[0].strftime('%Y-%m-%d')
        processed_data["shares_diluted"].append({
            "period": latest_period,
            "value": stock_info.get("sharesOutstanding")
        })

    print("✅ Financial statements fetched and processed successfully.")
    return processed_data

def calculate_stock_returns(ticker: str, as_of_date: str) -> dict:
    """
    Calculate precise stock returns with full validation and methodology.
    Args:
        ticker: Stock ticker symbol
        as_of_date: End date for calculations (YYYY-MM-DD format)
    """


    try:
        # Get price data
        price_data = get_prices(ticker)
        if not price_data.get("items"):
            return {"error": "No price data available", "warnings": ["No price data found"]}

        # Convert to DataFrame for easier calculation
        df = pd.DataFrame(price_data["items"])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Calculate end date
        end_date = datetime.strptime(as_of_date, "%Y-%m-%d")

        # Find closest trading day to end_date
        end_row = df[df['date'] <= end_date].tail(1)
        if end_row.empty:
            return {"error": "No price data for end date", "warnings": ["No data for specified end date"]}

        actual_end_date = end_row.iloc[0]['date']
        end_price = float(end_row.iloc[0]['close'])

        results = {}
        calculation_details = {}
        warnings = []

        periods = {
            "d1m": DateOffset(months=1),
            "d3m": DateOffset(months=3),
            "d6m": DateOffset(months=6),
            "d12m": DateOffset(months=12)
        }

        for period_name, offset in periods.items():
            # Calculate start date using the calendar offset
            start_date = actual_end_date - offset

            # Find closest trading day to start date
            start_candidates = df[df['date'] >= start_date]
            if start_candidates.empty:
                results[period_name] = None
                warnings.append(f"Insufficient data for {period_name} calculation")
                continue

            start_row = start_candidates.head(1)
            actual_start_date = start_row.iloc[0]['date']
            start_price = float(start_row.iloc[0]['close'])

            # Calculate return: (end_price / start_price - 1) * 100
            if start_price > 0:
                calculated_return = ((end_price / start_price) - 1) * 100
                results[period_name] = round(calculated_return, 1)

                # Store calculation details
                calculation_details[period_name] = {
                    "start_date": actual_start_date.strftime("%Y-%m-%d"),
                    "end_date": actual_end_date.strftime("%Y-%m-%d"),
                    "start_price": round(start_price, 2),
                    "end_price": round(end_price, 2),
                    "calculated_return": round(calculated_return, 1),
                    "data_points_used": len(df[(df['date'] >= actual_start_date) & (df['date'] <= actual_end_date)])
                }

                # Validation check
                if abs(calculated_return) > 200:  # > 200% move
                    warnings.append(f"{period_name}: {calculated_return:.1f}% return seems high - please verify")

            else:
                results[period_name] = None
                warnings.append(f"Invalid start price for {period_name}")

        # Calculate volatility
        if len(df) > 30:  # Need sufficient data
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            # Annualized volatility = std(daily log returns) * sqrt(252)
            daily_vol = df['log_returns'].std()
            if not math.isnan(daily_vol):
                annualized_vol = daily_vol * math.sqrt(252)
                results["vol_proxy"] = round(annualized_vol, 3)
            else:
                results["vol_proxy"] = None
                warnings.append("Could not calculate volatility")
        else:
            results["vol_proxy"] = None
            warnings.append("Insufficient data for volatility calculation")

        return {
            "returns": results,
            "calculation_details": calculation_details,
            "methodology_notes": [
                "Returns calculated as (end_price / start_price - 1) * 100",
                "Dates anchored to nearest available trading day",
                "Volatility = std(daily log returns) * sqrt(252)",
                f"Calculations as of {actual_end_date.strftime('%Y-%m-%d')}"
            ],
            "data_quality_score": min(1.0, len([v for v in results.values() if v is not None]) / 4),
            "warnings": warnings,
            "raw_input": {"ticker": ticker, "as_of_date": as_of_date, "data_rows": len(df)}
        }

    except Exception as e:
        logger.exception("Error in calculate_stock_returns for %s", ticker)
        return {
            "error": str(e),
            "warnings": [f"Calculation failed: {str(e)}"],
            "raw_input": {"ticker": ticker, "as_of_date": as_of_date}
        }


def _friendly_from_error(err: Dict[str, Any]) -> str:
    """Return a compact human-friendly string for a single pydantic error entry."""
    loc = ".".join(str(x) for x in err.get("loc", []))
    msg = err.get("msg", "")
    return f"{loc}: {msg}" if loc else msg


def validate_output(model_resolver: Callable[[], Type[BaseModel]]):
    """
    Decorator that resolves a Pydantic model at tool-run time and validates
    the raw tool output against it.

    On success: returns model.model_dump() (a dict).
    On failure: returns a best-effort dict that:
      - includes validated sub-items where possible,
      - always includes 'warnings' (List[str]) and 'raw_input' (original payload).
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Dict[str, Any]:
            raw = None
            try:
                raw = fn(*args, **kwargs)
            except Exception as e:
                logger.exception("Tool %s raised during execution", fn.__name__)
                return {"warnings": [f"tool_error: {e}"], "raw_input": None}

            # Resolve the target Pydantic model at runtime to avoid circular imports
            try:
                model_cls = model_resolver()
            except Exception as e:
                logger.exception("Failed to resolve model for %s: %s", fn.__name__, e)
                return {"warnings": [f"model_resolve_error: {e}"], "raw_input": raw}

            # Validate full payload
            try:
                validated = model_cls.model_validate(raw)
                out = validated.model_dump()
                # Ensure diagnostics keys exist in the returned dict (BaseSchema should provide them)
                if isinstance(out, dict):
                    out.setdefault("warnings", [])
                    out.setdefault("raw_input", None)
                return out
            except ValidationError as exc:
                errs = exc.errors()
                warnings: List[str] = [_friendly_from_error(e) for e in errs]
                logger.warning("Validation failed for %s: %s", fn.__name__, warnings)

                # Build a best-effort partial output matching model_cls field names.
                partial: Dict[str, Any] = {}
                raw_dict = raw if isinstance(raw, dict) else {"value": raw}

                # If model has an 'items' field (we create list-wrapper models that use 'items'),
                # attempt to validate individual list elements and include valid ones.
                if "items" in model_cls.model_fields:
                    items_valid: List[Any] = []
                    if isinstance(raw, list):
                        for item in raw:
                            try:
                                # Create a small instance of the element schema to validate
                                # We attempt per-element validation by wrapping into the items type via model
                                elem_model = create_model(
                                    "ElemWrapper",
                                    __base__=model_cls.__class__ if False else BaseModel,  # placeholder
                                )
                                # Simpler approach: try to validate element via model_cls by calling model_validate on {"items":[item]}
                                tmp = model_cls.model_validate({"items": [item]})
                                dumped = tmp.model_dump()
                                items = dumped.get("items", [])
                                if items:
                                    # append validated items
                                    items_valid.extend(items)
                                else:
                                    # fallback to raw item
                                    items_valid.append(item)
                            except Exception:
                                # couldn't validate this item; skip it
                                continue
                    partial["items"] = items_valid
                else:
                    # Try per-field salvage for top-level fields
                    for fld in model_cls.model_fields.keys():
                        if fld in raw_dict:
                            try:
                                tmp = model_cls.model_validate({fld: raw_dict[fld]})
                                dumped = tmp.model_dump()
                                if fld in dumped:
                                    partial[fld] = dumped[fld]
                            except Exception:
                                continue

                partial["raw_input"] = raw_dict
                partial["warnings"] = warnings
                return partial

        return wrapper
    return decorator


# Deferred model resolvers that import crew at runtime and return models shaped
# to include 'items' for list outputs so returned dicts are consistent.
def _sec_info_model() -> Type[BaseModel]:
    crew = importlib.import_module("crew")
    # EdgarBasicsSchema inherits BaseSchema and thus has warnings/raw_input fields
    return crew.EdgarBasicsSchema


def _recent_filings_model() -> Type[BaseModel]:
    crew = importlib.import_module("crew")
    # Create a wrapper model with items: List[FilingSchema]; inherits crew.BaseSchema so warnings/raw_input exist
    cls = create_model(
        "FilingsWrapper",
        __base__=crew.BaseSchema,
        items=(List[crew.FilingSchema], Field(default_factory=list)),
    )
    return cls


def _company_facts_model() -> Type[BaseModel]:
    crew = importlib.import_module("crew")
    return crew.CompanyFactsSnapshotSchema


def _news_list_model() -> Type[BaseModel]:
    crew = importlib.import_module("crew")
    cls = create_model(
        "NewsWrapper",
        __base__=crew.BaseSchema,
        items=(List[crew.NewsItemSchema], Field(default_factory=list)),
    )
    return cls


def _prices_list_model() -> Type[BaseModel]:
    import sys
    import os

    # Add the src directory to Python path if not already there
    src_path = os.path.join(os.path.dirname(__file__), '..', '..')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    try:
        from finance_buddy_crew import crew
    except ImportError:
        crew = importlib.import_module("crew")

    cls = create_model(
        "PricesWrapper",
        __base__=crew.BaseSchema,
        items=(List[Dict[str, Any]], Field(default_factory=list)),
    )
    return cls


# ---- Tool functions wrapped with validation using the real crew models ----

@tool("Get company SEC info by ticker")
@validate_output(_sec_info_model)
def get_sec_info(ticker: str) -> dict:
    """Get SEC company information. Pass only the ticker symbol as string."""
    return sec_get_cik_record(ticker.strip().upper())


@tool("Get recent SEC filings by CIK")
@validate_output(_recent_filings_model)
def get_recent_filings(cik: str) -> list:
    """Get recent SEC filings. Pass only the 10-digit CIK as string."""
    return sec_recent_filings(cik.strip())


@tool("Get processed and summarized company financial facts by CIK")
# The tool now takes a timeframe argument to pass to the processor
def get_company_facts(cik: str, timeframe: str = "LTM") -> dict:
    """
    Get SEC XBRL facts, processed into a clean summary.
    Pass the 10-digit CIK as a string and an optional timeframe.
    """
    raw_data = sec_company_facts(cik.strip())
    # Process the raw data before returning it
    processed_summary = _process_sec_facts(raw_data, timeframe)
    return processed_summary


@tool("Get recent news by ticker")
@validate_output(_news_list_model)
def get_news(ticker: str) -> list:
    """Get recent news. Pass only the ticker symbol."""
    return gdelt_news(f"{ticker} OR {ticker.strip().upper()}", days=30, limit=20)


# changed this to be an internal 'helper' function for main tool agents will now us. this was too much context for the agents. and was causing errors
def get_prices(ticker: str) -> dict:
    """Get stock prices. Pass only the ticker symbol. Returns wrapper with items and metadata."""
    sym = ticker.strip().upper()
    try:
        rows = yahoo_prices(sym)
    except Exception as e:
        logger.exception("Failed to fetch prices for %s: %s", sym, e)
        # Let decorator handle packaging the error into warnings/raw_input if needed
        raise

    meta = {
        "price_source": "yahoo",
        "ticker": sym,
        "as_of": datetime.utcnow().strftime("%Y-%m-%d"),
    }

    # Return wrapper matching the validation model: contains 'items' (list) and diagnostic raw_input
    return {"items": rows, "raw_input": meta}