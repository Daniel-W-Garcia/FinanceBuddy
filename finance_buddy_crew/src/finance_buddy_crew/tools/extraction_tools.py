# custom_extraction_tools.py
from typing import Any, Callable, Dict, List, Optional, Type
import functools
import logging
import importlib

from crewai.tools import tool
from pydantic import ValidationError, BaseModel, create_model, Field

from .feeds import (
    sec_get_cik_record,
    sec_recent_filings,
    sec_company_facts,
    gdelt_news,
    stooq_prices,
)

logger = logging.getLogger(__name__)


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
    crew = importlib.import_module("crew")
    # Use generic dict items for prices, but still wrap in 'items' for consistency
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


@tool("Get company financial facts by CIK")
@validate_output(_company_facts_model)
def get_company_facts(cik: str) -> dict:
    """Get SEC XBRL facts. Pass only the 10-digit CIK as string."""
    return sec_company_facts(cik.strip())


@tool("Get recent news by ticker")
@validate_output(_news_list_model)
def get_news(ticker: str) -> list:
    """Get recent news. Pass only the ticker symbol."""
    return gdelt_news(f"{ticker} OR {ticker.strip().upper()}", days=30, limit=20)


@tool("Get stock prices by ticker")
@validate_output(_prices_list_model)
def get_prices(ticker: str) -> list:
    """Get stock prices. Pass only the ticker symbol."""
    return stooq_prices(ticker.strip().upper())
