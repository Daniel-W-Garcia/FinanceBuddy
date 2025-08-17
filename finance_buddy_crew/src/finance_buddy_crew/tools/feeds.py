# free_feeds.py -- refactored
# Requirements: requests, urllib3 (requests already depends on urllib3)
# Env: export SEC_USER_AGENT="FinanceBuddy/1.0 you@domain.com"
from __future__ import annotations
import os
import csv
import io
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, date
from functools import lru_cache

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# logging setup - library code should not configure root logger
logger = logging.getLogger(__name__)

SEC_BASE = "https://data.sec.gov"
SEC_HEADERS = {
    "User-Agent": os.getenv("SEC_USER_AGENT", "FinanceBuddy/0.1 you@example.com")
}
APP_USER_AGENT = os.getenv(
    "FINBUDDY_USER_AGENT",
    os.getenv("SEC_USER_AGENT", "FinanceBuddy/1.0 (contact@example.com)")
)

# default timeouts (connect, read)
DEFAULT_TIMEOUT: Tuple[float, float] = (3.05, 30.0)


class FeedError(Exception):
    """Base class for feed errors."""


class NotFoundError(FeedError):
    """Raised when an entity (ticker / price) is not found."""


class HTTPFeedError(FeedError):
    """Raised when upstream returns an error status."""


class RequestsClient:
    """
    Thin wrapper around a requests.Session configured with urllib3 Retry + HTTPAdapter.
    Use dependency injection for easier unit testing / mocking.
    """

    def __init__(
        self,
        user_agent: Optional[str] = None,
        retries: int = 3,
        backoff_factor: float = 0.5,
        status_forcelist: Optional[List[int]] = None,
        timeout: Tuple[float, float] = DEFAULT_TIMEOUT,
        trust_env: bool = False,
    ):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = trust_env  # avoid reading ~/.netrc / env proxies by default
        self.session.headers.update(
            {"User-Agent": user_agent or APP_USER_AGENT}
        )
        status_forcelist = status_forcelist or [429, 502, 503, 504]
        # Configure urllib3 Retry
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=frozenset(["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"]),
            raise_on_status=False,  # we'll handle raise_for_status ourselves
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        # Mount for both http and https
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def close(self):
        try:
            self.session.close()
        except Exception:
            pass

    def get(self, url: str, params: Optional[dict] = None, headers: Optional[dict] = None, timeout: Optional[Tuple[float, float]] = None) -> requests.Response:
        t = timeout or self.timeout
        hdrs = {}
        if headers:
            hdrs.update(headers)
        resp = self.session.get(url, params=params, headers=hdrs, timeout=t)
        # raise helpful exception with context
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            logger.debug("HTTP error for %s (status %s): %s", url, resp.status_code, resp.text)
            raise HTTPFeedError(f"HTTP {resp.status_code} for {url}") from exc
        return resp

    def get_json(self, url: str, params: Optional[dict] = None, headers: Optional[dict] = None, timeout: Optional[Tuple[float, float]] = None) -> Any:
        r = self.get(url, params=params, headers=headers, timeout=timeout)
        try:
            return r.json()
        except ValueError as exc:
            logger.debug("JSON decode error for %s: %s", url, r.text[:500])
            raise FeedError(f"JSON decode error from {url}") from exc


# default client instance for simple callers
_default_client = RequestsClient(user_agent=APP_USER_AGENT)


def _today_utc_date() -> date:
    override = os.getenv("FINBUDDY_TODAY")
    if override:
        try:
            return datetime.strptime(override, "%Y-%m-%d").date()
        except ValueError:
            logger.warning("Malformed FINBUDDY_TODAY=%r; falling back to UTC today", override)
    return datetime.now(timezone.utc).date()


def _to_yyyymmddhhmmss(d: date, end_of_day: bool = False) -> str:
    if end_of_day:
        dt = datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)
    else:
        dt = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
    return dt.strftime("%Y%m%d%H%M%S")


# -------------------------------
# SEC: Ticker -> CIK (identifier)
# -------------------------------
@lru_cache(maxsize=1)
def _fetch_company_tickers_cached() -> Dict[str, Any]:
    url = "https://www.sec.gov/files/company_tickers.json"
    return _default_client.get_json(url, headers=SEC_HEADERS)

def _fetch_company_tickers(client: RequestsClient = _default_client) -> Dict[str, Any]:
    if client is _default_client:
        return _fetch_company_tickers_cached()
    # If caller passed a custom client, fetch fresh (no caching)
    url = "https://www.sec.gov/files/company_tickers.json"
    return client.get_json(url)


def sec_get_cik_record(ticker: str, client: RequestsClient = _default_client) -> Dict[str, Any]:
    if not ticker or not ticker.strip():
        raise ValueError("ticker must be a non-empty string")
    t = ticker.strip().upper()
    data = _fetch_company_tickers(client)
    # data keys are numeric strings; values contain 'ticker' and 'cik_str'
    for _, rec in data.items():
        if rec.get("ticker", "").upper() == t:
            cik_int = int(rec["cik_str"])
            return {
                "ticker": t,
                "cik10": f"{cik_int:010d}",
                "cik": str(cik_int),
                "name": rec.get("title"),
            }
    raise NotFoundError(f"Ticker {ticker} not found in SEC list")


# --------------------------------------
# SEC: Recent filings (submissions feed)
# --------------------------------------
def sec_recent_filings(
    cik10: str,
    forms: Optional[List[str]] = None,
    limit: int = 10,
    client: RequestsClient = _default_client
) -> List[Dict[str, Any]]:
    forms = forms or ["10-K", "10-Q", "8-K"]
    if not cik10 or len(cik10) != 10 or not cik10.isdigit():
        raise ValueError("cik10 must be a zero-padded 10-digit string, e.g. '0000320193'")
    url = f"{SEC_BASE}/submissions/CIK{cik10}.json"
    data = client.get_json(url, headers=SEC_HEADERS)
    cik_nozeros = str(int(cik10))
    rec = data.get("filings", {}).get("recent", {})
    results: List[Dict[str, Any]] = []

    rows = zip(
        rec.get("form", []),
        rec.get("filingDate", []),
        rec.get("reportDate", []),
        rec.get("accessionNumber", []),
        rec.get("primaryDocument", []),
        rec.get("items", []),
    )
    for form, filing_date, report_date, accn, primary_doc, items in rows:
        if form not in forms:
            continue
        accn_nodash = (accn or "").replace("-", "")
        doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_nozeros}/{accn_nodash}/{primary_doc}"
        index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_nozeros}/{accn_nodash}-index.html"
        results.append({
            "form": form,
            "filing_date": filing_date,
            "report_date": report_date or None,
            "accession": accn,
            "primary_doc": primary_doc,
            "items": items or "",
            "doc_url": doc_url,
            "index_url": index_url,
        })
        if len(results) >= limit:
            break
    return results


# --------------------------------
# SEC: Company facts (XBRL JSON)
# --------------------------------
def sec_company_facts(cik10: str, client: RequestsClient = _default_client) -> Dict[str, Any]:
    if not cik10 or len(cik10) != 10 or not cik10.isdigit():
        raise ValueError("cik10 must be a zero-padded 10-digit string, e.g. '0000320193'")
    url = f"{SEC_BASE}/api/xbrl/companyfacts/CIK{cik10}.json"
    return client.get_json(url, headers=SEC_HEADERS)


# -----------------------
# GDELT: Recent news API
# -----------------------
def gdelt_news(
    query: str,
    days: int = 14,
    limit: int = 20,
    end_date: Optional[str] = None,
    start_date: Optional[str] = None,
    client: RequestsClient = _default_client
) -> List[Dict[str, Any]]:
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    params: Dict[str, Any] = {
        "query": query,            # DO NOT pre-quote; requests will encode
        "mode": "ArtList",
        "format": "JSON",
        "maxrecords": int(limit),
        "sort": "DateDesc",
    }

    # Determine effective end date
    if end_date:
        try:
            eff_end = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError("end_date must be YYYY-MM-DD") from exc
    else:
        eff_end = _today_utc_date()

    if start_date:
        try:
            eff_start = datetime.strptime(start_date, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError("start_date must be YYYY-MM-DD") from exc
        params["STARTDATETIME"] = _to_yyyymmddhhmmss(eff_start, end_of_day=False)
        params["ENDDATETIME"] = _to_yyyymmddhhmmss(eff_end, end_of_day=True)
    else:
        params["timespan"] = f"{days}days"
        params["ENDDATETIME"] = _to_yyyymmddhhmmss(eff_end, end_of_day=True)

    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    js = client.get_json(url, params=params)
    arts = js.get("articles", []) or []
    out: List[Dict[str, Any]] = []
    seen = set()
    for a in arts:
        key = a.get("url") or a.get("title")
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "title": a.get("title"),
            "url": a.get("url"),
            "seendate": a.get("seendate"),
            "domain": a.get("domain"),
            "language": a.get("language"),
        })
    return out[:limit]


# -----------------------
# Stooq: Daily OHLC CSV
# -----------------------
def stooq_prices(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    client: RequestsClient = _default_client
) -> List[Dict[str, Any]]:
    if not symbol or not symbol.strip():
        raise ValueError("symbol must be non-empty")
    sym = symbol.strip().lower()
    urls = [f"https://stooq.com/q/d/l/?s={sym}&i=d"]
    if not sym.endswith(".us"):
        urls.append(f"https://stooq.com/q/d/l/?s={sym}.us&i=d")

    # Determine filter window
    start_dt: Optional[date] = None
    if start_date:
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError("start_date must be YYYY-MM-DD") from exc

    end_dt: date
    if end_date:
        try:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError("end_date must be YYYY-MM-DD") from exc
    else:
        end_dt = _today_utc_date()

    last_err: Optional[str] = None
    for url in urls:
        try:
            r = client.get(url)
        except HTTPFeedError as exc:
            last_err = f"HTTP error when fetching {url}: {exc}"
            logger.debug(last_err)
            continue
        except requests.RequestException as exc:
            last_err = f"Network error when fetching {url}: {exc}"
            logger.debug(last_err)
            continue

        txt = r.text or ""
        # Stooq returns plain "No data" or an error body for unknown symbols
        if r.status_code != 200 or txt.strip().lower().startswith("error") or not txt.strip():
            last_err = f"No data at {url}"
            logger.debug("Stooq returned no data for %s: status=%s, body=%r", url, r.status_code, txt[:200])
            continue

        f = io.StringIO(txt)
        rows: List[Dict[str, Any]] = []
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows missing mandatory fields
            if not row.get("Date") or not row.get("Close"):
                continue
            try:
                d = datetime.strptime(row["Date"].strip(), "%Y-%m-%d").date()
            except Exception:
                # Malformed date; skip
                logger.debug("Skipping malformed date row: %r", row)
                continue
            if start_dt and d < start_dt:
                continue
            if d > end_dt:
                continue

            def _to_float(v: Optional[str]) -> Optional[float]:
                if v is None:
                    return None
                v = v.strip()
                if v == "" or v.lower() == "nan":
                    return None
                try:
                    return float(v.replace(",", ""))
                except Exception:
                    return None

            def _to_int(v: Optional[str]) -> Optional[int]:
                if v is None:
                    return None
                v = v.strip()
                if v == "" or v.lower() == "nan":
                    return None
                try:
                    return int(float(v.replace(",", "")))
                except Exception:
                    return None

            rows.append({
                "date": row["Date"].strip(),
                "open": _to_float(row.get("Open")),
                "high": _to_float(row.get("High")),
                "low": _to_float(row.get("Low")),
                "close": _to_float(row.get("Close")),
                "volume": _to_int(row.get("Volume")),
            })

        if rows:
            return rows

    # Nothing found on any candidate URL
    raise NotFoundError(last_err or f"Price data not found for symbol {symbol}")
