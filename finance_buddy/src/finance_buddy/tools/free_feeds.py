# Minimal helpers for SEC EDGAR, GDELT news, and Stooq prices.
# Env: export SEC_USER_AGENT="FinanceBuddy/1.0 you@domain.com"

import os
import time
import csv
import io
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus
import requests
from datetime import datetime, timezone, date

SEC_BASE = "https://data.sec.gov"
SEC_HEADERS = {
    "User-Agent": os.getenv("SEC_USER_AGENT", "FinanceBuddy/0.1 you@example.com")
}
APP_USER_AGENT = os.getenv(
    "FINBUDDY_USER_AGENT",
    os.getenv("SEC_USER_AGENT", "FinanceBuddy/1.0 (contact@example.com)")
)

def _today_utc_date() -> date:
    """
    Centralized 'today' in UTC with optional override via FINBUDDY_TODAY (YYYY-MM-DD).
    This allows reproducible runs/tests and ensures all feeds align on the same 'today'.
    """
    override = os.getenv("FINBUDDY_TODAY")
    if override:
        try:
            return datetime.strptime(override, "%Y-%m-%d").date()
        except ValueError:
            # Fallback to actual UTC today if override is malformed
            pass
    return datetime.now(timezone.utc).date()

def _to_yyyymmddhhmmss(d: date, end_of_day: bool = False) -> str:
    """
    Convert date to GDELT's YYYYMMDDHHMMSS format.
    """
    if end_of_day:
        dt = datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)
    else:
        dt = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
    return dt.strftime("%Y%m%d%H%M%S")

def _get(
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    retries: int = 3,
    backoff: float = 1.0,
    timeout: int = 20,
) -> requests.Response:
    """
    Internal GET with polite headers and simple retry/backoff for rate-limited endpoints.
    Uses SEC headers automatically when hitting *.sec.gov, otherwise uses APP_USER_AGENT.
    """
    # Default headers depend on domain; allow caller to add/override via 'headers'
    base_headers = SEC_HEADERS if "sec.gov" in url else {"User-Agent": APP_USER_AGENT}
    h = dict(base_headers)
    if headers:
        h.update(headers)
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=h, timeout=timeout)
            if r.status_code in (200, 204):
                return r
            # Retry on common transient errors / rate limits
            if r.status_code in (429, 403, 500, 502, 503, 504):
                time.sleep(backoff * attempt)
                continue
            r.raise_for_status()
        except Exception as e:
            last_exc = e
            time.sleep(backoff * attempt)
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Failed GET {url}")

# -------------------------------
# SEC: Ticker -> CIK (identifier)
# -------------------------------
def sec_get_cik_record(ticker: str) -> Dict[str, Any]:
    """
    Resolve a stock ticker to SEC identifiers using the official mapping.

    Returns:
        dict: { 'ticker', 'cik10' (zero-padded 10-digit), 'cik' (no leading zeros), 'name' }
    Raises:
        ValueError: if the ticker is not found.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    r = _get(url)
    data = r.json()  # { "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ... }
    t = ticker.strip().upper()
    for _, rec in data.items():
        if rec.get("ticker", "").upper() == t:
            cik_int = int(rec["cik_str"])
            return {
                "ticker": t,
                "cik10": f"{cik_int:010d}",
                "cik": str(cik_int),
                "name": rec.get("title"),
            }
    raise ValueError(f"Ticker {ticker} not found in SEC list")

# --------------------------------------
# SEC: Recent filings (submissions feed)
# --------------------------------------
def sec_recent_filings(
    cik10: str,
    forms: Optional[List[str]] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    List recent filings for a company by CIK (10-K/10-Q/8-K by default).

    Args:
        cik10: 10-digit zero-padded CIK (e.g., '0000320193').
        forms: list of form types to include.
        limit: max results.

    Returns:
        List of dicts with keys:
        ['form','filing_date','report_date','accession','primary_doc','items','doc_url','index_url']
    """
    forms = forms or ["10-K", "10-Q", "8-K"]
    url = f"{SEC_BASE}/submissions/CIK{cik10}.json"
    r = _get(url)
    data = r.json()
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
def sec_company_facts(cik10: str) -> Dict[str, Any]:
    """
    Fetch the raw XBRL company facts JSON for a given 10-digit CIK.

    Note: The JSON contains multiple taxonomies (e.g., 'us-gaap', 'dei'), and each tag
    may have multiple units. Your downstream logic should pick appropriate tags/units.

    Returns:
        dict: Full JSON from SEC companyfacts endpoint.
    """
    url = f"{SEC_BASE}/api/xbrl/companyfacts/CIK{cik10}.json"
    return _get(url).json()

# -----------------------
# GDELT: Recent news API
# -----------------------
def gdelt_news(
    query: str,
    days: int = 14,
    limit: int = 20,
    end_date: Optional[str] = None,
    start_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Fetch recent news mentions via GDELT Doc API.

    Args:
        query: Search string, e.g., '"Apple Inc" OR AAPL'
        days: Timespan in days (ignored if start_date/end_date provided)
        limit: Max number of records
        end_date: Optional YYYY-MM-DD string (defaults to UTC 'today' via FINBUDDY_TODAY or real today)
        start_date: Optional YYYY-MM-DD string (if provided, uses GDELT START/END instead of timespan)

    Returns:
        List of dicts: ['title','url','seendate','domain','language']
    """
    params = {
        "query": quote_plus(query),
        "mode": "ArtList",
        "format": "JSON",
        "maxrecords": str(limit),
        "sort": "DateDesc",
    }

    # Determine the effective end date (defaults to centralized 'today')
    eff_end = (
        datetime.strptime(end_date, "%Y-%m-%d").date()
        if end_date else _today_utc_date()
    )

    if start_date:
        eff_start = datetime.strptime(start_date, "%Y-%m-%d").date()
        params["STARTDATETIME"] = _to_yyyymmddhhmmss(eff_start, end_of_day=False)
        params["ENDDATETIME"] = _to_yyyymmddhhmmss(eff_end, end_of_day=True)
    else:
        # Keep original behavior but clamp to effective 'today'
        params["timespan"] = f"{days}days"
        params["ENDDATETIME"] = _to_yyyymmddhhmmss(eff_end, end_of_day=True)

    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    r = _get(url, params=params)  # Use default APP_USER_AGENT unless overridden by env
    js = r.json()
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
    end_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Fetch daily OHLC price history from Stooq for a given symbol.

    Tries both {symbol} and {symbol}.us (lowercased) to improve US coverage.

    Args:
        symbol: Ticker symbol
        start_date: Optional YYYY-MM-DD to filter results (inclusive)
        end_date: Optional YYYY-MM-DD to filter results (inclusive, defaults to UTC 'today')

    Returns:
        List of dicts: ['date','open','high','low','close','volume']
    """
    sym = symbol.strip().lower()
    urls = [f"https://stooq.com/q/d/l/?s={sym}&i=d"]
    if not sym.endswith(".us"):
        urls.append(f"https://stooq.com/q/d/l/?s={sym}.us&i=d")

    # Determine filter window
    start_dt: Optional[date] = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
    end_dt: date = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else _today_utc_date()

    last_err: Optional[str] = None
    for url in urls:
        r = _get(url)  # Use default APP_USER_AGENT unless overridden by env
        txt = r.text or ""
        if r.status_code != 200 or txt.lower().startswith("error"):
            last_err = f"No data at {url}"
            continue
        f = io.StringIO(txt)
        rows: List[Dict[str, Any]] = []
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("Date") or not row.get("Close"):
                continue
            try:
                d = datetime.strptime(row["Date"], "%Y-%m-%d").date()
                if start_dt and d < start_dt:
                    continue
                if d > end_dt:
                    continue
                rows.append({
                    "date": row["Date"],
                    "open": float(row["Open"]) if row["Open"] else None,
                    "high": float(row["High"]) if row["High"] else None,
                    "low": float(row["Low"]) if row["Low"] else None,
                    "close": float(row["Close"]) if row["Close"] else None,
                    "volume": int(row["Volume"]) if row["Volume"] else None,
                })
            except Exception:
                # Skip malformed lines
                continue
        if rows:
            return rows
    raise ValueError(last_err or "Price data not found")