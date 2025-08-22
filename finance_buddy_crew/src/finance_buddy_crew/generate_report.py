#!/usr/bin/env python3
"""
generate_report.py

Clean, self-contained report generator using jinja2.

Outputs:
 - output/report_preview.html
 - output/report.pdf (if WeasyPrint/pdfkit available)
 - output/figs/revenue_net_income.png
 - output/figs/returns.png

Required packages:
 - pandas
 - matplotlib
 - seaborn
 - jinja2
 - weasyprint (optional, for PDF)
 - pdfkit (optional, for PDF)
"""
from __future__ import annotations
import json
import base64
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template


def find_project_root(marker="pyproject.toml") -> Path:

    """Finds the project root by searching upwards for a marker file."""
    current_path = Path(__file__).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Project root marker '{marker}' not found from {current_path}.")

sns.set_style("whitegrid")


# PDF rendering: try WeasyPrint, then pdfkit
def render_pdf(html: str, out_pdf: Path) -> bool:
    """Try to render HTML to PDF using WeasyPrint or pdfkit."""
    try:
        from weasyprint import HTML  # type: ignore
        HTML(string=html).write_pdf(str(out_pdf))
        return True
    except Exception:
        try:
            import pdfkit  # type: ignore
            pdfkit.from_string(html, str(out_pdf))
            return True
        except Exception:
            return False

try:
    ROOT = find_project_root()
    OUT_DIR = ROOT / "output"
    FIG_DIR = OUT_DIR / "figs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
except FileNotFoundError as e:
    print(f"Error: Could not find project root. {e}")
    exit(1)


def find_json(name_tokens: List[str]) -> Optional[Path]:
    """Finds the first existing JSON file by searching for name patterns in the output directory."""
    for name in name_tokens:
        p = OUT_DIR / name
        if p.exists():
            return p
    # Fallback to pattern matching
    for p in OUT_DIR.glob("*.json"):
        nm = p.name.lower()
        for tok in name_tokens:
            if tok.replace(".json", "").lower() in nm:
                return p
    return None

def load_json(p: Optional[Path]) -> Dict[str, Any]:
    """Safely loads a JSON file, returning an empty dict if it fails."""
    if not p or not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Warning: Failed to load or parse JSON file {p.name}. Error: {e}")
        return {}

FILE_PATTERNS = {
    "financial_facts": ["company_facts_snapshot.json"],
    "edgar_basics": ["edgar_basics.json"],
    "price_context": ["price_context.json"],
    "news_and_catalysts": ["news_and_catalysts.json"],
    "synthesis_brief": ["synthesis_brief.json"],
    "fundamental_analysis": ["fundamental_analysis.json"],
    "qualitative_analysis": ["qualitative_analysis.json"],
}

def load_all_data(patterns: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
    """Finds and loads all necessary JSON files based on patterns into a single dictionary."""
    print("--- Loading Report Data ---")
    loaded_data = {}
    for key, file_patterns in patterns.items():
        found_path = find_json(file_patterns)
        if found_path:
            print(f"  - Found '{key}' data at: {found_path.name}")
            loaded_data[key] = load_json(found_path)
        else:
            print(f"  - WARNING: Could not find data file for '{key}'. Using empty data.")
            loaded_data[key] = {}
    print("---------------------------")
    return loaded_data

DATA = load_all_data(FILE_PATTERNS)



# Data Processing Helpers
def normalize_return_value(raw: Any) -> Optional[float]:
    """Normalize return value, expecting it to already be in percentage format."""
    if raw is None:
        return None
    try:
        v = float(raw)
    except (ValueError, TypeError):
        return None
    if math.isnan(v):
        return None
    # Returns are already in percentage format from price_context_task
    return v


def sanitize_list(raw: Any) -> List[str]:
    """Convert raw data to clean list of strings."""
    if not raw or not isinstance(raw, list):
        return []
    return [str(x).strip() for x in raw if x and str(x).strip()]


def first_val(arr: Optional[List[Dict[str, Any]]]) -> Optional[Any]:
    """Extract first non-null value from array of objects."""
    if not arr or not isinstance(arr, list):
        return None
    for el in arr:
        if isinstance(el, dict) and el.get("value") is not None:
            return el.get("value")
    return None


# Metadata Extraction
def extract_ticker() -> str:
    """Extract ticker symbol from multiple possible sources."""
    for src in (DATA["synthesis_brief"], DATA["financial_facts"], DATA["edgar_basics"], DATA["price_context"], DATA["news_and_catalysts"]):
        if not isinstance(src, dict):
            continue

        # Direct ticker field
        if src.get("ticker"):
            return src.get("ticker")

        # Check raw_input
        ri = src.get("raw_input") or {}
        if isinstance(ri, dict) and ri.get("ticker"):
            return ri.get("ticker")

        # Check raw_facts entity
        rf = src.get("raw_facts") or {}
        ent = rf.get("entity") or {}
        if isinstance(ent, dict) and ent.get("tradingSymbol"):
            return ent.get("tradingSymbol")

    return "UNKNOWN"


def extract_company() -> str:
    """Extract company name from multiple possible sources."""
    for src in (DATA["edgar_basics"], DATA["financial_facts"], DATA["synthesis_brief"]):
        if not isinstance(src, dict):
            continue

        if src.get("company_name"):
            return src.get("company_name")

        # Check raw_facts entity
        rf = src.get("raw_facts") or {}
        ent = rf.get("entity") or {}
        if isinstance(ent, dict) and ent.get("entityRegistrantName"):
            return ent.get("entityRegistrantName")

        # Check raw_input
        ri = src.get("raw_input") or {}
        if isinstance(ri, dict) and ri.get("company"):
            return ri.get("company")

    return extract_ticker()


def extract_as_of() -> str:
    """Extract as_of date from multiple possible sources."""
    for src in (DATA["synthesis_brief"], DATA["financial_facts"], DATA["price_context"], DATA["edgar_basics"]):
        if not isinstance(src, dict):
            continue

        ri = src.get("raw_input") or {}
        if isinstance(ri, dict) and ri.get("as_of"):
            return ri.get("as_of")

        rf = src.get("raw_facts") or {}
        if isinstance(rf, dict) and rf.get("asOf"):
            return rf.get("asOf")

        if src.get("as_of"):
            return src.get("as_of")

    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# Extract primary metadata
TICKER = extract_ticker()
COMPANY = extract_company()
AS_OF = extract_as_of()

print("Processing synthesis brief...")
tldr = sanitize_list(DATA["synthesis_brief"].get("tldr") or [])
positives = sanitize_list(DATA["synthesis_brief"].get("positives") or [])
risks = sanitize_list(DATA["synthesis_brief"].get("risks") or [])
next_steps = sanitize_list(DATA["synthesis_brief"].get("next_steps") or [])
disclaimer = DATA["synthesis_brief"].get("disclaimer") or ""

print("Processing specialist analysis...")
fundamental_takeaways = sanitize_list(DATA["fundamental_analysis"].get("fundamental_takeaways") or [])
qualitative_takeaways = sanitize_list(DATA["qualitative_analysis"].get("qualitative_takeaways") or [])

print("Processing data points...")
notable_filings = sanitize_list(DATA["synthesis_brief"].get("notable_filings") or [])
recent_catalysts = sanitize_list(DATA["synthesis_brief"].get("recent_catalysts") or [])

# Financial metrics
revenue = first_val(DATA["financial_facts"].get("revenue"))
cash = first_val(DATA["financial_facts"].get("cash_or_fcf_proxy"))
debt = first_val(DATA["financial_facts"].get("debt"))
eps = first_val(DATA["financial_facts"].get("diluted_eps"))
shares = first_val(DATA["financial_facts"].get("shares_diluted"))

# Returns processing
provided_returns = DATA["price_context"].get("returns", {}) if isinstance(DATA["price_context"], dict) else {}
final_returns: Dict[str, Optional[float]] = {}
for k in ("d1m", "d3m", "d6m", "d12m"):
    raw = provided_returns.get(k)
    final_returns[k] = normalize_return_value(raw) if raw is not None else None

vol_proxy = provided_returns.get("vol_proxy")
try:
    vol_percent = float(vol_proxy) * 100.0 if vol_proxy is not None else None
except (ValueError, TypeError):
    vol_percent = None


# Chart Generation
def make_financial_trend_chart(facts_obj: Dict[str, Any], outpath: Path) -> None:
    """Generate financial trend chart showing revenue, operating income, and net income."""

    def to_df(arr):
        rows = []
        for el in (arr or []):
            period = el.get("period") or ""
            val = el.get("value")
            if val is None:
                continue
            rows.append({"period": period, "value": float(val)})
        return pd.DataFrame(rows)

    rev_df = to_df(facts_obj.get("revenue"))
    op_df = to_df(facts_obj.get("operating_income"))
    net_df = to_df(facts_obj.get("net_income"))

    # Handle empty data
    if rev_df.empty and op_df.empty and net_df.empty:
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "No financial series available",
                 ha="center", va="center")
        plt.axis("off")
        plt.savefig(outpath, dpi=150)
        plt.close()
        return

    # Build unified periods
    periods = []
    for d in (rev_df, op_df, net_df):
        if not d.empty:
            periods.extend(list(d["period"].astype(str)))
    periods = list(dict.fromkeys(periods))  # Preserve order, remove duplicates

    # Create consolidated data
    rows = []
    for p in periods:
        rev = rev_df[rev_df["period"] == p]["value"].values
        op = op_df[op_df["period"] == p]["value"].values
        net = net_df[net_df["period"] == p]["value"].values

        rows.append({
            "period": p,
            "revenue": float(rev[0]) if len(rev) else 0.0,
            "operating": float(op[0]) if len(op) else 0.0,
            "net": float(net[0]) if len(net) else 0.0
        })

    df = pd.DataFrame(rows)

    # Create chart
    x = list(range(len(df)))
    width = 0.25
    plt.figure(figsize=(8, 4))

    plt.bar([i - width for i in x], df["revenue"],
            width=width, label="Revenue", color="#2b7bba")
    plt.bar(x, df["operating"],
            width=width, label="Operating Income", color="#2bba7b")
    plt.bar([i + width for i in x], df["net"],
            width=width, label="Net Income", color="#f2b134")

    plt.xticks(x, df["period"], rotation=10, fontsize=9)
    plt.title(f"{COMPANY} — Revenue & Profitability")

    # Format y-axis
    import matplotlib.ticker as mtick
    plt.gca().yaxis.set_major_formatter(
        mtick.FuncFormatter(
            lambda v, pos: f"${v / 1e9:.1f}B" if abs(v) >= 1e9 else f"${v:,.0f}"
        )
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def make_returns_chart(returns_obj: Dict[str, Optional[float]], outpath: Path) -> None:
    """Generate returns chart showing performance over different periods."""
    labels = []
    vals = []

    for key, label in (("d1m", "1M"), ("d3m", "3M"), ("d6m", "6M"), ("d12m", "12M")):
        v = returns_obj.get(key)
        if v is None:
            continue
        labels.append(label)
        vals.append(float(v))

    if not labels:
        plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "Returns data not available",
                 ha="center", va="center")
        plt.axis("off")
        plt.savefig(outpath, dpi=150)
        plt.close()
        return

    plt.figure(figsize=(6, 3))
    pal = sns.color_palette("RdYlGn", len(labels))
    sns.barplot(x=labels, y=vals, palette=pal)
    plt.ylabel("Total return (%)")
    plt.title(f"{TICKER} — Returns")

    # Add value labels on bars
    max_val = max(vals) if vals else 1
    for i, v in enumerate(vals):
        plt.text(i, v + (0.01 * max(1, abs(max_val))),
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# Generate charts
FIG_REV = FIG_DIR / "revenue_net_income.png"
FIG_RET = FIG_DIR / "returns.png"
make_financial_trend_chart(DATA["financial_facts"], FIG_REV)
make_returns_chart(final_returns, FIG_RET)


def img_to_datauri(p: Path) -> str:
    """Convert image file to data URI for embedding in HTML."""
    if not p.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


rev_uri = img_to_datauri(FIG_REV)
ret_uri = img_to_datauri(FIG_RET)

# HTML Template
TEMPLATE = """
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>{{ company }} — Research Brief</title>
    <style>
        @page { size: A4; margin: 18mm; }
        body { font-family: Arial, Helvetica, sans-serif; color: #222; }
        header { 
            display: flex; 
            justify-content: space-between; 
            border-bottom: 1px solid #eee; 
            padding-bottom: 6px; 
        }
        h1 { font-size: 18px; margin: 0; }
        .muted { color: #666; font-size: 12px; }
        .two-col { display: flex; gap: 18px; }
        .left { flex: 1.4; }
        .right { flex: 0.9; }
        .metrics { 
            width: 100%; 
            border-collapse: collapse; 
            margin-top: 6px; 
        }
        .metrics th { 
            background: #f6f6f6; 
            padding: 6px 8px; 
            text-align: left; 
            width: 40%; 
        }
        .metrics td { padding: 6px 8px; }
        .fig { 
            border: 1px solid #f0f0f0; 
            padding: 6px; 
            background: #fff; 
        }
        ul { margin: 6px 0 12px 18px; padding: 0; }
        p.muted { margin-top: 6px; font-style: italic; }
        footer { 
            border-top: 1px solid #f0f0f0; 
            color: #888; 
            padding-top: 8px; 
            text-align: center; 
            font-size: 11px; 
            margin-top: 14px; 
        }
        pre.raw { 
            background: #fafafa; 
            border: 1px solid #eee; 
            padding: 8px; 
            font-size: 11px; 
            overflow: auto; 
            white-space: pre-wrap;
            word-break: break-all;
        }
    </style>
</head>
<body>
    <header>
        <div>
            <h1>{{ company }} <small>({{ ticker }})</small></h1>
            <div class="muted">Research Brief • As of {{ as_of }}</div>
        </div>
        <div class="muted">
            Generated: {{ generated_at }}<br/>
            Source: CrewAI outputs
        </div>
    </header>

    <div class="two-col" style="margin-top:12px">
        <div class="left">
            <h2>TL;DR</h2>
            {% if tldr %}
                <ul>
                    {% for b in tldr %}<li>{{ b }}</li>{% endfor %}
                </ul>
            {% else %}
                <p class="muted">No TL;DR summary was generated.</p>
            {% endif %}

            <h2>Key Financials</h2>
            <table class="metrics">
                <tr><th>Ticker</th><td>{{ ticker }}</td></tr>
                <tr><th>Company</th><td>{{ company }}</td></tr>
                <tr><th>As of</th><td>{{ as_of }}</td></tr>
                <tr><th>Revenue (LTM)</th><td>{{ revenue }}</td></tr>
                <tr><th>Operating cash / proxy</th><td>{{ cash }}</td></tr>
                <tr><th>Reported debt</th><td>{{ debt }}</td></tr>
                <tr><th>Diluted EPS (LTM)</th><td>{{ eps }}</td></tr>
                <tr><th>Diluted shares (approx)</th><td>{{ shares }}</td></tr>
            </table>

            <h2>Charts</h2>
            <div class="fig">
                <img src="{{ rev_uri }}" style="width:100%">
            </div>
            <div style="height:8px"></div>
            <div class="fig">
                <img src="{{ ret_uri }}" style="width:60%">
            </div>

            <!-- ADDED: Direct output from the Fundamental Analyst -->
            <h2>Fundamental Analysis</h2>
            {% if fundamental_takeaways %}
                <ul>
                    {% for b in fundamental_takeaways %}<li>{{ b }}</li>{% endfor %}
                </ul>
            {% else %}
                <p class="muted">No fundamental analysis takeaways were generated.</p>
            {% endif %}

            <h2>Notable Filings</h2>
            {% if notable_filings %}
                <ul>
                    {% for f in notable_filings %}<li>{{ f }}</li>{% endfor %}
                </ul>
            {% else %}
                <p class="muted">No notable filings were identified.</p>
            {% endif %}
        </div>

        <div class="right">
            <h2>Positives</h2>
            {% if positives %}
                <ul>
                    {% for p in positives %}<li>{{ p }}</li>{% endfor %}
                </ul>
            {% else %}
                <p class="muted">No positives were identified.</p>
            {% endif %}

            <h2>Risks</h2>
            {% if risks %}
                <ul>
                    {% for r in risks %}<li>{{ r }}</li>{% endfor %}
                </ul>
            {% else %}
                <p class="muted">No risks were identified.</p>
            {% endif %}

            <!-- ADDED: Direct output from the Qualitative Analyst -->
            <h2>Qualitative Analysis</h2>
            {% if qualitative_takeaways %}
                <ul>
                    {% for b in qualitative_takeaways %}<li>{{ b }}</li>{% endfor %}
                </ul>
            {% else %}
                <p class="muted">No qualitative analysis takeaways were generated.</p>
            {% endif %}

            <h2>Recent Catalysts</h2>
            {% if recent_catalysts %}
                <ul>
                    {% for c in recent_catalysts %}<li>{{ c }}</li>{% endfor %}
                </ul>
            {% else %}
                <p class="muted">No recent catalysts were identified.</p>
            {% endif %}

            <h2>Next Steps</h2>
            {% if next_steps %}
                <ul>
                    {% for n in next_steps %}<li>{{ n }}</li>{% endfor %}
                </ul>
            {% else %}
                <p class="muted">No next steps were identified.</p>
            {% endif %}

            <div style="margin-top:12px;color:#666">
                <strong>Disclaimer</strong>
                <div>{{ disclaimer }}</div>
            </div>
        </div>
    </div>

    <h2>Data sources (raw)</h2>
    <pre class="raw">{{ raw_audit | tojson(indent=2) }}</pre>

    <footer>
        Confidential • For internal use only • Generated by FinanceBuddy
    </footer>
</body>
</html>
"""

# --- FINAL: RENDER REPORT ---

# Create a safe, default raw_audit dictionary to prevent errors
raw_audit_data = {
    "provided_returns": provided_returns or {},
    "normalized_returns_percent": final_returns or {},
    "vol_percent": vol_percent if vol_percent is not None else "N/A"
}

# Build the complete context dictionary with all keys the template needs
context = {
    # Metadata
    "company": COMPANY,
    "ticker": TICKER,
    "as_of": AS_OF,
    # Best practice: use timezone.utc for timezone-aware datetimes
    "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),

    # Synthesis Brief Content
    "tldr": tldr,
    "positives": positives,
    "risks": risks,
    "notable_filings": notable_filings,
    "recent_catalysts": recent_catalysts,
    "next_steps": next_steps,
    "disclaimer": disclaimer,

    # Specialist Analysis Content
    "fundamental_takeaways": fundamental_takeaways,
    "qualitative_takeaways": qualitative_takeaways,

    # Key Financial Metrics (formatted for display with safe defaults)
    "revenue": f"${int(revenue):,}" if revenue is not None else "N/A",
    "cash": f"${int(cash):,}" if cash is not None else "N/A",
    "debt": f"${int(debt):,}" if debt is not None else "N/A",
    "eps": f"{eps:.2f}" if eps is not None else "N/A",
    "shares": f"{int(shares):,}" if shares is not None else "N/A",

    # Image Data URIs for embedding charts
    "rev_uri": rev_uri,
    "ret_uri": ret_uri,

    # Raw data for the audit section
    "raw_audit": raw_audit_data,
}

# Render the HTML using the Jinja2 template
print("Rendering final HTML report...")
template = Template(TEMPLATE)
html = template.render(context)

# Write the HTML preview file
preview = OUT_DIR / "report_preview.html"
preview.write_text(html, encoding="utf-8")

# Attempt to render the PDF
pdf_out = OUT_DIR / "report.pdf"
ok = render_pdf(html, pdf_out)

# Print final status messages
print("\n--- Report Generation Complete ---")
print(f"✅ HTML preview written to: {preview}")
if ok:
    print(f"✅ PDF written to: {pdf_out}")
else:
    print("⚠️ PDF renderer not available or failed; open the HTML preview and print to PDF manually.")