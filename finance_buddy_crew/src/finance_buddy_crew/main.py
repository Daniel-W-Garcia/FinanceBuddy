#!/usr/bin/env python
"""
Main CLI / script for FinanceBuddy crew.

This file preserves the legacy run() function for interactive use while also
providing argparse-driven CLI subcommands (batch, train, replay, test).
Inputs are validated via Pydantic v2 InputModel.
"""
from __future__ import annotations

import sys
import warnings
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict, ValidationError, model_validator

from .crew import FinanceBuddy
from .tools.extraction_tools import calculate_stock_returns, get_financial_statements

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# configure simple CLI logging (libraries should not configure root logger)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def find_project_root(marker="pyproject.toml") -> Path:
    """Finds the project root by searching upwards for a marker file."""
    current_path = Path(__file__).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Project root marker '{marker}' not found from {current_path}.")


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: Optional[str] = None
    ticker: Optional[str] = None
    timeframe: str = Field(default="LTM")
    focus_areas: List[str] = Field(default_factory=lambda: ["fundamentals", "recent_news", "price_performance"])
    output_format: str = Field(default="executive_summary")
    constraints: List[str] = Field(default_factory=lambda: ["public_data_only", "no_investment_advice"])
    # Derived fields for convenience
    current_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    current_year: str = Field(default_factory=lambda: str(datetime.now().year))

    @model_validator(mode="after")
    def normalize_and_require(self) -> "InputModel":
        # Normalize ticker
        if self.ticker:
            self.ticker = self.ticker.strip().upper()
        # If ticker present but company not, provide a placeholder
        if not self.company and self.ticker:
            self.company = f"Company with ticker {self.ticker}"
        # Require at least one of company or ticker
        if not (self.company or self.ticker):
            raise ValueError("Either 'company' or 'ticker' must be provided.")
        # Ensure lists are strings
        self.focus_areas = [str(x) for x in (self.focus_areas or [])]
        self.constraints = [str(x) for x in (self.constraints or [])]
        return self


def validate_inputs(inputs: Dict[str, Any]) -> InputModel:
    """
    Validate inputs dict and return validated InputModel.
    Raises ValidationError or ValueError on invalid input.
    """
    try:
        im = InputModel.model_validate(inputs)
        return im
    except ValidationError as e:
        logger.error("Input validation error(s): %s", e)
        raise


def run():

    inputs: Dict[str, Any] = {}

    if len(sys.argv) > 1:
        first = sys.argv[1]
        if first not in ("batch", "train", "replay", "test"):
            if "." in first or len(first) <= 5:
                inputs["ticker"] = first
            else:
                inputs["company"] = first

    # Interactive prompt
    if not (inputs.get("ticker") or inputs.get("company")):
        try:
            ticker = input("Enter the stock ticker (e.g., AAPL): ").strip()
            while not ticker:
                ticker = input("Ticker cannot be empty. Enter the stock ticker (e.g., AAPL): ").strip()
            inputs["ticker"] = ticker
        except (EOFError, KeyboardInterrupt):
            logger.error("No input provided; exiting.")
            return

    # Default research params
    inputs.setdefault("timeframe", "LTM")
    inputs.setdefault("focus_areas", ["fundamentals", "recent_news", "price_performance", "key_risks"])
    inputs.setdefault("output_format", "executive_summary")

    try:
        validated = validate_inputs(inputs)
        print(f"\n🔍 Starting research for: {validated.company} ({validated.ticker})")

        print("📊 Calculating price context...")
        price_data = calculate_stock_returns(
            ticker=validated.ticker,
            as_of_date=validated.current_date
        )

        crew_inputs = validated.model_dump()

        returns = price_data.get("returns", {})
        crew_inputs['d1m_return'] = returns.get('d1m', 'N/A')
        crew_inputs['d3m_return'] = returns.get('d3m', 'N/A')
        crew_inputs['d6m_return'] = returns.get('d6m', 'N/A')
        crew_inputs['d12m_return'] = returns.get('d12m', 'N/A')

        print("✅ Price context values extracted.")

        print("📊 Fetching and processing financial statements...")
        company_facts_data = get_financial_statements(ticker=validated.ticker)
        crew_inputs['company_facts_data'] = json.dumps(company_facts_data)
        print("✅ Financial statements processed.")

        # DEBUG: inspect CrewAI's registered output-pydantic mapping keys so we can align tasks.yaml
        try:
            import crewai.project.crew_base as _cb
            for attr in ("output_pydantic_functions", "OUTPUT_PYDANTIC", "output_pydantic"):
                if hasattr(_cb, attr):
                    obj = getattr(_cb, attr)
                    try:
                        keys = list(obj.keys())
                    except Exception:
                        # Fall back to string representation if not a dict
                        keys = str(obj)
                    print("DEBUG: crew_base.%s keys:" % attr, keys)
        except Exception as _e:
            print("DEBUG: couldn't inspect crew_base mapping:", repr(_e))

        fb = FinanceBuddy()
        crew = fb.crew()
        result = crew.kickoff(inputs=crew_inputs)

        # After crew run, validate task outputs (post-run)
        try:
            project_root = find_project_root()
            tasks_yaml_path = str(project_root / "tasks.yaml")
            diagnostics_dir = str(project_root / "output" / "diagnostics")

            # Import and run validator (import inside to avoid import-time cost and circular issues)
            from .tools.validate_task_outputs import validate_all_task_outputs
            validate_all_task_outputs(
                tasks_yaml_path=tasks_yaml_path,
                base_dir=str(project_root),
                diagnostics_dir=diagnostics_dir,
            )
        except Exception as e:
            logger.exception("Post-run validation failed: %s", e)

        # Simple markdown generation (ADD THIS)
        try:
            _create_markdown_report()
            print("📖 Human-readable report: output/synthesis_brief.md")
        except Exception as e:
            logger.warning("Markdown generation failed: %s", e)

        print("\n✅ Research completed successfully!")
        print("📁 JSON reports saved in 'output' directory")
        return result

    except Exception as e:
        logger.exception("Error during run(): %s", e)
        raise


def _create_markdown_report():
    import json
    from pathlib import Path

    project_root = find_project_root()
    output_dir = project_root / "output"
    json_file = output_dir / "synthesis_brief.json"
    md_file = output_dir / "synthesis_brief.md"
    if not json_file.exists():
        return

    with open(json_file) as f:
        data = json.load(f)

    ticker = data.get('raw_input', {}).get('ticker', 'UNKNOWN')

    md_lines = [
        f"# 📊 {ticker} Research Brief",
        "",
        "## Summary",
    ]

    for item in data.get('tldr', []):
        md_lines.append(f"• {item}")

    md_lines.extend(["", "## Key Positives"])
    for item in data.get('positives', []):
        md_lines.append(f"• {item}")

    md_lines.extend(["", "## Key Risks"])
    for item in data.get('risks', []):
        md_lines.append(f"• {item}")

    md_lines.extend(["", "## Next Steps"])
    for item in data.get('next_steps', []):
        md_lines.append(f"• {item}")

    md_lines.extend([
        "",
        "---",
        "",
        data.get('disclaimer', 'This is not investment advice.')
    ])

    with open(md_file, 'w') as f:
        f.write('\n'.join(md_lines))


def batch_run(batch_file: str):
    project_root = find_project_root()
    path = Path(batch_file)
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        logger.error("Batch file not found: %s", batch_file)
        return
    with path.open("r", encoding="utf-8") as fh:
        jobs = json.load(fh)
    results = []
    for job in jobs:
        try:
            validated = validate_inputs(job)
            fb = FinanceBuddy()
            crew = fb.crew()
            res = crew.kickoff(inputs=validated.model_dump())
            results.append({"input": job, "status": "success", "result": res})
        except Exception as e:
            logger.exception("Batch job failed for %s: %s", job, e)
            results.append({"input": job, "status": "error", "error": str(e)})
    out = project_root / "output"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "batch_results.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n✅ Batch processing complete. Results saved to {out / 'batch_results.json'}")


def train(n_iterations: int, filename: str, inputs: Dict[str, Any]):
    validated = validate_inputs(inputs)
    fb = FinanceBuddy()
    crew = fb.crew()
    crew.train(n_iterations=n_iterations, filename=filename, inputs=validated.model_dump())


def replay(task_id: str):
    fb = FinanceBuddy()
    crew = fb.crew()
    crew.replay(task_id=task_id)


def test(n_iterations: int, eval_llm: str, inputs: Dict[str, Any]):
    validated = validate_inputs(inputs)
    fb = FinanceBuddy()
    crew = fb.crew()
    crew.test(n_iterations=n_iterations, eval_llm=eval_llm, inputs=validated.model_dump())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "batch":
            if len(sys.argv) < 3:
                print("Usage: python main.py batch <companies.json>")
            else:
                batch_run(sys.argv[2])
        elif cmd == "train":
            # old train usage: train <n> <filename>
            train(int(sys.argv[2]), sys.argv[3], {"ticker": "AAPL"})
        elif cmd == "replay":
            replay(sys.argv[2])
        elif cmd == "test":
            test(int(sys.argv[2]), sys.argv[3], {"ticker": "MSFT"})
        else:
            # treat as ticker/company
            run()
    else:
        run()
