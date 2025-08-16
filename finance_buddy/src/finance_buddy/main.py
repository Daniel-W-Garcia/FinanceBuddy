#!/usr/bin/env python
import sys
import warnings
import json
from datetime import datetime
from typing import Dict, Any, Optional

from .crew import FinanceBuddy

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def validate_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and enhance input parameters"""

    # Required fields with fallback
    if 'company' not in inputs and 'ticker' not in inputs:
        # Provide a default fallback for testing
        print("⚠️  No ticker provided, using default: AAPL")
        inputs['ticker'] = 'AAPL'
        inputs['company'] = 'Apple Inc'

    # Set defaults
    validated = {
        'company': inputs.get('company', inputs.get('ticker', 'Unknown Company')),  # Fallback to ticker if no company
        'ticker': inputs.get('ticker', '').upper(),  # Ensure uppercase
        'timeframe': inputs.get('timeframe', 'LTM'),
        'focus_areas': inputs.get('focus_areas', ['fundamentals', 'recent_news', 'price_performance']),
        'current_date': datetime.now().strftime('%Y-%m-%d'),
        'current_year': str(datetime.now().year),
        'output_format': inputs.get('output_format', 'executive_summary'),
        'constraints': inputs.get('constraints', ['public_data_only', 'no_investment_advice'])
    }

    # If we have ticker but no company name, we'll let the research_analyst figure it out
    if validated['ticker'] and not validated['company']:
        validated['company'] = f"Company with ticker {validated['ticker']}"

    return validated


def run():
    """
    Run the crew for company research.
    """
    inputs = {}

    # If a CLI arg is provided (and it's not a command), use it to set ticker/company
    if len(sys.argv) > 1:
        ticker_or_company = sys.argv[1]
        if ticker_or_company not in ("batch", "train", "replay", "test"):
            if '.' in ticker_or_company or len(ticker_or_company) <= 5:
                inputs['ticker'] = ticker_or_company
            else:
                inputs['company'] = ticker_or_company
    else:
        # Interactive prompt for ticker
        ticker = input("Enter the stock ticker (e.g., AAPL): ").strip()
        while not ticker:
            ticker = input("Ticker cannot be empty. Enter the stock ticker (e.g., AAPL): ").strip()
        inputs['ticker'] = ticker

    # Set default research parameters if not provided
    inputs.setdefault('timeframe', 'LTM')  # Last Twelve Months
    inputs.setdefault('focus_areas', ['fundamentals', 'recent_news', 'price_performance', 'key_risks'])
    inputs.setdefault('output_format', 'executive_summary')

    try:
        validated_inputs = validate_inputs(inputs)
        print(f"\n🔍 Starting research for: {validated_inputs.get('company') or validated_inputs.get('ticker')}")
        print(f"📅 Timeframe: {validated_inputs['timeframe']}")
        print(f"🎯 Focus areas: {', '.join(validated_inputs['focus_areas'])}\n")

        result = FinanceBuddy().crew().kickoff(inputs=validated_inputs)

        print("\n✅ Research completed successfully!")
        print(f"📁 Reports saved in 'output' directory")

        return result

    except Exception as e:
        print(f"❌ An error occurred while running the crew: {e}")
        raise

# ... existing code ...


def batch_run():
    """
    Run research for multiple companies from a JSON file.
    """
    if len(sys.argv) < 2:
        print("Usage: python main.py batch <companies.json>")
        return

    with open(sys.argv[2], 'r') as f:
        companies = json.load(f)

    results = []
    for company_input in companies:
        try:
            print(f"\n{'=' * 50}")
            print(f"Processing: {company_input.get('company') or company_input.get('ticker')}")
            print(f"{'=' * 50}\n")

            validated_inputs = validate_inputs(company_input)
            result = FinanceBuddy().crew().kickoff(inputs=validated_inputs)
            results.append({
                'company': company_input,
                'status': 'success',
                'result': result
            })
        except Exception as e:
            results.append({
                'company': company_input,
                'status': 'error',
                'error': str(e)
            })

    # Save batch results
    with open('output/batch_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Batch processing complete. Results saved to output/batch_results.json")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'ticker': 'AAPL',
        'timeframe': 'LTM',
        'focus_areas': ['fundamentals', 'recent_news']
    }

    try:
        validated_inputs = validate_inputs(inputs)
        FinanceBuddy().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=validated_inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        FinanceBuddy().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        'ticker': 'MSFT',
        'timeframe': 'LTM'
    }

    try:
        validated_inputs = validate_inputs(inputs)
        FinanceBuddy().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=validated_inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "batch":
            batch_run()
        elif command == "train":
            train()
        elif command == "replay":
            replay()
        elif command == "test":
            test()
        else:
            # Treat as ticker/company name
            run()
    else:
        run()
