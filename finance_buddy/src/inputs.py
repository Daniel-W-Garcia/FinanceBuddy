from datetime import datetime, timezone

def get_inputs():
    """
    Get ticker input from user when using crewai run
    """
    print("\n" + "=" * 50)
    print("🏦 FinanceBuddy - Company Research Assistant")
    print("=" * 50)

    ticker = input("\n📊 Enter the stock ticker (e.g., AAPL, MSFT, TSLA): ").strip().upper()
    while not ticker:
        ticker = input("❌ Ticker cannot be empty. Please enter a valid ticker: ").strip().upper()

    print(f"\n✅ Researching: {ticker}")
    print("🚀 Starting analysis...\n")

    today = datetime.now(timezone.utc).date()
    return {
        'ticker': ticker,
        'company': f"Company with ticker {ticker}",  # Will be resolved by agents
        'timeframe': 'LTM',
        'focus_areas': ['fundamentals', 'recent_news', 'price_performance'],
        'current_date': today.isoformat(),           # e.g., 2025-08-15
        'current_year': str(today.year),             # e.g., 2025
        'as_of': today.isoformat(),
        'output_format': 'executive_summary',
        'constraints': ['public_data_only', 'no_investment_advice']
    }