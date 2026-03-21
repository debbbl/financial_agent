import sys
import os
import json

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.market_data import fetch_macro_context, fetch_options_data, fetch_sec_filings

def test_new_tools():
    print("--- New Tools Verification ---")

    # 1. Test Options Data (yfinance)
    print("\n[Testing Options Data for TSLA]")
    options = fetch_options_data("TSLA")
    if "error" in options:
        print(f"Error fetching options: {options['error']}")
    else:
        print(f"Ticker: {options['ticker']}")
        print(f"Nearest Expiry: {options['nearest_expiry']}")
        print(f"Put/Call Ratio: {options['put_call_ratio']}")
        print(f"Sentiment: {options['options_sentiment']}")
        print(f"Interpretation: {options['interpretation']}")

    # 2. Test SEC Filings
    print("\n[Testing SEC Filings for AAPL]")
    sec = fetch_sec_filings("AAPL")
    if "error" in sec:
        print(f"Error fetching SEC filings: {sec['error']}")
    else:
        print(f"Company: {sec['company']}")
        for f in sec['recent_filings']:
            print(f"  {f['date']} | {f['type']} | {f['description']}")

    # 3. Test Macro Data (FRED)
    print("\n[Testing Macro Data (FRED)]")
    # This will likely fail with 'invalid key' but we want to see the error handling
    macro = fetch_macro_context("2024-01-01", "2024-03-21")
    if "error" in macro:
        print(f"Macro Data Error (Expected if no key): {macro['error']}")
    else:
        print("Macro indicators fetched successfully:")
        for series_id, data in macro.items():
            print(f"  {data['label']}: {data['latest']} ({data['change']} {data['trend']})")

if __name__ == "__main__":
    test_new_tools()
