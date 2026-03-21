import os
import sys
import json
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.financial_agent import FinancialAgent
from tools.market_data import StockData, fetch_stock_data
from database.db import create_session

def test_multi_agent():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not found in .env")
        return

    print("--- Multi-Agent Architecture Verification ---")
    
    # Initialize session and agent
    session_id = create_session(ticker="TSLA")
    from agents.financial_agent import ResearcherAgent
    print("\n[Debug] ResearcherAgent TOOLS_SUBSET (First Tool):")
    print(json.dumps(ResearcherAgent.TOOLS_SUBSET[0], indent=2))
    
    agent = FinancialAgent(api_key=api_key, session_id=session_id)
    
    # Mock stock data for TSLA
    ticker = "TSLA"
    print(f"\n[1] Fetching stock data for {ticker}...")
    stock = fetch_stock_data(ticker)
    if not stock:
        print(f"Failed to fetch stock data for {ticker}")
        return
    agent.set_stock_data(stock)
    
    # Test a complex query that triggers the multi-agent flow
    user_query = "Based on the recent news and macro environment, what is the bull and bear case for Tesla's stock over the next 30 days?"
    print(f"\n[2] Sending Query: {user_query}")
    
    try:
        response = agent.chat(user_query)
        print("\n--- Final Agent Response ---")
        print(response)
        
        # Check if the output looks like a synthesis of research, thesis, and risk
        # (Since we are using real LLM calls, we can't assert exact text, but we can look for keywords)
        keywords = ["bull", "bear", "risk", "research", "thesis", "scenario"]
        found_keywords = [k for k in keywords if k.lower() in response.lower()]
        print(f"\n[Verification] Found keywords: {found_keywords}")
        
        if len(found_keywords) >= 3:
            print("\n[SUCCESS] Multi-agent flow seems operational.")
        else:
            print("\n[WARNING] Response might be lacking multi-perspective synthesis.")
            
    except Exception as e:
        print(f"\n[ERROR] Multi-agent flow failed: {e}")

if __name__ == "__main__":
    test_multi_agent()
