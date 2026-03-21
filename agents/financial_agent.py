"""
agents/financial_agent.py — PRODUCTION VERSION with DB persistence
Reverted to Groq (meta-llama/llama-4-scout-17b-16e-instruct) with:
  - Token-aware context management & adaptive history trimming
  - SQLite-persisted conversation history
  - ChromaDB semantic pattern search
"""

import os
import json
from typing import Optional
from groq import Groq
from dotenv import load_dotenv
from tools.market_data import StockData, get_range_news, compute_overall_sentiment
from database.db import (
    save_message, load_chat_history, clear_chat_history,
    find_similar_patterns_sql, semantic_pattern_search,
)

load_dotenv()

# Standard OpenAI/Groq Tool format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_price_range",
            "description": "Analyze a specific date range on the stock chart. Returns price movement statistics and all news events in that window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
                    "end_date":   {"type": "string", "description": "End date YYYY-MM-DD"},
                    "ticker":     {"type": "string", "description": "Stock ticker symbol"},
                },
                "required": ["start_date", "end_date", "ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forecast_trend",
            "description": "Generate a bullish/bearish probability forecast based on recent news sentiment and price momentum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker":  {"type": "string"},
                    "horizon": {"type": "string", "enum": ["7d", "30d"]},
                },
                "required": ["ticker", "horizon"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar_periods",
            "description": "Find historical date ranges with similar news sentiment and market conditions. Searches a real database of past patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "current_sentiment_score": {"type": "number", "description": "Current sentiment score -1.0 to 1.0"},
                    "query_description": {"type": "string", "description": "Natural language description of current market setup for semantic search"},
                },
                "required": ["ticker", "current_sentiment_score"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_news_category",
            "description": "Summarize news events by category and their market impact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker":   {"type": "string"},
                    "category": {"type": "string", "enum": ["earnings","product","management","policy","market","competition"]},
                },
                "required": ["ticker", "category"],
            },
        },
    },
]


def execute_tool(tool_name: str, tool_input: dict, stock_data: StockData) -> str:

    if tool_name == "analyze_price_range":
        start, end = tool_input["start_date"], tool_input["end_date"]
        prices = stock_data.prices
        mask   = (prices.index >= start) & (prices.index <= end)
        rng    = prices.loc[mask]
        if rng.empty:
            return json.dumps({"error": "No price data found for that date range."})
        p_start  = float(rng["Close"].iloc[0])
        p_end    = float(rng["Close"].iloc[-1])
        pct      = ((p_end - p_start) / p_start) * 100
        rng_news = get_range_news(stock_data.news, start, end)
        return json.dumps({
            "ticker": tool_input["ticker"],
            "start_date": start, "end_date": end,
            "price_start": round(p_start, 2), "price_end": round(p_end, 2),
            "pct_change": round(pct, 2),
            "period_high": round(float(rng["High"].max()), 2),
            "period_low":  round(float(rng["Low"].min()), 2),
            "avg_daily_volume": int(rng["Volume"].mean()),
            "news_events": [{"date": n.date, "title": n.title, "sentiment": n.sentiment,
                             "category": n.category, "score": n.sentiment_score} for n in rng_news],
            "bullish_count": sum(1 for n in rng_news if n.sentiment == "bullish"),
            "bearish_count": sum(1 for n in rng_news if n.sentiment == "bearish"),
        })

    elif tool_name == "forecast_trend":
        news     = stock_data.news
        overall  = compute_overall_sentiment(news)
        prices   = stock_data.prices["Close"].values
        momentum = (prices[-1] - prices[-min(10, len(prices))]) / prices[-min(10, len(prices))]
        raw      = (overall * 0.6) + (momentum * 0.4 * 5)
        bull_prob = min(95, max(5, 50 + raw * 40))
        bear_prob = 100 - bull_prob
        high      = [n for n in news if n.impact == "high"]
        return json.dumps({
            "ticker": tool_input["ticker"],
            "horizon": tool_input["horizon"],
            "bullish_probability": round(bull_prob, 1),
            "bearish_probability": round(bear_prob, 1),
            "verdict": "Bullish" if bull_prob > 55 else "Bearish" if bear_prob > 55 else "Neutral",
            "overall_sentiment_score": overall,
            "momentum_signal": "Positive" if momentum > 0 else "Negative",
            "key_risk_events":  [n.title for n in high if n.sentiment == "bearish"][:3],
            "key_catalysts":    [n.title for n in high if n.sentiment == "bullish"][:3],
        })

    elif tool_name == "find_similar_periods":
        ticker  = tool_input["ticker"]
        score   = tool_input["current_sentiment_score"]
        query   = tool_input.get("query_description", f"{ticker} sentiment {score:+.2f}")

        # Try semantic vector search first
        patterns = semantic_pattern_search(query_text=query, ticker=ticker, top_k=3)

        # Fall back to SQL similarity search
        if not patterns:
            sql_rows = find_similar_patterns_sql(ticker, score, top_k=3)
            patterns = [
                {
                    "ticker": r["ticker"],
                    "period": f"{r['period_start']} to {r['period_end']}",
                    "sentiment_score": r["sentiment_score"],
                    "price_change_pct": r.get("price_change_pct", 0),
                    "outcome_30d": r["outcome_30d"],
                    "dominant_category": r["dominant_category"],
                    "context_summary": r["context_summary"],
                    "similarity": round(1 - abs(r["sentiment_score"] - score), 3),
                }
                for r in sql_rows
            ]

        return json.dumps({
            "ticker": ticker,
            "current_sentiment_score": score,
            "search_method": "semantic" if patterns and "similarity" in patterns[0] else "sql",
            "similar_periods": patterns,
            "note": "Past performance does not guarantee future results.",
        })

    elif tool_name == "summarize_news_category":
        category = tool_input["category"]
        cat_news = [n for n in stock_data.news if n.category == category]
        if not cat_news:
            return json.dumps({"message": f"No {category} news found for {tool_input['ticker']}"})
        avg_score = sum(n.sentiment_score for n in cat_news) / len(cat_news)
        return json.dumps({
            "ticker": tool_input["ticker"], "category": category,
            "total_events": len(cat_news),
            "avg_sentiment_score": round(avg_score, 2),
            "overall_tone": "Bullish" if avg_score > 0.1 else "Bearish" if avg_score < -0.1 else "Neutral",
            "events": [{"date": n.date, "title": n.title, "sentiment": n.sentiment, "source": n.source}
                       for n in cat_news],
        })

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


class FinancialAgent:
    """
    Agentic AI using Groq with full DB persistence:
    - Smart adaptive context management (retries on 413 errors)
    - Token-aware history trimming
    """

    def __init__(self, api_key: str, session_id: str, model: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        self.client     = Groq(api_key=api_key)
        self.model      = model
        self.session_id = session_id
        self.stock_data: Optional[StockData] = None

        # Load persisted conversation history from DB
        self.conversation_history: list[dict] = load_chat_history(session_id)
        print(f"[Agent] Loaded {len(self.conversation_history)} messages from DB for session {session_id[:8]}...")

        self.system_prompt = """You are an expert financial analyst AI agent specialising in 
event-driven market analysis. You help investors—especially beginners—understand WHY stocks 
move by connecting price action to real news events and narratives.

You have access to tools that let you:
- Analyze specific date ranges on the chart to explain price movements
- Generate AI forecasts based on news sentiment and momentum  
- Find similar historical periods using semantic vector search over a real pattern database
- Summarize news by category (earnings, policy, product, etc.)

Your communication style:
- Clear and educational — explain concepts simply for beginner investors
- Evidence-based — always ground analysis in specific news events
- Balanced — present both bullish and bearish perspectives
- Actionable — give concrete takeaways, not vague generalities

Always call the relevant tools before giving your analysis. Think step by step."""

    def set_stock_data(self, stock_data: StockData):
        self.stock_data = stock_data

    def _estimate_tokens(self, text: str) -> int:
        """Rough estimate of tokens: 4 characters per token."""
        return len(text) // 4

    def _trimmed_history(self, max_tokens: int = 4000) -> list[dict]:
        """Keep conversation history within token limits by trimming older messages."""
        current_tokens = 0
        valid_history = []
        for msg in reversed(self.conversation_history):
            content_tokens = self._estimate_tokens(msg.get("content", ""))
            if current_tokens + content_tokens < max_tokens:
                valid_history.insert(0, msg)
                current_tokens += content_tokens
            else:
                break
        
        trimmed_count = len(self.conversation_history) - len(valid_history)
        if trimmed_count > 0:
            print(f"[Agent] Trimmed {trimmed_count} old messages to stay under {max_tokens} tokens.")
        return valid_history

    def chat(self, user_message: str) -> str:
        if not self.stock_data:
            return "Please load a stock first by selecting a ticker."

        if self._estimate_tokens(user_message) > 4000:
            return "Your message is too long for a single request. Please try splitting it."

        context = (
            f"[Context] Ticker: {self.stock_data.ticker} | "
            f"Price: ${self.stock_data.current_price:.2f} | "
            f"Change: {self.stock_data.price_change_pct:+.2f}% | "
            f"News events: {len(self.stock_data.news)}\n\n"
        )
        full_message = context + user_message

        # Save user message to DB
        save_message(self.session_id, "user", full_message)
        self.conversation_history.append({"role": "user", "content": full_message})

        max_context_tokens = 5000
        for attempt in range(3):
            try:
                trimmed_history = self._trimmed_history(max_tokens=max_context_tokens)
                
                max_iterations = 5
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    *trimmed_history
                ]

                for _ in range(max_iterations):
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="auto",
                        max_tokens=2048,
                        temperature=0.3,
                    )

                    msg = response.choices[0].message
                    if not msg.tool_calls:
                        final = (msg.content or "").strip()
                        save_message(self.session_id, "assistant", final)
                        self.conversation_history.append({"role": "assistant", "content": final})
                        return final

                    tc_list = [
                        {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ]
                    # Persistence of assistant message with tool calls
                    save_message(self.session_id, "assistant", msg.content or "", tool_calls=tc_list)
                    self.conversation_history.append({"role": "assistant", "content": msg.content or "", "tool_calls": tc_list})
                    messages.append({"role": "assistant", "content": msg.content, "tool_calls": tc_list})

                    for tc in msg.tool_calls:
                        try:
                            tool_input = json.loads(tc.function.arguments)
                            result = execute_tool(tc.function.name, tool_input, self.stock_data)
                        except Exception as tool_err:
                            result = json.dumps({"error": f"Tool failed: {tool_err}"})
                        
                        save_message(self.session_id, "tool", result)
                        self.conversation_history.append({"role": "tool", "content": result, "tool_call_id": tc.id, "name": tc.function.name})
                        messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": result})

                return "Maximum reasoning steps reached."

            except Exception as e:
                err_str = str(e).lower()
                if ("413" in err_str or "too large" in err_str or "rate limit" in err_str or "limit 6000" in err_str) and attempt < 2:
                    print(f"[Agent] Payload too large or rate limit. Reducing context to {max_context_tokens // 2}...")
                    max_context_tokens //= 2
                    continue
                else:
                    return f"Error: {e}"

        return "Persistent token limit issues. Please try a shorter question."

    def reset_conversation(self):
        """Clear history from both memory and DB."""
        clear_chat_history(self.session_id)
        self.conversation_history = []