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
                    "current_sentiment_score": {"type": "string", "description": "Current sentiment score -1.0 to 1.0 (as a string)"},
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
    {
        "type": "function",
        "function": {
            "name": "get_macro_context",
            "description": (
                "Fetch current macroeconomic indicators from FRED — "
                "interest rates, inflation, unemployment, yield curve, VIX. "
                "Use this to contextualise stock movements within the broader economy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
                    "end_date":   {"type": "string", "description": "End date YYYY-MM-DD"},
                },
                "required": ["start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_options_flow",
            "description": (
                "Fetch options chain data including put/call ratio, implied volatility, "
                "and unusual options activity. Use to gauge institutional sentiment — "
                "high PCR suggests bearish hedging, low PCR suggests bullish positioning."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sec_filings",
            "description": "Fetch recent SEC regulatory filings for a company including 8-K material events, 10-Q quarterly reports, and 10-K annual reports from EDGAR.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                },
                "required": ["ticker"],
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
        score   = float(tool_input["current_sentiment_score"])
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

    elif tool_name == "get_macro_context":
        from tools.market_data import fetch_macro_context
        result = fetch_macro_context(
            tool_input["start_date"],
            tool_input["end_date"]
        )
        return json.dumps(result)

    elif tool_name == "get_options_flow":
        from tools.market_data import fetch_options_data
        return json.dumps(fetch_options_data(tool_input["ticker"]))

    elif tool_name == "get_sec_filings":
        from tools.market_data import fetch_sec_filings
        return json.dumps(fetch_sec_filings(tool_input["ticker"]))

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


class ResearcherAgent:
    """Fetches and summarises raw data — news, prices, filings, macro."""
    # TOOLS indices: 0: analyze_price_range, 2: find_similar_periods, 3: summarize_news_category, 4: get_macro_context, 5: get_options_flow, 6: get_sec_filings
    TOOLS_SUBSET = [TOOLS[0], TOOLS[2], TOOLS[3], TOOLS[4], TOOLS[5], TOOLS[6]]
    SYSTEM = """You are a financial research assistant. Your task is to gather 
            raw data (prices, news, macro, filings, options) using your tools.
            IMPORTANT: Use tools to find specific numbers and headlines. 
            Do not provide any analysis or interpretation. 
            Once you have all the data, provide a structured summary."""

    REPORT_FORMAT = """
            ## Price Action
            - Period: [start] to [end]
            - Change: [+/-X.X%]
            - High: $X | Low: $X | Avg volume: X

            ## News Summary ([N] events)
            - [DATE] [CATEGORY] [BULLISH/BEARISH] — [headline]

            ## Sentiment Score
            - Overall: [score] ([Bullish/Bearish/Neutral])

            ## Macro & Filings
            - Fed rate: X% | CPI: X% | UNRATE: X%
            - Last SEC Filing: [DATE] [TYPE]
            - Put/Call ratio: X | Avg IV: X%
            """

class AnalystAgent:
    """Interprets data and builds investment thesis."""
    # TOOLS indices: 1: forecast_trend
    TOOLS_SUBSET = [TOOLS[1]]
    SYSTEM = """You are a senior financial analyst. Build a structured investment thesis 
            based on research data.
            Rules: 
            1. Every claim MUST cite a specific data point from the research.
            2. ONLY use the tools explicitly provided to you. DO NOT hallucinate tools."""

    REPORT_FORMAT = """
            ## Investment Thesis — [TICKER]
            **Verdict:** [BULLISH / BEARISH / NEUTRAL] ([X]% confidence)

            ## Bull/Bear Case
            - [Reason 1 with data]
            - [Reason 2 with data]
            - [Risk factor with data]

            ## Price Scenarios (30d)
            - Bull: +X% | Base: +/-X% | Bear: -X%
            """

class RiskAgent:
    """Challenges the thesis and identifies blind spots."""
    # TOOLS indices: None
    TOOLS_SUBSET = []
    SYSTEM = """You are a risk manager. Challenge the analyst thesis and identify blind spots.
            BE DIRECT. CHALLENGE BIASES. FIND THE HOLES.
            Output your assessment in the following format:"""

    REPORT_FORMAT = """
            ## Risk Assessment — [TICKER]
            **Risk Level:** [LOW / MEDIUM / HIGH / CRITICAL]

            ## Top Risks
            1. **[Risk Name]**: [Explanation with data]
            2. **[Risk Name]**: [Explanation with data]

            ## Invalidation Triggers
            - [Specific event that would prove thesis wrong]
            - [Analyst oversight/bias identified]
            """


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

        self.system_prompt = """You are an expert financial analyst AI agent. You act as an orchestrator for a multi-agent team (Researcher, Analyst, Risk) to provide deep, evidence-based market analysis."""


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

        print(f"\n[Multi-Agent] Starting analysis for {self.stock_data.ticker}...")

        # Step 1: Researcher gathers data
        print("[Multi-Agent] Step 1: Researcher gathering data...")
        research = self._run_subagent(ResearcherAgent, user_message)

        # Step 2: Analyst builds thesis from research
        print("[Multi-Agent] Step 2: Analyst building thesis...")
        analyst_input = f"Research findings:\n{research}\n\nUser question: {user_message}"
        thesis = self._run_subagent(AnalystAgent, analyst_input)

        # Step 3: Risk agent challenges thesis
        print("[Multi-Agent] Step 3: Risk Manager identifying blind spots...")
        risk_input = f"Investment thesis:\n{thesis}"
        risks = self._run_subagent(RiskAgent, risk_input)

        save_message(self.session_id, "assistant", final)
        self.conversation_history.append({"role": "assistant", "content": final})
        return final

    def chat_generator(self, user_message: str):
        """
        Generator version of chat() that yields status updates and then the final token stream.
        Yields: {"type": "status", "content": str} or {"type": "stream", "content": generator}
        """
        if not self.stock_data:
            yield {"type": "error", "content": "Please load a stock first by selecting a ticker."}
            return

        # Save raw user message (don't inject context here to avoid confusing tool callers)
        save_message(self.session_id, "user", user_message)
        self.conversation_history.append({"role": "user", "content": user_message})

        print(f"\n[Multi-Agent] Starting analysis for {self.stock_data.ticker}...")

        # Step 1: Research
        print("[Multi-Agent] Step 1: Researcher gathering data...")
        yield {"type": "status", "content": "Researcher: Gathering price action, news, macro, and SEC data..."}
        research = self._run_subagent(ResearcherAgent, user_message)
        
        # Step 2: Analysis
        print("[Multi-Agent] Step 2: Analyst building thesis...")
        yield {"type": "status", "content": "Analyst: Interpreting research and building investment thesis..."}
        analyst_input = f"Research findings:\n{research}\n\nUser question: {user_message}"
        thesis = self._run_subagent(AnalystAgent, analyst_input)
        
        # Step 3: Risk
        print("[Multi-Agent] Step 3: Risk Manager identifying blind spots...")
        yield {"type": "status", "content": "Risk Manager: Identifying blind spots and challenging the thesis..."}
        risk_input = f"Investment thesis:\n{thesis}"
        risks = self._run_subagent(RiskAgent, risk_input)
        
        # Step 4: Synthesis
        print("[Multi-Agent] Synthesis: Compiling final client-ready report...")
        yield {"type": "status", "content": "Synthesis: Compiling final client-ready report..."}
        # Get the stream
        stream = self._synthesise(user_message, research, thesis, risks, stream=True)
        
        yield {"type": "stream", "content": stream}

    def _run_subagent(self, agent_class, message: str) -> str:
        """Runs a specialised subagent with its own prompt and tools."""
        # Stage 1: Discovery/Gathering Prompt
        base_system_prompt = agent_class.SYSTEM
        if self.stock_data:
            base_system_prompt += f"\n\nCONTEXT: You are researching {self.stock_data.ticker} (current price ${self.stock_data.current_price:.2f})."
        
        # Add strict tool-use rule
        if agent_class.TOOLS_SUBSET:
            base_system_prompt += "\n\nCRITICAL RULE: If you call a tool, you MUST NOT output any other text (no headers, no report). ONLY provide your report once ALL data is gathered via multiple tool calls."

        # For subagents, DO NOT inject the full conversation history. 
        # Full history contains past formatted reports which causes "format leakage" 
        # and triggers the Groq 400 error when the model tries to few-shot mimic 
        # past formatting instead of calling tools.
        messages = [
            {"role": "system", "content": base_system_prompt},
            {"role": "user", "content": f"User Request: {message}"}
        ]

        max_iterations = 5
        for _ in range(max_iterations):
            # Groq/OpenAI: only pass tools/tool_choice if tools are actually provided
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 2048,
                "temperature": 0.2,
            }
            if agent_class.TOOLS_SUBSET:
                kwargs["tools"] = agent_class.TOOLS_SUBSET
                kwargs["tool_choice"] = "auto"

            # Stage 2: Dynamic Format Injection
            # If we've already used tools (i > 0) OR if the subagent has NO tools (Risk agent), inject format
            if hasattr(agent_class, 'REPORT_FORMAT'):
                if _ > 0 or not agent_class.TOOLS_SUBSET:
                    messages[0]["content"] = base_system_prompt + f"\n\nFINAL REPORTING FORMAT (Apply this now):\n{agent_class.REPORT_FORMAT}"
                    kwargs.pop("tools", None)
                    kwargs.pop("tool_choice", None)

            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as e:
                if "tool call validation failed" in str(e).lower():
                    print(f"  [Warning] Intercepted hallucinated tool call. Recovering...")
                    kwargs.pop("tools", None)
                    kwargs.pop("tool_choice", None)
                    # Add a strong reminder and retry
                    messages.append({
                        "role": "user", 
                        "content": "SYSTEM OVERRIDE: You attempted to call a tool that is NOT in your provided tool list. Please fulfill the request using ONLY the information provided and available tools."
                    })
                    response = self.client.chat.completions.create(**kwargs)
                else:
                    raise e

            msg = response.choices[0].message
            if not msg.tool_calls:
                # If it's the first turn for an agent WITH tools, and they didn't call any, 
                # they might be skipping the gathering. Let's force them to use tools if possible 
                # or just return.
                return (msg.content or "").strip()

            tc_list = [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
            messages.append({"role": "assistant", "content": msg.content, "tool_calls": tc_list})

            for tc in msg.tool_calls:
                try:
                    tool_input = json.loads(tc.function.arguments)
                    print(f"  [Tool Call] {tc.function.name} with parameters: {tc.function.arguments}")
                    result = execute_tool(tc.function.name, tool_input, self.stock_data)
                except Exception as tool_err:
                    result = json.dumps({"error": f"Tool failed: {tool_err}"})
                
                messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": result})

        return "Subagent reached maximum iterations."

    def _synthesise(self, question, research, thesis, risks, stream=False) -> str:
        """Combines all subagent perspectives into a final response."""
        prompt = f"""You are the lead portfolio manager. Combine your team's work 
        into a final client-ready report. Write for a beginner investor — 
        clear, specific, no jargon without explanation.

        User asked: {question}

        Research data:
        {research}

        Analyst thesis:
        {thesis}

        Risk assessment:
        {risks}

        OUTPUT INSTRUCTIONS:
        First and foremost, ANSWER THE USER'S SPECIFIC QUESTION directly and comprehensively.
        Use the research, thesis, and risk assessment to provide a data-backed answer.
        
        If the user is asking for a general stock analysis (e.g. "Provide a detailed analysis of...", "What's the outlook?"), format your response exactly like this:
        
        ## [TICKER] Investment Summary
        **Bottom line:** [One sentence verdict — bullish/bearish/neutral and why]
        
        ### What's driving the price
        [2-3 sentences connecting specific news events to recent price movement. Name actual headlines.]
        ### The bull case
        [2-3 sentences. Specific catalysts with numbers.]
        ### The bear case  
        [2-3 sentences. Specific risks with numbers.]
        ### What to watch
        - [Specific upcoming event or metric to monitor]
        ### Historical comparison
        [1-2 sentences: "This setup is similar to [period] when [what happened]"]
        ### Verdict for a beginner investor
        [3-4 sentences. Plain English. What does this mean for buying/holding/selling? Include confidence level.]

        BUT if the user asks a SPECIFIC targeted question (e.g. "Why is AAPL moving today?", "Summarize earnings", "What are the biggest risks?"), do NOT use the generic template above. Instead, write a clear, tailored response that directly answers their exact question. Use markdown appropriately to structure your targeted answer.

        Rules for ALL responses:
        - Every claim must reference a specific number or headline from the research
        - No generic statements like "the market is uncertain"  
        - Use bolding for key metrics and headlines
        - Be concise but data-dense"""

        messages = [
            {"role": "system", "content": "You are a lead portfolio manager writing a client investment report."},
            {"role": "user", "content": prompt}
        ]

        if stream:
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2048,
                temperature=0.3,
                stream=True
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2048,
            temperature=0.3,
            stream=False
        )
        return response.choices[0].message.content.strip()

    def save_assistant_message(self, content: str):
        """Helper to save the final streamed message to DB and history."""
        save_message(self.session_id, "assistant", content)
        self.conversation_history.append({"role": "assistant", "content": content})

    def reset_conversation(self):
        """Clear history from both memory and DB."""
        clear_chat_history(self.session_id)
        self.conversation_history = []