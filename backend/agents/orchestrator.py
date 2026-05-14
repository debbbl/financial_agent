"""
agents/orchestrator.py — Async version of FinancialAgent for FastAPI/SSE
Preserves all logic from financial_agent.py; adds async + event queue emission.
"""

import json
import asyncio
from typing import AsyncIterator, Optional

from groq import AsyncGroq

from tools.market_data import StockData, get_range_news, compute_overall_sentiment
from db.db import save_message, load_chat_history, find_similar_patterns_sql, semantic_pattern_search

# ── TOOLS (exact copy from financial_agent.py — do not modify) ─────────────
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
                    "end_date": {"type": "string", "description": "End date YYYY-MM-DD"},
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
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
                    "ticker": {"type": "string"},
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
                    "ticker": {"type": "string"},
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
                    "end_date": {"type": "string", "description": "End date YYYY-MM-DD"},
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
        mask = (prices.index >= start) & (prices.index <= end)
        rng = prices.loc[mask]
        if rng.empty:
            return json.dumps({"error": "No price data found for that date range."})
        p_start = float(rng["Close"].iloc[0])
        p_end = float(rng["Close"].iloc[-1])
        pct = ((p_end - p_start) / p_start) * 100
        rng_news = get_range_news(stock_data.news, start, end)
        return json.dumps(
            {
                "ticker": tool_input["ticker"],
                "start_date": start,
                "end_date": end,
                "price_start": round(p_start, 2),
                "price_end": round(p_end, 2),
                "pct_change": round(pct, 2),
                "period_high": round(float(rng["High"].max()), 2),
                "period_low": round(float(rng["Low"].min()), 2),
                "avg_daily_volume": int(rng["Volume"].mean()),
                "news_events": [
                    {"date": n.date, "title": n.title, "sentiment": n.sentiment, "category": n.category, "score": n.sentiment_score}
                    for n in rng_news
                ],
                "bullish_count": sum(1 for n in rng_news if n.sentiment == "bullish"),
                "bearish_count": sum(1 for n in rng_news if n.sentiment == "bearish"),
            }
        )

    elif tool_name == "forecast_trend":
        news = stock_data.news
        overall = compute_overall_sentiment(news)
        prices = stock_data.prices["Close"].values
        momentum = (prices[-1] - prices[-min(10, len(prices))]) / prices[-min(10, len(prices))]
        raw = (overall * 0.6) + (momentum * 0.4 * 5)
        bull_prob = min(95, max(5, 50 + raw * 40))
        bear_prob = 100 - bull_prob
        high = [n for n in news if n.impact == "high"]
        return json.dumps(
            {
                "ticker": tool_input["ticker"],
                "horizon": tool_input["horizon"],
                "bullish_probability": round(bull_prob, 1),
                "bearish_probability": round(bear_prob, 1),
                "verdict": "Bullish" if bull_prob > 55 else "Bearish" if bear_prob > 55 else "Neutral",
                "overall_sentiment_score": overall,
                "momentum_signal": "Positive" if momentum > 0 else "Negative",
                "key_risk_events": [n.title for n in high if n.sentiment == "bearish"][:3],
                "key_catalysts": [n.title for n in high if n.sentiment == "bullish"][:3],
            }
        )

    elif tool_name == "find_similar_periods":
        ticker = tool_input["ticker"]
        score = float(tool_input["current_sentiment_score"])
        query = tool_input.get("query_description", f"{ticker} sentiment {score:+.2f}")

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

        return json.dumps(
            {
                "ticker": ticker,
                "current_sentiment_score": score,
                "search_method": "semantic" if patterns and "similarity" in patterns[0] else "sql",
                "similar_periods": patterns,
                "note": "Past performance does not guarantee future results.",
            }
        )

    elif tool_name == "summarize_news_category":
        category = tool_input["category"]
        cat_news = [n for n in stock_data.news if n.category == category]
        if not cat_news:
            return json.dumps({"message": f"No {category} news found for {tool_input['ticker']}"})
        avg_score = sum(n.sentiment_score for n in cat_news) / len(cat_news)
        return json.dumps(
            {
                "ticker": tool_input["ticker"],
                "category": category,
                "total_events": len(cat_news),
                "avg_sentiment_score": round(avg_score, 2),
                "overall_tone": "Bullish" if avg_score > 0.1 else "Bearish" if avg_score < -0.1 else "Neutral",
                "events": [{"date": n.date, "title": n.title, "sentiment": n.sentiment, "source": n.source} for n in cat_news],
            }
        )

    elif tool_name == "get_macro_context":
        from tools.market_data import fetch_macro_context

        result = fetch_macro_context(
            tool_input["start_date"],
            tool_input["end_date"],
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


class AgentOrchestrator:
    """
    Async version of FinancialAgent.
    Emits SSE events via asyncio.Queue as each stage completes.
    Preserves all logic from the original _run_subagent() and _synthesise().
    """

    def __init__(self, api_key: str, session_id: str, model: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        self.client = AsyncGroq(api_key=api_key)
        self.model = model
        self.session_id = session_id
        self.stock_data: Optional[StockData] = None
        self.conversation_history: list[dict] = []

    async def load_history(self):
        """Load persisted chat history from DB (call after __init__)."""
        self.conversation_history = await asyncio.to_thread(load_chat_history, self.session_id)

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

    async def run(
        self,
        user_message: str,
        event_queue: asyncio.Queue,
        chart_context: dict | None = None,
    ) -> str:
        """
        Main entry point. Mirrors chat_generator() logic but async + queue-based.

        Steps (same as original chat_generator, plus debate round):
        1. Save user message
        2. Researcher stage → emit stage_start, run, emit stage_complete
        3. Analyst stage → same
        4. Risk stage → same
        5. Debate round → emit debate_turn ×3 (analyst t1, risk t1, analyst t2)
        6. Synthesis → emit tokens via synthesis_token events
        7. Save assistant message, emit done
        """
        if not self.stock_data:
            await event_queue.put({"type": "error", "message": "No stock data loaded."})
            return ""

        await asyncio.to_thread(save_message, self.session_id, "user", user_message)
        self.conversation_history.append({"role": "user", "content": user_message})

        full_response = ""
        try:
            print(f"\n[Multi-Agent] Starting analysis for {self.stock_data.ticker}...")

            await event_queue.put({"type": "stage_start", "stage": "researcher"})
            print("[Multi-Agent] Step 1: Researcher gathering data...")
            researcher_message = user_message
            if chart_context:
                top_lines = "\n".join(
                    f"  - {n.get('date')}: {n.get('title')} [{n.get('sentiment')}]"
                    for n in chart_context.get("top_news") or []
                )
                researcher_message = (
                    "[Chart Analysis Context]\n"
                    f"Ticker: {chart_context['ticker']}\n"
                    f"Range: {chart_context['from_date']} → {chart_context['to_date']}\n"
                    f"Price change: {chart_context['price_change_pct']:+.2f}%\n"
                    f"Open: ${chart_context['open_price']} | Close: ${chart_context['close_price']} | "
                    f"High: ${chart_context['high']} | Low: ${chart_context['low']}\n"
                    f"News events in range: {chart_context['news_count']}\n"
                    + "Key events:\n"
                    + top_lines
                    + "\n\n"
                    + user_message
                )
            research = await self._run_subagent_async(ResearcherAgent, researcher_message)
            await event_queue.put({"type": "stage_complete", "stage": "researcher", "summary": research[:500]})

            await event_queue.put({"type": "stage_start", "stage": "analyst"})
            print("[Multi-Agent] Step 2: Analyst building thesis...")
            analyst_input = f"Research findings:\n{research}\n\nUser question: {user_message}"
            thesis = await self._run_subagent_async(AnalystAgent, analyst_input)
            await event_queue.put({"type": "stage_complete", "stage": "analyst", "summary": thesis[:500]})

            await event_queue.put({"type": "stage_start", "stage": "risk"})
            print("[Multi-Agent] Step 3: Risk Manager identifying blind spots...")
            risk_input = f"Investment thesis:\n{thesis}"
            risks = await self._run_subagent_async(RiskAgent, risk_input)
            await event_queue.put({"type": "stage_complete", "stage": "risk", "summary": risks[:500]})

            # ── Debate round: Analyst defends thesis against Risk critique ──────
            # Re-surface the original thesis & risk assessment as turns 1, then
            # make one additional Groq call for the analyst's rebuttal (turn 2).
            print("[Multi-Agent] Step 3.5: Analyst ↔ Risk debate round...")
            await event_queue.put(
                {"type": "debate_turn", "speaker": "analyst", "content": thesis, "turn": 1}
            )
            await event_queue.put(
                {"type": "debate_turn", "speaker": "risk", "content": risks, "turn": 1}
            )

            analyst_rebuttal = ""
            try:
                analyst_rebuttal = await self._analyst_rebuttal_async(thesis, risks)
            except Exception as debate_err:
                print(f"  [Warning] Analyst rebuttal failed: {debate_err}")

            if analyst_rebuttal:
                await event_queue.put(
                    {
                        "type": "debate_turn",
                        "speaker": "analyst",
                        "content": analyst_rebuttal,
                        "turn": 2,
                    }
                )

            await event_queue.put({"type": "stage_start", "stage": "synthesis"})
            print("[Multi-Agent] Synthesis: Compiling final client-ready report...")
            async for token in self._synthesise_async(
                user_message, research, thesis, risks, analyst_rebuttal
            ):
                full_response += token
                await event_queue.put({"type": "synthesis_token", "token": token})

            await asyncio.to_thread(save_message, self.session_id, "assistant", full_response)
            self.conversation_history.append({"role": "assistant", "content": full_response})
            await event_queue.put({"type": "done", "session_id": self.session_id})

        except Exception as e:
            await event_queue.put({"type": "error", "message": str(e)})

        return full_response

    async def _run_subagent_async(self, agent_class, message: str) -> str:
        """
        Async version of _run_subagent().
        Preserves ALL logic from the original:
        - Dynamic format injection (inject REPORT_FORMAT after first tool use)
        - Hallucinated tool call recovery (catch "tool call validation failed")
        - max_iterations = 5
        - CRITICAL RULE in system prompt for tool use
        - No conversation history injection (prevents format leakage)

        Only difference: self.client.chat.completions.create() is now awaited.
        execute_tool() remains sync (wrapped in asyncio.to_thread).
        """
        base_system_prompt = agent_class.SYSTEM
        if self.stock_data:
            base_system_prompt += f"\n\nCONTEXT: You are researching {self.stock_data.ticker} (current price ${self.stock_data.current_price:.2f})."

        if agent_class.TOOLS_SUBSET:
            base_system_prompt += "\n\nCRITICAL RULE: If you call a tool, you MUST NOT output any other text (no headers, no report). ONLY provide your report once ALL data is gathered via multiple tool calls."

        messages = [
            {"role": "system", "content": base_system_prompt},
            {"role": "user", "content": f"User Request: {message}"},
        ]

        max_iterations = 5
        for i in range(max_iterations):
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 2048,
                "temperature": 0.2,
            }
            if agent_class.TOOLS_SUBSET:
                kwargs["tools"] = agent_class.TOOLS_SUBSET
                kwargs["tool_choice"] = "auto"

            if hasattr(agent_class, "REPORT_FORMAT"):
                if i > 0 or not agent_class.TOOLS_SUBSET:
                    messages[0]["content"] = base_system_prompt + f"\n\nFINAL REPORTING FORMAT (Apply this now):\n{agent_class.REPORT_FORMAT}"
                    kwargs.pop("tools", None)
                    kwargs.pop("tool_choice", None)

            try:
                response = await self.client.chat.completions.create(**kwargs)
            except Exception as e:
                if "tool call validation failed" in str(e).lower():
                    print("  [Warning] Intercepted hallucinated tool call. Recovering...")
                    kwargs.pop("tools", None)
                    kwargs.pop("tool_choice", None)
                    messages.append(
                        {
                            "role": "user",
                            "content": "SYSTEM OVERRIDE: You attempted to call a tool that is NOT in your provided tool list. Please fulfill the request using ONLY the information provided and available tools.",
                        }
                    )
                    response = await self.client.chat.completions.create(**kwargs)
                else:
                    raise

            msg = response.choices[0].message
            if not msg.tool_calls:
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
                    result = await asyncio.to_thread(execute_tool, tc.function.name, tool_input, self.stock_data)
                except Exception as tool_err:
                    result = json.dumps({"error": f"Tool failed: {tool_err}"})

                messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": result})

        return "Subagent reached maximum iterations."

    async def _analyst_rebuttal_async(self, thesis: str, risks: str) -> str:
        """
        One additional Groq call: the Analyst defends their thesis after seeing
        the Risk Manager's critique. Kept short and conversational on purpose
        to bound token / cost spend (single round, no further back-and-forth).
        """
        system_prompt = (
            "You are the same senior financial analyst who wrote the thesis above. "
            "The Risk Manager has just challenged your view. Defend your thesis "
            "directly and concisely.\n"
            "- Acknowledge any genuinely strong risk points (one sentence each).\n"
            "- Push back firmly where you have data to support your view.\n"
            "- Cite specific numbers or headlines from the research where you can.\n"
            "- Be conversational — this is a debate, not a report. "
            "No markdown headers, no bullet lists.\n"
            "- 4-6 sentences max."
        )
        if self.stock_data:
            system_prompt += (
                f"\n\nCONTEXT: {self.stock_data.ticker} @ "
                f"${self.stock_data.current_price:.2f}."
            )

        user_prompt = (
            f"Your earlier investment thesis:\n{thesis}\n\n"
            f"Risk Manager's critique:\n{risks}\n\n"
            "Write your rebuttal now."
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=512,
            temperature=0.4,
        )
        return (response.choices[0].message.content or "").strip()

    async def _synthesise_async(
        self,
        question: str,
        research: str,
        thesis: str,
        risks: str,
        analyst_rebuttal: str = "",
    ) -> AsyncIterator[str]:
        """
        Async streaming version of _synthesise().
        Uses the EXACT same prompt from financial_agent.py _synthesise(),
        with an optional Analyst↔Risk debate context appended.
        Yields string tokens.
        """
        debate_block = (
            f"\n\n        Analyst rebuttal to risk critique:\n        {analyst_rebuttal}"
            if analyst_rebuttal
            else ""
        )

        prompt = f"""You are the lead portfolio manager. Combine your team's work 
        into a final client-ready report. Write for a beginner investor — 
        clear, specific, no jargon without explanation.

        User asked: {question}

        Research data:
        {research}

        Analyst thesis:
        {thesis}

        Risk assessment:
        {risks}{debate_block}

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
        - Be concise but data-dense
        - If an Analyst rebuttal is provided, weigh both the original risk critique and the analyst's counter-arguments when forming the final verdict (do not simply parrot one side)."""

        messages = [
            {"role": "system", "content": "You are a lead portfolio manager writing a client investment report."},
            {"role": "user", "content": prompt},
        ]

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2048,
            temperature=0.3,
            stream=True,
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token
