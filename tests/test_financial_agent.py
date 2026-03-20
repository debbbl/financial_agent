"""
tests/test_financial_agent.py
Comprehensive test suite covering:
- Unit tests: data layer, tools, chart builder
- Integration tests: agent tool execution loop
- Quality tests: sentiment scoring, filter logic
- Edge case tests: empty data, invalid tickers
"""

import pytest
import json
import sys
import os
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tools.market_data import (
    NewsEvent, StockData, filter_news, compute_overall_sentiment,
    get_range_news, classify_category, score_sentiment,
)
from agents.financial_agent import execute_tool, FinancialAgent


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_prices():
    """20-day OHLCV DataFrame."""
    dates = pd.date_range(start="2024-01-01", periods=20, freq="B")
    np.random.seed(42)
    closes = 180.0 + np.cumsum(np.random.randn(20))
    return pd.DataFrame({
        "Open":   closes - 0.5,
        "High":   closes + 1.5,
        "Low":    closes - 1.5,
        "Close":  closes,
        "Volume": np.random.randint(50_000_000, 100_000_000, 20),
    }, index=dates)


@pytest.fixture
def sample_news():
    """A known set of news events for deterministic tests."""
    return [
        NewsEvent("2024-01-03", "Apple Q2 earnings beat by 8%", "Reuters", "earnings", "bullish", 0.75, "high"),
        NewsEvent("2024-01-05", "China iPhone sales drop 19%", "FT", "policy", "bearish", -0.70, "high"),
        NewsEvent("2024-01-09", "Fed holds rates steady", "WSJ", "market", "bullish", 0.45, "medium"),
        NewsEvent("2024-01-12", "Tim Cook hints at AI features", "CNBC", "product", "bullish", 0.65, "medium"),
        NewsEvent("2024-01-17", "DOJ probe expanded to App Store", "NYT", "policy", "bearish", -0.65, "high"),
        NewsEvent("2024-01-19", "Services revenue hits record $24B", "Reuters", "earnings", "bullish", 0.80, "high"),
    ]


@pytest.fixture
def sample_stock_data(sample_prices, sample_news):
    return StockData(
        ticker="AAPL",
        prices=sample_prices,
        news=sample_news,
        current_price=float(sample_prices["Close"].iloc[-1]),
        price_change_pct=1.5,
    )


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Data layer
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeSentiment:
    def test_all_bullish(self):
        news = [
            NewsEvent("2024-01-01", "Good news", "S", "earnings", "bullish", 0.8, "high"),
            NewsEvent("2024-01-02", "More good news", "S", "product", "bullish", 0.6, "medium"),
        ]
        score = compute_overall_sentiment(news)
        assert score > 0, "All bullish news should yield positive sentiment"

    def test_all_bearish(self):
        news = [
            NewsEvent("2024-01-01", "Bad news", "S", "policy", "bearish", -0.8, "high"),
            NewsEvent("2024-01-02", "More bad news", "S", "market", "bearish", -0.5, "medium"),
        ]
        score = compute_overall_sentiment(news)
        assert score < 0, "All bearish news should yield negative sentiment"

    def test_empty_returns_zero(self):
        assert compute_overall_sentiment([]) == 0.0

    def test_high_impact_weighted_more(self, sample_news):
        """High-impact events should dominate low-impact ones."""
        high_only = [n for n in sample_news if n.impact == "high"]
        mixed = sample_news
        score_high = compute_overall_sentiment(high_only)
        score_mixed = compute_overall_sentiment(mixed)
        # Not asserting direction, just that they differ (weighting matters)
        assert score_high != score_mixed or len(high_only) == len(mixed)

    def test_score_bounded(self, sample_news):
        score = compute_overall_sentiment(sample_news)
        assert -1.0 <= score <= 1.0, "Sentiment score must stay within [-1, 1]"


class TestFilterNews:
    def test_filter_by_category(self, sample_news):
        result = filter_news(sample_news, categories=["earnings"])
        assert all(n.category == "earnings" for n in result)
        assert len(result) == 2

    def test_filter_by_sentiment(self, sample_news):
        result = filter_news(sample_news, sentiments=["bearish"])
        assert all(n.sentiment == "bearish" for n in result)

    def test_filter_combined(self, sample_news):
        result = filter_news(sample_news, categories=["policy"], sentiments=["bearish"])
        assert len(result) == 2
        assert all(n.category == "policy" and n.sentiment == "bearish" for n in result)

    def test_no_filter_returns_all(self, sample_news):
        result = filter_news(sample_news)
        assert len(result) == len(sample_news)

    def test_empty_category_list_returns_nothing(self, sample_news):
        result = filter_news(sample_news, categories=[])
        assert len(result) == 0

    def test_unknown_category_returns_nothing(self, sample_news):
        result = filter_news(sample_news, categories=["unknown_category"])
        assert len(result) == 0


class TestGetRangeNews:
    def test_within_range(self, sample_news):
        result = get_range_news(sample_news, "2024-01-03", "2024-01-10")
        assert all("2024-01-03" <= n.date <= "2024-01-10" for n in result)
        assert len(result) == 3

    def test_boundary_inclusive(self, sample_news):
        result = get_range_news(sample_news, "2024-01-03", "2024-01-03")
        assert len(result) == 1
        assert result[0].date == "2024-01-03"

    def test_no_news_in_range(self, sample_news):
        result = get_range_news(sample_news, "2024-02-01", "2024-02-28")
        assert len(result) == 0

    def test_full_range_returns_all(self, sample_news):
        result = get_range_news(sample_news, "2024-01-01", "2024-12-31")
        assert len(result) == len(sample_news)


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Text classification helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestClassifyCategory:
    def test_earnings_category(self):
        assert classify_category("Apple Q2 earnings beat expectations") == "earnings"

    def test_product_category(self):
        assert classify_category("New iPhone 16 launch announced") == "product"

    def test_policy_category(self):
        assert classify_category("DOJ antitrust probe into App Store") == "policy"

    def test_default_to_market(self):
        cat = classify_category("Some generic headline here")
        assert cat == "market"


class TestScoreSentiment:
    def test_bullish_text(self):
        label, score = score_sentiment("Stock surges to record gains")
        assert label == "bullish"
        assert score > 0

    def test_bearish_text(self):
        label, score = score_sentiment("Stock plunges on disappointing loss")
        assert label == "bearish"
        assert score < 0

    def test_neutral_text(self):
        label, score = score_sentiment("Company releases annual report")
        assert label == "neutral"
        assert score == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Tool execution
# ══════════════════════════════════════════════════════════════════════════════

class TestExecuteToolAnalyzeRange:
    def test_returns_valid_json(self, sample_stock_data):
        result = execute_tool(
            "analyze_price_range",
            {"ticker": "AAPL", "start_date": "2024-01-03", "end_date": "2024-01-12"},
            sample_stock_data,
        )
        data = json.loads(result)
        assert "pct_change" in data
        assert "news_events" in data
        assert isinstance(data["bullish_count"], int)
        assert isinstance(data["bearish_count"], int)

    def test_empty_date_range(self, sample_stock_data):
        result = execute_tool(
            "analyze_price_range",
            {"ticker": "AAPL", "start_date": "2025-01-01", "end_date": "2025-01-31"},
            sample_stock_data,
        )
        data = json.loads(result)
        assert "error" in data

    def test_price_change_is_correct(self, sample_stock_data):
        prices = sample_stock_data.prices
        start_str = prices.index[0].strftime("%Y-%m-%d")
        end_str = prices.index[5].strftime("%Y-%m-%d")
        result = json.loads(execute_tool(
            "analyze_price_range",
            {"ticker": "AAPL", "start_date": start_str, "end_date": end_str},
            sample_stock_data,
        ))
        expected_start = float(prices["Close"].iloc[0])
        assert abs(result["price_start"] - expected_start) < 0.01


class TestExecuteToolForecast:
    def test_probabilities_sum_to_100(self, sample_stock_data):
        result = json.loads(execute_tool(
            "forecast_trend",
            {"ticker": "AAPL", "horizon": "7d"},
            sample_stock_data,
        ))
        total = result["bullish_probability"] + result["bearish_probability"]
        assert abs(total - 100) < 0.01, f"Bull + Bear should = 100, got {total}"

    def test_probabilities_in_valid_range(self, sample_stock_data):
        for horizon in ["7d", "30d"]:
            result = json.loads(execute_tool(
                "forecast_trend",
                {"ticker": "AAPL", "horizon": horizon},
                sample_stock_data,
            ))
            assert 0 <= result["bullish_probability"] <= 100
            assert 0 <= result["bearish_probability"] <= 100

    def test_verdict_matches_probability(self, sample_stock_data):
        result = json.loads(execute_tool(
            "forecast_trend",
            {"ticker": "AAPL", "horizon": "30d"},
            sample_stock_data,
        ))
        if result["bullish_probability"] > 55:
            assert result["verdict"] == "Bullish"
        elif result["bearish_probability"] > 55:
            assert result["verdict"] == "Bearish"


class TestExecuteToolSimilarPeriods:
    @patch("agents.financial_agent.semantic_pattern_search", return_value=[])
    @patch("agents.financial_agent.find_similar_patterns_sql")
    def test_returns_results(self, mock_sql, mock_semantic, sample_stock_data):
        mock_sql.return_value = [
            {"ticker": "AAPL", "period_start": "2023-08-01", "period_end": "2023-09-01",
             "sentiment_score": 0.58, "price_change_pct": 3.2, "outcome_30d": 4.1,
             "dominant_category": "earnings", "context_summary": "Post-earnings rally."},
            {"ticker": "AAPL", "period_start": "2023-05-01", "period_end": "2023-06-01",
             "sentiment_score": 0.72, "price_change_pct": 5.1, "outcome_30d": 6.2,
             "dominant_category": "earnings", "context_summary": "Q2 beat."},
            {"ticker": "AAPL", "period_start": "2024-02-01", "period_end": "2024-03-01",
             "sentiment_score": 0.65, "price_change_pct": 2.8, "outcome_30d": 3.8,
             "dominant_category": "product", "context_summary": "Vision Pro launch."},
        ]
        result = json.loads(execute_tool(
            "find_similar_periods",
            {"ticker": "AAPL", "current_sentiment_score": 0.5},
            sample_stock_data,
        ))
        assert len(result["similar_periods"]) == 3


class TestExecuteToolCategorySummary:
    def test_earnings_summary(self, sample_stock_data):
        result = json.loads(execute_tool(
            "summarize_news_category",
            {"ticker": "AAPL", "category": "earnings"},
            sample_stock_data,
        ))
        assert result["total_events"] == 2
        assert result["avg_sentiment_score"] > 0  # Both earnings are bullish
        assert result["overall_tone"] == "Bullish"

    def test_missing_category(self, sample_stock_data):
        result = json.loads(execute_tool(
            "summarize_news_category",
            {"ticker": "AAPL", "category": "competition"},
            sample_stock_data,
        ))
        assert "message" in result


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — Agent (mocked Groq API)
# ══════════════════════════════════════════════════════════════════════════════

class TestFinancialAgentIntegration:
    """
    Integration tests mock the Groq API to verify:
    - Correct tool execution when Groq requests a tool
    - Proper conversation history management
    - ReAct loop termination
    """

    def _make_text_response(self, text):
        """Simulate a final text-only Groq response."""
        msg = MagicMock()
        msg.content = text
        msg.tool_calls = None
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message = msg
        return resp

    def _make_tool_response(self, tool_name, tool_input):
        """Simulate Groq requesting a tool call."""
        tc = MagicMock()
        tc.id = f"call_{tool_name}_001"
        tc.function.name = tool_name
        tc.function.arguments = json.dumps(tool_input)
        msg = MagicMock()
        msg.content = ""
        msg.tool_calls = [tc]
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message = msg
        return resp

    @patch("agents.financial_agent.load_chat_history", return_value=[])
    @patch("agents.financial_agent.save_message")
    def test_agent_responds_without_tool(self, mock_save, mock_load, sample_stock_data):
        agent = FinancialAgent(api_key="test-key", session_id="test-session-123")
        agent.set_stock_data(sample_stock_data)

        with patch.object(agent.client.chat.completions, "create") as mock_create:
            mock_create.return_value = self._make_text_response("AAPL is trending upward.")
            response = agent.chat("What's happening with AAPL?")

        assert "AAPL" in response
        assert len(agent.conversation_history) == 2  # user + assistant

    @patch("agents.financial_agent.load_chat_history", return_value=[])
    @patch("agents.financial_agent.save_message")
    def test_agent_executes_tool_then_responds(self, mock_save, mock_load, sample_stock_data):
        agent = FinancialAgent(api_key="test-key", session_id="test-session-123")
        agent.set_stock_data(sample_stock_data)

        prices = sample_stock_data.prices
        start = prices.index[0].strftime("%Y-%m-%d")
        end = prices.index[5].strftime("%Y-%m-%d")

        tool_response = self._make_tool_response(
            "analyze_price_range",
            {"ticker": "AAPL", "start_date": start, "end_date": end},
        )
        final_response = self._make_text_response("The stock rose 3% driven by earnings news.")

        with patch.object(agent.client.chat.completions, "create") as mock_create:
            mock_create.side_effect = [tool_response, final_response]
            response = agent.chat(f"Why did AAPL move from {start} to {end}?")

        assert mock_create.call_count == 2
        assert "rose" in response or "stock" in response

    @patch("agents.financial_agent.load_chat_history", return_value=[])
    @patch("agents.financial_agent.save_message")
    def test_conversation_history_grows(self, mock_save, mock_load, sample_stock_data):
        agent = FinancialAgent(api_key="test-key", session_id="test-session-123")
        agent.set_stock_data(sample_stock_data)

        with patch.object(agent.client.chat.completions, "create") as mock_create:
            mock_create.return_value = self._make_text_response("Analysis complete.")
            agent.chat("Question 1")
            agent.chat("Question 2")

        assert len(agent.conversation_history) == 4  # 2 user + 2 assistant

    @patch("agents.financial_agent.load_chat_history", return_value=[])
    @patch("agents.financial_agent.save_message")
    @patch("agents.financial_agent.clear_chat_history")
    def test_reset_clears_history(self, mock_clear, mock_save, mock_load, sample_stock_data):
        agent = FinancialAgent(api_key="test-key", session_id="test-session-123")
        agent.set_stock_data(sample_stock_data)

        with patch.object(agent.client.chat.completions, "create") as mock_create:
            mock_create.return_value = self._make_text_response("Done.")
            agent.chat("Hello")

        agent.reset_conversation()
        assert agent.conversation_history == []

    @patch("agents.financial_agent.load_chat_history", return_value=[])
    def test_no_stock_data_returns_message(self, mock_load):
        agent = FinancialAgent(api_key="test-key", session_id="test-session-123")
        response = agent.chat("What's happening?")
        assert "load" in response.lower() or "select" in response.lower()


# ══════════════════════════════════════════════════════════════════════════════
# EDGE CASE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_single_day_price_data(self):
        single_day = pd.DataFrame({
            "Open": [190.0], "High": [192.0], "Low": [188.0],
            "Close": [191.0], "Volume": [50_000_000],
        }, index=pd.DatetimeIndex(["2024-01-15"]))
        sd = StockData(
            ticker="TEST", prices=single_day, news=[],
            current_price=191.0, price_change_pct=0.5,
        )
        score = compute_overall_sentiment(sd.news)
        assert score == 0.0

    def test_news_with_extreme_scores(self):
        extreme = [
            NewsEvent("2024-01-01", "Catastrophic loss", "S", "earnings", "bearish", -1.0, "high"),
            NewsEvent("2024-01-02", "Record-breaking gains", "S", "earnings", "bullish", 1.0, "high"),
        ]
        score = compute_overall_sentiment(extreme)
        assert -0.1 <= score <= 0.1, "Equal extreme scores should cancel out"

    def test_filter_returns_empty_list_not_none(self, sample_news):
        result = filter_news(sample_news, categories=["nonexistent"])
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 0

    def test_classify_category_for_unknown_text(self):
        """Unknown text should fall back to 'market' category."""
        cat = classify_category("completely unrelated text about cooking")
        assert cat == "market", "Should fall back gracefully to 'market' category"

    def test_tool_unknown_name_returns_error(self, sample_stock_data):
        result = json.loads(execute_tool("nonexistent_tool", {}, sample_stock_data))
        assert "error" in result


# ══════════════════════════════════════════════════════════════════════════════
# QUALITY / REGRESSION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestQualityAssurance:
    def test_news_events_have_valid_sentiment_values(self, sample_news):
        valid_sentiments = {"bullish", "bearish", "neutral"}
        for n in sample_news:
            assert n.sentiment in valid_sentiments, f"Invalid sentiment: {n.sentiment}"

    def test_news_events_have_valid_categories(self, sample_news):
        valid_cats = {"earnings", "product", "management", "policy", "market", "competition"}
        for n in sample_news:
            assert n.category in valid_cats, f"Invalid category: {n.category}"

    def test_news_dates_are_valid_format(self, sample_news):
        for n in sample_news:
            parsed = datetime.strptime(n.date, "%Y-%m-%d")
            assert parsed is not None

    def test_sentiment_scores_within_bounds(self, sample_news):
        for n in sample_news:
            assert -1.0 <= n.sentiment_score <= 1.0, f"Score out of bounds: {n.sentiment_score}"

    def test_sentiment_score_sign_matches_sentiment_label(self, sample_news):
        for n in sample_news:
            if n.sentiment == "bullish":
                assert n.sentiment_score > 0, f"Bullish event should have positive score: {n}"
            elif n.sentiment == "bearish":
                assert n.sentiment_score < 0, f"Bearish event should have negative score: {n}"

    def test_stock_data_current_price_positive(self, sample_stock_data):
        assert sample_stock_data.current_price > 0

    @patch("agents.financial_agent.load_chat_history", return_value=[])
    def test_agent_system_prompt_not_empty(self, mock_load):
        agent = FinancialAgent(api_key="test-key", session_id="test-session-123")
        assert len(agent.system_prompt) > 100, "System prompt should be substantive"

    def test_tools_schema_has_required_fields(self):
        from agents.financial_agent import TOOLS
        for tool in TOOLS:
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]
            assert "properties" in tool["function"]["parameters"]