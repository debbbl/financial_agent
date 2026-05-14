"""Test execute_tool() with the sample StockData fixture."""

import json

import pytest

from agents.orchestrator import execute_tool


def test_analyze_price_range(sample_stock_data):
    result_json = execute_tool(
        "analyze_price_range",
        {"start_date": "2024-01-10", "end_date": "2024-01-25", "ticker": "AAPL"},
        sample_stock_data,
    )
    result = json.loads(result_json)
    assert "pct_change" in result
    assert "news_events" in result
    assert result["ticker"] == "AAPL"


def test_analyze_price_range_no_data(sample_stock_data):
    result_json = execute_tool(
        "analyze_price_range",
        {"start_date": "2020-01-01", "end_date": "2020-01-31", "ticker": "AAPL"},
        sample_stock_data,
    )
    result = json.loads(result_json)
    assert "error" in result


def test_forecast_trend(sample_stock_data):
    result_json = execute_tool(
        "forecast_trend",
        {"ticker": "AAPL", "horizon": "7d"},
        sample_stock_data,
    )
    result = json.loads(result_json)
    assert "bullish_probability" in result
    assert 0 <= result["bullish_probability"] <= 100
    assert result["verdict"] in ("Bullish", "Bearish", "Neutral")


def test_summarize_news_category(sample_stock_data):
    result_json = execute_tool(
        "summarize_news_category",
        {"ticker": "AAPL", "category": "earnings"},
        sample_stock_data,
    )
    result = json.loads(result_json)
    assert result["total_events"] == 1
    assert result["overall_tone"] == "Bullish"


def test_unknown_tool(sample_stock_data):
    result_json = execute_tool("nonexistent_tool", {}, sample_stock_data)
    result = json.loads(result_json)
    assert "error" in result
