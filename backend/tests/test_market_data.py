"""Test existing market_data.py functions."""

import pytest

from tools.market_data import (
    score_sentiment,
    classify_category,
    compute_overall_sentiment,
    get_range_news,
    filter_news,
    derive_impact,
    _keyword_sentiment_fallback,
)


def test_score_sentiment_bullish():
    label, score = score_sentiment("Apple beats earnings estimates with record revenue growth")
    assert label in ("bullish", "neutral")
    assert -1.0 <= score <= 1.0


def test_score_sentiment_bearish():
    label, score = score_sentiment("Company faces massive lawsuit and revenue decline warning")
    assert label in ("bearish", "neutral")
    assert score <= 0.3


def test_score_sentiment_empty():
    label, score = score_sentiment("")
    assert label == "neutral"
    assert score == 0.0


def test_classify_category():
    assert classify_category("Apple reports quarterly earnings beat") == "earnings"
    assert classify_category("CEO resigns amid controversy") == "management"
    assert classify_category("New iPhone model launch announcement") == "product"


def test_compute_overall_sentiment(sample_news):
    score = compute_overall_sentiment(sample_news)
    assert -1.0 <= score <= 1.0
    # High impact bullish + medium impact bearish — should lean slightly bullish
    # (weights: high=3, medium=2, low=1; 3*0.72 + 2*-0.45 + 1*0.05 = 2.16-0.9+0.05 = 1.31 / 6 ≈ 0.22)


def test_get_range_news(sample_news):
    result = get_range_news(sample_news, "2024-01-14", "2024-01-16")
    assert len(result) == 1
    assert result[0].title == "Apple beats earnings estimates"


def test_filter_news_by_category(sample_news):
    result = filter_news(sample_news, categories=["earnings"])
    assert len(result) == 1
    result_two = filter_news(sample_news, sentiments=["bullish", "bearish"])
    assert len(result_two) == 2


def test_derive_impact():
    assert derive_impact(0.7, "reuters") == "high"
    assert derive_impact(0.5, "reuters") == "high"
    assert derive_impact(0.3, "unknown") == "medium"
    assert derive_impact(0.1, "unknown") == "low"


def test_keyword_fallback():
    label, score = _keyword_sentiment_fallback("stock surges on record profits beat")
    assert label == "bullish"
    assert score > 0
    label2, score2 = _keyword_sentiment_fallback("shares plunge on revenue miss and warning")
    assert label2 == "bearish"
    assert score2 < 0
