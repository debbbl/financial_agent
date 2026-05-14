import pytest
import pandas as pd

from tools.market_data import StockData, NewsEvent


@pytest.fixture
def sample_news():
    return [
        NewsEvent(
            date="2024-01-15",
            title="Apple beats earnings estimates",
            source="Reuters",
            category="earnings",
            sentiment="bullish",
            sentiment_score=0.72,
            impact="high",
            url="",
            summary="",
        ),
        NewsEvent(
            date="2024-01-20",
            title="Apple faces antitrust probe",
            source="Bloomberg",
            category="policy",
            sentiment="bearish",
            sentiment_score=-0.45,
            impact="medium",
            url="",
            summary="",
        ),
        NewsEvent(
            date="2024-01-25",
            title="iPhone sales stable",
            source="CNBC",
            category="product",
            sentiment="neutral",
            sentiment_score=0.05,
            impact="low",
            url="",
            summary="",
        ),
    ]


@pytest.fixture
def sample_prices():
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    return pd.DataFrame(
        {
            "Open": [180.0 + i * 0.5 for i in range(30)],
            "High": [182.0 + i * 0.5 for i in range(30)],
            "Low": [179.0 + i * 0.5 for i in range(30)],
            "Close": [181.0 + i * 0.5 for i in range(30)],
            "Volume": [50_000_000] * 30,
        },
        index=dates,
    )


@pytest.fixture
def sample_stock_data(sample_prices, sample_news):
    return StockData(
        ticker="AAPL",
        prices=sample_prices,
        news=sample_news,
        current_price=195.5,
        price_change_pct=0.82,
    )
