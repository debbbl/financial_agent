"""Integration tests for FastAPI routes."""

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import patch

from tools.market_data import StockData, NewsEvent


@pytest.fixture(scope="module", autouse=True)
def _ensure_db_initialized():
    """httpx ASGITransport does not run FastAPI lifespan; create tables like production startup."""
    from db.db import startup

    startup()
    yield


@pytest.fixture
def mock_stock_data():
    prices = pd.DataFrame(
        {
            "Open": [180.0],
            "High": [182.0],
            "Low": [179.0],
            "Close": [181.0],
            "Volume": [50_000_000],
        },
        index=pd.DatetimeIndex(["2024-01-15"]),
    )
    news = [
        NewsEvent(
            date="2024-01-15",
            title="AAPL beats earnings",
            source="Reuters",
            category="earnings",
            sentiment="bullish",
            sentiment_score=0.72,
            impact="high",
            url="",
            summary="",
        )
    ]
    return StockData("AAPL", prices, news, 181.0, 0.55)


@pytest.mark.asyncio
async def test_health():
    from main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_create_and_list_sessions():
    from main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Create
        r = await ac.post("/api/v1/sessions", json={"ticker": "AAPL", "period": "3mo"})
        assert r.status_code == 200
        sid = r.json()["session_id"]
        assert len(sid) == 36
        # List
        r2 = await ac.get("/api/v1/sessions")
        assert r2.status_code == 200
        assert r2.json()["count"] >= 1


@pytest.mark.asyncio
async def test_portfolio_crud():
    from main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post("/api/v1/portfolio", json={"ticker": "NVDA", "notes": "AI play"})
        assert r.status_code == 200
        r2 = await ac.get("/api/v1/portfolio")
        assert any(i["ticker"] == "NVDA" for i in r2.json()["portfolio"])
        r3 = await ac.delete("/api/v1/portfolio/NVDA")
        assert r3.status_code == 200


@pytest.mark.asyncio
async def test_market_data_404(mock_stock_data):
    from main import app

    with patch("tools.market_data.fetch_stock_data", side_effect=ValueError("No data for ZZZZ")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.get("/api/v1/market/ZZZZ")
        assert r.status_code == 404
