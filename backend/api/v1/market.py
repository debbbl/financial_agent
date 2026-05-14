from fastapi import APIRouter, HTTPException, Query
import asyncio

import pandas as pd

router = APIRouter()


@router.get("/{ticker}")
async def get_market_data(
    ticker: str,
    period: str = Query(default="3mo", description="yfinance period or 14d for two-week window"),
    interval: str | None = Query(default=None, description="yfinance interval e.g. 1d, 1h, 5m, 1wk"),
    include_news: bool = Query(
        default=True,
        description="If false, returns OHLCV and prices only (skips news fetch and FinBERT).",
    ),
):
    """
    Load OHLCV + news for a ticker.
    Uses existing fetch_stock_data() from tools/market_data.py.
    Returns JSON-serializable version of StockData.
    """
    from tools.market_data import fetch_stock_data

    ticker = ticker.upper()

    try:
        # fetch_stock_data is sync (yfinance + FinBERT) — run in thread
        stock_data = await asyncio.to_thread(
            fetch_stock_data,
            ticker,
            period,
            None,
            None,
            interval,
            include_news=include_news,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data fetch failed: {e}")

    # Serialize StockData — prices DataFrame → list of dicts, NewsEvent → dict
    prices_list = []
    for date, row in stock_data.prices.iterrows():
        ts = pd.Timestamp(date)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        day_key = ts.strftime("%Y-%m-%d")
        prices_list.append(
            {
                "date": day_key,
                "t": int(ts.timestamp()),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
                "volume": int(row["Volume"]),
            }
        )

    news_list = [
        {
            "date": n.date,
            "title": n.title,
            "source": n.source,
            "category": n.category,
            "sentiment": n.sentiment,
            "sentiment_score": n.sentiment_score,
            "impact": n.impact,
            "url": n.url,
            "summary": n.summary,
            "image_url": n.image_url,
        }
        for n in stock_data.news
    ]

    return {
        "ticker": stock_data.ticker,
        "current_price": stock_data.current_price,
        "price_change_pct": round(stock_data.price_change_pct, 4),
        "ohlcv": prices_list,
        "news": news_list,
        "news_count": len(news_list),
    }


@router.get("/{ticker}/sentiment")
async def get_sentiment_summary(ticker: str):
    """Quick sentiment summary without full OHLCV load."""
    from tools.market_data import fetch_stock_data, compute_overall_sentiment, filter_news

    ticker = ticker.upper()
    try:
        stock_data = await asyncio.to_thread(fetch_stock_data, ticker, "1mo")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    overall = compute_overall_sentiment(stock_data.news)
    by_category = {}
    for cat in ["earnings", "product", "management", "policy", "competition", "market"]:
        cat_news = filter_news(stock_data.news, categories=[cat])
        if cat_news:
            by_category[cat] = {
                "count": len(cat_news),
                "avg_score": round(sum(n.sentiment_score for n in cat_news) / len(cat_news), 3),
            }

    return {
        "ticker": ticker,
        "overall_sentiment": overall,
        "label": "bullish" if overall > 0.1 else "bearish" if overall < -0.1 else "neutral",
        "by_category": by_category,
        "total_events": len(stock_data.news),
    }
