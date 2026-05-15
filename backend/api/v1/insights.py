import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

_SYSTEM = (
    "You are a concise financial assistant. Given the ticker, that session's OHLC, "
    "day-over-day return vs the prior bar's close, and one news headline (plus optional metadata), "
    "write exactly 1 or 2 short sentences explaining how this news could plausibly relate to "
    "the stock's move that day. Use neutral, factual language; no investment advice; "
    "do not claim causation with certainty — use 'may', 'could', 'often'. "
    "Do not repeat the headline verbatim."
)


class NewsPriceBlurbBody(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=32)
    bar_date: str = Field(..., description="Trading day YYYY-MM-DD")
    open: float
    high: float
    low: float
    close: float
    day_change_pct: float | None = Field(
        default=None, description="Intraday (close-open)/open * 100 if available"
    )
    prev_bar_change_pct: float | None = Field(
        default=None, description="Close vs previous bar's close, * 100"
    )
    news_title: str = Field(..., min_length=1, max_length=500)
    news_summary: str | None = Field(default=None, max_length=4000)
    news_sentiment: str | None = None
    news_impact: str | None = None
    news_category: str | None = None


@router.post("/news-price-blurb")
async def news_price_blurb(body: NewsPriceBlurbBody):
    """
    One short LLM paragraph: how the given headline may relate to the bar's price action.
    """
    from groq import AsyncGroq

    from core.config import get_settings

    settings = get_settings()
    if not settings.groq_api_key:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured; AI blurbs are disabled.",
        )

    day_pct = body.day_change_pct
    if day_pct is None and abs(body.open) > 1e-12:
        day_pct = (body.close - body.open) / body.open * 100.0

    facts = {
        "ticker": body.ticker.upper(),
        "bar_date": body.bar_date,
        "ohlc": {
            "open": body.open,
            "high": body.high,
            "low": body.low,
            "close": body.close,
        },
        "intraday_change_pct_vs_open": round(day_pct, 4) if day_pct is not None else None,
        "change_pct_vs_prior_bar_close": (
            round(body.prev_bar_change_pct, 4) if body.prev_bar_change_pct is not None else None
        ),
        "news": {
            "title": body.news_title,
            "summary": body.news_summary,
            "sentiment": body.news_sentiment,
            "impact": body.news_impact,
            "category": body.news_category,
        },
    }

    client = AsyncGroq(api_key=settings.groq_api_key)
    try:
        completion = await client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": (
                        "Facts (JSON):\n"
                        f"{json.dumps(facts, ensure_ascii=False)}\n\n"
                        "Reply with only the 1–2 sentences, no prefix."
                    ),
                },
            ],
            max_tokens=180,
            temperature=0.25,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {e}") from e

    text = (completion.choices[0].message.content or "").strip()
    if not text:
        raise HTTPException(status_code=502, detail="Empty model response")

    # Hard cap for UI / safety
    if len(text) > 600:
        text = text[:600]
        dot = text.rfind(".")
        if dot > 80:
            text = text[: dot + 1]

    return {"blurb": text}
