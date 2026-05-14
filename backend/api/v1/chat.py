from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import asyncio
import json

from sse_starlette.sse import EventSourceResponse

router = APIRouter()


class NewsContextItem(BaseModel):
    date: str
    title: str
    sentiment: str | None = None
    impact: str | None = None


class ChartContext(BaseModel):
    ticker: str
    from_date: str
    to_date: str
    price_change_pct: float
    open_price: float
    close_price: float
    high: float
    low: float
    news_count: int
    top_news: list[NewsContextItem] = []


class ChatRequest(BaseModel):
    session_id: str
    message: str
    ticker: str
    chart_context: ChartContext | None = None


@router.post("/stream")
async def chat_stream(body: ChatRequest):
    """
    Stream a chat response through the 4-stage agent pipeline.

    SSE event types:
      stage_start   → {"type": "stage_start", "stage": "researcher"}
      stage_complete→ {"type": "stage_complete", "stage": "researcher", "summary": "..."}
      debate_turn   → {"type": "debate_turn", "speaker": "analyst"|"risk", "content": "...", "turn": 1|2}
      synthesis_token→{"type": "synthesis_token", "token": "..."}
      done          → {"type": "done", "session_id": "..."}
      error         → {"type": "error", "message": "..."}
    """
    from tools.market_data import fetch_stock_data
    from agents.orchestrator import AgentOrchestrator
    from db.db import get_session
    from core.config import get_settings

    settings = get_settings()

    # Validate session exists
    session = await asyncio.to_thread(get_session, body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    event_queue: asyncio.Queue = asyncio.Queue()

    async def generate():
        # Load stock data
        ticker = body.ticker.upper()
        try:
            stock_data = await asyncio.to_thread(fetch_stock_data, ticker, "3mo")
        except Exception as e:
            yield {"data": json.dumps({"type": "error", "message": f"Failed to load {ticker}: {e}"})}
            return

        orchestrator = AgentOrchestrator(
            api_key=settings.groq_api_key,
            session_id=body.session_id,
            model=settings.groq_model,
        )
        await orchestrator.load_history()
        orchestrator.set_stock_data(stock_data)

        # Run pipeline, stream events
        chart_ctx = body.chart_context.model_dump() if body.chart_context else None
        task = asyncio.create_task(orchestrator.run(body.message, event_queue, chart_context=chart_ctx))

        while True:
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=180.0)
                yield {"data": json.dumps(event)}
                if event.get("type") in ("done", "error"):
                    break
            except asyncio.TimeoutError:
                yield {"data": json.dumps({"type": "error", "message": "Request timed out (180s)"})}
                break

        await task

    return EventSourceResponse(generate())
