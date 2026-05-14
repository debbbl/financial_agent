from fastapi import APIRouter
from pydantic import BaseModel
import asyncio
import json

from sse_starlette.sse import EventSourceResponse

router = APIRouter()


class RangeAnalysisRequest(BaseModel):
    ticker: str
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD
    session_id: str


@router.post("/range")
async def analyze_range(body: RangeAnalysisRequest):
    """
    Trigger the full 4-stage agent pipeline for a specific date range.
    Returns SSE stream — same format as /chat/stream.
    """
    from tools.market_data import fetch_stock_data
    from agents.orchestrator import AgentOrchestrator
    from core.config import get_settings

    settings = get_settings()
    event_queue: asyncio.Queue = asyncio.Queue()

    async def generate():
        try:
            stock_data = await asyncio.to_thread(
                fetch_stock_data, body.ticker.upper(), None, body.start_date, body.end_date
            )
        except Exception as e:
            yield {"data": json.dumps({"type": "error", "message": str(e)})}
            return

        orchestrator = AgentOrchestrator(
            api_key=settings.groq_api_key,
            session_id=body.session_id,
            model=settings.groq_model,
        )
        await orchestrator.load_history()
        orchestrator.set_stock_data(stock_data)

        message = f"Analyze {body.ticker} from {body.start_date} to {body.end_date}. What happened in this period and what drove the price movement?"

        # Run pipeline in background, stream events
        task = asyncio.create_task(orchestrator.run(message, event_queue))

        while True:
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=120.0)
                yield {"data": json.dumps(event)}
                if event.get("type") in ("done", "error"):
                    break
            except asyncio.TimeoutError:
                yield {"data": json.dumps({"type": "error", "message": "Analysis timed out"})}
                break

        await task

    return EventSourceResponse(generate())
