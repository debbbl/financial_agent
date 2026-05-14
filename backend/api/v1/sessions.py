from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio

router = APIRouter()


class CreateSessionRequest(BaseModel):
    ticker: Optional[str] = None
    period: str = "3mo"


@router.get("")
async def list_sessions(limit: int = 20):
    from db.db import get_all_sessions

    sessions = await asyncio.to_thread(get_all_sessions, limit)
    return {"sessions": sessions, "count": len(sessions)}


@router.post("")
async def create_session(body: CreateSessionRequest):
    from db.db import create_session

    session_id = await asyncio.to_thread(create_session, body.ticker, body.period)
    return {"session_id": session_id, "ticker": body.ticker, "period": body.period}


@router.get("/{session_id}")
async def get_session(session_id: str):
    from db.db import get_session as db_get_session, load_chat_history

    session = await asyncio.to_thread(db_get_session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    history = await asyncio.to_thread(load_chat_history, session_id)
    return {"session": session, "history": history, "message_count": len(history)}


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    from db.db import clear_chat_history, get_connection

    await asyncio.to_thread(clear_chat_history, session_id)

    def _delete():
        with get_connection() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))

    await asyncio.to_thread(_delete)
    return {"deleted": True, "session_id": session_id}
