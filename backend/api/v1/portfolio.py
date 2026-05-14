from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import asyncio

router = APIRouter()


class AddTickerRequest(BaseModel):
    ticker: str
    notes: str = ""


@router.get("")
async def list_portfolio():
    from db.db import get_global_watchlist

    items = await asyncio.to_thread(get_global_watchlist)
    return {"portfolio": items, "count": len(items)}


@router.post("")
async def add_to_portfolio(body: AddTickerRequest):
    from db.db import add_to_global_watchlist

    item = await asyncio.to_thread(add_to_global_watchlist, body.ticker.upper(), body.notes)
    return item


@router.delete("/{ticker}")
async def remove_from_portfolio(ticker: str):
    from db.db import remove_from_global_watchlist

    deleted = await asyncio.to_thread(remove_from_global_watchlist, ticker.upper())
    if not deleted:
        raise HTTPException(status_code=404, detail=f"{ticker} not in portfolio")
    return {"deleted": True, "ticker": ticker.upper()}
