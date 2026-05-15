from fastapi import APIRouter

from .market import router as market_router
from .chat import router as chat_router
from .analysis import router as analysis_router
from .sessions import router as sessions_router
from .portfolio import router as portfolio_router
from .insights import router as insights_router

router = APIRouter(prefix="/api/v1")
router.include_router(market_router, prefix="/market", tags=["market"])
router.include_router(chat_router, prefix="/chat", tags=["chat"])
router.include_router(analysis_router, prefix="/analysis", tags=["analysis"])
router.include_router(sessions_router, prefix="/sessions", tags=["sessions"])
router.include_router(portfolio_router, prefix="/portfolio", tags=["portfolio"])
router.include_router(insights_router, prefix="/insights", tags=["insights"])
