from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import get_settings

# These imports come from my existing code (will be copied to backend/)
# from db.db import startup as db_startup
# from tools.market_data import get_finbert  # warm FinBERT on startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    settings = get_settings()

    # 1. Init DB + seed patterns (existing startup() function from db.py)
    from db.db import startup as db_startup

    await asyncio.to_thread(db_startup)

    # 2. Warm FinBERT model in background (non-blocking)
    async def warm_finbert():
        try:
            from tools.market_data import get_finbert

            await asyncio.to_thread(get_finbert)
            print("[Startup] FinBERT warmed successfully")
        except Exception as e:
            print(f"[Startup] FinBERT warm failed (will load on first request): {e}")

    asyncio.create_task(warm_finbert())

    print(f"[Startup] Financial Agent v2 ready — model: {settings.groq_model}")
    yield
    # Shutdown (nothing needed)


app = FastAPI(
    title="Financial Agent API",
    version="2.0.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.v1.router import router
from api.v1.ws import router as ws_router

app.include_router(router)
app.include_router(ws_router, prefix="/api/v1/ws")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0", "model": get_settings().groq_model}
