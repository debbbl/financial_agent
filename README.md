# Financial Agent

Event-driven stock analysis with a **React** dashboard and **FastAPI** backend. The UI loads OHLCV and news, overlays sentiment on charts, and runs a **multi-stage Groq LLM pipeline** (researcher → analyst → risk → synthesis) with tool calls (range analysis, similar periods, macro, options, SEC filings).

## Repository layout

| Path | Role |
|------|------|
| `backend/` | FastAPI app (`main.py`), REST + SSE chat, SQLite + ChromaDB, `tools/market_data.py`, `agents/orchestrator.py`. |
| `frontend/` | Vite + React + Tailwind; proxies `/api` to the backend in dev. |
| `docs/HANDOFF.md` | Engineering handoff, verification checklist, known gaps. |
| `requirements.txt` | Same dependency set as `backend/requirements.txt` (install from repo root). |
| `pytest.ini` | Runs `backend/tests` with `pythonpath=backend`. |

## Prerequisites

- Python 3.11+ recommended  
- Node 20+ for the frontend  
- `GROQ_API_KEY` (required). Optional: `FRED_API_KEY`, `FINNHUB_API_KEY`, `NEWSAPI_API_KEY`, `ALPHAVANTAGE_API_KEY` — copy `backend/.env.example` to **`backend/.env`** (the API loads that file by path, not a root `.env`).

## Quick start

**Backend** (from repo root or `backend/`):

```bash
python -m venv venv
venv\Scripts\activate
pip install -r backend/requirements.txt
copy backend\.env.example backend\.env
# Edit backend\.env — set GROQ_API_KEY at minimum
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend** (separate terminal):

```bash
cd frontend
npm ci
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`). API requests go to `/api/v1/*` and are proxied to `http://localhost:8000`.

## Tests

From repository root:

```bash
pip install -r requirements.txt
python -m pytest -q
```

## Docker

`backend/Dockerfile` and `frontend/Dockerfile` exist for container builds; wire them in compose if you deploy as a pair (API + static nginx).

## Claude for Financial Services (external)

Institutional-style skills and slash commands (e.g. comps, DCF workflows) live in the separate [anthropics/claude-for-financial-services](https://github.com/anthropics/claude-for-financial-services) ecosystem. This app does not bundle them; use that repo in Claude Code when you want those workflows next to this dashboard.

## Disclaimer

Outputs are analytical drafts, not investment advice.
