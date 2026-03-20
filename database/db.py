"""
database/db.py — Persistence layer

Two databases:
  1. SQLite (financial_agent.db)
     - sessions         → user sessions with metadata
     - chat_history     → full conversation per session
     - news_cache       → persisted news events per ticker+date
     - watchlist        → tickers a session has loaded

  2. Vector store (ChromaDB, local file-based)
     - historical_patterns collection
     → each record = one 30-day window of sentiment+price data
     → queried by cosine similarity to find matching past periods
"""

import sqlite3
import json
import os
import uuid
from datetime import datetime, timedelta
from dataclasses import asdict
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "financial_agent.db")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_store")


# ════════════════════════════════════════════════════════════════════════════
# SQLITE — sessions, chat history, news cache, watchlist
# ════════════════════════════════════════════════════════════════════════════

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # safe concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables if they don't exist. Safe to call on every app start."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                created_at   TEXT NOT NULL,
                last_active  TEXT NOT NULL,
                ticker       TEXT,
                period       TEXT DEFAULT '3mo'
            );

            CREATE TABLE IF NOT EXISTS chat_history (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT NOT NULL REFERENCES sessions(session_id),
                role         TEXT NOT NULL,
                content      TEXT NOT NULL,
                tool_calls   TEXT,
                created_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS news_cache (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT NOT NULL,
                date            TEXT NOT NULL,
                title           TEXT NOT NULL,
                source          TEXT,
                category        TEXT,
                sentiment       TEXT,
                sentiment_score REAL,
                impact          TEXT,
                url             TEXT,
                summary         TEXT,
                fetched_at      TEXT NOT NULL,
                UNIQUE(ticker, date, title)
            );

            CREATE TABLE IF NOT EXISTS watchlist (
                session_id  TEXT NOT NULL REFERENCES sessions(session_id),
                ticker      TEXT NOT NULL,
                added_at    TEXT NOT NULL,
                PRIMARY KEY (session_id, ticker)
            );

            CREATE TABLE IF NOT EXISTS historical_patterns (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker           TEXT NOT NULL,
                period_start     TEXT NOT NULL,
                period_end       TEXT NOT NULL,
                sentiment_score  REAL NOT NULL,
                bull_pct         REAL,
                bear_pct         REAL,
                price_change_pct REAL,
                dominant_category TEXT,
                news_count       INTEGER,
                outcome_30d      REAL,
                context_summary  TEXT,
                embedding        TEXT,
                created_at       TEXT NOT NULL,
                UNIQUE(ticker, period_start)
            );

            CREATE INDEX IF NOT EXISTS idx_chat_session
                ON chat_history(session_id);
            CREATE INDEX IF NOT EXISTS idx_news_ticker_date
                ON news_cache(ticker, date);
            CREATE INDEX IF NOT EXISTS idx_patterns_ticker
                ON historical_patterns(ticker);
        """)
    print(f"[DB] Initialised SQLite at {DB_PATH}")


# ── Session management ─────────────────────────────────────────────────────────

def create_session(ticker: str = None, period: str = "3mo") -> str:
    """Create a new session and return its ID."""
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, created_at, last_active, ticker, period) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, now, now, ticker, period)
        )
    return session_id


def update_session(session_id: str, ticker: str = None, period: str = None):
    """Update session metadata and last_active timestamp."""
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        if ticker and period:
            conn.execute(
                "UPDATE sessions SET last_active=?, ticker=?, period=? WHERE session_id=?",
                (now, ticker, period, session_id)
            )
        else:
            conn.execute(
                "UPDATE sessions SET last_active=? WHERE session_id=?",
                (now, session_id)
            )


def get_session(session_id: str) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
    return dict(row) if row else None


# ── Chat history ───────────────────────────────────────────────────────────────

def save_message(session_id: str, role: str, content: str, tool_calls: list = None):
    """Persist a single chat message."""
    now = datetime.utcnow().isoformat()
    tc_json = json.dumps(tool_calls) if tool_calls else None
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO chat_history (session_id, role, content, tool_calls, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, tc_json, now)
        )
    update_session(session_id)


def load_chat_history(session_id: str) -> list[dict]:
    """
    Load full conversation history for a session.
    Returns list of dicts compatible with Groq's messages format.
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT role, content, tool_calls FROM chat_history "
            "WHERE session_id=? ORDER BY id ASC",
            (session_id,)
        ).fetchall()

    messages = []
    for row in rows:
        msg = {"role": row["role"], "content": row["content"]}
        if row["tool_calls"]:
            msg["tool_calls"] = json.loads(row["tool_calls"])
        messages.append(msg)
    return messages


def clear_chat_history(session_id: str):
    """Reset conversation for a session without deleting the session itself."""
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM chat_history WHERE session_id=?", (session_id,)
        )


def get_all_sessions(limit: int = 50) -> list[dict]:
    """List recent sessions for a session browser UI."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT s.*, COUNT(c.id) as message_count "
            "FROM sessions s "
            "LEFT JOIN chat_history c ON s.session_id = c.session_id "
            "GROUP BY s.session_id "
            "ORDER BY s.last_active DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ── News cache ─────────────────────────────────────────────────────────────────

def save_news_to_db(ticker: str, news_events: list):
    """
    Persist news events to SQLite for offline access and historical analysis.
    Uses INSERT OR IGNORE to avoid duplicates.
    """
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.executemany(
            """INSERT OR IGNORE INTO news_cache
               (ticker, date, title, source, category, sentiment,
                sentiment_score, impact, url, summary, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    ticker, n.date, n.title, n.source, n.category,
                    n.sentiment, n.sentiment_score, n.impact,
                    getattr(n, "url", ""), getattr(n, "summary", ""), now
                )
                for n in news_events
            ]
        )
    print(f"[DB] Saved {len(news_events)} news events for {ticker}")


def load_news_from_db(ticker: str, start_date: str, end_date: str) -> list[dict]:
    """Load cached news from SQLite for a ticker+date range."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT * FROM news_cache
               WHERE ticker=? AND date>=? AND date<=?
               ORDER BY date ASC""",
            (ticker, start_date, end_date)
        ).fetchall()
    return [dict(r) for r in rows]


def news_db_is_fresh(ticker: str, start_date: str, end_date: str,
                     max_age_hours: int = 6) -> bool:
    """Check if DB news is recent enough to skip API calls."""
    cutoff = (datetime.utcnow() - timedelta(hours=max_age_hours)).isoformat()
    with get_connection() as conn:
        row = conn.execute(
            """SELECT COUNT(*) as cnt FROM news_cache
               WHERE ticker=? AND date>=? AND date<=? AND fetched_at>=?""",
            (ticker, start_date, end_date, cutoff)
        ).fetchone()
    return row["cnt"] > 0


# ── Watchlist ──────────────────────────────────────────────────────────────────

def add_to_watchlist(session_id: str, ticker: str):
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO watchlist (session_id, ticker, added_at) VALUES (?, ?, ?)",
            (session_id, ticker, now)
        )


def get_watchlist(session_id: str) -> list[str]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT ticker FROM watchlist WHERE session_id=? ORDER BY added_at DESC",
            (session_id,)
        ).fetchall()
    return [r["ticker"] for r in rows]


# ── Historical patterns (SQLite side) ─────────────────────────────────────────

def save_historical_pattern(
    ticker: str,
    period_start: str,
    period_end: str,
    sentiment_score: float,
    price_change_pct: float,
    bull_pct: float,
    bear_pct: float,
    dominant_category: str,
    news_count: int,
    outcome_30d: float,
    context_summary: str,
    embedding: list[float] = None,
):
    now = datetime.utcnow().isoformat()
    emb_json = json.dumps(embedding) if embedding else None
    with get_connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO historical_patterns
               (ticker, period_start, period_end, sentiment_score, bull_pct,
                bear_pct, price_change_pct, dominant_category, news_count,
                outcome_30d, context_summary, embedding, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ticker, period_start, period_end, sentiment_score, bull_pct,
             bear_pct, price_change_pct, dominant_category, news_count,
             outcome_30d, context_summary, emb_json, now)
        )


def find_similar_patterns_sql(
    ticker: str,
    sentiment_score: float,
    top_k: int = 3,
) -> list[dict]:
    """
    Simple similarity search using SQLite:
    Finds historical periods whose sentiment_score is closest to the query.
    Falls back to any ticker if not enough results for the requested ticker.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT *, ABS(sentiment_score - ?) as diff
               FROM historical_patterns
               WHERE ticker=?
               ORDER BY diff ASC
               LIMIT ?""",
            (sentiment_score, ticker, top_k)
        ).fetchall()

        if len(rows) < top_k:
            extra = conn.execute(
                """SELECT *, ABS(sentiment_score - ?) as diff
                   FROM historical_patterns
                   WHERE ticker != ?
                   ORDER BY diff ASC
                   LIMIT ?""",
                (sentiment_score, ticker, top_k - len(rows))
            ).fetchall()
            rows = list(rows) + list(extra)

    return [dict(r) for r in rows]


# ════════════════════════════════════════════════════════════════════════════
# VECTOR STORE — ChromaDB for semantic pattern matching
# ════════════════════════════════════════════════════════════════════════════

def get_chroma_collection():
    """
    Returns the ChromaDB collection for historical patterns.
    ChromaDB stores embeddings locally in CHROMA_PATH.
    Uses sentence-transformers to embed pattern descriptions.
    """
    try:
        import chromadb
        from chromadb.utils import embedding_functions

        client = chromadb.PersistentClient(path=CHROMA_PATH)
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"   # lightweight, runs locally
        )
        collection = client.get_or_create_collection(
            name="historical_patterns",
            embedding_function=emb_fn,
            metadata={"hnsw:space": "cosine"},
        )
        return collection
    except ImportError:
        return None


def upsert_pattern_to_vector_store(
    ticker: str,
    period_start: str,
    sentiment_score: float,
    price_change_pct: float,
    outcome_30d: float,
    context_summary: str,
    dominant_category: str,
):
    """
    Add or update a historical pattern in ChromaDB.
    The document text is a natural language description of the period —
    this is what gets embedded and searched semantically.
    """
    collection = get_chroma_collection()
    if collection is None:
        return  # ChromaDB not installed, skip silently

    doc_id = f"{ticker}_{period_start}"
    document = (
        f"{ticker} stock in {period_start}: "
        f"sentiment score {sentiment_score:+.2f}, "
        f"dominant news category {dominant_category}, "
        f"price moved {price_change_pct:+.1f}%, "
        f"30-day outcome {outcome_30d:+.1f}%. "
        f"{context_summary}"
    )
    metadata = {
        "ticker": ticker,
        "period_start": period_start,
        "sentiment_score": float(sentiment_score),
        "price_change_pct": float(price_change_pct),
        "outcome_30d": float(outcome_30d),
        "dominant_category": dominant_category,
    }

    try:
        collection.upsert(ids=[doc_id], documents=[document], metadatas=[metadata])
    except Exception as e:
        print(f"[ChromaDB] Upsert error: {e}")


def semantic_pattern_search(
    query_text: str,
    ticker: str = None,
    top_k: int = 3,
) -> list[dict]:
    """
    Search historical patterns by semantic similarity.
    query_text can be natural language like:
    'earnings miss + policy headwinds + bearish momentum'

    Falls back to SQL similarity search if ChromaDB unavailable.
    """
    collection = get_chroma_collection()
    if collection is None:
        print("[ChromaDB] Not available — falling back to SQL search")
        return []

    try:
        where = {"ticker": ticker} if ticker else None
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where,
        )

        patterns = []
        if results and results["metadatas"]:
            for meta, doc, dist in zip(
                results["metadatas"][0],
                results["documents"][0],
                results["distances"][0],
            ):
                patterns.append({
                    **meta,
                    "context_summary": doc,
                    "similarity": round(1 - dist, 3),  # cosine distance → similarity
                })
        return patterns

    except Exception as e:
        print(f"[ChromaDB] Query error: {e}")
        return []


# ── Seed historical patterns ───────────────────────────────────────────────────

SEED_PATTERNS = [
    # AAPL
    ("AAPL","2023-08-01","2023-09-01", 0.58, 3.2, 68, 32, "earnings",  8, 4.1,  "Post-earnings rally. Services revenue beat drove optimism despite China concerns."),
    ("AAPL","2024-02-01","2024-03-01", 0.65, 2.8, 72, 28, "product",   6, 3.8,  "Vision Pro launch hype mixed with services growth narrative dominating."),
    ("AAPL","2022-11-01","2022-12-01",-0.30,-4.1, 35, 65, "policy",    9,-2.1,  "China Foxconn production halt fears + Fed rate hike pressure on growth stocks."),
    ("AAPL","2023-05-01","2023-06-01", 0.72, 5.1, 75, 25, "earnings",  7, 6.2,  "Q2 beat across all segments. Warren Buffett increased Berkshire stake."),
    ("AAPL","2024-07-01","2024-08-01",-0.45,-3.2, 38, 62, "competition",10,-1.8,"Google AI search threat + antitrust ruling on App Store commissions."),
    # NVDA
    ("NVDA","2023-05-01","2023-06-01", 0.88,24.0, 91,  9, "product",   5,24.0,  "AI hype initial surge. Q1 guidance raised 50%. ChatGPT demand confirmed."),
    ("NVDA","2024-01-01","2024-02-01", 0.82,16.4, 85, 15, "market",    6,16.4,  "Data center spending boom confirmed. Microsoft Azure GPU order expansion."),
    ("NVDA","2022-09-01","2022-10-01",-0.42,-8.2, 28, 72, "policy",    8,-8.2,  "US export controls on H100 to China + gaming GPU demand collapse."),
    ("NVDA","2023-11-01","2023-12-01", 0.76,12.3, 80, 20, "earnings",  4,10.1,  "Q3 blowout earnings. Data center revenue tripled year-over-year."),
    # TSLA
    ("TSLA","2023-01-01","2023-02-01", 0.55,40.2, 65, 35, "product",   7,35.0,  "Price cuts drove volume optimism. Austin Gigafactory ramping strongly."),
    ("TSLA","2024-04-01","2024-05-01",-0.70,-15.3,25, 75, "management",9,-12.1, "Q1 delivery miss + Musk distraction narrative + BYD surpassing globally."),
    ("TSLA","2023-07-01","2023-08-01", 0.60, 8.1, 70, 30, "earnings",  5, 6.5,  "Q2 earnings beat on margins. Supercharger network partnership with Ford."),
    # MSFT
    ("MSFT","2023-10-01","2023-11-01", 0.75, 7.2, 78, 22, "product",   6, 8.3,  "Copilot AI launch across Office 365. Azure growth reacceleration confirmed."),
    ("MSFT","2024-01-01","2024-02-01", 0.68, 4.1, 71, 29, "earnings",  5, 5.2,  "Q2 FY2024 beat. Cloud + AI segment outperformed all analyst estimates."),
    # GOOGL
    ("GOOGL","2023-10-01","2023-11-01", 0.62, 9.8, 73, 27, "earnings", 6, 7.4,  "Q3 ad revenue recovery. YouTube monetization inflection. Cloud growing 28%."),
    ("GOOGL","2024-02-01","2024-03-01",-0.38,-4.5, 40, 60, "product",  8,-2.3,  "Gemini launch fumble. AI search cannibalisation fears dominated sentiment."),
    # AMZN
    ("AMZN","2023-10-01","2023-11-01", 0.71, 6.8, 75, 25, "earnings",  5, 8.1,  "AWS growth reacceleration to 12%. Ad revenue hit record. Margins improved."),
    ("AMZN","2022-10-01","2022-11-01",-0.52,-7.3, 30, 70, "management",7,-9.2,  "Layoff announcements. AWS slowdown fears. Consumer spending concerns."),
]


def seed_historical_patterns():
    """
    Populate the database with real historical patterns on first run.
    Safe to call multiple times — uses INSERT OR REPLACE.
    """
    with get_connection() as conn:
        existing = conn.execute("SELECT COUNT(*) as cnt FROM historical_patterns").fetchone()
        if existing["cnt"] >= len(SEED_PATTERNS):
            print(f"[DB] Historical patterns already seeded ({existing['cnt']} records)")
            return

    print(f"[DB] Seeding {len(SEED_PATTERNS)} historical patterns...")
    for p in SEED_PATTERNS:
        (ticker, period_start, period_end, sentiment_score, price_change_pct,
         bull_pct, bear_pct, dominant_category, news_count,
         outcome_30d, context_summary) = p

        save_historical_pattern(
            ticker=ticker, period_start=period_start, period_end=period_end,
            sentiment_score=sentiment_score, price_change_pct=price_change_pct,
            bull_pct=bull_pct, bear_pct=bear_pct,
            dominant_category=dominant_category, news_count=news_count,
            outcome_30d=outcome_30d, context_summary=context_summary,
        )
        upsert_pattern_to_vector_store(
            ticker=ticker, period_start=period_start,
            sentiment_score=sentiment_score,
            price_change_pct=price_change_pct,
            outcome_30d=outcome_30d,
            context_summary=context_summary,
            dominant_category=dominant_category,
        )

    print(f"[DB] Seeding complete.")


def startup():
    """Call once on app start — initialises DB and seeds patterns."""
    init_db()
    seed_historical_patterns()