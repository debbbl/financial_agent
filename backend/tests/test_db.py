"""Test DB functions using a temporary database."""

import importlib
import os

import pytest
from unittest.mock import patch


@pytest.fixture
def temp_db(tmp_path):
    db_file = str(tmp_path / "test.db").replace("\\", "/")
    chroma_dir = str(tmp_path / "chroma").replace("\\", "/")
    with patch.dict(os.environ, {"DATABASE_URL": f"sqlite:///{db_file}", "CHROMA_PATH": chroma_dir}):
        import db.db as db_module

        importlib.reload(db_module)
        db_module.init_db()
        yield db_module
    import db.db as db_module

    importlib.reload(db_module)


def test_create_and_get_session(temp_db):
    sid = temp_db.create_session("AAPL", "3mo")
    assert len(sid) == 36  # UUID format
    session = temp_db.get_session(sid)
    assert session["ticker"] == "AAPL"
    assert session["period"] == "3mo"


def test_save_and_load_messages(temp_db):
    sid = temp_db.create_session("NVDA")
    temp_db.save_message(sid, "user", "What is NVDA doing?")
    temp_db.save_message(sid, "assistant", "NVDA is surging on AI demand.")
    history = temp_db.load_chat_history(sid)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_clear_chat_history(temp_db):
    sid = temp_db.create_session("TSLA")
    temp_db.save_message(sid, "user", "Tesla analysis?")
    temp_db.clear_chat_history(sid)
    history = temp_db.load_chat_history(sid)
    assert len(history) == 0


def test_global_watchlist(temp_db):
    temp_db.add_to_global_watchlist("MSFT", "tracking AI pivot")
    temp_db.add_to_global_watchlist("GOOGL")
    items = temp_db.get_global_watchlist()
    tickers = [i["ticker"] for i in items]
    assert "MSFT" in tickers
    assert "GOOGL" in tickers
    deleted = temp_db.remove_from_global_watchlist("MSFT")
    assert deleted is True
    items_after = temp_db.get_global_watchlist()
    assert len(items_after) == 1
