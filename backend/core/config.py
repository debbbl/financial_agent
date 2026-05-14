from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/core/config.py -> parents[1] == backend/, parents[2] == repo root
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Prefer repo-root `.env` for API keys; load `backend/.env` first (if present) so root entries
# override duplicates (pydantic loads later files on top of earlier ones).
_env_files_list: list[Path] = []
if (_BACKEND_ROOT / ".env").is_file():
    _env_files_list.append(_BACKEND_ROOT / ".env")
if (_REPO_ROOT / ".env").is_file():
    _env_files_list.append(_REPO_ROOT / ".env")
_env_files = tuple(_env_files_list) if _env_files_list else None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_env_files if _env_files else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = ""
    fred_api_key: str = ""
    finnhub_api_key: str = ""
    alphavantage_api_key: str = ""
    newsapi_api_key: str = ""
    database_url: str = "sqlite:///./data/financial_agent.db"
    chroma_path: str = "./data/chroma_store"
    cors_origins: list[str] = ["http://localhost:5173"]
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"


@lru_cache
def get_settings() -> Settings:
    return Settings()
