"""Shared FastAPI dependencies."""

from core.config import get_settings


def get_groq_key() -> str:
    settings = get_settings()
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY not set in .env")
    return settings.groq_api_key
