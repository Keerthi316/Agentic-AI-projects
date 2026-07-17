"""Central configuration loaded from .env"""
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

APP_TITLE: str = os.getenv("APP_TITLE", "AI Recruitment Multi-Agent System")
MAX_REVISIONS: int = int(os.getenv("MAX_REVISIONS", "3"))
DB_PATH: str = os.getenv("DB_PATH", "./database/recruitment.db")
