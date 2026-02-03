from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _default_db_url() -> str:
    """Use legalai.db in project root (works for local dev on any OS)."""
    project_root = Path(__file__).resolve().parents[2]
    db_path = project_root / "legalai.db"
    return f"sqlite:///{db_path.as_posix()}"


@dataclass(frozen=True)
class Settings:
    # DB: default = legalai.db in project root; override via DATABASE_URL
    DATABASE_URL: str = os.getenv("DATABASE_URL", _default_db_url())

    # Auth
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_DEV_SECRET")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "10080"))  # 7 days

    # App
    ENV: str = os.getenv("ENV", "dev")


settings = Settings()
