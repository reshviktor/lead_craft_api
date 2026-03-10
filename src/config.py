"""
LeadCraft Application Settings

Central configuration loaded from environment variables / .env file.
All values can be overridden via environment variables or a .env file
placed in the project root.
"""

from pathlib import Path
from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    api_title: str = "LeadCraft API"
    api_version: str = "0.1.0"
    api_description: str = (
        "Bioactivity retrieval and molecular similarity search via ChEMBL"
    )
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    db_path: str = str(PROJECT_ROOT / "data" / "molecular_activities.db")

    @computed_field
    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.db_path}"

    default_organism: str = "Homo sapiens"
    default_min_similarity: float = 0.8
    default_max_molecules: int = 10

    log_level: str = "INFO"


settings = Settings()