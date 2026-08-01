"""Runtime settings loaded from environment variables."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="CAPTIONER_")
    artifact_dir: Path = Path("artifacts")
    model_path: Path = Path("artifacts/model.pt")
    vocab_path: Path = Path("artifacts/vocab.json")
    max_upload_bytes: int = 10 * 1024 * 1024
    device: str = "auto"


settings = Settings()
