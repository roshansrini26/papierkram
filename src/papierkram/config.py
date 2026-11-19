"""Every setting in one place, read from environment variables or .env file"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="PAPIERKRAM_", extra="ignore")

    # storage
    data_dir: Path = Path("data")
    chroma_dir: Path = Path("chroma_db")
    collection_name: str = "knowledge_base"

    # models
    llm_provider: Literal["ollama"] = "ollama"
    llm_model: str = "llama3.1:8b"
    llm_temperature: float = 0.2
    embedding_model: str = "nomic-embed-test"
    ollama_base_url: str = "http://localhost:11434"

    # retrieval and chunking
    top_k: int = 5
    chunk_max_chars: int = 800
    chunk_overlap_sentences: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()  # build once and reuse it
