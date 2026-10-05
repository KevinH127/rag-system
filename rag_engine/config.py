"""
Centralised configuration for the RAG engine.

All settings are read from environment variables (or a .env file).
Any missing required variable raises a ValidationError at startup
— not buried inside a query call at runtime.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # DATABASE
    database_url: str = Field(
        ...,
        description=(
            "PostgreSQL connection string. Must use the psycopg driver. "
            "Example: postgresql+psycopg://postgres:postgres@localhost:5432/ragdb"
        ),
    )

    # OLLAMA LLM
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Base URL of the locally running Ollama daemon.",
    )
    ollama_model: str = Field(
        default="llama3.2:3b",
        description="Ollama model name to use for answer generation.",
    )

    # EMBEDDINGS
    embed_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="FastEmbed model name. Must produce 384-dim vectors.",
    )

    # CHUNKING
    chunk_size: int = Field(
        default=600,
        ge=100,
        le=4000,
        description="Maximum character length per document chunk.",
    )
    chunk_overlap: int = Field(
        default=100,
        ge=0,
        description="Character overlap between consecutive chunks.",
    )

    # RETRIEVAL
    retrieval_top_k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of chunks to retrieve per query.",
    )
    retrieval_min_score: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cosine similarity score. Chunks below this threshold are "
            "discarded; if none pass, the graceful fallback response is returned."
        ),
    )

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 600)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})"
            )
        return v


# import this everywhere instead of re-instantiating.
settings = Settings()
