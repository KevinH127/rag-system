"""Settings, read from the environment or `.env` (see `.env.example`)."""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Services. The defaults match docker-compose.yml and a local `ollama serve`.
    database_url: str = "postgresql://rag:rag@localhost:5433/rag"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    # Changing it means re-ingesting; chunks.embedding in knowledge/schema.sql is 768-dimensional.
    embed_model: str = "nomic-embed-text"
    # Markdown files under this directory (subfolders included) are the knowledge base.
    knowledge_base_dir: Path = Path("knowledge_base")

    # Sections retrieved per message and shown to the model.
    top_k: int = Field(4, ge=1, le=20)
    # Questions the bot may ask about one vague message before handing off to staff.
    max_clarify_turns: int = Field(2, ge=0, le=5)

    # How many of the closest sections are checked for a quotable answer (one LLM call each).
    verify_top_n: int = Field(3, ge=1, le=10)

    # Cosine-distance thresholds for off-topic handling, tuned on evals/golden.jsonl (on-topic
    # messages measured <= 0.39, unrelated ones >= 0.45). Distance only says a message is ABOUT
    # the docs, never that the docs ANSWER it; answers are verified separately (verify.py).
    # The LLM may only call a message off-topic if it is at least this far from the docs.
    llm_off_topic_min_distance: float = Field(0.40, gt=0, le=1)
    # This far away with no Trevona term in the message: declined without calling the LLM.
    off_topic_min_distance: float = Field(0.44, gt=0, le=1)

    # Turn logging to Postgres (rag_engine/logbook). The debug trace is bulkier and internal, so
    # it can be switched off on its own.
    log_questions: bool = True
    log_debug: bool = True

    # Discord bot (`rag discord`). The token comes from the Discord Developer Portal; keep it in
    # .env only. Members of the staff role are pinged on a handoff, and their messages in a
    # ticket take it over from the bot.
    discord_token: SecretStr | None = None
    discord_staff_role_id: int | None = None


# The one instance every module imports.
settings = Settings()
