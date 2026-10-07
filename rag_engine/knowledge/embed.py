"""Embeddings from Ollama's nomic-embed-text, which expects a task prefix on every text."""

import ollama

from rag_engine.config import settings

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

_client = ollama.Client(host=settings.ollama_host)


def _embed(inputs: list[str]) -> list[list[float]]:
    return _client.embed(model=settings.embed_model, input=inputs).embeddings


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _embed([DOC_PREFIX + t for t in texts])


def embed_query(text: str) -> list[float]:
    return _embed([QUERY_PREFIX + text])[0]
