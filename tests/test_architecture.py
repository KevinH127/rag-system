"""Guard the layering described in docs/architecture.md."""

import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "rag_engine"
IO_LIBRARIES = {"ollama", "psycopg", "pgvector"}
# Decide replies only; must stay free of I/O so the rules are testable without services.
PURE_ASSISTANT_MODULES = [
    "policy",
    "verify",
    "grounding",
    "vocabulary",
    "redact",
    "replies",
    "prompt",
]


def imported_modules(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


@pytest.mark.parametrize("path", sorted((PKG / "knowledge").glob("*.py")), ids=lambda p: p.name)
def test_knowledge_layer_knows_nothing_about_conversations(path):
    bad = {
        m
        for m in imported_modules(path)
        if m.startswith(("rag_engine.assistant", "rag_engine.logbook", "rag_engine.interfaces"))
    }
    assert not bad


@pytest.mark.parametrize("path", sorted((PKG / "assistant").glob("*.py")), ids=lambda p: p.name)
def test_assistant_layer_does_not_depend_on_logbook_or_interfaces(path):
    # The log store is injected as Engine's recorder, so the engine runs without Postgres.
    bad = {
        m
        for m in imported_modules(path)
        if m.startswith(("rag_engine.logbook", "rag_engine.interfaces"))
    }
    assert not bad


@pytest.mark.parametrize("path", sorted((PKG / "logbook").glob("*.py")), ids=lambda p: p.name)
def test_logbook_only_stores_what_it_is_given(path):
    bad = {
        m
        for m in imported_modules(path)
        if m.startswith(("rag_engine.assistant", "rag_engine.interfaces"))
    }
    assert not bad


@pytest.mark.parametrize("name", PURE_ASSISTANT_MODULES)
def test_rule_modules_do_no_io(name):
    modules = imported_modules(PKG / "assistant" / f"{name}.py")
    bad = {
        m
        for m in modules
        if m.split(".")[0] in IO_LIBRARIES or m.startswith("rag_engine.knowledge")
    }
    assert not bad
