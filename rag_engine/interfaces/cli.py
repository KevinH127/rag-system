import time
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
import typer

from rag_engine.assistant.engine import Engine
from rag_engine.knowledge import db, ingest
from rag_engine.logbook import store
from rag_engine.models import Action, Response

app = typer.Typer(help="Trevona ACO customer-service RAG engine", no_args_is_help=True)
db_app = typer.Typer(help="Database commands", no_args_is_help=True)
app.add_typer(db_app, name="db")


@contextmanager
def _services() -> Iterator[None]:
    """Turn a missing Ollama or Postgres into a one-line hint instead of a traceback."""
    try:
        yield
    except ConnectionError:  # raised by the ollama client
        typer.secho(
            "Can't reach Ollama. Start the Ollama app (or `ollama serve`).", err=True, fg="red"
        )
        raise typer.Exit(1) from None
    except psycopg.OperationalError:
        typer.secho("Can't reach Postgres. Run `docker compose up -d`.", err=True, fg="red")
        raise typer.Exit(1) from None


def _show(r: Response, elapsed: float, debug: bool) -> None:
    typer.echo(f"\n{r.reply}\n")
    if r.summary and r.action is Action.HANDOFF:
        typer.echo(f"[ticket summary] {r.summary}\n")
    if debug:
        if r.redacted:
            typer.echo(f"[redacted: {', '.join(r.redacted)}]")
        typer.echo(f"[{r.intent} / {r.action} / {elapsed:.1f}s]")
        if r.source:
            typer.echo(f"[answered from: {r.source.path} | {r.source.heading}]")
        for h in r.hits:
            typer.echo(f"  {h.distance:.3f} {h.path} | {h.heading}")


def _engine(debug: bool) -> Engine:
    """A new session, logged to Postgres."""
    engine = Engine(recorder=store.recorder("cli"))
    if debug:
        typer.echo(f"[session {engine.session_id}]")
    return engine


@app.command("ask")
def ask_cmd(message: str, debug: bool = typer.Option(False, "--debug")) -> None:
    """Ask a single question (a one-message session)."""
    engine = _engine(debug)
    start = time.perf_counter()
    with _services():
        response = engine.respond(message)
    _show(response, time.perf_counter() - start, debug)


@app.command("chat")
def chat_cmd(debug: bool = typer.Option(False, "--debug")) -> None:
    """Interactive conversation, one session (empty line to quit; ends after a handoff)."""
    engine = _engine(debug)
    while message := typer.prompt("you", default="", show_default=False).strip():
        start = time.perf_counter()
        with _services():
            response = engine.respond(message)
        _show(response, time.perf_counter() - start, debug)
        if engine.closed:
            typer.echo("[session closed: handed off to staff]")
            break


@app.command("ingest")
def ingest_cmd() -> None:
    """Embed the knowledge base into Postgres (skips unchanged files)."""
    with _services():
        r = ingest.ingest()
    typer.echo(
        f"added={r.added} updated={r.updated} unchanged={r.unchanged} "
        f"removed={r.removed} chunks_written={r.chunks}"
    )


@db_app.command("init")
def db_init() -> None:
    """Create the pgvector extension, knowledge tables and log tables."""
    with _services():
        db.init_schema()
        store.init_schema()
    typer.echo("Schema ready.")
