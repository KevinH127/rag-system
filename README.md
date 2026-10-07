# Trevona ACO RAG Engine

Customer-service RAG engine: Ollama `llama3.2:3b` + `nomic-embed-text` v1.5, Postgres + pgvector.
Terminal CLI today; a Discord bot will call the same `Engine` later.

Requires Python 3.12+, Docker (for Postgres) and a running [Ollama](https://ollama.com) server.

## Setup

The knowledge base is not in this repo (`knowledge_base/` is git-ignored). Put the Markdown docs
under `knowledge_base/` before ingesting; subfolders are fine.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
docker compose up -d                      # Postgres + pgvector on :5433
ollama pull llama3.2:3b && ollama pull nomic-embed-text   # Ollama server must be running
rag db init                               # create tables (safe to re-run)
rag ingest                                # embed knowledge_base/ (skips unchanged files)
```

## Use

```bash
rag ask "How much is the ACO fee for an Elite Trainer Box?" --debug
rag chat                                  # interactive; ends on an empty line or a handoff
```

## Logs
Every message is logged to Postgres (`rag db init` creates the tables). One `rag chat` run is one
session; each `rag ask` is its own. `--debug` prints the session id.

| Table | One row per | Holds |
|---|---|---|
| `sessions` | conversation | channel, started / last active, `ended_at` + `end_reason` (`handoff`) |
| `question_log` | message | `session_id`, `turn`, redacted message, intent, action, reply, source section, `composed` (answer in the model's words, or the section verbatim), latency |
| `question_debug` | message (1:1) | retrieved sections, gate, the model's own proposal, judge checks, timings, settings |

Only redacted text is stored. `LOG_DEBUG=false` drops the debug rows; `LOG_QUESTIONS=false` turns
logging off. Query with `docker exec -it rag-db psql -U rag -d rag`:

```sql
-- Recent sessions
SELECT s.id, s.started_at, count(*) AS turns
FROM sessions s JOIN question_log q ON q.session_id = s.id
GROUP BY s.id ORDER BY s.started_at DESC LIMIT 20;

-- One session's transcript
SELECT turn, message, action, reply FROM question_log
WHERE session_id = '<session id>' ORDER BY turn;

-- Questions the docs could not answer: candidates for new sections
SELECT message, count(*) FROM question_log
WHERE intent = 'question' AND action = 'handoff'
GROUP BY message ORDER BY count(*) DESC;

-- Answers whose wording failed the check and were sent as the doc section
SELECT message, reply, source_heading FROM question_log
WHERE action = 'answer' AND NOT composed ORDER BY created_at DESC LIMIT 20;

-- Why one reply was given
SELECT d.query, d.gated, d.decision, d.checks, d.timings_ms
FROM question_debug d JOIN question_log q ON q.id = d.question_id
WHERE q.session_id = '<session id>' AND q.turn = 2;
```

## How it is organised
```
rag_engine/
  config.py      settings from .env
  models.py      types shared by every layer
  knowledge/     chunking, embeddings, ingestion, hybrid search (pgvector + full-text)
  assistant/     conversation pipeline and business rules
  logbook/       session and turn logging to Postgres
  interfaces/    the CLI, later the Discord bot
```

Each message is redacted, retrieved for, gated, and sent through one structured LLM call.
Retrieval is hybrid: vector and keyword rankings are fused with reciprocal rank fusion. The bot
answers like a search engine that knows the docs: in its own words, summarising across retailers
when the answer differs by retailer, and never asking which retailer. But it only answers a
question the docs verifiably answer: a judge must quote the sentence that answers it, and the
quote is checked. The model's wording is then checked against the docs (amounts, links, which
retailer and product each fee is for, yes or no); if anything does not match, the doc section is
sent as written instead. Anything the docs do not answer, and every request, goes to staff with a
ticket summary of what the customer asked that was not answered.

See [docs/architecture.md](docs/architecture.md) for the dependency rules (enforced by
`tests/test_architecture.py`), the pipeline step by step and the decision log.

## Quality checks

```bash
pytest                                    # unit tests, no Ollama or Postgres needed
ruff check .
python evals/run_eval.py                  # golden set: intent/action accuracy, retrieval,
                                          # hallucinated amounts, answers in the model's own
                                          # words, latency (budget 10 s)
```

The eval needs Ollama, Postgres and an ingested knowledge base. The real golden set quotes the
private docs, so it is git-ignored; start one from the template:

```bash
cp evals/golden.example.jsonl evals/golden.jsonl
```

Each line is one case:

| Field | Meaning |
|---|---|
| `id` | Unique name; the prefix groups cases for `--only` (e.g. `--only nm-,off-`) |
| `message` | What the customer writes |
| `history` | Optional: earlier messages of the same conversation, sent first; only `message` is scored |
| `intent` | Expected intent: `question`, `request` or `off_topic` |
| `actions` | Acceptable actions: any of `answer`, `clarify`, `handoff`, `decline` |
| `sources` | Doc filename prefixes that should be retrieved (`[]` when none should) |
| `headings` | Optional: section headings, any of which should be retrieved |
| `must_contain` | Optional: text an answer must include (all of it, or any with `contain_any: true`) |
| `must_not` | Optional: text an answer must never include |

Cases whose `actions` leave out `answer` are documentation gaps: answering them fails the eval.
Add 1-2 phrasings whenever you add documentation.

## Editing the knowledge base
One chunk is made per `##` section, prefixed with the document's `#` title and the section
heading so it reads on its own; sections over 1,500 characters are split on blank lines. Keep each
`##` section about one topic and state facts plainly: the bot's answers are checked against
them, and a section is sent as written when the check fails. Start each section with a sentence
that answers its heading: the judge has to quote one, and a section that is only a list is often
skipped. A section that applies to one retailer must name it in its heading (e.g.
`## ACO fee at Costco`), and every fee should name its retailer and product in the same clause:
that is how the bot knows which fee belongs to which retailer when it summarises.

The fee overview ("How much is the ACO fee in general?" in `04-fees-and-payment.md`) repeats every
fee, so customers who name no retailer get a full summary. Update it whenever a fee changes.

After editing run `rag ingest` (only changed files are re-embedded), then the eval.
