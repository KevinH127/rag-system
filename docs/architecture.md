# Architecture

## Layout

```
rag_engine/
  config.py          settings, including every tuned threshold
  models.py          shared types: Intent, Action, Decision, Hit, Response, Trace, TurnRecord
  knowledge/         the knowledge base: storage, ingestion, search
    schema.sql, db.py, chunking.py, embed.py, ingest.py, retrieve.py
  assistant/         deciding what to reply
    engine.py        one conversation: state + the per-message pipeline
    policy.py        business rules around the LLM (pure)
    verify.py        only documented, quote-checked text becomes an answer (pure)
    vocabulary.py    word normalisation + Trevona domain terms (pure)
    redact.py        secret detection (pure)
    replies.py       engine-owned customer texts (pure)
    prompt.py        system prompt (pure)
    llm.py           the Ollama calls: decide() and quote_answer()
  logbook/           turn logging: sessions, question_log, question_debug
    schema.sql, store.py
  interfaces/        ways in: cli.py today, the Discord bot next
tests/               unit tests, no Ollama or Postgres; test_architecture.py checks the rules below
evals/
  run_eval.py        golden-set eval against the real model and database
  golden.example.jsonl  case format; the real golden.jsonl quotes private docs and is git-ignored
```

## Dependency rules (enforced by `tests/test_architecture.py`)
- `knowledge/` never imports `assistant/`, `logbook/` or `interfaces/`.
- `assistant/` never imports `logbook/` or `interfaces/`.
- `logbook/` never imports `assistant/` or `interfaces/`; it stores the `TurnRecord` it is given.
- The pure assistant modules (`policy`, `verify`, `vocabulary`, `redact`, `replies`, `prompt`) do
  no I/O: no Ollama, no Postgres. All business rules are testable without services.
- `Engine` receives its retriever, LLM, answer judge and recorder as injectable callables; tests
  pass stubs. The default recorder does nothing; interfaces pass `logbook.store.recorder(channel)`.

## One message, end to end

```
message
  -> redact            secrets removed; the rest of the message is still answered
  -> query             this message + earlier unresolved ones; a bare "what about Costco?" after a
                       resolved turn re-asks that turn's question with the retailer swapped
  -> retrieve          hybrid vector + keyword search over knowledge/ (top_k sections)
  -> policy.before_llm far from the docs and no Trevona word: decline, no LLM call
  -> llm.decide        one JSON call: intent (question/request/off_topic) + action + reply
  -> verify            questions only: the judge must quote the sentence that answers it from
                       one of the closest sections; the quote is checked against the text
  -> policy.after_llm  rules decide the final reply (see below)
  -> Engine            history, clarify counter, pending retrieval context updated; a handoff
                       closes the session
  -> record            TurnRecord (redacted message, reply, trace) handed to the recorder
```

Rules in `policy.after_llm`:
1. An off-topic verdict is only accepted far from the docs; otherwise it is treated as a question.
2. A question naming no retailer, whose closest section is one retailer's answer while other close
   sections answer it for other retailers ("how much is the fee?"), gets "which retailer?" instead
   of a guess (`policy.asks_which_retailer`, within `max_clarify_turns`; the judge is skipped).
3. **Only documented text is ever an answer**: a verified section is returned verbatim, and the
   model's own written answer is never shown. Requests are never answered.
4. No verified section: ask the model's clarifying question if it asked one (at most
   `max_clarify_turns`), otherwise hand off to staff with a ticket summary.
5. Handoff, off-topic and which-retailer texts come from `replies.py`, never from the model.
6. Unreadable model output is treated as "hand off" / "no answer", never a crash.
7. A handoff ends the conversation: `Engine.closed` is set, interfaces stop (`rag chat` exits), and
   anything sent later gets `replies.CLOSED` without running the pipeline.

## Decision log
| Decision | Why | Alternatives rejected |
|---|---|---|
| One structured LLM call per message, plus up to 3 short judge calls for questions | p95 7.5 s against a 10 s budget | Separate classify + answer calls |
| Business rules in code, not the prompt | A 3B model mislabels actions and promises things it cannot do | Prompt-only guardrails |
| Answers are verified, verbatim doc text | The model answered 14/14 undocumented questions, several with invented facts ("your card is encrypted") | Trusting the model's answer when retrieval looks relevant |
| Judge quotes first, then gives a *named* verdict | With `answers: bool` the model marked correct "No, not accepted" answers false (22/40 recall); a named verdict reached 35/40 | Yes/no flag; a three-way verdict (no better); ordering candidates by heading similarity (worse) |
| Word-overlap FAQ shortcut removed | Patched three times and still matched "same card" to the Vault/Venn answer | More word lists |
| Hybrid retrieval (vector + Postgres full-text, RRF) | Pure vector search missed exact terms such as "booster box" | Vector only |
| Deterministic off-topic gate before the LLM | The model treated "write me a python script" as a request | Trusting the model's classification |
| Distance thresholds only for off-topic, never for answerability | Measured: undocumented questions sit as close to the docs (0.12-0.40) as answerable ones | Distance as a relevance gate |
| In-memory conversation history | Enough for CLI testing | Postgres-backed history (planned with Discord) |
| Sessions group the logs; conversation state stays in memory | Every logged turn keeps its message and reply, so resuming can later be rebuilt from `question_log` without a schema change | Restoring state from Postgres now |
| Debug traces in their own table, switchable with `LOG_DEBUG` | Bulky and internal; can be disabled or pruned without losing the question log | One wide log table |
| Recorder injected into `Engine`; its failures never raise | A logging outage must not cost a customer their reply, and tests and evals run without Postgres | Engine writing to Postgres directly |
| "Which retailer?" decided in code from section headings | The judge accepted the Walmart fee section for "how much is the aco fee" (any retailer's fee answers it); a section that applies to one retailer names it in its `##` heading (README, "Editing the knowledge base"). Measured on the golden set: no answerable case affected | Asking the judge or the prompt to spot missing retailers |
| Follow-ups that only name a retailer ("what about Costco?") re-ask the last resolved question in code | Retrieval and the judge only saw "what about for costco" and answered with Costco's requirements. Asked to rewrite follow-ups into standalone questions, llama3.2:3b answered them instead (0/6 rewritten, several invented facts) | An LLM rewrite call per follow-up; always merging the previous question (breaks topic switches) |
| A handoff closes the session | Staff own the conversation from then on, in a ticket | Starting a fresh session in the same chat |
| Logs store sections by path and heading, not chunk id | `rag ingest` deletes and re-creates chunks, which would break or cascade-delete history | Foreign keys into `chunks` |

## Logging (`logbook/`)
One `Engine` is one session. Every message becomes a `question_log` row tied to its `sessions` row
by `session_id` and numbered by `turn`. Its `question_debug` row (1:1) holds how the reply was
reached: retrieved sections and distances, whether the off-topic gate fired, the model's own
proposal, every judge check, and stage timings. A handoff stamps the session's `ended_at` and
`end_reason = 'handoff'`. Only redacted text is ever stored. A recorder failure is logged and
swallowed, so the customer still gets the reply. `LOG_QUESTIONS=false` turns logging off,
`LOG_DEBUG=false` keeps the question log but drops the debug rows. Evals do not log.

## Evals (`evals/`)
`run_eval.py` sends every case in `evals/golden.jsonl` through a fresh `Engine` with the real
model and database, and reports intent and action accuracy, retrieval and section hits, answer
facts and latency. It exits non-zero on any of the failures that matter most for a support bot:
an amount or forbidden claim in an answer that the retrieved docs do not contain, an answer to a
question the docs do not cover, or p95 latency over 10 s. Per-case results go to
`evals/results/latest.json`. The golden set quotes the private knowledge base, so only its format
is committed (`golden.example.jsonl`, fields described in the README).

## Known limits (measured on the golden set, 79 cases)
- Answerable questions: 39/40 get the right facts; occasional handoffs where the judge rejects the
  right section.
- 3/14 undocumented questions get a true but partial doc answer ("Is there a deadline to pay?" ->
  "You pay after delivery"). These disappear when the docs gain the missing section, because the
  real answer is then the closest one.

## Adding the Discord bot
Create `interfaces/discord_bot.py`. Keep one `Engine` per Discord user or thread, created with
`recorder=store.recorder("discord")`, call `engine.respond(text)`, and post `response.reply`. On a
handoff, post `response.summary` into the staff ticket. No changes to `knowledge/` or
`assistant/` should be needed.
