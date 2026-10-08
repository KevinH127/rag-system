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
    verify.py        which section answers: judge quotes checked against the text (pure)
    grounding.py     whether the model's own wording says only what the docs say (pure)
    vocabulary.py    word normalisation, Trevona terms, retailers, products, small talk (pure)
    redact.py        secret detection (pure)
    replies.py       engine-owned customer texts (pure)
    prompt.py        system prompt (pure)
    llm.py           the Ollama calls: decide(), quote_answer() and summarize()
  logbook/           turn logging: sessions, question_log, question_debug
    schema.sql, store.py
  interfaces/        ways in: cli.py and discord_bot.py
tests/               unit tests, no Ollama or Postgres; test_architecture.py checks the rules below
evals/
  run_eval.py        golden-set eval against the real model and database
  golden.example.jsonl  case format; the real golden.jsonl quotes private docs and is git-ignored
```

## Dependency rules (enforced by `tests/test_architecture.py`)
- `knowledge/` never imports `assistant/`, `logbook/` or `interfaces/`.
- `assistant/` never imports `logbook/` or `interfaces/`.
- `logbook/` never imports `assistant/` or `interfaces/`; it stores the `TurnRecord` it is given.
- The pure assistant modules (`policy`, `verify`, `grounding`, `vocabulary`, `redact`, `replies`,
  `prompt`) do no I/O: no Ollama, no Postgres. All business rules are testable without services.
- `Engine` receives its retriever, LLM, answer judge, summarizer and recorder as injectable
  callables; tests pass stubs. The default recorder does nothing; interfaces pass
  `logbook.store.recorder(channel)`.

## One message, end to end

```
message
  -> redact            secrets removed; the rest of the message is still answered
  -> query             this message + earlier unresolved ones; a bare follow-up after a resolved
                       turn ("what about Costco?", "in general") re-asks that turn's question
  -> retrieve          hybrid vector + keyword search over knowledge/ (top_k sections)
  -> policy.before_llm small talk gets a friendly reply; far from the docs with no Trevona word:
                       decline. Neither calls the LLM
  -> llm.decide        one JSON call: intent (question/request/off_topic) + action + the reply in
                       the model's own words. When the answer should cover every retailer,
                       sections tied to no retailer come first in its context
  -> verify            questions only: the judge must quote the sentence that answers it from
                       one of the closest sections; the quote is checked against the text.
                       Sections about a retailer the question does not ask about are skipped
  -> policy.after_llm  rules decide the final reply (see below)
  -> llm.summarize     handoffs only: a ticket summary of the messages not answered since the
                       last resolved turn
  -> Engine            history, clarify counter, pending retrieval context updated; a handoff
                       closes the session
  -> record            TurnRecord (redacted message, reply, trace) handed to the recorder
```

Rules in `policy.after_llm`:
1. An off-topic verdict is only accepted far from the docs; otherwise it is treated as a question.
   A bare follow-up is always a question, whatever the model labels it.
2. The bot never asks which retailer. A question naming no retailer or product, whose closest
   section answers it for one retailer while other close sections answer it for others ("how much
   is the fee?"), or that asks "in general", is answered for every retailer: sections tied to no
   retailer (the fee overview) are judged first (`policy.wants_general_answer`).
3. **Only documented facts are ever an answer.** Only a question with a verified section is
   answered. The reply is the model's own wording if `grounding.grounded` passes: every amount,
   number, link and channel is in the sections; each fee is said for the retailer and product the
   docs give it for, clause by clause; and it says yes or no the same way as the judge's quote.
   Otherwise the verified section is sent verbatim (`Response.composed` records which).
   Requests are never answered.
4. No verified section, or a request: hand off to staff. The ticket summary is written by a
   separate short call that sees only the messages not answered since the last resolved turn, so
   answered questions never reach the ticket; if it returns nothing, those messages are used as
   written.
5. The only question the bot asks is the model's own, for a question too vague to look up ("it's
   not working"), at most `max_clarify_turns` times; then nothing is verified (`policy.asks_back`).
   Never for a request, a bare follow-up, or an answer that varies by retailer. There are no
   hard-coded clarifying questions.
6. Handoff, off-topic and small-talk texts come from `replies.py`, never from the model.
7. Unreadable model output is treated as "hand off" / "no answer", never a crash.
8. A handoff ends the conversation: `Engine.closed` is set, interfaces stop (`rag chat` exits), and
   anything sent later gets `replies.CLOSED` without running the pipeline.

## Decision log
| Decision | Why | Alternatives rejected |
|---|---|---|
| One structured LLM call per message, plus up to 3 short judge calls for questions | p95 7.5 s against a 10 s budget | Separate classify + answer calls |
| Business rules in code, not the prompt | A 3B model mislabels actions and promises things it cannot do | Prompt-only guardrails |
| Only questions with a verified section are answered | The model answered 14/14 undocumented questions, several with invented facts ("your card is encrypted") | Trusting the model's answer when retrieval looks relevant |
| The model words the answer; `grounding.grounded` checks it, the verbatim section is the fallback | Verbatim sections read like a pasted FAQ and could not combine sections ("every Pokémon Center fee", "in general"). The check caught the model's own drafts "The fee is $2.50 to $5 per item" (Walmart's fee as the general one) and "$25 CAD for pre-orders on Pokémon Center Canada" ($25 is the Elite Trainer Box fee only). The model's existing reply is reused, so no extra LLM call | Verbatim sections only; a separate rewrite call per answer (p95 was already 9.7 s); handing off when the wording fails (customers wait for staff on documented questions) |
| No "which retailer?" question: answers that vary by retailer are summarised | Asked "in general", the bot asked the same question again; customers expect an answer, and staff for anything it cannot answer | A hard-coded clarifying question, asked once |
| Sections about another retailer are never judged (heading, or text when the heading names none) | The judge accepted the Pokémon Center "other items" fee for "the ACO fee for Bandai drops", and the fee overview (every retailer but Bandai) for "bandai" | Trusting the judge's prompt rule about retailers |
| Small talk answered in code, without the LLM | "how are you doing today?" was declined as off-topic | Letting the model chat freely |
| Judge quotes first, then gives a *named* verdict | With `answers: bool` the model marked correct "No, not accepted" answers false (22/40 recall); a named verdict reached 35/40 | Yes/no flag; a three-way verdict (no better); ordering candidates by heading similarity (worse) |
| Word-overlap FAQ shortcut removed | Patched three times and still matched "same card" to the Vault/Venn answer | More word lists |
| Hybrid retrieval (vector + Postgres full-text, RRF) | Pure vector search missed exact terms such as "booster box" | Vector only |
| Deterministic off-topic gate before the LLM | The model treated "write me a python script" as a request | Trusting the model's classification |
| Distance thresholds only for off-topic, never for answerability | Measured: undocumented questions sit as close to the docs (0.12-0.40) as answerable ones | Distance as a relevance gate |
| In-memory conversation history; after a restart the bot answers only in tickets opened since | Tickets are short-lived, and a restarted bot that resumed old tickets would have forgotten both their conversation and that staff had taken them over, and could reply over staff. Every turn is still logged | Postgres-backed history; rereading each ticket's Discord history on restart |
| The Discord bot goes silent in a ticket once staff post or it hands off | Staff own the ticket from then on; the bot must never talk over them or answer the customer's replies to staff | Answering until the ticket closes |
| Sessions group the logs; conversation state stays in memory | Every logged turn keeps its message and reply, so resuming can later be rebuilt from `question_log` without a schema change | Restoring state from Postgres now |
| Debug traces in their own table, switchable with `LOG_DEBUG` | Bulky and internal; can be disabled or pruned without losing the question log | One wide log table |
| Recorder injected into `Engine`; its failures never raise | A logging outage must not cost a customer their reply, and tests and evals run without Postgres | Engine writing to Postgres directly |
| Whether an answer varies by retailer is decided in code from section headings | The judge accepted the Walmart fee section for "how much is the aco fee" (any retailer's fee answers it); a section that applies to one retailer names it in its `##` heading (README, "Editing the knowledge base") | Asking the judge or the prompt to spot missing retailers |
| Bare follow-ups ("what about Costco?", "in general") re-ask the last resolved question in code | Retrieval and the judge only saw "what about for costco" and answered with Costco's requirements. Asked to rewrite follow-ups into standalone questions, llama3.2:3b answered them instead (0/6 rewritten, several invented facts) | An LLM rewrite call per follow-up; always merging the previous question (breaks topic switches) |
| A handoff closes the session | Staff own the conversation from then on, in a ticket | Starting a fresh session in the same chat |
| Ticket summaries come from a separate call on handoff, given only the unanswered messages | The summary used to be written in the main call only when the model itself chose to hand off; otherwise every message of the session was joined, answered questions included. The main call no longer writes one, which offsets the extra call | Summarising in every main call (tokens on every message, and it sees answered questions) |
| The model's own question about a vague message is asked, and nothing is verified | Support words ("not working") now reach the model; for "it's not working" the judge verified "A decline can show up twice in the team's reports" | Letting any verified section win over the model's question |
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

## Known limits (measured on the golden set, 113 cases)
- No hallucinated amounts, and no undocumented question answered. Intent is right on all 113.
- 55 of 80 answers use the model's own wording; the other 25 failed the grounding check and were
  sent as the doc section. Most of those 25 contained an invented or contradicting claim; a few
  were correct but phrased the opposite way to the section ("Can I get a refund?" against "Do I
  still pay?").
- The grounding check works on words, so a reply that recombines doc words into a new claim can
  pass ("check the current fee in the ACO Fees and Payment section of our Discord channel"). The
  prompt forbids adding steps, places or reasons; the eval's `must_contain` and `must_not` checks
  catch some of the rest.
- Two answerable questions hand off: the model labels "I only want the Elite Trainer Boxes" a
  request, and the judge rejects the right section for one ETB fee phrasing.
- `must_contain` matches exact text, so a correct paraphrase can fail "answer facts" ("you don't
  need Amazon Prime" for "not required"): 71/75.
- p95 latency is 10.6 s against the 10 s budget: answers in the model's own words are longer, and
  handoffs make the extra summary call.
- The judge sometimes quotes a section's heading back instead of its text; that never counts, so a
  section needs a plain sentence that answers its heading, not only a list.

## Discord bot (`interfaces/discord_bot.py`)
One `Engine` per ticket channel, created with `recorder=store.recorder("discord")`. The bot
answers in every channel it can read; Discord permissions limit it to ticket channels. On a
handoff it posts the reply, then the ticket summary with a ping for the staff role, and goes
silent in that channel; a staff-role member posting in a ticket also silences it. The engine
blocks on Ollama, so each call runs in a worker thread (`asyncio.to_thread`) and one lock per
channel keeps a ticket's messages in order. If the engine fails, the customer gets
`replies.UNAVAILABLE` and staff are pinged. `Tickets` and `messages_for` hold this logic without
touching Discord, so they are unit-tested; `create_client` wires them to the client. It answers
only in channels created after the bot started (see the decision log). Nothing in
`knowledge/` or `assistant/` changed for it.
