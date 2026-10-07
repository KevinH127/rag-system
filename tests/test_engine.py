from rag_engine.assistant import replies
from rag_engine.assistant.engine import Engine
from rag_engine.models import Action, Decision, Hit, Intent


def section(heading: str, body: str, distance: float) -> Hit:
    return Hit("doc.md", heading, f"Doc > {heading}\n\n{body}", distance)


FEE = section("How much is the fee?", "The ACO fee is $25 CAD, paid after delivery.", 0.2)
CARDS = section("Can I use a Vault card?", "No. Vault or Venn cards are not accepted.", 0.25)
FAR = [section("Something unrelated?", "Nothing relevant here at all.", 0.9)]


def d(intent, action, reply="ok?", summary=None):
    return Decision(intent=intent, action=action, reply=reply, summary=summary)


def passage_body(passage):
    return passage.partition("\n\n")[2]


def quotes_whole_section(question, passage):
    """A judge that always finds an answer: it quotes the section body."""
    return passage_body(passage)


def finds_nothing(question, passage):
    return None


def boom(*_):
    raise AssertionError("must not be called")


def make(hits, *decisions, judge=finds_nothing):
    queue = list(decisions)
    return Engine(retriever=lambda q: hits, decider=lambda s, h: queue.pop(0), judge=judge)


# --- Only documented text is ever an answer -------------------------------------------------


def test_verified_section_is_the_answer_verbatim_not_the_models_wording():
    e = make([FEE], d(Intent.QUESTION, Action.ANSWER, "It's $99!"), judge=quotes_whole_section)
    r = e.respond("what is the fee?")
    assert r.action is Action.ANSWER
    assert r.reply == "The ACO fee is $25 CAD, paid after delivery."
    assert r.source == FEE


def test_unverified_answer_is_never_shown_and_goes_to_staff():
    # The bug this design fixes: the model invents an answer the docs do not contain.
    invented = "Yes, you can use the same card on every profile."
    r = make([CARDS], d(Intent.QUESTION, Action.ANSWER, invented)).respond("same card twice?")
    assert r.action is Action.HANDOFF and r.reply == replies.NO_ANSWER
    assert "same card" not in r.reply


def test_documented_answer_wins_even_if_the_model_chose_another_action():
    for action in (Action.DECLINE, Action.HANDOFF, Action.CLARIFY):
        e = make([FEE], d(Intent.QUESTION, action, "Which fee?"), judge=quotes_whole_section)
        assert e.respond("fee?").action is Action.ANSWER


def test_requests_are_never_answered_or_verified():
    r = make([FEE], d(Intent.REQUEST, Action.ANSWER, "Done!"), judge=boom).respond("refund me")
    assert r.action is Action.HANDOFF and r.reply == replies.REQUEST_HANDOFF
    assert r.summary == "refund me"


def test_handoff_reply_never_uses_the_models_promises():
    promise = "I'll create a ticket for you and email you!"
    r = make([FEE], d(Intent.REQUEST, Action.HANDOFF, promise, "refund")).respond("refund me")
    assert r.reply == replies.REQUEST_HANDOFF and r.summary == "refund"


# --- Clarifying questions -------------------------------------------------------------------


def test_model_question_is_asked_when_nothing_is_documented():
    r = make([CARDS], d(Intent.QUESTION, Action.CLARIFY, "Which retailer?")).respond("cards?")
    assert r.action is Action.CLARIFY and r.reply == "Which retailer?"


def test_handoff_that_asks_a_question_becomes_clarify():
    r = make([FEE], d(Intent.REQUEST, Action.HANDOFF, "Which order?")).respond("cancel it")
    assert r.action is Action.CLARIFY and r.reply == "Which order?"


def test_clarify_without_a_question_mark_uses_the_fallback():
    r = make([CARDS], d(Intent.QUESTION, Action.CLARIFY, "Tell me more.")).respond("card?")
    assert r.reply == replies.CLARIFY_FALLBACK


def test_clarify_capped_then_handoff():
    e = make([CARDS], *[d(Intent.QUESTION, Action.CLARIFY, "Which retailer?")] * 3)
    assert [e.respond(m).action for m in ("card a", "b", "c")] == [
        Action.CLARIFY,
        Action.CLARIFY,
        Action.HANDOFF,
    ]


def test_answer_check_uses_the_whole_unresolved_conversation():
    asked = []
    e = make(
        [FEE],
        d(Intent.QUESTION, Action.CLARIFY, "Which retailer?"),
        d(Intent.QUESTION, Action.ANSWER, "x"),
        judge=lambda q, p: asked.append(q),
    )
    e.respond("what is the fee?")
    e.respond("Walmart")
    assert asked[-1] == "what is the fee? Walmart"


def test_handoff_closes_the_session_and_nothing_runs_after_it():
    e = make([CARDS], d(Intent.QUESTION, Action.HANDOFF, "Staff will help", "need help"))
    assert e.respond("card x").action is Action.HANDOFF and e.closed
    e.retriever = e.decider = e.judge = boom
    r = e.respond("card y")
    assert r.action is Action.HANDOFF and r.reply == replies.CLOSED


def test_answers_and_clarifying_questions_keep_the_session_open():
    e = make(
        [FEE],
        d(Intent.QUESTION, Action.CLARIFY, "Which retailer?"),
        d(Intent.QUESTION, Action.ANSWER, "x"),
        judge=lambda q, p: None if q == "fee?" else passage_body(p),
    )
    assert e.respond("fee?").action is Action.CLARIFY and not e.closed
    assert e.respond("walmart").action is Action.ANSWER and not e.closed


# --- Which retailer -------------------------------------------------------------------------

WALMART_FEE = section(
    "How much is the ACO fee for Walmart Canada items?", "Walmart items cost $2.50 to $5.", 0.245
)
AMAZON_FEE = section(
    "How much is the ACO fee for Amazon Canada items?", "Amazon fees are set after the drop.", 0.274
)
COSTCO_FEE = section(
    "How much is the ACO fee for Costco Canada items?", "Costco fees are set after the drop.", 0.216
)
COSTCO_TIPS = section(
    "What are the requirements for Costco Canada ACO?", "A membership is needed.", 0.328
)
FEES_GENERAL = section("How do ACO fees work?", "The ACO fee is paid after delivery.", 0.2)


def fee_search(query):
    """Stands in for retrieval as measured: a named retailer's fee section comes first."""
    if "costco" in query:
        return [COSTCO_FEE, WALMART_FEE] if "fee" in query else [COSTCO_TIPS]
    return [WALMART_FEE, AMAZON_FEE]


def test_question_naming_no_retailer_asks_which_one_without_verifying():
    e = make([WALMART_FEE, AMAZON_FEE], d(Intent.QUESTION, Action.ANSWER, "$2.50"), judge=boom)
    r = e.respond("how much is the aco fee")
    assert r.action is Action.CLARIFY and r.reply == replies.WHICH_RETAILER


def test_question_naming_the_retailer_is_answered():
    hits = [WALMART_FEE, AMAZON_FEE]
    e = make(hits, d(Intent.QUESTION, Action.ANSWER, "x"), judge=quotes_whole_section)
    assert e.respond("walmart fee?").source == WALMART_FEE


def test_general_section_closest_is_answered_without_asking():
    hits = [FEES_GENERAL, WALMART_FEE, AMAZON_FEE]
    e = make(hits, d(Intent.QUESTION, Action.ANSWER, "x"), judge=quotes_whole_section)
    assert e.respond("how do the fees work?").source == FEES_GENERAL


def test_which_retailer_stays_within_the_clarify_budget():
    hits = [WALMART_FEE, AMAZON_FEE]
    e = make(hits, *[d(Intent.QUESTION, Action.ANSWER, "x")] * 3, judge=quotes_whole_section)
    assert [e.respond(m).action for m in ("fee?", "the fee", "fee please")] == [
        Action.CLARIFY,
        Action.CLARIFY,
        Action.ANSWER,
    ]


def test_follow_up_naming_another_retailer_re_asks_the_last_question():
    # Reported: "what about for costco" after a Walmart fee answer got Costco's requirements.
    e = Engine(
        retriever=fee_search,
        decider=lambda s, h: d(Intent.QUESTION, Action.ANSWER, "x"),
        judge=quotes_whole_section,
    )
    turns = ("how much is the aco fee", "walmart", "what about for costco")
    answers = [e.respond(m) for m in turns]
    assert [r.action for r in answers] == [Action.CLARIFY, Action.ANSWER, Action.ANSWER]
    assert answers[1].source == WALMART_FEE and answers[2].source == COSTCO_FEE


def test_a_new_question_after_an_answer_is_not_merged_with_the_last():
    queries = []
    e = Engine(
        retriever=lambda q: queries.append(q) or [WALMART_FEE, AMAZON_FEE],
        decider=lambda s, h: d(Intent.QUESTION, Action.ANSWER, "x"),
        judge=quotes_whole_section,
    )
    e.respond("walmart fee?")
    e.respond("how do I sign up for amazon?")
    assert queries[-1] == "how do I sign up for amazon?"


def test_a_bare_retailer_with_nothing_asked_before_is_taken_as_is():
    queries = []
    e = Engine(
        retriever=lambda q: queries.append(q) or [COSTCO_TIPS],
        decider=lambda s, h: d(Intent.QUESTION, Action.CLARIFY, "What about Costco?"),
        judge=finds_nothing,
    )
    e.respond("costco?")
    assert queries == ["costco?"]


# --- Off-topic ------------------------------------------------------------------------------


def test_distant_message_without_trevona_terms_is_declined_before_the_llm():
    e = Engine(retriever=lambda q: FAR, decider=boom, judge=boom)
    r = e.respond("Write me a python script")
    assert r.action is Action.DECLINE and r.reply == replies.OFF_TOPIC
    assert e.history[-1]["role"] == "assistant"


def test_trevona_term_bypasses_the_distance_gate():
    r = make(FAR, d(Intent.QUESTION, Action.HANDOFF, "Staff will help", "x")).respond("Bandai fee?")
    assert r.action is Action.HANDOFF


def test_gate_is_skipped_mid_clarification():
    e = make(
        [FEE],
        d(Intent.REQUEST, Action.CLARIFY, "Which retailer?"),
        d(Intent.REQUEST, Action.HANDOFF, "Staff will help", "x"),
    )
    assert e.respond("fix it please").action is Action.CLARIFY
    e.retriever = lambda q: FAR  # the short follow-up looks unrelated, but we are mid-flow
    assert e.respond("the blue one").action is Action.HANDOFF


def test_llm_off_topic_verdict_far_from_the_docs_is_declined():
    e = make([section("x?", "y", 0.42)], d(Intent.OFF_TOPIC, Action.DECLINE), judge=boom)
    assert e.respond("tell me about my team").action is Action.DECLINE


def test_llm_off_topic_verdict_close_to_the_docs_is_treated_as_a_question():
    e = make([FEE], d(Intent.OFF_TOPIC, Action.DECLINE, "no"), judge=quotes_whole_section)
    r = e.respond("can I DM the team about fees?")
    assert r.intent is Intent.QUESTION and r.action is Action.ANSWER


def test_off_topic_does_not_use_clarify_budget_or_pollute_retrieval():
    queries = []
    e = Engine(retriever=lambda q: queries.append(q) or FAR, decider=boom, judge=boom)
    e.respond("weather today?")
    assert e.clarify_count == 0
    e.decider = lambda s, h: d(Intent.QUESTION, Action.HANDOFF, "Staff", "x")
    e.judge = finds_nothing
    e.respond("fee?")
    assert queries[-1] == "fee?"


# --- Secrets --------------------------------------------------------------------------------


def test_secrets_never_reach_retriever_llm_judge_or_history():
    seen = []
    e = Engine(
        retriever=lambda q: seen.append(q) or [FEE],
        decider=lambda s, h: seen.append(str(h)) or d(Intent.QUESTION, Action.ANSWER, "x"),
        judge=lambda q, p: seen.append(q),
    )
    r = e.respond("my otp is 483920, why did checkout fail?")
    assert not any("483920" in x for x in seen)
    assert "483920" not in str(e.history)
    assert r.redacted == ("one-time code",) and r.reply.startswith("For your safety")


def test_message_that_is_only_a_secret_skips_everything():
    e = Engine(retriever=boom, decider=boom, judge=boom)
    r = e.respond("my otp is 483920")
    assert r.action is Action.CLARIFY and r.redacted == ("one-time code",)


# --- Recording ------------------------------------------------------------------------------


def recorded(engine):
    """Give the engine a recorder and return the list it fills."""
    records = []
    engine.recorder = records.append
    return records


def test_each_message_is_recorded_as_a_numbered_turn_of_one_session():
    e = make([CARDS], *[d(Intent.QUESTION, Action.CLARIFY, "Which retailer?")] * 2)
    records = recorded(e)
    e.respond("card a")
    e.respond("Walmart")
    assert [r.turn for r in records] == [1, 2]
    assert {r.session_id for r in records} == {e.session_id}
    assert records[1].trace.query == "card a Walmart"
    assert make([CARDS]).session_id != e.session_id


def test_trace_keeps_the_models_proposal_and_the_judge_checks():
    e = make([FEE], d(Intent.QUESTION, Action.ANSWER, "It's $99!"), judge=quotes_whole_section)
    records = recorded(e)
    r = e.respond("what is the fee?")
    trace = records[0].trace
    assert trace.decision.reply == "It's $99!"
    assert r.reply == "The ACO fee is $25 CAD, paid after delivery."
    assert [(c.hit, c.supported) for c in trace.checks] == [(FEE, True)]
    assert not trace.gated and set(trace.timings_ms) == {"retrieve", "decide", "verify"}


def test_trace_of_a_declined_message_shows_the_llm_was_never_called():
    e = Engine(retriever=lambda q: FAR, decider=boom, judge=boom)
    records = recorded(e)
    e.respond("Write me a python script")
    trace = records[0].trace
    assert trace.gated and trace.decision is None and trace.checks == ()
    assert set(trace.timings_ms) == {"retrieve"}


def test_secrets_never_reach_the_record():
    e = make([FEE], d(Intent.QUESTION, Action.ANSWER, "x"), judge=quotes_whole_section)
    records = recorded(e)
    e.respond("my otp is 483920, why did checkout fail?")
    assert "483920" not in str(records)
    assert "[REDACTED]" in records[0].message


def test_message_that_is_only_a_secret_is_recorded_without_a_trace():
    e = Engine(retriever=boom, decider=boom, judge=boom)
    records = recorded(e)
    e.respond("my otp is 483920")
    assert records[0].trace is None and records[0].response.action is Action.CLARIFY


def test_the_handoff_turn_ends_the_session_in_the_record():
    e = make(
        [CARDS],
        d(Intent.QUESTION, Action.CLARIFY, "Which retailer?"),
        d(Intent.QUESTION, Action.HANDOFF, "Staff will help", "card question"),
    )
    records = recorded(e)
    for m in ("card?", "walmart", "my otp is 483920, hello?"):
        e.respond(m)
    assert [r.end_reason for r in records] == [None, "handoff", None]
    after = records[-1]
    assert after.trace is None and after.response.reply.endswith(replies.CLOSED)
    assert after.response.reply.startswith(replies.SECRET_WARNING)
    assert "483920" not in str(records)


def test_a_failing_recorder_never_costs_the_customer_their_reply(caplog):
    def broken(record):
        raise RuntimeError("database down")

    e = make([FEE], d(Intent.QUESTION, Action.ANSWER, "x"), judge=quotes_whole_section)
    e.recorder = broken
    assert e.respond("what is the fee?").action is Action.ANSWER
    assert "Could not record turn 1" in caplog.text


def test_repeated_follow_ups_keep_re_asking_the_original_question():
    queries = []
    e = Engine(
        retriever=lambda q: queries.append(q) or fee_search(q),
        decider=lambda s, h: d(Intent.QUESTION, Action.ANSWER, "x"),
        judge=quotes_whole_section,
    )
    for m in ("how much is the aco fee", "walmart", "what about for costco", "and amazon?"):
        e.respond(m)
    assert queries[-1] == "how much is the aco fee and amazon?"
