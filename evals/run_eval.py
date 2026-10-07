"""Run the golden question set against the engine and report quality and latency.

Usage: python evals/run_eval.py [--only PREFIX,PREFIX] [--limit N]
A case's "history" messages are sent first, as earlier turns of the same conversation.
Exits non-zero on any hallucinated amount / forbidden claim, any answer to an undocumented
question, or p95 latency over budget.
"""

import json
import re
import statistics
import sys
import time
from pathlib import Path

from rag_engine.assistant.engine import Engine

GOLDEN = Path(__file__).parent / "golden.jsonl"
RESULTS = Path(__file__).parent / "results"
LATENCY_BUDGET_S = 10.0
_MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _unsupported_amounts(reply: str, context: str) -> list[str]:
    ctx = _squash(context)
    return [m for m in _MONEY.findall(reply) if _squash(m) not in ctx]


def run_case(case: dict) -> dict:
    engine = Engine()
    # Earlier turns of a conversation; only the last message is scored and timed.
    for earlier in case.get("history", []):
        engine.respond(earlier)
    start = time.perf_counter()
    r = engine.respond(case["message"])
    elapsed = time.perf_counter() - start

    reply = r.reply.lower()
    context = "\n".join(h.content for h in r.hits)
    needles = [s.lower() for s in case.get("must_contain", [])]
    if not needles:
        facts_ok = True
    elif case.get("contain_any"):
        facts_ok = any(n in reply for n in needles)
    else:
        facts_ok = all(n in reply for n in needles)

    sources = case.get("sources", [])
    return {
        "id": case["id"],
        "message": case["message"],
        "intent": r.intent.value,
        "action": r.action.value,
        "reply": r.reply,
        "composed": r.composed,
        "seconds": round(elapsed, 2),
        "intent_ok": r.intent.value == case["intent"],
        "action_ok": r.action.value in case["actions"],
        "retrieval_ok": (not sources) or any(s in h.path for h in r.hits for s in sources),
        "section_ok": (not case.get("headings"))
        or any(n.lower() in h.heading.lower() for h in r.hits for n in case["headings"]),
        "facts_ok": facts_ok or r.action.value != "answer",
        "unsupported_amounts": _unsupported_amounts(r.reply, context)
        if r.action.value == "answer"
        else [],
        "forbidden": [n for n in case.get("must_not", []) if n.lower() in reply],
    }


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p * (len(ordered) - 1)))]


def main() -> int:
    if not GOLDEN.exists():
        print(f"{GOLDEN} not found: copy golden.example.jsonl to it and write real cases.")
        return 2
    cases = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    if "--only" in sys.argv:  # e.g. --only nm-,long-
        prefixes = tuple(sys.argv[sys.argv.index("--only") + 1].split(","))
        cases = [c for c in cases if c["id"].startswith(prefixes)]
    if "--limit" in sys.argv:
        cases = cases[: int(sys.argv[sys.argv.index("--limit") + 1])]

    # Warm the model; excluded from latency stats. A greeting would not call it.
    Engine().respond("how do I sign up?")
    results = []
    for case in cases:
        res = run_case(case)
        results.append(res)
        flags = "".join(
            f" !{k}"
            for k in ("intent_ok", "action_ok", "retrieval_ok", "section_ok", "facts_ok")
            if not res[k]
        )
        print(f"{res['seconds']:5.1f}s {case['id']:<20} {res['intent']}/{res['action']}{flags}")

    n = len(results)
    times = [r["seconds"] for r in results]
    hallucinated = [r for r in results if r["unsupported_amounts"] or r["forbidden"]]
    # Answered although the docs do not answer it: the worst failure for a support bot.
    wrongly_answered = [
        r
        for r, c in zip(results, cases, strict=True)
        if r["action"] == "answer" and "answer" not in c["actions"]
    ]
    with_source = [r for r, c in zip(results, cases, strict=True) if c.get("sources")]
    answered = [r for r, c in zip(results, cases, strict=True) if c["actions"] == ["answer"]]

    print("\n== Summary ==")
    print(f"intent accuracy : {sum(r['intent_ok'] for r in results)}/{n}")
    print(f"action accuracy : {sum(r['action_ok'] for r in results)}/{n}")
    print(f"retrieval hit   : {sum(r['retrieval_ok'] for r in with_source)}/{len(with_source)}")
    with_heading = [r for r, c in zip(results, cases, strict=True) if c.get("headings")]
    print(f"section hit     : {sum(r['section_ok'] for r in with_heading)}/{len(with_heading)}")
    print(f"answer facts    : {sum(r['facts_ok'] for r in answered)}/{len(answered)}")
    given = [r for r in results if r["action"] == "answer"]
    print(
        f"own wording     : {sum(r['composed'] for r in given)}/{len(given)} answers "
        "(the rest sent the doc section verbatim)"
    )
    print(f"hallucinations  : {len(hallucinated)}")
    print(f"answered undocumented : {len(wrongly_answered)}")
    print(
        f"latency         : p50 {statistics.median(times):.1f}s  p95 {pct(times, 0.95):.1f}s  "
        f"max {max(times):.1f}s  (budget {LATENCY_BUDGET_S:.0f}s)"
    )
    for r in hallucinated:
        print(
            f"  HALLUCINATION {r['id']}: {r['unsupported_amounts'] or r['forbidden']} -> {r['reply']}"
        )

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "latest.json").write_text(json.dumps(results, indent=2))
    for r in wrongly_answered:
        print(f"  ANSWERED UNDOCUMENTED {r['id']}: {r['reply'][:120]}")
    failed = hallucinated or wrongly_answered or pct(times, 0.95) > LATENCY_BUDGET_S
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
