"""Run the golden question set against the engine and report quality and latency.

Usage: python evals/run_eval.py [--only PREFIX,PREFIX] [--limit N]

Each case runs in a fresh conversation; its "history" messages are sent first as earlier turns,
and only its "message" is scored and timed. Exits non-zero on any hallucinated amount or
forbidden claim, any answer to an undocumented question, or p95 latency over budget. The case
format is described in the README ("Quality checks").
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
_CHECKS = ("intent_ok", "action_ok", "retrieval_ok", "section_ok", "facts_ok")


def main() -> int:
    if not GOLDEN.exists():
        print(f"{GOLDEN} not found: copy golden.example.jsonl to it and write real cases.")
        return 2
    cases = _selected(_load(GOLDEN), sys.argv[1:])

    # Warm the model; excluded from latency stats. A greeting would not call it.
    Engine().respond("how do I sign up?")
    results = []
    for case in cases:
        result = run_case(case)
        results.append(result)
        flags = "".join(f" !{k}" for k in _CHECKS if not result[k])
        print(
            f"{result['seconds']:5.1f}s {case['id']:<20} {result['intent']}/{result['action']}{flags}"
        )

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "latest.json").write_text(json.dumps(results, indent=2))
    return 1 if _report(results, cases) else 0


def run_case(case: dict) -> dict:
    """Send one case through a fresh Engine and score the reply to its message."""
    engine = Engine()
    for earlier in case.get("history", []):
        engine.respond(earlier)
    start = time.perf_counter()
    r = engine.respond(case["message"])
    elapsed = time.perf_counter() - start

    reply = r.reply.lower()
    answered = r.action.value == "answer"
    sources = case.get("sources", [])
    headings = [h.lower() for h in case.get("headings", [])]
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
        "retrieval_ok": not sources or any(s in h.path for h in r.hits for s in sources),
        "section_ok": not headings or any(n in h.heading.lower() for h in r.hits for n in headings),
        "facts_ok": not answered or _has_facts(reply, case),
        "unsupported_amounts": _unsupported_amounts(r.reply, r.hits) if answered else [],
        "forbidden": [n for n in case.get("must_not", []) if n.lower() in reply],
    }


def _report(results: list[dict], cases: list[dict]) -> bool:
    """Print the summary; True if the eval failed."""
    pairs = list(zip(results, cases, strict=True))
    times = [r["seconds"] for r in results]
    hallucinated = [r for r in results if r["unsupported_amounts"] or r["forbidden"]]
    # Answered although the docs do not answer it: the worst failure for a support bot.
    wrongly_answered = [
        r for r, c in pairs if r["action"] == "answer" and "answer" not in c["actions"]
    ]
    with_source = [r for r, c in pairs if c.get("sources")]
    with_heading = [r for r, c in pairs if c.get("headings")]
    answerable = [r for r, c in pairs if c["actions"] == ["answer"]]
    given = [r for r in results if r["action"] == "answer"]
    p95 = _percentile(times, 0.95)

    print("\n== Summary ==")
    print(f"intent accuracy : {_count(results, 'intent_ok')}")
    print(f"action accuracy : {_count(results, 'action_ok')}")
    print(f"retrieval hit   : {_count(with_source, 'retrieval_ok')}")
    print(f"section hit     : {_count(with_heading, 'section_ok')}")
    print(f"answer facts    : {_count(answerable, 'facts_ok')}")
    print(
        f"own wording     : {_count(given, 'composed')} answers (the rest sent the doc section verbatim)"
    )
    print(f"hallucinations  : {len(hallucinated)}")
    print(f"answered undocumented : {len(wrongly_answered)}")
    print(
        f"latency         : p50 {statistics.median(times):.1f}s  p95 {p95:.1f}s  "
        f"max {max(times):.1f}s  (budget {LATENCY_BUDGET_S:.0f}s)"
    )
    for r in hallucinated:
        print(
            f"  HALLUCINATION {r['id']}: {r['unsupported_amounts'] or r['forbidden']} -> {r['reply']}"
        )
    for r in wrongly_answered:
        print(f"  ANSWERED UNDOCUMENTED {r['id']}: {r['reply'][:120]}")
    return bool(hallucinated or wrongly_answered or p95 > LATENCY_BUDGET_S)


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _selected(cases: list[dict], args: list[str]) -> list[dict]:
    """Apply --only PREFIX,PREFIX (case ids) and --limit N."""
    if "--only" in args:
        prefixes = tuple(args[args.index("--only") + 1].split(","))
        cases = [c for c in cases if c["id"].startswith(prefixes)]
    if "--limit" in args:
        cases = cases[: int(args[args.index("--limit") + 1])]
    return cases


def _has_facts(reply: str, case: dict) -> bool:
    """The answer contains the case's must_contain texts (any one of them with contain_any)."""
    needles = [s.lower() for s in case.get("must_contain", [])]
    if not needles:
        return True
    found = (n in reply for n in needles)
    return any(found) if case.get("contain_any") else all(found)


def _unsupported_amounts(reply: str, hits: list) -> list[str]:
    """Dollar amounts in the reply that none of the retrieved sections contain."""
    context = _squash("\n".join(h.content for h in hits))
    return [m for m in _MONEY.findall(reply) if _squash(m) not in context]


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _count(results: list[dict], key: str) -> str:
    return f"{sum(r[key] for r in results)}/{len(results)}"


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p * (len(ordered) - 1)))]


if __name__ == "__main__":
    sys.exit(main())
