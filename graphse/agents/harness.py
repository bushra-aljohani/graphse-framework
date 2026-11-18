"""Deterministic agent harness.

The application, not the model, decides what the model sees, which output is
accepted and what happens on failure:

    for attempt in 1..max_attempts:
        raw    = model(messages)              # provider failure -> ProviderError (not hidden)
        result = parse(raw)                   # structural normalisation, e.g. citation filtering
        problem = check(result)               # deterministic quality checks
        log(attempt, raw, result, problem)
        if no problem: accept
        messages += [raw, repair instruction naming the problem]
    if still failing: fallback(...)

Checks are structural. A valid citation index does not prove that the cited
passage supports the claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .llm import LLMClient, ProviderError


@dataclass
class HarnessResult:
    result: dict
    outcome: str                     # "first_pass", "repaired" or "fallback"
    attempts: list[dict] = field(default_factory=list)


def run_harness(llm: LLMClient, system: str, messages: list[dict],
                parse: Callable[[str], dict], check: Callable[[dict], str | None],
                fallback: Callable[[], dict], max_attempts: int = 3,
                temperatures: tuple[float, float] = (0.1, 0.0),
                log: Callable[[dict], Any] | None = None) -> HarnessResult:
    messages = list(messages)
    attempts: list[dict] = []
    problem = None
    for attempt in range(1, max_attempts + 1):
        temperature = temperatures[0] if attempt == 1 else temperatures[1]
        try:
            raw = llm.complete(system, messages, temperature)
        except ProviderError as exc:
            record = {"attempt": attempt, "error": str(exc)}
            attempts.append(record)
            if log:
                log(record)
            raise
        result = parse(raw)
        problem = check(result)
        record = {"attempt": attempt, "raw": raw, "problem": problem}
        attempts.append(record)
        if log:
            log(record)
        if problem is None:
            return HarnessResult(result, "first_pass" if attempt == 1 else "repaired", attempts)
        messages += [{"role": "assistant", "content": raw},
                     {"role": "user", "content": f"{problem} Return the same JSON shape."}]
    return HarnessResult(fallback(), "fallback", attempts)
