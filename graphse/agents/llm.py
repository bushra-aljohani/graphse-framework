"""Model interface. Agents depend on this protocol only, never on a provider SDK.

ScriptedLLM replays fixed responses so the framework runs and is tested offline.
To use a hosted model, implement `complete` with the provider's chat API and
JSON output mode.
"""
from __future__ import annotations

import json
from typing import Protocol


class ProviderError(RuntimeError):
    """The model provider failed (network, quota, server error). Never hidden by a fallback."""


class LLMClient(Protocol):
    def complete(self, system: str, messages: list[dict], temperature: float) -> str: ...


class ScriptedLLM:
    """Returns queued responses in order; the string 'PROVIDER_ERROR' raises ProviderError."""

    def __init__(self, responses: list[object] | None = None):
        self.queue = list(responses or [])
        self.calls: list[dict] = []

    def push(self, *responses: object) -> None:
        self.queue.extend(responses)

    def complete(self, system: str, messages: list[dict], temperature: float) -> str:
        self.calls.append({"system": system, "messages": messages, "temperature": temperature})
        if not self.queue:
            raise AssertionError("ScriptedLLM: no response queued")
        r = self.queue.pop(0)
        if r == "PROVIDER_ERROR":
            raise ProviderError("injected provider failure")
        return r if isinstance(r, str) else json.dumps(r)
