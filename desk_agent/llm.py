"""Chat clients. The real one speaks the OpenAI-style chat completions protocol,
so it works with Ollama, vLLM, LM Studio or a hosted API by changing three
environment variables. ``ScriptedLLM`` replays fixed replies for tests."""

from __future__ import annotations

import os
from typing import Any, Protocol

import httpx


class LLM(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Return one assistant message: ``{"content": str|None, "tool_calls": [...]|None}``."""


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "qwen2.5:3b")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.timeout = timeout

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {"model": self.model, "messages": messages, "tools": tools, "temperature": 0}
        resp = httpx.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]


class ScriptedLLM:
    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.seen: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        self.seen.append(list(messages))
        if not self.replies:
            return {"content": "(script exhausted)", "tool_calls": None}
        return self.replies.pop(0)
