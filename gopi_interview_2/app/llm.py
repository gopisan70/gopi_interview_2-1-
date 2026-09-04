"""Thin LLM wrapper with native tool calling.

Two backends behind one tiny interface:
  * AnthropicLLM - primary (Claude Messages API with native tool use).
  * OllamaLLM    - local fallback so the project runs without an API key (Ollama /api/chat with tools).

Messages are kept in a neutral format and converted per provider:
  {"role": "user", "content": str}
  {"role": "assistant", "content": str, "tool_calls": [ToolCall], "raw": <provider blocks, optional>}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}
Tool schemas use the Anthropic shape: {"name", "description", "input_schema"}.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import uuid4

from app import config

log = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised for any provider failure; the agent turns it into a safe user-facing message."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    raw_content: Any = None          # provider-native assistant content, replayed verbatim in tool loops


class LLM(Protocol):
    def chat(self, system: str, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse: ...


# --------------------------------------------------------------------------- Anthropic
class AnthropicLLM:
    def __init__(self, model: str = config.ANTHROPIC_MODEL, effort: str = config.ANTHROPIC_EFFORT):
        import anthropic  # imported lazily so the Ollama path works without the package configured

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model, self.effort = model, effort

    def chat(self, system: str, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=_to_anthropic_messages(messages),
            output_config={"effort": self.effort},
        )
        if tools:
            kwargs["tools"] = tools
        try:
            response = self.client.messages.create(**kwargs)
        except self._anthropic.RateLimitError as exc:
            raise LLMError("rate limited") from exc
        except self._anthropic.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise LLMError("connection error") from exc

        if response.stop_reason == "refusal":
            return LLMResponse(stop_reason="refusal", raw_content=response.content)
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in response.content if b.type == "tool_use"]
        return LLMResponse(text, calls, response.stop_reason, response.content)


def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            if m.get("raw") is not None:
                content = m["raw"]                      # keeps thinking + tool_use blocks intact
            else:
                content = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                content += [{"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                            for tc in m.get("tool_calls", [])]
            out.append({"role": "assistant", "content": content})
        elif m["role"] == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            previous = out[-1] if out else None
            if previous and previous["role"] == "user" and isinstance(previous["content"], list):
                previous["content"].append(block)       # all results of one round go in ONE user message
            else:
                out.append({"role": "user", "content": [block]})
    return out


# --------------------------------------------------------------------------- Ollama (local fallback)
class OllamaLLM:
    def __init__(self, model: str = config.OLLAMA_MODEL, url: str = config.OLLAMA_URL):
        self.model, self.url = model, url.rstrip("/")

    def chat(self, system: str, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        import requests

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + _to_ollama_messages(messages),
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 8192},
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}} for t in tools]
        try:
            resp = requests.post(f"{self.url}/api/chat", json=payload, timeout=600)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        message = resp.json().get("message", {})
        calls = [ToolCall(f"call_{uuid4().hex[:8]}", tc["function"]["name"], dict(tc["function"].get("arguments") or {}))
                 for tc in message.get("tool_calls", [])]
        return LLMResponse((message.get("content") or "").strip(), calls)


def _to_ollama_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m["role"] == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": m.get("content", "")}
            if m.get("tool_calls"):
                entry["tool_calls"] = [{"function": {"name": tc.name, "arguments": tc.arguments}} for tc in m["tool_calls"]]
            out.append(entry)
        elif m["role"] == "tool":
            out.append({"role": "tool", "content": m["content"], "tool_name": m.get("name", "")})
        else:
            out.append({"role": "user", "content": m["content"]})
    return out


# --------------------------------------------------------------------------- factory
def get_llm() -> LLM:
    if config.LLM_PROVIDER == "anthropic":
        log.info("LLM provider: anthropic (%s, effort=%s)", config.ANTHROPIC_MODEL, config.ANTHROPIC_EFFORT)
        return AnthropicLLM()
    if config.LLM_PROVIDER == "ollama":
        log.info("LLM provider: ollama (%s at %s)", config.OLLAMA_MODEL, config.OLLAMA_URL)
        return OllamaLLM()
    raise LLMError(f"Unknown LLM_PROVIDER {config.LLM_PROVIDER!r}; use 'anthropic' or 'ollama'.")
