"""Model backends. Ollama for local models, Anthropic for frontier ones.

Both are thin: one ``chat(system, messages) -> LLMResponse`` call. Temperature
is fixed per run and reported, because it is the free exogenous variation that
makes the influence identifiable at all.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
ANTHROPIC_VERSION = "2023-06-01"


@dataclass
class LLMResponse:
    text: str
    latency_s: float
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    #: reasoning emitted on a separate channel, where the backend exposes one.
    #: Kept apart from ``text`` so it never collides with the output format,
    #: and so the judge can be pointed at the chain of thought explicitly.
    thinking: str = ""


class LLMClient(Protocol):
    name: str
    temperature: float

    def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse: ...

    def describe(self) -> dict[str, Any]: ...


class OllamaClient:
    """Local models through the Ollama chat API."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        seed: int | None = None,
        num_ctx: int = 16384,
        num_predict: int = 768,
        host: str = DEFAULT_OLLAMA_HOST,
        timeout: float = 600.0,
        think: bool | None = None,
        keep_alive: str = "30m",
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.name = f"ollama:{model}"
        self.temperature = temperature
        self.seed = seed
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.host = host.rstrip("/")
        self.think = think
        self.keep_alive = keep_alive
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=timeout)

    def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }
        if self.seed is not None:
            payload["options"]["seed"] = self.seed
        if self.think is not None:
            payload["think"] = self.think
        if schema is not None:
            # Constrained decoding. Without it a mid-sized model echoes the
            # example template often enough to wreck the label distribution.
            payload["format"] = schema

        last: Exception | None = None
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                r = self._client.post(f"{self.host}/api/chat", json=payload)
                r.raise_for_status()
                data = r.json()
            except Exception as exc:  # network hiccup / model load timeout
                last = exc
                time.sleep(2.0 * (attempt + 1))
                continue
            msg = data.get("message", {}) or {}
            text = msg.get("content", "") or ""
            thinking = msg.get("thinking", "") or ""
            if not text.strip() and thinking:
                # The budget went entirely on reasoning and nothing came out
                # the other side; hand the reasoning over so the caller can at
                # least try to parse it, and flag the truncation.
                text = thinking
            return LLMResponse(
                text=text,
                thinking=thinking,
                latency_s=time.time() - t0,
                usage={
                    "prompt_eval_count": data.get("prompt_eval_count"),
                    "eval_count": data.get("eval_count"),
                    "truncated": data.get("done_reason") == "length",
                },
                raw={k: v for k, v in data.items() if k != "message"},
            )
        raise RuntimeError(f"ollama call failed after {self.max_retries} attempts: {last}")

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "ollama",
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
            "num_ctx": self.num_ctx,
            "think": self.think,
        }


class AnthropicClient:
    """Frontier models through the Anthropic Messages API."""

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        api_key: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 4,
    ) -> None:
        self.model = model
        self.name = f"anthropic:{model}"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.max_retries = max_retries
        self._client = httpx.Client(
            timeout=timeout,
            base_url="https://api.anthropic.com",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
        )

    def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        del schema  # the prompt carries the shape for this backend
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system,
            "messages": messages,
        }
        last: Exception | None = None
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                r = self._client.post("/v1/messages", json=payload)
                if r.status_code in (429, 500, 502, 503, 529):
                    raise httpx.HTTPError(f"retryable status {r.status_code}")
                r.raise_for_status()
                data = r.json()
            except Exception as exc:
                last = exc
                time.sleep(min(2.0**attempt, 30.0))
                continue
            text = "".join(
                block.get("text", "")
                for block in data.get("content", [])
                if block.get("type") == "text"
            )
            return LLMResponse(
                text=text,
                latency_s=time.time() - t0,
                usage=data.get("usage", {}),
                raw={"id": data.get("id"), "stop_reason": data.get("stop_reason")},
            )
        raise RuntimeError(f"anthropic call failed after {self.max_retries} attempts: {last}")

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "anthropic",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


class EchoClient:
    """Deterministic stand-in used by the tests and by ``--dry-run``.

    Plays the role of an unaware adaptive forecaster: naive persistence plus a
    small deterministic wobble, so the whole pipeline can be exercised without
    a model server.
    """

    def __init__(self, temperature: float = 0.0, wobble: float = 1.0) -> None:
        self.name = "echo:adaptive"
        self.temperature = temperature
        self.wobble = wobble
        self._calls = 0

    def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        import re

        del schema

        self._calls += 1
        text = messages[-1]["content"]
        nums = re.findall(r"^\s*\d+\s+([-+]?\d+\.\d+)", text, re.MULTILINE)
        last = float(nums[-1]) if nums else 100.0
        guess = last + self.wobble * ((self._calls % 3) - 1)
        return LLMResponse(
            text=f"FORECAST: {guess:.2f}\nREASONING: persistence.\nNOTES:",
            latency_s=0.0,
        )

    def describe(self) -> dict[str, Any]:
        return {"backend": "echo", "model": "adaptive", "temperature": self.temperature}


def make_client(spec: str, **kwargs) -> LLMClient:
    """``ollama:qwen3:32b`` / ``anthropic:claude-sonnet-5`` / ``echo``."""
    if spec == "echo":
        return EchoClient(**{k: v for k, v in kwargs.items() if k in ("temperature", "wobble")})
    backend, _, model = spec.partition(":")
    if backend == "ollama":
        allowed = {"temperature", "seed", "num_ctx", "num_predict", "host", "timeout", "think", "keep_alive"}
        return OllamaClient(model, **{k: v for k, v in kwargs.items() if k in allowed})
    if backend == "anthropic":
        allowed = {"temperature", "max_tokens", "api_key", "timeout"}
        return AnthropicClient(model, **{k: v for k, v in kwargs.items() if k in allowed})
    raise ValueError(f"unknown model spec {spec!r}")


def list_ollama_models(host: str = DEFAULT_OLLAMA_HOST, timeout: float = 10.0) -> list[dict]:
    r = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
    r.raise_for_status()
    return r.json().get("models", [])
