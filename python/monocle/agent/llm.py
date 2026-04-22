"""Thin Ollama chat wrapper. Shared by rewrite + validate nodes."""

from __future__ import annotations

import ollama

DEFAULT_MODEL = "llama3.2:3b"


class OllamaClient:
    """Minimal Ollama wrapper. Single-turn chat with optional system prompt."""

    def __init__(self, model: str = DEFAULT_MODEL, host: str | None = None):
        self.model = model
        self._client = ollama.Client(host=host) if host else ollama.Client()

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        """Run one chat turn. Returns the assistant text, stripped."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return resp["message"]["content"].strip()
