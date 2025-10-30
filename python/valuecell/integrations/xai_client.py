"""Async client helpers for interacting with xAI Grok models."""

from __future__ import annotations

import os
from typing import Iterable, Optional

import httpx
from loguru import logger

DEFAULT_MODEL = "grok-4-fast"
DEFAULT_BASE_URL = "https://api.x.ai/v1"


class XAIClientError(RuntimeError):
    """Raised when the xAI API returns an unexpected response."""


class XAIClient:
    """Thin asynchronous client for xAI chat completions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise XAIClientError(
                "XAI_API_KEY is not set. Please configure it in your environment."
            )

        self.model = model
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def chat_completion(
        self,
        messages: Iterable[dict],
        *,
        temperature: float = 0.3,
        max_output_tokens: Optional[int] = None,
        extra_body: Optional[dict] = None,
    ) -> str:
        """Call xAI chat completions and return the assistant content."""

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if extra_body:
            payload.update(extra_body)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, json=payload, headers=headers)

        if response.status_code >= 400:
            logger.error(
                "xAI API error %s: %s", response.status_code, response.text[:500]
            )
            raise XAIClientError(
                f"xAI API request failed with status {response.status_code}"
            )

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Unexpected xAI response structure: %s", data)
            raise XAIClientError("Unexpected xAI API response format") from exc


async def grok_search(
    query: str,
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_output_tokens: Optional[int] = None,
    extra_body: Optional[dict] = None,
) -> str:
    """Convenience wrapper to execute a Grok chat completion."""

    client = XAIClient(model=model or DEFAULT_MODEL)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})

    return await client.chat_completion(
        messages,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        extra_body=extra_body,
    )
