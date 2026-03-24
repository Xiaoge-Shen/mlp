from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import EndpointConfig


@dataclass
class ChatResponse:
    raw: dict[str, Any]
    text: str
    latency_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None


class OpenAICompatibleClient:
    def __init__(self, endpoint: EndpointConfig) -> None:
        self.endpoint = endpoint
        api_key = os.environ.get(endpoint.api_key_env, "")
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_output_tokens: int,
    ) -> ChatResponse:
        payload = {
            "model": self.endpoint.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        url = self.endpoint.base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers,
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.endpoint.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI-compatible request failed with {exc.code}: {body}") from exc
        latency = time.perf_counter() - start
        choice = raw["choices"][0]["message"]
        usage = raw.get("usage", {})
        return ChatResponse(
            raw=raw,
            text=choice.get("content", "") or "",
            latency_seconds=latency,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
