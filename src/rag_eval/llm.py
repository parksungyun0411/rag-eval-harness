"""LLM 공급자 인터페이스.

합성 질의 생성과 LLM 판정은 이 프로토콜에만 의존한다. 키가 없는 환경에서는
``get_provider`` 가 명확한 오류를 내고, 테스트는 가짜 공급자를 주입한다.

환경 변수
- ``LLM_PROVIDER``: ``anthropic`` | ``openai_compat``
- ``LLM_MODEL``: 모델 ID
- ``ANTHROPIC_API_KEY`` (anthropic)
- ``OPENAI_BASE_URL``, ``OPENAI_API_KEY`` (openai_compat: OpenAI 호환 서버, 로컬 vLLM/Ollama 포함)
"""

from __future__ import annotations

import os
from typing import Protocol


class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str: ...


class AnthropicProvider:
    def __init__(self, model: str, api_key: str | None = None):
        import anthropic

        self.name = f"anthropic:{model}"
        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in msg.content if block.type == "text")


class OpenAICompatProvider:
    """``/v1/chat/completions`` 를 직접 호출한다 (SDK 의존 없음)."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
    ):
        import httpx

        self.name = f"openai_compat:{model}"
        self.model = model
        self.temperature = temperature
        base = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip(
            "/"
        )
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = httpx.Client(
            base_url=base, headers={"Authorization": f"Bearer {key}"}, timeout=120
        )

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        resp = self._client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def get_provider() -> LLMProvider:
    kind = os.environ.get("LLM_PROVIDER", "").strip()
    model = os.environ.get("LLM_MODEL", "").strip()
    if not kind or not model:
        raise RuntimeError("LLM_PROVIDER와 LLM_MODEL 환경 변수가 필요합니다 (.env.example 참고)")
    if kind == "anthropic":
        return AnthropicProvider(model)
    if kind == "openai_compat":
        return OpenAICompatProvider(model)
    raise RuntimeError(f"지원하지 않는 LLM_PROVIDER: {kind}")
