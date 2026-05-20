from __future__ import annotations

import logging
import time
from collections.abc import Iterator

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from .settings import normalize_base_url

logger = logging.getLogger("kb_web_agent.llm")


class OpenAICompatibleClient:
    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        model: str,
        timeout_seconds: int = 90,
        sdk_client=None,
    ) -> None:
        self.api_key = api_key
        self.base_url = normalize_base_url(base_url)
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.sdk_client = sdk_client

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("Missing ARK_API_KEY. Configure it in .env or environment variables.")
        client = self.sdk_client or OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
        )
        logger.info(
            "[LLM] 发起请求 model=%s messages=%d max_tokens=%d temperature=%s",
            self.model, len(messages), max_tokens, temperature,
        )
        t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except APIStatusError as exc:
            elapsed = time.perf_counter() - t0
            logger.error("[LLM] HTTP错误 status=%d elapsed=%.2fs body=%s", exc.status_code, elapsed, exc.response.text)
            raise RuntimeError(f"LLM request failed with HTTP {exc.status_code}: {exc.response.text}") from exc
        except (APIConnectionError, APITimeoutError) as exc:
            elapsed = time.perf_counter() - t0
            logger.error("[LLM] 连接/超时错误 elapsed=%.2fs err=%s", elapsed, exc)
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        elapsed = time.perf_counter() - t0
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError(f"Unexpected empty LLM response: {response!r}")
        usage = response.usage
        logger.info(
            "[LLM] 响应完成 elapsed=%.2fs prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            elapsed,
            usage.prompt_tokens if usage else "?",
            usage.completion_tokens if usage else "?",
            usage.total_tokens if usage else "?",
        )
        return content

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> Iterator[str]:
        """流式返回每个 token，供 SSE 端点使用。"""
        if not self.api_key:
            raise RuntimeError("Missing ARK_API_KEY.")
        client = self.sdk_client or OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
        )
        logger.info(
            "[LLM] 流式请求 model=%s messages=%d max_tokens=%d",
            self.model, len(messages), max_tokens,
        )
        t0 = time.perf_counter()
        token_count = 0
        try:
            stream = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    token_count += 1
                    yield delta
        except APIStatusError as exc:
            logger.error("[LLM] 流式HTTP错误 status=%d", exc.status_code)
            raise RuntimeError(f"LLM stream failed with HTTP {exc.status_code}: {exc.response.text}") from exc
        except (APIConnectionError, APITimeoutError) as exc:
            logger.error("[LLM] 流式连接错误 err=%s", exc)
            raise RuntimeError(f"LLM stream failed: {exc}") from exc
        elapsed = time.perf_counter() - t0
        logger.info("[LLM] 流式完成 elapsed=%.2fs tokens≈%d", elapsed, token_count)
