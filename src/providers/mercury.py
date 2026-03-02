"""Mercury 2 LLM Provider - Diffusion-based fast inference.

Mercury 2 from Inception Labs achieves ~1000 tokens/second output throughput
using parallel diffusion-based generation instead of sequential autoregressive.

This is 10x faster than Claude Haiku and 14x faster than GPT-5 Mini.

API Docs: https://platform.inceptionlabs.ai
Get API Key: https://platform.inceptionlabs.ai

Usage:
    client = MercuryClient(api_key="your-key")
    response = await client.generate(
        messages=[{"role": "user", "content": "Hello"}],
        reasoning_effort="low",  # "low" for speed, "high" for quality
    )
"""

import logging
from typing import AsyncIterator, Literal, Optional

import httpx

logger = logging.getLogger(__name__)


class MercuryError(Exception):
    """Mercury API error."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class MercuryClient:
    """Async client for Mercury 2 API.

    Mercury 2 uses diffusion-based parallel generation for ~10x faster inference.
    Best suited for latency-critical applications like RAG agent loops.
    """

    BASE_URL = "https://api.inceptionlabs.ai/v1"

    def __init__(
        self,
        api_key: str,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """Initialize Mercury client.

        Args:
            api_key: Inception Labs API key
            timeout: Request timeout in seconds
            max_retries: Max retry attempts on failure
        """
        if not api_key:
            raise MercuryError("Mercury API key is required")

        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def generate(
        self,
        messages: list[dict],
        model: str = "mercury-2",
        reasoning_effort: Literal["low", "high"] = "low",
        max_tokens: int = 1024,
        temperature: float = 0.1,
        stream: bool = False,
        json_mode: bool = False,
    ) -> str:
        """Generate a response using Mercury 2.

        Args:
            messages: Chat messages in OpenAI format
            model: Model ID (default: mercury-2)
            reasoning_effort: "low" for speed, "high" for quality
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
            stream: Whether to stream response
            json_mode: Force JSON output format

        Returns:
            Generated text response

        Raises:
            MercuryError: On API errors
        """
        client = await self._get_client()

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        if stream:
            return await self._stream_generate(client, payload)

        for attempt in range(self.max_retries):
            try:
                response = await client.post("/chat/completions", json=payload)

                if response.status_code == 429:
                    # Rate limited, wait and retry
                    wait_time = 2 ** attempt
                    logger.warning(f"Mercury rate limited, retrying in {wait_time}s")
                    import asyncio
                    await asyncio.sleep(wait_time)
                    continue

                if response.status_code != 200:
                    error_text = response.text
                    raise MercuryError(
                        f"Mercury API error: {error_text}",
                        status_code=response.status_code,
                    )

                data = response.json()
                return data["choices"][0]["message"]["content"]

            except httpx.TimeoutException:
                if attempt == self.max_retries - 1:
                    raise MercuryError("Mercury request timed out")
                logger.warning(f"Mercury timeout, retrying ({attempt + 1}/{self.max_retries})")

            except httpx.RequestError as e:
                if attempt == self.max_retries - 1:
                    raise MercuryError(f"Mercury request failed: {e}")
                logger.warning(f"Mercury request error, retrying: {e}")

        raise MercuryError("Mercury max retries exceeded")

    async def _stream_generate(
        self,
        client: httpx.AsyncClient,
        payload: dict,
    ) -> str:
        """Stream generate response.

        Note: Mercury 2 uses diffusion-based generation, so streaming
        behavior differs from autoregressive models. Instead of token-by-token,
        it streams refined chunks that converge to the final response.
        """
        payload["stream"] = True
        full_response = ""

        async with client.stream("POST", "/chat/completions", json=payload) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                raise MercuryError(
                    f"Mercury API error: {error_text.decode()}",
                    status_code=response.status_code,
                )

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        import json
                        chunk = json.loads(data)
                        if "choices" in chunk and chunk["choices"]:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            full_response += content
                    except json.JSONDecodeError:
                        continue

        return full_response

    async def stream_generate(
        self,
        messages: list[dict],
        model: str = "mercury-2",
        reasoning_effort: Literal["low", "high"] = "low",
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> AsyncIterator[str]:
        """Stream generate response, yielding chunks.

        Yields:
            Text chunks as they are generated
        """
        client = await self._get_client()

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
            "stream": True,
        }

        async with client.stream("POST", "/chat/completions", json=payload) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                raise MercuryError(
                    f"Mercury API error: {error_text.decode()}",
                    status_code=response.status_code,
                )

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        import json
                        chunk = json.loads(data)
                        if "choices" in chunk and chunk["choices"]:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue


def get_mercury_client() -> Optional[MercuryClient]:
    """Get Mercury client if API key is configured.

    Returns:
        MercuryClient if MERCURY_API_KEY is set, None otherwise
    """
    from src.config.settings import get_settings

    settings = get_settings()
    api_key = settings.mercury_api_key.get_secret_value()

    if not api_key:
        return None

    return MercuryClient(api_key=api_key)
